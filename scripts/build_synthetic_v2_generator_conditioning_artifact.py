#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (BACKEND_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.synthetic_generation_service import (  # noqa: E402
    CONTROL_FEATURES_BY_CAPABILITY,
    TARGET_FEATURES_BY_CAPABILITY,
    _generate_sample_values,
    _normalize_covariates,
    _realized_features,
    _seed_for,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)
from app.services.synthetic_generator_conditioning import GeneratorConditioning  # noqa: E402
from build_synthetic_v2_feature_gate_artifact import (  # noqa: E402
    DEFAULT_CALIBRATION_FRACTION,
    DEFAULT_GATE_REFERENCE_FRACTION,
    DEFAULT_MAX_WINDOWS,
    FEATURE_GATE_BUCKET_SPECS,
    split_real_rows_three_way,
)
from run_synthetic_v2_near_distance_calibration import load_real_bucket  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "runtime/research"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "backend/app/data/synthetic_v2_generator_conditioning_artifact.json"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "runtime/research/synthetic-v2-generator-conditioning/summary.json"
DEFAULT_SEED = 20260715
DEFAULT_CALIBRATION_SAMPLES = 20
TARGET_PERCENTILE_LEVELS = (0.10, 0.30, 0.50, 0.70, 0.90)
STRUCTURE_SCALE_GRID = (0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0)
LAMBDA_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
PRIMARY_TARGET_FEATURE = {
    "trend": "trend_strength",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "level_shift_strength",
    "nonlinear_persistence": "nonlinear_multi_lag_gain",
    "predictable_intermittency": "spike_rate",
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    "covariate_response": "covariate_incremental_r2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit real-profile-conditioned nuisance parameters and intensity mappings."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--calibration-fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    parser.add_argument("--gate-reference-fraction", type=float, default=DEFAULT_GATE_REFERENCE_FRACTION)
    parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--buckets", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact, summary = build_artifact(
        data_dir=args.data_dir,
        max_windows=args.max_windows,
        calibration_fraction=args.calibration_fraction,
        gate_reference_fraction=args.gate_reference_fraction,
        calibration_samples=args.calibration_samples,
        seed=args.seed,
        bucket_ids=tuple(args.buckets) if args.buckets else None,
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote artifact: {args.artifact}")
    print(f"wrote summary: {args.summary}")
    return 0


def build_artifact(
    *,
    data_dir: Path,
    max_windows: int,
    calibration_fraction: float,
    gate_reference_fraction: float,
    calibration_samples: int,
    seed: int,
    bucket_ids: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [
        spec
        for spec in FEATURE_GATE_BUCKET_SPECS
        if bucket_ids is None or spec.profile_id in bucket_ids
    ]
    if not selected:
        raise ValueError("no bucket specs selected")
    if calibration_samples < 4:
        raise ValueError("calibration_samples must be at least 4")

    created_at = datetime.now(timezone.utc).isoformat()
    profiles: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for spec in selected:
        rows = load_real_bucket(spec, data_dir / spec.asset_name, max_windows=max_windows)
        parameter_rows, _gate_reference, _gate_calibration, split_summary = split_real_rows_three_way(
            rows,
            spec,
            calibration_fraction=calibration_fraction,
            gate_reference_fraction=gate_reference_fraction,
            seed=_seed_for(seed, spec.profile_id, 0),
        )
        real_feature_summary = summarize_real_features(parameter_rows)
        profile_nuisance = derive_profile_nuisance(real_feature_summary, spec.context_length, spec.season_length)
        capability_configs: dict[str, dict[str, Any]] = {}
        capability_summaries: dict[str, dict[str, Any]] = {}
        for capability_id in spec.synthetic_capabilities:
            target_feature_targets = {
                name: quantiles_for_levels(
                    finite_values(parameter_rows, name),
                    TARGET_PERCENTILE_LEVELS,
                )
                for name in TARGET_FEATURES_BY_CAPABILITY[capability_id]
                if finite_values(parameter_rows, name).size
            }
            parameters, intensity_lambdas, calibration_summary = calibrate_capability_conditioning(
                spec=spec,
                capability_id=capability_id,
                profile_nuisance=profile_nuisance,
                real_feature_summary=real_feature_summary,
                target_feature_targets=target_feature_targets,
                sample_count=calibration_samples,
                seed=_seed_for(seed, spec.profile_id, len(capability_configs) + 1),
            )
            capability_configs[capability_id] = {
                "parameters": parameters,
                "intensity_lambdas": intensity_lambdas,
                "target_percentile_levels": list(TARGET_PERCENTILE_LEVELS),
                "target_feature_targets": target_feature_targets,
                "calibration_method": (
                    "parameter-split quantile matching; discrete effect-size grid"
                    if capability_id == "predictable_intermittency"
                    else "parameter-split quantile matching"
                ),
            }
            capability_summaries[capability_id] = calibration_summary

        frequency = profile_frequency(spec.profile_id)
        profiles[spec.profile_id] = {
            "profile_id": spec.profile_id,
            "context_length": int(spec.context_length),
            "horizon": int(spec.horizon),
            "target_dim": int(spec.target_dim),
            "covariate_dim": int(spec.covariate_dim),
            "season_length": int(spec.season_length),
            "frequency": frequency,
            "selection_weight": 1.0,
            "nuisance_parameters": profile_nuisance,
            "real_parameter_feature_summary": real_feature_summary,
            "split": split_summary,
            "capabilities": capability_configs,
        }
        summaries.append(
            {
                "profile_id": spec.profile_id,
                "parameter_window_count": len(parameter_rows),
                "nuisance_parameters": profile_nuisance,
                "capabilities": capability_summaries,
            }
        )

    try:
        data_dir_label = str(data_dir.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        data_dir_label = str(data_dir)
    config = {
        "data_dir": data_dir_label,
        "max_windows_per_bucket": int(max_windows),
        "calibration_fraction": float(calibration_fraction),
        "gate_reference_fraction_of_development": float(gate_reference_fraction),
        "calibration_samples_per_grid_cell": int(calibration_samples),
        "seed": int(seed),
        "split_policy": "generator parameters are fit only on the parameter split",
        "profile_selection_policy": "balanced uniform over exact task/window profiles",
        "intensity_policy": "profile-specific monotone map to real parameter-split target quantiles",
    }
    artifact = {
        "schema_version": "synthetic_v2_generator_conditioning_artifact.v1",
        "created_at": created_at,
        "config": config,
        "profiles": profiles,
    }
    summary = {
        "schema_version": "synthetic_v2_generator_conditioning_calibration.v1",
        "created_at": created_at,
        "config": config,
        "profiles": summaries,
    }
    return artifact, summary


def summarize_real_features(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = sorted({name for row in rows for name in row.get("features", {})})
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        values = finite_values(rows, name)
        if values.size:
            q25, q50, q75 = np.quantile(values, (0.25, 0.50, 0.75))
            summary[name] = {
                "p25": round_float(q25),
                "p50": round_float(q50),
                "p75": round_float(q75),
                "iqr": round_float(max(float(q75 - q25), 1e-6)),
            }
    return summary


def derive_profile_nuisance(
    features: dict[str, dict[str, float]],
    context_length: int,
    season_length: int,
) -> dict[str, float]:
    noise_ratio = median_feature(features, "noise_ratio", 0.12)
    seasonal_strength = median_feature(features, "seasonal_strength", 0.85)
    trend_strength = median_feature(features, "trend_strength", 0.05)
    spike_rate = max(
        median_feature(features, "spike_rate", 0.0),
        median_feature(features, "covariate_residual_spike_rate", 0.0),
    )
    residual_acf = median_feature(features, "covariate_residual_acf_abs_mean", 0.0)
    noise_odds = max(noise_ratio, 1e-4) / max(1.0 - noise_ratio, 1e-4)
    reference_odds = 0.12 / 0.88
    noise_multiplier = float(np.clip(np.sqrt(noise_odds / reference_odds), 0.50, 3.00))
    seasonal_odds = max(seasonal_strength, 1e-4) / max(1.0 - seasonal_strength, 1e-4)
    reference_seasonal_odds = 0.85 / 0.15
    seasonal_multiplier = float(np.clip(np.sqrt(seasonal_odds / reference_seasonal_odds), 0.50, 2.00))
    if spike_rate < 0.005:
        noise_df = 0.0
    else:
        noise_df = float(np.clip(0.28 / spike_rate, 2.25, 20.0))
    context_periods = max(1.0, context_length / max(4, season_length))
    background_trend_scale = float(
        np.clip(0.012 * np.sqrt(max(trend_strength, 0.0)) * 7.0 / context_periods, 0.0, 0.025)
    )
    return {
        "noise_scale_multiplier": round_float(noise_multiplier),
        "noise_degrees_of_freedom": round_float(noise_df),
        "seasonal_amplitude_multiplier": round_float(seasonal_multiplier),
        "slow_amplitude_multiplier": round_float(np.clip(0.15 + 2.0 * residual_acf, 0.15, 1.25)),
        "background_trend_scale": round_float(background_trend_scale),
        "local_amplitude_multiplier": 1.0,
        "residual_ar_phi": round_float(np.clip(1.15 * residual_acf, 0.0, 0.65)),
    }


def calibrate_capability_conditioning(
    *,
    spec: Any,
    capability_id: str,
    profile_nuisance: dict[str, float],
    real_feature_summary: dict[str, dict[str, float]],
    target_feature_targets: dict[str, list[float]],
    sample_count: int,
    seed: int,
) -> tuple[dict[str, float], list[float], dict[str, Any]]:
    capability_nuisance = derive_capability_nuisance(
        capability_id,
        profile_nuisance,
        real_feature_summary,
    )
    primary_feature = PRIMARY_TARGET_FEATURE[capability_id]
    desired = target_feature_targets.get(primary_feature)
    if desired is None:
        values = finite_summary_target(real_feature_summary, primary_feature)
        desired = [values] * 5
    target_scale = max(
        real_feature_summary.get(primary_feature, {}).get("iqr", 0.05),
        0.01,
    )
    scale_results: dict[float, float] = {}
    for structure_scale in structure_scale_grid(capability_id):
        generated = simulate_feature_medians(
            spec=spec,
            capability_id=capability_id,
            parameters={**profile_nuisance, **capability_nuisance, "structure_scale": structure_scale},
            intensity_lambda=1.0,
            feature_names=(primary_feature,),
            sample_count=sample_count,
            seed=_seed_for(seed, capability_id, 1),
        )
        scale_results[structure_scale] = generated[primary_feature]
    structure_scale = min(
        structure_scale_grid(capability_id),
        key=lambda value: abs(scale_results[value] - desired[-1]) / target_scale,
    )
    parameters = {**capability_nuisance, "structure_scale": float(structure_scale)}

    lambda_feature_values: dict[float, float] = {}
    for index, intensity_lambda in enumerate(LAMBDA_GRID):
        generated = simulate_feature_medians(
            spec=spec,
            capability_id=capability_id,
            parameters={**profile_nuisance, **parameters},
            intensity_lambda=intensity_lambda,
            feature_names=(primary_feature,),
            sample_count=sample_count,
            seed=_seed_for(seed, capability_id, 10_000),
        )
        lambda_feature_values[intensity_lambda] = generated[primary_feature]

    if capability_id == "predictable_intermittency":
        intensity_lambdas = [0.0, 0.25, 0.50, 0.75, 1.0]
    else:
        intensity_lambdas = [
            min(
                LAMBDA_GRID,
                key=lambda value: abs(lambda_feature_values[value] - target),
            )
            for target in desired
        ]
        intensity_lambdas = strictly_monotone_lambdas(intensity_lambdas)
    return (
        {name: round_float(value) for name, value in parameters.items()},
        [round_float(value) for value in intensity_lambdas],
        {
            "primary_target_feature": primary_feature,
            "target_values": desired,
            "structure_scale": round_float(structure_scale),
            "intensity_lambdas": [round_float(value) for value in intensity_lambdas],
            "lambda_grid_feature_medians": {
                str(round_float(key)): round_float(value)
                for key, value in lambda_feature_values.items()
            },
        },
    )


def derive_capability_nuisance(
    capability_id: str,
    profile_nuisance: dict[str, float],
    real_features: dict[str, dict[str, float]],
) -> dict[str, float]:
    overrides: dict[str, float] = {}
    seasonal_target = median_feature(real_features, "seasonal_strength", 0.85)
    noise_target = median_feature(real_features, "noise_ratio", 0.12)
    if capability_id == "trend":
        overrides["seasonal_amplitude_multiplier"] = ratio_multiplier(
            seasonal_target,
            baseline=0.872,
            lower=0.40,
            upper=2.50,
        )
        overrides["noise_scale_multiplier"] = ratio_multiplier(
            noise_target,
            baseline=0.117,
            lower=0.30,
            upper=2.50,
        )
    elif capability_id == "nonlinear_persistence":
        overrides["seasonal_amplitude_multiplier"] = ratio_multiplier(
            seasonal_target,
            baseline=0.614,
            lower=0.50,
            upper=3.00,
        )
        overrides["noise_scale_multiplier"] = ratio_multiplier(
            noise_target,
            baseline=0.378,
            lower=0.25,
            upper=2.00,
        )
    elif capability_id == "hierarchical_coherence":
        overrides.update(
            {
                "noise_scale_multiplier": 0.75,
                "noise_degrees_of_freedom": 8.0,
            }
        )
    return {name: round_float(value) for name, value in overrides.items()}


def ratio_multiplier(
    target: float,
    *,
    baseline: float,
    lower: float,
    upper: float,
) -> float:
    target_odds = max(target, 1e-4) / max(1.0 - target, 1e-4)
    baseline_odds = max(baseline, 1e-4) / max(1.0 - baseline, 1e-4)
    return float(np.clip(np.sqrt(target_odds / baseline_odds), lower, upper))


def structure_scale_grid(capability_id: str) -> tuple[float, ...]:
    if capability_id == "nonlinear_persistence":
        return tuple(value for value in STRUCTURE_SCALE_GRID if value <= 1.0)
    return STRUCTURE_SCALE_GRID


def simulate_feature_medians(
    *,
    spec: Any,
    capability_id: str,
    parameters: dict[str, float],
    intensity_lambda: float,
    feature_names: tuple[str, ...],
    sample_count: int,
    seed: int,
) -> dict[str, float]:
    conditioning = GeneratorConditioning(
        profile_id=spec.profile_id,
        capability_id=capability_id,
        context_length=int(spec.context_length),
        horizon=int(spec.horizon),
        target_dim=int(spec.target_dim),
        season_length=int(spec.season_length),
        frequency=profile_frequency(spec.profile_id),
        parameters=parameters,
        intensity_lambdas=(intensity_lambda,) * 5,
        target_percentile_levels=TARGET_PERCENTILE_LEVELS,
        target_feature_targets={},
        artifact_schema_version="calibration-in-memory",
        artifact_created_at=None,
        calibration_method="in-memory grid",
    )
    rows: list[dict[str, float]] = []
    length = int(spec.context_length + spec.horizon)
    for sample_index in range(sample_count):
        rng = np.random.default_rng(_seed_for(seed, capability_id, sample_index))
        target, _latent, covariates = _generate_sample_values(
            capability_id,
            length,
            int(spec.context_length),
            int(spec.target_dim),
            int(spec.season_length),
            3,
            rng,
            generator_conditioning=conditioning,
        )
        target = (
            _standardize_hierarchy_by_context(target, int(spec.context_length))
            if capability_id == "hierarchical_coherence"
            else _standardize_by_context(target, int(spec.context_length))
        )
        if covariates is not None and covariates.size:
            covariates = _normalize_covariates(covariates, int(spec.context_length))
        rows.append(
            _realized_features(
                target,
                covariates,
                int(spec.season_length),
                int(spec.context_length),
            )
        )
    return {
        name: float(np.median([row[name] for row in rows]))
        for name in feature_names
    }


def strictly_monotone_lambdas(values: list[float], minimum_gap: float = 0.05) -> list[float]:
    result = [float(np.clip(value, 0.0, 1.0)) for value in values]
    for index in range(1, len(result)):
        result[index] = max(result[index], result[index - 1] + minimum_gap)
    if result[-1] > 1.0:
        result[-1] = 1.0
    for index in range(len(result) - 2, -1, -1):
        result[index] = min(result[index], result[index + 1] - minimum_gap)
    if result[0] < 0.0:
        result[0] = 0.0
        for index in range(1, len(result)):
            result[index] = max(result[index], result[index - 1] + minimum_gap)
    return [float(np.clip(value, 0.0, 1.0)) for value in result]


def finite_values(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = np.asarray(
        [row.get("features", {}).get(name, float("nan")) for row in rows],
        dtype=float,
    )
    return values[np.isfinite(values)]


def quantiles_for_levels(values: np.ndarray, levels: tuple[float, ...]) -> list[float]:
    return [round_float(value) for value in np.quantile(values, levels)]


def median_feature(features: dict[str, dict[str, float]], name: str, default: float) -> float:
    return float(features.get(name, {}).get("p50", default))


def finite_summary_target(features: dict[str, dict[str, float]], name: str) -> float:
    return float(features.get(name, {}).get("p50", 0.0))


def profile_frequency(profile_id: str) -> str:
    return "d" if profile_id.startswith("m5_daily_") else "h"


def round_float(value: float, digits: int = 8) -> float:
    return float(round(float(value), digits))


if __name__ == "__main__":
    raise SystemExit(main())
