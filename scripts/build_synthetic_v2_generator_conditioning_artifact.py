#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import dataclass
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
DEFAULT_CALIBRATION_SAMPLES = 64
TARGET_PERCENTILE_LEVELS = (0.20, 0.35, 0.50, 0.70, 0.90)
CAPABILITY_REFERENCE_PERCENTILE_LEVELS = {
    "nonlinear_persistence": (0.35, 0.50, 0.60, 0.75, 0.90),
}
STRUCTURE_SCALE_GRID = (
    0.0025,
    0.005,
    0.006,
    0.0075,
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.375,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    1.0,
    1.25,
    1.50,
    2.0,
    2.50,
    3.0,
    4.0,
)
LAMBDA_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
CANONICAL_CALIBRATION_TOLERANCE = 0.20
CANONICAL_SCALE_ID = "synthetic-v2-paper-v1-development-2026-07"
CANONICAL_REFERENCE_CORPUS_ROLE = (
    "development calibration only; freeze before model evaluation and keep external "
    "dataset-level validation splits out of scale fitting"
)
CANONICAL_REFERENCE_PROFILE_IDS = (
    "m4_hourly_daily_168ctx",
    "electricity_hourly_daily_168ctx",
    "traffic_hourly_daily_168ctx",
    "electricity_hourly_panel_168ctx",
    "traffic_hourly_panel_168ctx",
    "m5_daily_covariate_365ctx_28h",
    "m5_daily_hierarchy_365ctx_28h",
    "gefcom2014_load_hourly_covariate_168ctx_24h",
)
CONDITIONING_PROFILE_IDS = (
    *CANONICAL_REFERENCE_PROFILE_IDS,
    "electricity_hourly_daily_2048ctx_24h",
)
PRIMARY_TARGET_FEATURE = {
    "trend": "trend_strength",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "change_point_shift_energy",
    "nonlinear_persistence": "nonlinear_multi_lag_gain",
    "predictable_intermittency": "spike_rate",
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    "covariate_response": "covariate_incremental_r2",
}


@dataclass(frozen=True)
class ProfileCalibrationInput:
    spec: Any
    parameter_window_count: int
    split_summary: dict[str, Any]
    real_feature_summary: dict[str, dict[str, float]]
    profile_nuisance: dict[str, float]
    local_target_quantiles: dict[str, dict[str, list[float]]]
    primary_values: dict[str, np.ndarray]


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
    specs_by_id = {spec.profile_id: spec for spec in FEATURE_GATE_BUCKET_SPECS}
    missing_reference_profiles = sorted(set(CONDITIONING_PROFILE_IDS) - set(specs_by_id))
    if missing_reference_profiles:
        raise ValueError(
            "conditioning profiles are not registered: "
            + ", ".join(missing_reference_profiles)
        )
    selected = [
        specs_by_id[profile_id]
        for profile_id in CONDITIONING_PROFILE_IDS
        if bucket_ids is None or profile_id in bucket_ids
    ]
    if not selected:
        raise ValueError("no bucket specs selected")
    unknown_bucket_ids = sorted(set(bucket_ids or ()) - set(CONDITIONING_PROFILE_IDS))
    if unknown_bucket_ids:
        raise ValueError("unknown conditioning bucket ids: " + ", ".join(unknown_bucket_ids))
    if calibration_samples < 4:
        raise ValueError("calibration_samples must be at least 4")

    created_at = datetime.now(timezone.utc).isoformat()
    calibration_inputs: dict[str, ProfileCalibrationInput] = {}
    for profile_id in CONDITIONING_PROFILE_IDS:
        spec = specs_by_id[profile_id]
        rows = load_real_bucket(spec, data_dir / spec.asset_name, max_windows=max_windows)
        parameter_rows, _gate_reference, _gate_calibration, split_summary = split_real_rows_three_way(
            rows,
            spec,
            calibration_fraction=calibration_fraction,
            gate_reference_fraction=gate_reference_fraction,
            seed=_seed_for(seed, spec.profile_id, 0),
        )
        real_feature_summary = summarize_real_features(parameter_rows)
        profile_nuisance = derive_profile_nuisance(
            real_feature_summary,
            spec.context_length,
            spec.season_length,
        )
        local_target_quantiles: dict[str, dict[str, list[float]]] = {}
        primary_values: dict[str, np.ndarray] = {}
        for capability_id in spec.synthetic_capabilities:
            percentile_levels = reference_percentile_levels(capability_id)
            local_target_quantiles[capability_id] = {
                name: quantiles_for_levels(
                    finite_values(parameter_rows, name),
                    percentile_levels,
                )
                for name in TARGET_FEATURES_BY_CAPABILITY[capability_id]
                if finite_values(parameter_rows, name).size
            }
            primary_feature = PRIMARY_TARGET_FEATURE[capability_id]
            values = finite_values(parameter_rows, primary_feature)
            if not values.size:
                raise ValueError(
                    f"{spec.profile_id}/{capability_id} has no finite {primary_feature} values"
                )
            primary_values[capability_id] = values
        calibration_inputs[spec.profile_id] = ProfileCalibrationInput(
            spec=spec,
            parameter_window_count=len(parameter_rows),
            split_summary=split_summary,
            real_feature_summary=real_feature_summary,
            profile_nuisance=profile_nuisance,
            local_target_quantiles=local_target_quantiles,
            primary_values=primary_values,
        )
        del rows, parameter_rows
        gc.collect()

    canonical_definitions = derive_canonical_target_definitions(calibration_inputs)
    scale_fingerprint = canonical_scale_fingerprint(canonical_definitions)
    profiles: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    unsupported_cells: list[str] = []
    for spec in selected:
        calibration_input = calibration_inputs[spec.profile_id]
        real_feature_summary = calibration_input.real_feature_summary
        profile_nuisance = calibration_input.profile_nuisance
        capability_configs: dict[str, dict[str, Any]] = {}
        capability_summaries: dict[str, dict[str, Any]] = {}
        for capability_id in spec.synthetic_capabilities:
            canonical_definition = canonical_definitions[capability_id]
            canonical_target_values = canonical_definition["target_values"]
            local_real_percentiles = empirical_percentiles(
                calibration_input.primary_values[capability_id],
                canonical_target_values,
            )
            parameters, intensity_lambdas, calibration_summary = calibrate_capability_conditioning(
                spec=spec,
                capability_id=capability_id,
                profile_nuisance=profile_nuisance,
                real_feature_summary=real_feature_summary,
                canonical_target_values=canonical_target_values,
                sample_count=calibration_samples,
                seed=_seed_for(seed, spec.profile_id, len(capability_configs) + 1),
            )
            if calibration_summary["status"] != "supported":
                unsupported_cells.append(
                    f"{spec.profile_id}/{capability_id}"
                    f"(max_error={calibration_summary['max_normalized_error']},"
                    f"monotone={calibration_summary['monotone_realized']})"
                )
            capability_configs[capability_id] = {
                "parameters": parameters,
                "intensity_lambdas": intensity_lambdas,
                "canonical_reference_percentile_levels": canonical_definition[
                    "reference_percentile_levels"
                ],
                "canonical_target_feature": canonical_definition["primary_feature"],
                "canonical_target_values": canonical_target_values,
                "calibrated_realized_strengths": calibration_summary["realized_values"],
                "local_real_percentiles_at_canonical_targets": local_real_percentiles,
                "local_real_target_quantiles": calibration_input.local_target_quantiles[
                    capability_id
                ],
                "canonical_calibration": {
                    key: value
                    for key, value in calibration_summary.items()
                    if key
                    in {
                        "status",
                        "normalized_absolute_errors",
                        "max_normalized_error",
                        "tolerance",
                        "target_scale",
                        "fit_sample_count",
                        "validation_sample_count",
                        "validation_seed_is_independent",
                    }
                },
                "calibration_method": "capability-global canonical target inverse calibration",
            }
            capability_summaries[capability_id] = {
                **calibration_summary,
                "local_real_percentiles_at_canonical_targets": local_real_percentiles,
            }

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
            "split": calibration_input.split_summary,
            "capabilities": capability_configs,
        }
        summaries.append(
            {
                "profile_id": spec.profile_id,
                "parameter_window_count": calibration_input.parameter_window_count,
                "nuisance_parameters": profile_nuisance,
                "capabilities": capability_summaries,
            }
        )

    if unsupported_cells:
        raise ValueError(
            "canonical intensity calibration is unsupported for: " + ", ".join(unsupported_cells)
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
        "intensity_policy": (
            "capability-global canonical realized-strength targets with profile-specific inverse maps"
        ),
        "canonical_reference_profile_ids": list(CANONICAL_REFERENCE_PROFILE_IDS),
        "conditioning_profile_ids": list(CONDITIONING_PROFILE_IDS),
        "canonical_profile_weighting": "equal profile weight",
        "canonical_scale_id": CANONICAL_SCALE_ID,
        "canonical_scale_fingerprint": scale_fingerprint,
        "canonical_reference_corpus_role": CANONICAL_REFERENCE_CORPUS_ROLE,
        "canonical_scale_change_policy": (
            "adding/removing reference profiles or changing target curves requires a new scale_id"
        ),
    }
    artifact = {
        "schema_version": "synthetic_v2_generator_conditioning_artifact.v2",
        "created_at": created_at,
        "config": config,
        "canonical_intensity": {
            "scale_id": CANONICAL_SCALE_ID,
            "scale_fingerprint": scale_fingerprint,
            "reference_corpus_role": CANONICAL_REFERENCE_CORPUS_ROLE,
            "policy": (
                "coordinate-wise median of equal-profile local quantile curves, constrained to "
                "capability-observable reference levels"
            ),
            "default_reference_percentile_levels": list(TARGET_PERCENTILE_LEVELS),
            "capabilities": canonical_definitions,
        },
        "profiles": profiles,
    }
    summary = {
        "schema_version": "synthetic_v2_generator_conditioning_calibration.v2",
        "created_at": created_at,
        "config": config,
        "canonical_intensity": artifact["canonical_intensity"],
        "profiles": summaries,
    }
    return artifact, summary


def canonical_scale_fingerprint(
    canonical_definitions: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "scale_id": CANONICAL_SCALE_ID,
        "reference_profile_ids": list(CANONICAL_REFERENCE_PROFILE_IDS),
        "capabilities": canonical_definitions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def derive_canonical_target_definitions(
    calibration_inputs: dict[str, ProfileCalibrationInput],
) -> dict[str, dict[str, Any]]:
    profile_curves: dict[str, list[tuple[str, list[float]]]] = {}
    for profile_id in CANONICAL_REFERENCE_PROFILE_IDS:
        calibration_input = calibration_inputs[profile_id]
        for capability_id in calibration_input.spec.synthetic_capabilities:
            primary_feature = PRIMARY_TARGET_FEATURE[capability_id]
            curve = calibration_input.local_target_quantiles[capability_id].get(primary_feature)
            if curve is None or len(curve) != len(reference_percentile_levels(capability_id)):
                raise ValueError(
                    f"{profile_id}/{capability_id} has no complete local quantile curve for "
                    f"{primary_feature}"
                )
            profile_curves.setdefault(capability_id, []).append((profile_id, curve))

    definitions: dict[str, dict[str, Any]] = {}
    for capability_id, curves in sorted(profile_curves.items()):
        matrix = np.asarray([curve for _profile_id, curve in curves], dtype=float)
        target_values = np.median(matrix, axis=0)
        target_values = np.maximum.accumulate(target_values)
        percentile_levels = reference_percentile_levels(capability_id)
        definitions[capability_id] = {
            "primary_feature": PRIMARY_TARGET_FEATURE[capability_id],
            "target_values": [round_float(value) for value in target_values],
            "reference_percentile_levels": list(percentile_levels),
            "contributing_profile_ids": [profile_id for profile_id, _curve in curves],
            "profile_weighting": "equal",
            "aggregation": "coordinate-wise median of local parameter-split quantile curves",
        }
    return definitions


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
    canonical_target_values: list[float],
    sample_count: int,
    seed: int,
) -> tuple[dict[str, float], list[float], dict[str, Any]]:
    capability_nuisance = derive_capability_nuisance(
        capability_id,
        profile_nuisance,
        real_feature_summary,
    )
    primary_feature = PRIMARY_TARGET_FEATURE[capability_id]
    desired = [float(value) for value in canonical_target_values]
    if len(desired) != 5 or any(
        right < left for left, right in zip(desired, desired[1:])
    ):
        raise ValueError(f"invalid canonical targets for {capability_id}: {desired}")
    target_scale = max(desired[-1] - desired[0], 0.05)
    fit_seed = _seed_for(seed, capability_id, 10_000)
    validation_seed = _seed_for(seed, capability_id, 20_000)
    validation_sample_count = max(sample_count, 32)
    scale_results: dict[float, float] = {}
    for structure_scale in structure_scale_grid(capability_id):
        generated = simulate_feature_medians(
            spec=spec,
            capability_id=capability_id,
            parameters={**profile_nuisance, **capability_nuisance, "structure_scale": structure_scale},
            intensity_lambda=1.0,
            feature_names=(primary_feature,),
            sample_count=sample_count,
            seed=fit_seed,
        )
        scale_results[structure_scale] = generated[primary_feature]
    structure_scale = min(
        structure_scale_grid(capability_id),
        key=lambda value: abs(scale_results[value] - desired[-1]) / target_scale,
    )
    parameters = {**capability_nuisance, "structure_scale": float(structure_scale)}

    lambda_feature_values: dict[float, float] = {}
    for intensity_lambda in LAMBDA_GRID:
        generated = simulate_feature_medians(
            spec=spec,
            capability_id=capability_id,
            parameters={**profile_nuisance, **parameters},
            intensity_lambda=intensity_lambda,
            feature_names=(primary_feature,),
            sample_count=sample_count,
            seed=fit_seed,
        )
        lambda_feature_values[intensity_lambda] = generated[primary_feature]

    intensity_lambdas = invert_monotone_feature_curve(
        lambda_feature_values,
        desired,
    )
    intensity_lambdas = strictly_monotone_lambdas(intensity_lambdas)
    realized_values = [
        simulate_feature_medians(
            spec=spec,
            capability_id=capability_id,
            parameters={**profile_nuisance, **parameters},
            intensity_lambda=intensity_lambda,
            feature_names=(primary_feature,),
            sample_count=validation_sample_count,
            seed=validation_seed,
        )[primary_feature]
        for intensity_lambda in intensity_lambdas
    ]
    normalized_errors = [
        abs(realized - target) / target_scale
        for realized, target in zip(realized_values, desired)
    ]
    monotone_realized = all(
        right >= left for left, right in zip(realized_values, realized_values[1:])
    )
    max_normalized_error = max(normalized_errors)
    status = (
        "supported"
        if monotone_realized and max_normalized_error <= CANONICAL_CALIBRATION_TOLERANCE
        else "unsupported"
    )
    return (
        {name: round_float(value) for name, value in parameters.items()},
        [round_float(value) for value in intensity_lambdas],
        {
            "status": status,
            "primary_target_feature": primary_feature,
            "canonical_target_values": [round_float(value) for value in desired],
            "realized_values": [round_float(value) for value in realized_values],
            "normalized_absolute_errors": [round_float(value) for value in normalized_errors],
            "max_normalized_error": round_float(max_normalized_error),
            "tolerance": CANONICAL_CALIBRATION_TOLERANCE,
            "target_scale": round_float(target_scale),
            "fit_sample_count": int(sample_count),
            "validation_sample_count": int(validation_sample_count),
            "validation_seed_is_independent": True,
            "monotone_realized": monotone_realized,
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
        canonical_reference_percentile_levels=reference_percentile_levels(capability_id),
        canonical_target_feature=PRIMARY_TARGET_FEATURE[capability_id],
        canonical_target_values=(0.0,) * 5,
        calibrated_realized_strengths=(0.0,) * 5,
        local_real_percentiles=(0.0,) * 5,
        local_real_target_quantiles={},
        calibration_max_normalized_error=0.0,
        canonical_scale_id=CANONICAL_SCALE_ID,
        canonical_scale_fingerprint="calibration-in-progress",
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


def strictly_monotone_lambdas(values: list[float], minimum_gap: float = 0.01) -> list[float]:
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


def invert_monotone_feature_curve(
    feature_values_by_lambda: dict[float, float],
    targets: list[float],
) -> list[float]:
    ordered = sorted((float(lam), float(value)) for lam, value in feature_values_by_lambda.items())
    if len(ordered) < 2:
        raise ValueError("at least two lambda grid points are required")
    lambdas = np.asarray([item[0] for item in ordered], dtype=float)
    feature_values = np.maximum.accumulate(
        np.asarray([item[1] for item in ordered], dtype=float)
    )
    inverted: list[float] = []
    for target in targets:
        if target <= feature_values[0]:
            inverted.append(float(lambdas[0]))
            continue
        if target >= feature_values[-1]:
            inverted.append(float(lambdas[-1]))
            continue
        right_index = int(np.searchsorted(feature_values, target, side="left"))
        left_index = right_index - 1
        left_value = float(feature_values[left_index])
        right_value = float(feature_values[right_index])
        if right_value <= left_value + 1e-12:
            inverted.append(float(lambdas[right_index]))
            continue
        fraction = (float(target) - left_value) / (right_value - left_value)
        inverted.append(
            float(lambdas[left_index] + fraction * (lambdas[right_index] - lambdas[left_index]))
        )
    return inverted


def finite_values(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = np.asarray(
        [row.get("features", {}).get(name, float("nan")) for row in rows],
        dtype=float,
    )
    return values[np.isfinite(values)]


def quantiles_for_levels(values: np.ndarray, levels: tuple[float, ...]) -> list[float]:
    return [round_float(value) for value in np.quantile(values, levels)]


def reference_percentile_levels(capability_id: str) -> tuple[float, ...]:
    return CAPABILITY_REFERENCE_PERCENTILE_LEVELS.get(
        capability_id,
        TARGET_PERCENTILE_LEVELS,
    )


def empirical_percentiles(values: np.ndarray, targets: list[float]) -> list[float]:
    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not finite.size:
        raise ValueError("cannot compute local real percentiles from an empty sample")
    return [
        round_float(np.searchsorted(finite, target, side="right") / finite.size)
        for target in targets
    ]


def median_feature(features: dict[str, dict[str, float]], name: str, default: float) -> float:
    return float(features.get(name, {}).get("p50", default))


def profile_frequency(profile_id: str) -> str:
    return "d" if profile_id.startswith("m5_daily_") else "h"


def round_float(value: float, digits: int = 8) -> float:
    return float(round(float(value), digits))


if __name__ == "__main__":
    raise SystemExit(main())
