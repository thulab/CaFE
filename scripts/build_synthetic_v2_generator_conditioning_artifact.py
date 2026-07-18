#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
    PAPER_GENERATOR_VERSION,
    TARGET_FEATURES_BY_CAPABILITY,
    _generate_sample_values,
    _normalize_covariates,
    _realized_features,
    _regime_clock_history_incremental_r2,
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
from run_synthetic_v2_near_distance_calibration import BucketSpec, load_real_bucket  # noqa: E402
from synthetic_capability_qualification import regime_clock_features  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "runtime/research"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "backend/app/data/synthetic_v2_generator_conditioning_artifact.json"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "runtime/research/synthetic-v2-generator-conditioning/summary.json"
DEFAULT_SEED = 20260715
DEFAULT_CALIBRATION_SAMPLES = 64
FIT_SEED_BANK_COUNT = 2
HIGH_VARIANCE_FIT_SEED_BANK_COUNT = 4
HIGH_VARIANCE_CAPABILITY_IDS = frozenset(
    {"nonlinear_persistence", "covariate_response"}
)
TARGET_PERCENTILE_LEVELS = (0.10, 0.30, 0.50, 0.70, 0.90)
INTENSITY_POLICY_ID = "dataset-local-relative-quantiles-v1"
LOCAL_CALIBRATION_TOLERANCE = 0.20
LOCAL_MIN_TARGET_RANGE = 1e-6
LOCAL_MIN_ADJACENT_GAP_FRACTION = 0.02
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
MIN_REGIME_QUALIFIED_PARAMETER_WINDOWS = 30
MIN_REGIME_QUALIFIED_RATE = 0.10
CONDITIONING_PROFILE_IDS = (
    "m4_hourly_daily_168ctx",
    "electricity_hourly_daily_168ctx",
    "traffic_hourly_daily_168ctx",
    "electricity_hourly_panel_168ctx",
    "traffic_hourly_panel_168ctx",
    "m5_daily_covariate_365ctx_28h",
    "m5_daily_hierarchy_365ctx_28h",
    "gefcom2014_load_hourly_covariate_168ctx_24h",
    "electricity_hourly_daily_2048ctx_24h",
)
RESEARCH_ONLY_CONDITIONING_PROFILE_IDS = (
    "electricity_hourly_daily_2048ctx_24h",
)
ONLINE_CONDITIONING_PROFILE_IDS = tuple(
    profile_id
    for profile_id in CONDITIONING_PROFILE_IDS
    if profile_id not in RESEARCH_ONLY_CONDITIONING_PROFILE_IDS
)
DATASET_ID_BY_PROFILE_ID = {
    "m4_hourly_daily_168ctx": "m4_hourly",
    "electricity_hourly_daily_168ctx": "electricity",
    "traffic_hourly_daily_168ctx": "traffic",
    "electricity_hourly_panel_168ctx": "electricity",
    "traffic_hourly_panel_168ctx": "traffic",
    "m5_daily_covariate_365ctx_28h": "m5",
    "m5_daily_hierarchy_365ctx_28h": "m5",
    "gefcom2014_load_hourly_covariate_168ctx_24h": "gefcom2014_load",
    "electricity_hourly_daily_2048ctx_24h": "electricity",
}
EXPECTED_ASSET_IDENTITIES = {
    "m5_sha256": "0349ba38a2efd30d0f5acc6394c1110e140e1a990c650d7b5ca44c5b25dd12f5",
    "gefcom2014_sha256": "d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154",
    "m4_hourly_sha256": "18085bd3c34e41cdc07441aa61c5610dac9e916b9489a6a381f8e89fd01c8a66",
    "electricity_hourly_sha256": "eff447075dde68dca0105ab7e2851c5637967ae3bb21556fd8b931f196d5968c",
    "traffic_hourly_sha256": "3db12ba866a9c9d3c8109b7b6d189a990c38d0e5002fa2617022157358d08299",
}
PRIMARY_TARGET_FEATURE = {
    "trend": "trend_strength",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "regime_clock_history_incremental_r2",
    "nonlinear_persistence": "nonlinear_conditional_gain",
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
    capability_parameter_counts: dict[str, int]
    capability_qualification_summaries: dict[str, dict[str, Any]]


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
    parser.add_argument("--skip-asset-identity-check", action="store_true")
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
        validate_asset_identities=not args.skip_asset_identity_check,
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
    validate_asset_identities: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    specs_by_id = {spec.profile_id: spec for spec in FEATURE_GATE_BUCKET_SPECS}
    missing_profiles = sorted(set(CONDITIONING_PROFILE_IDS) - set(specs_by_id))
    if missing_profiles:
        raise ValueError(
            "conditioning profiles are not registered: "
            + ", ".join(missing_profiles)
        )
    asset_identities = resolve_asset_identities(data_dir=data_dir)
    if validate_asset_identities:
        assert_expected_asset_identities(asset_identities)
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
    input_profile_ids = tuple(spec.profile_id for spec in selected)
    for profile_id in input_profile_ids:
        spec = specs_by_id[profile_id]
        asset_path = resolve_profile_asset_path(
            spec,
            data_dir=data_dir,
        )
        rows = load_real_bucket(spec, asset_path, max_windows=max_windows)
        parameter_rows, _gate_reference, _gate_calibration, split_summary = split_real_rows_three_way(
            rows,
            spec,
            calibration_fraction=calibration_fraction,
            gate_reference_fraction=gate_reference_fraction,
            seed=_seed_for(seed, spec.profile_id, 0),
        )
        excluded_test_tail_steps = sorted(
            {
                int(row["source_tail_excluded_steps"])
                for row in rows
                if row.get("source_tail_excluded_steps") is not None
            }
        )
        if len(excluded_test_tail_steps) > 1:
            raise ValueError(
                f"{spec.profile_id} has inconsistent source test-tail exclusions: "
                f"{excluded_test_tail_steps}"
            )
        if excluded_test_tail_steps:
            split_summary = {
                **split_summary,
                "source_test_tail_excluded_steps": excluded_test_tail_steps[0],
                "source_test_tail_policy": (
                    "GIFT-Eval short-term official test windows excluded before windowing"
                ),
            }
        real_feature_summary = summarize_real_features(parameter_rows)
        profile_nuisance = derive_profile_nuisance(
            real_feature_summary,
            spec.context_length,
            spec.season_length,
        )
        local_target_quantiles: dict[str, dict[str, list[float]]] = {}
        primary_values: dict[str, np.ndarray] = {}
        capability_parameter_counts: dict[str, int] = {}
        capability_qualification_summaries: dict[str, dict[str, Any]] = {}
        for capability_id in spec.synthetic_capabilities:
            capability_rows = parameter_rows
            if capability_id == "regime_switching":
                annotated_rows, regime_audits = annotate_regime_clock_rows(
                    parameter_rows,
                    spec,
                )
                try:
                    capability_rows, qualification_summary = (
                        qualify_regime_reference_rows(
                            annotated_rows,
                            spec,
                            audits=regime_audits,
                        )
                    )
                    capability_qualification_summaries[capability_id] = {
                        "status": "supported",
                        **qualification_summary,
                    }
                except ValueError as error:
                    capability_rows = []
                    capability_qualification_summaries[capability_id] = {
                        "status": "unsupported",
                        "reason": "insufficient_dataset_local_recurring_regime_structure",
                        "detail": str(error),
                    }
            capability_parameter_counts[capability_id] = len(capability_rows)
            percentile_levels = reference_percentile_levels(capability_id)
            local_target_quantiles[capability_id] = {
                name: quantiles_for_levels(
                    finite_values(capability_rows, name),
                    percentile_levels,
                )
                for name in TARGET_FEATURES_BY_CAPABILITY[capability_id]
                if finite_values(capability_rows, name).size
            }
            primary_feature = PRIMARY_TARGET_FEATURE[capability_id]
            values = finite_values(capability_rows, primary_feature)
            primary_values[capability_id] = values
        calibration_inputs[spec.profile_id] = ProfileCalibrationInput(
            spec=spec,
            parameter_window_count=len(parameter_rows),
            split_summary=split_summary,
            real_feature_summary=real_feature_summary,
            profile_nuisance=profile_nuisance,
            local_target_quantiles=local_target_quantiles,
            primary_values=primary_values,
            capability_parameter_counts=capability_parameter_counts,
            capability_qualification_summaries=capability_qualification_summaries,
        )
        del rows, parameter_rows
        gc.collect()

    profiles: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    unsupported_cells: list[dict[str, Any]] = []
    for spec in selected:
        calibration_input = calibration_inputs[spec.profile_id]
        real_feature_summary = calibration_input.real_feature_summary
        profile_nuisance = calibration_input.profile_nuisance
        capability_configs: dict[str, dict[str, Any]] = {}
        capability_summaries: dict[str, dict[str, Any]] = {}
        for capability_id in spec.synthetic_capabilities:
            percentile_levels = reference_percentile_levels(capability_id)
            primary_feature = PRIMARY_TARGET_FEATURE[capability_id]
            target_values = calibration_input.local_target_quantiles.get(
                capability_id, {}
            ).get(primary_feature, [])
            structural = calibration_input.capability_qualification_summaries.get(
                capability_id,
                {"status": "supported"},
            )
            support = local_target_support(
                target_values,
                structural_status=structural,
            )
            base_config: dict[str, Any] = {
                "status": support["status"],
                "target_feature": primary_feature,
                "target_percentile_levels": list(percentile_levels),
                "target_values": target_values,
                "dataset_local_target_quantiles": calibration_input.local_target_quantiles.get(
                    capability_id, {}
                ),
                "structural_qualification": structural,
                "calibration_method": (
                    "dataset-local parameter-split quantile targets with "
                    "dataset-local inverse calibration"
                ),
            }
            if support["status"] != "supported":
                unsupported_reason = str(support["reason"])
                unsupported_cells.append(
                    {
                        "dataset_id": DATASET_ID_BY_PROFILE_ID[spec.profile_id],
                        "profile_id": spec.profile_id,
                        "capability_id": capability_id,
                        "reason": unsupported_reason,
                        "detail": support.get("detail"),
                    }
                )
                capability_configs[capability_id] = {
                    **base_config,
                    "unsupported_reason": unsupported_reason,
                    "unsupported_detail": support.get("detail"),
                }
                capability_summaries[capability_id] = capability_configs[
                    capability_id
                ]
                continue

            parameters, intensity_lambdas, calibration_summary = (
                calibrate_capability_conditioning(
                    spec=spec,
                    capability_id=capability_id,
                    profile_nuisance=profile_nuisance,
                    real_feature_summary=real_feature_summary,
                    target_values=target_values,
                    sample_count=calibration_samples,
                    seed=_seed_for(
                        seed,
                        spec.profile_id,
                        len(capability_configs) + 1,
                    ),
                )
            )
            status = str(calibration_summary["status"])
            capability_configs[capability_id] = {
                **base_config,
                "status": status,
                "parameters": parameters,
                "intensity_lambdas": intensity_lambdas,
                "calibrated_realized_strengths": calibration_summary[
                    "realized_values"
                ],
                "calibration": calibration_summary,
            }
            if status != "supported":
                capability_configs[capability_id].update(
                    {
                        "unsupported_reason": "inverse_calibration_failed",
                        "unsupported_detail": {
                            "max_normalized_error": calibration_summary[
                                "max_normalized_error"
                            ],
                            "monotone_realized": calibration_summary[
                                "monotone_realized"
                            ],
                        },
                    }
                )
            capability_summaries[capability_id] = capability_configs[
                capability_id
            ]
            if status != "supported":
                unsupported_cells.append(
                    {
                        "dataset_id": DATASET_ID_BY_PROFILE_ID[spec.profile_id],
                        "profile_id": spec.profile_id,
                        "capability_id": capability_id,
                        "reason": "inverse_calibration_failed",
                        "detail": {
                            "max_normalized_error": calibration_summary[
                                "max_normalized_error"
                            ],
                            "monotone_realized": calibration_summary[
                                "monotone_realized"
                            ],
                        },
                    }
                )

        frequency = profile_frequency(spec.profile_id)
        profiles[spec.profile_id] = {
            "profile_id": spec.profile_id,
            "dataset_id": DATASET_ID_BY_PROFILE_ID[spec.profile_id],
            "conditioning_role": (
                "research_only_pending_near_distance_gate"
                if spec.profile_id in RESEARCH_ONLY_CONDITIONING_PROFILE_IDS
                else "dataset_local_online"
            ),
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
                "dataset_id": DATASET_ID_BY_PROFILE_ID[spec.profile_id],
                "profile_id": spec.profile_id,
                "parameter_window_count": calibration_input.parameter_window_count,
                "nuisance_parameters": profile_nuisance,
                "capabilities": capability_summaries,
            }
        )

    try:
        data_dir_label = str(data_dir.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        data_dir_label = str(data_dir)
    config = {
        "generator_version": PAPER_GENERATOR_VERSION,
        "data_dir": data_dir_label,
        "max_windows_per_bucket": int(max_windows),
        "calibration_fraction": float(calibration_fraction),
        "gate_reference_fraction_of_development": float(gate_reference_fraction),
        "calibration_samples_per_grid_cell": int(calibration_samples),
        "calibration_fit_seed_banks_default": FIT_SEED_BANK_COUNT,
        "calibration_fit_samples_total_per_grid_cell_default": int(
            calibration_samples * FIT_SEED_BANK_COUNT
        ),
        "high_variance_calibration": {
            "capability_ids": sorted(HIGH_VARIANCE_CAPABILITY_IDS),
            "fit_seed_bank_count": HIGH_VARIANCE_FIT_SEED_BANK_COUNT,
            "fit_samples_per_seed_bank": max(calibration_samples, 128),
            "fit_samples_total_per_grid_cell": int(
                max(calibration_samples, 128)
                * HIGH_VARIANCE_FIT_SEED_BANK_COUNT
            ),
            "minimum_validation_sample_count": 1024,
        },
        "seed": int(seed),
        "split_policy": "generator parameters are fit only on the parameter split",
        "profile_selection_policy": (
            "explicit dataset profile; no cross-dataset pooling or fallback"
        ),
        "intensity_policy": (
            "dataset-bucket-local parameter-split quantiles; intensity is ordinal "
            "only within one dataset/profile/capability"
        ),
        "intensity_policy_id": INTENSITY_POLICY_ID,
        "target_percentile_levels": list(TARGET_PERCENTILE_LEVELS),
        "conditioning_profile_ids": list(CONDITIONING_PROFILE_IDS),
        "online_conditioning_profile_ids": list(ONLINE_CONDITIONING_PROFILE_IDS),
        "research_only_conditioning_profile_ids": list(
            RESEARCH_ONLY_CONDITIONING_PROFILE_IDS
        ),
        "research_only_conditioning_policy": (
            "inverse conditioning is retained for window-length research, but a profile is not "
            "paper-v2 online until feature-support and near-distance gates are both calibrated"
        ),
        "gift_eval_test_tail_policy": (
            "exclude prediction_length * windows using frozen short-term protocol before "
            "candidate window construction"
        ),
        "asset_identities": asset_identities,
        "unsupported_cells": unsupported_cells,
        "unsupported_policy": (
            "record unsupported dataset/profile/capability cells and exclude "
            "them from online generation without failing other datasets"
        ),
    }
    artifact = {
        "schema_version": "synthetic_v2_generator_conditioning_artifact.v4",
        "generator_version": PAPER_GENERATOR_VERSION,
        "created_at": created_at,
        "config": config,
        "intensity_policy": {
            "policy_id": INTENSITY_POLICY_ID,
            "percentile_levels": list(TARGET_PERCENTILE_LEVELS),
            "definition": (
                "intensity 1..5 denotes dataset/profile-local relative realized "
                "strength; target values are not absolute-comparable across datasets"
            ),
            "target_source": "the selected dataset bucket parameter split only",
        },
        "profiles": profiles,
    }
    summary = {
        "schema_version": "synthetic_v2_generator_conditioning_calibration.v4",
        "created_at": created_at,
        "config": config,
        "intensity_policy": artifact["intensity_policy"],
        "profiles": summaries,
    }
    return artifact, summary


def resolve_profile_asset_path(
    spec: BucketSpec,
    *,
    data_dir: Path,
) -> Path:
    if spec.profile_id not in CONDITIONING_PROFILE_IDS:
        raise ValueError(f"{spec.profile_id} is not a dataset-local conditioning profile")
    return data_dir / spec.asset_name


def resolve_asset_identities(
    *,
    data_dir: Path,
) -> dict[str, str]:
    required_files = {
        "m5_sha256": data_dir / "m5-forecasting-accuracy.zip",
        "gefcom2014_sha256": data_dir / "GEFCom2014.zip",
        "m4_hourly_sha256": data_dir / "m4_hourly_dataset.zip",
        "electricity_hourly_sha256": data_dir / "electricity_hourly_dataset.zip",
        "traffic_hourly_sha256": data_dir / "traffic_hourly_dataset.zip",
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "dataset-local conditioning assets are missing: " + ", ".join(missing)
        )
    return {name: sha256_file(path) for name, path in required_files.items()}


def assert_expected_asset_identities(actual: dict[str, str]) -> None:
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in EXPECTED_ASSET_IDENTITIES.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            "dataset-local conditioning asset identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_target_support(
    values: list[float],
    *,
    structural_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structural = structural_status or {"status": "supported"}
    if structural.get("status") != "supported":
        return {
            "status": "unsupported",
            "reason": str(structural.get("reason", "unsupported_dataset_structure")),
            "detail": structural.get("detail"),
        }
    target = np.asarray(values, dtype=float)
    if target.shape != (5,) or not np.isfinite(target).all():
        return {
            "status": "unsupported",
            "reason": "missing_or_nonfinite_local_target_quantiles",
            "detail": {"target_values": list(values)},
        }
    gaps = np.diff(target)
    target_range = float(target[-1] - target[0])
    if target_range < LOCAL_MIN_TARGET_RANGE:
        return {
            "status": "unsupported",
            "reason": "insufficient_local_target_range",
            "detail": {
                "target_range": round_float(target_range),
                "minimum": LOCAL_MIN_TARGET_RANGE,
            },
        }
    minimum_gap = max(
        LOCAL_MIN_TARGET_RANGE,
        LOCAL_MIN_ADJACENT_GAP_FRACTION * target_range,
    )
    if np.any(gaps < minimum_gap):
        return {
            "status": "unsupported",
            "reason": "insufficient_local_intensity_spacing",
            "detail": {
                "target_values": [round_float(value) for value in target],
                "adjacent_gaps": [round_float(value) for value in gaps],
                "minimum_gap": round_float(minimum_gap),
            },
        }
    return {
        "status": "supported",
        "target_range": round_float(target_range),
        "minimum_adjacent_gap": round_float(float(np.min(gaps))),
    }


def qualify_regime_reference_rows(
    rows: list[dict[str, Any]],
    spec: Any,
    *,
    audits: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if audits is None:
        rows, audits = annotate_regime_clock_rows(rows, spec)
    qualified_rows = [
        row
        for row, audit in zip(rows, audits, strict=True)
        if audit["qualified"]
    ]
    qualified_rate = len(qualified_rows) / max(len(rows), 1)
    if (
        len(qualified_rows) < MIN_REGIME_QUALIFIED_PARAMETER_WINDOWS
        or qualified_rate < MIN_REGIME_QUALIFIED_RATE
    ):
        raise ValueError(
            f"{spec.profile_id} has insufficient qualified recurring-regime parameter windows: "
            f"qualified={len(qualified_rows)}/{len(rows)}, rate={qualified_rate:.3f}"
        )
    metric_names = (
        "history_incremental_r2",
        "future_mse_gain",
        "amplitude_ratio",
        "history_level_shift_ratio",
        "future_level_shift_ratio",
        "history_direction_consistency",
        "future_direction_consistency",
        "history_state_coverage",
        "future_state_coverage",
        "context_absolute_skew",
    )
    qualified_audits = [audit for audit in audits if audit["qualified"]]
    selected_period_counts: dict[str, int] = {}
    for audit in qualified_audits:
        key = str(audit["selected_period"])
        selected_period_counts[key] = selected_period_counts.get(key, 0) + 1
    return qualified_rows, {
        "method": "history-selected recurring clock with untouched-future validation",
        "candidate_window_count": len(rows),
        "qualified_window_count": len(qualified_rows),
        "qualified_rate": round_float(qualified_rate),
        "minimum_qualified_window_count": MIN_REGIME_QUALIFIED_PARAMETER_WINDOWS,
        "minimum_qualified_rate": MIN_REGIME_QUALIFIED_RATE,
        "selected_period_counts": selected_period_counts,
        "qualified_metric_medians": {
            name: round_float(np.median([float(audit[name]) for audit in qualified_audits]))
            for name in metric_names
        },
        "thresholds": qualified_audits[0]["thresholds"],
    }


def annotate_regime_clock_rows(
    rows: list[dict[str, Any]],
    spec: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audits = [
        regime_clock_features(
            row["target"],
            context_length=int(spec.context_length),
            season_length=int(spec.season_length),
        )
        for row in rows
    ]
    annotated_rows = [
        {
            **row,
            "features": {
                **row.get("features", {}),
                "regime_clock_history_incremental_r2": float(
                    audit["history_incremental_r2"]
                ),
            },
        }
        for row, audit in zip(rows, audits, strict=True)
    ]
    return annotated_rows, audits


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
    target_values: list[float] | None,
    sample_count: int,
    seed: int,
    primary_feature: str | None = None,
    real_tolerance_bounds: tuple[float, float] | None = None,
    relative_dose_levels: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
) -> tuple[dict[str, float], list[float], dict[str, Any]]:
    capability_nuisance = derive_capability_nuisance(
        capability_id,
        profile_nuisance,
        real_feature_summary,
    )
    primary_feature = primary_feature or PRIMARY_TARGET_FEATURE[capability_id]
    if (target_values is None) == (real_tolerance_bounds is None):
        raise ValueError(
            "provide exactly one of target_values or real_tolerance_bounds"
        )
    if real_tolerance_bounds is not None:
        real_lower, real_upper = (
            float(real_tolerance_bounds[0]),
            float(real_tolerance_bounds[1]),
        )
        if (
            not math.isfinite(real_lower)
            or not math.isfinite(real_upper)
            or real_upper <= real_lower
        ):
            raise ValueError(
                f"invalid real tolerance bounds for {capability_id}: "
                f"{real_tolerance_bounds}"
            )
        if (
            len(relative_dose_levels) != 5
            or relative_dose_levels[0] != 0.0
            or relative_dose_levels[-1] != 1.0
            or any(
                right <= left
                for left, right in zip(
                    relative_dose_levels,
                    relative_dose_levels[1:],
                )
            )
        ):
            raise ValueError(
                f"invalid relative dose levels: {relative_dose_levels}"
            )
        desired: list[float] = []
        upper_scale_target = real_upper
        target_selection_method = (
            "five evenly spaced relative doses inside the intersection of the "
            "dataset-local real tolerance interval and generator response range"
        )
    else:
        desired = [float(value) for value in target_values or ()]
        if len(desired) != 5 or any(
            right < left for left, right in zip(desired, desired[1:])
        ):
            raise ValueError(
                f"invalid dataset-local targets for {capability_id}: {desired}"
            )
        real_lower = float("nan")
        real_upper = float("nan")
        upper_scale_target = desired[-1]
        target_selection_method = "exact dataset-local target values"
    high_variance_calibration = (
        capability_id in HIGH_VARIANCE_CAPABILITY_IDS
    )
    fit_seed_bank_count = (
        HIGH_VARIANCE_FIT_SEED_BANK_COUNT
        if high_variance_calibration
        else FIT_SEED_BANK_COUNT
    )
    fit_samples_per_seed_bank = (
        max(sample_count, 128)
        if high_variance_calibration
        else sample_count
    )
    fit_seeds = tuple(
        _seed_for(seed, capability_id, 10_000 + bank_index)
        for bank_index in range(fit_seed_bank_count)
    )
    validation_seed = _seed_for(seed, capability_id, 20_000)
    validation_sample_count = max(
        fit_samples_per_seed_bank * fit_seed_bank_count,
        1024 if high_variance_calibration else 256,
    )
    scale_results: dict[float, float] = {}
    for structure_scale in structure_scale_grid(capability_id):
        scale_results[structure_scale] = mean_feature_over_seed_banks(
            spec=spec,
            capability_id=capability_id,
            parameters={
                **profile_nuisance,
                **capability_nuisance,
                "structure_scale": structure_scale,
            },
            intensity_lambda=1.0,
            feature_name=primary_feature,
            sample_count=fit_samples_per_seed_bank,
            seeds=fit_seeds,
        )
    structure_scale = invert_monotone_feature_curve(
        scale_results,
        [upper_scale_target],
    )[0]
    parameters = {**capability_nuisance, "structure_scale": float(structure_scale)}

    lambda_feature_values: dict[float, float] = {}
    for intensity_lambda in LAMBDA_GRID:
        lambda_feature_values[intensity_lambda] = mean_feature_over_seed_banks(
            spec=spec,
            capability_id=capability_id,
            parameters={**profile_nuisance, **parameters},
            intensity_lambda=intensity_lambda,
            feature_name=primary_feature,
            sample_count=fit_samples_per_seed_bank,
            seeds=fit_seeds,
        )

    generator_lower = float(lambda_feature_values[min(lambda_feature_values)])
    generator_upper = float(
        np.maximum.accumulate(
            np.asarray(
                [
                    lambda_feature_values[key]
                    for key in sorted(lambda_feature_values)
                ],
                dtype=float,
            )
        )[-1]
    )
    if real_tolerance_bounds is not None:
        feasible_lower = max(real_lower, generator_lower)
        feasible_upper = min(real_upper, generator_upper)
        magnitude = max(abs(feasible_lower), abs(feasible_upper), 1.0)
        minimum_span = LOCAL_MIN_TARGET_RANGE * magnitude
        if feasible_upper - feasible_lower <= minimum_span:
            return (
                {name: round_float(value) for name, value in parameters.items()},
                [],
                {
                    "status": "unsupported",
                    "reason_code": "no_real_generator_tolerance_overlap",
                    "primary_target_feature": primary_feature,
                    "target_values": [],
                    "realized_values": [],
                    "real_tolerance_lower": round_float(real_lower),
                    "real_tolerance_upper": round_float(real_upper),
                    "generator_response_lower": round_float(generator_lower),
                    "generator_response_upper": round_float(generator_upper),
                    "feasible_lower": round_float(feasible_lower),
                    "feasible_upper": round_float(feasible_upper),
                    "minimum_feasible_span": round_float(minimum_span),
                    "structure_scale": round_float(structure_scale),
                    "lambda_grid_feature_means": {
                        str(round_float(key)): round_float(value)
                        for key, value in lambda_feature_values.items()
                    },
                    "structure_scale_grid_feature_means": {
                        str(round_float(key)): round_float(value)
                        for key, value in scale_results.items()
                    },
                },
            )
        desired = [
            feasible_lower + float(level) * (feasible_upper - feasible_lower)
            for level in relative_dose_levels
        ]

    target_scale_floor = (
        0.005 if primary_feature == "nonlinear_conditional_gain" else 0.05
    )
    target_scale = max(desired[-1] - desired[0], target_scale_floor)
    intensity_lambdas = invert_monotone_feature_curve(
        lambda_feature_values,
        desired,
    )
    intensity_lambdas = strictly_monotone_lambdas(intensity_lambdas)
    realized_values = [
        simulate_feature_means(
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
        if monotone_realized and max_normalized_error <= LOCAL_CALIBRATION_TOLERANCE
        else "unsupported"
    )
    return (
        {name: round_float(value) for name, value in parameters.items()},
        [round_float(value) for value in intensity_lambdas],
        {
            "status": status,
            "primary_target_feature": primary_feature,
            "target_values": [round_float(value) for value in desired],
            "target_selection_method": target_selection_method,
            "relative_dose_levels": [
                round_float(value) for value in relative_dose_levels
            ],
            "real_tolerance_lower": (
                round_float(real_lower)
                if real_tolerance_bounds is not None
                else None
            ),
            "real_tolerance_upper": (
                round_float(real_upper)
                if real_tolerance_bounds is not None
                else None
            ),
            "generator_response_lower": round_float(generator_lower),
            "generator_response_upper": round_float(generator_upper),
            "feasible_lower": (
                round_float(desired[0])
                if real_tolerance_bounds is not None
                else None
            ),
            "feasible_upper": (
                round_float(desired[-1])
                if real_tolerance_bounds is not None
                else None
            ),
            "realized_values": [round_float(value) for value in realized_values],
            "normalized_absolute_errors": [round_float(value) for value in normalized_errors],
            "max_normalized_error": round_float(max_normalized_error),
            "tolerance": LOCAL_CALIBRATION_TOLERANCE,
            "target_scale": round_float(target_scale),
            "fit_sample_count": int(
                fit_samples_per_seed_bank * fit_seed_bank_count
            ),
            "fit_seed_bank_count": fit_seed_bank_count,
            "fit_samples_per_seed_bank": int(fit_samples_per_seed_bank),
            "validation_sample_count": int(validation_sample_count),
            "validation_seed_is_independent": True,
            "monotone_realized": monotone_realized,
            "structure_scale": round_float(structure_scale),
            "intensity_lambdas": [round_float(value) for value in intensity_lambdas],
            "lambda_grid_feature_means": {
                str(round_float(key)): round_float(value)
                for key, value in lambda_feature_values.items()
            },
            "structure_scale_grid_feature_means": {
                str(round_float(key)): round_float(value)
                for key, value in scale_results.items()
            },
        },
    )


def mean_feature_over_seed_banks(
    *,
    spec: Any,
    capability_id: str,
    parameters: dict[str, float],
    intensity_lambda: float,
    feature_name: str,
    sample_count: int,
    seeds: tuple[int, ...],
) -> float:
    values = [
        simulate_feature_means(
            spec=spec,
            capability_id=capability_id,
            parameters=parameters,
            intensity_lambda=intensity_lambda,
            feature_names=(feature_name,),
            sample_count=sample_count,
            seed=seed,
        )[feature_name]
        for seed in seeds
    ]
    return float(np.mean(values))


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
        overrides.update(
            {
                "seasonal_amplitude_multiplier": 1.0,
                "noise_scale_multiplier": 1.0,
                "noise_degrees_of_freedom": 0.0,
                "nonlinear_transform_version": 2.0,
            }
        )
    elif capability_id == "predictable_intermittency":
        trend_target = median_feature(real_features, "trend_strength", 0.05)
        trend_odds = max(trend_target, 1e-4) / max(
            1.0 - trend_target,
            1e-4,
        )
        overrides["background_trend_scale"] = float(
            np.clip(0.0215 * np.sqrt(trend_odds), 0.001, 0.15)
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


def simulate_feature_means(
    *,
    spec: Any,
    capability_id: str,
    parameters: dict[str, float],
    intensity_lambda: float,
    feature_names: tuple[str, ...],
    sample_count: int,
    seed: int,
) -> dict[str, float]:
    feature_measurement_horizon = int(
        getattr(spec, "feature_measurement_horizon", spec.horizon)
    )
    if not 1 <= feature_measurement_horizon <= int(spec.horizon):
        raise ValueError(
            f"{spec.profile_id} has invalid feature_measurement_horizon="
            f"{feature_measurement_horizon} for horizon={spec.horizon}"
        )
    conditioning = GeneratorConditioning(
        profile_id=spec.profile_id,
        dataset_id=DATASET_ID_BY_PROFILE_ID.get(spec.profile_id, spec.profile_id),
        capability_id=capability_id,
        context_length=int(spec.context_length),
        horizon=int(spec.horizon),
        target_dim=int(spec.target_dim),
        season_length=int(spec.season_length),
        frequency=str(
            getattr(spec, "frequency", profile_frequency(spec.profile_id))
        ),
        parameters=parameters,
        intensity_lambdas=(intensity_lambda,) * 5,
        target_percentile_levels=reference_percentile_levels(capability_id),
        target_feature=PRIMARY_TARGET_FEATURE[capability_id],
        target_values=(0.0, 0.25, 0.5, 0.75, 1.0),
        calibrated_realized_strengths=(0.0, 0.25, 0.5, 0.75, 1.0),
        calibration_max_normalized_error=0.0,
        intensity_policy_id=INTENSITY_POLICY_ID,
        artifact_schema_version="calibration-in-memory",
        artifact_created_at=None,
        calibration_method="in-memory grid",
    )
    rows: list[dict[str, float]] = []
    length = int(spec.context_length + spec.horizon)
    for sample_index in range(sample_count):
        rng = np.random.default_rng(_seed_for(seed, capability_id, sample_index))
        target, latent, covariates = _generate_sample_values(
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
        measurement_end = int(spec.context_length) + feature_measurement_horizon
        measurement_target = target[:measurement_end]
        measurement_covariates = (
            covariates[:measurement_end] if covariates is not None else None
        )
        features = _realized_features(
            measurement_target,
            measurement_covariates,
            int(spec.season_length),
            int(spec.context_length),
        )
        if capability_id == "regime_switching":
            features["regime_clock_history_incremental_r2"] = (
                _regime_clock_history_incremental_r2(
                    measurement_target,
                    context_length=int(spec.context_length),
                    season_length=int(spec.season_length),
                    cut_points=latent["cut_points"],
                    dwell_length=int(latent["dwell_length"]),
                )
            )
        rows.append(features)
    return {name: float(np.mean([row[name] for row in rows])) for name in feature_names}


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


def reference_percentile_levels(_capability_id: str) -> tuple[float, ...]:
    return TARGET_PERCENTILE_LEVELS


def median_feature(features: dict[str, dict[str, float]], name: str, default: float) -> float:
    return float(features.get(name, {}).get("p50", default))


def profile_frequency(profile_id: str) -> str:
    if profile_id.startswith("uci_hydraulic_"):
        return "s"
    if profile_id.startswith("skchange_hvac_"):
        return "10min"
    if profile_id.startswith("m5_daily_"):
        return "d"
    if profile_id.startswith("gift_hospital_monthly_") or profile_id.startswith(
        "nixtla_"
    ):
        return "m"
    return "h"


def round_float(value: float, digits: int = 8) -> float:
    return float(round(float(value), digits))


if __name__ == "__main__":
    raise SystemExit(main())
