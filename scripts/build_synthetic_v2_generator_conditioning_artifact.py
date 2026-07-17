#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
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
from run_synthetic_v2_near_distance_calibration import BucketSpec, load_real_bucket  # noqa: E402
from synthetic_capability_qualification import regime_clock_features  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "runtime/research"
DEFAULT_GIFT_EVAL_DIR = Path.home() / "xmy/gift-eval"
DEFAULT_GIFT_EVAL_CODE_DIR = Path.home() / "xmy/gift-eval-code"
DEFAULT_NIXTLA_HIERARCHY_ZIP = Path.home() / "xmy/reference-data/nixtla-hierarchical/datasets.zip"
DEFAULT_UCI_HYDRAULIC_ZIP = (
    Path.home() / "xmy/reference-data/uci-condition-monitoring-hydraulic-systems.zip"
)
DEFAULT_SKCHANGE_DIR = Path.home() / "xmy/reference-data/skchange"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "backend/app/data/synthetic_v2_generator_conditioning_artifact.json"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "runtime/research/synthetic-v2-generator-conditioning/summary.json"
DEFAULT_SEED = 20260715
DEFAULT_CALIBRATION_SAMPLES = 64
FIT_SEED_BANK_COUNT = 2
TARGET_PERCENTILE_LEVELS = (0.20, 0.35, 0.50, 0.70, 0.90)
CAPABILITY_REFERENCE_PERCENTILE_LEVELS = {
    # Below the real median, multi-lag nonlinear gain is not distinguishable
    # from the estimator floor induced by seasonal nuisance in every profile.
    "nonlinear_persistence": (0.55, 0.625, 0.70, 0.80, 0.90),
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
CANONICAL_MIN_ADJACENT_GAP_FRACTION = 0.10
CANONICAL_SCALE_ID = "synthetic-v2-paper-v1-frozen-2026-07-16"
CANONICAL_REFERENCE_CORPUS_ROLE = (
    "paper-v1 frozen development calibration only; external dataset-family validation "
    "and all GIFT-Eval official test windows are excluded from scale fitting"
)
MIN_REGIME_QUALIFIED_PARAMETER_WINDOWS = 30
MIN_REGIME_QUALIFIED_RATE = 0.10
PAPER_UNIVARIATE_CAPABILITY_IDS = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
)
CANONICAL_ONLY_BUCKET_SPECS = (
    BucketSpec(
        "gift_hospital_monthly_60ctx_12h",
        "gift_univariate",
        "hospital",
        60,
        12,
        12,
        12,
        synthetic_capabilities=PAPER_UNIVARIATE_CAPABILITY_IDS,
    ),
    BucketSpec(
        "gift_jena_weather_hourly_168ctx_24h",
        "gift_univariate",
        "jena_weather/H",
        168,
        24,
        24,
        24,
        synthetic_capabilities=PAPER_UNIVARIATE_CAPABILITY_IDS,
    ),
    BucketSpec(
        "gift_bizitobs_l2c_hourly_168ctx_24h",
        "gift_univariate",
        "bizitobs_l2c/H",
        168,
        24,
        24,
        24,
        synthetic_capabilities=PAPER_UNIVARIATE_CAPABILITY_IDS,
    ),
    BucketSpec(
        "gift_jena_weather_hourly_panel_168ctx",
        "gift_panel",
        "jena_weather/H",
        168,
        24,
        24,
        24,
        target_dim=3,
        synthetic_capabilities=("common_factor",),
    ),
    BucketSpec(
        "gift_bizitobs_l2c_hourly_panel_168ctx",
        "gift_panel",
        "bizitobs_l2c/H",
        168,
        24,
        12,
        24,
        target_dim=3,
        synthetic_capabilities=("common_factor",),
    ),
    BucketSpec(
        "nixtla_labour_monthly_hierarchy_60ctx_12h",
        "nixtla_binary_hierarchy",
        "Labour",
        60,
        12,
        12,
        12,
        target_dim=3,
        hierarchy="additive_first",
        max_groups=16,
        synthetic_capabilities=("hierarchical_coherence",),
    ),
    BucketSpec(
        "nixtla_tourism_large_monthly_hierarchy_60ctx_12h",
        "nixtla_binary_hierarchy",
        "TourismLarge",
        60,
        12,
        12,
        12,
        target_dim=3,
        hierarchy="additive_first",
        max_groups=24,
        synthetic_capabilities=("hierarchical_coherence",),
    ),
    BucketSpec(
        "gefcom2014_solar_hourly_covariate_168ctx_24h",
        "gefcom2014_solar",
        "GEFCom2014.zip",
        168,
        24,
        24,
        24,
        covariate_dim=12,
        task=1,
        synthetic_capabilities=("covariate_response",),
    ),
    BucketSpec(
        "uci_hydraulic_eps1_420ctx_60h",
        "uci_hydraulic_cycle",
        "uci-condition-monitoring-hydraulic-systems.zip:EPS1",
        420,
        60,
        60,
        60,
        synthetic_capabilities=("regime_switching",),
    ),
    BucketSpec(
        "skchange_hvac_unit0_504ctx_144h",
        "skchange_hvac",
        "skchange/datasets/data/hvac_system/data.csv:unit=0",
        504,
        144,
        8,
        144,
        task=0,
        synthetic_capabilities=("regime_switching",),
    ),
)
GENERIC_UNIVARIATE_REFERENCE_PROFILE_IDS = (
    "m4_hourly_daily_168ctx",
    "electricity_hourly_daily_168ctx",
    "traffic_hourly_daily_168ctx",
    "gift_hospital_monthly_60ctx_12h",
    "gift_jena_weather_hourly_168ctx_24h",
    "gift_bizitobs_l2c_hourly_168ctx_24h",
)
CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY = {
    "trend": GENERIC_UNIVARIATE_REFERENCE_PROFILE_IDS,
    "multi_seasonal": GENERIC_UNIVARIATE_REFERENCE_PROFILE_IDS,
    "time_varying_seasonality": GENERIC_UNIVARIATE_REFERENCE_PROFILE_IDS,
    "regime_switching": (
        "uci_hydraulic_eps1_420ctx_60h",
        "skchange_hvac_unit0_504ctx_144h",
    ),
    "nonlinear_persistence": GENERIC_UNIVARIATE_REFERENCE_PROFILE_IDS,
    "predictable_intermittency": GENERIC_UNIVARIATE_REFERENCE_PROFILE_IDS,
    "common_factor": (
        "electricity_hourly_panel_168ctx",
        "traffic_hourly_panel_168ctx",
        "gift_jena_weather_hourly_panel_168ctx",
        "gift_bizitobs_l2c_hourly_panel_168ctx",
    ),
    "hierarchical_coherence": (
        "m5_daily_hierarchy_365ctx_28h",
        "nixtla_labour_monthly_hierarchy_60ctx_12h",
        "nixtla_tourism_large_monthly_hierarchy_60ctx_12h",
    ),
    "covariate_response": (
        "m5_daily_covariate_365ctx_28h",
        "gefcom2014_load_hourly_covariate_168ctx_24h",
        "gefcom2014_solar_hourly_covariate_168ctx_24h",
    ),
}
CANONICAL_REFERENCE_PROFILE_IDS = tuple(
    dict.fromkeys(
        profile_id
        for profile_ids in CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY.values()
        for profile_id in profile_ids
    )
)
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
GIFT_EVAL_HELD_OUT_FAMILIES = (
    "solar",
    "covid_deaths",
    "kdd_cup_2018_with_missing",
    "restaurant",
    "hierarchical_sales",
    "LOOP_SEATTLE",
    "SZ_TAXI",
    "M_DENSE",
    "ett1",
    "ett2",
    "bitbrains_fast_storage",
    "bitbrains_rnd",
    "bizitobs_application",
    "bizitobs_service",
)
EXPECTED_ASSET_IDENTITIES = {
    "gift_eval_arrow_manifest_sha256": "0f410dd0eadce583886e7141e556f3a40c069472ad6a1b6c3bd1663d5860c120",
    "gift_eval_protocol_git_commit": "6fdb10df9c17411f0aef5ff862afbec23627c12f",
    "nixtla_hierarchy_sha256": "6512d9aa80f111ee26480bc6f3f4eb3b5655d4ceecc384933100edc85adf704b",
    "m5_sha256": "0349ba38a2efd30d0f5acc6394c1110e140e1a990c650d7b5ca44c5b25dd12f5",
    "gefcom2014_sha256": "d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154",
    "m4_hourly_sha256": "18085bd3c34e41cdc07441aa61c5610dac9e916b9489a6a381f8e89fd01c8a66",
    "electricity_hourly_sha256": "eff447075dde68dca0105ab7e2851c5637967ae3bb21556fd8b931f196d5968c",
    "traffic_hourly_sha256": "3db12ba866a9c9d3c8109b7b6d189a990c38d0e5002fa2617022157358d08299",
    "uci_hydraulic_sha256": "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2",
    "skchange_hvac_csv_sha256": "1da08ee5922db6d4d6f4ab32a0e6a9666fc41680ed75dcffe53ac9e1819fff99",
    "skchange_git_commit": "f209def94199607b11b1ae9b3108d80e3e87e624",
}
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
    capability_parameter_counts: dict[str, int]
    capability_qualification_summaries: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit real-profile-conditioned nuisance parameters and intensity mappings."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gift-eval-dir", type=Path, default=DEFAULT_GIFT_EVAL_DIR)
    parser.add_argument("--gift-eval-code-dir", type=Path, default=DEFAULT_GIFT_EVAL_CODE_DIR)
    parser.add_argument(
        "--nixtla-hierarchy-zip",
        type=Path,
        default=DEFAULT_NIXTLA_HIERARCHY_ZIP,
    )
    parser.add_argument("--uci-hydraulic-zip", type=Path, default=DEFAULT_UCI_HYDRAULIC_ZIP)
    parser.add_argument("--skchange-dir", type=Path, default=DEFAULT_SKCHANGE_DIR)
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
        gift_eval_dir=args.gift_eval_dir,
        gift_eval_code_dir=args.gift_eval_code_dir,
        nixtla_hierarchy_zip=args.nixtla_hierarchy_zip,
        uci_hydraulic_zip=args.uci_hydraulic_zip,
        skchange_dir=args.skchange_dir,
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
    gift_eval_dir: Path = DEFAULT_GIFT_EVAL_DIR,
    gift_eval_code_dir: Path = DEFAULT_GIFT_EVAL_CODE_DIR,
    nixtla_hierarchy_zip: Path = DEFAULT_NIXTLA_HIERARCHY_ZIP,
    uci_hydraulic_zip: Path = DEFAULT_UCI_HYDRAULIC_ZIP,
    skchange_dir: Path = DEFAULT_SKCHANGE_DIR,
    max_windows: int,
    calibration_fraction: float,
    gate_reference_fraction: float,
    calibration_samples: int,
    seed: int,
    bucket_ids: tuple[str, ...] | None = None,
    validate_asset_identities: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_specs = (*FEATURE_GATE_BUCKET_SPECS, *CANONICAL_ONLY_BUCKET_SPECS)
    specs_by_id = {spec.profile_id: spec for spec in all_specs}
    required_profile_ids = (*CANONICAL_REFERENCE_PROFILE_IDS, *CONDITIONING_PROFILE_IDS)
    missing_reference_profiles = sorted(set(required_profile_ids) - set(specs_by_id))
    if missing_reference_profiles:
        raise ValueError(
            "canonical or conditioning profiles are not registered: "
            + ", ".join(missing_reference_profiles)
        )
    asset_identities = resolve_asset_identities(
        data_dir=data_dir,
        gift_eval_dir=gift_eval_dir,
        gift_eval_code_dir=gift_eval_code_dir,
        nixtla_hierarchy_zip=nixtla_hierarchy_zip,
        uci_hydraulic_zip=uci_hydraulic_zip,
        skchange_dir=skchange_dir,
    )
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
    input_profile_ids = tuple(dict.fromkeys(required_profile_ids))
    for profile_id in input_profile_ids:
        spec = specs_by_id[profile_id]
        asset_path = resolve_profile_asset_path(
            spec,
            data_dir=data_dir,
            gift_eval_dir=gift_eval_dir,
            nixtla_hierarchy_zip=nixtla_hierarchy_zip,
            uci_hydraulic_zip=uci_hydraulic_zip,
            skchange_dir=skchange_dir,
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
            if (
                capability_id == "regime_switching"
                and spec.profile_id
                in CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY["regime_switching"]
            ):
                capability_rows, qualification_summary = qualify_regime_reference_rows(
                    parameter_rows,
                    spec,
                )
                capability_qualification_summaries[capability_id] = qualification_summary
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
            capability_parameter_counts=capability_parameter_counts,
            capability_qualification_summaries=capability_qualification_summaries,
        )
        del rows, parameter_rows
        gc.collect()

    canonical_definitions = derive_canonical_target_definitions(calibration_inputs)
    scale_fingerprint = canonical_scale_fingerprint(
        canonical_definitions,
        asset_identities=asset_identities,
    )
    canonical_reference_summaries = [
        canonical_reference_summary(calibration_inputs[profile_id])
        for profile_id in CANONICAL_REFERENCE_PROFILE_IDS
    ]
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
                "canonical_raw_reference_quantile_values": canonical_definition[
                    "raw_reference_quantile_values"
                ],
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
                        "fit_seed_bank_count",
                        "fit_samples_per_seed_bank",
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
            "conditioning_role": (
                "research_only_pending_near_distance_gate"
                if spec.profile_id in RESEARCH_ONLY_CONDITIONING_PROFILE_IDS
                else "paper_v1_online"
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
        "calibration_fit_seed_banks": FIT_SEED_BANK_COUNT,
        "calibration_fit_samples_total_per_grid_cell": int(
            calibration_samples * FIT_SEED_BANK_COUNT
        ),
        "seed": int(seed),
        "split_policy": "generator parameters are fit only on the parameter split",
        "profile_selection_policy": "balanced uniform over exact task/window profiles",
        "intensity_policy": (
            "capability-global canonical realized-strength targets with endpoint-preserving 10% "
            "minimum adjacent resolution and profile-specific inverse maps"
        ),
        "canonical_reference_profile_ids": list(CANONICAL_REFERENCE_PROFILE_IDS),
        "canonical_reference_profile_ids_by_capability": {
            capability_id: list(profile_ids)
            for capability_id, profile_ids in CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY.items()
        },
        "conditioning_profile_ids": list(CONDITIONING_PROFILE_IDS),
        "online_conditioning_profile_ids": list(ONLINE_CONDITIONING_PROFILE_IDS),
        "research_only_conditioning_profile_ids": list(
            RESEARCH_ONLY_CONDITIONING_PROFILE_IDS
        ),
        "research_only_conditioning_policy": (
            "inverse conditioning is retained for window-length research, but a profile is not "
            "paper-v1 online until feature-support and near-distance gates are both calibrated"
        ),
        "canonical_profile_weighting": "equal profile weight",
        "canonical_scale_id": CANONICAL_SCALE_ID,
        "canonical_scale_fingerprint": scale_fingerprint,
        "canonical_reference_corpus_role": CANONICAL_REFERENCE_CORPUS_ROLE,
        "gift_eval_test_tail_policy": (
            "exclude prediction_length * windows using frozen short-term protocol before "
            "candidate window construction"
        ),
        "canonical_reference_asset_identities": asset_identities,
        "gift_eval_held_out_families": list(GIFT_EVAL_HELD_OUT_FAMILIES),
        "other_held_out_sources": {
            "nixtla_hierarchy": ["Traffic", "Wiki2", "TourismSmall"],
            "gefcom2014": ["Wind"],
            "skchange_hvac": ["unit_1"],
        },
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
                "coordinate-wise median of equal-profile local quantile curves, followed by an "
                "endpoint-preserving 10%-of-range adjacent-resolution projection"
            ),
            "default_reference_percentile_levels": list(TARGET_PERCENTILE_LEVELS),
            "asset_identities": asset_identities,
            "reference_qualification": {
                profile_id: calibration_inputs[
                    profile_id
                ].capability_qualification_summaries
                for profile_id in CANONICAL_REFERENCE_PROFILE_IDS
                if calibration_inputs[profile_id].capability_qualification_summaries
            },
            "reference_preprocessing": {
                "uci_hydraulic_eps1_420ctx_60h": (
                    "block-average each official 60-second EPS1 cycle from 100 Hz to 1 Hz, "
                    "then concatenate cycles in source order"
                ),
                "skchange_hvac_unit0_504ctx_144h": (
                    "regularize unit 0 to a 10-minute grid and linearly interpolate short gaps "
                    "only when total missingness is at most 1%"
                ),
            },
            "held_out": {
                "gift_eval_families": list(GIFT_EVAL_HELD_OUT_FAMILIES),
                "nixtla_hierarchy": ["Traffic", "Wiki2", "TourismSmall"],
                "gefcom2014": ["Wind"],
                "skchange_hvac": ["unit_1"],
            },
            "capabilities": canonical_definitions,
        },
        "profiles": profiles,
    }
    summary = {
        "schema_version": "synthetic_v2_generator_conditioning_calibration.v2",
        "created_at": created_at,
        "config": config,
        "canonical_intensity": artifact["canonical_intensity"],
        "canonical_reference_profiles": canonical_reference_summaries,
        "profiles": summaries,
    }
    return artifact, summary


def resolve_profile_asset_path(
    spec: BucketSpec,
    *,
    data_dir: Path,
    gift_eval_dir: Path,
    nixtla_hierarchy_zip: Path,
    uci_hydraulic_zip: Path,
    skchange_dir: Path,
) -> Path:
    if spec.kind in {"gift_univariate", "gift_panel"}:
        return gift_eval_dir / spec.asset_name
    if spec.kind == "nixtla_binary_hierarchy":
        return nixtla_hierarchy_zip
    if spec.kind == "uci_hydraulic_cycle":
        return uci_hydraulic_zip
    if spec.kind == "skchange_hvac":
        return skchange_dir
    return data_dir / spec.asset_name


def resolve_asset_identities(
    *,
    data_dir: Path,
    gift_eval_dir: Path,
    gift_eval_code_dir: Path,
    nixtla_hierarchy_zip: Path,
    uci_hydraulic_zip: Path,
    skchange_dir: Path,
) -> dict[str, str]:
    skchange_hvac_csv = skchange_dir / "skchange/datasets/data/hvac_system/data.csv"
    required_files = {
        "nixtla_hierarchy_sha256": nixtla_hierarchy_zip,
        "m5_sha256": data_dir / "m5-forecasting-accuracy.zip",
        "gefcom2014_sha256": data_dir / "GEFCom2014.zip",
        "m4_hourly_sha256": data_dir / "m4_hourly_dataset.zip",
        "electricity_hourly_sha256": data_dir / "electricity_hourly_dataset.zip",
        "traffic_hourly_sha256": data_dir / "traffic_hourly_dataset.zip",
        "uci_hydraulic_sha256": uci_hydraulic_zip,
        "skchange_hvac_csv_sha256": skchange_hvac_csv,
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("canonical reference assets are missing: " + ", ".join(missing))
    if not gift_eval_dir.is_dir():
        raise FileNotFoundError(f"GIFT-Eval Arrow directory not found: {gift_eval_dir}")
    if not gift_eval_code_dir.is_dir():
        raise FileNotFoundError(f"GIFT-Eval protocol repository not found: {gift_eval_code_dir}")
    if not skchange_dir.is_dir():
        raise FileNotFoundError(f"skchange repository not found: {skchange_dir}")
    identities = {
        "gift_eval_arrow_manifest_sha256": gift_eval_arrow_manifest_sha256(gift_eval_dir),
        "gift_eval_protocol_git_commit": git_head(gift_eval_code_dir),
        "skchange_git_commit": git_head(skchange_dir),
    }
    identities.update({name: sha256_file(path) for name, path in required_files.items()})
    return identities


def assert_expected_asset_identities(actual: dict[str, str]) -> None:
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in EXPECTED_ASSET_IDENTITIES.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            "canonical reference asset identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gift_eval_arrow_manifest_sha256(root: Path) -> str:
    names = {"dataset_info.json", "state.json"}
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name in names or (path.name.startswith("data-") and path.suffix == ".arrow"))
    )
    if not paths:
        raise ValueError(f"GIFT-Eval Arrow manifest is empty: {root}")
    manifest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        manifest.update(f"{sha256_file(path)}  ./{relative}\n".encode("utf-8"))
    return manifest.hexdigest()


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def canonical_reference_summary(calibration_input: ProfileCalibrationInput) -> dict[str, Any]:
    spec = calibration_input.spec
    return {
        "profile_id": spec.profile_id,
        "source_kind": spec.kind,
        "source_asset": spec.asset_name,
        "role": (
            "canonical_and_conditioning"
            if spec.profile_id in CONDITIONING_PROFILE_IDS
            else "canonical_only"
        ),
        "context_length": int(spec.context_length),
        "horizon": int(spec.horizon),
        "season_length": int(spec.season_length),
        "target_dim": int(spec.target_dim),
        "covariate_dim": int(spec.covariate_dim),
        "parameter_window_count": calibration_input.parameter_window_count,
        "capability_parameter_counts": calibration_input.capability_parameter_counts,
        "capability_qualification": calibration_input.capability_qualification_summaries,
        "contributed_capabilities": [
            capability_id
            for capability_id, profile_ids in CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY.items()
            if spec.profile_id in profile_ids
        ],
        "split": calibration_input.split_summary,
        "local_target_quantiles": calibration_input.local_target_quantiles,
        "real_feature_summary": calibration_input.real_feature_summary,
    }


def canonical_scale_fingerprint(
    canonical_definitions: dict[str, dict[str, Any]],
    *,
    asset_identities: dict[str, str] | None = None,
) -> str:
    payload = {
        "scale_id": CANONICAL_SCALE_ID,
        "reference_profile_ids": list(CANONICAL_REFERENCE_PROFILE_IDS),
        "reference_profile_ids_by_capability": {
            capability_id: list(profile_ids)
            for capability_id, profile_ids in CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY.items()
        },
        "asset_identities": asset_identities or EXPECTED_ASSET_IDENTITIES,
        "capabilities": canonical_definitions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def derive_canonical_target_definitions(
    calibration_inputs: dict[str, ProfileCalibrationInput],
) -> dict[str, dict[str, Any]]:
    profile_curves: dict[str, list[tuple[str, list[float]]]] = {}
    for capability_id, profile_ids in CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY.items():
        for profile_id in profile_ids:
            calibration_input = calibration_inputs[profile_id]
            if capability_id not in calibration_input.spec.synthetic_capabilities:
                raise ValueError(
                    f"{profile_id} does not declare canonical capability {capability_id}"
                )
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
        minimum_profile_count = 2 if capability_id == "regime_switching" else 3
        if len(curves) < minimum_profile_count:
            raise ValueError(
                f"{capability_id} needs at least {minimum_profile_count} canonical profiles, "
                f"got {len(curves)}"
            )
        matrix = np.asarray([curve for _profile_id, curve in curves], dtype=float)
        raw_target_values = np.maximum.accumulate(np.median(matrix, axis=0))
        target_values = enforce_target_resolution(raw_target_values)
        if any(right <= left for left, right in zip(target_values, target_values[1:])):
            raise ValueError(
                f"{capability_id} canonical target curve is not strictly increasing: "
                f"{target_values.tolist()}"
            )
        percentile_levels = reference_percentile_levels(capability_id)
        definitions[capability_id] = {
            "primary_feature": PRIMARY_TARGET_FEATURE[capability_id],
            "target_values": [round_float(value) for value in target_values],
            "raw_reference_quantile_values": [
                round_float(value) for value in raw_target_values
            ],
            "reference_percentile_levels": list(percentile_levels),
            "contributing_profile_ids": [profile_id for profile_id, _curve in curves],
            "contributing_parameter_window_counts": {
                profile_id: calibration_inputs[profile_id].capability_parameter_counts[
                    capability_id
                ]
                for profile_id, _curve in curves
            },
            "profile_weighting": "equal",
            "aggregation": (
                "coordinate-wise median of local parameter-split quantile curves, then "
                "endpoint-preserving minimum-gap projection"
            ),
            "target_resolution": {
                "minimum_adjacent_gap_fraction_of_raw_range": (
                    CANONICAL_MIN_ADJACENT_GAP_FRACTION
                ),
                "applied": bool(
                    not np.allclose(target_values, raw_target_values, rtol=0.0, atol=1e-12)
                ),
            },
        }
    return definitions


def enforce_target_resolution(values: np.ndarray) -> np.ndarray:
    resolved = np.asarray(values, dtype=float).copy()
    if resolved.shape != (5,) or not np.isfinite(resolved).all():
        raise ValueError(f"canonical target resolution expects five finite values: {resolved}")
    if np.any(np.diff(resolved) < 0.0):
        raise ValueError(f"canonical target resolution expects a monotone curve: {resolved}")
    raw_start = float(resolved[0])
    raw_end = float(resolved[-1])
    raw_range = raw_end - raw_start
    if raw_range <= 0.0:
        raise ValueError(f"canonical target curve has no usable range: {resolved}")
    minimum_gap = CANONICAL_MIN_ADJACENT_GAP_FRACTION * raw_range
    for index in range(1, len(resolved)):
        resolved[index] = max(resolved[index], resolved[index - 1] + minimum_gap)
    if resolved[-1] > raw_end:
        resolved[-1] = raw_end
        for index in range(len(resolved) - 2, -1, -1):
            resolved[index] = min(resolved[index], resolved[index + 1] - minimum_gap)
    resolved[0] = raw_start
    resolved[-1] = raw_end
    if np.any(np.diff(resolved) < minimum_gap - 1e-12):
        raise ValueError(f"failed to enforce canonical target resolution: {resolved}")
    return resolved


def qualify_regime_reference_rows(
    rows: list[dict[str, Any]],
    spec: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audits = [
        regime_clock_features(
            row["target"],
            context_length=int(spec.context_length),
            season_length=int(spec.season_length),
        )
        for row in rows
    ]
    qualified_rows = [row for row, audit in zip(rows, audits, strict=True) if audit["qualified"]]
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
    primary_feature: str | None = None,
) -> tuple[dict[str, float], list[float], dict[str, Any]]:
    capability_nuisance = derive_capability_nuisance(
        capability_id,
        profile_nuisance,
        real_feature_summary,
    )
    primary_feature = primary_feature or PRIMARY_TARGET_FEATURE[capability_id]
    desired = [float(value) for value in canonical_target_values]
    if len(desired) != 5 or any(
        right < left for left, right in zip(desired, desired[1:])
    ):
        raise ValueError(f"invalid canonical targets for {capability_id}: {desired}")
    target_scale_floor = (
        0.005 if primary_feature == "nonlinear_conditional_gain" else 0.05
    )
    target_scale = max(desired[-1] - desired[0], target_scale_floor)
    fit_seeds = tuple(
        _seed_for(seed, capability_id, 10_000 + bank_index)
        for bank_index in range(FIT_SEED_BANK_COUNT)
    )
    validation_seed = _seed_for(seed, capability_id, 20_000)
    validation_sample_count = max(sample_count * FIT_SEED_BANK_COUNT, 256)
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
            sample_count=sample_count,
            seeds=fit_seeds,
        )
    structure_scale = invert_monotone_feature_curve(scale_results, [desired[-1]])[0]
    parameters = {**capability_nuisance, "structure_scale": float(structure_scale)}

    lambda_feature_values: dict[float, float] = {}
    for intensity_lambda in LAMBDA_GRID:
        lambda_feature_values[intensity_lambda] = mean_feature_over_seed_banks(
            spec=spec,
            capability_id=capability_id,
            parameters={**profile_nuisance, **parameters},
            intensity_lambda=intensity_lambda,
            feature_name=primary_feature,
            sample_count=sample_count,
            seeds=fit_seeds,
        )

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
            "fit_sample_count": int(sample_count * FIT_SEED_BANK_COUNT),
            "fit_seed_bank_count": FIT_SEED_BANK_COUNT,
            "fit_samples_per_seed_bank": int(sample_count),
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
        measurement_end = int(spec.context_length) + feature_measurement_horizon
        measurement_target = target[:measurement_end]
        measurement_covariates = (
            covariates[:measurement_end] if covariates is not None else None
        )
        rows.append(
            _realized_features(
                measurement_target,
                measurement_covariates,
                int(spec.season_length),
                int(spec.context_length),
            )
        )
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
