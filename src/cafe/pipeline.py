#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cafe import provenance
from cafe import protocol
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_GENERATOR_VERSION,
    REAL_ANCHORED_MINIMUM_COMPONENT_RMS_RATIO,
    REAL_ANCHORED_MINIMUM_CYCLES,
    REAL_ANCHORED_MINIMUM_ELIGIBLE_BACKGROUNDS,
    REAL_ANCHORED_MINIMUM_FUTURE_COMPONENT_RMS_RATIO,
    REAL_ANCHORED_SUPPORTED_CAPABILITIES,
)
from cafe.generation.real_anchored_policy import (
    REAL_ANCHORED_CANONICAL_STRENGTH_GRID,
    REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION,
    REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION,
    REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION,
    REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION,
    REAL_ANCHORED_PROTOCOL_SCHEMA,
    protocol_decisions as real_anchored_protocol_decisions,
)
from cafe.inference import runner as inference

DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"
DEFAULT_MODELS = inference.DEFAULT_MODELS
STEPS = ("calibration", "generation", "validation", "inference", "analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete formal CaFE pipeline."
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help=(
            "One registered dataset id. Repeat the flag to run several "
            "datasets. Defaults to gift_electricity_h."
        ),
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=None,
        help="Convenience form for passing several registered dataset ids.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--analysis-source-experiment-root",
        type=Path,
        default=None,
        help=(
            "For an analysis-only run, reuse immutable generation and "
            "inference artifacts from this completed experiment."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "Stable experiment identity. When omitted, derive one from the "
            "generator version, calibration-stage hash, and UTC start time."
        ),
    )
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Real-data root for a single selected adapter-backed dataset "
            "(for example /root/xmy/M5 for m5_daily)."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument(
        "--calibration-sample-seed",
        type=int,
        default=protocol.CALIBRATION_SAMPLE_SEED,
        help=(
            "Deterministic batch seed for ordinary real calibration and "
            "real-accuracy forecast origins."
        ),
    )
    parser.add_argument(
        "--real-anchored-sample-seed",
        type=int,
        default=protocol.REAL_ANCHORED_SAMPLE_SEED,
        help=(
            "Deterministic batch seed for L504 real-anchored candidate "
            "origins before reference/evaluation splitting."
        ),
    )
    parser.add_argument(
        "--preparation-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Capability-level processes used by calibration and generation.",
    )
    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=1,
        help=(
            "Datasets prepared concurrently. Values above one are allowed "
            "only when the selected range ends at validation."
        ),
    )
    parser.add_argument(
        "--inference-preprocess-workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help=(
            "CPU workers that prepare per-model dataset tasks before the "
            "model-major inference phases."
        ),
    )
    parser.add_argument(
        "--calibration-seeds",
        type=int,
        default=protocol.DEFAULT_CALIBRATION_PATH_COUNT,
    )
    parser.add_argument(
        "--max-calibration-seeds",
        type=int,
        default=protocol.MAX_CALIBRATION_PATH_COUNT,
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=5,
        help=(
            "Maximum deterministic candidates for one capability/seed "
            "bundle, including attempt zero."
        ),
    )
    parser.add_argument(
        "--near-distance-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the anchor-internal DCR/NNDR anti-copy gate during " "generation."
        ),
    )
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=protocol.CAPABILITIES,
        default=list(protocol.CAPABILITIES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=list(inference.DEFAULT_ENDPOINTS),
    )
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    inference.add_endpoint_topology_arguments(parser)
    parser.add_argument("--start-at", choices=STEPS, default="calibration")
    parser.add_argument("--stop-after", choices=STEPS, default="analysis")
    parser.add_argument("--resume-inference", action="store_true")
    parser.add_argument(
        "--resume-analysis",
        action="store_true",
        help=(
            "Reuse a complete analysis shard only after validating its "
            "inference-manifest binding, requested models, and output files."
        ),
    )
    parser.add_argument(
        "--analysis-profile",
        choices=("full", "scores_only"),
        default="full",
        help=(
            "Select full diagnostics or score-only analysis containing "
            "model MASE and primary mechanism scores."
        ),
    )
    return parser.parse_args()


def run(
    module: str,
    arguments: list[str],
    *,
    log_path: Path | None = None,
) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("+ " + " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=protocol.REPO_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{protocol.utc_now()}] + {' '.join(command)}\n")
        log.flush()
        subprocess.run(
            command,
            cwd=protocol.REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def requested_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.dataset_id and args.dataset_ids:
        raise ValueError("use either --dataset-id or --dataset-ids, not both")
    values = list(args.dataset_ids or args.dataset_id or ["gift_electricity_h"])
    if len(values) != len(set(values)):
        raise ValueError("cafe dataset ids must be unique")
    for dataset_id in values:
        protocol.resolve_dataset(dataset_id)
    return values


def validate_dataset_parallelism(
    *,
    dataset_workers: int,
    start_index: int,
    stop_index: int,
) -> None:
    if dataset_workers < 1:
        raise ValueError("dataset_workers must be positive")
    preparation_only = stop_index <= STEPS.index("validation")
    analysis_only = start_index == STEPS.index(
        "analysis"
    ) and stop_index == STEPS.index("analysis")
    if dataset_workers > 1 and not (preparation_only or analysis_only):
        raise ValueError(
            "dataset-level parallelism supports preparation-only or "
            "analysis-only ranges; use --dataset-workers 1 when inference "
            "or a mixed inference/analysis range is selected"
        )


def commands_for_dataset(
    args: argparse.Namespace,
    dataset_id: str,
    *,
    experiment_root: Path,
) -> dict[str, tuple[str, list[str]]]:
    common = [
        "--dataset-id",
        dataset_id,
        "--output-root",
        str(experiment_root),
    ]
    seed = [
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
    ]
    return {
        "calibration": (
            "cafe.calibration.runner",
            [
                *common,
                "--gift-eval-dir",
                str(args.gift_eval_dir.resolve()),
                *(
                    [
                        "--source-root",
                        str(getattr(args, "source_root").resolve()),
                    ]
                    if getattr(args, "source_root", None) is not None
                    else []
                ),
                "--max-anchors",
                str(args.max_anchors),
                "--calibration-sample-seed",
                str(
                    getattr(
                        args,
                        "calibration_sample_seed",
                        protocol.CALIBRATION_SAMPLE_SEED,
                    )
                ),
                "--real-anchored-sample-seed",
                str(
                    getattr(
                        args,
                        "real_anchored_sample_seed",
                        protocol.REAL_ANCHORED_SAMPLE_SEED,
                    )
                ),
                "--calibration-seeds",
                str(args.calibration_seeds),
                "--max-calibration-seeds",
                str(args.max_calibration_seeds),
                "--workers",
                str(args.preparation_workers),
                "--capabilities",
                *args.capabilities,
            ],
        ),
        "generation": (
            "cafe.generation.runner",
            [
                *common,
                *seed,
                "--workers",
                str(args.preparation_workers),
                "--max-generation-attempts",
                str(args.max_generation_attempts),
                (
                    "--near-distance-gate"
                    if args.near_distance_gate
                    else "--no-near-distance-gate"
                ),
                "--capabilities",
                *args.capabilities,
            ],
        ),
        "validation": (
            "cafe.validation.runner",
            [*common, *seed],
        ),
        "inference": (
            "cafe.inference.runner",
            [
                *common,
                *seed,
                "--models",
                *args.models,
                "--endpoints",
                *args.endpoints,
                "--api-prefix",
                str(args.api_prefix),
                *inference.endpoint_topology_cli_arguments(args),
                *(["--resume"] if args.resume_inference else []),
            ],
        ),
        "analysis": (
            "cafe.analysis.runner",
            [
                *common,
                *seed,
                "--models",
                *args.models,
                "--analysis-profile",
                str(getattr(args, "analysis_profile", "full")),
                *(
                    [
                        "--source-experiment-root",
                        str(
                            getattr(
                                args,
                                "analysis_source_experiment_root",
                            ).resolve()
                        ),
                    ]
                    if getattr(
                        args,
                        "analysis_source_experiment_root",
                        None,
                    )
                    is not None
                    else []
                ),
            ],
        ),
    }


def model_major_inference_arguments(
    args: argparse.Namespace,
    dataset_ids: list[str],
    *,
    experiment_root: Path,
) -> list[str]:
    return [
        "--dataset-ids",
        *dataset_ids,
        "--output-root",
        str(experiment_root),
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
        "--models",
        *args.models,
        "--endpoints",
        *args.endpoints,
        "--api-prefix",
        str(args.api_prefix),
        *inference.endpoint_topology_cli_arguments(args),
        "--preprocess-workers",
        str(args.inference_preprocess_workers),
        "--input-capability-contract",
        str(experiment_root / "stage_contracts" / "inference.json"),
        *(["--resume"] if args.resume_inference else []),
    ]


def experiment_analysis_arguments(
    args: argparse.Namespace,
    *,
    experiment_root: Path,
) -> list[str]:
    return [
        "--aggregate-experiment",
        "--output-root",
        str(experiment_root),
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
        "--models",
        *args.models,
        "--analysis-profile",
        str(getattr(args, "analysis_profile", "full")),
        *(
            [
                "--source-experiment-root",
                str(
                    getattr(
                        args,
                        "analysis_source_experiment_root",
                    ).resolve()
                ),
            ]
            if getattr(
                args,
                "analysis_source_experiment_root",
                None,
            )
            is not None
            else []
        ),
        *(["--reuse-existing-aggregate"] if args.resume_analysis else []),
    ]


def run_experiment_analysis(
    args: argparse.Namespace,
    *,
    experiment_root: Path,
    log_path: Path | None = None,
) -> None:
    run(
        "cafe.analysis.runner",
        experiment_analysis_arguments(
            args,
            experiment_root=experiment_root,
        ),
        log_path=log_path,
    )


def protocol_config(
    args: argparse.Namespace,
    dataset_ids: list[str],
) -> dict[str, Any]:
    missing_configs = sorted(set(args.models) - set(inference.MODEL_EXECUTION_CONFIG))
    if missing_configs:
        raise ValueError(
            "missing model execution configs: " + ", ".join(missing_configs)
        )
    return {
        "schema_version": "cafe.experiment_protocol.v5",
        "pipeline_schema_version": protocol.SCHEMA_VERSION,
        "generator_version": protocol.GENERATOR_VERSION,
        "benchmark_tracks": [
            "real_accuracy",
            "real_anchored_counterfactual",
            "deterministic_synthetic",
        ],
        "real_anchored_protocol": {
            **real_anchored_protocol_decisions(),
            "schema_version": REAL_ANCHORED_PROTOCOL_SCHEMA,
            "generator_version": REAL_ANCHORED_GENERATOR_VERSION,
            "univariate_supported_capabilities": list(
                REAL_ANCHORED_SUPPORTED_CAPABILITIES
            ),
            "decomposition_fit_scope": "history_only_l504",
            "decomposition_context_length": (
                protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
            ),
            "model_master_context_length": (
                protocol.REAL_ANCHORED_CONTEXT_LENGTH
            ),
            "forecast_horizon": protocol.HORIZON,
            "rank_context_length": protocol.FIXED_CONTEXT_LENGTH,
            "minimum_complete_cycles": REAL_ANCHORED_MINIMUM_CYCLES,
            "minimum_controlled_component_rms_ratio": (
                REAL_ANCHORED_MINIMUM_COMPONENT_RMS_RATIO
            ),
            "minimum_future_controlled_component_rms_ratio": (
                REAL_ANCHORED_MINIMUM_FUTURE_COMPONENT_RMS_RATIO
            ),
            "future_controlled_component_gate": (
                "history_only_analytic_h48_rms_relative_to_shared_l336_scale"
            ),
            "period_resolution": (
                "calibration_carrier_plus_separated_history_fft_peaks_v1"
            ),
            "canonical_strength_grid": list(
                REAL_ANCHORED_CANONICAL_STRENGTH_GRID
            ),
            "physical_alpha_grid": (
                "dataset_capability_fixed_reference_q75_mapping"
            ),
            "paired_minimum_separation_gate": {
                "status": "mandatory_not_cli_disableable",
                "history_window": "trailing_l168",
                "future_window": "history_only_or_known_future_legal_h48_delta",
                "minimum_acceptance_fraction": (
                    REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
                ),
                "maximum_history_macro_separation": (
                    REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION
                ),
                "maximum_future_macro_separation": (
                    REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION
                ),
                "maximum_affected_channel_separation": (
                    REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION
                ),
                "distinct_from_synthetic_anti_copy": True,
            },
            "multi_seasonal_intervention": (
                "fixed_carrier_scale_secondary_harmonic_sum_v1"
            ),
            "trend_intervention": (
                "fixed_level_and_linear_scale_c1_local_nonlinearity_w96_v1"
            ),
            "time_varying_seasonality_intervention": (
                "carrier_phase_locked_symmetric_constrained_am_v1"
            ),
            "nonlinear_future_innovation_main": (
                "zero_future_innovation_paired_rollout_v1"
            ),
            "nonlinear_future_innovation_sensitivity": (
                "history_residual_replay_qualification_only_v1"
            ),
            "normalization": "shared_unmodified_real_l336_history",
            "future_semantics": (
                "observed_real_nuisance_plus_deterministic_intervention"
            ),
            "official_test_tail": (
                "excluded_before_window_sampling_gift_short_term_or_m4_h48"
            ),
            "minimum_eligible_backgrounds": (
                REAL_ANCHORED_MINIMUM_ELIGIBLE_BACKGROUNDS
            ),
            "anti_copy_applicability": (
                "not_applicable_intentional_real_anchor_counterfactual"
            ),
            "qualification_bank": (
                "source_time_disjoint_reference_bank_never_evaluation_"
                "origins_v1"
            ),
            "formal_panel_minimum_dimension": 3,
            "panel_d2_policy": "sensitivity_only_never_formal_rank",
            "hierarchical_coherence_policy": (
                "qualification_only_no_generation_no_formal_rank"
            ),
            "structural_input_ablation": (
                "mandatory_common_cross_attribution_separate_not_score_"
                "weighted_v1"
            ),
            "ranking": "separate_from_deterministic_synthetic",
        },
        "dataset_ids": list(dataset_ids),
        "real_data_adapters": {
            dataset_id: protocol.resolve_dataset(dataset_id).real_data_adapter
            for dataset_id in dataset_ids
        },
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "max_anchors": int(args.max_anchors),
        "calibration_sample_seed": int(
            getattr(
                args,
                "calibration_sample_seed",
                protocol.CALIBRATION_SAMPLE_SEED,
            )
        ),
        "real_anchored_sample_seed": int(
            getattr(
                args,
                "real_anchored_sample_seed",
                protocol.REAL_ANCHORED_SAMPLE_SEED,
            )
        ),
        "calibration_seeds": int(args.calibration_seeds),
        "max_calibration_seeds": int(args.max_calibration_seeds),
        "generation_acceptance": {
            "max_attempts_per_capability_seed_bundle": int(
                args.max_generation_attempts
            ),
            "feature_support": (
                "diagnostic_only_primary_feature_anchor_minmax_with_"
                "0.1_span_each_side_when_real_reference_exists"
            ),
            "synthetic_near_distance_enabled": bool(args.near_distance_gate),
            "synthetic_near_distance": (
                "anchor_internal_leave_one_out_dcr_p01_and_nndr_p01_"
                "with_multivariate_majority_vote"
            ),
            "real_anchored_paired_minimum_separation": (
                "mandatory_treatment_source_l168_distance_"
                "with_local_augmentation_budget_v1"
            ),
            "retry_identity": (
                "formal seed, anchor, sample IDs, and pairing remain fixed"
            ),
            "structural_identifiability": (
                "real_calibrated_common_factor_and_cross_series_pairs_use_"
                "observable_aligned_selected_dose_gates_plus_separate_"
                "lambda1_blind_positive_controls"
            ),
            "cross_series_minimum_incremental_history_holdout_gain": (
                protocol.CROSS_SERIES_MIN_INCREMENTAL_HOLDOUT_GAIN
            ),
            "common_factor_identifiability": (
                "pca_share_above_finite_panel_floor_at_selected_dose_plus_"
                "blind_rank1_counterfactual_control_at_lambda1"
            ),
            "cross_series_intensity_coordinate": (
                "time_reverse_null_corrected_history_only_incremental_gain_"
                "with_public_lag_range_1_to_24"
            ),
            "family_intensity_scale": (
                "joint_primary_secondary_family_mean_inverse_on_dataset_real_"
                "support_no_generator_relative_fallback"
            ),
            "unavailable_capability_policy": (
                "skip_dataset_capability_when_real_coordinate_or_joint_"
                "generator_support_is_unavailable"
            ),
        },
        "calibration_path_policy": (
            "independent_family_response_qualification_bank_"
            "fixed_base_hard_failure_only_expansion_v1"
        ),
        "mase_scale_policy": (
            "seasonal_lag_with_per_target_lag1_degeneracy_fallback_v1"
        ),
        "capabilities": list(args.capabilities),
        "models": list(args.models),
        "api_prefix": str(args.api_prefix),
        "input_adaptation_policy": inference.INPUT_ADAPTATION_POLICY_ID,
        "analysis_profile": str(getattr(args, "analysis_profile", "full")),
        "model_execution_config": {
            model_id: dict(inference.MODEL_EXECUTION_CONFIG[model_id])
            for model_id in args.models
        },
        "dataset_execution_policy": (
            "preparation_dataset_parallelism_is_execution_only_"
            "inference_remains_sequential_in_declared_order"
        ),
        "model_scheduling_policy": {
            "policy_id": inference.SCHEDULING_POLICY_ID,
            "phase_order": "models_in_declared_order",
            "service_collaboration": (
                "all_compatible_services_run_deterministic_parts_of_each_model"
            ),
            "resume_part_identity": "preserved_when_service_count_changes",
        },
        "real_calibration_context_length": (protocol.REAL_CALIBRATION_CONTEXT_LENGTH),
        "synthetic_master_context_length": protocol.CONTEXT_LENGTH,
        "fixed_context_length": protocol.FIXED_CONTEXT_LENGTH,
        "horizon": protocol.HORIZON,
        "view_context_lengths": list(protocol.VIEW_CONTEXT_LENGTHS),
        "intensities": list(protocol.INTENSITIES),
        "aggregation_policy": (
            "dataset-isolated outputs and reports; no implicit "
            "cross-dataset averaging"
        ),
        "primary_mechanism_score_policy": {
            "hierarchical_coherence": (
                "i5-child-contrast-nmae-plus-unit-weight-native-"
                "coherence-nmae-penalty"
            ),
            "common_factor": ("all-seed-i5-protected-target-paired-effect-nrmse"),
            "cross_series_dependence": (
                "all-seed-i5-active-history-covered-prefix-paired-effect-" "nrmse"
            ),
            "factual_accuracy": ("unchanged-single-member-i1-i5-seed-group-mean-mase"),
        },
        "analysis_source_experiment": (
            {
                "path": str(
                    getattr(
                        args,
                        "analysis_source_experiment_root",
                    ).resolve()
                ),
                "experiment_sha256": protocol.file_sha256(
                    getattr(
                        args,
                        "analysis_source_experiment_root",
                    ).resolve()
                    / "experiment.json"
                ),
            }
            if getattr(
                args,
                "analysis_source_experiment_root",
                None,
            )
            is not None
            else None
        ),
    }


def resolve_stage_input_capabilities(
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Resolve one consistent live input contract for every formal model."""

    observed: dict[str, list[tuple[str, dict[str, Any]]]] = {
        model_id: [] for model_id in args.models
    }
    with ThreadPoolExecutor(max_workers=len(args.endpoints)) as executor:
        futures = {
            executor.submit(
                inference.health_catalog,
                endpoint,
                args.api_prefix,
            ): endpoint
            for endpoint in args.endpoints
        }
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            endpoint, catalog = result
            for model_id in args.models:
                if model_id in catalog:
                    observed[model_id].append(
                        (
                            endpoint,
                            inference.resolve_input_capability(catalog[model_id]),
                        )
                    )

    resolved: dict[str, dict[str, Any]] = {}
    for model_id, rows in observed.items():
        if not rows:
            raise RuntimeError(
                f"model {model_id!r} unavailable while freezing inference contract"
            )
        first = rows[0][1]
        mismatched = [endpoint for endpoint, value in rows if value != first]
        if mismatched:
            raise ValueError(
                f"inconsistent input capability for {model_id}: "
                + ", ".join(endpoint for endpoint, _value in rows)
            )
        resolved[model_id] = first
    return resolved


def default_experiment_id(
    preparation_sha256: str,
    *,
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    generator_tag = protocol.safe_id(protocol.GENERATOR_VERSION)
    return f"{generator_tag}_{preparation_sha256[:12]}_{timestamp}"


def code_provenance() -> dict[str, Any]:
    return provenance.code_provenance(protocol.REPO_ROOT)


def stage_protocol_configs(
    full_protocol: dict[str, Any],
    *,
    endpoints: list[str],
    endpoint_profiles: dict[str, dict[str, Any]],
    preparation_execution: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    shared_structure = {
        key: full_protocol[key]
        for key in (
            "pipeline_schema_version",
            "dataset_ids",
            "real_data_adapters",
            "capabilities",
            "real_calibration_context_length",
            "synthetic_master_context_length",
            "fixed_context_length",
            "horizon",
            "view_context_lengths",
            "intensities",
            "benchmark_tracks",
            "real_anchored_protocol",
        )
    }
    return {
        "calibration": {
            "schema_version": "cafe.calibration_stage.v5",
            **shared_structure,
            "max_anchors": full_protocol["max_anchors"],
            "calibration_sample_seed": full_protocol.get(
                "calibration_sample_seed",
                protocol.CALIBRATION_SAMPLE_SEED,
            ),
            "real_anchored_sample_seed": full_protocol.get(
                "real_anchored_sample_seed",
                protocol.REAL_ANCHORED_SAMPLE_SEED,
            ),
            "calibration_seeds": full_protocol["calibration_seeds"],
            "max_calibration_seeds": full_protocol["max_calibration_seeds"],
            "calibration_path_policy": full_protocol["calibration_path_policy"],
            "execution": preparation_execution,
        },
        "generation": {
            "schema_version": "cafe.generation_stage.v5",
            **shared_structure,
            "generator_version": full_protocol["generator_version"],
            "seed_start": full_protocol["seed_start"],
            "seed_count": full_protocol["seed_count"],
            "generation_acceptance": full_protocol["generation_acceptance"],
            "mase_scale_policy": full_protocol["mase_scale_policy"],
            "execution": preparation_execution,
        },
        "validation": {
            "schema_version": "cafe.validation_stage.v5",
            **shared_structure,
            "generator_version": full_protocol["generator_version"],
        },
        "inference": {
            "schema_version": "cafe.inference_stage.v5",
            "dataset_ids": full_protocol["dataset_ids"],
            "seed_start": full_protocol["seed_start"],
            "seed_count": full_protocol["seed_count"],
            "models": full_protocol["models"],
            "model_execution_config": full_protocol["model_execution_config"],
            "model_scheduling_policy": full_protocol["model_scheduling_policy"],
            "input_adaptation_policy": full_protocol["input_adaptation_policy"],
            "view_context_lengths": full_protocol["view_context_lengths"],
            "fixed_context_length": full_protocol["fixed_context_length"],
            "horizon": full_protocol["horizon"],
            "benchmark_tracks": full_protocol["benchmark_tracks"],
            "real_anchored_protocol": full_protocol[
                "real_anchored_protocol"
            ],
            "requested_endpoints": endpoints,
            "requested_api_prefix": full_protocol["api_prefix"],
            "endpoint_profiles": endpoint_profiles,
        },
        "analysis": {
            "schema_version": "cafe.analysis_stage.v5",
            "dataset_ids": full_protocol["dataset_ids"],
            "seed_start": full_protocol["seed_start"],
            "seed_count": full_protocol["seed_count"],
            "models": full_protocol["models"],
            "capabilities": full_protocol["capabilities"],
            "analysis_profile": full_protocol["analysis_profile"],
            "fixed_context_length": full_protocol["fixed_context_length"],
            "view_context_lengths": full_protocol["view_context_lengths"],
            "benchmark_tracks": full_protocol["benchmark_tracks"],
            "real_anchored_protocol": full_protocol[
                "real_anchored_protocol"
            ],
            "aggregation_policy": full_protocol["aggregation_policy"],
            "primary_mechanism_score_policy": (
                full_protocol["primary_mechanism_score_policy"]
            ),
            "analysis_source_experiment": (full_protocol["analysis_source_experiment"]),
        },
    }


def initialize_experiment(
    *,
    storage_root: Path,
    experiment_id: str,
) -> tuple[Path, dict[str, Any]]:
    if protocol.safe_id(experiment_id) != experiment_id:
        raise ValueError("experiment-id may contain only letters, digits, '_' and '-'")
    experiment_root = storage_root.resolve() / experiment_id
    existing_path = experiment_root / "experiment.json"
    if (
        experiment_root.exists()
        and any(experiment_root.iterdir())
        and not existing_path.exists()
    ):
        raise ValueError(
            "refusing to use a non-empty experiment directory without "
            "experiment.json"
        )
    manifest = provenance.initialize_experiment(
        experiment_root,
        experiment_id=experiment_id,
        created_at=protocol.utc_now(),
    )
    return experiment_root, manifest


def initialize_stage_contracts(
    experiment_root: Path,
    *,
    stage_configs: dict[str, dict[str, Any]],
    requested_steps: list[str],
    analysis_source_experiment_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for stage in requested_steps:
        stage_index = STEPS.index(stage)
        upstream: list[dict[str, Any]] = []
        if stage_index:
            previous_stage = STEPS[stage_index - 1]
            previous_root = experiment_root
            if stage == "analysis" and analysis_source_experiment_root is not None:
                previous_root = analysis_source_experiment_root.resolve()
            previous_path = previous_root / "stage_contracts" / f"{previous_stage}.json"
            if not previous_path.is_file():
                raise ValueError(
                    f"{stage} requires the frozen {previous_stage} stage contract: "
                    f"{previous_path}"
                )
            upstream = provenance.upstream_records(
                [previous_path],
                relative_to=experiment_root,
            )
        contracts[stage] = provenance.ensure_stage_contract(
            experiment_root,
            stage=stage,
            created_at=protocol.utc_now(),
            repository_root=protocol.REPO_ROOT,
            config=stage_configs[stage],
            upstream=upstream,
        )
    return contracts


def write_pipeline_status(
    experiment_root: Path,
    *,
    experiment_id: str,
    run_plan_sha256: str,
    state: str,
    start_at: str,
    stop_after: str,
    completed: list[dict[str, Any]],
    active_dataset_id: str | None = None,
    active_step: str | None = None,
    active_dataset_ids: list[str] | None = None,
    active_jobs: list[dict[str, Any]] | None = None,
    failed: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> None:
    protocol.write_json(
        experiment_root / "pipeline_status.json",
        {
            "schema_version": "cafe.pipeline_status.v1",
            "updated_at": protocol.utc_now(),
            "experiment_id": experiment_id,
            "run_plan_sha256": run_plan_sha256,
            "state": state,
            "start_at": start_at,
            "stop_after": stop_after,
            "active_dataset_id": active_dataset_id,
            "active_step": active_step,
            "active_dataset_ids": list(active_dataset_ids or []),
            "active_jobs": list(active_jobs or []),
            "completed": completed,
            "failed": list(failed or []),
            "error": error,
        },
    )


def write_dataset_preparation_status(
    experiment_root: Path,
    *,
    dataset_id: str,
    state: str,
    requested_steps: list[str],
    completed_steps: list[str],
    active_step: str | None = None,
    elapsed_seconds: float | None = None,
    error: str | None = None,
) -> None:
    protocol.write_json(
        experiment_root / dataset_id / "preparation_status.json",
        {
            "schema_version": "cafe.dataset_preparation_status.v1",
            "updated_at": protocol.utc_now(),
            "dataset_id": dataset_id,
            "state": state,
            "requested_steps": requested_steps,
            "completed_steps": completed_steps,
            "active_step": active_step,
            "elapsed_seconds": elapsed_seconds,
            "error": error,
        },
    )


def execute_dataset_steps(
    args: argparse.Namespace,
    dataset_id: str,
    *,
    experiment_root: Path,
    steps: list[str],
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Execute one dataset as an isolated preparation job.

    Errors are returned as data so a bad dataset cannot prevent the scheduler
    from attempting the remaining declared datasets.
    """

    started = time.monotonic()
    completed_steps: list[str] = []
    commands = commands_for_dataset(
        args,
        dataset_id,
        experiment_root=experiment_root,
    )
    write_dataset_preparation_status(
        experiment_root,
        dataset_id=dataset_id,
        state="running",
        requested_steps=steps,
        completed_steps=completed_steps,
        active_step=steps[0] if steps else None,
    )
    for step in steps:
        write_dataset_preparation_status(
            experiment_root,
            dataset_id=dataset_id,
            state="running",
            requested_steps=steps,
            completed_steps=completed_steps,
            active_step=step,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        script, arguments = commands[step]
        try:
            run(script, arguments, log_path=log_path)
        except Exception as error:
            elapsed = round(time.monotonic() - started, 3)
            error_text = f"{type(error).__name__}: {error}"
            outcome = {
                "dataset_id": dataset_id,
                "state": "failed",
                "steps": list(completed_steps),
                "failed_step": step,
                "output_dir": str(experiment_root / dataset_id),
                "log_path": str(log_path) if log_path is not None else None,
                "elapsed_seconds": elapsed,
                "error": error_text,
            }
            write_dataset_preparation_status(
                experiment_root,
                dataset_id=dataset_id,
                state="failed",
                requested_steps=steps,
                completed_steps=completed_steps,
                active_step=step,
                elapsed_seconds=elapsed,
                error=error_text,
            )
            return outcome
        completed_steps.append(step)
    elapsed = round(time.monotonic() - started, 3)
    outcome = {
        "dataset_id": dataset_id,
        "state": "complete",
        "steps": list(completed_steps),
        "output_dir": str(experiment_root / dataset_id),
        "log_path": str(log_path) if log_path is not None else None,
        "elapsed_seconds": elapsed,
    }
    write_dataset_preparation_status(
        experiment_root,
        dataset_id=dataset_id,
        state="complete",
        requested_steps=steps,
        completed_steps=completed_steps,
        elapsed_seconds=elapsed,
    )
    return outcome


def run_parallel_preparation(
    args: argparse.Namespace,
    dataset_ids: list[str],
    *,
    experiment_root: Path,
    experiment_id: str,
    run_plan_sha256: str,
    steps: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare datasets with a bounded, work-conserving process schedule."""

    outcomes: dict[str, dict[str, Any]] = {}
    active: dict[Future[dict[str, Any]], str] = {}
    next_index = 0
    log_root = experiment_root / "preparation_logs"

    def ordered(state: str) -> list[dict[str, Any]]:
        return [
            outcomes[dataset_id]
            for dataset_id in dataset_ids
            if outcomes.get(dataset_id, {}).get("state") == state
        ]

    def submit_available(executor: ThreadPoolExecutor) -> None:
        nonlocal next_index
        while len(active) < args.dataset_workers and next_index < len(dataset_ids):
            dataset_id = dataset_ids[next_index]
            next_index += 1
            future = executor.submit(
                execute_dataset_steps,
                args,
                dataset_id,
                experiment_root=experiment_root,
                steps=steps,
                log_path=log_root / f"{dataset_id}.log",
            )
            active[future] = dataset_id

    def write_running_status() -> None:
        active_ids = [
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id in set(active.values())
        ]
        write_pipeline_status(
            experiment_root,
            experiment_id=experiment_id,
            run_plan_sha256=run_plan_sha256,
            state="running",
            start_at=args.start_at,
            stop_after=args.stop_after,
            completed=ordered("complete"),
            failed=ordered("failed"),
            active_step="concurrent_preparation",
            active_dataset_ids=active_ids,
            active_jobs=[
                {
                    "dataset_id": dataset_id,
                    "status_path": str(
                        experiment_root / dataset_id / "preparation_status.json"
                    ),
                    "log_path": str(log_root / f"{dataset_id}.log"),
                }
                for dataset_id in active_ids
            ],
        )

    with ThreadPoolExecutor(max_workers=args.dataset_workers) as executor:
        submit_available(executor)
        write_running_status()
        while active:
            done, _ = wait(set(active), return_when=FIRST_COMPLETED)
            for future in done:
                dataset_id = active.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = {
                        "dataset_id": dataset_id,
                        "state": "failed",
                        "steps": [],
                        "failed_step": "scheduler",
                        "output_dir": str(experiment_root / dataset_id),
                        "log_path": str(log_root / f"{dataset_id}.log"),
                        "elapsed_seconds": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                outcomes[dataset_id] = outcome
                print(protocol.canonical_json(outcome), flush=True)
            submit_available(executor)
            write_running_status()
    return ordered("complete"), ordered("failed")


def reusable_analysis_manifest(
    experiment_root: Path,
    *,
    dataset_id: str,
    seed_start: int,
    seed_count: int,
    models: list[str],
    source_experiment_root: Path | None = None,
    analysis_profile: str = "full",
) -> bool:
    shard_name = f"seed_{seed_start:06d}_{seed_start + seed_count:06d}"
    inference_manifest_path = (
        (source_experiment_root or experiment_root)
        / dataset_id
        / "03_inference"
        / shard_name
        / "inference_manifest.json"
    )
    analysis_manifest_path = (
        experiment_root
        / dataset_id
        / "04_analysis"
        / shard_name
        / "analysis_manifest.json"
    )
    if not inference_manifest_path.is_file() or not analysis_manifest_path.is_file():
        return False
    try:
        inference_manifest = protocol.read_json(inference_manifest_path)
        analysis_manifest = protocol.read_json(analysis_manifest_path)
        if not bool(inference_manifest.get("complete")):
            return False
        if analysis_manifest.get("schema_version") not in {
            "cafe.analysis_manifest.v1",
            "cafe.analysis_manifest.v2",
        }:
            return False
        if str(analysis_manifest.get("dataset_id")) != dataset_id:
            return False
        if list(analysis_manifest.get("models") or []) != list(models):
            return False
        if analysis_manifest.get("analysis_profile") != analysis_profile:
            return False
        if str(analysis_manifest.get("inference_manifest_sha256")) != (
            protocol.file_sha256(inference_manifest_path)
        ):
            return False
        coverage = {
            str(row.get("model_id")): row
            for row in analysis_manifest.get("coverage", [])
            if isinstance(row, dict)
        }
        for model_id in models:
            row = coverage.get(model_id)
            if row is None or int(row.get("missing_prediction_count", -1)) != 0:
                return False
        files = analysis_manifest.get("files")
        if not isinstance(files, dict) or not files:
            return False
        for record in files.values():
            if not isinstance(record, dict):
                return False
            path = Path(str(record.get("path", "")))
            if (
                not path.is_file()
                or record.get("bytes") is None
                or int(record["bytes"]) != path.stat().st_size
                or not record.get("sha256")
                or str(record["sha256"]) != protocol.file_sha256(path)
            ):
                return False
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def execute_analysis_job(
    args: argparse.Namespace,
    dataset_id: str,
    *,
    experiment_root: Path,
    log_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if args.resume_analysis and reusable_analysis_manifest(
        experiment_root,
        dataset_id=dataset_id,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        models=list(args.models),
        source_experiment_root=(
            getattr(args, "analysis_source_experiment_root").resolve()
            if getattr(args, "analysis_source_experiment_root", None) is not None
            else experiment_root
        ),
        analysis_profile=str(getattr(args, "analysis_profile", "full")),
    ):
        return {
            "dataset_id": dataset_id,
            "state": "complete",
            "steps": ["analysis"],
            "analysis_status": "already_complete",
            "output_dir": str(experiment_root / dataset_id),
            "log_path": str(log_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    script, arguments = commands_for_dataset(
        args,
        dataset_id,
        experiment_root=experiment_root,
    )["analysis"]
    try:
        run(script, arguments, log_path=log_path)
    except Exception as error:
        return {
            "dataset_id": dataset_id,
            "state": "failed",
            "steps": [],
            "failed_step": "analysis",
            "output_dir": str(experiment_root / dataset_id),
            "log_path": str(log_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "dataset_id": dataset_id,
        "state": "complete",
        "steps": ["analysis"],
        "analysis_status": "computed",
        "output_dir": str(experiment_root / dataset_id),
        "log_path": str(log_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_parallel_analysis(
    args: argparse.Namespace,
    dataset_ids: list[str],
    *,
    experiment_root: Path,
    experiment_id: str,
    run_plan_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcomes: dict[str, dict[str, Any]] = {}
    active: dict[Future[dict[str, Any]], str] = {}
    next_index = 0
    log_root = experiment_root / "analysis_logs"

    def ordered(state: str) -> list[dict[str, Any]]:
        return [
            outcomes[dataset_id]
            for dataset_id in dataset_ids
            if outcomes.get(dataset_id, {}).get("state") == state
        ]

    def submit_available(executor: ThreadPoolExecutor) -> None:
        nonlocal next_index
        while len(active) < args.dataset_workers and next_index < len(dataset_ids):
            dataset_id = dataset_ids[next_index]
            next_index += 1
            future = executor.submit(
                execute_analysis_job,
                args,
                dataset_id,
                experiment_root=experiment_root,
                log_path=log_root / f"{dataset_id}.log",
            )
            active[future] = dataset_id

    def write_running_status() -> None:
        active_ids = [
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id in set(active.values())
        ]
        write_pipeline_status(
            experiment_root,
            experiment_id=experiment_id,
            run_plan_sha256=run_plan_sha256,
            state="running",
            start_at=args.start_at,
            stop_after=args.stop_after,
            completed=ordered("complete"),
            failed=ordered("failed"),
            active_step="concurrent_analysis",
            active_dataset_ids=active_ids,
            active_jobs=[
                {
                    "dataset_id": dataset_id,
                    "log_path": str(log_root / f"{dataset_id}.log"),
                }
                for dataset_id in active_ids
            ],
        )

    with ThreadPoolExecutor(max_workers=args.dataset_workers) as executor:
        submit_available(executor)
        write_running_status()
        while active:
            done, _ = wait(set(active), return_when=FIRST_COMPLETED)
            for future in done:
                dataset_id = active.pop(future)
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = {
                        "dataset_id": dataset_id,
                        "state": "failed",
                        "steps": [],
                        "failed_step": "analysis_scheduler",
                        "output_dir": str(experiment_root / dataset_id),
                        "log_path": str(log_root / f"{dataset_id}.log"),
                        "elapsed_seconds": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                outcomes[dataset_id] = outcome
                print(protocol.canonical_json(outcome), flush=True)
            submit_available(executor)
            write_running_status()
    return ordered("complete"), ordered("failed")


def main() -> int:
    args = parse_args()
    dataset_ids = requested_dataset_ids(args)
    non_gift_datasets = [
        dataset_id
        for dataset_id in dataset_ids
        if (
            protocol.resolve_dataset(dataset_id).real_data_adapter
            not in protocol.GIFT_EVAL_REAL_DATA_ADAPTERS
        )
    ]
    if non_gift_datasets and args.source_root is None:
        raise ValueError(
            "non-GIFT datasets require --source-root: " + ", ".join(non_gift_datasets)
        )
    if len(args.models) != len(set(args.models)):
        raise ValueError("model ids must be unique")
    if len(args.endpoints) != len(set(args.endpoints)):
        raise ValueError("inference endpoints must be unique")
    endpoint_presets = inference.endpoint_presets_with_defaults(
        list(args.endpoints),
        list(args.endpoint_preset),
    )
    endpoint_profiles = inference.build_endpoint_profiles(
        list(args.endpoints),
        default_devices=args.devices,
        endpoint_presets=endpoint_presets,
        endpoint_devices=list(args.endpoint_devices),
        endpoint_capacities=list(args.endpoint_capacity),
        endpoint_concurrency_scales=list(args.endpoint_concurrency_scale),
        endpoint_model_capacities=list(args.endpoint_model_capacity),
        endpoint_model_concurrencies=list(args.endpoint_model_concurrency),
    )
    if args.seed_start < 0 or args.seed_count < 1:
        raise ValueError("seed_start must be non-negative and seed_count positive")
    if args.preparation_workers < 1:
        raise ValueError("preparation_workers must be positive")
    if args.inference_preprocess_workers < 1:
        raise ValueError("inference_preprocess_workers must be positive")
    if (
        args.max_anchors < 1
        or args.calibration_seeds < 1
        or args.max_calibration_seeds < args.calibration_seeds
    ):
        raise ValueError(
            "anchor and calibration path budgets must be positive and "
            "maximums must not be smaller than base counts"
        )
    start = STEPS.index(args.start_at)
    stop = STEPS.index(args.stop_after)
    if stop < start:
        raise ValueError("stop-after must not precede start-at")
    if args.analysis_source_experiment_root is not None and not (
        start == STEPS.index("analysis") and stop == STEPS.index("analysis")
    ):
        raise ValueError(
            "--analysis-source-experiment-root requires an analysis-only " "stage range"
        )
    validation_index = STEPS.index("validation")
    validate_dataset_parallelism(
        dataset_workers=args.dataset_workers,
        start_index=start,
        stop_index=stop,
    )
    full_protocol = protocol_config(args, dataset_ids)
    serialized_endpoint_profiles = {
        endpoint: profile.as_dict() for endpoint, profile in endpoint_profiles.items()
    }
    preparation_execution = {
        "dataset_workers": int(args.dataset_workers),
        "capability_workers_per_dataset": int(args.preparation_workers),
        "maximum_capability_worker_processes": int(
            args.dataset_workers * args.preparation_workers
        ),
    }
    stage_configs = stage_protocol_configs(
        full_protocol,
        endpoints=list(args.endpoints),
        endpoint_profiles=serialized_endpoint_profiles,
        preparation_execution=preparation_execution,
    )
    preparation_sha256 = protocol.json_sha256(stage_configs["calibration"])
    experiment_id = args.experiment_id or default_experiment_id(preparation_sha256)
    experiment_root, manifest = initialize_experiment(
        storage_root=args.output_root,
        experiment_id=experiment_id,
    )
    requested_steps = list(STEPS[start : stop + 1])
    preparation_steps = {
        "calibration",
        "generation",
        "validation",
    }
    initial_contract_steps = [
        stage for stage in requested_steps if stage in preparation_steps
    ]
    if not initial_contract_steps:
        if requested_steps[0] == "inference":
            stage_configs["inference"] = {
                **stage_configs["inference"],
                "resolved_model_input_capabilities": (
                    resolve_stage_input_capabilities(args)
                ),
            }
        initial_contract_steps = [requested_steps[0]]
    contracts = initialize_stage_contracts(
        experiment_root,
        stage_configs=stage_configs,
        requested_steps=initial_contract_steps,
        analysis_source_experiment_root=args.analysis_source_experiment_root,
    )
    run_plan_sha256 = protocol.json_sha256(
        {
            "requested_steps": requested_steps,
            "stage_configs": {stage: stage_configs[stage] for stage in requested_steps},
            "launch_code": code_provenance(),
        }
    )
    completed: list[dict[str, Any]] = []
    write_pipeline_status(
        experiment_root,
        experiment_id=experiment_id,
        run_plan_sha256=run_plan_sha256,
        state="running",
        start_at=args.start_at,
        stop_after=args.stop_after,
        completed=completed,
    )
    if start == STEPS.index("analysis") and stop == start:
        completed, failed = run_parallel_analysis(
            args,
            dataset_ids,
            experiment_root=experiment_root,
            experiment_id=experiment_id,
            run_plan_sha256=run_plan_sha256,
        )
        if failed:
            error_text = (
                f"{len(failed)} of {len(dataset_ids)} dataset analysis " "jobs failed"
            )
            write_pipeline_status(
                experiment_root,
                experiment_id=experiment_id,
                run_plan_sha256=run_plan_sha256,
                state="failed",
                start_at=args.start_at,
                stop_after=args.stop_after,
                completed=completed,
                failed=failed,
                error=error_text,
            )
            raise RuntimeError(error_text)
        run_experiment_analysis(
            args,
            experiment_root=experiment_root,
            log_path=(experiment_root / "analysis_logs" / "experiment_summary.log"),
        )
        write_pipeline_status(
            experiment_root,
            experiment_id=experiment_id,
            run_plan_sha256=run_plan_sha256,
            state="complete",
            start_at=args.start_at,
            stop_after=args.stop_after,
            completed=completed,
        )
        print(
            protocol.canonical_json(
                {
                    "experiment_id": experiment_id,
                    "run_plan_sha256": run_plan_sha256,
                    "dataset_count": len(dataset_ids),
                    "output": str(experiment_root),
                }
            )
        )
        return 0
    if stop <= validation_index:
        completed, failed = run_parallel_preparation(
            args,
            dataset_ids,
            experiment_root=experiment_root,
            experiment_id=experiment_id,
            run_plan_sha256=run_plan_sha256,
            steps=list(STEPS[start : stop + 1]),
        )
        if failed:
            error_text = (
                f"{len(failed)} of {len(dataset_ids)} dataset preparation "
                "jobs failed"
            )
            write_pipeline_status(
                experiment_root,
                experiment_id=experiment_id,
                run_plan_sha256=run_plan_sha256,
                state="failed",
                start_at=args.start_at,
                stop_after=args.stop_after,
                completed=completed,
                failed=failed,
                error=error_text,
            )
            raise RuntimeError(error_text)
        write_pipeline_status(
            experiment_root,
            experiment_id=experiment_id,
            run_plan_sha256=run_plan_sha256,
            state="complete",
            start_at=args.start_at,
            stop_after=args.stop_after,
            completed=completed,
        )
        print(
            protocol.canonical_json(
                {
                    "experiment_id": experiment_id,
                    "run_plan_sha256": run_plan_sha256,
                    "dataset_count": len(dataset_ids),
                    "output": str(experiment_root),
                }
            )
        )
        return 0

    active_dataset_id: str | None = None
    active_dataset_ids: list[str] | None = None
    active_step: str | None = None
    completed_steps_by_dataset: dict[str, list[str]] = {
        dataset_id: [] for dataset_id in dataset_ids
    }

    def completed_records() -> list[dict[str, Any]]:
        return [
            {
                "dataset_id": dataset_id,
                "steps": list(completed_steps_by_dataset[dataset_id]),
                "output_dir": str(experiment_root / dataset_id),
            }
            for dataset_id in dataset_ids
            if completed_steps_by_dataset[dataset_id]
        ]

    try:
        for step in STEPS[start : stop + 1]:
            active_step = step
            if step == "inference":
                stage_configs["inference"] = {
                    **stage_configs["inference"],
                    "resolved_model_input_capabilities": (
                        resolve_stage_input_capabilities(args)
                    ),
                }
            if step not in contracts:
                contracts.update(
                    initialize_stage_contracts(
                        experiment_root,
                        stage_configs=stage_configs,
                        requested_steps=[step],
                        analysis_source_experiment_root=(
                            args.analysis_source_experiment_root
                        ),
                    )
                )
            if step == "inference":
                active_dataset_id = None
                active_dataset_ids = list(dataset_ids)
                write_pipeline_status(
                    experiment_root,
                    experiment_id=experiment_id,
                    run_plan_sha256=run_plan_sha256,
                    state="running",
                    start_at=args.start_at,
                    stop_after=args.stop_after,
                    completed=completed_records(),
                    active_dataset_ids=active_dataset_ids,
                    active_step=step,
                )
                run(
                    "cafe.inference.runner",
                    model_major_inference_arguments(
                        args,
                        dataset_ids,
                        experiment_root=experiment_root,
                    ),
                )
                for dataset_id in dataset_ids:
                    completed_steps_by_dataset[dataset_id].append(step)
                active_dataset_ids = None
                continue

            for dataset_id in dataset_ids:
                active_dataset_id = dataset_id
                active_dataset_ids = None
                write_pipeline_status(
                    experiment_root,
                    experiment_id=experiment_id,
                    run_plan_sha256=run_plan_sha256,
                    state="running",
                    start_at=args.start_at,
                    stop_after=args.stop_after,
                    completed=completed_records(),
                    active_dataset_id=dataset_id,
                    active_step=step,
                )
                commands = commands_for_dataset(
                    args,
                    dataset_id,
                    experiment_root=experiment_root,
                )
                script, arguments = commands[step]
                run(script, arguments)
                completed_steps_by_dataset[dataset_id].append(step)
            if step == "analysis":
                active_dataset_id = None
                active_step = "experiment_analysis"
                run_experiment_analysis(
                    args,
                    experiment_root=experiment_root,
                )
    except Exception as error:
        write_pipeline_status(
            experiment_root,
            experiment_id=experiment_id,
            run_plan_sha256=run_plan_sha256,
            state="failed",
            start_at=args.start_at,
            stop_after=args.stop_after,
            completed=completed_records(),
            active_dataset_id=active_dataset_id,
            active_dataset_ids=active_dataset_ids,
            active_step=active_step,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    completed = completed_records()
    write_pipeline_status(
        experiment_root,
        experiment_id=experiment_id,
        run_plan_sha256=run_plan_sha256,
        state="complete",
        start_at=args.start_at,
        stop_after=args.stop_after,
        completed=completed,
    )
    print(
        protocol.canonical_json(
            {
                "experiment_id": experiment_id,
                "run_plan_sha256": run_plan_sha256,
                "dataset_count": len(dataset_ids),
                "output": str(experiment_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
