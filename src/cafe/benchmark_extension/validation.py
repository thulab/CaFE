from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from cafe import core as protocol
from cafe.benchmark_extension.generation import (
    CONTRACT_SCHEMA,
    GENERATION_SCHEMA,
    PIPELINE_SCHEMA,
    _compact_record_batch,
    _parallel_work_batches,
    compact_contract_row,
    materialized_samples_for_instance,
)
from cafe.benchmark_extension.gift_eval import iter_gift_eval_instances
from cafe.benchmark_extension.mechanisms import (
    COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO,
    MECHANISM_EFFECT_MINIMUM_MASE_RMS,
    NONLINEAR_MAXIMUM_FUTURE_PEAK_FRACTION,
    NONLINEAR_MAXIMUM_TAIL_TO_PEAK_RATIO,
    NONLINEAR_FUTURE_INNOVATION_MINIMUM_BLOCK_LENGTH,
    NONLINEAR_FUTURE_INNOVATION_PATH_COUNT,
    NONLINEAR_MINIMUM_COEFFICIENT_ABS,
    NONLINEAR_MINIMUM_FUTURE_PROFILE_RANGE,
    NONLINEAR_MINIMUM_HALF_COEFFICIENT_RATIO,
    NONLINEAR_MINIMUM_HOLDOUT_R2_GAIN,
    NONLINEAR_MINIMUM_MULTISTEP_HOLDOUT_R2_GAIN,
    NONLINEAR_MULTISTEP_AUDIT_ORIGIN_COUNT,
    NONLINEAR_PERSISTENCE_INTERVALS,
    NONLINEAR_STABILITY_LIMIT,
    SOURCE_DISTANCE_MAXIMUM_CHANNEL,
    SOURCE_DISTANCE_MAXIMUM_MACRO,
    SOURCE_DISTANCE_MINIMUM_MACRO,
    SOURCE_DISTANCE_MODEL_MAX_CONTEXTS,
    STRICT_FUTURE_EFFECT_CAPABILITIES,
    TVS_ENVELOPE_ACTIVE_AMPLITUDE_FRACTION,
    TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION,
)
from cafe.benchmark_extension.storage import (
    iter_compact_parquet,
    validate_parquet_record,
)


VALIDATION_SCHEMA = "cafe.benchmark_extension_validation.v11"
VALIDATION_MODES = ("research", "publication")
DEFAULT_VALIDATION_WORKERS = max(1, min(8, os.cpu_count() or 1))
MAX_RECORDED_FAILURES = 100
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate compact GIFT-Eval capability contracts."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    parser.add_argument(
        "--mode",
        choices=VALIDATION_MODES,
        default="research",
        help=(
            "research scans every stored treatment distance gate; publication "
            "also verifies hashes and exactly replays every contract"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_VALIDATION_WORKERS,
        help="Process workers used for Parquet row groups or publication replay.",
    )
    return parser.parse_args()


def _next_or_none(iterator: Any) -> dict[str, Any] | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def _distance_gate_reason(row: dict[str, Any]) -> str | None:
    gate = row.get("source_distance_gate")
    if not isinstance(gate, dict):
        return "source_distance_gate_missing"
    if gate.get("schema_version") != "cafe.treatment_source_distance_gate.v3":
        return "source_distance_gate_schema"
    if gate.get("scope") != "treatment_history_vs_authentic_official_history":
        return "source_distance_gate_scope"
    if gate.get("treatment_only") is not True:
        return "source_distance_gate_not_treatment_only"
    if gate.get("strength_reference") != (
        "full_official_history_macro_normalized_rms"
    ):
        return "source_distance_gate_strength_reference"
    try:
        required = float(gate["minimum_required_macro_distance"])
        maximum_macro = float(gate["maximum_allowed_macro_distance"])
        maximum_channel = float(gate["maximum_allowed_channel_distance"])
        observed_minimum = float(gate["minimum_observed_macro_distance"])
        observed_maximum = float(gate["maximum_observed_macro_distance"])
        observed_channel_maximum = float(gate["maximum_observed_channel_distance"])
        full_context = int(gate["full_history_context_length"])
        full_macro = float(gate["full_history_macro_normalized_rms"])
    except (KeyError, TypeError, ValueError):
        return "source_distance_gate_invalid_distance"
    expected_thresholds = (
        (required, SOURCE_DISTANCE_MINIMUM_MACRO),
        (maximum_macro, SOURCE_DISTANCE_MAXIMUM_MACRO),
        (maximum_channel, SOURCE_DISTANCE_MAXIMUM_CHANNEL),
    )
    if any(
        not math.isfinite(value)
        or not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
        for value, expected in expected_thresholds
    ):
        return "source_distance_gate_threshold"
    if full_context <= 0 or not _finite_nonnegative(full_macro):
        return "source_distance_gate_full_history_invalid"
    if gate.get("model_max_contexts") != SOURCE_DISTANCE_MODEL_MAX_CONTEXTS:
        return "source_distance_gate_model_context_policy"
    by_context = gate.get("by_model_context")
    if not isinstance(by_context, list) or not by_context:
        return "source_distance_gate_contexts_missing"
    context_macros: list[float] = []
    context_channel_maxima: list[float] = []
    observed_contexts: list[int] = []
    observed_model_ids: set[str] = set()
    for context_row in by_context:
        if not isinstance(context_row, dict):
            return "source_distance_gate_context_invalid"
        context = int(context_row.get("context_length") or 0)
        if context <= 0 or context > full_context:
            return "source_distance_gate_context_invalid"
        model_ids = context_row.get("model_ids")
        if not isinstance(model_ids, list) or not model_ids:
            return "source_distance_gate_model_ids_missing"
        if any(not isinstance(model_id, str) for model_id in model_ids):
            return "source_distance_gate_model_ids_invalid"
        if model_ids != sorted(set(model_ids)):
            return "source_distance_gate_model_ids_order"
        if any(
            min(full_context, int(SOURCE_DISTANCE_MODEL_MAX_CONTEXTS.get(model_id, -1)))
            != context
            for model_id in model_ids
        ):
            return "source_distance_gate_model_context_mismatch"
        observed_model_ids.update(model_ids)
        observed_contexts.append(context)
        macro = context_row.get("macro_normalized_rms")
        channels = context_row.get("channel_normalized_rms")
        if not _finite_nonnegative(macro):
            return "source_distance_gate_context_macro_invalid"
        if not isinstance(channels, list) or not channels:
            return "source_distance_gate_channels_missing"
        if not all(_finite_nonnegative(value) for value in channels):
            return "source_distance_gate_channel_invalid"
        calculated_macro = sum(float(value) for value in channels) / len(channels)
        if not math.isclose(
            float(macro), calculated_macro, rel_tol=1e-9, abs_tol=1e-12
        ):
            return "source_distance_gate_context_macro_mismatch"
        context_macros.append(float(macro))
        context_channel_maxima.append(max(float(value) for value in channels))
    if observed_model_ids != set(SOURCE_DISTANCE_MODEL_MAX_CONTEXTS):
        return "source_distance_gate_model_coverage"
    if observed_contexts != sorted(set(observed_contexts)):
        return "source_distance_gate_context_order"
    if gate.get("evaluated_model_contexts") != observed_contexts:
        return "source_distance_gate_context_list_mismatch"
    full_channels = gate.get("full_history_channel_normalized_rms")
    if not isinstance(full_channels, list) or not full_channels:
        return "source_distance_gate_full_channels_missing"
    if not all(_finite_nonnegative(value) for value in full_channels):
        return "source_distance_gate_full_channel_invalid"
    if not math.isclose(
        full_macro,
        sum(float(value) for value in full_channels) / len(full_channels),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        return "source_distance_gate_full_macro_mismatch"
    if not math.isclose(
        observed_minimum,
        min(context_macros),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        return "source_distance_gate_minimum_mismatch"
    if not math.isclose(
        observed_maximum, max(context_macros), rel_tol=1e-9, abs_tol=1e-12
    ):
        return "source_distance_gate_maximum_mismatch"
    if not math.isclose(
        observed_channel_maximum,
        max(context_channel_maxima),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        return "source_distance_gate_channel_maximum_mismatch"
    if observed_minimum < required - 1e-12:
        return "source_distance_below_minimum"
    if observed_maximum > maximum_macro + 1e-12:
        return "source_distance_above_macro_maximum"
    if observed_channel_maximum > maximum_channel + 1e-12:
        return "source_distance_above_channel_maximum"
    if gate.get("accepted") is not True or gate.get("reason") is not None:
        return "source_distance_rejected"
    return None


def _mechanism_scoring_gate_reason(row: dict[str, Any]) -> str | None:
    gate = row.get("mechanism_scoring_gate")
    if not isinstance(gate, dict):
        return "mechanism_scoring_gate_missing"
    if gate.get("schema_version") != "cafe.mechanism_scoring_gate.v1":
        return "mechanism_scoring_gate_schema"
    if gate.get("metric") != "observed_affected_future_mase_standardized_rms":
        return "mechanism_scoring_gate_metric"
    try:
        required = float(gate["minimum_required_mase_rms"])
        mase_rms = float(gate["truth_effect_mase_rms"])
        raw_rms = float(gate["truth_effect_raw_rms"])
        observed_count = int(gate["observed_future_cell_count"])
    except (KeyError, TypeError, ValueError):
        return "mechanism_scoring_gate_values"
    if not math.isclose(
        required,
        MECHANISM_EFFECT_MINIMUM_MASE_RMS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "mechanism_scoring_gate_threshold"
    if not _finite_nonnegative(mase_rms) or not _finite_nonnegative(raw_rms):
        return "mechanism_scoring_gate_nonfinite"
    expected_accepted = observed_count > 0 and mase_rms >= required - 1e-12
    if gate.get("accepted") != expected_accepted:
        return "mechanism_scoring_gate_status"
    expected_reason = None
    if observed_count <= 0:
        expected_reason = "no_observed_affected_future_cell"
    elif not expected_accepted:
        expected_reason = "future_truth_effect_below_minimum"
    if gate.get("reason") != expected_reason:
        return "mechanism_scoring_gate_reason"
    if bool(row.get("included_in_capability_ranking")) != expected_accepted:
        return "mechanism_scoring_gate_ranking_flag"
    if (
        row.get("capability_id") in STRICT_FUTURE_EFFECT_CAPABILITIES
        and not expected_accepted
    ):
        return f"{row.get('capability_id')}_future_effect_not_scoreable"
    return None


def _horizon_support_gate_reason(row: dict[str, Any]) -> str | None:
    capability_id = row.get("capability_id")
    gate = row.get("horizon_support_gate")
    if capability_id not in {
        "time_varying_seasonality",
        "nonlinear_persistence",
        "common_factor",
    }:
        return None if gate is None else "horizon_support_gate_not_applicable"
    if not isinstance(gate, dict):
        return "horizon_support_gate_missing"
    if gate.get("schema_version") != "cafe.capability_horizon_support_gate.v1":
        return "horizon_support_gate_schema"
    if gate.get("capability_id") != capability_id:
        return "horizon_support_gate_capability"
    if gate.get("target_future_values_used") is not False:
        return "horizon_support_gate_future_leakage"
    if gate.get("accepted") is not True or gate.get("reason") is not None:
        return "horizon_support_gate_rejected"
    if capability_id == "time_varying_seasonality":
        if gate.get("metric") != (
            "future_envelope_active_fraction_by_affected_target"
        ):
            return "horizon_support_gate_metric"
        if gate.get("horizon_partition") != "whole_forecast_horizon":
            return "horizon_support_gate_partition"
        try:
            active_threshold = float(gate["active_amplitude_fraction"])
            required = float(gate["minimum_required_active_fraction"])
            observed_minimum = float(gate["minimum_observed_active_fraction"])
        except (KeyError, TypeError, ValueError):
            return "horizon_support_gate_values"
        if not math.isclose(
            active_threshold,
            TVS_ENVELOPE_ACTIVE_AMPLITUDE_FRACTION,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            required,
            TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return "horizon_support_gate_threshold"
        by_target = gate.get("by_target")
        if not isinstance(by_target, dict) or not by_target:
            return "horizon_support_gate_targets"
        fractions: list[float] = []
        for target in by_target.values():
            if not isinstance(target, dict):
                return "horizon_support_gate_target_values"
            try:
                observed = int(target["observed_future_count"])
                active = int(target["active_future_count"])
                fraction = float(target["active_fraction"])
            except (KeyError, TypeError, ValueError):
                return "horizon_support_gate_target_values"
            if observed <= 0 or active < 0 or active > observed:
                return "horizon_support_gate_target_counts"
            if not math.isclose(
                fraction,
                active / observed,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                return "horizon_support_gate_target_fraction"
            fractions.append(fraction)
        if not math.isclose(
            observed_minimum,
            min(fractions),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return "horizon_support_gate_minimum_mismatch"
        if observed_minimum < required - 1e-12:
            return "horizon_support_gate_below_minimum"
        return None
    if capability_id == "nonlinear_persistence":
        if gate.get("metric") != (
            "innovation_marginalized_future_effect_decay_profile"
        ):
            return "horizon_support_gate_metric"
        if gate.get("horizon_partition") != "whole_forecast_horizon":
            return "horizon_support_gate_partition"
        try:
            required_range = float(gate["minimum_required_relative_range"])
            maximum_peak_fraction = float(
                gate["maximum_allowed_peak_fraction"]
            )
            maximum_tail_ratio = float(
                gate["maximum_allowed_tail_peak_ratio"]
            )
            observed_range = float(gate["observed_relative_range"])
            peak_index = int(gate["observed_peak_index"])
            peak = float(gate["observed_peak_history_scale"])
            tail_ratio = float(gate["observed_tail_peak_ratio"])
        except (KeyError, TypeError, ValueError):
            return "horizon_support_gate_values"
        expected_thresholds = (
            math.isclose(
                required_range,
                NONLINEAR_MINIMUM_FUTURE_PROFILE_RANGE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                maximum_peak_fraction,
                NONLINEAR_MAXIMUM_FUTURE_PEAK_FRACTION,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                maximum_tail_ratio,
                NONLINEAR_MAXIMUM_TAIL_TO_PEAK_RATIO,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if not expected_thresholds:
            return "horizon_support_gate_threshold"
        horizon = int(row.get("horizon", 0))
        latest_peak = int(
            math.floor(maximum_peak_fraction * max(0, horizon - 1))
        )
        if (
            peak <= 0.0
            or peak_index < 0
            or peak_index > latest_peak
            or observed_range < required_range - 1e-12
            or tail_ratio > maximum_tail_ratio + 1e-12
        ):
            return "horizon_support_gate_nonlinear_decay"
        return None
    if gate.get("metric") != (
        "common_factor_tail_to_head_macro_normalized_rms_ratio"
    ):
        return "horizon_support_gate_metric"
    if gate.get("horizon_partition") != "three_equal_relative_sections":
        return "horizon_support_gate_partition"
    try:
        required = float(gate["minimum_required_tail_head_ratio"])
        observed = float(gate["observed_tail_head_ratio"])
    except (KeyError, TypeError, ValueError):
        return "horizon_support_gate_values"
    if not math.isclose(
        required,
        COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "horizon_support_gate_threshold"
    by_section = gate.get("by_section")
    if not isinstance(by_section, dict) or set(by_section) != {
        "head",
        "middle",
        "tail",
    }:
        return "horizon_support_gate_sections"
    macros: dict[str, float] = {}
    for name, section in by_section.items():
        if not isinstance(section, dict):
            return "horizon_support_gate_section_values"
        try:
            time_count = int(section["time_count"])
            target_count = int(section["observed_target_count"])
            macro = float(section["macro_normalized_rms"])
        except (KeyError, TypeError, ValueError):
            return "horizon_support_gate_section_values"
        if time_count <= 0 or target_count <= 0 or not _finite_nonnegative(macro):
            return "horizon_support_gate_section_values"
        macros[name] = macro
    calculated = macros["tail"] / max(macros["head"], 1e-12)
    if not math.isclose(
        observed, calculated, rel_tol=1e-9, abs_tol=1e-12
    ):
        return "horizon_support_gate_ratio_mismatch"
    if macros["head"] <= 0.0 or macros["tail"] <= 0.0:
        return "horizon_support_gate_zero_endpoint"
    if observed < required - 1e-12:
        return "horizon_support_gate_below_minimum"
    return None


def _nonlinear_identifiability_gate_reason(row: dict[str, Any]) -> str | None:
    if row.get("capability_id") != "nonlinear_persistence":
        return None
    group = row.get("group_metadata")
    metadata = row.get("mechanism_metadata")
    if not isinstance(group, dict) or not isinstance(metadata, dict):
        return "nonlinear_metadata_missing"
    gate = group.get("nonlinear_identifiability_gate")
    if not isinstance(gate, dict):
        return "nonlinear_identifiability_gate_missing"
    if gate.get("schema_version") != "cafe.nonlinear_identifiability_gate.v2":
        return "nonlinear_identifiability_gate_schema"
    if gate.get("method") != (
        "blocked_suffix_one_step_and_rolling_multistep_"
        "nonlinear_vs_linear_ar1"
    ):
        return "nonlinear_identifiability_gate_method"
    if (
        gate.get("target_future_values_used") is not False
        or gate.get("accepted") is not True
        or gate.get("reason") is not None
    ):
        return "nonlinear_identifiability_gate_status"
    affected = [int(value) for value in row.get("affected_target_indices") or []]
    if affected != [int(value) for value in gate.get("affected_target_indices") or []]:
        return "nonlinear_identifiability_gate_targets"
    diagnostics = gate.get("diagnostics_by_target")
    if not isinstance(diagnostics, dict) or not affected:
        return "nonlinear_identifiability_gate_diagnostics"
    for channel in affected:
        audit = diagnostics.get(str(channel))
        if not isinstance(audit, dict) or audit.get("accepted") is not True:
            return "nonlinear_identifiability_target_rejected"
        try:
            gain = float(audit["holdout_incremental_r2"])
            required_gain = float(
                audit["minimum_required_holdout_incremental_r2"]
            )
            minimum_coefficient = float(
                audit["minimum_required_coefficient_abs"]
            )
            coefficients = [
                float(audit["training_nonlinear_coefficient"]),
                float(audit["first_half_nonlinear_coefficient"]),
                float(audit["second_half_nonlinear_coefficient"]),
                float(audit["full_history_nonlinear_coefficient"]),
            ]
            half_ratio = float(audit["half_coefficient_magnitude_ratio"])
            required_half_ratio = float(
                audit["minimum_required_half_coefficient_ratio"]
            )
            linear_persistence = float(
                audit["linear_persistence_coefficient"]
            )
            direction = float(audit["nonlinear_direction"])
            stability_limit = float(audit["stability_limit"])
            headroom = float(audit["stability_headroom"])
            multistep = audit["multistep_holdout"]
            multistep_gain = float(multistep["incremental_r2"])
            required_multistep_gain = float(
                multistep["minimum_required_incremental_r2"]
            )
            multistep_origin_count = int(multistep["origin_count"])
            multistep_horizon = int(multistep["forecast_horizon"])
            multistep_linear_mse = float(multistep["linear_mse"])
            multistep_nonlinear_mse = float(multistep["nonlinear_mse"])
            multistep_bootstrap = multistep["innovation_bootstrap"]
        except (KeyError, TypeError, ValueError):
            return "nonlinear_identifiability_target_values"
        if (
            not math.isclose(
                required_gain,
                NONLINEAR_MINIMUM_HOLDOUT_R2_GAIN,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                minimum_coefficient,
                NONLINEAR_MINIMUM_COEFFICIENT_ABS,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                required_half_ratio,
                NONLINEAR_MINIMUM_HALF_COEFFICIENT_RATIO,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                stability_limit,
                NONLINEAR_STABILITY_LIMIT,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                required_multistep_gain,
                NONLINEAR_MINIMUM_MULTISTEP_HOLDOUT_R2_GAIN,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return "nonlinear_identifiability_target_threshold"
        signs = {math.copysign(1.0, value) for value in coefficients}
        calculated_multistep_gain = (
            0.0
            if multistep_linear_mse <= 1e-12
            else 1.0 - multistep_nonlinear_mse / multistep_linear_mse
        )
        if (
            gain < required_gain - 1e-12
            or any(abs(value) < minimum_coefficient - 1e-12 for value in coefficients)
            or len(signs) != 1
            or audit.get("coefficient_sign_stable") is not True
            or half_ratio < required_half_ratio - 1e-12
            or abs(linear_persistence) >= stability_limit
            or direction not in {-1.0, 1.0}
            or headroom <= 0.0
            or not isinstance(multistep, dict)
            or multistep.get("accepted") is not True
            or multistep.get("reason") is not None
            or multistep_gain <= required_multistep_gain + 1e-12
            or multistep_origin_count < 1
            or multistep_origin_count > NONLINEAR_MULTISTEP_AUDIT_ORIGIN_COUNT
            or multistep_horizon < 1
            or not math.isclose(
                multistep_gain,
                calculated_multistep_gain,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or not isinstance(multistep_bootstrap, dict)
            or int(multistep_bootstrap.get("path_count", -1))
            != NONLINEAR_FUTURE_INNOVATION_PATH_COUNT
            or multistep_bootstrap.get("method")
            != "centered_circular_moving_block_bootstrap"
            or multistep_bootstrap.get("innovation_pool")
            != "linear_skeleton_training_prefix_residuals"
        ):
            return "nonlinear_identifiability_target_invalid"
    distances = group.get("full_history_distance_by_level")
    if (
        not isinstance(distances, list)
        or len(distances) != 5
        or any(not _finite_nonnegative(value) for value in distances)
        or any(
            float(current) <= float(previous) + 1e-12
            for previous, current in zip(distances, distances[1:])
        )
    ):
        return "nonlinear_level_distance_not_monotone"
    if (
        group.get("future_estimand")
        != "paired_conditional_mean_over_shared_bootstrapped_innovations"
        or metadata.get("component")
        != "same_innovation_state_dependent_persistence_recurrence"
        or metadata.get("state_response") != "z_abs_z_over_one_plus_abs_z"
        or metadata.get("future_innovation_policy")
        != "history_innovation_marginalized_shared_path_mean"
        or metadata.get("target_future_used_for_delta") is not False
        or row.get("controlled_coordinate")
        != "stable_persistence_headroom_fraction"
        or not math.isclose(
            float(row.get("applied_component_gain", float("nan"))),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return "nonlinear_treatment_semantics"
    level_details = metadata.get("level_diagnostics_by_target")
    if not isinstance(level_details, dict):
        return "nonlinear_level_diagnostics_missing"
    bootstrap_by_target = metadata.get("future_innovation_bootstrap_by_target")
    if not isinstance(bootstrap_by_target, dict):
        return "nonlinear_future_bootstrap_missing"
    coordinate = float(row["sampled_coordinate"])
    level = int(row.get("capability_level", 0))
    if level < 1 or level > len(NONLINEAR_PERSISTENCE_INTERVALS):
        return "nonlinear_level_invalid"
    expected_interval = NONLINEAR_PERSISTENCE_INTERVALS[level - 1]
    stored_interval = row.get("coordinate_interval")
    if (
        not isinstance(stored_interval, list)
        or len(stored_interval) != 2
        or not all(
            math.isclose(
                float(stored), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
            for stored, expected in zip(
                stored_interval, expected_interval, strict=True
            )
        )
        or coordinate < expected_interval[0] - 1e-12
        or coordinate > expected_interval[1] + 1e-12
    ):
        return "nonlinear_level_coordinate"
    for channel in affected:
        audit = diagnostics[str(channel)]
        detail = level_details.get(str(channel))
        if not isinstance(detail, dict):
            return "nonlinear_level_diagnostics_missing"
        expected = (
            float(audit["nonlinear_direction"])
            * coordinate
            * float(audit["stability_headroom"])
        )
        coefficient = float(detail["nonlinear_persistence_coefficient"])
        effective = float(detail["effective_extreme_persistence_limit"])
        if not math.isclose(
            coefficient, expected, rel_tol=1e-9, abs_tol=1e-12
        ) or abs(effective) >= NONLINEAR_STABILITY_LIMIT:
            return "nonlinear_level_coefficient_mismatch"
        bootstrap = bootstrap_by_target.get(str(channel))
        if not isinstance(bootstrap, dict):
            return "nonlinear_future_bootstrap_missing"
        expected_block_length = min(
            int(audit["transition_count"]),
            max(
                NONLINEAR_FUTURE_INNOVATION_MINIMUM_BLOCK_LENGTH,
                int(math.ceil(math.sqrt(int(row.get("horizon", 0))))),
            ),
        )
        try:
            path_count = int(bootstrap["path_count"])
            block_length = int(bootstrap["block_length"])
            seed = int(bootstrap["seed"])
        except (KeyError, TypeError, ValueError):
            return "nonlinear_future_bootstrap_values"
        if (
            bootstrap.get("schema_version")
            != "cafe.nonlinear_innovation_bootstrap.v1"
            or bootstrap.get("method")
            != "centered_circular_moving_block_bootstrap"
            or bootstrap.get("innovation_pool")
            != "linear_skeleton_full_history_residuals"
            or bootstrap.get("aggregation") != "paired_path_mean"
            or bootstrap.get("shared_across_linear_and_nonlinear_branches")
            is not True
            or bootstrap.get("ensemble_centered_at_each_horizon_step") is not True
            or bootstrap.get("target_future_values_used") is not False
            or path_count != NONLINEAR_FUTURE_INNOVATION_PATH_COUNT
            or block_length != expected_block_length
            or seed < 0
            or int(detail.get("future_innovation_path_count", -1)) != path_count
        ):
            return "nonlinear_future_bootstrap_invalid"
    return None


def _treatment_contract_reason(row: dict[str, Any]) -> str | None:
    return (
        _distance_gate_reason(row)
        or _nonlinear_identifiability_gate_reason(row)
        or _horizon_support_gate_reason(row)
        or _mechanism_scoring_gate_reason(row)
    )


def _scan_treatment_row_group(
    work: tuple[str, int],
) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    path_string, row_group_index = work
    parquet = pq.ParquetFile(path_string)
    table = parquet.read_row_group(row_group_index, columns=("payload_json",))
    payloads = table.column(0).to_pylist()
    failures: list[dict[str, Any]] = []
    failure_count = 0
    horizon_support_count = 0
    nonlinear_identifiability_count = 0
    for payload in payloads:
        sample_id: Any = None
        try:
            row = json.loads(str(payload))
            if not isinstance(row, dict):
                raise TypeError("payload is not an object")
            sample_id = row.get("sample_id")
            horizon_support_count += int(
                row.get("horizon_support_gate") is not None
            )
            nonlinear_identifiability_count += int(
                row.get("capability_id") == "nonlinear_persistence"
            )
            reason = _treatment_contract_reason(row)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            reason = f"source_distance_payload:{error}"
        if reason is not None:
            failure_count += 1
            if len(failures) < MAX_RECORDED_FAILURES:
                failures.append(
                    {
                        "scope": "capability_treatments",
                        "sample_id": sample_id,
                        "reason": reason,
                    }
                )
    return (
        len(payloads),
        horizon_support_count,
        nonlinear_identifiability_count,
        failure_count,
        failures,
    )


def _research_validation(
    manifest: dict[str, Any],
    *,
    workers: int,
) -> tuple[dict[str, int], int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    failure_count = 0
    counts = {
        "official_baselines": int(manifest.get("official_instance_count", 0)),
        "capability_treatments": 0,
        "horizon_support_gates": 0,
        "nonlinear_identifiability_gates": 0,
        "input_ablations": int(manifest.get("input_ablation_count", 0)),
        "availability": int(
            ((manifest.get("files") or {}).get("availability") or {}).get(
                "row_count", 0
            )
        ),
    }
    try:
        treatment_path = Path(
            str(manifest["files"]["capability_treatments"]["path"])
        )
        parquet = pq.ParquetFile(treatment_path)
        work = [
            (str(treatment_path), row_group_index)
            for row_group_index in range(parquet.num_row_groups)
        ]
        if int(workers) > 1 and len(work) > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                results = executor.map(_scan_treatment_row_group, work)
                for count, support_count, nonlinear_count, rejected, rows in results:
                    counts["capability_treatments"] += count
                    counts["horizon_support_gates"] += support_count
                    counts["nonlinear_identifiability_gates"] += nonlinear_count
                    failure_count += rejected
                    remaining = MAX_RECORDED_FAILURES - len(failures)
                    failures.extend(rows[: max(0, remaining)])
        else:
            for item in work:
                count, support_count, nonlinear_count, rejected, rows = (
                    _scan_treatment_row_group(item)
                )
                counts["capability_treatments"] += count
                counts["horizon_support_gates"] += support_count
                counts["nonlinear_identifiability_gates"] += nonlinear_count
                failure_count += rejected
                remaining = MAX_RECORDED_FAILURES - len(failures)
                failures.extend(rows[: max(0, remaining)])
    except Exception as error:
        # A malformed or unreadable treatment artifact must yield a rejected
        # report rather than letting inference proceed without a gate audit.
        failure_count += 1
        failures.append(
            {
                "scope": "capability_treatments",
                "sample_id": None,
                "reason": f"source_distance_scan:{error}",
            }
        )
    return counts, failure_count, failures


def _publication_expected_batches(
    manifest: dict[str, Any],
    gift_root: Path,
    workers: int,
) -> Iterator[dict[str, list[dict[str, Any]]]]:
    config = manifest["config"]
    shard_size = int(config.get("generation_execution", {}).get("shard_size", 256))
    if int(workers) <= 1:
        for instance_index, instance in enumerate(
            iter_gift_eval_instances(
                str(config["dataset_id"]),
                gift_root,
                term=str(config["term"]),
                max_instances=config.get("max_instances"),
            )
        ):
            output = {
                "official_baselines": [],
                "capability_treatments": [],
                "input_ablations": [],
                "availability": [],
            }
            for kind, dense_row in materialized_samples_for_instance(
                instance,
                augmentation_seed=int(config["augmentation_seed"]),
                capability_ids=tuple(str(value) for value in config["capability_ids"]),
                source_shard_index=instance_index // max(1, shard_size),
            ):
                output[kind].append(compact_contract_row(dense_row))
            yield output
        return

    batches = iter(
        _parallel_work_batches(
            str(config["dataset_id"]),
            gift_eval_dir=gift_root,
            term=str(config["term"]),
            augmentation_seed=int(config["augmentation_seed"]),
            capability_ids=tuple(str(value) for value in config["capability_ids"]),
            max_instances=config.get("max_instances"),
            shard_size=shard_size,
        )
    )
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        pending: list[Any] = []
        for _ in range(max(1, int(workers) * 2)):
            try:
                pending.append(executor.submit(_compact_record_batch, next(batches)))
            except StopIteration:
                break
        while pending:
            future = pending.pop(0)
            yield future.result()
            try:
                pending.append(executor.submit(_compact_record_batch, next(batches)))
            except StopIteration:
                pass


def _publication_validation(
    manifest: dict[str, Any],
    *,
    gift_root: Path,
    workers: int,
) -> tuple[dict[str, int], int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    config = manifest.get("config")
    if manifest.get("schema_version") != GENERATION_SCHEMA:
        failures.append({"scope": "manifest", "reason": "schema_version"})
    if not isinstance(config, dict) or config.get("pipeline_schema_version") != PIPELINE_SCHEMA:
        failures.append({"scope": "manifest", "reason": "pipeline_schema"})
    elif manifest.get("config_sha256") != protocol.json_sha256(config):
        failures.append({"scope": "manifest", "reason": "config_hash"})
    if isinstance(config, dict):
        storage = config.get("artifact_storage")
        if (
            not isinstance(storage, dict)
            or storage.get("format") != "parquet"
            or storage.get("dense_targets_stored") is not False
            or storage.get("dense_covariates_stored") is not False
        ):
            failures.append({"scope": "manifest", "reason": "dense_storage_policy"})

    artifact_keys = (
        "official_baselines",
        "capability_treatments",
        "input_ablations",
        "availability",
    )
    paths: dict[str, Path] = {}
    for key in artifact_keys:
        try:
            paths[key] = validate_parquet_record(manifest["files"][key])
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            failures.append({"scope": "manifest", "reason": f"{key}:{error}"})
    for record in manifest.get("source_files") or []:
        try:
            source_path = Path(str(record["path"]))
            if protocol.file_sha256(source_path) != record["sha256"]:
                raise ValueError("source_hash")
        except (KeyError, OSError, ValueError) as error:
            failures.append({"scope": "manifest", "reason": f"source:{error}"})

    counts = {key: 0 for key in artifact_keys}
    horizon_support_count = 0
    nonlinear_identifiability_count = 0
    if not failures and isinstance(config, dict):
        observed = {
            key: iter(iter_compact_parquet(path)) for key, path in paths.items()
        }
        for batch in _publication_expected_batches(manifest, gift_root, workers):
            for kind in artifact_keys:
                for expected in batch[kind]:
                    actual = _next_or_none(observed[kind])
                    counts[kind] += 1
                    if actual is None:
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": expected.get("sample_id"),
                                "reason": "missing_compact_contract",
                            }
                        )
                        continue
                    if actual.get("schema_version") != CONTRACT_SCHEMA:
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": actual.get("sample_id"),
                                "reason": "contract_schema",
                            }
                        )
                    if any(
                        field in actual
                        for field in ("target", "covariates", "future_observed_mask")
                    ):
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": actual.get("sample_id"),
                                "reason": "dense_payload_present",
                            }
                        )
                    if protocol.canonical_json(actual) != protocol.canonical_json(expected):
                        failures.append(
                            {
                                "scope": kind,
                                "sample_id": expected.get("sample_id"),
                                "reason": "deterministic_replay_mismatch",
                            }
                        )
                    if kind == "capability_treatments":
                        horizon_support_count += int(
                            expected.get("horizon_support_gate") is not None
                        )
                        nonlinear_identifiability_count += int(
                            expected.get("capability_id")
                            == "nonlinear_persistence"
                        )
                        reason = _treatment_contract_reason(expected)
                        if reason is not None:
                            failures.append(
                                {
                                    "scope": kind,
                                    "sample_id": expected.get("sample_id"),
                                    "reason": reason,
                                }
                            )
        for kind, iterator in observed.items():
            extra = _next_or_none(iterator)
            if extra is not None:
                failures.append(
                    {
                        "scope": kind,
                        "sample_id": extra.get("sample_id"),
                        "reason": "unexpected_compact_contract",
                    }
                )
        for kind, count in counts.items():
            declared = int((manifest.get("files", {}).get(kind) or {}).get("row_count", -1))
            if count != declared:
                failures.append(
                    {
                        "scope": "manifest",
                        "reason": f"{kind}_count:{count}!={declared}",
                    }
                )
    counts["horizon_support_gates"] = horizon_support_count
    counts["nonlinear_identifiability_gates"] = (
        nonlinear_identifiability_count
    )
    return counts, len(failures), failures[:MAX_RECORDED_FAILURES]


def validate_generation(
    dataset_root: Path,
    *,
    gift_eval_dir: Path | None = None,
    mode: str = "research",
    workers: int = DEFAULT_VALIDATION_WORKERS,
) -> dict[str, Any]:
    """Validate treatment gates, with exact replay reserved for publication."""

    if mode not in VALIDATION_MODES:
        raise ValueError(f"unsupported validation mode: {mode}")
    if int(workers) < 1:
        raise ValueError("validation workers must be positive")
    manifest_path = dataset_root / "01_generation" / "manifest.json"
    manifest = protocol.read_json(manifest_path)
    if mode == "publication":
        config = manifest.get("config") or {}
        gift_root = (
            Path(str(config.get("gift_eval_source_root"))).resolve()
            if gift_eval_dir is None and config.get("gift_eval_source_root")
            else (
                protocol.REPO_ROOT / "data" / "gift-eval"
                if gift_eval_dir is None
                else gift_eval_dir.resolve()
            )
        )
        counts, failure_count, failures = _publication_validation(
            manifest,
            gift_root=gift_root,
            workers=int(workers),
        )
        policy = "publication_full_hash_and_exact_source_replay_v1"
    else:
        counts, failure_count, failures = _research_validation(
            manifest,
            workers=int(workers),
        )
        policy = (
            "research_all_treatment_source_distance_and_mechanism_scoring_"
            "capability_horizon_support_and_nonlinear_identifiability_gates_v5"
        )

    report = {
        "schema_version": VALIDATION_SCHEMA,
        "created_at": protocol.utc_now(),
        "dataset_id": manifest.get("dataset_id"),
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "generation_manifest_sha256": protocol.file_sha256(manifest_path),
        "validation_mode": mode,
        "validation_policy": policy,
        "validation_workers": int(workers),
        "accepted": failure_count == 0,
        "official_baseline_count": counts["official_baselines"],
        "treatment_count": counts["capability_treatments"],
        "input_ablation_count": counts["input_ablations"],
        "availability_count": counts["availability"],
        "source_distance_gate_checked_count": counts["capability_treatments"],
        "horizon_support_gate_checked_count": counts.get(
            "horizon_support_gates", 0
        ),
        "nonlinear_identifiability_gate_checked_count": counts.get(
            "nonlinear_identifiability_gates", 0
        ),
        "mechanism_scoring_gate_checked_count": counts[
            "capability_treatments"
        ],
        "failure_count": int(failure_count),
        "failures_truncated": failure_count > len(failures),
        "failures": failures,
    }
    protocol.write_json(dataset_root / "02_validation" / "report.json", report)
    return report


def main() -> int:
    args = parse_args()
    dataset_root = args.output_root.resolve() / args.dataset_id
    report_path = dataset_root / "02_validation" / "report.json"
    if report_path.exists():
        raise FileExistsError(
            f"validation artifact already exists; use a new experiment root: {report_path}"
        )
    report = validate_generation(
        dataset_root,
        gift_eval_dir=args.gift_eval_dir,
        mode=args.mode,
        workers=args.workers,
    )
    print(protocol.canonical_json(report))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
