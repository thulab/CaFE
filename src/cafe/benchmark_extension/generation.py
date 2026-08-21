from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.gift_eval import (
    GIFT_EVAL_ADAPTER_SCHEMA,
    GIFT_EVAL_SOURCE_REVISION,
    GiftEvalInstance,
    gift_arrow_target_summary,
    gift_eval_asset_path,
    gift_eval_instances_for_record,
    iter_gift_arrow_records,
    iter_gift_eval_instances,
    official_window_count_from_minimum_length,
    prediction_length,
)
from cafe.benchmark_extension.mechanisms import (
    CAPABILITY_IDS,
    COMMON_FACTOR_MINIMUM_HARMONIC_SHARE,
    COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO,
    MECHANISM_EFFECT_MINIMUM_MASE_RMS,
    NONLINEAR_EXTREME_STATE_MINIMUM_ABS,
    NONLINEAR_HOLDOUT_FRACTION,
    NONLINEAR_MAXIMUM_FUTURE_PEAK_FRACTION,
    NONLINEAR_MAXIMUM_TAIL_TO_PEAK_RATIO,
    NONLINEAR_MINIMUM_FUTURE_PROFILE_RANGE,
    NONLINEAR_MINIMUM_HISTORY,
    NONLINEAR_MINIMUM_HOLDOUT_R2_GAIN,
    NONLINEAR_ORDINARY_STATE_MAXIMUM_ABS,
    NONLINEAR_PERSISTENCE_INTERVALS,
    NONLINEAR_STABILITY_LIMIT,
    SOURCE_DISTANCE_MAXIMUM_CHANNEL,
    SOURCE_DISTANCE_MAXIMUM_MACRO,
    SOURCE_DISTANCE_MINIMUM_MACRO,
    SOURCE_DISTANCE_MODEL_MAX_CONTEXTS,
    TVS_ENVELOPE_ACTIVE_AMPLITUDE_FRACTION,
    TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION,
    TVS_MINIMUM_INCREMENTAL_R2,
    CapabilityGroup,
    CapabilityTreatment,
    build_capability_group,
    mechanism_effect_signal,
    replay_treatment_deltas,
)
from cafe.benchmark_extension.storage import (
    CompactParquetWriter,
    iter_compact_parquet,
    parquet_file_record,
)


PIPELINE_SCHEMA = "cafe.pipeline.v10"
GENERATION_SCHEMA = "cafe.benchmark_extension_generation.v8"
SAMPLE_SCHEMA = "cafe.benchmark_extension_sample.v7"
CONTRACT_SCHEMA = "cafe.benchmark_extension_contract.v5"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend official GIFT-Eval instances with capability treatments."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    parser.add_argument("--term", choices=("short", "medium", "long"), default="short")
    parser.add_argument("--augmentation-seed", type=int, default=2026081601)
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=CAPABILITY_IDS,
        default=list(CAPABILITY_IDS),
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Non-formal source-order prefix for smoke tests.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=256)
    return parser.parse_args()


def _target_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _mase_scale(history: np.ndarray, period: int) -> tuple[float, list[float]]:
    values = np.asarray(history, dtype=float)
    lag = min(max(1, int(period)), max(1, values.shape[0] - 1))
    differences = np.abs(values[lag:] - values[:-lag])
    by_target = np.mean(differences, axis=0) if differences.size else np.ones(values.shape[1])
    fallback = np.mean(np.abs(np.diff(values, axis=0)), axis=0)
    by_target = np.where(np.isfinite(by_target) & (by_target > 1e-8), by_target, fallback)
    by_target = np.where(np.isfinite(by_target) & (by_target > 1e-8), by_target, 1.0)
    return float(np.mean(by_target)), [float(value) for value in by_target]


def _mechanism_scoring_gate(
    future_delta: np.ndarray,
    instance: GiftEvalInstance,
    mase_scale_by_target: list[float],
    affected_target_indices: tuple[int, ...],
) -> dict[str, Any]:
    raw_rms, mase_rms, observed_count = mechanism_effect_signal(
        future_delta,
        instance.future_observed_mask,
        np.asarray(mase_scale_by_target, dtype=float),
        affected_target_indices,
    )
    accepted = (
        observed_count > 0
        and mase_rms >= MECHANISM_EFFECT_MINIMUM_MASE_RMS - 1e-12
    )
    reason = None
    if observed_count == 0:
        reason = "no_observed_affected_future_cell"
    elif not accepted:
        reason = "future_truth_effect_below_minimum"
    return {
        "schema_version": "cafe.mechanism_scoring_gate.v1",
        "metric": "observed_affected_future_mase_standardized_rms",
        "scope": "treatment_future_delta_vs_authentic_official_future",
        "minimum_required_mase_rms": MECHANISM_EFFECT_MINIMUM_MASE_RMS,
        "observed_future_cell_count": observed_count,
        "truth_effect_raw_rms": raw_rms,
        "truth_effect_mase_rms": mase_rms,
        "target_future_values_used_for_parameter_selection": False,
        "accepted": accepted,
        "reason": reason,
    }


def _season_length(frequency: str) -> int:
    raw = str(frequency)
    if raw.endswith(("H", "h")):
        return 24
    if raw.endswith("D"):
        return 7
    if raw.endswith(("M", "ME")):
        return 12
    if raw.endswith("W") or raw.startswith("W-"):
        return 52
    return 1


def _baseline_row(instance: GiftEvalInstance) -> dict[str, Any]:
    target = np.vstack((instance.history, instance.future))
    covariates = np.vstack(
        (instance.history_covariates, instance.future_covariates)
    )
    season = _season_length(instance.frequency)
    mase, mase_by_target = _mase_scale(instance.history, season)
    return {
        "schema_version": SAMPLE_SCHEMA,
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "benchmark_track": "gift_eval_capability_extension",
        "evaluation_table": "gift_eval_official_baseline",
        "sample_id": f"{instance.official_instance_id}__baseline",
        "official_instance_id": instance.official_instance_id,
        "baseline_sample_id": None,
        "counterfactual_pair_id": None,
        "counterfactual_member": 0,
        "dataset_id": instance.dataset_id,
        "config_id": instance.config_id,
        "item_id": instance.item_id,
        "window_index": instance.window_index,
        "window_count": instance.window_count,
        "forecast_origin": instance.forecast_origin,
        "source_target_length": instance.source_target_length,
        "capability_id": None,
        "capability_level": 0,
        "augmentation_seed": None,
        "context_length": instance.context_length,
        "horizon": instance.prediction_length,
        "target_dim": instance.target_dim,
        "target_column_names": list(instance.target_column_names),
        "covariate_dim": int(covariates.shape[1]),
        "covariate_column_names": list(instance.covariate_column_names),
        "covariate_availability": list(instance.covariate_availability),
        "future_covariate_visible": list(instance.future_covariate_visible),
        "covariates": covariates if covariates.shape[1] else None,
        "frequency": instance.frequency,
        "term": instance.term,
        "season_length": season,
        "target": target,
        "future_observed_mask": instance.future_observed_mask,
        "history_imputation": instance.history_imputation,
        "mase_scale": mase,
        "mase_scale_by_target": mase_by_target,
        "mase_period": season,
        "target_sha256": _target_sha256(target),
        "history_sha256": _target_sha256(instance.history),
        "future_sha256": _target_sha256(instance.future),
        "scoring_target_semantics": "gift_eval_official_future",
        "input_history_semantics": "gift_eval_official_history_after_history_only_imputation",
        "included_in_capability_ranking": False,
    }


def _treatment_row(
    instance: GiftEvalInstance,
    group: CapabilityGroup,
    treatment: CapabilityTreatment,
    *,
    augmentation_seed: int,
) -> dict[str, Any]:
    baseline_id = f"{instance.official_instance_id}__baseline"
    pair_id = (
        f"{instance.official_instance_id}__{group.capability_id}__"
        f"level{treatment.level}__aug{augmentation_seed}"
    )
    history = instance.history + treatment.history_delta
    future = instance.future + treatment.future_delta
    stored_history_delta = history - instance.history
    stored_future_delta = future - instance.future
    target = np.vstack((history, future))
    history_covariates = (
        instance.history_covariates + treatment.history_covariate_delta
    )
    future_covariates = (
        instance.future_covariates + treatment.future_covariate_delta
    )
    covariates = np.vstack((history_covariates, future_covariates))
    season = _season_length(instance.frequency)
    mase, mase_by_target = _mase_scale(instance.history, season)
    mechanism_gate = _mechanism_scoring_gate(
        stored_future_delta,
        instance,
        mase_by_target,
        treatment.affected_target_indices,
    )
    parameter_draw = {
        "capability_level": treatment.level,
        "controlled_coordinate": treatment.controlled_coordinate,
        "coordinate_interval": list(treatment.coordinate_interval),
        "sampled_coordinate": treatment.sampled_coordinate,
        "applied_component_gain": treatment.applied_component_gain,
        "augmentation_seed": int(augmentation_seed),
    }
    return {
        "schema_version": SAMPLE_SCHEMA,
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "benchmark_track": "gift_eval_capability_extension",
        "evaluation_table": "gift_eval_capability_treatment",
        "sample_id": f"{pair_id}__treatment",
        "official_instance_id": instance.official_instance_id,
        "baseline_sample_id": baseline_id,
        "counterfactual_pair_id": pair_id,
        "counterfactual_member": 1,
        "dataset_id": instance.dataset_id,
        "config_id": instance.config_id,
        "item_id": instance.item_id,
        "window_index": instance.window_index,
        "window_count": instance.window_count,
        "forecast_origin": instance.forecast_origin,
        "source_target_length": instance.source_target_length,
        "capability_id": group.capability_id,
        "capability_level": treatment.level,
        "augmentation_seed": int(augmentation_seed),
        "controlled_coordinate": treatment.controlled_coordinate,
        "coordinate_interval": list(treatment.coordinate_interval),
        "sampled_coordinate": treatment.sampled_coordinate,
        "applied_component_gain": treatment.applied_component_gain,
        "parameter_draw_sha256": protocol.json_sha256(parameter_draw),
        "context_length": instance.context_length,
        "horizon": instance.prediction_length,
        "target_dim": instance.target_dim,
        "target_column_names": list(instance.target_column_names),
        "affected_target_indices": list(treatment.affected_target_indices),
        "covariate_dim": int(covariates.shape[1]),
        "covariate_column_names": list(instance.covariate_column_names),
        "covariate_availability": list(instance.covariate_availability),
        "future_covariate_visible": list(instance.future_covariate_visible),
        "covariates": covariates if covariates.shape[1] else None,
        "frequency": instance.frequency,
        "term": instance.term,
        "season_length": season,
        "target": target,
        "future_observed_mask": instance.future_observed_mask,
        "history_imputation": instance.history_imputation,
        "mase_scale": mase,
        "mase_scale_by_target": mase_by_target,
        "mase_period": season,
        "target_sha256": _target_sha256(target),
        "history_sha256": _target_sha256(history),
        "future_sha256": _target_sha256(future),
        "source_history_sha256": _target_sha256(instance.history),
        "source_future_sha256": _target_sha256(instance.future),
        "history_delta_sha256": _target_sha256(stored_history_delta),
        "future_delta_sha256": _target_sha256(stored_future_delta),
        "covariate_sha256": _target_sha256(covariates),
        "history_covariate_delta_sha256": _target_sha256(
            treatment.history_covariate_delta
        ),
        "future_covariate_delta_sha256": _target_sha256(
            treatment.future_covariate_delta
        ),
        "source_distance_gate": treatment.source_distance_gate,
        "horizon_support_gate": treatment.horizon_support_gate,
        "mechanism_scoring_gate": mechanism_gate,
        "anti_copy_gate": {
            "policy": "treatment_to_authentic_source_distance_v1",
            "status": "accepted",
            "treatment_only": True,
        },
        "mechanism_metadata": treatment.metadata,
        "group_metadata": group.group_metadata,
        "scoring_target_semantics": (
            "gift_eval_official_future_plus_history_only_capability_delta"
        ),
        "input_history_semantics": (
            "entire_gift_eval_official_history_plus_capability_treatment"
        ),
        "included_in_capability_ranking": bool(mechanism_gate["accepted"]),
    }


def _least_aligned_circular_shift(
    values: np.ndarray,
    *,
    official_instance_id: str,
    capability_id: str,
    capability_level: int,
    augmentation_seed: int,
    channel: int,
) -> tuple[np.ndarray, int, float | None]:
    """Return a deterministic, marginal-preserving temporal donor.

    Candidate shifts come only from the model-visible history.  Choosing the
    least aligned candidate avoids accidentally retaining a dominant period,
    while circular shifting preserves the exact empirical marginal scale.
    """

    source = np.asarray(values, dtype=float)
    length = int(source.size)
    if length < 4:
        raise ValueError("input_ablation_requires_history_length_at_least_four")
    rng = np.random.default_rng(
        protocol.stable_seed(
            official_instance_id,
            capability_id,
            capability_level,
            channel,
            "input_ablation",
            base=int(augmentation_seed),
        )
    )
    lower = max(1, length // 8)
    upper = max(lower + 1, length - lower)
    candidate_count = min(24, max(1, upper - lower))
    candidates = sorted(
        {
            int(value)
            for value in rng.integers(lower, upper, size=candidate_count)
            if int(value) % length
        }
    )
    if not candidates:
        candidates = [max(1, length // 2)]
    source_std = float(np.std(source))
    best: tuple[float, int, np.ndarray, float | None] | None = None
    for shift in candidates:
        shifted = np.roll(source, shift)
        shifted_std = float(np.std(shifted))
        correlation = (
            None
            if source_std <= 1e-12 or shifted_std <= 1e-12
            else float(np.corrcoef(source, shifted)[0, 1])
        )
        score = 0.0 if correlation is None else abs(correlation)
        candidate = (score, shift, shifted, correlation)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], int(best[1]), best[3]


def _input_ablation_row(
    instance: GiftEvalInstance,
    group: CapabilityGroup,
    treatment: CapabilityTreatment,
    treatment_row: dict[str, Any],
    *,
    augmentation_seed: int,
) -> dict[str, Any] | None:
    """Build the mandatory auxiliary-input attribution task.

    The assessed target history and the complete scored future stay identical
    to the main treatment.  Only auxiliary histories are temporally misaligned,
    so the row measures whether a model actually uses cross-channel inputs.
    """

    if group.capability_id not in {
        "common_factor",
        "cross_series_dependence",
        "covariate_impulse_response",
    }:
        return None
    metadata = treatment.metadata
    if group.capability_id == "covariate_impulse_response":
        assessed = (int(metadata["eligible_target_index"]),)
        covariate = int(metadata["covariate_index"])
        impulse_delta = np.asarray(
            treatment.history_covariate_delta[:, covariate], dtype=float
        )
        shifted, shift, correlation = _least_aligned_circular_shift(
            impulse_delta,
            official_instance_id=instance.official_instance_id,
            capability_id=group.capability_id,
            capability_level=treatment.level,
            augmentation_seed=augmentation_seed,
            channel=covariate,
        )
        source_covariates = np.asarray(treatment_row["covariates"], dtype=float)
        result_covariates = source_covariates.copy()
        context = int(treatment_row["context_length"])
        result_covariates[:context, covariate] = (
            instance.history_covariates[:, covariate] + shifted
        )
        target = np.asarray(treatment_row["target"], dtype=float)
        row = dict(treatment_row)
        row.update(
            {
                "evaluation_table": "gift_eval_capability_input_ablation",
                "sample_id": f"{treatment_row['sample_id']}__input_ablation",
                "counterfactual_pair_id": (
                    f"{treatment_row['counterfactual_pair_id']}__input_ablation"
                ),
                "counterfactual_member": 2,
                "input_ablation_source_sample_id": treatment_row["sample_id"],
                "input_ablation_source_target_sha256": treatment_row["target_sha256"],
                "assessed_target_indices": list(assessed),
                "ablated_input_indices": [covariate],
                "target": target.copy(),
                "covariates": result_covariates,
                "target_sha256": treatment_row["target_sha256"],
                "history_sha256": treatment_row["history_sha256"],
                "future_sha256": treatment_row["future_sha256"],
                "covariate_sha256": _target_sha256(result_covariates),
                "input_ablation_delta_sha256": _target_sha256(
                    result_covariates - source_covariates
                ),
                "source_distance_gate": {
                    "policy": "not_applicable_auxiliary_input_ablation",
                    "status": "not_applicable",
                },
                "anti_copy_gate": {
                    "policy": "not_applicable_auxiliary_input_ablation",
                    "status": "not_applicable",
                },
                "input_ablation_metadata": {
                    "policy": "shift_only_constructed_covariate_impulse_v1",
                    "assessed_target_history_unchanged": True,
                    "scored_future_unchanged": True,
                    "authentic_covariate_path_unchanged": True,
                    "channel_audit": {
                        str(covariate): {
                            "circular_shift": shift,
                            "absolute_alignment_correlation": (
                                None if correlation is None else abs(correlation)
                            ),
                            "source_impulse_delta_sha256": _target_sha256(
                                impulse_delta
                            ),
                            "ablated_impulse_delta_sha256": _target_sha256(shifted),
                        }
                    },
                },
                "included_in_capability_ranking": False,
                "excluded_from_primary_score": True,
            }
        )
        return row
    if group.capability_id == "cross_series_dependence":
        assessed = (int(metadata["responder_target_index"]),)
        ablated = (int(metadata["driver_target_index"]),)
    else:
        loading = np.asarray(metadata["loading"], dtype=float)
        assessed = (int(np.argmax(np.abs(loading))),)
        ablated = tuple(index for index in range(instance.target_dim) if index not in assessed)
    if not ablated:
        raise ValueError("input_ablation_has_no_auxiliary_channel")

    target = np.asarray(treatment_row["target"], dtype=float)
    context = int(treatment_row["context_length"])
    result = target.copy()
    channel_audit: dict[str, Any] = {}
    for channel in ablated:
        source = target[:context, channel]
        shifted, shift, correlation = _least_aligned_circular_shift(
            source,
            official_instance_id=instance.official_instance_id,
            capability_id=group.capability_id,
            capability_level=treatment.level,
            augmentation_seed=augmentation_seed,
            channel=channel,
        )
        result[:context, channel] = shifted
        channel_audit[str(channel)] = {
            "circular_shift": shift,
            "absolute_alignment_correlation": (
                None if correlation is None else abs(correlation)
            ),
            "source_history_sha256": _target_sha256(source),
            "ablated_history_sha256": _target_sha256(shifted),
            "source_mean": float(np.mean(source)),
            "source_std": float(np.std(source)),
            "ablated_mean": float(np.mean(shifted)),
            "ablated_std": float(np.std(shifted)),
        }
    np.testing.assert_array_equal(
        result[:context, list(assessed)], target[:context, list(assessed)]
    )
    np.testing.assert_array_equal(result[context:], target[context:])

    row = dict(treatment_row)
    row.update(
        {
            "evaluation_table": "gift_eval_capability_input_ablation",
            "sample_id": f"{treatment_row['sample_id']}__input_ablation",
            "counterfactual_pair_id": (
                f"{treatment_row['counterfactual_pair_id']}__input_ablation"
            ),
            "counterfactual_member": 2,
            "input_ablation_source_sample_id": treatment_row["sample_id"],
            "input_ablation_source_target_sha256": treatment_row["target_sha256"],
            "assessed_target_indices": list(assessed),
            "ablated_input_indices": list(ablated),
            "target": result,
            "target_sha256": _target_sha256(result),
            "history_sha256": _target_sha256(result[:context]),
            "future_sha256": _target_sha256(result[context:]),
            "history_delta_sha256": _target_sha256(
                result[:context] - instance.history
            ),
            "future_delta_sha256": _target_sha256(
                result[context:] - instance.future
            ),
            "input_ablation_delta_sha256": _target_sha256(
                result[:context] - target[:context]
            ),
            "source_distance_gate": {
                "policy": "not_applicable_auxiliary_input_ablation",
                "status": "not_applicable",
            },
            "anti_copy_gate": {
                "policy": "not_applicable_auxiliary_input_ablation",
                "status": "not_applicable",
            },
            "input_ablation_metadata": {
                "policy": "deterministic_least_aligned_circular_shift_v1",
                "assessed_target_history_unchanged": True,
                "scored_future_unchanged": True,
                "empirical_marginal_preserved": True,
                "channel_audit": channel_audit,
            },
            "included_in_capability_ranking": False,
            "excluded_from_primary_score": True,
        }
    )
    return row


def _availability_row(
    instance: GiftEvalInstance,
    group: CapabilityGroup,
) -> dict[str, Any]:
    return {
        "schema_version": "cafe.instance_capability_availability.v1",
        "dataset_id": instance.dataset_id,
        "official_instance_id": instance.official_instance_id,
        "capability_id": group.capability_id,
        "available": group.available,
        "reason": group.reason,
        "generated_level_count": len(group.treatments),
        "context_length": instance.context_length,
        "horizon": instance.prediction_length,
        "target_dim": instance.target_dim,
        "group_metadata": group.group_metadata,
    }


_DENSE_GENERATION_FIELDS = {
    "target",
    "covariates",
    "future_observed_mask",
}


def compact_contract_row(row: dict[str, Any]) -> dict[str, Any]:
    """Remove replayable dense arrays from a generated sample row."""

    compact = {
        key: value for key, value in row.items() if key not in _DENSE_GENERATION_FIELDS
    }
    compact.update(
        {
            "schema_version": CONTRACT_SCHEMA,
            "source_sample_schema_version": row.get("schema_version"),
            "record_kind": row.get("evaluation_table"),
            "dense_payload_policy": (
                "source_arrow_reference_plus_deterministic_mechanism_replay"
            ),
        }
    )
    return compact


def materialized_samples_for_instance(
    instance: GiftEvalInstance,
    *,
    augmentation_seed: int,
    capability_ids: tuple[str, ...],
    source_shard_index: int | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield dense rows transiently; callers decide whether to send or score."""

    def tagged(row: dict[str, Any]) -> dict[str, Any]:
        if source_shard_index is not None:
            row["source_shard_index"] = int(source_shard_index)
        return row

    yield "official_baselines", tagged(_baseline_row(instance))
    for capability_id in capability_ids:
        for kind, row in _materialized_capability_rows(
            instance,
            capability_id,
            augmentation_seed=augmentation_seed,
        ):
            yield kind, tagged(row)


def _materialized_capability_rows(
    instance: GiftEvalInstance,
    capability_id: str,
    *,
    augmentation_seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    group = build_capability_group(
        instance,
        capability_id,
        augmentation_seed=augmentation_seed,
    )
    rows: list[tuple[str, dict[str, Any]]] = [
        ("availability", _availability_row(instance, group))
    ]
    for treatment in group.treatments:
        treatment_row = _treatment_row(
            instance,
            group,
            treatment,
            augmentation_seed=augmentation_seed,
        )
        rows.append(("capability_treatments", treatment_row))
        ablation_row = _input_ablation_row(
            instance,
            group,
            treatment,
            treatment_row,
            augmentation_seed=augmentation_seed,
        )
        if ablation_row is not None:
            rows.append(("input_ablations", ablation_row))
    return rows


def iter_replayed_samples(
    manifest: dict[str, Any],
    *,
    gift_eval_dir: Path,
    replay_workers: int = 1,
) -> Iterator[dict[str, Any]]:
    """Reconstruct dense rows from frozen compact contracts and source Arrow.

    Capability selection and parameter fitting happen only in generation.  This
    path streams the compact Parquet contracts, applies their frozen mechanism
    parameters to authentic instances, and prefetches independent instances.
    """

    workers = max(1, int(replay_workers))
    if workers == 1:
        for work in iter_replay_contract_work_items(
            manifest, gift_eval_dir=gift_eval_dir
        ):
            yield from _replay_contract_instance(*work)
        return
    iterator = iter(
        iter_replay_contract_work_items(manifest, gift_eval_dir=gift_eval_dir)
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Any] = deque()
        for _index in range(workers):
            work = next(iterator, None)
            if work is None:
                break
            pending.append(executor.submit(_replay_contract_instance, *work))
        while pending:
            yield from pending.popleft().result()
            work = next(iterator, None)
            if work is not None:
                pending.append(executor.submit(_replay_contract_instance, *work))


ReplayContractWorkItem = tuple[
    GiftEvalInstance,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]


def iter_replay_contract_work_items(
    manifest: dict[str, Any],
    *,
    gift_eval_dir: Path,
) -> Iterator[ReplayContractWorkItem]:
    """Join official instances with their compact contracts exactly once."""

    config = manifest["config"]
    files = manifest["files"]
    baseline_rows = iter(iter_compact_parquet(Path(files["official_baselines"]["path"])))
    treatment_groups = iter(
        _group_contract_rows(
            iter_compact_parquet(Path(files["capability_treatments"]["path"]))
        )
    )
    ablation_groups = iter(
        _group_contract_rows(
            iter_compact_parquet(Path(files["input_ablations"]["path"]))
        )
    )
    treatment_group = next(treatment_groups, None)
    ablation_group = next(ablation_groups, None)

    for instance in iter_gift_eval_instances(
        str(config["dataset_id"]),
        gift_eval_dir,
        term=str(config["term"]),
        max_instances=config.get("max_instances"),
    ):
        baseline_contract = next(baseline_rows, None)
        if baseline_contract is None:
            raise ValueError("official baseline contract stream ended early")
        official_id = str(instance.official_instance_id)
        if str(baseline_contract["official_instance_id"]) != official_id:
            raise ValueError("official baseline contract order mismatch")
        treatments: list[dict[str, Any]] = []
        if treatment_group is not None and treatment_group[0] == official_id:
            treatments = treatment_group[1]
            treatment_group = next(treatment_groups, None)
        ablations: list[dict[str, Any]] = []
        if ablation_group is not None and ablation_group[0] == official_id:
            ablations = ablation_group[1]
            ablation_group = next(ablation_groups, None)
        yield instance, baseline_contract, treatments, ablations
    if next(baseline_rows, None) is not None:
        raise ValueError("official baseline contract stream has extra rows")
    if treatment_group is not None or next(treatment_groups, None) is not None:
        raise ValueError("treatment contract stream has unmatched rows")
    if ablation_group is not None or next(ablation_groups, None) is not None:
        raise ValueError("ablation contract stream has unmatched rows")


def _group_contract_rows(
    rows: Iterator[dict[str, Any]],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    for row in rows:
        official_id = str(row["official_instance_id"])
        if current_id is None:
            current_id = official_id
        elif official_id != current_id:
            yield current_id, current
            current_id = official_id
            current = []
        current.append(row)
    if current_id is not None:
        yield current_id, current


def _dense_contract_row(
    contract: dict[str, Any],
    *,
    target: np.ndarray,
    covariates: np.ndarray,
    future_observed_mask: np.ndarray,
) -> dict[str, Any]:
    row = dict(contract)
    source_schema = row.pop("source_sample_schema_version", SAMPLE_SCHEMA)
    row.pop("record_kind", None)
    row.pop("dense_payload_policy", None)
    row.update(
        {
            "schema_version": source_schema,
            "target": np.asarray(target, dtype=float),
            "covariates": (
                np.asarray(covariates, dtype=float)
                if covariates.shape[1]
                else None
            ),
            "future_observed_mask": np.asarray(
                future_observed_mask, dtype=bool
            ),
        }
    )
    return row


def _replay_contract_instance(
    instance: GiftEvalInstance,
    baseline_contract: dict[str, Any],
    treatment_contracts: list[dict[str, Any]],
    ablation_contracts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    covariates = np.vstack(
        (instance.history_covariates, instance.future_covariates)
    )
    baseline = _dense_contract_row(
        baseline_contract,
        target=np.vstack((instance.history, instance.future)),
        covariates=covariates,
        future_observed_mask=instance.future_observed_mask,
    )
    output = [baseline]
    treatments_by_sample: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(treatment_contracts):
        capability_id = str(treatment_contracts[index]["capability_id"])
        stop = index + 1
        while (
            stop < len(treatment_contracts)
            and str(treatment_contracts[stop]["capability_id"])
            == capability_id
        ):
            stop += 1
        contracts = treatment_contracts[index:stop]
        deltas = replay_treatment_deltas(instance, contracts)
        for contract in contracts:
            (
                history_delta,
                future_delta,
                history_covariate_delta,
                future_covariate_delta,
            ) = deltas[str(contract["sample_id"])]
            target = np.vstack(
                (
                    instance.history + history_delta,
                    instance.future + future_delta,
                )
            )
            treatment_covariates = np.vstack(
                (
                    instance.history_covariates + history_covariate_delta,
                    instance.future_covariates + future_covariate_delta,
                )
            )
            row = _dense_contract_row(
                contract,
                target=target,
                covariates=treatment_covariates,
                future_observed_mask=instance.future_observed_mask,
            )
            treatments_by_sample[str(row["sample_id"])] = row
            output.append(row)
        index = stop
    for contract in ablation_contracts:
        source_id = str(contract["input_ablation_source_sample_id"])
        source = treatments_by_sample.get(source_id)
        if source is None:
            raise ValueError("ablation contract has no matching treatment")
        target = np.asarray(source["target"], dtype=float).copy()
        ablated_covariates = np.asarray(source["covariates"], dtype=float).copy()
        context = int(contract["context_length"])
        audit = contract["input_ablation_metadata"]["channel_audit"]
        if str(contract["capability_id"]) == "covariate_impulse_response":
            metadata = contract["mechanism_metadata"]
            channel = int(metadata["covariate_index"])
            shift = int(audit[str(channel)]["circular_shift"])
            source_impulse = (
                np.asarray(source["covariates"], dtype=float)[:context, channel]
                - instance.history_covariates[:, channel]
            )
            ablated_covariates[:context, channel] = (
                instance.history_covariates[:, channel]
                + np.roll(source_impulse, shift)
            )
        else:
            for raw_channel in contract["ablated_input_indices"]:
                channel = int(raw_channel)
                shift = int(audit[str(channel)]["circular_shift"])
                target[:context, channel] = np.roll(
                    target[:context, channel], shift
                )
        output.append(
            _dense_contract_row(
                contract,
                target=target,
                covariates=ablated_covariates,
                future_observed_mask=instance.future_observed_mask,
            )
        )
    return output


def _compact_record_batch(
    work: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Worker entry point: fit contracts for source records, return no arrays."""

    output = {
        "official_baselines": [],
        "capability_treatments": [],
        "input_ablations": [],
        "availability": [],
    }
    instance_index = int(work["start_instance_index"])
    for item in work["records"]:
        for instance in gift_eval_instances_for_record(
            dataset_id=str(work["dataset_id"]),
            config_id=str(work["config_id"]),
            item_id=str(item["item_id"]),
            frequency=str(work["frequency"]),
            term=str(work["term"]),
            raw_target=np.asarray(item["target"], dtype=float),
            prediction_length_value=int(work["horizon"]),
            window_count=int(work["window_count"]),
            maximum_windows=int(item["maximum_windows"]),
            raw_past_covariates=(
                None
                if item.get("past_covariates") is None
                else np.asarray(item["past_covariates"], dtype=float)
            ),
            raw_known_future_covariates=(
                None
                if item.get("known_future_covariates") is None
                else np.asarray(item["known_future_covariates"], dtype=float)
            ),
        ):
            for kind, row in materialized_samples_for_instance(
                instance,
                augmentation_seed=int(work["augmentation_seed"]),
                capability_ids=tuple(str(value) for value in work["capability_ids"]),
                source_shard_index=(
                    instance_index // max(1, int(work["source_shard_size"]))
                ),
            ):
                output[kind].append(compact_contract_row(row))
            instance_index += 1
    return output


def _parallel_work_batches(
    dataset_id: str,
    *,
    gift_eval_dir: Path,
    term: str,
    augmentation_seed: int,
    capability_ids: tuple[str, ...],
    max_instances: int | None,
    shard_size: int,
) -> Iterator[dict[str, Any]]:
    dataset = protocol.resolve_dataset(dataset_id)
    asset_path = gift_eval_asset_path(dataset_id, gift_eval_dir)
    frequency, minimum_length, _record_count = gift_arrow_target_summary(asset_path)
    horizon = prediction_length(dataset_id, frequency, term=term)
    windows = official_window_count_from_minimum_length(
        dataset_id, minimum_length, horizon
    )
    remaining = max_instances
    current: list[dict[str, Any]] = []
    current_instances = 0

    next_instance_index = 0

    def work(records: list[dict[str, Any]], start_index: int) -> dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "config_id": dataset.config_id,
            "frequency": frequency,
            "term": term,
            "horizon": horizon,
            "window_count": windows,
            "augmentation_seed": augmentation_seed,
            "capability_ids": capability_ids,
            "start_instance_index": int(start_index),
            "source_shard_size": int(shard_size),
            "records": records,
        }

    for record in iter_gift_arrow_records(asset_path):
        if record.frequency != frequency:
            raise ValueError("GIFT-Eval config must have one frequency")
        maximum = windows if remaining is None else min(windows, int(remaining))
        if maximum <= 0:
            break
        if current and current_instances + maximum > int(shard_size):
            yield work(current, next_instance_index)
            next_instance_index += current_instances
            current = []
            current_instances = 0
        item = {
            "item_id": str(record.item_id),
            "target": np.asarray(record.target, dtype=float),
            "past_covariates": record.past_covariates,
            "known_future_covariates": record.known_future_covariates,
            "maximum_windows": int(maximum),
        }
        current.append(item)
        current_instances += maximum
        if remaining is not None:
            remaining -= maximum
            if remaining <= 0:
                break
    if current:
        yield work(current, next_instance_index)


def generate_dataset(
    dataset_id: str,
    *,
    gift_eval_dir: Path,
    dataset_root: Path,
    term: str,
    augmentation_seed: int,
    capability_ids: tuple[str, ...],
    max_instances: int | None,
    workers: int = 1,
    shard_size: int = 256,
) -> dict[str, Any]:
    generation_dir = dataset_root / "01_generation"
    baseline_path = generation_dir / "official_instances.parquet"
    treatment_path = generation_dir / "treatment_contracts.parquet"
    ablation_path = generation_dir / "input_ablation_contracts.parquet"
    availability_path = generation_dir / "availability.parquet"
    paths = (baseline_path, treatment_path, ablation_path, availability_path)
    baseline_count = treatment_count = ablation_count = availability_count = 0
    instance_count = 0
    available_counts = {capability: 0 for capability in capability_ids}
    unavailable_reason_counts: dict[str, dict[str, int]] = {
        capability: {} for capability in capability_ids
    }
    writers = {
        "official_baselines": CompactParquetWriter(baseline_path),
        "capability_treatments": CompactParquetWriter(treatment_path),
        "input_ablations": CompactParquetWriter(ablation_path),
        "availability": CompactParquetWriter(availability_path),
    }

    def consume(kind: str, compact: dict[str, Any]) -> None:
        nonlocal baseline_count, treatment_count, ablation_count, availability_count
        writers[kind].write(compact)
        if kind == "official_baselines":
            baseline_count += 1
        elif kind == "capability_treatments":
            treatment_count += 1
        elif kind == "input_ablations":
            ablation_count += 1
        else:
            availability_count += 1
            if bool(compact["available"]):
                available_counts[str(compact["capability_id"])] += 1
            else:
                capability = str(compact["capability_id"])
                reason = str(compact["reason"])
                reasons = unavailable_reason_counts[capability]
                reasons[reason] = reasons.get(reason, 0) + 1

    try:
        if int(workers) > 1:
            batches = iter(
                _parallel_work_batches(
                dataset_id,
                gift_eval_dir=gift_eval_dir,
                term=term,
                augmentation_seed=augmentation_seed,
                capability_ids=capability_ids,
                max_instances=max_instances,
                shard_size=shard_size,
                )
            )
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                pending: list[Any] = []
                for _index in range(max(1, int(workers) * 2)):
                    try:
                        pending.append(executor.submit(_compact_record_batch, next(batches)))
                    except StopIteration:
                        break
                while pending:
                    future = pending.pop(0)
                    result = future.result()
                    for kind in (
                        "official_baselines",
                        "capability_treatments",
                        "input_ablations",
                        "availability",
                    ):
                        for compact in result[kind]:
                            consume(kind, compact)
                    try:
                        pending.append(executor.submit(_compact_record_batch, next(batches)))
                    except StopIteration:
                        pass
            instance_count = baseline_count
        else:
            for instance_index, instance in enumerate(iter_gift_eval_instances(
                dataset_id,
                gift_eval_dir,
                term=term,
                max_instances=max_instances,
            )):
                instance_count += 1
                for kind, row in materialized_samples_for_instance(
                    instance,
                    augmentation_seed=augmentation_seed,
                    capability_ids=capability_ids,
                    source_shard_index=instance_index // max(1, int(shard_size)),
                ):
                    consume(kind, compact_contract_row(row))
        for writer in writers.values():
            writer.close()
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise
    source_path = gift_eval_asset_path(dataset_id, gift_eval_dir)
    source_files = [
        {**protocol.file_record(path), "path": str(path.resolve())}
        for path in sorted(source_path.glob("data-*.arrow"))
    ]
    config = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "adapter_schema_version": GIFT_EVAL_ADAPTER_SCHEMA,
        "gift_eval_split_source": GIFT_EVAL_SOURCE_REVISION,
        "dataset_id": dataset_id,
        "gift_eval_source_root": str(gift_eval_dir.resolve()),
        "term": term,
        "augmentation_seed": int(augmentation_seed),
        "capability_ids": list(capability_ids),
        "max_instances": max_instances,
        "formal": max_instances is None,
        "instance_selection": (
            "all_official_test_instances"
            if max_instances is None
            else "nonformal_source_order_prefix"
        ),
        "native_target_policy": "preserve_gift_eval_target_dimension",
        "native_covariate_policy": (
            "preserve_source_fields_and_declared_future_visibility"
        ),
        "treatment_history_scope": "entire_official_input_history",
        "randomness_policy": (
            "counter_based_by_official_instance_capability_level_and_augmentation_seed"
        ),
        "source_distance_policy": (
            "full_history_strength_actual_model_context_bounds_v3"
        ),
        "mechanism_scoring_policy": {
            "metric": "observed_affected_future_mase_standardized_rms",
            "minimum_required_mase_rms": MECHANISM_EFFECT_MINIMUM_MASE_RMS,
            "low_signal_policy": "treatment_accuracy_retained_mechanism_score_unavailable",
        },
        "capability_horizon_support_policy": {
            "time_varying_seasonality": {
                "continuation": "history_fitted_constrained_am",
                "minimum_incremental_r2": TVS_MINIMUM_INCREMENTAL_R2,
                "active_amplitude_fraction": (
                    TVS_ENVELOPE_ACTIVE_AMPLITUDE_FRACTION
                ),
                "minimum_future_active_fraction": (
                    TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION
                ),
            },
            "common_factor": {
                "continuation": "history_fitted_stable_latent_harmonic",
                "minimum_latent_harmonic_share": (
                    COMMON_FACTOR_MINIMUM_HARMONIC_SHARE
                ),
                "minimum_tail_head_rms_ratio": (
                    COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO
                ),
            },
            "nonlinear_persistence": {
                "mechanism": (
                    "same_innovation_state_dependent_persistence_recurrence"
                ),
                "identifiability": "single_blocked_holdout_nonlinear_vs_linear_ar1",
                "holdout_fraction": NONLINEAR_HOLDOUT_FRACTION,
                "minimum_history_length": NONLINEAR_MINIMUM_HISTORY,
                "minimum_holdout_incremental_r2": (
                    NONLINEAR_MINIMUM_HOLDOUT_R2_GAIN
                ),
                "ordinary_state_maximum_abs": (
                    NONLINEAR_ORDINARY_STATE_MAXIMUM_ABS
                ),
                "extreme_state_minimum_abs": (
                    NONLINEAR_EXTREME_STATE_MINIMUM_ABS
                ),
                "stability_limit": NONLINEAR_STABILITY_LIMIT,
                "headroom_fraction_intervals": [
                    list(interval) for interval in NONLINEAR_PERSISTENCE_INTERVALS
                ],
                "minimum_future_profile_range": (
                    NONLINEAR_MINIMUM_FUTURE_PROFILE_RANGE
                ),
                "maximum_future_peak_fraction": (
                    NONLINEAR_MAXIMUM_FUTURE_PEAK_FRACTION
                ),
                "maximum_tail_peak_ratio": (
                    NONLINEAR_MAXIMUM_TAIL_TO_PEAK_RATIO
                ),
            },
            "covariate_impulse_response": {
                "continuation": "fixed_causal_kernel_from_visible_impulses",
                "future_energy": (
                    "minimum_0.05_mase_rms_for_all_levels_by_construction"
                ),
                "past_only_future_input": "omitted",
            },
            "other_capabilities": "capability_specific_or_not_applicable",
        },
        "source_distance_configuration": {
            "strength_reference": "full_official_history_macro_normalized_rms",
            "model_max_contexts": dict(SOURCE_DISTANCE_MODEL_MAX_CONTEXTS),
            "minimum_model_context_macro_distance": (
                SOURCE_DISTANCE_MINIMUM_MACRO
            ),
            "maximum_model_context_macro_distance": (
                SOURCE_DISTANCE_MAXIMUM_MACRO
            ),
            "maximum_model_context_channel_distance": (
                SOURCE_DISTANCE_MAXIMUM_CHANNEL
            ),
        },
        "input_ablation_policy": (
            "common_cross_auxiliary_or_covariate_impulse_alignment_shift_v2"
        ),
        "artifact_storage": {
            "format": "parquet",
            "compression": "zstd",
            "dense_targets_stored": False,
            "dense_covariates_stored": False,
            "replay_policy": "source_arrow_plus_deterministic_contract_v2",
        },
        "generation_execution": {
            "workers": int(workers),
            "shard_size": int(shard_size),
            "parallelism_status": (
                "enabled" if int(workers) > 1 else "single_worker"
            ),
        },
    }
    manifest = {
        "schema_version": GENERATION_SCHEMA,
        "created_at": protocol.utc_now(),
        "dataset_id": dataset_id,
        "config": config,
        "config_sha256": protocol.json_sha256(config),
        "source_files": source_files,
        "files": {
            "official_baselines": {
                **parquet_file_record(baseline_path, row_count=baseline_count),
            },
            "capability_treatments": {
                **parquet_file_record(treatment_path, row_count=treatment_count),
            },
            "input_ablations": {
                **parquet_file_record(ablation_path, row_count=ablation_count),
            },
            "availability": {
                **parquet_file_record(availability_path, row_count=availability_count),
            },
        },
        "official_instance_count": instance_count,
        "available_instance_count_by_capability": available_counts,
        "unavailable_reason_count_by_capability": {
            capability: dict(sorted(reasons.items()))
            for capability, reasons in unavailable_reason_counts.items()
        },
        "treatment_count": treatment_count,
        "input_ablation_count": ablation_count,
    }
    protocol.write_json(generation_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be positive")
    if len(args.capabilities) != len(set(args.capabilities)):
        raise ValueError("capabilities must be unique")
    dataset_root = args.output_root.resolve() / args.dataset_id
    manifest_path = dataset_root / "01_generation" / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"generation artifact already exists; use a new experiment root: {manifest_path}"
        )
    manifest = generate_dataset(
        args.dataset_id,
        gift_eval_dir=args.gift_eval_dir,
        dataset_root=dataset_root,
        term=args.term,
        augmentation_seed=args.augmentation_seed,
        capability_ids=tuple(args.capabilities),
        max_instances=args.max_instances,
        workers=args.workers,
        shard_size=args.shard_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
