from __future__ import annotations

import argparse
import math
import os
import shutil
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cafe import core as protocol
from cafe.benchmark_extension.generation import (
    PIPELINE_SCHEMA,
    ReplayContractWorkItem,
    _replay_contract_instance,
    iter_replay_contract_work_items,
)
from cafe.benchmark_extension.inference import INFERENCE_SCHEMA
from cafe.benchmark_extension.mechanisms import (
    MECHANISM_EFFECT_MINIMUM_MASE_RMS,
)
from cafe.benchmark_extension.storage import (
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL,
    TypedParquetWriter,
    iter_prediction_parquet,
    parquet_file_record,
    validate_parquet_record,
)


ANALYSIS_SCHEMA = "cafe.benchmark_extension_analysis.v12"
SUITE_ANALYSIS_SCHEMA = "cafe.benchmark_extension_suite_analysis.v1"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"

ACCURACY_METRIC_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("model_id", pa.string()),
        ("dataset_id", pa.string()),
        ("official_instance_id", pa.string()),
        ("sample_id", pa.string()),
        ("sample_kind", pa.string()),
        ("capability_id", pa.string()),
        ("capability_level", pa.int8()),
        ("mase", pa.float64()),
        ("mae", pa.float64()),
    ]
)
EFFECT_METRIC_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("model_id", pa.string()),
        ("dataset_id", pa.string()),
        ("official_instance_id", pa.string()),
        ("sample_id", pa.string()),
        ("capability_id", pa.string()),
        ("capability_level", pa.int8()),
        ("controlled_coordinate", pa.string()),
        ("sampled_coordinate", pa.float64()),
        ("effect_score_status", pa.string()),
        ("effect_nrmse", pa.float64()),
        ("effect_correlation", pa.float64()),
        ("effect_amplitude_ratio", pa.float64()),
        ("effect_decay_shape_nrmse", pa.float64()),
        ("effect_half_life_status", pa.string()),
        ("truth_effect_half_life", pa.float64()),
        ("forecast_effect_half_life", pa.float64()),
        ("effect_half_life_absolute_error", pa.float64()),
        ("truth_effect_rms", pa.float64()),
        ("truth_effect_mase_rms", pa.float64()),
        ("observed_effect_cell_count", pa.int64()),
        ("standardized_squared_error_sum", pa.float64()),
        ("standardized_truth_squared_sum", pa.float64()),
        ("standardized_forecast_squared_sum", pa.float64()),
        ("treatment_mase", pa.float64()),
        ("treatment_mae", pa.float64()),
    ]
)
ABLATION_METRIC_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("model_id", pa.string()),
        ("dataset_id", pa.string()),
        ("official_instance_id", pa.string()),
        ("sample_id", pa.string()),
        ("source_treatment_sample_id", pa.string()),
        ("capability_id", pa.string()),
        ("capability_level", pa.int8()),
        ("assessed_target_indices", pa.list_(pa.int32())),
        ("ablation_target_indices", pa.list_(pa.int32())),
        ("ablated_input_indices", pa.list_(pa.int32())),
        ("full_input_mase", pa.float64()),
        ("ablated_input_mase", pa.float64()),
        ("input_ablation_mase_degradation", pa.float64()),
        ("input_ablation_forecast_change_rms", pa.float64()),
        ("input_ablation_response_ratio", pa.float64()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse official accuracy and capability-treatment effects."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _future(row: dict[str, Any]) -> np.ndarray:
    target = np.asarray(row["target"], dtype=float)
    return target[int(row["context_length"]) :]


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _effect_measurement(
    truth_delta: np.ndarray,
    forecast_delta: np.ndarray,
    observed_mask: np.ndarray,
    affected_target_indices: Iterable[int],
    mase_scale_by_target: np.ndarray,
) -> dict[str, Any]:
    """Measure a paired forecast effect without inventing a denominator."""

    mask = np.asarray(observed_mask, dtype=bool)
    assessed = np.zeros_like(mask, dtype=bool)
    affected = [int(value) for value in affected_target_indices]
    assessed[:, affected] = mask[:, affected]
    observed_count = int(np.count_nonzero(assessed))
    if observed_count == 0:
        return {
            "status": "unavailable_no_observed_effect_cell",
            "observed_count": 0,
            "truth_raw_rms": 0.0,
            "truth_mase_rms": 0.0,
            "nrmse": None,
            "correlation": None,
            "amplitude_ratio": None,
            "squared_error_sum": 0.0,
            "truth_squared_sum": 0.0,
            "forecast_squared_sum": 0.0,
        }
    scales = np.asarray(mase_scale_by_target, dtype=float)
    truth = np.asarray(truth_delta, dtype=float)
    forecast = np.asarray(forecast_delta, dtype=float)
    truth_raw = truth[assessed]
    truth_standardized = (truth / scales[None, :])[assessed]
    forecast_standardized = (forecast / scales[None, :])[assessed]
    truth_raw_rms = float(np.sqrt(np.mean(np.square(truth_raw))))
    truth_mase_rms = float(
        np.sqrt(np.mean(np.square(truth_standardized)))
    )
    difference = forecast_standardized - truth_standardized
    squared_error_sum = float(np.sum(np.square(difference)))
    truth_squared_sum = float(np.sum(np.square(truth_standardized)))
    forecast_squared_sum = float(np.sum(np.square(forecast_standardized)))
    if truth_mase_rms < MECHANISM_EFFECT_MINIMUM_MASE_RMS - 1e-12:
        return {
            "status": "unavailable_low_truth_effect",
            "observed_count": observed_count,
            "truth_raw_rms": truth_raw_rms,
            "truth_mase_rms": truth_mase_rms,
            "nrmse": None,
            "correlation": None,
            "amplitude_ratio": None,
            "squared_error_sum": squared_error_sum,
            "truth_squared_sum": truth_squared_sum,
            "forecast_squared_sum": forecast_squared_sum,
        }
    return {
        "status": "scored",
        "observed_count": observed_count,
        "truth_raw_rms": truth_raw_rms,
        "truth_mase_rms": truth_mase_rms,
        "nrmse": float(np.sqrt(squared_error_sum / truth_squared_sum)),
        "correlation": _correlation(
            forecast_standardized, truth_standardized
        ),
        "amplitude_ratio": float(
            np.sqrt(forecast_squared_sum / truth_squared_sum)
        ),
        "squared_error_sum": squared_error_sum,
        "truth_squared_sum": truth_squared_sum,
        "forecast_squared_sum": forecast_squared_sum,
    }


def _effect_decay_measurement(
    truth_delta: np.ndarray,
    forecast_delta: np.ndarray,
    observed_mask: np.ndarray,
    affected_target_indices: Iterable[int],
    mase_scale_by_target: np.ndarray,
) -> dict[str, Any]:
    """Compare amplitude-free response shape and peak-relative half-life."""

    truth = np.asarray(truth_delta, dtype=float)
    forecast = np.asarray(forecast_delta, dtype=float)
    mask = np.asarray(observed_mask, dtype=bool)
    scales = np.asarray(mase_scale_by_target, dtype=float)
    affected = [int(value) for value in affected_target_indices]
    truth_profile: list[float] = []
    forecast_profile: list[float] = []
    time_indexes: list[int] = []
    for step in range(truth.shape[0]):
        observed = mask[step, affected]
        if not np.any(observed):
            continue
        selected = np.asarray(affected, dtype=int)[observed]
        truth_values = truth[step, selected] / scales[selected]
        forecast_values = forecast[step, selected] / scales[selected]
        truth_profile.append(float(np.sqrt(np.mean(np.square(truth_values)))))
        forecast_profile.append(
            float(np.sqrt(np.mean(np.square(forecast_values))))
        )
        time_indexes.append(step)
    if not truth_profile:
        return {
            "shape_nrmse": None,
            "half_life_status": "unavailable_no_observed_effect_cell",
            "truth_half_life": None,
            "forecast_half_life": None,
            "half_life_absolute_error": None,
        }
    truth_values = np.asarray(truth_profile, dtype=float)
    forecast_values = np.asarray(forecast_profile, dtype=float)
    times = np.asarray(time_indexes, dtype=int)
    truth_peak = float(np.max(truth_values))
    forecast_peak = float(np.max(forecast_values))
    if truth_peak <= 1e-12:
        return {
            "shape_nrmse": None,
            "half_life_status": "unavailable_zero_truth_profile",
            "truth_half_life": None,
            "forecast_half_life": None,
            "half_life_absolute_error": None,
        }
    truth_normalized = truth_values / truth_peak
    forecast_normalized = (
        forecast_values / forecast_peak
        if forecast_peak > 1e-12
        else np.zeros_like(forecast_values)
    )
    shape_nrmse = float(
        np.sqrt(
            np.sum(np.square(forecast_normalized - truth_normalized))
            / max(np.sum(np.square(truth_normalized)), 1e-12)
        )
    )

    def half_life(profile: np.ndarray, peak: float) -> float | None:
        if peak <= 1e-12:
            return None
        peak_position = int(np.argmax(profile))
        crossing = np.flatnonzero(
            (np.arange(profile.size) >= peak_position)
            & (profile <= 0.5 * peak)
        )
        if not crossing.size:
            return None
        return float(times[int(crossing[0])] - times[peak_position])

    truth_half_life = half_life(truth_values, truth_peak)
    forecast_half_life = half_life(forecast_values, forecast_peak)
    status = "scored"
    if truth_half_life is None:
        status = "censored_truth_not_halved_in_horizon"
    elif forecast_half_life is None:
        status = "censored_forecast_not_halved_in_horizon"
    return {
        "shape_nrmse": shape_nrmse,
        "half_life_status": status,
        "truth_half_life": truth_half_life,
        "forecast_half_life": forecast_half_life,
        "half_life_absolute_error": (
            abs(forecast_half_life - truth_half_life)
            if truth_half_life is not None and forecast_half_life is not None
            else None
        ),
    }


def analyse_model(
    model_id: str,
    baselines: dict[str, dict[str, Any]],
    treatments: Iterable[dict[str, Any]],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_metrics: list[float] = []
    baseline_mae: list[float] = []
    for baseline in baselines.values():
        forecast = predictions.get(str(baseline["sample_id"]))
        if forecast is None:
            continue
        truth = _future(baseline)
        mask = np.asarray(baseline["future_observed_mask"], dtype=bool)
        error = np.abs(_masked(forecast - truth, mask))
        baseline_mae.append(float(np.mean(error)))
        scales = np.asarray(baseline["mase_scale_by_target"], dtype=float)
        scaled = np.abs(forecast - truth) / scales[None, :]
        baseline_metrics.append(float(np.mean(_masked(scaled, mask))))
    effect_rows: list[dict[str, Any]] = []
    for treatment in treatments:
        baseline = baselines[str(treatment["baseline_sample_id"])]
        treatment_forecast = predictions.get(str(treatment["sample_id"]))
        baseline_forecast = predictions.get(str(baseline["sample_id"]))
        if treatment_forecast is None or baseline_forecast is None:
            continue
        truth_delta = _future(treatment) - _future(baseline)
        forecast_delta = treatment_forecast - baseline_forecast
        mask = np.asarray(treatment["future_observed_mask"], dtype=bool)
        affected = [int(value) for value in treatment["affected_target_indices"]]
        scales = np.asarray(baseline["mase_scale_by_target"], dtype=float)
        measurement = _effect_measurement(
            truth_delta,
            forecast_delta,
            mask,
            affected,
            scales,
        )
        decay = (
            _effect_decay_measurement(
                truth_delta,
                forecast_delta,
                mask,
                affected,
                scales,
            )
            if treatment["capability_id"] == "nonlinear_persistence"
            else {
                "shape_nrmse": None,
                "half_life_status": "not_applicable",
                "truth_half_life": None,
                "forecast_half_life": None,
                "half_life_absolute_error": None,
            }
        )
        treatment_error = treatment_forecast - _future(treatment)
        treatment_mase = float(
            np.mean(_masked(np.abs(treatment_error) / scales[None, :], mask))
        )
        treatment_mae = float(np.mean(np.abs(_masked(treatment_error, mask))))
        effect_rows.append(
            {
                "schema_version": "cafe.capability_effect_metric.v4",
                "model_id": model_id,
                "dataset_id": treatment["dataset_id"],
                "official_instance_id": treatment["official_instance_id"],
                "sample_id": treatment["sample_id"],
                "capability_id": treatment["capability_id"],
                "capability_level": int(treatment["capability_level"]),
                "controlled_coordinate": treatment["controlled_coordinate"],
                "sampled_coordinate": float(treatment["sampled_coordinate"]),
                "effect_score_status": measurement["status"],
                "effect_nrmse": measurement["nrmse"],
                "effect_correlation": measurement["correlation"],
                "effect_amplitude_ratio": measurement["amplitude_ratio"],
                "effect_decay_shape_nrmse": decay["shape_nrmse"],
                "effect_half_life_status": decay["half_life_status"],
                "truth_effect_half_life": decay["truth_half_life"],
                "forecast_effect_half_life": decay["forecast_half_life"],
                "effect_half_life_absolute_error": decay[
                    "half_life_absolute_error"
                ],
                "truth_effect_rms": measurement["truth_raw_rms"],
                "truth_effect_mase_rms": measurement["truth_mase_rms"],
                "observed_effect_cell_count": measurement["observed_count"],
                "standardized_squared_error_sum": measurement[
                    "squared_error_sum"
                ],
                "standardized_truth_squared_sum": measurement[
                    "truth_squared_sum"
                ],
                "standardized_forecast_squared_sum": measurement[
                    "forecast_squared_sum"
                ],
                "treatment_mase": treatment_mase,
                "treatment_mae": treatment_mae,
            }
        )
    model_summary = {
        "schema_version": "cafe.official_accuracy_summary.v1",
        "model_id": model_id,
        "official_instance_count": len(baseline_metrics),
        "official_mase_mean": (
            float(np.mean(baseline_metrics)) if baseline_metrics else None
        ),
        "official_mae_mean": float(np.mean(baseline_mae)) if baseline_mae else None,
        "capability_treatment_count": len(effect_rows),
    }
    return model_summary, effect_rows


def _accuracy_rows(
    model_id: str,
    baselines: dict[str, dict[str, Any]],
    treatments: Iterable[dict[str, Any]],
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_kind, samples in (
        ("official_baseline", baselines.values()),
        ("capability_treatment", treatments),
    ):
        for sample in samples:
            forecast = predictions.get(str(sample["sample_id"]))
            if forecast is None:
                continue
            truth = _future(sample)
            mask = np.asarray(sample["future_observed_mask"], dtype=bool)
            scales = np.asarray(sample["mase_scale_by_target"], dtype=float)
            error = forecast - truth
            rows.append(
                {
                    "schema_version": "cafe.forecast_accuracy_metric.v1",
                    "model_id": model_id,
                    "dataset_id": sample["dataset_id"],
                    "official_instance_id": sample["official_instance_id"],
                    "sample_id": sample["sample_id"],
                    "sample_kind": sample_kind,
                    "capability_id": sample.get("capability_id"),
                    "capability_level": int(sample.get("capability_level") or 0),
                    "mase": float(
                        np.mean(_masked(np.abs(error) / scales[None, :], mask))
                    ),
                    "mae": float(np.mean(np.abs(_masked(error, mask)))),
                }
            )
    return rows


def _aggregate_accuracy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str | None, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model_id"]),
            str(row["sample_kind"]),
            None if row["capability_id"] is None else str(row["capability_id"]),
            int(row["capability_level"]),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for (model_id, sample_kind, capability_id, level), members in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        output.append(
            {
                "schema_version": "cafe.forecast_accuracy_summary.v1",
                "model_id": model_id,
                "sample_kind": sample_kind,
                "capability_id": capability_id,
                "capability_level": level,
                "official_instance_count": len(
                    {row["official_instance_id"] for row in members}
                ),
                "mase_mean": float(np.mean([row["mase"] for row in members])),
                "mae_mean": float(np.mean([row["mae"] for row in members])),
            }
        )
    return output


def _input_ablation_rows(
    model_id: str,
    baselines: dict[str, dict[str, Any]],
    treatments: dict[str, dict[str, Any]],
    ablations: Iterable[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    input_adaptations: dict[str, dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ablation in ablations:
        source = treatments[str(ablation["input_ablation_source_sample_id"])]
        baseline = baselines[str(source["baseline_sample_id"])]
        if input_adaptations is not None and not _input_ablation_is_exposed(
            source,
            ablation,
            input_adaptations.get(str(source["sample_id"])),
        ):
            continue
        full_forecast = predictions.get(str(source["sample_id"]))
        ablated_forecast = predictions.get(str(ablation["sample_id"]))
        if full_forecast is None or ablated_forecast is None:
            continue
        truth = _future(source)
        assessed = [int(value) for value in ablation["assessed_target_indices"]]
        ablation_targets = [
            int(value)
            for value in ablation.get("ablation_target_indices", assessed)
        ]
        if len(assessed) != len(ablation_targets):
            raise ValueError("input ablation target index mapping is invalid")
        ablated_truth = _future(ablation)[:, ablation_targets]
        assessed_truth = truth[:, assessed]
        if not np.array_equal(assessed_truth, ablated_truth):
            raise ValueError("input ablation changed scored future")
        assessed_mask = np.asarray(
            source["future_observed_mask"], dtype=bool
        )[:, assessed]
        scales = np.asarray(
            baseline["mase_scale_by_target"], dtype=float
        )[assessed]
        full_assessed = full_forecast[:, assessed]
        ablated_assessed = ablated_forecast[:, ablation_targets]
        full_scaled_error = (
            np.abs(full_assessed - assessed_truth) / scales[None, :]
        )
        ablated_scaled_error = (
            np.abs(ablated_assessed - assessed_truth) / scales[None, :]
        )
        full_mase = float(np.mean(_masked(full_scaled_error, assessed_mask)))
        ablated_mase = float(np.mean(_masked(ablated_scaled_error, assessed_mask)))
        forecast_change = _masked(
            ablated_assessed - full_assessed, assessed_mask
        )
        truth_effect = _masked(
            assessed_truth - _future(baseline)[:, assessed], assessed_mask
        )
        truth_effect_rms = max(
            float(np.sqrt(np.mean(np.square(truth_effect)))), 1e-8
        )
        rows.append(
            {
                "schema_version": "cafe.capability_input_ablation_metric.v3",
                "model_id": model_id,
                "dataset_id": source["dataset_id"],
                "official_instance_id": source["official_instance_id"],
                "sample_id": ablation["sample_id"],
                "source_treatment_sample_id": source["sample_id"],
                "capability_id": source["capability_id"],
                "capability_level": int(source["capability_level"]),
                "assessed_target_indices": assessed,
                "ablation_target_indices": ablation_targets,
                "ablated_input_indices": [
                    int(value) for value in ablation["ablated_input_indices"]
                ],
                "full_input_mase": full_mase,
                "ablated_input_mase": ablated_mase,
                "input_ablation_mase_degradation": ablated_mase - full_mase,
                "input_ablation_forecast_change_rms": float(
                    np.sqrt(np.mean(np.square(forecast_change)))
                ),
                "input_ablation_response_ratio": float(
                    np.sqrt(np.mean(np.square(forecast_change))) / truth_effect_rms
                ),
            }
        )
    return rows


def _input_ablation_is_exposed(
    source: dict[str, Any],
    ablation: dict[str, Any],
    input_adaptation: dict[str, Any] | None,
) -> bool:
    """Whether the full request actually contained the removed input."""

    if input_adaptation is None:
        # Prediction artifacts written before adaptation metadata was recorded
        # remain analysable with their historical all-pairs behavior.
        return True
    policy = str((ablation.get("input_ablation_metadata") or {}).get("policy"))
    if policy in {
        "deterministic_least_aligned_circular_shift_v1",
        "shift_only_constructed_covariate_impulse_v1",
    }:
        return True
    capability = str(source["capability_id"])
    if capability in {"common_factor", "cross_series_dependence"}:
        return input_adaptation.get("target_mode") == "native_multivariate"
    if capability != "covariate_impulse_response":
        return False
    covariate_mode = str(input_adaptation.get("covariate_mode"))
    if covariate_mode in {"none", "omitted_unsupported"}:
        return False
    if covariate_mode != "paired_known_future_only":
        return True
    removed = int(ablation["ablated_input_indices"][0])
    visible = [bool(value) for value in source["future_covariate_visible"]]
    return removed < len(visible) and visible[removed]


def _aggregate_input_ablations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["capability_id"], row["capability_level"])].append(row)
    output: list[dict[str, Any]] = []
    for (model_id, capability, level), members in sorted(grouped.items()):
        output.append(
            {
                "schema_version": "cafe.capability_input_ablation_summary.v1",
                "model_id": model_id,
                "capability_id": capability,
                "capability_level": level,
                "official_instance_count": len(
                    {row["official_instance_id"] for row in members}
                ),
                "full_input_mase_mean": float(
                    np.mean([row["full_input_mase"] for row in members])
                ),
                "ablated_input_mase_mean": float(
                    np.mean([row["ablated_input_mase"] for row in members])
                ),
                "input_ablation_mase_degradation_mean": float(
                    np.mean(
                        [row["input_ablation_mase_degradation"] for row in members]
                    )
                ),
                "input_ablation_response_ratio_mean": float(
                    np.mean([row["input_ablation_response_ratio"] for row in members])
                ),
            }
        )
    rank_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        rank_groups[(row["capability_id"], row["capability_level"])].append(row)
    for members in rank_groups.values():
        ordered = sorted(
            members,
            key=lambda row: (
                -row["input_ablation_mase_degradation_mean"],
                row["model_id"],
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row["input_attribution_rank"] = rank
    return output


def _aggregate_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["capability_id"], row["capability_level"])].append(row)
    aggregates: list[dict[str, Any]] = []
    for (model_id, capability, level), members in sorted(grouped.items()):
        scored = [
            row for row in members if row.get("effect_score_status") == "scored"
        ]
        correlations = [
            float(row["effect_correlation"])
            for row in scored
            if row["effect_correlation"] is not None
        ]
        truth_squared_sum = float(
            sum(row["standardized_truth_squared_sum"] for row in scored)
        )
        squared_error_sum = float(
            sum(row["standardized_squared_error_sum"] for row in scored)
        )
        decay_shapes = [
            float(row["effect_decay_shape_nrmse"])
            for row in scored
            if row.get("effect_decay_shape_nrmse") is not None
        ]
        half_life_errors = [
            float(row["effect_half_life_absolute_error"])
            for row in scored
            if row.get("effect_half_life_status") == "scored"
            and row.get("effect_half_life_absolute_error") is not None
        ]
        aggregates.append(
            {
                "schema_version": "cafe.capability_effect_summary.v4",
                "model_id": model_id,
                "capability_id": capability,
                "capability_level": level,
                "official_instance_count": len(scored),
                "effect_candidate_count": len(members),
                "effect_unavailable_low_signal_count": sum(
                    row.get("effect_score_status")
                    == "unavailable_low_truth_effect"
                    for row in members
                ),
                "effect_scoring_coverage": (
                    len(scored) / len(members) if members else None
                ),
                "effect_nrmse_pooled": (
                    float(np.sqrt(squared_error_sum / truth_squared_sum))
                    if truth_squared_sum > 0.0
                    else None
                ),
                "effect_nrmse_mean": (
                    float(np.mean([row["effect_nrmse"] for row in scored]))
                    if scored
                    else None
                ),
                "effect_correlation_mean": (
                    float(np.mean(correlations)) if correlations else None
                ),
                "effect_amplitude_ratio_mean": (
                    float(
                        np.mean(
                            [row["effect_amplitude_ratio"] for row in scored]
                        )
                    )
                    if scored
                    else None
                ),
                "effect_decay_shape_nrmse_mean": (
                    float(np.mean(decay_shapes)) if decay_shapes else None
                ),
                "effect_half_life_mae": (
                    float(np.mean(half_life_errors))
                    if half_life_errors
                    else None
                ),
                "effect_half_life_scored_count": len(half_life_errors),
                "effect_half_life_censored_count": sum(
                    row.get("effect_half_life_status", "not_applicable").startswith(
                        "censored_"
                    )
                    for row in scored
                ),
            }
        )
    rank_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        if row["effect_nrmse_pooled"] is not None:
            rank_groups[(row["capability_id"], row["capability_level"])].append(row)
    for members in rank_groups.values():
        ordered = sorted(
            members,
            key=lambda row: (row["effect_nrmse_pooled"], row["model_id"]),
        )
        for rank, row in enumerate(ordered, start=1):
            row["effect_rank"] = rank
    return aggregates


def _load_prediction_part(
    record: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if record is None:
        return {}
    path = validate_parquet_record(record)
    output: dict[str, dict[str, Any]] = {}
    for row in iter_prediction_parquet(path):
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate prediction for {sample_id} in {path}")
        output[sample_id] = {
            "forecast": np.asarray(row["forecast"], dtype=float),
            "input_adaptation": row.get("input_adaptation"),
        }
    return output


def _mean_or_none(total: float, count: int) -> float | None:
    return None if count <= 0 else float(total / count)


def _accuracy_aggregate() -> dict[str, float]:
    return {"count": 0.0, "mase": 0.0, "mae": 0.0}


def _effect_aggregate() -> dict[str, float]:
    return {
        "candidate_count": 0.0,
        "count": 0.0,
        "low_signal_count": 0.0,
        "unobserved_count": 0.0,
        "nrmse": 0.0,
        "amplitude": 0.0,
        "correlation": 0.0,
        "correlation_count": 0.0,
        "squared_error_sum": 0.0,
        "truth_squared_sum": 0.0,
        "forecast_squared_sum": 0.0,
        "observed_cell_count": 0.0,
        "decay_shape_nrmse": 0.0,
        "decay_shape_count": 0.0,
        "half_life_absolute_error": 0.0,
        "half_life_count": 0.0,
        "half_life_censored_count": 0.0,
    }


def _ablation_aggregate() -> dict[str, float]:
    return {
        "count": 0.0,
        "full_mase": 0.0,
        "ablated_mase": 0.0,
        "degradation": 0.0,
        "response_ratio": 0.0,
    }


def _consume_replayed_samples(
    samples: Iterable[dict[str, Any]],
    *,
    model_ids: list[str],
    predictions_by_model: dict[str, dict[str, dict[str, Any]]],
    accuracy_writer: TypedParquetWriter,
    effect_writer: TypedParquetWriter,
    ablation_writer: TypedParquetWriter,
) -> dict[str, Any]:
    accuracy_aggregates: defaultdict[
        tuple[str, str, str | None, int], dict[str, float]
    ] = defaultdict(_accuracy_aggregate)
    effect_aggregates: defaultdict[
        tuple[str, str, int], dict[str, float]
    ] = defaultdict(_effect_aggregate)
    ablation_aggregates: defaultdict[
        tuple[str, str, int], dict[str, float]
    ] = defaultdict(_ablation_aggregate)
    official_counts: defaultdict[str, int] = defaultdict(int)
    treatment_counts: defaultdict[str, int] = defaultdict(int)
    baselines: dict[str, dict[str, Any]] = {}
    treatments: dict[str, dict[str, Any]] = {}

    for sample in samples:
        sample_id = str(sample["sample_id"])
        prediction_rows = {
            model_id: predictions_by_model[model_id].get(sample_id)
            for model_id in model_ids
        }
        forecasts = {
            model_id: (
                None if prediction is None else prediction["forecast"]
            )
            for model_id, prediction in prediction_rows.items()
        }
        input_adaptations = {
            model_id: (
                None if prediction is None else prediction["input_adaptation"]
            )
            for model_id, prediction in prediction_rows.items()
        }
        table = str(sample["evaluation_table"])
        if table in {"gift_eval_official_baseline", "benchmark_official_baseline"}:
            future = _future(sample)
            state = {
                "row": sample,
                "future": future,
                "mask": np.asarray(sample["future_observed_mask"], dtype=bool),
                "scales": np.asarray(sample["mase_scale_by_target"], dtype=float),
                "forecasts": forecasts,
            }
            baselines[sample_id] = state
            if not np.any(state["mask"]):
                continue
            for model_id, forecast in forecasts.items():
                if forecast is None:
                    continue
                error = forecast - future
                mase = float(
                    np.mean(
                        _masked(
                            np.abs(error) / state["scales"][None, :],
                            state["mask"],
                        )
                    )
                )
                mae = float(np.mean(np.abs(_masked(error, state["mask"]))))
                accuracy_writer.write(
                    {
                        "schema_version": "cafe.forecast_accuracy_metric.v2",
                        "model_id": model_id,
                        "dataset_id": sample["dataset_id"],
                        "official_instance_id": sample["official_instance_id"],
                        "sample_id": sample_id,
                        "sample_kind": "official_baseline",
                        "capability_id": None,
                        "capability_level": 0,
                        "mase": mase,
                        "mae": mae,
                    }
                )
                aggregate = accuracy_aggregates[
                    (model_id, "official_baseline", None, 0)
                ]
                aggregate["count"] += 1
                aggregate["mase"] += mase
                aggregate["mae"] += mae
                official_counts[model_id] += 1
            continue
        if table in {
            "gift_eval_capability_treatment",
            "benchmark_capability_treatment",
        }:
            baseline = baselines[str(sample["baseline_sample_id"])]
            future = _future(sample)
            treatments[sample_id] = {
                "row": sample,
                "future": future,
                "forecasts": forecasts,
                "input_adaptations": input_adaptations,
                "baseline": baseline,
            }
            mask = baseline["mask"]
            scales = baseline["scales"]
            truth_delta = future - baseline["future"]
            affected = [int(value) for value in sample["affected_target_indices"]]
            for model_id, forecast in forecasts.items():
                baseline_forecast = baseline["forecasts"][model_id]
                if forecast is None or baseline_forecast is None:
                    continue
                treatment_mase: float | None = None
                treatment_mae: float | None = None
                if np.any(mask):
                    error = forecast - future
                    treatment_mase = float(
                        np.mean(_masked(np.abs(error) / scales[None, :], mask))
                    )
                    treatment_mae = float(np.mean(np.abs(_masked(error, mask))))
                    accuracy_writer.write(
                        {
                            "schema_version": "cafe.forecast_accuracy_metric.v2",
                            "model_id": model_id,
                            "dataset_id": sample["dataset_id"],
                            "official_instance_id": sample["official_instance_id"],
                            "sample_id": sample_id,
                            "sample_kind": "capability_treatment",
                            "capability_id": sample["capability_id"],
                            "capability_level": int(sample["capability_level"]),
                            "mase": treatment_mase,
                            "mae": treatment_mae,
                        }
                    )
                    accuracy_key = (
                        model_id,
                        "capability_treatment",
                        str(sample["capability_id"]),
                        int(sample["capability_level"]),
                    )
                    accuracy_aggregate = accuracy_aggregates[accuracy_key]
                    accuracy_aggregate["count"] += 1
                    accuracy_aggregate["mase"] += treatment_mase
                    accuracy_aggregate["mae"] += treatment_mae
                    treatment_counts[model_id] += 1
                forecast_delta = forecast - baseline_forecast
                measurement = _effect_measurement(
                    truth_delta,
                    forecast_delta,
                    mask,
                    affected,
                    scales,
                )
                decay = (
                    _effect_decay_measurement(
                        truth_delta,
                        forecast_delta,
                        mask,
                        affected,
                        scales,
                    )
                    if sample["capability_id"] == "nonlinear_persistence"
                    else {
                        "shape_nrmse": None,
                        "half_life_status": "not_applicable",
                        "truth_half_life": None,
                        "forecast_half_life": None,
                        "half_life_absolute_error": None,
                    }
                )
                effect_writer.write(
                    {
                        "schema_version": "cafe.capability_effect_metric.v4",
                        "model_id": model_id,
                        "dataset_id": sample["dataset_id"],
                        "official_instance_id": sample["official_instance_id"],
                        "sample_id": sample_id,
                        "capability_id": sample["capability_id"],
                        "capability_level": int(sample["capability_level"]),
                        "controlled_coordinate": sample["controlled_coordinate"],
                        "sampled_coordinate": float(sample["sampled_coordinate"]),
                        "effect_score_status": measurement["status"],
                        "effect_nrmse": measurement["nrmse"],
                        "effect_correlation": measurement["correlation"],
                        "effect_amplitude_ratio": measurement[
                            "amplitude_ratio"
                        ],
                        "effect_decay_shape_nrmse": decay["shape_nrmse"],
                        "effect_half_life_status": decay[
                            "half_life_status"
                        ],
                        "truth_effect_half_life": decay["truth_half_life"],
                        "forecast_effect_half_life": decay[
                            "forecast_half_life"
                        ],
                        "effect_half_life_absolute_error": decay[
                            "half_life_absolute_error"
                        ],
                        "truth_effect_rms": measurement["truth_raw_rms"],
                        "truth_effect_mase_rms": measurement[
                            "truth_mase_rms"
                        ],
                        "observed_effect_cell_count": measurement[
                            "observed_count"
                        ],
                        "standardized_squared_error_sum": measurement[
                            "squared_error_sum"
                        ],
                        "standardized_truth_squared_sum": measurement[
                            "truth_squared_sum"
                        ],
                        "standardized_forecast_squared_sum": measurement[
                            "forecast_squared_sum"
                        ],
                        "treatment_mase": treatment_mase,
                        "treatment_mae": treatment_mae,
                    }
                )
                effect_key = (
                    model_id,
                    str(sample["capability_id"]),
                    int(sample["capability_level"]),
                )
                effect_aggregate = effect_aggregates[effect_key]
                effect_aggregate["candidate_count"] += 1
                if measurement["status"] == "unavailable_low_truth_effect":
                    effect_aggregate["low_signal_count"] += 1
                    continue
                if measurement["status"] != "scored":
                    effect_aggregate["unobserved_count"] += 1
                    continue
                nrmse = float(measurement["nrmse"])
                amplitude = float(measurement["amplitude_ratio"])
                correlation = measurement["correlation"]
                effect_aggregate["count"] += 1
                effect_aggregate["nrmse"] += nrmse
                effect_aggregate["amplitude"] += amplitude
                effect_aggregate["squared_error_sum"] += measurement[
                    "squared_error_sum"
                ]
                effect_aggregate["truth_squared_sum"] += measurement[
                    "truth_squared_sum"
                ]
                effect_aggregate["forecast_squared_sum"] += measurement[
                    "forecast_squared_sum"
                ]
                effect_aggregate["observed_cell_count"] += measurement[
                    "observed_count"
                ]
                if correlation is not None:
                    effect_aggregate["correlation"] += correlation
                    effect_aggregate["correlation_count"] += 1
                if decay["shape_nrmse"] is not None:
                    effect_aggregate["decay_shape_nrmse"] += float(
                        decay["shape_nrmse"]
                    )
                    effect_aggregate["decay_shape_count"] += 1
                if decay["half_life_status"] == "scored":
                    effect_aggregate["half_life_absolute_error"] += float(
                        decay["half_life_absolute_error"]
                    )
                    effect_aggregate["half_life_count"] += 1
                elif sample["capability_id"] == "nonlinear_persistence":
                    effect_aggregate["half_life_censored_count"] += 1
            continue
        if table not in {
            "gift_eval_capability_input_ablation",
            "benchmark_capability_input_ablation",
        }:
            raise ValueError(f"unknown evaluation table {table}")
        source = treatments[str(sample["input_ablation_source_sample_id"])]
        baseline = source["baseline"]
        truth = source["future"]
        mask = baseline["mask"]
        assessed = [int(value) for value in sample["assessed_target_indices"]]
        ablation_targets = [
            int(value)
            for value in sample.get("ablation_target_indices", assessed)
        ]
        if len(assessed) != len(ablation_targets):
            raise ValueError("input ablation target index mapping is invalid")
        assessed_mask = mask[:, assessed]
        if not np.any(assessed_mask):
            continue
        scales = baseline["scales"][assessed]
        assessed_truth = truth[:, assessed]
        ablated_truth = _future(sample)[:, ablation_targets]
        if not np.array_equal(assessed_truth, ablated_truth):
            raise ValueError("input ablation changed scored future")
        truth_effect = _masked(
            assessed_truth - baseline["future"][:, assessed], assessed_mask
        )
        truth_effect_rms = max(
            float(np.sqrt(np.mean(np.square(truth_effect)))), 1e-8
        )
        for model_id, forecast in forecasts.items():
            full_forecast = source["forecasts"][model_id]
            if forecast is None or full_forecast is None:
                continue
            if not _input_ablation_is_exposed(
                source["row"],
                sample,
                source["input_adaptations"][model_id],
            ):
                continue
            full_assessed = full_forecast[:, assessed]
            ablated_assessed = forecast[:, ablation_targets]
            full_mase = float(
                np.mean(
                    _masked(
                        np.abs(full_assessed - assessed_truth) / scales[None, :],
                        assessed_mask,
                    )
                )
            )
            ablated_mase = float(
                np.mean(
                    _masked(
                        np.abs(ablated_assessed - assessed_truth)
                        / scales[None, :],
                        assessed_mask,
                    )
                )
            )
            forecast_change = _masked(
                ablated_assessed - full_assessed, assessed_mask
            )
            response_ratio = float(
                np.sqrt(np.mean(np.square(forecast_change))) / truth_effect_rms
            )
            ablation_writer.write(
                {
                    "schema_version": "cafe.capability_input_ablation_metric.v3",
                    "model_id": model_id,
                    "dataset_id": sample["dataset_id"],
                    "official_instance_id": sample["official_instance_id"],
                    "sample_id": sample_id,
                    "source_treatment_sample_id": source["row"]["sample_id"],
                    "capability_id": source["row"]["capability_id"],
                    "capability_level": int(source["row"]["capability_level"]),
                    "assessed_target_indices": assessed,
                    "ablation_target_indices": ablation_targets,
                    "ablated_input_indices": [
                        int(value) for value in sample["ablated_input_indices"]
                    ],
                    "full_input_mase": full_mase,
                    "ablated_input_mase": ablated_mase,
                    "input_ablation_mase_degradation": ablated_mase - full_mase,
                    "input_ablation_forecast_change_rms": float(
                        np.sqrt(np.mean(np.square(forecast_change)))
                    ),
                    "input_ablation_response_ratio": response_ratio,
                }
            )
            ablation_key = (
                model_id,
                str(source["row"]["capability_id"]),
                int(source["row"]["capability_level"]),
            )
            ablation_aggregate = ablation_aggregates[ablation_key]
            ablation_aggregate["count"] += 1
            ablation_aggregate["full_mase"] += full_mase
            ablation_aggregate["ablated_mase"] += ablated_mase
            ablation_aggregate["degradation"] += ablated_mase - full_mase
            ablation_aggregate["response_ratio"] += response_ratio
    return {
        "accuracy_aggregates": dict(accuracy_aggregates),
        "effect_aggregates": dict(effect_aggregates),
        "ablation_aggregates": dict(ablation_aggregates),
        "official_counts": dict(official_counts),
        "treatment_counts": dict(treatment_counts),
    }


def _analyse_source_shard(job: dict[str, Any]) -> dict[str, Any]:
    shard = int(job["source_shard_index"])
    parts_dir = Path(job["parts_dir"])
    model_ids = [str(value) for value in job["model_ids"]]
    records = job["prediction_records"]
    predictions_by_model = {
        model_id: _load_prediction_part(records.get(model_id))
        for model_id in model_ids
    }
    paths = {
        "accuracy": parts_dir / f"accuracy_{shard:06d}.parquet",
        "effect": parts_dir / f"effect_{shard:06d}.parquet",
        "ablation": parts_dir / f"ablation_{shard:06d}.parquet",
    }
    writers = {
        "accuracy": TypedParquetWriter(paths["accuracy"], schema=ACCURACY_METRIC_SCHEMA),
        "effect": TypedParquetWriter(paths["effect"], schema=EFFECT_METRIC_SCHEMA),
        "ablation": TypedParquetWriter(paths["ablation"], schema=ABLATION_METRIC_SCHEMA),
    }
    try:
        samples = (
            sample
            for work in job["work_items"]
            for sample in _replay_contract_instance(*work)
        )
        result = _consume_replayed_samples(
            samples,
            model_ids=model_ids,
            predictions_by_model=predictions_by_model,
            accuracy_writer=writers["accuracy"],
            effect_writer=writers["effect"],
            ablation_writer=writers["ablation"],
        )
        result["row_counts"] = {
            name: writer.close() for name, writer in writers.items()
        }
        result["paths"] = {name: str(path) for name, path in paths.items()}
        result["source_shard_index"] = shard
        return result
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise


def _merge_parquet_parts(
    part_paths: list[Path],
    output_path: Path,
    *,
    schema: pa.Schema,
) -> int:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for path in part_paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches():
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        schema,
                        compression=DEFAULT_COMPRESSION,
                        compression_level=DEFAULT_COMPRESSION_LEVEL,
                        use_dictionary=True,
                        write_statistics=True,
                    )
                writer.write_table(pa.Table.from_batches([batch], schema=schema))
                count += batch.num_rows
        if writer is None:
            pq.write_table(
                pa.Table.from_pylist([], schema=schema),
                temporary,
                compression=DEFAULT_COMPRESSION,
                compression_level=DEFAULT_COMPRESSION_LEVEL,
            )
        else:
            writer.close()
            writer = None
        os.replace(temporary, output_path)
        return count
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise


def _merge_numeric_aggregates(
    destination: defaultdict[Any, dict[str, float]],
    source: dict[Any, dict[str, float]],
) -> None:
    for key, values in source.items():
        aggregate = destination[key]
        for name, value in values.items():
            aggregate[name] += float(value)


def run_analysis(
    dataset_root: Path,
    *,
    gift_eval_dir: Path | None = None,
    replay_workers: int = 1,
) -> dict[str, Any]:
    """Analyse source shards in parallel after one official-source scan."""

    return _run_analysis_sharded(
        dataset_root,
        gift_eval_dir=gift_eval_dir,
        shard_workers=max(1, int(replay_workers)),
    )


def _bootstrap_task_mean(
    values: list[float],
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty task set")
    if array.size == 1 or repetitions < 1:
        value = float(np.mean(array))
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(repetitions), array.size))
    means = np.mean(array[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def aggregate_analysis_tasks(
    experiment_root: Path,
    task_ids: Iterable[str],
    *,
    bootstrap_seed: int = 20260826,
    bootstrap_repetitions: int = 2000,
) -> dict[str, Any]:
    """Write task-equal suite metrics inside the analysis stage.

    Each GIFT dataset or FEV task contributes one value. Pairwise uncertainty
    resamples only the common task set, preserving model pairing.
    """

    task_ids = tuple(dict.fromkeys(str(value) for value in task_ids))
    if not task_ids:
        raise ValueError("suite analysis requires at least one task")
    observations: defaultdict[
        tuple[str, str, str | None, int | None], dict[str, float]
    ] = defaultdict(dict)
    benchmark_ids: set[str] = set()
    suite_ids: set[str] = set()
    upstream: list[Path] = []
    for task_id in task_ids:
        task_root = experiment_root / task_id
        generation_path = task_root / "01_generation" / "manifest.json"
        analysis_path = task_root / "04_analysis" / "manifest.json"
        generation = protocol.read_json(generation_path)
        analysis = protocol.read_json(analysis_path)
        config = generation.get("config") or {}
        benchmark_ids.add(str(config.get("benchmark_id") or "gift_eval"))
        suite_ids.add(str(config.get("suite_id") or config.get("term") or "native"))
        upstream.append(analysis_path)

        accuracy = protocol.read_json(
            Path(str(analysis["files"]["official_accuracy"]["path"]))
        )
        for row in accuracy.get("models") or []:
            value = row.get("official_mase_mean")
            if value is not None and math.isfinite(float(value)):
                observations[("official_mase", str(row["model_id"]), None, None)][
                    task_id
                ] = float(value)

        effects = protocol.read_json(
            Path(str(analysis["files"]["capability_effect_summary"]["path"]))
        )
        for row in effects.get("rows") or []:
            value = row.get("effect_nrmse_pooled")
            if value is not None and math.isfinite(float(value)):
                observations[
                    (
                        "capability_effect_nrmse",
                        str(row["model_id"]),
                        str(row["capability_id"]),
                        int(row["capability_level"]),
                    )
                ][task_id] = float(value)

        ablations = protocol.read_json(
            Path(str(analysis["files"]["input_ablation_summary"]["path"]))
        )
        for row in ablations.get("rows") or []:
            value = row.get("input_ablation_mase_degradation_mean")
            if value is not None and math.isfinite(float(value)):
                observations[
                    (
                        "input_ablation_mase_degradation",
                        str(row["model_id"]),
                        str(row["capability_id"]),
                        int(row["capability_level"]),
                    )
                ][task_id] = float(value)

    rows: list[dict[str, Any]] = []
    for index, (key, by_task) in enumerate(sorted(observations.items())):
        metric, model_id, capability_id, level = key
        values = list(by_task.values())
        lower, upper = _bootstrap_task_mean(
            values,
            seed=int(bootstrap_seed) + index,
            repetitions=int(bootstrap_repetitions),
        )
        rows.append(
            {
                "metric": metric,
                "model_id": model_id,
                "capability_id": capability_id,
                "capability_level": level,
                "task_count": len(values),
                "suite_task_count": len(task_ids),
                "task_coverage": len(values) / len(task_ids),
                "task_equal_mean": float(np.mean(values)),
                "task_bootstrap_95_ci_lower": lower,
                "task_bootstrap_95_ci_upper": upper,
            }
        )

    pairwise: list[dict[str, Any]] = []
    groups: defaultdict[
        tuple[str, str | None, int | None],
        dict[str, dict[str, float]],
    ] = defaultdict(dict)
    for (metric, model_id, capability_id, level), by_task in observations.items():
        groups[(metric, capability_id, level)][model_id] = by_task
    pair_index = 0
    for (metric, capability_id, level), by_model in sorted(groups.items()):
        model_ids = sorted(by_model)
        for left_index, left_model in enumerate(model_ids):
            for right_model in model_ids[left_index + 1 :]:
                common = sorted(
                    set(by_model[left_model]) & set(by_model[right_model])
                )
                if not common:
                    continue
                differences = [
                    by_model[left_model][task] - by_model[right_model][task]
                    for task in common
                ]
                lower, upper = _bootstrap_task_mean(
                    differences,
                    seed=int(bootstrap_seed) + 100_000 + pair_index,
                    repetitions=int(bootstrap_repetitions),
                )
                pair_index += 1
                pairwise.append(
                    {
                        "metric": metric,
                        "capability_id": capability_id,
                        "capability_level": level,
                        "left_model_id": left_model,
                        "right_model_id": right_model,
                        "paired_task_count": len(common),
                        "task_equal_mean_difference_left_minus_right": float(
                            np.mean(differences)
                        ),
                        "paired_task_bootstrap_95_ci_lower": lower,
                        "paired_task_bootstrap_95_ci_upper": upper,
                    }
                )

    output_dir = experiment_root / "04_analysis_suite"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "task_equal_summary.json"
    protocol.write_json(
        summary_path,
        {
            "schema_version": SUITE_ANALYSIS_SCHEMA,
            "aggregation_unit": "benchmark_task",
            "task_ids": list(task_ids),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_repetitions": int(bootstrap_repetitions),
            "rows": rows,
            "paired_model_comparisons": pairwise,
        },
    )
    config = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "benchmark_ids": sorted(benchmark_ids),
        "suite_ids": sorted(suite_ids),
        "task_ids": list(task_ids),
        "aggregation": "arithmetic_mean_of_task_level_estimands",
        "uncertainty": "paired_nonparametric_task_bootstrap",
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_repetitions": int(bootstrap_repetitions),
    }
    manifest = {
        "schema_version": SUITE_ANALYSIS_SCHEMA,
        "created_at": protocol.utc_now(),
        "config": config,
        "config_sha256": protocol.json_sha256(config),
        "upstream_analysis": [
            {
                "path": str(path.resolve()),
                "sha256": protocol.file_sha256(path),
            }
            for path in upstream
        ],
        "files": {"task_equal_summary": protocol.file_record(summary_path)},
    }
    protocol.write_json(output_dir / "manifest.json", manifest)
    return manifest


def _write_sharded_analysis_outputs(
    *,
    analysis_dir: Path,
    inference_manifest: dict[str, Any],
    inference_manifest_path: Path,
    model_ids: list[str],
    accuracy_aggregates: defaultdict[
        tuple[str, str, str | None, int], dict[str, float]
    ],
    effect_aggregates: defaultdict[tuple[str, str, int], dict[str, float]],
    ablation_aggregates: defaultdict[tuple[str, str, int], dict[str, float]],
    official_counts: defaultdict[str, int],
    treatment_counts: defaultdict[str, int],
    row_counts: dict[str, int],
    shard_workers: int,
    source_shard_count: int,
) -> dict[str, Any]:
    model_summaries: list[dict[str, Any]] = []
    for model_id in model_ids:
        official_values = accuracy_aggregates[
            (model_id, "official_baseline", None, 0)
        ]
        model_summaries.append(
            {
                "schema_version": "cafe.official_accuracy_summary.v2",
                "model_id": model_id,
                "official_instance_count": official_counts[model_id],
                "official_mase_mean": _mean_or_none(
                    official_values["mase"], int(official_values["count"])
                ),
                "official_mae_mean": _mean_or_none(
                    official_values["mae"], int(official_values["count"])
                ),
                "capability_treatment_count": treatment_counts[model_id],
            }
        )

    accuracy_summary: list[dict[str, Any]] = []
    for (model_id, kind, capability, level), values in sorted(
        accuracy_aggregates.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        count = int(values["count"])
        accuracy_summary.append(
            {
                "schema_version": "cafe.forecast_accuracy_summary.v2",
                "model_id": model_id,
                "sample_kind": kind,
                "capability_id": capability,
                "capability_level": level,
                "official_instance_count": count,
                "mase_mean": _mean_or_none(values["mase"], count),
                "mae_mean": _mean_or_none(values["mae"], count),
            }
        )
    effect_summary: list[dict[str, Any]] = []
    for (model_id, capability, level), values in sorted(effect_aggregates.items()):
        count = int(values["count"])
        candidate_count = int(values["candidate_count"])
        truth_squared_sum = float(values["truth_squared_sum"])
        pooled_nrmse = (
            float(
                np.sqrt(
                    float(values["squared_error_sum"]) / truth_squared_sum
                )
            )
            if truth_squared_sum > 0.0
            else None
        )
        pooled_amplitude = (
            float(
                np.sqrt(
                    float(values["forecast_squared_sum"])
                    / truth_squared_sum
                )
            )
            if truth_squared_sum > 0.0
            else None
        )
        effect_summary.append(
            {
                "schema_version": "cafe.capability_effect_summary.v4",
                "model_id": model_id,
                "capability_id": capability,
                "capability_level": level,
                "official_instance_count": count,
                "effect_candidate_count": candidate_count,
                "effect_unavailable_low_signal_count": int(
                    values["low_signal_count"]
                ),
                "effect_unavailable_unobserved_count": int(
                    values["unobserved_count"]
                ),
                "effect_scoring_coverage": (
                    float(count / candidate_count)
                    if candidate_count > 0
                    else None
                ),
                "effect_score_metric": (
                    "mase_standardized_pooled_nrmse_v1"
                ),
                "minimum_truth_effect_mase_rms": (
                    MECHANISM_EFFECT_MINIMUM_MASE_RMS
                ),
                "effect_nrmse_pooled": pooled_nrmse,
                "effect_nrmse_mean": _mean_or_none(values["nrmse"], count),
                "effect_correlation_mean": _mean_or_none(
                    values["correlation"], int(values["correlation_count"])
                ),
                "effect_amplitude_ratio_pooled": pooled_amplitude,
                "effect_amplitude_ratio_mean": _mean_or_none(
                    values["amplitude"], count
                ),
                "observed_effect_cell_count": int(
                    values["observed_cell_count"]
                ),
                "effect_decay_shape_nrmse_mean": _mean_or_none(
                    values["decay_shape_nrmse"],
                    int(values["decay_shape_count"]),
                ),
                "effect_decay_shape_scored_count": int(
                    values["decay_shape_count"]
                ),
                "effect_half_life_mae": _mean_or_none(
                    values["half_life_absolute_error"],
                    int(values["half_life_count"]),
                ),
                "effect_half_life_scored_count": int(
                    values["half_life_count"]
                ),
                "effect_half_life_censored_count": int(
                    values["half_life_censored_count"]
                ),
            }
        )
    rank_groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in effect_summary:
        if row["effect_nrmse_pooled"] is not None:
            rank_groups[(row["capability_id"], row["capability_level"])].append(row)
    for members in rank_groups.values():
        for rank, row in enumerate(
            sorted(
                members,
                key=lambda value: (
                    value["effect_nrmse_pooled"],
                    value["model_id"],
                ),
            ),
            start=1,
        ):
            row["effect_rank"] = rank

    ablation_summary: list[dict[str, Any]] = []
    for (model_id, capability, level), values in sorted(
        ablation_aggregates.items()
    ):
        count = int(values["count"])
        ablation_summary.append(
            {
                "schema_version": "cafe.capability_input_ablation_summary.v2",
                "model_id": model_id,
                "capability_id": capability,
                "capability_level": level,
                "official_instance_count": count,
                "full_input_mase_mean": _mean_or_none(values["full_mase"], count),
                "ablated_input_mase_mean": _mean_or_none(
                    values["ablated_mase"], count
                ),
                "input_ablation_mase_degradation_mean": _mean_or_none(
                    values["degradation"], count
                ),
                "input_ablation_response_ratio_mean": _mean_or_none(
                    values["response_ratio"], count
                ),
            }
        )
    ablation_rank_groups: defaultdict[
        tuple[str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in ablation_summary:
        ablation_rank_groups[
            (row["capability_id"], row["capability_level"])
        ].append(row)
    for members in ablation_rank_groups.values():
        for rank, row in enumerate(
            sorted(
                members,
                key=lambda value: (
                    -value["input_ablation_mase_degradation_mean"],
                    value["model_id"],
                ),
            ),
            start=1,
        ):
            row["input_attribution_rank"] = rank

    accuracy_path = analysis_dir / "official_accuracy.json"
    accuracy_summary_path = analysis_dir / "accuracy_summary.json"
    effect_summary_path = analysis_dir / "capability_effect_summary.json"
    ablation_summary_path = analysis_dir / "input_ablation_summary.json"
    protocol.write_json(accuracy_path, {"models": model_summaries})
    protocol.write_json(accuracy_summary_path, {"rows": accuracy_summary})
    protocol.write_json(effect_summary_path, {"rows": effect_summary})
    protocol.write_json(ablation_summary_path, {"rows": ablation_summary})
    config = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "execution": "process_parallel_source_shards_all_model_prediction_join",
        "row_artifact_format": "parquet_zstd",
        "shard_workers": int(shard_workers),
        "source_shard_count": int(source_shard_count),
        "source_scan_count": 1,
        "estimands": {
            "official_accuracy": "benchmark-native official future MASE/MAE",
            "treatment_accuracy": "treatment future MASE/MAE on authentic MASE scale",
            "capability_effect": (
                "mase_standardized_pooled_forecast_delta_vs_truth_delta_"
                "on_scoreable_affected_targets"
            ),
            "input_ablation_attribution": (
                "same_assessed_treatment_truth_with_exposed_relevant_auxiliary_"
                "inputs_removed"
            ),
        },
        "mechanism_scoring_policy": {
            "minimum_truth_effect_mase_rms": (
                MECHANISM_EFFECT_MINIMUM_MASE_RMS
            ),
            "low_signal_status": "unavailable_low_truth_effect",
            "low_signal_treatment_accuracy_retained": True,
            "primary_metric": "mase_standardized_pooled_nrmse_v1",
            "aggregation": "ratio_of_standardized_squared_sums",
            "nonlinear_persistence_diagnostics": {
                "decay_shape_metric": "peak_normalized_profile_nrmse",
                "half_life_metric": "first_half_peak_crossing_after_peak",
                "censoring_policy": "report_count_and_exclude_from_half_life_mae",
                "ranking_use": "diagnostic_only",
            },
        },
    }
    row_paths = {
        "accuracy": analysis_dir / "accuracy_rows.parquet",
        "effect": analysis_dir / "capability_effect_rows.parquet",
        "ablation": analysis_dir / "input_ablation_rows.parquet",
    }
    manifest = {
        "schema_version": ANALYSIS_SCHEMA,
        "created_at": protocol.utc_now(),
        "dataset_id": inference_manifest["dataset_id"],
        "config": config,
        "config_sha256": protocol.json_sha256(config),
        "inference_manifest_sha256": protocol.file_sha256(inference_manifest_path),
        "files": {
            "official_accuracy": protocol.file_record(accuracy_path),
            "accuracy_rows": parquet_file_record(
                row_paths["accuracy"], row_count=row_counts["accuracy"]
            ),
            "accuracy_summary": protocol.file_record(accuracy_summary_path),
            "capability_effect_rows": parquet_file_record(
                row_paths["effect"], row_count=row_counts["effect"]
            ),
            "capability_effect_summary": protocol.file_record(effect_summary_path),
            "input_ablation_rows": parquet_file_record(
                row_paths["ablation"], row_count=row_counts["ablation"]
            ),
            "input_ablation_summary": protocol.file_record(ablation_summary_path),
        },
    }
    protocol.write_json(analysis_dir / "manifest.json", manifest)
    return manifest


def _run_analysis_sharded(
    dataset_root: Path,
    *,
    gift_eval_dir: Path | None,
    shard_workers: int,
) -> dict[str, Any]:
    generation_dir = dataset_root / "01_generation"
    inference_dir = dataset_root / "03_inference"
    analysis_dir = dataset_root / "04_analysis"
    generation_manifest = protocol.read_json(generation_dir / "manifest.json")
    inference_manifest_path = inference_dir / "manifest.json"
    inference_manifest = protocol.read_json(inference_manifest_path)
    if inference_manifest.get("schema_version") != INFERENCE_SCHEMA:
        raise ValueError("unsupported inference manifest")
    if (
        inference_manifest.get("config", {}).get("pipeline_schema_version")
        != PIPELINE_SCHEMA
    ):
        raise ValueError("inference is not current pipeline v15")
    if not inference_manifest.get("complete"):
        raise ValueError("inference is incomplete")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = analysis_dir / ".source_shard_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    model_ids = [str(value) for value in inference_manifest["config"]["models"]]
    prediction_parts_by_model = {
        model_id: {
            int(record["source_shard_index"]): record
            for record in (
                inference_manifest["model_predictions"][model_id].get("parts") or []
            )
        }
        for model_id in model_ids
    }
    accuracy_aggregates: defaultdict[
        tuple[str, str, str | None, int], dict[str, float]
    ] = defaultdict(_accuracy_aggregate)
    effect_aggregates: defaultdict[
        tuple[str, str, int], dict[str, float]
    ] = defaultdict(_effect_aggregate)
    ablation_aggregates: defaultdict[
        tuple[str, str, int], dict[str, float]
    ] = defaultdict(_ablation_aggregate)
    official_counts: defaultdict[str, int] = defaultdict(int)
    treatment_counts: defaultdict[str, int] = defaultdict(int)
    results: list[dict[str, Any]] = []

    def consume(result: dict[str, Any]) -> None:
        _merge_numeric_aggregates(
            accuracy_aggregates, result["accuracy_aggregates"]
        )
        _merge_numeric_aggregates(effect_aggregates, result["effect_aggregates"])
        _merge_numeric_aggregates(
            ablation_aggregates, result["ablation_aggregates"]
        )
        for model_id, count in result["official_counts"].items():
            official_counts[model_id] += int(count)
        for model_id, count in result["treatment_counts"].items():
            treatment_counts[model_id] += int(count)
        results.append(result)

    workers = max(1, int(shard_workers))

    def job(shard: int, work_items: list[ReplayContractWorkItem]) -> dict[str, Any]:
        return {
            "source_shard_index": shard,
            "work_items": work_items,
            "model_ids": model_ids,
            "prediction_records": {
                model_id: prediction_parts_by_model[model_id].get(shard)
                for model_id in model_ids
            },
            "parts_dir": str(parts_dir),
        }

    def iter_jobs() -> Iterable[dict[str, Any]]:
        current_shard: int | None = None
        current_items: list[ReplayContractWorkItem] = []
        for work in iter_replay_contract_work_items(
            generation_manifest, gift_eval_dir=gift_eval_dir
        ):
            shard = int(work[1].get("source_shard_index", 0))
            if current_shard is None:
                current_shard = shard
            if shard < current_shard:
                raise ValueError("source shard indices are not monotonic")
            if shard != current_shard:
                yield job(current_shard, current_items)
                current_shard = shard
                current_items = []
            current_items.append(work)
        if current_shard is not None:
            yield job(current_shard, current_items)

    try:
        if workers == 1:
            for shard_job in iter_jobs():
                consume(_analyse_source_shard(shard_job))
        else:
            pending: deque[Any] = deque()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for shard_job in iter_jobs():
                    pending.append(
                        executor.submit(_analyse_source_shard, shard_job)
                    )
                    if len(pending) >= workers * 2:
                        consume(pending.popleft().result())
                while pending:
                    consume(pending.popleft().result())

        results.sort(key=lambda value: int(value["source_shard_index"]))
        row_counts = {
            "accuracy": _merge_parquet_parts(
                [Path(result["paths"]["accuracy"]) for result in results],
                analysis_dir / "accuracy_rows.parquet",
                schema=ACCURACY_METRIC_SCHEMA,
            ),
            "effect": _merge_parquet_parts(
                [Path(result["paths"]["effect"]) for result in results],
                analysis_dir / "capability_effect_rows.parquet",
                schema=EFFECT_METRIC_SCHEMA,
            ),
            "ablation": _merge_parquet_parts(
                [Path(result["paths"]["ablation"]) for result in results],
                analysis_dir / "input_ablation_rows.parquet",
                schema=ABLATION_METRIC_SCHEMA,
            ),
        }
        expected_counts = {
            name: sum(int(result["row_counts"][name]) for result in results)
            for name in ("accuracy", "effect", "ablation")
        }
        if row_counts != expected_counts:
            raise ValueError("merged analysis row count mismatch")
        manifest = _write_sharded_analysis_outputs(
            analysis_dir=analysis_dir,
            inference_manifest=inference_manifest,
            inference_manifest_path=inference_manifest_path,
            model_ids=model_ids,
            accuracy_aggregates=accuracy_aggregates,
            effect_aggregates=effect_aggregates,
            ablation_aggregates=ablation_aggregates,
            official_counts=official_counts,
            treatment_counts=treatment_counts,
            row_counts=row_counts,
            shard_workers=workers,
            source_shard_count=len(results),
        )
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)
    return manifest


def main() -> int:
    args = parse_args()
    dataset_root = args.output_root.resolve() / args.dataset_id
    manifest_path = dataset_root / "04_analysis" / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"analysis artifact already exists; use a new experiment root: {manifest_path}"
        )
    manifest = run_analysis(
        dataset_root,
        gift_eval_dir=args.gift_eval_dir,
        replay_workers=args.workers,
    )
    print(protocol.canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
