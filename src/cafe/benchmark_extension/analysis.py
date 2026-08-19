from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa

from cafe import core as protocol
from cafe.benchmark_extension.generation import PIPELINE_SCHEMA
from cafe.benchmark_extension.generation import iter_replayed_samples
from cafe.benchmark_extension.inference import INFERENCE_SCHEMA
from cafe.benchmark_extension.storage import (
    TypedParquetWriter,
    iter_prediction_parquet,
    parquet_file_record,
    validate_parquet_record,
)


ANALYSIS_SCHEMA = "cafe.benchmark_extension_analysis.v3"
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
        ("effect_nrmse", pa.float64()),
        ("effect_correlation", pa.float64()),
        ("effect_amplitude_ratio", pa.float64()),
        ("truth_effect_rms", pa.float64()),
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
        assessed_mask = np.zeros_like(mask, dtype=bool)
        assessed_mask[:, affected] = mask[:, affected]
        truth_values = _masked(truth_delta, assessed_mask)
        forecast_values = _masked(forecast_delta, assessed_mask)
        denominator = max(float(np.sqrt(np.mean(np.square(truth_values)))), 1e-8)
        scales = np.asarray(baseline["mase_scale_by_target"], dtype=float)
        treatment_error = treatment_forecast - _future(treatment)
        treatment_mase = float(
            np.mean(_masked(np.abs(treatment_error) / scales[None, :], mask))
        )
        treatment_mae = float(np.mean(np.abs(_masked(treatment_error, mask))))
        effect_nrmse = float(
            np.sqrt(np.mean(np.square(forecast_values - truth_values))) / denominator
        )
        amplitude_ratio = float(
            np.sqrt(np.mean(np.square(forecast_values))) / denominator
        )
        effect_rows.append(
            {
                "schema_version": "cafe.capability_effect_metric.v1",
                "model_id": model_id,
                "dataset_id": treatment["dataset_id"],
                "official_instance_id": treatment["official_instance_id"],
                "sample_id": treatment["sample_id"],
                "capability_id": treatment["capability_id"],
                "capability_level": int(treatment["capability_level"]),
                "controlled_coordinate": treatment["controlled_coordinate"],
                "sampled_coordinate": float(treatment["sampled_coordinate"]),
                "effect_nrmse": effect_nrmse,
                "effect_correlation": _correlation(forecast_values, truth_values),
                "effect_amplitude_ratio": amplitude_ratio,
                "truth_effect_rms": denominator,
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ablation in ablations:
        source = treatments[str(ablation["input_ablation_source_sample_id"])]
        baseline = baselines[str(source["baseline_sample_id"])]
        full_forecast = predictions.get(str(source["sample_id"]))
        ablated_forecast = predictions.get(str(ablation["sample_id"]))
        if full_forecast is None or ablated_forecast is None:
            continue
        truth = _future(source)
        if not np.array_equal(truth, _future(ablation)):
            raise ValueError("input ablation changed scored future")
        mask = np.asarray(source["future_observed_mask"], dtype=bool)
        assessed = [int(value) for value in ablation["assessed_target_indices"]]
        assessed_mask = np.zeros_like(mask, dtype=bool)
        assessed_mask[:, assessed] = mask[:, assessed]
        scales = np.asarray(baseline["mase_scale_by_target"], dtype=float)
        full_scaled_error = np.abs(full_forecast - truth) / scales[None, :]
        ablated_scaled_error = np.abs(ablated_forecast - truth) / scales[None, :]
        full_mase = float(np.mean(_masked(full_scaled_error, assessed_mask)))
        ablated_mase = float(np.mean(_masked(ablated_scaled_error, assessed_mask)))
        forecast_change = _masked(ablated_forecast - full_forecast, assessed_mask)
        truth_effect = _masked(truth - _future(baseline), assessed_mask)
        truth_effect_rms = max(
            float(np.sqrt(np.mean(np.square(truth_effect)))), 1e-8
        )
        rows.append(
            {
                "schema_version": "cafe.capability_input_ablation_metric.v1",
                "model_id": model_id,
                "dataset_id": source["dataset_id"],
                "official_instance_id": source["official_instance_id"],
                "sample_id": ablation["sample_id"],
                "source_treatment_sample_id": source["sample_id"],
                "capability_id": source["capability_id"],
                "capability_level": int(source["capability_level"]),
                "assessed_target_indices": assessed,
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
        correlations = [
            float(row["effect_correlation"])
            for row in members
            if row["effect_correlation"] is not None
        ]
        aggregates.append(
            {
                "schema_version": "cafe.capability_effect_summary.v1",
                "model_id": model_id,
                "capability_id": capability,
                "capability_level": level,
                "official_instance_count": len(
                    {row["official_instance_id"] for row in members}
                ),
                "effect_nrmse_mean": float(
                    np.mean([row["effect_nrmse"] for row in members])
                ),
                "effect_correlation_mean": (
                    float(np.mean(correlations)) if correlations else None
                ),
                "effect_amplitude_ratio_mean": float(
                    np.mean([row["effect_amplitude_ratio"] for row in members])
                ),
            }
        )
    rank_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        rank_groups[(row["capability_id"], row["capability_level"])].append(row)
    for members in rank_groups.values():
        ordered = sorted(members, key=lambda row: (row["effect_nrmse_mean"], row["model_id"]))
        for rank, row in enumerate(ordered, start=1):
            row["effect_rank"] = rank
    return aggregates


def _load_prediction_part(
    record: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    if record is None:
        return {}
    path = validate_parquet_record(record)
    output: dict[str, np.ndarray] = {}
    for row in iter_prediction_parquet(path):
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate prediction for {sample_id} in {path}")
        output[sample_id] = np.asarray(row["forecast"], dtype=float)
    return output


def _mean_or_none(total: float, count: int) -> float | None:
    return None if count <= 0 else float(total / count)


def run_analysis(
    dataset_root: Path,
    *,
    gift_eval_dir: Path | None = None,
    replay_workers: int = 1,
) -> dict[str, Any]:
    """Analyse one source shard at a time; never load a model's full run."""

    generation_dir = dataset_root / "01_generation"
    inference_dir = dataset_root / "03_inference"
    analysis_dir = dataset_root / "04_analysis"
    generation_manifest = protocol.read_json(generation_dir / "manifest.json")
    source_root = generation_manifest.get("config", {}).get("gift_eval_source_root")
    gift_root = (
        Path(str(source_root)).resolve()
        if gift_eval_dir is None and source_root
        else (
            protocol.REPO_ROOT / "data" / "gift-eval"
            if gift_eval_dir is None
            else gift_eval_dir.resolve()
        )
    )
    inference_manifest_path = inference_dir / "manifest.json"
    inference_manifest = protocol.read_json(inference_manifest_path)
    if inference_manifest.get("schema_version") != INFERENCE_SCHEMA:
        raise ValueError("unsupported inference manifest")
    if not inference_manifest.get("complete"):
        raise ValueError("inference is incomplete")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    accuracy_rows_path = analysis_dir / "accuracy_rows.parquet"
    effect_rows_path = analysis_dir / "capability_effect_rows.parquet"
    ablation_rows_path = analysis_dir / "input_ablation_rows.parquet"
    accuracy_writer = TypedParquetWriter(
        accuracy_rows_path, schema=ACCURACY_METRIC_SCHEMA
    )
    effect_writer = TypedParquetWriter(
        effect_rows_path, schema=EFFECT_METRIC_SCHEMA
    )
    ablation_writer = TypedParquetWriter(
        ablation_rows_path, schema=ABLATION_METRIC_SCHEMA
    )
    accuracy_aggregates: defaultdict[tuple[str, str, str | None, int], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "mase": 0.0, "mae": 0.0}
    )
    effect_aggregates: defaultdict[tuple[str, str, int], dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "nrmse": 0.0,
            "amplitude": 0.0,
            "correlation": 0.0,
            "correlation_count": 0.0,
        }
    )
    ablation_aggregates: defaultdict[tuple[str, str, int], dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "full_mase": 0.0,
            "ablated_mase": 0.0,
            "degradation": 0.0,
            "response_ratio": 0.0,
        }
    )
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
    official_counts: defaultdict[str, int] = defaultdict(int)
    treatment_counts: defaultdict[str, int] = defaultdict(int)
    try:
        current_shard: int | None = None
        predictions_by_model: dict[str, dict[str, np.ndarray]] = {}
        baselines: dict[str, dict[str, Any]] = {}
        treatments: dict[str, dict[str, Any]] = {}
        for sample in iter_replayed_samples(
            generation_manifest,
            gift_eval_dir=gift_root,
            replay_workers=max(1, int(replay_workers)),
        ):
            shard = int(sample.get("source_shard_index", 0))
            if current_shard != shard:
                current_shard = shard
                predictions_by_model = {
                    model_id: _load_prediction_part(
                        prediction_parts_by_model[model_id].get(shard)
                    )
                    for model_id in model_ids
                }
                baselines.clear()
                treatments.clear()
            sample_id = str(sample["sample_id"])
            forecasts = {
                model_id: predictions_by_model[model_id].get(sample_id)
                for model_id in model_ids
            }
            table = str(sample["evaluation_table"])
            if table == "gift_eval_official_baseline":
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
            if table == "gift_eval_capability_treatment":
                baseline = baselines[str(sample["baseline_sample_id"])]
                future = _future(sample)
                treatments[sample_id] = {
                    "row": sample,
                    "future": future,
                    "forecasts": forecasts,
                    "baseline": baseline,
                }
                mask = baseline["mask"]
                scales = baseline["scales"]
                truth_delta = future - baseline["future"]
                affected = [int(value) for value in sample["affected_target_indices"]]
                assessed_mask = np.zeros_like(mask, dtype=bool)
                assessed_mask[:, affected] = mask[:, affected]
                effect_is_observed = bool(np.any(assessed_mask))
                truth_values = (
                    _masked(truth_delta, assessed_mask)
                    if effect_is_observed
                    else np.empty(0, dtype=float)
                )
                denominator = (
                    max(float(np.sqrt(np.mean(np.square(truth_values)))), 1e-8)
                    if effect_is_observed
                    else None
                )
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

                    if not effect_is_observed:
                        continue
                    assert denominator is not None

                    forecast_delta = forecast - baseline_forecast
                    forecast_values = _masked(forecast_delta, assessed_mask)
                    correlation = _correlation(forecast_values, truth_values)
                    nrmse = float(
                        np.sqrt(np.mean(np.square(forecast_values - truth_values)))
                        / denominator
                    )
                    amplitude = float(
                        np.sqrt(np.mean(np.square(forecast_values))) / denominator
                    )
                    effect_writer.write(
                        {
                            "schema_version": "cafe.capability_effect_metric.v2",
                            "model_id": model_id,
                            "dataset_id": sample["dataset_id"],
                            "official_instance_id": sample["official_instance_id"],
                            "sample_id": sample_id,
                            "capability_id": sample["capability_id"],
                            "capability_level": int(sample["capability_level"]),
                            "controlled_coordinate": sample["controlled_coordinate"],
                            "sampled_coordinate": float(sample["sampled_coordinate"]),
                            "effect_nrmse": nrmse,
                            "effect_correlation": correlation,
                            "effect_amplitude_ratio": amplitude,
                            "truth_effect_rms": denominator,
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
                    effect_aggregate["count"] += 1
                    effect_aggregate["nrmse"] += nrmse
                    effect_aggregate["amplitude"] += amplitude
                    if correlation is not None:
                        effect_aggregate["correlation"] += correlation
                        effect_aggregate["correlation_count"] += 1
                continue
            if table != "gift_eval_capability_input_ablation":
                raise ValueError(f"unknown evaluation table {table}")
            source = treatments[str(sample["input_ablation_source_sample_id"])]
            baseline = source["baseline"]
            truth = source["future"]
            mask = baseline["mask"]
            assessed = [int(value) for value in sample["assessed_target_indices"]]
            assessed_mask = np.zeros_like(mask, dtype=bool)
            assessed_mask[:, assessed] = mask[:, assessed]
            if not np.any(assessed_mask):
                continue
            scales = baseline["scales"]
            truth_effect = _masked(truth - baseline["future"], assessed_mask)
            truth_effect_rms = max(
                float(np.sqrt(np.mean(np.square(truth_effect)))), 1e-8
            )
            for model_id, forecast in forecasts.items():
                full_forecast = source["forecasts"][model_id]
                if forecast is None or full_forecast is None:
                    continue
                full_mase = float(
                    np.mean(
                        _masked(
                            np.abs(full_forecast - truth) / scales[None, :],
                            assessed_mask,
                        )
                    )
                )
                ablated_mase = float(
                    np.mean(
                        _masked(
                            np.abs(forecast - truth) / scales[None, :],
                            assessed_mask,
                        )
                    )
                )
                forecast_change = _masked(forecast - full_forecast, assessed_mask)
                response_ratio = float(
                    np.sqrt(np.mean(np.square(forecast_change))) / truth_effect_rms
                )
                ablation_writer.write(
                    {
                        "schema_version": "cafe.capability_input_ablation_metric.v2",
                        "model_id": model_id,
                        "dataset_id": sample["dataset_id"],
                        "official_instance_id": sample["official_instance_id"],
                        "sample_id": sample_id,
                        "source_treatment_sample_id": source["row"]["sample_id"],
                        "capability_id": source["row"]["capability_id"],
                        "capability_level": int(source["row"]["capability_level"]),
                        "assessed_target_indices": assessed,
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
        accuracy_count = accuracy_writer.close()
        effect_count = effect_writer.close()
        ablation_count = ablation_writer.close()
    except Exception:
        accuracy_writer.abort()
        effect_writer.abort()
        ablation_writer.abort()
        raise

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
        accuracy_aggregates.items(), key=lambda item: tuple(str(value) for value in item[0])
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
        effect_summary.append(
            {
                "schema_version": "cafe.capability_effect_summary.v2",
                "model_id": model_id,
                "capability_id": capability,
                "capability_level": level,
                "official_instance_count": count,
                "effect_nrmse_mean": _mean_or_none(values["nrmse"], count),
                "effect_correlation_mean": _mean_or_none(
                    values["correlation"], int(values["correlation_count"])
                ),
                "effect_amplitude_ratio_mean": _mean_or_none(
                    values["amplitude"], count
                ),
            }
        )
    for members in (
        [row for row in effect_summary if row["capability_id"] == capability and row["capability_level"] == level]
        for capability, level in {
            (row["capability_id"], row["capability_level"]) for row in effect_summary
        }
    ):
        for rank, row in enumerate(
            sorted(members, key=lambda value: (value["effect_nrmse_mean"], value["model_id"])),
            start=1,
        ):
            row["effect_rank"] = rank
    ablation_summary: list[dict[str, Any]] = []
    for (model_id, capability, level), values in sorted(ablation_aggregates.items()):
        count = int(values["count"])
        ablation_summary.append(
            {
                "schema_version": "cafe.capability_input_ablation_summary.v2",
                "model_id": model_id,
                "capability_id": capability,
                "capability_level": level,
                "official_instance_count": count,
                "full_input_mase_mean": _mean_or_none(values["full_mase"], count),
                "ablated_input_mase_mean": _mean_or_none(values["ablated_mase"], count),
                "input_ablation_mase_degradation_mean": _mean_or_none(
                    values["degradation"], count
                ),
                "input_ablation_response_ratio_mean": _mean_or_none(
                    values["response_ratio"], count
                ),
            }
        )
    for members in (
        [row for row in ablation_summary if row["capability_id"] == capability and row["capability_level"] == level]
        for capability, level in {
            (row["capability_id"], row["capability_level"])
            for row in ablation_summary
        }
    ):
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
        "execution": "single_replay_source_shard_all_model_prediction_join",
        "row_artifact_format": "parquet_zstd",
        "replay_workers": int(replay_workers),
        "estimands": {
            "official_accuracy": "GIFT-Eval official future MASE/MAE",
            "treatment_accuracy": "treatment future MASE/MAE on authentic MASE scale",
            "capability_effect": "forecast_delta_vs_truth_delta_on_affected_targets",
            "input_ablation_attribution": (
                "same_treatment_truth_with_auxiliary_histories_temporally_misaligned"
            ),
        },
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
            "accuracy_rows": parquet_file_record(accuracy_rows_path, row_count=accuracy_count),
            "accuracy_summary": protocol.file_record(accuracy_summary_path),
            "capability_effect_rows": parquet_file_record(effect_rows_path, row_count=effect_count),
            "capability_effect_summary": protocol.file_record(effect_summary_path),
            "input_ablation_rows": parquet_file_record(ablation_rows_path, row_count=ablation_count),
            "input_ablation_summary": protocol.file_record(ablation_summary_path),
        },
    }
    protocol.write_json(analysis_dir / "manifest.json", manifest)
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
