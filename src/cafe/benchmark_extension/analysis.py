from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.generation import PIPELINE_SCHEMA
from cafe.benchmark_extension.inference import INFERENCE_SCHEMA
from cafe.inference.runner import safe_filename


ANALYSIS_SCHEMA = "cafe.benchmark_extension_analysis.v1"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse official accuracy and capability-treatment effects."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _predictions(path: Path) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for row in protocol.iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate prediction for {sample_id}")
        output[sample_id] = np.asarray(row["forecast"], dtype=float)
    return output


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


def run_analysis(dataset_root: Path) -> dict[str, Any]:
    generation_dir = dataset_root / "01_generation"
    inference_dir = dataset_root / "03_inference"
    analysis_dir = dataset_root / "04_analysis"
    inference_manifest_path = inference_dir / "manifest.json"
    inference_manifest = protocol.read_json(inference_manifest_path)
    if inference_manifest.get("schema_version") != INFERENCE_SCHEMA:
        raise ValueError("unsupported inference manifest")
    if not inference_manifest.get("complete"):
        raise ValueError("inference is incomplete")
    baselines = {
        str(row["sample_id"]): row
        for row in protocol.iter_jsonl(generation_dir / "official_baselines.jsonl")
    }
    treatments = list(
        protocol.iter_jsonl(generation_dir / "capability_treatments.jsonl")
    )
    summaries: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for model_id in inference_manifest["config"]["models"]:
        prediction_path = (
            inference_dir
            / "models"
            / safe_filename(model_id)
            / "predictions"
            / f"{safe_filename(model_id)}.jsonl"
        )
        model_summary, model_effects = analyse_model(
            model_id,
            baselines,
            treatments,
            _predictions(prediction_path),
        )
        summaries.append(model_summary)
        effects.extend(model_effects)
    aggregate_effects = _aggregate_effects(effects)
    accuracy_path = analysis_dir / "official_accuracy.json"
    effect_rows_path = analysis_dir / "capability_effect_rows.jsonl"
    effect_summary_path = analysis_dir / "capability_effect_summary.json"
    protocol.write_json(accuracy_path, {"models": summaries})
    protocol.write_jsonl(effect_rows_path, effects)
    protocol.write_json(effect_summary_path, {"rows": aggregate_effects})
    config = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "estimands": {
            "official_accuracy": "GIFT-Eval official future MASE/MAE",
            "capability_effect": "forecast_delta_vs_truth_delta_on_affected_targets",
        },
        "ranking_policy": "separate_rank_by_capability_and_level_lower_effect_nrmse",
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
            "capability_effect_rows": {
                **protocol.file_record(effect_rows_path),
                "row_count": len(effects),
            },
            "capability_effect_summary": protocol.file_record(effect_summary_path),
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
    manifest = run_analysis(dataset_root)
    print(protocol.canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
