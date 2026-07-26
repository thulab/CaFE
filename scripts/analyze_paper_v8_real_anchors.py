#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

import paper_v8_pipeline_common as v8
import run_paper_e2_dynamic_stability as engine


SHARD_TEMPLATE = "seed_{seed_start:06d}_{seed_stop:06d}"
FIXED_CONTEXT_POLICY = f"fixed_l{v8.FIXED_CONTEXT_LENGTH}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze auxiliary Paper v8 real-anchor MASE and compare its "
            "ranking with synthetic capability rankings."
        )
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validated_record_path(
    record: dict[str, Any],
    *,
    expected_rows: int | None = None,
) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"recorded file is missing: {path}")
    if (
        record.get("bytes") is None
        or int(record["bytes"]) != path.stat().st_size
    ):
        raise ValueError(f"recorded byte size mismatch: {path}")
    if (
        not record.get("sha256")
        or str(record["sha256"]) != v8.file_sha256(path)
    ):
        raise ValueError(f"recorded sha256 mismatch: {path}")
    if expected_rows is not None and int(record.get("row_count", -1)) != (
        expected_rows
    ):
        raise ValueError(f"recorded row count mismatch: {path}")
    return path


def unique_rows_by_sample_id(
    path: Path,
    *,
    expected_rows: int,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in v8.iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        output[sample_id] = row
    if len(output) != expected_rows:
        raise ValueError(
            f"row count mismatch for {path}: {len(output)} != {expected_rows}"
        )
    return output


def competition_ranks(
    values: dict[str, float],
) -> dict[str, int]:
    return {
        key: 1 + sum(other < value for other in values.values())
        for key, value in values.items()
    }


def score_real_anchor_dataset(
    *,
    dataset_id: str,
    views: dict[str, dict[str, Any]],
    predictions_by_model: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_ids = set(views)
    metric_rows: list[dict[str, Any]] = []
    for model_id, predictions in predictions_by_model.items():
        if set(predictions) != expected_ids:
            missing = sorted(expected_ids - set(predictions))
            extra = sorted(set(predictions) - expected_ids)
            raise ValueError(
                f"{dataset_id}/{model_id} real-anchor identity mismatch: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        for sample_id, view in views.items():
            target = np.asarray(view["target"], dtype=float)
            if target.ndim == 1:
                target = target[:, None]
            context = int(view["context_length"])
            horizon = int(view["horizon"])
            truth = target[context : context + horizon]
            forecast = np.asarray(
                predictions[sample_id]["forecast"],
                dtype=float,
            )
            if forecast.ndim == 1:
                forecast = forecast[:, None]
            if truth.shape != forecast.shape:
                raise ValueError(
                    f"{dataset_id}/{model_id}/{sample_id} forecast shape "
                    f"mismatch: {forecast.shape} != {truth.shape}"
                )
            scale = float(view["mase_scale"])
            if (
                scale <= 0.0
                or not math.isfinite(scale)
                or not np.isfinite(truth).all()
                or not np.isfinite(forecast).all()
            ):
                raise ValueError(
                    f"invalid real-anchor scoring values: "
                    f"{dataset_id}/{model_id}/{sample_id}"
                )
            mae = float(np.mean(np.abs(truth - forecast)))
            metric_rows.append(
                {
                    "schema_version": (
                        "paper_v8_real_anchor_prediction_metric.v1"
                    ),
                    "dataset_id": dataset_id,
                    "model_id": model_id,
                    "sample_id": sample_id,
                    "anchor_id": view.get("anchor_id"),
                    "context_length": context,
                    "horizon": horizon,
                    "target_dim": int(view["target_dim"]),
                    "mase_scale": scale,
                    "mae": mae,
                    "mase": mae / scale,
                }
            )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_model[str(row["model_id"])].append(row)
    mean_mase = {
        model_id: float(np.mean([row["mase"] for row in rows]))
        for model_id, rows in by_model.items()
    }
    ranks = competition_ranks(mean_mase)
    dataset_rows = [
        {
            "schema_version": "paper_v8_real_anchor_dataset_score.v1",
            "dataset_id": dataset_id,
            "model_id": model_id,
            "anchor_count": len(rows),
            "mean_mase": mean_mase[model_id],
            "median_mase": float(median(row["mase"] for row in rows)),
            "mase_rank": ranks[model_id],
        }
        for model_id, rows in sorted(by_model.items())
    ]
    return metric_rows, dataset_rows


def synthetic_dataset_rows(
    scores: Iterable[dict[str, Any]],
    *,
    dataset_id: str,
    models: list[str],
    capabilities: list[str],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in scores
        if row["dataset_id"] == dataset_id
        and row["context_policy"] == FIXED_CONTEXT_POLICY
        and row["evaluation_table"] == "main"
        and row["generator_family_role"] == "primary"
        and row["model_id"] in models
    ]
    expected = len(models) * len(capabilities)
    if len(rows) != expected:
        raise ValueError(
            f"{dataset_id} fixed synthetic score count mismatch: "
            f"{len(rows)} != {expected}"
        )
    output = []
    accuracy_means: dict[str, float] = {}
    mechanism_means: dict[str, float] = {}
    for model_id in models:
        model_rows = [
            row for row in rows if str(row["model_id"]) == model_id
        ]
        present = {str(row["capability_id"]) for row in model_rows}
        if present != set(capabilities):
            raise ValueError(
                f"{dataset_id}/{model_id} capability coverage mismatch"
            )
        accuracy_means[model_id] = float(
            np.mean([row["accuracy_rank"] for row in model_rows])
        )
        mechanism_means[model_id] = float(
            np.mean([row["mechanism_rank"] for row in model_rows])
        )
    accuracy_ranks = competition_ranks(accuracy_means)
    mechanism_ranks = competition_ranks(mechanism_means)
    for model_id in models:
        output.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "mean_capability_accuracy_rank": accuracy_means[model_id],
                "capability_accuracy_rank": accuracy_ranks[model_id],
                "mean_capability_mechanism_rank": mechanism_means[model_id],
                "capability_mechanism_rank": mechanism_ranks[model_id],
            }
        )
    return output


def kendall_for_models(
    left: dict[str, int],
    right: dict[str, int],
    models: list[str],
) -> float:
    return float(
        engine.kendall_tau_b(
            np.asarray([left[model_id] for model_id in models], dtype=float),
            np.asarray(
                [right[model_id] for model_id in models],
                dtype=float,
            ),
        )
    )


def ranking_comparison(
    *,
    real_metrics: list[dict[str, Any]],
    real_dataset_scores: list[dict[str, Any]],
    synthetic_scores: list[dict[str, Any]],
    synthetic_dataset_scores: list[dict[str, Any]],
    models: list[str],
    datasets: list[str],
    capabilities: list[str],
) -> dict[str, Any]:
    real_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in real_dataset_scores:
        real_by_model[str(row["model_id"])].append(row)
    metric_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in real_metrics:
        metric_by_model[str(row["model_id"])].append(row)
    synthetic_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in synthetic_dataset_scores:
        synthetic_by_model[str(row["model_id"])].append(row)

    real_macro_mase = {
        model_id: float(
            np.mean([row["mean_mase"] for row in real_by_model[model_id]])
        )
        for model_id in models
    }
    real_macro_ranks = competition_ranks(real_macro_mase)
    synthetic_accuracy_means = {
        model_id: float(
            np.mean(
                [
                    row["mean_capability_accuracy_rank"]
                    for row in synthetic_by_model[model_id]
                ]
            )
        )
        for model_id in models
    }
    synthetic_mechanism_means = {
        model_id: float(
            np.mean(
                [
                    row["mean_capability_mechanism_rank"]
                    for row in synthetic_by_model[model_id]
                ]
            )
        )
        for model_id in models
    }
    synthetic_accuracy_ranks = competition_ranks(
        synthetic_accuracy_means
    )
    synthetic_mechanism_ranks = competition_ranks(
        synthetic_mechanism_means
    )
    overall = []
    for model_id in models:
        dataset_rows = real_by_model[model_id]
        metric_rows = metric_by_model[model_id]
        overall.append(
            {
                "model_id": model_id,
                "real_anchor_macro_mean_dataset_mase": (
                    real_macro_mase[model_id]
                ),
                "real_anchor_weighted_mean_mase": float(
                    np.mean([row["mase"] for row in metric_rows])
                ),
                "real_anchor_macro_rank": real_macro_ranks[model_id],
                "real_anchor_mean_dataset_rank": float(
                    np.mean([row["mase_rank"] for row in dataset_rows])
                ),
                "real_anchor_dataset_wins": sum(
                    int(row["mase_rank"]) == 1 for row in dataset_rows
                ),
                "synthetic_mean_capability_accuracy_rank": (
                    synthetic_accuracy_means[model_id]
                ),
                "synthetic_capability_accuracy_rank": (
                    synthetic_accuracy_ranks[model_id]
                ),
                "synthetic_mean_capability_mechanism_rank": (
                    synthetic_mechanism_means[model_id]
                ),
                "synthetic_capability_mechanism_rank": (
                    synthetic_mechanism_ranks[model_id]
                ),
            }
        )

    dataset_comparisons = []
    real_index = {
        (str(row["dataset_id"]), str(row["model_id"])): row
        for row in real_dataset_scores
    }
    synthetic_index = {
        (str(row["dataset_id"]), str(row["model_id"])): row
        for row in synthetic_dataset_scores
    }
    for dataset_id in datasets:
        real_ranks = {
            model_id: int(
                real_index[(dataset_id, model_id)]["mase_rank"]
            )
            for model_id in models
        }
        accuracy_ranks = {
            model_id: int(
                synthetic_index[(dataset_id, model_id)][
                    "capability_accuracy_rank"
                ]
            )
            for model_id in models
        }
        mechanism_ranks = {
            model_id: int(
                synthetic_index[(dataset_id, model_id)][
                    "capability_mechanism_rank"
                ]
            )
            for model_id in models
        }
        dataset_comparisons.append(
            {
                "dataset_id": dataset_id,
                "real_anchor_accuracy_kendall_tau_b": (
                    kendall_for_models(
                        real_ranks,
                        accuracy_ranks,
                        models,
                    )
                ),
                "real_anchor_mechanism_kendall_tau_b": (
                    kendall_for_models(
                        real_ranks,
                        mechanism_ranks,
                        models,
                    )
                ),
            }
        )

    fixed_scores = [
        row
        for row in synthetic_scores
        if row["context_policy"] == FIXED_CONTEXT_POLICY
        and row["evaluation_table"] == "main"
        and row["generator_family_role"] == "primary"
        and row["model_id"] in models
    ]
    capability_comparisons = []
    for capability in capabilities:
        capability_rows = [
            row
            for row in fixed_scores
            if str(row["capability_id"]) == capability
        ]
        accuracy_mean = {
            model_id: float(
                np.mean(
                    [
                        row["accuracy_rank"]
                        for row in capability_rows
                        if row["model_id"] == model_id
                    ]
                )
            )
            for model_id in models
        }
        mechanism_mean = {
            model_id: float(
                np.mean(
                    [
                        row["mechanism_rank"]
                        for row in capability_rows
                        if row["model_id"] == model_id
                    ]
                )
            )
            for model_id in models
        }
        accuracy_ranks = competition_ranks(accuracy_mean)
        mechanism_ranks = competition_ranks(mechanism_mean)
        capability_comparisons.append(
            {
                "capability_id": capability,
                "accuracy_top_model": min(
                    models,
                    key=lambda model_id: (
                        accuracy_mean[model_id],
                        model_id,
                    ),
                ),
                "mechanism_top_model": min(
                    models,
                    key=lambda model_id: (
                        mechanism_mean[model_id],
                        model_id,
                    ),
                ),
                "real_anchor_accuracy_kendall_tau_b": (
                    kendall_for_models(
                        real_macro_ranks,
                        accuracy_ranks,
                        models,
                    )
                ),
                "real_anchor_mechanism_kendall_tau_b": (
                    kendall_for_models(
                        real_macro_ranks,
                        mechanism_ranks,
                        models,
                    )
                ),
                "model_rows": [
                    {
                        "model_id": model_id,
                        "mean_accuracy_rank": accuracy_mean[model_id],
                        "accuracy_rank": accuracy_ranks[model_id],
                        "mean_mechanism_rank": mechanism_mean[model_id],
                        "mechanism_rank": mechanism_ranks[model_id],
                    }
                    for model_id in models
                ],
            }
        )

    return {
        "schema_version": (
            "paper_v8_real_anchor_capability_ranking_comparison.v1"
        ),
        "scope": {
            "real_anchor_role": "auxiliary_real_data_accuracy_table",
            "included_in_synthetic_mechanism_ranking": False,
            "real_anchor_aggregation": (
                "mean MASE within dataset, then explicit equal-dataset "
                "macro mean; anchor-weighted mean reported separately"
            ),
            "synthetic_ranking": (
                f"{FIXED_CONTEXT_POLICY} clean primary main-table ranks"
            ),
        },
        "overall_model_rows": overall,
        "overall_rank_correlations": {
            "real_anchor_vs_synthetic_accuracy_kendall_tau_b": (
                kendall_for_models(
                    real_macro_ranks,
                    synthetic_accuracy_ranks,
                    models,
                )
            ),
            "real_anchor_vs_synthetic_mechanism_kendall_tau_b": (
                kendall_for_models(
                    real_macro_ranks,
                    synthetic_mechanism_ranks,
                    models,
                )
            ),
        },
        "dataset_rank_correlations": dataset_comparisons,
        "dataset_rank_correlation_summary": {
            "median_real_anchor_vs_synthetic_accuracy_kendall_tau_b": (
                float(
                    median(
                        row["real_anchor_accuracy_kendall_tau_b"]
                        for row in dataset_comparisons
                    )
                )
            ),
            "median_real_anchor_vs_synthetic_mechanism_kendall_tau_b": (
                float(
                    median(
                        row["real_anchor_mechanism_kendall_tau_b"]
                        for row in dataset_comparisons
                    )
                )
            ),
        },
        "capability_rank_comparisons": capability_comparisons,
    }


def render_report(
    *,
    experiment_id: str,
    models: list[str],
    datasets: list[str],
    real_dataset_scores: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    lines = [
        "# Paper v8 real-anchor 预测与能力排名对比",
        "",
        f"- 实验：`{experiment_id}`",
        "- real-anchor 是辅助真实数据 Accuracy 表，不进入 synthetic mechanism 排名。",
        "- 先在每个数据集内平均 anchor MASE；跨数据集主汇总采用等数据集权重 macro mean。",
        "- `weighted MASE` 另外按 anchor 数量加权；它会偏向拥有 256 个 anchor 的数据集。",
        "",
        "## 跨数据集汇总",
        "",
        (
            "| model | macro mean MASE | macro rank | weighted MASE | "
            "mean dataset rank | dataset wins | synthetic accuracy rank | "
            "synthetic mechanism rank |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        comparison["overall_model_rows"],
        key=lambda value: (
            value["real_anchor_macro_rank"],
            value["model_id"],
        ),
    ):
        lines.append(
            f"| {row['model_id']} | "
            f"{row['real_anchor_macro_mean_dataset_mase']:.3f} | "
            f"{row['real_anchor_macro_rank']} | "
            f"{row['real_anchor_weighted_mean_mase']:.3f} | "
            f"{row['real_anchor_mean_dataset_rank']:.2f} | "
            f"{row['real_anchor_dataset_wins']} | "
            f"{row['synthetic_capability_accuracy_rank']} | "
            f"{row['synthetic_capability_mechanism_rank']} |"
        )
    correlations = comparison["overall_rank_correlations"]
    dataset_correlations = comparison[
        "dataset_rank_correlation_summary"
    ]
    overall_accuracy_tau = correlations[
        "real_anchor_vs_synthetic_accuracy_kendall_tau_b"
    ]
    overall_mechanism_tau = correlations[
        "real_anchor_vs_synthetic_mechanism_kendall_tau_b"
    ]
    median_accuracy_tau = dataset_correlations[
        "median_real_anchor_vs_synthetic_accuracy_kendall_tau_b"
    ]
    median_mechanism_tau = dataset_correlations[
        "median_real_anchor_vs_synthetic_mechanism_kendall_tau_b"
    ]
    lines.extend(
        [
            "",
            "## 排名相关性",
            "",
            (
                "- 跨数据集总体 real-anchor vs synthetic Accuracy "
                "Kendall tau-b："
                f"{overall_accuracy_tau:.3f}"
            ),
            (
                "- 跨数据集总体 real-anchor vs synthetic Mechanism "
                "Kendall tau-b："
                f"{overall_mechanism_tau:.3f}"
            ),
            (
                "- 数据集内相关性的中位数：Accuracy "
                f"{median_accuracy_tau:.3f}"
                "，Mechanism "
                f"{median_mechanism_tau:.3f}"
            ),
            "",
            "## 各能力与真实预测排名的关系",
            "",
            (
                "| capability | synthetic accuracy top | tau-b vs real | "
                "synthetic mechanism top | tau-b vs real |"
            ),
            "|---|---|---:|---|---:|",
        ]
    )
    for row in comparison["capability_rank_comparisons"]:
        lines.append(
            f"| {row['capability_id']} | "
            f"{row['accuracy_top_model']} | "
            f"{row['real_anchor_accuracy_kendall_tau_b']:.3f} | "
            f"{row['mechanism_top_model']} | "
            f"{row['real_anchor_mechanism_kendall_tau_b']:.3f} |"
        )

    score_index = {
        (str(row["dataset_id"]), str(row["model_id"])): row
        for row in real_dataset_scores
    }
    lines.extend(
        [
            "",
            "## 各数据集 real-anchor mean MASE（括号内为数据集内排名）",
            "",
            "| dataset | anchors | "
            + " | ".join(models)
            + " |",
            "|---|---:|" + "---:|" * len(models),
        ]
    )
    for dataset_id in datasets:
        first = score_index[(dataset_id, models[0])]
        values = []
        for model_id in models:
            row = score_index[(dataset_id, model_id)]
            values.append(
                f"{row['mean_mase']:.3f} ({row['mase_rank']})"
            )
        lines.append(
            f"| {dataset_id} | {first['anchor_count']} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    experiment_manifest_path = experiment_root / "experiment_manifest.json"
    experiment_manifest = v8.read_json(experiment_manifest_path)
    protocol = experiment_manifest["protocol"]
    datasets = list(protocol["dataset_ids"])
    models = list(args.models or protocol["models"])
    if len(models) != len(set(models)):
        raise ValueError("model ids must be unique")
    if set(models) - set(protocol["models"]):
        raise ValueError("requested models are outside the experiment protocol")
    seed_start = (
        int(args.seed_start)
        if args.seed_start is not None
        else int(protocol["seed_start"])
    )
    seed_count = (
        int(args.seed_count)
        if args.seed_count is not None
        else int(protocol["seed_count"])
    )
    if seed_start < 0 or seed_count < 1:
        raise ValueError("invalid seed range")
    shard_name = SHARD_TEMPLATE.format(
        seed_start=seed_start,
        seed_stop=seed_start + seed_count,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else experiment_root / "posthoc_real_anchor_comparison"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"refusing to overwrite non-empty posthoc directory: {output_dir}"
        )

    all_metrics: list[dict[str, Any]] = []
    all_dataset_scores: list[dict[str, Any]] = []
    all_synthetic_scores: list[dict[str, Any]] = []
    all_synthetic_dataset_scores: list[dict[str, Any]] = []
    inputs = []
    for dataset_id in datasets:
        inference_dir = (
            experiment_root / dataset_id / "03_inference" / shard_name
        )
        inference_manifest_path = (
            inference_dir / "inference_manifest.json"
        )
        task_manifest_path = inference_dir / "task_manifest.json"
        inference_manifest = v8.read_json(inference_manifest_path)
        if not bool(inference_manifest.get("complete")):
            raise ValueError(f"incomplete inference: {dataset_id}")
        if str(inference_manifest.get("task_manifest_sha256")) != (
            v8.file_sha256(task_manifest_path)
        ):
            raise ValueError(f"task manifest binding mismatch: {dataset_id}")
        task_manifest = v8.read_json(task_manifest_path)
        component = task_manifest["task_components"]["real_anchors"]
        expected_rows = int(component["row_count"])
        if expected_rows != int(task_manifest["real_anchor_view_count"]):
            raise ValueError(f"real-anchor task count mismatch: {dataset_id}")
        view_path = validated_record_path(
            component,
            expected_rows=expected_rows,
        )
        views = unique_rows_by_sample_id(
            view_path,
            expected_rows=expected_rows,
        )

        prediction_records = {
            str(record["model_id"]): record
            for record in inference_manifest["predictions"]["real_anchor"][
                "files"
            ]
        }
        predictions_by_model = {}
        for model_id in models:
            record = prediction_records.get(model_id)
            if record is None:
                raise ValueError(
                    f"missing real-anchor predictions: "
                    f"{dataset_id}/{model_id}"
                )
            prediction_path = validated_record_path(
                record,
                expected_rows=expected_rows,
            )
            predictions_by_model[model_id] = unique_rows_by_sample_id(
                prediction_path,
                expected_rows=expected_rows,
            )

        metrics, dataset_scores = score_real_anchor_dataset(
            dataset_id=dataset_id,
            views=views,
            predictions_by_model=predictions_by_model,
        )
        all_metrics.extend(metrics)
        all_dataset_scores.extend(dataset_scores)

        analysis_dir = (
            experiment_root / dataset_id / "04_analysis" / shard_name
        )
        analysis_manifest_path = analysis_dir / "analysis_manifest.json"
        analysis_manifest = v8.read_json(analysis_manifest_path)
        scores_record = analysis_manifest["files"]["scores"]
        scores_path = validated_record_path(scores_record)
        synthetic_scores = v8.read_json(scores_path)["scores"]
        all_synthetic_scores.extend(synthetic_scores)
        all_synthetic_dataset_scores.extend(
            synthetic_dataset_rows(
                synthetic_scores,
                dataset_id=dataset_id,
                models=models,
                capabilities=list(protocol["capabilities"]),
            )
        )
        inputs.append(
            {
                "dataset_id": dataset_id,
                "anchor_count": expected_rows,
                "task_manifest_sha256": v8.file_sha256(
                    task_manifest_path
                ),
                "inference_manifest_sha256": v8.file_sha256(
                    inference_manifest_path
                ),
                "analysis_manifest_sha256": v8.file_sha256(
                    analysis_manifest_path
                ),
            }
        )

    comparison = ranking_comparison(
        real_metrics=all_metrics,
        real_dataset_scores=all_dataset_scores,
        synthetic_scores=all_synthetic_scores,
        synthetic_dataset_scores=all_synthetic_dataset_scores,
        models=models,
        datasets=datasets,
        capabilities=list(protocol["capabilities"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metric_path = output_dir / "real_anchor_metrics.jsonl"
    dataset_score_path = output_dir / "dataset_real_anchor_scores.json"
    comparison_path = output_dir / "ranking_comparison.json"
    report_path = output_dir / "REPORT_ZH.md"
    v8.write_jsonl(metric_path, all_metrics)
    v8.write_json(
        dataset_score_path,
        {"dataset_real_anchor_scores": all_dataset_scores},
    )
    v8.write_json(comparison_path, comparison)
    report_path.write_text(
        render_report(
            experiment_id=str(experiment_manifest["experiment_id"]),
            models=models,
            datasets=datasets,
            real_dataset_scores=all_dataset_scores,
            comparison=comparison,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "paper_v8_real_anchor_posthoc_manifest.v1",
        "created_at": v8.utc_now(),
        "experiment_id": experiment_manifest["experiment_id"],
        "experiment_manifest_sha256": v8.file_sha256(
            experiment_manifest_path
        ),
        "seed_start": seed_start,
        "seed_count": seed_count,
        "models": models,
        "datasets": datasets,
        "scope": comparison["scope"],
        "coverage": {
            "dataset_count": len(datasets),
            "model_count": len(models),
            "real_anchor_metric_count": len(all_metrics),
            "dataset_score_count": len(all_dataset_scores),
        },
        "inputs": inputs,
        "files": {
            "real_anchor_metrics": v8.file_record(metric_path),
            "dataset_real_anchor_scores": v8.file_record(
                dataset_score_path
            ),
            "ranking_comparison": v8.file_record(comparison_path),
            "report": v8.file_record(report_path),
        },
    }
    v8.write_json(output_dir / "analysis_manifest.json", manifest)
    print(
        v8.canonical_json(
            {
                "dataset_count": len(datasets),
                "model_count": len(models),
                "metric_count": len(all_metrics),
                "output": str(output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
