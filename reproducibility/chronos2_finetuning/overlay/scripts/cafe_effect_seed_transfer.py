#!/usr/bin/env python3
"""Evaluate CaFE paired official/treatment effect NRMSE across fine-tuning steps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import cafe_seed_transfer as transfer


def _prepared_arrays(row: Mapping[str, Any]) -> dict[str, Any]:
    target = np.asarray(row["target"], dtype=np.float32)
    horizon = int(row["horizon"])
    covariates, n_future_covariates = transfer._ordered_covariates(row)
    context = np.concatenate((target.T, covariates.T), axis=0)
    future = np.full((context.shape[0], horizon), np.nan, dtype=np.float32)
    if n_future_covariates:
        future[-n_future_covariates:] = covariates[-horizon:, -n_future_covariates:].T
    return {
        "context": context.tolist(),
        "future_covariates": future.tolist(),
        "n_targets": int(target.shape[1]),
        "n_covariates": int(covariates.shape[1]),
        "n_future_covariates": n_future_covariates,
    }


def _effect_features() -> Any:
    import datasets

    matrix = datasets.List(datasets.List(datasets.Value("float32")))
    return datasets.Features(
        {
            "baseline_context": matrix,
            "treatment_context": matrix,
            "baseline_future_covariates": matrix,
            "treatment_future_covariates": matrix,
            "n_targets": datasets.Value("int64"),
            "n_covariates": datasets.Value("int64"),
            "n_future_covariates": datasets.Value("int64"),
            "horizon": datasets.Value("int64"),
            "dataset_id": datasets.Value("string"),
            "official_instance_id": datasets.Value("string"),
            "sample_id": datasets.Value("string"),
            "capability_id": datasets.Value("string"),
            "capability_level": datasets.Value("int64"),
            "augmentation_seed": datasets.Value("int64"),
            "affected_target_indices": datasets.List(datasets.Value("int64")),
            "future_observed_mask": datasets.List(
                datasets.List(datasets.Value("bool"))
            ),
            "mase_scale_by_target": datasets.List(datasets.Value("float64")),
        }
    )


def _iter_dense_pairs(
    args: argparse.Namespace,
    selections: Mapping[str, set[str] | None],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    from cafe.benchmark_extension.generation import (
        _replay_contract_instance,
        iter_replay_contract_work_items,
    )

    for dataset_root in transfer._dataset_roots(args.experiment_root):
        if dataset_root.name not in selections:
            continue
        manifest = transfer._load_local_manifest(dataset_root)
        if transfer._manifest_horizon(manifest) != args.horizon:
            continue
        selected = selections[dataset_root.name]
        for instance, baseline, treatments, _ablations in iter_replay_contract_work_items(
            manifest, gift_eval_dir=args.gift_eval_dir
        ):
            if not transfer._fold_matches(
                str(baseline["official_instance_id"]),
                fold_count=args.official_fold_count,
                heldout_fold=args.heldout_fold,
                role=args.fold_role,
                fold_salt=args.fold_salt,
            ):
                continue
            filtered = [
                row
                for row in treatments
                if str(row["capability_id"]) in args.capabilities
                and (selected is None or str(row["sample_id"]) in selected)
            ]
            if not filtered:
                continue
            history_start = max(0, int(instance.context_length) - args.maximum_context)
            dense = _replay_contract_instance(
                instance, baseline, filtered, [], history_start=history_start
            )
            baseline_dense = dense[0]
            for treatment_dense in dense[1:]:
                yield baseline_dense, treatment_dense


def command_prepare(args: argparse.Namespace) -> None:
    import datasets

    transfer._configure_cafe_import(args.cafe_root)
    if args.output.exists():
        raise FileExistsError(args.output)
    selections, audit = transfer._selection_audit(args)

    def generate() -> Iterator[dict[str, Any]]:
        for baseline, treatment in _iter_dense_pairs(args, selections):
            baseline_arrays = _prepared_arrays(baseline)
            treatment_arrays = _prepared_arrays(treatment)
            for key in ("n_targets", "n_covariates", "n_future_covariates"):
                if baseline_arrays[key] != treatment_arrays[key]:
                    raise ValueError(f"Baseline/treatment {key} mismatch")
            yield {
                "baseline_context": baseline_arrays["context"],
                "treatment_context": treatment_arrays["context"],
                "baseline_future_covariates": baseline_arrays["future_covariates"],
                "treatment_future_covariates": treatment_arrays["future_covariates"],
                "n_targets": treatment_arrays["n_targets"],
                "n_covariates": treatment_arrays["n_covariates"],
                "n_future_covariates": treatment_arrays["n_future_covariates"],
                "horizon": int(treatment["horizon"]),
                "dataset_id": str(treatment["dataset_id"]),
                "official_instance_id": str(treatment["official_instance_id"]),
                "sample_id": str(treatment["sample_id"]),
                "capability_id": str(treatment["capability_id"]),
                "capability_level": int(treatment["capability_level"]),
                "augmentation_seed": int(treatment["augmentation_seed"]),
                "affected_target_indices": [
                    int(value) for value in treatment["affected_target_indices"]
                ],
                "future_observed_mask": np.asarray(
                    treatment["future_observed_mask"], dtype=bool
                ).tolist(),
                "mase_scale_by_target": np.asarray(
                    baseline["mase_scale_by_target"], dtype=np.float64
                ).tolist(),
            }

    dataset = datasets.Dataset.from_generator(
        generate, features=_effect_features(), writer_batch_size=args.writer_batch_size
    )
    dataset.save_to_disk(str(args.output))
    audit.update(
        {
            "schema_version": "chronos2.cafe_effect_seed_transfer_selection.v1",
            "horizon": args.horizon,
            "materialized_pair_count": len(dataset),
            "maximum_context": args.maximum_context,
        }
    )
    (args.output / "cafe_effect_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {len(dataset)} official/treatment pairs at {args.output}")


def _pair_input(row: Mapping[str, Any], prefix: str) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    context = torch.as_tensor(row[f"{prefix}_context"], dtype=torch.float32)
    horizon = int(row["horizon"])
    n_targets = int(row["n_targets"])
    origin = context.shape[-1] - horizon
    return (
        {
            "context": context[:, :origin],
            "future_covariates": torch.as_tensor(
                row[f"{prefix}_future_covariates"], dtype=torch.float32
            ),
            "n_targets": n_targets,
            "n_covariates": int(row["n_covariates"]),
            "n_future_covariates": int(row["n_future_covariates"]),
        },
        context[:n_targets, origin:].T.numpy(),
    )


def _effect_sums(
    row: Mapping[str, Any],
    baseline_truth: np.ndarray,
    treatment_truth: np.ndarray,
    baseline_forecast: np.ndarray,
    treatment_forecast: np.ndarray,
) -> tuple[float, float, float, int, bool]:
    truth = np.asarray(treatment_truth - baseline_truth, dtype=np.float64)
    forecast = np.asarray(treatment_forecast - baseline_forecast, dtype=np.float64)
    mask = np.asarray(row["future_observed_mask"], dtype=bool)
    assessed = np.zeros_like(mask)
    affected = np.asarray(row["affected_target_indices"], dtype=int)
    assessed[:, affected] = mask[:, affected]
    scales = np.asarray(row["mase_scale_by_target"], dtype=np.float64)
    valid = assessed & np.isfinite(truth) & np.isfinite(forecast)
    truth_standardized = (truth / scales[None, :])[valid]
    forecast_standardized = (forecast / scales[None, :])[valid]
    difference = forecast_standardized - truth_standardized
    truth_squared = float(np.sum(np.square(truth_standardized)))
    low_signal = (
        truth_standardized.size == 0
        or float(np.sqrt(np.mean(np.square(truth_standardized)))) < 0.05 - 1e-12
    )
    return (
        float(np.sum(np.square(difference))),
        truth_squared,
        float(np.sum(np.square(forecast_standardized))),
        int(truth_standardized.size),
        low_signal,
    )


def command_evaluate(args: argparse.Namespace) -> None:
    import datasets
    import torch

    from chronos import Chronos2Pipeline

    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = datasets.load_from_disk(str(args.dataset)).shard(
        num_shards=args.world_size, index=args.rank, contiguous=True
    ).with_format("torch")
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
    )
    aggregates: defaultdict[tuple[str, str, int], list[float]] = defaultdict(
        lambda: [0.0] * 7
    )
    batch_inputs: list[dict[str, Any]] = []
    batch_rows: list[Mapping[str, Any]] = []
    batch_truth: list[tuple[np.ndarray, np.ndarray]] = []

    def flush() -> None:
        if not batch_rows:
            return
        _quantiles, means = pipeline.predict_quantiles(
            batch_inputs,
            prediction_length=args.horizon,
            quantile_levels=[0.5],
            batch_size=args.batch_size,
        )
        for index, (row, truths) in enumerate(zip(batch_rows, batch_truth, strict=True)):
            baseline_forecast = means[2 * index].float().cpu().numpy().T
            treatment_forecast = means[2 * index + 1].float().cpu().numpy().T
            values = _effect_sums(
                row, truths[0], truths[1], baseline_forecast, treatment_forecast
            )
            key = (
                str(row["dataset_id"]),
                str(row["capability_id"]),
                int(row["capability_level"]),
            )
            aggregates[key][5] += 1.0
            if values[4]:
                aggregates[key][4] += 1.0
            else:
                for position, value in enumerate(values[:4]):
                    aggregates[key][position] += float(value)
                aggregates[key][6] += 1.0
        batch_inputs.clear()
        batch_rows.clear()
        batch_truth.clear()

    for row in dataset:
        baseline_input, baseline_truth = _pair_input(row, "baseline")
        treatment_input, treatment_truth = _pair_input(row, "treatment")
        batch_inputs.extend((baseline_input, treatment_input))
        batch_rows.append(row)
        batch_truth.append((baseline_truth, treatment_truth))
        if len(batch_rows) >= args.input_batch_size:
            flush()
    flush()
    result = {
        "schema_version": "chronos2.cafe_effect_evaluation_part.v1",
        "corpus": args.corpus,
        "horizon": args.horizon,
        "model": args.model_label,
        "model_path": args.model,
        "step": args.step,
        "rank": args.rank,
        "world_size": args.world_size,
        "pair_count": len(dataset),
        "strata": [
            {
                "dataset_id": dataset_id,
                "capability_id": capability_id,
                "capability_level": level,
                "standardized_squared_error_sum": values[0],
                "standardized_truth_squared_sum": values[1],
                "standardized_forecast_squared_sum": values[2],
                "observed_effect_cell_count": int(values[3]),
                "low_signal_pair_count": int(values[4]),
                "candidate_pair_count": int(values[5]),
                "scored_pair_count": int(values[6]),
            }
            for (dataset_id, capability_id, level), values in sorted(aggregates.items())
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def command_aggregate(args: argparse.Namespace) -> None:
    parts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.parts.rglob("*.json"))
    ]
    parts = [
        part
        for part in parts
        if part.get("schema_version") == "chronos2.cafe_effect_evaluation_part.v1"
    ]
    grouped: defaultdict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for part in parts:
        grouped[(part["corpus"], int(part["horizon"]), int(part["step"]))].append(part)
    rows: list[dict[str, Any]] = []
    for (corpus, horizon, step), group in sorted(grouped.items()):
        corpus_label = {"v13": "train_seed", "v14": "cross_seed"}.get(
            corpus, corpus
        )
        world_size = int(group[0]["world_size"])
        if {int(part["rank"]) for part in group} != set(range(world_size)):
            raise ValueError(f"Incomplete ranks for {(corpus, horizon, step)}")
        strata: defaultdict[tuple[str, str, int], list[float]] = defaultdict(
            lambda: [0.0] * 7
        )
        for part in group:
            for item in part["strata"]:
                key = (
                    item["dataset_id"],
                    item["capability_id"],
                    int(item["capability_level"]),
                )
                fields = (
                    "standardized_squared_error_sum",
                    "standardized_truth_squared_sum",
                    "standardized_forecast_squared_sum",
                    "observed_effect_cell_count",
                    "low_signal_pair_count",
                    "candidate_pair_count",
                    "scored_pair_count",
                )
                for index, field in enumerate(fields):
                    strata[key][index] += float(item[field])
        scored = {key: value for key, value in strata.items() if value[1] > 0.0}
        stratum_scores = [math.sqrt(value[0] / value[1]) for value in scored.values()]
        datasets_acc: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        capabilities_acc: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for (dataset_id, capability_id, _level), value in scored.items():
            datasets_acc[dataset_id][0] += value[0]
            datasets_acc[dataset_id][1] += value[1]
            capabilities_acc[capability_id][0] += value[0]
            capabilities_acc[capability_id][1] += value[1]
        total_error = sum(value[0] for value in scored.values())
        total_truth = sum(value[1] for value in scored.values())
        rows.append(
            {
                "corpus": corpus_label,
                "horizon": horizon,
                "step": step,
                "model": group[0]["model"],
                "pair_count": sum(int(part["pair_count"]) for part in group),
                "scored_pair_count": int(sum(value[6] for value in scored.values())),
                "low_signal_pair_count": int(sum(value[4] for value in strata.values())),
                "stratum_count": len(scored),
                "dataset_count": len(datasets_acc),
                "pooled_effect_nrmse": math.sqrt(total_error / total_truth),
                "macro_stratum_effect_nrmse": float(np.mean(stratum_scores)),
                "macro_dataset_effect_nrmse": float(
                    np.mean([math.sqrt(a / b) for a, b in datasets_acc.values()])
                ),
                "capability_effect_nrmse": {
                    capability: math.sqrt(a / b)
                    for capability, (a, b) in sorted(capabilities_acc.items())
                },
            }
        )
    args.output.write_text(
        json.dumps(
            {"schema_version": "chronos2.cafe_effect_curve.v1", "rows": rows},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_rows = [{**row, "capability_effect_nrmse": json.dumps(row["capability_effect_nrmse"], sort_keys=True)} for row in rows]
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    colors = {"train_seed": "#2563eb", "cross_seed": "#dc2626"}
    labels = {"train_seed": args.train_label, "cross_seed": args.cross_label}
    for axis, metric, title in zip(
        axes,
        ("macro_stratum_effect_nrmse", "pooled_effect_nrmse"),
        ("Macro-stratum effect NRMSE", "Pooled effect NRMSE"),
        strict=True,
    ):
        for corpus in ("train_seed", "cross_seed"):
            selected = sorted(
                (row for row in rows if row["corpus"] == corpus),
                key=lambda row: int(row["step"]),
            )
            axis.plot(
                [row["step"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                linewidth=2,
                color=colors[corpus],
                label=labels[corpus],
            )
        axis.axhline(1.0, color="#64748b", linewidth=1, alpha=0.7)
        axis.set_title(title)
        axis.set_xlabel("Training steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("NRMSE (lower is better)")
    axes[-1].legend(frameon=False)
    figure.savefig(args.output.with_suffix(".png"), dpi=180)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)
    print(f"aggregated {len(rows)} effect curve rows at {args.output}")


def _model_steps(models_root: Path) -> list[tuple[int, str]]:
    manifest_paths = (
        models_root / "cafe_effect_training_manifest.json",
        models_root / "cafe_training_manifest.json",
    )
    manifest_path = next((path for path in manifest_paths if path.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"No training manifest found in {models_root}")
    manifest = json.loads(manifest_path.read_text())
    return [
        (0, "amazon/chronos-2"),
        *[
            (int(step), str(models_root / f"checkpoint-{step}"))
            for step in manifest["checkpoint_steps"]
        ],
    ]


def command_run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    models_root = (args.models_root or root / "models/horizon-48").resolve()
    parts_root = (args.parts or root / "effect-parts").resolve()
    output = (args.output or root / "effect-curve.json").resolve()
    tasks: dict[int, list[tuple[str, int, str, Path]]] = {
        rank: [] for rank in range(args.world_size)
    }
    for step, model in _model_steps(models_root):
        for corpus, dataset in (
            ("v13", root / "effect-data/horizon-48"),
            ("v14", root / "effect-eval-data/horizon-48"),
        ):
            for rank in range(args.world_size):
                part_output = parts_root / f"rank-{rank}" / f"{corpus}-h48-step{step}.json"
                tasks[rank].append((corpus, step, model, part_output))

    def run_rank(rank: int) -> None:
        environment = {
            **dict(__import__("os").environ),
            "CUDA_VISIBLE_DEVICES": str(rank),
            "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        for corpus, step, model, output in tasks[rank]:
            if output.exists():
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            dataset = root / ("effect-data" if corpus == "v13" else "effect-eval-data") / "horizon-48"
            with output.with_suffix(".log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "evaluate",
                        "--dataset", str(dataset),
                        "--horizon", "48",
                        "--model", model,
                        "--model-label", f"chronos2-step-{step}",
                        "--step", str(step),
                        "--corpus", corpus,
                        "--rank", str(rank),
                        "--world-size", str(args.world_size),
                        "--output", str(output),
                    ],
                    check=True,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

    with ThreadPoolExecutor(max_workers=args.world_size) as executor:
        futures = {executor.submit(run_rank, rank): rank for rank in tasks}
        for future in as_completed(futures):
            future.result()
            print(f"rank {futures[future]} complete", flush=True)
    command_aggregate(
        argparse.Namespace(
            parts=parts_root,
            output=output,
            train_label=args.train_label,
            cross_label=args.cross_label,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    transfer._add_selection_arguments(prepare)
    prepare.add_argument("--horizon", type=int, choices=(30, 48, 60), required=True)
    prepare.add_argument("--writer-batch-size", type=int, default=128)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=command_prepare)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--horizon", type=int, required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--model-label", required=True)
    evaluate.add_argument("--step", type=int, required=True)
    evaluate.add_argument("--corpus", choices=("v13", "v14"), required=True)
    evaluate.add_argument("--rank", type=int, default=0)
    evaluate.add_argument("--world-size", type=int, default=1)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--input-batch-size", type=int, default=128)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=command_evaluate)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--parts", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--train-label", default="training seed")
    aggregate.add_argument("--cross-label", default="cross seed")
    aggregate.set_defaults(func=command_aggregate)

    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--models-root", type=Path)
    run.add_argument("--parts", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--world-size", type=int, default=4)
    run.add_argument("--train-label", default="seed2026082701 (train)")
    run.add_argument("--cross-label", default="seed2026082702 (cross-seed)")
    run.set_defaults(func=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
