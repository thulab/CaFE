#!/usr/bin/env python3
"""Fine-tune Chronos-2 on CaFE seed batch A and evaluate on A/B.

Input JSONL files are produced by ``experiments/finetuning/prepare.py``.
The experiment follows the CaFE fixed-context main-table task (L168, H48)
and evaluates the pretrained model plus regular checkpoints along one
continuous full-fine-tuning trajectory on A.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from chronos import Chronos2Pipeline


CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 48
EXPECTED_SCHEMA = "cafe.chronos_finetune_split.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-a", type=Path, required=True)
    parser.add_argument("--data-b", type=Path, required=True)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--training-seed", type=int, default=20260727)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 loads all rows; positive values are for smoke tests only",
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class DataBundle:
    name: str
    source: Path
    training_inputs: list[dict[str, Any]]
    prediction_inputs: list[dict[str, Any]]
    futures: list[np.ndarray]
    mase_scales: np.ndarray
    metadata: list[dict[str, Any]]
    audit: dict[str, Any]


def prepared_input(
    target: np.ndarray,
    covariates: np.ndarray | None,
    *,
    prediction: bool,
) -> dict[str, Any]:
    if prediction:
        target_context = target[:CONTEXT_LENGTH]
        context_covariates = (
            np.empty((CONTEXT_LENGTH, 0), dtype=np.float32)
            if covariates is None
            else covariates[:CONTEXT_LENGTH]
        )
        future_covariates = (
            np.empty((PREDICTION_LENGTH, 0), dtype=np.float32)
            if covariates is None
            else covariates[CONTEXT_LENGTH:]
        )
        context = np.concatenate(
            [target_context[:, None], context_covariates],
            axis=1,
        ).T
        future = np.concatenate(
            [
                np.full((PREDICTION_LENGTH, 1), np.nan, dtype=np.float32),
                future_covariates,
            ],
            axis=1,
        ).T
    else:
        all_covariates = (
            np.empty((len(target), 0), dtype=np.float32)
            if covariates is None
            else covariates
        )
        context = np.concatenate([target[:, None], all_covariates], axis=1).T
        future = np.full(
            (context.shape[0], PREDICTION_LENGTH),
            np.nan,
            dtype=np.float32,
        )
    covariate_dim = 0 if covariates is None else int(covariates.shape[1])
    return {
        "context": torch.from_numpy(np.ascontiguousarray(context)),
        "future_covariates": torch.from_numpy(np.ascontiguousarray(future)),
        "n_targets": 1,
        "n_covariates": covariate_dim,
        "n_future_covariates": covariate_dim,
    }


def load_jsonl(name: str, path: Path, max_rows: int) -> DataBundle:
    started = time.perf_counter()
    training_inputs: list[dict[str, Any]] = []
    prediction_inputs: list[dict[str, Any]] = []
    futures: list[np.ndarray] = []
    mase_scales: list[float] = []
    metadata: list[dict[str, Any]] = []
    master_rows = 0
    master_target_dims: Counter[int] = Counter()
    covariate_dims: Counter[int] = Counter()
    source_protocol_sha256: str | None = None
    source_experiment_id: str | None = None

    with path.open() as handle:
        for line_index, line in enumerate(handle):
            if max_rows and line_index >= max_rows:
                break
            row = json.loads(line)
            if row["schema_version"] != EXPECTED_SCHEMA:
                raise ValueError(
                    f"{path}:{line_index + 1}: unexpected schema "
                    f"{row['schema_version']!r}"
                )
            if row["split"] != name:
                raise ValueError(
                    f"{path}:{line_index + 1}: split={row['split']!r}, "
                    f"expected {name!r}"
                )
            if (
                int(row["context_length"]),
                int(row["horizon"]),
            ) != (CONTEXT_LENGTH, PREDICTION_LENGTH):
                raise ValueError(
                    f"{path}:{line_index + 1}: expected L168/H48"
                )

            target = np.asarray(row["target"], dtype=np.float32)
            target_dim = int(row["target_dim"])
            expected_shape = (
                CONTEXT_LENGTH + PREDICTION_LENGTH,
                target_dim,
            )
            if target.shape != expected_shape:
                raise ValueError(
                    f"{path}:{line_index + 1}: target shape {target.shape}, "
                    f"expected {expected_shape}"
                )
            covariate_dim = int(row["covariate_dim"])
            covariates = (
                None
                if row.get("covariates") is None
                else np.asarray(row["covariates"], dtype=np.float32)
            )
            if covariate_dim == 0 and covariates is not None:
                raise ValueError(f"{path}:{line_index + 1}: unexpected covariates")
            if covariate_dim:
                expected_covariate_shape = (
                    CONTEXT_LENGTH + PREDICTION_LENGTH,
                    covariate_dim,
                )
                if covariates is None or covariates.shape != expected_covariate_shape:
                    raise ValueError(
                        f"{path}:{line_index + 1}: covariate shape "
                        f"{None if covariates is None else covariates.shape}, "
                        f"expected {expected_covariate_shape}"
                    )
            scales = np.asarray(row["mase_scale_by_target"], dtype=np.float64)
            if scales.shape != (target_dim,) or not np.all(
                np.isfinite(scales) & (scales > 0)
            ):
                raise ValueError(f"{path}:{line_index + 1}: invalid MASE scales")
            if not np.isfinite(target).all() or (
                covariates is not None and not np.isfinite(covariates).all()
            ):
                raise ValueError(f"{path}:{line_index + 1}: non-finite values")

            observed_protocol = str(row["source_protocol_sha256"])
            observed_experiment = str(row["source_experiment_id"])
            source_protocol_sha256 = source_protocol_sha256 or observed_protocol
            source_experiment_id = source_experiment_id or observed_experiment
            if observed_protocol != source_protocol_sha256:
                raise ValueError("multiple source protocols in one split")
            if observed_experiment != source_experiment_id:
                raise ValueError("multiple source experiments in one split")

            master_rows += 1
            master_target_dims[target_dim] += 1
            covariate_dims[covariate_dim] += 1
            for target_index in range(target_dim):
                one_target = target[:, target_index]
                training_inputs.append(
                    prepared_input(one_target, covariates, prediction=False)
                )
                prediction_inputs.append(
                    prepared_input(one_target, covariates, prediction=True)
                )
                futures.append(
                    np.ascontiguousarray(
                        one_target[CONTEXT_LENGTH:],
                        dtype=np.float32,
                    )
                )
                mase_scales.append(float(scales[target_index]))
                metadata.append(
                    {
                        "sample_id": row["sample_id"],
                        "target_index": target_index,
                        "dataset_id": row["dataset_id"],
                        "capability_id": row["capability_id"],
                        "cell": (
                            f"{row['dataset_id']}::{row['capability_id']}"
                        ),
                        "intensity": int(row["intensity"]),
                        "seed_index": int(row["seed_index"]),
                        "counterfactual_member": row.get(
                            "counterfactual_member"
                        ),
                    }
                )

    if not training_inputs:
        raise ValueError(f"no records loaded from {path}")
    seed_indices = sorted({int(row["seed_index"]) for row in metadata})
    audit = {
        "dataset": name,
        "source": str(path.resolve()),
        "source_bytes": path.stat().st_size,
        "source_experiment_id": source_experiment_id,
        "source_protocol_sha256": source_protocol_sha256,
        "master_rows": master_rows,
        "target_tasks": len(training_inputs),
        "forecast_points": len(training_inputs) * PREDICTION_LENGTH,
        "master_target_dims": {
            str(key): value for key, value in sorted(master_target_dims.items())
        },
        "covariate_dims": {
            str(key): value for key, value in sorted(covariate_dims.items())
        },
        "dataset_count": len({row["dataset_id"] for row in metadata}),
        "capability_count": len({row["capability_id"] for row in metadata}),
        "cell_count": len({row["cell"] for row in metadata}),
        "intensity_counts": dict(
            sorted(Counter(row["intensity"] for row in metadata).items())
        ),
        "seed_indices": seed_indices,
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "input_adaptation": (
            "each target is an independent univariate Chronos task; known-future "
            "covariates remain grouped with that target"
        ),
        "load_seconds": time.perf_counter() - started,
    }
    print(
        f"Loaded {name}: {master_rows:,} masters, "
        f"{len(training_inputs):,} target tasks, "
        f"{audit['forecast_points']:,} forecast points",
        flush=True,
    )
    return DataBundle(
        name=name,
        source=path,
        training_inputs=training_inputs,
        prediction_inputs=prediction_inputs,
        futures=futures,
        mase_scales=np.asarray(mase_scales, dtype=np.float64),
        metadata=metadata,
        audit=audit,
    )


def aggregate_evaluation(
    model: torch.nn.Module,
    bundle: DataBundle,
    inference_batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    pipeline = Chronos2Pipeline(model=model)
    with torch.inference_mode():
        outputs = pipeline.predict(
            bundle.prediction_inputs,
            prediction_length=PREDICTION_LENGTH,
            context_length=CONTEXT_LENGTH,
            batch_size=inference_batch_size,
            cross_learning=False,
        )
    quantiles = np.asarray(pipeline.quantiles, dtype=np.float64)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    smape_component_sum = 0.0
    mase_error_sum = 0.0
    target_abs_sum = 0.0
    pinball_sums = np.zeros(len(quantiles), dtype=np.float64)

    for index, (future, output) in enumerate(
        zip(bundle.futures, outputs, strict=True)
    ):
        prediction = output.detach().float().cpu().numpy()[0]
        median = prediction[median_index]
        residual = future - median
        absolute_residual = np.abs(residual)
        absolute_error_sum += float(absolute_residual.sum(dtype=np.float64))
        squared_error_sum += float(
            np.square(residual).sum(dtype=np.float64)
        )
        smape_component_sum += float(
            (
                absolute_residual
                / np.maximum(np.abs(future) + np.abs(median), 1e-8)
            ).sum(dtype=np.float64)
        )
        mase_error_sum += float(
            (absolute_residual / bundle.mase_scales[index]).sum(
                dtype=np.float64
            )
        )
        target_abs_sum += float(np.abs(future).sum(dtype=np.float64))
        quantile_errors = future[None, :] - prediction
        q = quantiles[:, None]
        pinball_sums += np.maximum(
            q * quantile_errors,
            (q - 1.0) * quantile_errors,
        ).sum(axis=1, dtype=np.float64)

    point_count = len(bundle.futures) * PREDICTION_LENGTH
    normalized_wql_per_quantile = (
        2.0 * pinball_sums / max(target_abs_sum, 1e-8)
    )
    metrics = {
        "master_sample_count": int(bundle.audit["master_rows"]),
        "target_task_count": len(bundle.futures),
        "point_count": point_count,
        "mae": absolute_error_sum / point_count,
        "rmse": float(np.sqrt(squared_error_sum / point_count)),
        "smape_percent": 200.0 * smape_component_sum / point_count,
        "mase": mase_error_sum / point_count,
        "normalized_wql": float(np.mean(normalized_wql_per_quantile)),
        "evaluation_seconds": time.perf_counter() - started,
    }
    del outputs, pipeline
    gc.collect()
    torch.cuda.empty_cache()
    if was_training:
        model.train()
    return metrics


def write_curve(rows: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(rows).sort_values(["step", "dataset"])
    frame.to_csv(output_dir / "step_metrics.csv", index=False)
    wide = frame.pivot(index="step", columns="dataset", values="mae").sort_index()
    steps = wide.index.to_numpy()
    base = wide.loc[0]
    relative_improvement = 100.0 * (base - wide) / base

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"A": "#1677ff", "B": "#f97316"}
    fig, (error_ax, improvement_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.45]},
    )
    for dataset in ["A", "B"]:
        error_ax.plot(
            steps,
            wide[dataset],
            marker="o",
            markersize=5,
            linewidth=2.3,
            color=colors[dataset],
            label=f"Evaluate on {dataset}",
        )
        improvement_ax.plot(
            steps,
            relative_improvement[dataset],
            marker="o",
            markersize=4.5,
            linewidth=2.1,
            color=colors[dataset],
            label=dataset,
        )
    error_ax.set_ylabel("MAE (lower is better)")
    error_ax.set_title(
        "Chronos-2 fine-tuned on seed batch A, evaluated on disjoint A/B\n"
        "CaFE primary/main samples · fixed context L168 · horizon H48"
    )
    error_ax.legend(frameon=True)
    improvement_ax.axhline(0.0, color="#667085", linewidth=1.0)
    improvement_ax.set_ylabel("MAE improvement\nfrom pretrained (%)")
    improvement_ax.set_xlabel("Full fine-tuning steps on A")
    improvement_ax.set_xticks(steps)
    improvement_ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value):,}")
    )
    improvement_ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "mae_transfer_curve.png", dpi=220)
    fig.savefig(output_dir / "mae_transfer_curve.svg")
    plt.close(fig)

    metric_names = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("mase", "MASE"),
        ("normalized_wql", "normalized WQL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True)
    for axis, (metric, label) in zip(axes.flat, metric_names, strict=True):
        values = frame.pivot(
            index="step",
            columns="dataset",
            values=metric,
        ).sort_index()
        normalized = 100.0 * values / values.loc[0]
        for dataset in ["A", "B"]:
            axis.plot(
                normalized.index,
                normalized[dataset],
                marker="o",
                markersize=4,
                linewidth=2,
                color=colors[dataset],
                label=f"Evaluate on {dataset}",
            )
        axis.axhline(100.0, color="#98a2b3", linewidth=0.9)
        axis.set_title(label)
        axis.set_ylabel("% of pretrained error")
    for axis in axes[-1]:
        axis.set_xlabel("Full fine-tuning steps on A")
        axis.tick_params(axis="x", rotation=45)
    axes[0, 0].legend()
    fig.suptitle(
        "Chronos-2 A→A versus A→B transfer curves (lower is better)",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "all_metric_transfer_curves.png", dpi=200)
    fig.savefig(output_dir / "all_metric_transfer_curves.svg")
    plt.close(fig)


def write_status(
    output_dir: Path,
    *,
    status: str,
    last_evaluated_step: int | None,
    detail: str,
) -> None:
    payload = {
        "status": status,
        "last_evaluated_step": last_evaluated_step,
        "detail": detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    temporary = output_dir / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output_dir / "status.json")


class StepCurveCallback(TrainerCallback):
    def __init__(
        self,
        bundle_a: DataBundle,
        bundle_b: DataBundle,
        output_dir: Path,
        inference_batch_size: int,
        eval_interval: int,
        early_stop_patience: int,
        early_stop_min_delta: float,
        initial_rows: list[dict[str, Any]],
    ) -> None:
        self.bundle_a = bundle_a
        self.bundle_b = bundle_b
        self.output_dir = output_dir
        self.inference_batch_size = inference_batch_size
        self.eval_interval = eval_interval
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.rows = initial_rows
        initial_a = next(row for row in initial_rows if row["dataset"] == "A")
        self.best_a = float(initial_a["mae"])
        self.best_step = 0
        self.stale_checks = 0
        self.stop_reason = "max_steps"

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        step = int(state.global_step)
        if step == 0 or step % self.eval_interval:
            return control
        if model is None:
            raise RuntimeError("Trainer callback did not provide the model")

        print(f"\nCurve evaluation at step {step:,}", flush=True)
        for bundle in (self.bundle_a, self.bundle_b):
            metrics = aggregate_evaluation(
                model,
                bundle,
                self.inference_batch_size,
            )
            self.rows.append({"step": step, "dataset": bundle.name, **metrics})
            print(
                f"  {bundle.name}: MAE={metrics['mae']:.6f}, "
                f"MASE={metrics['mase']:.6f}, "
                f"WQL={metrics['normalized_wql']:.6f}",
                flush=True,
            )
        write_curve(self.rows, self.output_dir)
        write_status(
            self.output_dir,
            status="running",
            last_evaluated_step=step,
            detail="checkpoint evaluation complete",
        )

        current_a = float(
            next(
                row["mae"]
                for row in reversed(self.rows)
                if row["step"] == step and row["dataset"] == "A"
            )
        )
        if self.best_a - current_a >= self.early_stop_min_delta:
            self.best_a = current_a
            self.best_step = step
            self.stale_checks = 0
        else:
            self.stale_checks += 1
        if (
            self.early_stop_patience > 0
            and self.stale_checks >= self.early_stop_patience
        ):
            self.stop_reason = (
                f"A MAE failed to improve by {self.early_stop_min_delta:g} "
                f"for {self.early_stop_patience} evaluations"
            )
            control.should_training_stop = True
            print(f"Early stopping: {self.stop_reason}", flush=True)
        return control


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_steps <= 0 or args.eval_interval <= 0:
        raise ValueError("max steps and evaluation interval must be positive")
    if args.max_steps % args.eval_interval:
        raise ValueError("max steps must be divisible by evaluation interval")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty {args.output_dir}"
        )

    args.output_dir.mkdir(parents=True)
    shutil.copy2(__file__, args.output_dir / "experiment_script.py")
    write_status(
        args.output_dir,
        status="initializing",
        last_evaluated_step=None,
        detail="loading and auditing A/B data",
    )
    set_all_seeds(args.training_seed)
    torch.set_float32_matmul_precision("high")
    bundle_a = load_jsonl("A", args.data_a, args.max_rows)
    bundle_b = load_jsonl("B", args.data_b, args.max_rows)
    if bundle_a.audit["source_protocol_sha256"] != bundle_b.audit[
        "source_protocol_sha256"
    ]:
        raise ValueError("A/B source protocol hashes differ")
    if bundle_a.audit["master_rows"] != bundle_b.audit["master_rows"]:
        raise ValueError("A/B master row counts differ")
    if bundle_a.audit["target_tasks"] != bundle_b.audit["target_tasks"]:
        raise ValueError("A/B target task counts differ")
    if set(bundle_a.audit["seed_indices"]) & set(bundle_b.audit["seed_indices"]):
        raise ValueError("A/B seed sets overlap")
    if not args.max_rows and (
        len(bundle_a.audit["seed_indices"]) != 32
        or len(bundle_b.audit["seed_indices"]) != 32
    ):
        raise ValueError("full experiment requires 32 seeds in each split")

    config = {
        "protocol": {
            "source_experiment_id": bundle_a.audit["source_experiment_id"],
            "source_protocol_sha256": bundle_a.audit[
                "source_protocol_sha256"
            ],
            "selection": (
                "evaluation_table=main and generator_family_role=primary"
            ),
            "seed_partition": {
                "A": bundle_a.audit["seed_indices"],
                "B": bundle_b.audit["seed_indices"],
            },
            "context": "master_target[168:336]",
            "labels": "master_target[336:384]",
            "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "standardization": (
                "slice exact L336-standardized master without re-standardization"
            ),
            "target_adaptation": (
                "multi-target masters are split into independent univariate "
                "Chronos tasks"
            ),
            "covariate_adaptation": (
                "history and known-future covariates remain grouped with each "
                "target"
            ),
            "training_semantics": (
                "A labels participate in supervised fine-tuning; A evaluation "
                "is an in-sample memorization probe and B is disjoint-seed "
                "transfer"
            ),
        },
        "training": {
            "model_id": args.model_id,
            "finetune_mode": "full",
            "training_seed": args.training_seed,
            "max_steps": args.max_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "adamw_torch_fused",
            "lr_scheduler": "linear over max_steps",
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "precision": "BF16 with TF32",
            "inference_batch_size": args.inference_batch_size,
            "cross_learning_at_inference": False,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
        },
        "data_a": bundle_a.audit,
        "data_b": bundle_b.audit,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "visible_gpu_count": torch.cuda.device_count(),
            "chronos_repo_commit": git_commit(),
        },
        "command_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    print(f"Loading pretrained model: {args.model_id}", flush=True)
    write_status(
        args.output_dir,
        status="initializing",
        last_evaluated_step=None,
        detail="loading pretrained Chronos-2",
    )
    base = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cuda")
    rows: list[dict[str, Any]] = []
    print("Curve evaluation at step 0", flush=True)
    for bundle in (bundle_a, bundle_b):
        metrics = aggregate_evaluation(
            base.model,
            bundle,
            args.inference_batch_size,
        )
        rows.append({"step": 0, "dataset": bundle.name, **metrics})
        print(
            f"  {bundle.name}: MAE={metrics['mae']:.6f}, "
            f"MASE={metrics['mase']:.6f}, "
            f"WQL={metrics['normalized_wql']:.6f}",
            flush=True,
        )
    write_curve(rows, args.output_dir)
    write_status(
        args.output_dir,
        status="running",
        last_evaluated_step=0,
        detail="pretrained baseline evaluation complete; fine-tuning A",
    )

    callback = StepCurveCallback(
        bundle_a=bundle_a,
        bundle_b=bundle_b,
        output_dir=args.output_dir,
        inference_batch_size=args.inference_batch_size,
        eval_interval=args.eval_interval,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        initial_rows=rows,
    )
    started = time.perf_counter()
    finetuned = base.fit(
        inputs=bundle_a.training_inputs,
        prediction_length=PREDICTION_LENGTH,
        finetune_mode="full",
        context_length=CONTEXT_LENGTH,
        min_past=CONTEXT_LENGTH,
        learning_rate=args.learning_rate,
        num_steps=args.max_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir / "training",
        finetuned_ckpt_name="finetuned-ckpt",
        callbacks=[callback],
        logging_steps=args.eval_interval,
        report_to="none",
        seed=args.training_seed,
        data_seed=args.training_seed,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    elapsed = time.perf_counter() - started
    final_step = max(int(row["step"]) for row in callback.rows)
    final_a = next(
        row
        for row in callback.rows
        if row["step"] == final_step and row["dataset"] == "A"
    )
    final_b = next(
        row
        for row in callback.rows
        if row["step"] == final_step and row["dataset"] == "B"
    )
    base_a = next(
        row
        for row in callback.rows
        if row["step"] == 0 and row["dataset"] == "A"
    )
    base_b = next(
        row
        for row in callback.rows
        if row["step"] == 0 and row["dataset"] == "B"
    )
    result = {
        "status": "complete",
        "final_evaluated_step": final_step,
        "stop_reason": callback.stop_reason,
        "best_A_mae": callback.best_a,
        "best_A_step": callback.best_step,
        "wall_seconds_including_evaluations": elapsed,
        "final_mae": {"A": final_a["mae"], "B": final_b["mae"]},
        "mae_relative_improvement_percent": {
            "A": 100.0 * (base_a["mae"] - final_a["mae"]) / base_a["mae"],
            "B": 100.0 * (base_b["mae"] - final_b["mae"]) / base_b["mae"],
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_curve(callback.rows, args.output_dir)
    write_status(
        args.output_dir,
        status="complete",
        last_evaluated_step=final_step,
        detail="fine-tuning and all checkpoint evaluations complete",
    )
    del finetuned
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(result, indent=2), flush=True)
    print(f"Artifacts: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
