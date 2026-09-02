#!/usr/bin/env python3
"""Prepare data, fine-tune Chronos-2, and reconstruct the paper curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Runner:
    def __init__(self, *, chronos_root: Path, dry_run: bool) -> None:
        self.chronos_root = chronos_root
        self.python = chronos_root / ".venv" / "bin" / "python"
        self.dry_run = dry_run

    def run(
        self,
        command: list[str | Path],
        *,
        environment: dict[str, str] | None = None,
    ) -> None:
        rendered = [str(value) for value in command]
        prefix = ""
        if environment:
            prefix = " ".join(f"{key}={value}" for key, value in environment.items()) + " "
        print(prefix + " ".join(rendered), flush=True)
        if self.dry_run:
            return
        merged_environment = os.environ.copy()
        if environment:
            merged_environment.update(environment)
        subprocess.run(
            rendered,
            cwd=self.chronos_root,
            env=merged_environment,
            check=True,
        )


def resolve_model(*, dry_run: bool) -> Path:
    model = CONFIG["base_model"]
    if dry_run:
        return Path("${PINNED_CHRONOS2_SNAPSHOT}")
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=model["repository"],
            revision=model["revision"],
            allow_patterns=["config.json", "model.safetensors"],
        )
    )
    expected = {
        "config.json": model["config_sha256"],
        "model.safetensors": model["weights_sha256"],
    }
    for name, expected_hash in expected.items():
        observed = sha256(snapshot / name)
        if observed != expected_hash:
            raise RuntimeError(f"base-model hash mismatch for {name}: {observed}")
    return snapshot


def experiment_root(cafe_root: Path, corpus: str) -> Path:
    experiment_id = CONFIG["data"][corpus]["experiment_id"]
    return cafe_root / "runtime" / "experiments" / experiment_id


def selection_arguments(cafe_root: Path, corpus: str) -> list[str | Path]:
    selection = CONFIG["data"][corpus]
    return [
        "--cafe-root", cafe_root,
        "--experiment-root", experiment_root(cafe_root, corpus),
        "--gift-eval-dir", cafe_root / "data" / "gift-eval",
        "--official-fold-count", str(selection["fold_count"]),
        "--heldout-fold", str(selection["fold_index"]),
        "--fold-role", str(selection["fold_role"]),
        "--fold-salt", str(selection["fold_salt"]),
        "--maximum-context", str(CONFIG["data"]["materialized_maximum_context"]),
    ]


def prepare(args: argparse.Namespace, runner: Runner) -> None:
    data_root = args.work_root / "data"
    for corpus in ("fit", "evaluation"):
        treatment_output = data_root / corpus / "treatments"
        if not (treatment_output / "cafe_adapter_manifest.json").is_file():
            runner.run(
                [
                    runner.python,
                    "scripts/cafe_seed_transfer.py",
                    "prepare",
                    *selection_arguments(args.cafe_root, corpus),
                    "--horizon", str(CONFIG["data"]["horizon"]),
                    "--output", treatment_output,
                ]
            )
        evaluation_output = data_root / corpus / "direct-evaluation"
        if not (evaluation_output / "manifest.json").is_file():
            selection = CONFIG["data"][corpus]
            runner.run(
                [
                    runner.python,
                    "scripts/prepare_cafe_direct_evaluation.py",
                    "--cafe-root", args.cafe_root,
                    "--experiment-root", experiment_root(args.cafe_root, corpus),
                    "--gift-eval-dir", args.cafe_root / "data" / "gift-eval",
                    "--datasets", *CONFIG["data"]["datasets"],
                    "--horizon", str(CONFIG["data"]["horizon"]),
                    "--fold-count", str(selection["fold_count"]),
                    "--fold-index", str(selection["fold_index"]),
                    "--fold-salt", str(selection["fold_salt"]),
                    "--maximum-context", str(CONFIG["data"]["materialized_maximum_context"]),
                    "--output", evaluation_output,
                ]
            )
    effect_output = data_root / "fit" / "effect-pairs"
    if not (effect_output / "cafe_effect_manifest.json").is_file():
        runner.run(
            [
                runner.python,
                "scripts/cafe_effect_seed_transfer.py",
                "prepare",
                *selection_arguments(args.cafe_root, "fit"),
                "--horizon", str(CONFIG["data"]["horizon"]),
                "--output", effect_output,
            ]
        )


def train(args: argparse.Namespace, runner: Runner) -> None:
    model = resolve_model(dry_run=args.dry_run)
    shared = CONFIG["shared_training"]
    default = CONFIG["objectives"]["default"]
    runner.run(
        [
            runner.python,
            "scripts/cafe_seed_transfer.py",
            "finetune",
            "--dataset", args.work_root / "data" / "fit" / "treatments",
            "--model", model,
            "--horizon", str(CONFIG["data"]["horizon"]),
            "--output", args.work_root / "models" / "default",
            "--device", args.device,
            "--finetune-mode", str(shared["finetune_mode"]),
            "--context-length", str(shared["context_length"]),
            "--min-past", str(default["min_past"]),
            "--learning-rate", str(default["learning_rate"]),
            "--num-steps", str(shared["steps"]),
            "--training-sampling", str(shared["sampling"]),
            "--checkpoint-interval", str(shared["checkpoint_interval"]),
            "--batch-size", str(shared["batch_size_series_budget"]),
            "--seed", str(shared["seed"]),
        ],
        environment={"CUDA_VISIBLE_DEVICES": args.device_index},
    )
    effect = CONFIG["objectives"]["effect_nrmse"]
    runner.run(
        [
            runner.python,
            "scripts/cafe_effect_finetune.py",
            "--dataset", args.work_root / "data" / "fit" / "effect-pairs",
            "--model", model,
            "--horizon", str(CONFIG["data"]["horizon"]),
            "--context-length", str(shared["context_length"]),
            "--batch-size", str(shared["batch_size_series_budget"]),
            "--learning-rate", str(effect["learning_rate"]),
            "--num-steps", str(shared["steps"]),
            "--checkpoint-interval", str(shared["checkpoint_interval"]),
            "--training-sampling", str(shared["sampling"]),
            "--finetune-mode", str(shared["finetune_mode"]),
            "--seed", str(shared["seed"]),
            "--device", args.device,
            "--output", args.work_root / "models" / "effect-nrmse",
        ],
        environment={"CUDA_VISIBLE_DEVICES": args.device_index},
    )


def evaluate_base(
    args: argparse.Namespace,
    runner: Runner,
    *,
    model: Path,
    output_root: Path,
) -> None:
    for corpus in ("fit", "evaluation"):
        label = "train" if corpus == "fit" else "cross"
        world_size = len(args.gpus)
        for rank, gpu in enumerate(args.gpus):
            output = output_root / "step-0" / label / f"rank-{rank}.json"
            runner.run(
                [
                    runner.python,
                    "scripts/cafe_direct_checkpoint_evaluation.py",
                    "--model", model,
                    "--treatment-dataset", args.work_root / "data" / corpus / "treatments",
                    "--evaluation-data", args.work_root / "data" / corpus / "direct-evaluation",
                    "--corpus", label,
                    "--step", "0",
                    "--rank", str(rank),
                    "--world-size", str(world_size),
                    "--device", "cuda",
                    "--dtype", "float32",
                    "--context-length", str(CONFIG["evaluation"]["context_length"]),
                    "--prediction-length", str(CONFIG["data"]["horizon"]),
                    "--output", output,
                ],
                environment={"CUDA_VISIBLE_DEVICES": gpu},
            )


def evaluate(args: argparse.Namespace, runner: Runner) -> None:
    model = resolve_model(dry_run=args.dry_run)
    nonzero_steps = [str(value) for value in CONFIG["evaluation"]["checkpoint_steps"] if value]
    objective_labels = {
        "default": "default_loss_random_with_replacement",
        "effect-nrmse": "paired_effect_squared_nrmse_random_with_replacement",
    }
    for objective, label in objective_labels.items():
        parts = args.work_root / "results" / objective / "metric-parts"
        evaluate_base(args, runner, model=model, output_root=parts)
        runner.run(
            [
                runner.python,
                "scripts/run_cafe_direct_checkpoint_curve.py",
                "--checkpoint-root", args.work_root / "models" / objective,
                "--train-dataset", args.work_root / "data" / "fit" / "treatments",
                "--cross-dataset", args.work_root / "data" / "evaluation" / "treatments",
                "--train-evaluation-data", args.work_root / "data" / "fit" / "direct-evaluation",
                "--cross-evaluation-data", args.work_root / "data" / "evaluation" / "direct-evaluation",
                "--output-root", parts,
                "--steps", *nonzero_steps,
                "--gpus", *args.gpus,
                "--context-length", str(CONFIG["evaluation"]["context_length"]),
            ]
        )
        runner.run(
            [
                runner.python,
                "scripts/aggregate_cafe_direct_curve.py",
                parts,
                "--output", args.work_root / "results" / objective / "curve.json",
                "--objective", label,
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "all"))
    parser.add_argument("--cafe-root", type=Path, required=True)
    parser.add_argument("--chronos-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.cafe_root = args.cafe_root.resolve()
    args.chronos_root = args.chronos_root.resolve()
    args.work_root = args.work_root.resolve()
    runner = Runner(chronos_root=args.chronos_root, dry_run=args.dry_run)
    if not args.dry_run and not runner.python.is_file():
        raise FileNotFoundError(f"missing Chronos environment: {runner.python}")
    if args.command in {"prepare", "all"}:
        prepare(args, runner)
    if args.command in {"train", "all"}:
        train(args, runner)
    if args.command in {"evaluate", "all"}:
        evaluate(args, runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
