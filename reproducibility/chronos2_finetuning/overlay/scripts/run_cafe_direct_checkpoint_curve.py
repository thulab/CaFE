#!/usr/bin/env python3
"""Evaluate Chronos-2 checkpoints directly on GPUs and keep only metric parts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--cross-dataset", type=Path, required=True)
    parser.add_argument("--train-evaluation-data", type=Path, required=True)
    parser.add_argument("--cross-evaluation-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", default=["1", "2", "3"])
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=("train", "cross"),
        default=["train", "cross"],
    )
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--wait-seconds", type=float, default=15.0)
    return parser.parse_args()


def _complete(path: Path, *, step: int, rank: int, world_size: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("complete")
        and int(payload.get("step", -1)) == step
        and int(payload.get("rank", -1)) == rank
        and int(payload.get("world_size", -1)) == world_size
        and payload.get("dtype") == "float32"
    )


def _wait_for_checkpoint(root: Path, step: int, wait_seconds: float) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    marker = checkpoint / "adapter_model.safetensors"
    announced = False
    while not marker.is_file():
        if not announced:
            print(f"Waiting for checkpoint-{step}", flush=True)
            announced = True
        time.sleep(wait_seconds)
    return checkpoint


def _merge_checkpoint(source: Path, output: Path) -> None:
    import torch
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        str(source), device_map="cpu", dtype=torch.float32
    )
    pipeline.save_pretrained(output)
    del pipeline


def _evaluate_corpus(
    *,
    model: Path,
    dataset: Path,
    evaluation_data: Path,
    corpus: str,
    step: int,
    output_root: Path,
    gpus: list[str],
    context_length: int,
    batch_size: int,
) -> None:
    world_size = len(gpus)
    destination = output_root / f"step-{step}" / corpus
    destination.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    for rank, gpu in enumerate(gpus):
        output = destination / f"rank-{rank}.json"
        if _complete(output, step=step, rank=rank, world_size=world_size):
            continue
        command = [
            ".venv/bin/python",
            "scripts/cafe_direct_checkpoint_evaluation.py",
            "--model", str(model),
            "--treatment-dataset", str(dataset),
            "--evaluation-data", str(evaluation_data),
            "--corpus", corpus,
            "--step", str(step),
            "--rank", str(rank),
            "--world-size", str(world_size),
            "--device", "cuda",
            "--dtype", "float32",
            "--context-length", str(context_length),
            "--prediction-length", "48",
            "--batch-size", str(batch_size),
            "--output", str(output),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        processes.append((rank, subprocess.Popen(command, env=environment)))
    failures = []
    for rank, process in processes:
        returncode = process.wait()
        if returncode:
            failures.append((rank, returncode))
    if failures:
        raise RuntimeError(f"{corpus} step {step} evaluator failures: {failures}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        checkpoint = _wait_for_checkpoint(
            args.checkpoint_root, step, args.wait_seconds
        )
        with tempfile.TemporaryDirectory(prefix=f"chronos2-step-{step}-") as tmp:
            merged = Path(tmp) / "model"
            print(f"Merging checkpoint-{step}", flush=True)
            _merge_checkpoint(checkpoint, merged)
            corpora = {
                ("train", args.train_dataset, args.train_evaluation_data),
                ("cross", args.cross_dataset, args.cross_evaluation_data),
            }
            for corpus, dataset, evaluation_data in sorted(corpora):
                if corpus not in args.corpora:
                    continue
                print(f"Evaluating step {step} on {corpus}", flush=True)
                _evaluate_corpus(
                    model=merged,
                    dataset=dataset,
                    evaluation_data=evaluation_data,
                    corpus=corpus,
                    step=step,
                    output_root=args.output_root,
                    gpus=args.gpus,
                    context_length=args.context_length,
                    batch_size=args.batch_size,
                )
        print(f"Completed step {step}", flush=True)


if __name__ == "__main__":
    main()
