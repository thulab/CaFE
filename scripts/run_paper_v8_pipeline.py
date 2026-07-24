#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import paper_v8_pipeline_common as v8


DEFAULT_OUTPUT_ROOT = (
    v8.REPO_ROOT / "runtime" / "paper_exp" / "v8_test" / "full_pipeline"
)
STEPS = ("calibration", "generation", "validation", "inference", "analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete formal Paper v8 pipeline."
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help=(
            "One registered dataset id. Repeat the flag to run several "
            "datasets. Defaults to gift_electricity_h."
        ),
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=None,
        help="Convenience form for passing several registered dataset ids.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gift-eval-dir", type=Path, default=Path("/root/xmy/gift-eval"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument("--calibration-seeds", type=int, default=12)
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=v8.CAPABILITIES,
        default=list(v8.CAPABILITIES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Chronos-2", "toto2.0", "tirex2", "timesfm2.5"],
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=[
            "http://127.0.0.1:10810",
            "http://192.168.99.17:10811",
            "http://192.168.99.18:10810",
        ],
    )
    parser.add_argument("--start-at", choices=STEPS, default="calibration")
    parser.add_argument("--stop-after", choices=STEPS, default="analysis")
    parser.add_argument("--resume-inference", action="store_true")
    return parser.parse_args()


def run(script: str, arguments: list[str]) -> None:
    command = [sys.executable, str(v8.REPO_ROOT / "scripts" / script), *arguments]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=v8.REPO_ROOT / "backend", check=True)


def requested_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.dataset_id and args.dataset_ids:
        raise ValueError("use either --dataset-id or --dataset-ids, not both")
    values = list(
        args.dataset_ids
        or args.dataset_id
        or ["gift_electricity_h"]
    )
    if len(values) != len(set(values)):
        raise ValueError("v8 dataset ids must be unique")
    for dataset_id in values:
        v8.resolve_dataset(dataset_id)
    return values


def commands_for_dataset(
    args: argparse.Namespace,
    dataset_id: str,
) -> dict[str, tuple[str, list[str]]]:
    common = [
        "--dataset-id",
        dataset_id,
        "--output-root",
        str(args.output_root.resolve()),
    ]
    seed = [
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
    ]
    return {
        "calibration": (
            "calibrate_paper_v8.py",
            [
                *common,
                "--gift-eval-dir",
                str(args.gift_eval_dir.resolve()),
                "--max-anchors",
                str(args.max_anchors),
                "--calibration-seeds",
                str(args.calibration_seeds),
                "--capabilities",
                *args.capabilities,
            ],
        ),
        "generation": (
            "generate_paper_v8_samples.py",
            [*common, *seed, "--capabilities", *args.capabilities],
        ),
        "validation": (
            "validate_paper_v8_samples.py",
            [*common, *seed],
        ),
        "inference": (
            "run_paper_v8_inference.py",
            [
                *common,
                *seed,
                "--models",
                *args.models,
                "--endpoints",
                *args.endpoints,
                *(["--resume"] if args.resume_inference else []),
            ],
        ),
        "analysis": (
            "analyze_paper_v8.py",
            [*common, *seed, "--models", *args.models],
        ),
    }


def main() -> int:
    args = parse_args()
    dataset_ids = requested_dataset_ids(args)
    start = STEPS.index(args.start_at)
    stop = STEPS.index(args.stop_after)
    if stop < start:
        raise ValueError("stop-after must not precede start-at")
    completed: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        commands = commands_for_dataset(args, dataset_id)
        for step in STEPS[start : stop + 1]:
            script, arguments = commands[step]
            run(script, arguments)
        completed.append(
            {
                "dataset_id": dataset_id,
                "steps": list(STEPS[start : stop + 1]),
                "output_dir": str(
                    args.output_root.resolve() / dataset_id
                ),
            }
        )
    manifest = {
        "schema_version": "paper_v8_multi_dataset_run.v1",
        "created_at": v8.utc_now(),
        "dataset_ids": dataset_ids,
        "dataset_count": len(dataset_ids),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "capabilities": list(args.capabilities),
        "models": list(args.models),
        "start_at": args.start_at,
        "stop_after": args.stop_after,
        "aggregation_policy": (
            "dataset-isolated outputs and reports; no implicit "
            "cross-dataset averaging"
        ),
        "completed": completed,
    }
    manifest_name = (
        f"pipeline_run__seed_{args.seed_start:06d}_"
        f"{args.seed_start + args.seed_count:06d}.json"
    )
    v8.write_json(args.output_root.resolve() / manifest_name, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
