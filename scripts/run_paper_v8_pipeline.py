#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import paper_v8_pipeline_common as v8


DEFAULT_OUTPUT_ROOT = (
    v8.REPO_ROOT / "runtime" / "paper_exp" / "v8_test" / "full_pipeline"
)
STEPS = ("calibration", "generation", "validation", "inference", "analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete formal Paper v8 pipeline."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gift-eval-dir", type=Path, default=Path("/root/xmy/gift-eval"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument("--calibration-seeds", type=int, default=12)
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


def main() -> int:
    args = parse_args()
    start = STEPS.index(args.start_at)
    stop = STEPS.index(args.stop_after)
    if stop < start:
        raise ValueError("stop-after must not precede start-at")
    common = [
        "--dataset-id",
        args.dataset_id,
        "--output-root",
        str(args.output_root.resolve()),
    ]
    seed = [
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
    ]
    commands = {
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
            ],
        ),
        "generation": (
            "generate_paper_v8_samples.py",
            [*common, *seed],
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
    for step in STEPS[start : stop + 1]:
        script, arguments = commands[step]
        run(script, arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
