#!/usr/bin/env python3
"""Reviewer-facing entry point for CaFE paper reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "configs" / "paper_experiments.json").read_text(encoding="utf-8"))
RESULTS_MANIFEST = HERE / "expected_results.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> int:
    manifest = json.loads(RESULTS_MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            failures.append(f"missing: {record['path']}")
            continue
        observed_hash = sha256(path)
        if observed_hash != record["sha256"]:
            failures.append(
                f"hash mismatch: {record['path']} ({observed_hash})"
            )
        if "csv_rows" in record:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = sum(1 for _ in csv.DictReader(handle))
            if rows != int(record["csv_rows"]):
                failures.append(
                    f"row-count mismatch: {record['path']} ({rows})"
                )
    if failures:
        print("verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"verified {len(manifest['files'])} frozen paper artifacts")
    return 0


def common_arguments(experiment: dict[str, Any]) -> list[str]:
    return [
        "--experiment-id", str(experiment["experiment_id"]),
        "--augmentation-seed", str(experiment["augmentation_seed"]),
        "--capabilities", *CONFIG["capabilities"],
        "--models", *CONFIG["public_models"],
        "--backend", "native",
        "--model-root", "${CAFE_MODEL_ROOT}",
        "--model-code-root", "${CAFE_MODEL_CODE_ROOT}",
        "--devices", "${CUDA_DEVICES}",
        "--output-root", "${CAFE_OUTPUT_ROOT}",
    ]


def command_for(name: str) -> list[str]:
    experiment = CONFIG["main_experiments"][name]
    if experiment["runner"] == "gift":
        return [
            "uv", "run", "cafe", "run",
            *common_arguments(experiment),
            "--term", str(experiment["term"]),
            "--gift-eval-dir", "${GIFT_EVAL_DIR}",
        ]
    return [
        "uv", "run", "python", "-m", "cafe.fev_pipeline",
        *common_arguments(experiment),
        "--source-root", "${FEV_DATA_ROOT}",
    ]


def format_command(command: list[str]) -> str:
    return (" \\" + "\n  ").join(command)


def print_commands(name: str) -> int:
    names = (
        list(CONFIG["main_experiments"])
        if name == "all"
        else ([] if name == "stability" else [name])
    )
    for current in names:
        print(f"# {current}")
        print(format_command(command_for(current)))
    if name in {"all", "stability"}:
        stability = CONFIG["stability"]
        for seed in stability["augmentation_seeds"]:
            experiment = {
                "experiment_id": stability["experiment_id_template"].format(seed=seed),
                "augmentation_seed": seed,
            }
            command = [
                "uv", "run", "cafe", "run",
                *common_arguments(experiment),
                "--term", "short",
                "--dataset-ids", *stability["dataset_ids"],
                "--gift-eval-dir", "${GIFT_EVAL_DIR}",
            ]
            print(f"# stability seed {seed}")
            print(format_command(command))
    return 0


def figures(target: str, *, no_plots: bool) -> int:
    commands: dict[str, list[str]] = {
        "main": [
            "uv", "run", "--extra", "plots", "python",
            "paper_results/work/main_experiments/analyze_main_experiments.py",
        ],
        "stability": [
            "uv", "run", "--extra", "plots", "python",
            "paper_results/work/stability/analyze_stability.py",
        ],
        "finetuning": [
            "uv", "run", "--extra", "plots", "python",
            "paper_results/work/finetuning/analyze_finetuning.py",
            *( ["--no-plots"] if no_plots else [] ),
        ],
        "ablation": [
            "uv", "run", "--extra", "plots", "python",
            "paper_results/work/ablation/plot_ablation.py",
            "--input", "paper_results/data/ablation_collapsed_summary.csv",
            "--output-stem", "paper_results/work/ablation/fig_target_only_ablation",
            "--output-data", "paper_results/work/ablation/ablation_collapsed_summary_public.csv",
        ],
    }
    selected = list(commands) if target == "all" else [target]
    for name in selected:
        if no_plots and name == "ablation":
            continue
        print(f"running {name} reconstruction", flush=True)
        subprocess.run(commands[name], cwd=ROOT, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="verify the frozen reviewer artifacts")
    commands = subparsers.add_parser("commands", help="print full-run commands")
    commands.add_argument(
        "experiment",
        choices=(*CONFIG["main_experiments"], "stability", "all"),
    )
    reconstruct = subparsers.add_parser(
        "figures", help="reconstruct analyses and publication figures"
    )
    reconstruct.add_argument(
        "target", choices=("main", "stability", "finetuning", "ablation", "all")
    )
    reconstruct.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        return verify()
    if args.command == "commands":
        return print_commands(args.experiment)
    return figures(args.target, no_plots=args.no_plots)


if __name__ == "__main__":
    raise SystemExit(main())
