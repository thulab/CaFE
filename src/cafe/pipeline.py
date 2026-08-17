#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cafe import core as protocol
from cafe import provenance
from cafe.benchmark_extension.analysis import run_analysis
from cafe.benchmark_extension.generation import (
    DEFAULT_OUTPUT_ROOT,
    PIPELINE_SCHEMA,
    generate_dataset,
)
from cafe.benchmark_extension.mechanisms import CAPABILITY_IDS
from cafe.benchmark_extension.validation import validate_generation
from cafe.inference.runner import DEFAULT_ENDPOINTS, DEFAULT_MODELS


STAGES = ("generation", "validation", "inference", "analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GIFT-Eval Capability-Focused Extension pipeline."
    )
    parser.add_argument("--dataset-id", action="append", default=None)
    parser.add_argument("--dataset-ids", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    parser.add_argument("--term", choices=("short", "medium", "long"), default="short")
    parser.add_argument("--augmentation-seed", type=int, default=2026081601)
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=CAPABILITY_IDS,
        default=list(CAPABILITY_IDS),
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--start-at", choices=STAGES, default="generation")
    parser.add_argument("--stop-after", choices=STAGES, default="analysis")
    parser.add_argument("--resume-inference", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def selected_dataset_ids(args: argparse.Namespace) -> list[str]:
    values = list(args.dataset_ids or args.dataset_id or ["gift_electricity_h"])
    output = list(dict.fromkeys(str(value) for value in values))
    for dataset_id in output:
        dataset = protocol.resolve_dataset(dataset_id)
        if dataset.real_data_adapter not in {"gift_arrow", "gift_hierarchical_sales"}:
            raise ValueError(f"v6 supports GIFT-Eval only: {dataset_id}")
    return output


def _experiment_id(args: argparse.Namespace) -> str:
    if args.experiment_id:
        return str(args.experiment_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gift-extension-v6-{args.augmentation_seed}-{timestamp}"


def _stage_range(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = STAGES.index(start_at)
    stop = STAGES.index(stop_after)
    if start > stop:
        raise ValueError("start-at must not follow stop-after")
    return STAGES[start : stop + 1]


def _contract(
    experiment_root: Path,
    stage: str,
    config: dict[str, Any],
    upstream_paths: list[Path],
) -> dict[str, Any]:
    return provenance.ensure_stage_contract(
        experiment_root,
        stage=stage,
        created_at=protocol.utc_now(),
        repository_root=protocol.REPO_ROOT,
        config=config,
        upstream=provenance.upstream_records(
            upstream_paths,
            relative_to=experiment_root,
        ),
    )


def _inference_command(
    args: argparse.Namespace,
    experiment_root: Path,
    dataset_id: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "cafe.benchmark_extension.inference",
        "--dataset-id",
        dataset_id,
        "--output-root",
        str(experiment_root),
        "--models",
        *args.models,
        "--endpoints",
        *args.endpoints,
        "--api-prefix",
        args.api_prefix,
        "--devices",
        args.devices,
    ]
    if args.resume_inference:
        command.append("--resume")
    if args.prepare_only:
        command.append("--prepare-only")
    return command


def run_pipeline(args: argparse.Namespace) -> Path:
    dataset_ids = selected_dataset_ids(args)
    stages = _stage_range(args.start_at, args.stop_after)
    experiment_id = _experiment_id(args)
    experiment_root = args.output_root.resolve() / experiment_id
    provenance.initialize_experiment(
        experiment_root,
        experiment_id=experiment_id,
        created_at=protocol.utc_now(),
    )
    common = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "dataset_ids": dataset_ids,
        "term": args.term,
        "capability_ids": list(args.capabilities),
        "augmentation_seed": int(args.augmentation_seed),
        "max_instances": args.max_instances,
        "formal": args.max_instances is None,
    }
    if "generation" in stages:
        _contract(experiment_root, "generation", common, [])
        for dataset_id in dataset_ids:
            manifest_path = (
                experiment_root / dataset_id / "01_generation" / "manifest.json"
            )
            if manifest_path.exists():
                raise FileExistsError(
                    f"completed generation is immutable: {manifest_path}"
                )
            generate_dataset(
                dataset_id,
                gift_eval_dir=args.gift_eval_dir,
                dataset_root=experiment_root / dataset_id,
                term=args.term,
                augmentation_seed=args.augmentation_seed,
                capability_ids=tuple(args.capabilities),
                max_instances=args.max_instances,
            )
    generation_manifests = [
        experiment_root / dataset_id / "01_generation" / "manifest.json"
        for dataset_id in dataset_ids
    ]
    if "validation" in stages:
        _contract(experiment_root, "validation", common, generation_manifests)
        for dataset_id in dataset_ids:
            report_path = (
                experiment_root / dataset_id / "02_validation" / "report.json"
            )
            if report_path.exists():
                raise FileExistsError(
                    f"completed validation is immutable: {report_path}"
                )
            report = validate_generation(experiment_root / dataset_id)
            if not report["accepted"]:
                raise RuntimeError(f"generation validation failed: {dataset_id}")
    validation_reports = [
        experiment_root / dataset_id / "02_validation" / "report.json"
        for dataset_id in dataset_ids
    ]
    if "inference" in stages:
        inference_config = {
            **common,
            "models": list(args.models),
            "endpoints": list(args.endpoints),
            "model_input_policy": (
                "full_treatment_then_model_max_context_truncation"
            ),
        }
        _contract(
            experiment_root,
            "inference",
            inference_config,
            [*generation_manifests, *validation_reports],
        )
        for dataset_id in dataset_ids:
            subprocess.run(
                _inference_command(args, experiment_root, dataset_id),
                cwd=protocol.REPO_ROOT,
                check=True,
            )
    inference_manifests = [
        experiment_root / dataset_id / "03_inference" / "manifest.json"
        for dataset_id in dataset_ids
    ]
    if "analysis" in stages:
        _contract(
            experiment_root,
            "analysis",
            {
                **common,
                "analysis_policy": (
                    "baseline_treatment_accuracy_paired_effects_and_input_ablation"
                ),
            },
            inference_manifests,
        )
        for dataset_id in dataset_ids:
            manifest_path = (
                experiment_root / dataset_id / "04_analysis" / "manifest.json"
            )
            if manifest_path.exists():
                raise FileExistsError(
                    f"completed analysis is immutable: {manifest_path}"
                )
            run_analysis(experiment_root / dataset_id)
    return experiment_root


def main() -> int:
    args = parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be positive")
    if len(args.capabilities) != len(set(args.capabilities)):
        raise ValueError("capabilities must be unique")
    root = run_pipeline(args)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
