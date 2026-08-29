#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cafe import core as protocol
from cafe import provenance
from cafe.benchmark_extension.analysis import (
    aggregate_analysis_tasks,
    run_analysis,
)
from cafe.benchmark_extension.fev_bench import (
    FEV_MINI_SUITE_ID,
    FevBenchAdapter,
)
from cafe.benchmark_extension.generation import (
    DEFAULT_NATIVE_GENERATION_BATCH_BYTES,
    DEFAULT_OUTPUT_ROOT,
    PIPELINE_SCHEMA,
    generate_benchmark_task,
)
from cafe.benchmark_extension.inference import health_catalog
from cafe.benchmark_extension.mechanisms import (
    CAPABILITY_IDS,
    DEFAULT_CAPABILITY_IDS,
)
from cafe.benchmark_extension.validation import validate_generation
from cafe.inference.runner import DEFAULT_MODELS


STAGES = ("generation", "validation", "inference", "analysis")
FEV_SOURCE_REVISION = (
    "fev-v0.8.0-f1afffbf97bc51a4a233080d331633c6f7ab32f6+"
    "fev-datasets-f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CaFE benchmark extension on official FEV Mini-20 tasks."
    )
    parser.add_argument(
        "--suite-path",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "fev-mini-v0.8.0" / "tasks_mini.yaml",
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--source-revision", default=FEV_SOURCE_REVISION)
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--augmentation-seed", type=int, default=2026082601)
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=CAPABILITY_IDS,
        default=list(DEFAULT_CAPABILITY_IDS),
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["http://100.102.176.45:10810"],
    )
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--start-at", choices=STAGES, default="generation")
    parser.add_argument("--stop-after", choices=STAGES, default="analysis")
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--generation-shard-size", type=int, default=256)
    parser.add_argument(
        "--generation-batch-mib",
        type=int,
        default=DEFAULT_NATIVE_GENERATION_BATCH_BYTES // (1024 * 1024),
    )
    parser.add_argument("--validation-workers", type=int, default=4)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--analysis-workers", type=int, default=4)
    parser.add_argument("--max-open-shape-groups", type=int, default=64)
    parser.add_argument("--max-inflight-batches", type=int, default=8)
    parser.add_argument("--max-inflight-mib", type=int, default=2048)
    parser.add_argument("--resume-inference", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def _stage_range(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = STAGES.index(start_at)
    stop = STAGES.index(stop_after)
    if start > stop:
        raise ValueError("start-at must not follow stop-after")
    return STAGES[start : stop + 1]


def _experiment_id(args: argparse.Namespace) -> str:
    if args.experiment_id:
        return str(args.experiment_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"fev-mini-extension-v1-{args.augmentation_seed}-{timestamp}"


def _contract(
    experiment_root: Path,
    stage: str,
    config: dict[str, Any],
    upstream_paths: list[Path],
) -> None:
    provenance.ensure_stage_contract(
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


def _service_model_contracts(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    results: list[tuple[str, dict[str, dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=len(args.endpoints)) as executor:
        futures = [
            executor.submit(health_catalog, endpoint, args.api_prefix)
            for endpoint in args.endpoints
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    if not results:
        raise RuntimeError("no inference service is available for model preflight")
    contracts: dict[str, dict[str, Any]] = {}
    for model_id in args.models:
        observed = [catalog[model_id] for _endpoint, catalog in results if model_id in catalog]
        if not observed:
            raise ValueError(f"model {model_id!r} is unavailable on all endpoints")
        limits = observed[0].get("forecast_limits") or {}
        signature = protocol.canonical_json(limits)
        if any(
            protocol.canonical_json(row.get("forecast_limits") or {}) != signature
            for row in observed[1:]
        ):
            raise ValueError(f"model {model_id!r} has inconsistent endpoint limits")
        maximum_context = int(limits.get("max_input_length") or -1)
        if maximum_context < 1:
            raise ValueError(f"model {model_id!r} has no finite positive context limit")
        contracts[model_id] = {
            "maximum_context": maximum_context,
            "maximum_horizon": int(limits.get("max_output_length") or -1),
            "forecast_limits": limits,
        }
    return contracts


def _inference_command(
    args: argparse.Namespace,
    experiment_root: Path,
    task_id: str,
    *,
    execute_model: str | None = None,
    reuse_loaded_model: bool = False,
    preserve_loaded_model: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "cafe.benchmark_extension.inference",
        "--dataset-id",
        task_id,
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
        "--preprocess-workers",
        str(args.preprocess_workers),
        "--max-open-shape-groups",
        str(args.max_open_shape_groups),
        "--max-inflight-batches",
        str(args.max_inflight_batches),
        "--max-inflight-mib",
        str(args.max_inflight_mib),
    ]
    if execute_model is not None:
        command.extend(("--execute-models", execute_model, "--resume"))
    elif args.resume_inference:
        command.append("--resume")
    if reuse_loaded_model:
        command.append("--reuse-loaded-model")
    if preserve_loaded_model:
        command.append("--preserve-loaded-model")
    if args.prepare_only:
        command.append("--prepare-only")
    return command


def run_pipeline(args: argparse.Namespace) -> Path:
    if args.max_instances is not None and int(args.max_instances) < 1:
        raise ValueError("max-instances must be positive")
    if len(args.models) != len(set(args.models)):
        raise ValueError("models must be unique")
    stages = _stage_range(args.start_at, args.stop_after)
    adapter = FevBenchAdapter(
        args.suite_path,
        source_root=args.source_root,
        source_revision=args.source_revision,
    )
    available_task_ids = adapter.available_task_ids(FEV_MINI_SUITE_ID)
    requested = list(args.task_ids or args.task_id or available_task_ids)
    unknown = sorted(set(requested) - set(available_task_ids))
    if unknown:
        raise ValueError("unknown FEV task ids: " + ", ".join(unknown))
    selected_ids = tuple(dict.fromkeys(requested))
    selected_tasks = adapter.list_tasks(
        FEV_MINI_SUITE_ID,
        selected_task_ids=selected_ids,
    )
    by_id = {task.task_id: task for task in selected_tasks}
    tasks = [by_id[task_id] for task_id in selected_ids]

    experiment_root = args.output_root.resolve() / _experiment_id(args)
    provenance.initialize_experiment(
        experiment_root,
        experiment_id=experiment_root.name,
        created_at=protocol.utc_now(),
    )
    service_contracts = _service_model_contracts(args)
    for task in tasks:
        incompatible = [
            model_id
            for model_id, contract in service_contracts.items()
            if 0 <= int(contract["maximum_horizon"]) < int(task.horizon)
        ]
        if incompatible:
            raise ValueError(
                f"FEV task {task.task_id} H={task.horizon} exceeds model limits: "
                + ", ".join(incompatible)
            )
    model_contexts = {
        model_id: int(contract["maximum_context"])
        for model_id, contract in service_contracts.items()
    }
    task_ids = [task.task_id for task in tasks]
    common = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "benchmark_id": "fev_bench",
        "suite_id": FEV_MINI_SUITE_ID,
        "task_ids": task_ids,
        "model_ids": list(args.models),
        "capability_ids": list(args.capabilities),
        "augmentation_seed": int(args.augmentation_seed),
        "max_instances_per_task": args.max_instances,
        "generation_batch_bytes": int(args.generation_batch_mib) * 1024 * 1024,
        "validation_mode": "research",
        "model_service_contracts": service_contracts,
    }

    if "generation" in stages:
        _contract(experiment_root, "generation", common, [])
        for task in tasks:
            manifest_path = (
                experiment_root
                / task.task_id
                / "01_generation"
                / "manifest.json"
            )
            if manifest_path.exists():
                raise FileExistsError(
                    f"completed generation is immutable: {manifest_path}"
                )
            generate_benchmark_task(
                adapter,
                task,
                dataset_root=experiment_root / task.task_id,
                augmentation_seed=int(args.augmentation_seed),
                capability_ids=tuple(args.capabilities),
                model_max_contexts=model_contexts,
                max_instances=args.max_instances,
                workers=int(args.generation_workers),
                shard_size=int(args.generation_shard_size),
                maximum_batch_bytes=(
                    int(args.generation_batch_mib) * 1024 * 1024
                ),
            )
    generation_manifests = [
        experiment_root / task_id / "01_generation" / "manifest.json"
        for task_id in task_ids
    ]

    if "validation" in stages:
        _contract(experiment_root, "validation", common, generation_manifests)
        for task_id in task_ids:
            report_path = (
                experiment_root / task_id / "02_validation" / "report.json"
            )
            if report_path.exists():
                raise FileExistsError(
                    f"completed validation is immutable: {report_path}"
                )
            report = validate_generation(
                experiment_root / task_id,
                mode="research",
                workers=int(args.validation_workers),
            )
            if not report["accepted"]:
                raise RuntimeError(f"research validation failed: {task_id}")
    validation_reports = [
        experiment_root / task_id / "02_validation" / "report.json"
        for task_id in task_ids
    ]

    if "inference" in stages:
        _contract(
            experiment_root,
            "inference",
            {**common, "endpoints": list(args.endpoints)},
            [*generation_manifests, *validation_reports],
        )
        if args.prepare_only:
            for task_id in task_ids:
                subprocess.run(
                    _inference_command(args, experiment_root, task_id),
                    cwd=protocol.REPO_ROOT,
                    check=True,
                )
        else:
            for model_id in args.models:
                for index, task_id in enumerate(task_ids):
                    subprocess.run(
                        _inference_command(
                            args,
                            experiment_root,
                            task_id,
                            execute_model=model_id,
                            reuse_loaded_model=index > 0,
                            preserve_loaded_model=index < len(task_ids) - 1,
                        ),
                        cwd=protocol.REPO_ROOT,
                        check=True,
                    )
    inference_manifests = [
        experiment_root / task_id / "03_inference" / "manifest.json"
        for task_id in task_ids
    ]

    if "analysis" in stages:
        _contract(experiment_root, "analysis", common, inference_manifests)
        for task_id in task_ids:
            manifest_path = (
                experiment_root / task_id / "04_analysis" / "manifest.json"
            )
            if manifest_path.exists():
                raise FileExistsError(
                    f"completed analysis is immutable: {manifest_path}"
                )
            run_analysis(
                experiment_root / task_id,
                replay_workers=int(args.analysis_workers),
            )
        aggregate_analysis_tasks(experiment_root, task_ids)
    return experiment_root


def main() -> int:
    root = run_pipeline(parse_args())
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
