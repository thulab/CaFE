#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
from cafe.benchmark_extension.validation import (
    DEFAULT_VALIDATION_WORKERS,
    VALIDATION_MODES,
    validate_generation,
)
from cafe.benchmark_extension.gift_eval import (
    gift_arrow_target_summary,
    gift_eval_asset_path,
    iter_gift_arrow_target_records,
    official_window_count_from_minimum_length,
    prediction_length,
)
from cafe.inference.runner import DEFAULT_ENDPOINTS, DEFAULT_MODELS


STAGES = ("generation", "validation", "inference", "analysis")


def _analyse_dataset_process(
    job: tuple[str, str, str],
) -> str:
    dataset_id, experiment_root_value, gift_eval_dir_value = job
    experiment_root = Path(experiment_root_value)
    manifest_path = (
        experiment_root / dataset_id / "04_analysis" / "manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(f"completed analysis is immutable: {manifest_path}")
    run_analysis(
        experiment_root / dataset_id,
        gift_eval_dir=Path(gift_eval_dir_value),
        replay_workers=1,
    )
    return dataset_id


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
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--generation-shard-size", type=int, default=256)
    parser.add_argument("--validation-dataset-workers", type=int, default=2)
    parser.add_argument(
        "--validation-mode",
        choices=VALIDATION_MODES,
        default="research",
    )
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=DEFAULT_VALIDATION_WORKERS,
        help="Per-dataset process workers for validation.",
    )
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--analysis-workers", type=int, default=4)
    parser.add_argument("--max-open-shape-groups", type=int, default=64)
    parser.add_argument("--max-inflight-batches", type=int, default=8)
    parser.add_argument("--max-inflight-mib", type=int, default=2048)
    parser.add_argument("--max-request-input-tokens", type=int, default=None)
    parser.add_argument("--client-inflight-input-tokens", type=int, default=None)
    parser.add_argument("--disk-budget-gb", type=float, default=40.0)
    return parser.parse_args()


def selected_dataset_ids(args: argparse.Namespace) -> list[str]:
    values = list(args.dataset_ids or args.dataset_id or ["gift_electricity_h"])
    output = list(dict.fromkeys(str(value) for value in values))
    for dataset_id in output:
        dataset = protocol.resolve_dataset(dataset_id)
        if dataset.real_data_adapter not in {"gift_arrow", "gift_hierarchical_sales"}:
            raise ValueError(f"v7 supports GIFT-Eval only: {dataset_id}")
    return output


def _experiment_id(args: argparse.Namespace) -> str:
    if args.experiment_id:
        return str(args.experiment_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gift-extension-v7-{args.augmentation_seed}-{timestamp}"


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
        "--gift-eval-dir",
        str(args.gift_eval_dir),
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
        command.extend(("--execute-models", execute_model))
    if reuse_loaded_model:
        command.append("--reuse-loaded-model")
    if preserve_loaded_model:
        command.append("--preserve-loaded-model")
    if args.resume_inference or execute_model is not None:
        command.append("--resume")
    if args.prepare_only:
        command.append("--prepare-only")
    max_request_input_tokens = getattr(args, "max_request_input_tokens", None)
    client_inflight_input_tokens = getattr(
        args, "client_inflight_input_tokens", None
    )
    if max_request_input_tokens is not None:
        command.extend(
            (
                "--max-request-input-tokens",
                str(max_request_input_tokens),
            )
        )
    if client_inflight_input_tokens is not None:
        command.extend(
            (
                "--client-inflight-input-tokens",
                str(client_inflight_input_tokens),
            )
        )
    return command


def _storage_preflight(
    dataset_ids: list[str],
    *,
    gift_eval_dir: Path,
    term: str,
    capability_ids: list[str],
    model_count: int,
) -> dict[str, Any]:
    generated_capabilities = sum(
        capability != "hierarchical_coherence" for capability in capability_ids
    )
    ablation_capabilities = sum(
        capability in {"common_factor", "cross_series_dependence"}
        for capability in capability_ids
    )
    maximum_views = 1 + generated_capabilities * 5 + ablation_capabilities * 5
    rows: list[dict[str, Any]] = []
    total_forecast_values = 0
    total_instances = 0
    for dataset_id in dataset_ids:
        asset = gift_eval_asset_path(dataset_id, gift_eval_dir)
        frequency, minimum_length, _record_count = gift_arrow_target_summary(asset)
        horizon = prediction_length(dataset_id, frequency, term=term)
        windows = official_window_count_from_minimum_length(
            dataset_id, minimum_length, horizon
        )
        instance_count = 0
        forecast_values = 0
        for _item_id, _frequency, target in iter_gift_arrow_target_records(asset):
            dimension = 1 if target.ndim == 1 else int(target.shape[0])
            instance_count += windows
            forecast_values += windows * maximum_views * horizon * dimension
        rows.append(
            {
                "dataset_id": dataset_id,
                "official_instance_upper_bound": instance_count,
                "maximum_generated_views_per_instance": maximum_views,
                "forecast_float_count_per_model_upper_bound": forecast_values,
            }
        )
        total_instances += instance_count
        total_forecast_values += forecast_values
    # Forecast float32 plus Parquet metadata; compact contracts and scalar metric rows
    # use conservative empirical byte allowances. Availability can only reduce this.
    prediction_bytes = total_forecast_values * max(1, model_count) * 4 * 1.25
    contract_rows = total_instances * maximum_views
    contract_bytes = contract_rows * 700
    metric_bytes = contract_rows * max(1, model_count) * 220
    estimated = int(prediction_bytes + contract_bytes + metric_bytes)
    peak = int(estimated * 1.2)
    return {
        "schema_version": "cafe.storage_preflight.v1",
        "policy": "conservative_supported_capability_upper_bound",
        "dataset_count": len(dataset_ids),
        "model_count": int(model_count),
        "maximum_views_per_instance": maximum_views,
        "estimated_steady_state_bytes": estimated,
        "estimated_steady_state_gib": estimated / (1024**3),
        "estimated_peak_bytes": peak,
        "estimated_peak_gib": peak / (1024**3),
        "datasets": rows,
    }


def _freeze_storage_preflight(
    experiment_root: Path,
    computed: dict[str, Any],
    *,
    disk_budget_gb: float,
) -> dict[str, Any]:
    """Freeze reproducible estimates while rechecking current free space on resume."""

    path = experiment_root / "storage_preflight.json"
    budget_bytes = int(float(disk_budget_gb) * (1024**3))
    free_bytes = shutil.disk_usage(experiment_root.parent).free
    accepted_now = bool(
        computed["estimated_peak_bytes"] <= budget_bytes
        and computed["estimated_peak_bytes"] <= int(free_bytes * 0.9)
    )
    if path.is_file():
        frozen = protocol.read_json(path)
        comparable_keys = (
            "schema_version",
            "policy",
            "dataset_count",
            "model_count",
            "maximum_views_per_instance",
            "estimated_steady_state_bytes",
            "estimated_peak_bytes",
            "datasets",
        )
        if any(frozen.get(key) != computed.get(key) for key in comparable_keys):
            raise ValueError(
                "existing storage preflight does not match the resumed experiment"
            )
        if int(frozen.get("configured_budget_bytes", -1)) != budget_bytes:
            raise ValueError(
                "existing storage preflight uses a different disk budget"
            )
        if not accepted_now:
            raise RuntimeError(
                "current free space is below the frozen preflight requirement"
            )
        return frozen
    frozen = {
        **computed,
        "configured_budget_bytes": budget_bytes,
        "configured_budget_gib": float(disk_budget_gb),
        "filesystem_free_bytes": int(free_bytes),
        "accepted": accepted_now,
    }
    protocol.write_json(path, frozen)
    if not accepted_now:
        raise RuntimeError(
            "estimated artifact footprint exceeds the configured disk budget or "
            "90% of available space; see storage_preflight.json"
        )
    return frozen


def run_pipeline(args: argparse.Namespace) -> Path:
    dataset_ids = selected_dataset_ids(args)
    stages = _stage_range(args.start_at, args.stop_after)
    validation_workers = int(
        getattr(args, "validation_workers", DEFAULT_VALIDATION_WORKERS)
    )
    requested_validation_dataset_workers = max(
        1, int(args.validation_dataset_workers)
    )
    validation_dataset_workers = (
        1 if validation_workers > 1 else requested_validation_dataset_workers
    )
    experiment_id = _experiment_id(args)
    experiment_root = args.output_root.resolve() / experiment_id
    provenance.initialize_experiment(
        experiment_root,
        experiment_id=experiment_id,
        created_at=protocol.utc_now(),
    )
    computed_preflight = _storage_preflight(
        dataset_ids,
        gift_eval_dir=args.gift_eval_dir,
        term=args.term,
        capability_ids=list(args.capabilities),
        model_count=len(args.models),
    )
    storage_preflight = _freeze_storage_preflight(
        experiment_root,
        computed_preflight,
        disk_budget_gb=float(args.disk_budget_gb),
    )
    common = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "dataset_ids": dataset_ids,
        "term": args.term,
        "capability_ids": list(args.capabilities),
        "augmentation_seed": int(args.augmentation_seed),
        "max_instances": args.max_instances,
        "formal": args.max_instances is None,
        "artifact_format": "parquet_zstd",
        "storage_preflight_sha256": protocol.file_sha256(
            experiment_root / "storage_preflight.json"
        ),
        "execution": {
            "generation_workers": int(args.generation_workers),
            "generation_shard_size": int(args.generation_shard_size),
            "validation_dataset_workers": validation_dataset_workers,
            "requested_validation_dataset_workers": (
                requested_validation_dataset_workers
            ),
            "validation_mode": str(getattr(args, "validation_mode", "research")),
            "validation_workers": validation_workers,
            "validation_parallelism_policy": (
                "one_dataset_at_a_time_with_row_group_process_pool"
                if validation_workers > 1
                else "concurrent_datasets_with_single_process_scans"
            ),
            "preprocess_workers": int(args.preprocess_workers),
            "analysis_workers": int(args.analysis_workers),
            "max_open_shape_groups": int(args.max_open_shape_groups),
            "max_inflight_batches": int(args.max_inflight_batches),
            "max_inflight_mib": int(args.max_inflight_mib),
            "max_request_input_tokens_override": getattr(
                args, "max_request_input_tokens", None
            ),
            "client_inflight_input_tokens_override": (
                getattr(args, "client_inflight_input_tokens", None)
            ),
        },
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
                workers=args.generation_workers,
                shard_size=args.generation_shard_size,
            )
    generation_manifests = [
        experiment_root / dataset_id / "01_generation" / "manifest.json"
        for dataset_id in dataset_ids
    ]
    if "validation" in stages:
        _contract(experiment_root, "validation", common, generation_manifests)
        def validate_one(dataset_id: str) -> tuple[str, dict[str, Any]]:
            report_path = (
                experiment_root / dataset_id / "02_validation" / "report.json"
            )
            if report_path.exists():
                raise FileExistsError(
                    f"completed validation is immutable: {report_path}"
                )
            report = validate_generation(
                experiment_root / dataset_id,
                gift_eval_dir=args.gift_eval_dir,
                mode=str(getattr(args, "validation_mode", "research")),
                workers=validation_workers,
            )
            return dataset_id, report

        with ThreadPoolExecutor(
            max_workers=validation_dataset_workers
        ) as executor:
            for dataset_id, report in executor.map(validate_one, dataset_ids):
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
        if args.prepare_only:
            for dataset_id in dataset_ids:
                subprocess.run(
                    _inference_command(args, experiment_root, dataset_id),
                    cwd=protocol.REPO_ROOT,
                    check=True,
                )
        else:
            # Model-major execution loads each model once across all compatible
            # endpoints/GPUs, streams every dataset, then unloads before the next.
            for model_id in args.models:
                pending_datasets: list[str] = []
                for dataset_id in dataset_ids:
                    manifest_path = (
                        experiment_root / dataset_id / "03_inference" / "manifest.json"
                    )
                    if manifest_path.is_file():
                        existing = protocol.read_json(manifest_path)
                        complete_models = {
                            str(row["model_id"])
                            for row in existing.get("model_statuses") or []
                            if row.get("status") == "complete"
                        }
                        record = (existing.get("model_predictions") or {}).get(model_id)
                        parts = record.get("parts") if isinstance(record, dict) else None
                        zero_row_complete = bool(
                            isinstance(record, dict)
                            and int(record.get("row_count", -1)) == 0
                        )
                        artifacts_valid = (zero_row_complete or bool(parts)) and all(
                            Path(str(part["path"])).is_file()
                            and protocol.file_sha256(Path(str(part["path"])))
                            == part["sha256"]
                            for part in (parts or [])
                        )
                        if model_id in complete_models and artifacts_valid:
                            continue
                    pending_datasets.append(dataset_id)
                for dataset_index, dataset_id in enumerate(pending_datasets):
                    subprocess.run(
                        _inference_command(
                            args,
                            experiment_root,
                            dataset_id,
                            execute_model=model_id,
                            reuse_loaded_model=dataset_index > 0,
                            preserve_loaded_model=(
                                dataset_index < len(pending_datasets) - 1
                            ),
                        ),
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
        # A process owns one dataset and replays each source shard once for every
        # model.  Dataset processes avoid the Python GIL without nested worker
        # pools or repeated source-Arrow scans.
        dataset_analysis_workers = max(
            1,
            min(len(dataset_ids), max(1, int(args.analysis_workers))),
        )
        jobs = [
            (dataset_id, str(experiment_root), str(args.gift_eval_dir.resolve()))
            for dataset_id in dataset_ids
        ]
        with ProcessPoolExecutor(max_workers=dataset_analysis_workers) as executor:
            list(executor.map(_analyse_dataset_process, jobs))
    return experiment_root


def main() -> int:
    args = parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("max_instances must be positive")
    max_request_input_tokens = getattr(args, "max_request_input_tokens", None)
    client_inflight_input_tokens = getattr(
        args, "client_inflight_input_tokens", None
    )
    if max_request_input_tokens is not None and max_request_input_tokens < 1:
        raise ValueError("max-request-input-tokens must be positive")
    if (
        client_inflight_input_tokens is not None
        and client_inflight_input_tokens < 1
    ):
        raise ValueError("client-inflight-input-tokens must be positive")
    if len(args.capabilities) != len(set(args.capabilities)):
        raise ValueError("capabilities must be unique")
    root = run_pipeline(args)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
