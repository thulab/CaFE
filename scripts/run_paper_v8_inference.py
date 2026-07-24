#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import paper_v8_pipeline_common as v8
import run_paper_e2_dynamic_stability as engine
import run_paper_v5_e2_inference as v7_inference


DEFAULT_OUTPUT_ROOT = (
    v8.REPO_ROOT / "runtime" / "paper_exp" / "v8_test" / "full_pipeline"
)
DEFAULT_ENDPOINTS = (
    "http://127.0.0.1:10810",
    "http://192.168.99.17:10811",
    "http://192.168.99.18:10810",
)
DEFAULT_MODELS = ("Chronos-2", "toto2.0", "tirex2", "timesfm2.5")
MODEL_COST = {
    "tirex2": 5.0,
    "toto2.0": 4.0,
    "Chronos-2": 3.0,
    "timesfm2.5": 2.0,
    "Timer-3.5": 4.0,
    "Timer-3.0": 4.0,
    "moirai2": 3.0,
    "tabpfn-ts3": 2.0,
}
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class InferenceWork:
    model_id: str
    sample_path: Path
    output_dir: Path
    work_id: str
    tail_part_index: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal Paper v8 multi-service model inference."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--disable-tail-sharding",
        action="store_true",
        help="Do not split the queued tail model across otherwise idle services.",
    )
    return parser.parse_args()


def iter_forecast_samples(path: Path) -> Iterator[dict[str, Any]]:
    yield from v8.iter_jsonl(path)


def prediction_path_for(
    output_dir: Path,
    model_id: str,
    *,
    prediction_kind: str = "synthetic",
) -> Path:
    return (
        output_dir
        / "predictions"
        / f"{engine.safe_filename(model_id)}.jsonl"
    )


def prediction_row(
    model_id: str,
    model_group: str,
    sample: dict[str, Any],
    forecast: np.ndarray | list[list[float]],
) -> dict[str, Any]:
    values = np.asarray(forecast, dtype=float)
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    truth = target[context:]
    expected_shape = (int(sample["horizon"]), int(sample["target_dim"]))
    if values.shape != expected_shape:
        raise ValueError(
            f"forecast shape mismatch: {values.shape} != {expected_shape}"
        )
    mae = float(np.mean(np.abs(values - truth)))
    scale = float(sample["mase_scale"])
    return {
        "schema_version": "paper_v8_inference_prediction.v1",
        "model_id": model_id,
        "model_group": model_group,
        "sample_id": sample["sample_id"],
        "view_id": sample["sample_id"],
        "master_sample_id": sample["master_sample_id"],
        "dataset_id": sample["dataset_id"],
        "config_id": sample["config_id"],
        "profile_id": sample["profile_id"],
        "capability_id": sample["capability_id"],
        "generator_family_role": sample["generator_family_role"],
        "generator_family_id": sample["generator_family_id"],
        "evaluation_table": sample.get("evaluation_table", "main"),
        "intensity": int(sample["intensity"]),
        "seed_index": int(sample["seed_index"]),
        "counterfactual_pair_id": sample.get("counterfactual_pair_id"),
        "counterfactual_member": sample.get("counterfactual_member"),
        "context_length": context,
        "horizon": int(sample["horizon"]),
        "target_dim": int(sample["target_dim"]),
        "covariate_dim": int(sample["covariate_dim"]),
        "mase_scale": scale,
        "metrics": {"mae": mae, "mase": mae / scale},
        "forecast": values.tolist(),
        "target_future": truth.tolist(),
        "future_sha256": sample["future_sha256"],
    }


def install_engine_hooks() -> None:
    engine.iter_forecast_samples = iter_forecast_samples
    engine.prediction_row = prediction_row
    engine.prediction_path_for = prediction_path_for


def prepare_view_tasks(
    generation_manifest: dict[str, Any],
    *,
    inference_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    source_records = [
        generation_manifest["files"]["clean"],
        generation_manifest["files"]["robustness"],
    ]
    masters = (
        row
        for record in source_records
        for row in v8.iter_jsonl(Path(record["path"]))
    )
    task_path = inference_dir / "forecast_views.jsonl"
    view_count = v8.write_jsonl(
        task_path,
        v8.iter_master_views(masters),
    )
    manifest = {
        "schema_version": "paper_v8_inference_task_manifest.v1",
        "created_at": v8.utc_now(),
        "generation_config_sha256": generation_manifest["config_sha256"],
        "generation_files": source_records,
        "context_lengths": list(v8.VIEW_CONTEXT_LENGTHS),
        "view_count": view_count,
        "task_file": {
            **v8.file_record(task_path),
            "row_count": view_count,
        },
        "mase_policy": "shared_clean_l504_denominator_across_views",
    }
    v8.write_json(inference_dir / "task_manifest.json", manifest)
    return task_path, manifest


def health_catalog(
    endpoint: str,
    api_prefix: str,
) -> tuple[str, dict[str, dict[str, Any]]] | None:
    client = engine.TimerServiceClient(endpoint, api_prefix, timeout_seconds=30)
    try:
        catalog = {
            str(row["model_id"]): row for row in client.list_models()
        }
        return endpoint, catalog
    except Exception as error:  # noqa: BLE001
        with _PRINT_LOCK:
            print(f"endpoint unavailable {endpoint}: {type(error).__name__}: {error}")
        return None
    finally:
        client.close()


def assign_models(
    models: list[str],
    services: list[tuple[str, dict[str, dict[str, Any]]]],
) -> dict[str, list[str]]:
    loads = {endpoint: 0.0 for endpoint, _catalog in services}
    assignments = {endpoint: [] for endpoint, _catalog in services}
    catalogs = {endpoint: catalog for endpoint, catalog in services}
    for model_id in sorted(
        models,
        key=lambda name: (-MODEL_COST.get(name, 1.0), name),
    ):
        eligible = [
            endpoint
            for endpoint in assignments
            if model_id in catalogs[endpoint]
        ]
        if not eligible:
            raise ValueError(f"model {model_id!r} unavailable on all services")
        endpoint = min(eligible, key=lambda name: (loads[name], name))
        assignments[endpoint].append(model_id)
        loads[endpoint] += MODEL_COST.get(model_id, 1.0)
    return assignments


def _tail_manifest_path(inference_dir: Path) -> Path:
    return inference_dir / "tail_shard_manifest.json"


def prepare_tail_task_shards(
    task_path: Path,
    *,
    model_id: str,
    part_count: int,
    inference_dir: Path,
) -> dict[str, Any]:
    if part_count < 2:
        raise ValueError("tail sharding requires at least two parts")
    manifest_path = _tail_manifest_path(inference_dir)
    if manifest_path.exists():
        manifest = v8.read_json(manifest_path)
        if manifest["model_id"] != model_id:
            raise ValueError("existing tail shard model mismatch")
        if int(manifest["part_count"]) != part_count:
            raise ValueError("existing tail shard count mismatch")
        if manifest["source_task_sha256"] != v8.file_sha256(task_path):
            raise ValueError("existing tail shard source task mismatch")
        for part in manifest["parts"]:
            path = Path(part["path"])
            if v8.file_sha256(path) != part["sha256"]:
                raise ValueError(f"tail task shard hash mismatch: {path}")
        return manifest

    shard_dir = (
        inference_dir
        / "tail_task_shards"
        / engine.safe_filename(model_id)
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    part_paths = [
        shard_dir / f"part_{index:03d}.jsonl"
        for index in range(part_count)
    ]
    handles = [
        path.open("w", encoding="utf-8") for path in part_paths
    ]
    counts = [0] * part_count
    try:
        for row in v8.iter_jsonl(task_path):
            part_index = (
                v8.stable_seed(
                    "inference_tail_shard",
                    model_id,
                    str(row["sample_id"]),
                )
                % part_count
            )
            handles[part_index].write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            counts[part_index] += 1
    finally:
        for handle in handles:
            handle.close()
    if not all(counts):
        raise ValueError("tail task partition produced an empty shard")
    parts = [
        {
            "part_index": index,
            "row_count": counts[index],
            **v8.file_record(path),
        }
        for index, path in enumerate(part_paths)
    ]
    manifest = {
        "schema_version": "paper_v8_tail_shard_manifest.v1",
        "created_at": v8.utc_now(),
        "model_id": model_id,
        "part_count": part_count,
        "partition_policy": "stable_hash_of_model_and_sample_id",
        "source_task_sha256": v8.file_sha256(task_path),
        "source_task_row_count": sum(counts),
        "parts": parts,
    }
    v8.write_json(manifest_path, manifest)
    return manifest


def plan_inference_work(
    models: list[str],
    services: list[tuple[str, dict[str, dict[str, Any]]]],
    *,
    task_path: Path,
    inference_dir: Path,
    enable_tail_sharding: bool,
) -> tuple[
    dict[str, list[InferenceWork]],
    dict[str, list[str]],
    dict[str, Any] | None,
]:
    assignments = assign_models(models, services)
    catalogs = dict(services)
    work = {
        endpoint: [
            InferenceWork(
                model_id=model_id,
                sample_path=task_path,
                output_dir=(
                    inference_dir
                    / "model_shards"
                    / engine.safe_filename(model_id)
                ),
                work_id=f"{model_id}__full",
            )
            for model_id in model_ids
        ]
        for endpoint, model_ids in assignments.items()
    }
    if not enable_tail_sharding:
        return work, assignments, None

    existing_manifest = (
        v8.read_json(_tail_manifest_path(inference_dir))
        if _tail_manifest_path(inference_dir).exists()
        else None
    )
    if existing_manifest is not None:
        tail_model = str(existing_manifest["model_id"])
        if tail_model not in models:
            raise ValueError(
                f"tail shard manifest contains unrequested model {tail_model!r}"
            )
        eligible = sorted(
            endpoint
            for endpoint, catalog in services
            if tail_model in catalog
        )
        if not eligible:
            raise ValueError(
                f"tail-sharded model {tail_model!r} unavailable"
            )
        for endpoint in work:
            work[endpoint] = [
                item for item in work[endpoint]
                if item.model_id != tail_model
            ]
        manifest = prepare_tail_task_shards(
            task_path,
            model_id=tail_model,
            part_count=int(existing_manifest["part_count"]),
            inference_dir=inference_dir,
        )
    else:
        queued_endpoints = [
            endpoint
            for endpoint, model_ids in assignments.items()
            if len(model_ids) > 1
        ]
        if not queued_endpoints:
            return work, assignments, None
        tail_endpoint = max(
            queued_endpoints,
            key=lambda endpoint: (
                sum(
                    MODEL_COST.get(model_id, 1.0)
                    for model_id in assignments[endpoint]
                ),
                endpoint,
            ),
        )
        tail_model = assignments[tail_endpoint][-1]
        canonical_prediction = prediction_path_for(
            inference_dir
            / "model_shards"
            / engine.safe_filename(tail_model),
            tail_model,
        )
        if canonical_prediction.exists():
            return work, assignments, None
        eligible = sorted(
            endpoint
            for endpoint, catalog in services
            if tail_model in catalog
        )
        if len(eligible) < 2:
            return work, assignments, None
        work[tail_endpoint] = [
            item
            for item in work[tail_endpoint]
            if item.model_id != tail_model
        ]
        manifest = prepare_tail_task_shards(
            task_path,
            model_id=tail_model,
            part_count=len(eligible),
            inference_dir=inference_dir,
        )

    for part in manifest["parts"]:
        part_index = int(part["part_index"])
        endpoint = eligible[part_index % len(eligible)]
        model_root = (
            inference_dir
            / "model_shards"
            / engine.safe_filename(tail_model)
        )
        work.setdefault(endpoint, []).append(
            InferenceWork(
                model_id=tail_model,
                sample_path=Path(part["path"]),
                output_dir=(
                    model_root
                    / "tail_parts"
                    / f"part_{part_index:03d}"
                ),
                work_id=f"{tail_model}__tail_part_{part_index:03d}",
                tail_part_index=part_index,
            )
        )
    return work, assignments, manifest


def run_service_queue(
    endpoint: str,
    work_items: list[InferenceWork],
    catalog: dict[str, dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    client = engine.TimerServiceClient(
        endpoint,
        args.api_prefix,
        timeout_seconds=30,
    )
    try:
        for item in work_items:
            model_id = item.model_id
            model_dir = item.output_dir
            for directory in ("predictions", "failures"):
                (model_dir / directory).mkdir(parents=True, exist_ok=True)
            execution = dict(v7_inference.MODEL_EXECUTION_CONFIG[model_id])
            with _PRINT_LOCK:
                print(
                    f"{endpoint}: starting {item.work_id}, "
                    f"replicas={execution['replicas_per_device']}, "
                    f"concurrency={execution['http_concurrency']}",
                    flush=True,
                )
            started = time.monotonic()
            try:
                status = engine.run_one_model(
                    client,
                    catalog[model_id],
                    output_dir=model_dir,
                    execution=execution,
                    devices=args.devices,
                    request_max_attempts=args.max_attempts,
                    forecast_timeout_seconds=args.forecast_timeout_seconds,
                    load_timeout_seconds=args.load_timeout_seconds,
                    keep_loaded=False,
                    sample_path=item.sample_path,
                    prediction_kind="synthetic",
                    status_filename="model_status.json",
                    input_adaptation_policy=engine.INPUT_ADAPTATION_POLICY_ID,
                )
            except Exception as error:  # noqa: BLE001
                status = {
                    "model_id": model_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.monotonic() - started,
                }
            status["endpoint"] = endpoint
            status["work_id"] = item.work_id
            status["tail_part_index"] = item.tail_part_index
            status["sample_path"] = str(item.sample_path)
            statuses.append(status)
            v8.write_json(
                model_dir / "service_status.json",
                status,
            )
            with _PRINT_LOCK:
                print(
                    f"{endpoint}: {item.work_id} {status['status']} "
                    f"{status.get('succeeded_count', 0)}/"
                    f"{status.get('compatible_sample_count', 0)}",
                    flush=True,
                )
    finally:
        try:
            client.unload_all_loaded()
        except Exception:  # noqa: BLE001
            pass
        client.close()
    return statuses


def consolidate_tail_predictions(
    inference_dir: Path,
    tail_manifest: dict[str, Any] | None,
) -> None:
    if tail_manifest is None:
        return
    model_id = str(tail_manifest["model_id"])
    model_root = (
        inference_dir / "model_shards" / engine.safe_filename(model_id)
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    complete = True
    for part in tail_manifest["parts"]:
        part_index = int(part["part_index"])
        part_root = (
            model_root / "tail_parts" / f"part_{part_index:03d}"
        )
        path = prediction_path_for(part_root, model_id)
        if not path.exists():
            complete = False
            continue
        part_rows = list(v8.iter_jsonl(path))
        if len(part_rows) != int(part["row_count"]):
            complete = False
        for row in part_rows:
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(
                    f"duplicate tail prediction for {model_id}: {sample_id}"
                )
            seen.add(sample_id)
            rows.append(row)
    if not complete:
        return
    expected = int(tail_manifest["source_task_row_count"])
    if len(rows) != expected:
        raise ValueError(
            f"tail prediction coverage mismatch: {len(rows)} != {expected}"
        )
    rows.sort(key=lambda row: str(row["sample_id"]))
    canonical_path = prediction_path_for(model_root, model_id)
    v8.write_jsonl(canonical_path, rows)


def aggregate_model_statuses(
    models: list[str],
    work_statuses: list[dict[str, Any]],
    *,
    inference_dir: Path,
    expected_view_count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    count_fields = (
        "compatible_sample_count",
        "expected_original_view_count",
        "succeeded_count",
        "succeeded_original_view_count",
        "native_view_count",
        "adapted_view_count",
        "split_target_view_count",
        "covariates_omitted_view_count",
        "expected_http_request_count",
        "successful_http_request_count",
        "failed_request_count_this_attempt",
    )
    for model_id in models:
        matching = [
            row for row in work_statuses
            if row.get("model_id") == model_id
        ]
        canonical_path = prediction_path_for(
            inference_dir
            / "model_shards"
            / engine.safe_filename(model_id),
            model_id,
        )
        observed = (
            engine.count_jsonl(canonical_path)
            if canonical_path.exists()
            else 0
        )
        status: dict[str, Any] = {
            "model_id": model_id,
            "status": (
                "complete"
                if observed == expected_view_count
                and matching
                and all(
                    row.get("status") == "complete"
                    for row in matching
                )
                else "failed"
            ),
            "expected_original_view_count": expected_view_count,
            "succeeded_original_view_count": observed,
            "endpoints": sorted(
                {
                    str(row["endpoint"])
                    for row in matching
                    if row.get("endpoint")
                }
            ),
            "work_statuses": matching,
            "prediction_path": str(canonical_path),
        }
        for field in count_fields:
            values = [
                int(row[field])
                for row in matching
                if row.get(field) is not None
            ]
            if values:
                status[field] = sum(values)
        output.append(status)
        v8.write_json(
            inference_dir
            / "model_shards"
            / engine.safe_filename(model_id)
            / "service_status.json",
            status,
        )
    return output


def merge_predictions(
    inference_dir: Path,
    models: list[str],
) -> tuple[Path, int]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for model_id in models:
        path = prediction_path_for(
            inference_dir / "model_shards" / engine.safe_filename(model_id),
            model_id,
        )
        if not path.exists():
            continue
        for row in v8.iter_jsonl(path):
            key = (str(row["model_id"]), str(row["sample_id"]))
            if key in seen:
                raise ValueError(f"duplicate inference prediction {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["model_id"], row["sample_id"]))
    merged_path = inference_dir / "predictions.jsonl"
    count = v8.write_jsonl(merged_path, rows)
    return merged_path, count


def main() -> int:
    args = parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError("model ids must be unique")
    missing_configs = sorted(
        set(args.models) - set(v7_inference.MODEL_EXECUTION_CONFIG)
    )
    if missing_configs:
        raise ValueError(
            "missing model execution configs: " + ", ".join(missing_configs)
        )
    install_engine_hooks()
    dataset = v8.resolve_dataset(args.dataset_id)
    dataset_root = args.output_root.resolve() / dataset.dataset_id
    generation_dir = dataset_root / "02_generation"
    shard_name = (
        f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    )
    generation_manifest_path = (
        generation_dir / f"manifest__{shard_name}.json"
    )
    validation_path = generation_dir / f"validation__{shard_name}.json"
    validation = v8.read_json(validation_path)
    if not validation["accepted"]:
        raise ValueError("generation validation is not accepted")
    generation_manifest = v8.read_json(generation_manifest_path)
    inference_dir = dataset_root / "03_inference" / shard_name
    inference_dir.mkdir(parents=True, exist_ok=True)
    task_manifest_path = inference_dir / "task_manifest.json"
    if task_manifest_path.exists() and args.resume:
        task_manifest = v8.read_json(task_manifest_path)
        task_path = Path(task_manifest["task_file"]["path"])
        if v8.file_sha256(task_path) != task_manifest["task_file"]["sha256"]:
            raise ValueError("existing inference task file hash mismatch")
        if (
            task_manifest["generation_config_sha256"]
            != generation_manifest["config_sha256"]
        ):
            raise ValueError("resume generation config mismatch")
    else:
        task_path, task_manifest = prepare_view_tasks(
            generation_manifest,
            inference_dir=inference_dir,
        )

    health_results: list[tuple[str, dict[str, dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=len(args.endpoints)) as executor:
        futures = [
            executor.submit(health_catalog, endpoint, args.api_prefix)
            for endpoint in args.endpoints
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                health_results.append(result)
    if not health_results:
        raise RuntimeError("no inference service is available")
    catalogs = dict(health_results)
    work_assignments, assignments, tail_manifest = plan_inference_work(
        list(args.models),
        health_results,
        task_path=task_path,
        inference_dir=inference_dir,
        enable_tail_sharding=not args.disable_tail_sharding,
    )
    work_statuses: list[dict[str, Any]] = []
    active_work = {
        endpoint: items
        for endpoint, items in work_assignments.items()
        if items
    }
    with ThreadPoolExecutor(max_workers=len(active_work)) as executor:
        future_map = {
            executor.submit(
                run_service_queue,
                endpoint,
                items,
                catalogs[endpoint],
                args=args,
            ): endpoint
            for endpoint, items in active_work.items()
        }
        for future in as_completed(future_map):
            work_statuses.extend(future.result())
    consolidate_tail_predictions(inference_dir, tail_manifest)
    statuses = aggregate_model_statuses(
        list(args.models),
        work_statuses,
        inference_dir=inference_dir,
        expected_view_count=int(task_manifest["view_count"]),
    )
    merged_path, prediction_count = merge_predictions(
        inference_dir,
        list(args.models),
    )
    manifest = {
        "schema_version": "paper_v8_inference_manifest.v1",
        "created_at": v8.utc_now(),
        "task_manifest_sha256": v8.file_sha256(task_manifest_path),
        "models": list(args.models),
        "available_endpoints": sorted(catalogs),
        "assignments": assignments,
        "work_assignments": {
            endpoint: [
                {
                    "work_id": item.work_id,
                    "model_id": item.model_id,
                    "sample_path": str(item.sample_path),
                    "output_dir": str(item.output_dir),
                    "tail_part_index": item.tail_part_index,
                }
                for item in items
            ]
            for endpoint, items in work_assignments.items()
        },
        "tail_sharding": tail_manifest,
        "statuses": statuses,
        "predictions": {
            **v8.file_record(merged_path),
            "row_count": prediction_count,
        },
    }
    manifest["complete"] = all(
        row.get("status") == "complete" for row in statuses
    ) and len(statuses) == len(args.models)
    v8.write_json(inference_dir / "inference_manifest.json", manifest)
    print(
        v8.canonical_json(
            {
                "complete": manifest["complete"],
                "prediction_count": prediction_count,
                "assignments": assignments,
                "output": str(inference_dir),
            }
        )
    )
    if not manifest["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
