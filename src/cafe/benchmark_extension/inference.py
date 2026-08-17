from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.generation import GENERATION_SCHEMA, PIPELINE_SCHEMA
from cafe.benchmark_extension.validation import VALIDATION_SCHEMA
from cafe.inference.runner import (
    DEFAULT_ENDPOINTS,
    DEFAULT_MODELS,
    INPUT_ADAPTATION_POLICY_ID,
    MODEL_EXECUTION_CONFIG,
    TimerServiceClient,
    health_catalog,
    resolve_input_capability,
    run_one_model,
    safe_filename,
)


INFERENCE_SCHEMA = "cafe.benchmark_extension_inference.v2"
TASK_SCHEMA = "cafe.benchmark_extension_forecast_task.v2"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast native GIFT-Eval baselines and capability treatments."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def _maximum_context(model: dict[str, Any]) -> int | None:
    value = (model.get("forecast_limits") or {}).get("max_input_length")
    if value is None:
        return None
    parsed = int(value)
    return None if parsed < 0 else parsed


def model_task_row(sample: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(sample))
    source_context = int(row["context_length"])
    maximum = _maximum_context(model)
    context = source_context if maximum is None else min(source_context, maximum)
    minimum = int((model.get("forecast_limits") or {}).get("min_input_length") or 0)
    if context < minimum:
        raise ValueError(
            f"sample {row['sample_id']} has L{context}, below model minimum L{minimum}"
        )
    target = np.asarray(row["target"], dtype=float)
    covariates = (
        None
        if row.get("covariates") is None
        else np.asarray(row["covariates"], dtype=float)
    )
    start = source_context - context
    sliced = target[start:]
    row.update(
        {
            "schema_version": TASK_SCHEMA,
            "source_sample_schema_version": sample["schema_version"],
            "source_context_length": source_context,
            "context_length": context,
            "model_context_policy": (
                "truncate_authentic_treated_history_to_model_maximum_context"
                if context < source_context
                else "use_entire_authentic_treated_history"
            ),
            "target": sliced.tolist(),
            "covariates": (
                None if covariates is None else covariates[start:].tolist()
            ),
            "target_sha256": _array_sha256(sliced),
        }
    )
    return row


def iter_generation_samples(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for key in ("official_baselines", "capability_treatments", "input_ablations"):
        record = manifest["files"][key]
        path = Path(record["path"])
        if protocol.file_sha256(path) != record["sha256"]:
            raise ValueError(f"generation input hash mismatch: {path}")
        yield from protocol.iter_jsonl(path)


def prepare_model_tasks(
    samples: Iterable[dict[str, Any]],
    model: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    count = protocol.write_jsonl(
        output_path,
        (model_task_row(sample, model) for sample in samples),
    )
    return {
        **protocol.file_record(output_path),
        "row_count": count,
        "model_id": str(model["model_id"]),
        "resolved_input_capability": resolve_input_capability(model),
        "input_adaptation_policy": INPUT_ADAPTATION_POLICY_ID,
    }


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _validated_inputs(dataset_root: Path) -> tuple[dict[str, Any], Path, Path]:
    generation_manifest_path = dataset_root / "01_generation" / "manifest.json"
    validation_path = dataset_root / "02_validation" / "report.json"
    generation = protocol.read_json(generation_manifest_path)
    validation = protocol.read_json(validation_path)
    if generation.get("schema_version") != GENERATION_SCHEMA:
        raise ValueError("unsupported generation manifest")
    if generation.get("config", {}).get("pipeline_schema_version") != PIPELINE_SCHEMA:
        raise ValueError("generation is not current pipeline v6")
    if validation.get("schema_version") != VALIDATION_SCHEMA or not validation.get("accepted"):
        raise ValueError("generation validation is not accepted")
    if validation.get("generation_manifest_sha256") != protocol.file_sha256(
        generation_manifest_path
    ):
        raise ValueError("validation is not bound to generation manifest")
    return generation, generation_manifest_path, validation_path


def main() -> int:
    args = parse_args()
    if len(args.models) != len(set(args.models)):
        raise ValueError("models must be unique")
    missing = sorted(set(args.models) - set(MODEL_EXECUTION_CONFIG))
    if missing:
        raise ValueError("missing model execution configs: " + ", ".join(missing))
    dataset_root = args.output_root.resolve() / args.dataset_id
    generation, generation_manifest_path, validation_path = _validated_inputs(dataset_root)
    inference_dir = dataset_root / "03_inference"
    manifest_path = inference_dir / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"inference artifact already exists; use --resume or a new experiment: {manifest_path}"
        )
    inference_dir.mkdir(parents=True, exist_ok=True)
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
    health_results.sort(key=lambda item: item[0])
    task_records: dict[str, Any] = {}
    statuses: list[dict[str, Any]] = []
    for model_index, model_id in enumerate(args.models):
        candidates = [
            (endpoint, catalog)
            for endpoint, catalog in health_results
            if model_id in catalog
        ]
        if not candidates:
            raise ValueError(f"model {model_id!r} unavailable on all endpoints")
        endpoint, catalog = candidates[model_index % len(candidates)]
        model = catalog[model_id]
        task_path = inference_dir / "tasks" / f"{safe_filename(model_id)}.jsonl"
        if not task_path.exists() or not args.resume:
            task_records[model_id] = prepare_model_tasks(
                iter_generation_samples(generation),
                model,
                task_path,
            )
        else:
            task_records[model_id] = {
                **protocol.file_record(task_path),
                "row_count": sum(1 for _ in protocol.iter_jsonl(task_path)),
                "model_id": model_id,
                "resolved_input_capability": resolve_input_capability(model),
                "input_adaptation_policy": INPUT_ADAPTATION_POLICY_ID,
            }
        if args.prepare_only:
            continue
        model_root = inference_dir / "models" / safe_filename(model_id)
        for directory in ("predictions", "failures"):
            (model_root / directory).mkdir(parents=True, exist_ok=True)
        client = TimerServiceClient(endpoint, args.api_prefix, timeout_seconds=30)
        try:
            status = run_one_model(
                client,
                model,
                output_dir=model_root,
                execution=dict(MODEL_EXECUTION_CONFIG[model_id]),
                devices=args.devices,
                request_max_attempts=args.max_attempts,
                forecast_timeout_seconds=args.forecast_timeout_seconds,
                load_timeout_seconds=args.load_timeout_seconds,
                keep_loaded=False,
                sample_path=task_path,
                prediction_kind="benchmark_extension",
                status_filename="status.json",
                input_adaptation_policy=INPUT_ADAPTATION_POLICY_ID,
            )
        finally:
            client.close()
        status["endpoint"] = endpoint
        statuses.append(status)
        protocol.write_json(model_root / "status.json", status)
    config = {
        "pipeline_schema_version": PIPELINE_SCHEMA,
        "dataset_id": args.dataset_id,
        "models": list(args.models),
        "input_history_policy": (
            "treatment_applied_to_entire_official_history_then_model_max_context_suffix"
        ),
        "native_target_policy": (
            "native_if_supported_else_inference_only_independent_univariate_reassembly"
        ),
    }
    manifest = {
        "schema_version": INFERENCE_SCHEMA,
        "created_at": protocol.utc_now(),
        "dataset_id": args.dataset_id,
        "config": config,
        "config_sha256": protocol.json_sha256(config),
        "generation_manifest": {
            "path": str(generation_manifest_path),
            "sha256": protocol.file_sha256(generation_manifest_path),
        },
        "validation_report": {
            "path": str(validation_path),
            "sha256": protocol.file_sha256(validation_path),
        },
        "model_tasks": task_records,
        "model_statuses": statuses,
        "complete": bool(
            not args.prepare_only
            and len(statuses) == len(args.models)
            and all(status.get("status") == "complete" for status in statuses)
        ),
        "prepare_only": bool(args.prepare_only),
    }
    protocol.write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if args.prepare_only or manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
