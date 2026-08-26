from __future__ import annotations

import argparse
import socket
from pathlib import Path
from typing import Any

from cafe import core as protocol
from cafe.benchmark_extension.generation import iter_replayed_samples
from cafe.benchmark_extension.inference import (
    DEFAULT_CLIENT_INFLIGHT_INPUT_TOKENS,
    DEFAULT_MAX_REQUEST_INPUT_TOKENS,
    MODEL_INPUT_TOKEN_CONFIG,
    _maximum_context,
    _validate_distance_context_contract,
    _validate_forecast_limits,
    _validated_inputs,
    run_streaming_model,
)
from cafe.benchmark_extension.mechanisms import source_distance_model_max_contexts
from cafe.inference.runner import (
    INPUT_ADAPTATION_POLICY_ID,
    MODEL_EXECUTION_CONFIG,
    health_catalog,
    resolve_input_capability,
)


WORKER_STATUS_SCHEMA = "cafe.distributed_inference_worker.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic source-shard partition near an endpoint."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gift-eval-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--part-index", type=int, required=True)
    parser.add_argument("--part-count", type=int, required=True)
    parser.add_argument("--worker-output-dir", type=Path, required=True)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--max-open-shape-groups", type=int, default=64)
    parser.add_argument("--max-inflight-batches", type=int, default=8)
    parser.add_argument("--max-inflight-mib", type=int, default=2048)
    parser.add_argument("--max-request-input-tokens", type=int, default=None)
    parser.add_argument("--client-inflight-input-tokens", type=int, default=None)
    parser.add_argument("--load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--reuse-loaded-model", action="store_true")
    parser.add_argument("--preserve-loaded-model", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _completed_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    status = protocol.read_json(path)
    if (
        status.get("schema_version") != WORKER_STATUS_SCHEMA
        or status.get("status") != "complete"
    ):
        return None
    for record in status.get("prediction_parts") or []:
        part = Path(str(record["path"]))
        if not part.is_file() or protocol.file_sha256(part) != record["sha256"]:
            return None
    return status


def _validate_model_protocol(
    generation: dict[str, Any],
    model_id: str,
    model: dict[str, Any],
) -> None:
    config = generation.get("config") or {}
    raw_contexts = config.get("source_distance_configuration", {}).get(
        "model_max_contexts"
    )
    if not isinstance(raw_contexts, dict) or not raw_contexts:
        raise ValueError("generation source-distance model protocol is inconsistent")
    expected_contexts = {
        str(key): int(value) for key, value in raw_contexts.items()
    }
    if str(config.get("benchmark_id") or "gift_eval") == "gift_eval":
        term_contexts = source_distance_model_max_contexts(str(config.get("term")))
        if any(
            model_id not in term_contexts
            or maximum != int(term_contexts[model_id])
            for model_id, maximum in expected_contexts.items()
        ):
            raise ValueError(
                "generation source-distance model protocol is inconsistent"
            )
    _validate_distance_context_contract(model_id, model, expected_contexts)
    _validate_forecast_limits(model_id, model, generation)


def _relocate_generation_files(
    generation: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    """Point replay at hash-identical files synchronized beside the manifest."""

    relocated: dict[str, dict[str, Any]] = {}
    generation_dir = dataset_root / "01_generation"
    for name, value in (generation.get("files") or {}).items():
        record = dict(value)
        candidate = generation_dir / Path(str(record["path"])).name
        if (
            not candidate.is_file()
            or protocol.file_sha256(candidate) != str(record["sha256"])
        ):
            raise ValueError(f"synchronized generation file is invalid: {candidate}")
        record["path"] = str(candidate.resolve())
        relocated[str(name)] = record
    if not relocated:
        raise ValueError("generation manifest has no replay files")
    return {**generation, "files": relocated}


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.part_count < 1 or not 0 <= args.part_index < args.part_count:
        raise ValueError("invalid distributed worker partition")
    if args.preprocess_workers < 1:
        raise ValueError("preprocess-workers must be positive")
    worker_dir = args.worker_output_dir.resolve()
    status_path = worker_dir / "status.json"
    if args.resume:
        completed = _completed_status(status_path)
        if completed is not None:
            return completed

    dataset_root = args.output_root.resolve() / args.dataset_id
    generation, _generation_path, _validation_path = _validated_inputs(dataset_root)
    generation = _relocate_generation_files(generation, dataset_root)
    health = health_catalog(args.endpoint, args.api_prefix)
    if health is None or args.model_id not in health[1]:
        raise RuntimeError(
            f"model {args.model_id!r} is unavailable at {args.endpoint!r}"
        )
    model = health[1][args.model_id]
    _validate_model_protocol(generation, args.model_id, model)
    execution = {
        **dict(MODEL_EXECUTION_CONFIG[args.model_id]),
        **dict(MODEL_INPUT_TOKEN_CONFIG[args.model_id]),
    }
    if args.max_request_input_tokens is not None:
        execution["maximum_request_input_tokens"] = int(
            args.max_request_input_tokens
        )
    if args.client_inflight_input_tokens is not None:
        execution["client_inflight_input_tokens"] = int(
            args.client_inflight_input_tokens
        )

    prediction_dir = worker_dir / "predictions"
    failure_path = worker_dir / "failures.jsonl"
    status = run_streaming_model(
        model_id=args.model_id,
        model=model,
        execution=execution,
        endpoints=[args.endpoint],
        api_prefix=args.api_prefix,
        devices=args.devices,
        sample_factory=lambda: iter_replayed_samples(
            generation,
            gift_eval_dir=args.gift_eval_dir.resolve(),
            replay_workers=max(1, int(args.preprocess_workers)),
            source_shard_count=int(args.part_count),
            source_shard_index=int(args.part_index),
            maximum_context=_maximum_context(model),
        ),
        prediction_dir=prediction_dir,
        failure_path=failure_path,
        load_timeout_seconds=int(args.load_timeout_seconds),
        forecast_timeout_seconds=int(args.forecast_timeout_seconds),
        max_attempts=int(args.max_attempts),
        maximum_open_groups=int(args.max_open_shape_groups),
        maximum_inflight_batches=int(args.max_inflight_batches),
        maximum_inflight_bytes=max(1, int(args.max_inflight_mib)) * 1024 * 1024,
        unload_before_load=not bool(args.reuse_loaded_model),
        unload_after=not bool(args.preserve_loaded_model),
    )
    status.update(
        {
            "schema_version": WORKER_STATUS_SCHEMA,
            "dataset_id": args.dataset_id,
            "worker_hostname": socket.gethostname(),
            "service_endpoint": args.endpoint,
            "source_shard_partition": {
                "part_index": int(args.part_index),
                "part_count": int(args.part_count),
                "policy": "source_shard_index_modulo_worker_count_v1",
            },
            "maximum_context_materialization": _maximum_context(model),
            "resolved_input_capability": resolve_input_capability(model),
            "input_adaptation_policy": INPUT_ADAPTATION_POLICY_ID,
            "maximum_request_input_tokens": int(
                execution.get(
                    "maximum_request_input_tokens",
                    DEFAULT_MAX_REQUEST_INPUT_TOKENS,
                )
            ),
            "client_inflight_input_tokens": int(
                execution.get(
                    "client_inflight_input_tokens",
                    DEFAULT_CLIENT_INFLIGHT_INPUT_TOKENS,
                )
            ),
        }
    )
    worker_dir.mkdir(parents=True, exist_ok=True)
    protocol.write_json(status_path, status)
    return status


def main() -> int:
    args = parse_args()
    status = run_worker(args)
    print(protocol.canonical_json(status))
    return 0 if status.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
