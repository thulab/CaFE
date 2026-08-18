from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx
import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.generation import (
    GENERATION_SCHEMA,
    PIPELINE_SCHEMA,
    iter_replayed_samples,
)
from cafe.benchmark_extension.storage import (
    PredictionParquetWriter,
    parquet_file_record,
)
from cafe.benchmark_extension.validation import VALIDATION_SCHEMA
from cafe.inference.runner import (
    DEFAULT_ENDPOINTS,
    DEFAULT_MODELS,
    INPUT_ADAPTATION_POLICY_ID,
    MODEL_EXECUTION_CONFIG,
    TimerServiceClient,
    _forecast_bulk_with_retry,
    adapted_request_samples,
    health_catalog,
    input_adaptation_plan,
    request_group_key,
    resolve_input_capability,
    safe_filename,
)


INFERENCE_SCHEMA = "cafe.benchmark_extension_inference.v4"
TASK_SCHEMA = "cafe.benchmark_extension_forecast_task.v3"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"
SERVICE_DEVICE_BATCH_INPUT_TOKENS = (11520 + 10000) * 50 + 11520
DEFAULT_MAX_REQUEST_INPUT_TOKENS = round(
    SERVICE_DEVICE_BATCH_INPUT_TOKENS * 0.75
)
DEFAULT_CLIENT_INFLIGHT_INPUT_TOKENS = DEFAULT_MAX_REQUEST_INPUT_TOKENS * 8
MODEL_INPUT_TOKEN_CONFIG = {
    model_id: {
        "maximum_request_input_tokens": DEFAULT_MAX_REQUEST_INPUT_TOKENS,
        "client_inflight_input_tokens": DEFAULT_CLIENT_INFLIGHT_INPUT_TOKENS,
        "native_multivariate_input_token_multiplier": 1.0,
        "maximum_bulk_rows": 64,
        "native_multivariate_maximum_bulk_rows": 64,
    }
    for model_id in MODEL_EXECUTION_CONFIG
}
MODEL_INPUT_TOKEN_CONFIG["Timer-4.0"].update(
    {
        "client_inflight_input_tokens": (
            DEFAULT_CLIENT_INFLIGHT_INPUT_TOKENS * 2
        ),
        "native_multivariate_input_token_multiplier": 2.0,
    }
)
MODEL_INPUT_TOKEN_CONFIG["Chronos-2"].update({"http_concurrency": 32})
MODEL_INPUT_TOKEN_CONFIG["timesfm2.5"].update(
    {
        "very_long_context_threshold": 8192,
        "very_long_context_input_token_multiplier": 2.0,
    }
)
MODEL_INPUT_TOKEN_CONFIG["tirex2"].update(
    {
        "http_concurrency": 32,
        "native_multivariate_http_concurrency": 2,
        "native_multivariate_maximum_bulk_rows": 8,
        "large_panel_context_threshold": 512,
        "large_panel_target_dim_threshold": 16,
        "large_panel_http_concurrency": 8,
    }
)
MODEL_INPUT_TOKEN_CONFIG["moirai2"].update(
    {
        "http_concurrency": 32,
        "client_inflight_input_tokens": (
            DEFAULT_CLIENT_INFLIGHT_INPUT_TOKENS * 2
        ),
    }
)
MODEL_INPUT_TOKEN_CONFIG["Timer-3.5"].update({"http_concurrency": 32})
MODEL_INPUT_TOKEN_CONFIG["toto2.0"].update(
    {
        "http_concurrency": 8,
        "maximum_request_input_tokens": 128 * 1024,
        "client_inflight_input_tokens": 1024 * 1024,
        "maximum_bulk_rows": 256,
        "native_multivariate_maximum_bulk_rows": 256,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast native GIFT-Eval baselines and capability treatments."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--execute-models", nargs="+", default=None)
    parser.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=protocol.REPO_ROOT / "data" / "gift-eval",
    )
    parser.add_argument("--max-open-shape-groups", type=int, default=64)
    parser.add_argument("--max-inflight-batches", type=int, default=8)
    parser.add_argument("--max-inflight-mib", type=int, default=2048)
    parser.add_argument("--max-request-input-tokens", type=int, default=None)
    parser.add_argument("--client-inflight-input-tokens", type=int, default=None)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--reuse-loaded-model", action="store_true")
    parser.add_argument("--preserve-loaded-model", action="store_true")
    return parser.parse_args()


def _maximum_context(model: dict[str, Any]) -> int | None:
    value = (model.get("forecast_limits") or {}).get("max_input_length")
    if value is None:
        return None
    parsed = int(value)
    return None if parsed < 0 else parsed


def model_task_row(sample: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    row = dict(sample)
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
            "target": sliced,
            "covariates": (
                None if covariates is None else covariates[start:]
            ),
            "target_sha256": _array_sha256(sliced),
        }
    )
    return row


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
        raise ValueError("generation is not current pipeline v7")
    if validation.get("schema_version") != VALIDATION_SCHEMA or not validation.get("accepted"):
        raise ValueError("generation validation is not accepted")
    if validation.get("generation_manifest_sha256") != protocol.file_sha256(
        generation_manifest_path
    ):
        raise ValueError("validation is not bound to generation manifest")
    return generation, generation_manifest_path, validation_path


def _iter_model_bulk_batches(
    samples: Iterable[dict[str, Any]],
    *,
    model: dict[str, Any],
    execution: dict[str, Any],
    maximum_open_groups: int,
    maximum_buffered_bytes: int,
    summary: dict[str, int],
) -> Iterator[tuple[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]]]:
    """Bound shape buffers while preserving homogeneous service requests."""

    buffers: OrderedDict[
        tuple[Any, ...],
        dict[str, Any],
    ] = OrderedDict()
    current_shard: int | None = None
    configured_batch_size = int(execution["task_batch_size"])
    task_batch_size = min(
        configured_batch_size,
        int(execution.get("maximum_bulk_rows", configured_batch_size)),
    )
    maximum_request_input_tokens = int(
        execution.get(
            "maximum_request_input_tokens",
            DEFAULT_MAX_REQUEST_INPUT_TOKENS,
        )
    )
    for dense_sample in samples:
        source_shard = int(dense_sample.get("source_shard_index", 0))
        if current_shard is None:
            current_shard = source_shard
        elif source_shard != current_shard:
            for key, state in buffers.items():
                if state["items"]:
                    yield current_shard, key, state["items"]
            buffers.clear()
            current_shard = source_shard
        summary["expected_original_view_count"] += 1
        sample = model_task_row(dense_sample, model)
        plan = input_adaptation_plan(
            model,
            sample,
            policy_id=INPUT_ADAPTATION_POLICY_ID,
        )
        if plan is None:
            summary["unsupported_window_view_count"] += 1
            continue
        summary["compatible_sample_count"] += 1
        summary["expected_http_request_count"] += int(plan["target_request_count"])
        summary[
            "adapted_view_count" if plan["adapted"] else "native_view_count"
        ] += 1
        if plan["target_mode"] == "independent_univariate":
            summary["split_target_view_count"] += 1
        if plan["covariate_mode"] == "omitted_unsupported":
            summary["covariates_omitted_view_count"] += 1
        children = adapted_request_samples(sample, plan)
        key = request_group_key(sample, plan=plan)
        child_count = int(plan["target_request_count"])
        item = (sample, plan, children)
        item_input_tokens = _batch_input_tokens([item])
        item_batch_size = task_batch_size
        if int(plan["request_target_dim"]) > 1:
            item_batch_size = min(
                item_batch_size,
                int(
                    execution.get(
                        "native_multivariate_maximum_bulk_rows",
                        item_batch_size,
                    )
                ),
            )
        limit = max(
            1,
            min(
                item_batch_size // child_count,
                maximum_request_input_tokens // item_input_tokens,
            ),
        )
        item_bytes = _batch_bytes([item])
        per_group_budget = max(
            1,
            int(maximum_buffered_bytes) // max(1, int(maximum_open_groups)),
        )
        state = buffers.setdefault(
            key, {"limit": limit, "items": [], "bytes": 0}
        )
        buffers.move_to_end(key)
        if state["items"] and int(state["bytes"]) + item_bytes > per_group_budget:
            yield current_shard, key, state["items"]
            state = {"limit": limit, "items": [], "bytes": 0}
            buffers[key] = state
        state["items"].append(item)
        state["bytes"] += item_bytes
        if len(state["items"]) >= int(state["limit"]):
            yield current_shard, key, state["items"]
            del buffers[key]
        while len(buffers) > max(1, int(maximum_open_groups)):
            old_key, old_state = buffers.popitem(last=False)
            yield current_shard, old_key, old_state["items"]
    for key, state in buffers.items():
        if state["items"]:
            yield int(current_shard or 0), key, state["items"]


def _adaptation_summary() -> dict[str, int]:
    return {
        "expected_original_view_count": 0,
        "compatible_sample_count": 0,
        "unsupported_window_view_count": 0,
        "native_view_count": 0,
        "adapted_view_count": 0,
        "split_target_view_count": 0,
        "covariates_omitted_view_count": 0,
        "expected_http_request_count": 0,
    }


def _next_batch(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _requested_execution_complete(
    execute_models: list[str],
    status_by_model: dict[str, dict[str, Any]],
    prediction_records: dict[str, Any],
) -> bool:
    """Report success for this invocation, not for the whole resumed manifest."""

    for model_id in execute_models:
        if status_by_model.get(model_id, {}).get("status") != "complete":
            return False
        record = prediction_records.get(model_id)
        if not isinstance(record, dict):
            return False
        if int(record.get("row_count", -1)) > 0 and not record.get("parts"):
            return False
    return True


def _batch_bytes(
    chunk: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
) -> int:
    total = 0
    for _sample, _plan, children in chunk:
        for child in children:
            total += int(np.asarray(child["target"], dtype=np.float32).nbytes)
            if child.get("covariates") is not None:
                total += int(
                    np.asarray(child["covariates"], dtype=np.float32).nbytes
                )
    return max(1, total)


def _child_input_tokens(child: dict[str, Any]) -> int:
    context = int(child["context_length"])
    horizon = int(child["horizon"])
    target_dim = int(child["target_dim"])
    covariate_dim = int(child["covariate_dim"])
    return max(
        1,
        context * target_dim
        + context * covariate_dim
        + horizon * covariate_dim,
    )


def _batch_input_tokens(
    chunk: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
) -> int:
    return max(
        1,
        sum(
            _child_input_tokens(child)
            for _sample, _plan, children in chunk
            for child in children
        ),
    )


class _InputTokenLimiter:
    """Bound an endpoint by request cost while allowing one oversized request."""

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.inflight = 0
        self.condition = asyncio.Condition()

    async def acquire(self, amount: int) -> None:
        amount = max(1, int(amount))
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.inflight == 0
                or self.inflight + amount <= self.capacity
            )
            self.inflight += amount

    async def release(self, amount: int) -> None:
        amount = max(1, int(amount))
        async with self.condition:
            self.inflight -= amount
            self.condition.notify_all()


async def _run_streaming_bulk_model(
    *,
    model_id: str,
    model: dict[str, Any],
    execution: dict[str, Any],
    endpoints: list[str],
    api_prefix: str,
    sample_factory: Any,
    prediction_dir: Path,
    failure_path: Path,
    forecast_timeout_seconds: int,
    max_attempts: int,
    maximum_open_groups: int,
    maximum_inflight_batches: int,
    maximum_inflight_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary = _adaptation_summary()
    batch_iterator = iter(
        _iter_model_bulk_batches(
            sample_factory(),
            model=model,
            execution=execution,
            maximum_open_groups=maximum_open_groups,
            maximum_buffered_bytes=max(1, int(maximum_inflight_bytes) // 2),
            summary=summary,
        )
    )
    concurrency = int(execution["http_concurrency"])
    client_inflight_input_tokens = int(
        execution.get(
            "client_inflight_input_tokens",
            DEFAULT_CLIENT_INFLIGHT_INPUT_TOKENS,
        )
    )
    native_multivariate_input_token_multiplier = float(
        execution.get("native_multivariate_input_token_multiplier", 1.0)
    )
    native_multivariate_http_concurrency = max(
        1,
        min(
            concurrency,
            int(
                execution.get(
                    "native_multivariate_http_concurrency",
                    concurrency,
                )
            ),
        ),
    )
    very_long_context_threshold = int(
        execution.get("very_long_context_threshold", -1)
    )
    very_long_context_input_token_multiplier = float(
        execution.get("very_long_context_input_token_multiplier", 1.0)
    )
    large_panel_context_threshold = int(
        execution.get("large_panel_context_threshold", -1)
    )
    large_panel_target_dim_threshold = int(
        execution.get("large_panel_target_dim_threshold", -1)
    )
    large_panel_http_concurrency = max(
        1,
        min(
            concurrency,
            int(
                execution.get(
                    "large_panel_http_concurrency",
                    native_multivariate_http_concurrency,
                )
            ),
        ),
    )
    queues = [
        asyncio.Queue(maxsize=max(1, int(maximum_inflight_batches)))
        for _endpoint in endpoints
    ]
    token_limiters = [
        _InputTokenLimiter(client_inflight_input_tokens)
        for _endpoint in endpoints
    ]
    native_multivariate_semaphores = [
        asyncio.Semaphore(native_multivariate_http_concurrency)
        for _endpoint in endpoints
    ]
    large_panel_semaphores = [
        asyncio.Semaphore(large_panel_http_concurrency)
        for _endpoint in endpoints
    ]
    group_stats: defaultdict[tuple[Any, ...], dict[str, int]] = defaultdict(
        lambda: {
            "logical_view_count": 0,
            "successful_view_count": 0,
            "failed_view_count": 0,
            "bulk_request_count": 0,
            "attempt_count": 0,
            "maximum_request_input_tokens": 0,
            "maximum_weighted_request_input_tokens": 0,
            "minimum_dynamic_concurrency_ceiling": concurrency,
        }
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    part_records: list[dict[str, Any]] = []
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_handle = failure_path.open("w", encoding="utf-8")
    endpoint_index = 0
    byte_condition = asyncio.Condition()
    inflight_bytes = 0

    async def acquire_bytes(amount: int) -> None:
        nonlocal inflight_bytes
        async with byte_condition:
            await byte_condition.wait_for(
                lambda: inflight_bytes == 0
                or inflight_bytes + amount <= max(1, int(maximum_inflight_bytes))
            )
            inflight_bytes += amount

    async def release_bytes(amount: int) -> None:
        nonlocal inflight_bytes
        async with byte_condition:
            inflight_bytes -= amount
            byte_condition.notify_all()

    async def producer() -> None:
        nonlocal endpoint_index
        current_shard: int | None = None
        current_writer: PredictionParquetWriter | None = None

        async def finish_shard() -> None:
            nonlocal current_writer
            if current_writer is None or current_shard is None:
                return
            for queue in queues:
                await queue.join()
            row_count = current_writer.close()
            part_records.append(
                {
                    **parquet_file_record(current_writer.path, row_count=row_count),
                    "source_shard_index": int(current_shard),
                }
            )
            current_writer = None

        while True:
            batch = await asyncio.to_thread(_next_batch, batch_iterator)
            if batch is None:
                break
            source_shard, group_key, chunk = batch
            if current_shard is None or int(source_shard) != int(current_shard):
                await finish_shard()
                current_shard = int(source_shard)
                current_writer = PredictionParquetWriter(
                    prediction_dir / f"part_{current_shard:06d}.parquet"
                )
            queue = queues[endpoint_index % len(queues)]
            endpoint_index += 1
            payload_bytes = _batch_bytes(chunk)
            await acquire_bytes(payload_bytes)
            await queue.put((group_key, chunk, current_writer, payload_bytes))
        await finish_shard()
        for queue in queues:
            for _worker in range(concurrency):
                await queue.put(None)

    async def endpoint_workers(
        endpoint: str,
        queue: asyncio.Queue[Any],
        token_limiter: _InputTokenLimiter,
        native_multivariate_semaphore: asyncio.Semaphore,
        large_panel_semaphore: asyncio.Semaphore,
    ) -> None:
        limits = httpx.Limits(
            max_connections=concurrency,
            max_keepalive_connections=concurrency,
            keepalive_expiry=120.0,
        )
        forecast_url = endpoint.rstrip("/") + api_prefix + "/forecast/bulk"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(forecast_timeout_seconds),
            limits=limits,
            trust_env=False,
        ) as async_client:

            async def worker() -> None:
                while True:
                    job = await queue.get()
                    try:
                        if job is None:
                            return
                        group_key, chunk, writer, payload_bytes = job
                        stats = group_stats[group_key]
                        stats["logical_view_count"] += len(chunk)
                        children = [
                            child
                            for _sample, _plan, child_rows in chunk
                            for child in child_rows
                        ]
                        request_input_tokens = sum(
                            _child_input_tokens(child) for child in children
                        )
                        native_target_dim = max(
                            int(child["target_dim"]) for child in children
                        )
                        weighted_input_tokens = round(
                            request_input_tokens
                            * (
                                native_multivariate_input_token_multiplier
                                if native_target_dim > 1
                                else 1.0
                            )
                        )
                        request_context = max(
                            int(child["context_length"]) for child in children
                        )
                        if (
                            very_long_context_threshold > 0
                            and request_context > very_long_context_threshold
                        ):
                            weighted_input_tokens = round(
                                weighted_input_tokens
                                * very_long_context_input_token_multiplier
                            )
                        stats["maximum_request_input_tokens"] = max(
                            stats["maximum_request_input_tokens"],
                            request_input_tokens,
                        )
                        stats["maximum_weighted_request_input_tokens"] = max(
                            stats["maximum_weighted_request_input_tokens"],
                            weighted_input_tokens,
                        )
                        stats["minimum_dynamic_concurrency_ceiling"] = min(
                            stats["minimum_dynamic_concurrency_ceiling"],
                            max(
                                1,
                                client_inflight_input_tokens
                                // max(1, weighted_input_tokens),
                            ),
                        )
                        await token_limiter.acquire(weighted_input_tokens)
                        native_multivariate_acquired = False
                        selected_panel_semaphore = native_multivariate_semaphore
                        try:
                            if native_target_dim > 1:
                                if (
                                    large_panel_context_threshold > 0
                                    and large_panel_target_dim_threshold > 1
                                    and request_context
                                    > large_panel_context_threshold
                                    and native_target_dim
                                    >= large_panel_target_dim_threshold
                                ):
                                    selected_panel_semaphore = large_panel_semaphore
                                await selected_panel_semaphore.acquire()
                                native_multivariate_acquired = True
                            result = await _forecast_bulk_with_retry(
                                async_client,
                                forecast_url=forecast_url,
                                model_id=model_id,
                                children=children,
                                max_attempts=max_attempts,
                            )
                        finally:
                            if native_multivariate_acquired:
                                selected_panel_semaphore.release()
                            await token_limiter.release(weighted_input_tokens)
                        stats["attempt_count"] += int(result["attempts"])
                        forecasts = result["forecasts"]
                        if forecasts is None:
                            stats["failed_view_count"] += len(chunk)
                            for sample, plan, _children in chunk:
                                failure_handle.write(
                                    protocol.canonical_json(
                                        {
                                            "model_id": model_id,
                                            "sample_id": sample["sample_id"],
                                            "endpoint": endpoint,
                                            "error": result["error"],
                                            "attempts": result["attempts"],
                                            "input_adaptation": plan,
                                        }
                                    )
                                    + "\n"
                                )
                            continue
                        stats["bulk_request_count"] += 1
                        child_offset = 0
                        prediction_rows = []
                        for sample, plan, child_rows in chunk:
                            selected = forecasts[
                                child_offset : child_offset + len(child_rows)
                            ]
                            child_offset += len(child_rows)
                            if plan["target_mode"] == "independent_univariate":
                                forecast = np.concatenate(
                                    [value.T for value in selected], axis=1
                                )
                            else:
                                forecast = selected[0].T
                            prediction_rows.append(
                                {
                                    "model_id": model_id,
                                    "sample_id": str(sample["sample_id"]),
                                    "forecast": forecast,
                                    "input_adaptation": plan,
                                }
                            )
                            stats["successful_view_count"] += 1
                        writer.write_batch(prediction_rows)
                    finally:
                        if job is not None:
                            await release_bytes(payload_bytes)
                        queue.task_done()

            await asyncio.gather(*(worker() for _index in range(concurrency)))

    try:
        await asyncio.gather(
            producer(),
            *(
                endpoint_workers(
                    endpoint,
                    queue,
                    token_limiter,
                    native_multivariate_semaphore,
                    large_panel_semaphore,
                )
                for (
                    endpoint,
                    queue,
                    token_limiter,
                    native_multivariate_semaphore,
                    large_panel_semaphore,
                ) in zip(
                    endpoints,
                    queues,
                    token_limiters,
                    native_multivariate_semaphores,
                    large_panel_semaphores,
                    strict=True,
                )
            ),
        )
        prediction_count = sum(int(record["row_count"]) for record in part_records)
        failure_handle.flush()
        failure_count = sum(row["failed_view_count"] for row in group_stats.values())
    except Exception:
        for path in prediction_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        raise
    finally:
        failure_handle.close()
    stats = {
        "prediction_count": prediction_count,
        "failure_count": failure_count,
        "endpoint_count": len(endpoints),
        "http_concurrency_per_endpoint": concurrency,
        "bulk_request_count": sum(row["bulk_request_count"] for row in group_stats.values()),
        "attempt_count": sum(row["attempt_count"] for row in group_stats.values()),
        "shape_group_count": len(group_stats),
        "maximum_inflight_bytes": int(maximum_inflight_bytes),
        "maximum_request_input_tokens": int(
            execution.get(
                "maximum_request_input_tokens",
                DEFAULT_MAX_REQUEST_INPUT_TOKENS,
            )
        ),
        "client_inflight_input_tokens_per_endpoint": (
            client_inflight_input_tokens
        ),
        "native_multivariate_input_token_multiplier": (
            native_multivariate_input_token_multiplier
        ),
        "native_multivariate_http_concurrency": (
            native_multivariate_http_concurrency
        ),
        "very_long_context_threshold": very_long_context_threshold,
        "very_long_context_input_token_multiplier": (
            very_long_context_input_token_multiplier
        ),
        "large_panel_context_threshold": large_panel_context_threshold,
        "large_panel_target_dim_threshold": large_panel_target_dim_threshold,
        "large_panel_http_concurrency": large_panel_http_concurrency,
        "shape_group_token_stats": [
            {
                "request_group": list(group_key),
                "maximum_request_input_tokens": int(
                    row["maximum_request_input_tokens"]
                ),
                "maximum_weighted_request_input_tokens": int(
                    row["maximum_weighted_request_input_tokens"]
                ),
                "minimum_dynamic_concurrency_ceiling": int(
                    row["minimum_dynamic_concurrency_ceiling"]
                ),
            }
            for group_key, row in sorted(
                group_stats.items(), key=lambda item: tuple(map(str, item[0]))
            )
        ],
    }
    return summary, stats, part_records


def run_streaming_model(
    *,
    model_id: str,
    model: dict[str, Any],
    execution: dict[str, Any],
    endpoints: list[str],
    api_prefix: str,
    devices: str,
    sample_factory: Any,
    prediction_dir: Path,
    failure_path: Path,
    load_timeout_seconds: int,
    forecast_timeout_seconds: int,
    max_attempts: int,
    maximum_open_groups: int,
    maximum_inflight_batches: int,
    maximum_inflight_bytes: int,
    unload_before_load: bool,
    unload_after: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    clients = [TimerServiceClient(endpoint, api_prefix, timeout_seconds=30) for endpoint in endpoints]

    def load(client: TimerServiceClient) -> tuple[float, dict[str, Any]]:
        if unload_before_load:
            client.unload_all_loaded()
        return client.ensure_loaded(
            model_id,
            devices=devices,
            replicas_per_device=int(execution["replicas_per_device"]),
            timeout_seconds=load_timeout_seconds,
        )

    try:
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            loaded = list(executor.map(load, clients))
        summary, request_stats, part_records = asyncio.run(
            _run_streaming_bulk_model(
                model_id=model_id,
                model=model,
                execution=execution,
                endpoints=endpoints,
                api_prefix=api_prefix,
                sample_factory=sample_factory,
                prediction_dir=prediction_dir,
                failure_path=failure_path,
                forecast_timeout_seconds=forecast_timeout_seconds,
                max_attempts=max_attempts,
                maximum_open_groups=maximum_open_groups,
                maximum_inflight_batches=maximum_inflight_batches,
                maximum_inflight_bytes=maximum_inflight_bytes,
            )
        )
    finally:
        if unload_after:
            with ThreadPoolExecutor(max_workers=len(clients)) as executor:
                list(executor.map(lambda client: client.unload_model(model_id), clients))
        for client in clients:
            client.close()
    complete = (
        request_stats["failure_count"] == 0
        and request_stats["prediction_count"] == summary["compatible_sample_count"]
    )
    return {
        "model_id": model_id,
        "status": "complete" if complete else "incomplete",
        **summary,
        **request_stats,
        "endpoints": endpoints,
        "loaded_topologies": [topology for _seconds, topology in loaded],
        "load_seconds_max": max((seconds for seconds, _topology in loaded), default=0.0),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "prediction_parts": part_records,
    }


def main() -> int:
    args = parse_args()
    if args.max_request_input_tokens is not None and args.max_request_input_tokens < 1:
        raise ValueError("max-request-input-tokens must be positive")
    if (
        args.client_inflight_input_tokens is not None
        and args.client_inflight_input_tokens < 1
    ):
        raise ValueError("client-inflight-input-tokens must be positive")
    if len(args.models) != len(set(args.models)):
        raise ValueError("models must be unique")
    execute_models = list(args.execute_models or args.models)
    if len(execute_models) != len(set(execute_models)):
        raise ValueError("execute-models must be unique")
    if not set(execute_models).issubset(args.models):
        raise ValueError("execute-models must be a subset of models")
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
    previous_manifest = (
        protocol.read_json(manifest_path)
        if manifest_path.exists() and args.resume
        else None
    )
    prediction_records: dict[str, Any] = dict(
        (previous_manifest or {}).get("model_predictions") or {}
    )
    status_by_model = {
        str(row["model_id"]): row
        for row in (previous_manifest or {}).get("model_statuses", [])
    }
    for model_id in args.models:
        if model_id not in execute_models:
            continue
        candidates = [
            (endpoint, catalog)
            for endpoint, catalog in health_results
            if model_id in catalog
        ]
        if not candidates:
            raise ValueError(f"model {model_id!r} unavailable on all endpoints")
        endpoint_list = [endpoint for endpoint, _catalog in candidates]
        model = candidates[0][1][model_id]
        model_contract = protocol.canonical_json(
            {
                "forecast_limits": model.get("forecast_limits") or {},
                "input_capability": resolve_input_capability(model),
            }
        )
        if any(
            protocol.canonical_json(
                {
                    "forecast_limits": catalog[model_id].get("forecast_limits") or {},
                    "input_capability": resolve_input_capability(catalog[model_id]),
                }
            )
            != model_contract
            for _endpoint, catalog in candidates[1:]
        ):
            raise ValueError(
                f"model {model_id!r} advertises inconsistent limits across endpoints"
            )
        if args.prepare_only:
            continue
        previous_status = next(
            (
                row
                for row in (previous_manifest or {}).get("model_statuses", [])
                if row.get("model_id") == model_id and row.get("status") == "complete"
            ),
            None,
        )
        previous_record = (previous_manifest or {}).get("model_predictions", {}).get(
            model_id
        )
        if previous_status is not None and isinstance(previous_record, dict):
            parts = previous_record.get("parts") or []
            zero_row_complete = int(previous_record.get("row_count", -1)) == 0
            if (zero_row_complete or bool(parts)) and all(
                Path(str(record["path"])).is_file()
                and protocol.file_sha256(Path(str(record["path"]))) == record["sha256"]
                for record in parts
            ):
                status_by_model[model_id] = previous_status
                prediction_records[model_id] = previous_record
                continue
        model_root = inference_dir / "models" / safe_filename(model_id)
        prediction_dir = model_root / "predictions"
        failure_path = model_root / "failures" / f"{safe_filename(model_id)}.jsonl"
        execution = {
            **dict(MODEL_EXECUTION_CONFIG[model_id]),
            **dict(MODEL_INPUT_TOKEN_CONFIG[model_id]),
        }
        if args.max_request_input_tokens is not None:
            execution["maximum_request_input_tokens"] = int(
                args.max_request_input_tokens
            )
        if args.client_inflight_input_tokens is not None:
            execution["client_inflight_input_tokens"] = int(
                args.client_inflight_input_tokens
            )
        status = run_streaming_model(
            model_id=model_id,
            model=model,
            execution=execution,
            endpoints=endpoint_list,
            api_prefix=args.api_prefix,
            devices=args.devices,
            sample_factory=lambda: iter_replayed_samples(
                generation,
                gift_eval_dir=args.gift_eval_dir.resolve(),
                replay_workers=max(1, int(args.preprocess_workers)),
            ),
            prediction_dir=prediction_dir,
            failure_path=failure_path,
            load_timeout_seconds=args.load_timeout_seconds,
            forecast_timeout_seconds=args.forecast_timeout_seconds,
            max_attempts=args.max_attempts,
            maximum_open_groups=args.max_open_shape_groups,
            maximum_inflight_batches=args.max_inflight_batches,
            maximum_inflight_bytes=max(1, int(args.max_inflight_mib)) * 1024 * 1024,
            unload_before_load=not bool(args.reuse_loaded_model),
            unload_after=not bool(args.preserve_loaded_model),
        )
        status_by_model[model_id] = status
        prediction_records[model_id] = {
            "format": "partitioned_parquet",
            "compression": "zstd",
            "row_count": int(status["prediction_count"]),
            "parts": status.pop("prediction_parts"),
            "model_id": model_id,
            "resolved_input_capability": resolve_input_capability(model),
            "input_adaptation_policy": INPUT_ADAPTATION_POLICY_ID,
        }
        protocol.write_json(model_root / "status.json", status)
    statuses = [status_by_model[model_id] for model_id in args.models if model_id in status_by_model]
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
        "request_materialization": (
            "bounded_in_memory_source_contract_replay_no_model_task_artifact"
        ),
        "transport": "msgpack_bulk",
        "endpoint_policy": "model_major_all_compatible_endpoints",
        "bulk_request_input_token_policy": (
            "shape_aware_max_request_and_weighted_endpoint_inflight_v1"
        ),
        "default_maximum_request_input_tokens": (
            DEFAULT_MAX_REQUEST_INPUT_TOKENS
        ),
        "default_client_inflight_input_tokens_per_endpoint": (
            DEFAULT_CLIENT_INFLIGHT_INPUT_TOKENS
        ),
        "model_input_token_config": MODEL_INPUT_TOKEN_CONFIG,
        "max_request_input_tokens_override": args.max_request_input_tokens,
        "client_inflight_input_tokens_override": (
            args.client_inflight_input_tokens
        ),
        "overload_retry_policy": (
            "http_429_503_exponential_1s_2s_with_0_to_25pct_jitter"
        ),
        "max_open_shape_groups": int(args.max_open_shape_groups),
        "max_inflight_batches_per_endpoint": int(args.max_inflight_batches),
        "max_inflight_mib": int(args.max_inflight_mib),
        "preprocess_workers": int(args.preprocess_workers),
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
        "model_tasks": {},
        "model_predictions": prediction_records,
        "model_statuses": statuses,
        "complete": bool(
            not args.prepare_only
            and len(statuses) == len(args.models)
            and all(status.get("status") == "complete" for status in statuses)
        ),
        "prepare_only": bool(args.prepare_only),
    }
    protocol.write_json(manifest_path, manifest)
    invocation_complete = _requested_execution_complete(
        execute_models,
        status_by_model,
        prediction_records,
    )
    print(
        protocol.canonical_json(
            {
                "manifest_path": str(manifest_path),
                "executed_models": execute_models,
                "invocation_complete": invocation_complete,
                "experiment_complete": manifest["complete"],
            }
        )
    )
    return 0 if args.prepare_only or invocation_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
