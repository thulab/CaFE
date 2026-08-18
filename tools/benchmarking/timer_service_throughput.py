#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import msgpack
import numpy as np

from cafe.inference.runner import (
    MODEL_EXECUTION_CONFIG,
    TimerServiceClient,
    _bulk_request_content,
)


DEFAULT_MODELS = (
    "Timer-4.0",
    "Chronos-2",
    "timesfm2.5",
    "tirex2",
    "moirai2",
    "Timer-3.5",
    "toto2.0",
)
DEFAULT_CONTEXTS = (96, 512, 2048, 8192)
DEFAULT_TARGET_DIMS = (1, 7, 21, 64, 137)
DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32)
DEVICE_BATCH_TOKEN_BUDGET = (11520 + 10000) * 50 + 11520
TARGET_REQUEST_TOKEN_FRACTION = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Timer REST Service bulk throughput on one endpoint."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument(
        "--replicas-per-device",
        type=int,
        default=None,
        help=(
            "Override the model execution preset when comparing worker replicas."
        ),
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=list(DEFAULT_CONTEXTS)
    )
    parser.add_argument(
        "--target-dims", nargs="+", type=int, default=list(DEFAULT_TARGET_DIMS)
    )
    parser.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=list(DEFAULT_CONCURRENCIES),
    )
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--requests-per-case", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--maximum-bulk-rows", type=int, default=None)
    parser.add_argument(
        "--maximum-request-input-tokens",
        type=int,
        default=None,
        help="Optional per-request input-value ceiling used to size bulk rows.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preserve-loaded-model", action="store_true")
    parser.add_argument(
        "--exact-contexts",
        action="store_true",
        help="Do not automatically add the model maximum context length.",
    )
    return parser.parse_args()


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _supports_native_dimension(model: dict[str, Any], target_dim: int) -> bool:
    limits = model.get("forecast_limits") or {}
    mode = limits.get("input_mode") or {}
    maximum = int(mode.get("max_target_count", -1))
    maximum_group_rows = int(limits.get("max_group_rows") or -1)
    return (
        (maximum < 0 or target_dim <= maximum)
        and (
            maximum_group_rows < 0
            or target_dim <= maximum_group_rows
        )
    )


def _case_covariate_dimension(
    model: dict[str, Any], target_dim: int
) -> int:
    covariate_dim = _covariate_dimension(model)
    maximum_group_rows = int(
        (model.get("forecast_limits") or {}).get("max_group_rows") or -1
    )
    if (
        maximum_group_rows > 0
        and target_dim + covariate_dim > maximum_group_rows
    ):
        return 0
    return covariate_dim


def _covariate_dimension(model: dict[str, Any]) -> int:
    mode = (model.get("forecast_limits") or {}).get("input_mode") or {}
    if not bool(mode.get("supports_future_covariates", False)):
        return 0
    maximum = int(mode.get("max_history_covariate_count", -1))
    return 2 if maximum < 0 else min(2, maximum)


def _contexts(
    model: dict[str, Any],
    requested: list[int],
    *,
    include_model_maximum: bool = True,
) -> list[int]:
    limits = model.get("forecast_limits") or {}
    maximum = int(limits.get("max_input_length", -1))
    values = set(int(value) for value in requested if value > 0)
    if maximum > 0:
        if include_model_maximum:
            values.add(maximum)
        values = {min(value, maximum) for value in values}
    return sorted(values)


def _batch_rows(
    model_id: str,
    model: dict[str, Any],
    *,
    context: int,
    target_dim: int,
    covariate_dim: int,
    horizon: int,
    maximum_bulk_rows: int | None,
    maximum_request_input_tokens: int | None,
) -> tuple[int, int]:
    per_row_tokens = context * (target_dim + covariate_dim) + horizon * covariate_dim
    configured = int(MODEL_EXECUTION_CONFIG[model_id]["task_batch_size"])
    row_limit = (
        configured
        if maximum_bulk_rows is None
        else min(configured, max(1, int(maximum_bulk_rows)))
    )
    request_token_budget = (
        DEVICE_BATCH_TOKEN_BUDGET * TARGET_REQUEST_TOKEN_FRACTION
        if maximum_request_input_tokens is None
        else max(1, int(maximum_request_input_tokens))
    )
    token_limit = max(
        1,
        math.floor(
            request_token_budget / max(1, per_row_tokens)
        ),
    )
    rows = max(1, min(row_limit, token_limit))
    return rows, per_row_tokens * rows


def _children(
    *,
    batch_rows: int,
    context: int,
    horizon: int,
    target_dim: int,
    covariate_dim: int,
) -> list[dict[str, Any]]:
    time_index = np.arange(context + horizon, dtype=np.float32)
    children: list[dict[str, Any]] = []
    for row_index in range(batch_rows):
        target = np.stack(
            [
                np.sin(time_index / (7.0 + column % 13))
                + 0.01 * time_index / max(1, context)
                + 0.001 * (row_index + column)
                for column in range(target_dim)
            ],
            axis=1,
        ).astype(np.float32)
        covariates = None
        if covariate_dim:
            covariates = np.stack(
                [
                    np.cos(time_index / (11.0 + column * 3))
                    for column in range(covariate_dim)
                ],
                axis=1,
            ).astype(np.float32)
        children.append(
            {
                "context_length": context,
                "horizon": horizon,
                "target_dim": target_dim,
                "covariate_dim": covariate_dim,
                "target": target,
                "covariates": covariates,
            }
        )
    return children


async def _run_case(
    *,
    endpoint: str,
    api_prefix: str,
    model_id: str,
    children: list[dict[str, Any]],
    concurrency: int,
    request_count: int,
    warmup_requests: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    content, expected_shape, horizon = _bulk_request_content(model_id, children)
    url = endpoint.rstrip("/") + "/" + api_prefix.strip("/") + "/forecast/bulk"
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
        keepalive_expiry=120.0,
    )
    semaphore = asyncio.Semaphore(concurrency)
    statuses: Counter[int | str] = Counter()
    latencies: list[float] = []

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds), limits=limits, trust_env=False
    ) as client:
        async def request_once() -> None:
            async with semaphore:
                started = time.monotonic()
                try:
                    response = await client.post(
                        url,
                        content=content,
                        headers={"Content-Type": "application/msgpack"},
                    )
                    elapsed = time.monotonic() - started
                    statuses[response.status_code] += 1
                    if response.status_code == 200:
                        payload = msgpack.unpackb(response.content, raw=False)
                        observed = tuple(int(value) for value in payload["shape"])
                        expected = (*expected_shape[:2], horizon)
                        if observed != expected:
                            raise ValueError(
                                f"forecast shape {observed} != expected {expected}"
                            )
                        latencies.append(elapsed)
                except Exception as error:  # benchmark records transport failures
                    statuses[type(error).__name__] += 1

        warmup_count = max(1, int(warmup_requests))
        for _index in range(warmup_count):
            await request_once()
        statuses.clear()
        latencies.clear()
        started = time.monotonic()
        await asyncio.gather(*(request_once() for _index in range(request_count)))
        elapsed = time.monotonic() - started

    successes = int(statuses.get(200, 0))
    batch_rows = len(children)
    target_dim = int(children[0]["target_dim"])
    return {
        "request_count": request_count,
        "status_counts": {str(key): value for key, value in sorted(statuses.items(), key=lambda x: str(x[0]))},
        "success_count": successes,
        "failure_count": request_count - successes,
        "elapsed_seconds": elapsed,
        "request_per_second": successes / elapsed if elapsed else 0.0,
        "view_per_second": successes * batch_rows / elapsed if elapsed else 0.0,
        "target_series_per_second": (
            successes * batch_rows * target_dim / elapsed if elapsed else 0.0
        ),
        "latency_p50_seconds": _quantile(latencies, 0.50),
        "latency_p95_seconds": _quantile(latencies, 0.95),
    }


def _case_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["model_id"],
        row["context_length"],
        row["target_dim"],
        row["covariate_dim"],
        row["batch_rows"],
        row["concurrency"],
        row.get("replicas_per_device", 1),
    )


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[Any, ...]] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                completed.add(_case_key(json.loads(line)))

    control = TimerServiceClient(
        args.endpoint, args.api_prefix, timeout_seconds=30
    )
    try:
        catalog = {str(row["model_id"]): row for row in control.list_models()}
        with args.output.open("a", encoding="utf-8") as output:
            for model_id in args.models:
                model = catalog[model_id]
                control.unload_all_loaded()
                replicas_per_device = int(
                    args.replicas_per_device
                    if args.replicas_per_device is not None
                    else MODEL_EXECUTION_CONFIG[model_id]["replicas_per_device"]
                )
                if replicas_per_device <= 0:
                    raise ValueError("replicas_per_device must be positive")
                load_seconds, topology = control.ensure_loaded(
                    model_id,
                    devices=args.devices,
                    replicas_per_device=replicas_per_device,
                    timeout_seconds=args.load_timeout_seconds,
                )
                dimensions = [
                    value
                    for value in args.target_dims
                    if _supports_native_dimension(model, value)
                ]
                for context in _contexts(
                    model,
                    list(args.contexts),
                    include_model_maximum=not args.exact_contexts,
                ):
                    for target_dim in dimensions:
                        covariate_dim = _case_covariate_dimension(
                            model, target_dim
                        )
                        batch_rows, request_tokens = _batch_rows(
                            model_id,
                            model,
                            context=context,
                            target_dim=target_dim,
                            covariate_dim=covariate_dim,
                            horizon=args.horizon,
                            maximum_bulk_rows=args.maximum_bulk_rows,
                            maximum_request_input_tokens=(
                                args.maximum_request_input_tokens
                            ),
                        )
                        children = _children(
                            batch_rows=batch_rows,
                            context=context,
                            horizon=args.horizon,
                            target_dim=target_dim,
                            covariate_dim=covariate_dim,
                        )
                        for concurrency in args.concurrencies:
                            base = {
                                "schema_version": "cafe.timer_service_throughput.v2",
                                "model_id": model_id,
                                "replicas_per_device": replicas_per_device,
                                "context_length": context,
                                "horizon": args.horizon,
                                "target_dim": target_dim,
                                "covariate_dim": covariate_dim,
                                "batch_rows": batch_rows,
                                "request_input_tokens": request_tokens,
                                "concurrency": concurrency,
                            }
                            if _case_key(base) in completed:
                                continue
                            request_count = max(
                                int(args.requests_per_case), concurrency
                            )
                            result = asyncio.run(
                                _run_case(
                                    endpoint=args.endpoint,
                                    api_prefix=args.api_prefix,
                                    model_id=model_id,
                                    children=children,
                                    concurrency=concurrency,
                                    request_count=request_count,
                                    warmup_requests=args.warmup_requests,
                                    timeout_seconds=args.timeout_seconds,
                                )
                            )
                            row = {
                                **base,
                                **result,
                                "load_seconds": load_seconds,
                                "loaded_endpoint_count": len(
                                    topology.get("endpoints") or []
                                ),
                                "measured_at": time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                ),
                            }
                            output.write(
                                json.dumps(row, ensure_ascii=False, sort_keys=True)
                                + "\n"
                            )
                            output.flush()
                            print(json.dumps(row, ensure_ascii=False), flush=True)
                if not args.preserve_loaded_model:
                    control.unload_model(model_id)
    finally:
        control.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
