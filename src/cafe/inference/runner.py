#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

import httpx
import msgpack
import numpy as np
from cafe import protocol

INPUT_ADAPTATION_POLICY_ID = "cafe-input-adaptation-v2-input-mode"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"
DEFAULT_ENDPOINTS = (
    "http://127.0.0.1:10810",
    "http://192.168.99.17:10811",
    "http://192.168.99.18:10810",
    "http://192.168.99.89:10810",
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
MODEL_EXECUTION_CONFIG = {
    # CaFE has a fixed H=48 and runs on RTX 5090 services. These defaults were
    # measured end-to-end on one four-card endpoint on 2026-08-04 using both
    # the main native-target mix and a hierarchy/covariate request mix. The x8
    # preset below overrides whole-service concurrency where applicable.
    "Timer-4.0": {
        "replicas_per_device": 4,
        "http_concurrency": 32,
        "task_batch_size": 192,
        "transport": "msgpack_bulk",
    },
    "Timer-3.5": {
        "replicas_per_device": 1,
        "http_concurrency": 8,
        "task_batch_size": 1024,
        "transport": "msgpack_bulk",
    },
    "Chronos-2": {
        "replicas_per_device": 2,
        "http_concurrency": 16,
        "task_batch_size": 192,
        "transport": "msgpack_bulk",
    },
    "moirai2": {
        "replicas_per_device": 1,
        "http_concurrency": 8,
        "task_batch_size": 256,
        "transport": "msgpack_bulk",
    },
    "toto2.0": {
        "replicas_per_device": 2,
        "http_concurrency": 16,
        "task_batch_size": 4,
        "transport": "msgpack_bulk",
    },
    "timesfm2.5": {
        "replicas_per_device": 4,
        "http_concurrency": 32,
        "task_batch_size": 64,
        "transport": "msgpack_bulk",
    },
    "tirex2": {
        "replicas_per_device": 1,
        "http_concurrency": 8,
        "task_batch_size": 512,
        "transport": "msgpack_bulk",
    },
    "TimePFN": {
        "replicas_per_device": 1,
        "http_concurrency": 2,
        "task_batch_size": 512,
        "transport": "msgpack_bulk",
    },
}
MODEL_MAJOR_DATASET_PARALLELISM = {
    "Timer-4.0": 2,
    "Chronos-2": 4,
    "timesfm2.5": 2,
    "tirex2": 2,
    "moirai2": 2,
    "Timer-3.5": 2,
    "toto2.0": 4,
    "TimePFN": 4,
}
RTX5090X8_H48_B1_PRESET = "rtx5090x8-h48-b1-v1"
DEFAULT_ENDPOINT_PRESETS = (f"http://192.168.99.89:10810={RTX5090X8_H48_B1_PRESET}",)
ENDPOINT_PERFORMANCE_PRESETS = {
    RTX5090X8_H48_B1_PRESET: {
        "devices": "0,1,2,3,4,5,6,7",
        # Capacity is relative to one dual-card endpoint. These weights use the
        # measured end-to-end bulk-request ratios; sub-linear entries reflect
        # host/preprocessing work that does not scale with GPU count.
        "model_capacity_units": {
            "Chronos-2": 3.65,
            "toto2.0": 3,
            "tirex2": 3,
            "timesfm2.5": 3.8,
            "Timer-3.5": 2.4,
            "moirai2": 2.8,
            "TimePFN": 4,
        },
        "model_http_concurrency": {
            "Chronos-2": 16,
            "toto2.0": 16,
            "tirex2": 8,
            "timesfm2.5": 8,
            "Timer-3.5": 8,
            "moirai2": 8,
            "TimePFN": 8,
        },
    },
}
SCHEDULING_POLICY_ID = (
    "model-major-weighted-endpoint-shards-interleaved-context-bulk-v2"
)
REAL_ANCHORED_GENERATION_FILE_KEY = "real_anchored_counterfactuals"
REAL_ANCHORED_BENCHMARK_TRACK = "real_anchored_counterfactual"
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class InferenceWork:
    model_id: str
    sample_path: Path
    output_dir: Path
    work_id: str
    part_index: int


@dataclass(frozen=True)
class EndpointProfile:
    endpoint: str
    preset_name: str | None
    devices: str
    capacity_units: float
    concurrency_scale: float
    model_capacity_units: dict[str, float]
    model_http_concurrency: dict[str, int]

    def capacity_for(self, model_id: str) -> float:
        return self.model_capacity_units.get(
            model_id,
            self.capacity_units,
        )

    def http_concurrency_for(
        self,
        model_id: str,
        default: int,
    ) -> int:
        if model_id in self.model_http_concurrency:
            return self.model_http_concurrency[model_id]
        return max(1, round(default * self.concurrency_scale))

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset_name": self.preset_name,
            "devices": self.devices,
            "capacity_units": self.capacity_units,
            "concurrency_scale": self.concurrency_scale,
            "model_capacity_units": dict(sorted(self.model_capacity_units.items())),
            "model_http_concurrency": dict(sorted(self.model_http_concurrency.items())),
        }


def add_endpoint_topology_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--devices",
        default="0,1",
        help="Default device list for endpoints without an override.",
    )
    parser.add_argument(
        "--endpoint-preset",
        action="append",
        default=[],
        metavar="ENDPOINT=PRESET",
        help=(
            "Apply a named, explicitly measured/estimated endpoint profile. "
            f"Available: {', '.join(sorted(ENDPOINT_PERFORMANCE_PRESETS))}."
        ),
    )
    parser.add_argument(
        "--endpoint-devices",
        action="append",
        default=[],
        metavar="ENDPOINT=DEVICE_CSV",
        help=(
            "Override devices for one endpoint. Repeat as needed. Device "
            "count never changes task share or HTTP concurrency implicitly."
        ),
    )
    parser.add_argument(
        "--endpoint-capacity",
        action="append",
        default=[],
        metavar="ENDPOINT=UNITS",
        help=(
            "Set the fallback deterministic task-share units for one "
            "endpoint. Defaults to 1 regardless of device count."
        ),
    )
    parser.add_argument(
        "--endpoint-concurrency-scale",
        action="append",
        default=[],
        metavar="ENDPOINT=SCALE",
        help=(
            "Set the fallback HTTP concurrency multiplier for one endpoint. "
            "Defaults to 1 and is independent of devices and task share."
        ),
    )
    parser.add_argument(
        "--endpoint-model-capacity",
        action="append",
        default=[],
        metavar="ENDPOINT|MODEL=UNITS",
        help=(
            "Override deterministic task-share units for one endpoint and "
            "model. Repeat for independently measured model throughputs."
        ),
    )
    parser.add_argument(
        "--endpoint-model-concurrency",
        action="append",
        default=[],
        metavar="ENDPOINT|MODEL=COUNT",
        help=(
            "Set absolute HTTP concurrency for one endpoint and model. "
            "This takes precedence over the fallback concurrency scale."
        ),
    )


def endpoint_presets_with_defaults(
    endpoints: list[str],
    explicit_presets: list[str],
) -> list[str]:
    presets = list(explicit_presets)
    configured = {value.rsplit("=", 1)[0].strip() for value in presets}
    for default_preset in DEFAULT_ENDPOINT_PRESETS:
        endpoint = default_preset.rsplit("=", 1)[0]
        if endpoint in endpoints and endpoint not in configured:
            presets.append(default_preset)
    return presets


_OverrideValue = TypeVar("_OverrideValue")


def _parse_endpoint_overrides(
    values: list[str],
    *,
    option_name: str,
    convert: Callable[[str], _OverrideValue],
) -> dict[str, _OverrideValue]:
    output: dict[str, _OverrideValue] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} must use ENDPOINT=VALUE: {value!r}")
        endpoint, raw = value.split("=", 1)
        endpoint = endpoint.strip()
        raw = raw.strip()
        if not endpoint or not raw:
            raise ValueError(f"{option_name} must use non-empty ENDPOINT=VALUE")
        if endpoint in output:
            raise ValueError(f"duplicate {option_name} for endpoint {endpoint!r}")
        try:
            output[endpoint] = convert(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid {option_name} for endpoint {endpoint!r}: {raw!r}"
            ) from error
    return output


def _normalize_devices(value: str) -> str:
    devices = [item.strip() for item in value.split(",")]
    if (
        not devices
        or any(not item or not item.isdecimal() for item in devices)
        or len(devices) != len(set(devices))
    ):
        raise ValueError("devices must be unique non-negative integer ids")
    return ",".join(devices)


def _parse_endpoint_model_overrides(
    values: list[str],
    *,
    option_name: str,
    convert: Callable[[str], _OverrideValue],
) -> dict[tuple[str, str], _OverrideValue]:
    output: dict[tuple[str, str], _OverrideValue] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} must use ENDPOINT|MODEL=VALUE: {value!r}")
        key, raw = value.rsplit("=", 1)
        if "|" not in key:
            raise ValueError(f"{option_name} must use ENDPOINT|MODEL=VALUE: {value!r}")
        endpoint, model_id = (item.strip() for item in key.split("|", 1))
        raw = raw.strip()
        if not endpoint or not model_id or not raw:
            raise ValueError(f"{option_name} must use non-empty ENDPOINT|MODEL=VALUE")
        pair = (endpoint, model_id)
        if pair in output:
            raise ValueError(f"duplicate {option_name} for {endpoint!r}, {model_id!r}")
        try:
            parsed = convert(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid {option_name} for {endpoint!r}, " f"{model_id!r}: {raw!r}"
            ) from error
        if parsed <= 0:
            raise ValueError(f"{option_name} values must be positive")
        output[pair] = parsed
    return output


def build_endpoint_profiles(
    endpoints: list[str],
    *,
    default_devices: str,
    endpoint_presets: list[str],
    endpoint_devices: list[str],
    endpoint_capacities: list[str],
    endpoint_concurrency_scales: list[str],
    endpoint_model_capacities: list[str],
    endpoint_model_concurrencies: list[str],
) -> dict[str, EndpointProfile]:
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("inference endpoints must be unique")
    normalized_default = _normalize_devices(default_devices)
    preset_overrides = _parse_endpoint_overrides(
        endpoint_presets,
        option_name="--endpoint-preset",
        convert=str,
    )
    unknown_presets = sorted(
        set(preset_overrides.values()) - set(ENDPOINT_PERFORMANCE_PRESETS)
    )
    if unknown_presets:
        raise ValueError(
            "unknown endpoint performance preset: " + ", ".join(unknown_presets)
        )
    device_overrides = _parse_endpoint_overrides(
        endpoint_devices,
        option_name="--endpoint-devices",
        convert=_normalize_devices,
    )
    capacity_overrides = _parse_endpoint_overrides(
        endpoint_capacities,
        option_name="--endpoint-capacity",
        convert=float,
    )
    concurrency_overrides = _parse_endpoint_overrides(
        endpoint_concurrency_scales,
        option_name="--endpoint-concurrency-scale",
        convert=float,
    )
    model_capacity_overrides = _parse_endpoint_model_overrides(
        endpoint_model_capacities,
        option_name="--endpoint-model-capacity",
        convert=float,
    )
    model_concurrency_overrides = _parse_endpoint_model_overrides(
        endpoint_model_concurrencies,
        option_name="--endpoint-model-concurrency",
        convert=int,
    )
    configured_endpoints = (
        set(preset_overrides)
        | set(device_overrides)
        | set(capacity_overrides)
        | set(concurrency_overrides)
        | {endpoint for endpoint, _model_id in model_capacity_overrides}
        | {endpoint for endpoint, _model_id in model_concurrency_overrides}
    )
    unknown = sorted(configured_endpoints - set(endpoints))
    if unknown:
        raise ValueError(
            "endpoint topology override does not match --endpoints: "
            + ", ".join(unknown)
        )
    configured_models = {
        model_id
        for _endpoint, model_id in (
            set(model_capacity_overrides) | set(model_concurrency_overrides)
        )
    }
    unknown_models = sorted(configured_models - set(MODEL_EXECUTION_CONFIG))
    if unknown_models:
        raise ValueError(
            "endpoint model override has no execution config: "
            + ", ".join(unknown_models)
        )

    profiles: dict[str, EndpointProfile] = {}
    for endpoint in endpoints:
        preset_name = preset_overrides.get(endpoint)
        preset = (
            ENDPOINT_PERFORMANCE_PRESETS[preset_name] if preset_name is not None else {}
        )
        devices = device_overrides.get(
            endpoint,
            str(preset.get("devices", normalized_default)),
        )
        capacity_units = capacity_overrides.get(endpoint, 1.0)
        if capacity_units <= 0:
            raise ValueError("endpoint capacity units must be positive")
        concurrency_scale = concurrency_overrides.get(endpoint, 1.0)
        if concurrency_scale <= 0:
            raise ValueError("endpoint concurrency scales must be positive")
        profiles[endpoint] = EndpointProfile(
            endpoint=endpoint,
            preset_name=preset_name,
            devices=devices,
            capacity_units=capacity_units,
            concurrency_scale=concurrency_scale,
            model_capacity_units={
                **dict(preset.get("model_capacity_units", {})),
                **{
                    model_id: value
                    for (
                        configured_endpoint,
                        model_id,
                    ), value in model_capacity_overrides.items()
                    if configured_endpoint == endpoint
                },
            },
            model_http_concurrency={
                **dict(preset.get("model_http_concurrency", {})),
                **{
                    model_id: value
                    for (
                        configured_endpoint,
                        model_id,
                    ), value in model_concurrency_overrides.items()
                    if configured_endpoint == endpoint
                },
            },
        )
    return profiles


def endpoint_topology_cli_arguments(args: argparse.Namespace) -> list[str]:
    arguments = ["--devices", str(args.devices)]
    endpoint_presets = endpoint_presets_with_defaults(
        list(args.endpoints),
        list(args.endpoint_preset),
    )
    for option, values in (
        ("--endpoint-preset", endpoint_presets),
        ("--endpoint-devices", args.endpoint_devices),
        ("--endpoint-capacity", args.endpoint_capacity),
        (
            "--endpoint-concurrency-scale",
            args.endpoint_concurrency_scale,
        ),
        (
            "--endpoint-model-capacity",
            args.endpoint_model_capacity,
        ),
        (
            "--endpoint-model-concurrency",
            args.endpoint_model_concurrency,
        ),
    ):
        for value in values:
            arguments.extend((option, value))
    return arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal CaFE multi-service model inference."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        help=(
            "Run several generated datasets model-major, keeping each model "
            "loaded until every listed dataset is complete."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument(
        "--input-capability-contract",
        type=Path,
        default=None,
        help=(
            "Frozen inference stage contract containing the normalized live "
            "model input capabilities."
        ),
    )
    add_endpoint_topology_arguments(parser)
    parser.add_argument("--load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the task view and model endpoint shards without inference.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel dataset preprocessors used by model-major execution.",
    )
    parser.add_argument(
        "--request-concurrency-divisor",
        type=int,
        default=1,
        help=(
            "Divide endpoint HTTP concurrency by this value when several "
            "datasets share one loaded model."
        ),
    )
    parser.add_argument(
        "--keep-loaded-between-runs",
        action="store_true",
        help=(
            "Leave the selected model loaded so a model-major controller can "
            "reuse it for the next dataset."
        ),
    )
    return parser.parse_args()


def safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def count_jsonl(path: Path) -> int:
    return sum(1 for _row in protocol.iter_jsonl(path))


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(protocol.REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_envelope(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"{response.request.url} returned non-json: {response.text[:200]}"
        ) from error
    if response.status_code != 200:
        raise RuntimeError(
            f"{response.request.url} returned {response.status_code}: "
            f"{payload.get('message', response.text)}"
        )
    if payload.get("code") not in (None, 200):
        raise RuntimeError(
            f"{response.request.url} returned code {payload.get('code')}: "
            f"{payload.get('message')}"
        )
    return payload


class TimerServiceClient:
    def __init__(self, base_url: str, api_prefix: str, *, timeout_seconds: int):
        self.base = base_url.rstrip("/") + "/" + api_prefix.strip("/")
        self.client = httpx.Client(timeout=timeout_seconds, trust_env=False)
        self.timeout_seconds = timeout_seconds

    def close(self) -> None:
        self.client.close()

    def list_models(self) -> list[dict[str, Any]]:
        return list(self._get("/models/list")["data"]["models"])

    def list_loaded_models(self) -> list[dict[str, Any]]:
        return list(self._get("/models/list_loaded")["data"]["models"])

    def unload_all_loaded(self) -> None:
        for model in self.list_loaded_models():
            endpoints = model.get("endpoints") or []
            if any(
                str(endpoint.get("device", "")).lower() != "cpu"
                for endpoint in endpoints
            ):
                self.unload_model(str(model["model_id"]))

    def unload_model(self, model_id: str) -> None:
        deadline = time.monotonic() + max(self.timeout_seconds, 600)
        next_submit = time.monotonic()
        last_transient_error: Exception | None = None
        while True:
            now = time.monotonic()
            if now >= next_submit:
                try:
                    self._post(
                        "/models/unload",
                        {"model_id": model_id},
                        timeout_seconds=max(self.timeout_seconds, 600),
                    )
                    last_transient_error = None
                except RuntimeError as error:
                    message = str(error).lower()
                    if "409" in message and "not loaded" in message:
                        return
                    if not self._is_transient_control_error(error):
                        raise
                    last_transient_error = error
                except (httpx.TimeoutException, httpx.TransportError) as error:
                    last_transient_error = error
                next_submit = time.monotonic() + 10

            try:
                state = self._loaded_state(model_id)
            except (
                RuntimeError,
                httpx.TimeoutException,
                httpx.TransportError,
            ) as error:
                if not self._is_transient_control_error(error):
                    raise
                last_transient_error = error
                state = object()
            if state is None:
                return
            if time.monotonic() >= deadline:
                detail = (
                    f"; last transient error: {last_transient_error}"
                    if last_transient_error is not None
                    else ""
                )
                raise TimeoutError(f"timed out unloading model {model_id}{detail}")
            time.sleep(1)

    def ensure_loaded(
        self,
        model_id: str,
        *,
        devices: str,
        replicas_per_device: int,
        timeout_seconds: int,
    ) -> tuple[float, dict[str, Any]]:
        started = time.monotonic()
        device_indexes = [part.strip() for part in devices.split(",") if part.strip()]
        expected_devices = {f"cuda:{index}" for index in device_indexes}
        expected_endpoints = len(device_indexes) * replicas_per_device
        deadline = time.monotonic() + timeout_seconds
        next_submit = started
        first_observation = True
        last_control_error: Exception | None = None
        while True:
            state_unavailable = False
            try:
                state = self._loaded_state(model_id)
            except (
                RuntimeError,
                httpx.TimeoutException,
                httpx.TransportError,
            ) as error:
                if not self._is_transient_control_error(error):
                    raise
                last_control_error = error
                state = None
                state_unavailable = True

            if state is not None:
                status = str(state.get("status", "")).lower()
                if status == "loaded":
                    self._validate_loaded_topology(
                        model_id,
                        state,
                        expected_devices=expected_devices,
                        replicas_per_device=replicas_per_device,
                        expected_endpoints=expected_endpoints,
                    )
                    elapsed = 0.0 if first_observation else time.monotonic() - started
                    return elapsed, state
                if status not in {"loading", "unloading", "pending", "queued"}:
                    raise RuntimeError(
                        f"model {model_id} entered unexpected state before becoming "
                        f"ready: {state}"
                    )
            elif not state_unavailable and time.monotonic() >= next_submit:
                try:
                    self._post(
                        "/models/load",
                        {
                            "model_id": model_id,
                            "devices": devices,
                            "replicas_per_device": replicas_per_device,
                        },
                        timeout_seconds=timeout_seconds,
                    )
                    last_control_error = None
                except RuntimeError as error:
                    if not (
                        self._is_load_in_progress_error(error)
                        or self._is_transient_control_error(error)
                    ):
                        raise
                    last_control_error = error
                except (httpx.TimeoutException, httpx.TransportError) as error:
                    last_control_error = error
                next_submit = time.monotonic() + 10

            first_observation = False
            if time.monotonic() >= deadline:
                detail = (
                    f"; last control error: {last_control_error}"
                    if last_control_error is not None
                    else ""
                )
                raise TimeoutError(f"timed out loading model {model_id}{detail}")
            time.sleep(1)

    @staticmethod
    def _is_load_in_progress_error(error: Exception) -> bool:
        message = str(error).lower()
        return "409" in message and any(
            marker in message
            for marker in (
                "already loaded",
                "being loaded",
                "loading",
                "in progress",
                "conflict",
            )
        )

    @staticmethod
    def _is_transient_control_error(error: Exception) -> bool:
        if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
            return True
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                " 429",
                " 502",
                " 503",
                " 504",
                "coordinator unreachable",
                "resource temporarily unavailable",
                "timed out",
                "timeout",
            )
        )

    def _loaded_state(self, model_id: str) -> dict[str, Any] | None:
        return next(
            (
                model
                for model in self.list_loaded_models()
                if str(model.get("model_id", "")).lower() == model_id.lower()
            ),
            None,
        )

    @staticmethod
    def _validate_loaded_topology(
        model_id: str,
        state: dict[str, Any],
        *,
        expected_devices: set[str],
        replicas_per_device: int,
        expected_endpoints: int,
    ) -> None:
        endpoints = list(state.get("endpoints") or [])
        observed_devices = {str(endpoint.get("device")) for endpoint in endpoints}
        per_device = {
            device: sum(str(endpoint.get("device")) == device for endpoint in endpoints)
            for device in expected_devices
        }
        pids = [endpoint.get("worker_pid") for endpoint in endpoints]
        if (
            state.get("status") != "loaded"
            or len(endpoints) != expected_endpoints
            or observed_devices != expected_devices
            or any(count != replicas_per_device for count in per_device.values())
            or len(set(pids)) != expected_endpoints
        ):
            raise RuntimeError(
                f"model {model_id} loaded topology does not match frozen cafe config: "
                f"expected_devices={sorted(expected_devices)}, "
                f"replicas_per_device={replicas_per_device}, state={state}"
            )

    def _get(self, path: str) -> dict[str, Any]:
        return parse_envelope(
            self.client.get(self.base + path, timeout=self.timeout_seconds)
        )

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        return parse_envelope(
            self.client.post(self.base + path, json=body, timeout=timeout_seconds)
        )


def iter_forecast_samples(path: Path) -> Iterator[dict[str, Any]]:
    yield from protocol.iter_jsonl(path)


def prediction_path_for(
    output_dir: Path,
    model_id: str,
    *,
    prediction_kind: str = "synthetic",
) -> Path:
    return output_dir / "predictions" / f"{safe_filename(model_id)}.jsonl"


def prediction_row(
    model_id: str,
    model_group: str,
    sample: dict[str, Any],
    forecast: np.ndarray | list[list[float]],
) -> dict[str, Any]:
    values = np.asarray(forecast, dtype=float)
    expected_shape = (int(sample["horizon"]), int(sample["target_dim"]))
    if values.shape != expected_shape:
        raise ValueError(f"forecast shape mismatch: {values.shape} != {expected_shape}")
    return {
        "schema_version": "cafe.inference_prediction.v1",
        "model_id": model_id,
        "sample_id": sample["sample_id"],
        "forecast": values.tolist(),
    }


def model_supports_window(
    model: dict[str, Any],
    sample: dict[str, Any],
) -> bool:
    limits = model.get("forecast_limits") or {}
    context = int(sample["context_length"])
    horizon = int(sample["horizon"])
    if context < int(limits.get("min_input_length") or 0):
        return False
    maximum_input = _normalize_unbounded_count(
        limits.get("max_input_length"),
        default=None,
    )
    if maximum_input is not None and context > maximum_input:
        return False
    maximum_output = _normalize_unbounded_count(
        limits.get("max_output_length"),
        default=None,
    )
    if maximum_output is not None and horizon > maximum_output:
        return False
    return True


def model_supports_sample(
    model: dict[str, Any],
    sample: dict[str, Any],
) -> bool:
    if not model_supports_window(model, sample):
        return False
    horizon = int(sample["horizon"])
    target_dim = int(sample["target_dim"])
    covariate_dim = int(sample["covariate_dim"])
    capability = resolve_input_capability(model)
    maximum_targets = capability["max_target_count"]
    if maximum_targets is not None and target_dim > maximum_targets:
        return False
    if covariate_dim:
        maximum_covariates = capability["max_history_covariate_count"]
        if maximum_covariates is not None and covariate_dim > maximum_covariates:
            return False
        if not capability["supports_future_covariates"]:
            return False
        maximum_future_length = capability["max_future_covariate_length"]
        if maximum_future_length is not None and horizon > maximum_future_length:
            return False
    return True


def _normalize_unbounded_count(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    normalized = int(value)
    return None if normalized < 0 else normalized


def resolve_input_capability(model: dict[str, Any]) -> dict[str, Any]:
    """Normalize current and legacy Timer service input capability schemas."""

    limits = model.get("forecast_limits") or {}
    input_mode = limits.get("input_mode")
    if isinstance(input_mode, dict):
        source_schema = "input_mode"
        maximum_targets = _normalize_unbounded_count(
            input_mode.get("max_target_count"),
            default=(None if "max_target_count" in input_mode else 1),
        )
        maximum_history_covariates = _normalize_unbounded_count(
            input_mode.get("max_history_covariate_count"),
            default=(None if "max_history_covariate_count" in input_mode else 0),
        )
        supports_future_covariates = bool(
            input_mode.get("supports_future_covariates", False)
        )
        maximum_future_length = _normalize_unbounded_count(
            limits.get("max_future_covs_length"),
            default=None,
        )
    else:
        source_schema = "legacy_forecast_limits"
        maximum_targets = _normalize_unbounded_count(
            limits.get("max_target_count"),
            default=(None if "max_target_count" in limits else 1),
        )
        maximum_history_covariates = _normalize_unbounded_count(
            limits.get("max_covariate_count"),
            default=(None if "max_covariate_count" in limits else 0),
        )
        legacy_future_length = limits.get("max_future_covs_length")
        supports_future_covariates = (
            maximum_history_covariates != 0 and legacy_future_length is not None
        )
        maximum_future_length = _normalize_unbounded_count(
            legacy_future_length,
            default=None,
        )
    return {
        "schema_version": "cafe.resolved_input_capability.v1",
        "source_schema": source_schema,
        "max_target_count": maximum_targets,
        "max_history_covariate_count": maximum_history_covariates,
        "supports_future_covariates": supports_future_covariates,
        "max_future_covariate_length": maximum_future_length,
    }


def _supports_native_targets(
    capability: dict[str, Any],
    target_dim: int,
) -> bool:
    if target_dim <= 1:
        return True
    maximum_targets = capability["max_target_count"]
    return maximum_targets is None or target_dim <= maximum_targets


def _supports_native_covariates(
    capability: dict[str, Any],
    *,
    covariate_dim: int,
    horizon: int,
) -> bool:
    if covariate_dim <= 0:
        return True
    maximum_covariates = capability["max_history_covariate_count"]
    if maximum_covariates is not None and covariate_dim > maximum_covariates:
        return False
    if not capability["supports_future_covariates"]:
        return False
    maximum_future = capability["max_future_covariate_length"]
    return maximum_future is None or horizon <= maximum_future


def input_adaptation_plan(
    model: dict[str, Any],
    sample: dict[str, Any],
    *,
    policy_id: str | None,
) -> dict[str, Any] | None:
    if not model_supports_window(model, sample):
        return None
    if policy_id not in {None, INPUT_ADAPTATION_POLICY_ID}:
        raise ValueError(f"unknown input adaptation policy: {policy_id}")
    if policy_id is None and not model_supports_sample(model, sample):
        return None

    capability = resolve_input_capability(model)
    target_dim = int(sample["target_dim"])
    covariate_dim = int(sample["covariate_dim"])
    horizon = int(sample["horizon"])
    target_native = (
        True if policy_id is None else _supports_native_targets(capability, target_dim)
    )
    covariates_native = (
        True
        if policy_id is None
        else _supports_native_covariates(
            capability,
            covariate_dim=covariate_dim,
            horizon=horizon,
        )
    )
    if target_native:
        target_mode = "native_univariate" if target_dim == 1 else "native_multivariate"
    else:
        target_mode = "independent_univariate"
    if covariate_dim == 0:
        covariate_mode = "none"
    elif covariates_native:
        covariate_mode = "native"
    else:
        covariate_mode = "omitted_unsupported"
    target_request_count = target_dim if target_mode == "independent_univariate" else 1
    return {
        "policy_id": (
            INPUT_ADAPTATION_POLICY_ID if policy_id is not None else "native-only"
        ),
        "target_mode": target_mode,
        "covariate_mode": covariate_mode,
        "adapted": (
            target_mode == "independent_univariate"
            or covariate_mode == "omitted_unsupported"
        ),
        "original_target_dim": target_dim,
        "request_target_dim": (
            1 if target_mode == "independent_univariate" else target_dim
        ),
        "target_request_count": target_request_count,
        "original_covariate_dim": covariate_dim,
        "request_covariate_dim": (covariate_dim if covariate_mode == "native" else 0),
        "resolved_input_capability": capability,
    }


def _target_column_names(sample: dict[str, Any]) -> list[str]:
    target_dim = int(sample["target_dim"])
    configured = sample.get("target_column_names")
    if configured is None:
        configured = sample.get("target_columns")
    if configured is None:
        return [f"target_{index}" for index in range(target_dim)]
    names = [str(value) for value in configured]
    if len(names) != target_dim or len(set(names)) != len(names):
        raise ValueError("target_column_names must be unique and match target_dim")
    return names


def adapted_request_samples(
    sample: dict[str, Any],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    target = np.asarray(sample["target"], dtype=float)
    target_names = _target_column_names(sample)
    if plan["target_mode"] == "independent_univariate":
        target_indexes: list[int | None] = list(range(int(sample["target_dim"])))
    else:
        target_indexes = [None]
    requests: list[dict[str, Any]] = []
    for target_index in target_indexes:
        child = dict(sample)
        if target_index is not None:
            child["target"] = target[:, target_index : target_index + 1].tolist()
            child["target_dim"] = 1
            child["target_column_names"] = [target_names[target_index]]
            child["_adaptation_target_index"] = int(target_index)
        else:
            child["target_column_names"] = target_names
            child["_adaptation_target_index"] = None
        if plan["covariate_mode"] == "omitted_unsupported":
            child["covariates"] = None
            child["covariate_dim"] = 0
            child["covariate_column_names"] = []
        requests.append(child)
    return requests


def request_group_key(
    sample: dict[str, Any],
    *,
    plan: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    base = (
        sample["context_length"],
        sample["horizon"],
        sample["target_dim"],
        sample["covariate_dim"],
        sample["frequency"],
    )
    if plan is None or plan["policy_id"] == "native-only":
        return base
    return (
        *base,
        plan["target_request_count"],
        plan["target_mode"],
        plan["covariate_mode"],
        plan["request_target_dim"],
        plan["request_covariate_dim"],
    )


def request_group_sort_key(group: tuple[Any, ...]) -> tuple[Any, ...]:
    context, horizon, target_dim, covariate_dim, frequency = group[:5]
    request_covariate_dim = int(group[9]) if len(group) > 5 else int(covariate_dim)
    return (
        int(request_covariate_dim > 0),
        int(context),
        int(target_dim),
        int(horizon),
        str(frequency),
    )


def request_group_label(group: tuple[Any, ...]) -> str:
    context, horizon, target_dim, covariate_dim, frequency = group[:5]
    base = f"ctx{context}_h{horizon}_t{target_dim}_c{covariate_dim}_{frequency}"
    if len(group) == 5:
        return base
    (
        target_request_count,
        target_mode,
        covariate_mode,
        request_target_dim,
        request_covariate_dim,
    ) = group[5:]
    return (
        f"{base}__{target_mode}__{covariate_mode}__"
        f"requests{target_request_count}_t{request_target_dim}_"
        f"c{request_covariate_dim}"
    )


def successful_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    identifiers: set[str] = set()
    for row in protocol.iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in identifiers:
            raise ValueError(
                f"duplicate successful prediction for {sample_id} in {path}"
            )
        identifiers.add(sample_id)
    return identifiers


def summarize_model_input_adaptation(
    sample_path: Path,
    *,
    model: dict[str, Any],
    input_adaptation_policy: str | None,
) -> dict[str, int]:
    summary = {
        "expected_original_view_count": 0,
        "compatible_sample_count": 0,
        "unsupported_window_view_count": 0,
        "native_view_count": 0,
        "adapted_view_count": 0,
        "split_target_view_count": 0,
        "covariates_omitted_view_count": 0,
        "expected_http_request_count": 0,
    }
    for sample in iter_forecast_samples(sample_path):
        summary["expected_original_view_count"] += 1
        plan = input_adaptation_plan(
            model,
            sample,
            policy_id=input_adaptation_policy,
        )
        if plan is None:
            summary["unsupported_window_view_count"] += 1
            continue
        summary["compatible_sample_count"] += 1
        summary["expected_http_request_count"] += int(plan["target_request_count"])
        if plan["adapted"]:
            summary["adapted_view_count"] += 1
        else:
            summary["native_view_count"] += 1
        if plan["target_mode"] == "independent_univariate":
            summary["split_target_view_count"] += 1
        if plan["covariate_mode"] == "omitted_unsupported":
            summary["covariates_omitted_view_count"] += 1
    return summary


def persisted_http_request_count(path: Path) -> int:
    if not path.is_file():
        return 0
    total = 0
    for row in protocol.iter_jsonl(path):
        adaptation = row.get("input_adaptation")
        total += int(
            adaptation.get("target_request_count", 1)
            if isinstance(adaptation, dict)
            else 1
        )
    return total


def pending_request_group_counts(
    sample_path: Path,
    *,
    model: dict[str, Any],
    done: set[str],
    input_adaptation_policy: str | None = None,
) -> dict[tuple[Any, ...], int]:
    counts: defaultdict[tuple[Any, ...], int] = defaultdict(int)
    for sample in iter_forecast_samples(sample_path):
        plan = input_adaptation_plan(
            model,
            sample,
            policy_id=input_adaptation_policy,
        )
        if sample["sample_id"] in done or plan is None:
            continue
        counts[request_group_key(sample, plan=plan)] += 1
    return dict(counts)


def run_one_model(
    client: TimerServiceClient,
    model: dict[str, Any],
    *,
    output_dir: Path,
    execution: dict[str, Any],
    devices: str,
    request_max_attempts: int,
    forecast_timeout_seconds: int,
    load_timeout_seconds: int,
    keep_loaded: bool,
    sample_path: Path | None = None,
    prediction_kind: str = "synthetic",
    status_filename: str = "model_status.json",
    input_adaptation_policy: str | None = None,
) -> dict[str, Any]:
    sample_path = sample_path or output_dir / "samples.jsonl"
    model_id = str(model["model_id"])
    prediction_path = prediction_path_for(
        output_dir,
        model_id,
        prediction_kind=prediction_kind,
    )
    done = successful_sample_ids(prediction_path)
    adaptation_summary = summarize_model_input_adaptation(
        sample_path,
        model=model,
        input_adaptation_policy=input_adaptation_policy,
    )
    compatible_count = adaptation_summary["compatible_sample_count"]
    if len(done) > compatible_count:
        raise ValueError(f"prediction file for {model_id} has too many unique samples")
    if len(done) == compatible_count:
        status_path = output_dir / status_filename
        previous = (
            protocol.read_json(status_path).get("models", {}).get(model_id)
            if status_path.is_file()
            else None
        )
        if (
            previous
            and previous.get("status") == "complete"
            and int(previous.get("succeeded_count", -1)) == compatible_count
        ):
            return previous
        return {
            "model_id": model_id,
            "status": "complete",
            **adaptation_summary,
            "succeeded_count": compatible_count,
            "succeeded_original_view_count": compatible_count,
            "successful_http_request_count": persisted_http_request_count(
                prediction_path
            ),
            "already_complete_on_entry": True,
            "prediction_path": relative_path(prediction_path),
            "elapsed_seconds": 0.0,
        }

    started = time.monotonic()
    load_seconds = 0.0
    failures = 0
    bucket_stats: list[dict[str, Any]] = []
    loaded_topology: dict[str, Any] | None = None
    pending_groups = pending_request_group_counts(
        sample_path,
        model=model,
        done=done,
        input_adaptation_policy=input_adaptation_policy,
    )
    try:
        if len(done) < compatible_count:
            if not keep_loaded:
                client.unload_all_loaded()
            load_seconds, loaded_topology = client.ensure_loaded(
                model_id,
                devices=devices,
                replicas_per_device=int(execution["replicas_per_device"]),
                timeout_seconds=load_timeout_seconds,
            )
            with prediction_path.open("a", encoding="utf-8") as output_handle:
                failure_path = (
                    output_dir / "failures" / f"{safe_filename(model_id)}.jsonl"
                )
                if prediction_kind == "real":
                    failure_path = (
                        output_dir
                        / "real_failures"
                        / f"{safe_filename(model_id)}.jsonl"
                    )
                with failure_path.open("a", encoding="utf-8") as failure_handle:
                    bucket_stats = asyncio.run(
                        run_model_requests_v8(
                            forecast_url=client.base + "/forecast",
                            model_id=model_id,
                            model=model,
                            sample_path=sample_path,
                            done=done,
                            pending_groups=pending_groups,
                            http_concurrency=int(execution["http_concurrency"]),
                            timeout_seconds=forecast_timeout_seconds,
                            max_attempts=request_max_attempts,
                            output_handle=output_handle,
                            failure_handle=failure_handle,
                            compatible_count=compatible_count,
                            initial_persisted=len(done),
                            input_adaptation_policy=input_adaptation_policy,
                        )
                    )
                    failures = sum(row["failed_count"] for row in bucket_stats)
    finally:
        if not keep_loaded:
            try:
                client.unload_model(model_id)
            except Exception as error:  # noqa: BLE001
                print(f"warning: failed to unload {model_id}: {error}", flush=True)

    succeeded = count_jsonl(prediction_path) if prediction_path.exists() else 0
    return {
        "model_id": model_id,
        "prediction_kind": prediction_kind,
        "status": "complete" if succeeded == compatible_count else "incomplete",
        **adaptation_summary,
        "succeeded_count": succeeded,
        "succeeded_original_view_count": succeeded,
        "failed_request_count_this_attempt": failures,
        "successful_http_request_count": persisted_http_request_count(prediction_path),
        "attempted_http_request_count_this_attempt": sum(
            int(row.get("attempted_http_request_count", 0)) for row in bucket_stats
        ),
        "execution": {
            "devices": devices,
            "replicas_per_device": int(execution["replicas_per_device"]),
            "http_concurrency": int(execution["http_concurrency"]),
            "tasks_per_http_request": 1,
        },
        "loaded_topology": loaded_topology,
        "bucket_stats": bucket_stats,
        "load_seconds": round(load_seconds, 3),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "prediction_path": relative_path(prediction_path),
    }


def _bulk_request_content(
    model_id: str,
    children: list[dict[str, Any]],
) -> tuple[bytes, tuple[int, int, int], int]:
    if not children:
        raise ValueError("a bulk forecast request requires at least one child")
    context = int(children[0]["context_length"])
    horizon = int(children[0]["horizon"])
    target_dim = int(children[0]["target_dim"])
    covariate_dim = int(children[0]["covariate_dim"])
    for child in children[1:]:
        observed = (
            int(child["context_length"]),
            int(child["horizon"]),
            int(child["target_dim"]),
            int(child["covariate_dim"]),
        )
        if observed != (context, horizon, target_dim, covariate_dim):
            raise ValueError(
                "bulk forecast children must have identical request shapes"
            )

    targets = np.stack(
        [
            np.asarray(child["target"], dtype=np.float32)[:context].T
            for child in children
        ]
    )
    payload: dict[str, Any] = {
        "model_id": model_id,
        "shape": list(targets.shape),
        "targets": np.ascontiguousarray(targets).tobytes(),
        "output_length": horizon,
    }
    if covariate_dim:
        history_covariates = np.stack(
            [
                np.asarray(child["covariates"], dtype=np.float32)[:context].T
                for child in children
            ]
        )
        future_covariates = np.stack(
            [
                np.asarray(child["covariates"], dtype=np.float32)[
                    context : context + horizon
                ].T
                for child in children
            ]
        )
        payload.update(
            {
                "history_covariates_shape": list(history_covariates.shape),
                "history_covariates": np.ascontiguousarray(
                    history_covariates
                ).tobytes(),
                "future_covariates_shape": list(future_covariates.shape),
                "future_covariates": np.ascontiguousarray(future_covariates).tobytes(),
            }
        )
    return (
        msgpack.packb(payload, use_bin_type=True),
        tuple(int(value) for value in targets.shape),
        horizon,
    )


async def _forecast_bulk_with_retry(
    client: httpx.AsyncClient,
    *,
    forecast_url: str,
    model_id: str,
    children: list[dict[str, Any]],
    max_attempts: int,
) -> dict[str, Any]:
    started = time.monotonic()
    content, input_shape, horizon = _bulk_request_content(
        model_id,
        children,
    )
    last_error = "unknown bulk forecast error"
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(
                forecast_url,
                content=content,
                headers={"Content-Type": "application/msgpack"},
            )
            response.raise_for_status()
            payload = msgpack.unpackb(response.content, raw=False)
            if payload.get("encoding") != "float32":
                raise ValueError("bulk forecast returned an unknown encoding")
            output_shape = tuple(int(value) for value in payload["shape"])
            expected_shape = (input_shape[0], input_shape[1], horizon)
            if output_shape != expected_shape:
                raise ValueError(
                    f"bulk forecast shape {output_shape} != {expected_shape}"
                )
            forecasts = np.frombuffer(
                payload["forecasts"],
                dtype=np.float32,
            )
            if forecasts.size != int(np.prod(output_shape)):
                raise ValueError(
                    "bulk forecast byte length does not match output shape"
                )
            forecasts = forecasts.reshape(output_shape)
            if not np.isfinite(forecasts).all():
                raise ValueError("bulk forecast contains non-finite values")
            return {
                "forecasts": forecasts,
                "attempts": attempt,
                "elapsed_seconds": time.monotonic() - started,
                "error": None,
            }
        except (
            httpx.HTTPError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            msgpack.ExtraData,
            msgpack.FormatError,
        ) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < max_attempts:
                await asyncio.sleep(0.25 * (2 ** (attempt - 1)))
    return {
        "forecasts": None,
        "attempts": max_attempts,
        "elapsed_seconds": time.monotonic() - started,
        "error": last_error,
    }


async def _run_model_requests_v8_serial(
    *,
    forecast_url: str,
    model_id: str,
    model: dict[str, Any],
    sample_path: Path,
    done: set[str],
    pending_groups: dict[tuple[Any, ...], int],
    http_concurrency: int,
    timeout_seconds: int,
    max_attempts: int,
    output_handle: Any,
    failure_handle: Any,
    compatible_count: int,
    initial_persisted: int,
    input_adaptation_policy: str | None = None,
) -> list[dict[str, Any]]:
    execution = MODEL_EXECUTION_CONFIG[model_id]
    if execution.get("transport") != "msgpack_bulk":
        raise ValueError(f"paper cafe requires msgpack_bulk transport for {model_id}")

    task_batch_size = int(execution["task_batch_size"])
    limits = httpx.Limits(
        max_connections=http_concurrency,
        max_keepalive_connections=http_concurrency,
        keepalive_expiry=120.0,
    )
    timeout = httpx.Timeout(timeout_seconds)
    bucket_stats: list[dict[str, Any]] = []
    persisted = initial_persisted
    bulk_url = forecast_url.rsplit("/", 1)[0] + "/forecast/bulk"
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as async_client:
        ordered_groups = sorted(
            pending_groups,
            key=request_group_sort_key,
        )
        for bucket_index, group_key in enumerate(ordered_groups, start=1):
            pending_count = pending_groups[group_key]
            label = request_group_label(group_key)
            group_items: list[
                tuple[
                    dict[str, Any],
                    dict[str, Any],
                    list[dict[str, Any]],
                ]
            ] = []
            for sample in iter_forecast_samples(sample_path):
                plan = input_adaptation_plan(
                    model,
                    sample,
                    policy_id=input_adaptation_policy,
                )
                if (
                    sample["sample_id"] in done
                    or plan is None
                    or request_group_key(sample, plan=plan) != group_key
                ):
                    continue
                group_items.append(
                    (
                        sample,
                        plan,
                        adapted_request_samples(sample, plan),
                    )
                )
            if len(group_items) != pending_count:
                raise RuntimeError(
                    f"{model_id}/{label} producer count "
                    f"{len(group_items)} != {pending_count}"
                )

            child_count = int(group_items[0][1]["target_request_count"])
            views_per_request = max(1, task_batch_size // child_count)
            chunks = [
                group_items[offset : offset + views_per_request]
                for offset in range(0, len(group_items), views_per_request)
            ]
            print(
                f"{model_id}: bucket {bucket_index}/{len(ordered_groups)} "
                f"{label}, pending={pending_count}, bulk_batch="
                f"{task_batch_size}, views/request={views_per_request}, "
                f"requests={len(chunks)}, concurrency={http_concurrency}",
                flush=True,
            )
            queue: asyncio.Queue[
                list[
                    tuple[
                        dict[str, Any],
                        dict[str, Any],
                        list[dict[str, Any]],
                    ]
                ]
                | None
            ] = asyncio.Queue()
            for chunk in chunks:
                queue.put_nowait(chunk)
            for _worker_index in range(http_concurrency):
                queue.put_nowait(None)

            bucket_started = time.monotonic()
            succeeded_count = 0
            failed_count = 0
            successful_transport_request_count = 0
            attempted_transport_request_count = 0

            async def worker() -> None:
                nonlocal succeeded_count, failed_count, persisted
                nonlocal successful_transport_request_count
                nonlocal attempted_transport_request_count
                while True:
                    chunk = await queue.get()
                    try:
                        if chunk is None:
                            return
                        children = [
                            child
                            for _sample, _plan, child_samples in chunk
                            for child in child_samples
                        ]
                        result = await _forecast_bulk_with_retry(
                            async_client,
                            forecast_url=bulk_url,
                            model_id=model_id,
                            children=children,
                            max_attempts=max_attempts,
                        )
                        attempted_transport_request_count += int(result["attempts"])
                        forecasts = result["forecasts"]
                        if forecasts is None:
                            for sample, plan, _children in chunk:
                                failure_handle.write(
                                    json.dumps(
                                        {
                                            "model_id": model_id,
                                            "sample_id": sample["sample_id"],
                                            "request_group": label,
                                            "attempts": result["attempts"],
                                            "request_seconds": result[
                                                "elapsed_seconds"
                                            ],
                                            "error": result["error"],
                                            "input_adaptation": plan,
                                            "failed_target_index": None,
                                            "transport": "msgpack_bulk",
                                            "created_at": protocol.utc_now(),
                                        },
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                                failed_count += 1
                            continue

                        successful_transport_request_count += 1
                        offset = 0
                        for sample, plan, child_samples in chunk:
                            child_forecasts = forecasts[
                                offset : offset + len(child_samples)
                            ]
                            offset += len(child_samples)
                            if plan["target_mode"] == "independent_univariate":
                                forecast = np.concatenate(
                                    [
                                        child_forecast.T
                                        for child_forecast in child_forecasts
                                    ],
                                    axis=1,
                                )
                            else:
                                forecast = child_forecasts[0].T
                            row = prediction_row(
                                model_id,
                                "timer_service",
                                sample,
                                forecast,
                            )
                            row["input_adaptation"] = plan
                            output_handle.write(
                                json.dumps(
                                    row,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            succeeded_count += 1
                            persisted += 1
                            if persisted % 500 == 0:
                                output_handle.flush()
                                print(
                                    f"{model_id}: persisted={persisted}/"
                                    f"{compatible_count}",
                                    flush=True,
                                )
                    finally:
                        queue.task_done()

            tasks = [
                asyncio.create_task(worker())
                for _worker_index in range(http_concurrency)
            ]
            await asyncio.gather(*tasks)
            output_handle.flush()
            failure_handle.flush()
            elapsed = time.monotonic() - bucket_started
            if succeeded_count + failed_count != pending_count:
                raise RuntimeError(
                    f"{model_id}/{label} processed count mismatch: "
                    f"{succeeded_count}+{failed_count}!={pending_count}"
                )
            bucket_stats.append(
                {
                    "request_group": label,
                    "context_length": group_key[0],
                    "horizon": group_key[1],
                    "target_dim": group_key[2],
                    "covariate_dim": group_key[3],
                    "frequency": group_key[4],
                    "pending_count": pending_count,
                    "succeeded_count": succeeded_count,
                    "failed_count": failed_count,
                    "transport": "msgpack_bulk",
                    "task_batch_size": task_batch_size,
                    "views_per_http_request": views_per_request,
                    "expected_http_request_count": len(chunks),
                    "successful_http_request_count": (
                        successful_transport_request_count
                    ),
                    "attempted_http_request_count": (attempted_transport_request_count),
                    "logical_model_input_count": pending_count * child_count,
                    "elapsed_seconds": round(elapsed, 3),
                    "successful_tasks_per_second": round(
                        succeeded_count / max(elapsed, 1e-12),
                        3,
                    ),
                    **(
                        {
                            "target_request_count_per_view": int(group_key[5]),
                            "target_mode": str(group_key[6]),
                            "covariate_mode": str(group_key[7]),
                            "request_target_dim": int(group_key[8]),
                            "request_covariate_dim": int(group_key[9]),
                        }
                        if len(group_key) > 5
                        else {}
                    ),
                }
            )
            print(
                f"{model_id}: bucket {label} complete, "
                f"{succeeded_count}/{pending_count} in {elapsed:.1f}s "
                f"({successful_transport_request_count} bulk requests)",
                flush=True,
            )
    return bucket_stats


async def _run_model_requests_v8_group_scans(
    *,
    forecast_url: str,
    model_id: str,
    model: dict[str, Any],
    sample_path: Path,
    done: set[str],
    pending_groups: dict[tuple[Any, ...], int],
    http_concurrency: int,
    timeout_seconds: int,
    max_attempts: int,
    output_handle: Any,
    failure_handle: Any,
    compatible_count: int,
    initial_persisted: int,
    input_adaptation_policy: str | None = None,
) -> list[dict[str, Any]]:
    """Interleave homogeneous context buckets in one endpoint concurrency budget.

    Each individual msgpack request remains shape-homogeneous, while several
    context/target/covariate groups run concurrently. This is important for an
    eight-card endpoint: a small tail bucket must not leave the other replicas
    idle merely because groups were processed serially.
    """
    execution = MODEL_EXECUTION_CONFIG[model_id]
    if execution.get("transport") != "msgpack_bulk" or len(pending_groups) <= 1:
        return await _run_model_requests_v8_serial(
            forecast_url=forecast_url,
            model_id=model_id,
            model=model,
            sample_path=sample_path,
            done=done,
            pending_groups=pending_groups,
            http_concurrency=http_concurrency,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            output_handle=output_handle,
            failure_handle=failure_handle,
            compatible_count=compatible_count,
            initial_persisted=initial_persisted,
            input_adaptation_policy=input_adaptation_policy,
        )

    ordered_groups = sorted(pending_groups, key=request_group_sort_key)
    wave_size = min(http_concurrency, len(ordered_groups))
    all_stats: list[dict[str, Any]] = []
    for offset in range(0, len(ordered_groups), wave_size):
        wave = ordered_groups[offset : offset + wave_size]
        group_concurrency = max(1, http_concurrency // len(wave))
        results = await asyncio.gather(
            *(
                _run_model_requests_v8_serial(
                    forecast_url=forecast_url,
                    model_id=model_id,
                    model=model,
                    sample_path=sample_path,
                    done=done,
                    pending_groups={group_key: pending_groups[group_key]},
                    http_concurrency=group_concurrency,
                    timeout_seconds=timeout_seconds,
                    max_attempts=max_attempts,
                    output_handle=output_handle,
                    failure_handle=failure_handle,
                    compatible_count=compatible_count,
                    initial_persisted=initial_persisted,
                    input_adaptation_policy=input_adaptation_policy,
                )
                for group_key in wave
            )
        )
        for group_stats in results:
            all_stats.extend(group_stats)
    all_stats.sort(
        key=lambda row: (
            int(row["context_length"]),
            int(row["target_dim"]),
            int(row["covariate_dim"]),
        )
    )
    return all_stats


async def run_model_requests_v8(
    *,
    forecast_url: str,
    model_id: str,
    model: dict[str, Any],
    sample_path: Path,
    done: set[str],
    pending_groups: dict[tuple[Any, ...], int],
    http_concurrency: int,
    timeout_seconds: int,
    max_attempts: int,
    output_handle: Any,
    failure_handle: Any,
    compatible_count: int,
    initial_persisted: int,
    input_adaptation_policy: str | None = None,
) -> list[dict[str, Any]]:
    """Scan one shard once, then interleave homogeneous bulk requests."""
    execution = MODEL_EXECUTION_CONFIG[model_id]
    if execution.get("transport") != "msgpack_bulk":
        raise ValueError(f"paper cafe requires msgpack_bulk transport for {model_id}")

    task_batch_size = int(execution["task_batch_size"])
    ordered_groups = sorted(pending_groups, key=request_group_sort_key)
    states: dict[tuple[Any, ...], dict[str, Any]] = {
        group_key: {
            "label": request_group_label(group_key),
            "pending_count": int(pending_groups[group_key]),
            "items": [],
            "succeeded_count": 0,
            "failed_count": 0,
            "successful_request_count": 0,
            "attempted_request_count": 0,
            "started": None,
            "finished": None,
        }
        for group_key in ordered_groups
    }

    for sample in iter_forecast_samples(sample_path):
        plan = input_adaptation_plan(
            model,
            sample,
            policy_id=input_adaptation_policy,
        )
        if sample["sample_id"] in done or plan is None:
            continue
        group_key = request_group_key(sample, plan=plan)
        state = states.get(group_key)
        if state is not None:
            state["items"].append(
                (
                    sample,
                    plan,
                    adapted_request_samples(sample, plan),
                )
            )

    grouped_chunks: dict[tuple[Any, ...], list[list[Any]]] = {}
    for group_key in ordered_groups:
        state = states[group_key]
        items = state["items"]
        if len(items) != state["pending_count"]:
            raise RuntimeError(
                f"{model_id}/{state['label']} producer count "
                f"{len(items)} != {state['pending_count']}"
            )
        child_count = int(items[0][1]["target_request_count"])
        views_per_request = max(1, task_batch_size // child_count)
        chunks = [
            items[offset : offset + views_per_request]
            for offset in range(0, len(items), views_per_request)
        ]
        state["child_count"] = child_count
        state["views_per_request"] = views_per_request
        state["expected_request_count"] = len(chunks)
        grouped_chunks[group_key] = chunks

    jobs: list[tuple[tuple[Any, ...], list[Any]]] = []
    max_chunk_count = max(
        (len(chunks) for chunks in grouped_chunks.values()),
        default=0,
    )
    for chunk_index in range(max_chunk_count):
        for group_key in ordered_groups:
            chunks = grouped_chunks[group_key]
            if chunk_index < len(chunks):
                jobs.append((group_key, chunks[chunk_index]))
    print(
        f"{model_id}: one-scan scheduler prepared {len(jobs)} bulk "
        f"requests across {len(ordered_groups)} shape groups; "
        f"concurrency={http_concurrency}, batch={task_batch_size}",
        flush=True,
    )

    queue: asyncio.Queue[tuple[tuple[Any, ...], list[Any]] | None] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    for _worker_index in range(http_concurrency):
        queue.put_nowait(None)

    persisted = initial_persisted
    bulk_url = forecast_url.rsplit("/", 1)[0] + "/forecast/bulk"
    limits = httpx.Limits(
        max_connections=http_concurrency,
        max_keepalive_connections=http_concurrency,
        keepalive_expiry=120.0,
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        limits=limits,
        trust_env=False,
    ) as async_client:

        async def worker() -> None:
            nonlocal persisted
            while True:
                job = await queue.get()
                try:
                    if job is None:
                        return
                    group_key, chunk = job
                    state = states[group_key]
                    if state["started"] is None:
                        state["started"] = time.monotonic()
                    children = [
                        child
                        for _sample, _plan, child_samples in chunk
                        for child in child_samples
                    ]
                    result = await _forecast_bulk_with_retry(
                        async_client,
                        forecast_url=bulk_url,
                        model_id=model_id,
                        children=children,
                        max_attempts=max_attempts,
                    )
                    state["attempted_request_count"] += int(result["attempts"])
                    forecasts = result["forecasts"]
                    if forecasts is None:
                        for sample, plan, _children in chunk:
                            failure_handle.write(
                                json.dumps(
                                    {
                                        "model_id": model_id,
                                        "sample_id": sample["sample_id"],
                                        "request_group": state["label"],
                                        "attempts": result["attempts"],
                                        "request_seconds": result["elapsed_seconds"],
                                        "error": result["error"],
                                        "input_adaptation": plan,
                                        "failed_target_index": None,
                                        "transport": "msgpack_bulk",
                                        "created_at": protocol.utc_now(),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            state["failed_count"] += 1
                        state["finished"] = time.monotonic()
                        continue

                    state["successful_request_count"] += 1
                    child_offset = 0
                    for sample, plan, child_samples in chunk:
                        child_forecasts = forecasts[
                            child_offset : child_offset + len(child_samples)
                        ]
                        child_offset += len(child_samples)
                        if plan["target_mode"] == "independent_univariate":
                            forecast = np.concatenate(
                                [
                                    child_forecast.T
                                    for child_forecast in child_forecasts
                                ],
                                axis=1,
                            )
                        else:
                            forecast = child_forecasts[0].T
                        row = prediction_row(
                            model_id,
                            "timer_service",
                            sample,
                            forecast,
                        )
                        row["input_adaptation"] = plan
                        output_handle.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        state["succeeded_count"] += 1
                        persisted += 1
                        if persisted % 500 == 0:
                            output_handle.flush()
                    state["finished"] = time.monotonic()
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(worker()) for _worker_index in range(http_concurrency)
        ]
        await asyncio.gather(*workers)

    output_handle.flush()
    failure_handle.flush()
    bucket_stats: list[dict[str, Any]] = []
    for group_key in ordered_groups:
        state = states[group_key]
        pending_count = int(state["pending_count"])
        succeeded_count = int(state["succeeded_count"])
        failed_count = int(state["failed_count"])
        if succeeded_count + failed_count != pending_count:
            raise RuntimeError(
                f"{model_id}/{state['label']} processed count mismatch: "
                f"{succeeded_count}+{failed_count}!={pending_count}"
            )
        elapsed = max(
            float(state["finished"] - state["started"]),
            1e-12,
        )
        bucket_stats.append(
            {
                "request_group": state["label"],
                "context_length": group_key[0],
                "horizon": group_key[1],
                "target_dim": group_key[2],
                "covariate_dim": group_key[3],
                "frequency": group_key[4],
                "pending_count": pending_count,
                "succeeded_count": succeeded_count,
                "failed_count": failed_count,
                "transport": "msgpack_bulk",
                "task_batch_size": task_batch_size,
                "views_per_http_request": int(state["views_per_request"]),
                "expected_http_request_count": int(state["expected_request_count"]),
                "successful_http_request_count": int(state["successful_request_count"]),
                "attempted_http_request_count": int(state["attempted_request_count"]),
                "logical_model_input_count": (
                    pending_count * int(state["child_count"])
                ),
                "elapsed_seconds": round(elapsed, 3),
                "successful_tasks_per_second": round(
                    succeeded_count / elapsed,
                    3,
                ),
                **(
                    {
                        "target_request_count_per_view": int(group_key[5]),
                        "target_mode": str(group_key[6]),
                        "covariate_mode": str(group_key[7]),
                        "request_target_dim": int(group_key[8]),
                        "request_covariate_dim": int(group_key[9]),
                    }
                    if len(group_key) > 5
                    else {}
                ),
            }
        )
    return bucket_stats


def validated_file_record_path(
    record: dict[str, Any],
    *,
    label: str,
    validate_row_count: bool = False,
) -> Path:
    """Resolve a manifest file record only after validating its content."""

    path_value = record.get("path")
    if not path_value:
        raise ValueError(f"{label} file record is missing path")
    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(f"{label} file is missing: {path}")
    expected_bytes = record.get("bytes")
    if expected_bytes is None or int(expected_bytes) != path.stat().st_size:
        raise ValueError(f"{label} file byte-size mismatch: {path}")
    expected_sha256 = record.get("sha256")
    if not expected_sha256 or str(expected_sha256) != protocol.file_sha256(path):
        raise ValueError(f"{label} file hash mismatch: {path}")
    if validate_row_count:
        expected_rows = record.get("row_count")
        if expected_rows is None:
            raise ValueError(f"{label} file record is missing row_count")
        observed_rows = count_jsonl(path)
        if observed_rows != int(expected_rows):
            raise ValueError(
                f"{label} file row-count mismatch: "
                f"{observed_rows} != {expected_rows}"
            )
    return path


def formal_real_anchor_source_record(
    calibration_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Require the auxiliary real-anchor source for a formal inference run."""

    if calibration_bundle is None:
        raise ValueError(
            "formal Paper-cafe inference requires calibration_bundle.json "
            "with real_anchor_masters"
        )
    record = calibration_bundle.get("files", {}).get("real_anchor_masters")
    if not isinstance(record, dict):
        raise ValueError(
            "formal Paper-cafe inference calibration bundle is missing "
            "files.real_anchor_masters"
        )
    validated_file_record_path(
        record,
        label="real-anchor source",
    )
    return record


def optional_real_anchored_source_record(
    generation_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve the optional real-anchored generation component."""

    files = generation_manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("generation manifest is missing files")
    record = files.get(REAL_ANCHORED_GENERATION_FILE_KEY)
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError(
            "generation manifest real-anchored source file record must be "
            "an object"
        )
    validated_file_record_path(
        record,
        label="real-anchored counterfactual source",
        validate_row_count=True,
    )
    return record


def validate_inference_task_manifest_files(
    task_manifest: dict[str, Any],
) -> Path:
    """Validate the combined task and all independently consumed components."""

    if task_manifest.get("schema_version") not in {
        "cafe.inference_task_manifest.v1",
        "cafe.inference_task_manifest.v2",
    }:
        raise ValueError("unsupported Paper-cafe inference task manifest")
    task_components = task_manifest.get("task_components")
    if not isinstance(task_components, dict):
        raise ValueError("inference task manifest is missing task_components")
    component_paths: dict[str, Path] = {}
    for name in ("synthetic", "real_anchors"):
        record = task_components.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"inference task manifest is missing {name} component")
        component_paths[name] = validated_file_record_path(
            record,
            label=f"inference task component {name}",
            validate_row_count=True,
        )
    real_anchored_record = task_components.get(
        REAL_ANCHORED_GENERATION_FILE_KEY
    )
    if real_anchored_record is not None:
        if not isinstance(real_anchored_record, dict):
            raise ValueError(
                "inference task manifest real-anchored component must be an "
                "object"
            )
        component_paths[REAL_ANCHORED_GENERATION_FILE_KEY] = (
            validated_file_record_path(
                real_anchored_record,
                label="inference task component real_anchored_counterfactuals",
                validate_row_count=True,
            )
        )
    task_record = task_manifest.get("task_file")
    if not isinstance(task_record, dict):
        raise ValueError("inference task manifest is missing task_file")
    task_path = validated_file_record_path(
        task_record,
        label="combined inference task",
        validate_row_count=True,
    )
    synthetic_count = int(task_components["synthetic"]["row_count"])
    real_count = int(task_components["real_anchors"]["row_count"])
    real_anchored_count = (
        0
        if real_anchored_record is None
        else int(real_anchored_record["row_count"])
    )
    if synthetic_count != int(task_manifest.get("synthetic_view_count", -1)):
        raise ValueError("synthetic inference task count disagrees with manifest")
    if real_count != int(task_manifest.get("real_anchor_view_count", -1)):
        raise ValueError("real-anchor inference task count disagrees with manifest")
    declared_real_anchored_count = int(
        task_manifest.get("real_anchored_view_count", 0)
    )
    if real_anchored_count != declared_real_anchored_count:
        raise ValueError(
            "real-anchored inference task count disagrees with manifest"
        )
    if (
        synthetic_count + real_count + real_anchored_count
        != int(task_manifest.get("view_count", -1))
    ):
        raise ValueError("combined inference task count disagrees with components")
    if int(task_record["row_count"]) != int(task_manifest["view_count"]):
        raise ValueError("combined inference task record count mismatch")
    return task_path


def prepare_view_tasks(
    generation_manifest: dict[str, Any],
    *,
    inference_dir: Path,
    calibration_bundle: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    source_records = [
        generation_manifest["files"]["clean"],
        generation_manifest["files"]["robustness"],
        generation_manifest["files"]["input_ablations"],
    ]
    masters = (
        row
        for record in source_records
        for row in protocol.iter_jsonl(Path(record["path"]))
    )
    synthetic_task_path = inference_dir / "synthetic_forecast_views.jsonl"
    synthetic_view_count = protocol.write_jsonl(
        synthetic_task_path,
        protocol.iter_master_views(masters),
    )
    real_anchored_source_record = optional_real_anchored_source_record(
        generation_manifest
    )
    real_anchored_task_path = (
        inference_dir / "real_anchored_forecast_views.jsonl"
    )

    def real_anchored_tasks() -> Iterator[dict[str, Any]]:
        if real_anchored_source_record is None:
            return
        source_masters = protocol.iter_jsonl(
            Path(real_anchored_source_record["path"])
        )
        for view in protocol.iter_master_views(
            source_masters,
            context_lengths=(protocol.FIXED_CONTEXT_LENGTH,),
        ):
            row = dict(view)
            row["schema_version"] = (
                "cafe.real_anchored_forecast_view.v2"
            )
            row.setdefault(
                "evaluation_table",
                REAL_ANCHORED_BENCHMARK_TRACK,
            )
            row["benchmark_track"] = REAL_ANCHORED_BENCHMARK_TRACK
            row["context_policy"] = (
                f"fixed_l{protocol.FIXED_CONTEXT_LENGTH}"
            )
            row["context_policy_candidates"] = [
                protocol.FIXED_CONTEXT_LENGTH
            ]
            yield row

    real_anchored_view_count = protocol.write_jsonl(
        real_anchored_task_path,
        real_anchored_tasks(),
    )
    real_source_record = (
        (calibration_bundle or {}).get("files", {}).get("real_anchor_masters")
    )
    if real_source_record is not None:
        if not isinstance(real_source_record, dict):
            raise ValueError("real-anchor source file record must be an object")
        validated_file_record_path(
            real_source_record,
            label="real-anchor source",
        )
    real_task_path = inference_dir / "real_anchor_views.jsonl"

    def real_anchor_tasks() -> Iterator[dict[str, Any]]:
        if real_source_record is None:
            return
        for master in protocol.iter_jsonl(Path(real_source_record["path"])):
            row = dict(master)
            row["schema_version"] = "cafe.real_anchor_forecast_view.v1"
            row["evaluation_table"] = "real_anchor_forecast"
            row["view_id"] = row["sample_id"]
            row["context_policy"] = f"fixed_l{protocol.REAL_CALIBRATION_CONTEXT_LENGTH}"
            row["context_policy_candidates"] = [
                protocol.REAL_CALIBRATION_CONTEXT_LENGTH
            ]
            row["scoring_target_semantics"] = "held_out_real_future"
            yield row

    real_anchor_view_count = protocol.write_jsonl(
        real_task_path,
        real_anchor_tasks(),
    )
    task_path = inference_dir / "forecast_views.jsonl"
    view_count = protocol.write_jsonl(
        task_path,
        (
            row
            for path in (
                synthetic_task_path,
                real_anchored_task_path,
                real_task_path,
            )
            for row in protocol.iter_jsonl(path)
        ),
    )
    task_components = {
        "synthetic": {
            **protocol.file_record(synthetic_task_path),
            "row_count": synthetic_view_count,
        },
        "real_anchors": {
            **protocol.file_record(real_task_path),
            "row_count": real_anchor_view_count,
        },
    }
    if real_anchored_source_record is not None:
        task_components[REAL_ANCHORED_GENERATION_FILE_KEY] = {
            **protocol.file_record(real_anchored_task_path),
            "row_count": real_anchored_view_count,
        }
    manifest = {
        "schema_version": "cafe.inference_task_manifest.v2",
        "created_at": protocol.utc_now(),
        "generation_config_sha256": generation_manifest["config_sha256"],
        "generation_files": source_records + (
            []
            if real_anchored_source_record is None
            else [real_anchored_source_record]
        ),
        "calibration_bundle_content_sha256": (
            None
            if calibration_bundle is None
            else calibration_bundle.get("bundle_content_sha256")
        ),
        "real_anchor_source": real_source_record,
        "real_anchored_source": real_anchored_source_record,
        "context_lengths": list(protocol.VIEW_CONTEXT_LENGTHS),
        "fixed_context_length": protocol.FIXED_CONTEXT_LENGTH,
        "synthetic_view_count": synthetic_view_count,
        "real_anchored_view_count": real_anchored_view_count,
        "real_anchor_view_count": real_anchor_view_count,
        "view_count": view_count,
        "task_components": task_components,
        "task_file": {
            **protocol.file_record(task_path),
            "row_count": view_count,
        },
        "mase_policy": (
            "synthetic views share clean L336 denominator; real-anchored "
            "counterfactual pairs share their unmodified real L336 history "
            "denominator; real anchors use their own clean "
            f"L{protocol.REAL_CALIBRATION_CONTEXT_LENGTH} history denominator"
        ),
    }
    protocol.write_json(inference_dir / "task_manifest.json", manifest)
    return task_path, manifest


def write_real_anchor_prediction_subset(
    canonical_prediction_path: Path,
    *,
    real_anchor_task_path: Path,
    output_path: Path,
) -> int:
    """Materialize an auxiliary real-anchor result without duplicating tasks."""

    real_ids = {
        str(row["sample_id"]) for row in protocol.iter_jsonl(real_anchor_task_path)
    }
    if not real_ids:
        return protocol.write_jsonl(output_path, ())
    return protocol.write_jsonl(
        output_path,
        (
            row
            for row in protocol.iter_jsonl(canonical_prediction_path)
            if str(row["sample_id"]) in real_ids
        ),
    )


def health_catalog(
    endpoint: str,
    api_prefix: str,
) -> tuple[str, dict[str, dict[str, Any]]] | None:
    client = TimerServiceClient(endpoint, api_prefix, timeout_seconds=30)
    try:
        catalog = {str(row["model_id"]): row for row in client.list_models()}
        return endpoint, catalog
    except Exception as error:  # noqa: BLE001
        with _PRINT_LOCK:
            print(f"endpoint unavailable {endpoint}: {type(error).__name__}: {error}")
        return None
    finally:
        client.close()


def validate_catalog_input_capabilities(
    catalogs: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    *,
    contract_path: Path | None,
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] | None = None
    if contract_path is not None:
        contract = protocol.read_json(contract_path.resolve())
        configured = contract.get("config", {}).get("resolved_model_input_capabilities")
        if not isinstance(configured, dict):
            raise ValueError(
                "inference contract is missing resolved model input capabilities"
            )
        expected = {
            str(model_id): dict(capability)
            for model_id, capability in configured.items()
        }

    resolved: dict[str, dict[str, Any]] = {}
    for model_id in models:
        observations = [
            (endpoint, resolve_input_capability(catalog[model_id]))
            for endpoint, catalog in sorted(catalogs.items())
            if model_id in catalog
        ]
        if not observations:
            raise ValueError(f"model {model_id!r} unavailable on all services")
        first = observations[0][1]
        if any(value != first for _endpoint, value in observations[1:]):
            raise ValueError(
                f"inconsistent live input capabilities for {model_id}: "
                + ", ".join(endpoint for endpoint, _value in observations)
            )
        if expected is not None:
            if model_id not in expected:
                raise ValueError(
                    f"inference contract is missing input capability for {model_id}"
                )
            if first != expected[model_id]:
                raise ValueError(
                    f"live input capability changed after the inference contract "
                    f"was frozen for {model_id}"
                )
        resolved[model_id] = first
    return resolved


def model_root(inference_dir: Path, model_id: str) -> Path:
    return inference_dir / "model_shards" / safe_filename(model_id)


def model_part_root(
    inference_dir: Path,
    model_id: str,
    part_index: int,
) -> Path:
    return model_root(inference_dir, model_id) / "parts" / (f"part_{part_index:03d}")


def compact_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "schema_version": "cafe.inference_prediction.v1",
        "model_id": str(row["model_id"]),
        "sample_id": str(row["sample_id"]),
        "forecast": row["forecast"],
    }
    if row.get("input_adaptation") is not None:
        compact["input_adaptation"] = row["input_adaptation"]
    return compact


def canonical_prediction_file_is_compact(path: Path) -> bool:
    """Check the writer format without reparsing and rewriting the whole file."""

    allowed_keys = {
        "schema_version",
        "model_id",
        "sample_id",
        "forecast",
        "input_adaptation",
    }
    required_keys = allowed_keys - {"input_adaptation"}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys = set(json.loads(line))
                return required_keys <= keys <= allowed_keys
    return True


def compact_canonical_prediction_file(path: Path) -> int:
    return protocol.write_jsonl(
        path,
        (compact_prediction_row(row) for row in protocol.iter_jsonl(path)),
    )


def cleanup_completed_model_intermediates(
    inference_dir: Path,
    model_id: str,
) -> int:
    """Remove restart-only artifacts after a canonical model file exists."""

    canonical_path = prediction_path_for(
        model_root(inference_dir, model_id),
        model_id,
    )
    if not canonical_path.is_file():
        return 0
    removed_bytes = 0
    parts_root = model_root(inference_dir, model_id) / "parts"
    for part_root in sorted(parts_root.glob("part_*")):
        for directory_name in ("predictions", "failures"):
            directory = part_root / directory_name
            if directory.is_dir():
                removed_bytes += sum(
                    path.stat().st_size
                    for path in directory.rglob("*")
                    if path.is_file()
                )
                shutil.rmtree(directory)
    shard_dir = _model_task_shard_manifest_path(
        inference_dir,
        model_id,
    ).parent
    if shard_dir.is_dir():
        removed_bytes += sum(
            path.stat().st_size for path in shard_dir.rglob("*") if path.is_file()
        )
        shutil.rmtree(shard_dir)
    return removed_bytes


def task_sample_ids(path: Path) -> set[str]:
    sample_ids: set[str] = set()
    for row in protocol.iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate inference task sample_id: {sample_id}")
        sample_ids.add(sample_id)
    return sample_ids


def canonical_prediction_sample_ids(
    path: Path,
    *,
    model_id: str,
) -> set[str]:
    """Return canonical IDs while rejecting duplicate or foreign model rows."""

    sample_ids: set[str] = set()
    for row in protocol.iter_jsonl(path):
        observed_model = str(row.get("model_id", ""))
        if observed_model != model_id:
            raise ValueError(
                f"canonical prediction model mismatch for {path}: "
                f"{observed_model!r} != {model_id!r}"
            )
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(
                f"duplicate canonical prediction for {model_id}: {sample_id}"
            )
        sample_ids.add(sample_id)
    return sample_ids


def cached_complete_model_records(
    inference_dir: Path,
    *,
    expected_sample_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Reuse a complete manifest when its canonical files are unchanged."""

    manifest_path = inference_dir / "inference_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = protocol.read_json(manifest_path)
    if manifest.get(
        "schema_version"
    ) != "cafe.inference_manifest.v1" or not manifest.get("complete"):
        return {}
    statuses = {str(row["model_id"]): row for row in manifest.get("statuses", [])}
    cached: dict[str, dict[str, Any]] = {}
    for record in manifest.get("predictions", {}).get("files", []):
        model_id = str(record["model_id"])
        status = statuses.get(model_id, {})
        canonical_path = prediction_path_for(
            model_root(inference_dir, model_id),
            model_id,
        )
        if (
            status.get("status") == "complete"
            and int(status.get("succeeded_original_view_count", -1))
            == len(expected_sample_ids)
            and int(record.get("row_count", -1)) == len(expected_sample_ids)
            and Path(record.get("path", "")).resolve() == canonical_path.resolve()
            and canonical_path.is_file()
            and int(record.get("bytes", -1)) == canonical_path.stat().st_size
            and str(record.get("sha256", "")) == protocol.file_sha256(canonical_path)
            and canonical_prediction_sample_ids(
                canonical_path,
                model_id=model_id,
            )
            == expected_sample_ids
        ):
            cached[model_id] = dict(record)
    return cached


def _model_task_shard_manifest_path(
    inference_dir: Path,
    model_id: str,
) -> Path:
    return (
        inference_dir / "model_task_shards" / safe_filename(model_id) / "manifest.json"
    )


def _model_task_shard_manifest_is_reusable(
    manifest: dict[str, Any],
    *,
    model_id: str,
    task_path: Path,
    part_weights: list[int],
) -> bool:
    if manifest.get("model_id") != model_id or manifest.get(
        "source_task_sha256"
    ) != protocol.file_sha256(task_path):
        return False
    if list(manifest.get("part_weights") or []) != part_weights:
        return False
    parts = list(manifest.get("parts") or [])
    if len(parts) != int(manifest.get("part_count", -1)):
        return False
    for part in parts:
        path = Path(part["path"])
        if not path.is_file() or protocol.file_sha256(path) != part["sha256"]:
            return False
    return True


def prepare_model_task_shards(
    task_path: Path,
    *,
    model_id: str,
    part_count: int,
    inference_dir: Path,
    part_weights: list[int] | None = None,
) -> dict[str, Any]:
    """Create or reuse deterministic task parts for one model.

    An existing manifest is reused when its source hash and capacity weights
    match. Capacity changes rebuild the shards so one endpoint still receives
    one large, weighted part instead of many tiny sequential parts.
    """

    if part_count < 1:
        raise ValueError("model task sharding requires at least one part")
    if part_weights is None:
        part_weights = [1] * part_count
    if len(part_weights) != part_count or any(weight < 1 for weight in part_weights):
        raise ValueError("part_weights must contain one positive weight per part")
    manifest_path = _model_task_shard_manifest_path(
        inference_dir,
        model_id,
    )
    if manifest_path.exists():
        manifest = protocol.read_json(manifest_path)
        if _model_task_shard_manifest_is_reusable(
            manifest,
            model_id=model_id,
            task_path=task_path,
            part_weights=part_weights,
        ):
            return manifest

    shard_dir = manifest_path.parent
    shard_dir.mkdir(parents=True, exist_ok=True)
    part_paths = [shard_dir / f"part_{index:03d}.jsonl" for index in range(part_count)]
    handles = [path.open("w", encoding="utf-8") for path in part_paths]
    counts = [0] * part_count
    total_weight = sum(part_weights)
    try:
        for row in protocol.iter_jsonl(task_path):
            slot = (
                protocol.stable_seed(
                    "cafe.model_task_shard",
                    model_id,
                    str(row["sample_id"]),
                )
                % total_weight
            )
            cumulative_weight = 0
            part_index = 0
            for candidate, weight in enumerate(part_weights):
                cumulative_weight += weight
                if slot < cumulative_weight:
                    part_index = candidate
                    break
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
        raise ValueError(f"model task partition for {model_id} produced an empty part")
    parts = [
        {
            "part_index": index,
            "row_count": counts[index],
            **protocol.file_record(path),
        }
        for index, path in enumerate(part_paths)
    ]
    manifest = {
        "schema_version": "cafe.model_task_shard_manifest.v1",
        "created_at": protocol.utc_now(),
        "model_id": model_id,
        "part_count": part_count,
        "part_weights": part_weights,
        "partition_policy": (
            "stable_hash_of_policy_model_and_sample_id_weighted_by_endpoint"
        ),
        "source_task_sha256": protocol.file_sha256(task_path),
        "source_task_row_count": sum(counts),
        "parts": parts,
    }
    protocol.write_json(manifest_path, manifest)
    return manifest


def plan_model_phase(
    model_id: str,
    services: list[tuple[str, dict[str, dict[str, Any]]]],
    *,
    task_path: Path,
    inference_dir: Path,
    endpoint_profiles: dict[str, EndpointProfile] | None = None,
) -> tuple[dict[str, list[InferenceWork]], dict[str, Any]]:
    eligible = sorted(endpoint for endpoint, catalog in services if model_id in catalog)
    if not eligible:
        raise ValueError(f"model {model_id!r} unavailable on all services")
    capacities = {
        endpoint: (
            endpoint_profiles[endpoint].capacity_for(model_id)
            if endpoint_profiles is not None
            else 1
        )
        for endpoint in eligible
    }
    # Keep shards coarse enough that every homogeneous context bucket can fill
    # the model batch. Exact rational normalization (for example 3.65 ->
    # 73/20) creates hundreds of tiny parts and destroys GPU utilization.
    target_part_count = max(
        len(eligible),
        int(sum(capacities.values()) + 0.5),
    )
    normalized_capacities = {endpoint: 1 for endpoint in eligible}
    for _part_index in range(target_part_count - len(eligible)):
        endpoint = min(
            eligible,
            key=lambda candidate: (
                normalized_capacities[candidate] / capacities[candidate],
                candidate,
            ),
        )
        normalized_capacities[endpoint] += 1
    manifest = prepare_model_task_shards(
        task_path,
        model_id=model_id,
        part_count=len(eligible),
        inference_dir=inference_dir,
        part_weights=[normalized_capacities[endpoint] for endpoint in eligible],
    )
    work = {endpoint: [] for endpoint in eligible}
    for part in sorted(
        manifest["parts"],
        key=lambda row: int(row["part_index"]),
    ):
        part_index = int(part["part_index"])
        endpoint = eligible[part_index]
        work[endpoint].append(
            InferenceWork(
                model_id=model_id,
                sample_path=Path(part["path"]),
                output_dir=model_part_root(
                    inference_dir,
                    model_id,
                    part_index,
                ),
                work_id=f"{model_id}__part_{part_index:03d}",
                part_index=part_index,
            )
        )
    return work, manifest


def run_service_queue(
    endpoint: str,
    work_items: list[InferenceWork],
    catalog: dict[str, dict[str, Any]],
    *,
    args: argparse.Namespace,
    endpoint_profile: EndpointProfile,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    client = TimerServiceClient(
        endpoint,
        args.api_prefix,
        timeout_seconds=30,
    )
    try:
        if len({item.model_id for item in work_items}) > 1:
            raise ValueError(
                "one service queue may contain parts for only one model phase"
            )
        if not args.keep_loaded_between_runs:
            client.unload_all_loaded()
        for item in work_items:
            model_id = item.model_id
            model_dir = item.output_dir
            for directory in ("predictions", "failures"):
                (model_dir / directory).mkdir(parents=True, exist_ok=True)
            execution = dict(MODEL_EXECUTION_CONFIG[model_id])
            endpoint_concurrency = endpoint_profile.http_concurrency_for(
                model_id,
                execution["http_concurrency"],
            )
            execution["http_concurrency"] = max(
                1,
                endpoint_concurrency // max(1, args.request_concurrency_divisor),
            )
            with _PRINT_LOCK:
                print(
                    f"{endpoint}: starting {item.work_id}, "
                    f"devices={endpoint_profile.devices}, "
                    f"capacity={endpoint_profile.capacity_for(model_id)}, "
                    f"replicas={execution['replicas_per_device']}, "
                    f"concurrency={execution['http_concurrency']}, "
                    f"transport={execution.get('transport', 'json')}, "
                    f"batch={execution.get('task_batch_size', 1)}",
                    flush=True,
                )
            started = time.monotonic()
            try:
                status = run_one_model(
                    client,
                    catalog[model_id],
                    output_dir=model_dir,
                    execution=execution,
                    devices=endpoint_profile.devices,
                    request_max_attempts=args.max_attempts,
                    forecast_timeout_seconds=args.forecast_timeout_seconds,
                    load_timeout_seconds=args.load_timeout_seconds,
                    keep_loaded=True,
                    sample_path=item.sample_path,
                    prediction_kind="synthetic",
                    status_filename="model_status.json",
                    input_adaptation_policy=INPUT_ADAPTATION_POLICY_ID,
                )
                if execution.get("transport") == "msgpack_bulk":
                    bucket_stats = list(status.get("bucket_stats") or [])
                    status["successful_http_request_count"] = sum(
                        int(row.get("successful_http_request_count", 0))
                        for row in bucket_stats
                    )
                    status["attempted_http_request_count_this_attempt"] = sum(
                        int(row.get("attempted_http_request_count", 0))
                        for row in bucket_stats
                    )
                    status.setdefault("execution", {}).update(
                        {
                            "transport": "msgpack_bulk",
                            "task_batch_size": int(execution["task_batch_size"]),
                        }
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
            status["part_index"] = item.part_index
            status["sample_path"] = str(item.sample_path)
            status["endpoint_profile"] = endpoint_profile.as_dict()
            statuses.append(status)
            protocol.write_json(
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
        if not args.keep_loaded_between_runs:
            try:
                client.unload_all_loaded()
            except Exception:  # noqa: BLE001
                pass
        client.close()
    return statuses


def consolidate_model_predictions(
    inference_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    model_id = str(manifest["model_id"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    complete = True
    for part in manifest["parts"]:
        part_index = int(part["part_index"])
        part_root = model_part_root(
            inference_dir,
            model_id,
            part_index,
        )
        path = prediction_path_for(part_root, model_id)
        if not path.exists():
            complete = False
            continue
        part_rows = list(protocol.iter_jsonl(path))
        expected_ids = {
            str(row["sample_id"]) for row in protocol.iter_jsonl(Path(part["path"]))
        }
        observed_ids = {str(row["sample_id"]) for row in part_rows}
        if len(part_rows) != int(part["row_count"]) or observed_ids != expected_ids:
            complete = False
        for row in part_rows:
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(
                    f"duplicate model prediction for {model_id}: {sample_id}"
                )
            seen.add(sample_id)
            rows.append(row)
    if not complete:
        return False
    expected = int(manifest["source_task_row_count"])
    if len(rows) != expected:
        raise ValueError(
            f"model prediction coverage mismatch: {len(rows)} != {expected}"
        )
    rows.sort(key=lambda row: str(row["sample_id"]))
    canonical_path = prediction_path_for(
        model_root(inference_dir, model_id),
        model_id,
    )
    written = protocol.write_jsonl(
        canonical_path,
        (compact_prediction_row(row) for row in rows),
    )
    if written != expected:
        raise RuntimeError(
            f"canonical model prediction count mismatch: {written} != {expected}"
        )
    removed_bytes = cleanup_completed_model_intermediates(
        inference_dir,
        model_id,
    )
    print(
        f"{model_id}: removed {removed_bytes / (1024**2):.1f} MiB "
        "of completed endpoint intermediates",
        flush=True,
    )
    return True


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
        matching = [row for row in work_statuses if row.get("model_id") == model_id]
        canonical_path = prediction_path_for(
            inference_dir / "model_shards" / safe_filename(model_id),
            model_id,
        )
        observed = count_jsonl(canonical_path) if canonical_path.exists() else 0
        status: dict[str, Any] = {
            "model_id": model_id,
            "status": (
                "complete"
                if observed == expected_view_count
                and all(row.get("status") == "complete" for row in matching)
                else "failed"
            ),
            "expected_original_view_count": expected_view_count,
            "succeeded_original_view_count": observed,
            "endpoints": sorted(
                {str(row["endpoint"]) for row in matching if row.get("endpoint")}
            ),
            "work_statuses": matching,
            "prediction_path": str(canonical_path),
        }
        for field in count_fields:
            values = [int(row[field]) for row in matching if row.get(field) is not None]
            if values:
                status[field] = sum(values)
        output.append(status)
        protocol.write_json(
            inference_dir
            / "model_shards"
            / safe_filename(model_id)
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
            inference_dir / "model_shards" / safe_filename(model_id),
            model_id,
        )
        if not path.exists():
            continue
        for row in protocol.iter_jsonl(path):
            key = (str(row["model_id"]), str(row["sample_id"]))
            if key in seen:
                raise ValueError(f"duplicate inference prediction {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["model_id"], row["sample_id"]))
    merged_path = inference_dir / "predictions.jsonl"
    count = protocol.write_jsonl(merged_path, rows)
    return merged_path, count


def _unload_accelerator_models(
    endpoints: list[str],
    api_prefix: str,
) -> None:
    def unload(endpoint: str) -> None:
        client = TimerServiceClient(
            endpoint,
            api_prefix,
            timeout_seconds=30,
        )
        try:
            client.unload_all_loaded()
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = [executor.submit(unload, endpoint) for endpoint in endpoints]
        for future in as_completed(futures):
            future.result()


def _single_dataset_child_arguments(
    args: argparse.Namespace,
    *,
    dataset_id: str,
    models: list[str],
    keep_loaded: bool,
    request_concurrency_divisor: int = 1,
) -> list[str]:
    arguments = [
        "--dataset-id",
        dataset_id,
        "--output-root",
        str(args.output_root.resolve()),
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
        "--models",
        *models,
        "--endpoints",
        *args.endpoints,
        "--api-prefix",
        args.api_prefix,
        "--load-timeout-seconds",
        str(args.load_timeout_seconds),
        "--forecast-timeout-seconds",
        str(args.forecast_timeout_seconds),
        "--max-attempts",
        str(args.max_attempts),
        "--resume",
        "--request-concurrency-divisor",
        str(request_concurrency_divisor),
        *endpoint_topology_cli_arguments(args),
    ]
    if keep_loaded:
        arguments.append("--keep-loaded-between-runs")
    if args.input_capability_contract is not None:
        arguments.extend(
            (
                "--input-capability-contract",
                str(args.input_capability_contract.resolve()),
            )
        )
    return arguments


def _prepare_model_dataset(
    args: argparse.Namespace,
    *,
    dataset_id: str,
    model_id: str,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *_single_dataset_child_arguments(
                args,
                dataset_id=dataset_id,
                models=[model_id],
                keep_loaded=True,
            ),
            "--prepare-only",
        ],
        cwd=protocol.REPO_ROOT,
        check=True,
    )


def _prepare_model_datasets(
    args: argparse.Namespace,
    *,
    dataset_ids: list[str],
    model_id: str,
) -> None:
    pending_dataset_ids = [
        dataset_id
        for dataset_id in dataset_ids
        if not _model_dataset_is_complete(
            args,
            dataset_id=dataset_id,
            model_id=model_id,
        )
    ]
    if not pending_dataset_ids:
        print(
            f"{model_id}: all dataset canonical predictions already exist; "
            "preparation skipped",
            flush=True,
        )
        return
    workers = min(max(1, args.preprocess_workers), len(pending_dataset_ids))
    print(
        f"{model_id}: preparing {len(pending_dataset_ids)} datasets with "
        f"{workers} CPU workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _prepare_model_dataset,
                args,
                dataset_id=dataset_id,
                model_id=model_id,
            ): dataset_id
            for dataset_id in pending_dataset_ids
        }
        for future in as_completed(futures):
            future.result()


def _model_dataset_is_complete(
    args: argparse.Namespace,
    *,
    dataset_id: str,
    model_id: str,
) -> bool:
    shard_name = f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    path = prediction_path_for(
        model_root(
            args.output_root.resolve() / dataset_id / "03_inference" / shard_name,
            model_id,
        ),
        model_id,
    )
    return path.is_file()


def _run_model_dataset(
    args: argparse.Namespace,
    *,
    dataset_id: str,
    model_id: str,
    request_concurrency_divisor: int,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *_single_dataset_child_arguments(
                args,
                dataset_id=dataset_id,
                models=[model_id],
                keep_loaded=True,
                request_concurrency_divisor=request_concurrency_divisor,
            ),
        ],
        cwd=protocol.REPO_ROOT,
        check=True,
    )


def run_model_major_controller(args: argparse.Namespace) -> int:
    dataset_ids = list(dict.fromkeys(args.dataset_ids or []))
    if not dataset_ids:
        raise ValueError("--dataset-ids requires at least one dataset")
    for dataset_id in dataset_ids:
        protocol.resolve_dataset(dataset_id)

    status_path = args.output_root.resolve() / "model_major_inference_status.json"
    completed: list[dict[str, Any]] = []
    status = {
        "schema_version": "cafe.model_major_inference_status.v1",
        "started_at": protocol.utc_now(),
        "state": "running",
        "dataset_ids": dataset_ids,
        "models": list(args.models),
        "execution": {
            "preprocess_workers": int(args.preprocess_workers),
            "dataset_parallelism_by_model": {
                model_id: int(MODEL_MAJOR_DATASET_PARALLELISM.get(model_id, 1))
                for model_id in args.models
            },
            "request_concurrency_policy": (
                "divide_each_endpoint_model_http_concurrency_by_the_number_"
                "of_concurrent_datasets"
            ),
            "first_pending_dataset_role": (
                "load_once_before_concurrent_dataset_batches"
            ),
        },
        "completed": completed,
    }
    protocol.write_json(status_path, status)
    try:
        for model_index, model_id in enumerate(args.models, start=1):
            print(
                f"model-major phase {model_index}/{len(args.models)}: "
                f"{model_id} across {len(dataset_ids)} datasets",
                flush=True,
            )
            _prepare_model_datasets(
                args,
                dataset_ids=dataset_ids,
                model_id=model_id,
            )
            _unload_accelerator_models(list(args.endpoints), args.api_prefix)

            pending_dataset_ids: list[str] = []
            for dataset_id in dataset_ids:
                if _model_dataset_is_complete(
                    args,
                    dataset_id=dataset_id,
                    model_id=model_id,
                ):
                    completed.append(
                        {
                            "model_id": model_id,
                            "dataset_id": dataset_id,
                            "completed_at": protocol.utc_now(),
                            "resumed": True,
                        }
                    )
                else:
                    pending_dataset_ids.append(dataset_id)
            status["completed"] = completed
            protocol.write_json(status_path, status)

            def record_completed(dataset_id: str) -> None:
                completed.append(
                    {
                        "model_id": model_id,
                        "dataset_id": dataset_id,
                        "completed_at": protocol.utc_now(),
                    }
                )
                status["completed"] = completed
                status["active_model_id"] = model_id
                status["active_dataset_id"] = dataset_id
                protocol.write_json(status_path, status)

            if pending_dataset_ids:
                first_dataset_id = pending_dataset_ids.pop(0)
                print(
                    f"{model_id}: loading model with dataset " f"{first_dataset_id}",
                    flush=True,
                )
                _run_model_dataset(
                    args,
                    dataset_id=first_dataset_id,
                    model_id=model_id,
                    request_concurrency_divisor=1,
                )
                record_completed(first_dataset_id)

            dataset_parallelism = MODEL_MAJOR_DATASET_PARALLELISM.get(model_id, 1)
            for offset in range(0, len(pending_dataset_ids), dataset_parallelism):
                batch = pending_dataset_ids[offset : offset + dataset_parallelism]
                divisor = len(batch)
                status["active_model_id"] = model_id
                status["active_dataset_ids"] = batch
                protocol.write_json(status_path, status)
                print(
                    f"{model_id}: concurrent dataset batch "
                    f"{offset // dataset_parallelism + 1}, "
                    f"datasets={batch}, concurrency_divisor={divisor}",
                    flush=True,
                )
                with ThreadPoolExecutor(max_workers=divisor) as executor:
                    futures = {
                        executor.submit(
                            _run_model_dataset,
                            args,
                            dataset_id=dataset_id,
                            model_id=model_id,
                            request_concurrency_divisor=divisor,
                        ): dataset_id
                        for dataset_id in batch
                    }
                    for future in as_completed(futures):
                        future.result()
                        record_completed(futures[future])
            _unload_accelerator_models(list(args.endpoints), args.api_prefix)

        # Rebuild dataset manifests in parallel. At this point every canonical
        # prediction is immutable and accelerator models have been unloaded,
        # so the children only validate cached file records and write metadata.
        finalization_workers = min(
            max(1, args.preprocess_workers),
            len(dataset_ids),
        )
        print(
            f"rebuilding {len(dataset_ids)} dataset manifests with "
            f"{finalization_workers} workers",
            flush=True,
        )

        def rebuild_dataset_manifest(dataset_id: str) -> None:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *_single_dataset_child_arguments(
                        args,
                        dataset_id=dataset_id,
                        models=list(args.models),
                        keep_loaded=False,
                    ),
                ],
                cwd=protocol.REPO_ROOT,
                check=True,
            )

        with ThreadPoolExecutor(max_workers=finalization_workers) as executor:
            futures = {
                executor.submit(rebuild_dataset_manifest, dataset_id): dataset_id
                for dataset_id in dataset_ids
            }
            for future in as_completed(futures):
                future.result()
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "finished_at": protocol.utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        protocol.write_json(status_path, status)
        try:
            _unload_accelerator_models(list(args.endpoints), args.api_prefix)
        except Exception as cleanup_error:
            print(
                f"best-effort model cleanup failed after {type(error).__name__}: "
                f"{cleanup_error}",
                file=sys.stderr,
                flush=True,
            )
        raise

    status.update(
        {
            "state": "complete",
            "finished_at": protocol.utc_now(),
            "active_model_id": None,
            "active_dataset_id": None,
            "active_dataset_ids": [],
        }
    )
    protocol.write_json(status_path, status)
    return 0


def main() -> int:
    args = parse_args()
    if args.dataset_ids:
        return run_model_major_controller(args)
    if len(set(args.models)) != len(args.models):
        raise ValueError("model ids must be unique")
    endpoint_presets = endpoint_presets_with_defaults(
        list(args.endpoints),
        list(args.endpoint_preset),
    )
    endpoint_profiles = build_endpoint_profiles(
        list(args.endpoints),
        default_devices=args.devices,
        endpoint_presets=endpoint_presets,
        endpoint_devices=list(args.endpoint_devices),
        endpoint_capacities=list(args.endpoint_capacity),
        endpoint_concurrency_scales=list(args.endpoint_concurrency_scale),
        endpoint_model_capacities=list(args.endpoint_model_capacity),
        endpoint_model_concurrencies=list(args.endpoint_model_concurrency),
    )
    missing_configs = sorted(set(args.models) - set(MODEL_EXECUTION_CONFIG))
    if missing_configs:
        raise ValueError(
            "missing model execution configs: " + ", ".join(missing_configs)
        )
    dataset = protocol.resolve_dataset(args.dataset_id)
    dataset_root = args.output_root.resolve() / dataset.dataset_id
    generation_dir = dataset_root / "02_generation"
    shard_name = f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    generation_manifest_path = generation_dir / f"manifest__{shard_name}.json"
    validation_path = generation_dir / f"validation__{shard_name}.json"
    validation = protocol.read_json(validation_path)
    if not validation["accepted"]:
        raise ValueError("generation validation is not accepted")
    if validation.get("generation_manifest_sha256") != (
        protocol.file_sha256(generation_manifest_path)
    ):
        raise ValueError(
            "generation validation is not bound to the current manifest"
        )
    generation_manifest = protocol.read_json(generation_manifest_path)
    real_anchored_source_record = optional_real_anchored_source_record(
        generation_manifest
    )
    calibration_bundle_path = (
        dataset_root / "01_calibration" / "calibration_bundle.json"
    )
    calibration_bundle = (
        protocol.read_json(calibration_bundle_path)
        if calibration_bundle_path.is_file()
        else None
    )
    formal_real_anchor_source_record(calibration_bundle)
    inference_dir = dataset_root / "03_inference" / shard_name
    if inference_dir.exists() and not args.resume:
        expected_parent = (dataset_root / "03_inference").resolve()
        if inference_dir.resolve().parent != expected_parent:
            raise ValueError("refusing to replace inference output outside dataset")
        shutil.rmtree(inference_dir)
    inference_dir.mkdir(parents=True, exist_ok=True)
    task_manifest_path = inference_dir / "task_manifest.json"
    if task_manifest_path.exists() and args.resume:
        task_manifest = protocol.read_json(task_manifest_path)
        task_path = validate_inference_task_manifest_files(task_manifest)
        if (
            task_manifest["generation_config_sha256"]
            != generation_manifest["config_sha256"]
        ):
            raise ValueError("resume generation config mismatch")
        expected_calibration_sha256 = (
            None
            if calibration_bundle is None
            else calibration_bundle.get("bundle_content_sha256")
        )
        if (
            task_manifest.get("calibration_bundle_content_sha256")
            != expected_calibration_sha256
        ):
            raise ValueError("resume calibration bundle mismatch")
        task_real_anchored_source = task_manifest.get(
            "real_anchored_source"
        )
        expected_real_anchored_sha256 = (
            None
            if real_anchored_source_record is None
            else real_anchored_source_record.get("sha256")
        )
        task_real_anchored_sha256 = (
            None
            if not isinstance(task_real_anchored_source, dict)
            else task_real_anchored_source.get("sha256")
        )
        if task_real_anchored_sha256 != expected_real_anchored_sha256:
            raise ValueError("resume real-anchored generation source mismatch")
    else:
        task_path, task_manifest = prepare_view_tasks(
            generation_manifest,
            inference_dir=inference_dir,
            calibration_bundle=calibration_bundle,
        )
        validate_inference_task_manifest_files(task_manifest)

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
    catalogs = dict(health_results)
    resolved_input_capabilities = validate_catalog_input_capabilities(
        catalogs,
        list(args.models),
        contract_path=args.input_capability_contract,
    )
    if args.prepare_only:
        for model_id in args.models:
            _work_assignments, shard_manifest = plan_model_phase(
                model_id,
                health_results,
                task_path=task_path,
                inference_dir=inference_dir,
                endpoint_profiles=endpoint_profiles,
            )
            print(
                f"prepared {dataset.dataset_id}/{model_id}: "
                f"{shard_manifest['source_task_row_count']} tasks, "
                f"{shard_manifest['part_count']} endpoint shards",
                flush=True,
            )
        return 0
    work_statuses: list[dict[str, Any]] = []
    model_phases: list[dict[str, Any]] = []
    expected_sample_ids = task_sample_ids(task_path)
    expected_view_count = int(task_manifest["view_count"])
    if len(expected_sample_ids) != expected_view_count:
        raise ValueError(
            "combined inference task contains duplicate or missing sample IDs"
        )
    cached_model_records: dict[str, dict[str, Any]] = (
        cached_complete_model_records(
            inference_dir,
            expected_sample_ids=expected_sample_ids,
        )
        if args.resume
        else {}
    )
    for phase_index, model_id in enumerate(args.models):
        canonical_path = prediction_path_for(
            model_root(inference_dir, model_id),
            model_id,
        )
        manifest_cached = model_id in cached_model_records
        canonical_matches_task = False
        if args.resume and canonical_path.exists():
            canonical_matches_task = (
                canonical_prediction_sample_ids(
                    canonical_path,
                    model_id=model_id,
                )
                == expected_sample_ids
            )
        if (
            args.resume
            and canonical_path.exists()
            and (manifest_cached or canonical_matches_task)
        ):
            already_compact = manifest_cached or canonical_prediction_file_is_compact(
                canonical_path
            )
            compacted_count = (
                expected_view_count
                if already_compact
                else compact_canonical_prediction_file(canonical_path)
            )
            if compacted_count != expected_view_count:
                raise RuntimeError(
                    f"compacted prediction count mismatch: "
                    f"{compacted_count} != {expected_view_count}"
                )
            removed_bytes = cleanup_completed_model_intermediates(
                inference_dir,
                model_id,
            )
            print(
                f"model phase {phase_index + 1}/{len(args.models)} "
                f"{model_id}: canonical prediction already complete; "
                f"verified={'manifest' if manifest_cached else 'row_count'}, "
                f"format={'already_compact' if already_compact else 'rewritten'}, "
                f"cleaned={removed_bytes / (1024**2):.1f} MiB",
                flush=True,
            )
            model_phases.append(
                {
                    "phase_index": phase_index,
                    "model_id": model_id,
                    "status": "already_complete",
                    "eligible_endpoints": sorted(
                        endpoint
                        for endpoint, catalog in health_results
                        if model_id in catalog
                    ),
                    "part_count": 0,
                    "work_assignments": {},
                    "shard_manifest": None,
                }
            )
            continue

        work_assignments, shard_manifest = plan_model_phase(
            model_id,
            health_results,
            task_path=task_path,
            inference_dir=inference_dir,
            endpoint_profiles=endpoint_profiles,
        )
        active_work = {
            endpoint: items for endpoint, items in work_assignments.items() if items
        }
        print(
            f"model phase {phase_index + 1}/{len(args.models)} "
            f"{model_id}: {len(shard_manifest['parts'])} parts on "
            f"{len(active_work)} services",
            flush=True,
        )
        phase_statuses: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(active_work)) as executor:
            future_map = {
                executor.submit(
                    run_service_queue,
                    endpoint,
                    items,
                    catalogs[endpoint],
                    args=args,
                    endpoint_profile=endpoint_profiles[endpoint],
                ): endpoint
                for endpoint, items in active_work.items()
            }
            for future in as_completed(future_map):
                phase_statuses.extend(future.result())
        work_statuses.extend(phase_statuses)
        consolidated = consolidate_model_predictions(
            inference_dir,
            shard_manifest,
        )
        model_phases.append(
            {
                "phase_index": phase_index,
                "model_id": model_id,
                "status": "complete" if consolidated else "incomplete",
                "eligible_endpoints": sorted(active_work),
                "endpoint_profiles": {
                    endpoint: endpoint_profiles[endpoint].as_dict()
                    for endpoint in sorted(active_work)
                },
                "part_count": int(shard_manifest["part_count"]),
                "work_assignments": {
                    endpoint: [
                        {
                            "work_id": item.work_id,
                            "part_index": item.part_index,
                            "sample_path": str(item.sample_path),
                            "output_dir": str(item.output_dir),
                        }
                        for item in items
                    ]
                    for endpoint, items in work_assignments.items()
                },
                "shard_manifest": shard_manifest,
            }
        )
    if args.keep_loaded_between_runs and len(args.models) == 1:
        complete = all(
            row.get("status") in {"complete", "already_complete"}
            for row in model_phases
        )
        print(
            protocol.canonical_json(
                {
                    "complete": complete,
                    "dataset_manifest_deferred": True,
                    "output": str(inference_dir),
                    "scheduling_policy": SCHEDULING_POLICY_ID,
                }
            )
        )
        return 0 if complete else 2
    statuses = aggregate_model_statuses(
        list(args.models),
        work_statuses,
        inference_dir=inference_dir,
        expected_view_count=expected_view_count,
    )
    stale_merged_path = inference_dir / "predictions.jsonl"
    stale_merged_path.unlink(missing_ok=True)
    prediction_files = []
    real_anchor_prediction_files = []
    prediction_count = 0
    real_anchor_prediction_count = 0
    validate_inference_task_manifest_files(task_manifest)
    real_anchor_task_path = Path(
        task_manifest["task_components"]["real_anchors"]["path"]
    )
    expected_real_anchor_count = int(task_manifest["real_anchor_view_count"])
    for model_id in args.models:
        path = prediction_path_for(
            model_root(inference_dir, model_id),
            model_id,
        )
        if not path.is_file():
            continue
        cached_record = cached_model_records.get(model_id)
        prediction_files.append(
            {
                "model_id": model_id,
                "row_count": expected_view_count,
                **(
                    cached_record
                    if cached_record is not None
                    else protocol.file_record(path)
                ),
            }
        )
        prediction_count += expected_view_count
        real_output_path = (
            inference_dir
            / "real_anchor_predictions"
            / f"{safe_filename(model_id)}.jsonl"
        )
        observed_real_anchor_count = write_real_anchor_prediction_subset(
            path,
            real_anchor_task_path=real_anchor_task_path,
            output_path=real_output_path,
        )
        if observed_real_anchor_count != expected_real_anchor_count:
            raise RuntimeError(
                f"{model_id} real-anchor prediction count mismatch: "
                f"{observed_real_anchor_count} != "
                f"{expected_real_anchor_count}"
            )
        real_anchor_prediction_files.append(
            {
                "model_id": model_id,
                "row_count": observed_real_anchor_count,
                **protocol.file_record(real_output_path),
            }
        )
        real_anchor_prediction_count += observed_real_anchor_count
    manifest = {
        "schema_version": "cafe.inference_manifest.v1",
        "created_at": protocol.utc_now(),
        "task_manifest_sha256": protocol.file_sha256(task_manifest_path),
        "models": list(args.models),
        "available_endpoints": sorted(catalogs),
        "endpoint_profiles": {
            endpoint: endpoint_profiles[endpoint].as_dict()
            for endpoint in sorted(catalogs)
        },
        "scheduling": {
            "policy_id": SCHEDULING_POLICY_ID,
            "phase_order": list(args.models),
            "model_phases": model_phases,
        },
        "model_execution_config": {
            model_id: dict(MODEL_EXECUTION_CONFIG[model_id]) for model_id in args.models
        },
        "input_capabilities": {
            "adaptation_policy_id": INPUT_ADAPTATION_POLICY_ID,
            "resolved": resolved_input_capabilities,
            "stage_contract": (
                None
                if args.input_capability_contract is None
                else protocol.file_record(args.input_capability_contract.resolve())
            ),
        },
        "statuses": statuses,
        "predictions": {
            "storage": (
                "per_model_jsonl_including_synthetic_real_anchored_and_"
                "real_anchor_tasks"
            ),
            "files": prediction_files,
            "row_count": prediction_count,
            "synthetic_row_count_per_model": int(task_manifest["synthetic_view_count"]),
            "real_anchored_counterfactual": {
                "storage": "included_in_canonical_per_model_jsonl",
                "rows_per_model": int(
                    task_manifest.get("real_anchored_view_count", 0)
                ),
                "included_in_synthetic_mechanism_ranking": False,
                "benchmark_track": REAL_ANCHORED_BENCHMARK_TRACK,
            },
            "real_anchor": {
                "storage": "separate_per_model_auxiliary_jsonl",
                "files": real_anchor_prediction_files,
                "row_count": real_anchor_prediction_count,
                "rows_per_model": expected_real_anchor_count,
                "included_in_mechanism_ranking": False,
            },
        },
    }
    manifest["complete"] = all(
        row.get("status") == "complete" for row in statuses
    ) and len(statuses) == len(args.models)
    protocol.write_json(inference_dir / "inference_manifest.json", manifest)
    print(
        protocol.canonical_json(
            {
                "complete": manifest["complete"],
                "prediction_count": prediction_count,
                "scheduling_policy": SCHEDULING_POLICY_ID,
                "output": str(inference_dir),
            }
        )
    )
    if not manifest["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
