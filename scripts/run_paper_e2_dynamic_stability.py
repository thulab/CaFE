#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import httpx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.services.metric_service import compute_sample_metrics  # noqa: E402
from app.services.synthetic_generation_service import (  # noqa: E402
    CAPABILITIES_BY_ID,
    _generate_accepted_sample_values,
    _seed_for,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    resolve_generator_conditioning,
)


SCHEMA_VERSION = "paper_e2_dynamic_stability.v1"
EXPERIMENT_VERSION = "v1"
EXPERIMENT_ID = "E2_dynamic_stability"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp" / EXPERIMENT_VERSION / EXPERIMENT_ID
GENERATOR_ARTIFACT_PATH = (
    REPO_ROOT / "backend/app/data/synthetic_v2_generator_conditioning_artifact.json"
)
FEATURE_GATE_ARTIFACT_PATH = (
    REPO_ROOT / "backend/app/data/synthetic_v2_feature_gate_artifact.json"
)
NEAR_DISTANCE_ARTIFACT_PATH = (
    REPO_ROOT / "backend/app/data/synthetic_v2_near_distance_artifact.json"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-16-paper-e2-dynamic-stability-protocol.md"
)
RUNNER_PATH = Path(__file__).resolve()
EXECUTION_CALIBRATION = {
    "report": "timer-rest-service/data/concurrency-benchmark/20260716T182346Z-replicas/REPLICA_OPTIMIZATION_ZH.md",
    "report_sha256": "cd16830aa17985e1c45701aa6a56454b7a42bec85e1a724ca51563e319cfec46",
    "timer_rest_service_git_commit": "3b5dc776c4c7846416048ed290fa0ae56e3eb870",
}
DEFAULT_MODELS = (
    "Timer-3.5",
    "Timer-3.0",
    "Chronos-2",
    "moirai2",
    "toto2.0",
    "timesfm2.5",
    "tirex2",
)
BASELINE_MODELS = ("naive", "seasonal_naive")
MODEL_EXECUTION_CONFIG = {
    "Timer-3.5": {"replicas_per_device": 1, "http_concurrency": 64},
    "Timer-3.0": {"replicas_per_device": 1, "http_concurrency": 32},
    "Chronos-2": {"replicas_per_device": 4, "http_concurrency": 32},
    "moirai2": {"replicas_per_device": 2, "http_concurrency": 16},
    "toto2.0": {"replicas_per_device": 2, "http_concurrency": 16},
    "timesfm2.5": {"replicas_per_device": 8, "http_concurrency": 32},
    "tirex2": {"replicas_per_device": 1, "http_concurrency": 32},
}
DEFAULT_DEVICES = "0,1"
DEFAULT_REQUEST_MAX_ATTEMPTS = 3
DEFAULT_ROUND_SEEDS = (
    2026071621,
    2026071622,
    2026071623,
    2026071624,
    2026071625,
)
DEFAULT_SAMPLES_PER_ROUND = 32
DEFAULT_BOOTSTRAP_REPLICATES = 1000
INTENSITIES = (1, 2, 3, 4, 5)
TIME_COLUMN = "time"

# Preregistered operational stability criteria.
MAX_MEDIAN_SCORE_CV = 0.10
MAX_P95_SCORE_CV = 0.25
MIN_MODEL_PROFILE_ICC = 0.90
MIN_MEDIAN_CELL_KENDALL = 0.80
MIN_P10_CELL_KENDALL = 0.50
MAX_MEDIAN_RELATIVE_CI_WIDTH = 0.20
MAX_P95_RELATIVE_CI_WIDTH = 0.50
MAX_CROSS_ROUND_DUPLICATE_RATE = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-v1 E2 dynamic benchmark stability experiment."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--round-seeds", nargs="+", type=int, default=list(DEFAULT_ROUND_SEEDS))
    parser.add_argument("--samples-per-round", type=int, default=DEFAULT_SAMPLES_PER_ROUND)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument(
        "--request-max-attempts",
        type=int,
        default=DEFAULT_REQUEST_MAX_ATTEMPTS,
    )
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "infer", "analyze"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-loaded", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_cli_args(args)
    output_dir = args.output_dir.resolve()
    generator_artifact = read_json(GENERATOR_ARTIFACT_PATH)
    config = experiment_config(args, generator_artifact)
    prepare_or_resume_output(output_dir, config=config, resume=args.resume)

    if args.stage in {"all", "generate"}:
        generate_samples_if_needed(output_dir, config=config, artifact=generator_artifact)
    if args.stage in {"all", "infer"}:
        require_file(output_dir / "samples.jsonl")
        run_inference(output_dir, config=config, args=args)
    if args.stage in {"all", "analyze"}:
        require_file(output_dir / "samples.jsonl")
        summary = analyze_experiment(
            output_dir,
            config=config,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        write_json(output_dir / "summary.json", summary)
        (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
        write_manifest(output_dir, config=config)
        print(
            f"E2 criteria: {summary['criteria']['passed_count']}/"
            f"{summary['criteria']['criterion_count']}, "
            f"overall={summary['criteria']['overall_passed']}",
            flush=True,
        )
    print(f"E2 output: {output_dir}", flush=True)
    return 0


def validate_cli_args(args: argparse.Namespace) -> None:
    if len(args.round_seeds) < 2:
        raise ValueError("E2 requires at least two independent generation rounds")
    if len(set(args.round_seeds)) != len(args.round_seeds):
        raise ValueError("round seeds must be unique")
    if args.samples_per_round < 2:
        raise ValueError("samples-per-round must be at least 2")
    device_parts = [part.strip() for part in args.devices.split(",") if part.strip()]
    if not device_parts or any(not part.isdigit() for part in device_parts):
        raise ValueError("devices must be a comma-separated list of device indexes")
    if len(set(device_parts)) != len(device_parts):
        raise ValueError("devices must not contain duplicates")
    if args.request_max_attempts < 1:
        raise ValueError("request-max-attempts must be positive")
    unknown_models = sorted(set(args.models) - set(MODEL_EXECUTION_CONFIG))
    if unknown_models:
        raise ValueError(f"missing frozen execution config for models: {unknown_models}")
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")


def experiment_config(args: argparse.Namespace, artifact: dict[str, Any]) -> dict[str, Any]:
    online = list(artifact["config"]["online_conditioning_profile_ids"])
    profile_capability_count = sum(
        len(artifact["profiles"][profile_id]["capabilities"]) for profile_id in online
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "canonical_scale_id": artifact["canonical_intensity"]["scale_id"],
        "canonical_scale_fingerprint": artifact["canonical_intensity"]["scale_fingerprint"],
        "online_conditioning_profile_ids": online,
        "profile_capability_count": profile_capability_count,
        "intensities": list(INTENSITIES),
        "round_seeds": [int(seed) for seed in args.round_seeds],
        "samples_per_round_per_cell": int(args.samples_per_round),
        "expected_generated_sample_count": int(
            profile_capability_count
            * len(INTENSITIES)
            * len(args.round_seeds)
            * args.samples_per_round
        ),
        "paired_seed_policy": (
            "within profile/capability/round/sample_index, all models and all intensities use "
            "the same generated base seed"
        ),
        "requested_models": list(args.models),
        "baseline_models": list(BASELINE_MODELS),
        "model_execution": {
            model_id: dict(MODEL_EXECUTION_CONFIG[model_id]) for model_id in args.models
        },
        "devices": ",".join(part.strip() for part in args.devices.split(",") if part.strip()),
        "tasks_per_http_request": 1,
        "shape_schedule": "complete each request_group_key bucket before the next bucket",
        "request_max_attempts": int(args.request_max_attempts),
        "execution_calibration": dict(EXECUTION_CALIBRATION),
        "primary_metric": "mase",
        "secondary_metric": "mae",
        "forecast_timeout_seconds": int(args.forecast_timeout_seconds),
        "model_load_timeout_seconds": int(args.model_load_timeout_seconds),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "service": {
            "base_url": args.base_url,
            "api_prefix": args.api_prefix,
        },
        "criteria": {
            "max_median_score_cv": MAX_MEDIAN_SCORE_CV,
            "max_p95_score_cv": MAX_P95_SCORE_CV,
            "min_model_profile_icc": MIN_MODEL_PROFILE_ICC,
            "min_median_cell_kendall": MIN_MEDIAN_CELL_KENDALL,
            "min_p10_cell_kendall": MIN_P10_CELL_KENDALL,
            "max_median_relative_ci_width": MAX_MEDIAN_RELATIVE_CI_WIDTH,
            "max_p95_relative_ci_width": MAX_P95_RELATIVE_CI_WIDTH,
            "max_cross_round_duplicate_rate": MAX_CROSS_ROUND_DUPLICATE_RATE,
        },
        "retention_policy": (
            "samples and successful predictions are append-safe; --resume never overwrites them; "
            "manifest.json seals a completed analysis"
        ),
    }


def prepare_or_resume_output(
    output_dir: Path,
    *,
    config: dict[str, Any],
    resume: bool,
) -> None:
    config_path = output_dir / "config.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"E2 output already exists; use --resume with exactly the same config: {output_dir}"
            )
        if (output_dir / "manifest.json").exists():
            raise FileExistsError(f"completed E2 output is sealed by manifest.json: {output_dir}")
        existing = read_json(config_path)
        if canonical_json(existing) != canonical_json(config):
            raise ValueError("resume config does not match the existing E2 config.json")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "failures").mkdir(exist_ok=True)
    write_json(config_path, config)


def generate_samples_if_needed(
    output_dir: Path,
    *,
    config: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    sample_path = output_dir / "samples.jsonl"
    if sample_path.exists():
        observed = count_jsonl(sample_path)
        expected = int(config["expected_generated_sample_count"])
        if observed != expected:
            raise ValueError(
                f"existing samples.jsonl is incomplete: observed={observed}, expected={expected}"
            )
        print(f"samples already complete: {observed}", flush=True)
        return

    temporary = output_dir / "samples.jsonl.in_progress"
    if temporary.exists():
        raise FileExistsError(
            f"partial sample file exists and is retained for diagnosis: {temporary}"
        )
    created = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for profile_id in config["online_conditioning_profile_ids"]:
            profile = artifact["profiles"][profile_id]
            for capability_id in sorted(profile["capabilities"]):
                conditioning = resolve_generator_conditioning(
                    capability_id=capability_id,
                    profile_id=profile_id,
                    context_length=int(profile["context_length"]),
                    horizon=int(profile["horizon"]),
                    target_dim=int(profile["target_dim"]),
                    artifact=artifact,
                )
                if conditioning is None:
                    raise RuntimeError(f"missing conditioning for {profile_id}/{capability_id}")
                for round_index, round_seed in enumerate(config["round_seeds"], start=1):
                    for sample_index in range(config["samples_per_round_per_cell"]):
                        sample_seed = _seed_for(
                            int(round_seed),
                            f"{profile_id}:{capability_id}",
                            sample_index,
                        )
                        for intensity in INTENSITIES:
                            target, latent, covariates, features = _generate_accepted_sample_values(
                                capability_id,
                                int(profile["context_length"]) + int(profile["horizon"]),
                                int(profile["context_length"]),
                                int(profile["target_dim"]),
                                int(profile["season_length"]),
                                intensity,
                                sample_seed,
                                anchor_profile_id=profile_id,
                                generator_conditioning=conditioning,
                            )
                            row = sample_row(
                                profile=profile,
                                profile_id=profile_id,
                                capability_id=capability_id,
                                intensity=intensity,
                                round_index=round_index,
                                round_seed=int(round_seed),
                                sample_index=sample_index,
                                sample_seed=sample_seed,
                                target=target,
                                covariates=covariates,
                                features=features,
                                latent=latent,
                            )
                            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                            created += 1
                            if created % 500 == 0:
                                print(
                                    f"generated {created}/{config['expected_generated_sample_count']}",
                                    flush=True,
                                )
    os.replace(temporary, sample_path)
    if created != int(config["expected_generated_sample_count"]):
        raise AssertionError(f"unexpected generated sample count: {created}")
    print(f"generated samples complete: {created}", flush=True)


def sample_row(
    *,
    profile: dict[str, Any],
    profile_id: str,
    capability_id: str,
    intensity: int,
    round_index: int,
    round_seed: int,
    sample_index: int,
    sample_seed: int,
    target: np.ndarray,
    covariates: np.ndarray | None,
    features: dict[str, float],
    latent: dict[str, Any],
) -> dict[str, Any]:
    sample_id = (
        f"{profile_id}__{capability_id}__i{intensity}__r{round_index}__s{sample_index:03d}"
    )
    return {
        "schema_version": "paper_e2_sample.v1",
        "sample_id": sample_id,
        "profile_id": profile_id,
        "capability_id": capability_id,
        "intensity": int(intensity),
        "round_index": int(round_index),
        "round_seed": int(round_seed),
        "sample_index": int(sample_index),
        "sample_seed": int(sample_seed),
        "context_length": int(profile["context_length"]),
        "horizon": int(profile["horizon"]),
        "season_length": int(profile["season_length"]),
        "frequency": str(profile["frequency"]),
        "target_dim": int(profile["target_dim"]),
        "covariate_dim": 0 if covariates is None else int(covariates.shape[1]),
        "target": np.asarray(target, dtype=float).tolist(),
        "covariates": None if covariates is None else np.asarray(covariates, dtype=float).tolist(),
        "realized_features": clean_float_mapping(features),
        "acceptance_attempts": int(latent["acceptance"]["attempts"]),
        "canonical_target_feature": latent["generator_conditioning"][
            "canonical_target_feature"
        ],
        "canonical_target_strength": latent["generator_conditioning"][
            "canonical_target_strength"
        ],
    }


def run_inference(
    output_dir: Path,
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    client = TimerServiceClient(
        config["service"]["base_url"],
        config["service"]["api_prefix"],
        timeout_seconds=30,
    )
    try:
        catalog = client.list_models()
        write_json(output_dir / "model_catalog.json", {"models": catalog})
        requested = resolve_requested_models(catalog, config["requested_models"])
        run_baselines(output_dir)
        statuses = read_json_if_exists(output_dir / "model_status.json", default={"models": {}})
        for model in requested:
            model_id = str(model["model_id"])
            print(f"starting model: {model_id}", flush=True)
            started = time.monotonic()
            try:
                status = run_one_model(
                    client,
                    model,
                    output_dir=output_dir,
                    execution=config["model_execution"][model_id],
                    devices=str(config["devices"]),
                    request_max_attempts=int(config["request_max_attempts"]),
                    forecast_timeout_seconds=int(config["forecast_timeout_seconds"]),
                    load_timeout_seconds=int(config["model_load_timeout_seconds"]),
                    keep_loaded=args.keep_loaded,
                )
            except Exception as error:  # noqa: BLE001
                prediction_path = prediction_path_for(output_dir, model_id)
                succeeded = count_jsonl(prediction_path) if prediction_path.exists() else 0
                compatible_count = sum(
                    model_supports_sample(model, sample)
                    for sample in iter_jsonl(output_dir / "samples.jsonl")
                )
                status = {
                    "model_id": model_id,
                    "status": "failed",
                    "compatible_sample_count": compatible_count,
                    "succeeded_count": succeeded,
                    "error": str(error),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "prediction_path": relative_path(prediction_path),
                }
            statuses["models"][model_id] = status
            write_json(output_dir / "model_status.json", statuses)
            print(
                f"model {model_id}: {status['status']} "
                f"{status['succeeded_count']}/{status.get('compatible_sample_count')} "
                f"in {status['elapsed_seconds']:.1f}s",
                flush=True,
            )
    finally:
        client.close()


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

    def find_model(self, model_id: str) -> dict[str, Any]:
        model = next(
            (item for item in self.list_models() if item.get("model_id") == model_id),
            None,
        )
        if model is None:
            raise RuntimeError(f"model not found: {model_id}")
        return model

    def unload_all_loaded(self) -> None:
        for model in self.list_loaded_models():
            endpoints = model.get("endpoints") or []
            if any(str(endpoint.get("device", "")).lower() != "cpu" for endpoint in endpoints):
                self.unload_model(str(model["model_id"]))

    def unload_model(self, model_id: str) -> None:
        try:
            self._post(
                "/models/unload",
                {"model_id": model_id},
                timeout_seconds=max(self.timeout_seconds, 600),
            )
        except RuntimeError as error:
            if "409" in str(error) and "not loaded" in str(error).lower():
                return
            raise
        deadline = time.monotonic() + max(self.timeout_seconds, 600)
        while any(
            str(model.get("model_id", "")).lower() == model_id.lower()
            for model in self.list_loaded_models()
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out unloading model {model_id}")
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
        existing = self._loaded_state(model_id)
        if existing is not None:
            self._validate_loaded_topology(
                model_id,
                existing,
                expected_devices=expected_devices,
                replicas_per_device=replicas_per_device,
                expected_endpoints=expected_endpoints,
            )
            return 0.0, existing
        self._post(
            "/models/load",
            {
                "model_id": model_id,
                "devices": devices,
                "replicas_per_device": replicas_per_device,
            },
            timeout_seconds=timeout_seconds,
        )
        deadline = time.monotonic() + timeout_seconds
        while True:
            state = self._loaded_state(model_id)
            if state is not None and state.get("status") == "loaded":
                self._validate_loaded_topology(
                    model_id,
                    state,
                    expected_devices=expected_devices,
                    replicas_per_device=replicas_per_device,
                    expected_endpoints=expected_endpoints,
                )
                return time.monotonic() - started, state
            if state is None and time.monotonic() - started >= 60:
                raise RuntimeError(
                    f"model {model_id} workers disappeared before becoming ready"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out loading model {model_id}")
            time.sleep(1)

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
                f"model {model_id} loaded topology does not match frozen E2 config: "
                f"expected_devices={sorted(expected_devices)}, "
                f"replicas_per_device={replicas_per_device}, state={state}"
            )

    def _get(self, path: str) -> dict[str, Any]:
        return parse_envelope(self.client.get(self.base + path, timeout=self.timeout_seconds))

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


def resolve_requested_models(
    catalog: list[dict[str, Any]],
    requested_ids: list[str],
) -> list[dict[str, Any]]:
    by_id = {str(model.get("model_id")): model for model in catalog}
    resolved: list[dict[str, Any]] = []
    for model_id in requested_ids:
        model = by_id.get(model_id)
        if model is None:
            raise ValueError(f"requested E2 model is not registered: {model_id}")
        if str(model.get("state", "")).lower() == "inactive":
            raise ValueError(f"requested E2 model is inactive: {model_id}")
        resolved.append(model)
    return resolved


def run_baselines(output_dir: Path) -> None:
    for model_id in BASELINE_MODELS:
        prediction_path = prediction_path_for(output_dir, model_id)
        if prediction_path.exists():
            expected = count_jsonl(output_dir / "samples.jsonl")
            observed = count_jsonl(prediction_path)
            if observed != expected:
                raise ValueError(
                    f"baseline {model_id} is incomplete: {observed}/{expected}"
                )
            continue
        temporary = prediction_path.with_suffix(".jsonl.in_progress")
        with temporary.open("w", encoding="utf-8") as handle:
            for sample in iter_jsonl(output_dir / "samples.jsonl"):
                target = np.asarray(sample["target"], dtype=float)
                context = int(sample["context_length"])
                history = target[:context]
                horizon = int(sample["horizon"])
                if model_id == "naive":
                    forecast = np.repeat(history[-1:], horizon, axis=0)
                else:
                    period = min(int(sample["season_length"]), len(history))
                    pattern = history[-period:]
                    forecast = np.vstack(
                        [pattern[index % period] for index in range(horizon)]
                    )
                row = prediction_row(model_id, "baseline", sample, forecast)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, prediction_path)
        print(f"baseline complete: {model_id}", flush=True)


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
) -> dict[str, Any]:
    model_id = str(model["model_id"])
    prediction_path = prediction_path_for(output_dir, model_id)
    done = successful_sample_ids(prediction_path)
    compatible_count = sum(
        model_supports_sample(model, sample)
        for sample in iter_jsonl(output_dir / "samples.jsonl")
    )
    if len(done) > compatible_count:
        raise ValueError(f"prediction file for {model_id} has too many unique samples")
    if len(done) == compatible_count:
        previous = read_json_if_exists(
            output_dir / "model_status.json",
            default={"models": {}},
        ).get("models", {}).get(model_id)
        if (
            previous
            and previous.get("status") == "complete"
            and int(previous.get("succeeded_count", -1)) == compatible_count
        ):
            return previous
        return {
            "model_id": model_id,
            "status": "complete",
            "compatible_sample_count": compatible_count,
            "succeeded_count": compatible_count,
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
        output_dir / "samples.jsonl",
        model=model,
        done=done,
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
                failure_path = output_dir / "failures" / f"{safe_filename(model_id)}.jsonl"
                with failure_path.open("a", encoding="utf-8") as failure_handle:
                    bucket_stats = asyncio.run(
                        run_model_requests(
                            forecast_url=client.base + "/forecast",
                            model_id=model_id,
                            model=model,
                            sample_path=output_dir / "samples.jsonl",
                            done=done,
                            pending_groups=pending_groups,
                            http_concurrency=int(execution["http_concurrency"]),
                            timeout_seconds=forecast_timeout_seconds,
                            max_attempts=request_max_attempts,
                            output_handle=output_handle,
                            failure_handle=failure_handle,
                            compatible_count=compatible_count,
                            initial_persisted=len(done),
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
        "status": "complete" if succeeded == compatible_count else "incomplete",
        "compatible_sample_count": compatible_count,
        "succeeded_count": succeeded,
        "failed_request_count_this_attempt": failures,
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


def pending_request_group_counts(
    sample_path: Path,
    *,
    model: dict[str, Any],
    done: set[str],
) -> dict[tuple[Any, ...], int]:
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for sample in iter_jsonl(sample_path):
        if sample["sample_id"] in done or not model_supports_sample(model, sample):
            continue
        counts[request_group_key(sample)] += 1
    return dict(counts)


async def run_model_requests(
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
) -> list[dict[str, Any]]:
    limits = httpx.Limits(
        max_connections=http_concurrency,
        max_keepalive_connections=http_concurrency,
        keepalive_expiry=120.0,
    )
    timeout = httpx.Timeout(timeout_seconds)
    bucket_stats: list[dict[str, Any]] = []
    persisted = initial_persisted
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as async_client:
        ordered_groups = sorted(pending_groups, key=request_group_sort_key)
        for bucket_index, group_key in enumerate(ordered_groups, start=1):
            pending_count = pending_groups[group_key]
            label = request_group_label(group_key)
            print(
                f"{model_id}: bucket {bucket_index}/{len(ordered_groups)} "
                f"{label}, pending={pending_count}, concurrency={http_concurrency}",
                flush=True,
            )
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
                maxsize=max(2 * http_concurrency, 1)
            )
            bucket_started = time.monotonic()
            succeeded_count = 0
            failed_count = 0

            async def producer() -> None:
                produced = 0
                for sample in iter_jsonl(sample_path):
                    if (
                        sample["sample_id"] in done
                        or not model_supports_sample(model, sample)
                        or request_group_key(sample) != group_key
                    ):
                        continue
                    await queue.put(sample)
                    produced += 1
                if produced != pending_count:
                    raise RuntimeError(
                        f"{model_id}/{label} producer count {produced} != {pending_count}"
                    )
                for _worker_index in range(http_concurrency):
                    await queue.put(None)

            async def worker() -> None:
                nonlocal succeeded_count, failed_count, persisted
                while True:
                    sample = await queue.get()
                    try:
                        if sample is None:
                            return
                        result = await forecast_one_with_retry(
                            async_client,
                            forecast_url=forecast_url,
                            model_id=model_id,
                            sample=sample,
                            max_attempts=max_attempts,
                        )
                        if result["forecast"] is not None:
                            row = prediction_row(
                                model_id,
                                "timer_service",
                                sample,
                                result["forecast"],
                            )
                            row["request_seconds"] = result["elapsed_seconds"]
                            row["request_attempts"] = result["attempts"]
                            row["request_group"] = label
                            output_handle.write(
                                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            succeeded_count += 1
                            persisted += 1
                            if persisted % 500 == 0:
                                output_handle.flush()
                                print(
                                    f"{model_id}: persisted={persisted}/{compatible_count}",
                                    flush=True,
                                )
                        else:
                            failure_handle.write(
                                json.dumps(
                                    {
                                        "model_id": model_id,
                                        "sample_id": sample["sample_id"],
                                        "request_group": label,
                                        "attempts": result["attempts"],
                                        "request_seconds": result["elapsed_seconds"],
                                        "error": result["error"],
                                        "created_at": datetime.now(timezone.utc).isoformat(),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            failed_count += 1
                    finally:
                        queue.task_done()

            tasks = [asyncio.create_task(producer())]
            tasks.extend(
                asyncio.create_task(worker()) for _worker_index in range(http_concurrency)
            )
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
                    "elapsed_seconds": round(elapsed, 3),
                    "successful_tasks_per_second": round(
                        succeeded_count / max(elapsed, 1e-12), 3
                    ),
                }
            )
            print(
                f"{model_id}: bucket {label} complete, "
                f"{succeeded_count}/{pending_count} in {elapsed:.1f}s",
                flush=True,
            )
    return bucket_stats


async def forecast_one_with_retry(
    client: httpx.AsyncClient,
    *,
    forecast_url: str,
    model_id: str,
    sample: dict[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    started = time.monotonic()
    last_error = "unknown forecast error"
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(
                forecast_url,
                json=forecast_request_body(model_id, sample),
            )
            payload = parse_envelope(response)
            results = payload.get("data", {}).get("results", [])
            if len(results) != 1:
                raise RuntimeError(
                    f"forecast returned {len(results)} results for one sample"
                )
            forecast = parse_forecast_result(results[0], horizon=int(sample["horizon"]))
            observed_shape = np.asarray(forecast, dtype=float).shape
            expected_shape = (int(sample["horizon"]), int(sample["target_dim"]))
            if observed_shape != expected_shape:
                raise ValueError(
                    f"forecast shape {observed_shape} does not match {expected_shape}"
                )
            return {
                "forecast": forecast,
                "attempts": attempt,
                "elapsed_seconds": time.monotonic() - started,
                "error": None,
            }
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < max_attempts:
                await asyncio.sleep(0.25 * (2 ** (attempt - 1)))
    return {
        "forecast": None,
        "attempts": max_attempts,
        "elapsed_seconds": time.monotonic() - started,
        "error": last_error,
    }


def forecast_request_body(model_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model_id": model_id,
        "targets": [forecast_target(sample)],
        "output_length": [int(sample["horizon"])],
        "time_col": [TIME_COLUMN],
    }
    if sample["covariate_dim"]:
        body["history_covs"] = [forecast_covariates(sample, history=True)]
        body["future_covs"] = [forecast_covariates(sample, history=False)]
    return body


def model_supports_sample(model: dict[str, Any], sample: dict[str, Any]) -> bool:
    limits = model.get("forecast_limits") or {}
    context = int(sample["context_length"])
    horizon = int(sample["horizon"])
    target_dim = int(sample["target_dim"])
    covariate_dim = int(sample["covariate_dim"])
    if context < int(limits.get("min_input_length") or 0):
        return False
    maximum_input = limits.get("max_input_length")
    if maximum_input is not None and context > int(maximum_input):
        return False
    maximum_output = limits.get("max_output_length")
    if maximum_output is not None and horizon > int(maximum_output):
        return False
    maximum_targets = limits.get("max_target_count")
    if maximum_targets is not None and target_dim > int(maximum_targets):
        return False
    maximum_covariates = int(limits.get("max_covariate_count") or 0)
    if covariate_dim > maximum_covariates:
        return False
    maximum_future_covs = limits.get("max_future_covs_length")
    if covariate_dim and maximum_future_covs is None:
        return False
    if maximum_future_covs is not None and horizon > int(maximum_future_covs):
        return False
    return True


def request_group_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sample["context_length"],
        sample["horizon"],
        sample["target_dim"],
        sample["covariate_dim"],
        sample["frequency"],
    )


def request_group_sort_key(group: tuple[Any, ...]) -> tuple[Any, ...]:
    context, horizon, target_dim, covariate_dim, frequency = group
    return (
        int(int(covariate_dim) > 0),
        int(context),
        int(target_dim),
        int(horizon),
        str(frequency),
    )


def request_group_label(group: tuple[Any, ...]) -> str:
    context, horizon, target_dim, covariate_dim, frequency = group
    return (
        f"ctx{context}_h{horizon}_t{target_dim}_c{covariate_dim}_{frequency}"
    )


def forecast_target(sample: dict[str, Any]) -> dict[str, Any]:
    context = int(sample["context_length"])
    timestamps = sample_timestamps(sample)[:context]
    target = sample["target"][:context]
    columns = [TIME_COLUMN, *[f"target_{index}" for index in range(sample["target_dim"])]]
    return {
        "columns": columns,
        "data": [[timestamp, *row] for timestamp, row in zip(timestamps, target, strict=True)],
    }


def forecast_covariates(sample: dict[str, Any], *, history: bool) -> dict[str, Any]:
    context = int(sample["context_length"])
    timestamps = sample_timestamps(sample)
    covariates = sample["covariates"]
    if history:
        timestamps = timestamps[:context]
        covariates = covariates[:context]
    else:
        timestamps = timestamps[context:]
        covariates = covariates[context:]
    names = list(CAPABILITIES_BY_ID[sample["capability_id"]].covariate_columns)
    return {
        "columns": [TIME_COLUMN, *names],
        "data": [
            [timestamp, *row]
            for timestamp, row in zip(timestamps, covariates, strict=True)
        ],
    }


def sample_timestamps(sample: dict[str, Any]) -> list[str]:
    frequency = str(sample["frequency"]).lower()
    delta = timedelta(days=1) if frequency == "d" else timedelta(hours=1)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    length = int(sample["context_length"]) + int(sample["horizon"])
    return [(base + index * delta).isoformat() for index in range(length)]


def parse_forecast_result(result: dict[str, Any], *, horizon: int) -> list[list[float]]:
    columns = result["columns"]
    indexes = [index for index, column in enumerate(columns) if column != TIME_COLUMN]
    rows = [[float(row[index]) for index in indexes] for row in result["data"][:horizon]]
    if len(rows) != horizon:
        raise ValueError(f"forecast length {len(rows)} does not match horizon {horizon}")
    return rows


def prediction_row(
    model_id: str,
    model_group: str,
    sample: dict[str, Any],
    forecast: np.ndarray | list[list[float]],
) -> dict[str, Any]:
    values = np.asarray(forecast, dtype=float)
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    metrics = compute_sample_metrics(
        target[context:].tolist(),
        values.tolist(),
        target[:context].tolist(),
        seasonal_period=int(sample["season_length"]),
    )
    if sample["capability_id"] == "hierarchical_coherence" and values.shape[1] >= 3:
        residual = values[:, 0] - np.sum(values[:, 1:], axis=1)
        metrics["coherence_mae"] = float(np.mean(np.abs(residual)))
    return {
        "schema_version": "paper_e2_prediction.v1",
        "model_id": model_id,
        "model_group": model_group,
        "sample_id": sample["sample_id"],
        "profile_id": sample["profile_id"],
        "capability_id": sample["capability_id"],
        "intensity": int(sample["intensity"]),
        "round_index": int(sample["round_index"]),
        "sample_index": int(sample["sample_index"]),
        "metrics": clean_float_mapping(metrics),
        "forecast": values.tolist(),
    }


def analyze_experiment(
    output_dir: Path,
    *,
    config: dict[str, Any],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    catalog_payload = read_json(output_dir / "model_catalog.json")
    catalog = {str(model["model_id"]): model for model in catalog_payload["models"]}
    expected_by_model = expected_prediction_counts(
        output_dir / "samples.jsonl",
        catalog=catalog,
        requested_models=config["requested_models"],
    )
    expected_by_model.update(
        {model_id: int(config["expected_generated_sample_count"]) for model_id in BASELINE_MODELS}
    )
    prediction_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for model_id, expected in [
        *( (model_id, expected_by_model[model_id]) for model_id in BASELINE_MODELS ),
        *( (model_id, expected_by_model[model_id]) for model_id in config["requested_models"] ),
    ]:
        path = prediction_path_for(output_dir, model_id)
        observed = count_jsonl(path) if path.exists() else 0
        coverage_rows.append(
            {
                "model_id": model_id,
                "expected_count": expected,
                "observed_count": observed,
                "coverage": float(observed / expected) if expected else 0.0,
                "complete": observed == expected,
            }
        )
        if observed != expected:
            raise RuntimeError(
                f"cannot analyze incomplete model {model_id}: {observed}/{expected}; resume inference"
            )
        prediction_rows.extend(iter_jsonl(path))
    write_csv(output_dir / "model_coverage.csv", coverage_rows)

    round_rows = round_score_rows(prediction_rows)
    cv_rows = score_cv_rows(round_rows)
    bootstrap_rows = bootstrap_ci_rows(
        prediction_rows,
        replicates=bootstrap_replicates,
    )
    rank_rows = rank_stability_rows(round_rows)
    icc_rows = model_profile_icc_rows(round_rows)
    distance_rows = cross_round_distance_rows(output_dir / "samples.jsonl")
    outputs = {
        "round_scores.csv": round_rows,
        "score_cv.csv": cv_rows,
        "bootstrap_ci.csv": bootstrap_rows,
        "rank_stability.csv": rank_rows,
        "model_profile_icc.csv": icc_rows,
        "cross_round_distance.csv": distance_rows,
    }
    for filename, rows in outputs.items():
        write_csv(output_dir / filename, rows)

    statistics = summarize_stability(
        cv_rows=cv_rows,
        bootstrap_rows=bootstrap_rows,
        rank_rows=rank_rows,
        icc_rows=icc_rows,
        distance_rows=distance_rows,
    )
    criteria = stability_criteria(statistics)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "model_coverage": coverage_rows,
        "statistics": statistics,
        "criteria": criteria,
        "table_rows": {filename: len(rows) for filename, rows in outputs.items()},
    }


def round_score_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_rows(
        predictions,
        "model_id",
        "profile_id",
        "capability_id",
        "intensity",
        "round_index",
    )
    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        model_id, profile_id, capability_id, intensity, round_index = key
        model_groups = {str(row["model_group"]) for row in group}
        if len(model_groups) != 1:
            raise ValueError(f"model {model_id} has inconsistent model_group values")
        mase = finite_metrics(group, "mase")
        mae = finite_metrics(group, "mae")
        rows.append(
            {
                "model_id": model_id,
                "model_group": next(iter(model_groups)),
                "profile_id": profile_id,
                "capability_id": capability_id,
                "intensity": intensity,
                "round_index": round_index,
                "sample_count": len(group),
                "mase_mean": float(np.mean(mase)),
                "mase_std": float(np.std(mase, ddof=1)),
                "mae_mean": float(np.mean(mae)),
            }
        )
    return rows


def score_cv_rows(round_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_rows(
        round_rows,
        "model_id",
        "profile_id",
        "capability_id",
        "intensity",
    )
    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["round_index"]))
        scores = np.asarray([row["mase_mean"] for row in ordered], dtype=float)
        mean = float(np.mean(scores))
        rows.append(
            {
                "model_id": key[0],
                "model_group": ordered[0]["model_group"],
                "profile_id": key[1],
                "capability_id": key[2],
                "intensity": key[3],
                "round_count": len(scores),
                "mase_round_mean": mean,
                "mase_round_std": float(np.std(scores, ddof=1)),
                "mase_round_cv": float(np.std(scores, ddof=1) / max(abs(mean), 1e-12)),
                "mase_round_min": float(np.min(scores)),
                "mase_round_max": float(np.max(scores)),
            }
        )
    return rows


def bootstrap_ci_rows(
    predictions: list[dict[str, Any]],
    *,
    replicates: int,
) -> list[dict[str, Any]]:
    grouped = group_rows(
        predictions,
        "model_id",
        "profile_id",
        "capability_id",
        "intensity",
    )
    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        by_round = group_rows(group, "round_index")
        round_values = [
            finite_metrics(by_round[(round_index,)], "mase")
            for round_index in sorted(value[0] for value in by_round)
        ]
        seed = stable_seed("bootstrap", *key)
        estimates = hierarchical_bootstrap_means(
            round_values,
            replicates=replicates,
            seed=seed,
        )
        pooled = np.concatenate(round_values)
        mean = float(np.mean(pooled))
        low, high = np.quantile(estimates, [0.025, 0.975])
        rows.append(
            {
                "model_id": key[0],
                "model_group": group[0]["model_group"],
                "profile_id": key[1],
                "capability_id": key[2],
                "intensity": key[3],
                "sample_count": len(pooled),
                "mase_mean": mean,
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "relative_ci_width": float((high - low) / max(abs(mean), 1e-12)),
                "bootstrap_replicates": replicates,
            }
        )
    return rows


def rank_stability_rows(round_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_rows(round_rows, "profile_id", "capability_id", "intensity")
    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        for scope, scoped_group in (
            (
                "foundation_models",
                [row for row in group if row["model_group"] == "timer_service"],
            ),
            ("all_predictors", group),
        ):
            by_round = group_rows(scoped_group, "round_index")
            model_sets = [
                set(row["model_id"] for row in values) for values in by_round.values()
            ]
            models = sorted(set.intersection(*model_sets))
            if len(models) < 2:
                raise ValueError(f"ranking scope {scope} has fewer than two models in {key}")
            rounds = sorted(value[0] for value in by_round)
            matrix = np.asarray(
                [
                    [
                        next(
                            row["mase_mean"]
                            for row in by_round[(round_index,)]
                            if row["model_id"] == model_id
                        )
                        for round_index in rounds
                    ]
                    for model_id in models
                ],
                dtype=float,
            )
            taus = [
                kendall_tau_b(matrix[:, left], matrix[:, right])
                for left in range(len(rounds))
                for right in range(left + 1, len(rounds))
            ]
            rows.append(
                {
                    "ranking_scope": scope,
                    "profile_id": key[0],
                    "capability_id": key[1],
                    "intensity": key[2],
                    "model_count": len(models),
                    "models": ";".join(models),
                    "round_count": len(rounds),
                    "round_pair_count": len(taus),
                    "kendall_tau_mean": float(np.mean(taus)),
                    "kendall_tau_min": float(np.min(taus)),
                    "kendall_tau_p10": float(np.quantile(taus, 0.10)),
                    "model_by_round_icc_a1": icc_a1(matrix),
                }
            )
    return rows


def model_profile_icc_rows(round_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_rows(round_rows, "model_id")
    rows: list[dict[str, Any]] = []
    for (model_id,), group in sorted(grouped.items()):
        cells = sorted(
            {
                (row["profile_id"], row["capability_id"], int(row["intensity"]))
                for row in group
            }
        )
        rounds = sorted({int(row["round_index"]) for row in group})
        lookup = {
            (
                row["profile_id"],
                row["capability_id"],
                int(row["intensity"]),
                int(row["round_index"]),
            ): float(row["mase_mean"])
            for row in group
        }
        matrix = np.asarray(
            [[lookup[(*cell, round_index)] for round_index in rounds] for cell in cells],
            dtype=float,
        )
        rows.append(
            {
                "model_id": model_id,
                "model_group": group[0]["model_group"],
                "cell_count": len(cells),
                "round_count": len(rounds),
                "profile_score_icc_a1": icc_a1(matrix),
            }
        )
    return rows


def cross_round_distance_rows(sample_path: Path) -> list[dict[str, Any]]:
    samples = list(iter_jsonl(sample_path))
    grouped = group_rows(samples, "profile_id", "capability_id", "intensity")
    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        by_round = group_rows(group, "round_index")
        round_indexes = sorted(value[0] for value in by_round)
        dcr_values: list[float] = []
        nndr_values: list[float] = []
        exact_duplicates = 0
        rounded_duplicates = 0
        query_count = 0
        for left_index in range(len(round_indexes)):
            left_rows = by_round[(round_indexes[left_index],)]
            left = np.vstack(
                [np.asarray(row["target"], dtype=float).reshape(-1) for row in left_rows]
            )
            exact_left = {array_hash(row["target"]) for row in left_rows}
            rounded_left = {array_hash(np.round(row["target"], 6)) for row in left_rows}
            for right_index in range(left_index + 1, len(round_indexes)):
                right_rows = by_round[(round_indexes[right_index],)]
                right = np.vstack(
                    [np.asarray(row["target"], dtype=float).reshape(-1) for row in right_rows]
                )
                right_to_left = nearest_mae_distances(right, left)
                left_to_right = nearest_mae_distances(left, right)
                dcr_values.extend(right_to_left["d1"].tolist())
                dcr_values.extend(left_to_right["d1"].tolist())
                nndr_values.extend(right_to_left["nndr"].tolist())
                nndr_values.extend(left_to_right["nndr"].tolist())
                exact_right = {array_hash(row["target"]) for row in right_rows}
                rounded_right = {
                    array_hash(np.round(row["target"], 6)) for row in right_rows
                }
                exact_duplicates += sum(
                    array_hash(row["target"]) in exact_left for row in right_rows
                )
                exact_duplicates += sum(
                    array_hash(row["target"]) in exact_right for row in left_rows
                )
                rounded_duplicates += sum(
                    array_hash(np.round(row["target"], 6)) in rounded_left
                    for row in right_rows
                )
                rounded_duplicates += sum(
                    array_hash(np.round(row["target"], 6)) in rounded_right
                    for row in left_rows
                )
                query_count += len(left_rows) + len(right_rows)
        dcr = np.asarray(dcr_values, dtype=float)
        nndr = np.asarray(nndr_values, dtype=float)
        rows.append(
            {
                "profile_id": key[0],
                "capability_id": key[1],
                "intensity": key[2],
                "round_count": len(round_indexes),
                "round_pair_count": len(round_indexes) * (len(round_indexes) - 1) // 2,
                "query_count": query_count,
                "cross_round_dcr_q01": float(np.quantile(dcr, 0.01)),
                "cross_round_dcr_q05": float(np.quantile(dcr, 0.05)),
                "cross_round_dcr_p50": float(np.quantile(dcr, 0.50)),
                "cross_round_nndr_q05": float(np.quantile(nndr, 0.05)),
                "cross_round_nndr_p50": float(np.quantile(nndr, 0.50)),
                "exact_duplicate_rate": float(exact_duplicates / max(query_count, 1)),
                "rounded_1e6_duplicate_rate": float(
                    rounded_duplicates / max(query_count, 1)
                ),
                "near_duplicate_mae_le_1e6_rate": float(np.mean(dcr <= 1e-6)),
            }
        )
    return rows


def summarize_stability(
    *,
    cv_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    icc_rows: list[dict[str, Any]],
    distance_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    foundation_cv_rows = [
        row for row in cv_rows if row["model_group"] == "timer_service"
    ]
    foundation_bootstrap_rows = [
        row for row in bootstrap_rows if row["model_group"] == "timer_service"
    ]
    foundation_rank_rows = [
        row for row in rank_rows if row["ranking_scope"] == "foundation_models"
    ]
    foundation_icc_rows = [
        row for row in icc_rows if row["model_group"] == "timer_service"
    ]
    cv = np.asarray([row["mase_round_cv"] for row in foundation_cv_rows], dtype=float)
    widths = np.asarray(
        [row["relative_ci_width"] for row in foundation_bootstrap_rows], dtype=float
    )
    kendall = np.asarray(
        [row["kendall_tau_mean"] for row in foundation_rank_rows], dtype=float
    )
    icc = np.asarray(
        [row["profile_score_icc_a1"] for row in foundation_icc_rows], dtype=float
    )
    return {
        "score_cv": {
            "scope": "foundation_models",
            "cell_model_count": len(foundation_cv_rows),
            "median": float(np.median(cv)),
            "p90": float(np.quantile(cv, 0.90)),
            "p95": float(np.quantile(cv, 0.95)),
            "maximum": float(np.max(cv)),
            "by_model": aggregate_scalar_by_key(
                foundation_cv_rows, "model_id", "mase_round_cv"
            ),
        },
        "bootstrap_ci": {
            "scope": "foundation_models",
            "cell_model_count": len(foundation_bootstrap_rows),
            "relative_width_median": float(np.median(widths)),
            "relative_width_p90": float(np.quantile(widths, 0.90)),
            "relative_width_p95": float(np.quantile(widths, 0.95)),
            "relative_width_maximum": float(np.max(widths)),
        },
        "rank_stability": {
            "scope": "foundation_models",
            "cell_count": len(foundation_rank_rows),
            "kendall_mean_median": float(np.median(kendall)),
            "kendall_mean_p10": float(np.quantile(kendall, 0.10)),
            "kendall_mean_minimum": float(np.min(kendall)),
            "cell_icc_median": float(
                np.median(
                    [row["model_by_round_icc_a1"] for row in foundation_rank_rows]
                )
            ),
        },
        "model_profile_icc": {
            "scope": "foundation_models",
            "model_count": len(foundation_icc_rows),
            "minimum": float(np.min(icc)),
            "median": float(np.median(icc)),
            "by_model": {
                row["model_id"]: row["profile_score_icc_a1"]
                for row in foundation_icc_rows
            },
        },
        "cross_round_distance": {
            "cell_count": len(distance_rows),
            "minimum_dcr_q01": min(row["cross_round_dcr_q01"] for row in distance_rows),
            "minimum_nndr_q05": min(row["cross_round_nndr_q05"] for row in distance_rows),
            "maximum_exact_duplicate_rate": max(
                row["exact_duplicate_rate"] for row in distance_rows
            ),
            "maximum_rounded_duplicate_rate": max(
                row["rounded_1e6_duplicate_rate"] for row in distance_rows
            ),
            "maximum_near_duplicate_rate": max(
                row["near_duplicate_mae_le_1e6_rate"] for row in distance_rows
            ),
        },
    }


def stability_criteria(statistics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "score_cv": bool(
            statistics["score_cv"]["median"] <= MAX_MEDIAN_SCORE_CV
            and statistics["score_cv"]["p95"] <= MAX_P95_SCORE_CV
        ),
        "model_profile_icc": bool(
            statistics["model_profile_icc"]["minimum"] >= MIN_MODEL_PROFILE_ICC
        ),
        "model_ranking": bool(
            statistics["rank_stability"]["kendall_mean_median"]
            >= MIN_MEDIAN_CELL_KENDALL
            and statistics["rank_stability"]["kendall_mean_p10"]
            >= MIN_P10_CELL_KENDALL
        ),
        "bootstrap_ci": bool(
            statistics["bootstrap_ci"]["relative_width_median"]
            <= MAX_MEDIAN_RELATIVE_CI_WIDTH
            and statistics["bootstrap_ci"]["relative_width_p95"]
            <= MAX_P95_RELATIVE_CI_WIDTH
        ),
        "cross_round_diversity": bool(
            statistics["cross_round_distance"]["maximum_exact_duplicate_rate"]
            <= MAX_CROSS_ROUND_DUPLICATE_RATE
            and statistics["cross_round_distance"]["maximum_rounded_duplicate_rate"]
            <= MAX_CROSS_ROUND_DUPLICATE_RATE
            and statistics["cross_round_distance"]["maximum_near_duplicate_rate"]
            <= MAX_CROSS_ROUND_DUPLICATE_RATE
        ),
    }
    return {
        "checks": checks,
        "passed_count": sum(checks.values()),
        "criterion_count": len(checks),
        "overall_passed": all(checks.values()),
    }


def hierarchical_bootstrap_means(
    round_values: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if len(round_values) < 2 or any(len(values) < 2 for values in round_values):
        raise ValueError("hierarchical bootstrap needs at least two rounds with two samples each")
    rng = np.random.default_rng(seed)
    within_means = np.column_stack(
        [
            np.mean(
                values[rng.integers(0, len(values), size=(replicates, len(values)))],
                axis=1,
            )
            for values in round_values
        ]
    )
    selected_rounds = rng.integers(
        0,
        len(round_values),
        size=(replicates, len(round_values)),
    )
    return np.mean(np.take_along_axis(within_means, selected_rounds, axis=1), axis=1)


def icc_a1(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return float("nan")
    subjects, raters = values.shape
    grand = float(np.mean(values))
    row_means = np.mean(values, axis=1)
    column_means = np.mean(values, axis=0)
    ms_rows = raters * float(np.sum((row_means - grand) ** 2)) / (subjects - 1)
    ms_columns = subjects * float(np.sum((column_means - grand) ** 2)) / (raters - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = float(np.sum(residual**2)) / ((subjects - 1) * (raters - 1))
    denominator = (
        ms_rows
        + (raters - 1) * ms_error
        + raters * (ms_columns - ms_error) / subjects
    )
    return float((ms_rows - ms_error) / denominator) if abs(denominator) > 1e-12 else 0.0


def kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            delta_left = np.sign(left[first] - left[second])
            delta_right = np.sign(right[first] - right[second])
            if delta_left == 0 and delta_right == 0:
                continue
            if delta_left == 0:
                ties_left += 1
            elif delta_right == 0:
                ties_right += 1
            elif delta_left == delta_right:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    return float((concordant - discordant) / denominator) if denominator else 0.0


def nearest_mae_distances(query: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    distances = np.mean(np.abs(query[:, None, :] - reference[None, :, :]), axis=2)
    part = np.partition(distances, kth=1, axis=1)
    d1 = part[:, 0]
    d2 = part[:, 1]
    return {"d1": d1, "d2": d2, "nndr": d1 / np.maximum(d2, 1e-12)}


def expected_prediction_counts(
    sample_path: Path,
    *,
    catalog: dict[str, dict[str, Any]],
    requested_models: list[str],
) -> dict[str, int]:
    counts = {model_id: 0 for model_id in requested_models}
    for sample in iter_jsonl(sample_path):
        for model_id in requested_models:
            counts[model_id] += int(model_supports_sample(catalog[model_id], sample))
    return counts


def successful_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    identifiers: set[str] = set()
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in identifiers:
            raise ValueError(f"duplicate successful prediction for {sample_id} in {path}")
        identifiers.add(sample_id)
    return identifiers


def prediction_path_for(output_dir: Path, model_id: str) -> Path:
    return output_dir / "predictions" / f"{safe_filename(model_id)}.jsonl"


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


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


def finite_metrics(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = np.asarray(
        [float(row["metrics"][metric]) for row in rows if is_finite(row["metrics"].get(metric))],
        dtype=float,
    )
    if len(values) != len(rows):
        raise ValueError(f"metric {metric} is absent or non-finite in {len(rows) - len(values)} rows")
    return values


def aggregate_scalar_by_key(
    rows: list[dict[str, Any]],
    key: str,
    value: str,
) -> dict[str, dict[str, float]]:
    grouped = group_rows(rows, key)
    return {
        str(group_key[0]): {
            "count": len(group),
            "median": float(np.median([row[value] for row in group])),
            "p95": float(np.quantile([row[value] for row in group], 0.95)),
            "maximum": float(np.max([row[value] for row in group])),
        }
        for group_key, group in sorted(grouped.items())
    }


def render_report(summary: dict[str, Any]) -> str:
    stats = summary["statistics"]
    criteria = summary["criteria"]
    lines = [
        "# E2 — Dynamic evaluation stability",
        "",
        f"- Scale: `{summary['config']['canonical_scale_id']}` / `{summary['config']['canonical_scale_fingerprint']}`",
        f"- Rounds × samples: {len(summary['config']['round_seeds'])} × {summary['config']['samples_per_round_per_cell']} per profile/capability/intensity.",
        f"- Requested foundation models: {', '.join(summary['config']['requested_models'])}.",
        f"- Criteria: {criteria['passed_count']} / {criteria['criterion_count']} passed.",
        "",
        "## Criteria",
        "",
        "| Criterion | Passed |",
        "| --- | --- |",
    ]
    for name, passed in criteria["checks"].items():
        lines.append(f"| `{name}` | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Main statistics",
            "",
            f"- Round-score CV median / p95: {stats['score_cv']['median']:.4f} / {stats['score_cv']['p95']:.4f}.",
            f"- Model performance-profile ICC minimum / median: {stats['model_profile_icc']['minimum']:.4f} / {stats['model_profile_icc']['median']:.4f}.",
            f"- Cell ranking Kendall mean median / p10: {stats['rank_stability']['kendall_mean_median']:.4f} / {stats['rank_stability']['kendall_mean_p10']:.4f}.",
            f"- Hierarchical bootstrap relative CI width median / p95: {stats['bootstrap_ci']['relative_width_median']:.4f} / {stats['bootstrap_ci']['relative_width_p95']:.4f}.",
            f"- Cross-round minimum DCR q01: {stats['cross_round_distance']['minimum_dcr_q01']:.6f}; maximum duplicate rate: {max(stats['cross_round_distance']['maximum_exact_duplicate_rate'], stats['cross_round_distance']['maximum_rounded_duplicate_rate'], stats['cross_round_distance']['maximum_near_duplicate_rate']):.6f}.",
            "",
            "## ICC by model",
            "",
            "| Model | ICC(A,1) |",
            "| --- | ---: |",
        ]
    )
    for model_id, value in sorted(stats["model_profile_icc"]["by_model"].items()):
        lines.append(f"| `{model_id}` | {value:.4f} |")
    lines.append("")
    lines.append("Detailed cell/model/round results are retained in the CSV and JSONL files in this directory.")
    return "\n".join(lines) + "\n"


def write_manifest(output_dir: Path, *, config: dict[str, Any]) -> None:
    inputs = {
        "generator_conditioning_artifact": GENERATOR_ARTIFACT_PATH,
        "feature_gate_artifact": FEATURE_GATE_ARTIFACT_PATH,
        "near_distance_artifact": NEAR_DISTANCE_ARTIFACT_PATH,
        "runner": RUNNER_PATH,
        "protocol": PROTOCOL_PATH,
    }
    files = {
        str(path.relative_to(output_dir)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and path.name != "manifest.json"
        and not path.name.endswith(".in_progress")
    }
    manifest = {
        "schema_version": "paper_experiment_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": git_head(REPO_ROOT),
        "inputs": {
            name: {"path": relative_path(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "files": files,
    }
    write_json(output_dir / "manifest.json", manifest)


def group_rows(rows: Iterable[Any], *keys: str) -> dict[tuple[Any, ...], list[Any]]:
    grouped: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def array_hash(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big")


def clean_float_mapping(values: dict[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in values.items()
        if is_finite(value)
    }


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def count_jsonl(path: Path) -> int:
    return sum(1 for _row in iter_jsonl(path))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty E2 table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    return read_json(path) if path.exists() else default


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required E2 file is missing: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
