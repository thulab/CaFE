#!/usr/bin/env python3
"""Run resumable four-context inference for the formal Paper v5 E2 inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.services.metric_service import (  # noqa: E402
    compute_sample_metrics,
    mase_unavailable_reason,
)
from build_paper_v4_nine_capability_suite import (  # noqa: E402
    CONTEXT_LENGTHS,
    HORIZON,
    MAX_CONTEXT_LENGTH,
    array_sha256,
    synthetic_paired_view,
)
import run_paper_e2_dynamic_stability as engine  # noqa: E402


SCHEMA_VERSION = "paper_v5_e2_inference.v1"
PREDICTION_SCHEMA_VERSION = "paper_v5_e2_view_prediction.v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp/v5/E2_dynamic_stability"
)
DEFAULT_SYNTHETIC_MASTER_PATH = DEFAULT_OUTPUT_DIR / "samples.jsonl"
DEFAULT_SYNTHETIC_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "sample_manifest.json"
DEFAULT_REAL_SOURCE_DIR = (
    REPO_ROOT / "runtime/paper_exp/v5/02_real_source_window_suite"
)
DEFAULT_REAL_SOURCE_PATH = (
    DEFAULT_REAL_SOURCE_DIR / "real_source_samples.jsonl"
)
DEFAULT_REAL_SOURCE_MANIFEST_PATH = DEFAULT_REAL_SOURCE_DIR / "manifest.json"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/"
    "2026-07-19-paper-v5-e2-inference-and-analysis-plan.md"
)
DEFAULT_MODELS = (
    "Timer-3.5",
    "Timer-3.0",
    "Chronos-2",
    "moirai2",
    "toto2.0",
    "timesfm2.5",
    "tirex2",
    "tabpfn-ts3",
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
    "tabpfn-ts3": {"replicas_per_device": 8, "http_concurrency": 24},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run formal Paper v5 E2 inference over four suffix contexts."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--synthetic-master-path",
        type=Path,
        default=DEFAULT_SYNTHETIC_MASTER_PATH,
    )
    parser.add_argument(
        "--real-source-path",
        type=Path,
        default=DEFAULT_REAL_SOURCE_PATH,
    )
    parser.add_argument(
        "--synthetic-manifest-path",
        type=Path,
        default=DEFAULT_SYNTHETIC_MANIFEST_PATH,
    )
    parser.add_argument(
        "--real-source-manifest-path",
        type=Path,
        default=DEFAULT_REAL_SOURCE_MANIFEST_PATH,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--capabilities",
        nargs="+",
        default=None,
        help=(
            "Optionally infer only the listed synthetic capabilities. "
            "This creates a deterministic derived input inside output-dir "
            "and skips real-source inference."
        ),
    )
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--request-max-attempts", type=int, default=3)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-load-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--stage",
        choices=("baselines", "models", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-real-source", action="store_true")
    parser.add_argument(
        "--preflight-one-per-cell",
        action="store_true",
        help=(
            "Build and use a deterministic one-master-per-cell synthetic "
            "subset; intended only for a separate preflight output directory."
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error


def count_jsonl(path: Path) -> int:
    return sum(1 for _row in iter_jsonl(path))


def build_preflight_master_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(source):
        key = (
            str(row["dataset_id"]),
            str(row["task_id"]),
            str(row["capability_id"]),
        )
        selected.setdefault(key, row)
    if not selected:
        raise ValueError("preflight source has no synthetic cells")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(selected):
            handle.write(
                json.dumps(
                    selected[key],
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, destination)
    return destination


def build_capability_subset_master_file(
    source: Path,
    destination: Path,
    capabilities: list[str],
) -> Path:
    requested = set(capabilities)
    selected = [
        row
        for row in iter_jsonl(source)
        if str(row.get("capability_id")) in requested
    ]
    observed = {
        str(row["capability_id"])
        for row in selected
    }
    missing = sorted(requested - observed)
    if missing:
        raise ValueError(
            f"synthetic input is missing requested capabilities: {missing}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, destination)
    return destination


def validate_master(master: dict[str, Any]) -> None:
    target = np.asarray(master.get("target"), dtype=float)
    expected_shape = (
        MAX_CONTEXT_LENGTH + HORIZON,
        int(master["target_dim"]),
    )
    if target.shape != expected_shape:
        raise ValueError(
            f"master target shape mismatch for {master.get('sample_id')}: "
            f"{target.shape} != {expected_shape}"
        )
    if not np.isfinite(target).all():
        raise ValueError(f"non-finite master target: {master.get('sample_id')}")
    if int(master["horizon"]) != HORIZON:
        raise ValueError(f"unexpected horizon: {master.get('sample_id')}")
    if list(master["context_lengths"]) != list(CONTEXT_LENGTHS):
        raise ValueError(
            f"unexpected context contract: {master.get('sample_id')}"
        )
    covariates = master.get("covariates")
    if covariates is None:
        if int(master.get("covariate_dim", 0)) != 0:
            raise ValueError(
                f"missing master covariates: {master.get('sample_id')}"
            )
        return
    covariate_array = np.asarray(covariates, dtype=float)
    expected_covariate_shape = (
        MAX_CONTEXT_LENGTH + HORIZON,
        int(master["covariate_dim"]),
    )
    if covariate_array.shape != expected_covariate_shape:
        raise ValueError(
            f"master covariate shape mismatch: "
            f"{covariate_array.shape} != {expected_covariate_shape}"
        )
    if not np.isfinite(covariate_array).all():
        raise ValueError(
            f"non-finite master covariates: {master.get('sample_id')}"
        )


def master_view(
    master: dict[str, Any],
    context_length: int,
) -> dict[str, Any]:
    validate_master(master)
    target = np.asarray(master["target"], dtype=float)
    covariates = (
        None
        if master.get("covariates") is None
        else np.asarray(master["covariates"], dtype=float)
    )
    view_target, view_covariates = synthetic_paired_view(
        target,
        covariates,
        context_length=context_length,
        hierarchy=master.get("hierarchy"),
    )
    master_sample_id = str(
        master.get("master_sample_id", master["sample_id"])
    )
    view_id = f"{master_sample_id}__L{context_length}"
    prediction_kind = (
        "synthetic" if "capability_id" in master else "real_source"
    )
    return {
        **master,
        "schema_version": "paper_v5_e2_forecast_view.v1",
        "sample_id": view_id,
        "view_id": view_id,
        "master_sample_id": master_sample_id,
        "prediction_kind": prediction_kind,
        "context_length": int(context_length),
        "target": view_target.tolist(),
        "covariates": (
            None
            if view_covariates is None
            else view_covariates.tolist()
        ),
        "target_sha256": array_sha256(view_target),
        "future_sha256": array_sha256(view_target[context_length:]),
        "master_future_sha256": master["future_sha256"],
    }


def iter_forecast_views(path: Path) -> Iterator[dict[str, Any]]:
    for master in iter_jsonl(path):
        for context_length in CONTEXT_LENGTHS:
            yield master_view(master, context_length)


def prediction_path_for(
    output_dir: Path,
    model_id: str,
    *,
    prediction_kind: str = "synthetic",
) -> Path:
    directory = (
        "real_source_predictions"
        if prediction_kind == "real"
        else "predictions"
    )
    return (
        output_dir
        / directory
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
    expected_shape = (int(sample["horizon"]), int(sample["target_dim"]))
    if values.shape != expected_shape:
        raise ValueError(
            f"forecast shape mismatch: {values.shape} != {expected_shape}"
        )
    metrics = compute_sample_metrics(
        target[context:].tolist(),
        values.tolist(),
        target[:context].tolist(),
        seasonal_period=int(sample["season_length"]),
    )
    if (
        sample.get("capability_id") == "hierarchical_coherence"
        and values.shape[1] >= 3
    ):
        residual = values[:, 0] - np.sum(values[:, 1:], axis=1)
        metrics["coherence_mae"] = float(np.mean(np.abs(residual)))
    row = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_kind": sample["prediction_kind"],
        "model_id": model_id,
        "model_group": model_group,
        "sample_id": sample["view_id"],
        "view_id": sample["view_id"],
        "master_sample_id": sample["master_sample_id"],
        "dataset_id": sample["dataset_id"],
        "task_id": sample["task_id"],
        "profile_id": sample["profile_id"],
        "context_length": context,
        "horizon": int(sample["horizon"]),
        "target_dim": int(sample["target_dim"]),
        "covariate_dim": int(sample["covariate_dim"]),
        "metrics": {
            str(name): float(value) for name, value in metrics.items()
        },
        "mase_unavailable_reason": mase_unavailable_reason(metrics),
        "forecast": values.tolist(),
        "target_future": target[context:].tolist(),
        "view_future_sha256": sample["future_sha256"],
        "master_future_sha256": sample["master_future_sha256"],
    }
    if sample["prediction_kind"] == "synthetic":
        row.update(
            {
                "capability_id": sample["capability_id"],
                "intensity": int(sample["intensity"]),
                "round_index": int(sample["round_index"]),
                "round_seed": int(sample["round_seed"]),
                "sample_index": int(sample["sample_index"]),
                "paired_group_id": sample["paired_group_id"],
            }
        )
    else:
        row.update(
            {
                "source_reference_index": int(
                    sample["source_reference_index"]
                ),
                "supported_capabilities": list(
                    sample["supported_capabilities"]
                ),
            }
        )
    return row


def install_engine_hooks() -> None:
    engine.iter_forecast_samples = iter_forecast_views
    engine.prediction_row = prediction_row
    engine.prediction_path_for = prediction_path_for


def validate_cli_args(args: argparse.Namespace) -> None:
    unknown = sorted(set(args.models) - set(MODEL_EXECUTION_CONFIG))
    if unknown:
        raise ValueError(f"missing execution config for models: {unknown}")
    if len(set(args.models)) != len(args.models):
        raise ValueError("model ids must be unique")
    devices = [
        value.strip() for value in str(args.devices).split(",") if value.strip()
    ]
    if not devices or any(not value.isdigit() for value in devices):
        raise ValueError("devices must be comma-separated GPU indexes")
    if args.request_max_attempts < 1:
        raise ValueError("request-max-attempts must be positive")
    if args.capabilities is not None:
        if len(set(args.capabilities)) != len(args.capabilities):
            raise ValueError("capability ids must be unique")
        if any(not str(value).strip() for value in args.capabilities):
            raise ValueError("capability ids must be non-empty")


def input_record(
    path: Path,
    *,
    manifest_path: Path | None,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing inference input: {path}")
    record = {
        "path": str(path.relative_to(REPO_ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "master_sample_count": count_jsonl(path),
    }
    record["view_count"] = (
        int(record["master_sample_count"]) * len(CONTEXT_LENGTHS)
    )
    if manifest_path is None:
        return record
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing input manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    expected: str | None = None
    filename = path.name
    files = manifest.get("files", {})
    if isinstance(files, dict):
        file_record = files.get(filename)
        if isinstance(file_record, dict):
            expected = file_record.get("sha256")
    if expected is None and filename == "samples.jsonl":
        file_record = files.get("samples.jsonl")
        if isinstance(file_record, dict):
            expected = file_record.get("sha256")
    if expected != record["sha256"]:
        raise ValueError(
            f"input manifest hash mismatch for {path}: "
            f"{record['sha256']} != {expected}"
        )
    record["manifest_path"] = str(manifest_path.relative_to(REPO_ROOT))
    record["manifest_sha256"] = file_sha256(manifest_path)
    return record


def inference_config(
    args: argparse.Namespace,
    *,
    synthetic: dict[str, Any],
    real_source: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "runner_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "runner_sha256": file_sha256(Path(__file__)),
        "synthetic_input": synthetic,
        "real_source_input": real_source,
        "context_lengths": list(CONTEXT_LENGTHS),
        "horizon": HORIZON,
        "oracle_context_policy": (
            "per model and master sample choose minimum MASE; exact ties "
            "choose contexts in ascending order"
        ),
        "requested_models": list(args.models),
        "synthetic_capability_filter": (
            None
            if args.capabilities is None
            else sorted(args.capabilities)
        ),
        "baseline_models": list(BASELINE_MODELS),
        "model_execution": {
            model_id: dict(MODEL_EXECUTION_CONFIG[model_id])
            for model_id in args.models
        },
        "devices": str(args.devices),
        "request_max_attempts": int(args.request_max_attempts),
        "forecast_timeout_seconds": int(args.forecast_timeout_seconds),
        "model_load_timeout_seconds": int(
            args.model_load_timeout_seconds
        ),
        "service": {
            "base_url": str(args.base_url),
            "api_prefix": str(args.api_prefix),
        },
        "retention_policy": (
            "successful view predictions are append-only and resume by "
            "deterministic view_id; failures are retained separately"
        ),
    }


def prepare_output(
    output_dir: Path,
    *,
    config: dict[str, Any],
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "inference_config.json"
    if config_path.exists():
        if not resume:
            raise FileExistsError(
                "inference_config.json exists; use --resume"
            )
        existing = read_json(config_path)
        comparable_new = {
            key: value for key, value in config.items() if key != "created_at"
        }
        comparable_existing = {
            key: value
            for key, value in existing.items()
            if key != "created_at"
        }
        if comparable_new != comparable_existing:
            raise ValueError("resume inference config does not match")
    else:
        write_json(config_path, config)
    for directory in (
        "predictions",
        "real_source_predictions",
        "failures",
        "real_failures",
        "logs",
    ):
        (output_dir / directory).mkdir(exist_ok=True)


def run_baseline(
    output_dir: Path,
    *,
    model_id: str,
    master_path: Path,
    prediction_kind: str,
) -> dict[str, Any]:
    path = prediction_path_for(
        output_dir,
        model_id,
        prediction_kind=prediction_kind,
    )
    expected = count_jsonl(master_path) * len(CONTEXT_LENGTHS)
    if path.exists():
        observed = count_jsonl(path)
        if observed != expected:
            raise ValueError(
                f"baseline {model_id}/{prediction_kind} incomplete: "
                f"{observed}/{expected}"
            )
        return {
            "model_id": model_id,
            "prediction_kind": prediction_kind,
            "status": "complete",
            "compatible_sample_count": expected,
            "succeeded_count": observed,
            "already_complete_on_entry": True,
        }
    temporary = path.with_suffix(".jsonl.in_progress")
    if temporary.exists():
        temporary.unlink()
    started = time.monotonic()
    with temporary.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(
            iter_forecast_views(master_path),
            start=1,
        ):
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
                    [pattern[offset % period] for offset in range(horizon)]
                )
            row = prediction_row(model_id, "baseline", sample, forecast)
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            if index % 500 == 0:
                handle.flush()
                print(
                    f"{prediction_kind} {model_id}: {index}/{expected}",
                    flush=True,
                )
    os.replace(temporary, path)
    return {
        "model_id": model_id,
        "prediction_kind": prediction_kind,
        "status": "complete",
        "compatible_sample_count": expected,
        "succeeded_count": expected,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "prediction_path": str(path.relative_to(REPO_ROOT)),
    }


def status_on_error(
    *,
    output_dir: Path,
    model: dict[str, Any],
    model_id: str,
    master_path: Path,
    prediction_kind: str,
    error: Exception,
    started: float,
) -> dict[str, Any]:
    path = prediction_path_for(
        output_dir,
        model_id,
        prediction_kind=prediction_kind,
    )
    observed = count_jsonl(path) if path.exists() else 0
    expected = sum(
        engine.model_supports_sample(model, sample)
        for sample in iter_forecast_views(master_path)
    )
    return {
        "model_id": model_id,
        "prediction_kind": prediction_kind,
        "status": "failed",
        "compatible_sample_count": expected,
        "succeeded_count": observed,
        "error": f"{type(error).__name__}: {error}",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "prediction_path": str(path.relative_to(REPO_ROOT)),
    }


def run_models(
    output_dir: Path,
    *,
    config: dict[str, Any],
    synthetic_master_path: Path,
    real_source_path: Path | None,
    args: argparse.Namespace,
) -> None:
    client = engine.TimerServiceClient(
        config["service"]["base_url"],
        config["service"]["api_prefix"],
        timeout_seconds=30,
    )
    install_engine_hooks()
    try:
        catalog = client.list_models()
        write_json(
            output_dir / "inference_model_catalog.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "models": catalog,
            },
        )
        requested = engine.resolve_requested_models(
            catalog,
            config["requested_models"],
        )
        synthetic_statuses = read_status(
            output_dir / "model_status.json"
        )
        real_statuses = read_status(
            output_dir / "real_source_model_status.json"
        )
        for model in requested:
            model_id = str(model["model_id"])
            execution = config["model_execution"][model_id]
            print(
                f"starting {model_id}: replicas/GPU="
                f"{execution['replicas_per_device']}, "
                f"concurrency={execution['http_concurrency']}",
                flush=True,
            )
            client.unload_all_loaded()
            synthetic_started = time.monotonic()
            try:
                synthetic_status = engine.run_one_model(
                    client,
                    model,
                    output_dir=output_dir,
                    execution=execution,
                    devices=config["devices"],
                    request_max_attempts=config["request_max_attempts"],
                    forecast_timeout_seconds=config[
                        "forecast_timeout_seconds"
                    ],
                    load_timeout_seconds=config[
                        "model_load_timeout_seconds"
                    ],
                    keep_loaded=real_source_path is not None,
                    sample_path=synthetic_master_path,
                    prediction_kind="synthetic",
                    status_filename="model_status.json",
                )
            except Exception as error:  # noqa: BLE001
                synthetic_status = status_on_error(
                    output_dir=output_dir,
                    model=model,
                    model_id=model_id,
                    master_path=synthetic_master_path,
                    prediction_kind="synthetic",
                    error=error,
                    started=synthetic_started,
                )
            synthetic_statuses["models"][model_id] = synthetic_status
            write_json(
                output_dir / "model_status.json",
                synthetic_statuses,
            )
            print(
                f"{model_id} synthetic: {synthetic_status['status']} "
                f"{synthetic_status['succeeded_count']}/"
                f"{synthetic_status['compatible_sample_count']}",
                flush=True,
            )

            if real_source_path is not None:
                real_started = time.monotonic()
                try:
                    real_status = engine.run_one_model(
                        client,
                        model,
                        output_dir=output_dir,
                        execution=execution,
                        devices=config["devices"],
                        request_max_attempts=config[
                            "request_max_attempts"
                        ],
                        forecast_timeout_seconds=config[
                            "forecast_timeout_seconds"
                        ],
                        load_timeout_seconds=config[
                            "model_load_timeout_seconds"
                        ],
                        keep_loaded=False,
                        sample_path=real_source_path,
                        prediction_kind="real",
                        status_filename="real_source_model_status.json",
                    )
                except Exception as error:  # noqa: BLE001
                    real_status = status_on_error(
                        output_dir=output_dir,
                        model=model,
                        model_id=model_id,
                        master_path=real_source_path,
                        prediction_kind="real",
                        error=error,
                        started=real_started,
                    )
                real_statuses["models"][model_id] = real_status
                write_json(
                    output_dir / "real_source_model_status.json",
                    real_statuses,
                )
                print(
                    f"{model_id} real source: {real_status['status']} "
                    f"{real_status['succeeded_count']}/"
                    f"{real_status['compatible_sample_count']}",
                    flush=True,
                )
            client.unload_all_loaded()
    finally:
        try:
            client.unload_all_loaded()
        except Exception:  # noqa: BLE001
            pass
        client.close()


def read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "paper_v5_e2_model_status.v1",
            "models": {},
        }
    return read_json(path)


def run_all_baselines(
    output_dir: Path,
    *,
    synthetic_master_path: Path,
    real_source_path: Path | None,
) -> None:
    statuses = read_status(output_dir / "baseline_status.json")
    for model_id in BASELINE_MODELS:
        statuses["models"][model_id] = run_baseline(
            output_dir,
            model_id=model_id,
            master_path=synthetic_master_path,
            prediction_kind="synthetic",
        )
        write_json(output_dir / "baseline_status.json", statuses)
    if real_source_path is None:
        return
    real_statuses = read_status(
        output_dir / "real_source_baseline_status.json"
    )
    for model_id in BASELINE_MODELS:
        real_statuses["models"][model_id] = run_baseline(
            output_dir,
            model_id=model_id,
            master_path=real_source_path,
            prediction_kind="real",
        )
        write_json(
            output_dir / "real_source_baseline_status.json",
            real_statuses,
        )


def main() -> int:
    args = parse_args()
    validate_cli_args(args)
    output_dir = args.output_dir.resolve()
    synthetic_master_path = args.synthetic_master_path.resolve()
    synthetic_manifest_path: Path | None = (
        args.synthetic_manifest_path
        if args.synthetic_manifest_path.is_file()
        else None
    )
    if args.capabilities is not None:
        synthetic_master_path = build_capability_subset_master_file(
            synthetic_master_path,
            output_dir / "capability_subset_master_samples.jsonl",
            args.capabilities,
        )
        synthetic_manifest_path = None
        args.skip_real_source = True
    if args.preflight_one_per_cell:
        if args.resume:
            raise ValueError(
                "--preflight-one-per-cell creates a fresh preflight run"
            )
        synthetic_master_path = build_preflight_master_file(
            synthetic_master_path,
            output_dir / "preflight_master_samples.jsonl",
        )
        synthetic_manifest_path = None
        args.skip_real_source = True
    synthetic_manifest = (
        synthetic_manifest_path
    )
    synthetic = input_record(
        synthetic_master_path,
        manifest_path=synthetic_manifest,
    )
    real_source_path: Path | None = None
    real_source: dict[str, Any] | None = None
    if not args.skip_real_source:
        real_source_path = args.real_source_path.resolve()
        real_source = input_record(
            real_source_path,
            manifest_path=args.real_source_manifest_path,
        )
    config = inference_config(
        args,
        synthetic=synthetic,
        real_source=real_source,
    )
    prepare_output(output_dir, config=config, resume=args.resume)
    if args.stage in {"baselines", "all"}:
        run_all_baselines(
            output_dir,
            synthetic_master_path=synthetic_master_path,
            real_source_path=real_source_path,
        )
    if args.stage in {"models", "all"}:
        run_models(
            output_dir,
            config=config,
            synthetic_master_path=synthetic_master_path,
            real_source_path=real_source_path,
            args=args,
        )
    print(f"Paper v5 E2 inference output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
