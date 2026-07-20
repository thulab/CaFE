#!/usr/bin/env python3
"""Run the formal Paper v7 E3 future-covariate paired ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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

import analyze_paper_v5_e3_mechanism_fidelity as e3  # noqa: E402
import run_paper_e2_dynamic_stability as engine  # noqa: E402
import run_paper_v5_e2_inference as e2  # noqa: E402


SCHEMA_VERSION = "paper_v7_e3_covariate_ablation.v1"
ABLATION_ID = "future_covariates_zero"
EXPECTED_PAIRED_GROUPS_PER_CELL = 320
DEFAULT_E2_DIR = REPO_ROOT / "runtime/paper_exp/v7/E2_dynamic_stability"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "runtime/paper_exp/v7/E3_mechanism_fidelity/"
    "covariate_ablation_predictions"
)
DEFAULT_MODELS = tuple(e2.DEFAULT_MODELS)
MODEL_EXECUTION_CONFIG = {
    model_id: dict(e2.MODEL_EXECUTION_CONFIG[model_id])
    for model_id in DEFAULT_MODELS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or reuse one oracle-context counterfactual per supported "
            "covariate-response sample after zeroing only the known-future "
            "covariates."
        )
    )
    parser.add_argument("--e2-dir", type=Path, default=DEFAULT_E2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--request-max-attempts", type=int, default=3)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-load-timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Do not contact a model service; validate all requested model "
            "files and write manifest.json."
        ),
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_ablation_view(
    master: dict[str, Any],
    *,
    context_length: int,
) -> dict[str, Any]:
    view = e2.master_view(master, context_length)
    covariates = np.asarray(view["covariates"], dtype=float)
    expected_shape = (
        context_length + int(view["horizon"]),
        int(view["covariate_dim"]),
    )
    if covariates.shape != expected_shape:
        raise ValueError(
            f"unexpected covariate shape: {covariates.shape} != "
            f"{expected_shape}"
        )
    history_sha256 = array_sha256(covariates[:context_length])
    covariates[context_length:] = 0.0
    master_id = str(view["master_sample_id"])
    sample_id = (
        f"{master_id}__L{context_length}__{ABLATION_ID}"
    )
    return {
        **view,
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "view_id": sample_id,
        "ablation": ABLATION_ID,
        "covariates": covariates.tolist(),
        "history_covariates_sha256": history_sha256,
        "ablated_future_covariates_sha256": array_sha256(
            covariates[context_length:]
        ),
    }


def build_model_input(
    destination: Path,
    *,
    samples: dict[str, dict[str, Any]],
    oracle_selection: dict[str, dict[str, Any]],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for master_id in sorted(oracle_selection):
            master = samples[master_id]
            view = build_ablation_view(
                master,
                context_length=int(
                    oracle_selection[master_id]["oracle_context"]
                ),
            )
            handle.write(
                json.dumps(
                    view,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, destination)
    return destination


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def partition_counterfactual_samples(
    intact_predictions: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str]]:
    native_http: set[str] = set()
    reused: set[str] = set()
    for master_id, prediction in intact_predictions.items():
        mode = e3.expected_counterfactual_mode(prediction)
        if mode == "reuse_intact_forecast_covariates_omitted":
            reused.add(master_id)
        elif mode == "native_future_covariate_ablation_http":
            native_http.add(master_id)
        else:
            raise ValueError(f"unknown counterfactual mode: {mode}")
    if native_http & reused:
        raise AssertionError("counterfactual partitions overlap")
    if native_http | reused != set(intact_predictions):
        raise AssertionError("counterfactual partitions are incomplete")
    return native_http, reused


def reused_counterfactual_rows(
    *,
    model_id: str,
    samples: dict[str, dict[str, Any]],
    oracle_selection: dict[str, dict[str, Any]],
    intact_predictions: dict[str, dict[str, Any]],
    master_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for master_id in sorted(master_ids):
        view = build_ablation_view(
            samples[master_id],
            context_length=int(
                oracle_selection[master_id]["oracle_context"]
            ),
        )
        intact = intact_predictions[master_id]
        row = ablation_prediction_row(
            model_id,
            str(intact.get("model_group", "timer_service")),
            view,
            intact["forecast"],
            counterfactual_mode=(
                "reuse_intact_forecast_covariates_omitted"
            ),
            counterfactual_http_request_count=0,
            intact_input_adaptation=intact.get("input_adaptation"),
        )
        row["counterfactual_effect_mae"] = 0.0
        rows.append(row)
    return rows


def native_counterfactual_rows(
    path: Path,
    *,
    intact_predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in iter_jsonl(path):
        master_id = str(raw["master_sample_id"])
        intact = intact_predictions[master_id]
        forecast = np.asarray(raw["forecast"], dtype=float)
        intact_forecast = np.asarray(intact["forecast"], dtype=float)
        raw.update(
            {
                "counterfactual_mode": (
                    "native_future_covariate_ablation_http"
                ),
                "counterfactual_http_request_count": int(
                    raw.get("successful_http_request_count", 1)
                ),
                "reused_intact_forecast": False,
                "intact_input_adaptation": intact.get(
                    "input_adaptation"
                ),
                "counterfactual_effect_mae": float(
                    np.mean(np.abs(intact_forecast - forecast))
                ),
            }
        )
        rows.append(raw)
    return rows


def iter_ablation_samples(path: Path) -> Iterator[dict[str, Any]]:
    yield from iter_jsonl(path)


def ablation_prediction_path(
    output_dir: Path,
    model_id: str,
    *,
    prediction_kind: str = "synthetic",
) -> Path:
    del prediction_kind
    return output_dir / f"{engine.safe_filename(model_id)}.jsonl"


def ablation_prediction_row(
    model_id: str,
    model_group: str,
    sample: dict[str, Any],
    forecast: np.ndarray | list[list[float]],
    *,
    counterfactual_mode: str | None = None,
    counterfactual_http_request_count: int | None = None,
    intact_input_adaptation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(forecast, dtype=float)
    expected_shape = (
        int(sample["horizon"]),
        int(sample["target_dim"]),
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"forecast shape mismatch: {values.shape} != {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("ablation forecast contains non-finite values")
    row = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "model_group": model_group,
        "sample_id": str(sample["sample_id"]),
        "master_sample_id": str(sample["master_sample_id"]),
        "dataset_id": str(sample["dataset_id"]),
        "task_id": str(sample["task_id"]),
        "capability_id": str(sample["capability_id"]),
        "intensity": int(sample["intensity"]),
        "paired_group_id": str(sample["paired_group_id"]),
        "context_length": int(sample["context_length"]),
        "horizon": int(sample["horizon"]),
        "target_dim": int(sample["target_dim"]),
        "covariate_dim": int(sample["covariate_dim"]),
        "ablation": ABLATION_ID,
        "history_covariates_sha256": str(
            sample["history_covariates_sha256"]
        ),
        "ablated_future_covariates_sha256": str(
            sample["ablated_future_covariates_sha256"]
        ),
        "forecast": values.tolist(),
    }
    if counterfactual_mode is not None:
        row.update(
            {
                "counterfactual_mode": counterfactual_mode,
                "counterfactual_http_request_count": int(
                    counterfactual_http_request_count or 0
                ),
                "reused_intact_forecast": counterfactual_mode.startswith(
                    "reuse_"
                ),
                "intact_input_adaptation": intact_input_adaptation,
            }
        )
    return row


def install_engine_hooks() -> None:
    engine.iter_forecast_samples = iter_ablation_samples
    engine.prediction_path_for = ablation_prediction_path
    engine.prediction_row = ablation_prediction_row


def selected_samples_and_oracles(
    e2_dir: Path,
    model_ids: list[str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
]:
    supported_cells = tuple(
        cell
        for cell in e3.discover_supported_cells(e2_dir)
        if cell[2] == "covariate_response"
    )
    if not supported_cells:
        raise ValueError(
            "sealed generation suite has no supported covariate_response cells"
        )
    samples = e3.load_selected_samples(
        e2_dir,
        supported_cells=supported_cells,
        max_paired_groups_per_cell=0,
    )
    counts: dict[tuple[str, str, str], set[str]] = {}
    for sample in samples.values():
        key = (
            str(sample["dataset_id"]),
            str(sample["task_id"]),
            str(sample["capability_id"]),
        )
        counts.setdefault(key, set()).add(
            str(sample["paired_group_id"])
        )
    incomplete = {
        key: len(group_ids)
        for key, group_ids in counts.items()
        if len(group_ids) != EXPECTED_PAIRED_GROUPS_PER_CELL
    }
    if incomplete:
        raise ValueError(
            "v7 covariate ablation requires 320 paired groups per cell: "
            f"{incomplete}"
        )
    selections, _scores = e3.load_oracle_selection(
        e2_dir,
        model_ids=model_ids,
        samples=samples,
    )
    for model_id in model_ids:
        if set(selections[model_id]) != set(samples):
            raise ValueError(
                f"{model_id} does not fully support covariate_response"
            )
    return samples, selections


def run_models(args: argparse.Namespace) -> None:
    model_ids = [str(value) for value in args.models]
    unknown = sorted(set(model_ids) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(
            f"models do not support the formal ablation: {unknown}"
        )
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("model ids must be unique")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inputs").mkdir(exist_ok=True)
    (output_dir / "failures").mkdir(exist_ok=True)
    (output_dir / "status").mkdir(exist_ok=True)
    samples, selections = selected_samples_and_oracles(
        args.e2_dir,
        model_ids,
    )
    intact_predictions = {
        model_id: e3.load_selected_predictions(
            args.e2_dir,
            model_id=model_id,
            oracle_selection=selections[model_id],
            require_input_adaptation=True,
        )
        for model_id in model_ids
    }
    install_engine_hooks()
    client = engine.TimerServiceClient(
        str(args.base_url),
        str(args.api_prefix),
        timeout_seconds=30,
    )
    try:
        catalog = client.list_models()
        models = engine.resolve_requested_models(catalog, model_ids)
        for model in models:
            model_id = str(model["model_id"])
            prediction_path = ablation_prediction_path(
                output_dir,
                model_id,
            )
            expected_ids = set(selections[model_id])
            if prediction_path.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"{prediction_path} exists; use --resume"
                    )
                existing = list(iter_jsonl(prediction_path))
                existing_ids = {
                    str(row["master_sample_id"]) for row in existing
                }
                if (
                    len(existing) == len(expected_ids)
                    and existing_ids == expected_ids
                ):
                    print(
                        f"{model_id} ablation already complete; skipping",
                        flush=True,
                    )
                    continue
                raise ValueError(
                    f"{model_id} final ablation file is incomplete; "
                    "remove it and resume from the HTTP staging directory"
                )
            native_ids, reused_ids = partition_counterfactual_samples(
                intact_predictions[model_id]
            )
            print(
                f"starting {model_id} future-covariate ablation: "
                f"{len(expected_ids)} oracle views "
                f"({len(native_ids)} native HTTP, "
                f"{len(reused_ids)} reused with zero HTTP)",
                flush=True,
            )
            combined_rows = reused_counterfactual_rows(
                model_id=model_id,
                samples=samples,
                oracle_selection=selections[model_id],
                intact_predictions=intact_predictions[model_id],
                master_ids=reused_ids,
            )
            http_status: dict[str, Any] | None = None
            input_path: Path | None = None
            if native_ids:
                native_selection = {
                    master_id: selections[model_id][master_id]
                    for master_id in sorted(native_ids)
                }
                input_path = build_model_input(
                    output_dir
                    / "inputs"
                    / f"{engine.safe_filename(model_id)}.jsonl",
                    samples=samples,
                    oracle_selection=native_selection,
                )
                staging_dir = (
                    output_dir
                    / "http_counterfactual"
                    / engine.safe_filename(model_id)
                )
                (staging_dir / "failures").mkdir(
                    parents=True,
                    exist_ok=True,
                )
                (staging_dir / "status").mkdir(exist_ok=True)
                http_status = engine.run_one_model(
                    client,
                    model,
                    output_dir=staging_dir,
                    execution=MODEL_EXECUTION_CONFIG[model_id],
                    devices=str(args.devices),
                    request_max_attempts=int(args.request_max_attempts),
                    forecast_timeout_seconds=int(
                        args.forecast_timeout_seconds
                    ),
                    load_timeout_seconds=int(
                        args.model_load_timeout_seconds
                    ),
                    keep_loaded=False,
                    sample_path=input_path,
                    prediction_kind="synthetic",
                    status_filename=(
                        f"status/{engine.safe_filename(model_id)}.json"
                    ),
                    input_adaptation_policy=(
                        engine.INPUT_ADAPTATION_POLICY_ID
                    ),
                )
                if http_status["status"] != "complete":
                    raise RuntimeError(
                        f"{model_id} native ablation is incomplete: "
                        f"{http_status}"
                    )
                combined_rows.extend(
                    native_counterfactual_rows(
                        ablation_prediction_path(staging_dir, model_id),
                        intact_predictions=intact_predictions[model_id],
                    )
                )
            combined_rows.sort(
                key=lambda row: str(row["master_sample_id"])
            )
            combined_ids = {
                str(row["master_sample_id"]) for row in combined_rows
            }
            if (
                len(combined_rows) != len(expected_ids)
                or combined_ids != expected_ids
            ):
                raise ValueError(
                    f"{model_id} combined ablation coverage mismatch: "
                    f"{len(combined_rows)}/{len(expected_ids)}"
                )
            write_jsonl(prediction_path, combined_rows)
            successful_http_count = sum(
                int(row["counterfactual_http_request_count"])
                for row in combined_rows
            )
            status = {
                "model_id": model_id,
                "status": "complete",
                "expected_original_view_count": len(expected_ids),
                "succeeded_original_view_count": len(combined_rows),
                "native_counterfactual_view_count": len(native_ids),
                "reused_counterfactual_view_count": len(reused_ids),
                "counterfactual_http_request_count": (
                    successful_http_count
                ),
                "http_engine_status": http_status,
            }
            status.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "ablation": ABLATION_ID,
                    "service_base_url": str(args.base_url),
                    "input_path": (
                        str(input_path) if input_path is not None else None
                    ),
                    "input_sha256": (
                        file_sha256(input_path)
                        if input_path is not None
                        else None
                    ),
                    "input_adaptation_policy": (
                        engine.INPUT_ADAPTATION_POLICY_ID
                    ),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_json(
                output_dir
                / "status"
                / f"{engine.safe_filename(model_id)}.json",
                status,
            )
            print(
                f"{model_id} ablation complete: "
                f"{status['succeeded_original_view_count']}/"
                f"{status['expected_original_view_count']} original views; "
                f"{successful_http_count} HTTP requests",
                flush=True,
            )
    finally:
        try:
            client.unload_all_loaded()
        except Exception as error:  # noqa: BLE001
            print(f"warning: final unload failed: {error}", flush=True)
        client.close()


def finalize_manifest(
    output_dir: Path,
    *,
    e2_dir: Path,
    model_ids: list[str],
) -> None:
    samples, selections = selected_samples_and_oracles(e2_dir, model_ids)
    intact_predictions = {
        model_id: e3.load_selected_predictions(
            e2_dir,
            model_id=model_id,
            oracle_selection=selections[model_id],
            require_input_adaptation=True,
        )
        for model_id in model_ids
    }
    files: dict[str, Any] = {}
    for model_id in model_ids:
        path = ablation_prediction_path(output_dir, model_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing ablation output: {path}")
        rows = list(iter_jsonl(path))
        expected_ids = set(selections[model_id])
        observed_ids = {str(row["master_sample_id"]) for row in rows}
        if len(rows) != len(expected_ids) or observed_ids != expected_ids:
            raise ValueError(
                f"{model_id} ablation coverage mismatch: "
                f"{len(rows)}/{len(expected_ids)}"
            )
        if any(
            str(row.get("ablation")) != ABLATION_ID
            for row in rows
        ):
            raise ValueError(f"{model_id} contains a wrong ablation id")
        if any(
            int(row["context_length"])
            != int(
                selections[model_id][str(row["master_sample_id"])][
                    "oracle_context"
                ]
            )
            for row in rows
        ):
            raise ValueError(
                f"{model_id} contains a non-oracle context"
            )
        native_ids, reused_ids = partition_counterfactual_samples(
            intact_predictions[model_id]
        )
        rows_by_id = {
            str(row["master_sample_id"]): row for row in rows
        }
        for master_id in expected_ids:
            row = rows_by_id[master_id]
            expected_mode = e3.expected_counterfactual_mode(
                intact_predictions[model_id][master_id]
            )
            if str(row.get("counterfactual_mode")) != expected_mode:
                raise ValueError(
                    f"{model_id}/{master_id} counterfactual mode mismatch"
                )
            if master_id in reused_ids:
                if int(
                    row.get("counterfactual_http_request_count", -1)
                ) != 0:
                    raise ValueError(
                        f"{model_id}/{master_id} reused forecast used HTTP"
                    )
                if not np.array_equal(
                    np.asarray(row["forecast"], dtype=float),
                    np.asarray(
                        intact_predictions[model_id][master_id]["forecast"],
                        dtype=float,
                    ),
                ):
                    raise ValueError(
                        f"{model_id}/{master_id} did not exactly reuse intact "
                        "forecast"
                    )
                if float(row.get("counterfactual_effect_mae", -1.0)) != 0.0:
                    raise ValueError(
                        f"{model_id}/{master_id} reused effect is not zero"
                    )
            elif master_id in native_ids:
                if int(
                    row.get("counterfactual_http_request_count", 0)
                ) < 1:
                    raise ValueError(
                        f"{model_id}/{master_id} native counterfactual has "
                        "no successful HTTP request"
                    )
                counterfactual_plan = row.get("input_adaptation")
                if (
                    not isinstance(counterfactual_plan, dict)
                    or counterfactual_plan.get("covariate_mode") != "native"
                ):
                    raise ValueError(
                        f"{model_id}/{master_id} native counterfactual lost "
                        "covariate provenance"
                    )
        http_request_count = sum(
            int(row.get("counterfactual_http_request_count", 0))
            for row in rows
        )
        files[path.name] = {
            "model_id": model_id,
            "row_count": len(rows),
            "native_counterfactual_view_count": len(native_ids),
            "reused_counterfactual_view_count": len(reused_ids),
            "counterfactual_http_request_count": http_request_count,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    cell_group_counts: dict[tuple[str, str, str], int] = {}
    for cell_key in sorted(
        {
            (
                str(sample["dataset_id"]),
                str(sample["task_id"]),
                str(sample["capability_id"]),
            )
            for sample in samples.values()
        }
    ):
        group_ids = {
            str(sample["paired_group_id"])
            for sample in samples.values()
            if (
                str(sample["dataset_id"]),
                str(sample["task_id"]),
                str(sample["capability_id"]),
            )
            == cell_key
        }
        cell_group_counts[cell_key] = len(group_ids)
    supported_cells = [
        {
            "dataset_id": key[0],
            "task_id": key[1],
            "capability_id": key[2],
            "paired_group_count": count,
        }
        for key, count in sorted(cell_group_counts.items())
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ablation": ABLATION_ID,
        "models": model_ids,
        "supported_cells": supported_cells,
        "supported_cell_count": len(supported_cells),
        "paired_group_count": sum(cell_group_counts.values()),
        "paired_group_count_per_cell": {
            "/".join(key): count
            for key, count in sorted(cell_group_counts.items())
        },
        "intensity_count": 5,
        "expected_rows_per_model": len(samples),
        "context_policy": (
            "reuse each model and master sample's intact E2 oracle context"
        ),
        "intervention": (
            "keep normalized history covariates intact and set only all "
            "known-future covariates to zero"
        ),
        "input_adaptation_policy": {
            "policy_id": engine.INPUT_ADAPTATION_POLICY_ID,
            "covariates_omitted": (
                "reuse_intact_forecast_effect_zero_no_http"
            ),
            "covariates_native": (
                "execute_true_zero_future_covariate_counterfactual"
            ),
        },
        "source_e2_dir": str(e2_dir.resolve()),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "files": files,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"ablation manifest complete: {output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    model_ids = [str(value) for value in args.models]
    if args.finalize_only:
        finalize_manifest(
            args.output_dir.resolve(),
            e2_dir=args.e2_dir.resolve(),
            model_ids=model_ids,
        )
        return
    run_models(args)


if __name__ == "__main__":
    main()
