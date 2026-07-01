#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (BACKEND_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.metric_service import compute_sample_metrics  # noqa: E402
from app.services.synthetic_generation_service import CAPABILITIES_BY_ID, _generate_accepted_sample_values, _seed_for  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/research/synthetic-v2-real-model-experiment"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs/superpowers/baselines/2026-07-01-synthetic-v2-real-model-experiment.md"

DEFAULT_MODEL_ORDER = ("Timer-3.5", "Timer-3.0", "Chronos-2", "timesfm2.5", "AutoARIMA", "Holt-Winters")
DEFAULT_CAPABILITIES = ("trend", "multi_seasonal")
MULTI_TARGET_DIM = 3
CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24
TIME_COLUMN = "time"


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    capability_id: str
    difficulty: int
    sample_index: int
    history_timestamps: list[str]
    future_timestamps: list[str]
    target_column_names: list[str]
    target_history: list[list[float]]
    target_future: list[list[float]]
    covariate_column_names: list[str]
    history_cov: list[list[float]]
    future_cov: list[list[float]]
    realized_features: dict[str, float]


class TimerServiceClient:
    def __init__(self, base_url: str, api_prefix: str, timeout_seconds: int):
        self.base = base_url.rstrip("/") + "/" + api_prefix.strip("/")
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(timeout=timeout_seconds, trust_env=False)

    def close(self) -> None:
        self.client.close()

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._get("/models/list")
        return list(payload["data"]["models"])

    def unload_all_loaded(self) -> None:
        for model in self.list_models():
            if model.get("loaded"):
                self.unload_model(str(model.get("model_id")))

    def unload_model(self, model_id: str) -> None:
        try:
            self._post("/models/unload", {"model_id": model_id}, timeout_seconds=max(self.timeout_seconds, 600))
        except RuntimeError as exc:
            message = str(exc).lower()
            if "409" in message and "not loaded" in message:
                return
            raise

    def ensure_model_loaded(self, model_id: str, timeout_seconds: int) -> float:
        start = time.monotonic()
        model = self.find_model(model_id)
        if model.get("loaded"):
            return 0.0
        self._post("/models/load", {"model_id": model_id}, timeout_seconds=timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        while True:
            model = self.find_model(model_id)
            if model.get("loaded"):
                return time.monotonic() - start
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for model {model_id} to load")
            time.sleep(min(2.0, remaining))

    def find_model(self, model_id: str) -> dict[str, Any]:
        model = next((item for item in self.list_models() if item.get("model_id") == model_id), None)
        if model is None:
            raise RuntimeError(f"model not found in timer service: {model_id}")
        return model

    def forecast_batch(
        self,
        model_id: str,
        samples: list[ProbeSample],
        *,
        timeout_seconds: int,
        model_params: dict[str, Any] | None = None,
    ) -> list[list[list[float]]]:
        body: dict[str, Any] = {
            "model_id": model_id,
            "targets": [forecast_target(sample) for sample in samples],
            "output_length": [HORIZON for _ in samples],
            "time_col": [TIME_COLUMN for _ in samples],
        }
        if any(sample.covariate_column_names for sample in samples):
            if not all(sample.covariate_column_names for sample in samples):
                raise RuntimeError("cannot mix covariate and non-covariate samples in one forecast batch")
            body["history_covs"] = [forecast_covariates(sample, history=True) for sample in samples]
            body["future_covs"] = [forecast_covariates(sample, history=False) for sample in samples]
        if model_params:
            body["model_params"] = model_params
        payload = self._post("/forecast", body, timeout_seconds=timeout_seconds)
        results = payload.get("data", {}).get("results", [])
        if len(results) != len(samples):
            raise RuntimeError(f"forecast returned {len(results)} results for {len(samples)} samples")
        return [parse_forecast_result(result) for result in results]

    def _get(self, path: str) -> dict[str, Any]:
        response = self.client.get(self.base + path, timeout=self.timeout_seconds)
        return parse_envelope(response)

    def _post(self, path: str, body: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
        response = self.client.post(self.base + path, json=body, timeout=timeout_seconds)
        return parse_envelope(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic v2 probe against real timer-rest-service models.")
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODEL_ORDER), help="Model ids to evaluate. Use 'all-active' for every active service model.")
    parser.add_argument("--capabilities", nargs="+", default=list(DEFAULT_CAPABILITIES), help="Synthetic capability ids to evaluate.")
    parser.add_argument("--sample-count", type=int, default=12, help="Samples per capability and difficulty.")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=900)
    parser.add_argument("--model-load-timeout-seconds", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--keep-loaded", action="store_true", help="Do not unload loaded service models before/after each model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = TimerServiceClient(args.base_url, args.api_prefix, timeout_seconds=30)
    try:
        capabilities = validate_capabilities(args.capabilities)
        requirements = capability_requirements(capabilities)
        service_models = client.list_models()
        selected, skipped = select_models(service_models, args.models, requirements=requirements)
        samples = generate_probe_samples(args.sample_count, capabilities)
        baseline_rows = baseline_metric_rows(samples)
        model_rows, model_run_status = run_models(
            client,
            selected,
            samples,
            batch_size=args.batch_size,
            forecast_timeout_seconds=args.forecast_timeout_seconds,
            load_timeout_seconds=args.model_load_timeout_seconds,
            keep_loaded=args.keep_loaded,
            output_dir=args.output_dir,
        )
    finally:
        client.close()

    rows = [*baseline_rows, *model_rows]
    summary = summarize_results(
        rows,
        selected_models=selected,
        skipped_models=skipped,
        model_run_status=model_run_status,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        base_url=args.base_url,
        requested_models=args.models,
        requested_capabilities=capabilities,
        requirements=requirements,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(summary, output_dir=args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote report: {args.report}")
    print(f"wrote summary: {args.output_dir / 'summary.json'}")
    return 0


def validate_capabilities(capabilities: list[str]) -> list[str]:
    missing = [capability_id for capability_id in capabilities if capability_id not in CAPABILITIES_BY_ID]
    if missing:
        raise SystemExit(f"unknown synthetic capabilities: {', '.join(missing)}")
    return capabilities


def capability_requirements(capabilities: list[str]) -> dict[str, int]:
    target_dim = max(target_dim_for_capability(capability_id) for capability_id in capabilities)
    covariate_dim = max(len(CAPABILITIES_BY_ID[capability_id].covariate_columns) for capability_id in capabilities)
    return {"target_dim": target_dim, "covariate_dim": covariate_dim}


def target_dim_for_capability(capability_id: str) -> int:
    capability = CAPABILITIES_BY_ID[capability_id]
    if capability.target_dim_mode == "multi":
        return MULTI_TARGET_DIM
    return 1


def select_models(
    service_models: list[dict[str, Any]],
    requested: list[str],
    *,
    requirements: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requirements = requirements or {"target_dim": 1, "covariate_dim": 0}
    by_id = {str(model.get("model_id")): model for model in service_models if model.get("model_id")}
    requested_ids = [model["model_id"] for model in service_models if str(model.get("state")).lower() != "inactive"] if requested == ["all-active"] else requested
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for model_id in requested_ids:
        model = by_id.get(model_id)
        if model is None:
            skipped.append({"model_id": model_id, "reason": "not_registered"})
            continue
        if str(model.get("state") or "").lower() == "inactive":
            skipped.append({"model_id": model_id, "reason": "inactive"})
            continue
        limits = model.get("forecast_limits") or {}
        if int(limits.get("min_input_length") or 0) > CONTEXT_LENGTH or int(limits.get("max_output_length") or HORIZON) < HORIZON:
            skipped.append({"model_id": model_id, "reason": "window_unsupported", "forecast_limits": limits})
            continue
        target_dim = int(requirements.get("target_dim") or 1)
        max_target_count = limits.get("max_target_count")
        if target_dim > 1 and ("max_target_count" not in limits or (max_target_count is not None and int(max_target_count) < target_dim)):
            skipped.append({"model_id": model_id, "reason": "target_dim_unsupported", "forecast_limits": limits, "required_target_dim": target_dim})
            continue
        covariate_dim = int(requirements.get("covariate_dim") or 0)
        max_covariate_count = int(limits.get("max_covariate_count") or 0)
        if covariate_dim > 0 and max_covariate_count < covariate_dim:
            skipped.append({"model_id": model_id, "reason": "covariate_dim_unsupported", "forecast_limits": limits, "required_covariate_dim": covariate_dim})
            continue
        selected.append(model)
    return selected, skipped


def generate_probe_samples(sample_count: int, capabilities: list[str]) -> list[ProbeSample]:
    samples: list[ProbeSample] = []
    base_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for capability_id in capabilities:
        target_dim = target_dim_for_capability(capability_id)
        covariate_names = list(CAPABILITIES_BY_ID[capability_id].covariate_columns)
        for difficulty in range(1, 6):
            for sample_index in range(sample_count):
                seed = _seed_for(20260701, capability_id, difficulty * 10_000 + sample_index)
                values, _latent, covariates, features = _generate_accepted_sample_values(
                    capability_id,
                    CONTEXT_LENGTH + HORIZON,
                    CONTEXT_LENGTH,
                    target_dim,
                    SEASON_LENGTH,
                    difficulty,
                    seed,
                )
                offset = len(samples) * (CONTEXT_LENGTH + HORIZON)
                timestamps = [(base_start + timedelta(hours=offset + i)).isoformat() for i in range(CONTEXT_LENGTH + HORIZON)]
                samples.append(
                    ProbeSample(
                        sample_id=f"{capability_id}-d{difficulty}-{sample_index:03d}",
                        capability_id=capability_id,
                        difficulty=difficulty,
                        sample_index=sample_index,
                        history_timestamps=timestamps[:CONTEXT_LENGTH],
                        future_timestamps=timestamps[CONTEXT_LENGTH:],
                        target_column_names=[f"target_{index}" for index in range(target_dim)],
                        target_history=values[:CONTEXT_LENGTH].astype(float).tolist(),
                        target_future=values[CONTEXT_LENGTH:].astype(float).tolist(),
                        covariate_column_names=covariate_names,
                        history_cov=(covariates[:CONTEXT_LENGTH].astype(float).tolist() if covariates is not None else []),
                        future_cov=(covariates[CONTEXT_LENGTH:].astype(float).tolist() if covariates is not None else []),
                        realized_features=features,
                    )
                )
    return samples


def forecast_target(sample: ProbeSample) -> dict[str, Any]:
    return {
        "columns": [TIME_COLUMN, *sample.target_column_names],
        "data": [
            [timestamp, *row]
            for timestamp, row in zip(sample.history_timestamps, sample.target_history, strict=True)
        ],
    }


def forecast_covariates(sample: ProbeSample, *, history: bool) -> dict[str, Any]:
    timestamps = sample.history_timestamps if history else sample.future_timestamps
    rows = sample.history_cov if history else sample.future_cov
    return {
        "columns": [TIME_COLUMN, *sample.covariate_column_names],
        "data": [
            [timestamp, *row]
            for timestamp, row in zip(timestamps, rows, strict=True)
        ],
    }


def parse_forecast_result(result: dict[str, Any]) -> list[list[float]]:
    columns = result["columns"]
    value_indexes = [index for index, column in enumerate(columns) if column != TIME_COLUMN]
    rows = result["data"]
    return [[float(row[index]) for index in value_indexes] for row in rows[:HORIZON]]


def baseline_metric_rows(samples: list[ProbeSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        history = np.asarray(sample.target_history, dtype=float)
        actual = np.asarray(sample.target_future, dtype=float)
        rows.append(metric_row("naive", "baseline", sample, np.repeat(history[-1:], HORIZON, axis=0).tolist()))
        rows.append(metric_row("seasonal_naive", "baseline", sample, history[-SEASON_LENGTH:][:HORIZON].tolist()))
    return rows


def run_models(
    client: TimerServiceClient,
    models: list[dict[str, Any]],
    samples: list[ProbeSample],
    *,
    batch_size: int,
    forecast_timeout_seconds: int,
    load_timeout_seconds: int,
    keep_loaded: bool,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    raw_path = output_dir / "sample_metrics.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    for model in models:
        model_id = str(model["model_id"])
        status: dict[str, Any] = {"model_id": model_id, "status": "pending", "sample_count": len(samples), "failed_count": 0}
        start = time.monotonic()
        try:
            if not keep_loaded:
                client.unload_all_loaded()
            status["load_seconds"] = round(client.ensure_model_loaded(model_id, timeout_seconds=load_timeout_seconds), 3)
            model_rows = forecast_model_samples(
                client,
                model_id,
                samples,
                batch_size=batch_size,
                forecast_timeout_seconds=forecast_timeout_seconds,
                raw_path=raw_path,
            )
            all_rows.extend(model_rows)
            failed_rows = [row for row in model_rows if row.get("status") != "succeeded"]
            status["failed_count"] = len(failed_rows)
            status["succeeded_count"] = len(model_rows) - len(failed_rows)
            if failed_rows:
                status["first_error"] = failed_rows[0].get("error")
            if status["succeeded_count"] == len(samples):
                status["status"] = "succeeded"
            elif status["succeeded_count"] == 0:
                status["status"] = "failed"
            else:
                status["status"] = "partial_succeeded"
        except Exception as exc:  # noqa: BLE001 - record model-level failure and keep later models running.
            status["status"] = "failed"
            status["error"] = str(exc)
        finally:
            status["elapsed_seconds"] = round(time.monotonic() - start, 3)
            statuses.append(status)
            if not keep_loaded:
                try:
                    client.unload_model(model_id)
                except Exception as exc:  # noqa: BLE001
                    status["unload_error"] = str(exc)
    return all_rows, statuses


def forecast_model_samples(
    client: TimerServiceClient,
    model_id: str,
    samples: list[ProbeSample],
    *,
    batch_size: int,
    forecast_timeout_seconds: int,
    raw_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with raw_path.open("a", encoding="utf-8") as handle:
        for group in sample_request_groups(samples):
            for batch in chunks(group, batch_size):
                started = time.monotonic()
                try:
                    forecasts = client.forecast_batch(
                        model_id,
                        batch,
                        timeout_seconds=forecast_timeout_seconds,
                        model_params=model_params_for(model_id),
                    )
                    batch_seconds = time.monotonic() - started
                    for sample, forecast in zip(batch, forecasts, strict=True):
                        row = metric_row(model_id, "timer_service", sample, forecast)
                        row["batch_seconds"] = batch_seconds
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                except Exception as exc:  # noqa: BLE001 - one failed batch should not hide other capability/difficulty batches.
                    batch_seconds = time.monotonic() - started
                    for sample in batch:
                        row = failed_metric_row(model_id, sample, str(exc), batch_seconds)
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return rows


def sample_request_groups(samples: list[ProbeSample]) -> list[list[ProbeSample]]:
    groups: list[list[ProbeSample]] = []
    current: list[ProbeSample] = []
    current_key: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    for sample in samples:
        key = (tuple(sample.target_column_names), tuple(sample.covariate_column_names))
        if current_key is not None and key != current_key:
            groups.append(current)
            current = []
        current_key = key
        current.append(sample)
    if current:
        groups.append(current)
    return groups


def model_params_for(model_id: str) -> dict[str, Any] | None:
    if model_id in {"AutoARIMA", "Holt-Winters"}:
        return {"sp": SEASON_LENGTH, "suppress_warnings": True}
    return None


def metric_row(model_id: str, model_group: str, sample: ProbeSample, forecast: list[list[float]]) -> dict[str, Any]:
    metrics = compute_sample_metrics(sample.target_future, forecast, sample.target_history)
    extra_metrics = extra_sample_metrics(sample, forecast)
    metrics.update(extra_metrics)
    return {
        "model_id": model_id,
        "model_group": model_group,
        "sample_id": sample.sample_id,
        "capability_id": sample.capability_id,
        "difficulty": sample.difficulty,
        "target_dim": len(sample.target_column_names),
        "covariate_dim": len(sample.covariate_column_names),
        "status": "succeeded",
        "metrics": dict(metrics),
        "realized_features": sample.realized_features,
    }


def extra_sample_metrics(sample: ProbeSample, forecast: list[list[float]]) -> dict[str, float]:
    if sample.capability_id != "hierarchical_coherence":
        return {}
    values = np.asarray(forecast, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3:
        return {}
    residual = values[:, 0] - np.sum(values[:, 1:], axis=1)
    return {"coherence_mae": float(np.mean(np.abs(residual)))}


def failed_metric_row(model_id: str, sample: ProbeSample, error: str, batch_seconds: float) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model_group": "timer_service",
        "sample_id": sample.sample_id,
        "capability_id": sample.capability_id,
        "difficulty": sample.difficulty,
        "target_dim": len(sample.target_column_names),
        "covariate_dim": len(sample.covariate_column_names),
        "status": "failed",
        "error": error,
        "batch_seconds": batch_seconds,
    }


def summarize_results(
    rows: list[dict[str, Any]],
    *,
    selected_models: list[dict[str, Any]],
    skipped_models: list[dict[str, Any]],
    model_run_status: list[dict[str, Any]],
    sample_count: int,
    batch_size: int,
    base_url: str,
    requested_models: list[str],
    requested_capabilities: list[str],
    requirements: dict[str, int],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    failed_grouped: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in rows:
        if row.get("status") == "succeeded":
            grouped[(row["model_id"], row["capability_id"], int(row["difficulty"]))].append(row)
        else:
            failed_grouped[(row["model_id"], row["capability_id"], int(row["difficulty"]))] += 1
    summaries = []
    for (model_id, capability_id, difficulty), group_rows in sorted(grouped.items()):
        summaries.append(
            {
                "model_id": model_id,
                "capability_id": capability_id,
                "difficulty": difficulty,
                "sample_count": len(group_rows),
                "target_dim": max(int(row.get("target_dim") or 1) for row in group_rows),
                "covariate_dim": max(int(row.get("covariate_dim") or 0) for row in group_rows),
                "metrics": summarize_metric_rows(group_rows),
                "features": summarize_feature_rows(group_rows),
            }
        )
    return {
        "schema_version": "synthetic_v2_real_model_experiment.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "season_length": SEASON_LENGTH,
        "sample_count_per_capability_difficulty": sample_count,
        "batch_size": batch_size,
        "requirements": requirements,
        "requested_models": requested_models,
        "requested_capabilities": requested_capabilities,
        "selected_models": [model.get("model_id") for model in selected_models],
        "skipped_models": skipped_models,
        "model_run_status": model_run_status,
        "failure_counts": [
            {"model_id": model_id, "capability_id": capability_id, "difficulty": difficulty, "failed_count": failed_count}
            for (model_id, capability_id, difficulty), failed_count in sorted(failed_grouped.items())
        ],
        "summaries": summaries,
        "comparisons": build_comparisons(summaries),
        "reproduction_command": reproduction_command(requested_models, requested_capabilities, sample_count, batch_size),
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.get("metrics", {}).items() if is_finite(value)})
    return {key: float(np.mean([row["metrics"][key] for row in rows if key in row.get("metrics", {}) and is_finite(row["metrics"][key])])) for key in keys}


def summarize_feature_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.get("realized_features", {}).items() if is_finite(value)})
    return {key: float(np.mean([row["realized_features"][key] for row in rows if key in row.get("realized_features", {}) and is_finite(row["realized_features"][key])])) for key in keys}


def build_comparisons(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model_id"], row["capability_id"], row["difficulty"]): row for row in summaries}
    out: list[dict[str, Any]] = []
    for row in summaries:
        if row["model_id"] in {"naive", "seasonal_naive"}:
            continue
        seasonal = by_key.get(("seasonal_naive", row["capability_id"], row["difficulty"]))
        naive = by_key.get(("naive", row["capability_id"], row["difficulty"]))
        out.append(
            {
                "model_id": row["model_id"],
                "capability_id": row["capability_id"],
                "difficulty": row["difficulty"],
                "mae_vs_seasonal_naive": ratio(metric(row, "mae"), metric(seasonal, "mae") if seasonal else None),
                "mae_vs_naive": ratio(metric(row, "mae"), metric(naive, "mae") if naive else None),
                "mase_vs_seasonal_naive": ratio(metric(row, "mase"), metric(seasonal, "mase") if seasonal else None),
            }
        )
    return out


def render_report(summary: dict[str, Any], *, output_dir: Path) -> str:
    capabilities = summary.get("requested_capabilities") or DEFAULT_CAPABILITIES
    capability_text = " / ".join(f"`{capability_id}`" for capability_id in capabilities)
    selected_models = ", ".join(summary["selected_models"]) or "none"
    skipped = ", ".join(f"{item['model_id']} ({item['reason']})" for item in summary["skipped_models"]) or "none"
    status_lines = [
        f"- `{item['model_id']}`: `{item['status']}`, failed={item.get('failed_count', '-')}, elapsed={item.get('elapsed_seconds', '-')}s"
        + (f", error={item.get('error') or item.get('first_error')}" if (item.get("error") or item.get("first_error")) else "")
        for item in summary["model_run_status"]
    ]
    tables = []
    for capability_id in capabilities:
        tables.extend(capability_table(summary, capability_id))
    best_lines = best_model_lines(summary)
    failed_lines = failed_model_lines(summary)
    skipped_observation = skipped_model_observation(summary["skipped_models"])
    return "\n".join(
        [
            "# Synthetic v2 真实模型响应实验",
            "",
            "日期：2026-07-01",
            "",
            "## 目的",
            "",
            f"用本机 `timer-rest-service` 的真实模型验证 synthetic v2 {capability_text} probe 是否能呈现模型能力差异。实验直接调用 `http://127.0.0.1:10810/ai/api/v1/forecast`，并保留 naive / seasonal naive 作为基线。",
            "",
            "## 配置",
            "",
            f"- 服务：`{summary['base_url']}`",
            f"- context / horizon / season：`{summary['context_length']} / {summary['horizon']} / {summary['season_length']}`",
            f"- 每个能力每个难度样本数：`{summary['sample_count_per_capability_difficulty']}`",
            f"- batch size：`{summary['batch_size']}`",
            f"- 能力维度：{capability_text}",
            f"- required target / covariate dim：`{summary.get('requirements', {}).get('target_dim', 1)} / {summary.get('requirements', {}).get('covariate_dim', 0)}`",
            f"- requested 模型：{', '.join(summary['requested_models'])}",
            f"- 参评 active 模型：{selected_models}",
            f"- 跳过模型：{skipped}",
            f"- runtime 输出：`{display_path(output_dir)}`",
            "",
            "## 模型运行状态",
            "",
            *status_lines,
            "",
            "## 结果汇总",
            "",
            *tables,
            "",
            "## 初步观察",
            "",
            *best_lines,
            *failed_lines,
            *skipped_observation,
            "- 这份结果用于观察真实模型响应，不替代后续更大样本、多随机种子和更多能力维度的论文主实验。",
            "",
            "## 复现",
            "",
            "```bash",
            summary["reproduction_command"],
            "```",
            "",
        ]
    )


def reproduction_command(requested_models: list[str], requested_capabilities: list[str], sample_count: int, batch_size: int) -> str:
    model_args = " ".join(shlex.quote(model_id) for model_id in requested_models)
    capability_args = " ".join(shlex.quote(capability_id) for capability_id in requested_capabilities)
    return (
        "cd backend && PYTHONPATH=.:../scripts uv run python "
        f"../scripts/run_synthetic_v2_real_model_experiment.py --models {model_args} "
        f"--capabilities {capability_args} "
        f"--sample-count {sample_count} --batch-size {batch_size}"
    )


def skipped_model_observation(skipped_models: list[dict[str, Any]]) -> list[str]:
    if not skipped_models:
        return []
    skipped = ", ".join(f"`{item['model_id']}`（{item['reason']}）" for item in skipped_models)
    return [f"- 本轮跳过模型：{skipped}。"]


def failed_model_lines(summary: dict[str, Any]) -> list[str]:
    failed = [
        item
        for item in summary.get("model_run_status", [])
        if item.get("status") == "failed" and int(item.get("failed_count", 0)) > 0
    ]
    if not failed:
        return []
    models = ", ".join(f"`{item['model_id']}`" for item in failed)
    return [f"- {models} 本轮所有样本均返回失败，暂不纳入能力排序；需要先排查对应推理 worker。"]


def capability_table(summary: dict[str, Any], capability_id: str) -> list[str]:
    model_ids = ["naive", "seasonal_naive", *summary["selected_models"]]
    lines = [
        f"### `{capability_id}`",
        "",
        "| Model | Fail | MAE d1 | MAE d3 | MAE d5 | MASE d1 | MASE d3 | MASE d5 | MAE d5 / SNaive d5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id in model_ids:
        by_diff = {
            row["difficulty"]: row
            for row in summary["summaries"]
            if row["model_id"] == model_id and row["capability_id"] == capability_id
        }
        fail_count = capability_fail_count(summary, model_id, capability_id)
        if not by_diff and fail_count == 0:
            continue
        ratio_d5 = comparison_ratio(summary, model_id, capability_id, 5, "mae_vs_seasonal_naive")
        lines.append(
            "| "
            + " | ".join(
                [
                    model_id,
                    str(fail_count),
                    fmt(metric(by_diff.get(1), "mae")),
                    fmt(metric(by_diff.get(3), "mae")),
                    fmt(metric(by_diff.get(5), "mae")),
                    fmt(metric(by_diff.get(1), "mase")),
                    fmt(metric(by_diff.get(3), "mase")),
                    fmt(metric(by_diff.get(5), "mase")),
                    fmt(ratio_d5),
                ]
            )
            + " |"
        )
    return [*lines, ""]


def capability_fail_count(summary: dict[str, Any], model_id: str, capability_id: str) -> int:
    failure_counts = summary.get("failure_counts")
    if failure_counts:
        return sum(
            int(row.get("failed_count", 0))
            for row in failure_counts
            if row.get("model_id") == model_id and row.get("capability_id") == capability_id
        )
    return model_fail_count(summary, model_id)


def best_model_lines(summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for capability_id in summary.get("requested_capabilities") or DEFAULT_CAPABILITIES:
        rows = [
            row
            for row in summary["summaries"]
            if row["capability_id"] == capability_id and row["model_id"] not in {"naive", "seasonal_naive"}
        ]
        by_model: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = metric(row, "mae")
            if value is not None:
                by_model[row["model_id"]].append(value)
        if not by_model:
            lines.append(f"- `{capability_id}`：没有真实模型成功结果。")
            continue
        averages = {model_id: float(np.mean(values)) for model_id, values in by_model.items()}
        best = min(averages.items(), key=lambda item: item[1])
        lines.append(f"- `{capability_id}`：平均 MAE 最低的是 `{best[0]}`（{fmt(best[1])}）。")
    return lines


def model_fail_count(summary: dict[str, Any], model_id: str) -> int:
    status = next((item for item in summary["model_run_status"] if item["model_id"] == model_id), None)
    return int(status.get("failed_count", 0)) if status else 0


def comparison_ratio(summary: dict[str, Any], model_id: str, capability_id: str, difficulty: int, key: str) -> float | None:
    item = next(
        (
            row
            for row in summary["comparisons"]
            if row["model_id"] == model_id and row["capability_id"] == capability_id and row["difficulty"] == difficulty
        ),
        None,
    )
    return item.get(key) if item else None


def parse_envelope(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{response.request.url} returned non-json response: {response.text[:200]}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"{response.request.url} returned {response.status_code}: {payload.get('message', response.text)}")
    if isinstance(payload, dict) and payload.get("code") not in (None, 200):
        raise RuntimeError(f"{response.request.url} returned service code {payload.get('code')}: {payload.get('message')}")
    return payload


def chunks(values: list[ProbeSample], size: int) -> list[list[ProbeSample]]:
    return [values[index : index + size] for index in range(0, len(values), max(1, size))]


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get("metrics", {}).get(key)
    return float(value) if is_finite(value) else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value: float | None) -> str:
    if value is None or not is_finite(value):
        return "-"
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
