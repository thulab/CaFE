#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (BACKEND_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.metric_service import compute_sample_metrics, mase_unavailable_reason  # noqa: E402
from app.services.synthetic_generation_service import (  # noqa: E402
    CAPABILITIES_BY_ID,
    _generate_accepted_sample_values,
    _seed_for,
)
from app.services.synthetic_near_distance_gate import evaluate_near_distance_gate  # noqa: E402
from synthetic_feature_profile import (  # noqa: E402
    DEFAULT_FEATURES,
    TSFSeriesRecord,
    WindowSpec,
    feature_vector,
    read_tsf_series_records,
    select_tsf_windows,
)


DEFAULT_DATASET = REPO_ROOT / "runtime/research/m4_hourly_dataset.zip"
DEFAULT_PROFILE = REPO_ROOT / "runtime/research/synthetic-v2-profile-smoke-expanded/m4_hourly_daily_168ctx.json"
DEFAULT_CALIBRATION_SUMMARY = REPO_ROOT / "runtime/research/synthetic-v2-near-distance-calibration/summary.json"
DEFAULT_NEAR_DISTANCE_ARTIFACT = BACKEND_DIR / "app/data/synthetic_v2_near_distance_artifact.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/research/m4-hourly-full-chain"

CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24
TIME_COLUMN = "time"
M4_PROFILE_ID = "m4_hourly_daily_168ctx"

DEFAULT_CAPABILITIES = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
)
DEFAULT_MODEL_ORDER = (
    "Timer-3.5",
    "Timer-3.0",
    "Chronos-2",
    "moirai2",
    "toto2.0",
    "timesfm2.5",
    "AutoARIMA",
    "Holt-Winters",
)
BASELINE_MODEL_IDS = ("naive", "seasonal_naive")
RANK_METRIC = "mase"

CAPABILITY_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "trend": ("trend_strength", "slope_abs", "curvature_abs"),
    "multi_seasonal": ("seasonal_strength", "multi_period_score"),
    "time_varying_seasonality": ("seasonal_amplitude_modulation", "seasonal_phase_variation"),
    "regime_switching": ("level_shift_strength", "volatility_shift_strength", "change_point_shift_energy"),
    "nonlinear_persistence": ("nonlinear_multi_lag_gain",),
    "predictable_intermittency": ("burst_rate", "spike_rate", "outlier_rate", "noise_ratio"),
}


@dataclass(frozen=True)
class ExperimentSample:
    sample_id: str
    dataset_kind: str
    source_dataset: str
    capability_id: str
    intensity: int
    sample_index: int
    series_id: str | None
    window_start: int | None
    history_timestamps: list[str]
    future_timestamps: list[str]
    target_column_names: list[str]
    target_history: list[list[float]]
    target_future: list[list[float]]
    realized_features: dict[str, float]
    validation: dict[str, Any]

    @property
    def feature_dimension(self) -> str:
        if self.dataset_kind == "real":
            return "m4_original"
        return self.capability_id


class TimerServiceClient:
    def __init__(self, base_url: str, api_prefix: str, timeout_seconds: int = 30):
        self.base = base_url.rstrip("/") + "/" + api_prefix.strip("/")
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(timeout=timeout_seconds, trust_env=False)

    def close(self) -> None:
        self.client.close()

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._get("/models/list")
        return list(payload["data"]["models"])

    def find_model(self, model_id: str) -> dict[str, Any]:
        model = next((item for item in self.list_models() if item.get("model_id") == model_id), None)
        if model is None:
            raise RuntimeError(f"model not found in timer service: {model_id}")
        return model

    def ensure_model_loaded(self, model_id: str, timeout_seconds: int) -> float:
        start = time.monotonic()
        if self.find_model(model_id).get("loaded"):
            return 0.0
        self._post("/models/load", {"model_id": model_id}, timeout_seconds=timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        poll_seconds = 1.0
        while True:
            if self.find_model(model_id).get("loaded"):
                return time.monotonic() - start
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for model {model_id} to load")
            time.sleep(min(poll_seconds, remaining))
            poll_seconds = min(poll_seconds * 2.0, 30.0)

    def unload_model(self, model_id: str, *, timeout_seconds: int = 120) -> None:
        try:
            self._post("/models/unload", {"model_id": model_id}, timeout_seconds=timeout_seconds)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "409" in message and "not loaded" in message:
                return
            raise

    def forecast_batch(
        self,
        model_id: str,
        samples: list[ExperimentSample],
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
    parser = argparse.ArgumentParser(
        description="Run the M4 hourly distribution/generation/validation/model full-chain experiment."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--calibration-summary", type=Path, default=DEFAULT_CALIBRATION_SUMMARY)
    parser.add_argument("--near-distance-artifact", type=Path, default=DEFAULT_NEAR_DISTANCE_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["all-active"],
        help="Model ids to evaluate. Use all-active for every active service model.",
    )
    parser.add_argument("--capabilities", nargs="+", default=list(DEFAULT_CAPABILITIES))
    parser.add_argument("--sample-count", type=int, default=3, help="Synthetic samples per capability and intensity.")
    parser.add_argument("--real-sample-count", type=int, default=48, help="Original M4 hourly windows to evaluate.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=900)
    parser.add_argument("--model-load-timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-unload-timeout-seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--skip-service", action="store_true", help="Run baselines only and skip timer service models.")
    parser.set_defaults(unload_after_model=True)
    parser.add_argument(
        "--unload-after-model",
        action="store_true",
        dest="unload_after_model",
        help="Unload each timer service model after it finishes. This is the default.",
    )
    parser.add_argument(
        "--keep-model-loaded",
        action="store_false",
        dest="unload_after_model",
        help="Keep timer service models loaded after evaluation. Use only for local adapter debugging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing outputs to {display_path(args.output_dir)}")

    distribution_rows = distribution_status_rows(
        profile_path=args.profile,
        artifact_path=args.near_distance_artifact,
        calibration_summary_path=args.calibration_summary,
    )
    print("loading M4 hourly real windows")
    real_samples = load_real_m4_samples(args.dataset, sample_count=args.real_sample_count)
    print(f"generated {len(real_samples)} real M4 windows")

    print("generating synthetic samples with feature and near-distance acceptance")
    synthetic_samples, generation_failures = generate_synthetic_samples(
        capabilities=args.capabilities,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    print(f"generated {len(synthetic_samples)} accepted synthetic windows")
    if generation_failures:
        print(f"generation failures: {len(generation_failures)}")

    samples = [*real_samples, *synthetic_samples]
    write_jsonl(args.output_dir / "samples.jsonl", [sample_row(sample) for sample in samples])
    if generation_failures:
        write_jsonl(args.output_dir / "generation_failures.jsonl", generation_failures)

    metric_rows: list[dict[str, Any]] = []
    forecast_points: list[dict[str, Any]] = []
    print("running naive and seasonal naive baselines")
    baseline_rows, baseline_points = baseline_predictions(samples)
    metric_rows.extend(baseline_rows)
    forecast_points.extend(baseline_points)

    selected_models: list[dict[str, Any]] = []
    skipped_models: list[dict[str, Any]] = []
    model_status: list[dict[str, Any]] = []
    if args.skip_service:
        model_status.append({"model_id": "timer_service", "status": "skipped", "reason": "skip_service"})
    else:
        client = TimerServiceClient(args.base_url, args.api_prefix)
        try:
            service_models = client.list_models()
            selected_models, skipped_models = select_models(service_models, args.models)
            print("selected service models: " + ", ".join(model["model_id"] for model in selected_models))
            service_rows, service_points, model_status = run_service_models(
                client,
                selected_models,
                samples,
                batch_size=args.batch_size,
                forecast_timeout_seconds=args.forecast_timeout_seconds,
                load_timeout_seconds=args.model_load_timeout_seconds,
                unload_timeout_seconds=args.model_unload_timeout_seconds,
                unload_after_model=args.unload_after_model,
            )
            metric_rows.extend(service_rows)
            forecast_points.extend(service_points)
        except Exception as exc:  # noqa: BLE001 - keep the offline part available when the service is down.
            model_status.append({"model_id": "timer_service", "status": "failed", "error": str(exc)})
            print(f"timer service failed: {exc}")
        finally:
            client.close()

    write_jsonl(args.output_dir / "sample_metrics.jsonl", metric_rows)
    write_jsonl(args.output_dir / "forecast_points.jsonl", forecast_points)

    rankings = build_rankings(metric_rows)
    real_ranking = [row for row in rankings if row["rank_scope"] == "real_original"]
    synthetic_rankings = [row for row in rankings if row["rank_scope"] == "synthetic_feature_intensity"]
    rank_comparison = build_rank_comparison(real_ranking, synthetic_rankings)
    sample_feature_rows = [sample_feature_row(sample) for sample in samples]
    model_status_rows = model_status_table(selected_models, skipped_models, model_status)
    run_config_rows = run_config_table(args, samples, synthetic_samples, real_samples, selected_models, skipped_models)

    workbook_path = args.output_dir / "m4_hourly_full_chain.xlsx"
    write_xlsx(
        workbook_path,
        {
            "run_config": run_config_rows,
            "distribution_status": distribution_rows,
            "model_status": model_status_rows,
            "real_model_ranking": real_ranking,
            "feature_intensity_rankings": synthetic_rankings,
            "rank_comparison": rank_comparison,
            "sample_metrics": excel_metric_rows(metric_rows),
            "sample_features": sample_feature_rows,
            "near_distance": near_distance_rows(samples),
            "forecast_points": forecast_points,
        },
    )

    summary = {
        "schema_version": "m4_hourly_full_chain_experiment.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "season_length": SEASON_LENGTH,
        "source_dataset": "M4 hourly",
        "profile_id": M4_PROFILE_ID,
        "real_sample_count": len(real_samples),
        "synthetic_sample_count": len(synthetic_samples),
        "generation_failure_count": len(generation_failures),
        "selected_models": [model.get("model_id") for model in selected_models],
        "skipped_models": skipped_models,
        "model_status": model_status,
        "outputs": {
            "workbook": str(workbook_path),
            "report": str(args.output_dir / "report.md"),
            "samples_jsonl": str(args.output_dir / "samples.jsonl"),
            "sample_metrics_jsonl": str(args.output_dir / "sample_metrics.jsonl"),
            "forecast_points_jsonl": str(args.output_dir / "forecast_points.jsonl"),
        },
        "top_real_ranking": real_ranking[:8],
        "rank_comparison": rank_comparison,
        "reproduction_command": reproduction_command(args),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_report(
        args=args,
        real_ranking=real_ranking,
        synthetic_rankings=synthetic_rankings,
        rank_comparison=rank_comparison,
        distribution_rows=distribution_rows,
        model_status_rows=model_status_rows,
        generation_failures=generation_failures,
        workbook_path=workbook_path,
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"wrote workbook: {display_path(workbook_path)}")
    print(f"wrote report: {display_path(args.output_dir / 'report.md')}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise SystemExit(f"M4 hourly dataset not found: {args.dataset}")
    if not args.profile.exists():
        raise SystemExit(f"M4 hourly feature profile not found: {args.profile}")
    if args.sample_count <= 0 or args.real_sample_count <= 0:
        raise SystemExit("sample counts must be positive")
    unknown = [capability for capability in args.capabilities if capability not in CAPABILITIES_BY_ID]
    if unknown:
        raise SystemExit(f"unknown synthetic capabilities: {', '.join(unknown)}")
    non_m4 = [
        capability
        for capability in args.capabilities
        if CAPABILITIES_BY_ID[capability].target_dim_mode != "fixed_1" or CAPABILITIES_BY_ID[capability].covariate_columns
    ]
    if non_m4:
        raise SystemExit(
            "this M4 hourly unit only supports univariate no-covariate capabilities: "
            + ", ".join(non_m4)
        )


def load_real_m4_samples(path: Path, *, sample_count: int) -> list[ExperimentSample]:
    metadata, records = read_tsf_series_records(path)
    series = [(record.series_id, record.values) for record in records]
    spec = WindowSpec(CONTEXT_LENGTH, HORIZON, HORIZON)
    selected = select_tsf_windows(series, spec, max_windows=sample_count * 2)
    samples: list[ExperimentSample] = []
    base_start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    for selected_index, (series_index, window_start, window) in enumerate(selected):
        if len(samples) >= sample_count:
            break
        if window.shape[0] != spec.length or not np.isfinite(window).all():
            continue
        values = window.astype(float)
        record = records[series_index]
        timestamps, timestamp_source = tsf_window_timestamps(
            record,
            metadata=metadata,
            window_start=window_start,
            length=spec.length,
            fallback_start=base_start + timedelta(hours=selected_index * spec.length),
        )
        features = feature_vector(values, season_length=SEASON_LENGTH, context_length=CONTEXT_LENGTH)
        series_id = series[series_index][0]
        samples.append(
            ExperimentSample(
                sample_id=f"m4_real-{len(samples):04d}",
                dataset_kind="real",
                source_dataset=str(metadata.get("frequency", "M4 hourly")),
                capability_id="m4_original",
                intensity=0,
                sample_index=len(samples),
                series_id=str(series_id),
                window_start=int(window_start),
                history_timestamps=timestamps[:CONTEXT_LENGTH],
                future_timestamps=timestamps[CONTEXT_LENGTH:],
                target_column_names=["target_0"],
                target_history=values[:CONTEXT_LENGTH].tolist(),
                target_future=values[CONTEXT_LENGTH:].tolist(),
                realized_features=features,
                validation={
                    "feature_gate": {"accepted": True, "enforced": False, "reason": "real_original_reference"},
                    "near_distance": {"accepted": True, "enforced": False, "reason": "real_original_reference"},
                    "timestamp_source": timestamp_source,
                    "source_start_timestamp": record.attributes.get("start_timestamp"),
                },
            )
        )
    if len(samples) < sample_count:
        raise RuntimeError(f"only found {len(samples)} finite M4 hourly windows, requested {sample_count}")
    return samples


def tsf_window_timestamps(
    record: TSFSeriesRecord,
    *,
    metadata: dict[str, str],
    window_start: int,
    length: int,
    fallback_start: datetime,
) -> tuple[list[str], str]:
    frequency = str(metadata.get("frequency", "hourly"))
    step = tsf_frequency_delta(frequency)
    series_start = parse_tsf_datetime(record.attributes.get("start_timestamp"))
    if series_start is None:
        start = fallback_start
        source = "fallback_synthetic_time"
    else:
        start = series_start + step * int(window_start)
        source = "tsf_start_timestamp"
    return [(start + step * offset).isoformat() for offset in range(length)], source


def parse_tsf_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d %H-%M-%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def tsf_frequency_delta(value: str) -> timedelta:
    normalized = (value or "").strip().lower()
    aliases = {
        "hourly": "h",
        "hour": "h",
        "h": "h",
        "daily": "d",
        "day": "d",
        "d": "d",
        "weekly": "w",
        "week": "w",
        "w": "w",
    }
    unit = aliases.get(normalized, normalized)
    if unit == "h":
        return timedelta(hours=1)
    if unit == "d":
        return timedelta(days=1)
    if unit == "w":
        return timedelta(weeks=1)
    raise ValueError(f"unsupported fixed TSF frequency for timestamp reconstruction: {value}")


def generate_synthetic_samples(
    *,
    capabilities: list[str],
    sample_count: int,
    seed: int,
) -> tuple[list[ExperimentSample], list[dict[str, Any]]]:
    samples: list[ExperimentSample] = []
    failures: list[dict[str, Any]] = []
    base_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sample_length = CONTEXT_LENGTH + HORIZON
    for capability_id in capabilities:
        for intensity in range(1, 6):
            for sample_index in range(sample_count):
                sample_seed = _seed_for(seed, capability_id, sample_index)
                try:
                    target, latent_params, _covariates, realized_features = _generate_accepted_sample_values(
                        capability_id,
                        sample_length,
                        CONTEXT_LENGTH,
                        1,
                        SEASON_LENGTH,
                        intensity,
                        sample_seed,
                        anchor_profile_id=M4_PROFILE_ID,
                    )
                    m4_near = evaluate_near_distance_gate(
                        target=target,
                        features=realized_features,
                        profile_ids=(M4_PROFILE_ID,),
                        context_length=CONTEXT_LENGTH,
                        horizon=HORIZON,
                    )
                    acceptance = latent_params.get("acceptance", {})
                    validation = dict(acceptance.get("validation", {}))
                    validation["m4_near_distance"] = m4_near
                    if not bool(acceptance.get("accepted")) or not bool(m4_near.get("accepted")):
                        raise RuntimeError("generated sample did not pass feature and M4 near-distance gates")
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "capability_id": capability_id,
                            "intensity": intensity,
                            "sample_index": sample_index,
                            "sample_seed": sample_seed,
                            "error": str(exc),
                        }
                    )
                    continue

                offset = len(samples) * sample_length
                timestamps = [(base_start + timedelta(hours=offset + step)).isoformat() for step in range(sample_length)]
                samples.append(
                    ExperimentSample(
                        sample_id=f"{capability_id}-i{intensity}-{sample_index:03d}",
                        dataset_kind="synthetic",
                        source_dataset="synthetic_v2_m4_hourly_anchor",
                        capability_id=capability_id,
                        intensity=intensity,
                        sample_index=sample_index,
                        series_id=None,
                        window_start=None,
                        history_timestamps=timestamps[:CONTEXT_LENGTH],
                        future_timestamps=timestamps[CONTEXT_LENGTH:],
                        target_column_names=["target_0"],
                        target_history=target[:CONTEXT_LENGTH].astype(float).tolist(),
                        target_future=target[CONTEXT_LENGTH:].astype(float).tolist(),
                        realized_features=realized_features,
                        validation=validation,
                    )
                )
    return samples, failures


def baseline_predictions(samples: list[ExperimentSample]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for sample in samples:
        history = np.asarray(sample.target_history, dtype=float)
        naive = np.repeat(history[-1:], HORIZON, axis=0).tolist()
        seasonal = history[-SEASON_LENGTH:][:HORIZON].tolist()
        for model_id, forecast in (("naive", naive), ("seasonal_naive", seasonal)):
            row, point_rows = metric_and_points(model_id, "baseline", sample, forecast, elapsed_seconds=0.0)
            rows.append(row)
            points.extend(point_rows)
    return rows, points


def run_service_models(
    client: TimerServiceClient,
    models: list[dict[str, Any]],
    samples: list[ExperimentSample],
    *,
    batch_size: int,
    forecast_timeout_seconds: int,
    load_timeout_seconds: int,
    unload_timeout_seconds: int,
    unload_after_model: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["model_id"])
        print(f"running model {model_id}")
        status: dict[str, Any] = {
            "model_id": model_id,
            "status": "pending",
            "sample_count": len(samples),
            "succeeded_count": 0,
            "failed_count": 0,
        }
        started = time.monotonic()
        try:
            status["load_seconds"] = round(client.ensure_model_loaded(model_id, timeout_seconds=load_timeout_seconds), 3)
            for batch in chunks(samples, batch_size):
                batch_started = time.monotonic()
                try:
                    forecasts = client.forecast_batch(
                        model_id,
                        batch,
                        timeout_seconds=forecast_timeout_seconds,
                        model_params=model_params_for(model_id),
                    )
                    elapsed = time.monotonic() - batch_started
                    for sample, forecast in zip(batch, forecasts, strict=True):
                        row, point_rows = metric_and_points(model_id, "timer_service", sample, forecast, elapsed_seconds=elapsed)
                        rows.append(row)
                        points.extend(point_rows)
                        status["succeeded_count"] += 1
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.monotonic() - batch_started
                    status["failed_count"] += len(batch)
                    status.setdefault("first_error", str(exc))
                    for sample in batch:
                        rows.append(failed_metric_row(model_id, "timer_service", sample, str(exc), elapsed))
            if status["succeeded_count"] == len(samples):
                status["status"] = "succeeded"
            elif status["succeeded_count"] == 0:
                status["status"] = "failed"
            else:
                status["status"] = "partial_succeeded"
        except Exception as exc:  # noqa: BLE001
            status["status"] = "failed"
            status["error"] = str(exc)
            status["failed_count"] = len(samples)
            for sample in samples:
                rows.append(failed_metric_row(model_id, "timer_service", sample, str(exc), 0.0))
        finally:
            status["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if unload_after_model:
                try:
                    if client.find_model(model_id).get("loaded"):
                        client.unload_model(model_id, timeout_seconds=unload_timeout_seconds)
                        status["unloaded"] = True
                    else:
                        status["unloaded"] = False
                        status["unload_skipped_reason"] = "not_loaded"
                except Exception as exc:  # noqa: BLE001
                    status["unload_error"] = str(exc)
            statuses.append(status)
    return rows, points, statuses


def forecast_target(sample: ExperimentSample) -> dict[str, Any]:
    return {
        "columns": [TIME_COLUMN, *sample.target_column_names],
        "data": [
            [timestamp, *row]
            for timestamp, row in zip(sample.history_timestamps, sample.target_history, strict=True)
        ],
    }


def parse_forecast_result(result: dict[str, Any]) -> list[list[float]]:
    columns = result["columns"]
    value_indexes = [index for index, column in enumerate(columns) if column != TIME_COLUMN]
    rows = result["data"]
    parsed = [[float(row[index]) for index in value_indexes] for row in rows[:HORIZON]]
    if len(parsed) != HORIZON:
        raise RuntimeError(f"forecast has {len(parsed)} rows, expected {HORIZON}")
    return parsed


def metric_and_points(
    model_id: str,
    model_group: str,
    sample: ExperimentSample,
    forecast: list[list[float]],
    *,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = compute_sample_metrics(
        sample.target_future,
        forecast,
        sample.target_history,
        seasonal_period=SEASON_LENGTH,
    )
    row = {
        "model_id": model_id,
        "model_group": model_group,
        "dataset_kind": sample.dataset_kind,
        "sample_id": sample.sample_id,
        "source_dataset": sample.source_dataset,
        "capability_id": sample.capability_id,
        "feature_dimension": sample.feature_dimension,
        "intensity": sample.intensity,
        "sample_index": sample.sample_index,
        "series_id": sample.series_id,
        "window_start": sample.window_start,
        "target_dim": len(sample.target_column_names),
        "mase_period": SEASON_LENGTH,
        "status": "succeeded",
        "metrics": dict(metrics),
        "mase_unavailable_reason": mase_unavailable_reason(metrics),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
    }
    point_rows: list[dict[str, Any]] = []
    for step, (timestamp, actual_row, forecast_row) in enumerate(
        zip(sample.future_timestamps, sample.target_future, forecast, strict=True)
    ):
        for target_index, target_name in enumerate(sample.target_column_names):
            actual = float(actual_row[target_index])
            predicted = float(forecast_row[target_index])
            point_rows.append(
                {
                    "model_id": model_id,
                    "model_group": model_group,
                    "dataset_kind": sample.dataset_kind,
                    "sample_id": sample.sample_id,
                    "capability_id": sample.capability_id,
                    "feature_dimension": sample.feature_dimension,
                    "intensity": sample.intensity,
                    "horizon_step": step + 1,
                    "timestamp": timestamp,
                    "target": target_name,
                    "actual": actual,
                    "prediction": predicted,
                    "error": predicted - actual,
                    "absolute_error": abs(predicted - actual),
                }
            )
    return row, point_rows


def failed_metric_row(
    model_id: str,
    model_group: str,
    sample: ExperimentSample,
    error: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model_group": model_group,
        "dataset_kind": sample.dataset_kind,
        "sample_id": sample.sample_id,
        "source_dataset": sample.source_dataset,
        "capability_id": sample.capability_id,
        "feature_dimension": sample.feature_dimension,
        "intensity": sample.intensity,
        "sample_index": sample.sample_index,
        "series_id": sample.series_id,
        "window_start": sample.window_start,
        "target_dim": len(sample.target_column_names),
        "status": "failed",
        "error": error,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
    }


def select_models(service_models: list[dict[str, Any]], requested: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {str(model.get("model_id")): model for model in service_models if model.get("model_id")}
    if requested == ["all-active"]:
        active_ids = [str(model.get("model_id")) for model in service_models if str(model.get("state")).lower() == "active"]
        requested_ids = sorted(active_ids, key=model_sort_key)
    else:
        requested_ids = requested
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for model_id in requested_ids:
        model = by_id.get(model_id)
        if model is None:
            skipped.append({"model_id": model_id, "reason": "not_registered"})
            continue
        if str(model.get("state") or "").lower() != "active":
            skipped.append({"model_id": model_id, "reason": "inactive"})
            continue
        limits = model.get("forecast_limits") or {}
        if int(limits.get("min_input_length") or 0) > CONTEXT_LENGTH:
            skipped.append({"model_id": model_id, "reason": "min_input_length_unsupported", "forecast_limits": limits})
            continue
        if int(limits.get("max_output_length") or 0) < HORIZON:
            skipped.append({"model_id": model_id, "reason": "max_output_length_unsupported", "forecast_limits": limits})
            continue
        max_target_count = limits.get("max_target_count")
        if max_target_count is not None and int(max_target_count) < 1:
            skipped.append({"model_id": model_id, "reason": "target_dim_unsupported", "forecast_limits": limits})
            continue
        selected.append(model)
    return selected, skipped


def model_sort_key(model_id: str) -> tuple[int, str]:
    try:
        return (DEFAULT_MODEL_ORDER.index(model_id), model_id)
    except ValueError:
        return (len(DEFAULT_MODEL_ORDER), model_id)


def model_params_for(model_id: str) -> dict[str, Any] | None:
    if model_id in {"AutoARIMA", "Holt-Winters"}:
        return {"sp": SEASON_LENGTH, "suppress_warnings": True}
    return None


def build_rankings(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[tuple[str, str, int, str], int] = defaultdict(int)
    for row in metric_rows:
        scope = "real_original" if row.get("dataset_kind") == "real" else "synthetic_feature_intensity"
        key = (scope, str(row.get("feature_dimension")), int(row.get("intensity") or 0), str(row.get("model_id")))
        if row.get("status") == "succeeded":
            groups[key].append(row)
        else:
            failures[key] += 1

    aggregates: list[dict[str, Any]] = []
    for (scope, feature_dimension, intensity, model_id), rows in groups.items():
        metrics = aggregate_metrics(rows)
        rank_value = metrics.get(RANK_METRIC)
        if rank_value is None:
            rank_value = metrics.get("mae")
        aggregates.append(
            {
                "rank_scope": scope,
                "feature_dimension": feature_dimension,
                "capability_id": feature_dimension if scope != "real_original" else "m4_original",
                "intensity": intensity,
                "model_id": model_id,
                "model_group": rows[0].get("model_group"),
                "sample_count": len(rows),
                "failed_count": failures.get((scope, feature_dimension, intensity, model_id), 0),
                "rank_metric": RANK_METRIC if metrics.get(RANK_METRIC) is not None else "mae",
                "rank_metric_value": rank_value,
                **{f"mean_{key}": value for key, value in metrics.items()},
            }
        )

    ranked: list[dict[str, Any]] = []
    by_scope: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        by_scope[(row["rank_scope"], row["feature_dimension"], int(row["intensity"]))].append(row)
    for key, rows in sorted(by_scope.items()):
        rows = sorted(rows, key=lambda item: (float_or_inf(item.get("rank_metric_value")), item["model_id"]))
        for rank, row in enumerate(rows, start=1):
            ranked.append({**row, "rank": rank})
    return ranked


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.get("metrics", {}).items() if is_finite(value)})
    return {
        key: float(np.mean([float(row["metrics"][key]) for row in rows if is_finite(row.get("metrics", {}).get(key))]))
        for key in keys
    }


def build_rank_comparison(
    real_ranking: list[dict[str, Any]],
    synthetic_rankings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    real_by_model = {row["model_id"]: row for row in real_ranking}
    rows: list[dict[str, Any]] = []
    for synthetic in synthetic_rankings:
        real = real_by_model.get(synthetic["model_id"])
        if real is None:
            continue
        rows.append(
            {
                "model_id": synthetic["model_id"],
                "feature_dimension": synthetic["feature_dimension"],
                "intensity": synthetic["intensity"],
                "real_rank": real["rank"],
                "synthetic_rank": synthetic["rank"],
                "rank_delta_synthetic_minus_real": int(synthetic["rank"]) - int(real["rank"]),
                "real_rank_metric": real.get("rank_metric"),
                "real_rank_metric_value": real.get("rank_metric_value"),
                "synthetic_rank_metric": synthetic.get("rank_metric"),
                "synthetic_rank_metric_value": synthetic.get("rank_metric_value"),
                "real_mean_mae": real.get("mean_mae"),
                "synthetic_mean_mae": synthetic.get("mean_mae"),
                "real_mean_mase": real.get("mean_mase"),
                "synthetic_mean_mase": synthetic.get("mean_mase"),
            }
        )
    return rows


def distribution_status_rows(
    *,
    profile_path: Path,
    artifact_path: Path,
    calibration_summary_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile = load_json_if_exists(profile_path)
    if profile:
        bucket = profile.get("bucket") or {}
        rows.append(
            {
                "source": "feature_profile",
                "profile_id": profile.get("profile_id", M4_PROFILE_ID),
                "path": str(profile_path),
                "exists": True,
                "context_length": bucket.get("context_length"),
                "horizon": bucket.get("horizon"),
                "target_dim": bucket.get("target_dim"),
                "season_length": bucket.get("season_length"),
                "window_count": profile.get("window_count"),
                "candidate_window_count": profile.get("candidate_window_count"),
                "series_count": profile.get("series_count"),
                "used_series_count": profile.get("used_series_count"),
            }
        )
        for feature_name, quantiles in sorted((profile.get("features") or {}).items()):
            rows.append(
                {
                    "source": "feature_profile_quantile",
                    "profile_id": profile.get("profile_id", M4_PROFILE_ID),
                    "feature": feature_name,
                    **flatten_prefixed("q", quantiles),
                }
            )
    else:
        rows.append({"source": "feature_profile", "profile_id": M4_PROFILE_ID, "path": str(profile_path), "exists": False})

    artifact = load_json_if_exists(artifact_path)
    bucket = (artifact.get("buckets") or {}).get(M4_PROFILE_ID) if artifact else None
    if bucket:
        rows.append(
            {
                "source": "near_distance_online_artifact",
                "profile_id": M4_PROFILE_ID,
                "path": str(artifact_path),
                "exists": True,
                "context_length": bucket.get("context_length"),
                "horizon": bucket.get("horizon"),
                "target_dim": bucket.get("target_dim"),
                "season_length": bucket.get("season_length"),
                "reference_count": bucket.get("reference_count"),
                **flatten_prefixed("threshold", bucket.get("thresholds") or {}),
            }
        )
        rows.append(
            {
                "source": "near_distance_online_features",
                "profile_id": M4_PROFILE_ID,
                "feature_names": ", ".join(str(name) for name in bucket.get("feature_names", [])),
            }
        )
    else:
        rows.append({"source": "near_distance_online_artifact", "profile_id": M4_PROFILE_ID, "path": str(artifact_path), "exists": False})

    calibration = load_json_if_exists(calibration_summary_path)
    calibration_bucket = None
    if calibration:
        calibration_bucket = next(
            (item for item in calibration.get("buckets", []) if item.get("profile_id") == M4_PROFILE_ID),
            None,
        )
    if calibration_bucket:
        for metric_name, values in sorted((calibration_bucket.get("threshold_stability") or {}).items()):
            rows.append(
                {
                    "source": "near_distance_threshold_stability",
                    "profile_id": M4_PROFILE_ID,
                    "metric": metric_name,
                    **flatten_prefixed("stability", values),
                }
            )
        for control_name, control_values in sorted((calibration_bucket.get("control_summary") or {}).items()):
            row: dict[str, Any] = {
                "source": "near_distance_control_summary",
                "profile_id": M4_PROFILE_ID,
                "control": control_name,
            }
            for metric_name, values in sorted(control_values.items()):
                if isinstance(values, dict):
                    for key, value in values.items():
                        row[f"{metric_name}_{key}"] = value
            rows.append(row)
    elif calibration_summary_path.exists():
        rows.append({"source": "near_distance_calibration_summary", "profile_id": M4_PROFILE_ID, "path": str(calibration_summary_path), "exists": True, "status": "bucket_not_found"})
    else:
        rows.append({"source": "near_distance_calibration_summary", "profile_id": M4_PROFILE_ID, "path": str(calibration_summary_path), "exists": False})
    return rows


def sample_row(sample: ExperimentSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset_kind": sample.dataset_kind,
        "source_dataset": sample.source_dataset,
        "capability_id": sample.capability_id,
        "feature_dimension": sample.feature_dimension,
        "intensity": sample.intensity,
        "sample_index": sample.sample_index,
        "series_id": sample.series_id,
        "window_start": sample.window_start,
        "history_start": sample.history_timestamps[0] if sample.history_timestamps else None,
        "history_end": sample.history_timestamps[-1] if sample.history_timestamps else None,
        "future_start": sample.future_timestamps[0] if sample.future_timestamps else None,
        "future_end": sample.future_timestamps[-1] if sample.future_timestamps else None,
        "target_dim": len(sample.target_column_names),
        "target_columns": sample.target_column_names,
        "realized_features": sample.realized_features,
        "validation": sample.validation,
    }


def sample_feature_row(sample: ExperimentSample) -> dict[str, Any]:
    row = {
        "sample_id": sample.sample_id,
        "dataset_kind": sample.dataset_kind,
        "source_dataset": sample.source_dataset,
        "capability_id": sample.capability_id,
        "feature_dimension": sample.feature_dimension,
        "intensity": sample.intensity,
        "sample_index": sample.sample_index,
        "series_id": sample.series_id,
        "window_start": sample.window_start,
        "history_start": sample.history_timestamps[0] if sample.history_timestamps else None,
        "future_start": sample.future_timestamps[0] if sample.future_timestamps else None,
        "target_feature_group": ", ".join(CAPABILITY_FEATURE_GROUPS.get(sample.capability_id, ())),
        "feature_gate_accepted": nested_get(sample.validation, ("feature_gate", "accepted")),
        "near_distance_accepted": nested_get(sample.validation, ("near_distance", "accepted")),
        "m4_near_distance_accepted": nested_get(sample.validation, ("m4_near_distance", "accepted")),
    }
    for feature_name in DEFAULT_FEATURES:
        value = sample.realized_features.get(feature_name)
        if is_finite(value):
            row[feature_name] = float(value)
    return row


def near_distance_rows(samples: list[ExperimentSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for gate_name in ("near_distance", "m4_near_distance"):
            gate = sample.validation.get(gate_name) if isinstance(sample.validation, dict) else None
            if not isinstance(gate, dict):
                continue
            base = {
                "sample_id": sample.sample_id,
                "dataset_kind": sample.dataset_kind,
                "capability_id": sample.capability_id,
                "feature_dimension": sample.feature_dimension,
                "intensity": sample.intensity,
                "gate": gate_name,
                "accepted": gate.get("accepted"),
                "enforced": gate.get("enforced"),
                "status": gate.get("status"),
                "strict_risk": gate.get("strict_risk"),
                "combined_risk": gate.get("combined_risk"),
            }
            bucket_results = gate.get("bucket_results") or []
            if not bucket_results:
                rows.append(base)
                continue
            for bucket in bucket_results:
                thresholds = bucket.get("thresholds") or {}
                rows.append(
                    {
                        **base,
                        "profile_id": bucket.get("profile_id"),
                        "bucket_strict_risk": bucket.get("strict_risk"),
                        "bucket_combined_risk": bucket.get("combined_risk"),
                        "raw_mae_d1": bucket.get("raw_mae_d1"),
                        "raw_l2_d1": bucket.get("raw_l2_d1"),
                        "feature_l2_d1": bucket.get("feature_l2_d1"),
                        "raw_mae_nndr": bucket.get("raw_mae_nndr"),
                        **flatten_prefixed("threshold", thresholds),
                    }
                )
    return rows


def excel_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        out = {key: value for key, value in row.items() if key != "metrics"}
        out.update({f"metric_{key}": value for key, value in (row.get("metrics") or {}).items()})
        rows.append(out)
    return rows


def model_status_table(
    selected_models: list[dict[str, Any]],
    skipped_models: list[dict[str, Any]],
    model_status: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_ids = {str(model.get("model_id")) for model in selected_models}
    status_by_id = {str(row.get("model_id")): row for row in model_status}
    for model in selected_models:
        model_id = str(model.get("model_id"))
        limits = model.get("forecast_limits") or {}
        rows.append(
            {
                "model_id": model_id,
                "selection": "selected",
                "service_state": model.get("state"),
                "loaded_before_run": model.get("loaded"),
                **flatten_prefixed("limit", limits),
                **status_by_id.get(model_id, {}),
            }
        )
    for skipped in skipped_models:
        rows.append({**skipped, "selection": "skipped"})
    for status in model_status:
        if str(status.get("model_id")) not in selected_ids:
            rows.append({**status, "selection": "service_status"})
    return rows


def run_config_table(
    args: argparse.Namespace,
    samples: list[ExperimentSample],
    synthetic_samples: list[ExperimentSample],
    real_samples: list[ExperimentSample],
    selected_models: list[dict[str, Any]],
    skipped_models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "key": "schema_version",
            "value": "m4_hourly_full_chain_experiment.v1",
        },
        {"key": "created_at", "value": datetime.now(timezone.utc).isoformat()},
        {"key": "dataset", "value": str(args.dataset)},
        {"key": "profile_id", "value": M4_PROFILE_ID},
        {"key": "context_length", "value": CONTEXT_LENGTH},
        {"key": "horizon", "value": HORIZON},
        {"key": "season_length", "value": SEASON_LENGTH},
        {"key": "real_sample_count", "value": len(real_samples)},
        {"key": "synthetic_sample_count", "value": len(synthetic_samples)},
        {"key": "total_sample_count", "value": len(samples)},
        {"key": "synthetic_sample_count_per_feature_intensity", "value": args.sample_count},
        {"key": "capabilities", "value": ", ".join(args.capabilities)},
        {"key": "requested_models", "value": ", ".join(args.models)},
        {"key": "selected_models", "value": ", ".join(str(model.get("model_id")) for model in selected_models)},
        {"key": "skipped_models", "value": json.dumps(skipped_models, ensure_ascii=False)},
        {"key": "rank_metric", "value": RANK_METRIC},
        {"key": "output_dir", "value": str(args.output_dir)},
        {"key": "base_url", "value": args.base_url},
        {"key": "reproduction_command", "value": reproduction_command(args)},
    ]


def render_report(
    *,
    args: argparse.Namespace,
    real_ranking: list[dict[str, Any]],
    synthetic_rankings: list[dict[str, Any]],
    rank_comparison: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    model_status_rows: list[dict[str, Any]],
    generation_failures: list[dict[str, Any]],
    workbook_path: Path,
) -> str:
    distribution_summary = distribution_brief(distribution_rows)
    real_lines = ranking_markdown_lines(real_ranking[:12])
    status_lines = [
        f"- `{row.get('model_id')}`: `{row.get('status', row.get('selection'))}`, "
        f"succeeded={row.get('succeeded_count', '-')}, failed={row.get('failed_count', '-')}"
        + (f", error={row.get('error') or row.get('first_error')}" if (row.get("error") or row.get("first_error")) else "")
        for row in model_status_rows
        if row.get("selection") != "skipped"
    ]
    skipped_lines = [
        f"- `{row.get('model_id')}` skipped: {row.get('reason')}"
        for row in model_status_rows
        if row.get("selection") == "skipped"
    ]
    top_by_feature = best_synthetic_lines(synthetic_rankings)
    delta_lines = rank_delta_lines(rank_comparison)
    failure_text = (
        f"- 合成样本生成失败 {len(generation_failures)} 个，详见 `generation_failures.jsonl`。"
        if generation_failures
        else "- 合成样本全部通过特征阈值和近距离污染回验。"
    )
    return "\n".join(
        [
            "# M4 hourly 合成数据全链路实验",
            "",
            f"生成时间：{datetime.now(timezone.utc).isoformat()}",
            "",
            "## 实验单元",
            "",
            f"- 真实基底：M4 hourly，profile `{M4_PROFILE_ID}`。",
            f"- 窗口：context `{CONTEXT_LENGTH}`，horizon `{HORIZON}`，主季节周期 `{SEASON_LENGTH}`。",
            "- 原始 M4 窗口时间戳：优先使用 TSF `start_timestamp` 加窗口偏移重建。",
            f"- 原始 M4 评测窗口数：`{args.real_sample_count}`。",
            f"- 合成能力维度：{', '.join(f'`{capability}`' for capability in args.capabilities)}。",
            f"- 每个能力维度每个强度样本数：`{args.sample_count}`。",
            f"- 模型执行：按模型顺序加载、评测、{'卸载' if args.unload_after_model else '保留加载状态'}；卸载超时 `{args.model_unload_timeout_seconds}` 秒。",
            f"- 排名指标：`{RANK_METRIC}`，数值越低表示预测误差越低。",
            f"- Excel 结果：`{display_path(workbook_path)}`。",
            "",
            "## 分布提取与回验",
            "",
            *distribution_summary,
            failure_text,
            "",
            "## 模型运行状态",
            "",
            *(status_lines or ["- 没有成功连接模型服务，当前结果只包含 baseline。"]),
            *skipped_lines,
            "",
            "## 原始 M4 排名",
            "",
            *(real_lines or ["- 没有可用的原始 M4 模型排名。"]),
            "",
            "## 特征维度 x 强度排名观察",
            "",
            *top_by_feature,
            "",
            "## 与原始 M4 排名的差异",
            "",
            *delta_lines,
            "",
            "## 产物",
            "",
            f"- `m4_hourly_full_chain.xlsx`：配置、分布、回验、模型状态、排名、逐样本指标和逐点预测。",
            f"- `samples.jsonl`：真实窗口和合成窗口的样本元信息。",
            f"- `sample_metrics.jsonl`：每个模型在每个样本上的 MAE/MSE/MASE。",
            f"- `forecast_points.jsonl`：每个 horizon step 的 actual/prediction/error。",
            "",
            "## 复现命令",
            "",
            "```bash",
            reproduction_command(args),
            "```",
            "",
        ]
    )


def distribution_brief(rows: list[dict[str, Any]]) -> list[str]:
    profile = next((row for row in rows if row.get("source") == "feature_profile"), None)
    artifact = next((row for row in rows if row.get("source") == "near_distance_online_artifact"), None)
    stability = [row for row in rows if row.get("source") == "near_distance_threshold_stability"]
    control = [row for row in rows if row.get("source") == "near_distance_control_summary"]
    out: list[str] = []
    if profile:
        out.append(
            f"- 特征 profile 使用 `{display_path(Path(str(profile.get('path'))))}`，"
            f"窗口数 `{profile.get('window_count')}`，序列数 `{profile.get('series_count')}`。"
        )
    if artifact:
        out.append(
            f"- 近距离在线校验 artifact 使用 `{display_path(Path(str(artifact.get('path'))))}`，"
            f"reference_count `{artifact.get('reference_count')}`，raw_mae_p01 `{fmt(artifact.get('threshold_raw_mae_p01'))}`，"
            f"raw_mae_p05 `{fmt(artifact.get('threshold_raw_mae_p05'))}`。"
        )
    if stability:
        max_cv = max(float_or_zero(row.get("stability_cv")) for row in stability)
        out.append(f"- 阈值校准包含 train/holdout split 稳定性记录，本 bucket 最大 CV `{fmt(max_cv)}`。")
    if control:
        normal = next((row for row in control if row.get("control") == "normal_synthetic"), None)
        exact = next((row for row in control if row.get("control") == "exact_copy"), None)
        if normal and exact:
            out.append(
                "- 控制样本记录显示 exact copy 的 strict risk 均值 "
                f"`{fmt(exact.get('strict_risk_rate_mean'))}`，normal synthetic 的 strict risk 均值 "
                f"`{fmt(normal.get('strict_risk_rate_mean'))}`。"
            )
    return out


def ranking_markdown_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Rank | Model | MASE | MAE | Samples |", "| ---: | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row.get('rank')} | `{row.get('model_id')}` | {fmt(row.get('mean_mase'))} | "
            f"{fmt(row.get('mean_mae'))} | {row.get('sample_count')} |"
        )
    return lines if len(lines) > 2 else []


def best_synthetic_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- 没有可用的合成样本模型排名。"]
    best = [row for row in rows if int(row.get("rank") or 0) == 1]
    lines: list[str] = []
    for row in sorted(best, key=lambda item: (item["feature_dimension"], int(item["intensity"]))):
        lines.append(
            f"- `{row['feature_dimension']}` intensity `{row['intensity']}`："
            f"`{row['model_id']}` 排名第 1，{row.get('rank_metric')} `{fmt(row.get('rank_metric_value'))}`。"
        )
    return lines


def rank_delta_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- 没有可比较的模型排名差异。"]
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[str(row["feature_dimension"])].append(row)
    lines: list[str] = []
    for feature_dimension, feature_rows in sorted(by_feature.items()):
        deltas = [abs(int(row["rank_delta_synthetic_minus_real"])) for row in feature_rows]
        max_delta = max(deltas) if deltas else 0
        changed = [
            row
            for row in feature_rows
            if int(row.get("intensity") or 0) in {1, 3, 5}
            and abs(int(row["rank_delta_synthetic_minus_real"])) == max_delta
        ]
        example = changed[0] if changed else max(feature_rows, key=lambda row: abs(int(row["rank_delta_synthetic_minus_real"])))
        lines.append(
            f"- `{feature_dimension}`：最大排名差绝对值 `{max_delta}`；示例 "
            f"`{example['model_id']}` 在 intensity `{example['intensity']}` 上从原始 rank "
            f"`{example['real_rank']}` 变为合成 rank `{example['synthetic_rank']}`。"
        )
    return lines


def reproduction_command(args: argparse.Namespace) -> str:
    model_args = " ".join(shlex.quote(model) for model in args.models)
    capability_args = " ".join(shlex.quote(capability) for capability in args.capabilities)
    parts = [
        "cd /root/xmy/TSBenchmark",
        "&&",
        "PYTHONPATH=backend:scripts",
        "backend/.venv/bin/python" if (REPO_ROOT / "backend/.venv/bin/python").exists() else "python",
        "scripts/run_m4_hourly_full_chain_experiment.py",
        "--models",
        model_args,
        "--capabilities",
        capability_args,
        "--sample-count",
        str(args.sample_count),
        "--real-sample-count",
        str(args.real_sample_count),
        "--batch-size",
        str(args.batch_size),
        "--model-load-timeout-seconds",
        str(args.model_load_timeout_seconds),
        "--forecast-timeout-seconds",
        str(args.forecast_timeout_seconds),
        "--model-unload-timeout-seconds",
        str(args.model_unload_timeout_seconds),
    ]
    if not args.unload_after_model:
        parts.append("--keep-model-loaded")
    return " ".join(part for part in parts if part)


def write_xlsx(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_sheets = sanitize_sheets(sheets)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(safe_sheets)))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml([name for name, _rows in safe_sheets]))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(safe_sheets)))
        archive.writestr("xl/styles.xml", styles_xml())
        for index, (_name, rows) in enumerate(safe_sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(rows))


def sanitize_sheets(sheets: dict[str, list[dict[str, Any]]]) -> list[tuple[str, list[dict[str, Any]]]]:
    used: set[str] = set()
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for raw_name, rows in sheets.items():
        name = re.sub(r"[\[\]\:\*\?\/\\]", "_", raw_name)[:31] or "sheet"
        base = name
        suffix = 1
        while name in used:
            tail = f"_{suffix}"
            name = (base[: 31 - len(tail)] + tail)[:31]
            suffix += 1
        used.add(name)
        out.append((name, rows))
    return out


def content_types_xml(sheet_count: int) -> str:
    worksheet_overrides = "\n".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f"{worksheet_overrides}"
        "</Types>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, sheet_count + 1)
    )
    rels += (
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        "</styleSheet>"
    )


def sheet_xml(rows: list[dict[str, Any]]) -> str:
    headers = headers_for_rows(rows)
    xml_rows = []
    xml_rows.append(row_xml(1, headers, headers=True))
    for row_index, row in enumerate(rows, start=2):
        values = [row.get(header) for header in headers]
        xml_rows.append(row_xml(row_index, values, headers=False))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )


def headers_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_text = str(key)
            if key_text not in seen:
                headers.append(key_text)
                seen.add(key_text)
    return headers or ["empty"]


def row_xml(row_index: int, values: list[Any], *, headers: bool) -> str:
    cells = []
    for col_index, value in enumerate(values, start=1):
        ref = f"{column_letter(col_index)}{row_index}"
        cells.append(cell_xml(ref, value, force_string=headers))
    return f'<row r="{row_index}">{"".join(cells)}</row>'


def cell_xml(ref: str, value: Any, *, force_string: bool) -> str:
    if value is None:
        return f'<c r="{ref}" t="inlineStr"><is><t></t></is></c>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if not force_string and isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f'<c r="{ref}"><v>{float(value):.15g}</v></c>'
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def flatten_prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def nested_get(values: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = values
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def chunks(values: list[ExperimentSample], size: int) -> list[list[ExperimentSample]]:
    return [values[index : index + size] for index in range(0, len(values), max(1, size))]


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def float_or_inf(value: Any) -> float:
    return float(value) if is_finite(value) else float("inf")


def float_or_zero(value: Any) -> float:
    return float(value) if is_finite(value) else 0.0


def fmt(value: Any) -> str:
    if not is_finite(value):
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
