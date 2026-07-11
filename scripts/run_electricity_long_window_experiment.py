#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (BACKEND_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_m4_hourly_full_chain_experiment as shared  # noqa: E402
import run_synthetic_v2_near_distance_calibration as calibration  # noqa: E402
from app.services.synthetic_generation_service import (  # noqa: E402
    CAPABILITIES_BY_ID,
    PILOT_ACCEPTANCE_CAPS,
    PILOT_ACCEPTANCE_MINS,
    _accept_synthetic_features,
    _attempt_seed,
    _generate_sample_values,
    _realized_features,
    _seed_for,
    _standardize_by_context,
)
from app.services.synthetic_near_distance_gate import evaluate_near_distance_gate  # noqa: E402
from synthetic_feature_profile import (  # noqa: E402
    DEFAULT_FEATURES,
    WindowSpec,
    feature_vector,
    read_tsf_series_records,
    select_tsf_windows,
)


DEFAULT_DATASET = REPO_ROOT / "runtime/research/electricity_hourly_dataset.zip"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/research/electricity-long-window-2048ctx-24h"
DEFAULT_MODELS = ("Timer-3.5", "Timer-3.0", "Chronos-2", "moirai2", "toto2.0", "timesfm2.5")
DEFAULT_CAPABILITIES = shared.DEFAULT_CAPABILITIES
DATASET_ID = "electricity_hourly_long_2048ctx_24h"
PROFILE_ID = "electricity_hourly_daily_2048ctx_24h"
TIME_COLUMN = "time"


@dataclass(frozen=True)
class RealSample:
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
        return self.capability_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Electricity hourly long-window full-chain experiment.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--season-length", type=int, default=24)
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--synthetic-sample-count", type=int, default=3, help="Samples per capability and intensity.")
    parser.add_argument("--capabilities", nargs="+", default=list(DEFAULT_CAPABILITIES))
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--calibration-max-windows", type=int, default=240)
    parser.add_argument("--calibration-splits", type=int, default=5)
    parser.add_argument("--calibration-synthetic-count", type=int, default=48)
    parser.add_argument("--calibration-reference-count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-load-timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-unload-timeout-seconds", type=int, default=120)
    parser.set_defaults(unload_after_model=True)
    parser.add_argument("--unload-after-model", action="store_true", dest="unload_after_model")
    parser.add_argument("--keep-model-loaded", action="store_false", dest="unload_after_model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    apply_window_constants(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"writing outputs to {shared.display_path(args.output_dir)}")
    print("extracting Electricity long-window distribution and calibrating near-distance thresholds")
    distribution, artifact = calibrate_long_window_distribution(args)
    write_distribution_outputs(args, distribution, artifact)

    print("loading Electricity hourly long windows")
    real_samples = load_electricity_samples(args)
    print(f"loaded {len(real_samples)} Electricity windows")

    print("generating synthetic samples with feature and near-distance acceptance")
    synthetic_samples, generation_failures = generate_synthetic_samples(args, artifact)
    print(f"generated {len(synthetic_samples)} accepted synthetic windows")
    if generation_failures:
        print(f"generation failures: {len(generation_failures)}")
        shared.write_jsonl(args.output_dir / "generation_failures.jsonl", generation_failures)
    samples = [*real_samples, *synthetic_samples]

    shared.write_jsonl(args.output_dir / "samples.jsonl", [shared.sample_row(sample) for sample in samples])

    metric_rows: list[dict[str, Any]] = []
    forecast_points: list[dict[str, Any]] = []
    print("running naive and seasonal naive baselines")
    baseline_rows, baseline_points = shared.baseline_predictions(samples)
    metric_rows.extend(baseline_rows)
    forecast_points.extend(baseline_points)

    selected_models: list[dict[str, Any]] = []
    skipped_models: list[dict[str, Any]] = []
    model_status: list[dict[str, Any]] = []
    client = shared.TimerServiceClient(args.base_url, args.api_prefix, timeout_seconds=30)
    try:
        service_models = client.list_models()
        selected_models, skipped_models = select_models(service_models, args.models, args)
        print("selected service models: " + ", ".join(str(model["model_id"]) for model in selected_models))
        if skipped_models:
            print("skipped service models: " + ", ".join(f"{item['model_id']}({item['reason']})" for item in skipped_models))
        service_rows, service_points, model_status = shared.run_service_models(
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
    finally:
        client.close()

    shared.write_jsonl(args.output_dir / "sample_metrics.jsonl", metric_rows)
    shared.write_jsonl(args.output_dir / "forecast_points.jsonl", forecast_points)

    rankings = shared.build_rankings(metric_rows)
    real_rankings = [row for row in rankings if row["rank_scope"] == "real_original"]
    for row in real_rankings:
        row["capability_id"] = "electricity_original"
    synthetic_rankings = [row for row in rankings if row["rank_scope"] == "synthetic_feature_intensity"]
    rank_comparison = shared.build_rank_comparison(real_rankings, synthetic_rankings)
    model_status_rows = shared.model_status_table(selected_models, skipped_models, model_status)
    sample_feature_rows = [shared.sample_feature_row(sample) for sample in samples]
    workbook_path = args.output_dir / "electricity_long_window_2048ctx_24h.xlsx"
    shared.write_xlsx(
        workbook_path,
        {
            "run_config": run_config_rows(args, samples, real_samples, synthetic_samples, selected_models, skipped_models),
            "distribution_status": distribution_rows(distribution, artifact),
            "model_status": model_status_rows,
            "real_model_ranking": real_rankings,
            "feature_intensity_rankings": synthetic_rankings,
            "rank_comparison": rank_comparison,
            "sample_metrics": shared.excel_metric_rows(metric_rows),
            "sample_features": sample_feature_rows,
            "near_distance": shared.near_distance_rows(samples),
            "forecast_points": forecast_points,
        },
    )

    summary = {
        "schema_version": "electricity_long_window_full_chain_experiment.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "source_dataset": "Electricity hourly",
        "context_length": args.context_length,
        "horizon": args.horizon,
        "season_length": args.season_length,
        "real_sample_count": len(real_samples),
        "synthetic_sample_count": len(synthetic_samples),
        "generation_failure_count": len(generation_failures),
        "selected_models": [model.get("model_id") for model in selected_models],
        "skipped_models": skipped_models,
        "model_status": model_status,
        "top_real_ranking": real_rankings,
        "rank_comparison": rank_comparison,
        "outputs": {
            "workbook": str(workbook_path),
            "report": str(args.output_dir / "report.md"),
            "samples_jsonl": str(args.output_dir / "samples.jsonl"),
            "sample_metrics_jsonl": str(args.output_dir / "sample_metrics.jsonl"),
            "forecast_points_jsonl": str(args.output_dir / "forecast_points.jsonl"),
        },
        "reproduction_command": reproduction_command(args),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(
            args,
            real_rankings,
            synthetic_rankings,
            rank_comparison,
            model_status_rows,
            len(synthetic_samples),
            generation_failures,
            distribution,
            workbook_path,
        ),
        encoding="utf-8",
    )
    print(f"wrote workbook: {shared.display_path(workbook_path)}")
    print(f"wrote report: {shared.display_path(args.output_dir / 'report.md')}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise SystemExit(f"Electricity hourly dataset not found: {args.dataset}")
    if args.context_length <= 0 or args.horizon <= 0 or args.stride <= 0 or args.sample_count <= 0:
        raise SystemExit("context_length, horizon, stride, and sample_count must be positive")
    if args.synthetic_sample_count <= 0 or args.calibration_max_windows < 30 or args.calibration_splits <= 0:
        raise SystemExit("synthetic sample count and calibration settings must be positive")
    unknown = [capability for capability in args.capabilities if capability not in CAPABILITIES_BY_ID]
    if unknown:
        raise SystemExit(f"unknown synthetic capabilities: {', '.join(unknown)}")
    non_univariate = [
        capability
        for capability in args.capabilities
        if CAPABILITIES_BY_ID[capability].target_dim_mode != "fixed_1"
    ]
    if non_univariate:
        raise SystemExit("Electricity long-window experiment supports univariate capabilities only: " + ", ".join(non_univariate))


def apply_window_constants(args: argparse.Namespace) -> None:
    shared.CONTEXT_LENGTH = int(args.context_length)
    shared.HORIZON = int(args.horizon)
    shared.SEASON_LENGTH = int(args.season_length)


def calibration_spec(args: argparse.Namespace) -> calibration.BucketSpec:
    return calibration.BucketSpec(
        PROFILE_ID,
        "tsf_univariate",
        args.dataset.name,
        args.context_length,
        args.horizon,
        args.stride,
        args.season_length,
        synthetic_capabilities=tuple(args.capabilities),
    )


def calibrate_long_window_distribution(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = calibration_spec(args)
    real_rows = calibration.load_real_bucket(spec, args.dataset, max_windows=args.calibration_max_windows)
    bucket = calibration.calibrate_bucket(
        spec,
        real_rows,
        splits=args.calibration_splits,
        synthetic_count=args.calibration_synthetic_count,
        jitter_scale=calibration.DEFAULT_JITTER_SCALE,
        seed=_seed_for(args.seed, PROFILE_ID, 0),
    )
    thresholds = {name: values["mean"] for name, values in bucket["threshold_stability"].items()}
    online_bucket = calibration.online_artifact_bucket(
        spec,
        real_rows,
        thresholds=thresholds,
        reference_count=args.calibration_reference_count,
    )
    feature_quantiles: dict[str, dict[str, float]] = {}
    for feature_name in DEFAULT_FEATURES:
        values = np.asarray(
            [row["features"].get(feature_name, np.nan) for row in real_rows],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size:
            feature_quantiles[feature_name] = {
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
            }
    distribution = {
        "schema_version": "electricity_long_window_distribution.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": PROFILE_ID,
        "dataset": str(args.dataset),
        "context_length": args.context_length,
        "horizon": args.horizon,
        "season_length": args.season_length,
        "real_window_count": len(real_rows),
        "feature_quantiles": feature_quantiles,
        "threshold_stability": bucket["threshold_stability"],
        "control_summary": bucket["control_summary"],
        "splits": bucket["splits"],
    }
    artifact = {
        "schema_version": "synthetic_v2_near_distance_online.v1",
        "created_at": distribution["created_at"],
        "config": {
            "strict_rule": "raw_mae_d1 <= p01 AND raw_l2_d1 <= p01",
            "combined_rule": "raw_mae_d1 <= p05 AND raw_l2_d1 <= p05 AND (feature_l2_d1 <= p01 OR raw_mae_nndr <= p01)",
            "artifact_reference_count": args.calibration_reference_count,
        },
        "buckets": {PROFILE_ID: online_bucket},
    }
    return distribution, artifact


def write_distribution_outputs(args: argparse.Namespace, distribution: dict[str, Any], artifact: dict[str, Any]) -> None:
    (args.output_dir / "distribution_profile.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "near_distance_artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_synthetic_samples(
    args: argparse.Namespace,
    artifact: dict[str, Any],
) -> tuple[list[shared.ExperimentSample], list[dict[str, Any]]]:
    samples: list[shared.ExperimentSample] = []
    failures: list[dict[str, Any]] = []
    length = args.context_length + args.horizon
    base_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for capability_id in args.capabilities:
        for intensity in range(1, 6):
            for sample_index in range(args.synthetic_sample_count):
                sample_seed = _seed_for(args.seed, capability_id, intensity * 10_000 + sample_index)
                accepted_result: tuple[np.ndarray, dict[str, Any], dict[str, float], dict[str, Any], int] | None = None
                last_failure: dict[str, Any] = {}
                for attempt in range(24):
                    rng = np.random.default_rng(_attempt_seed(sample_seed, attempt))
                    target, latent_params, covariates = _generate_sample_values(
                        capability_id,
                        length,
                        args.context_length,
                        1,
                        args.season_length,
                        intensity,
                        rng,
                    )
                    target = _standardize_by_context(target, args.context_length)
                    realized_features = _realized_features(target, covariates, args.season_length, args.context_length)
                    feature_accepted, failed_features = _accept_synthetic_features(capability_id, realized_features)
                    near_distance = evaluate_near_distance_gate(
                        target=target,
                        features=realized_features,
                        profile_ids=(PROFILE_ID,),
                        context_length=args.context_length,
                        horizon=args.horizon,
                        artifact=artifact,
                    )
                    last_failure = {
                        "failed_features": failed_features,
                        "near_distance_status": near_distance.get("status"),
                        "strict_risk": near_distance.get("strict_risk"),
                        "combined_risk": near_distance.get("combined_risk"),
                    }
                    if feature_accepted and near_distance.get("accepted"):
                        accepted_result = (target, latent_params, realized_features, near_distance, attempt + 1)
                        break
                if accepted_result is None:
                    failures.append(
                        {
                            "capability_id": capability_id,
                            "intensity": intensity,
                            "sample_index": sample_index,
                            "sample_seed": sample_seed,
                            "attempts": 24,
                            **last_failure,
                        }
                    )
                    continue
                target, latent_params, realized_features, near_distance, attempts = accepted_result
                offset = len(samples) * length
                timestamps = [(base_start + timedelta(hours=offset + step)).isoformat() for step in range(length)]
                validation = {
                    "schema_version": "synthetic_post_generation_validation.v1",
                    "feature_gate": {
                        "accepted": True,
                        "caps": PILOT_ACCEPTANCE_CAPS.get(capability_id, {}),
                        "mins": PILOT_ACCEPTANCE_MINS.get(capability_id, {}),
                    },
                    "near_distance": near_distance,
                    "attempts": attempts,
                    "latent_params": latent_params,
                    "profile_id": PROFILE_ID,
                }
                samples.append(
                    shared.ExperimentSample(
                        sample_id=f"{capability_id}-i{intensity}-{sample_index:03d}",
                        dataset_kind="synthetic",
                        source_dataset="synthetic_v2_electricity_long_window_anchor",
                        capability_id=capability_id,
                        intensity=intensity,
                        sample_index=sample_index,
                        series_id=None,
                        window_start=None,
                        history_timestamps=timestamps[: args.context_length],
                        future_timestamps=timestamps[args.context_length :],
                        target_column_names=["target_0"],
                        target_history=target[: args.context_length].astype(float).tolist(),
                        target_future=target[args.context_length :].astype(float).tolist(),
                        realized_features=realized_features,
                        validation=validation,
                    )
                )
    return samples, failures


def load_electricity_samples(args: argparse.Namespace) -> list[RealSample]:
    metadata, records = read_tsf_series_records(args.dataset)
    spec = WindowSpec(args.context_length, args.horizon, args.stride)
    series = [(record.series_id, record.values) for record in records]
    selected = select_tsf_windows(series, spec, max_windows=args.sample_count)
    samples: list[RealSample] = []
    fallback_base = datetime(2012, 1, 1, tzinfo=timezone.utc)
    for selected_index, (series_index, window_start, window) in enumerate(selected):
        if window.shape[0] != spec.length or not np.isfinite(window).all():
            continue
        record = records[series_index]
        timestamps, timestamp_source = shared.tsf_window_timestamps(
            record,
            metadata=metadata,
            window_start=window_start,
            length=spec.length,
            fallback_start=fallback_base + timedelta(hours=selected_index * spec.length),
        )
        values = window.astype(float)
        samples.append(
            RealSample(
                sample_id=f"electricity_real-{len(samples):04d}",
                dataset_kind="real",
                source_dataset="electricity_hourly",
                capability_id="electricity_original",
                intensity=0,
                sample_index=len(samples),
                series_id=record.series_id,
                window_start=int(window_start),
                history_timestamps=timestamps[: args.context_length],
                future_timestamps=timestamps[args.context_length :],
                target_column_names=["target_0"],
                target_history=values[: args.context_length].tolist(),
                target_future=values[args.context_length :].tolist(),
                realized_features=feature_vector(values, season_length=args.season_length, context_length=args.context_length),
                validation={
                    "feature_gate": {"accepted": True, "enforced": False, "reason": "real_original_reference"},
                    "timestamp_source": timestamp_source,
                    "source_start_timestamp": record.attributes.get("start_timestamp"),
                },
            )
        )
        if len(samples) >= args.sample_count:
            break
    if len(samples) < args.sample_count:
        raise RuntimeError(f"only found {len(samples)} finite Electricity windows, requested {args.sample_count}")
    return samples


def select_models(
    service_models: list[dict[str, Any]],
    requested: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected, skipped = shared.select_models(service_models, requested)
    kept: list[dict[str, Any]] = []
    for model in selected:
        limits = model.get("forecast_limits") or {}
        max_input_length = limits.get("max_input_length")
        if max_input_length is not None and int(max_input_length) < int(args.context_length):
            skipped.append(
                {
                    "model_id": model.get("model_id"),
                    "reason": "max_input_length_unsupported",
                    "forecast_limits": limits,
                }
            )
            continue
        kept.append(model)
    return kept, skipped


def build_real_rankings(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    failures: dict[str, int] = {}
    for row in metric_rows:
        model_id = str(row.get("model_id"))
        if row.get("status") == "succeeded":
            by_model.setdefault(model_id, []).append(row)
        else:
            failures[model_id] = failures.get(model_id, 0) + 1
    for model_id, rows in by_model.items():
        metrics = shared.aggregate_metrics(rows)
        rank_value = metrics.get(shared.RANK_METRIC) or metrics.get("mae")
        aggregates.append(
            {
                "rank_scope": "real_original",
                "feature_dimension": "electricity_original",
                "capability_id": "electricity_original",
                "intensity": 0,
                "model_id": model_id,
                "model_group": rows[0].get("model_group"),
                "sample_count": len(rows),
                "failed_count": failures.get(model_id, 0),
                "rank_metric": shared.RANK_METRIC if metrics.get(shared.RANK_METRIC) is not None else "mae",
                "rank_metric_value": rank_value,
                **{f"mean_{key}": value for key, value in metrics.items()},
            }
        )
    aggregates = sorted(aggregates, key=lambda item: (shared.float_or_inf(item.get("rank_metric_value")), str(item.get("model_id"))))
    return [{**row, "rank": rank} for rank, row in enumerate(aggregates, start=1)]


def distribution_rows(distribution: dict[str, Any], artifact: dict[str, Any]) -> list[dict[str, Any]]:
    bucket = artifact["buckets"][PROFILE_ID]
    rows: list[dict[str, Any]] = [
        {
            "source": "feature_profile",
            "profile_id": PROFILE_ID,
            "context_length": distribution["context_length"],
            "horizon": distribution["horizon"],
            "season_length": distribution["season_length"],
            "window_count": distribution["real_window_count"],
        },
        {
            "source": "near_distance_online_artifact",
            "profile_id": PROFILE_ID,
            "reference_count": bucket["reference_count"],
            **shared.flatten_prefixed("threshold", bucket["thresholds"]),
        },
    ]
    for feature_name, values in sorted(distribution["feature_quantiles"].items()):
        rows.append({"source": "feature_profile_quantile", "profile_id": PROFILE_ID, "feature": feature_name, **values})
    for metric_name, values in sorted(distribution["threshold_stability"].items()):
        rows.append({"source": "near_distance_threshold_stability", "profile_id": PROFILE_ID, "metric": metric_name, **values})
    for control_name, metrics in sorted(distribution["control_summary"].items()):
        row: dict[str, Any] = {"source": "near_distance_control_summary", "profile_id": PROFILE_ID, "control": control_name}
        for metric_name, values in metrics.items():
            for key, value in values.items():
                row[f"{metric_name}_{key}"] = value
        rows.append(row)
    return rows


def run_config_rows(
    args: argparse.Namespace,
    samples: list[RealSample | shared.ExperimentSample],
    real_samples: list[RealSample],
    synthetic_samples: list[shared.ExperimentSample],
    selected_models: list[dict[str, Any]],
    skipped_models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"key": "dataset_id", "value": DATASET_ID},
        {"key": "dataset", "value": str(args.dataset)},
        {"key": "context_length", "value": args.context_length},
        {"key": "horizon", "value": args.horizon},
        {"key": "season_length", "value": args.season_length},
        {"key": "profile_id", "value": PROFILE_ID},
        {"key": "real_sample_count", "value": len(real_samples)},
        {"key": "synthetic_sample_count", "value": len(synthetic_samples)},
        {"key": "total_sample_count", "value": len(samples)},
        {"key": "synthetic_sample_count_per_feature_intensity", "value": args.synthetic_sample_count},
        {"key": "capabilities", "value": ", ".join(args.capabilities)},
        {"key": "stride", "value": args.stride},
        {"key": "calibration_max_windows", "value": args.calibration_max_windows},
        {"key": "calibration_splits", "value": args.calibration_splits},
        {"key": "calibration_reference_count", "value": args.calibration_reference_count},
        {"key": "batch_size", "value": args.batch_size},
        {"key": "models", "value": ", ".join(str(model.get("model_id")) for model in selected_models)},
        {"key": "skipped_models", "value": json.dumps(skipped_models, ensure_ascii=False, sort_keys=True)},
        {"key": "unload_after_model", "value": args.unload_after_model},
    ]


def render_report(
    args: argparse.Namespace,
    real_rankings: list[dict[str, Any]],
    synthetic_rankings: list[dict[str, Any]],
    rank_comparison: list[dict[str, Any]],
    model_status_rows: list[dict[str, Any]],
    synthetic_sample_count: int,
    generation_failures: list[dict[str, Any]],
    distribution: dict[str, Any],
    workbook_path: Path,
) -> str:
    status_lines = [
        f"- `{row.get('model_id')}`: {row.get('status')}, succeeded={row.get('succeeded_count')}, "
        f"failed={row.get('failed_count')}, load={shared.fmt(row.get('load_seconds'))}s, "
        f"elapsed={shared.fmt(row.get('elapsed_seconds'))}s, unloaded={row.get('unloaded')}"
        for row in model_status_rows
        if row.get("selection") != "skipped"
    ]
    return "\n".join(
        [
            "# Electricity hourly 长窗口合成数据全链路实验",
            "",
            f"生成时间：{datetime.now(timezone.utc).isoformat()}",
            "",
            "## 实验单元",
            "",
            f"- 真实数据：Electricity hourly，`{shared.display_path(args.dataset)}`。",
            f"- 窗口：context `{args.context_length}`，horizon `{args.horizon}`，season `{args.season_length}`。",
            f"- 真实评测窗口数：`{args.sample_count}`；分布和阈值校准窗口数：`{distribution['real_window_count']}`。",
            f"- 合成能力维度：{', '.join(f'`{capability}`' for capability in args.capabilities)}。",
            f"- 每个能力维度每档 intensity 样本数：`{args.synthetic_sample_count}`。",
            f"- 合成样本通过硬特征 acceptance 和 `{PROFILE_ID}` 近距离 acceptance 后进入模型评测。",
            f"- 模型执行：按模型顺序加载、评测、{'卸载' if args.unload_after_model else '保留加载状态'}。",
            f"- Excel 结果：`{shared.display_path(workbook_path)}`。",
            "",
            "## 模型运行状态",
            "",
            *(status_lines or ["- 没有成功运行服务模型。"]),
            "",
            "## 真实窗口排名",
            "",
            *shared.ranking_markdown_lines(real_rankings),
            "",
            "## 特征维度 x 强度排名观察",
            "",
            *shared.best_synthetic_lines(synthetic_rankings),
            "",
            "## 与真实 Electricity 排名的差异",
            "",
            *shared.rank_delta_lines(rank_comparison),
            "",
            "## 生成与回验",
            "",
            f"- 接受的合成样本：`{synthetic_sample_count}` 个。",
            f"- 生成失败请求：`{len(generation_failures)}` 个，记录在 `generation_failures.jsonl`（若非零）。",
            f"- 阈值由 `{args.calibration_splits}` 次 train/holdout split 校准，exact copy、jitter copy、normal synthetic 控制实验保存在 `distribution_profile.json`。",
            "",
            "## 产物",
            "",
            "- `electricity_long_window_2048ctx_24h.xlsx`：配置、分布、双门禁、模型状态、真实排名、特征×强度排名、排名对比、逐样本指标和逐点预测。",
            "- `distribution_profile.json`：2048+24 真实分布特征分位数、阈值稳定性与控制实验。",
            "- `near_distance_artifact.json`：本次实验使用的在线近距离门禁参考和阈值。",
            "- `samples.jsonl`：真实窗口和通过回验的合成窗口元信息。",
            "- `sample_metrics.jsonl`：每个模型在每个窗口上的 MAE/MSE/MASE。",
            "- `forecast_points.jsonl`：每个 horizon step 的 actual/prediction/error。",
            "",
            "## 复现命令",
            "",
            "```bash",
            reproduction_command(args),
            "```",
            "",
        ]
    )


def reproduction_command(args: argparse.Namespace) -> str:
    model_args = " ".join(shlex.quote(model) for model in args.models)
    parts = [
        "cd /root/xmy/TSBenchmark",
        "&&",
        "PYTHONPATH=backend:scripts",
        "backend/.venv/bin/python" if (REPO_ROOT / "backend/.venv/bin/python").exists() else "python",
        "scripts/run_electricity_long_window_experiment.py",
        "--models",
        model_args,
        "--context-length",
        str(args.context_length),
        "--horizon",
        str(args.horizon),
        "--season-length",
        str(args.season_length),
        "--sample-count",
        str(args.sample_count),
        "--synthetic-sample-count",
        str(args.synthetic_sample_count),
        "--capabilities",
        *args.capabilities,
        "--stride",
        str(args.stride),
        "--calibration-max-windows",
        str(args.calibration_max_windows),
        "--calibration-splits",
        str(args.calibration_splits),
        "--calibration-reference-count",
        str(args.calibration_reference_count),
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


if __name__ == "__main__":
    raise SystemExit(main())
