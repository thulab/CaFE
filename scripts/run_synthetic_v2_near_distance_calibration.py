#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (BACKEND_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.synthetic_generation_service import (  # noqa: E402
    _generate_sample_values,
    _realized_features,
    _seed_for,
)
from app.services.synthetic_generator_conditioning import resolve_generator_conditioning  # noqa: E402
from synthetic_feature_profile import (  # noqa: E402
    DEFAULT_FEATURES,
    WindowSpec,
    limit_candidates,
    m5_covariate_matrix,
    m5_hierarchy_values,
    read_gefcom2014_load_frame,
    read_m5_calendar_and_sales,
    read_m5_prices,
    read_tsf_series,
    sample_frame_evenly,
    sample_sequence_evenly,
    select_tsf_panel_windows,
    select_tsf_windows,
    window_starts,
)


DEFAULT_DATA_DIR = REPO_ROOT / "runtime/research"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/research/synthetic-v2-near-distance-calibration"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs/superpowers/baselines/2026-07-08-synthetic-v2-near-distance-calibration.md"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "backend/app/data/synthetic_v2_near_distance_artifact.json"
DEFAULT_MAX_WINDOWS = 600
DEFAULT_SPLITS = 5
DEFAULT_SYNTHETIC_COUNT = 48
DEFAULT_JITTER_SCALE = 0.02
DEFAULT_ARTIFACT_REFERENCE_COUNT = 192
DEFAULT_HOLDOUT_FRACTION = 0.2
MIN_REFERENCE_ROWS = 20
MIN_HOLDOUT_ROWS = 10


@dataclass(frozen=True)
class BucketSpec:
    profile_id: str
    kind: str
    asset_name: str
    context_length: int
    horizon: int
    stride: int
    season_length: int
    target_dim: int = 1
    covariate_dim: int = 0
    hierarchy: str | None = None
    max_series: int = 240
    max_groups: int = 20
    task: int = 1
    synthetic_capabilities: tuple[str, ...] = ("trend",)


BUCKET_SPECS: tuple[BucketSpec, ...] = (
    BucketSpec(
        "m4_hourly_daily_168ctx",
        "tsf_univariate",
        "m4_hourly_dataset.zip",
        168,
        24,
        24,
        24,
        synthetic_capabilities=(
            "trend",
            "multi_seasonal",
            "time_varying_seasonality",
            "regime_switching",
            "nonlinear_persistence",
            "predictable_intermittency",
        ),
    ),
    BucketSpec(
        "electricity_hourly_daily_168ctx",
        "tsf_univariate",
        "electricity_hourly_dataset.zip",
        168,
        24,
        24,
        24,
        synthetic_capabilities=(
            "trend",
            "multi_seasonal",
            "time_varying_seasonality",
            "regime_switching",
            "nonlinear_persistence",
            "predictable_intermittency",
        ),
    ),
    BucketSpec(
        "traffic_hourly_daily_168ctx",
        "tsf_univariate",
        "traffic_hourly_dataset.zip",
        168,
        24,
        24,
        24,
        synthetic_capabilities=(
            "trend",
            "multi_seasonal",
            "time_varying_seasonality",
            "regime_switching",
            "nonlinear_persistence",
            "predictable_intermittency",
        ),
    ),
    BucketSpec(
        "electricity_hourly_panel_168ctx",
        "tsf_panel",
        "electricity_hourly_dataset.zip",
        168,
        24,
        24,
        24,
        target_dim=3,
        synthetic_capabilities=("common_factor",),
    ),
    BucketSpec(
        "traffic_hourly_panel_168ctx",
        "tsf_panel",
        "traffic_hourly_dataset.zip",
        168,
        24,
        24,
        24,
        target_dim=3,
        synthetic_capabilities=("common_factor",),
    ),
    BucketSpec(
        "m5_daily_covariate_365ctx_28h",
        "m5_covariate",
        "m5-forecasting-accuracy.zip",
        365,
        28,
        28,
        7,
        covariate_dim=4,
        synthetic_capabilities=("covariate_response",),
    ),
    BucketSpec(
        "m5_daily_hierarchy_365ctx_28h",
        "m5_hierarchy",
        "m5-forecasting-accuracy.zip",
        365,
        28,
        28,
        7,
        target_dim=3,
        hierarchy="additive_first",
        synthetic_capabilities=("hierarchical_coherence",),
    ),
    BucketSpec(
        "gefcom2014_load_hourly_covariate_168ctx_24h",
        "gefcom2014_load",
        "GEFCom2014.zip",
        168,
        24,
        24,
        24,
        covariate_dim=25,
        synthetic_capabilities=("covariate_response",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate near-distance DCR/NNDR thresholds for synthetic v2 real buckets.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--synthetic-count", type=int, default=DEFAULT_SYNTHETIC_COUNT)
    parser.add_argument("--jitter-scale", type=float, default=DEFAULT_JITTER_SCALE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--artifact-reference-count", type=int, default=DEFAULT_ARTIFACT_REFERENCE_COUNT)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--buckets", nargs="*", default=None, help="Optional profile_id subset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_calibration(
        data_dir=args.data_dir,
        max_windows=args.max_windows,
        splits=args.splits,
        synthetic_count=args.synthetic_count,
        jitter_scale=args.jitter_scale,
        seed=args.seed,
        bucket_ids=tuple(args.buckets) if args.buckets else None,
        artifact_reference_count=args.artifact_reference_count,
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(summary, output_dir=args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    if args.artifact:
        artifact = build_online_artifact(summary)
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote summary: {summary_path}")
    print(f"wrote report: {args.report}")
    if args.artifact:
        print(f"wrote online artifact: {args.artifact}")
    return 0


def run_calibration(
    *,
    data_dir: Path,
    max_windows: int,
    splits: int,
    synthetic_count: int,
    jitter_scale: float,
    seed: int,
    bucket_ids: tuple[str, ...] | None = None,
    artifact_reference_count: int = DEFAULT_ARTIFACT_REFERENCE_COUNT,
) -> dict[str, Any]:
    selected_specs = [spec for spec in BUCKET_SPECS if bucket_ids is None or spec.profile_id in bucket_ids]
    if not selected_specs:
        raise ValueError("no bucket specs selected")
    buckets: list[dict[str, Any]] = []
    for spec in selected_specs:
        real_rows = load_real_bucket(spec, data_dir / spec.asset_name, max_windows=max_windows)
        bucket_summary = calibrate_bucket(
            spec,
            real_rows,
            splits=splits,
            synthetic_count=synthetic_count,
            jitter_scale=jitter_scale,
            seed=_seed_for(seed, spec.profile_id, 0),
            reference_count=artifact_reference_count,
        )
        buckets.append(bucket_summary)
    return {
        "schema_version": "synthetic_v2_near_distance_calibration.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "data_dir": str(data_dir),
            "max_windows_per_bucket": max_windows,
            "splits": splits,
            "synthetic_count": synthetic_count,
            "jitter_scale": jitter_scale,
            "artifact_reference_count": artifact_reference_count,
            "split_policy": "series/panel-group holdout; single-series temporal block with C+H embargo",
            "deployment_split_index": 0,
            "seed": seed,
            "strict_rule": "full-window OR context-only raw MAE/L2 DCR <= corresponding real-holdout p01",
            "combined_rule": "full-window combined rule OR context raw MAE/L2 <= p05 AND context NNDR <= p01",
        },
        "buckets": buckets,
        "overall": summarize_overall(buckets),
    }


def online_artifact_bucket(
    spec: BucketSpec,
    reference_rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float],
    split_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(reference_rows) < 2:
        raise ValueError("online near-distance artifact needs at least two reference rows")
    feature_names = feature_names_for_train(reference_rows)
    feature_values = feature_matrix(reference_rows, feature_names)
    feature_center = np.nanmedian(feature_values, axis=0)
    feature_scale = robust_feature_scale(feature_values)
    reference_features_z = (feature_values - feature_center) / feature_scale
    return {
        "profile_id": spec.profile_id,
        "context_length": spec.context_length,
        "horizon": spec.horizon,
        "season_length": spec.season_length,
        "target_dim": spec.target_dim,
        "covariate_dim": spec.covariate_dim,
        "reference_count": len(reference_rows),
        "split": split_summary or {},
        "feature_names": list(feature_names),
        "feature_center": round_nested(feature_center),
        "feature_scale": round_nested(feature_scale),
        "reference_raw": round_nested(np.vstack([row["raw"] for row in reference_rows])),
        "reference_context_raw": round_nested(np.vstack([row["context_raw"] for row in reference_rows])),
        "reference_features_z": round_nested(reference_features_z),
        "thresholds": {
            key: float(thresholds[key])
            for key in (
                "raw_mae_p01",
                "raw_mae_p05",
                "raw_l2_p01",
                "raw_l2_p05",
                "feature_l2_p01",
                "feature_l2_p05",
                "raw_mae_nndr_p01",
                "raw_mae_nndr_p05",
                "context_raw_mae_p01",
                "context_raw_mae_p05",
                "context_raw_l2_p01",
                "context_raw_l2_p05",
                "context_raw_mae_nndr_p01",
                "context_raw_mae_nndr_p05",
            )
        },
    }


def build_online_artifact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "synthetic_v2_near_distance_online.v2",
        "created_at": summary["created_at"],
        "source_summary_schema_version": summary["schema_version"],
        "config": {
            "strict_rule": summary["config"]["strict_rule"],
            "combined_rule": summary["config"]["combined_rule"],
            "artifact_reference_count": summary["config"]["artifact_reference_count"],
            "split_policy": summary["config"]["split_policy"],
            "deployment_split_index": summary["config"]["deployment_split_index"],
        },
        "buckets": {
            bucket["profile_id"]: bucket["online_artifact"]
            for bucket in summary["buckets"]
        },
    }


def round_nested(values: np.ndarray, digits: int = 6) -> list[Any]:
    arr = np.asarray(values, dtype=float)
    rounded = np.round(arr, digits)
    return rounded.tolist()


def load_real_bucket(spec: BucketSpec, path: Path, *, max_windows: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found for {spec.profile_id}: {path}")
    if spec.kind == "tsf_univariate":
        _metadata, series = read_tsf_series(path)
        windows = select_tsf_windows(series, WindowSpec(spec.context_length, spec.horizon, spec.stride), max_windows=max_windows)
        return [
            make_real_row(window, spec, group_id=f"series:{series_index}", window_start=start)
            for series_index, start, window in windows
            if np.isfinite(window).all() and is_informative_target(window, spec.context_length)
        ]
    if spec.kind == "tsf_panel":
        _metadata, series = read_tsf_series(path)
        windows = select_tsf_panel_windows(
            series,
            WindowSpec(spec.context_length, spec.horizon, spec.stride),
            target_dim=spec.target_dim,
            max_windows=max_windows,
        )
        return [
            make_real_row(
                window,
                spec,
                group_id="panel:" + ",".join(str(index) for index in group),
                window_start=start,
            )
            for group, start, window in windows
            if np.isfinite(window).all() and is_informative_target(window, spec.context_length)
        ]
    if spec.kind == "m5_covariate":
        return load_m5_covariate_rows(spec, path, max_windows=max_windows)
    if spec.kind == "m5_hierarchy":
        return load_m5_hierarchy_rows(spec, path, max_windows=max_windows)
    if spec.kind == "gefcom2014_load":
        return load_gefcom_rows(spec, path, max_windows=max_windows)
    raise ValueError(f"unsupported bucket kind: {spec.kind}")


def load_m5_covariate_rows(spec: BucketSpec, path: Path, *, max_windows: int) -> list[dict[str, Any]]:
    calendar, sales, day_columns = read_m5_calendar_and_sales(path)
    active_sales = sales.loc[sales[day_columns].sum(axis=1) > 0].reset_index(drop=True)
    selected_sales = sample_frame_evenly(active_sales if not active_sales.empty else sales, spec.max_series)
    prices = read_m5_prices(path, selected_sales[["store_id", "item_id"]].drop_duplicates())
    price_lookup = {
        (store_id, item_id): group.set_index("wm_yr_wk")["sell_price"].astype(float)
        for (store_id, item_id), group in prices.groupby(["store_id", "item_id"])
    }
    window_spec = WindowSpec(spec.context_length, spec.horizon, spec.stride)
    candidates = [
        (series_index, start)
        for series_index in range(len(selected_sales))
        for start in window_starts(len(day_columns), window_spec)
    ]
    rows: list[dict[str, Any]] = []
    for series_index, start in limit_candidates(candidates, max_windows):
        row = selected_sales.iloc[series_index]
        target = row[day_columns].to_numpy(dtype=float)[start : start + window_spec.length, None]
        covariates = m5_covariate_matrix(
            calendar,
            state_id=str(row["state_id"]),
            price_series=price_lookup.get((row["store_id"], row["item_id"])),
        )[start : start + window_spec.length]
        if (
            target.shape[0] == window_spec.length
            and np.isfinite(target).all()
            and np.isfinite(covariates).all()
            and is_informative_target(target, spec.context_length)
        ):
            rows.append(
                make_real_row(
                    target,
                    spec,
                    covariates=covariates,
                    group_id=f"m5-series:{series_index}",
                    window_start=start,
                )
            )
    return rows


def load_m5_hierarchy_rows(spec: BucketSpec, path: Path, *, max_windows: int) -> list[dict[str, Any]]:
    _calendar, sales, day_columns = read_m5_calendar_and_sales(path)
    group_sizes = sales.groupby(["store_id", "cat_id"])["dept_id"].nunique()
    groups = sample_sequence_evenly([group for group, count in group_sizes.items() if count == 2], spec.max_groups)
    group_values = [m5_hierarchy_values(sales, day_columns, group) for group in groups]
    window_spec = WindowSpec(spec.context_length, spec.horizon, spec.stride)
    candidates = [
        (group_index, start)
        for group_index, values in enumerate(group_values)
        for start in window_starts(values.shape[0], window_spec)
    ]
    rows: list[dict[str, Any]] = []
    for group_index, start in limit_candidates(candidates, max_windows):
        window = group_values[group_index][start : start + window_spec.length]
        if window.shape[0] == window_spec.length and np.isfinite(window).all() and is_informative_target(window, spec.context_length):
            rows.append(
                make_real_row(
                    window,
                    spec,
                    group_id=f"m5-hierarchy:{group_index}",
                    window_start=start,
                )
            )
    return rows


def load_gefcom_rows(spec: BucketSpec, path: Path, *, max_windows: int) -> list[dict[str, Any]]:
    frame, _source_name = read_gefcom2014_load_frame(path, task=spec.task)
    covariate_columns = [column for column in frame.columns if column.startswith("w")]
    frame = frame[["LOAD", *covariate_columns]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    target = frame["LOAD"].to_numpy(dtype=float)
    covariates = frame[covariate_columns].to_numpy(dtype=float)
    window_spec = WindowSpec(spec.context_length, spec.horizon, spec.stride)
    rows: list[dict[str, Any]] = []
    for start in window_starts(len(target), WindowSpec(spec.context_length, spec.horizon, spec.stride, max_windows=max_windows)):
        window = target[start : start + window_spec.length, None]
        covariate_window = covariates[start : start + window_spec.length]
        if np.isfinite(window).all() and np.isfinite(covariate_window).all() and is_informative_target(window, spec.context_length):
            rows.append(
                make_real_row(
                    window,
                    spec,
                    covariates=covariate_window,
                    group_id=f"gefcom-task:{spec.task}",
                    window_start=start,
                )
            )
    return rows


def make_real_row(
    target: np.ndarray,
    spec: BucketSpec,
    *,
    covariates: np.ndarray | None = None,
    group_id: str | None = None,
    window_start: int | None = None,
) -> dict[str, Any]:
    target_std = standardize_target(target, spec.context_length, hierarchy=spec.hierarchy)
    cov_std = normalize_covariates(covariates, spec.context_length) if covariates is not None and covariates.size else None
    row = make_row(target_std, spec, covariates=cov_std, label="real")
    row["group_id"] = group_id
    row["window_start"] = int(window_start) if window_start is not None else None
    return row


def make_row(target_std: np.ndarray, spec: BucketSpec, *, covariates: np.ndarray | None = None, label: str) -> dict[str, Any]:
    features = _realized_features(
        target_std,
        covariates,
        spec.season_length,
        spec.context_length,
    )
    return {
        "label": label,
        "target": target_std.astype(float),
        "covariates": covariates.astype(float) if covariates is not None else None,
        "raw": flatten_raw(target_std),
        "context_raw": flatten_raw(target_std[: spec.context_length]),
        "features": features,
    }


def calibrate_bucket(
    spec: BucketSpec,
    real_rows: list[dict[str, Any]],
    *,
    splits: int,
    synthetic_count: int,
    jitter_scale: float,
    seed: int,
    reference_count: int = DEFAULT_ARTIFACT_REFERENCE_COUNT,
) -> dict[str, Any]:
    if len(real_rows) < 30:
        raise ValueError(f"bucket {spec.profile_id} needs at least 30 windows, got {len(real_rows)}")
    if splits <= 0:
        raise ValueError("splits must be positive")
    if reference_count < 2:
        raise ValueError("reference_count must be at least 2")
    split_rows = []
    synthetic_rows = generate_synthetic_rows(spec, count=synthetic_count, seed=seed)
    deployment_artifact: dict[str, Any] | None = None
    for split_index in range(splits):
        reference_candidates, holdout, split_summary = split_rows_leakage_safe(
            real_rows,
            spec,
            seed=_seed_for(seed, spec.profile_id, split_index),
        )
        reference = sample_sequence_evenly(
            reference_candidates,
            min(reference_count, len(reference_candidates)),
        )
        split_summary = {
            **split_summary,
            "reference_candidate_count": len(reference_candidates),
            "reference_count": len(reference),
            "reference_sampling": "all" if len(reference) == len(reference_candidates) else "even",
        }
        thresholds, diagnostics = thresholds_from_split(reference, holdout)
        attack_source = reference[: min(synthetic_count, len(reference))]
        control_rows = {
            "real_holdout": holdout,
            "exact_copy": attack_source,
            "affine_copy": affine_copy_rows(attack_source, spec),
            "context_copy": context_copy_rows(
                attack_source,
                spec,
                seed=_seed_for(seed, "context-copy", split_index),
            ),
            "jitter_copy": jitter_rows(
                attack_source,
                spec,
                jitter_scale=jitter_scale,
                seed=_seed_for(seed, "jitter", split_index),
            ),
            "normal_synthetic": synthetic_rows,
        }
        controls = {
            label: evaluate_risk(
                rows,
                reference,
                diagnostics["feature_names"],
                diagnostics["feature_center"],
                diagnostics["feature_scale"],
                thresholds,
            )
            for label, rows in control_rows.items()
        }
        split_rows.append(
            {
                "split_index": split_index,
                "train_count": len(reference),
                "holdout_count": len(holdout),
                "split": split_summary,
                "thresholds": thresholds,
                "controls": controls,
            }
        )
        if split_index == 0:
            deployment_artifact = online_artifact_bucket(
                spec,
                reference,
                thresholds=thresholds,
                split_summary=split_summary,
            )
    assert deployment_artifact is not None
    return {
        "profile_id": spec.profile_id,
        "kind": spec.kind,
        "context_length": spec.context_length,
        "horizon": spec.horizon,
        "season_length": spec.season_length,
        "target_dim": spec.target_dim,
        "covariate_dim": spec.covariate_dim,
        "real_window_count": len(real_rows),
        "synthetic_capabilities": list(spec.synthetic_capabilities),
        "threshold_stability": summarize_threshold_stability(split_rows),
        "control_summary": summarize_controls(split_rows),
        "splits": split_rows,
        "deployment_split_index": 0,
        "online_artifact": deployment_artifact,
    }


def generate_synthetic_rows(spec: BucketSpec, *, count: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    length = spec.context_length + spec.horizon
    for sample_index in range(count):
        capability_id = spec.synthetic_capabilities[sample_index % len(spec.synthetic_capabilities)]
        intensity = 1 + (sample_index % 5)
        rng = np.random.default_rng(_seed_for(seed, capability_id, sample_index))
        target_dim = 1 if capability_id == "covariate_response" else spec.target_dim
        conditioning = resolve_generator_conditioning(
            capability_id=capability_id,
            profile_id=spec.profile_id,
            context_length=spec.context_length,
            horizon=spec.horizon,
            target_dim=target_dim,
        )
        if conditioning is None:
            raise RuntimeError(
                f"missing generator conditioning for {spec.profile_id}/{capability_id}; "
                "build the generator conditioning artifact first"
            )
        target, _latent, covariates = _generate_sample_values(
            capability_id,
            length,
            spec.context_length,
            target_dim,
            spec.season_length,
            intensity,
            rng,
            generator_conditioning=conditioning,
        )
        target_std = standardize_target(
            target,
            spec.context_length,
            hierarchy="additive_first" if capability_id == "hierarchical_coherence" else None,
        )
        cov_std = normalize_covariates(covariates, spec.context_length) if covariates is not None and covariates.size else None
        rows.append(make_row(target_std, spec, covariates=cov_std, label=f"synthetic:{capability_id}:i{intensity}"))
    return rows


def split_rows_leakage_safe(
    rows: list[dict[str, Any]],
    spec: BucketSpec,
    *,
    seed: int,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split real windows without sharing a source series or overlapping time.

    Multi-series/panel buckets assign complete groups to one side. A genuine
    single-series bucket uses a contiguous temporal holdout and excludes every
    reference window whose interval overlaps the holdout boundary.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_id = str(row.get("group_id") or "single-series")
        groups.setdefault(group_id, []).append(row)
    rng = np.random.default_rng(seed)
    target_holdout_count = max(MIN_HOLDOUT_ROWS, int(math.ceil(len(rows) * holdout_fraction)))

    if len(groups) >= 2:
        group_ids = np.asarray(sorted(groups), dtype=object)
        rng.shuffle(group_ids)
        holdout_groups: list[str] = []
        holdout_count = 0
        for raw_group_id in group_ids:
            group_id = str(raw_group_id)
            remaining = len(rows) - holdout_count - len(groups[group_id])
            if remaining < MIN_REFERENCE_ROWS:
                continue
            holdout_groups.append(group_id)
            holdout_count += len(groups[group_id])
            if holdout_count >= target_holdout_count:
                break
        holdout_group_set = set(holdout_groups)
        reference = [row for group_id, group_rows in groups.items() if group_id not in holdout_group_set for row in group_rows]
        holdout = [row for group_id, group_rows in groups.items() if group_id in holdout_group_set for row in group_rows]
        summary = {
            "policy": "group",
            "group_count": len(groups),
            "reference_group_count": len(groups) - len(holdout_group_set),
            "holdout_group_count": len(holdout_group_set),
            "embargo_steps": 0,
            "discarded_window_count": 0,
        }
    else:
        if any(row.get("window_start") is None for row in rows):
            raise ValueError(
                f"bucket {spec.profile_id} needs window_start metadata for a leakage-safe single-series split"
            )
        ordered = sorted(rows, key=lambda row: int(row["window_start"]))
        window_length = int(spec.context_length + spec.horizon)
        holdout_count = min(target_holdout_count, len(ordered) - MIN_REFERENCE_ROWS)
        valid_boundaries: list[tuple[int, list[dict[str, Any]]]] = []
        for boundary in range(MIN_HOLDOUT_ROWS, len(ordered) - holdout_count + 1):
            first_holdout_start = int(ordered[boundary]["window_start"])
            candidate_reference = [
                row
                for row in ordered[:boundary]
                if int(row["window_start"]) + window_length <= first_holdout_start
            ]
            if len(candidate_reference) >= MIN_REFERENCE_ROWS:
                valid_boundaries.append((boundary, candidate_reference))
        if not valid_boundaries:
            raise ValueError(
                f"bucket {spec.profile_id} cannot form a temporal split with a C+H embargo"
            )
        boundary, reference = valid_boundaries[int(rng.integers(0, len(valid_boundaries)))]
        holdout = ordered[boundary : boundary + holdout_count]
        summary = {
            "policy": "temporal_embargo",
            "group_count": 1,
            "reference_group_count": 1,
            "holdout_group_count": 1,
            "embargo_steps": window_length,
            "first_holdout_start": int(holdout[0]["window_start"]),
            "last_reference_start": int(reference[-1]["window_start"]),
            "discarded_window_count": len(rows) - len(reference) - len(holdout),
        }

    if len(reference) < MIN_REFERENCE_ROWS or len(holdout) < MIN_HOLDOUT_ROWS:
        raise ValueError(
            f"bucket {spec.profile_id} produced an undersized leakage-safe split: "
            f"reference={len(reference)}, holdout={len(holdout)}"
        )
    return reference, holdout, {
        **summary,
        "reference_count": len(reference),
        "holdout_count": len(holdout),
    }


def thresholds_from_split(train: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    train_raw = np.vstack([row["raw"] for row in train])
    holdout_raw = np.vstack([row["raw"] for row in holdout])
    train_context_raw = np.vstack([row["context_raw"] for row in train])
    holdout_context_raw = np.vstack([row["context_raw"] for row in holdout])
    feature_names = feature_names_for_train(train)
    train_features = feature_matrix(train, feature_names)
    holdout_features = feature_matrix(holdout, feature_names)
    feature_center = np.nanmedian(train_features, axis=0)
    feature_scale = robust_feature_scale(train_features)
    train_features_z = (train_features - feature_center) / feature_scale
    holdout_features_z = (holdout_features - feature_center) / feature_scale
    raw_mae = nearest_distances(holdout_raw, train_raw, metric="mae")
    raw_l2 = nearest_distances(holdout_raw, train_raw, metric="l2")
    context_raw_mae = nearest_distances(holdout_context_raw, train_context_raw, metric="mae")
    context_raw_l2 = nearest_distances(holdout_context_raw, train_context_raw, metric="l2")
    feature_l2 = nearest_distances(holdout_features_z, train_features_z, metric="l2")
    thresholds = {
        "raw_mae_p01": positive_lower_tail_quantile(raw_mae["d1"], 0.01),
        "raw_mae_p05": positive_lower_tail_quantile(raw_mae["d1"], 0.05),
        "raw_l2_p01": positive_lower_tail_quantile(raw_l2["d1"], 0.01),
        "raw_l2_p05": positive_lower_tail_quantile(raw_l2["d1"], 0.05),
        "feature_l2_p01": positive_lower_tail_quantile(feature_l2["d1"], 0.01),
        "feature_l2_p05": positive_lower_tail_quantile(feature_l2["d1"], 0.05),
        "raw_mae_nndr_p01": positive_lower_tail_quantile(raw_mae["nndr"], 0.01),
        "raw_mae_nndr_p05": positive_lower_tail_quantile(raw_mae["nndr"], 0.05),
        "context_raw_mae_p01": positive_lower_tail_quantile(context_raw_mae["d1"], 0.01),
        "context_raw_mae_p05": positive_lower_tail_quantile(context_raw_mae["d1"], 0.05),
        "context_raw_l2_p01": positive_lower_tail_quantile(context_raw_l2["d1"], 0.01),
        "context_raw_l2_p05": positive_lower_tail_quantile(context_raw_l2["d1"], 0.05),
        "context_raw_mae_nndr_p01": positive_lower_tail_quantile(context_raw_mae["nndr"], 0.01),
        "context_raw_mae_nndr_p05": positive_lower_tail_quantile(context_raw_mae["nndr"], 0.05),
        "feature_l2_nndr_p01": positive_lower_tail_quantile(feature_l2["nndr"], 0.01),
        "feature_l2_nndr_p05": positive_lower_tail_quantile(feature_l2["nndr"], 0.05),
    }
    return thresholds, {
        "feature_names": feature_names,
        "feature_center": feature_center,
        "feature_scale": feature_scale,
    }


def positive_lower_tail_quantile(values: np.ndarray, quantile: float) -> float:
    """Estimate a natural-distance tail without letting exact duplicates collapse it.

    Cross-group exact duplicates are themselves copy-risk examples. They remain
    flagged in control rates, but are not allowed to set the usable DCR/NNDR
    threshold to zero for affine or jittered copies.
    """
    arr = np.asarray(values, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 1e-12)]
    if positive.size == 0:
        raise ValueError("cannot calibrate a novelty threshold from an all-zero distance distribution")
    return float(np.quantile(positive, quantile))


def evaluate_risk(
    query_rows: list[dict[str, Any]],
    train: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    feature_center: np.ndarray,
    feature_scale: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, float]:
    train_raw = np.vstack([row["raw"] for row in train])
    query_raw = np.vstack([row["raw"] for row in query_rows])
    train_context_raw = np.vstack([row["context_raw"] for row in train])
    query_context_raw = np.vstack([row["context_raw"] for row in query_rows])
    train_features_z = (feature_matrix(train, feature_names) - feature_center) / feature_scale
    query_features_z = (feature_matrix(query_rows, feature_names) - feature_center) / feature_scale
    raw_mae = nearest_distances(query_raw, train_raw, metric="mae")
    raw_l2 = nearest_distances(query_raw, train_raw, metric="l2")
    context_raw_mae = nearest_distances(query_context_raw, train_context_raw, metric="mae")
    context_raw_l2 = nearest_distances(query_context_raw, train_context_raw, metric="l2")
    feature_l2 = nearest_distances(query_features_z, train_features_z, metric="l2")
    full_strict = (raw_mae["d1"] <= thresholds["raw_mae_p01"]) & (raw_l2["d1"] <= thresholds["raw_l2_p01"])
    context_strict = (
        (context_raw_mae["d1"] <= thresholds["context_raw_mae_p01"])
        & (context_raw_l2["d1"] <= thresholds["context_raw_l2_p01"])
    )
    full_combined = (
        (raw_mae["d1"] <= thresholds["raw_mae_p05"])
        & (raw_l2["d1"] <= thresholds["raw_l2_p05"])
        & ((feature_l2["d1"] <= thresholds["feature_l2_p01"]) | (raw_mae["nndr"] <= thresholds["raw_mae_nndr_p01"]))
    )
    context_combined = (
        (context_raw_mae["d1"] <= thresholds["context_raw_mae_p05"])
        & (context_raw_l2["d1"] <= thresholds["context_raw_l2_p05"])
        & (context_raw_mae["nndr"] <= thresholds["context_raw_mae_nndr_p01"])
    )
    strict = full_strict | context_strict
    combined = full_combined | context_combined
    return {
        "count": float(len(query_rows)),
        "strict_risk_rate": float(np.mean(strict)),
        "combined_risk_rate": float(np.mean(combined)),
        "full_strict_risk_rate": float(np.mean(full_strict)),
        "context_strict_risk_rate": float(np.mean(context_strict)),
        "full_combined_risk_rate": float(np.mean(full_combined)),
        "context_combined_risk_rate": float(np.mean(context_combined)),
        "raw_mae_p05_hit_rate": float(np.mean(raw_mae["d1"] <= thresholds["raw_mae_p05"])),
        "raw_l2_p05_hit_rate": float(np.mean(raw_l2["d1"] <= thresholds["raw_l2_p05"])),
        "feature_l2_p01_hit_rate": float(np.mean(feature_l2["d1"] <= thresholds["feature_l2_p01"])),
        "raw_mae_nndr_p01_hit_rate": float(np.mean(raw_mae["nndr"] <= thresholds["raw_mae_nndr_p01"])),
        "raw_mae_d1_p05": float(np.quantile(raw_mae["d1"], 0.05)),
        "raw_l2_d1_p05": float(np.quantile(raw_l2["d1"], 0.05)),
        "feature_l2_d1_p05": float(np.quantile(feature_l2["d1"], 0.05)),
        "raw_mae_nndr_p05": float(np.quantile(raw_mae["nndr"], 0.05)),
        "context_raw_mae_d1_p05": float(np.quantile(context_raw_mae["d1"], 0.05)),
        "context_raw_l2_d1_p05": float(np.quantile(context_raw_l2["d1"], 0.05)),
        "context_raw_mae_nndr_p05": float(np.quantile(context_raw_mae["nndr"], 0.05)),
    }


def jitter_rows(rows: list[dict[str, Any]], spec: BucketSpec, *, jitter_scale: float, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for row in rows:
        target = row["target"] + rng.normal(0.0, jitter_scale, size=row["target"].shape)
        out.append(make_row(target, spec, covariates=row.get("covariates"), label="jitter_copy"))
    return out


def affine_copy_rows(rows: list[dict[str, Any]], spec: BucketSpec) -> list[dict[str, Any]]:
    """Copies hidden behind a scale/offset transformation.

    Context standardization should make this attack identical to its source and
    the novelty gate must therefore reject it.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        transformed = np.asarray(row["target"], dtype=float) * 1.7 + 2.5
        standardized = standardize_target(
            transformed,
            spec.context_length,
            hierarchy=spec.hierarchy,
        )
        out.append(make_row(standardized, spec, covariates=row.get("covariates"), label="affine_copy"))
    return out


def context_copy_rows(
    rows: list[dict[str, Any]],
    spec: BucketSpec,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Keep the model-visible context exact while replacing the answer span."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for row in rows:
        target = np.asarray(row["target"], dtype=float).copy()
        future = target[spec.context_length :]
        target[spec.context_length :] = rng.normal(0.0, 1.0, size=future.shape)
        out.append(make_row(target, spec, covariates=row.get("covariates"), label="context_copy"))
    return out


def standardize_target(values: np.ndarray, context_length: int, *, hierarchy: str | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    context = arr[:context_length]
    if hierarchy == "additive_first" and arr.shape[1] > 1:
        mean = context.mean(axis=0, keepdims=True)
        centered = arr - mean
        scale = float(np.std(context[:, 0]))
        if scale <= 1e-6:
            scale = float(np.mean(np.std(context, axis=0)))
        if scale <= 1e-6:
            scale = 1.0
        return centered / scale
    mean = context.mean(axis=0, keepdims=True)
    std = context.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (arr - mean) / std


def is_informative_target(values: np.ndarray, context_length: int) -> bool:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    context = arr[:context_length]
    if context.size == 0 or not np.isfinite(context).all():
        return False
    return bool(float(np.mean(np.std(context, axis=0))) > 1e-6)


def normalize_covariates(covariates: np.ndarray | None, context_length: int) -> np.ndarray | None:
    if covariates is None:
        return None
    normalized = np.asarray(covariates, dtype=float).copy()
    for index in range(normalized.shape[1]):
        full = normalized[:, index]
        if set(np.unique(full)).issubset({0.0, 1.0}):
            continue
        context = normalized[:context_length, index]
        mean = float(np.mean(context))
        std = float(np.std(context)) or 1.0
        normalized[:, index] = (full - mean) / std
    return normalized


def flatten_raw(target_std: np.ndarray) -> np.ndarray:
    arr = np.asarray(target_std, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.reshape(-1)


def feature_names_for_train(train: list[dict[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for feature in DEFAULT_FEATURES:
        values = np.asarray([row["features"].get(feature, np.nan) for row in train], dtype=float)
        if np.isfinite(values).any() and float(np.nanstd(values)) > 1e-9:
            names.append(feature)
    return tuple(names)


def feature_matrix(rows: list[dict[str, Any]], names: tuple[str, ...]) -> np.ndarray:
    matrix = np.empty((len(rows), len(names)), dtype=float)
    for row_index, row in enumerate(rows):
        for col_index, name in enumerate(names):
            value = row["features"].get(name, np.nan)
            matrix[row_index, col_index] = float(value) if np.isfinite(value) else np.nan
    if matrix.size == 0:
        return np.zeros((len(rows), 1), dtype=float)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = ~np.isfinite(matrix)
    if missing.any():
        matrix[missing] = np.take(medians, np.where(missing)[1])
    return matrix


def robust_feature_scale(values: np.ndarray) -> np.ndarray:
    q75 = np.nanpercentile(values, 75, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    scale = q75 - q25
    std = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-9, scale, std)
    return np.where(scale > 1e-9, scale, 1.0)


def nearest_distances(query: np.ndarray, reference: np.ndarray, *, metric: str) -> dict[str, np.ndarray]:
    d1 = np.empty(query.shape[0], dtype=float)
    d2 = np.empty(query.shape[0], dtype=float)
    kth = min(1, reference.shape[0] - 1)
    for start in range(0, query.shape[0], 128):
        block = query[start : start + 128]
        diff = block[:, None, :] - reference[None, :, :]
        if metric == "mae":
            distances = np.mean(np.abs(diff), axis=2)
        elif metric == "l2":
            distances = np.sqrt(np.mean(diff * diff, axis=2))
        else:
            raise ValueError(f"unknown metric: {metric}")
        part = np.partition(distances, kth=kth, axis=1)
        d1[start : start + block.shape[0]] = part[:, 0]
        d2[start : start + block.shape[0]] = part[:, 1] if reference.shape[0] > 1 else part[:, 0]
    return {"d1": d1, "d2": d2, "nndr": d1 / np.maximum(d2, 1e-9)}


def summarize_threshold_stability(split_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = sorted(split_rows[0]["thresholds"])
    return {
        key: summarize_values([split_row["thresholds"][key] for split_row in split_rows])
        for key in keys
    }


def summarize_controls(split_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    labels = sorted(split_rows[0]["controls"])
    metrics = sorted(next(iter(split_rows[0]["controls"].values())))
    return {
        label: {
            metric: summarize_values([split_row["controls"][label][metric] for split_row in split_rows])
            for metric in metrics
        }
        for label in labels
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / abs(mean)) if abs(mean) > 1e-12 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize_overall(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    normal_combined = [
        bucket["control_summary"]["normal_synthetic"]["combined_risk_rate"]["mean"]
        for bucket in buckets
    ]
    exact_strict = [
        bucket["control_summary"]["exact_copy"]["strict_risk_rate"]["mean"]
        for bucket in buckets
    ]
    jitter_combined = [
        bucket["control_summary"]["jitter_copy"]["combined_risk_rate"]["mean"]
        for bucket in buckets
    ]
    affine_strict = [
        bucket["control_summary"]["affine_copy"]["strict_risk_rate"]["mean"]
        for bucket in buckets
    ]
    context_strict = [
        bucket["control_summary"]["context_copy"]["strict_risk_rate"]["mean"]
        for bucket in buckets
    ]
    return {
        "bucket_count": len(buckets),
        "normal_synthetic_combined_risk_max": float(np.max(normal_combined)) if normal_combined else 0.0,
        "normal_synthetic_combined_risk_mean": float(np.mean(normal_combined)) if normal_combined else 0.0,
        "exact_copy_strict_risk_min": float(np.min(exact_strict)) if exact_strict else 0.0,
        "jitter_copy_combined_risk_min": float(np.min(jitter_combined)) if jitter_combined else 0.0,
        "affine_copy_strict_risk_min": float(np.min(affine_strict)) if affine_strict else 0.0,
        "context_copy_strict_risk_min": float(np.min(context_strict)) if context_strict else 0.0,
    }


def render_report(summary: dict[str, Any], *, output_dir: Path) -> str:
    summary_path = (output_dir / "summary.json").resolve()
    try:
        summary_display_path = summary_path.relative_to(REPO_ROOT)
    except ValueError:
        summary_display_path = summary_path
    lines = [
        "# Synthetic v2 Near-Distance Calibration",
        "",
        f"日期：{datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Purpose",
        "",
        "校准 DCR/NNDR 近距离污染风险阈值：用 real holdout 到 real train 的自然最近邻距离定 p01/p05 基线，并用 exact copy、jitter copy、normal synthetic 检查阈值是否能区分复制与正常生成。",
        "",
        "## Design",
        "",
        f"- Buckets: {len(summary['buckets'])} real profile buckets.",
        f"- Real windows per bucket cap: {summary['config']['max_windows_per_bucket']}; splits: {summary['config']['splits']}; synthetic controls per bucket: {summary['config']['synthetic_count']}.",
        f"- Jitter copy scale: {summary['config']['jitter_scale']} on context-standardized target values.",
        "- Raw distance is computed on context-standardized target windows. Feature distance uses robust-z explicit features fitted on each split's real train set.",
        "- Source series/panel groups never cross train/holdout. Single-series buckets use temporal blocks with a C+H non-overlap embargo.",
        "- Full target-window and model-visible target-context DCR are both checked; the deployed threshold and reference rows come from the same fixed split.",
        "- Near-constant real target windows are excluded before split calibration because zero-information windows can make p01 DCR thresholds collapse to zero.",
        "- Scope: raw DCR covers target trajectories in the committed R_train reference. Known-future covariates enter feature DCR but are not concatenated into the raw vector; R_holdout and unknown pretraining corpora are not coverage claims.",
        f"- Strict risk: {summary['config']['strict_rule']}.",
        f"- Combined risk: {summary['config']['combined_rule']}.",
        "",
        "## Threshold Stability",
        "",
        "| Bucket | real windows | raw MAE p01 mean/cv | raw L2 p01 mean/cv | feature L2 p01 mean/cv | NNDR p01 mean/cv |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bucket in summary["buckets"]:
        stability = bucket["threshold_stability"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{bucket['profile_id']}`",
                    str(bucket["real_window_count"]),
                    mean_cv(stability["raw_mae_p01"]),
                    mean_cv(stability["raw_l2_p01"]),
                    mean_cv(stability["feature_l2_p01"]),
                    mean_cv(stability["raw_mae_nndr_p01"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Positive/Negative Controls",
            "",
            "| Bucket | holdout combined | exact strict/combined | affine strict | context-copy strict | jitter strict/combined | normal strict/combined |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket in summary["buckets"]:
        controls = bucket["control_summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{bucket['profile_id']}`",
                    fmt(controls["real_holdout"]["combined_risk_rate"]["mean"]),
                    f"{fmt(controls['exact_copy']['strict_risk_rate']['mean'])}/{fmt(controls['exact_copy']['combined_risk_rate']['mean'])}",
                    fmt(controls["affine_copy"]["strict_risk_rate"]["mean"]),
                    fmt(controls["context_copy"]["strict_risk_rate"]["mean"]),
                    f"{fmt(controls['jitter_copy']['strict_risk_rate']['mean'])}/{fmt(controls['jitter_copy']['combined_risk_rate']['mean'])}",
                    f"{fmt(controls['normal_synthetic']['strict_risk_rate']['mean'])}/{fmt(controls['normal_synthetic']['combined_risk_rate']['mean'])}",
                ]
            )
            + " |"
        )
    overall = summary["overall"]
    lines.extend(
        [
            "",
            "## Overall Checks",
            "",
            f"- Exact-copy strict-risk minimum across buckets: `{fmt(overall['exact_copy_strict_risk_min'])}`.",
            f"- Jitter-copy combined-risk minimum across buckets: `{fmt(overall['jitter_copy_combined_risk_min'])}`.",
            f"- Affine-copy strict-risk minimum across buckets: `{fmt(overall['affine_copy_strict_risk_min'])}`.",
            f"- Context-copy strict-risk minimum across buckets: `{fmt(overall['context_copy_strict_risk_min'])}`.",
            f"- Normal-synthetic combined-risk max across buckets: `{fmt(overall['normal_synthetic_combined_risk_max'])}`.",
            "",
            "## Bucket Flags",
            "",
            "| Bucket | reason |",
            "| --- | --- |",
        ]
    )
    flags = bucket_flags(summary["buckets"])
    if flags:
        for profile_id, reason in flags:
            lines.append(f"| `{profile_id}` | {reason} |")
    else:
        lines.append("| - | No bucket exceeded the current warning heuristics. |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This calibrates novelty thresholds and writes the online near-distance reference artifact used by generation acceptance.",
            "- A good threshold should flag exact copies almost always, flag small jitter copies frequently, and keep normal synthetic combined risk near or below the paper tolerance target.",
            "- If a bucket has high threshold CV or high normal-synthetic risk, rerun with a larger real-window cap and inspect that bucket before freezing paper thresholds.",
            "",
            f"Full JSON summary: `{summary_display_path}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def bucket_flags(buckets: list[dict[str, Any]]) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []
    for bucket in buckets:
        reasons: list[str] = []
        stability = bucket["threshold_stability"]
        if stability["raw_mae_p01"]["cv"] > 0.3:
            reasons.append(f"raw MAE p01 CV={fmt(stability['raw_mae_p01']['cv'])}")
        if stability["feature_l2_p01"]["cv"] > 0.3:
            reasons.append(f"feature L2 p01 CV={fmt(stability['feature_l2_p01']['cv'])}")
        if bucket["control_summary"]["normal_synthetic"]["combined_risk_rate"]["mean"] > 0.01:
            reasons.append(f"normal synthetic combined risk={fmt(bucket['control_summary']['normal_synthetic']['combined_risk_rate']['mean'])}")
        if bucket["control_summary"]["affine_copy"]["strict_risk_rate"]["mean"] < 0.95:
            reasons.append(f"affine-copy strict risk={fmt(bucket['control_summary']['affine_copy']['strict_risk_rate']['mean'])}")
        if bucket["control_summary"]["context_copy"]["strict_risk_rate"]["mean"] < 0.95:
            reasons.append(f"context-copy strict risk={fmt(bucket['control_summary']['context_copy']['strict_risk_rate']['mean'])}")
        if reasons:
            flags.append((bucket["profile_id"], "; ".join(reasons)))
    return flags


def mean_cv(row: dict[str, float]) -> str:
    return f"{fmt(row['mean'])}/{fmt(row['cv'])}"


def fmt(value: Any) -> str:
    number = float(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    if abs(number) >= 0.001:
        return f"{number:.4f}"
    return f"{number:.3g}"


if __name__ == "__main__":
    raise SystemExit(main())
