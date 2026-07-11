#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURES = (
    "trend_strength",
    "seasonal_strength",
    "slope_abs",
    "curvature_abs",
    "noise_ratio",
    "acf1",
    "acf_abs_mean",
    "outlier_rate",
    "spike_rate",
    "multi_period_score",
    "seasonal_drift_score",
    "seasonal_amplitude_cv",
    "change_point_shift_energy",
    "level_shift_strength",
    "volatility_shift_strength",
    "nonlinear_lag1_gain",
    "burst_rate",
    "diff_spike_rate",
    "avg_abs_target_corr",
    "pca_top1_explained",
    "pca_top2_explained",
    "effective_factor_rank",
    "lead_lag_peak_abs",
    "avg_abs_covariate_target_corr",
    "future_abs_covariate_target_corr",
    "event_lift_abs",
    "hierarchy_residual_mean_abs",
)
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
BOUNDED_FEATURES = {
    "trend_strength",
    "seasonal_strength",
    "noise_ratio",
    "outlier_rate",
    "spike_rate",
    "multi_period_score",
    "burst_rate",
    "diff_spike_rate",
    "avg_abs_target_corr",
    "pca_top1_explained",
    "pca_top2_explained",
    "lead_lag_peak_abs",
    "avg_abs_covariate_target_corr",
    "future_abs_covariate_target_corr",
}


@dataclass(frozen=True)
class WindowSpec:
    context_length: int
    horizon: int
    stride: int
    max_windows: int | None = None

    @property
    def length(self) -> int:
        return self.context_length + self.horizon


@dataclass(frozen=True)
class TSFSeriesRecord:
    series_id: str
    values: np.ndarray
    attributes: dict[str, str]


def read_csv_series(
    path: Path,
    time_column: str,
    target_columns: list[str] | None,
    covariate_columns: list[str] | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    if time_column not in frame.columns:
        raise ValueError(f"time column not found: {time_column}")
    time = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    if time.isna().any():
        raise ValueError(f"time column contains unparsable values: {time_column}")

    if target_columns is None or not target_columns:
        target_columns = [
            str(column)
            for column in frame.columns
            if column != time_column and pd.api.types.is_numeric_dtype(frame[column])
        ]
    missing = [column for column in target_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"target columns not found: {', '.join(missing)}")

    targets = frame[target_columns].apply(pd.to_numeric, errors="coerce")
    if targets.isna().any().any():
        raise ValueError("target columns contain missing or non-numeric values")
    resolved_covariate_columns = covariate_columns or []
    missing_covariates = [column for column in resolved_covariate_columns if column not in frame.columns]
    if missing_covariates:
        raise ValueError(f"covariate columns not found: {', '.join(missing_covariates)}")
    covariates = frame[resolved_covariate_columns].apply(pd.to_numeric, errors="coerce")
    if not covariates.empty and covariates.isna().any().any():
        raise ValueError("covariate columns contain missing or non-numeric values")
    return time, targets.astype(float), covariates.astype(float)


def read_tsf_series(path: Path) -> tuple[dict[str, str], list[tuple[str, np.ndarray]]]:
    metadata, records = read_tsf_series_records(path)
    return metadata, [(record.series_id, record.values) for record in records]


def read_tsf_series_records(path: Path) -> tuple[dict[str, str], list[TSFSeriesRecord]]:
    text = read_text_or_first_tsf_from_zip(path)
    attributes: list[str] = []
    metadata: dict[str, str] = {}
    records: list[TSFSeriesRecord] = []
    in_data = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            lower = line.lower()
            if lower == "@data":
                in_data = True
                continue
            if lower.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    attributes.append(parts[1])
                continue
            if lower.startswith("@"):
                parts = line.split(maxsplit=1)
                metadata[parts[0][1:].lower()] = parts[1].strip() if len(parts) > 1 else ""
            continue

        pieces = line.split(":", len(attributes))
        if len(pieces) < len(attributes) + 1:
            continue
        attr_values = pieces[: len(attributes)]
        values_text = ":".join(pieces[len(attributes) :])
        values = parse_tsf_values(values_text)
        if values.size:
            attr_map = {
                attribute: attr_values[index]
                for index, attribute in enumerate(attributes)
                if index < len(attr_values)
            }
            series_id = tsf_series_id(attr_map, fallback=f"series_{len(records)}")
            records.append(TSFSeriesRecord(series_id=series_id, values=values, attributes=attr_map))
    if not records:
        raise ValueError(f"no series found in TSF input: {path}")
    return metadata, records


def tsf_series_id(attributes: dict[str, str], *, fallback: str) -> str:
    for key in ("series_name", "series_id", "series", "id"):
        value = attributes.get(key)
        if value:
            return value
    if attributes:
        first_value = next(iter(attributes.values()))
        if first_value:
            return first_value
    return fallback


def read_text_or_first_tsf_from_zip(path: Path) -> str:
    if path.suffix.lower() != ".zip":
        return decode_tsf_bytes(path.read_bytes(), source=str(path))
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".tsf")]
        if not names:
            raise ValueError(f"zip does not contain a .tsf file: {path}")
        with archive.open(sorted(names)[0]) as handle:
            return decode_tsf_bytes(handle.read(), source=f"{path}:{sorted(names)[0]}")


def decode_tsf_bytes(data: bytes, *, source: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"unable to decode TSF input: {source}")


def parse_tsf_values(values_text: str) -> np.ndarray:
    values: list[float] = []
    for raw in values_text.split(","):
        item = raw.strip()
        if item in {"", "?"}:
            values.append(float("nan"))
            continue
        values.append(float(item))
    return np.asarray(values, dtype=float)


def iter_windows(values: np.ndarray, spec: WindowSpec) -> list[tuple[int, np.ndarray]]:
    if spec.context_length <= 0 or spec.horizon <= 0:
        raise ValueError("context_length and horizon must be positive")
    if spec.stride <= 0:
        raise ValueError("stride must be positive")
    starts = window_starts(len(values), spec)
    return [(start, values[start : start + spec.length]) for start in starts]


def window_starts(value_count: int, spec: WindowSpec) -> list[int]:
    if value_count < spec.length:
        return []
    starts = list(range(0, value_count - spec.length + 1, spec.stride))
    if spec.max_windows is not None and len(starts) > spec.max_windows:
        indexes = np.linspace(0, len(starts) - 1, spec.max_windows).round().astype(int)
        starts = [starts[index] for index in sorted(set(indexes.tolist()))]
    return starts


def profile_csv(
    path: Path,
    *,
    time_column: str = "time",
    target_columns: list[str] | None = None,
    covariate_columns: list[str] | None = None,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    season_length: int | None = None,
    domain: str | None = None,
    dataset_name: str | None = None,
    target_features: list[str] | None = None,
    target_max_multiplier: float = 2.0,
    hierarchy: str | None = None,
) -> dict[str, Any]:
    time, targets, covariates = read_csv_series(path, time_column, target_columns, covariate_columns)
    resolved_stride = stride or horizon
    spec = WindowSpec(context_length, horizon, resolved_stride, max_windows=max_windows)
    values = targets.to_numpy(dtype=float)
    covariate_values = covariates.to_numpy(dtype=float) if not covariates.empty else None
    windows = iter_windows(values, spec)
    feature_rows: list[dict[str, float]] = []
    for start, window in windows:
        covariate_window = (
            covariate_values[start : start + spec.length]
            if covariate_values is not None
            else None
        )
        features = feature_vector(
            window,
            season_length=season_length,
            covariates=covariate_window,
            context_length=context_length,
            hierarchy=hierarchy,
        )
        features["window_start"] = float(start)
        feature_rows.append(features)

    feature_names = [name for name in DEFAULT_FEATURES if any(name in row for row in feature_rows)]
    quantiles = summarize_feature_rows(feature_rows, feature_names)
    caps = suggested_target_caps(
        quantiles,
        target_features=target_features or default_target_features(quantiles),
        multiplier=target_max_multiplier,
    )
    return {
        "schema_version": "synthetic_feature_profile.v1",
        "dataset": dataset_name or path.stem,
        "source_path": str(path),
        "bucket": {
            "domain": domain or "unknown",
            "frequency": infer_frequency_label(time),
            "context_length": context_length,
            "horizon": horizon,
            "target_dim": int(values.shape[1]) if values.ndim == 2 else 1,
            "covariate_dim": int(covariate_values.shape[1]) if covariate_values is not None else 0,
            "season_length": season_length,
            "hierarchy": hierarchy,
        },
        "window_count": len(feature_rows),
        "target_columns": list(targets.columns),
        "covariate_columns": list(covariates.columns),
        "features": quantiles,
        "target_feature_caps": caps,
    }


def profile_tsf(
    path: Path,
    *,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    season_length: int | None = None,
    domain: str | None = None,
    dataset_name: str | None = None,
    target_features: list[str] | None = None,
    target_max_multiplier: float = 2.0,
) -> dict[str, Any]:
    metadata, series = read_tsf_series(path)
    resolved_stride = stride or horizon
    spec = WindowSpec(context_length, horizon, resolved_stride)
    selected_windows = select_tsf_windows(series, spec, max_windows=max_windows)
    feature_rows: list[dict[str, float]] = []
    used_series_ids = {series_id for series_id, _, _ in selected_windows}
    for series_id, start, window in selected_windows:
        if not np.isfinite(window).all():
            continue
        features = feature_vector(window, season_length=season_length)
        features["series_index"] = float(series_id)
        features["window_start"] = float(start)
        feature_rows.append(features)

    feature_names = [name for name in DEFAULT_FEATURES if any(name in row for row in feature_rows)]
    quantiles = summarize_feature_rows(feature_rows, feature_names)
    caps = suggested_target_caps(
        quantiles,
        target_features=target_features or default_target_features(quantiles),
        multiplier=target_max_multiplier,
    )
    frequency = normalize_tsf_frequency(metadata.get("frequency", "unknown"))
    return {
        "schema_version": "synthetic_feature_profile.v1",
        "dataset": dataset_name or path.stem,
        "source_path": str(path),
        "bucket": {
            "domain": domain or "unknown",
            "frequency": frequency,
            "context_length": context_length,
            "horizon": horizon,
            "target_dim": 1,
            "covariate_dim": 0,
            "season_length": season_length,
        },
        "window_count": len(feature_rows),
        "candidate_window_count": len(selected_windows),
        "series_count": len(series),
        "used_series_count": len(used_series_ids),
        "target_columns": ["target"],
        "features": quantiles,
        "target_feature_caps": caps,
    }


def profile_tsf_panel(
    path: Path,
    *,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    season_length: int | None = None,
    target_dim: int = 3,
    domain: str | None = None,
    dataset_name: str | None = None,
    target_features: list[str] | None = None,
    target_max_multiplier: float = 2.0,
) -> dict[str, Any]:
    metadata, series = read_tsf_series(path)
    resolved_stride = stride or horizon
    spec = WindowSpec(context_length, horizon, resolved_stride)
    selected_windows = select_tsf_panel_windows(series, spec, target_dim=target_dim, max_windows=max_windows)
    feature_rows: list[dict[str, float]] = []
    used_series_ids: set[int] = set()
    for series_indexes, start, window in selected_windows:
        if not np.isfinite(window).all():
            continue
        features = feature_vector(window, season_length=season_length)
        features["channel_group_start"] = float(series_indexes[0])
        features["window_start"] = float(start)
        feature_rows.append(features)
        used_series_ids.update(series_indexes)

    feature_names = [name for name in DEFAULT_FEATURES if any(name in row for row in feature_rows)]
    quantiles = summarize_feature_rows(feature_rows, feature_names)
    caps = suggested_target_caps(
        quantiles,
        target_features=target_features or default_target_features(quantiles),
        multiplier=target_max_multiplier,
    )
    frequency = normalize_tsf_frequency(metadata.get("frequency", "unknown"))
    return {
        "schema_version": "synthetic_feature_profile.v1",
        "dataset": dataset_name or f"{path.stem}_panel",
        "source_path": str(path),
        "bucket": {
            "domain": domain or "unknown",
            "frequency": frequency,
            "context_length": context_length,
            "horizon": horizon,
            "target_dim": int(target_dim),
            "covariate_dim": 0,
            "season_length": season_length,
        },
        "window_count": len(feature_rows),
        "candidate_window_count": len(selected_windows),
        "series_count": len(series),
        "used_series_count": len(used_series_ids),
        "target_columns": [f"target_{index}" for index in range(target_dim)],
        "features": quantiles,
        "target_feature_caps": caps,
    }


def profile_m5_covariate(
    path: Path,
    *,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    max_series: int = 240,
    season_length: int | None = 7,
    domain: str | None = "retail",
    dataset_name: str | None = None,
    target_features: list[str] | None = None,
    target_max_multiplier: float = 2.0,
) -> dict[str, Any]:
    calendar, sales, day_columns = read_m5_calendar_and_sales(path)
    resolved_stride = stride or horizon
    spec = WindowSpec(context_length, horizon, resolved_stride)
    active_sales = sales.loc[sales[day_columns].sum(axis=1) > 0].reset_index(drop=True)
    if active_sales.empty:
        active_sales = sales.reset_index(drop=True)
    selected_sales = sample_frame_evenly(active_sales, max_series)
    prices = read_m5_prices(path, selected_sales[["store_id", "item_id"]].drop_duplicates())
    price_lookup = {
        (store_id, item_id): group.set_index("wm_yr_wk")["sell_price"].astype(float)
        for (store_id, item_id), group in prices.groupby(["store_id", "item_id"])
    }
    candidates = [
        (series_index, start)
        for series_index in range(len(selected_sales))
        for start in window_starts(len(day_columns), spec)
    ]
    candidates = limit_candidates(candidates, max_windows)
    feature_rows: list[dict[str, float]] = []
    used_series_ids: set[int] = set()
    for series_index, start in candidates:
        row = selected_sales.iloc[series_index]
        target = row[day_columns].to_numpy(dtype=float)
        covariates = m5_covariate_matrix(
            calendar,
            state_id=str(row["state_id"]),
            price_series=price_lookup.get((row["store_id"], row["item_id"])),
        )
        window = target[start : start + spec.length, None]
        covariate_window = covariates[start : start + spec.length]
        if window.shape[0] != spec.length or not np.isfinite(window).all() or not np.isfinite(covariate_window).all():
            continue
        features = feature_vector(
            window,
            season_length=season_length,
            covariates=covariate_window,
            context_length=context_length,
        )
        features["series_index"] = float(series_index)
        features["window_start"] = float(start)
        feature_rows.append(features)
        used_series_ids.add(series_index)

    feature_names = [name for name in DEFAULT_FEATURES if any(name in row for row in feature_rows)]
    quantiles = summarize_feature_rows(feature_rows, feature_names)
    caps = suggested_target_caps(
        quantiles,
        target_features=target_features
        or ["future_abs_covariate_target_corr", "avg_abs_covariate_target_corr", "event_lift_abs"],
        multiplier=target_max_multiplier,
    )
    return {
        "schema_version": "synthetic_feature_profile.v1",
        "dataset": dataset_name or "M5 daily covariate profile",
        "source_path": str(path),
        "bucket": {
            "domain": domain or "retail",
            "frequency": "d",
            "context_length": context_length,
            "horizon": horizon,
            "target_dim": 1,
            "covariate_dim": 4,
            "season_length": season_length,
        },
        "window_count": len(feature_rows),
        "candidate_window_count": len(candidates),
        "series_count": int(len(active_sales)),
        "used_series_count": len(used_series_ids),
        "target_columns": ["sales"],
        "covariate_columns": ["event_count", "snap", "sell_price", "price_change"],
        "features": quantiles,
        "target_feature_caps": caps,
    }


def profile_m5_hierarchy(
    path: Path,
    *,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    max_groups: int = 20,
    season_length: int | None = 7,
    domain: str | None = "retail",
    dataset_name: str | None = None,
    target_features: list[str] | None = None,
    target_max_multiplier: float = 2.0,
) -> dict[str, Any]:
    _calendar, sales, day_columns = read_m5_calendar_and_sales(path)
    resolved_stride = stride or horizon
    spec = WindowSpec(context_length, horizon, resolved_stride)
    group_sizes = sales.groupby(["store_id", "cat_id"])["dept_id"].nunique()
    groups = [group for group, count in group_sizes.items() if count == 2]
    groups = sample_sequence_evenly(groups, max_groups)
    group_values = [m5_hierarchy_values(sales, day_columns, group) for group in groups]
    candidates = [
        (group_index, start)
        for group_index, values in enumerate(group_values)
        for start in window_starts(values.shape[0], spec)
    ]
    candidates = limit_candidates(candidates, max_windows)
    feature_rows: list[dict[str, float]] = []
    used_group_ids: set[int] = set()
    for group_index, start in candidates:
        values = group_values[group_index]
        window = values[start : start + spec.length]
        if window.shape[0] != spec.length or not np.isfinite(window).all():
            continue
        features = feature_vector(window, season_length=season_length, hierarchy="additive_first")
        features["group_index"] = float(group_index)
        features["window_start"] = float(start)
        feature_rows.append(features)
        used_group_ids.add(group_index)

    feature_names = [name for name in DEFAULT_FEATURES if any(name in row for row in feature_rows)]
    quantiles = summarize_feature_rows(feature_rows, feature_names)
    caps = suggested_target_caps(
        quantiles,
        target_features=target_features or ["hierarchy_residual_mean_abs", "avg_abs_target_corr"],
        multiplier=target_max_multiplier,
    )
    return {
        "schema_version": "synthetic_feature_profile.v1",
        "dataset": dataset_name or "M5 daily hierarchy profile",
        "source_path": str(path),
        "bucket": {
            "domain": domain or "retail",
            "frequency": "d",
            "context_length": context_length,
            "horizon": horizon,
            "target_dim": 3,
            "covariate_dim": 0,
            "season_length": season_length,
            "hierarchy": "additive_first",
        },
        "window_count": len(feature_rows),
        "candidate_window_count": len(candidates),
        "series_count": len(groups),
        "used_series_count": len(used_group_ids),
        "target_columns": ["parent", "child_0", "child_1"],
        "features": quantiles,
        "target_feature_caps": caps,
    }


def profile_gefcom2014_load(
    path: Path,
    *,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    season_length: int | None = 24,
    task: int = 1,
    domain: str | None = "energy",
    dataset_name: str | None = None,
    target_features: list[str] | None = None,
    target_max_multiplier: float = 2.0,
) -> dict[str, Any]:
    frame, source_name = read_gefcom2014_load_frame(path, task=task)
    covariate_columns = [column for column in frame.columns if column.startswith("w")]
    frame = frame[["LOAD", *covariate_columns]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    resolved_stride = stride or horizon
    spec = WindowSpec(context_length, horizon, resolved_stride, max_windows=max_windows)
    target = frame["LOAD"].to_numpy(dtype=float)
    covariates = frame[covariate_columns].to_numpy(dtype=float)
    starts = window_starts(len(target), spec)
    feature_rows: list[dict[str, float]] = []
    for start in starts:
        window = target[start : start + spec.length, None]
        covariate_window = covariates[start : start + spec.length]
        if not np.isfinite(window).all() or not np.isfinite(covariate_window).all():
            continue
        features = feature_vector(
            window,
            season_length=season_length,
            covariates=covariate_window,
            context_length=context_length,
        )
        features["window_start"] = float(start)
        feature_rows.append(features)

    feature_names = [name for name in DEFAULT_FEATURES if any(name in row for row in feature_rows)]
    quantiles = summarize_feature_rows(feature_rows, feature_names)
    caps = suggested_target_caps(
        quantiles,
        target_features=target_features or ["future_abs_covariate_target_corr", "avg_abs_covariate_target_corr"],
        multiplier=target_max_multiplier,
    )
    return {
        "schema_version": "synthetic_feature_profile.v1",
        "dataset": dataset_name or f"GEFCom2014 Load task {task}",
        "source_path": f"{path}:{source_name}",
        "bucket": {
            "domain": domain or "energy",
            "frequency": "h",
            "context_length": context_length,
            "horizon": horizon,
            "target_dim": 1,
            "covariate_dim": len(covariate_columns),
            "season_length": season_length,
        },
        "window_count": len(feature_rows),
        "candidate_window_count": len(starts),
        "series_count": 1,
        "used_series_count": 1 if feature_rows else 0,
        "target_columns": ["LOAD"],
        "covariate_columns": covariate_columns,
        "features": quantiles,
        "target_feature_caps": caps,
    }


def select_tsf_windows(
    series: list[tuple[str, np.ndarray]],
    spec: WindowSpec,
    *,
    max_windows: int | None,
) -> list[tuple[int, int, np.ndarray]]:
    candidates: list[tuple[int, int]] = []
    for series_index, (_series_id, values) in enumerate(series):
        starts = window_starts(len(values), spec)
        candidates.extend((series_index, start) for start in starts)
    if max_windows is not None and len(candidates) > max_windows:
        indexes = np.linspace(0, len(candidates) - 1, max_windows).round().astype(int)
        candidates = [candidates[index] for index in sorted(set(indexes.tolist()))]
    selected: list[tuple[int, int, np.ndarray]] = []
    for series_index, start in candidates:
        values = series[series_index][1]
        window = values[start : start + spec.length, None]
        selected.append((series_index, start, window))
    return selected


def select_tsf_panel_windows(
    series: list[tuple[str, np.ndarray]],
    spec: WindowSpec,
    *,
    target_dim: int,
    max_windows: int | None,
) -> list[tuple[tuple[int, ...], int, np.ndarray]]:
    if target_dim <= 1:
        raise ValueError("target_dim must be greater than 1 for panel profiling")
    usable = [index for index, (_series_id, values) in enumerate(series) if len(values) >= spec.length]
    if len(usable) < target_dim:
        return []
    min_len = min(len(series[index][1]) for index in usable)
    starts = window_starts(min_len, WindowSpec(spec.context_length, spec.horizon, spec.stride))
    channel_groups = [
        tuple(usable[start : start + target_dim])
        for start in range(0, len(usable) - target_dim + 1, target_dim)
    ]
    candidates = [(group, start) for group in channel_groups for start in starts]
    if max_windows is not None and len(candidates) > max_windows:
        indexes = np.linspace(0, len(candidates) - 1, max_windows).round().astype(int)
        candidates = [candidates[index] for index in sorted(set(indexes.tolist()))]

    selected: list[tuple[tuple[int, ...], int, np.ndarray]] = []
    for group, start in candidates:
        window = np.column_stack(
            [
                series[series_index][1][start : start + spec.length]
                for series_index in group
            ]
        )
        selected.append((group, start, window))
    return selected


def read_m5_calendar_and_sales(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    with zipfile.ZipFile(path) as archive:
        sales_name = first_existing_name(archive, ("sales_train_validation.csv", "sales_train_evaluation.csv"))
        with archive.open("calendar.csv") as handle:
            calendar = pd.read_csv(handle)
        with archive.open(sales_name) as handle:
            sales = pd.read_csv(handle)
    day_columns = sorted(
        [column for column in sales.columns if column.startswith("d_")],
        key=lambda value: int(value.split("_", 1)[1]),
    )
    calendar = calendar.loc[calendar["d"].isin(day_columns)].copy()
    day_order = {day: index for index, day in enumerate(day_columns)}
    calendar["_day_order"] = calendar["d"].map(day_order)
    calendar = calendar.sort_values("_day_order").drop(columns=["_day_order"]).reset_index(drop=True)
    return calendar, sales, day_columns


def read_m5_prices(path: Path, selected_keys: pd.DataFrame) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open("sell_prices.csv") as handle:
            prices = pd.read_csv(handle, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    if selected_keys.empty:
        return prices.iloc[0:0].copy()
    return prices.merge(selected_keys, on=["store_id", "item_id"], how="inner")


def first_existing_name(archive: zipfile.ZipFile, names: tuple[str, ...]) -> str:
    available = set(archive.namelist())
    for name in names:
        if name in available:
            return name
    raise ValueError(f"none of the expected files are present: {', '.join(names)}")


def m5_covariate_matrix(
    calendar: pd.DataFrame,
    *,
    state_id: str,
    price_series: pd.Series | None,
) -> np.ndarray:
    event_count = (
        calendar[[column for column in ("event_name_1", "event_name_2") if column in calendar.columns]]
        .notna()
        .sum(axis=1)
        .to_numpy(dtype=float)
    )
    snap_column = f"snap_{state_id}"
    snap = calendar[snap_column].to_numpy(dtype=float) if snap_column in calendar.columns else np.zeros(len(calendar))
    if price_series is None or price_series.empty:
        price = pd.Series(np.zeros(len(calendar), dtype=float))
    else:
        mapped = [float(price_series.get(week, np.nan)) for week in calendar["wm_yr_wk"].tolist()]
        price = pd.Series(mapped, dtype=float).ffill().bfill()
        if price.isna().any():
            fill_value = float(price.dropna().median()) if price.notna().any() else 0.0
            price = price.fillna(fill_value)
    price_change = price.diff().fillna(0.0)
    return np.column_stack(
        [
            event_count,
            snap,
            price.to_numpy(dtype=float),
            price_change.to_numpy(dtype=float),
        ]
    )


def m5_hierarchy_values(sales: pd.DataFrame, day_columns: list[str], group: tuple[str, str]) -> np.ndarray:
    store_id, cat_id = group
    group_rows = sales.loc[(sales["store_id"] == store_id) & (sales["cat_id"] == cat_id)]
    dept_values = [
        group_rows.loc[group_rows["dept_id"] == dept_id, day_columns].sum(axis=0).to_numpy(dtype=float)
        for dept_id in sorted(group_rows["dept_id"].unique())
    ]
    if len(dept_values) != 2:
        raise ValueError(f"M5 hierarchy profile expects exactly two child departments for {group!r}")
    parent = dept_values[0] + dept_values[1]
    return np.column_stack([parent, *dept_values])


def read_gefcom2014_load_frame(path: Path, *, task: int) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(path) as outer:
        names = outer.namelist()
        if any(name.startswith("Load/") and name.endswith("-train.csv") for name in names):
            return read_gefcom2014_load_frame_from_archive(outer, task=task)
        nested_name = first_matching_name(names, suffix="GEFCom2014-L_V2.zip")
        nested_data = outer.read(nested_name)
    with zipfile.ZipFile(io.BytesIO(nested_data)) as nested:
        frame, source_name = read_gefcom2014_load_frame_from_archive(nested, task=task)
    return frame, f"{nested_name}:{source_name}"


def read_gefcom2014_load_frame_from_archive(archive: zipfile.ZipFile, *, task: int) -> tuple[pd.DataFrame, str]:
    preferred = f"Load/Task {task}/L{task}-train.csv"
    names = archive.namelist()
    source_name = preferred if preferred in names else first_matching_name(names, prefix="Load/Task ", suffix="-train.csv")
    with archive.open(source_name) as handle:
        return pd.read_csv(handle), source_name


def first_matching_name(names: list[str], *, prefix: str | None = None, suffix: str | None = None) -> str:
    for name in names:
        if prefix is not None and not name.startswith(prefix):
            continue
        if suffix is not None and not name.endswith(suffix):
            continue
        return name
    criteria = ", ".join(part for part in [f"prefix={prefix!r}" if prefix else "", f"suffix={suffix!r}" if suffix else ""] if part)
    raise ValueError(f"no zip member matched {criteria}")


def sample_frame_evenly(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    indexes = np.linspace(0, len(frame) - 1, max_rows).round().astype(int)
    return frame.iloc[sorted(set(indexes.tolist()))].reset_index(drop=True)


def sample_sequence_evenly(values: list[Any], max_items: int) -> list[Any]:
    if max_items <= 0 or len(values) <= max_items:
        return list(values)
    indexes = np.linspace(0, len(values) - 1, max_items).round().astype(int)
    return [values[index] for index in sorted(set(indexes.tolist()))]


def limit_candidates(candidates: list[Any], max_items: int | None) -> list[Any]:
    if max_items is None or len(candidates) <= max_items:
        return candidates
    indexes = np.linspace(0, len(candidates) - 1, max_items).round().astype(int)
    return [candidates[index] for index in sorted(set(indexes.tolist()))]


def profile_input(
    path: Path,
    *,
    input_format: str = "auto",
    time_column: str = "time",
    target_columns: list[str] | None = None,
    covariate_columns: list[str] | None = None,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    season_length: int | None = None,
    domain: str | None = None,
    dataset_name: str | None = None,
    target_features: list[str] | None = None,
    target_max_multiplier: float = 2.0,
    hierarchy: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_input_format(path, input_format)
    if resolved == "tsf":
        return profile_tsf(
            path,
            context_length=context_length,
            horizon=horizon,
            stride=stride,
            max_windows=max_windows,
            season_length=season_length,
            domain=domain,
            dataset_name=dataset_name,
            target_features=target_features,
            target_max_multiplier=target_max_multiplier,
        )
    return profile_csv(
        path,
        time_column=time_column,
        target_columns=target_columns,
        covariate_columns=covariate_columns,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        max_windows=max_windows,
        season_length=season_length,
        domain=domain,
        dataset_name=dataset_name,
        target_features=target_features,
        target_max_multiplier=target_max_multiplier,
        hierarchy=hierarchy,
    )


def resolve_input_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    suffix = path.suffix.lower()
    if suffix == ".tsf":
        return "tsf"
    if suffix == ".zip":
        return "tsf"
    return "csv"


def feature_vector(
    window: np.ndarray,
    season_length: int | None = None,
    *,
    covariates: np.ndarray | None = None,
    context_length: int | None = None,
    hierarchy: str | None = None,
) -> dict[str, float]:
    if window.ndim == 1:
        window = window[:, None]
    per_target = [single_series_features(window[:, index], season_length=season_length) for index in range(window.shape[1])]
    out: dict[str, float] = {}
    for feature in DEFAULT_FEATURES:
        values = [row[feature] for row in per_target if feature in row and math.isfinite(row[feature])]
        if values:
            out[feature] = float(np.mean(values))
    out.update(structural_univariate_features(np.mean(window, axis=1), season_length=season_length))
    if window.shape[1] > 1:
        out.update(multitarget_features(window))
    if hierarchy == "additive_first" and window.shape[1] > 2:
        hierarchy_residual = window[:, 0] - np.sum(window[:, 1:], axis=1)
        out["hierarchy_residual_mean_abs"] = float(np.mean(np.abs(hierarchy_residual)))
    if covariates is not None and covariates.size:
        out.update(covariate_features(window, covariates, context_length or max(1, window.shape[0] // 2)))
    return out


def multitarget_features(window: np.ndarray) -> dict[str, float]:
    centered = window - np.mean(window, axis=0, keepdims=True)
    corr_values = [
        abs(safe_corr(window[:, left], window[:, right]))
        for left in range(window.shape[1])
        for right in range(window.shape[1])
        if left != right
    ]
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular**2
    total = float(np.sum(variance))
    explained = variance / total if total > 1e-12 else np.zeros_like(variance)
    entropy = -float(np.sum([value * np.log(value) for value in explained if value > 1e-12]))
    return {
        "avg_abs_target_corr": float(np.mean(corr_values)) if corr_values else 0.0,
        "pca_top1_explained": float(explained[0]) if explained.size else 0.0,
        "pca_top2_explained": float(np.sum(explained[:2])) if explained.size else 0.0,
        "effective_factor_rank": float(np.exp(entropy)) if explained.size else 0.0,
        "lead_lag_peak_abs": lead_lag_peak_abs(window),
    }


def lead_lag_peak_abs(window: np.ndarray, max_lag: int = 12) -> float:
    if window.shape[1] < 2:
        return 0.0
    peaks: list[float] = []
    lag_limit = min(max_lag, max(1, window.shape[0] // 4))
    for left in range(window.shape[1]):
        for right in range(window.shape[1]):
            if left == right:
                continue
            for lag in range(1, lag_limit + 1):
                peaks.append(abs(safe_corr(window[:-lag, left], window[lag:, right])))
    finite = [value for value in peaks if math.isfinite(value)]
    return float(max(finite)) if finite else 0.0


def single_series_features(values: np.ndarray, season_length: int | None = None) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return {}
    scaled = robust_scale(y)
    trend, seasonal, residual = decompose_for_features(scaled, season_length=season_length)
    total_var = safe_var(scaled)
    trend_resid_var = safe_var(trend + residual)
    seasonal_resid_var = safe_var(seasonal + residual)
    return {
        "trend_strength": strength(residual, trend + residual),
        "seasonal_strength": strength(residual, seasonal + residual),
        "slope_abs": abs(polyfit_coeff(scaled, degree=2, coeff_index=1)),
        "curvature_abs": abs(polyfit_coeff(scaled, degree=2, coeff_index=0)),
        "noise_ratio": clamp01(safe_var(residual) / total_var) if total_var > 0 else 0.0,
        "acf1": autocorrelation(scaled, 1),
        "acf_abs_mean": mean_abs_autocorrelation(scaled, max_lag=min(10, max(1, y.size // 4))),
        "outlier_rate": outlier_rate(scaled),
        "spike_rate": spike_rate(scaled),
        "trend_resid_var": trend_resid_var,
        "seasonal_resid_var": seasonal_resid_var,
    }


def structural_univariate_features(values: np.ndarray, season_length: int | None = None) -> dict[str, float]:
    y = robust_scale(np.asarray(values, dtype=float))
    y = y[np.isfinite(y)]
    n = y.size
    if n < 12:
        return {}
    min_seg = max(6, min(24, n // 8))
    level_scores: list[float] = []
    volatility_scores: list[float] = []
    std_all = float(np.std(y)) or 1.0
    for cut in range(min_seg, n - min_seg):
        left = y[:cut]
        right = y[cut:]
        level_scores.append(abs(float(np.mean(left) - np.mean(right))) / std_all)
        volatility_scores.append(abs(float(np.std(left) - np.std(right))) / std_all)
    seasonal_profile = phase_profile(y, season_length)
    half = max(1, n // 2)
    seasonal_left = phase_profile(y[:half], season_length)
    seasonal_right = phase_profile(y[half:], season_length)
    diff = np.diff(y)
    return {
        "level_shift_strength": float(max(level_scores)) if level_scores else 0.0,
        "volatility_shift_strength": float(max(volatility_scores)) if volatility_scores else 0.0,
        "change_point_shift_energy": float(np.mean(sorted(level_scores, reverse=True)[:3])) if level_scores else 0.0,
        "burst_rate": float(np.mean(np.abs(y) > 3.0)),
        "diff_spike_rate": float(np.mean(np.abs(robust_scale(diff)) > 3.0)) if diff.size else 0.0,
        "multi_period_score": multi_period_score(y, season_length),
        "seasonal_drift_score": float(np.mean(np.abs(seasonal_left - seasonal_right))) if seasonal_left.size and seasonal_right.size else 0.0,
        "seasonal_amplitude_cv": float(np.std(np.abs(seasonal_profile)) / (np.mean(np.abs(seasonal_profile)) + 1e-9)) if seasonal_profile.size else 0.0,
        "nonlinear_lag1_gain": nonlinear_lag1_gain(y),
    }


def covariate_features(target: np.ndarray, covariates: np.ndarray, context_length: int) -> dict[str, float]:
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    scores: list[float] = []
    future_scores: list[float] = []
    for cov_idx in range(covariates.shape[1]):
        for target_idx in range(target.shape[1]):
            corr_value = safe_corr(covariates[:, cov_idx], target[:, target_idx])
            if math.isfinite(corr_value):
                scores.append(abs(float(corr_value)))
            if context_length < len(target):
                future_corr = safe_corr(covariates[context_length:, cov_idx], target[context_length:, target_idx])
                if math.isfinite(future_corr):
                    future_scores.append(abs(float(future_corr)))
    event_lifts: list[float] = []
    for cov_idx in range(covariates.shape[1]):
        column = covariates[:, cov_idx]
        unique = np.unique(column)
        if unique.size <= 3 and np.any(column > 0):
            active = column > 0
            inactive = ~active
            if active.any() and inactive.any():
                event_lifts.append(abs(float(np.mean(target[active]) - np.mean(target[inactive]))))
    return {
        "avg_abs_covariate_target_corr": float(np.mean(scores)) if scores else 0.0,
        "future_abs_covariate_target_corr": float(np.mean(future_scores)) if future_scores else 0.0,
        "event_lift_abs": float(np.mean(event_lifts)) if event_lifts else 0.0,
    }


def decompose_for_features(values: np.ndarray, season_length: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float)
    trend = polynomial_trend(y)
    detrended = y - trend
    seasonal = seasonal_by_phase(detrended, season_length=season_length)
    residual = y - trend - seasonal
    return trend, seasonal, residual


def polynomial_trend(values: np.ndarray) -> np.ndarray:
    if values.size < 4:
        return np.full_like(values, float(np.mean(values)))
    t = np.linspace(-1.0, 1.0, values.size)
    degree = min(2, values.size - 1)
    coeffs = np.polyfit(t, values, degree)
    return np.polyval(coeffs, t)


def seasonal_by_phase(values: np.ndarray, season_length: int | None = None) -> np.ndarray:
    if not season_length or season_length < 2 or values.size < season_length * 2:
        return np.zeros_like(values)
    period = int(season_length)
    seasonal = np.zeros_like(values)
    for phase in range(period):
        mask = np.arange(values.size) % period == phase
        if mask.any():
            seasonal[mask] = float(np.mean(values[mask]))
    return seasonal - float(np.mean(seasonal))


def phase_profile(values: np.ndarray, season_length: int | None = None) -> np.ndarray:
    if not season_length or season_length < 2 or values.size < season_length:
        return np.asarray([], dtype=float)
    period = int(season_length)
    phases = np.arange(values.size) % period
    profile = np.asarray(
        [
            float(np.mean(values[phases == phase])) if np.any(phases == phase) else 0.0
            for phase in range(period)
        ],
        dtype=float,
    )
    return profile - float(np.mean(profile))


def multi_period_score(values: np.ndarray, season_length: int | None = None) -> float:
    if values.size < 8:
        return 0.0
    centered = values - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if spectrum.size <= 2:
        return 0.0
    spectrum[0] = 0.0
    total = float(np.sum(spectrum))
    if total <= 1e-12:
        return 0.0
    primary_index = int(round(values.size / max(2, season_length or 2)))
    exclude = {idx for idx in range(max(1, primary_index - 1), min(spectrum.size, primary_index + 2))}
    secondary = np.asarray([value for idx, value in enumerate(spectrum) if idx not in exclude and idx > 0], dtype=float)
    return float(np.max(secondary) / total) if secondary.size else 0.0


def nonlinear_lag1_gain(values: np.ndarray) -> float:
    if values.size < 8:
        return 0.0
    x = values[:-1]
    y = values[1:]
    linear = np.column_stack([np.ones_like(x), x])
    nonlinear = np.column_stack([np.ones_like(x), x, x**2, np.sin(x)])
    return max(0.0, r2(y, nonlinear) - r2(y, linear))


def r2(y: np.ndarray, design: np.ndarray) -> float:
    try:
        coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    fitted = design @ coeffs
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y - fitted) ** 2) / denom)


def robust_scale(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-9:
        return (values - median) / iqr
    std = float(np.std(values))
    if std > 1e-9:
        return (values - median) / std
    return values - median


def strength(residual: np.ndarray, residual_plus_component: np.ndarray) -> float:
    denom = safe_var(residual_plus_component)
    if denom <= 0:
        return 0.0
    return clamp01(1.0 - safe_var(residual) / denom)


def safe_var(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    value = float(np.var(values))
    return value if math.isfinite(value) else 0.0


def polyfit_coeff(values: np.ndarray, *, degree: int, coeff_index: int) -> float:
    if values.size <= degree:
        return 0.0
    t = np.linspace(-1.0, 1.0, values.size)
    coeffs = np.polyfit(t, values, degree)
    return float(coeffs[coeff_index])


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or values.size <= lag:
        return 0.0
    a = values[:-lag] - float(np.mean(values[:-lag]))
    b = values[lag:] - float(np.mean(values[lag:]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_abs_autocorrelation(values: np.ndarray, max_lag: int) -> float:
    if max_lag <= 0:
        return 0.0
    vals = [abs(autocorrelation(values, lag)) for lag in range(1, max_lag + 1)]
    return float(np.mean(vals)) if vals else 0.0


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size != right.size or left.size < 3:
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def outlier_rate(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-9:
        return 0.0
    robust_z = 0.6745 * np.abs(values - median) / mad
    return float(np.mean(robust_z > 4.0))


def spike_rate(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    diff = np.diff(values)
    median = float(np.median(diff))
    mad = float(np.median(np.abs(diff - median)))
    if mad <= 1e-9:
        return 0.0
    robust_z = 0.6745 * np.abs(diff - median) / mad
    return float(np.mean(robust_z > 4.0))


def summarize_feature_rows(rows: list[dict[str, float]], feature_names: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for feature in feature_names:
        values = np.asarray([row[feature] for row in rows if feature in row and math.isfinite(row[feature])], dtype=float)
        if values.size == 0:
            continue
        feature_summary = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for quantile in DEFAULT_QUANTILES:
            feature_summary[f"p{int(quantile * 100):02d}"] = float(np.quantile(values, quantile))
        summary[feature] = feature_summary
    return summary


def default_target_features(quantiles: dict[str, dict[str, float]]) -> list[str]:
    preferred = [
        "trend_strength",
        "seasonal_strength",
        "slope_abs",
        "curvature_abs",
        "pca_top1_explained",
        "avg_abs_target_corr",
    ]
    return [feature for feature in preferred if feature in quantiles]


def suggested_target_caps(
    quantiles: dict[str, dict[str, float]],
    *,
    target_features: list[str],
    multiplier: float,
) -> dict[str, dict[str, float]]:
    if multiplier <= 0:
        raise ValueError("target_max_multiplier must be positive")
    caps: dict[str, dict[str, float]] = {}
    for feature in target_features:
        stats = quantiles.get(feature)
        if not stats:
            continue
        anchor = float(stats.get("p95", stats.get("max", 0.0)))
        fallback = float(stats.get("max", anchor))
        basis = anchor if anchor > 0 else fallback
        max_allowed = basis * multiplier
        if feature in BOUNDED_FEATURES:
            max_allowed = min(max_allowed, 1.0)
        caps[feature] = {
            "basis_quantile": 0.95,
            "basis_value": basis,
            "multiplier": float(multiplier),
            "max_allowed": float(max_allowed),
        }
    return caps


def infer_frequency_label(time: pd.Series) -> str:
    if len(time) < 2:
        return "unknown"
    diffs = time.sort_values().diff().dropna()
    if diffs.empty:
        return "unknown"
    seconds = int(diffs.median().total_seconds())
    if seconds % 86400 == 0:
        days = seconds // 86400
        return "d" if days == 1 else f"{days}d"
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return "h" if hours == 1 else f"{hours}h"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return "min" if minutes == 1 else f"{minutes}min"
    return f"{seconds}s"


def normalize_tsf_frequency(value: str) -> str:
    normalized = (value or "unknown").strip().lower()
    aliases = {
        "hourly": "h",
        "daily": "d",
        "weekly": "w",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
        "annual": "Y",
    }
    return aliases.get(normalized, normalized or "unknown")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile real time-series feature distributions for synthetic v2 calibration.")
    parser.add_argument("input", type=Path, help="Input CSV, TSF, or TSF zip path.")
    parser.add_argument("--format", choices=["auto", "csv", "tsf"], default="auto", help="Input format. Defaults to extension-based auto detection.")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--target-column", action="append", dest="target_columns", help="Target column. Repeat for multiple targets.")
    parser.add_argument("--covariate-column", action="append", dest="covariate_columns", help="Known covariate column. Repeat for multiple covariates.")
    parser.add_argument("--hierarchy", choices=["additive_first"], help="Optional hierarchy convention. additive_first means target_0=sum(target_1:).")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--season-length", type=int)
    parser.add_argument("--domain")
    parser.add_argument("--dataset-name")
    parser.add_argument("--target-feature", action="append", dest="target_features")
    parser.add_argument("--target-max-multiplier", type=float, default=2.0)
    parser.add_argument("--out", type=Path, help="Output JSON path. Defaults to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = profile_input(
        args.input,
        input_format=args.format,
        time_column=args.time_column,
        target_columns=args.target_columns,
        covariate_columns=args.covariate_columns,
        context_length=args.context_length,
        horizon=args.horizon,
        stride=args.stride,
        max_windows=args.max_windows,
        season_length=args.season_length,
        domain=args.domain,
        dataset_name=args.dataset_name,
        target_features=args.target_features,
        target_max_multiplier=args.target_max_multiplier,
        hierarchy=args.hierarchy,
    )
    text = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
