#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc


DEFAULT_FEATURES = (
    "trend_strength",
    "seasonal_strength",
    "slope_abs",
    "curvature_abs",
    "noise_ratio",
    "acf1",
    "acf_abs_mean",
    "seasonal_acf",
    "dominant_period",
    "spectral_concentration",
    "outlier_rate",
    "spike_rate",
    "multi_period_score",
    "seasonal_drift_score",
    "seasonal_amplitude_cv",
    "seasonal_amplitude_modulation",
    "seasonal_phase_variation",
    "change_point_shift_energy",
    "level_shift_strength",
    "regime_sparse_transition_score",
    "volatility_shift_strength",
    "nonlinear_lag1_gain",
    "nonlinear_multi_lag_gain",
    "nonlinear_conditional_gain",
    "burst_rate",
    "diff_spike_rate",
    "intermittency_clock_incremental_r2",
    "avg_abs_target_corr",
    "pca_top1_explained",
    "pca_top2_explained",
    "effective_factor_rank",
    "lead_lag_peak_abs",
    "lead_lag_peak_lag_abs",
    "cross_series_incremental_r2",
    "avg_abs_covariate_target_corr",
    "future_abs_covariate_target_corr",
    "event_lift_abs",
    "covariate_incremental_r2",
    "covariate_residual_acf_abs_mean",
    "covariate_residual_outlier_rate",
    "covariate_residual_spike_rate",
    "hierarchy_residual_mean_abs",
    "hierarchy_child_heterogeneity",
    "hierarchy_aggregation_ratio",
    "hierarchy_aggregate_acf1",
    "hierarchy_contrast_acf1",
    "hierarchy_aggregate_seasonal_acf",
    "hierarchy_contrast_seasonal_acf",
    "hierarchy_contrast_to_aggregate_std_ratio",
    "hierarchy_aggregate_contrast_abs_corr",
    "factor_score_acf1",
    "factor_residual_acf1",
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
    "nonlinear_multi_lag_gain",
    "nonlinear_conditional_gain",
    "avg_abs_covariate_target_corr",
    "future_abs_covariate_target_corr",
}
M5_KNOWN_FUTURE_COVARIATES = (
    "day_of_week_sin",
    "day_of_week_cos",
    "event_count",
    "snap",
)
M5_COVARIATE_PROVENANCE = (
    "official calendar.csv fields fixed before the sales forecast issue time; "
    "sell_price and derived price_change are excluded"
)
GEFCOM2014_WIND_NWP_COLUMNS = ("U10", "V10", "U100", "V100")
GEFCOM2014_WIND_COVARIATE_PROVENANCE = (
    "official TaskExpVars competition release joined to subsequently observed "
    "targets by ZONEID and TIMESTAMP; forecasts are never stitched across releases"
)
PAPER_V7_SWISS_FACTOR_METER_INDICES = (0, 6, 12)
PAPER_V7_GEFCOM2012_FACTOR_ZONE_INDICES = (0, 9, 19)
PAPER_V7_SWISS_NWP_COLUMNS = (
    "ghi_backwards",
    "gni_backwards",
    "relativehumidity",
    "windspeed",
    "winddirection",
    "temperature",
)
PAPER_V7_GEFCOM2012_CALENDAR_COLUMNS = (
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "is_weekend",
    "is_holiday",
)
PAPER_V7_SWISS_COVARIATE_PROVENANCE = (
    "official Meteoblue 24-hour forecast cube; benchmark H48 repeats the "
    "single latest vintage available at the last history origin to 30-minute "
    "resolution and never stitches a later release"
)
PAPER_V7_GEFCOM2012_COVARIATE_PROVENANCE = (
    "deterministic hour/day-of-week/weekend/official-holiday calendar fields; "
    "observed temperature is excluded because it has no rolling issue-time vintage"
)


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_paper_v7_processed_npz(
    path: Path,
    *,
    expected_dataset_id: str,
    required_arrays: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read one provenance-checked numeric-only Paper v7 processed asset."""

    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"processed metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_id") != expected_dataset_id:
        raise ValueError(
            f"processed dataset id mismatch: expected {expected_dataset_id!r}, "
            f"got {metadata.get('dataset_id')!r}"
        )
    actual_hash = file_sha256(path)
    expected_hash = str((metadata.get("output_npz") or {}).get("sha256") or "")
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"processed NPZ hash mismatch for {path}: "
            f"expected={expected_hash!r}, actual={actual_hash!r}"
        )
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required_arrays) - set(archive.files))
        if missing:
            raise ValueError(
                f"processed NPZ {path} is missing arrays: {', '.join(missing)}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in required_arrays}
    row_counts = {array.shape[0] for array in arrays.values() if array.ndim > 0}
    if len(row_counts) != 1:
        raise ValueError(f"processed NPZ arrays have inconsistent rows: {row_counts}")
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"processed NPZ array {name!r} has object dtype")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"processed NPZ array {name!r} is non-finite")
    timestamps = arrays.get("timestamps")
    if timestamps is None or timestamps.dtype.kind != "M":
        raise ValueError("processed NPZ timestamps must be datetime64")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= np.timedelta64(0, "ns")):
        raise ValueError("processed NPZ timestamps must be strictly increasing")
    metadata["_processed_npz_sha256"] = actual_hash
    metadata["_processed_metadata_sha256"] = file_sha256(metadata_path)
    metadata["_processed_metadata_path"] = str(metadata_path)
    return arrays, metadata


def read_paper_v7_swiss_processed(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, metadata = read_paper_v7_processed_npz(
        path,
        expected_dataset_id="swiss_hierarchical_demand",
        required_arrays=(
            "timestamps",
            "meters",
            "aggregates",
            "nwp_asof_timestamps",
            "nwp_valid_timestamps",
            "nwp_forecasts",
        ),
    )
    if arrays["meters"].shape[1:] != (24,):
        raise ValueError("Swiss processed meters must have shape [time, 24]")
    if arrays["aggregates"].shape[1:] != (7,):
        raise ValueError("Swiss processed aggregates must have shape [time, 7]")
    if arrays["nwp_forecasts"].shape[1:] != (6, 24):
        raise ValueError(
            "Swiss processed NWP forecasts must have shape [time, 6, 24]"
        )
    if arrays["nwp_asof_timestamps"].dtype.kind != "M":
        raise ValueError("Swiss NWP as-of timestamps must be datetime64")
    valid = arrays["nwp_valid_timestamps"]
    if valid.shape[1:] != (24,) or valid.dtype.kind != "M":
        raise ValueError(
            "Swiss NWP valid timestamps must have datetime64 shape [time, 24]"
        )
    if not np.array_equal(valid[:, 0], arrays["nwp_asof_timestamps"]):
        raise ValueError("Swiss NWP lead-0 valid time must equal its as-of time")
    if valid.shape[1] > 1 and not np.all(
        np.diff(valid, axis=1) == np.timedelta64(1, "h")
    ):
        raise ValueError("Swiss NWP valid-time leads must increase hourly")
    if list(metadata.get("nwp_variable_order") or ()) != list(
        PAPER_V7_SWISS_NWP_COLUMNS
    ):
        raise ValueError("Swiss processed NWP variable order changed")
    return arrays, metadata


def read_paper_v7_gefcom2012_processed(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, metadata = read_paper_v7_processed_npz(
        path,
        expected_dataset_id="gefcom2012_load",
        required_arrays=(
            "timestamps",
            "segment_ids",
            "zones",
            "total",
            "canonical_hierarchy",
            "calendar_covariates",
        ),
    )
    if arrays["zones"].shape[1:] != (20,):
        raise ValueError("GEFCom2012 processed zones must have shape [time, 20]")
    if arrays["canonical_hierarchy"].shape[1:] != (3,):
        raise ValueError(
            "GEFCom2012 canonical hierarchy must have shape [time, 3]"
        )
    if arrays["calendar_covariates"].shape[1:] != (6,):
        raise ValueError(
            "GEFCom2012 calendar covariates must have shape [time, 6]"
        )
    segment_ids = arrays["segment_ids"]
    if segment_ids.ndim != 1 or segment_ids.dtype.kind not in "iu":
        raise ValueError("GEFCom2012 segment_ids must be a numeric integer vector")
    unique_segments = np.unique(segment_ids)
    if not np.array_equal(
        unique_segments,
        np.arange(len(unique_segments), dtype=unique_segments.dtype),
    ) or np.any(np.diff(segment_ids) < 0):
        raise ValueError("GEFCom2012 segment_ids must be ordered and contiguous")
    timestamps = arrays["timestamps"]
    for segment_id in unique_segments:
        positions = np.flatnonzero(segment_ids == segment_id)
        if positions.size == 0 or not np.array_equal(
            positions,
            np.arange(positions[0], positions[-1] + 1),
        ):
            raise ValueError("GEFCom2012 segment rows must form contiguous blocks")
        if positions.size > 1 and not np.all(
            np.diff(timestamps[positions]) == np.timedelta64(1, "h")
        ):
            raise ValueError("GEFCom2012 timestamps must be hourly within each segment")
    if list(metadata.get("calendar_covariate_columns") or ()) != list(
        PAPER_V7_GEFCOM2012_CALENDAR_COLUMNS
    ):
        raise ValueError("GEFCom2012 calendar covariate order changed")
    return arrays, metadata


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


def read_gift_arrow_targets(path: Path) -> tuple[str, list[tuple[str, np.ndarray]]]:
    """Read one GIFT-Eval load-from-disk directory without applying eval splits.

    Targets are returned in their native layout: ``[time]`` for univariate rows
    and ``[channel, time]`` for multivariate rows.  The caller owns the
    train/history window policy, including exclusion of official evaluation
    tails via :func:`gift_eval_short_term_test_holdout_steps`.
    """

    if not path.is_dir():
        raise FileNotFoundError(f"GIFT-Eval config directory not found: {path}")
    arrow_files = sorted(path.glob("data-*.arrow"))
    if len(arrow_files) != 1:
        raise ValueError(
            f"expected exactly one canonical data-*.arrow in {path}, got {len(arrow_files)}"
        )
    with pa.memory_map(str(arrow_files[0]), "r") as source:
        table = pa_ipc.open_stream(source).read_all()
    required = {"item_id", "target", "freq"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"GIFT-Eval config {path} is missing columns: {', '.join(missing)}")
    frequencies = {str(value) for value in table.column("freq").to_pylist()}
    if len(frequencies) != 1:
        raise ValueError(f"GIFT-Eval config {path} has non-unique frequencies: {frequencies}")
    records: list[tuple[str, np.ndarray]] = []
    for item_id, target in zip(
        table.column("item_id").to_pylist(),
        table.column("target").to_pylist(),
        strict=True,
    ):
        values = np.asarray(target, dtype=float)
        if values.ndim not in (1, 2) or values.size == 0:
            raise ValueError(
                f"GIFT-Eval item {item_id!r} in {path} has unsupported target shape {values.shape}"
            )
        records.append((str(item_id), values))
    return next(iter(frequencies)), records


def gift_eval_short_term_test_holdout_steps(
    frequency: str,
    records: list[tuple[str, np.ndarray]],
) -> int:
    """Reproduce GIFT-Eval's short-term test-tail length.

    This mirrors ``gift_eval.data.Dataset`` at the frozen protocol commit:
    ``windows = clip(ceil(0.1 * min_length / prediction_length), 1, 20)``.
    M4 is intentionally unsupported here because it has a separate prediction
    length table and is not loaded through the canonical-only Arrow profiles.
    """

    if not records:
        raise ValueError("GIFT-Eval holdout calculation needs at least one record")
    raw_frequency = str(frequency).strip()
    normalized = None
    for legacy_suffix in ("M", "W", "D", "H", "T", "S"):
        prefix = raw_frequency[: -len(legacy_suffix)]
        if raw_frequency.endswith(legacy_suffix) and (not prefix or prefix.isdigit()):
            normalized = legacy_suffix
            break
    offset_name = raw_frequency
    if normalized is None:
        offset_name = pd.tseries.frequencies.to_offset(raw_frequency).name
        normalized = {
            "ME": "M",
            "M": "M",
            "W": "W",
            "D": "D",
            "h": "H",
            "H": "H",
            "min": "T",
            "T": "T",
            "s": "S",
            "S": "S",
        }.get(offset_name)
    if normalized is None:
        # Anchored weekly aliases such as W-SUN retain the same base period.
        normalized = "W" if offset_name.startswith("W-") else None
    prediction_lengths = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
    if normalized not in prediction_lengths:
        raise ValueError(
            f"unsupported GIFT-Eval short-term frequency {frequency!r} ({offset_name!r})"
        )
    min_length = min(int(np.asarray(values).shape[-1]) for _item_id, values in records)
    prediction_length = prediction_lengths[normalized]
    windows = min(max(1, math.ceil(0.1 * min_length / prediction_length)), 20)
    return int(prediction_length * windows)


def truncate_gift_eval_official_test_tail(
    frequency: str,
    records: list[tuple[str, np.ndarray]],
) -> tuple[int, list[tuple[str, np.ndarray]]]:
    holdout_steps = gift_eval_short_term_test_holdout_steps(frequency, records)
    truncated: list[tuple[str, np.ndarray]] = []
    for item_id, values in records:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] <= holdout_steps:
            raise ValueError(
                f"GIFT-Eval item {item_id!r} is shorter than its official test tail: "
                f"length={array.shape[-1]}, holdout={holdout_steps}"
            )
        truncated.append((item_id, array[..., :-holdout_steps]))
    return holdout_steps, truncated


def read_uci_hydraulic_sensor_cycles(
    path: Path,
    *,
    sensor: str = "EPS1",
) -> np.ndarray:
    """Read and downsample one UCI hydraulic sensor to one value per second.

    The archive stores one row per official 60-second load cycle.  Sensors may
    have different native sampling rates, so each row is block-averaged to 60
    points and cycles are then concatenated in their original order.
    """

    member_name = f"{sensor}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"UCI hydraulic archive not found: {path}")
    with zipfile.ZipFile(path) as archive:
        names = {name.rsplit("/", 1)[-1]: name for name in archive.namelist()}
        if member_name not in names:
            raise ValueError(f"UCI hydraulic archive is missing {member_name}: {path}")
        with archive.open(names[member_name]) as handle:
            cycles = np.loadtxt(handle, dtype=np.float32)
    if cycles.ndim != 2 or cycles.shape[0] < 2 or cycles.shape[1] % 60:
        raise ValueError(
            f"invalid UCI hydraulic {sensor} matrix shape {cycles.shape}; "
            "expected [cycle, samples_per_60_seconds]"
        )
    points_per_second = cycles.shape[1] // 60
    per_second = cycles.reshape(cycles.shape[0], 60, points_per_second).mean(axis=2)
    values = np.asarray(per_second.reshape(-1), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"UCI hydraulic {sensor} contains non-finite values")
    return values


def read_skchange_hvac_series(path: Path, *, unit_id: int = 0) -> np.ndarray:
    """Read one bundled skchange HVAC unit in chronological 10-minute order."""

    csv_path = (
        path
        if path.is_file()
        else path / "skchange/datasets/data/hvac_system/data.csv"
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"skchange HVAC CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"time", "unit_id", "vibration"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"skchange HVAC CSV is missing columns: {', '.join(missing)}")
    selected = frame.loc[frame["unit_id"] == unit_id, ["time", "vibration"]].copy()
    if selected.empty:
        raise ValueError(f"skchange HVAC unit_id={unit_id} is absent from {csv_path}")
    selected["time"] = pd.to_datetime(selected["time"], utc=True, errors="coerce")
    selected["vibration"] = pd.to_numeric(selected["vibration"], errors="coerce")
    if selected.isna().any().any():
        raise ValueError(f"skchange HVAC unit_id={unit_id} contains invalid values")
    selected = selected.sort_values("time")
    selected = selected.set_index("time")
    regular_index = pd.date_range(
        selected.index.min(),
        selected.index.max(),
        freq="10min",
    )
    missing_count = len(regular_index) - len(selected)
    if missing_count < 0 or missing_count / len(regular_index) > 0.01:
        raise ValueError(
            f"skchange HVAC unit_id={unit_id} has too many missing 10-minute samples: "
            f"{missing_count}/{len(regular_index)}"
        )
    regular = selected.reindex(regular_index)
    values = regular["vibration"].interpolate(method="time", limit=18).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"skchange HVAC unit_id={unit_id} contains non-finite values")
    return values


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
        "covariate_columns": list(M5_KNOWN_FUTURE_COVARIATES),
        "covariate_provenance": M5_COVARIATE_PROVENANCE,
        "features": quantiles,
        "target_feature_caps": caps,
    }


def profile_m5_sibling_panel(
    path: Path,
    *,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows: int | None = None,
    max_groups: int = 20,
    target_dim: int = 3,
    season_length: int | None = 7,
    domain: str | None = "retail",
    dataset_name: str | None = None,
    target_features: list[str] | None = None,
    target_max_multiplier: float = 2.0,
) -> dict[str, Any]:
    _calendar, sales, day_columns = read_m5_calendar_and_sales(path)
    panels = sample_sequence_evenly(
        m5_sibling_leaf_panels(
            sales,
            day_columns,
            target_dim=target_dim,
        ),
        max_groups,
    )
    resolved_stride = stride or horizon
    spec = WindowSpec(context_length, horizon, resolved_stride)
    candidates = [
        (group_index, start)
        for group_index, (_group_id, _item_ids, values) in enumerate(panels)
        for start in window_starts(values.shape[0], spec)
    ]
    candidates = limit_candidates(candidates, max_windows)
    feature_rows: list[dict[str, float]] = []
    used_group_ids: set[int] = set()
    for group_index, start in candidates:
        values = panels[group_index][2]
        window = values[start : start + spec.length]
        if window.shape != (spec.length, target_dim) or not np.isfinite(window).all():
            continue
        features = feature_vector(window, season_length=season_length)
        features["group_index"] = float(group_index)
        features["window_start"] = float(start)
        feature_rows.append(features)
        used_group_ids.add(group_index)

    feature_names = [
        name for name in DEFAULT_FEATURES if any(name in row for row in feature_rows)
    ]
    quantiles = summarize_feature_rows(feature_rows, feature_names)
    caps = suggested_target_caps(
        quantiles,
        target_features=target_features
        or ["pca_top1_explained", "avg_abs_target_corr"],
        multiplier=target_max_multiplier,
    )
    return {
        "schema_version": "synthetic_feature_profile.v1",
        "dataset": dataset_name or "M5 daily sibling leaf panel",
        "source_path": str(path),
        "bucket": {
            "domain": domain or "retail",
            "frequency": "d",
            "context_length": context_length,
            "horizon": horizon,
            "target_dim": target_dim,
            "covariate_dim": 0,
            "season_length": season_length,
        },
        "window_count": len(feature_rows),
        "candidate_window_count": len(candidates),
        "series_count": len(panels),
        "used_series_count": len(used_group_ids),
        "target_columns": [f"sibling_leaf_{index}" for index in range(target_dim)],
        "panel_semantics": "disjoint item leaves sharing one store_id and dept_id",
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
) -> np.ndarray:
    if "date" in calendar.columns:
        dates = pd.to_datetime(calendar["date"], errors="raise")
        day_of_week = dates.dt.dayofweek.to_numpy(dtype=float)
    elif "wday" in calendar.columns:
        day_of_week = (
            pd.to_numeric(calendar["wday"], errors="raise").to_numpy(dtype=float)
            - 1.0
        )
    else:
        day_of_week = np.arange(len(calendar), dtype=float) % 7.0
    day_angle = 2.0 * np.pi * day_of_week / 7.0
    event_count = (
        calendar[[column for column in ("event_name_1", "event_name_2") if column in calendar.columns]]
        .notna()
        .sum(axis=1)
        .to_numpy(dtype=float)
    )
    snap_column = f"snap_{state_id}"
    snap = calendar[snap_column].to_numpy(dtype=float) if snap_column in calendar.columns else np.zeros(len(calendar))
    return np.column_stack(
        [
            np.sin(day_angle),
            np.cos(day_angle),
            event_count,
            snap,
        ]
    )


def m5_sibling_leaf_panels(
    sales: pd.DataFrame,
    day_columns: list[str],
    *,
    target_dim: int = 3,
) -> list[tuple[str, tuple[str, ...], np.ndarray]]:
    """Return disjoint sibling leaf panels without aggregate target leakage."""

    if target_dim < 2:
        raise ValueError("M5 sibling panels require target_dim >= 2")
    panels: list[tuple[str, tuple[str, ...], np.ndarray]] = []
    for (store_id, dept_id), siblings in sales.groupby(
        ["store_id", "dept_id"],
        sort=True,
    ):
        active = siblings.loc[siblings[day_columns].sum(axis=1) > 0].copy()
        if len(active) < target_dim:
            continue
        active = active.sort_values(["item_id", "id"], kind="stable")
        for start in range(0, len(active) - target_dim + 1, target_dim):
            selected = active.iloc[start : start + target_dim]
            item_ids = tuple(str(value) for value in selected["item_id"].tolist())
            if len(set(item_ids)) != target_dim:
                continue
            values = selected[day_columns].to_numpy(dtype=float).T
            panels.append(
                (
                    f"{store_id}:{dept_id}:chunk:{start // target_dim}",
                    item_ids,
                    values,
                )
            )
    return panels


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


def read_nixtla_binary_hierarchies(
    path: Path,
    *,
    group: str,
) -> list[tuple[str, np.ndarray]]:
    """Return disjoint two-child projections from an official Nixtla hierarchy.

    Paper-v1 uses only official aggregation rows whose support contains exactly
    two bottom series.  The supports must be pairwise disjoint, so a group split
    cannot leak the same bottom trajectory across calibration partitions.
    """

    with zipfile.ZipFile(path) as archive:
        matrix_name = f"{group}/agg_mat.csv"
        data_name = f"{group}/data.csv"
        available = set(archive.namelist())
        if matrix_name not in available or data_name not in available:
            raise ValueError(f"Nixtla hierarchy group {group!r} not found in {path}")
        with archive.open(matrix_name) as handle:
            summing = pd.read_csv(handle, index_col=0)
        with archive.open(data_name) as handle:
            values = pd.read_csv(handle, index_col=0)

    if list(summing.index) != list(values.columns):
        raise ValueError(f"Nixtla {group} data columns do not match summing-matrix rows")
    binary_rows = summing.loc[summing.astype(bool).sum(axis=1) == 2]
    if binary_rows.empty:
        raise ValueError(f"Nixtla {group} has no two-bottom aggregation rows")
    memberships = binary_rows.astype(bool).sum(axis=0)
    if bool((memberships > 1).any()):
        raise ValueError(f"Nixtla {group} two-bottom aggregation rows overlap")

    result: list[tuple[str, np.ndarray]] = []
    for parent_id, weights in binary_rows.iterrows():
        child_ids = [str(column) for column, weight in weights.items() if float(weight) != 0.0]
        child_values = [values[child_id].to_numpy(dtype=float) for child_id in child_ids]
        parent = child_values[0] + child_values[1]
        observed_parent = values[str(parent_id)].to_numpy(dtype=float)
        if not np.allclose(parent, observed_parent, rtol=1e-7, atol=1e-7, equal_nan=True):
            raise ValueError(f"Nixtla {group} aggregation invariant failed for {parent_id!r}")
        result.append((str(parent_id), np.column_stack([parent, *child_values])))
    return result


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


def gefcom2014_wind_archive_bytes(path: Path) -> bytes:
    """Return the official Wind track zip, accepting the full or track archive."""

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any(name.startswith("Wind/Task ") for name in names):
            return path.read_bytes()
        nested_name = first_matching_name(names, suffix="GEFCom2014-W_V2.zip")
        return archive.read(nested_name)


def read_gefcom2014_wind_nested_frames(
    archive: zipfile.ZipFile,
    *,
    task: int,
    predictors: bool,
) -> dict[int, pd.DataFrame]:
    nested_name = (
        f"Wind/Task {task}/TaskExpVars{task}_W_Zone1_10.zip"
        if predictors
        else f"Wind/Task {task}/Task{task}_W_Zone1_10.zip"
    )
    if nested_name not in archive.namelist():
        raise ValueError(
            f"GEFCom2014 Wind task {task} is missing "
            f"{'TaskExpVars' if predictors else 'target training'} archive"
        )
    with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested:
        csv_names = sorted(
            name
            for name in nested.namelist()
            if name.lower().endswith(".csv")
        )
        frames: dict[int, pd.DataFrame] = {}
        required = {
            "ZONEID",
            "TIMESTAMP",
            *GEFCOM2014_WIND_NWP_COLUMNS,
        }
        if not predictors:
            required.add("TARGETVAR")
        for csv_name in csv_names:
            with nested.open(csv_name) as handle:
                frame = pd.read_csv(handle)
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(
                    f"GEFCom2014 Wind {csv_name} is missing columns: "
                    + ", ".join(missing)
                )
            zone_values = pd.to_numeric(frame["ZONEID"], errors="raise")
            if zone_values.nunique() != 1:
                raise ValueError(f"GEFCom2014 Wind {csv_name} mixes zones")
            zone_id = int(zone_values.iloc[0])
            parsed = pd.DataFrame(
                {
                    "TIMESTAMP": pd.to_datetime(
                        frame["TIMESTAMP"],
                        format="%Y%m%d %H:%M",
                        errors="raise",
                    ),
                    **{
                        column: pd.to_numeric(frame[column], errors="coerce")
                        for column in (
                            *(("TARGETVAR",) if not predictors else ()),
                            *GEFCOM2014_WIND_NWP_COLUMNS,
                        )
                    },
                }
            ).sort_values("TIMESTAMP").reset_index(drop=True)
            if parsed["TIMESTAMP"].duplicated().any():
                raise ValueError(
                    f"GEFCom2014 Wind task {task} zone {zone_id} duplicates timestamps"
                )
            deltas = parsed["TIMESTAMP"].diff().dropna()
            if not deltas.empty and not bool((deltas == pd.Timedelta(hours=1)).all()):
                raise ValueError(
                    f"GEFCom2014 Wind task {task} zone {zone_id} is not hourly"
                )
            frames[zone_id] = parsed
    if len(frames) != 10:
        raise ValueError(
            f"GEFCom2014 Wind task {task} expected 10 zones, got {len(frames)}"
        )
    return frames


def read_gefcom2014_wind_training_frames(
    path: Path,
    *,
    task: int,
) -> dict[int, pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(gefcom2014_wind_archive_bytes(path))) as archive:
        return read_gefcom2014_wind_nested_frames(
            archive,
            task=task,
            predictors=False,
        )


def read_gefcom2014_wind_forecast_releases(
    path: Path,
    *,
    minimum_future_steps: int,
) -> list[dict[str, Any]]:
    """Build issue-time-valid NWP/target pairs from consecutive competition tasks."""

    if minimum_future_steps < 1:
        raise ValueError("minimum_future_steps must be positive")
    with zipfile.ZipFile(io.BytesIO(gefcom2014_wind_archive_bytes(path))) as archive:
        names = set(archive.namelist())
        task_numbers = sorted(
            {
                int(match.group(1))
                for name in names
                if (
                    match := re.fullmatch(
                        r"Wind/Task (\d+)/TaskExpVars\1_W_Zone1_10\.zip",
                        name,
                    )
                )
            }
        )
        training_cache: dict[int, dict[int, pd.DataFrame]] = {}

        def training(task: int) -> dict[int, pd.DataFrame]:
            if task not in training_cache:
                training_cache[task] = read_gefcom2014_wind_nested_frames(
                    archive,
                    task=task,
                    predictors=False,
                )
            return training_cache[task]

        releases: list[dict[str, Any]] = []
        for task in task_numbers:
            next_target_name = (
                f"Wind/Task {task + 1}/Task{task + 1}_W_Zone1_10.zip"
            )
            if next_target_name not in names:
                continue
            history = training(task)
            realized = training(task + 1)
            predictors = read_gefcom2014_wind_nested_frames(
                archive,
                task=task,
                predictors=True,
            )
            future: dict[int, pd.DataFrame] = {}
            release_valid_start: pd.Timestamp | None = None
            release_valid_end: pd.Timestamp | None = None
            complete = True
            for zone_id in sorted(predictors):
                forecast = predictors[zone_id]
                actual = realized[zone_id][["TIMESTAMP", "TARGETVAR"]]
                joined = forecast.merge(
                    actual,
                    on="TIMESTAMP",
                    how="left",
                    validate="one_to_one",
                )
                expected_start = history[zone_id]["TIMESTAMP"].iloc[-1] + pd.Timedelta(
                    hours=1
                )
                if (
                    len(joined) < minimum_future_steps
                    or joined["TIMESTAMP"].iloc[0] != expected_start
                    or joined.iloc[:minimum_future_steps].isna().any().any()
                ):
                    complete = False
                    break
                zone_start = joined["TIMESTAMP"].iloc[0]
                zone_end = joined["TIMESTAMP"].iloc[-1]
                if release_valid_start is None:
                    release_valid_start = zone_start
                    release_valid_end = zone_end
                elif (
                    zone_start != release_valid_start
                    or zone_end != release_valid_end
                ):
                    complete = False
                    break
                future[zone_id] = joined
            if complete and len(future) == 10:
                releases.append(
                    {
                        "task": task,
                        "release_id": f"GEFCom2014-W-TaskExpVars-{task}",
                        "valid_start": release_valid_start,
                        "valid_end": release_valid_end,
                        "history": history,
                        "future": future,
                        "available_future_steps": min(
                            len(frame) for frame in future.values()
                        ),
                        "covariate_provenance": (
                            GEFCOM2014_WIND_COVARIATE_PROVENANCE
                        ),
                    }
                )
    if not releases:
        raise ValueError(
            "GEFCom2014 Wind has no complete competition predictor release "
            f"covering H={minimum_future_steps}; covariate view fails closed"
        )
    return releases


def read_gefcom2014_solar_frames(
    path: Path,
    *,
    task: int,
) -> tuple[list[tuple[str, pd.DataFrame]], str]:
    """Read target/NWP intersections from one GEFCom2014 Solar task."""

    with zipfile.ZipFile(path) as outer:
        names = outer.namelist()
        if any(name.startswith("Solar/") and name.endswith(f"train{task}.csv") for name in names):
            return read_gefcom2014_solar_frames_from_archive(outer, task=task)
        nested_name = first_matching_name(names, suffix="GEFCom2014-S_V2.zip")
        nested_data = outer.read(nested_name)
    with zipfile.ZipFile(io.BytesIO(nested_data)) as nested:
        frames, source_name = read_gefcom2014_solar_frames_from_archive(nested, task=task)
    return frames, f"{nested_name}:{source_name}"


def read_gefcom2014_solar_frames_from_archive(
    archive: zipfile.ZipFile,
    *,
    task: int,
) -> tuple[list[tuple[str, pd.DataFrame]], str]:
    train_name = f"Solar/Task {task}/train{task}.csv"
    predictor_name = f"Solar/Task {task}/predictors{task}.csv"
    available = set(archive.namelist())
    if train_name not in available or predictor_name not in available:
        raise ValueError(f"GEFCom2014 Solar task {task} is incomplete")
    with archive.open(train_name) as handle:
        target = pd.read_csv(handle)
    with archive.open(predictor_name) as handle:
        predictors = pd.read_csv(handle)
    keys = ["ZONEID", "TIMESTAMP"]
    frame = target.merge(predictors, on=keys, how="inner", validate="one_to_one")
    frame["_timestamp"] = pd.to_datetime(frame["TIMESTAMP"], format="%Y%m%d %H:%M", errors="raise")
    covariate_columns = [column for column in frame.columns if column.startswith("VAR")]
    if len(covariate_columns) != 12:
        raise ValueError(
            f"GEFCom2014 Solar task {task} expected 12 NWP variables, got {len(covariate_columns)}"
        )
    frames: list[tuple[str, pd.DataFrame]] = []
    for zone_id, zone in frame.groupby("ZONEID", sort=True):
        zone = zone.sort_values("_timestamp").reset_index(drop=True)
        numeric = zone[["POWER", *covariate_columns]].apply(pd.to_numeric, errors="coerce")
        numeric.insert(0, "TIMESTAMP", zone["_timestamp"])
        frames.append((str(zone_id), numeric))
    return frames, f"{train_name}+{predictor_name}"


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
    include_cross_series_predictability: bool = True,
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
        out.update(
            multitarget_features(
                window,
                max_lag=min(
                    48,
                    max(12, int(season_length or 12)),
                ),
                include_cross_series_predictability=(
                    include_cross_series_predictability
                ),
            )
        )
    if hierarchy == "additive_first" and window.shape[1] > 2:
        out.update(hierarchy_coordinate_features(window, season_length))
    if covariates is not None and covariates.size:
        out.update(
            covariate_features(
                window,
                covariates,
                context_length or max(1, window.shape[0] // 2),
                season_length=season_length,
            )
        )
    return out


def multitarget_features(
    window: np.ndarray,
    *,
    max_lag: int = 12,
    include_cross_series_predictability: bool = True,
) -> dict[str, float]:
    centered = window - np.mean(window, axis=0, keepdims=True)
    corr_values = [
        abs(safe_corr(window[:, left], window[:, right]))
        for left in range(window.shape[1])
        for right in range(window.shape[1])
        if left != right
    ]
    try:
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        singular = np.zeros(min(centered.shape), dtype=float)
        right = np.zeros((0, centered.shape[1]), dtype=float)
    variance = singular**2
    total = float(np.sum(variance))
    explained = variance / total if total > 1e-12 else np.zeros_like(variance)
    entropy = -float(np.sum([value * np.log(value) for value in explained if value > 1e-12]))
    result = {
        "avg_abs_target_corr": float(np.mean(corr_values)) if corr_values else 0.0,
        "pca_top1_explained": float(explained[0]) if explained.size else 0.0,
        "pca_top2_explained": float(np.sum(explained[:2])) if explained.size else 0.0,
        "effective_factor_rank": float(np.exp(entropy)) if explained.size else 0.0,
    }
    if include_cross_series_predictability:
        result.update(
            {
                "lead_lag_peak_abs": lead_lag_peak_abs(
                    window,
                    max_lag=max_lag,
                ),
                "lead_lag_peak_lag_abs": lead_lag_peak_lag_abs(
                    window,
                    max_lag=max_lag,
                ),
                "cross_series_incremental_r2": (
                    cross_series_incremental_r2(
                        window,
                        max_lag=max_lag,
                    )
                ),
            }
        )
    if right.size:
        factor_score = centered @ right[0]
        factor_component = factor_score[:, None] * right[0][None, :]
        residual = centered - factor_component
        result["factor_score_acf1"] = autocorrelation(factor_score, 1)
        result["factor_residual_acf1"] = float(
            np.mean(
                [
                    autocorrelation(residual[:, index], 1)
                    for index in range(residual.shape[1])
                ]
            )
        )
    return result


def hierarchy_coordinate_features(
    window: np.ndarray,
    season_length: int | None,
) -> dict[str, float]:
    """Describe a hierarchy without assuming a particular latent DGP.

    The aggregate is the observed parent.  Child deviations are projected to
    their leading contrast score, which is invariant to the arbitrary sign of
    the singular vector.  These are descriptive calibration coordinates, not
    claims that the real data were generated by an aggregate/contrast model.
    """

    values = np.asarray(window, dtype=float)
    parent = values[:, 0]
    children = values[:, 1:]
    residual = parent - np.sum(children, axis=1)
    deviations = children - np.mean(children, axis=1, keepdims=True)
    try:
        left, singular, _ = np.linalg.svd(deviations, full_matrices=False)
        contrast = left[:, 0] * singular[0] if singular.size else np.zeros(len(values))
    except np.linalg.LinAlgError:
        contrast = deviations[:, 0]
    parent_std = float(np.std(parent))
    contrast_std = float(np.std(contrast))
    child_heterogeneity = float(np.mean(np.std(children, axis=1)))
    child_magnitude = float(np.mean(np.sum(np.abs(children), axis=1)))
    period = max(1, int(season_length or 1))
    return {
        "hierarchy_residual_mean_abs": float(np.mean(np.abs(residual))),
        "hierarchy_child_heterogeneity": child_heterogeneity,
        "hierarchy_aggregation_ratio": (
            float(np.mean(np.abs(parent))) / child_magnitude
            if child_magnitude > 1e-12
            else 0.0
        ),
        "hierarchy_aggregate_acf1": autocorrelation(parent, 1),
        "hierarchy_contrast_acf1": autocorrelation(contrast, 1),
        "hierarchy_aggregate_seasonal_acf": autocorrelation(parent, period),
        "hierarchy_contrast_seasonal_acf": autocorrelation(contrast, period),
        "hierarchy_contrast_to_aggregate_std_ratio": (
            contrast_std / parent_std if parent_std > 1e-12 else 0.0
        ),
        "hierarchy_aggregate_contrast_abs_corr": abs(
            safe_corr(parent, contrast)
        ),
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


def lead_lag_peak_lag_abs(
    window: np.ndarray,
    max_lag: int = 12,
) -> float:
    """Absolute lag of the strongest ordered cross-channel correlation."""

    if window.shape[1] < 2:
        return 0.0
    lag_limit = min(max_lag, max(1, window.shape[0] // 4))
    best_correlation = -1.0
    best_lag = 0
    for left in range(window.shape[1]):
        for right in range(window.shape[1]):
            if left == right:
                continue
            for lag in range(1, lag_limit + 1):
                correlation = abs(
                    safe_corr(window[:-lag, left], window[lag:, right])
                )
                if math.isfinite(correlation) and correlation > best_correlation:
                    best_correlation = correlation
                    best_lag = lag
    return float(best_lag)


def cross_series_incremental_r2(
    window: np.ndarray,
    max_lag: int = 12,
) -> float:
    """Held-out error reduction from other-channel lags over own lags.

    This is a descriptive Granger-style calibration coordinate, not a
    significance test.  Ridge stabilization and a chronological holdout keep
    the high-dimensional unrestricted regression from receiving free in-sample
    gains.
    """

    values = np.asarray(window, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        return 0.0
    lag_limit = min(
        max(2, int(max_lag)),
        48,
        max(2, values.shape[0] // 5),
    )
    if values.shape[0] < 4 * lag_limit:
        return 0.0
    scaled = np.column_stack(
        [robust_scale(values[:, index]) for index in range(values.shape[1])]
    )
    sample_count = values.shape[0] - lag_limit
    lagged = np.stack(
        [
            scaled[lag_limit - lag : values.shape[0] - lag]
            for lag in range(1, lag_limit + 1)
        ],
        axis=1,
    )
    split = max(2 * lag_limit, int(round(0.70 * sample_count)))
    split = min(split, sample_count - lag_limit)
    if split <= lag_limit or sample_count - split < lag_limit:
        return 0.0
    gains: list[float] = []
    for target_index in range(values.shape[1]):
        response = scaled[lag_limit:, target_index]
        own = lagged[:, :, target_index]
        full = lagged.reshape(sample_count, -1)
        own_prediction = ridge_holdout_prediction(own, response, split)
        full_prediction = ridge_holdout_prediction(full, response, split)
        actual = response[split:]
        own_error = float(np.sum((actual - own_prediction) ** 2))
        full_error = float(np.sum((actual - full_prediction) ** 2))
        if own_error > 1e-12:
            gains.append(
                clamp01((own_error - full_error) / own_error)
            )
    return float(np.mean(gains)) if gains else 0.0


def ridge_holdout_prediction(
    design: np.ndarray,
    response: np.ndarray,
    split: int,
) -> np.ndarray:
    train = np.asarray(design[:split], dtype=float)
    test = np.asarray(design[split:], dtype=float)
    center = np.mean(train, axis=0, keepdims=True)
    scale = np.std(train, axis=0, keepdims=True)
    scale = np.where(scale > 1e-9, scale, 1.0)
    train = (train - center) / scale
    test = (test - center) / scale
    train = np.column_stack([np.ones(train.shape[0]), train])
    test = np.column_stack([np.ones(test.shape[0]), test])
    penalty = np.eye(train.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    alpha = 0.05 * train.shape[0]
    try:
        coefficients = np.linalg.solve(
            train.T @ train + alpha * penalty,
            train.T @ response[:split],
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(
            train.T @ train + alpha * penalty
        ) @ (train.T @ response[:split])
    return test @ coefficients


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
        "seasonal_acf": autocorrelation(scaled, max(1, int(season_length or 1))),
        **spectral_time_scale_features(scaled),
        "outlier_rate": outlier_rate(scaled),
        "spike_rate": spike_rate(scaled),
        "trend_resid_var": trend_resid_var,
        "seasonal_resid_var": seasonal_resid_var,
    }


def spectral_time_scale_features(values: np.ndarray) -> dict[str, float]:
    """Return a robust dominant period and its share of detrended energy."""

    y = np.asarray(values, dtype=float)
    if y.size < 12:
        return {"dominant_period": 0.0, "spectral_concentration": 0.0}
    time = np.arange(y.size, dtype=float)
    design = np.column_stack([np.ones(y.size), time])
    try:
        detrended = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        detrended = y - float(np.mean(y))
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies = np.fft.rfftfreq(y.size)
    if power.size:
        power[0] = 0.0
    valid = (frequencies >= 2.0 / y.size) & (frequencies <= 1.0 / 3.0)
    if not np.any(valid) or float(np.sum(power)) <= 1e-12:
        return {"dominant_period": 0.0, "spectral_concentration": 0.0}
    valid_indices = np.flatnonzero(valid)
    peak = int(valid_indices[np.argmax(power[valid])])
    return {
        "dominant_period": float(1.0 / frequencies[peak]),
        "spectral_concentration": float(power[peak] / np.sum(power)),
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
    modulation_features = seasonal_modulation_features(y, season_length)
    diff = np.diff(y)
    return {
        "level_shift_strength": float(max(level_scores)) if level_scores else 0.0,
        "volatility_shift_strength": float(max(volatility_scores)) if volatility_scores else 0.0,
        "change_point_shift_energy": float(np.mean(sorted(level_scores, reverse=True)[:3])) if level_scores else 0.0,
        "regime_sparse_transition_score": regime_sparse_transition_score(y),
        "burst_rate": float(np.mean(np.abs(y) > 3.0)),
        "diff_spike_rate": float(np.mean(np.abs(robust_scale(diff)) > 3.0)) if diff.size else 0.0,
        "intermittency_clock_incremental_r2": intermittency_clock_incremental_r2(
            y,
            season_length,
        ),
        "multi_period_score": multi_period_score(y, season_length),
        "seasonal_drift_score": float(np.mean(np.abs(seasonal_left - seasonal_right))) if seasonal_left.size and seasonal_right.size else 0.0,
        "seasonal_amplitude_cv": float(np.std(np.abs(seasonal_profile)) / (np.mean(np.abs(seasonal_profile)) + 1e-9)) if seasonal_profile.size else 0.0,
        "nonlinear_lag1_gain": nonlinear_lag1_gain(y),
        "nonlinear_multi_lag_gain": nonlinear_multi_lag_gain(y, season_length),
        "nonlinear_conditional_gain": nonlinear_conditional_gain(y, season_length),
        **modulation_features,
    }


def seasonal_modulation_features(
    values: np.ndarray,
    season_length: int | None,
) -> dict[str, float]:
    """Measure cycle-to-cycle amplitude and phase variation of the base period."""

    period = max(4, int(season_length or 4))
    cycle_count = values.size // period
    if cycle_count < 3:
        return {
            "seasonal_amplitude_modulation": 0.0,
            "seasonal_phase_variation": 0.0,
        }
    phase_index = np.arange(period, dtype=float)
    sine = np.sin(2.0 * np.pi * phase_index / period)
    cosine = np.cos(2.0 * np.pi * phase_index / period)
    amplitudes: list[float] = []
    phases: list[float] = []
    for cycle in range(cycle_count):
        segment = values[cycle * period : (cycle + 1) * period]
        centered = segment - float(np.mean(segment))
        sine_coefficient = 2.0 * float(np.mean(centered * sine))
        cosine_coefficient = 2.0 * float(np.mean(centered * cosine))
        amplitudes.append(float(np.hypot(sine_coefficient, cosine_coefficient)))
        phases.append(float(np.arctan2(cosine_coefficient, sine_coefficient)))
    amplitude_array = np.asarray(amplitudes, dtype=float)
    phase_array = np.unwrap(np.asarray(phases, dtype=float))
    return {
        "seasonal_amplitude_modulation": float(
            np.std(amplitude_array) / (np.mean(amplitude_array) + 1e-9)
        ),
        "seasonal_phase_variation": float(np.std(phase_array) / np.pi),
    }


def intermittency_clock_incremental_r2(
    values: np.ndarray,
    season_length: int | None,
) -> float:
    """Extra adjusted R2 of a recurring event clock over smooth seasonality.

    The clock uses three base periods so it can represent a deterministic
    short/long/nominal interval motif without being given generator metadata.
    Ordinary trend and two base-period Fourier harmonics are residualized.
    """

    y = np.asarray(values, dtype=float)
    period = max(4, int(season_length or 4))
    clock_period = 3 * period
    if y.size < 3 * clock_period:
        return 0.0
    time = np.arange(y.size, dtype=float)
    normalized_time = (time - np.mean(time)) / max(y.size - 1, 1)
    baseline_columns = [np.ones(y.size), normalized_time]
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * time / period
        baseline_columns.extend([np.sin(angle), np.cos(angle)])
    baseline = np.column_stack(baseline_columns)
    phase = np.arange(y.size) % clock_period
    phase_design = np.eye(clock_period, dtype=float)[phase]
    full = np.column_stack([baseline, phase_design[:, 1:]])
    split = max(2 * clock_period, int(round(0.67 * y.size)))
    split = min(split, y.size - clock_period)
    if split <= clock_period or y.size - split < period:
        return 0.0
    try:
        baseline_coefficients = np.linalg.lstsq(
            baseline[:split],
            y[:split],
            rcond=None,
        )[0]
        full_coefficients = np.linalg.lstsq(
            full[:split],
            y[:split],
            rcond=None,
        )[0]
    except np.linalg.LinAlgError:
        return 0.0
    baseline_error = float(
        np.sum((y[split:] - baseline[split:] @ baseline_coefficients) ** 2)
    )
    full_error = float(
        np.sum((y[split:] - full[split:] @ full_coefficients) ** 2)
    )
    if baseline_error <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, (baseline_error - full_error) / baseline_error))


def regime_sparse_transition_score(values: np.ndarray) -> float:
    """Share of first-difference energy concentrated in sparse transitions."""

    differences = np.diff(np.asarray(values, dtype=float))
    energy = differences * differences
    total = float(np.sum(energy))
    if energy.size < 8 or total <= 1e-12:
        return 0.0
    count = max(2, int(math.ceil(0.05 * energy.size)))
    top = np.partition(energy, -count)[-count:]
    raw_share = float(np.sum(top) / total)
    # Uniformly distributed variation has expected share close to 5%; map
    # that baseline to zero while retaining a bounded [0, 1] score.
    baseline = count / energy.size
    return clamp01((raw_share - baseline) / max(1.0 - baseline, 1e-12))


def covariate_features(
    target: np.ndarray,
    covariates: np.ndarray,
    context_length: int,
    *,
    season_length: int | None = None,
) -> dict[str, float]:
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
    time = np.arange(target.shape[0], dtype=float)
    period = max(4, int(season_length or min(24, target.shape[0] // 4)))
    baseline_design = np.column_stack(
        [
            np.ones(target.shape[0], dtype=float),
            np.sin(2.0 * np.pi * time / period),
            np.cos(2.0 * np.pi * time / period),
        ]
    )
    covariate_design = np.column_stack([baseline_design, covariates])
    incremental_scores = [
        max(
            0.0,
            r2(target[:, target_idx], covariate_design)
            - r2(target[:, target_idx], baseline_design),
        )
        for target_idx in range(target.shape[1])
    ]
    residual_acf_scores: list[float] = []
    residual_outlier_scores: list[float] = []
    residual_spike_scores: list[float] = []
    for target_idx in range(target.shape[1]):
        coefficients = np.linalg.lstsq(
            covariate_design,
            target[:, target_idx],
            rcond=None,
        )[0]
        residual = target[:, target_idx] - covariate_design @ coefficients
        scaled_residual = robust_scale(residual)
        residual_acf_scores.append(
            mean_abs_autocorrelation(
                scaled_residual,
                max_lag=min(10, max(1, residual.size // 4)),
            )
        )
        residual_outlier_scores.append(outlier_rate(residual))
        residual_spike_scores.append(spike_rate(residual))
    return {
        "avg_abs_covariate_target_corr": float(np.mean(scores)) if scores else 0.0,
        "future_abs_covariate_target_corr": float(np.mean(future_scores)) if future_scores else 0.0,
        "event_lift_abs": float(np.mean(event_lifts)) if event_lifts else 0.0,
        "covariate_incremental_r2": float(np.mean(incremental_scores)) if incremental_scores else 0.0,
        "covariate_residual_acf_abs_mean": float(np.mean(residual_acf_scores)) if residual_acf_scores else 0.0,
        "covariate_residual_outlier_rate": float(np.mean(residual_outlier_scores)) if residual_outlier_scores else 0.0,
        "covariate_residual_spike_rate": float(np.mean(residual_spike_scores)) if residual_spike_scores else 0.0,
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


def nonlinear_multi_lag_gain(
    values: np.ndarray,
    season_length: int | None,
) -> float:
    seasonal_lag = max(4, int(season_length or 4))
    nonlinear_lag = max(2, seasonal_lag // 2)
    start = max(seasonal_lag, nonlinear_lag, 1)
    if values.size - start < 8:
        return 0.0
    target = values[start:]
    lag1 = values[start - 1 : -1]
    lag_seasonal = values[: values.size - seasonal_lag]
    if lag_seasonal.size > target.size:
        lag_seasonal = lag_seasonal[-target.size :]
    lag_nonlinear = values[
        start - nonlinear_lag : values.size - nonlinear_lag
    ]
    linear = np.column_stack([np.ones_like(target), lag1])
    nonlinear = np.column_stack(
        [
            np.ones_like(target),
            lag1,
            lag_seasonal,
            np.sin(2.0 * lag_nonlinear),
        ]
    )
    return max(0.0, r2(target, nonlinear) - r2(target, linear))


def nonlinear_conditional_gain(
    values: np.ndarray,
    season_length: int | None,
) -> float:
    """Bias-corrected nonlinear-lag gain after linear lag conditioning."""

    seasonal_lag = max(4, int(season_length or 4))
    nonlinear_lag = max(2, seasonal_lag // 2)
    start = max(seasonal_lag, nonlinear_lag, 1)
    if values.size - start < 8:
        return 0.0
    target = values[start:]
    lag1 = values[start - 1 : -1]
    lag_seasonal = values[: values.size - seasonal_lag]
    if lag_seasonal.size > target.size:
        lag_seasonal = lag_seasonal[-target.size :]
    lag_nonlinear = values[
        start - nonlinear_lag : values.size - nonlinear_lag
    ]
    linear = np.column_stack(
        [np.ones_like(target), lag1, lag_seasonal, lag_nonlinear]
    )
    nonlinear = np.column_stack(
        [linear, np.sin(1.1 * lag_nonlinear) ** 2]
    )
    return float(
        adjusted_r2(target, nonlinear) - adjusted_r2(target, linear)
    )


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


def adjusted_r2(y: np.ndarray, design: np.ndarray) -> float:
    observations = int(len(y))
    try:
        predictor_count = max(int(np.linalg.matrix_rank(design)) - 1, 0)
    except np.linalg.LinAlgError:
        return 0.0
    residual_degrees_of_freedom = observations - predictor_count - 1
    if observations <= 1 or residual_degrees_of_freedom <= 0:
        return 0.0
    raw_r2 = r2(y, design)
    return float(
        1.0
        - (1.0 - raw_r2)
        * (observations - 1)
        / residual_degrees_of_freedom
    )


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
