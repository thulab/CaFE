#!/usr/bin/env python3
"""Prepare the external Paper v7 P0 structured forecasting datasets.

The outputs are numeric-only ``.npz`` files that can be opened with
``allow_pickle=False`` plus explicit JSON provenance/audit sidecars.

The Swiss source stores its weather forecasts in a pandas HDF5 object table.
Reading it requires PyTables.  From the repository root, a reproducible
one-shot invocation is:

    cd backend
    uv run --with tables --with pytz \
      python ../scripts/prepare_paper_v7_p0_datasets.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Iterable
import zipfile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "runtime" / "research" / "v7-p0-data"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "processed"
DEFAULT_SWISS_POWER = DEFAULT_SOURCE_DIR / "swiss" / "power_data.p"
DEFAULT_SWISS_NWP = DEFAULT_SOURCE_DIR / "swiss" / "nwp_data.h5"
DEFAULT_GEFCOM2012_ZIP = DEFAULT_SOURCE_DIR / "gefcom2012" / "GEFCom2012.zip"

SCHEMA_VERSION = "paper_v7_p0_processed_dataset.v1"
SWISS_SOURCE_URL = "https://zenodo.org/records/3463137"
SWISS_POWER_URL = (
    "https://zenodo.org/api/records/3463137/files/power_data.p/content"
)
SWISS_NWP_URL = (
    "https://zenodo.org/api/records/3463137/files/nwp_data.h5/content"
)
GEFCOM2012_SOURCE_URL = (
    "https://blog.drhongtao.com/2016/07/"
    "gefcom2012-load-forecasting-data.html"
)
GEFCOM2012_DOWNLOAD_URL = (
    "https://www.dropbox.com/s/epj9b57eivn79j7/GEFCom2012.zip?dl=1"
)

SWISS_AGGREGATE_COLUMNS = ("all", "S1", "S2", "S11", "S12", "S21", "S22")
SWISS_NWP_CANONICAL_COLUMNS = (
    "ghi_backwards",
    "gni_backwards",
    "relativehumidity",
    "windspeed",
    "winddirection",
    "temperature",
)
GEFCOM_ZONE_COLUMNS = tuple(f"zone_{zone}" for zone in range(1, 21))
GEFCOM_HIERARCHY_COLUMNS = ("total", "sum_zones_1_10", "sum_zones_11_20")
GEFCOM_CALENDAR_COLUMNS = (
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "is_weekend",
    "is_holiday",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swiss-power", type=Path, default=DEFAULT_SWISS_POWER)
    parser.add_argument("--swiss-nwp", type=Path, default=DEFAULT_SWISS_NWP)
    parser.add_argument(
        "--gefcom2012-zip",
        type=Path,
        default=DEFAULT_GEFCOM2012_ZIP,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_record(path: Path, url: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def utc_index(values: Iterable[Any]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def datetime64_ns_utc(index: pd.DatetimeIndex) -> np.ndarray:
    normalized = utc_index(index).tz_localize(None)
    return normalized.to_numpy(dtype="datetime64[ns]")


def complete_trailing_30_minute_means(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Average complete trailing half-hour bins from 10-minute measurements.

    A source timestamp is the end of its 10-minute measurement interval.
    Therefore ``00:10``, ``00:20`` and ``00:30`` form the bin whose start is
    ``00:00``.  Bins without exactly those three observations are discarded.
    """

    if frame.empty:
        raise ValueError("cannot aggregate an empty Swiss power frame")
    index = utc_index(frame.index)
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Swiss power timestamps must be sorted and unique")
    if frame.isna().any().any():
        raise ValueError("Swiss power data contain missing values")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Swiss power data contain non-finite values")

    bin_starts = (index - pd.Timedelta(nanoseconds=1)).floor("30min")
    offsets = index - bin_starts
    expected_offsets = {
        pd.Timedelta(minutes=10),
        pd.Timedelta(minutes=20),
        pd.Timedelta(minutes=30),
    }
    offset_sets = pd.Series(offsets, index=index).groupby(
        np.asarray(bin_starts),
        sort=True,
    ).agg(lambda values: frozenset(values))
    complete_starts = offset_sets.index[
        offset_sets.map(lambda values: values == expected_offsets)
    ]
    grouped = frame.copy()
    grouped.index = bin_starts
    means = grouped.groupby(level=0, sort=True).mean().loc[complete_starts]
    means.index = utc_index(means.index)
    means.index.name = "timestamp"
    return means, {
        "source_rows": len(frame),
        "complete_bin_rows": len(means),
        "discarded_source_rows": len(frame) - 3 * len(means),
        "discarded_bins": len(offset_sets) - len(means),
    }


def swiss_hierarchy_audit(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [
        column for column in SWISS_AGGREGATE_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Swiss P_mean is missing aggregates: {', '.join(missing)}")
    meter_columns = [
        str(column)
        for column in frame.columns
        if str(column) not in SWISS_AGGREGATE_COLUMNS
    ]
    if len(meter_columns) != 24:
        raise ValueError(
            f"Swiss P_mean must contain 24 meters, found {len(meter_columns)}"
        )

    leaves = frame[meter_columns].to_numpy(dtype=float)
    actual = frame[list(SWISS_AGGREGATE_COLUMNS)].to_numpy(dtype=float)
    s11 = np.sum(leaves[:, 0:6], axis=1)
    s12 = np.sum(leaves[:, 6:12], axis=1)
    s21 = np.sum(leaves[:, 12:18], axis=1)
    s22 = np.sum(leaves[:, 18:24], axis=1)
    expected = np.column_stack(
        [
            s11 + s12 + s21 + s22,
            s11 + s12,
            s21 + s22,
            s11,
            s12,
            s21,
            s22,
        ]
    )
    absolute = np.abs(actual - expected)
    scale = max(1.0, float(np.max(np.abs(expected))))
    tolerance = scale * 1e-10
    max_error = float(np.max(absolute))
    if max_error > tolerance:
        raise ValueError(
            "Swiss hierarchy is not additive within tolerance: "
            f"max_error={max_error}, tolerance={tolerance}"
        )
    return {
        "meter_columns": meter_columns,
        "aggregate_columns": list(SWISS_AGGREGATE_COLUMNS),
        "max_absolute_error": max_error,
        "mean_absolute_error": float(np.mean(absolute)),
        "tolerance": tolerance,
        "relationships": {
            "S11": meter_columns[0:6],
            "S12": meter_columns[6:12],
            "S21": meter_columns[12:18],
            "S22": meter_columns[18:24],
            "S1": ["S11", "S12"],
            "S2": ["S21", "S22"],
            "all": ["S1", "S2"],
        },
    }


def normalized_nwp_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def resolve_swiss_nwp_columns(frame: pd.DataFrame) -> list[Any]:
    normalized = {
        normalized_nwp_column_name(column): column for column in frame.columns
    }
    resolved: list[Any] = []
    for canonical in SWISS_NWP_CANONICAL_COLUMNS:
        key = normalized_nwp_column_name(canonical)
        if key not in normalized:
            raise ValueError(
                "Swiss NWP does not contain the expected six variables; "
                f"missing {canonical!r}, available={list(frame.columns)!r}"
            )
        resolved.append(normalized[key])
    if len(frame.columns) != len(resolved):
        raise ValueError(
            "Swiss NWP variable contract changed: "
            f"expected 6 columns, found {len(frame.columns)}"
        )
    return resolved


def align_latest_swiss_nwp(
    target_starts: pd.DatetimeIndex,
    nwp_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align each half-hour target start to its latest available NWP row."""

    if nwp_frame.empty:
        raise ValueError("Swiss NWP frame is empty")
    nwp = nwp_frame.copy()
    nwp.index = utc_index(nwp.index)
    nwp = nwp.sort_index()
    if not nwp.index.is_unique:
        nwp = nwp.loc[~nwp.index.duplicated(keep="last")]
    columns = resolve_swiss_nwp_columns(nwp)
    starts = utc_index(target_starts)
    positions = nwp.index.searchsorted(starts, side="right") - 1
    available = positions >= 0
    if not available.any():
        raise ValueError("no Swiss NWP forecast is available at any target start")

    selected_positions = positions[available]
    selected = nwp.iloc[selected_positions]
    variable_arrays: list[np.ndarray] = []
    for column in columns:
        rows = [
            np.asarray(value, dtype=float).reshape(-1)
            for value in selected[column].tolist()
        ]
        lengths = {len(row) for row in rows}
        if lengths != {24}:
            raise ValueError(
                f"Swiss NWP {column!r} must contain 24-hour arrays, got {lengths}"
            )
        variable_arrays.append(np.stack(rows, axis=0))
    cube = np.stack(variable_arrays, axis=1)
    if cube.shape[1:] != (6, 24) or not np.isfinite(cube).all():
        raise ValueError("Swiss NWP forecast cube is non-finite or malformed")
    asof = datetime64_ns_utc(selected.index)
    valid_timestamps = asof[:, None] + np.arange(24, dtype="timedelta64[h]")
    return available, asof, valid_timestamps, cube


def load_swiss_power(path: Path) -> dict[str, pd.DataFrame]:
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)  # noqa: S301 - trusted official Zenodo file
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The official Swiss pickle requires pytz. Run with "
            "`cd backend && uv run --with tables --with pytz python "
            "../scripts/prepare_paper_v7_p0_datasets.py`."
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("P_mean"), pd.DataFrame):
        raise ValueError("Swiss power_data.p does not contain a P_mean DataFrame")
    return payload


def load_swiss_nwp(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_hdf(path, "df")
    except ImportError as exc:
        raise RuntimeError(
            "Reading the official Swiss nwp_data.h5 requires PyTables. Run with "
            "`cd backend && uv run --with tables --with pytz python "
            "../scripts/prepare_paper_v7_p0_datasets.py`."
        ) from exc
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Swiss nwp_data.h5 key 'df' is not a DataFrame")
    return frame


def prepare_swiss_arrays(
    power_frame: pd.DataFrame,
    nwp_frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_audit = swiss_hierarchy_audit(power_frame)
    half_hour, aggregation_audit = complete_trailing_30_minute_means(power_frame)
    aggregate_audit = swiss_hierarchy_audit(half_hour)
    available, nwp_asof, nwp_valid, nwp_cube = align_latest_swiss_nwp(
        half_hour.index,
        nwp_frame,
    )
    aligned = half_hour.loc[available]
    meter_columns = raw_audit["meter_columns"]
    arrays = {
        "timestamps": datetime64_ns_utc(aligned.index),
        "meters": aligned[meter_columns].to_numpy(dtype=np.float64),
        "aggregates": aligned[list(SWISS_AGGREGATE_COLUMNS)].to_numpy(
            dtype=np.float64
        ),
        "nwp_asof_timestamps": nwp_asof,
        "nwp_valid_timestamps": nwp_valid,
        "nwp_forecasts": nwp_cube.astype(np.float64, copy=False),
    }
    audit = {
        "raw_hierarchy": raw_audit,
        "half_hour_hierarchy": aggregate_audit,
        "aggregation": aggregation_audit,
        "nwp_alignment": {
            "half_hour_rows_before_alignment": len(half_hour),
            "rows_with_latest_available_forecast": int(np.sum(available)),
            "rows_without_prior_forecast": int(np.sum(~available)),
            "forecast_cube_shape": list(nwp_cube.shape),
            "valid_timestamp_shape": list(nwp_valid.shape),
            "lead_zero_semantics": (
                "lead 0 is valid at the selected HDF row timestamp; leads "
                "1..23 advance in exact one-hour increments"
            ),
        },
    }
    return arrays, audit


def numeric_hour_columns(frame: pd.DataFrame) -> pd.DataFrame:
    hour_columns = [f"h{hour}" for hour in range(1, 25)]
    missing = [column for column in hour_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"GEFCom2012 frame is missing hours: {', '.join(missing)}")
    return (
        frame[hour_columns]
        .replace({",": ""}, regex=True)
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )


def gefcom_wide_to_panel(
    frame: pd.DataFrame,
    *,
    id_column: str,
) -> pd.DataFrame:
    required = {id_column, "year", "month", "day"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"GEFCom2012 frame is missing columns: {', '.join(missing)}")
    values = numeric_hour_columns(frame).to_numpy(dtype=float)
    dates = pd.to_datetime(frame[["year", "month", "day"]])
    timestamps = np.repeat(dates.to_numpy(dtype="datetime64[ns]"), 24) + np.tile(
        np.arange(1, 25, dtype="timedelta64[h]"),
        len(frame),
    )
    long = pd.DataFrame(
        {
            "timestamp": timestamps,
            "series_id": np.repeat(frame[id_column].to_numpy(), 24),
            "value": values.reshape(-1),
        }
    )
    if long.duplicated(["timestamp", "series_id"]).any():
        raise ValueError("GEFCom2012 contains duplicate series/timestamp rows")
    panel = long.pivot(
        index="timestamp",
        columns="series_id",
        values="value",
    ).sort_index()
    panel.index = utc_index(panel.index)
    panel.index.name = "timestamp"
    return panel


def parse_holiday_value(value: Any, default_year: int) -> datetime | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    without_weekday = raw.split(",", maxsplit=1)[-1].strip()
    try:
        return datetime.strptime(without_weekday, "%B %d, %Y")
    except ValueError:
        pass
    try:
        return datetime.strptime(
            f"{without_weekday}, {default_year}",
            "%B %d, %Y",
        )
    except ValueError:
        pass
    raise ValueError(f"unrecognized GEFCom2012 holiday value: {raw!r}")


def parse_gefcom_holidays(frame: pd.DataFrame) -> set[pd.Timestamp]:
    holidays: set[pd.Timestamp] = set()
    for column in frame.columns:
        if not str(column).isdigit():
            continue
        year = int(column)
        for value in frame[column]:
            parsed = parse_holiday_value(value, year)
            if parsed is not None:
                holidays.add(pd.Timestamp(parsed.date()))
    return holidays


def deterministic_calendar_covariates(
    index: pd.DatetimeIndex,
    holidays: set[pd.Timestamp],
) -> np.ndarray:
    utc = utc_index(index)
    hours = utc.hour.to_numpy(dtype=float)
    day_of_week = utc.dayofweek.to_numpy(dtype=float)
    normalized_dates = utc.tz_localize(None).normalize()
    is_holiday = np.asarray(
        [float(value in holidays) for value in normalized_dates],
        dtype=float,
    )
    return np.column_stack(
        [
            np.sin(2.0 * np.pi * hours / 24.0),
            np.cos(2.0 * np.pi * hours / 24.0),
            np.sin(2.0 * np.pi * day_of_week / 7.0),
            np.cos(2.0 * np.pi * day_of_week / 7.0),
            (day_of_week >= 5).astype(float),
            is_holiday,
        ]
    ).astype(np.float64)


def prepare_gefcom2012_arrays(
    history: pd.DataFrame,
    solution: pd.DataFrame,
    holiday_frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    history_panel = gefcom_wide_to_panel(history, id_column="zone_id")
    solution_panel = gefcom_wide_to_panel(solution, id_column="zone_id")
    expected_history_zones = list(range(1, 21))
    if list(history_panel.columns) != expected_history_zones:
        raise ValueError(
            "GEFCom2012 Load_history must contain zones 1..20 exactly"
        )
    missing_solution_zones = [
        zone for zone in range(1, 22) if zone not in solution_panel.columns
    ]
    if missing_solution_zones:
        raise ValueError(
            "GEFCom2012 Load_solution is missing zones: "
            + ", ".join(map(str, missing_solution_zones))
        )

    missing_cells_by_time = history_panel.isna().sum(axis=1)
    missing_by_time = missing_cells_by_time > 0
    if not missing_by_time.any():
        raise ValueError("GEFCom2012 expected an incomplete official tail")
    if not bool(missing_by_time.iloc[-1]):
        raise ValueError("GEFCom2012 expected a terminal incomplete tail")
    complete_positions = np.flatnonzero(~missing_by_time.to_numpy())
    if complete_positions.size == 0:
        raise ValueError("GEFCom2012 contains no complete target timestamp")
    terminal_start_position = int(complete_positions[-1] + 1)
    first_incomplete = history_panel.index[terminal_start_position]
    tail_end = first_incomplete.normalize() + pd.Timedelta(days=1)
    tail_mask = (
        missing_by_time
        & (history_panel.index >= first_incomplete)
        & (history_panel.index <= tail_end)
    )
    incomplete_tail_hours = int(tail_mask.sum())
    incomplete_tail_cells = int(missing_cells_by_time.loc[tail_mask].sum())
    if incomplete_tail_hours != 18:
        raise ValueError(
            "GEFCom2012 expected the terminal source-day tail to contain 18 hours"
        )

    retained = history_panel.loc[~missing_by_time].copy()
    if retained.empty or retained.isna().any().any():
        raise ValueError("GEFCom2012 retained history must be finite")
    retained_deltas = retained.index.to_series().diff()
    segment_breaks = retained_deltas.ne(pd.Timedelta(hours=1)).to_numpy(copy=True)
    segment_breaks[0] = True
    segment_ids = (
        np.cumsum(segment_breaks, dtype=np.int64) - 1
    ).astype(np.int32)
    segment_lengths: list[int] = []
    segment_audit: list[dict[str, Any]] = []
    for segment_id in range(int(segment_ids[-1]) + 1):
        positions = np.flatnonzero(segment_ids == segment_id)
        if positions.size == 0 or not np.array_equal(
            positions,
            np.arange(positions[0], positions[-1] + 1),
        ):
            raise ValueError("GEFCom2012 segment ids must form contiguous blocks")
        segment_index = retained.index[positions]
        if len(segment_index) > 1 and not np.all(
            np.diff(segment_index.to_numpy()) == np.timedelta64(1, "h")
        ):
            raise ValueError("GEFCom2012 retained segment is not strictly hourly")
        segment_lengths.append(len(positions))
        segment_audit.append(
            {
                "segment_id": segment_id,
                "hours": len(positions),
                "start_timestamp": segment_index[0].isoformat(),
                "end_timestamp": segment_index[-1].isoformat(),
            }
        )

    comparable = solution_panel.dropna(subset=list(range(1, 22)))
    solution_residual = comparable[21].to_numpy(dtype=float) - comparable[
        list(range(1, 21))
    ].sum(axis=1).to_numpy(dtype=float)
    solution_max_error = float(np.max(np.abs(solution_residual)))
    if solution_max_error > 1e-8:
        raise ValueError(
            "GEFCom2012 solution Zone 21 is not the exact sum of zones 1..20"
        )

    zones = retained.to_numpy(dtype=np.float64)
    total = np.sum(zones, axis=1)
    sum_1_10 = np.sum(zones[:, 0:10], axis=1)
    sum_11_20 = np.sum(zones[:, 10:20], axis=1)
    hierarchy = np.column_stack([total, sum_1_10, sum_11_20])
    hierarchy_error = float(np.max(np.abs(total - sum_1_10 - sum_11_20)))
    holidays = parse_gefcom_holidays(holiday_frame)
    covariates = deterministic_calendar_covariates(retained.index, holidays)
    arrays = {
        "timestamps": datetime64_ns_utc(retained.index),
        "segment_ids": segment_ids,
        "zones": zones,
        "total": total.astype(np.float64, copy=False),
        "canonical_hierarchy": hierarchy.astype(np.float64, copy=False),
        "calendar_covariates": covariates,
    }
    audit = {
        "retained_history_hours": len(retained),
        "excluded_target_missing_hours": int(missing_by_time.sum()),
        "excluded_target_missing_cells": int(missing_cells_by_time.sum()),
        "excluded_official_hidden_hours": int(
            missing_by_time.sum() - incomplete_tail_hours
        ),
        "excluded_official_hidden_cells": int(
            missing_cells_by_time.sum() - incomplete_tail_cells
        ),
        "excluded_internal_hidden_hours": int(
            missing_by_time.loc[history_panel.index < first_incomplete].sum()
        ),
        "excluded_post_tail_evaluation_hours": int(
            missing_by_time.loc[history_panel.index > tail_end].sum()
        ),
        "excluded_incomplete_tail_hours": incomplete_tail_hours,
        "excluded_incomplete_tail_cells": incomplete_tail_cells,
        "first_incomplete_tail_timestamp": first_incomplete.isoformat(),
        "last_retained_timestamp": retained.index[-1].isoformat(),
        "segment_count": len(segment_lengths),
        "segment_lengths": segment_lengths,
        "segments": segment_audit,
        "solution_zone21_comparable_hours": len(comparable),
        "solution_zone21_max_absolute_error": solution_max_error,
        "canonical_hierarchy_max_absolute_error": hierarchy_error,
        "holiday_dates_in_source": len(holidays),
    }
    return arrays, audit


def read_gefcom2012_frames(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    members = {
        "history": "GEFCOM2012_Data/Load/Load_history.csv",
        "solution": "GEFCOM2012_Data/Load/Load_solution.csv",
        "holidays": "GEFCOM2012_Data/Load/Holiday_List.csv",
    }
    with zipfile.ZipFile(path) as archive:
        missing = [member for member in members.values() if member not in archive.namelist()]
        if missing:
            raise ValueError(
                f"GEFCom2012 archive is missing members: {', '.join(missing)}"
            )
        return tuple(
            pd.read_csv(archive.open(members[key]))
            for key in ("history", "solution", "holidays")
        )


def validate_numeric_npz_arrays(arrays: dict[str, np.ndarray]) -> None:
    if not arrays:
        raise ValueError("cannot write an empty NPZ")
    lengths: set[int] = set()
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"NPZ array {name!r} has forbidden object dtype")
        if array.ndim == 0:
            raise ValueError(f"NPZ array {name!r} must have a row dimension")
        lengths.add(array.shape[0])
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"NPZ array {name!r} contains non-finite values")
    if len(lengths) != 1:
        raise ValueError(f"NPZ arrays do not share a row count: {sorted(lengths)}")


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    validate_numeric_npz_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".npz",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    try:
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def array_inventory(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for name, value in arrays.items()
    }


def prepare_swiss(
    power_path: Path,
    nwp_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    payload = load_swiss_power(power_path)
    arrays, audit = prepare_swiss_arrays(
        payload["P_mean"],
        load_swiss_nwp(nwp_path),
    )
    npz_path = output_dir / "swiss_hierarchical_demand.npz"
    metadata_path = output_dir / "swiss_hierarchical_demand.metadata.json"
    write_npz_atomic(npz_path, arrays)
    meter_columns = audit["raw_hierarchy"]["meter_columns"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "swiss_hierarchical_demand",
        "dataset_name": "Hierarchical Demand Forecasting Benchmark for the Distribution Grid",
        "license": {
            "spdx": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "source_record_url": SWISS_SOURCE_URL,
        "source_files": [
            source_file_record(power_path, SWISS_POWER_URL),
            source_file_record(nwp_path, SWISS_NWP_URL),
        ],
        "frequency": "30min",
        "target_bin_semantics": (
            "mean of three 10-minute interval-ending measurements; output "
            "timestamp is the half-hour bin start; incomplete bins are dropped"
        ),
        "meter_columns": meter_columns,
        "aggregate_columns": list(SWISS_AGGREGATE_COLUMNS),
        "nwp_variable_order": list(SWISS_NWP_CANONICAL_COLUMNS),
        "nwp_forecast_horizon_hours": 24,
        "nwp_release_semantics": {
            "provider": "Meteoblue",
            "documented_update_cadence": "12h",
            "source_contract": (
                "each HDF row contains the most recent 24-hour forecast "
                "available at that row's timestamp"
            ),
            "alignment_rule": (
                "for each 30-minute target bin start, select the latest HDF "
                "row timestamp not later than the bin start"
            ),
            "stored_asof_array": "nwp_asof_timestamps",
            "stored_valid_time_array": "nwp_valid_timestamps",
            "valid_time_rule": (
                "valid[time, lead] = nwp_asof_timestamps[time] + lead hours; "
                "lead 0 is valid at the HDF row timestamp"
            ),
            "rolling_origin_rule": (
                "because power timestamps are half-hour bin starts, the "
                "forecast after history ending at origin uses the processed "
                "row origin+1, whose lead-0 valid time equals the first "
                "forecast target timestamp; every hourly lead is repeated "
                "over its two half-hour target bins"
            ),
            "no_forecast_stitching": True,
        },
        "hierarchy": {
            "native": True,
            "relationships": audit["raw_hierarchy"]["relationships"],
        },
        "audit": audit,
        "arrays": array_inventory(arrays),
        "output_npz": {
            "path": str(npz_path.resolve()),
            "bytes": npz_path.stat().st_size,
            "sha256": sha256_file(npz_path),
            "allow_pickle": False,
        },
    }
    write_json_atomic(metadata_path, metadata)
    return {
        "npz": str(npz_path),
        "metadata": str(metadata_path),
        "npz_sha256": metadata["output_npz"]["sha256"],
        "arrays": metadata["arrays"],
    }


def prepare_gefcom2012(path: Path, output_dir: Path) -> dict[str, Any]:
    history, solution, holidays = read_gefcom2012_frames(path)
    arrays, audit = prepare_gefcom2012_arrays(history, solution, holidays)
    npz_path = output_dir / "gefcom2012_load.npz"
    metadata_path = output_dir / "gefcom2012_load.metadata.json"
    write_npz_atomic(npz_path, arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "gefcom2012_load",
        "dataset_name": "GEFCom2012 Load Forecasting",
        "usage_status": {
            "status": "user_confirmed_usable",
            "confirmed_on": "2026-07-21",
            "license_note": (
                "The official archive contains no explicit license file; use "
                "in this experiment was explicitly confirmed by the user."
            ),
        },
        "source_record_url": GEFCOM2012_SOURCE_URL,
        "source_files": [
            source_file_record(path, GEFCOM2012_DOWNLOAD_URL),
        ],
        "frequency": "h",
        "source_hour_semantics": (
            "h1..h24 are hour-ending timestamps 01:00..next-day 00:00"
        ),
        "zone_columns": list(GEFCOM_ZONE_COLUMNS),
        "derived_total_column": "total",
        "canonical_hierarchy_columns": list(GEFCOM_HIERARCHY_COLUMNS),
        "canonical_hierarchy": {
            "native_total_validation": (
                "official solution Zone 21 equals the sum of zones 1..20"
            ),
            "stored_projection": [
                "total = sum(zones 1..20)",
                "sum_zones_1_10 = sum(zones 1..10)",
                "sum_zones_11_20 = sum(zones 11..20)",
                "total = sum_zones_1_10 + sum_zones_11_20",
            ],
        },
        "calendar_covariate_columns": list(GEFCOM_CALENDAR_COLUMNS),
        "known_future_covariate_policy": {
            "included": list(GEFCOM_CALENDAR_COLUMNS),
            "temperature_excluded": (
                "temperature observations/solutions have no issue-time "
                "forecast-vintage contract for arbitrary rolling origins"
            ),
        },
        "history_policy": (
            "Load_solution is never used as a target source. Every "
            "Load_history timestamp with any missing zone is excluded. "
            "Numeric segment_ids freeze the remaining strict-hourly "
            "components, and downstream windows may not cross segment gaps. "
            "Load_solution is used only to audit that Zone 21 equals the sum "
            "of zones 1..20."
        ),
        "audit": audit,
        "arrays": array_inventory(arrays),
        "output_npz": {
            "path": str(npz_path.resolve()),
            "bytes": npz_path.stat().st_size,
            "sha256": sha256_file(npz_path),
            "allow_pickle": False,
        },
    }
    write_json_atomic(metadata_path, metadata)
    return {
        "npz": str(npz_path),
        "metadata": str(metadata_path),
        "npz_sha256": metadata["output_npz"]["sha256"],
        "arrays": metadata["arrays"],
    }


def main() -> None:
    args = parse_args()
    for path in (args.swiss_power, args.swiss_nwp, args.gefcom2012_zip):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "swiss": prepare_swiss(
            args.swiss_power,
            args.swiss_nwp,
            args.output_dir,
        ),
        "gefcom2012": prepare_gefcom2012(
            args.gefcom2012_zip,
            args.output_dir,
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
