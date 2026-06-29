#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
)
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
BOUNDED_FEATURES = {
    "trend_strength",
    "seasonal_strength",
    "noise_ratio",
    "outlier_rate",
    "spike_rate",
    "avg_abs_target_corr",
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


def read_csv_series(path: Path, time_column: str, target_columns: list[str] | None) -> tuple[pd.Series, pd.DataFrame]:
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
    return time, targets.astype(float)


def read_tsf_series(path: Path) -> tuple[dict[str, str], list[tuple[str, np.ndarray]]]:
    text = read_text_or_first_tsf_from_zip(path)
    attributes: list[str] = []
    metadata: dict[str, str] = {}
    series: list[tuple[str, np.ndarray]] = []
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

        pieces = line.split(":")
        if len(pieces) < len(attributes) + 1:
            continue
        attr_values = pieces[: len(attributes)]
        values_text = ":".join(pieces[len(attributes) :])
        values = parse_tsf_values(values_text)
        if values.size:
            series_id = attr_values[0] if attr_values else f"series_{len(series)}"
            series.append((series_id, values))
    if not series:
        raise ValueError(f"no series found in TSF input: {path}")
    return metadata, series


def read_text_or_first_tsf_from_zip(path: Path) -> str:
    if path.suffix.lower() != ".zip":
        return path.read_text(encoding="utf-8")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".tsf")]
        if not names:
            raise ValueError(f"zip does not contain a .tsf file: {path}")
        with archive.open(names[0]) as handle:
            return handle.read().decode("utf-8")


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
    if len(values) < spec.length:
        return []
    starts = list(range(0, len(values) - spec.length + 1, spec.stride))
    if spec.max_windows is not None and len(starts) > spec.max_windows:
        indexes = np.linspace(0, len(starts) - 1, spec.max_windows).round().astype(int)
        starts = [starts[index] for index in sorted(set(indexes.tolist()))]
    return [(start, values[start : start + spec.length]) for start in starts]


def profile_csv(
    path: Path,
    *,
    time_column: str = "time",
    target_columns: list[str] | None = None,
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
    time, targets = read_csv_series(path, time_column, target_columns)
    resolved_stride = stride or horizon
    spec = WindowSpec(context_length, horizon, resolved_stride, max_windows=max_windows)
    values = targets.to_numpy(dtype=float)
    windows = iter_windows(values, spec)
    feature_rows: list[dict[str, float]] = []
    for start, window in windows:
        features = feature_vector(window, season_length=season_length)
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
            "covariate_dim": 0,
            "season_length": season_length,
        },
        "window_count": len(feature_rows),
        "target_columns": list(targets.columns),
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
    spec = WindowSpec(context_length, horizon, resolved_stride, max_windows=max_windows)
    feature_rows: list[dict[str, float]] = []
    used_series = 0
    for _series_id, values in series:
        windows = iter_windows(values[:, None], spec)
        for start, window in windows:
            if not np.isfinite(window).all():
                continue
            features = feature_vector(window, season_length=season_length)
            features["window_start"] = float(start)
            feature_rows.append(features)
        if windows:
            used_series += 1

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
        "series_count": len(series),
        "used_series_count": used_series,
        "target_columns": ["target"],
        "features": quantiles,
        "target_feature_caps": caps,
    }


def profile_input(
    path: Path,
    *,
    input_format: str = "auto",
    time_column: str = "time",
    target_columns: list[str] | None = None,
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


def resolve_input_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    suffix = path.suffix.lower()
    if suffix == ".tsf":
        return "tsf"
    if suffix == ".zip":
        return "tsf"
    return "csv"


def feature_vector(window: np.ndarray, season_length: int | None = None) -> dict[str, float]:
    if window.ndim == 1:
        window = window[:, None]
    per_target = [single_series_features(window[:, index], season_length=season_length) for index in range(window.shape[1])]
    out: dict[str, float] = {}
    for feature in DEFAULT_FEATURES:
        values = [row[feature] for row in per_target if feature in row and math.isfinite(row[feature])]
        if values:
            out[feature] = float(np.mean(values))
    if window.shape[1] > 1:
        corr = np.nan_to_num(np.corrcoef(window.T), nan=0.0)
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        out["avg_abs_target_corr"] = float(np.mean(np.abs(off_diag))) if off_diag.size else 0.0
    return out


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
    preferred = ["trend_strength", "seasonal_strength", "slope_abs", "curvature_abs"]
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
        context_length=args.context_length,
        horizon=args.horizon,
        stride=args.stride,
        max_windows=args.max_windows,
        season_length=args.season_length,
        domain=args.domain,
        dataset_name=args.dataset_name,
        target_features=args.target_features,
        target_max_multiplier=args.target_max_multiplier,
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
