from __future__ import annotations

from math import ceil

import numpy as np
from scipy.signal import periodogram

from .constants import FEATURE_COLUMNS


def _clean(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 8:
        raise ValueError("series is too short")
    return arr


def infer_season_length(values: np.ndarray) -> int:
    x = _clean(values)
    x = x - np.mean(x)
    max_lag = min(max(2, len(x) // 3), 96)
    if max_lag <= 2:
        return 1
    denom = np.dot(x, x) + 1e-12
    acfs = [(lag, float(np.dot(x[:-lag], x[lag:]) / denom)) for lag in range(2, max_lag + 1)]
    lag, value = max(acfs, key=lambda item: item[1])
    return int(lag if value > 0.25 else 1)


def infer_dominant_scale(values: np.ndarray) -> int:
    season_length = infer_season_length(values)
    if season_length > 1:
        return season_length
    n = len(values)
    freqs, power = periodogram(values)
    valid = (freqs > 0) & np.isfinite(power)
    if valid.sum() == 0:
        return min(max(12, n // 8), 96)
    dom = float(freqs[valid][np.argmax(power[valid])])
    if dom <= 0:
        return min(max(12, n // 8), 96)
    return int(np.clip(round(1 / dom), 2, 96))


def spectral_entropy(values: np.ndarray) -> float:
    _, power = periodogram(_clean(values))
    power = power[np.isfinite(power) & (power > 0)]
    if len(power) == 0:
        return 1.0
    probs = power / power.sum()
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    return float(entropy / np.log(len(probs)))


def trend_strength(values: np.ndarray) -> float:
    x = _clean(values)
    t = np.arange(len(x), dtype=float)
    coeffs = np.polyfit(t, x, deg=1)
    trend = np.polyval(coeffs, t)
    resid = x - trend
    return float(np.clip(1.0 - np.var(resid) / (np.var(x) + 1e-12), 0.0, 1.0))


def seasonal_strength(values: np.ndarray, season_length: int) -> float:
    x = _clean(values)
    if season_length <= 1 or len(x) < season_length * 2:
        return 0.0
    groups = [x[offset::season_length] for offset in range(season_length)]
    means = np.array([group.mean() if len(group) else 0.0 for group in groups])
    seasonal = np.resize(means - means.mean(), len(x))
    resid = x - seasonal
    return float(np.clip(1.0 - np.var(resid) / (np.var(x) + 1e-12), 0.0, 1.0))


def acf_half_life(values: np.ndarray) -> float:
    x = _clean(values)
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    max_lag = min(max(4, len(x) // 2), 128)
    for lag in range(1, max_lag + 1):
        acf = float(np.dot(x[:-lag], x[lag:]) / denom)
        if acf < 0.5:
            return float(lag)
    return float(max_lag)


def changepoint_density(values: np.ndarray) -> float:
    x = _clean(values)
    window = max(8, ceil(len(x) / 20))
    if len(x) <= window * 2:
        return 0.0
    threshold = x.std() * 0.75
    count = 0
    for idx in range(window, len(x) - window):
        left = x[idx - window : idx].mean()
        right = x[idx : idx + window].mean()
        if abs(right - left) > threshold:
            count += 1
    return float(count / max(1, len(x) - 2 * window))


def variance_shift(values: np.ndarray) -> float:
    x = _clean(values)
    window = max(10, len(x) // 12)
    if len(x) < window * 2:
        return 0.0
    step = max(1, window // 2)
    vars_ = np.array([np.var(x[i : i + window]) for i in range(0, len(x) - window + 1, step)])
    if len(vars_) == 0:
        return 0.0
    return float(np.clip((vars_.max() + 1e-12) / (vars_.min() + 1e-12) - 1.0, 0.0, 10.0))


def intermittency(values: np.ndarray) -> float:
    x = _clean(values)
    threshold = max(1e-8, np.quantile(np.abs(x), 0.1))
    return float(np.mean(np.abs(x) <= threshold))


def outlier_rate(values: np.ndarray) -> float:
    x = _clean(values)
    median = np.median(x)
    mad = np.median(np.abs(x - median)) + 1e-12
    scores = np.abs(x - median) / mad
    return float(np.mean(scores > 6.0))


def extract_features(values: np.ndarray) -> dict[str, float]:
    season_length = infer_season_length(values)
    return {
        "trend_strength": trend_strength(values),
        "seasonal_strength": seasonal_strength(values, season_length),
        "spectral_entropy": spectral_entropy(values),
        "acf_half_life": acf_half_life(values),
        "changepoint_density": changepoint_density(values),
        "variance_shift": variance_shift(values),
        "intermittency": intermittency(values),
        "outlier_rate": outlier_rate(values),
    }


def feature_vector(feature_map: dict[str, float]) -> np.ndarray:
    return np.array([float(feature_map[name]) for name in FEATURE_COLUMNS], dtype=float)

