from __future__ import annotations

import numpy as np


def mase(history: np.ndarray, target: np.ndarray, forecast: np.ndarray, season_length: int) -> float:
    history = np.asarray(history, dtype=float)
    target = np.asarray(target, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(forecast) != len(target):
        forecast = np.resize(forecast, len(target))
    m = max(1, int(season_length))
    if len(history) <= m:
        scale = np.mean(np.abs(np.diff(history))) + 1e-8 if len(history) > 1 else 1.0
    else:
        scale = np.mean(np.abs(history[m:] - history[:-m])) + 1e-8
    return float(np.mean(np.abs(target - forecast)) / scale)


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(forecast) != len(target):
        forecast = np.resize(forecast, len(target))
    denom = np.abs(target) + np.abs(forecast) + 1e-8
    return float(200.0 * np.mean(np.abs(target - forecast) / denom))


def relative_skill(model_mase: float, baseline_mase: float) -> float:
    return float(1.0 - model_mase / (baseline_mase + 1e-8))

