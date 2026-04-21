from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(slots=True)
class Predictor:
    name: str

    def predict(self, history: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
        raise NotImplementedError


class LastValuePredictor(Predictor):
    def __init__(self) -> None:
        super().__init__(name="last_value")

    def predict(self, history: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
        return np.repeat(float(np.asarray(history, dtype=float)[-1]), horizon)


class SeasonalNaivePredictor(Predictor):
    def __init__(self) -> None:
        super().__init__(name="seasonal_naive")

    def predict(self, history: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        m = max(1, int(season_length))
        if m <= 1 or len(history) < m:
            return np.repeat(float(history[-1]), horizon)
        return np.tile(history[-m:], int(np.ceil(horizon / m)))[:horizon]


class AutoThetaPredictor(Predictor):
    def __init__(self) -> None:
        super().__init__(name="auto_theta")

    def predict(self, history: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if len(history) < 8:
            return np.repeat(float(history[-1]), horizon)
        try:
            from statsmodels.tsa.api import ExponentialSmoothing
            from statsmodels.tsa.forecasting.theta import ThetaModel

            if season_length > 1 and len(history) >= season_length * 2:
                fit = ExponentialSmoothing(
                    history,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=int(season_length),
                    initialization_method="estimated",
                ).fit(optimized=True)
                return np.asarray(fit.forecast(horizon), dtype=float)
            fit = ThetaModel(history, period=max(1, int(season_length))).fit()
            return np.asarray(fit.forecast(horizon), dtype=float)
        except Exception:
            return np.repeat(float(history[-1]), horizon)


class RidgeARPredictor(Predictor):
    def __init__(self, lags: int = 24) -> None:
        super().__init__(name="ridge_ar")
        self.lags = lags

    def predict(self, history: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        lag = max(4, min(self.lags, max(4, len(history) // 3)))
        if len(history) <= lag + 2:
            return np.repeat(float(history[-1]), horizon)
        x_rows, y_rows = [], []
        for idx in range(lag, len(history)):
            x_rows.append(history[idx - lag : idx])
            y_rows.append(history[idx])
        model = Ridge(alpha=1.0)
        model.fit(np.asarray(x_rows), np.asarray(y_rows))
        buffer = history[-lag:].copy()
        preds = []
        for _ in range(horizon):
            pred = float(model.predict(buffer.reshape(1, -1))[0])
            preds.append(pred)
            buffer = np.concatenate([buffer[1:], [pred]])
        return np.asarray(preds, dtype=float)


def proxy_predictors() -> list[Predictor]:
    return [LastValuePredictor(), SeasonalNaivePredictor(), AutoThetaPredictor(), RidgeARPredictor()]


def baseline_by_name(name: str) -> Predictor:
    registry = {predictor.name: predictor for predictor in proxy_predictors()}
    if name not in registry:
        raise KeyError(f"unknown baseline predictor: {name}")
    return registry[name]
