from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .baselines import proxy_predictors
from .families import FAMILY_GENERATORS
from .features import infer_dominant_scale, infer_season_length
from .metrics import mase


@dataclass(slots=True)
class FamilyCalibration:
    family: str
    x_thresholds: list[float]
    y_thresholds: list[float]
    bin_edges: list[float]

    def score(self, control_lambda: float) -> float:
        return float(np.interp(control_lambda, self.x_thresholds, self.y_thresholds))

    def difficulty_bin(self, control_lambda: float) -> int:
        score = self.score(control_lambda)
        return int(np.digitize(score, self.bin_edges[1:-1], right=False) + 1)


def _proxy_difficulty(history: np.ndarray, target: np.ndarray, season_length: int) -> float:
    scores = []
    for predictor in proxy_predictors():
        forecast = predictor.predict(history, len(target), season_length)
        scores.append(mase(history, target, forecast, season_length))
    return float(np.asarray(scores, dtype=float).mean())


def calibrate_family(
    family: str,
    n_candidates: int,
    horizon_ratio: float,
    seed: int,
) -> tuple[FamilyCalibration, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    generator = FAMILY_GENERATORS[family]
    rows = []
    for idx in range(n_candidates):
        control_lambda = float(rng.uniform(0.0, 1.0))
        dominant_scale = int(rng.integers(16, 64))
        season_length = max(1, dominant_scale if family in {"multi_seasonal", "intermittent_heteroskedastic", "trend"} else dominant_scale // 2)
        horizon = int(np.clip(round(horizon_ratio * dominant_scale), 12, 96))
        context = int(min(8 * horizon, 512))
        total_length = max(context + horizon + 32, 240)
        output = generator(
            length=total_length,
            season_length=season_length,
            control_lambda=control_lambda,
            rng=rng,
            anchor_features={"spectral_entropy": rng.uniform(0.2, 0.9)},
        )
        values = output.values[-(context + horizon) :]
        history = values[:context]
        target = values[context:]
        proxy = _proxy_difficulty(history, target, infer_season_length(values))
        rows.append(
            {
                "family": family,
                "lambda": control_lambda,
                "proxy_difficulty": proxy,
                "dominant_scale": infer_dominant_scale(values),
                "season_length": infer_season_length(values),
                "candidate_id": idx,
            }
        )
    frame = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    scores = iso.fit_transform(frame["lambda"], frame["proxy_difficulty"])
    frame["difficulty_score"] = scores
    _, bin_edges = pd.qcut(scores, q=5, labels=False, retbins=True, duplicates="drop")
    if len(bin_edges) < 6:
        bin_edges = np.linspace(float(np.min(scores)), float(np.max(scores)) + 1e-9, 6)
    calibration = FamilyCalibration(
        family=family,
        x_thresholds=[float(x) for x in frame["lambda"]],
        y_thresholds=[float(y) for y in frame["difficulty_score"]],
        bin_edges=[float(x) for x in bin_edges],
    )
    frame["difficulty_bin"] = [calibration.difficulty_bin(value) for value in frame["lambda"]]
    return calibration, frame

