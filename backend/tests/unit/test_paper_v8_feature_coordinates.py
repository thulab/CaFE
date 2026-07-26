from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import paper_v8_features as features  # noqa: E402


def _persistent_common_background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = np.zeros(168, dtype=float)
    innovations = rng.normal(size=latent.size)
    for index in range(1, latent.size):
        latent[index] = 0.85 * latent[index - 1] + innovations[index]
    return latent[:, None] + rng.normal(scale=0.6, size=(latent.size, 4))


def test_xsd_coordinate_removes_independent_search_floor() -> None:
    corrected: list[float] = []
    raw: list[float] = []
    for seed in range(8):
        values = np.random.default_rng(seed).normal(size=(168, 4))
        raw.append(
            float(
                np.nanmean(
                    features._best_cross_series_holdout_gains(
                        values,
                        max_lag=48,
                    )
                )
            )
        )
        corrected.append(
            features._paper_v8_cross_series_incremental_r2(
                values,
                max_lag=48,
            )
        )

    assert np.mean(raw) > 0.06
    assert np.mean(corrected) < 0.02
    assert np.mean(corrected) < 0.25 * np.mean(raw)


def test_xsd_coordinate_treats_reversible_common_background_as_null() -> None:
    scores = [
        features._paper_v8_cross_series_incremental_r2(
            _persistent_common_background(seed),
            max_lag=48,
        )
        for seed in range(8)
    ]

    assert np.mean(scores) < 0.05
    assert max(scores) < 0.10


def test_xsd_coordinate_preserves_strong_directed_lag_monotonicity() -> None:
    rng = np.random.default_rng(91)
    driver = rng.normal(size=168)
    response_noise = rng.normal(scale=0.65, size=168)
    distractors = rng.normal(size=(168, 2))
    lag = 8
    scores: list[float] = []
    for strength in (0.0, 0.5, 1.0, 1.5):
        response = response_noise.copy()
        response[lag:] += strength * driver[:-lag]
        values = np.column_stack([driver, response, distractors])
        scores.append(
            features._paper_v8_cross_series_incremental_r2(
                values,
                max_lag=48,
            )
        )

    assert scores == sorted(scores)
    assert scores[0] < 0.01
    assert scores[-1] > 0.18


def test_v8_feature_vector_uses_deterministic_corrected_xsd_coordinate() -> None:
    values = _persistent_common_background(27)
    direct = features._paper_v8_cross_series_incremental_r2(
        values,
        max_lag=48,
    )

    first = features.v8_feature_vector(
        values,
        season_length=24,
        cross_series_max_lag=48,
    )
    second = features.v8_feature_vector(
        values,
        season_length=24,
        cross_series_max_lag=48,
    )

    assert first["cross_series_incremental_r2"] == pytest.approx(direct)
    assert second["cross_series_incremental_r2"] == pytest.approx(direct)
    assert first["v8_feature_history_length"] == 168.0
