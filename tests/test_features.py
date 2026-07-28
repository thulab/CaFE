from __future__ import annotations

import numpy as np
import pytest

from cafe.features import profile as features


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
            features._cafe_cross_series_incremental_r2(
                values,
                max_lag=48,
            )
        )

    assert np.mean(raw) > 0.06
    assert np.mean(corrected) < 0.02
    assert np.mean(corrected) < 0.25 * np.mean(raw)


def test_xsd_coordinate_treats_reversible_common_background_as_null() -> None:
    scores = [
        features._cafe_cross_series_incremental_r2(
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
            features._cafe_cross_series_incremental_r2(
                values,
                max_lag=48,
            )
        )

    assert scores == sorted(scores)
    assert scores[0] < 0.01
    assert scores[-1] > 0.18


def test_cafe_feature_vector_uses_deterministic_corrected_xsd_coordinate() -> None:
    values = _persistent_common_background(27)
    direct = features._cafe_cross_series_incremental_r2(
        values,
        max_lag=48,
    )

    first = features.cafe_feature_vector(
        values,
        season_length=24,
        cross_series_max_lag=48,
    )
    second = features.cafe_feature_vector(
        values,
        season_length=24,
        cross_series_max_lag=48,
    )

    assert first["cross_series_incremental_r2"] == pytest.approx(direct)
    assert second["cross_series_incremental_r2"] == pytest.approx(direct)
    assert first["cafe_feature_history_length"] == 168.0


def test_cross_series_effect_memory_distinguishes_persistent_transfer() -> None:
    rng = np.random.default_rng(113)
    driver = rng.normal(size=1024)
    direct = np.zeros_like(driver)
    direct[1:] = driver[:-1]
    persistent = np.zeros_like(driver)
    for index in range(1, len(driver)):
        persistent[index] = (
            0.96 * persistent[index - 1] + driver[index - 1]
        )

    direct_memory = features._cafe_cross_series_effect_memory(
        np.column_stack([driver, direct]),
        max_lag=24,
    )
    persistent_memory = features._cafe_cross_series_effect_memory(
        np.column_stack([driver, persistent]),
        max_lag=24,
    )

    assert direct_memory < 0.20
    assert persistent_memory > 0.50
    assert persistent_memory > direct_memory + 0.40
