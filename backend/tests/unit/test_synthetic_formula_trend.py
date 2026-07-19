from __future__ import annotations

import numpy as np

from app.services.synthetic_capability_contrast import (
    capability_contrast_forecasts,
)
from app.services.synthetic_generation_service import (
    _generate_sample_values,
    _single_target_profile,
)


CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24


def _generate(seed: int, intensity: int):
    return _generate_sample_values(
        "trend",
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        intensity,
        np.random.default_rng(seed),
    )


def test_trend_intensity_only_scales_a_seed_fixed_quadratic_basis():
    low_values, low_latent, _ = _generate(seed=37, intensity=1)
    high_values, high_latent, _ = _generate(seed=37, intensity=5)

    assert low_latent["predictability"]["evidence"]["trend_basis"] == (
        "dataset_relative_fixed_quadratic"
    )
    assert low_latent["trend_direction_by_target"] == high_latent[
        "trend_direction_by_target"
    ]
    assert low_latent["trend_shape_scale_by_target"] == high_latent[
        "trend_shape_scale_by_target"
    ]
    assert low_latent["curvature_ratio_by_target"] == high_latent[
        "curvature_ratio_by_target"
    ]

    low_slope = np.asarray(low_latent["slope_by_target"])
    high_slope = np.asarray(high_latent["slope_by_target"])
    low_curvature = np.asarray(low_latent["curvature_by_target"])
    high_curvature = np.asarray(high_latent["curvature_by_target"])
    np.testing.assert_allclose(
        high_slope / low_slope,
        high_curvature / low_curvature,
    )

    time = (
        np.arange(CONTEXT_LENGTH + HORIZON, dtype=float)
        - (CONTEXT_LENGTH - 1)
    ) / SEASON_LENGTH
    low_trend = (
        time[:, None] * low_slope[None, :]
        + time[:, None] ** 2 * low_curvature[None, :]
    )
    high_trend = (
        time[:, None] * high_slope[None, :]
        + time[:, None] ** 2 * high_curvature[None, :]
    )
    np.testing.assert_allclose(
        low_values - low_trend,
        high_values - high_trend,
    )


def test_trend_curvature_shape_is_fixed_across_samples():
    ratios = np.asarray(
        [
            _generate(seed=seed, intensity=5)[1][
                "curvature_ratio_by_target"
            ][0]
            for seed in range(32)
        ]
    )

    np.testing.assert_allclose(ratios, 0.06)


def test_high_intensity_trend_is_visible_and_predictably_continued():
    low_strengths: list[float] = []
    high_strengths: list[float] = []
    aware_errors: list[float] = []
    blind_errors: list[float] = []

    for seed in range(64):
        low_values, _, _ = _generate(seed=seed, intensity=1)
        high_values, high_latent, _ = _generate(seed=seed, intensity=5)
        low_strengths.append(
            _single_target_profile(
                low_values[:CONTEXT_LENGTH, 0],
                SEASON_LENGTH,
            )["trend_strength"]
        )
        high_strengths.append(
            _single_target_profile(
                high_values[:CONTEXT_LENGTH, 0],
                SEASON_LENGTH,
            )["trend_strength"]
        )

        forecasts = capability_contrast_forecasts(
            capability_id="trend",
            history=high_values[:CONTEXT_LENGTH],
            horizon=HORIZON,
            season_length=SEASON_LENGTH,
            latent_params=high_latent,
        )
        future = high_values[CONTEXT_LENGTH:]
        aware_errors.append(float(np.mean(np.abs(future - forecasts["aware"]))))
        blind_errors.append(float(np.mean(np.abs(future - forecasts["blind"]))))

    assert float(np.mean(high_strengths)) > float(np.mean(low_strengths)) + 0.2
    assert float(np.mean(aware_errors)) < 0.8 * float(np.mean(blind_errors))
