from __future__ import annotations

import copy

import numpy as np
import pytest

from app.services.synthetic_capability_contrast import (
    capability_contrast_forecasts,
    evaluate_capability_contrast,
    summarize_capability_contrasts,
)
from app.services.synthetic_generation_service import (
    _generate_accepted_sample_values,
    _generate_sample_values,
    _realized_features,
    _seed_for,
    _standardize_by_context,
)


CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24


def _generate(seed: int, intensity: int):
    return _generate_sample_values(
        "covariate_response",
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        intensity,
        np.random.default_rng(seed),
    )


def _features(
    values: np.ndarray,
    covariates: np.ndarray,
) -> dict[str, float]:
    return _realized_features(
        _standardize_by_context(values, CONTEXT_LENGTH),
        covariates,
        SEASON_LENGTH,
        CONTEXT_LENGTH,
    )


def test_event_schedule_is_sample_specific_and_fully_observed():
    schedules: set[tuple[int, ...]] = set()
    widths: set[int] = set()
    for seed in range(32):
        _, metadata, covariates = _generate(seed, 5)
        assert covariates is not None
        starts = tuple(int(value) for value in metadata["event_starts"])
        width = int(metadata["event_width"])
        schedules.add(starts)
        widths.add(width)

        historical = starts[:-1]
        future = starts[-1]
        assert len(historical) == 3
        assert all(
            0 <= start <= CONTEXT_LENGTH - width
            for start in historical
        )
        assert all(
            right - left >= width
            for left, right in zip(historical, historical[1:])
        )
        assert CONTEXT_LENGTH <= future
        assert future + width <= CONTEXT_LENGTH + HORIZON
        assert np.sum(covariates[:, 1]) == pytest.approx(4 * width)
        assert metadata["predictability"]["construction_validated"] is True

    assert len(schedules) == 32
    assert len(widths) >= 3


def test_intensity_only_scales_the_fixed_covariate_effect():
    low_values, low, low_covariates = _generate(41, 1)
    high_values, high, high_covariates = _generate(41, 5)
    assert low_covariates is not None
    assert high_covariates is not None

    np.testing.assert_allclose(low_covariates, high_covariates, atol=0.0)
    assert low["event_starts"] == high["event_starts"]
    assert low["event_width"] == high["event_width"]
    assert (
        low["weather_effect_ratio_by_target"]
        == high["weather_effect_ratio_by_target"]
    )
    assert (
        low["event_effect_ratio_by_target"]
        == high["event_effect_ratio_by_target"]
    )
    assert low["weather_process"] == high["weather_process"]
    assert low["baseline_process"] == high["baseline_process"]
    assert (
        low["covariate_path_checksum"]
        == high["covariate_path_checksum"]
    )
    assert (
        low["nuisance_component_checksum"]
        == high["nuisance_component_checksum"]
    )

    low_strength = float(low["effect_strength"])
    high_strength = float(high["effect_strength"])
    weather_ratio = np.asarray(
        low["weather_effect_ratio_by_target"],
        dtype=float,
    )
    event_ratio = np.asarray(
        low["event_effect_ratio_by_target"],
        dtype=float,
    )
    unit_effect = (
        low_covariates[:, :1] * weather_ratio[None, :]
        + low_covariates[:, 1:2] * event_ratio[None, :]
    )
    np.testing.assert_allclose(
        high_values - low_values,
        (high_strength - low_strength) * unit_effect,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        low_values - low_strength * unit_effect,
        high_values - high_strength * unit_effect,
        atol=1e-12,
    )


def test_covariate_strength_is_monotone_and_high_level_is_visible():
    incremental_r2_means: list[float] = []
    event_lift_means: list[float] = []
    for intensity in range(1, 6):
        rows = []
        for seed in range(48):
            values, _, covariates = _generate(seed, intensity)
            assert covariates is not None
            rows.append(_features(values, covariates))
        incremental_r2_means.append(
            float(np.mean([row["covariate_incremental_r2"] for row in rows]))
        )
        event_lift_means.append(
            float(np.mean([row["event_lift_abs"] for row in rows]))
        )

    assert incremental_r2_means == sorted(incremental_r2_means)
    assert incremental_r2_means[-1] >= 0.85
    assert incremental_r2_means[-1] - incremental_r2_means[0] >= 0.45
    assert event_lift_means[-1] > event_lift_means[0]
    assert event_lift_means[-1] >= 0.75


def test_contrast_estimates_coefficients_from_history_not_latent_beta():
    values, metadata, covariates = _generate(17, 5)
    assert covariates is not None
    first = capability_contrast_forecasts(
        capability_id="covariate_response",
        history=values[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=metadata,
        covariates=covariates,
    )
    changed = copy.deepcopy(metadata)
    changed["weather_effect_by_target"] = [1000.0]
    changed["event_effect_by_target"] = [-2000.0]
    second = capability_contrast_forecasts(
        capability_id="covariate_response",
        history=values[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=changed,
        covariates=covariates,
    )

    assert first["aware_method"] == (
        "history_estimated_known_future_covariate_regression"
    )
    assert first["aware_information_set"] == (
        "history_estimated_mechanism_plus_known_future_covariates"
    )
    np.testing.assert_allclose(first["blind"], second["blind"])
    np.testing.assert_allclose(first["aware"], second["aware"])

    changed_future_covariates = np.array(covariates, copy=True)
    changed_future_covariates[CONTEXT_LENGTH:, 0] += 2.0
    third = capability_contrast_forecasts(
        capability_id="covariate_response",
        history=values[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=changed,
        covariates=changed_future_covariates,
    )
    np.testing.assert_allclose(first["blind"], third["blind"])
    assert not np.allclose(first["aware"], third["aware"])


def test_high_intensity_history_estimated_contrast_is_stable():
    rows = []
    for seed in range(128):
        values, metadata, covariates = _generate(seed, 5)
        assert covariates is not None
        rows.append(
            evaluate_capability_contrast(
                capability_id="covariate_response",
                target=values,
                context_length=CONTEXT_LENGTH,
                season_length=SEASON_LENGTH,
                intensity=5,
                latent_params=metadata,
                covariates=covariates,
            )
        )
    summary = summarize_capability_contrasts(rows)
    assert summary["passed"] is True
    assert summary["aware_win_rate"] >= 0.90
    assert summary["mean_relative_loss_gain"] >= 0.60


@pytest.mark.parametrize(
    ("profile_id", "context_length", "horizon", "season_length"),
    (
        (
            "gefcom2014_load_hourly_covariate_168ctx_24h",
            168,
            24,
            24,
        ),
    ),
)
def test_existing_known_future_profiles_still_accept_high_intensity(
    profile_id: str,
    context_length: int,
    horizon: int,
    season_length: int,
):
    scores: list[float] = []
    for sample_index in range(8):
        _, metadata, covariates, features = (
            _generate_accepted_sample_values(
                "covariate_response",
                context_length + horizon,
                context_length,
                1,
                season_length,
                5,
                _seed_for(20260719, "covariate_response", sample_index),
                anchor_profile_id=profile_id,
            )
        )
        assert covariates is not None
        assert metadata["acceptance"]["accepted"] is True
        assert np.any(covariates[context_length:, 1] == 1.0)
        scores.append(float(features["covariate_incremental_r2"]))

    assert np.mean(scores) > 0.0


def test_horizon_extension_preserves_covariate_and_target_prefixes():
    short_values, short_metadata, short_covariates = _generate_sample_values(
        "covariate_response",
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        4,
        np.random.default_rng(91),
    )
    long_values, long_metadata, long_covariates = _generate_sample_values(
        "covariate_response",
        CONTEXT_LENGTH + 2 * HORIZON,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        4,
        np.random.default_rng(91),
    )
    assert short_covariates is not None
    assert long_covariates is not None

    np.testing.assert_allclose(
        short_values,
        long_values[: len(short_values)],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        short_covariates,
        long_covariates[: len(short_covariates)],
        atol=1e-12,
    )
    assert short_metadata["event_starts"] == long_metadata["event_starts"]
    assert short_metadata["event_width"] == long_metadata["event_width"]
