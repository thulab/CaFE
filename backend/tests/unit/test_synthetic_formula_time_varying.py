import numpy as np

from app.services.synthetic_capability_contrast import (
    _estimate_modulation_period_from_history,
    capability_contrast_forecasts,
)
from app.services.synthetic_generation_service import (
    _generate_sample_values,
    _realized_features,
)


CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24


def _generate(
    intensity: int,
    seed: int,
    *,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = HORIZON,
    season_length: int = SEASON_LENGTH,
):
    return _generate_sample_values(
        "time_varying_seasonality",
        context_length + horizon,
        context_length,
        1,
        season_length,
        intensity,
        np.random.default_rng(seed),
    )


def _modulated_carrier(
    metadata: dict,
    length: int,
) -> np.ndarray:
    time = np.arange(length, dtype=float)
    modulation_angle = (
        2
        * np.pi
        * time[:, None]
        / metadata["modulation_period"]
        + np.asarray(metadata["modulation_phase_by_target"])[None, :]
    )
    harmonic_ratio = metadata["modulation_second_harmonic_ratio"]
    modulation = (
        np.sin(modulation_angle)
        + harmonic_ratio
        * np.sin(
            2 * modulation_angle
            + metadata["modulation_second_harmonic_phase"]
        )
    ) / (1.0 + harmonic_ratio)
    amplitude = 1.0 + metadata["amplitude_depth"] * modulation
    phase_modulation = (
        2
        * np.pi
        * metadata["phase_modulation_depth_cycles"]
        * modulation
    )
    return amplitude * np.sin(
        2 * np.pi * time[:, None] / metadata["primary_period"]
        + np.asarray(metadata["carrier_phase_by_target"])[None, :]
        + phase_modulation
    )


def test_modulation_law_is_sample_specific_and_exposed_twice_in_context():
    periods: set[int] = set()
    waveform_parameters: set[tuple[float, float]] = set()
    for seed in range(48):
        _, metadata, _ = _generate(5, seed)
        evidence = metadata["predictability"]["evidence"]
        periods.add(metadata["modulation_period"])
        waveform_parameters.add(
            (
                round(metadata["modulation_second_harmonic_ratio"], 6),
                round(metadata["modulation_second_harmonic_phase"], 6),
            )
        )

        assert metadata["predictability"]["construction_validated"] is True
        assert evidence["modulation_cycles_in_context"] >= 2.0
        assert metadata["modulation_period"] in metadata[
            "modulation_period_candidates"
        ]
        assert metadata["modulation_basis"] == "bounded_two_harmonic_fourier"
        assert 0.12 <= metadata["modulation_second_harmonic_ratio"] <= 0.28

    assert len(periods) >= 3
    assert len(waveform_parameters) == 48


def test_intensity_changes_only_modulation_depth_not_law_or_background():
    low_values, low, _ = _generate(1, 91)
    high_values, high, _ = _generate(5, 91)

    for key in (
        "primary_period",
        "modulation_period",
        "modulation_period_candidates",
        "modulation_basis",
        "modulation_second_harmonic_ratio",
        "modulation_second_harmonic_phase",
        "modulation_phase_by_target",
        "carrier_phase_by_target",
        "background_slope_abs_mean",
        "noise_scale",
    ):
        assert low[key] == high[key]
    assert low["amplitude_depth"] < high["amplitude_depth"]
    assert (
        low["phase_modulation_depth_cycles"]
        < high["phase_modulation_depth_cycles"]
    )
    low_mechanism = _modulated_carrier(low, len(low_values))
    high_mechanism = _modulated_carrier(high, len(high_values))
    np.testing.assert_allclose(
        low_values - low_mechanism,
        high_values - high_mechanism,
        atol=1e-12,
    )


def test_high_intensity_has_visibly_stronger_measured_modulation():
    low_amplitude: list[float] = []
    high_amplitude: list[float] = []
    low_phase: list[float] = []
    high_phase: list[float] = []
    for seed in range(48):
        low_values, _, _ = _generate(1, seed)
        high_values, _, _ = _generate(5, seed)
        low_features = _realized_features(
            low_values,
            None,
            SEASON_LENGTH,
            CONTEXT_LENGTH,
        )
        high_features = _realized_features(
            high_values,
            None,
            SEASON_LENGTH,
            CONTEXT_LENGTH,
        )
        low_amplitude.append(low_features["seasonal_amplitude_modulation"])
        high_amplitude.append(high_features["seasonal_amplitude_modulation"])
        low_phase.append(low_features["seasonal_phase_variation"])
        high_phase.append(high_features["seasonal_phase_variation"])

    assert float(np.median(high_amplitude)) > 1.8 * float(
        np.median(low_amplitude)
    )
    assert float(np.median(high_phase)) > 1.8 * float(np.median(low_phase))


def test_history_estimated_contrast_ignores_latent_modulation_period():
    matched_periods = 0
    for seed in range(32):
        target, metadata, covariates = _generate(5, seed)
        estimated = _estimate_modulation_period_from_history(
            target[:CONTEXT_LENGTH],
            carrier_period=SEASON_LENGTH,
        )
        matched_periods += int(estimated == metadata["modulation_period"])

        first = capability_contrast_forecasts(
            capability_id="time_varying_seasonality",
            history=target[:CONTEXT_LENGTH],
            horizon=HORIZON,
            season_length=SEASON_LENGTH,
            latent_params=metadata,
            covariates=covariates,
        )
        changed = dict(metadata)
        changed["modulation_period"] = 10_000
        changed["predictability"] = {
            **metadata["predictability"],
            "evidence": {
                **metadata["predictability"]["evidence"],
                "modulation_period": 10_000,
            },
        }
        second = capability_contrast_forecasts(
            capability_id="time_varying_seasonality",
            history=target[:CONTEXT_LENGTH],
            horizon=HORIZON,
            season_length=SEASON_LENGTH,
            latent_params=changed,
            covariates=covariates,
        )

        assert first["aware_method"] == "history_estimated_sideband_continuation"
        np.testing.assert_allclose(first["aware"], second["aware"])

    assert matched_periods >= 24


def test_two_profile_shapes_are_stable_and_horizon_prefix_invariant():
    for context_length, horizon, season_length in (
        (168, 24, 24),
        (365, 28, 7),
    ):
        for seed in range(24):
            short, metadata, _ = _generate(
                5,
                seed,
                context_length=context_length,
                horizon=horizon,
                season_length=season_length,
            )
            long, _, _ = _generate(
                5,
                seed,
                context_length=context_length,
                horizon=2 * horizon,
                season_length=season_length,
            )
            assert metadata["predictability"]["construction_validated"] is True
            np.testing.assert_allclose(
                short,
                long[: context_length + horizon],
                atol=1e-12,
            )
