import numpy as np

from app.services.synthetic_capability_contrast import (
    _extrapolated_pulse_centers,
    capability_contrast_forecasts,
)
from app.services.synthetic_generation_service import _generate_sample_values


CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24


def _generate(intensity: int, seed: int = 19):
    return _generate_sample_values(
        "predictable_intermittency",
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        intensity,
        np.random.default_rng(seed),
    )


def test_interval_motif_is_dataset_relative_repeated_and_predictable():
    observed_motifs: set[tuple[int, ...]] = set()
    for seed in range(24):
        _, metadata, _ = _generate(5, seed)
        motif = tuple(metadata["pulse_interval_pattern"])
        centers = metadata["pulse_centers"]
        historical = [center for center in centers if center < CONTEXT_LENGTH]
        future = [center for center in centers if center >= CONTEXT_LENGTH]

        observed_motifs.add(motif)
        assert len(motif) == 3
        assert len(set(motif)) > 1
        assert len(historical) >= 7
        assert future
        np.testing.assert_array_equal(
            np.diff(historical)[3:],
            np.diff(historical)[:-3],
        )
        assert (
            metadata["predictability"]["evidence"][
                "interval_pattern_repetitions_in_context"
            ]
            >= 2.0
        )
        assert (
            metadata["predictability"]["evidence"][
                "interval_reconstruction_mae"
            ]
            == 0.0
        )

        extrapolated = _extrapolated_pulse_centers(
            CONTEXT_LENGTH,
            HORIZON,
            historical,
        )
        assert [
            center
            for center in extrapolated
            if CONTEXT_LENGTH <= center < CONTEXT_LENGTH + HORIZON
        ] == future

    assert observed_motifs == {(17, 29, 25)}


def test_intensity_changes_only_pulse_strength_not_schedule_or_width():
    low_values, low, _ = _generate(1, 71)
    high_values, high, _ = _generate(5, 71)

    for key in (
        "pulse_period",
        "requested_pulse_period",
        "pulse_interval_pattern",
        "pulse_interval_motif_length",
        "pulse_centers",
        "pulse_width",
        "pulse_support_radius",
        "burst_count",
    ):
        assert low[key] == high[key]
    assert low["pulse_strength"] < high["pulse_strength"]

    centers = np.asarray(high["pulse_centers"], dtype=int)
    distance = np.min(
        np.abs(np.arange(len(high_values))[:, None] - centers[None, :]),
        axis=1,
    )
    mechanism_delta = np.abs(high_values[:, 0] - low_values[:, 0])
    assert float(np.mean(mechanism_delta[distance <= 1])) > 20 * float(
        np.mean(mechanism_delta[distance >= 5])
    )


def test_high_intensity_pulses_are_visibly_separated_from_local_background():
    visibility: list[float] = []
    for seed in range(32):
        values, metadata, _ = _generate(5, seed)
        series = values[:, 0]
        width = max(1, int(np.ceil(2 * metadata["pulse_width"])))
        local_contrasts: list[float] = []
        for center in metadata["pulse_centers"]:
            left = max(0, center - 4 * width)
            right = min(len(series), center + 4 * width + 1)
            background_indices = [
                index
                for index in range(left, right)
                if abs(index - center) > width
            ]
            if background_indices:
                local_contrasts.append(
                    float(series[center] - np.median(series[background_indices]))
                )
        visibility.append(float(np.median(local_contrasts)))

    assert float(np.quantile(visibility, 0.1)) > 1.0


def test_aware_contrast_does_not_depend_on_latent_interval_pattern():
    target, metadata, covariates = _generate(5, 83)
    first = capability_contrast_forecasts(
        capability_id="predictable_intermittency",
        history=target[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=metadata,
        covariates=covariates,
    )
    changed = dict(metadata)
    changed["pulse_interval_pattern"] = [2, 97, 3, 101]
    second = capability_contrast_forecasts(
        capability_id="predictable_intermittency",
        history=target[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=changed,
        covariates=covariates,
    )

    assert first["aware_method"] == "history_estimated_nonuniform_pulse_clock"
    np.testing.assert_allclose(first["aware"], second["aware"])


def test_short_horizons_preserve_phase_and_existing_trajectory_prefix():
    for seed in range(48):
        generated = {}
        for horizon in (8, 12, 24):
            generated[horizon] = _generate_sample_values(
                "predictable_intermittency",
                CONTEXT_LENGTH + horizon,
                CONTEXT_LENGTH,
                1,
                SEASON_LENGTH,
                5,
                np.random.default_rng(seed),
            )

        long_values, long_metadata, _ = generated[24]
        assert long_metadata["predictability"]["construction_validated"] is True
        for horizon in (8, 12):
            values, metadata, _ = generated[horizon]
            np.testing.assert_allclose(
                values,
                long_values[: CONTEXT_LENGTH + horizon],
                atol=1e-12,
            )
            for key in (
                "pulse_period",
                "requested_pulse_period",
                "pulse_interval_pattern",
                "pulse_interval_motif_length",
                "pulse_width",
                "pulse_support_radius",
            ):
                assert metadata[key] == long_metadata[key]
            assert metadata["pulse_centers"] == [
                center
                for center in long_metadata["pulse_centers"]
                if center < CONTEXT_LENGTH + horizon
            ]
            if not any(
                CONTEXT_LENGTH <= center < CONTEXT_LENGTH + horizon
                for center in long_metadata["pulse_centers"]
            ):
                assert (
                    metadata["predictability"]["construction_validated"]
                    is False
                )
