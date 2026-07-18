from __future__ import annotations

import numpy as np
import pytest

from app.services.synthetic_capability_contrast import (
    _estimate_harmonic_periods_from_history,
    evaluate_capability_contrast,
)
from app.services.synthetic_generation_service import (
    _generate_accepted_sample_values,
    _generate_sample_values,
    _seed_for,
)


CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24


def _component(metadata: dict, *, additional_only: bool) -> np.ndarray:
    length = CONTEXT_LENGTH + HORIZON
    time = np.arange(length, dtype=float)
    result = np.zeros((length, 1), dtype=float)
    for item in metadata["period_components"]:
        if additional_only and not item["role"].startswith("additional_"):
            continue
        period = int(item["period"])
        phase = np.asarray(item["phase_by_target"], dtype=float)
        amplitude = np.asarray(item["amplitude_by_target"], dtype=float)
        result += amplitude[None, :] * np.sin(
            2 * np.pi * time[:, None] / period + phase[None, :]
        )
    return result


def test_period_set_is_sample_specific_resolved_and_history_observable():
    period_sets: set[tuple[int, ...]] = set()
    for seed in range(24):
        _, metadata, _ = _generate_sample_values(
            "multi_seasonal",
            CONTEXT_LENGTH + HORIZON,
            CONTEXT_LENGTH,
            1,
            SEASON_LENGTH,
            5,
            np.random.default_rng(seed),
        )
        periods = tuple(int(value) for value in metadata["periods"])
        period_sets.add(periods)

        assert len(periods) == 3
        assert periods[0] == SEASON_LENGTH
        assert len(set(periods)) == len(periods)
        assert max(periods) <= CONTEXT_LENGTH // 2
        assert CONTEXT_LENGTH / max(periods) >= 2.0
        pairwise_gaps = [
            abs(1.0 / left - 1.0 / right)
            for index, left in enumerate(periods)
            for right in periods[index + 1 :]
        ]
        assert min(pairwise_gaps) >= 1.0 / CONTEXT_LENGTH
        assert metadata["predictability"]["construction_validated"] is True

    assert len(period_sets) >= 8


def test_intensity_only_scales_additional_period_components():
    generated = {}
    for intensity in (1, 5):
        generated[intensity] = _generate_sample_values(
            "multi_seasonal",
            CONTEXT_LENGTH + HORIZON,
            CONTEXT_LENGTH,
            1,
            SEASON_LENGTH,
            intensity,
            np.random.default_rng(41),
        )

    low_values, low_metadata, _ = generated[1]
    high_values, high_metadata, _ = generated[5]
    assert low_metadata["periods"] == high_metadata["periods"]
    for low, high in zip(
        low_metadata["period_components"],
        high_metadata["period_components"],
    ):
        assert low["period"] == high["period"]
        assert low["role"] == high["role"]
        assert low["phase_by_target"] == high["phase_by_target"]
        assert (
            low["amplitude_multiplier_by_target"]
            == high["amplitude_multiplier_by_target"]
        )

    low_additional = _component(low_metadata, additional_only=True)
    high_additional = _component(high_metadata, additional_only=True)
    np.testing.assert_allclose(
        low_values - low_additional,
        high_values - high_additional,
        atol=1e-12,
    )
    assert high_metadata["additional_period_strength"] == pytest.approx(
        8.0 * low_metadata["additional_period_strength"]
    )
    assert np.sqrt(np.mean(high_additional**2)) > (
        5.0 * np.sqrt(np.mean(low_additional**2))
    )


def test_high_intensity_periods_are_visible_to_history_only_spectral_scan():
    matched_period_counts: list[int] = []
    relative_gains: list[float] = []
    for seed in range(24):
        values, metadata, _ = _generate_sample_values(
            "multi_seasonal",
            CONTEXT_LENGTH + HORIZON,
            CONTEXT_LENGTH,
            1,
            SEASON_LENGTH,
            5,
            np.random.default_rng(seed),
        )
        estimated = _estimate_harmonic_periods_from_history(
            values[:CONTEXT_LENGTH],
            fallback_period=SEASON_LENGTH,
            component_count=3,
        )
        matched_period_counts.append(
            sum(
                any(abs(estimate - truth) <= 2 for estimate in estimated)
                for truth in metadata["periods"]
            )
        )
        contrast = evaluate_capability_contrast(
            capability_id="multi_seasonal",
            target=values,
            context_length=CONTEXT_LENGTH,
            season_length=SEASON_LENGTH,
            intensity=5,
            latent_params=metadata,
        )
        assert contrast["aware_method"] == (
            "history_spectral_multi_harmonic_continuation"
        )
        relative_gains.append(float(contrast["relative_loss_gain"]))

    assert np.mean(matched_period_counts) >= 2.5
    assert np.mean(np.asarray(relative_gains) > 0.0) >= 0.80
    assert np.mean(relative_gains) >= 0.25


@pytest.mark.parametrize(
    "profile_id",
    (
        "electricity_hourly_daily_168ctx",
        "m4_hourly_daily_168ctx",
        "traffic_hourly_daily_168ctx",
    ),
)
def test_existing_real_profile_conditioning_still_accepts_high_intensity(
    profile_id: str,
):
    for sample_index in range(4):
        _, metadata, _, _ = _generate_accepted_sample_values(
            "multi_seasonal",
            CONTEXT_LENGTH + HORIZON,
            CONTEXT_LENGTH,
            1,
            SEASON_LENGTH,
            5,
            _seed_for(777, "multi_seasonal", sample_index),
            anchor_profile_id=profile_id,
        )
        assert metadata["acceptance"]["accepted"] is True
        assert metadata["acceptance"]["attempts"] <= 32
