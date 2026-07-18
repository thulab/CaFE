import numpy as np

from app.services.synthetic_capability_contrast import (
    _extrapolated_regime_cut_points,
    capability_contrast_forecasts,
)
from app.services.synthetic_generation_service import _generate_sample_values


CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24


def _generate(intensity: int, seed: int = 23):
    return _generate_sample_values(
        "regime_switching",
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        intensity,
        np.random.default_rng(seed),
    )


def _state_from_cuts(cut_points: list[int]) -> np.ndarray:
    state = np.ones(CONTEXT_LENGTH + HORIZON, dtype=float)
    sign = 1.0
    start = 0
    for cut in cut_points:
        state[start:cut] = sign
        sign *= -1.0
        start = cut
    state[start:] = sign
    return state


def test_regime_uses_a_sample_specific_repeated_explicit_duration_motif():
    motifs: set[tuple[int, ...]] = set()
    for seed in range(12):
        _, metadata, _ = _generate(5, seed)
        pattern = tuple(metadata["dwell_pattern"])
        cuts = metadata["cut_points"]
        historical = [cut for cut in cuts if cut < CONTEXT_LENGTH]

        assert len(pattern) == 4
        assert len(set(pattern)) > 1
        assert len(np.diff(historical)) >= 2 * len(pattern)
        assert any(
            all(
                interval
                == pattern[(rotation + index) % len(pattern)]
                for index, interval in enumerate(np.diff(cuts))
            )
            for rotation in range(len(pattern))
        )
        assert any(
            CONTEXT_LENGTH <= cut < CONTEXT_LENGTH + HORIZON
            for cut in cuts
        )
        assert metadata["predictability"]["construction_validated"] is True
        motifs.add(pattern)

    assert len(motifs) > 1


def test_regime_intensity_changes_strength_but_not_duration_clock():
    low_values, low, _ = _generate(1)
    high_values, high, _ = _generate(5)

    assert low["dwell_pattern"] == high["dwell_pattern"]
    assert low["cut_points"] == high["cut_points"]
    assert low["dwell_length"] == high["dwell_length"]
    assert high["regime_strength"] > low["regime_strength"]

    state = _state_from_cuts(low["cut_points"])
    intensity_effect = (high_values - low_values)[:, 0]
    assert abs(np.corrcoef(intensity_effect, state)[0, 1]) > 0.999
    assert np.mean(np.abs(intensity_effect)) > 0.5

    def standardized_state_gap(values: np.ndarray) -> float:
        history = values[:CONTEXT_LENGTH, 0]
        history_state = state[:CONTEXT_LENGTH]
        return abs(
            float(np.mean(history[history_state > 0]))
            - float(np.mean(history[history_state < 0]))
        ) / max(float(np.std(history)), 1e-9)

    assert standardized_state_gap(high_values) > 1.5
    assert standardized_state_gap(high_values) > standardized_state_gap(
        low_values
    )


def test_regime_contrast_recovers_future_cuts_from_history_only():
    target, metadata, _ = _generate(5)
    historical = [
        cut for cut in metadata["cut_points"] if cut < CONTEXT_LENGTH
    ]
    expected_future = [
        cut
        for cut in metadata["cut_points"]
        if CONTEXT_LENGTH <= cut < CONTEXT_LENGTH + HORIZON
    ]

    extrapolated = _extrapolated_regime_cut_points(
        CONTEXT_LENGTH,
        HORIZON,
        metadata["cut_points"],
    )
    assert [
        cut for cut in extrapolated if cut >= CONTEXT_LENGTH
    ] == expected_future

    first = capability_contrast_forecasts(
        capability_id="regime_switching",
        history=target[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=metadata,
    )
    changed = dict(metadata)
    changed["cut_points"] = [
        *historical,
        CONTEXT_LENGTH + 1,
        CONTEXT_LENGTH + 2,
    ]
    changed["dwell_pattern"] = [99, 98, 97, 96]
    second = capability_contrast_forecasts(
        capability_id="regime_switching",
        history=target[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=changed,
    )

    np.testing.assert_allclose(first["aware"], second["aware"])
    assert first["aware_method"] == (
        "history_inferred_explicit_duration_clock"
    )


def test_regime_horizon_extension_preserves_existing_prefix():
    short, short_metadata, _ = _generate(3, seed=41)
    long, long_metadata, _ = _generate_sample_values(
        "regime_switching",
        CONTEXT_LENGTH + 2 * HORIZON,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        3,
        np.random.default_rng(41),
    )

    np.testing.assert_allclose(short, long[: len(short)], atol=1e-12)
    assert short_metadata["dwell_pattern"] == long_metadata["dwell_pattern"]
    assert short_metadata["cut_points"] == [
        cut for cut in long_metadata["cut_points"] if cut < len(short)
    ]
