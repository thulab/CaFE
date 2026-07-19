import numpy as np

from app.services.synthetic_capability_contrast import (
    _nonlinear_forecast,
    capability_contrast_forecasts,
)
from app.services.synthetic_generation_service import (
    _bounded_nonlinear_response,
    _generate_sample_values,
)


CONTEXT_LENGTH = 168
HORIZON = 24
SEASON_LENGTH = 24


def _generate(intensity: int, seed: int = 29, horizon: int = HORIZON):
    return _generate_sample_values(
        "nonlinear_persistence",
        CONTEXT_LENGTH + horizon,
        CONTEXT_LENGTH,
        1,
        SEASON_LENGTH,
        intensity,
        np.random.default_rng(seed),
    )


def test_nonlinear_structure_and_innovations_are_intensity_invariant():
    _, low, _ = _generate(1)
    _, high, _ = _generate(5)

    structural_keys = (
        "ar_phi",
        "seasonal_lag",
        "seasonal_memory",
        "nonlinear_lag",
        "nonlinear_frequency",
        "nonlinear_offset",
        "nonlinear_transform",
        "burn_in_steps",
        "recurrence_amplitude",
        "innovation_probe",
    )
    for key in structural_keys:
        assert low[key] == high[key]

    assert high["nonlinear_strength"] > low["nonlinear_strength"]
    assert low["stability_bound"] < 0.91
    assert high["stability_bound"] < 0.91


def test_nonlinear_uses_one_dataset_relative_recurrence_law():
    structures = set()
    for seed in range(32):
        values, metadata, _ = _generate(5, seed)
        assert np.isfinite(values).all()
        assert float(np.max(np.abs(values))) < 20.0
        assert metadata["predictability"]["construction_validated"] is True
        structures.add(
            (
                metadata["nonlinear_transform"],
                metadata["nonlinear_lag"],
                metadata["nonlinear_frequency"],
                metadata["nonlinear_offset"],
            )
        )

    assert structures == {("shifted_tanh", 8, 1.4, 0.6)}


def test_high_intensity_has_a_visible_bounded_nonlinear_response():
    response_energy: dict[int, list[float]] = {1: [], 5: []}
    for seed in range(32):
        for intensity in (1, 5):
            values, metadata, _ = _generate(intensity, seed)
            lagged = values[
                CONTEXT_LENGTH // 2
                - metadata["nonlinear_lag"] : CONTEXT_LENGTH
                - metadata["nonlinear_lag"],
                0,
            ] / metadata["recurrence_amplitude"]
            response = _bounded_nonlinear_response(
                lagged,
                family=metadata["nonlinear_transform"],
                frequency=metadata["nonlinear_frequency"],
                offset=metadata["nonlinear_offset"],
            )
            response_energy[intensity].append(
                float(metadata["nonlinear_strength"])
                * float(np.std(response))
            )

    assert np.median(response_energy[5]) > 5 * np.median(
        response_energy[1]
    )
    assert np.quantile(response_energy[5], 0.1) > 0.01


def test_nonlinear_horizon_extension_preserves_existing_prefix():
    short, short_metadata, _ = _generate(4, seed=43)
    long, long_metadata, _ = _generate(4, seed=43, horizon=2 * HORIZON)

    np.testing.assert_allclose(short, long[: len(short)], atol=1e-12)
    for key in (
        "nonlinear_transform",
        "nonlinear_lag",
        "nonlinear_frequency",
        "nonlinear_offset",
        "innovation_probe",
    ):
        assert short_metadata[key] == long_metadata[key]


def test_nonlinear_contrast_is_history_estimated_and_recursively_bounded():
    target, metadata, _ = _generate(5, seed=47)
    first = capability_contrast_forecasts(
        capability_id="nonlinear_persistence",
        history=target[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=metadata,
    )
    changed = dict(metadata)
    changed.update(
        {
            "nonlinear_transform": "future_secret",
            "nonlinear_lag": 99,
            "nonlinear_frequency": 99.0,
            "nonlinear_offset": 99.0,
            "nonlinear_strength": 99.0,
        }
    )
    second = capability_contrast_forecasts(
        capability_id="nonlinear_persistence",
        history=target[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=changed,
    )

    np.testing.assert_allclose(first["aware"], second["aware"])
    np.testing.assert_allclose(
        first["aware"],
        first["blind"]
        + 0.5
        * (
            _nonlinear_forecast(
                target[:CONTEXT_LENGTH],
                HORIZON,
                SEASON_LENGTH,
                metadata,
            )
            - first["blind"]
        ),
    )
    assert np.isfinite(first["aware"]).all()
    assert first["aware_information_set"] == "history_estimated_mechanism"
