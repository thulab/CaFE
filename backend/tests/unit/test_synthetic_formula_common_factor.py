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
TARGET_DIM = 3


def _generate(seed: int, intensity: int):
    return _generate_sample_values(
        "common_factor",
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        TARGET_DIM,
        SEASON_LENGTH,
        intensity,
        np.random.default_rng(seed),
    )


def _features(values: np.ndarray) -> dict[str, float]:
    standardized = _standardize_by_context(values, CONTEXT_LENGTH)
    return _realized_features(
        standardized,
        None,
        SEASON_LENGTH,
        CONTEXT_LENGTH,
    )


def test_rank1_factor_law_is_sample_specific_stable_and_nonperiodic():
    root_pairs: set[tuple[float, float]] = set()
    loading_vectors: set[tuple[float, ...]] = set()
    for seed in range(32):
        values, metadata, _ = _generate(seed, 5)
        process = metadata["shared_factor_process"]
        root_pairs.add(
            (
                round(float(process["slow_root"]), 6),
                round(float(process["fast_root"]), 6),
            )
        )
        loading_vectors.add(
            tuple(round(float(value), 6) for value in metadata["loadings"])
        )

        assert np.isfinite(values).all()
        assert metadata["factor_rank"] == 1
        assert metadata["predictability"]["construction_validated"] is True
        assert process["law"] == (
            "sample_specific_stable_real_root_ar2_nonperiodic"
        )
        assert process["spectral_radius"] < 1.0
        assert process["phi_1"] == pytest.approx(
            process["slow_root"] + process["fast_root"]
        )
        assert process["phi_2"] == pytest.approx(
            -(process["slow_root"] * process["fast_root"])
        )
        assert metadata["loading_rms"] == pytest.approx(1.0)

    assert len(root_pairs) == 32
    assert len(loading_vectors) == 32


def test_intensity_only_scales_the_shared_rank1_contribution():
    low_values, low, _ = _generate(41, 1)
    high_values, high, _ = _generate(41, 5)

    assert low["factor_rank"] == high["factor_rank"] == 1
    assert low["loadings"] == high["loadings"]
    assert low["loading_log_scale"] == high["loading_log_scale"]
    assert (
        low["shared_factor_process"]["path_checksum"]
        == high["shared_factor_process"]["path_checksum"]
    )
    assert low["shared_factor_process"] == high["shared_factor_process"]
    assert low["local_process"] == high["local_process"]
    assert (
        low["nuisance_component_checksum"]
        == high["nuisance_component_checksum"]
    )

    low_strength = float(low["shared_factor_strength"])
    high_strength = float(high["shared_factor_strength"])
    assert high_strength > low_strength
    loadings = np.asarray(low["loadings"], dtype=float)
    difference = high_values - low_values
    factor = difference[:, 0] / (
        (high_strength - low_strength) * loadings[0]
    )
    expected_difference = (
        (high_strength - low_strength)
        * factor[:, None]
        * loadings[None, :]
    )
    np.testing.assert_allclose(difference, expected_difference, atol=1e-12)

    low_nuisance = (
        low_values
        - low_strength * factor[:, None] * loadings[None, :]
    )
    high_nuisance = (
        high_values
        - high_strength * factor[:, None] * loadings[None, :]
    )
    np.testing.assert_allclose(low_nuisance, high_nuisance, atol=1e-12)


def test_high_intensity_has_visible_rank1_structure_and_monotone_strength():
    pca_means: list[float] = []
    correlation_means: list[float] = []
    for intensity in range(1, 6):
        rows = [_features(_generate(seed, intensity)[0]) for seed in range(48)]
        pca_means.append(
            float(np.mean([row["pca_top1_explained"] for row in rows]))
        )
        correlation_means.append(
            float(np.mean([row["avg_abs_target_corr"] for row in rows]))
        )

    assert pca_means == sorted(pca_means)
    assert correlation_means == sorted(correlation_means)
    assert pca_means[-1] >= 0.88
    assert correlation_means[-1] >= 0.82
    assert pca_means[-1] - pca_means[0] >= 0.35


def test_history_factor_contrast_ignores_latent_factor_and_loadings():
    values, metadata, _ = _generate(17, 5)
    first = capability_contrast_forecasts(
        capability_id="common_factor",
        history=values[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=metadata,
    )
    changed = copy.deepcopy(metadata)
    changed["loadings"] = [1000.0, -2000.0, 3000.0]
    changed["shared_factor_process"] = {
        "law": "unrelated",
        "phi_1": -100.0,
        "phi_2": 100.0,
    }
    second = capability_contrast_forecasts(
        capability_id="common_factor",
        history=values[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=changed,
    )

    assert first["aware_method"] == (
        "history_estimated_rank1_dynamic_factor"
    )
    np.testing.assert_allclose(first["blind"], second["blind"])
    np.testing.assert_allclose(first["aware"], second["aware"])


def test_high_intensity_history_factor_contrast_is_stable_across_seeds():
    rows = []
    for seed in range(128):
        values, metadata, _ = _generate(seed, 5)
        rows.append(
            evaluate_capability_contrast(
                capability_id="common_factor",
                target=values,
                context_length=CONTEXT_LENGTH,
                season_length=SEASON_LENGTH,
                intensity=5,
                latent_params=metadata,
            )
        )
    summary = summarize_capability_contrasts(rows)
    assert summary["passed"] is True
    assert summary["aware_win_rate"] >= 0.65
    assert summary["mean_relative_loss_gain"] >= 0.10


@pytest.mark.parametrize(
    "profile_id",
    (
        "electricity_hourly_panel_168ctx",
        "traffic_hourly_panel_168ctx",
    ),
)
def test_panel_profile_conditioning_still_accepts_high_intensity(
    profile_id: str,
):
    pca_scores: list[float] = []
    correlations: list[float] = []
    for sample_index in range(8):
        _, metadata, _, features = _generate_accepted_sample_values(
            "common_factor",
            CONTEXT_LENGTH + HORIZON,
            CONTEXT_LENGTH,
            TARGET_DIM,
            SEASON_LENGTH,
            5,
            _seed_for(20260719, "common_factor", sample_index),
            anchor_profile_id=profile_id,
        )
        assert metadata["acceptance"]["accepted"] is True
        pca_scores.append(float(features["pca_top1_explained"]))
        correlations.append(float(features["avg_abs_target_corr"]))

    assert np.mean(pca_scores) >= 0.82
    assert np.mean(correlations) >= 0.72


def test_horizon_extension_preserves_common_factor_prefix():
    short, short_metadata, _ = _generate_sample_values(
        "common_factor",
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        TARGET_DIM,
        SEASON_LENGTH,
        4,
        np.random.default_rng(91),
    )
    long, long_metadata, _ = _generate_sample_values(
        "common_factor",
        CONTEXT_LENGTH + 2 * HORIZON,
        CONTEXT_LENGTH,
        TARGET_DIM,
        SEASON_LENGTH,
        4,
        np.random.default_rng(91),
    )

    np.testing.assert_allclose(short, long[: len(short)], atol=1e-12)
    assert short_metadata["loadings"] == long_metadata["loadings"]
    assert (
        short_metadata["shared_factor_process"]["process_seed"]
        == long_metadata["shared_factor_process"]["process_seed"]
    )
