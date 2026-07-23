from __future__ import annotations

import numpy as np
import pytest

from app.services.synthetic_v8_generation import (
    PRIMARY_FAMILY_BY_CAPABILITY,
    REQUIRED_REAL_FEATURES_BY_CAPABILITY,
    SECONDARY_FAMILY_BY_CAPABILITY,
    add_observation_noise_to_history,
    derive_deterministic_parameters,
    generate_deterministic_sample,
)


CAPABILITIES = tuple(PRIMARY_FAMILY_BY_CAPABILITY)


@pytest.mark.parametrize("capability_id", CAPABILITIES)
@pytest.mark.parametrize("family_role", ("primary", "secondary"))
def test_v8_families_are_clean_deterministic_and_prefix_invariant(
    capability_id: str,
    family_role: str,
) -> None:
    target_dim = 3 if capability_id in {
        "common_factor",
        "hierarchical_coherence",
        "cross_series_dependence",
    } else 1
    arguments = (
        capability_id,
        552,
        504,
        target_dim,
        24,
        3,
    )
    target, metadata, covariates = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(17),
        family_role=family_role,
    )
    repeated, repeated_metadata, repeated_covariates = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(17),
        family_role=family_role,
    )
    longer, _, _ = generate_deterministic_sample(
        capability_id,
        600,
        504,
        target_dim,
        24,
        3,
        np.random.default_rng(17),
        family_role=family_role,
    )

    assert target.shape == (552, target_dim)
    assert np.isfinite(target).all()
    assert np.array_equal(target, repeated)
    assert np.array_equal(target, longer[:552])
    assert metadata["clean_latent_sha256"] == repeated_metadata["clean_latent_sha256"]
    assert metadata["future_process_noise_scale"] == 0.0
    assert metadata["observation_noise_scale"] == 0.0
    assert metadata["clean_latent_is_target"] is True
    assert float(np.mean(np.std(target[504:], axis=0))) > 1e-4
    if covariates is None:
        assert repeated_covariates is None
    else:
        assert np.array_equal(covariates, repeated_covariates)


def test_v8_hierarchy_is_exactly_coherent() -> None:
    target, metadata, _ = generate_deterministic_sample(
        "hierarchical_coherence",
        552,
        504,
        4,
        24,
        5,
        np.random.default_rng(23),
    )

    assert np.max(np.abs(target[:, 0] - np.sum(target[:, 1:], axis=1))) < 1e-12
    assert metadata["future_only_shock_count"] == 0


@pytest.mark.parametrize("family_role", ("primary", "secondary"))
def test_v8_cross_series_dependence_has_observed_driver_for_future_response(
    family_role: str,
) -> None:
    target, metadata, _ = generate_deterministic_sample(
        "cross_series_dependence",
        552,
        504,
        3,
        24,
        5,
        np.random.default_rng(27),
        family_role=family_role,
    )

    delay = metadata["cross_lag_steps"]
    assert metadata["driver_index"] == 0
    assert metadata["responder_indices"] == [1, 2]
    assert delay == 64
    assert metadata["history_covered_forecast_steps"] == 48
    assert metadata["counterfactual_responder_history_invariant"] is True
    assert metadata["counterfactual_future_is_driver_determined"] is True
    assert metadata["future_only_shock_count"] == 0
    assert np.std(target[504:, 1:]) > 1e-4


@pytest.mark.parametrize("family_role", ("primary", "secondary"))
def test_v8_cross_series_counterfactual_pair_has_identical_responder_history(
    family_role: str,
) -> None:
    arguments = (
        "cross_series_dependence",
        552,
        504,
        3,
        24,
        5,
    )
    first, first_metadata, _ = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(43),
        family_role=family_role,
        counterfactual_variant=0,
    )
    second, second_metadata, _ = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(43),
        family_role=family_role,
        counterfactual_variant=1,
    )

    delay = first_metadata["cross_lag_steps"]
    assert first_metadata["counterfactual_variant"] == 0
    assert second_metadata["counterfactual_variant"] == 1
    assert np.array_equal(first[:504, 1:], second[:504, 1:])
    assert np.array_equal(first[: 504 - delay, 0], second[: 504 - delay, 0])
    assert not np.array_equal(first[504 - delay : 504, 0], second[504 - delay : 504, 0])
    assert not np.array_equal(first[504:, 1:], second[504:, 1:])
    assert first_metadata["counterfactual_alternative_rms"] > 0.1
    assert second_metadata["counterfactual_alternative_rms"] > 0.1


@pytest.mark.parametrize("family_role", ("primary", "secondary"))
def test_v8_covariate_counterfactual_pair_has_identical_history(
    family_role: str,
) -> None:
    arguments = (
        "covariate_response",
        552,
        504,
        1,
        24,
        5,
    )
    first, first_metadata, first_covariates = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(47),
        family_role=family_role,
        counterfactual_variant=0,
    )
    second, second_metadata, second_covariates = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(47),
        family_role=family_role,
        counterfactual_variant=1,
    )

    assert first_covariates is not None
    assert second_covariates is not None
    assert first_metadata["counterfactual_variant"] == 0
    assert second_metadata["counterfactual_variant"] == 1
    assert np.array_equal(first[:504], second[:504])
    assert np.array_equal(first_covariates[:504], second_covariates[:504])
    assert not np.array_equal(first_covariates[504:], second_covariates[504:])
    assert not np.array_equal(first[504:], second[504:])
    assert first_metadata["counterfactual_target_history_invariant"] is True
    assert first_metadata["counterfactual_past_covariates_invariant"] is True
    assert first_metadata["counterfactual_future_is_covariate_determined"] is True


def test_v8_robustness_noise_changes_only_history_and_keeps_clean_future() -> None:
    clean, _, _ = generate_deterministic_sample(
        "multi_seasonal",
        552,
        504,
        1,
        24,
        3,
        np.random.default_rng(29),
    )

    observed, metadata = add_observation_noise_to_history(
        clean,
        context_length=504,
        noise_ratio=0.15,
        rng=np.random.default_rng(31),
    )

    assert not np.array_equal(observed[:504], clean[:504])
    assert np.array_equal(observed[504:], clean[504:])
    assert metadata["future_noise_max_abs"] == 0.0
    assert metadata["realized_noise_to_history_std_ratio"] == pytest.approx(
        0.15,
        abs=0.02,
    )


def test_v8_hierarchy_robustness_noise_preserves_observed_coherence() -> None:
    clean, _, _ = generate_deterministic_sample(
        "hierarchical_coherence",
        552,
        504,
        4,
        24,
        3,
        np.random.default_rng(37),
    )

    observed, _ = add_observation_noise_to_history(
        clean,
        context_length=504,
        noise_ratio=0.15,
        rng=np.random.default_rng(41),
        preserve_additive_hierarchy=True,
    )

    assert np.max(
        np.abs(observed[:504, 0] - np.sum(observed[:504, 1:], axis=1))
    ) < 1e-12
    assert np.array_equal(observed[504:], clean[504:])


@pytest.mark.parametrize("capability_id", CAPABILITIES)
def test_every_required_real_feature_is_recorded_in_parameter_mapping(
    capability_id: str,
) -> None:
    summary = {
        feature: {"p50": 0.4, "p25": 0.3, "p75": 0.5}
        for feature in {
            "acf1",
            "seasonal_acf",
            "dominant_period",
            "spectral_concentration",
            *REQUIRED_REAL_FEATURES_BY_CAPABILITY[capability_id],
        }
    }
    _, mappings = derive_deterministic_parameters(
        capability_id,
        summary,
        season_length=24,
        context_length=504,
    )
    sources = {str(mapping["source_feature"]) for mapping in mappings}

    for feature in REQUIRED_REAL_FEATURES_BY_CAPABILITY[capability_id]:
        assert feature in sources or any(feature in source.split("/") for source in sources)
    assert PRIMARY_FAMILY_BY_CAPABILITY[capability_id]
    assert SECONDARY_FAMILY_BY_CAPABILITY[capability_id]
