from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from app.services.synthetic_v8_generation import (
    PRIMARY_FAMILY_BY_CAPABILITY,
    REQUIRED_REAL_FEATURES_BY_CAPABILITY,
    SECONDARY_FAMILY_BY_CAPABILITY,
    add_observation_noise_to_history,
    common_factor_identifiability_gate,
    cross_series_identifiability_gate,
    derive_deterministic_parameters,
    generate_deterministic_sample,
    standardize_common_factor_counterfactual_member,
    standardize_cross_series_counterfactual_member,
)
from app.services.synthetic_v8_feature_gate import (
    covariate_family_match_checks,
    nonlinear_mechanism_response_checks,
)


def test_covariate_real_feature_contract_is_history_only():
    assert "future_abs_covariate_target_corr" not in (
        REQUIRED_REAL_FEATURES_BY_CAPABILITY["covariate_response"]
    )


def test_nonlinear_observable_proxy_does_not_control_mechanism_dose():
    baseline = {
        "acf1": {"p50": 0.7},
        "seasonal_acf": {"p50": 0.4},
        "dominant_period": {"p50": 24.0},
        "spectral_concentration": {"p50": 0.3},
        "nonlinear_conditional_gain": {"p50": 0.0},
        "nonlinear_multi_lag_gain": {"p50": 0.0},
    }
    extreme = deepcopy(baseline)
    extreme["nonlinear_conditional_gain"] = {"p50": 1.0}
    extreme["nonlinear_multi_lag_gain"] = {"p50": 1.0}

    baseline_parameters, baseline_mappings = derive_deterministic_parameters(
        "nonlinear_persistence",
        baseline,
        season_length=24,
        context_length=504,
    )
    extreme_parameters, extreme_mappings = derive_deterministic_parameters(
        "nonlinear_persistence",
        extreme,
        season_length=24,
        context_length=504,
    )

    assert baseline_parameters == extreme_parameters
    assert baseline_parameters["nonlinear_gain_scale"] == pytest.approx(0.7)
    assert baseline_parameters["nonlinear_lag_scale"] == pytest.approx(1.0 / 3.0)
    nonlinear_sources = {
        mapping["source_feature"]
        for mapping in baseline_mappings
        if str(mapping["parameter"]).startswith("nonlinear_")
    }
    assert nonlinear_sources == {"synthetic_protocol_constant"}
    assert baseline_mappings == extreme_mappings


def test_nonlinear_mechanism_gate_separates_injected_dose_from_exact_lag_r2():
    rows = [
        {
            "dataset_id": "demo",
            "capability_id": "nonlinear_persistence",
            "generator_family_role": family_role,
            "evaluation_table": "main",
            "seed_index": seed,
            "intensity": intensity,
            "generation_metadata": {
                "nonlinear_strength": (
                    0.1 * intensity + 0.001 * seed
                ),
                "nonlinear_effect_to_recurrence_residual_std_ratio": (
                    0.05 * intensity + 0.0001 * seed
                ),
                "state_clip_fraction": 0.0,
            },
            "realized_features": {
                "nonlinear_actual_lag_gain": (
                    0.001 * (6 - intensity) + 0.00001 * seed
                ),
                "nonlinear_conditional_gain": (
                    0.002 * (6 - intensity) + 0.00001 * seed
                ),
            },
        }
        for family_role, intensities in (
            ("primary", (1, 2, 3, 4, 5)),
            ("secondary", (3, 5)),
        )
        for seed in (2, 7)
        for intensity in intensities
    ]

    results = nonlinear_mechanism_response_checks(rows)

    assert len(results) == 2
    assert all(result["accepted"] for result in results)
    assert all(
        result["paired_low_high_positive_fraction"] == pytest.approx(1.0)
        for result in results
    )
    assert all(
        result["actual_lag_gain_diagnostic"][
            "monotonicity_enforced"
        ]
        is False
        for result in results
    )
    assert all(
        result["actual_lag_gain_diagnostic"][
            "paired_low_high_positive_fraction"
        ]
        == pytest.approx(0.0)
        for result in results
    )
    assert all(
        result["observable_proxy_diagnostic"]["monotonicity_enforced"]
        is False
        for result in results
    )
    assert all(
        result["observable_proxy_diagnostic"][
            "paired_low_high_positive_fraction"
        ]
        == pytest.approx(0.0)
        for result in results
    )
    assert all(
        result["dynamic_activity_gate"]["accepted"]
        for result in results
    )

    for row in rows:
        if (
            row["generator_family_role"] == "secondary"
            and row["intensity"] == 5
        ):
            row["realized_features"] = {}
    missing = nonlinear_mechanism_response_checks(rows)

    secondary = next(
        result
        for result in missing
        if result["family_role"] == "secondary"
    )
    assert secondary["accepted"] is False

    for row in rows:
        if (
            row["generator_family_role"] == "secondary"
            and row["intensity"] == 5
        ):
            row["realized_features"] = {
                "nonlinear_actual_lag_gain": 0.001
            }
            row["generation_metadata"]["nonlinear_strength"] = 0.0
    reversed_strength = nonlinear_mechanism_response_checks(rows)
    secondary = next(
        result
        for result in reversed_strength
        if result["family_role"] == "secondary"
    )
    assert secondary["accepted"] is False


def test_v8_nonlinear_families_use_matched_bounded_quadratic_doses():
    generated = {}
    for family_role in ("primary", "secondary"):
        for intensity in (1, 5):
            _, metadata, _ = generate_deterministic_sample(
                "nonlinear_persistence",
                552,
                504,
                1,
                24,
                intensity,
                np.random.default_rng(43),
                family_role=family_role,
            )
            generated[(family_role, intensity)] = metadata
            assert metadata["state_clip_fraction"] == pytest.approx(0.0)
            assert (
                0.0
                <= metadata["nonlinear_response_curvature_fraction"]
                <= 1.0
            )
            assert (
                metadata[
                    "nonlinear_effect_to_recurrence_residual_std_ratio"
                ]
                > 0.0
            )

    assert generated[("primary", 1)]["nonlinear_transform"] == (
        "signed_rational_quadratic"
    )
    assert generated[("secondary", 1)]["nonlinear_transform"] == (
        "signed_softsign_quadratic"
    )
    for intensity in (1, 5):
        assert generated[("primary", intensity)]["nonlinear_strength"] == (
            pytest.approx(
                generated[("secondary", intensity)]["nonlinear_strength"]
            )
        )
    assert (
        generated[("primary", 5)]["nonlinear_strength"]
        > generated[("primary", 1)]["nonlinear_strength"]
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
def test_v8_common_factor_requires_joint_code_to_recover_future(
    family_role: str,
) -> None:
    arguments = (
        "common_factor",
        552,
        504,
        3,
        24,
        5,
    )
    first, first_metadata, _ = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(19),
        family_role=family_role,
        counterfactual_variant=0,
    )
    second, second_metadata, _ = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(19),
        family_role=family_role,
        counterfactual_variant=1,
    )
    first, first_normalization = (
        standardize_common_factor_counterfactual_member(
            first,
            context_length=504,
            metadata=first_metadata,
        )
    )
    second, second_normalization = (
        standardize_common_factor_counterfactual_member(
            second,
            context_length=504,
            metadata=second_metadata,
        )
    )
    protected = int(first_metadata["protected_target_index"])
    gate = common_factor_identifiability_gate(
        first,
        second,
        context_length=504,
        metadata=first_metadata,
        enforced=True,
    )

    assert first_normalization == second_normalization
    assert np.array_equal(
        first[:504, protected],
        second[:504, protected],
    )
    assert not np.array_equal(
        first[504:, protected],
        second[504:, protected],
    )
    assert first_metadata["code_matrix_rank"] == 2
    assert first_metadata["historical_episode_count"] >= 5
    assert first_metadata["response_basis_process"][
        "factor_persistence_parameter"
    ] == pytest.approx(first_metadata["factor_persistence"])
    assert first_metadata["local_factor_loading_orthogonalized"] is True
    assert first_metadata["local_code_shape_orthogonalized"] is True
    assert first_metadata["main_task_is_dense_dynamic_factor"] is True
    assert first_metadata["code_strength"] < 1.0
    assert len(first_metadata["code_shape"]) >= 24
    assert first_metadata["dense_factor_strength"] > 0.0
    assert gate["joint_holdout_r2"] >= 0.80
    assert gate["best_single_channel_holdout_r2"] <= 0.80
    assert gate["joint_minus_best_single_holdout_r2"] >= 0.15
    assert gate["positive_control_effect_nrmse"] <= 1e-6
    assert gate["positive_control_effect_correlation"] >= 0.95
    assert gate["accepted"] is True


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
    assert delay in metadata["cross_lag_candidate_steps"]
    assert delay >= 48
    assert delay <= int(0.40 * 504)
    assert (delay - 48) % metadata["cross_lag_step"] == 0
    assert metadata["cross_lag_sampling_policy"] == (
        "horizon_aligned_history_covered_lag"
    )
    assert metadata["history_covered_forecast_steps"] == 48
    assert metadata["counterfactual_responder_history_invariant"] is True
    assert metadata["counterfactual_future_is_driver_determined"] is True
    assert metadata["future_only_shock_count"] == 0
    assert np.std(target[504:, 1:]) > 1e-4


def test_v8_cross_series_master_pair_remains_well_formed_in_all_suffix_views(
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
        np.random.default_rng(59),
        counterfactual_variant=0,
    )
    second, second_metadata, _ = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(59),
        counterfactual_variant=1,
    )
    first, _ = standardize_cross_series_counterfactual_member(
        first,
        context_length=504,
        metadata=first_metadata,
    )
    second, _ = standardize_cross_series_counterfactual_member(
        second,
        context_length=504,
        metadata=second_metadata,
    )
    delay = int(first_metadata["cross_lag_steps"])

    for context_length in (96, 168, 336, 504):
        start = 504 - context_length
        first_view_metadata = deepcopy(first_metadata)
        second_view_metadata = deepcopy(second_metadata)
        first_view_metadata["counterfactual_driver_slice"] = [
            context_length - delay,
            context_length,
        ]
        second_view_metadata["counterfactual_driver_slice"] = [
            context_length - delay,
            context_length,
        ]
        first_view, first_normalization = (
            standardize_cross_series_counterfactual_member(
                first[start:],
                context_length=context_length,
                metadata=first_view_metadata,
            )
        )
        second_view, second_normalization = (
            standardize_cross_series_counterfactual_member(
                second[start:],
                context_length=context_length,
                metadata=second_view_metadata,
            )
        )

        invariant_stop = context_length - delay
        assert invariant_stop >= 16
        assert first_normalization == second_normalization
        assert np.array_equal(
            first_view[:invariant_stop, 0],
            second_view[:invariant_stop, 0],
        )
        assert np.array_equal(
            first_view[:context_length, 1:],
            second_view[:context_length, 1:],
        )
        assert not np.array_equal(
            first_view[context_length:, 1:],
            second_view[context_length:, 1:],
        )
        assert context_length - delay >= 0
        assert context_length - delay + 48 <= context_length


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


def test_v8_cross_series_primary_uses_signed_responder_edges() -> None:
    _, metadata, _ = generate_deterministic_sample(
        "cross_series_dependence",
        552,
        504,
        5,
        24,
        5,
        np.random.default_rng(41),
    )

    assert metadata["responder_signs"] == [1.0, -1.0, 1.0, -1.0]


@pytest.mark.parametrize("context_length", (96, 168, 336))
def test_v8_cross_series_pair_has_shared_scale_and_passes_identifiability_gate(
    context_length: int,
) -> None:
    arguments = (
        "cross_series_dependence",
        context_length + 48,
        context_length,
        3,
        24,
        5,
    )
    first, first_metadata, _ = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(43),
        counterfactual_variant=0,
    )
    second, second_metadata, _ = generate_deterministic_sample(
        *arguments,
        np.random.default_rng(43),
        counterfactual_variant=1,
    )

    first, first_normalization = (
        standardize_cross_series_counterfactual_member(
            first,
            context_length=context_length,
            metadata=first_metadata,
        )
    )
    second, second_normalization = (
        standardize_cross_series_counterfactual_member(
            second,
            context_length=context_length,
            metadata=second_metadata,
        )
    )
    invariant_stop = first_metadata["counterfactual_driver_slice"][0]
    gate = cross_series_identifiability_gate(
        first,
        second,
        context_length=context_length,
        metadata=first_metadata,
        enforced=True,
    )

    assert first_normalization == second_normalization
    assert np.array_equal(
        first[:invariant_stop, 0],
        second[:invariant_stop, 0],
    )
    assert np.array_equal(
        first[context_length:, 0],
        second[context_length:, 0],
    )
    assert np.array_equal(
        first[:context_length, 1:],
        second[:context_length, 1:],
    )
    assert first_metadata["counterfactual_path_is_dense"] is True
    assert first_metadata["counterfactual_path_is_in_support"] is True
    assert first_metadata["dense_teaching_fraction"] > 0.0
    assert first_metadata["driver_excitation_knot_spacing"] in {
        4,
        5,
        6,
    }
    assert np.max(
        np.abs(
            np.asarray(first_metadata["counterfactual_path_mean_by_member"])
        )
    ) < 1e-10
    assert np.asarray(
        first_metadata["counterfactual_path_std_by_member"]
    ) == pytest.approx(
        np.repeat(
            first_metadata["driver_excitation_scale"],
            2,
        )
    )
    assert gate["accepted"] is True
    assert gate["blind_best_driver"] == first_metadata["driver_index"]
    assert gate["blind_best_lag"] == first_metadata["cross_lag_steps"]
    assert gate["minimum_declared_holdout_r2"] >= 0.80
    assert gate["positive_control_effect_nrmse"] <= 0.15
    assert gate["positive_control_effect_correlation"] >= 0.95
    assert gate["positive_control_effect_amplitude_ratio"] == pytest.approx(
        1.0,
        abs=0.05,
    )


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


@pytest.mark.parametrize("intensity", (1, 2, 3, 4, 5))
def test_v8_covariate_secondary_matches_primary_history_effect_dose(
    intensity: int,
) -> None:
    arguments = (
        "covariate_response",
        552,
        504,
        1,
        24,
        intensity,
    )
    primary, primary_metadata, primary_covariates = (
        generate_deterministic_sample(
            *arguments,
            np.random.default_rng(59),
            family_role="primary",
            counterfactual_variant=0,
        )
    )
    secondary, secondary_metadata, secondary_covariates = (
        generate_deterministic_sample(
            *arguments,
            np.random.default_rng(59),
            family_role="secondary",
            counterfactual_variant=0,
        )
    )

    assert primary_covariates is not None
    assert secondary_covariates is not None
    assert np.array_equal(primary_covariates, secondary_covariates)
    assert not np.array_equal(primary, secondary)
    assert primary_metadata["response_law"] == "instantaneous_linear"
    assert (
        secondary_metadata["response_law"]
        == "semilinear_saturating_distributed_lag"
    )
    assert secondary_metadata["effect_strength"] == pytest.approx(
        primary_metadata["effect_strength"]
    )
    assert secondary_metadata[
        "covariate_effect_variance_share"
    ] == pytest.approx(
        primary_metadata["covariate_effect_variance_share"],
        abs=1e-12,
    )
    assert primary_metadata["response_normalization"] == (
        "primary_reference_unchanged"
    )
    assert secondary_metadata["response_normalization"] == (
        "affine_match_primary_reference_history_mean_and_std"
    )
    assert secondary_metadata[
        "response_history_mean_by_target"
    ] == pytest.approx(
        primary_metadata["response_history_mean_by_target"],
        abs=1e-12,
    )
    assert secondary_metadata[
        "response_history_std_by_target"
    ] == pytest.approx(
        primary_metadata["response_history_std_by_target"],
        abs=1e-12,
    )
    assert secondary_metadata["baseline_process"] == (
        primary_metadata["baseline_process"]
    )
    assert secondary_metadata["weather_effect_by_target"] == pytest.approx(
        primary_metadata["weather_effect_by_target"]
    )
    assert secondary_metadata["event_effect_by_target"] == pytest.approx(
        primary_metadata["event_effect_by_target"]
    )


def test_v8_covariate_family_gate_rejects_scale_confounding() -> None:
    metadata = {
        "effect_strength": 0.4,
        "baseline_process": {"motif_sha256": "same"},
    }
    primary = {
        "sample_id": "primary",
        "target_feature_value": 0.5,
        "mase_scale": 0.8,
        "covariates": [[0.0, 1.0], [1.0, 0.0]],
        "generation_metadata": metadata,
    }
    secondary = {
        **deepcopy(primary),
        "sample_id": "secondary",
        "mase_scale": 0.78,
    }

    assert covariate_family_match_checks(primary, secondary)["accepted"]
    secondary["mase_scale"] = 0.08
    rejected = covariate_family_match_checks(primary, secondary)
    assert rejected["accepted"] is False
    assert rejected["mase_scale_relative_difference"] == pytest.approx(0.9)


def test_v8_primary_nuisance_parameters_vary_across_seeds() -> None:
    target_dims = {
        "common_factor": 4,
        "hierarchical_coherence": 4,
        "cross_series_dependence": 4,
    }
    metadata_by_capability = {
        capability_id: [
            generate_deterministic_sample(
                capability_id,
                552,
                504,
                target_dims.get(capability_id, 1),
                24,
                5,
                np.random.default_rng(seed),
            )[1]
            for seed in range(12)
        ]
        for capability_id in CAPABILITIES
    }

    fingerprints = {
        "trend": lambda row: (
            tuple(np.round(row["slope_jitter_by_target"], 6)),
            tuple(row["curvature_sign_by_target"]),
        ),
        "multi_seasonal": lambda row: tuple(np.round(row["periods"], 6)),
        "time_varying_seasonality": lambda row: (
            round(row["primary_period"], 6),
            round(row["modulation_period"], 6),
        ),
        "regime_switching": lambda row: (
            tuple(row["dwell_pattern"]),
            row["dwell_anchor_offset"],
        ),
        "nonlinear_persistence": lambda row: (
            row["nonlinear_lag"],
            tuple(
                round(mode["period"], 6)
                for mode in row["deterministic_forcing"]["modes"]
            ),
        ),
        "predictable_intermittency": lambda row: (
            tuple(row["pulse_interval_pattern"]),
            row["pulse_anchor_offset"],
        ),
        "common_factor": lambda row: (
            tuple(
                np.round(
                    np.asarray(row["code_matrix"]).ravel(),
                    6,
                )
            ),
            tuple(np.round(row["response_loadings"], 6)),
            row["episode_span"],
            row["protected_target_index"],
        ),
        "hierarchical_coherence": lambda row: (
            tuple(row["child_permutation"]),
            tuple(np.round(row["aggregate_share_by_child"], 6)),
            tuple(
                np.round(
                    np.asarray(row["local_contrast_loadings"]).ravel(),
                    6,
                )
            ),
        ),
        "cross_series_dependence": lambda row: (
            row["cross_lag_steps"],
            row["driver_excitation_knot_spacing"],
            round(row["driver_excitation_scale"], 6),
            round(row["counterfactual_alternative_rms"], 6),
        ),
        "covariate_response": lambda row: (
            row["counterfactual_weather_transform_selected"],
            tuple(row["counterfactual_event_start_options"]),
            row["event_width"],
        ),
    }

    for capability_id, rows in metadata_by_capability.items():
        assert len({fingerprints[capability_id](row) for row in rows}) >= 8


@pytest.mark.parametrize("family_role", ["primary", "secondary"])
def test_intermittency_event_energy_dose_is_strictly_monotone(
    family_role,
) -> None:
    shares = []
    for intensity in range(1, 6):
        _target, metadata, _covariates = generate_deterministic_sample(
            "predictable_intermittency",
            552,
            504,
            1,
            24,
            intensity,
            np.random.default_rng(37),
            family_role=family_role,
        )
        shares.append(metadata["event_effect_energy_share"])

    assert all(
        right > left
        for left, right in zip(shares, shares[1:], strict=False)
    )
    assert all(0.0 < share < 1.0 for share in shares)


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
