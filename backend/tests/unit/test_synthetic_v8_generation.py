from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from app.services.synthetic_v8_generation import (
    GENERATOR_VERSION,
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
    basic_sample_checks,
    covariate_family_match_checks,
    nonlinear_mechanism_response_checks,
    paired_off_target_selectivity_matrix,
)


def test_basic_gate_rejects_non_real_intensity_grid() -> None:
    sample = {
        "target": np.zeros((384, 1)).tolist(),
        "context_length": 336,
        "horizon": 48,
        "target_dim": 1,
        "covariates": None,
        "covariate_dim": 0,
        "mase_scale": 1.0,
        "mase_period": 24,
        "target_feature_value": 0.5,
        "intensity_lambda": 0.5,
        "intensity_calibration": {
            "scope": "generator_relative_grid",
        },
    }

    result = basic_sample_checks(sample)

    assert result["accepted"] is False
    assert result["checks"]["intensity_grid_real_calibrated"] is False


def test_covariate_real_feature_contract_is_history_only():
    assert "future_abs_covariate_target_corr" not in (
        REQUIRED_REAL_FEATURES_BY_CAPABILITY["covariate_response"]
    )


def test_off_target_selectivity_matrix_is_paired_and_nonblocking():
    rows = []
    for capability_id, target_feature in (
        ("trend", "local_polynomial_energy_share_w96"),
        ("multi_seasonal", "multi_period_score"),
        (
            "time_varying_seasonality",
            "seasonal_amplitude_modulation",
        ),
    ):
        for seed in (2, 7):
            for intensity in (1, 5):
                dose = float(intensity)
                if capability_id == "multi_seasonal":
                    multi_period = dose + 0.01 * seed
                    amplitude_modulation = 2.0 * dose
                elif capability_id == "time_varying_seasonality":
                    multi_period = 0.1 * dose
                    amplitude_modulation = dose + 0.01 * seed
                else:
                    multi_period = 0.1 + 0.001 * dose
                    amplitude_modulation = 0.2 + 0.001 * dose
                rows.append(
                    {
                        "dataset_id": "demo",
                        "capability_id": capability_id,
                        "generator_family_role": "primary",
                        "evaluation_table": "main",
                        "counterfactual_member": None,
                        "seed_index": seed,
                        "intensity": intensity,
                        "target_feature_value": dose,
                        "realized_features": {
                            target_feature: dose + 0.01 * seed,
                            "local_polynomial_energy_share_w96": (
                                dose + 0.01 * seed
                                if capability_id == "trend"
                                else 0.2 + 0.001 * dose
                            ),
                            "multi_period_score": multi_period,
                            "seasonal_amplitude_modulation": (
                                amplitude_modulation
                            ),
                        },
                    }
                )

    result = paired_off_target_selectivity_matrix(rows)

    assert len(result) == 1
    assert result[0]["diagnostic_only"] is True
    assert result[0]["blocking"] is False
    assert result[0]["paired_seed_count_by_intervention"]["trend"] == 2
    assert result[0]["normalization"] == (
        "feature_owner_intervention_median_abs_paired_low_high_delta"
    )
    assert result[0]["feature_own_intervention_span"][
        "local_polynomial_energy_share_w96"
    ] == pytest.approx(4.0)
    assert result[0]["feature_own_intervention_span"][
        "pca_top1_explained"
    ] is None
    assert result[0]["normalized_absolute_delta_matrix"]["trend"][
        "pca_top1_explained"
    ] is None
    assert result[0]["normalized_absolute_delta_matrix"]["trend"][
        "local_polynomial_energy_share_w96"
    ] == pytest.approx(1.0)
    assert result[0]["normalized_absolute_delta_matrix"]["multi_seasonal"][
        "multi_period_score"
    ] == pytest.approx(1.0)
    assert result[0]["normalized_absolute_delta_matrix"][
        "time_varying_seasonality"
    ]["seasonal_amplitude_modulation"] == pytest.approx(1.0)
    assert result[0]["normalized_absolute_delta_matrix"]["multi_seasonal"][
        "seasonal_amplitude_modulation"
    ] == pytest.approx(2.0)
    assert result[0]["selectivity_summary"]["multi_seasonal"][
        "excluded_off_target_features"
    ] == ["seasonal_amplitude_modulation"]
    assert result[0]["selectivity_summary"]["multi_seasonal"][
        "maximum_nonexception_off_target_feature"
    ] == "local_polynomial_energy_share_w96"
    assert (
        result[0]["selectivity_summary"]["trend"][
            "on_to_max_off_target_ratio"
        ]
        > 1.0
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
    assert baseline_parameters["nonlinear_gain_scale"] == pytest.approx(2.1)
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
    assert all(
        result["dynamic_activity_gate"][
            "minimum_paired_positive_fraction"
        ]
        == pytest.approx(0.75)
        for result in results
    )

    # The activity ratio is a bounded diagnostic and can peak before I5 as
    # the nonlinear term begins to dominate the recurrence residual.  Its
    # paired low/high direction remains positive and is the hard condition.
    for row in rows:
        if (
            row["generator_family_role"] == "primary"
            and row["intensity"] == 4
        ):
            row["generation_metadata"][
                "nonlinear_effect_to_recurrence_residual_std_ratio"
            ] = 10.0
    folded_activity = nonlinear_mechanism_response_checks(rows)
    primary = next(
        result
        for result in folded_activity
        if result["family_role"] == "primary"
    )
    assert primary["dynamic_activity_gate"]["accepted"] is True

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
        "centered_rational_quadratic"
    )
    assert generated[("secondary", 1)]["nonlinear_transform"] == (
        "centered_tanh_quadratic"
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


@pytest.mark.parametrize(
    ("family_role", "expected_degree"),
    (("primary", 2), ("secondary", 3)),
)
def test_v8_trend_uses_local_c1_polynomial_with_tangent_extensions(
    family_role: str,
    expected_degree: int,
) -> None:
    target, metadata, _ = generate_deterministic_sample(
        "trend",
        600,
        504,
        3,
        24,
        5,
        np.random.default_rng(13),
        family_role=family_role,
    )

    join = int(metadata["trend_join_index"])
    design_stop = int(metadata["trend_design_stop_index"])
    direction = np.asarray(metadata["direction_by_target"])
    formal_differences = np.diff(target[:design_stop], axis=0)

    assert GENERATOR_VERSION == "capts-paper-v8-family-calibrated-v6"
    assert metadata["trend_local_evidence_window"] == 96
    assert join == 408
    assert metadata["trend_local_polynomial_degree"] == expected_degree
    assert metadata["trend_continuity_order"] == 1
    assert metadata["trend_prehistory_law"] == (
        "linear_tangent_at_local_join"
    )
    assert metadata["trend_postforecast_law"] == (
        "linear_tangent_at_design_horizon"
    )
    assert np.max(np.abs(np.diff(target[:join], n=2, axis=0))) < 1e-12
    assert np.all(
        formal_differences[: join - 1] * direction[None, :] > 0.0
    )
    assert max(
        np.abs(
            np.asarray(
                metadata[
                    "design_endpoint_derivative_ratio_by_target"
                ]
            )
            - 1.0
        )
    ) > 0.20
    assert metadata["slope_reversal_inside_design_window"] is True
    assert np.max(
        np.abs(np.diff(target[design_stop:], n=2, axis=0))
    ) < 1e-12

    local_coordinate = (
        np.arange(join, design_stop, dtype=float) - join
    ) / metadata["trend_local_evidence_window"]
    for target_index in range(target.shape[1]):
        coefficients = np.polyfit(
            local_coordinate,
            target[join:design_stop, target_index],
            deg=expected_degree,
        )
        fitted = np.polyval(coefficients, local_coordinate)
        assert np.max(
            np.abs(fitted - target[join:design_stop, target_index])
        ) < 1e-10
        assert abs(coefficients[0]) > 1e-6


@pytest.mark.parametrize("family_role", ("primary", "secondary"))
def test_v8_seasonal_time_scales_are_identifiable_in_l96(
    family_role: str,
) -> None:
    multi_metadata = generate_deterministic_sample(
        "multi_seasonal",
        552,
        504,
        1,
        400,
        5,
        np.random.default_rng(17),
        family_role=family_role,
    )[1]
    varying_metadata = generate_deterministic_sample(
        "time_varying_seasonality",
        552,
        504,
        1,
        400,
        5,
        np.random.default_rng(19),
        family_role=family_role,
    )[1]

    assert multi_metadata["period_evidence_window"] == 96
    assert multi_metadata["periods"][0] <= 32
    assert max(multi_metadata["periods"]) <= 48
    assert min(
        multi_metadata["cycles_in_shortest_evidence_window"]
    ) >= 2.0
    assert varying_metadata["period_evidence_window"] == 96
    assert varying_metadata["primary_period"] <= 32
    assert varying_metadata["modulation_period"] <= 96
    assert (
        varying_metadata["carrier_cycles_in_shortest_evidence_window"]
        >= 3.0
    )
    assert (
        varying_metadata[
            "modulation_cycles_in_shortest_evidence_window"
        ]
        >= 1.0
    )


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
def test_v8_common_factor_uses_blind_shared_state_to_recover_future(
    family_role: str,
) -> None:
    arguments = (
        "common_factor",
        552,
        504,
        5,
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
    assert first_metadata["local_factor_loading_orthogonalized"] is True
    assert first_metadata["main_task_is_dense_dynamic_factor"] is True
    assert first_metadata["generator_private_codebook_present"] is False
    assert first_metadata["directional_driver_present"] is False
    assert first_metadata["channel_specific_lag_present"] is False
    assert first_metadata["shared_state_evidence_width"] == 48
    assert first_metadata["shared_state_period"] in {12, 16, 24, 32}
    assert first_metadata["local_nuisance_path_pair_invariant"] is True
    assert first_metadata["dense_factor_strength"] > 0.0
    assert gate["generator_metadata_used_for_fitting"] is False
    assert (
        gate["observable_factor_share"]
        >= gate["minimum_observable_factor_share"]
    )
    assert gate["positive_control_effect_nrmse"] <= 0.15
    assert gate["positive_control_effect_correlation"] >= 0.95
    assert gate["accepted"] is True


def test_v8_zero_strength_regime_background_is_deterministic_but_not_exactly_seasonal(
) -> None:
    first, metadata, _ = generate_deterministic_sample(
        "regime_switching",
        384,
        336,
        1,
        24,
        1,
        np.random.default_rng(17),
    )
    repeated, _, _ = generate_deterministic_sample(
        "regime_switching",
        384,
        336,
        1,
        24,
        1,
        np.random.default_rng(17),
    )

    texture = metadata["deterministic_texture"]
    seasonal_scale = np.mean(
        np.abs(first[24:336] - first[:312])
    )

    assert metadata["regime_strength"] == 0.0
    assert np.array_equal(first, repeated)
    assert texture["law"] == "deterministic_quasiperiodic_two_tone"
    assert texture["period_ratio"] == pytest.approx(np.sqrt(2.0))
    assert texture["future_process_noise_scale"] == 0.0
    assert seasonal_scale > 1e-3


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
    assert 8 <= delay <= 24
    assert metadata["cross_lag_step"] == 1
    assert metadata["cross_lag_sampling_policy"] == (
        "real_anchor_lag_clipped_to_l96_identifiable_range"
    )
    expected_effect_steps = (
        delay if family_role == "primary" else delay + 2
    )
    assert metadata["history_covered_forecast_steps"] == (
        expected_effect_steps
    )
    assert metadata["counterfactual_effect_forecast_steps"] == (
        expected_effect_steps
    )
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
        effect_steps = int(
            first_metadata["counterfactual_effect_forecast_steps"]
        )
        assert context_length - delay + delay <= context_length
        assert not np.array_equal(
            first_view[
                context_length : context_length + effect_steps,
                1:,
            ],
            second_view[
                context_length : context_length + effect_steps,
                1:,
            ],
        )
        assert np.array_equal(
            first_view[context_length + effect_steps :, 1:],
            second_view[context_length + effect_steps :, 1:],
        )


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
    assert (
        abs(
            gate["blind_best_lag"]
            - first_metadata["cross_lag_steps"]
        )
        <= 2
    )
    assert (
        gate["aggregate_declared_incremental_holdout_gain"]
        >= gate["minimum_incremental_holdout_gain_threshold"]
    )
    assert first_metadata["effective_background_ratio"] == pytest.approx(
        first_metadata["calibrated_background_ratio"]
    )
    assert first_metadata["background_ratio_intensity_policy"] == (
        "fixed_calibrated_nuisance_across_intensity"
    )
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
        "intensity_target_feature_value": 0.48,
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
    secondary["target_feature_value"] = 0.52
    secondary["generation_metadata"] = {
        **metadata,
        "effect_strength": 0.43,
    }
    family_inverse = covariate_family_match_checks(primary, secondary)
    assert family_inverse["accepted"]
    assert family_inverse["family_specific_inverse_allowed"] is True
    secondary["intensity_target_feature_value"] = 0.49
    assert not covariate_family_match_checks(primary, secondary)["accepted"]
    secondary["intensity_target_feature_value"] = 0.48
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
            tuple(np.round(row["response_loadings"], 6)),
            row["shared_state_period"],
            row["shared_state_evidence_width"],
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

    noise_scale = np.asarray([0.4])
    observed, metadata = add_observation_noise_to_history(
        clean,
        context_length=504,
        noise_ratio=0.15,
        rng=np.random.default_rng(31),
        noise_scale_by_target=noise_scale,
        noise_scale_source="test_mase_denominator",
    )

    assert not np.array_equal(observed[:504], clean[:504])
    assert np.array_equal(observed[504:], clean[504:])
    assert metadata["future_noise_max_abs"] == 0.0
    assert metadata["noise_scale_source"] == "test_mase_denominator"
    assert metadata["requested_noise_scale_by_target"] == [0.4]
    assert metadata["requested_noise_to_scale_ratio"] == 0.15
    assert metadata["realized_noise_to_scale_ratio"] == pytest.approx(
        0.15,
        abs=0.02,
    )
    assert np.std(observed[:504] - clean[:504]) == pytest.approx(
        0.15 * noise_scale[0],
        abs=0.01,
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

    noise_scales = np.asarray([1.0, 0.2, 0.3, 0.4])
    observed, metadata = add_observation_noise_to_history(
        clean,
        context_length=504,
        noise_ratio=0.15,
        rng=np.random.default_rng(41),
        noise_scale_by_target=noise_scales,
        noise_scale_source="test_mase_denominator",
        preserve_additive_hierarchy=True,
    )

    assert np.max(
        np.abs(observed[:504, 0] - np.sum(observed[:504, 1:], axis=1))
    ) < 1e-12
    assert np.array_equal(observed[504:], clean[504:])
    assert metadata["effective_noise_scale_by_target"][0] == pytest.approx(
        np.sqrt(np.sum(np.square(noise_scales[1:])))
    )
    assert metadata["realized_noise_to_scale_ratio"] == pytest.approx(
        0.15,
        abs=0.02,
    )


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
