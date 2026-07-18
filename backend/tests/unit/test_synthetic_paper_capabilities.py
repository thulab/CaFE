import numpy as np

from app.services.synthetic_generation_service import (
    CAPABILITIES_BY_ID,
    CONTROL_FEATURES_BY_CAPABILITY,
    PAPER_CAPABILITY_IDS,
    PAPER_STRUCTURED_CAPABILITY_IDS,
    PAPER_UNIVARIATE_CAPABILITY_IDS,
    _generate_sample_values,
    _nonlinear_conditional_gain,
    _nonlinear_multi_lag_gain,
    _realized_features,
    _regime_clock_history_incremental_r2,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)


PRIMARY_INTENSITY_FEATURE = {
    "trend": "trend_strength",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "regime_clock_history_incremental_r2",
    "nonlinear_persistence": "nonlinear_conditional_gain",
    "predictable_intermittency": "spike_rate",
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    "covariate_response": "covariate_incremental_r2",
}


def test_paper_capability_set_is_six_univariate_plus_three_structured():
    assert len(PAPER_UNIVARIATE_CAPABILITY_IDS) == 6
    assert len(PAPER_STRUCTURED_CAPABILITY_IDS) == 3
    assert len(PAPER_CAPABILITY_IDS) == 9
    assert set(CAPABILITIES_BY_ID) == set(PAPER_CAPABILITY_IDS)
    assert CONTROL_FEATURES_BY_CAPABILITY["regime_switching"] == ()
    assert CONTROL_FEATURES_BY_CAPABILITY["predictable_intermittency"] == ()


def test_paper_generators_satisfy_predictability_construction_contracts():
    context_length = 168
    horizon = 24
    season_length = 24
    for capability_id in PAPER_CAPABILITY_IDS:
        target_dim = 3 if CAPABILITIES_BY_ID[capability_id].target_dim_mode == "multi" else 1
        values, metadata, covariates = _generate_sample_values(
            capability_id,
            context_length + horizon,
            context_length,
            target_dim,
            season_length,
            3,
            np.random.default_rng(20260715),
        )
        predictability = metadata["predictability"]
        assert predictability["construction_validated"] is True
        assert predictability["contract"]
        assert values.shape == (context_length + horizon, target_dim)
        if capability_id == "covariate_response":
            assert covariates is not None
            assert covariates.shape == (context_length + horizon, 2)
        else:
            assert covariates is None


def test_scheduled_and_structured_generators_do_not_hide_future_only_structure():
    context_length = 168
    horizon = 24
    season_length = 24

    _, regime, _ = _generate_sample_values(
        "regime_switching",
        context_length + horizon,
        context_length,
        1,
        season_length,
        4,
        np.random.default_rng(1),
    )
    cut_points = regime["cut_points"]
    assert sum(point < context_length for point in cut_points) >= 2
    assert any(point >= context_length for point in cut_points)
    dwell_pattern = regime["dwell_pattern"]
    assert len(dwell_pattern) >= 3
    assert len(set(dwell_pattern)) > 1
    observed_dwell = np.diff(cut_points)
    assert set(observed_dwell).issubset(set(dwell_pattern))
    assert (
        regime["predictability"]["evidence"][
            "duration_motif_repetitions_in_context"
        ]
        >= 2.0
    )

    _, intermittent, _ = _generate_sample_values(
        "predictable_intermittency",
        context_length + horizon,
        context_length,
        1,
        season_length,
        4,
        np.random.default_rng(2),
    )
    pulse_centers = intermittent["pulse_centers"]
    assert sum(center < context_length for center in pulse_centers) >= 2
    assert any(center >= context_length for center in pulse_centers)
    observed_intervals = set(np.diff(pulse_centers))
    assert len(observed_intervals) > 1
    assert observed_intervals == set(intermittent["pulse_interval_pattern"])

    hierarchy_values, hierarchy, _ = _generate_sample_values(
        "hierarchical_coherence",
        context_length + horizon,
        context_length,
        3,
        season_length,
        4,
        np.random.default_rng(3),
    )
    np.testing.assert_allclose(
        hierarchy_values[:, 0],
        np.sum(hierarchy_values[:, 1:], axis=1),
        atol=1e-12,
    )
    assert hierarchy["predictability"]["evidence"]["future_only_shock_count"] == 0

    _, covariate_metadata, covariates = _generate_sample_values(
        "covariate_response",
        context_length + horizon,
        context_length,
        1,
        season_length,
        4,
        np.random.default_rng(4),
    )
    assert covariates is not None
    assert np.any(covariates[context_length:, 1] == 1.0)
    assert covariate_metadata["future_event_start"] >= context_length
    assert len(covariate_metadata["weather_effect_by_target"]) == 1
    assert len(covariate_metadata["event_effect_by_target"]) == 1
    assert covariate_metadata["predictability"]["evidence"]["historical_event_count"] >= 2


def test_horizon_extension_preserves_the_existing_trajectory_prefix():
    context_length = 168
    short_horizon = 24
    long_horizon = 48
    season_length = 24
    for capability_id in PAPER_CAPABILITY_IDS:
        target_dim = 3 if CAPABILITIES_BY_ID[capability_id].target_dim_mode == "multi" else 1
        short_values, _, short_covariates = _generate_sample_values(
            capability_id,
            context_length + short_horizon,
            context_length,
            target_dim,
            season_length,
            3,
            np.random.default_rng(42),
        )
        long_values, _, long_covariates = _generate_sample_values(
            capability_id,
            context_length + long_horizon,
            context_length,
            target_dim,
            season_length,
            3,
            np.random.default_rng(42),
        )
        np.testing.assert_allclose(
            short_values,
            long_values[: context_length + short_horizon],
            atol=1e-12,
        )
        if short_covariates is None:
            assert long_covariates is None
        else:
            assert long_covariates is not None
            np.testing.assert_allclose(
                short_covariates,
                long_covariates[: context_length + short_horizon],
                atol=1e-12,
            )


def test_realized_primary_features_follow_intensity_direction():
    context_length = 168
    horizon = 24
    season_length = 24
    for capability_id, feature_name in PRIMARY_INTENSITY_FEATURE.items():
        target_dim = 3 if CAPABILITIES_BY_ID[capability_id].target_dim_mode == "multi" else 1
        intensity_means: list[float] = []
        for intensity in range(1, 6):
            values_for_intensity: list[float] = []
            for seed in range(24):
                values, metadata, covariates = _generate_sample_values(
                    capability_id,
                    context_length + horizon,
                    context_length,
                    target_dim,
                    season_length,
                    intensity,
                    np.random.default_rng(seed),
                )
                standardized = (
                    _standardize_hierarchy_by_context(values, context_length)
                    if capability_id == "hierarchical_coherence"
                    else _standardize_by_context(values, context_length)
                )
                features = _realized_features(
                    standardized,
                    covariates,
                    season_length,
                    context_length,
                )
                if capability_id == "regime_switching":
                    features["regime_clock_history_incremental_r2"] = (
                        _regime_clock_history_incremental_r2(
                            standardized,
                            context_length=context_length,
                            season_length=season_length,
                            cut_points=metadata["cut_points"],
                            dwell_length=int(metadata["dwell_length"]),
                        )
                    )
                values_for_intensity.append(features[feature_name])
            intensity_means.append(float(np.mean(values_for_intensity)))

        correlation = float(np.corrcoef(np.arange(1, 6), intensity_means)[0, 1])
        assert intensity_means[-1] > intensity_means[0], (
            capability_id,
            feature_name,
            intensity_means,
        )
        assert correlation > 0.8, (capability_id, feature_name, intensity_means)


def test_conditional_nonlinear_gain_does_not_label_plain_seasonality_as_nonlinear():
    rng = np.random.default_rng(20260717)
    time = np.arange(528, dtype=float)
    seasonal = np.sin(2 * np.pi * time / 24) + 0.08 * rng.normal(size=time.size)

    old_gain = _nonlinear_multi_lag_gain(seasonal, 24)
    conditional_gain = _nonlinear_conditional_gain(seasonal, 24)

    assert old_gain > 0.04
    assert abs(conditional_gain) < 0.01


def test_conditional_nonlinear_gain_detects_the_generated_recurrence():
    rng = np.random.default_rng(20260717)
    values = np.zeros(528, dtype=float)
    values[:24] = rng.normal(0.0, 1.00, size=24)
    for index in range(24, values.size):
        values[index] = (
            0.10 * values[index - 1]
            + 0.05 * values[index - 24]
            + 0.75
            * (np.sin(1.10 * values[index - 12]) ** 2 - 0.25)
            + 0.20 * rng.normal()
        )

    assert _nonlinear_conditional_gain(values, 24) > 0.005


def test_nonlinear_generator_discards_initialization_transient():
    ratios: list[float] = []
    for seed in range(64):
        values, metadata, _ = _generate_sample_values(
            "nonlinear_persistence",
            192,
            168,
            1,
            24,
            5,
            np.random.default_rng(seed),
        )
        first_cycle_scale = float(np.std(values[:24, 0]))
        stable_history_scale = float(np.std(values[24:168, 0]))
        ratios.append(
            first_cycle_scale / max(stable_history_scale, 1e-9)
        )
        assert metadata["burn_in_steps"] >= 256

    assert 0.5 < float(np.median(ratios)) < 2.0
    assert float(np.quantile(ratios, 0.90)) < 2.0
