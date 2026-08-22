from __future__ import annotations

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.gift_eval import GiftEvalInstance
from cafe.benchmark_extension.mechanisms import (
    COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO,
    MECHANISM_EFFECT_MINIMUM_MASE_RMS,
    SOURCE_DISTANCE_MAXIMUM_CHANNEL,
    SOURCE_DISTANCE_MAXIMUM_MACRO,
    SOURCE_DISTANCE_MODEL_MAX_CONTEXTS,
    SOURCE_DISTANCE_THRESHOLD,
    TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION,
    _distance_gate,
    _nonlinear_innovation_bootstrap_paths,
    _nonlinear_state_response,
    build_capability_group,
    mase_scale_by_target,
    mechanism_effect_signal,
)


def _instance(
    values: np.ndarray,
    *,
    horizon: int = 48,
    future_observed_mask: np.ndarray | None = None,
) -> GiftEvalInstance:
    history = np.asarray(values, dtype=float)
    if history.ndim == 1:
        history = history[:, None]
    return GiftEvalInstance(
        dataset_id="gift_fixture",
        config_id="fixture/H",
        item_id="item",
        official_instance_id="gift__fixture__item__w00__o600",
        frequency="H",
        term="short",
        window_index=0,
        window_count=1,
        forecast_origin=history.shape[0],
        prediction_length=horizon,
        history=history,
        future=np.zeros((horizon, history.shape[1])),
        future_observed_mask=(
            np.ones((horizon, history.shape[1]), dtype=bool)
            if future_observed_mask is None
            else np.asarray(future_observed_mask, dtype=bool)
        ),
        history_covariates=np.column_stack(
            (
                np.sin(2.0 * np.pi * np.arange(history.shape[0]) / 24.0),
                np.cos(2.0 * np.pi * np.arange(history.shape[0]) / 24.0),
            )
        ),
        future_covariates=np.column_stack(
            (
                np.sin(
                    2.0
                    * np.pi
                    * np.arange(history.shape[0], history.shape[0] + horizon)
                    / 24.0
                ),
                np.cos(
                    2.0
                    * np.pi
                    * np.arange(history.shape[0], history.shape[0] + horizon)
                    / 24.0
                ),
            )
        ),
        covariate_column_names=(
            "past_feat_dynamic_real_0",
            "past_feat_dynamic_real_1",
        ),
        covariate_availability=("past_only", "past_only"),
        future_covariate_visible=(False, False),
        target_column_names=tuple(f"target_{i}" for i in range(history.shape[1])),
        source_target_length=history.shape[0] + horizon,
        history_imputation={"policy": "fixture"},
    )


def test_trend_follows_original_stable_direction_and_spans_full_history() -> None:
    t = np.arange(600.0)
    instance = _instance(0.02 * t + 0.1 * np.sin(t / 6.0))
    group = build_capability_group(instance, "trend", augmentation_seed=7)
    assert group.available
    for treatment in group.treatments:
        assert treatment.history_delta[0, 0] == 0.0
        assert treatment.history_delta[-1, 0] > 0.0
        assert treatment.future_delta[-1, 0] > treatment.history_delta[-1, 0]
        assert treatment.source_distance_gate["accepted"]


def test_regime_levels_control_location_and_share_amplitude() -> None:
    instance = _instance(np.sin(np.arange(600.0) / 12.0))
    group = build_capability_group(instance, "regime_switching", augmentation_seed=8)
    assert group.available
    joins = [row.metadata["change_index"] for row in group.treatments]
    assert joins == sorted(joins)
    amplitudes = [np.max(np.abs(row.future_delta)) for row in group.treatments]
    np.testing.assert_allclose(amplitudes, amplitudes[0])


def test_nonlinear_persistence_is_holdout_identifiable_and_self_recursive() -> None:
    rng = np.random.default_rng(123)
    values = np.zeros(5000, dtype=float)
    values[0] = 1.0
    for index in range(1, values.size):
        values[index] = (
            0.05 * values[index - 1]
            + float(_nonlinear_state_response(values[index - 1]))
            + rng.normal(0.0, 0.6)
        )
        values[index] = float(np.clip(values[index], -6.0, 6.0))
    instance = _instance(values[:4970])
    group = build_capability_group(
        instance, "nonlinear_persistence", augmentation_seed=7
    )
    assert group.available
    gate = group.group_metadata["nonlinear_identifiability_gate"]
    audit = gate["diagnostics_by_target"]["0"]
    assert gate["accepted"]
    assert audit["holdout_incremental_r2"] >= audit[
        "minimum_required_holdout_incremental_r2"
    ]
    assert audit["coefficient_sign_stable"]
    assert audit["multistep_holdout"]["accepted"]
    assert audit["multistep_holdout"]["incremental_r2"] > 0.0

    distances = [
        treatment.source_distance_gate["full_history_macro_normalized_rms"]
        for treatment in group.treatments
    ]
    assert distances == sorted(distances)
    assert all(treatment.horizon_support_gate["accepted"] for treatment in group.treatments)
    treatment = group.treatments[2]
    scale = float(np.std(instance.history[:, 0]))
    mean = float(np.mean(instance.history[:, 0]))
    source = (instance.history[:, 0] - mean) / scale
    treated = source + treatment.history_delta[:, 0] / scale
    detail = treatment.metadata["level_diagnostics_by_target"]["0"]
    nonlinear_coefficient = float(
        detail["nonlinear_persistence_coefficient"]
    )
    intercept = float(audit["linear_intercept"])
    persistence = float(audit["linear_persistence_coefficient"])
    source_innovations = source[1:] - (
        intercept + persistence * source[:-1]
    )
    reconstructed = (
        intercept
        + persistence * treated[:-1]
        + nonlinear_coefficient
        * np.asarray(_nonlinear_state_response(treated[:-1]))
        + source_innovations
    )
    np.testing.assert_allclose(treated[1:], reconstructed, atol=1e-12)

    assert treatment.metadata["future_innovation_policy"] == (
        "history_innovation_marginalized_shared_path_mean"
    )
    bootstrap = treatment.metadata[
        "future_innovation_bootstrap_by_target"
    ]["0"]
    paths, replayed_bootstrap = _nonlinear_innovation_bootstrap_paths(
        source_innovations,
        horizon=instance.prediction_length,
        path_count=int(bootstrap["path_count"]),
        seed=int(bootstrap["seed"]),
    )
    assert np.any(np.abs(paths) > 0.0)
    assert replayed_bootstrap["block_length"] == bootstrap["block_length"]
    linear_states = np.full(paths.shape[0], source[-1], dtype=float)
    nonlinear_states = np.full(paths.shape[0], treated[-1], dtype=float)
    expected_future = np.empty(instance.prediction_length, dtype=float)
    deprecated_deterministic_future = np.empty(instance.prediction_length, dtype=float)
    deterministic_linear = float(source[-1])
    deterministic_nonlinear = float(treated[-1])
    for step in range(instance.prediction_length):
        linear_states = intercept + persistence * linear_states + paths[:, step]
        nonlinear_states = (
            intercept
            + persistence * nonlinear_states
            + nonlinear_coefficient
            * np.asarray(_nonlinear_state_response(nonlinear_states))
            + paths[:, step]
        )
        expected_future[step] = scale * float(
            np.mean(nonlinear_states - linear_states)
        )
        deterministic_linear = intercept + persistence * deterministic_linear
        deterministic_nonlinear = (
            intercept
            + persistence * deterministic_nonlinear
            + nonlinear_coefficient
            * float(_nonlinear_state_response(deterministic_nonlinear))
        )
        deprecated_deterministic_future[step] = scale * (
            deterministic_nonlinear - deterministic_linear
        )
    np.testing.assert_allclose(
        treatment.future_delta[:, 0], expected_future, atol=1e-12
    )
    assert not np.allclose(expected_future, deprecated_deterministic_future)


def test_nonlinear_empty_observed_future_is_finite_unavailable_metadata() -> None:
    rng = np.random.default_rng(123)
    values = np.zeros(5000, dtype=float)
    values[0] = 1.0
    for index in range(1, values.size):
        values[index] = (
            0.05 * values[index - 1]
            + float(_nonlinear_state_response(values[index - 1]))
            + rng.normal(0.0, 0.6)
        )
        values[index] = float(np.clip(values[index], -6.0, 6.0))
    instance = _instance(
        values[:4970],
        future_observed_mask=np.zeros((48, 1), dtype=bool),
    )
    group = build_capability_group(
        instance,
        "nonlinear_persistence",
        augmentation_seed=7,
    )
    assert not group.available
    assert group.reason == "level_1_nonlinear_future_effect_profile_empty"
    failed = group.group_metadata["failed_horizon_support_gate"]
    assert failed["observed_profile_count"] == 0
    assert failed["observed_tail_peak_ratio"] is None
    protocol.canonical_json(group.group_metadata)


def test_intermittency_levels_become_sparser_with_fixed_amplitude() -> None:
    instance = _instance(np.sin(np.arange(600.0) / 15.0))
    group = build_capability_group(
        instance,
        "predictable_intermittency",
        augmentation_seed=9,
    )
    assert group.available
    gaps = [row.metadata["event_gap"] for row in group.treatments]
    assert gaps == sorted(gaps)
    amplitudes = [np.max(row.history_delta) for row in group.treatments]
    np.testing.assert_allclose(amplitudes, amplitudes[0])
    assert all(row.metadata["future_event_count"] >= 1 for row in group.treatments)
    scales = mase_scale_by_target(instance.history, instance.frequency)
    for treatment in group.treatments:
        _, signal, observed_count = mechanism_effect_signal(
            treatment.future_delta,
            instance.future_observed_mask,
            scales,
            treatment.affected_target_indices,
        )
        assert observed_count > 0
        assert signal >= MECHANISM_EFFECT_MINIMUM_MASE_RMS - 1e-12


def test_intermittency_phase_targets_observed_future_cells() -> None:
    mask = np.zeros((48, 1), dtype=bool)
    mask[-1, 0] = True
    instance = _instance(
        np.sin(np.arange(600.0) / 15.0),
        future_observed_mask=mask,
    )
    group = build_capability_group(
        instance,
        "predictable_intermittency",
        augmentation_seed=13,
    )
    assert group.available
    scales = mase_scale_by_target(instance.history, instance.frequency)
    for treatment in group.treatments:
        _, signal, observed_count = mechanism_effect_signal(
            treatment.future_delta,
            mask,
            scales,
            treatment.affected_target_indices,
        )
        assert observed_count == 1
        assert signal >= MECHANISM_EFFECT_MINIMUM_MASE_RMS - 1e-12


def test_treatment_distance_gate_is_from_authentic_source_not_adjacent_level() -> None:
    t = np.arange(600.0)
    instance = _instance(0.03 * t + np.sin(t / 7.0))
    group = build_capability_group(instance, "trend", augmentation_seed=11)
    assert group.available
    assert all(
        treatment.source_distance_gate["scope"]
        == "treatment_history_vs_authentic_official_history"
        for treatment in group.treatments
    )
    assert all(
        treatment.source_distance_gate["minimum_observed_macro_distance"]
        >= SOURCE_DISTANCE_THRESHOLD - 1e-12
        for treatment in group.treatments
    )


def test_treatment_distance_gate_rejects_excessive_model_context_distance() -> None:
    history = np.column_stack((np.arange(120.0), np.arange(120.0)))
    delta = np.full_like(history, 1000.0)
    gate = _distance_gate(delta, history, (0, 1))
    assert not gate["accepted"]
    assert gate["reason"] == "above_maximum_model_context_macro_distance"
    assert gate["maximum_allowed_macro_distance"] == SOURCE_DISTANCE_MAXIMUM_MACRO
    assert (
        gate["maximum_allowed_channel_distance"]
        == SOURCE_DISTANCE_MAXIMUM_CHANNEL
    )


def test_distance_gate_uses_actual_model_contexts_and_separate_full_history() -> None:
    history = np.column_stack((np.arange(20_000.0), np.arange(20_000.0)))
    delta = np.ones_like(history)
    gate = _distance_gate(delta, history, (0, 1))
    expected = sorted(set(SOURCE_DISTANCE_MODEL_MAX_CONTEXTS.values()))
    assert gate["evaluated_model_contexts"] == expected
    assert [row["context_length"] for row in gate["by_model_context"]] == expected
    assert gate["full_history_context_length"] == 20_000
    assert 20_000 not in gate["evaluated_model_contexts"]


def test_strength_level_is_calibrated_on_full_history_not_weakest_context() -> None:
    t = np.arange(3000.0)
    instance = _instance(0.01 * t + np.sin(t / 11.0))
    group = build_capability_group(instance, "trend", augmentation_seed=19)
    assert group.available
    for treatment in group.treatments:
        gate = treatment.source_distance_gate
        assert treatment.controlled_coordinate == "full_history_macro_normalized_rms"
        assert np.isclose(
            gate["full_history_macro_normalized_rms"],
            treatment.sampled_coordinate,
        )


def test_common_and_cross_use_native_panel_without_channel_tasks() -> None:
    rng = np.random.default_rng(4)
    driver = np.sin(np.arange(600.0) / 9.0) + 0.05 * rng.normal(size=600)
    panel = np.column_stack(
        (
            driver,
            0.8 * np.roll(driver, 2) + 0.1 * rng.normal(size=600),
            -0.6 * driver + 0.1 * rng.normal(size=600),
        )
    )
    instance = _instance(panel)
    common = build_capability_group(instance, "common_factor", augmentation_seed=2)
    cross = build_capability_group(
        instance, "cross_series_dependence", augmentation_seed=2
    )
    assert common.available
    assert cross.available
    assert common.treatments[0].history_delta.shape == panel.shape
    assert cross.treatments[0].history_delta.shape == panel.shape


def test_tvs_fits_envelope_phase_and_keeps_future_envelope_supported() -> None:
    t = np.arange(600.0)
    carrier = np.sin(2.0 * np.pi * t / 12.0 + 0.2)
    envelope = 1.0 + 0.7 * np.sin(2.0 * np.pi * t / 120.0 + 0.8)
    instance = _instance(carrier * envelope, horizon=480)
    group = build_capability_group(
        instance,
        "time_varying_seasonality",
        augmentation_seed=23,
    )
    assert group.available
    treatment = group.treatments[0]
    details = treatment.metadata["resolved_periods_by_target"]["0"]
    assert treatment.metadata["component"] == (
        "history_fitted_constrained_am_carrier_envelope"
    )
    assert abs(details["envelope_cos_coefficient"]) > 1e-3
    assert details["am_incremental_r2"] > 0.5
    gate = treatment.horizon_support_gate
    assert gate is not None and gate["accepted"]
    assert gate["minimum_observed_active_fraction"] >= (
        TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION
    )


def test_common_uses_nondecaying_latent_harmonic_on_long_horizon() -> None:
    t = np.arange(600.0)
    factor = np.sin(2.0 * np.pi * t / 24.0 + 0.3)
    panel = np.column_stack((factor, 0.8 * factor, -0.6 * factor))
    instance = _instance(panel, horizon=480)
    group = build_capability_group(instance, "common_factor", augmentation_seed=29)
    assert group.available
    treatment = group.treatments[-1]
    assert treatment.metadata["component"] == (
        "history_pca_loading_with_stable_latent_harmonic"
    )
    assert "factor_ar1" not in treatment.metadata
    gate = treatment.horizon_support_gate
    assert gate is not None and gate["accepted"]
    assert gate["observed_tail_head_ratio"] >= (
        COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO
    )


def test_covariate_impulse_uses_native_past_only_path_and_has_future_energy() -> None:
    t = np.arange(200.0)
    instance = _instance(np.sin(2.0 * np.pi * t / 24.0))
    covariate = build_capability_group(
        instance, "covariate_impulse_response", augmentation_seed=1
    )
    hierarchy = build_capability_group(
        instance, "hierarchical_coherence", augmentation_seed=1
    )
    assert covariate.available
    treatment = covariate.treatments[0]
    assert treatment.metadata["covariate_availability"] == "past_only"
    assert not treatment.metadata["future_covariate_path_visible_to_model"]
    assert np.any(treatment.history_covariate_delta)
    assert not np.any(treatment.future_covariate_delta)
    scales = mase_scale_by_target(instance.history, instance.frequency)
    for row in covariate.treatments:
        _, signal, observed_count = mechanism_effect_signal(
            row.future_delta,
            instance.future_observed_mask,
            scales,
            row.affected_target_indices,
        )
        assert observed_count > 0
        assert signal >= MECHANISM_EFFECT_MINIMUM_MASE_RMS - 1e-12
        assert row.metadata[
            "constructed_minimum_future_effect_mase_rms"
        ] >= MECHANISM_EFFECT_MINIMUM_MASE_RMS
    assert not hierarchy.available
