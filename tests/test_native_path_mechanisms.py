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
    STRENGTH_INTERVALS,
    TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION,
    _dominant_frequency_indexes,
    _distance_gate,
    _independent_seasonal_period,
    _nonlinear_innovation_bootstrap_paths,
    _nonlinear_state_response,
    _strength_feasible_sampling_intervals,
    build_capability_group,
    mase_scale_by_target,
    mechanism_effect_signal,
    replay_treatment_deltas,
    replay_treatment_deltas_for_history_suffix,
)
from cafe.benchmark_extension.generation import (
    compact_contract_row,
    materialized_samples_for_instance,
)


def _instance(
    values: np.ndarray,
    *,
    horizon: int = 48,
    term: str = "short",
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
        term=term,
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


def test_medium_treatments_exclude_tirex_distance_context() -> None:
    t = np.arange(10_000.0)
    instance = _instance(
        0.01 * t + np.sin(t / 11.0), horizon=480, term="medium"
    )
    group = build_capability_group(instance, "trend", augmentation_seed=19)
    assert group.available
    for treatment in group.treatments:
        contexts = treatment.source_distance_gate["model_max_contexts"]
        assert "tirex2" not in contexts
        assert set(contexts) == {
            "Timer-4.0",
            "Chronos-2",
            "Timer-3.5",
            "timesfm2.5",
            "moirai2",
            "toto2.0",
        }


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


def test_multi_seasonal_levels_add_nested_independent_periods_at_fixed_rms() -> None:
    length = 1200
    t = np.arange(float(length))
    frequency_indexes = (13, 17, 23, 29, 37, 43)
    amplitudes = (2.0, 0.8, 0.7, 0.6, 0.5, 0.4)
    values = sum(
        amplitude
        * np.sin(
            2.0 * np.pi * frequency_index * t / length + 0.1 * index
        )
        for index, (frequency_index, amplitude) in enumerate(
            zip(frequency_indexes, amplitudes, strict=True)
        )
    )
    instance = _instance(values, horizon=120)
    group = build_capability_group(
        instance, "multi_seasonal", augmentation_seed=17
    )
    assert group.available
    shared_distances = []
    for level, treatment in enumerate(group.treatments, start=1):
        assert treatment.controlled_coordinate == (
            "additional_independent_period_count"
        )
        assert treatment.sampled_coordinate == float(level)
        details = treatment.metadata["resolved_periods_by_target"]["0"]
        assert details["anchor_source"] == "history_top3_stable_harmonic"
        assert len(details["components"]) == level + 1
        assert details["components"][0]["role"] == "anchor"
        assert details["components"][0]["frequency_index"] == (
            frequency_indexes[0]
        )
        assert details["history_anchor_search"]["accepted_rank"] == 1
        assert all(
            row["role"] == "additional"
            and row["source"] == "protocol_generated"
            and np.isclose(
                row["history_normalized_std_before_aggregate_gain"], 1.0
            )
            for row in details["components"][1:]
        )
        periods = [row["period"] for row in details["components"]]
        assert all(
            _independent_seasonal_period(
                left,
                right,
                min(length, 2048),
            )
            for index, left in enumerate(periods)
            for right in periods[index + 1 :]
        )
        shared_distances.append(
            treatment.source_distance_gate[
                "full_history_macro_normalized_rms"
            ]
        )
    np.testing.assert_allclose(shared_distances, shared_distances[0])

    dense_rows = [
        row
        for kind, row in materialized_samples_for_instance(
            instance,
            augmentation_seed=17,
            capability_ids=("multi_seasonal",),
        )
        if kind == "capability_treatments"
    ]
    contracts = [compact_contract_row(row) for row in dense_rows]
    replayed = replay_treatment_deltas(instance, contracts)
    for row in dense_rows:
        context = int(row["context_length"])
        history_delta, future_delta, _covariate_h, _covariate_f = replayed[
            str(row["sample_id"])
        ]
        np.testing.assert_array_equal(
            instance.history + history_delta,
            np.asarray(row["target"][:context]),
        )
        np.testing.assert_array_equal(
            instance.future + future_delta,
            np.asarray(row["target"][context:]),
        )


def test_multi_seasonal_checks_next_history_anchor_candidate() -> None:
    length = 1200
    t = np.arange(float(length))
    values = 2.0 * np.sin(2.0 * np.pi * 8 * t / length) + 1.5 * np.sin(
        2.0 * np.pi * 13 * t / length + 0.3
    )
    group = build_capability_group(
        _instance(values, horizon=48),
        "multi_seasonal",
        augmentation_seed=17,
    )
    assert group.available
    details = group.treatments[0].metadata["resolved_periods_by_target"]["0"]
    assert details["anchor_source"] == "history_top3_stable_harmonic"
    search = details["history_anchor_search"]
    assert search["accepted_rank"] == 2
    assert search["accepted_frequency_index"] == 13
    assert search["attempts"][0]["rejection_reasons"] == [
        "period_outside_supported_range"
    ]


def test_multi_seasonal_uses_protocol_anchor_when_history_has_none() -> None:
    length = 1200
    group = build_capability_group(
        _instance(np.zeros(length), horizon=120),
        "multi_seasonal",
        augmentation_seed=17,
    )
    assert group.available
    for level, treatment in enumerate(group.treatments, start=1):
        details = treatment.metadata["resolved_periods_by_target"]["0"]
        assert details["anchor_source"] == "protocol_generated"
        assert len(details["components"]) == level + 1
        assert all(
            component["source"] == "protocol_generated"
            for component in details["components"]
        )


def test_multi_seasonal_rejects_context_too_short_for_period_pool() -> None:
    group = build_capability_group(
        _instance(np.zeros(12), horizon=48),
        "multi_seasonal",
        augmentation_seed=17,
    )
    assert not group.available
    assert group.reason == "history_or_horizon_too_short_for_artificial_period_pool"


def test_dominant_frequency_indexes_never_leave_declared_range() -> None:
    values = np.arange(24.0)
    indexes = _dominant_frequency_indexes(values)
    assert indexes
    assert all(2 <= index <= len(values) // 4 for index in indexes)


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


def test_default_capabilities_draw_seed_specific_structures_shared_by_levels() -> None:
    length = 1200
    t = np.arange(float(length))
    rng = np.random.default_rng(77)
    carrier = np.sin(2.0 * np.pi * t / 24.0 + 0.2)
    envelope = 1.0 + 0.6 * np.sin(2.0 * np.pi * t / 240.0 + 0.8)
    driver = 0.006 * t + carrier * envelope + 0.05 * rng.normal(size=length)
    panel = np.column_stack(
        (
            driver,
            0.85 * np.roll(driver, 3) + 0.08 * rng.normal(size=length),
            -0.65 * driver + 0.08 * rng.normal(size=length),
            0.45 * np.roll(driver, 7) + 0.10 * rng.normal(size=length),
        )
    )
    instance = _instance(panel, horizon=120)
    capabilities = (
        "trend",
        "multi_seasonal",
        "time_varying_seasonality",
        "regime_switching",
        "predictable_intermittency",
        "common_factor",
        "cross_series_dependence",
        "covariate_impulse_response",
    )
    for capability_id in capabilities:
        first = build_capability_group(
            instance, capability_id, augmentation_seed=101
        )
        repeated = build_capability_group(
            instance, capability_id, augmentation_seed=101
        )
        alternatives = [
            build_capability_group(
                instance, capability_id, augmentation_seed=seed
            )
            for seed in (202, 303, 404)
        ]
        assert first.available and repeated.available
        assert all(group.available for group in alternatives)
        assert first.group_metadata["structure_shared_across_levels"] is True
        assert first.group_metadata["structure_draw_sha256"] == (
            repeated.group_metadata["structure_draw_sha256"]
        )
        structure_hashes = {
            first.group_metadata["structure_draw_sha256"],
            *(
                group.group_metadata["structure_draw_sha256"]
                for group in alternatives
            ),
        }
        assert len(structure_hashes) >= 2
        assert len(first.treatments) == 5


def test_qualified_structure_pools_are_seed_invariant_and_diverse() -> None:
    length = 1200
    t = np.arange(float(length))
    rng = np.random.default_rng(117)
    driver = np.sin(2.0 * np.pi * t / 24.0) + 0.04 * rng.normal(size=length)
    instance = _instance(
        np.column_stack(
            (
                driver,
                0.9 * np.roll(driver, 2) + 0.06 * rng.normal(size=length),
                -0.7 * np.roll(driver, 5) + 0.08 * rng.normal(size=length),
            )
        ),
        horizon=120,
    )
    for capability_id in (
        "cross_series_dependence",
        "covariate_impulse_response",
    ):
        groups = [
            build_capability_group(
                instance,
                capability_id,
                augmentation_seed=seed,
            )
            for seed in range(8)
        ]
        assert all(group.available for group in groups)
        assert len(
            {
                group.group_metadata["structure_draw_sha256"]
                for group in groups
            }
        ) >= 2
        for group in groups:
            assert all(
                treatment.source_distance_gate["accepted"]
                for treatment in group.treatments
            )
        if capability_id == "cross_series_dependence":
            counts = {
                group.group_metadata["structure_metadata"][
                    "source_distance_qualified_edge_count"
                ]
                for group in groups
            }
        else:
            counts = {
                group.group_metadata["qualified_candidate_count"]
                for group in groups
            }
        assert len(counts) == 1
        assert next(iter(counts)) >= 2


def test_strength_levels_use_expanded_nominal_and_feasible_subintervals() -> None:
    assert STRENGTH_INTERVALS == (
        (0.10, 0.15),
        (0.17, 0.22),
        (0.25, 0.32),
        (0.36, 0.46),
        (0.50, 0.65),
    )
    length = 12000
    rng = np.random.default_rng(181)
    instance = _instance(rng.normal(size=length), horizon=48)
    component = np.ones_like(instance.history)
    component[-2048:] *= 0.75
    intervals, metadata = _strength_feasible_sampling_intervals(
        instance,
        component,
        (0,),
    )
    assert intervals is not None
    assert STRENGTH_INTERVALS[0][0] < intervals[0][0] < intervals[0][1]
    assert intervals[0][1] == STRENGTH_INTERVALS[0][1]
    assert metadata["accepted"]
    assert metadata["empty_levels"] == []


def test_structurally_randomized_default_capabilities_replay_exactly() -> None:
    length = 1200
    t = np.arange(float(length))
    rng = np.random.default_rng(91)
    driver = (
        0.004 * t
        + np.sin(2.0 * np.pi * t / 24.0)
        * (1.0 + 0.5 * np.sin(2.0 * np.pi * t / 240.0 + 0.3))
        + 0.05 * rng.normal(size=length)
    )
    instance = _instance(
        np.column_stack(
            (
                driver,
                0.8 * np.roll(driver, 2) + 0.1 * rng.normal(size=length),
                -0.7 * driver + 0.1 * rng.normal(size=length),
            )
        ),
        horizon=120,
    )
    capabilities = (
        "trend",
        "multi_seasonal",
        "time_varying_seasonality",
        "regime_switching",
        "predictable_intermittency",
        "common_factor",
        "cross_series_dependence",
        "covariate_impulse_response",
    )
    dense_rows = [
        row
        for kind, row in materialized_samples_for_instance(
            instance,
            augmentation_seed=303,
            capability_ids=capabilities,
        )
        if kind == "capability_treatments"
    ]
    assert {row["capability_id"] for row in dense_rows} == set(capabilities)
    for capability_id in capabilities:
        rows = [
            row for row in dense_rows if row["capability_id"] == capability_id
        ]
        replayed = replay_treatment_deltas(
            instance, [compact_contract_row(row) for row in rows]
        )
        suffix_start = length - 257
        replayed_suffix = replay_treatment_deltas_for_history_suffix(
            instance,
            [compact_contract_row(row) for row in rows],
            history_start=suffix_start,
        )
        for row in rows:
            history_delta, future_delta, covariate_h, covariate_f = replayed[
                str(row["sample_id"])
            ]
            context = int(row["context_length"])
            np.testing.assert_array_equal(
                instance.history + history_delta,
                np.asarray(row["target"][:context]),
            )
            np.testing.assert_array_equal(
                instance.future + future_delta,
                np.asarray(row["target"][context:]),
            )
            expected_covariates = np.vstack(
                (
                    instance.history_covariates + covariate_h,
                    instance.future_covariates + covariate_f,
                )
            )
            np.testing.assert_array_equal(
                expected_covariates, np.asarray(row["covariates"])
            )
            suffix_history, suffix_future, suffix_covariate_h, suffix_covariate_f = (
                replayed_suffix[str(row["sample_id"])]
            )
            np.testing.assert_array_equal(
                suffix_history, history_delta[suffix_start:]
            )
            np.testing.assert_array_equal(suffix_future, future_delta)
            np.testing.assert_array_equal(
                suffix_covariate_h, covariate_h[suffix_start:]
            )
            np.testing.assert_array_equal(suffix_covariate_f, covariate_f)
