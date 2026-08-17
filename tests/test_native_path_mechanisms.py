from __future__ import annotations

import numpy as np

from cafe.benchmark_extension.gift_eval import GiftEvalInstance
from cafe.benchmark_extension.mechanisms import (
    SOURCE_DISTANCE_THRESHOLD,
    _distance_gate,
    build_capability_group,
)


def _instance(values: np.ndarray, *, horizon: int = 48) -> GiftEvalInstance:
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
        future_observed_mask=np.ones((horizon, history.shape[1]), dtype=bool),
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
        covariate_column_names=("calendar_sin_p24", "calendar_cos_p24"),
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


def test_treatment_distance_gate_has_no_upper_rejection_threshold() -> None:
    history = np.column_stack((np.arange(120.0), np.arange(120.0)))
    delta = np.full_like(history, 1000.0)
    gate = _distance_gate(delta, history, (0, 1))
    assert gate["accepted"]
    assert gate["maximum_observed_macro_distance"] > 1.0
    assert "maximum_macro_distance" not in gate
    assert "maximum_channel_distance" not in gate


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


def test_covariate_uses_known_calendar_path_and_hierarchy_is_qualification_only() -> None:
    t = np.arange(200.0)
    instance = _instance(np.sin(2.0 * np.pi * t / 24.0))
    covariate = build_capability_group(
        instance, "covariate_response", augmentation_seed=1
    )
    hierarchy = build_capability_group(
        instance, "hierarchical_coherence", augmentation_seed=1
    )
    assert covariate.available
    assert covariate.treatments[0].metadata[
        "known_future_covariate_path_used_for_delta"
    ]
    assert not hierarchy.available
