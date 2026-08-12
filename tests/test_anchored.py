from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pytest

from cafe.generation.anchored import (
    AnchoredDecompositionContract,
    anchored_pair_delta,
    apply_anchored_counterfactual,
    apply_real_anchored_contract,
    fit_anchored_decomposition,
    fit_real_anchored_contract,
)


CONTEXT_LENGTH = 168
HORIZON = 48
CARRIER_PERIOD = 24.0
SECONDARY_PERIODS = (12.0, 32.0)
MODULATION_PERIOD = 48.0
REGIME_JOIN_INDEX = 110


def _real_path() -> np.ndarray:
    length = CONTEXT_LENGTH + HORIZON
    time = np.arange(length, dtype=float)
    local = np.maximum((time - 72.0) / 96.0, 0.0)
    first = (
        8.0
        + 0.35 * local
        + 0.90 * local**2
        + 2.2 * np.sin(2.0 * np.pi * time / CARRIER_PERIOD + 0.2)
        + 0.8 * np.sin(2.0 * np.pi * time / SECONDARY_PERIODS[0] + 0.5)
        + 0.45 * np.cos(2.0 * np.pi * time / SECONDARY_PERIODS[1] - 0.3)
        + 0.06 * np.sin(2.0 * np.pi * time / 7.3)
    )
    second = (
        -3.0
        - 0.20 * local
        - 0.55 * local**2
        + 1.4 * np.cos(2.0 * np.pi * time / CARRIER_PERIOD - 0.4)
        + 0.5 * np.cos(2.0 * np.pi * time / SECONDARY_PERIODS[0] + 0.1)
        + 0.35 * np.sin(2.0 * np.pi * time / SECONDARY_PERIODS[1] + 0.8)
        + 0.04 * np.cos(2.0 * np.pi * time / 8.7)
    )
    path = np.column_stack([first, second])
    future = np.arange(HORIZON, dtype=float)
    path[CONTEXT_LENGTH:, 0] += 0.12 * np.sin(future / 2.7)
    path[CONTEXT_LENGTH:, 1] -= 0.10 * np.cos(future / 3.1)
    return path


def _fit(path: np.ndarray | None = None) -> AnchoredDecompositionContract:
    return fit_anchored_decomposition(
        _real_path() if path is None else path,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        fit_window=96,
        trend_degree=2,
        minimum_cycles=2.0,
    )


def _time_varying_path() -> np.ndarray:
    path = _real_path()
    time = np.arange(path.shape[0], dtype=float)
    carrier_phase = 2.0 * np.pi * time / CARRIER_PERIOD
    modulation_phase = 2.0 * np.pi * time / MODULATION_PERIOD
    path[:, 0] += (
        0.75
        * np.cos(modulation_phase + 0.6)
        * np.sin(carrier_phase + 0.2)
    )
    path[:, 1] += (
        0.55
        * np.sin(modulation_phase - 0.4)
        * np.cos(carrier_phase - 0.3)
    )
    return path


def _regime_path() -> np.ndarray:
    path = _real_path()
    path[REGIME_JOIN_INDEX:] += np.asarray([1.35, -0.90])
    return path


def test_fit_is_history_only_under_arbitrary_future_perturbation() -> None:
    baseline = _real_path()
    perturbed = baseline.copy()
    rng = np.random.default_rng(20260812)
    perturbed[CONTEXT_LENGTH:] = rng.normal(
        loc=1e6,
        scale=1e4,
        size=perturbed[CONTEXT_LENGTH:].shape,
    )

    original = _fit(baseline)
    changed_future = _fit(perturbed)

    assert original.to_dict() == changed_future.to_dict()
    assert original.history_sha256 == changed_future.history_sha256
    np.testing.assert_array_equal(
        original.components().secondary,
        changed_future.components().secondary,
    )


@pytest.mark.parametrize("capability_id", ("multi_seasonal", "trend"))
def test_alpha_one_is_exact_identity(capability_id: str) -> None:
    baseline = _real_path()
    contract = _fit(baseline)

    member = apply_anchored_counterfactual(
        baseline,
        contract,
        capability_id=capability_id,
        alpha=1.0,
    )

    np.testing.assert_array_equal(member.values, baseline)
    np.testing.assert_array_equal(member.intervention, np.zeros_like(baseline))
    assert member.contract_sha256 == contract.contract_sha256


def test_multi_seasonal_scales_only_secondary_with_exact_pair_delta() -> None:
    baseline = _real_path()
    contract = _fit(baseline)
    components = contract.components()
    high = apply_anchored_counterfactual(
        baseline,
        contract,
        capability_id="multi_seasonal",
        alpha=1.8,
    )
    declared_delta = anchored_pair_delta(
        contract,
        capability_id="multi_seasonal",
        alpha_from=1.0,
        alpha_to=1.8,
    )

    np.testing.assert_array_equal(high.intervention, 0.8 * components.secondary)
    np.testing.assert_array_equal(high.intervention, declared_delta)
    np.testing.assert_allclose(
        high.values - baseline,
        declared_delta,
        rtol=0.0,
        atol=2e-15,
    )
    assert np.linalg.norm(components.carrier) > 0.0
    assert not np.allclose(declared_delta, 0.8 * components.carrier)


def test_trend_scales_local_nonlinearity_not_level_or_linear_trend() -> None:
    baseline = _real_path()
    contract = _fit(baseline)
    components = contract.components()
    member = apply_anchored_counterfactual(
        baseline,
        contract,
        capability_id="trend",
        alpha=1.5,
    )

    np.testing.assert_array_equal(
        member.intervention,
        0.5 * components.trend_nonlinearity,
    )
    np.testing.assert_array_equal(
        member.intervention[: contract.fit_start],
        np.zeros((contract.fit_start, contract.target_dim)),
    )
    assert not np.allclose(
        member.intervention,
        0.5
        * (
            components.level_and_linear_trend
            + components.trend_nonlinearity
        ),
    )
    assert np.linalg.norm(member.intervention[CONTEXT_LENGTH:]) > 0.0


def test_time_varying_scales_only_bounded_carrier_sidebands() -> None:
    baseline = _time_varying_path()
    contract = fit_anchored_decomposition(
        baseline,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        fit_window=96,
        modulation_period=MODULATION_PERIOD,
        minimum_cycles=2.0,
    )
    identity = apply_anchored_counterfactual(
        baseline,
        contract,
        capability_id="time_varying_seasonality",
        alpha=1.0,
    )
    high = apply_anchored_counterfactual(
        baseline,
        contract,
        capability_id="time_varying_seasonality",
        alpha=1.7,
    )
    components = contract.components()
    restored = AnchoredDecompositionContract.from_dict(
        json.loads(json.dumps(contract.to_dict(), allow_nan=False))
    )
    declared_delta = anchored_pair_delta(
        contract,
        capability_id="time_varying_seasonality",
        alpha_from=1.0,
        alpha_to=1.7,
    )

    np.testing.assert_array_equal(identity.values, baseline)
    np.testing.assert_array_equal(
        high.intervention,
        (1.7 - 1.0) * components.amplitude_modulation,
    )
    np.testing.assert_array_equal(high.intervention, declared_delta)
    assert np.linalg.norm(components.carrier) > 0.0
    assert np.linalg.norm(components.secondary) > 0.0
    assert not np.allclose(high.intervention, components.carrier)
    assert not np.allclose(high.intervention, components.secondary)
    np.testing.assert_allclose(
        components.amplitude_modulation[CONTEXT_LENGTH:],
        components.amplitude_modulation[
            CONTEXT_LENGTH - int(MODULATION_PERIOD) : CONTEXT_LENGTH
        ],
        rtol=0.0,
        atol=2e-14,
    )
    doses = [
        apply_anchored_counterfactual(
            baseline,
            contract,
            capability_id="time_varying_seasonality",
            alpha=alpha,
        ).intervention_rms
        for alpha in (1.0, 1.2, 1.5, 2.0)
    ]
    assert doses[0] == 0.0
    assert all(left < right for left, right in zip(doses, doses[1:]))
    assert restored == contract
    assert len(contract.to_dict()["modulation_sidebands"]) == 2


def test_regime_scales_only_history_joinpoint_constant_level_extension() -> None:
    baseline = _regime_path()
    contract = fit_anchored_decomposition(
        baseline,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        fit_window=96,
        regime_join_index=REGIME_JOIN_INDEX,
        minimum_regime_segment_length=24,
    )
    identity = apply_anchored_counterfactual(
        baseline,
        contract,
        capability_id="regime_switching",
        alpha=1.0,
    )
    high = apply_anchored_counterfactual(
        baseline,
        contract,
        capability_id="regime_switching",
        alpha=1.8,
    )
    components = contract.components()
    restored = AnchoredDecompositionContract.from_dict(
        json.loads(json.dumps(contract.to_dict(), allow_nan=False))
    )
    declared_delta = anchored_pair_delta(
        contract,
        capability_id="regime_switching",
        alpha_from=1.0,
        alpha_to=1.8,
    )

    np.testing.assert_array_equal(identity.values, baseline)
    np.testing.assert_array_equal(
        high.intervention,
        (1.8 - 1.0) * components.regime_level_shift,
    )
    np.testing.assert_array_equal(high.intervention, declared_delta)
    np.testing.assert_array_equal(
        components.regime_level_shift[:REGIME_JOIN_INDEX],
        np.zeros((REGIME_JOIN_INDEX, baseline.shape[1])),
    )
    expected_post_join = np.repeat(
        components.regime_level_shift[REGIME_JOIN_INDEX][None, :],
        components.regime_level_shift.shape[0] - REGIME_JOIN_INDEX,
        axis=0,
    )
    np.testing.assert_array_equal(
        components.regime_level_shift[REGIME_JOIN_INDEX:],
        expected_post_join,
    )
    assert np.linalg.norm(components.carrier) > 0.0
    assert not np.allclose(high.intervention, components.carrier)
    doses = [
        apply_anchored_counterfactual(
            baseline,
            contract,
            capability_id="regime_switching",
            alpha=alpha,
        ).intervention_rms
        for alpha in (1.0, 1.2, 1.5, 2.0)
    ]
    assert doses[0] == 0.0
    assert all(left < right for left, right in zip(doses, doses[1:]))
    assert restored == contract
    assert contract.to_dict()["regime_extension"] == (
        "constant_post_join_level"
    )


@pytest.mark.parametrize(
    ("capability_id", "path_factory", "extra_fit_args"),
    (
        (
            "time_varying_seasonality",
            _time_varying_path,
            {"modulation_period": MODULATION_PERIOD},
        ),
        (
            "regime_switching",
            _regime_path,
            {
                "regime_join_index": REGIME_JOIN_INDEX,
                "minimum_regime_segment_length": 24,
            },
        ),
    ),
)
def test_new_capability_fit_is_future_blind_and_references_are_shared(
    capability_id: str,
    path_factory: Callable[[], np.ndarray],
    extra_fit_args: dict[str, float | int],
) -> None:
    baseline = path_factory()
    changed_future = baseline.copy()
    changed_future[CONTEXT_LENGTH:] = np.random.default_rng(91).normal(
        1e5,
        1e3,
        size=changed_future[CONTEXT_LENGTH:].shape,
    )
    fit_args = {
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "carrier_period": CARRIER_PERIOD,
        "secondary_periods": SECONDARY_PERIODS,
        "fit_window": 96,
        **extra_fit_args,
    }

    original = fit_anchored_decomposition(baseline, **fit_args)
    perturbed = fit_anchored_decomposition(changed_future, **fit_args)
    low = apply_anchored_counterfactual(
        baseline,
        original,
        capability_id=capability_id,
        alpha=1.2,
    )
    high = apply_anchored_counterfactual(
        baseline,
        original,
        capability_id=capability_id,
        alpha=1.9,
    )

    assert original.to_dict() == perturbed.to_dict()
    np.testing.assert_array_equal(
        low.normalization_mean,
        high.normalization_mean,
    )
    np.testing.assert_array_equal(
        low.normalization_scale,
        high.normalization_scale,
    )
    np.testing.assert_array_equal(
        low.mase_scale_by_target,
        high.mase_scale_by_target,
    )
    assert np.isfinite(high.values).all()
    assert np.isfinite(high.normalized_values).all()


def test_new_capability_wrappers_freeze_required_history_parameters() -> None:
    time_path = _time_varying_path()[:, 0]
    regime_path = _regime_path()[:, 0]
    missing_modulation = fit_real_anchored_contract(
        time_path[:CONTEXT_LENGTH],
        capability_id="time_varying_seasonality",
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
    )
    missing_join = fit_real_anchored_contract(
        regime_path[:CONTEXT_LENGTH],
        capability_id="regime_switching",
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
    )

    assert missing_modulation["available"] is False
    assert missing_modulation["unavailable_reason"] == (
        "modulation_period_required"
    )
    assert missing_join["available"] is False
    assert missing_join["unavailable_reason"] == "regime_join_index_required"

    time_contract = fit_real_anchored_contract(
        time_path[:CONTEXT_LENGTH],
        capability_id="time_varying_seasonality",
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        modulation_period=MODULATION_PERIOD,
        horizon=HORIZON,
    )
    regime_contract = fit_real_anchored_contract(
        regime_path[:CONTEXT_LENGTH],
        capability_id="regime_switching",
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        regime_join_index=REGIME_JOIN_INDEX,
        horizon=HORIZON,
    )
    time_target, time_metadata = apply_real_anchored_contract(
        time_path,
        time_contract,
        alpha=1.6,
    )
    regime_target, regime_metadata = apply_real_anchored_contract(
        regime_path,
        regime_contract,
        alpha=1.6,
    )

    assert time_contract["available"] is True
    assert regime_contract["available"] is True
    assert np.isfinite(time_target).all()
    assert np.isfinite(regime_target).all()
    assert time_metadata["controlled_component"] == (
        "carrier_amplitude_modulation_sidebands"
    )
    assert time_metadata["modulation_extension"] == (
        "bounded_stationary_carrier_sidebands"
    )
    assert time_metadata["carrier_fixed"] is True
    assert time_metadata["secondary_fixed"] is True
    assert regime_metadata["controlled_component"] == (
        "history_joinpoint_level_shift"
    )
    assert regime_metadata["regime_join_index"] == REGIME_JOIN_INDEX
    assert regime_metadata["regime_extension"] == (
        "constant_post_join_level"
    )


def test_all_doses_reuse_baseline_normalization_and_mase_reference() -> None:
    baseline = _real_path()
    contract = _fit(baseline)
    members = [
        apply_anchored_counterfactual(
            baseline,
            contract,
            capability_id="multi_seasonal",
            alpha=alpha,
        )
        for alpha in (1.0, 1.3, 1.7, 2.0)
    ]
    history = baseline[:CONTEXT_LENGTH]
    expected_mean = np.mean(history, axis=0)
    expected_scale = np.std(history, axis=0)
    expected_mase = np.mean(
        np.abs(
            history[int(CARRIER_PERIOD) :]
            - history[: -int(CARRIER_PERIOD)]
        ),
        axis=0,
    )

    for member in members:
        np.testing.assert_array_equal(member.normalization_mean, expected_mean)
        np.testing.assert_array_equal(member.normalization_scale, expected_scale)
        np.testing.assert_array_equal(member.mase_scale_by_target, expected_mase)
        np.testing.assert_allclose(
            member.normalized_values,
            (member.values - expected_mean) / expected_scale,
            rtol=0.0,
            atol=2e-15,
        )
        assert member.metadata()["mase_reference_policy"] == (
            "baseline_history_shared_by_pair_v1"
        )


def test_component_extension_is_finite_and_dose_response_is_monotone() -> None:
    baseline = _real_path()
    contract = _fit(baseline)
    components = contract.components(CONTEXT_LENGTH + HORIZON)
    for values in (
        components.level_and_linear_trend,
        components.trend_nonlinearity,
        components.carrier,
        components.secondary,
        components.fitted,
    ):
        assert values.shape == baseline.shape
        assert np.isfinite(values).all()

    for capability_id in ("multi_seasonal", "trend"):
        members = [
            apply_anchored_counterfactual(
                baseline,
                contract,
                capability_id=capability_id,
                alpha=alpha,
            )
            for alpha in (1.0, 1.2, 1.5, 2.0)
        ]
        rms = [member.intervention_rms for member in members]
        assert rms[0] == 0.0
        assert all(left < right for left, right in zip(rms[1:], rms[2:]))
        assert all(np.isfinite(member.values).all() for member in members)
        assert all(
            np.isfinite(member.normalized_values).all()
            for member in members
        )


def test_contract_json_round_trip_is_exact_and_tamper_evident() -> None:
    contract = _fit()
    payload = json.loads(json.dumps(contract.to_dict(), allow_nan=False))

    restored = AnchoredDecompositionContract.from_dict(payload)

    assert restored == contract
    assert restored.contract_sha256 == contract.contract_sha256
    np.testing.assert_array_equal(
        restored.components().fitted,
        contract.components().fitted,
    )
    payload["coefficients"][0][0] += 1.0
    with pytest.raises(ValueError, match="integrity hash mismatch"):
        AnchoredDecompositionContract.from_dict(payload)


def test_calibration_and_generation_dict_wrappers_freeze_availability() -> None:
    baseline = _real_path()[:, 0]
    unavailable = fit_real_anchored_contract(
        baseline[:CONTEXT_LENGTH],
        capability_id="multi_seasonal",
        carrier_period=CARRIER_PERIOD,
    )
    assert unavailable["available"] is False
    assert unavailable["unavailable_reason"] == "secondary_periods_required"

    future_gated = fit_real_anchored_contract(
        baseline[:CONTEXT_LENGTH],
        capability_id="multi_seasonal",
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        horizon=HORIZON,
        minimum_component_rms_ratio=0.0,
        minimum_future_component_rms_ratio=1e6,
    )
    assert future_gated["available"] is False
    assert future_gated["unavailable_reason"] == (
        "controlled_future_component_too_weak"
    )
    assert future_gated["controlled_component_history_rms"] > 0.0
    assert future_gated["controlled_component_future_rms"] > 0.0
    assert future_gated["future_component_source"] == (
        "analytic_history_fitted_component_extension"
    )

    frozen = fit_real_anchored_contract(
        baseline[:CONTEXT_LENGTH],
        capability_id="multi_seasonal",
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        horizon=HORIZON,
    )
    target, metadata = apply_real_anchored_contract(
        baseline,
        json.loads(json.dumps(frozen, allow_nan=False)),
        alpha=1.0,
        context_length=CONTEXT_LENGTH,
    )

    assert frozen["available"] is True
    np.testing.assert_array_equal(target, baseline)
    assert metadata["output_units"] == "baseline_raw_units"
    assert metadata["carrier_fixed"] is True
    assert metadata["source_history_sha256"] == (
        frozen["source_history_sha256"]
    )
    tampered = json.loads(json.dumps(frozen, allow_nan=False))
    tampered["capability_id"] = "trend"
    with pytest.raises(ValueError, match="integrity hash mismatch"):
        apply_real_anchored_contract(
            baseline,
            tampered,
            alpha=1.2,
        )


def test_l504_fit_uses_visible_l336_reference_then_applies_to_full_l552() -> None:
    fit_length = 504
    visible_length = 336
    total_length = fit_length + HORIZON
    visible_start = fit_length - visible_length
    time = np.arange(total_length, dtype=float)
    local = np.maximum((time - (fit_length - 96)) / 96.0, 0.0)
    full_source = (
        20.0
        + 0.25 * local
        + 0.75 * local**2
        + 2.0 * np.sin(2.0 * np.pi * time / CARRIER_PERIOD + 0.3)
        + 0.7 * np.cos(2.0 * np.pi * time / SECONDARY_PERIODS[0])
        + 0.4 * np.sin(2.0 * np.pi * time / SECONDARY_PERIODS[1])
        + 0.05 * np.sin(time / 2.9)
    )
    visible_history = full_source[visible_start:fit_length]
    contract = fit_anchored_decomposition(
        full_source,
        context_length=fit_length,
        horizon=HORIZON,
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        fit_window=fit_length,
        minimum_cycles=3.0,
        reference_history=visible_history,
    )

    identity = apply_anchored_counterfactual(
        full_source,
        contract,
        capability_id="multi_seasonal",
        alpha=1.0,
    )
    high = apply_anchored_counterfactual(
        full_source,
        contract,
        capability_id="multi_seasonal",
        alpha=1.6,
    )
    frozen = fit_real_anchored_contract(
        full_source[:fit_length],
        capability_id="multi_seasonal",
        carrier_period=CARRIER_PERIOD,
        secondary_periods=SECONDARY_PERIODS,
        horizon=HORIZON,
        fit_window=fit_length,
        minimum_cycles=3.0,
        reference_history=visible_history,
    )
    wrapped_high, wrapped_metadata = apply_real_anchored_contract(
        full_source,
        frozen,
        alpha=1.6,
        context_length=fit_length,
    )

    assert contract.context_length == fit_length
    assert contract.fit_start == 0
    assert contract.trend_start == fit_length - 96
    assert contract.trend_window == 96
    assert contract.reference_start == visible_start
    assert contract.reference_length == visible_length
    assert contract.history_sha256 != contract.reference_history_sha256
    np.testing.assert_array_equal(
        identity.values[visible_start:],
        full_source[visible_start:],
    )
    np.testing.assert_array_equal(
        identity.normalization_mean,
        np.asarray([np.mean(visible_history)]),
    )
    np.testing.assert_array_equal(
        identity.normalization_scale,
        np.asarray([np.std(visible_history)]),
    )
    assert high.values[visible_start:].shape == (
        visible_length + HORIZON,
    )
    np.testing.assert_array_equal(wrapped_high, high.values)
    assert wrapped_metadata["reference_length"] == visible_length
    np.testing.assert_allclose(
        high.intervention,
        (1.6 - 1.0) * contract.components(total_length).secondary[:, 0],
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="fit-history suffix"):
        fit_anchored_decomposition(
            full_source,
            context_length=fit_length,
            horizon=HORIZON,
            carrier_period=CARRIER_PERIOD,
            secondary_periods=SECONDARY_PERIODS,
            fit_window=fit_length,
            minimum_cycles=3.0,
            reference_history=full_source[visible_start - 1 : fit_length - 1],
        )
