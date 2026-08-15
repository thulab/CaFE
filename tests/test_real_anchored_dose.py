from __future__ import annotations

import copy

import numpy as np
import pytest

from cafe.generation.real_anchored_dose import (
    REAL_ANCHORED_CONTRACT_DOSE_CALIBRATION_SCHEMA,
    REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA,
    additive_dose_reference,
    build_dose_policy_summary,
    dose_targets,
    freeze_capability_dose_calibration,
    nonlinear_dose_reference,
    paired_minimum_separation_gate,
    resolve_contract_dose_calibration,
    validate_dose_calibration,
    validate_dose_policy_summary,
)


def _additive_row(
    background_id: str,
    *,
    history_unit_separation: float = 0.10,
    future_unit_separation: float = 0.15,
) -> dict:
    return {
        "background_id": background_id,
        "capability_id": "trend",
        "dose_design_reference": additive_dose_reference(
            capability_id="trend",
            background_id=background_id,
            unit_gain_history_separation=history_unit_separation,
            unit_gain_future_separation=future_unit_separation,
            affected_channel_indices=(0,),
        ),
    }


def test_reference_freezes_solver_but_contracts_resolve_distinct_alphas() -> None:
    rows = [
        _additive_row("b0", future_unit_separation=0.30),
        _additive_row("b1", future_unit_separation=0.20),
        _additive_row("b2", future_unit_separation=0.15),
        _additive_row(
            "b3",
            history_unit_separation=0.20,
            future_unit_separation=0.10,
        ),
    ]
    policy = freeze_capability_dose_calibration("trend", rows)

    assert policy["schema_version"] == REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA
    assert policy["status"] == "available"
    assert policy["mapping_scope"] == "contract_specific_history_only"
    assert policy["applied_alpha_grid"] == []
    assert policy["history_target_grid"][0] == pytest.approx(0.10)
    first = resolve_contract_dose_calibration(
        policy, rows[1]["dose_design_reference"]
    )
    last = resolve_contract_dose_calibration(
        policy, rows[-1]["dose_design_reference"]
    )
    assert first["schema_version"] == (
        REAL_ANCHORED_CONTRACT_DOSE_CALIBRATION_SCHEMA
    )
    assert first["applied_alpha_grid"] != last["applied_alpha_grid"]
    for resolved, row in ((first, rows[1]), (last, rows[-1])):
        gain = resolved["applied_alpha_grid"][0] - 1.0
        evidence = row["dose_design_reference"]
        assert gain * evidence["unit_gain_history_separation"] >= 0.10
        assert gain * evidence["unit_gain_future_separation"] >= 0.03
        validate_dose_calibration(resolved, capability_id="trend")


def test_additive_mapping_fails_closed_when_reference_requires_excess_gain() -> None:
    rows = [
        {
            "background_id": f"b{index}",
            "capability_id": "time_varying_seasonality",
            "dose_design_reference": additive_dose_reference(
                capability_id="time_varying_seasonality",
                background_id=f"b{index}",
                unit_gain_history_separation=0.005,
                unit_gain_future_separation=0.005,
                affected_channel_indices=(0,),
            ),
        }
        for index in range(4)
    ]
    policy = freeze_capability_dose_calibration(
        "time_varying_seasonality", rows
    )
    assert policy["status"] == "unavailable"
    assert policy["unavailable_reason"] == (
        "reference_contract_specific_dose_coverage_insufficient"
    )
    validate_dose_calibration(
        policy, capability_id="time_varying_seasonality"
    )


def test_history_future_imbalance_is_fail_closed() -> None:
    rows = [
        {
            "background_id": f"b{index}",
            "capability_id": "common_factor",
            "available": True,
            "dose_design_reference": additive_dose_reference(
                capability_id="common_factor",
                background_id=f"b{index}",
                unit_gain_history_separation=1.0,
                unit_gain_future_separation=0.05,
                affected_channel_indices=(0, 1, 2),
            ),
        }
        for index in range(4)
    ]
    policy = freeze_capability_dose_calibration("common_factor", rows)
    assert policy["status"] == "unavailable"
    assert policy["unavailable_reason"] == (
        "reference_contract_specific_dose_coverage_insufficient"
    )


def _nonlinear_curve(rate: float) -> list[dict]:
    return [
        {
            "alpha": float(1.0 + 0.005 * index),
            "history_separation": float(0.10 * rate * 0.005 * index),
            "future_separation": float(0.10 * rate * 0.005 * index),
            "safe": True,
        }
        for index in range(401)
    ]


def test_nonlinear_policy_resolves_history_only_contract_roots() -> None:
    rows = [
        {
            "background_id": f"n{index}",
            "capability_id": "nonlinear_persistence",
            "dose_design_reference": nonlinear_dose_reference(
                background_id=f"n{index}",
                zero_innovation_curve=_nonlinear_curve(rate),
                monotone=True,
            ),
        }
        for index, rate in enumerate((1.5, 1.25, 1.0, 0.75))
    ]
    policy = freeze_capability_dose_calibration(
        "nonlinear_persistence", rows
    )
    assert policy["status"] == "available"
    assert policy["applied_alpha_grid"] == []
    resolved = resolve_contract_dose_calibration(
        policy, rows[0]["dose_design_reference"]
    )
    assert len(resolved["applied_alpha_grid"]) == 5
    assert resolved["applied_alpha_grid"][-1] <= 3.0
    assert all(
        right > left
        for left, right in zip(
            resolved["applied_alpha_grid"],
            resolved["applied_alpha_grid"][1:],
        )
    )


def test_nonlinear_mapping_requires_three_monotone_reference_curves() -> None:
    rows = [
        {
            "background_id": f"n{index}",
            "capability_id": "nonlinear_persistence",
            "dose_design_reference": nonlinear_dose_reference(
                background_id=f"n{index}",
                zero_innovation_curve=_nonlinear_curve(1.5),
                monotone=index != 2,
            ),
        }
        for index in range(3)
    ]
    policy = freeze_capability_dose_calibration(
        "nonlinear_persistence", rows
    )
    assert policy["status"] == "unavailable"
    assert policy["unavailable_reason"] == (
        "insufficient_monotone_reference_dose_evidence"
    )


def test_nonlinear_mapping_rejects_flat_history_distance_grid() -> None:
    curve = [
        {
            "alpha": float(1.0 + 0.005 * index),
            "history_separation": float(min(0.10, 0.001 * index)),
            "future_separation": float(0.001 * index),
            "safe": True,
        }
        for index in range(401)
    ]
    rows = [
        {
            "background_id": f"flat{index}",
            "capability_id": "nonlinear_persistence",
            "dose_design_reference": nonlinear_dose_reference(
                background_id=f"flat{index}",
                zero_innovation_curve=curve,
                monotone=True,
            ),
        }
        for index in range(4)
    ]

    policy = freeze_capability_dose_calibration(
        "nonlinear_persistence", rows
    )

    assert policy["status"] == "unavailable"
    assert policy["unavailable_reason"] == (
        "nonlinear_reference_contract_specific_coverage_insufficient"
    )
    validate_dose_calibration(
        policy, capability_id="nonlinear_persistence"
    )


def _resolved_trend_calibration() -> dict:
    rows = [_additive_row(f"b{index}") for index in range(4)]
    policy = freeze_capability_dose_calibration("trend", rows)
    return resolve_contract_dose_calibration(
        policy, rows[0]["dose_design_reference"]
    )


def test_treatment_source_gate_uses_point_one_and_exempts_baseline() -> None:
    calibration = _resolved_trend_calibration()
    delta = np.zeros((336 + 48, 2), dtype=float)
    delta[336 - 168 : 336, :] = 0.10
    delta[336:, :] = 0.03
    result = paired_minimum_separation_gate(
        delta,
        context_length=336,
        dose_index=1,
        dose_calibration=calibration,
        affected_channel_indices=(0, 1),
    )
    assert result["accepted"] is True
    assert result["minimum_acceptance_fraction"] == 1.0
    assert result["minimum_history_macro_separation"] == pytest.approx(0.10)
    assert result["anti_copy_semantics"] == (
        "treatment_only_distance_from_authentic_source"
    )
    assert result["baseline_member_policy"] == "exact_authentic_source_exempt"

    too_close = delta.copy()
    too_close[336 - 168 : 336] = 0.099
    rejected = paired_minimum_separation_gate(
        too_close,
        context_length=336,
        dose_index=1,
        dose_calibration=calibration,
        affected_channel_indices=(0, 1),
    )
    assert rejected["accepted"] is False
    assert rejected["reason_code"] == (
        "real_anchor_treatment_too_close_to_authentic_source"
    )


def test_adjacent_dose_distance_is_diagnostic_only() -> None:
    calibration = _resolved_trend_calibration()
    current = np.zeros(336 + 48, dtype=float)
    current[336 - 168 : 336] = calibration["history_target_grid"][1]
    current[336:] = calibration["future_target_grid"][1]
    previous = current.copy()
    result = paired_minimum_separation_gate(
        current,
        context_length=336,
        dose_index=2,
        dose_calibration=calibration,
        affected_channel_indices=(0,),
        previous_delta=previous,
    )
    assert result["adjacent_accepted"] is False
    assert result["adjacent_distance_role"] == "diagnostic_only"
    assert result["accepted"] is True


def test_treatment_source_gate_rejects_over_amplified_path() -> None:
    calibration = _resolved_trend_calibration()
    delta = np.zeros(336 + 48, dtype=float)
    delta[336 - 168 : 336] = 1.2
    delta[336:] = 0.5
    result = paired_minimum_separation_gate(
        delta,
        context_length=336,
        dose_index=5,
        dose_calibration=calibration,
        affected_channel_indices=(0,),
    )
    assert result["local_augmentation_budget_passed"] is False
    assert result["accepted"] is False


def test_dose_policy_hash_binds_every_capability_mapping() -> None:
    calibration = freeze_capability_dose_calibration(
        "trend", [_additive_row(f"b{index}") for index in range(3)]
    )
    cells = {"trend": {"dose_calibration": calibration}}
    summary = build_dose_policy_summary(cells)
    validate_dose_policy_summary(summary, cells)
    tampered = copy.deepcopy(summary)
    tampered["capability_status"]["trend"] = "unavailable"
    with pytest.raises(ValueError, match="summary mismatch"):
        validate_dose_policy_summary(tampered, cells)


def test_dose_mapping_requires_three_eligible_reference_contracts() -> None:
    policy = freeze_capability_dose_calibration(
        "trend", [_additive_row(f"b{index}") for index in range(2)]
    )
    assert policy["status"] == "unavailable"
    assert policy["unavailable_reason"] == "insufficient_reference_dose_evidence"


def test_unavailable_reference_rows_do_not_enter_solver_coverage() -> None:
    eligible = [
        {
            **_additive_row(f"ok{index}"),
            "available": False,
            "sensitivity_available": True,
        }
        for index in range(3)
    ]
    for row in eligible:
        row["dose_design_reference"]["evidence_role"] = "sensitivity"
    ineligible = [
        {
            **_additive_row(
                f"bad{index}",
                history_unit_separation=0.001,
                future_unit_separation=0.001,
            ),
            "available": False,
            "sensitivity_available": False,
            "qualification_available": False,
        }
        for index in range(3)
    ]
    policy = freeze_capability_dose_calibration(
        "trend", [*eligible, *ineligible]
    )
    assert policy["status"] == "available"
    assert policy["reference_evidence_count"] == 3


def test_capability_targets_expose_source_distance_not_universal_alpha() -> None:
    additive = dose_targets("regime_switching")
    nonlinear = dose_targets("nonlinear_persistence")
    assert additive["source_distance_minimum"] == pytest.approx(0.10)
    assert additive["source_distance_member_scope"] == (
        "treatment_only_baseline_exempt"
    )
    assert additive["future_target_grid"][-1] == 0.15
    assert nonlinear["future_target_grid"][-1] == 0.10
