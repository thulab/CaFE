from __future__ import annotations

import copy

import pytest

from cafe import protocol
from cafe.generation.real_anchored_dose import additive_dose_reference
from cafe.generation.reference_bank import (
    build_combined_real_anchored_bank_split_audit,
    freeze_real_anchored_qualification_policy,
    split_real_anchored_background_banks,
    unavailable_real_anchored_qualification_policy,
    validate_evaluation_qualification_policy,
    validate_real_anchored_reference_chain,
)
from cafe.generation.real_anchored_policy import (
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
)


def _row(background_id: str, item_id: str, start: int) -> dict:
    return {
        "background_id": background_id,
        "item_id": item_id,
        "decomposition_start": start,
    }


def test_reference_and_evaluation_banks_are_temporally_disjoint() -> None:
    rows = [
        _row("a0", "panel-a", 0),
        _row("a1", "panel-a", 0),  # another synchronized channel
        _row("a2", "panel-a", 552),
        _row("a3", "panel-a", 1104),
        _row("b0", "panel-b", 0),
        _row("b1", "panel-b", 700),
    ]
    evaluation, reference, audit = split_real_anchored_background_banks(
        rows,
        maximum_evaluation_backgrounds=3,
        maximum_reference_backgrounds=3,
        source_window_length=552,
    )

    assert evaluation
    assert reference
    assert audit["cross_bank_temporal_overlap_count"] == 0
    roles = {
        row["background_id"]: row["background_bank_role"]
        for row in [*evaluation, *reference]
    }
    assert roles["a0"] == roles["a1"]
    for left in evaluation:
        for right in reference:
            if left["item_id"] != right["item_id"]:
                continue
            assert (
                left["decomposition_start"] + 552
                <= right["decomposition_start"]
                or right["decomposition_start"] + 552
                <= left["decomposition_start"]
            )


def test_reference_bank_split_is_deterministic_and_respects_caps() -> None:
    rows = [_row(f"x{index}", "x", index * 600) for index in range(12)]
    first = split_real_anchored_background_banks(
        rows,
        maximum_evaluation_backgrounds=4,
        maximum_reference_backgrounds=3,
        source_window_length=552,
    )
    second = split_real_anchored_background_banks(
        list(reversed(rows)),
        maximum_evaluation_backgrounds=4,
        maximum_reference_backgrounds=3,
        source_window_length=552,
    )

    assert first == second
    assert len(first[0]) == 4
    assert len(first[1]) == 3
    assert first[2]["dropped_background_count"] == 5


def test_reference_bank_freezes_thresholds_and_rejects_evaluation_drift() -> None:
    rows = [_row(f"x{index}", "x", index * 600) for index in range(8)]
    evaluation, reference, audit = split_real_anchored_background_banks(
        rows,
        maximum_evaluation_backgrounds=4,
        maximum_reference_backgrounds=4,
        source_window_length=552,
    )

    def contracts(bank: list[dict]) -> list[dict]:
        return [
            {
                "background_id": row["background_id"],
                "capability_id": "trend",
                "available": True,
                    "qualification_policy_id": "trend.reference.v1",
                    "qualification_threshold_source": (
                        QUALIFICATION_THRESHOLD_SOURCE_POLICY
                    ),
                "qualification_thresholds": {
                    "minimum_component_rms_ratio": 0.01,
                },
            }
            for row in bank
        ]

    policy = freeze_real_anchored_qualification_policy(
        contracts(reference),
        reference_background_ids=[row["background_id"] for row in reference],
        bank_split_audit=audit,
    )
    evaluation_contracts = contracts(evaluation)
    for row in evaluation_contracts:
        row["qualification_policy_sha256"] = policy[
            "qualification_policy_sha256"
        ]
    validate_evaluation_qualification_policy(evaluation_contracts, policy)
    drifted = contracts(evaluation)
    for row in drifted:
        row["qualification_policy_sha256"] = policy[
            "qualification_policy_sha256"
        ]
    drifted[0]["qualification_thresholds"] = {
        "minimum_component_rms_ratio": 0.02,
    }
    try:
        validate_evaluation_qualification_policy(drifted, policy)
    except ValueError as error:
        assert "differ" in str(error)
    else:
        raise AssertionError("evaluation threshold drift must be rejected")


def test_reference_policy_freezes_and_binds_capability_dose_mapping() -> None:
    rows = [_row(f"x{index}", "x", index * 600) for index in range(8)]
    evaluation, reference, audit = split_real_anchored_background_banks(
        rows,
        maximum_evaluation_backgrounds=4,
        maximum_reference_backgrounds=4,
        source_window_length=552,
    )

    def contracts(bank: list[dict], *, with_evidence: bool) -> list[dict]:
        result = []
        for index, row in enumerate(bank):
            contract = {
                "background_id": row["background_id"],
                "capability_id": "trend",
                "available": True,
                "qualification_policy_id": "trend.reference.v1",
                "qualification_threshold_source": (
                    QUALIFICATION_THRESHOLD_SOURCE_POLICY
                ),
                "qualification_thresholds": {"minimum": 0.01},
            }
            if with_evidence:
                contract["dose_design_reference"] = additive_dose_reference(
                    capability_id="trend",
                    background_id=row["background_id"],
                    unit_gain_history_separation=0.10,
                    unit_gain_future_separation=0.15 + 0.01 * index,
                    affected_channel_indices=(0,),
                )
            result.append(contract)
        return result

    policy = freeze_real_anchored_qualification_policy(
        contracts(reference, with_evidence=True),
        reference_background_ids=[row["background_id"] for row in reference],
        bank_split_audit=audit,
    )
    dose = policy["capabilities"]["trend"]["dose_calibration"]
    assert dose["status"] == "available"
    assert policy["dose_policy"]["capability_policy_sha256"]["trend"] == (
        dose["policy_sha256"]
    )

    evaluation_rows = contracts(evaluation, with_evidence=False)
    for row in evaluation_rows:
        row["qualification_policy_sha256"] = policy[
            "qualification_policy_sha256"
        ]
        row["dose_calibration"] = copy.deepcopy(dose)
    validate_evaluation_qualification_policy(evaluation_rows, policy)

    drifted = copy.deepcopy(evaluation_rows)
    drifted[0]["dose_calibration"]["history_target_grid"][-1] += 0.1
    with pytest.raises(ValueError, match="dose calibration policy hash"):
        validate_evaluation_qualification_policy(drifted, policy)


def test_empty_capability_reference_bank_freezes_fail_closed_policy() -> None:
    _evaluation, reference, audit = split_real_anchored_background_banks(
        [_row("x0", "x", 0), _row("x1", "x", 600)],
        maximum_evaluation_backgrounds=1,
        maximum_reference_backgrounds=1,
        source_window_length=552,
    )
    reference_ids = [str(row["background_id"]) for row in reference]

    policy = unavailable_real_anchored_qualification_policy(
        reference_background_ids=reference_ids,
        bank_split_audit=audit,
    )

    assert policy["capabilities"] == {}
    assert policy["formal_real_anchored_status"] == "unavailable"
    assert policy["formal_real_anchored_unavailable_reason"] == (
        "independent_reference_bank_unavailable"
    )
    assert policy["reference_bank"]["background_count"] == 1
    validate_evaluation_qualification_policy([], policy)


def test_qualification_policy_self_hash_and_decisions_fail_closed() -> None:
    rows = [_row(f"x{index}", "x", index * 600) for index in range(4)]
    evaluation, reference, audit = split_real_anchored_background_banks(
        rows,
        maximum_evaluation_backgrounds=2,
        maximum_reference_backgrounds=2,
        source_window_length=552,
    )
    reference_contracts = [
        {
            "background_id": row["background_id"],
            "capability_id": "trend",
            "available": True,
            "qualification_policy_id": "trend.reference.v1",
            "qualification_threshold_source": (
                QUALIFICATION_THRESHOLD_SOURCE_POLICY
            ),
            "qualification_thresholds": {"minimum": 0.01},
        }
        for row in reference
    ]
    policy = freeze_real_anchored_qualification_policy(
        reference_contracts,
        reference_background_ids=[row["background_id"] for row in reference],
        bank_split_audit=audit,
    )
    evaluation_rows = [
        {
            **reference_contracts[0],
            "background_id": row["background_id"],
            "qualification_policy_sha256": policy[
                "qualification_policy_sha256"
            ],
        }
        for row in evaluation
    ]
    tampered = dict(policy)
    tampered["decisions"] = {
        **policy["decisions"],
        "minimum_formal_background_count": 1,
    }
    with pytest.raises(ValueError, match="self-hash|decisions"):
        validate_evaluation_qualification_policy(evaluation_rows, tampered)


def test_combined_reference_chain_rejects_wrong_set_and_split_audit() -> None:
    candidates = [
        {
            **_row(f"x{index}", f"item-{index}", 0),
            "dataset_id": "fixture",
        }
        for index in range(6)
    ]
    evaluation, reference, base_audit = split_real_anchored_background_banks(
        candidates,
        maximum_evaluation_backgrounds=3,
        maximum_reference_backgrounds=3,
        source_window_length=552,
    )
    audit = build_combined_real_anchored_bank_split_audit(
        evaluation,
        reference,
        base_split_audit=base_audit,
    )
    reference_contracts = [
        {
            "background_id": row["background_id"],
            "capability_id": "trend",
            "available": True,
            "qualification_policy_id": "trend.reference.v1",
            "qualification_threshold_source": (
                QUALIFICATION_THRESHOLD_SOURCE_POLICY
            ),
            "qualification_thresholds": {"minimum": 0.01},
        }
        for row in reference
    ]
    policy = freeze_real_anchored_qualification_policy(
        reference_contracts,
        reference_background_ids=[row["background_id"] for row in reference],
        bank_split_audit=audit,
    )
    validate_real_anchored_reference_chain(
        evaluation,
        reference,
        audit,
        policy,
        reference_contract_rows=reference_contracts,
    )
    assert audit["component_assignment_sha256"] == base_audit[
        "component_assignment_sha256"
    ]
    assert audit["component_assignment_sha256"] != protocol.json_sha256(
        base_audit["component_assignment_sha256"]
    )

    with pytest.raises(ValueError, match="split audit disagrees"):
        validate_real_anchored_reference_chain(
            evaluation,
            reference[:-1],
            audit,
            policy,
            reference_contract_rows=reference_contracts,
        )

    forged_audit = copy.deepcopy(audit)
    forged_audit["combined_split"]["component_assignment_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="component assignment hash"):
        validate_real_anchored_reference_chain(
            evaluation,
            reference,
            forged_audit,
            policy,
            reference_contract_rows=reference_contracts,
        )
