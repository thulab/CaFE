"""Disjoint qualification/evaluation banks for real-anchored tasks.

The split happens after authentic windows have been selected from disjoint
source strata.  Windows whose source intervals overlap within one native item
are kept in the same bank, including different channels of a synchronized
record.  Qualification rows are never emitted as forecast tasks.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from cafe.generation.real_anchored_policy import (
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
    REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
    protocol_decisions,
)


REAL_ANCHORED_BANK_SPLIT_SCHEMA = "cafe.real_anchored_bank_split.v1"
REAL_ANCHORED_BANK_SPLIT_POLICY = (
    "native_item_temporal_overlap_components_balanced_without_replacement_v1"
)
REAL_ANCHORED_COMBINED_BANK_SPLIT_SCHEMA = (
    "cafe.real_anchored_combined_bank_split.v1"
)
REAL_ANCHORED_COMBINED_BANK_SPLIT_POLICY = (
    "univariate_and_structural_source_time_disjoint_reference_"
    "evaluation_banks_v1"
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_item(row: Mapping[str, Any]) -> str:
    item_id = str(row.get("item_id", ""))
    if not item_id:
        raise ValueError("real-anchored background is missing item_id")
    return item_id


def _interval(row: Mapping[str, Any], *, window_length: int) -> tuple[int, int]:
    start = row.get("decomposition_start")
    if not isinstance(start, int) or start < 0:
        raise ValueError(
            "real-anchored background has an invalid decomposition_start"
        )
    return start, start + int(window_length)


def _overlap_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    window_length: int,
) -> list[list[Mapping[str, Any]]]:
    by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item[_source_item(row)].append(row)

    components: list[list[Mapping[str, Any]]] = []
    for item_id, item_rows in sorted(by_item.items()):
        ordered = sorted(
            item_rows,
            key=lambda row: (
                _interval(row, window_length=window_length)[0],
                str(row.get("background_id", "")),
            ),
        )
        current: list[Mapping[str, Any]] = []
        current_stop = -1
        for row in ordered:
            start, stop = _interval(row, window_length=window_length)
            # Half-open intervals that only touch at one endpoint do not
            # share observations and may be assigned independently.
            if current and start >= current_stop:
                components.append(current)
                current = []
                current_stop = -1
            current.append(row)
            current_stop = max(current_stop, stop)
        if current:
            components.append(current)

    def component_identity(component: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
        starts = [
            _interval(row, window_length=window_length)[0]
            for row in component
        ]
        return (
            _source_item(component[0]),
            min(starts),
            max(starts) + window_length,
            sorted(str(row.get("background_id", "")) for row in component),
        )

    return sorted(
        components,
        key=lambda component: (
            _canonical_hash(component_identity(component)),
            component_identity(component),
        ),
    )


def split_real_anchored_background_banks(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_evaluation_backgrounds: int,
    maximum_reference_backgrounds: int,
    source_window_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split authentic windows into disjoint reference and evaluation banks.

    Allocation is deterministic, balanced by accepted row count, and keeps
    each connected temporal-overlap component wholly on one side.  Rows above
    a bank's requested cap are dropped; they are never reassigned to the other
    side because doing so could reintroduce source overlap.
    """

    evaluation_cap = int(maximum_evaluation_backgrounds)
    reference_cap = int(maximum_reference_backgrounds)
    window_length = int(source_window_length)
    if evaluation_cap < 1 or reference_cap < 1:
        raise ValueError("real-anchored bank caps must be positive")
    if window_length < 1:
        raise ValueError("real-anchored source window length must be positive")

    background_ids = [str(row.get("background_id", "")) for row in rows]
    if any(not value for value in background_ids):
        raise ValueError("real-anchored backgrounds require non-empty IDs")
    if len(background_ids) != len(set(background_ids)):
        raise ValueError("real-anchored background IDs must be unique")

    components = _overlap_components(rows, window_length=window_length)
    banks: dict[str, list[dict[str, Any]]] = {
        "evaluation": [],
        "reference": [],
    }
    caps = {"evaluation": evaluation_cap, "reference": reference_cap}
    component_roles: list[dict[str, Any]] = []
    dropped_count = 0
    for component in components:
        available_roles = [
            role for role in ("evaluation", "reference")
            if len(banks[role]) < caps[role]
        ]
        if not available_roles:
            dropped_count += len(component)
            continue
        role = min(
            available_roles,
            key=lambda candidate: (
                len(banks[candidate]) / caps[candidate],
                candidate,
            ),
        )
        remaining = caps[role] - len(banks[role])
        selected = sorted(
            component,
            key=lambda row: str(row["background_id"]),
        )[:remaining]
        dropped_count += len(component) - len(selected)
        banks[role].extend(
            {
                **dict(row),
                "background_bank_role": role,
                "background_bank_split_policy": (
                    REAL_ANCHORED_BANK_SPLIT_POLICY
                ),
            }
            for row in selected
        )
        starts = [
            _interval(row, window_length=window_length)[0]
            for row in component
        ]
        component_roles.append(
            {
                "item_id": _source_item(component[0]),
                "start": min(starts),
                "stop": max(starts) + window_length,
                "role": role,
                "source_row_count": len(component),
                "accepted_row_count": len(selected),
                "accepted_background_ids": sorted(
                    str(row["background_id"]) for row in selected
                ),
            }
        )

    evaluation = sorted(
        banks["evaluation"], key=lambda row: str(row["background_id"])
    )
    reference = sorted(
        banks["reference"], key=lambda row: str(row["background_id"])
    )

    overlap_pairs: list[tuple[str, str]] = []
    for left in evaluation:
        left_start, left_stop = _interval(left, window_length=window_length)
        for right in reference:
            if _source_item(left) != _source_item(right):
                continue
            right_start, right_stop = _interval(
                right, window_length=window_length
            )
            if left_start < right_stop and right_start < left_stop:
                overlap_pairs.append(
                    (str(left["background_id"]), str(right["background_id"]))
                )
    if overlap_pairs:
        raise ValueError(
            "reference/evaluation real-anchored banks overlap in source time"
        )

    audit = {
        "schema_version": REAL_ANCHORED_BANK_SPLIT_SCHEMA,
        "policy": REAL_ANCHORED_BANK_SPLIT_POLICY,
        "source_window_length": window_length,
        "source_background_count": len(rows),
        "temporal_overlap_component_count": len(components),
        "evaluation_background_count": len(evaluation),
        "reference_background_count": len(reference),
        "dropped_background_count": dropped_count,
        "cross_bank_temporal_overlap_count": 0,
        "evaluation_background_ids_sha256": _canonical_hash(
            [str(row["background_id"]) for row in evaluation]
        ),
        "evaluation_background_ids": [
            str(row["background_id"]) for row in evaluation
        ],
        "reference_background_ids_sha256": _canonical_hash(
            [str(row["background_id"]) for row in reference]
        ),
        "reference_background_ids": [
            str(row["background_id"]) for row in reference
        ],
        "component_assignment_sha256": _canonical_hash(component_roles),
        "component_assignments": component_roles,
        "threshold_tuning_policy": (
            "qualification_only_reference_bank_never_evaluation_origins"
        ),
    }
    return evaluation, reference, audit


def _bank_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> list[str]:
    ids = sorted(str(row.get("background_id", "")) for row in rows)
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{role} background IDs must be non-empty and unique")
    ordered = sorted(rows, key=lambda item: str(item.get("background_id", "")))
    mismatched_roles = [
        value
        for value, row in zip(ids, ordered)
        if row.get("background_bank_role") != role
    ]
    if mismatched_roles:
        raise ValueError(f"{role} backgrounds lost their frozen bank role")
    mismatched_policies = [
        value
        for value, row in zip(ids, ordered)
        if row.get("background_bank_split_policy")
        != REAL_ANCHORED_BANK_SPLIT_POLICY
    ]
    if mismatched_policies:
        raise ValueError(f"{role} backgrounds lost their frozen split policy")
    return ids


def _cross_bank_temporal_overlap_pairs(
    evaluation_backgrounds: Sequence[Mapping[str, Any]],
    reference_backgrounds: Sequence[Mapping[str, Any]],
    *,
    source_window_length: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for evaluation in evaluation_backgrounds:
        evaluation_start, evaluation_stop = _interval(
            evaluation,
            window_length=source_window_length,
        )
        for reference in reference_backgrounds:
            if _source_item(evaluation) != _source_item(reference):
                continue
            reference_start, reference_stop = _interval(
                reference,
                window_length=source_window_length,
            )
            if (
                evaluation_start < reference_stop
                and reference_start < evaluation_stop
            ):
                pairs.append(
                    (
                        str(evaluation["background_id"]),
                        str(reference["background_id"]),
                    )
                )
    return sorted(pairs)


def build_combined_real_anchored_bank_split_audit(
    evaluation_backgrounds: Sequence[Mapping[str, Any]],
    reference_backgrounds: Sequence[Mapping[str, Any]],
    *,
    base_split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the final univariate+structural banks to the base split audit."""

    if base_split_audit.get("schema_version") != REAL_ANCHORED_BANK_SPLIT_SCHEMA:
        raise ValueError("combined bank audit has an invalid base split schema")
    if base_split_audit.get("policy") != REAL_ANCHORED_BANK_SPLIT_POLICY:
        raise ValueError("combined bank audit has an invalid base split policy")
    source_window_length = base_split_audit.get("source_window_length")
    if not isinstance(source_window_length, int) or source_window_length < 1:
        raise ValueError("combined bank audit has an invalid source window")
    component_assignments = base_split_audit.get("component_assignments")
    if not isinstance(component_assignments, list) or (
        base_split_audit.get("component_assignment_sha256")
        != _canonical_hash(component_assignments)
    ):
        raise ValueError("base split component assignment hash mismatch")
    base_ids_by_role: dict[str, list[str]] = {}
    for role in ("evaluation", "reference"):
        raw_ids = base_split_audit.get(f"{role}_background_ids")
        if (
            not isinstance(raw_ids, list)
            or raw_ids != sorted(str(value) for value in raw_ids)
            or any(not value for value in raw_ids)
            or len(raw_ids) != len(set(raw_ids))
            or base_split_audit.get(f"{role}_background_ids_sha256")
            != _canonical_hash(raw_ids)
        ):
            raise ValueError(f"base split {role} background IDs are invalid")
        assigned_ids = sorted(
            str(background_id)
            for assignment in component_assignments
            if isinstance(assignment, Mapping)
            and assignment.get("role") == role
            for background_id in assignment.get("accepted_background_ids", [])
        )
        if assigned_ids != raw_ids:
            raise ValueError(
                f"base split {role} IDs disagree with component assignments"
            )
        base_ids_by_role[role] = raw_ids
    if base_split_audit.get("cross_bank_temporal_overlap_count") != 0:
        raise ValueError("base reference/evaluation split is not disjoint")

    evaluation_ids = _bank_ids(
        evaluation_backgrounds,
        role="evaluation",
    )
    reference_ids = _bank_ids(
        reference_backgrounds,
        role="reference",
    )
    if set(evaluation_ids).intersection(reference_ids):
        raise ValueError("reference/evaluation background IDs overlap")
    if not set(evaluation_ids).issubset(base_ids_by_role["evaluation"]):
        raise ValueError("final evaluation bank is not from the base split")
    if not set(reference_ids).issubset(base_ids_by_role["reference"]):
        raise ValueError("final reference bank is not from the base split")
    overlap_pairs = _cross_bank_temporal_overlap_pairs(
        evaluation_backgrounds,
        reference_backgrounds,
        source_window_length=source_window_length,
    )
    if overlap_pairs:
        raise ValueError(
            "reference/evaluation real-anchored banks overlap in source time"
        )

    base_payload = dict(base_split_audit)
    return {
        "schema_version": REAL_ANCHORED_COMBINED_BANK_SPLIT_SCHEMA,
        "policy": REAL_ANCHORED_COMBINED_BANK_SPLIT_POLICY,
        "source_window_length": source_window_length,
        "evaluation_background_count": len(evaluation_ids),
        "reference_background_count": len(reference_ids),
        "cross_bank_temporal_overlap_count": 0,
        "evaluation_background_ids_sha256": _canonical_hash(evaluation_ids),
        "reference_background_ids_sha256": _canonical_hash(reference_ids),
        # This is the base audit's component commitment itself, not a hash of
        # its hexadecimal representation.
        "component_assignment_sha256": base_split_audit[
            "component_assignment_sha256"
        ],
        "base_split_audit_sha256": _canonical_hash(base_payload),
        "combined_split": base_payload,
        "threshold_tuning_policy": (
            "qualification_only_reference_bank_never_evaluation_origins"
        ),
    }


def validate_real_anchored_reference_chain(
    evaluation_backgrounds: Sequence[Mapping[str, Any]],
    reference_backgrounds: Sequence[Mapping[str, Any]],
    bank_split_audit: Mapping[str, Any],
    qualification_policy: Mapping[str, Any],
    *,
    reference_contract_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Validate the bundle-bound split, final banks, and frozen policy chain."""

    base_split = bank_split_audit.get("combined_split")
    if not isinstance(base_split, Mapping):
        raise ValueError("combined bank split audit lacks its base audit")
    expected_audit = build_combined_real_anchored_bank_split_audit(
        evaluation_backgrounds,
        reference_backgrounds,
        base_split_audit=base_split,
    )
    if dict(bank_split_audit) != expected_audit:
        raise ValueError("combined bank split audit disagrees with frozen banks")

    reference_bank = qualification_policy.get("reference_bank")
    if not isinstance(reference_bank, Mapping):
        raise ValueError("qualification policy has no reference-bank binding")
    expected_fields = {
        "background_count": expected_audit["reference_background_count"],
        "background_ids_sha256": expected_audit[
            "reference_background_ids_sha256"
        ],
        "evaluation_background_count": expected_audit[
            "evaluation_background_count"
        ],
        "evaluation_background_ids_sha256": expected_audit[
            "evaluation_background_ids_sha256"
        ],
        "source_window_length": expected_audit["source_window_length"],
        "bank_split_schema": expected_audit["schema_version"],
        "bank_split_policy": expected_audit["policy"],
        "bank_split_audit_sha256": _canonical_hash(expected_audit),
        "component_assignment_sha256": expected_audit[
            "component_assignment_sha256"
        ],
        "cross_bank_temporal_overlap_count": 0,
    }
    for field, expected in expected_fields.items():
        if reference_bank.get(field) != expected:
            raise ValueError(
                "qualification policy reference-bank binding mismatch: "
                f"{field}"
            )
    if reference_contract_rows is not None:
        reference_ids = [
            str(row["background_id"]) for row in reference_backgrounds
        ]
        expected_policy = (
            freeze_real_anchored_qualification_policy(
                reference_contract_rows,
                reference_background_ids=reference_ids,
                bank_split_audit=expected_audit,
            )
            if reference_contract_rows
            else unavailable_real_anchored_qualification_policy(
                reference_background_ids=reference_ids,
                bank_split_audit=expected_audit,
            )
        )
        if dict(qualification_policy) != expected_policy:
            raise ValueError(
                "qualification policy disagrees with reference contracts"
            )


def freeze_real_anchored_qualification_policy(
    reference_contract_rows: Sequence[Mapping[str, Any]],
    *,
    reference_background_ids: Sequence[str],
    bank_split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze predeclared thresholds using reference-bank evidence only.

    The function does not optimize thresholds.  It verifies that every
    reference contract for one capability used one identical, predeclared
    threshold payload, then binds that payload to the disjoint reference bank
    and records coverage diagnostics.  Evaluation origins are not accepted as
    inputs and cannot alter the resulting policy.
    """

    reference_ids = tuple(sorted(str(value) for value in reference_background_ids))
    if not reference_ids or len(reference_ids) != len(set(reference_ids)):
        raise ValueError("reference background IDs must be non-empty and unique")
    reference_id_set = set(reference_ids)
    if bank_split_audit.get("cross_bank_temporal_overlap_count") != 0:
        raise ValueError("qualification bank is not disjoint from evaluation")
    expected_reference_hash = _canonical_hash(list(reference_ids))
    if bank_split_audit.get("reference_background_ids_sha256") != (
        expected_reference_hash
    ):
        raise ValueError("qualification bank IDs disagree with split audit")

    by_capability: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_contract_cells: set[tuple[str, str]] = set()
    for row in reference_contract_rows:
        background_id = str(row.get("background_id", ""))
        if background_id not in reference_id_set:
            raise ValueError(
                "qualification contract row is not from the reference bank"
            )
        capability_id = str(row.get("capability_id", ""))
        if not capability_id:
            raise ValueError("qualification contract is missing capability_id")
        cell = (background_id, capability_id)
        if cell in seen_contract_cells:
            raise ValueError(
                "qualification contracts contain a duplicate background/capability"
            )
        seen_contract_cells.add(cell)
        if row.get("qualification_threshold_source") != (
            QUALIFICATION_THRESHOLD_SOURCE_POLICY
        ):
            raise ValueError(
                "qualification contract threshold source is not reference-only"
            )
        by_capability[capability_id].append(row)
    if not by_capability:
        raise ValueError("qualification policy requires reference contracts")

    capability_policies: dict[str, dict[str, Any]] = {}
    for capability_id, rows in sorted(by_capability.items()):
        threshold_payloads = [row.get("qualification_thresholds") for row in rows]
        if not all(isinstance(value, Mapping) for value in threshold_payloads):
            raise ValueError(
                f"{capability_id} reference contracts lack qualification thresholds"
            )
        threshold_hashes = {
            _canonical_hash(dict(value))
            for value in threshold_payloads
            if isinstance(value, Mapping)
        }
        policy_ids = {
            str(row.get("qualification_policy_id", "")) for row in rows
        }
        if len(threshold_hashes) != 1 or len(policy_ids) != 1 or "" in policy_ids:
            raise ValueError(
                f"{capability_id} reference contracts did not use one frozen policy"
            )
        reason_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            if row.get("available") is not True:
                reason_counts[str(row.get("unavailable_reason", "unknown"))] += 1
        thresholds = dict(threshold_payloads[0])
        capability_policies[capability_id] = {
            "qualification_policy_id": next(iter(policy_ids)),
            "qualification_thresholds": thresholds,
            "qualification_thresholds_sha256": next(iter(threshold_hashes)),
            "reference_contract_count": len(rows),
            "reference_available_count": sum(
                row.get("available") is True for row in rows
            ),
            "reference_unavailable_reason_counts": dict(
                sorted(reason_counts.items())
            ),
            "qualification_threshold_source": (
                QUALIFICATION_THRESHOLD_SOURCE_POLICY
            ),
        }

    payload: dict[str, Any] = {
        "schema_version": REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
        "threshold_source_policy": QUALIFICATION_THRESHOLD_SOURCE_POLICY,
        "decisions": protocol_decisions(),
        "reference_bank": {
            "background_count": len(reference_ids),
            "background_ids_sha256": expected_reference_hash,
            "evaluation_background_count": bank_split_audit.get(
                "evaluation_background_count"
            ),
            "evaluation_background_ids_sha256": bank_split_audit.get(
                "evaluation_background_ids_sha256"
            ),
            "source_window_length": bank_split_audit.get(
                "source_window_length"
            ),
            "bank_split_schema": bank_split_audit.get("schema_version"),
            "bank_split_policy": bank_split_audit.get("policy"),
            "bank_split_audit_sha256": _canonical_hash(
                dict(bank_split_audit)
            ),
            "component_assignment_sha256": bank_split_audit.get(
                "component_assignment_sha256"
            ),
            "cross_bank_temporal_overlap_count": 0,
        },
        "capabilities": capability_policies,
        "evaluation_origin_role": "forbidden_for_threshold_tuning",
    }
    payload["qualification_policy_sha256"] = _canonical_hash(payload)
    return payload


def unavailable_real_anchored_qualification_policy(
    *,
    reference_background_ids: Sequence[str],
    bank_split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a fail-closed policy when no independent reference row exists.

    This artifact deliberately contains no capability thresholds.  It allows
    calibration of the legacy synthetic/real-accuracy tracks to continue while
    making every real-anchored cell unavailable; evaluation origins are never
    consulted as a substitute qualification bank.
    """

    if bank_split_audit.get("cross_bank_temporal_overlap_count") != 0:
        raise ValueError("qualification bank is not disjoint from evaluation")
    reference_ids = sorted(str(value) for value in reference_background_ids)
    reference_hash = bank_split_audit.get(
        "reference_background_ids_sha256"
    )
    if reference_hash != _canonical_hash(reference_ids):
        raise ValueError("qualification policy bank IDs disagree with audit")
    payload: dict[str, Any] = {
        "schema_version": REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
        "threshold_source_policy": QUALIFICATION_THRESHOLD_SOURCE_POLICY,
        "decisions": protocol_decisions(),
        "reference_bank": {
            "status": "unavailable",
            "unavailable_reason": "independent_reference_bank_unavailable",
            "background_count": len(reference_ids),
            "background_ids_sha256": reference_hash,
            "evaluation_background_count": bank_split_audit.get(
                "evaluation_background_count"
            ),
            "evaluation_background_ids_sha256": bank_split_audit.get(
                "evaluation_background_ids_sha256"
            ),
            "source_window_length": bank_split_audit.get(
                "source_window_length"
            ),
            "bank_split_schema": bank_split_audit.get("schema_version"),
            "bank_split_policy": bank_split_audit.get("policy"),
            "bank_split_audit_sha256": _canonical_hash(
                dict(bank_split_audit)
            ),
            "component_assignment_sha256": bank_split_audit.get(
                "component_assignment_sha256"
            ),
            "cross_bank_temporal_overlap_count": 0,
        },
        "capabilities": {},
        "evaluation_origin_role": "forbidden_for_threshold_tuning",
        "formal_real_anchored_status": "unavailable",
        "formal_real_anchored_unavailable_reason": (
            "independent_reference_bank_unavailable"
        ),
    }
    payload["qualification_policy_sha256"] = _canonical_hash(payload)
    return payload


def validate_evaluation_qualification_policy(
    evaluation_contract_rows: Sequence[Mapping[str, Any]],
    qualification_policy: Mapping[str, Any],
) -> None:
    """Require evaluation contracts to reuse reference-frozen thresholds."""

    if qualification_policy.get("schema_version") != (
        REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA
    ):
        raise ValueError("qualification policy has an invalid schema")
    observed_policy_hash = qualification_policy.get(
        "qualification_policy_sha256"
    )
    policy_payload = dict(qualification_policy)
    policy_payload.pop("qualification_policy_sha256", None)
    if observed_policy_hash != _canonical_hash(policy_payload):
        raise ValueError("qualification policy self-hash mismatch")
    if qualification_policy.get("decisions") != protocol_decisions():
        raise ValueError("qualification policy decisions changed")
    reference_bank = qualification_policy.get("reference_bank")
    if not isinstance(reference_bank, Mapping):
        raise ValueError("qualification policy has no reference-bank audit")
    if (
        not isinstance(reference_bank.get("background_count"), int)
        or int(reference_bank["background_count"]) < 0
        or not isinstance(reference_bank.get("background_ids_sha256"), str)
        or len(str(reference_bank["background_ids_sha256"])) != 64
        or not isinstance(
            reference_bank.get("evaluation_background_count"), int
        )
        or int(reference_bank["evaluation_background_count"]) < 0
        or not isinstance(
            reference_bank.get("evaluation_background_ids_sha256"), str
        )
        or len(str(reference_bank["evaluation_background_ids_sha256"])) != 64
        or not isinstance(reference_bank.get("source_window_length"), int)
        or int(reference_bank["source_window_length"]) < 1
        or reference_bank.get("cross_bank_temporal_overlap_count") != 0
        or not reference_bank.get("bank_split_schema")
        or not reference_bank.get("bank_split_policy")
        or not isinstance(
            reference_bank.get("bank_split_audit_sha256"), str
        )
        or len(str(reference_bank["bank_split_audit_sha256"])) != 64
        or not reference_bank.get("component_assignment_sha256")
    ):
        raise ValueError("qualification policy reference-bank audit is invalid")
    capabilities = qualification_policy.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ValueError("qualification policy has no capability thresholds")
    if qualification_policy.get("threshold_source_policy") != (
        QUALIFICATION_THRESHOLD_SOURCE_POLICY
    ):
        raise ValueError("qualification policy has an invalid threshold source")
    for capability_id, frozen in capabilities.items():
        if not isinstance(frozen, Mapping):
            raise ValueError(
                f"qualification policy cell is invalid: {capability_id}"
            )
        thresholds = frozen.get("qualification_thresholds")
        if not isinstance(thresholds, Mapping) or frozen.get(
            "qualification_thresholds_sha256"
        ) != _canonical_hash(dict(thresholds)):
            raise ValueError(
                f"qualification policy threshold hash mismatch: {capability_id}"
            )
    for row in evaluation_contract_rows:
        capability_id = str(row.get("capability_id", ""))
        frozen = capabilities.get(capability_id)
        if not isinstance(frozen, Mapping):
            raise ValueError(
                f"evaluation capability has no reference policy: {capability_id}"
            )
        thresholds = row.get("qualification_thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("evaluation contract lacks qualification thresholds")
        if _canonical_hash(dict(thresholds)) != frozen.get(
            "qualification_thresholds_sha256"
        ):
            raise ValueError(
                f"evaluation thresholds differ from reference policy: {capability_id}"
            )
        if str(row.get("qualification_policy_id", "")) != frozen.get(
            "qualification_policy_id"
        ):
            raise ValueError(
                f"evaluation policy ID differs from reference: {capability_id}"
            )
        if row.get("qualification_threshold_source") != (
            QUALIFICATION_THRESHOLD_SOURCE_POLICY
        ):
            raise ValueError(
                f"evaluation threshold source is not reference-frozen: "
                f"{capability_id}"
            )
        row_policy_hash = row.get("qualification_policy_sha256")
        if row_policy_hash is None:
            row_policy_hash = row.get(
                "frozen_qualification_policy_sha256"
            )
        if row_policy_hash != observed_policy_hash:
            raise ValueError(
                f"evaluation row is not bound to qualification policy: "
                f"{capability_id}"
            )
