"""Reference-frozen dose design for real-anchored counterfactuals.

``alpha`` remains the physical multiplier of the controlled component.  It is
not a cross-capability intensity coordinate: the same multiplier can produce
very different visible perturbations for a step, a sparse event, or a weak
seasonal sideband.  V5 therefore freezes the source-distance targets and the
solver from source-time-disjoint reference contracts, then resolves one alpha
grid per history-only contract.  The canonical strength grid remains the
cross-background aggregation coordinate.

The real-anchored proximity check is treatment-only: it enforces a minimum
fixed-L168 distance from the treatment to its own authentic source and a
maximum local-augmentation budget.  Adjacent-dose distances remain diagnostic
only.  The exact alpha-one baseline is intentionally exempt.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cafe.generation.real_anchored_policy import (
    REAL_ANCHORED_ADDITIVE_DOSE_CAPABILITIES,
    REAL_ANCHORED_ADDITIVE_FUTURE_TARGET_MAXIMUM,
    REAL_ANCHORED_ADDITIVE_HISTORY_TARGET_MAXIMUM,
    REAL_ANCHORED_ADDITIVE_MAXIMUM_ALPHA,
    REAL_ANCHORED_ADDITIVE_MAXIMUM_GAIN,
    REAL_ANCHORED_CANONICAL_STRENGTH_GRID,
    REAL_ANCHORED_NONLINEAR_DOSE_CAPABILITIES,
    REAL_ANCHORED_NONLINEAR_FUTURE_TARGET_MAXIMUM,
    REAL_ANCHORED_NONLINEAR_HISTORY_TARGET_MAXIMUM,
    REAL_ANCHORED_NONLINEAR_MAXIMUM_ALPHA,
    REAL_ANCHORED_NONLINEAR_REFERENCE_ALPHA_STEP,
    REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION,
    REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT,
    REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION,
    REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION,
    REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION,
    REAL_ANCHORED_REFERENCE_DOSE_QUANTILE,
    REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM,
)


REAL_ANCHORED_DOSE_REFERENCE_SCHEMA = (
    "cafe.real_anchored.dose_reference.v1"
)
REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA = (
    "cafe.real_anchored.dose_calibration.v2"
)
REAL_ANCHORED_CONTRACT_DOSE_CALIBRATION_SCHEMA = (
    "cafe.real_anchored.contract_dose_calibration.v1"
)
REAL_ANCHORED_DOSE_POLICY_SCHEMA = "cafe.real_anchored.dose_policy.v2"
REAL_ANCHORED_PAIRED_SEPARATION_SCHEMA = (
    "cafe.real_anchored.treatment_source_distance.v1"
)
REAL_ANCHORED_ADDITIVE_RESPONSE_LAW = (
    "additive_linear_in_alpha_minus_one"
)
REAL_ANCHORED_NONLINEAR_RESPONSE_LAW = (
    "dynamic_recursive_nonproportional"
)
REAL_ANCHORED_PHYSICAL_DOSE_PARAMETER = (
    "controlled_component_multiplier_alpha"
)
REAL_ANCHORED_REFERENCE_QUANTILE_METHOD = (
    "nearest_rank_order_statistic"
)
REAL_ANCHORED_PANEL_CHANNEL_TARGET_FRACTION = 0.5
REAL_ANCHORED_DOSE_EVIDENCE_ROLES = frozenset(
    {"formal", "sensitivity", "qualification_only"}
)
REAL_ANCHORED_FIXED_HISTORY_LENGTH = 168
REAL_ANCHORED_HORIZON = 48
_NUMERICAL_TOLERANCE = 1e-12


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_nonnegative(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _finite_positive(value: Any, *, name: str) -> float:
    number = _finite_nonnegative(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nearest_rank(values: Sequence[float], probability: float) -> tuple[float, int]:
    if not values:
        raise ValueError("nearest-rank quantile requires at least one value")
    if not 0.0 < float(probability) <= 1.0:
        raise ValueError("nearest-rank probability must lie in (0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("nearest-rank values must be finite")
    rank = max(1, int(math.ceil(float(probability) * len(ordered))))
    return ordered[rank - 1], rank


def _target_maxima(capability_id: str) -> tuple[float, float, float, str]:
    if capability_id in REAL_ANCHORED_NONLINEAR_DOSE_CAPABILITIES:
        return (
            REAL_ANCHORED_NONLINEAR_HISTORY_TARGET_MAXIMUM,
            REAL_ANCHORED_NONLINEAR_FUTURE_TARGET_MAXIMUM,
            REAL_ANCHORED_NONLINEAR_MAXIMUM_ALPHA,
            REAL_ANCHORED_NONLINEAR_RESPONSE_LAW,
        )
    if capability_id in REAL_ANCHORED_ADDITIVE_DOSE_CAPABILITIES:
        return (
            REAL_ANCHORED_ADDITIVE_HISTORY_TARGET_MAXIMUM,
            REAL_ANCHORED_ADDITIVE_FUTURE_TARGET_MAXIMUM,
            REAL_ANCHORED_ADDITIVE_MAXIMUM_ALPHA,
            REAL_ANCHORED_ADDITIVE_RESPONSE_LAW,
        )
    raise ValueError(f"unsupported real-anchored dose capability: {capability_id}")


def dose_targets(capability_id: str) -> dict[str, Any]:
    """Return the predeclared canonical strength and separation targets."""

    history_maximum, future_maximum, max_alpha, response_law = (
        _target_maxima(str(capability_id))
    )
    strength_grid = [
        float(value) for value in REAL_ANCHORED_CANONICAL_STRENGTH_GRID
    ]
    history_target_grid = np.linspace(
        REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM,
        history_maximum,
        len(strength_grid),
    ).astype(float).tolist()
    return {
        "capability_id": str(capability_id),
        "response_law": response_law,
        "strength_grid": strength_grid,
        "history_target_grid": history_target_grid,
        "future_target_grid": [
            float(future_maximum * value) for value in strength_grid
        ],
        "physical_parameter": REAL_ANCHORED_PHYSICAL_DOSE_PARAMETER,
        "source_distance_minimum": float(
            REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
        ),
        "source_distance_window": "fixed_l168_model_input_history",
        "source_distance_member_scope": "treatment_only_baseline_exempt",
        "max_alpha": float(max_alpha),
        "maximum_history_macro_separation": float(
            REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION
        ),
        "maximum_future_macro_separation": float(
            REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION
        ),
        "maximum_affected_channel_separation": float(
            REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION
        ),
        "history_window_length": REAL_ANCHORED_FIXED_HISTORY_LENGTH,
        "horizon": REAL_ANCHORED_HORIZON,
    }


def additive_dose_reference(
    *,
    capability_id: str,
    background_id: str,
    unit_gain_history_separation: float,
    unit_gain_future_separation: float,
    affected_channel_indices: Sequence[int],
    known_future_covariate_path_used: bool = False,
    evidence_role: str = "formal",
) -> dict[str, Any]:
    """Build JSON-safe additive reference evidence for one contract."""

    if capability_id not in REAL_ANCHORED_ADDITIVE_DOSE_CAPABILITIES:
        raise ValueError("additive dose evidence has a non-additive capability")
    affected = [int(value) for value in affected_channel_indices]
    if not affected or len(affected) != len(set(affected)) or min(affected) < 0:
        raise ValueError("additive dose evidence requires unique affected channels")
    role = str(evidence_role)
    if role not in REAL_ANCHORED_DOSE_EVIDENCE_ROLES:
        raise ValueError("additive dose evidence has an invalid role")
    return {
        "schema_version": REAL_ANCHORED_DOSE_REFERENCE_SCHEMA,
        "capability_id": str(capability_id),
        "background_id": str(background_id),
        "response_law": REAL_ANCHORED_ADDITIVE_RESPONSE_LAW,
        "history_window_length": REAL_ANCHORED_FIXED_HISTORY_LENGTH,
        "horizon": REAL_ANCHORED_HORIZON,
        "affected_channel_indices": affected,
        "evidence_role": role,
        "unit_gain_history_separation": _finite_positive(
            unit_gain_history_separation,
            name="unit_gain_history_separation",
        ),
        "unit_gain_future_separation": _finite_positive(
            unit_gain_future_separation,
            name="unit_gain_future_separation",
        ),
        "target_future_used": False,
        "known_future_covariate_path_used": bool(
            known_future_covariate_path_used
        ),
    }


def nonlinear_dose_reference(
    *,
    background_id: str,
    zero_innovation_curve: Sequence[Mapping[str, Any]],
    monotone: bool,
    evidence_role: str = "formal",
) -> dict[str, Any]:
    """Build JSON-safe nonlinear reference evidence for one contract."""

    curve = [dict(row) for row in zero_innovation_curve]
    role = str(evidence_role)
    if role not in REAL_ANCHORED_DOSE_EVIDENCE_ROLES:
        raise ValueError("nonlinear dose evidence has an invalid role")
    return {
        "schema_version": REAL_ANCHORED_DOSE_REFERENCE_SCHEMA,
        "capability_id": "nonlinear_persistence",
        "background_id": str(background_id),
        "response_law": REAL_ANCHORED_NONLINEAR_RESPONSE_LAW,
        "history_window_length": REAL_ANCHORED_FIXED_HISTORY_LENGTH,
        "horizon": REAL_ANCHORED_HORIZON,
        "affected_channel_indices": [0],
        "evidence_role": role,
        "zero_innovation_curve": curve,
        "monotone": bool(monotone),
        "target_future_used": False,
        "known_future_covariate_path_used": False,
    }


def _row_evidence(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    evidence = row.get("dose_design_reference")
    if isinstance(evidence, Mapping):
        return evidence
    contract = row.get("contract")
    if isinstance(contract, Mapping):
        evidence = contract.get("dose_design_reference")
        if isinstance(evidence, Mapping):
            return evidence

    # v3 univariate rows already expose the exact standardized fixed-L168 and
    # H48 unit-component RMS values.  Accept those fields as a migration path;
    # structural and nonlinear rows must publish explicit evidence because
    # their affected-channel and response-law semantics cannot be inferred.
    capability_id = str(row.get("capability_id", ""))
    if capability_id in REAL_ANCHORED_ADDITIVE_DOSE_CAPABILITIES:
        history = row.get("controlled_component_visible_history_rms")
        future = row.get("controlled_component_future_rms")
        if history is not None and future is not None:
            return additive_dose_reference(
                capability_id=capability_id,
                background_id=str(row.get("background_id", "")),
                unit_gain_history_separation=float(history),
                unit_gain_future_separation=float(future),
                affected_channel_indices=(0,),
            )
    return None


def _row_is_dose_eligible(row: Mapping[str, Any]) -> bool:
    """Use only mechanism-qualified reference contracts for dose fitting."""

    if row.get("available") is True:
        return True
    if row.get("sensitivity_available") is True:
        return True
    if row.get("qualification_available") is True:
        return True
    contract = row.get("contract")
    if isinstance(contract, Mapping):
        structural_flags = (
            "formal_main_eligible",
            "sensitivity_eligible",
            "qualification_only",
        )
        if any(field in contract for field in structural_flags):
            if contract.get("formal_main_eligible") is True:
                return True
            if contract.get("sensitivity_eligible") is True:
                return True
            diagnostics = contract.get("fit_diagnostics")
            return bool(
                contract.get("qualification_only") is True
                and isinstance(diagnostics, Mapping)
                and diagnostics.get("qualification_passed") is True
            )
    if any(
        field in row
        for field in (
            "available",
            "sensitivity_available",
            "qualification_available",
        )
    ):
        return False
    # Small direct unit fixtures may provide evidence without the surrounding
    # fitter status envelope.  Production rows always take one branch above.
    return True


def _validate_common_evidence(
    evidence: Mapping[str, Any],
    *,
    capability_id: str,
    background_id: str,
) -> None:
    if evidence.get("schema_version") != REAL_ANCHORED_DOSE_REFERENCE_SCHEMA:
        raise ValueError("dose reference evidence has an invalid schema")
    if str(evidence.get("capability_id", "")) != capability_id:
        raise ValueError("dose reference capability mismatch")
    if str(evidence.get("background_id", "")) != background_id:
        raise ValueError("dose reference background mismatch")
    if evidence.get("target_future_used") is not False:
        raise ValueError("dose reference evidence used held-out target future")
    if int(evidence.get("history_window_length", 0)) != (
        REAL_ANCHORED_FIXED_HISTORY_LENGTH
    ):
        raise ValueError("dose reference evidence changed the history window")
    if int(evidence.get("horizon", 0)) != REAL_ANCHORED_HORIZON:
        raise ValueError("dose reference evidence changed the horizon")
    affected = evidence.get("affected_channel_indices")
    if (
        not isinstance(affected, list)
        or not affected
        or any(not isinstance(value, int) or value < 0 for value in affected)
        or len(affected) != len(set(affected))
    ):
        raise ValueError("dose reference evidence has invalid affected channels")
    if evidence.get("evidence_role") not in REAL_ANCHORED_DOSE_EVIDENCE_ROLES:
        raise ValueError("dose reference evidence has an invalid role")


def _empty_calibration(
    capability_id: str,
    *,
    reason: str,
    evidence: Sequence[Mapping[str, Any]],
    excluded_nonmonotone_count: int = 0,
) -> dict[str, Any]:
    targets = dose_targets(capability_id)
    payload: dict[str, Any] = {
        "schema_version": REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA,
        "mapping_scope": "contract_specific_history_only",
        "status": "unavailable",
        "capability_id": capability_id,
        "response_law": targets["response_law"],
        "strength_grid": targets["strength_grid"],
        "history_target_grid": targets["history_target_grid"],
        "future_target_grid": targets["future_target_grid"],
        "applied_alpha_grid": [],
        "physical_parameter": targets["physical_parameter"],
        "max_alpha": targets["max_alpha"],
        "maximum_history_macro_separation": targets[
            "maximum_history_macro_separation"
        ],
        "maximum_future_macro_separation": targets[
            "maximum_future_macro_separation"
        ],
        "maximum_affected_channel_separation": targets[
            "maximum_affected_channel_separation"
        ],
        "reference_quantile": {
            "probability": REAL_ANCHORED_REFERENCE_DOSE_QUANTILE,
            "method": REAL_ANCHORED_REFERENCE_QUANTILE_METHOD,
        },
        "reference_evidence_count": len(evidence),
        "minimum_reference_evidence_count": (
            REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT
        ),
        "reference_evidence_sha256": _canonical_hash(list(evidence)),
        "reference_excluded_nonmonotone_count": int(
            excluded_nonmonotone_count
        ),
        "minimum_acceptance_fraction": (
            REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        ),
        "source_distance_minimum": float(
            REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
        ),
        "unavailable_reason": str(reason),
        "target_future_used_for_mapping": False,
    }
    payload["policy_sha256"] = _canonical_hash(payload)
    return payload


def _history_grid(maximum: float) -> list[float]:
    value = float(maximum)
    if value + _NUMERICAL_TOLERANCE < REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM:
        raise ValueError("history target maximum is below the source-distance floor")
    return np.linspace(
        REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM,
        value,
        len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID),
    ).astype(float).tolist()


def _candidate_history_maxima(capability_id: str) -> list[float]:
    targets = dose_targets(capability_id)
    maximum = float(targets["history_target_grid"][-1])
    minimum = float(REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM)
    count = max(1, int(round((maximum - minimum) / 0.025)) + 1)
    return sorted(
        set(
            float(value)
            for value in np.linspace(minimum, maximum, count)
            if float(value) > minimum + _NUMERICAL_TOLERANCE
        ),
        reverse=True,
    )


def _additive_alpha_grid(
    evidence: Mapping[str, Any],
    *,
    history_targets: Sequence[float],
    future_targets: Sequence[float],
) -> list[float] | None:
    history = _finite_positive(
        evidence.get("unit_gain_history_separation"),
        name="unit_gain_history_separation",
    )
    future = _finite_positive(
        evidence.get("unit_gain_future_separation"),
        name="unit_gain_future_separation",
    )
    gains = [
        max(float(history_target) / history, float(future_target) / future)
        for history_target, future_target in zip(
            history_targets,
            future_targets,
            strict=True,
        )
    ]
    if (
        gains[-1] > REAL_ANCHORED_ADDITIVE_MAXIMUM_GAIN + _NUMERICAL_TOLERANCE
        or gains[-1] * history
        > REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION
        + _NUMERICAL_TOLERANCE
        or gains[-1] * future
        > REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION
        + _NUMERICAL_TOLERANCE
    ):
        return None
    alphas = [float(1.0 + gain) for gain in gains]
    if any(
        right <= left + _NUMERICAL_TOLERANCE
        for left, right in zip(alphas, alphas[1:])
    ):
        return None
    return alphas


def _freeze_additive_calibration(
    capability_id: str,
    evidence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(evidence_rows) < REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT:
        return _empty_calibration(
            capability_id,
            reason="insufficient_reference_dose_evidence",
            evidence=evidence_rows,
        )
    for evidence in evidence_rows:
        if evidence.get("response_law") != REAL_ANCHORED_ADDITIVE_RESPONSE_LAW:
            raise ValueError("additive dose evidence changed the response law")
        _finite_positive(
            evidence.get("unit_gain_history_separation"),
            name="unit_gain_history_separation",
        )
        _finite_positive(
            evidence.get("unit_gain_future_separation"),
            name="unit_gain_future_separation",
        )
    targets = dose_targets(capability_id)
    minimum_balanced_count = int(
        math.ceil(
            REAL_ANCHORED_REFERENCE_DOSE_QUANTILE * len(evidence_rows)
        )
    )
    selected_history_grid: list[float] | None = None
    balanced_count = 0
    for maximum in _candidate_history_maxima(capability_id):
        candidate = _history_grid(maximum)
        count = sum(
            _additive_alpha_grid(
                evidence,
                history_targets=candidate,
                future_targets=targets["future_target_grid"],
            )
            is not None
            for evidence in evidence_rows
        )
        if count >= minimum_balanced_count:
            selected_history_grid = candidate
            balanced_count = count
            break
    if selected_history_grid is None:
        unavailable = _empty_calibration(
            capability_id,
            reason="reference_contract_specific_dose_coverage_insufficient",
            evidence=evidence_rows,
        )
        unavailable.update(
            {
                "reference_balanced_effect_count": int(balanced_count),
                "minimum_reference_balanced_effect_count": int(
                    minimum_balanced_count
                ),
            }
        )
        unavailable["policy_sha256"] = _canonical_hash(
            {
                key: value
                for key, value in unavailable.items()
                if key != "policy_sha256"
            }
        )
        return unavailable
    payload: dict[str, Any] = {
        "schema_version": REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA,
        "mapping_scope": "contract_specific_history_only",
        "status": "available",
        "capability_id": capability_id,
        "response_law": REAL_ANCHORED_ADDITIVE_RESPONSE_LAW,
        "strength_grid": targets["strength_grid"],
        "history_target_grid": selected_history_grid,
        "future_target_grid": targets["future_target_grid"],
        "applied_alpha_grid": [],
        "physical_parameter": targets["physical_parameter"],
        "max_alpha": targets["max_alpha"],
        "maximum_history_macro_separation": targets[
            "maximum_history_macro_separation"
        ],
        "maximum_future_macro_separation": targets[
            "maximum_future_macro_separation"
        ],
        "maximum_affected_channel_separation": targets[
            "maximum_affected_channel_separation"
        ],
        "reference_quantile": {
            "probability": REAL_ANCHORED_REFERENCE_DOSE_QUANTILE,
            "method": "minimum_feasible_contract_coverage",
        },
        "reference_evidence_count": len(evidence_rows),
        "reference_evidence_sha256": _canonical_hash(list(evidence_rows)),
        "reference_balanced_effect_count": int(balanced_count),
        "minimum_reference_balanced_effect_count": int(
            minimum_balanced_count
        ),
        "maximum_allowed_gain": REAL_ANCHORED_ADDITIVE_MAXIMUM_GAIN,
        "source_distance_minimum": float(
            REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
        ),
        "minimum_acceptance_fraction": (
            REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        ),
        "minimum_reference_evidence_count": (
            REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT
        ),
        "target_future_used_for_mapping": False,
        "evaluation_history_used_only_by_frozen_solver": True,
    }
    payload["policy_sha256"] = _canonical_hash(payload)
    return payload


def _expected_nonlinear_alpha_grid() -> list[float]:
    count = int(
        round(
            (REAL_ANCHORED_NONLINEAR_MAXIMUM_ALPHA - 1.0)
            / REAL_ANCHORED_NONLINEAR_REFERENCE_ALPHA_STEP
        )
    )
    return [
        float(1.0 + index * REAL_ANCHORED_NONLINEAR_REFERENCE_ALPHA_STEP)
        for index in range(count + 1)
    ]


def _validated_nonlinear_curve(
    evidence: Mapping[str, Any],
) -> list[tuple[float, float, float]] | None:
    if evidence.get("response_law") != REAL_ANCHORED_NONLINEAR_RESPONSE_LAW:
        raise ValueError("nonlinear dose evidence changed the response law")
    if (
        evidence.get("monotone") is not True
        and "monotone_safe_prefix_alpha_max" not in evidence
    ):
        return None
    raw_curve = evidence.get("zero_innovation_curve")
    if not isinstance(raw_curve, list):
        raise ValueError("nonlinear dose evidence has no zero-innovation curve")
    expected_alphas = _expected_nonlinear_alpha_grid()
    if len(raw_curve) != len(expected_alphas):
        raise ValueError("nonlinear dose curve changed the frozen alpha grid")
    curve: list[tuple[float, float, float]] = []
    for index, (row, expected_alpha) in enumerate(
        zip(raw_curve, expected_alphas, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise ValueError("nonlinear dose curve contains an invalid row")
        alpha = float(row.get("alpha"))
        history = _finite_nonnegative(
            row.get("history_separation"),
            name="history_separation",
        )
        future = _finite_nonnegative(
            row.get("future_separation"),
            name="future_separation",
        )
        if abs(alpha - expected_alpha) > _NUMERICAL_TOLERANCE:
            raise ValueError("nonlinear dose curve alpha grid mismatch")
        if row.get("safe") is not True:
            break
        if index == 0 and (
            history > _NUMERICAL_TOLERANCE or future > _NUMERICAL_TOLERANCE
        ):
            raise ValueError("nonlinear alpha-one separation is not zero")
        if curve and (
            history + _NUMERICAL_TOLERANCE < curve[-1][1]
            or future + _NUMERICAL_TOLERANCE < curve[-1][2]
        ):
            break
        curve.append((alpha, history, future))
    return curve if len(curve) >= 2 else None


def _freeze_nonlinear_calibration(
    capability_id: str,
    evidence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(evidence_rows) < REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT:
        return _empty_calibration(
            capability_id,
            reason="insufficient_reference_dose_evidence",
            evidence=evidence_rows,
        )
    valid_curves: list[list[tuple[float, float, float]]] = []
    excluded_nonmonotone = 0
    for evidence in evidence_rows:
        curve = _validated_nonlinear_curve(evidence)
        if curve is None:
            excluded_nonmonotone += 1
        else:
            valid_curves.append(curve)
    if len(valid_curves) < REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT:
        return _empty_calibration(
            capability_id,
            reason="insufficient_monotone_reference_dose_evidence",
            evidence=evidence_rows,
            excluded_nonmonotone_count=excluded_nonmonotone,
        )

    targets = dose_targets(capability_id)
    minimum_balanced_count = int(
        math.ceil(REAL_ANCHORED_REFERENCE_DOSE_QUANTILE * len(valid_curves))
    )
    selected_history_grid: list[float] | None = None
    balanced_count = 0
    for maximum in _candidate_history_maxima(capability_id):
        candidate = _history_grid(maximum)
        count = sum(
            _nonlinear_alpha_grid(
                curve,
                history_targets=candidate,
                future_targets=targets["future_target_grid"],
            )
            is not None
            for curve in valid_curves
        )
        if count >= minimum_balanced_count:
            selected_history_grid = candidate
            balanced_count = count
            break
    if selected_history_grid is None:
        unavailable = _empty_calibration(
            capability_id,
            reason="nonlinear_reference_contract_specific_coverage_insufficient",
            evidence=evidence_rows,
            excluded_nonmonotone_count=excluded_nonmonotone,
        )
        unavailable.update(
            {
                "reference_valid_curve_count": len(valid_curves),
                "reference_balanced_effect_count": int(balanced_count),
                "minimum_reference_balanced_effect_count": int(
                    minimum_balanced_count
                ),
            }
        )
        unavailable["policy_sha256"] = _canonical_hash(
            {
                key: value
                for key, value in unavailable.items()
                if key != "policy_sha256"
            }
        )
        return unavailable

    payload: dict[str, Any] = {
        "schema_version": REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA,
        "mapping_scope": "contract_specific_history_only",
        "status": "available",
        "capability_id": capability_id,
        "response_law": REAL_ANCHORED_NONLINEAR_RESPONSE_LAW,
        "strength_grid": targets["strength_grid"],
        "history_target_grid": selected_history_grid,
        "future_target_grid": targets["future_target_grid"],
        "applied_alpha_grid": [],
        "physical_parameter": targets["physical_parameter"],
        "max_alpha": targets["max_alpha"],
        "maximum_history_macro_separation": targets[
            "maximum_history_macro_separation"
        ],
        "maximum_future_macro_separation": targets[
            "maximum_future_macro_separation"
        ],
        "maximum_affected_channel_separation": targets[
            "maximum_affected_channel_separation"
        ],
        "reference_quantile": {
            "probability": REAL_ANCHORED_REFERENCE_DOSE_QUANTILE,
            "method": "minimum_feasible_contract_coverage",
        },
        "reference_evidence_count": len(evidence_rows),
        "reference_valid_curve_count": len(valid_curves),
        "reference_excluded_nonmonotone_count": excluded_nonmonotone,
        "reference_balanced_effect_count": int(balanced_count),
        "minimum_reference_balanced_effect_count": int(
            minimum_balanced_count
        ),
        "reference_evidence_sha256": _canonical_hash(list(evidence_rows)),
        "candidate_alpha_grid_sha256": _canonical_hash(
            _expected_nonlinear_alpha_grid()
        ),
        "candidate_alpha_step": (
            REAL_ANCHORED_NONLINEAR_REFERENCE_ALPHA_STEP
        ),
        "source_distance_minimum": float(
            REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
        ),
        "minimum_acceptance_fraction": (
            REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        ),
        "minimum_reference_evidence_count": (
            REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT
        ),
        "target_future_used_for_mapping": False,
        "evaluation_history_used_only_by_frozen_solver": True,
    }
    payload["policy_sha256"] = _canonical_hash(payload)
    return payload


def _nonlinear_alpha_grid(
    curve: Sequence[tuple[float, float, float]],
    *,
    history_targets: Sequence[float],
    future_targets: Sequence[float],
) -> list[float] | None:
    """Resolve the smallest safe, strictly increasing alpha at every level."""

    selected: list[float] = []
    previous = 1.0
    for history_target, future_target in zip(
        history_targets,
        future_targets,
        strict=True,
    ):
        match = next(
            (
                (alpha, history, future)
                for alpha, history, future in curve
                if alpha > previous + _NUMERICAL_TOLERANCE
                and history + _NUMERICAL_TOLERANCE >= float(history_target)
                and future + _NUMERICAL_TOLERANCE >= float(future_target)
                and history
                <= REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION
                + _NUMERICAL_TOLERANCE
                and future
                <= REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION
                + _NUMERICAL_TOLERANCE
            ),
            None,
        )
        if match is None:
            return None
        previous = float(match[0])
        selected.append(previous)
    return selected


def freeze_capability_dose_calibration(
    capability_id: str,
    reference_contract_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze one capability's alpha grid from reference rows only."""

    capability_id = str(capability_id)
    evidence_rows: list[dict[str, Any]] = []
    for row in sorted(
        reference_contract_rows,
        key=lambda value: str(value.get("background_id", "")),
    ):
        if str(row.get("capability_id", "")) != capability_id:
            raise ValueError("dose calibration rows contain another capability")
        if not _row_is_dose_eligible(row):
            continue
        evidence = _row_evidence(row)
        if evidence is None:
            continue
        background_id = str(row.get("background_id", ""))
        _validate_common_evidence(
            evidence,
            capability_id=capability_id,
            background_id=background_id,
        )
        evidence_rows.append(dict(evidence))

    if capability_id in REAL_ANCHORED_NONLINEAR_DOSE_CAPABILITIES:
        return _freeze_nonlinear_calibration(capability_id, evidence_rows)
    if capability_id in REAL_ANCHORED_ADDITIVE_DOSE_CAPABILITIES:
        return _freeze_additive_calibration(capability_id, evidence_rows)
    raise ValueError(f"unsupported real-anchored dose capability: {capability_id}")


def validate_dose_calibration(
    calibration: Mapping[str, Any],
    *,
    capability_id: str | None = None,
) -> None:
    """Validate a frozen dose calibration and its self-hash."""

    schema = calibration.get("schema_version")
    if schema not in {
        REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA,
        REAL_ANCHORED_CONTRACT_DOSE_CALIBRATION_SCHEMA,
    }:
        raise ValueError("dose calibration has an invalid schema")
    observed_hash = calibration.get("policy_sha256")
    payload = dict(calibration)
    payload.pop("policy_sha256", None)
    if observed_hash != _canonical_hash(payload):
        raise ValueError("dose calibration policy hash mismatch")
    observed_capability = str(calibration.get("capability_id", ""))
    if capability_id is not None and observed_capability != str(capability_id):
        raise ValueError("dose calibration capability mismatch")
    targets = dose_targets(observed_capability)
    for field in (
        "response_law",
        "strength_grid",
        "future_target_grid",
        "physical_parameter",
        "max_alpha",
        "maximum_history_macro_separation",
        "maximum_future_macro_separation",
        "maximum_affected_channel_separation",
    ):
        if calibration.get(field) != targets[field]:
            raise ValueError(f"dose calibration changed frozen field: {field}")
    history_targets = calibration.get("history_target_grid")
    if (
        not isinstance(history_targets, list)
        or len(history_targets) != len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
        or any(not math.isfinite(float(value)) for value in history_targets)
        or abs(float(history_targets[0]) - REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM)
        > _NUMERICAL_TOLERANCE
        or any(
            float(right) <= float(left)
            for left, right in zip(history_targets, history_targets[1:])
        )
        or float(history_targets[-1])
        > float(targets["history_target_grid"][-1]) + _NUMERICAL_TOLERANCE
    ):
        raise ValueError("dose calibration has an invalid source-distance grid")
    if calibration.get("mapping_scope") != "contract_specific_history_only":
        raise ValueError("dose calibration changed the mapping scope")
    if calibration.get("source_distance_minimum") != (
        REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
    ):
        raise ValueError("dose calibration changed the source-distance floor")
    if calibration.get("minimum_acceptance_fraction") != (
        REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
    ):
        raise ValueError("dose calibration changed the acceptance fraction")
    if calibration.get("minimum_reference_evidence_count") != (
        REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT
    ):
        raise ValueError("dose calibration changed the minimum evidence count")
    status = calibration.get("status")
    alphas = calibration.get("applied_alpha_grid")
    if status == "unavailable":
        if alphas != [] or not calibration.get("unavailable_reason"):
            raise ValueError("unavailable dose calibration is malformed")
        return
    if status != "available" or not isinstance(alphas, list):
        raise ValueError("dose calibration has an invalid status")
    if schema == REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA:
        if alphas != []:
            raise ValueError("policy-level dose calibration must not fix alpha")
        return
    if len(alphas) != len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID):
        raise ValueError("dose calibration has the wrong alpha-grid length")
    alpha_values = [float(value) for value in alphas]
    if (
        not all(math.isfinite(value) for value in alpha_values)
        or any(value <= 1.0 for value in alpha_values)
        or any(
            right <= left
            for left, right in zip(alpha_values, alpha_values[1:])
        )
        or alpha_values[-1] > float(targets["max_alpha"]) + _NUMERICAL_TOLERANCE
    ):
        raise ValueError("dose calibration alpha grid is invalid")


def resolve_contract_dose_calibration(
    policy_calibration: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen history-only solver to one evaluation contract."""

    validate_dose_calibration(policy_calibration)
    if policy_calibration.get("status") != "available":
        raise ValueError("cannot resolve an unavailable dose policy")
    capability_id = str(policy_calibration["capability_id"])
    _validate_common_evidence(
        evidence,
        capability_id=capability_id,
        background_id=str(evidence.get("background_id", "")),
    )
    if capability_id in REAL_ANCHORED_ADDITIVE_DOSE_CAPABILITIES:
        alphas = _additive_alpha_grid(
            evidence,
            history_targets=policy_calibration["history_target_grid"],
            future_targets=policy_calibration["future_target_grid"],
        )
    elif capability_id in REAL_ANCHORED_NONLINEAR_DOSE_CAPABILITIES:
        curve = _validated_nonlinear_curve(evidence)
        alphas = None if curve is None else _nonlinear_alpha_grid(
            curve,
            history_targets=policy_calibration["history_target_grid"],
            future_targets=policy_calibration["future_target_grid"],
        )
    else:
        raise ValueError("unsupported contract-specific dose capability")
    if alphas is None:
        raise ValueError("contract cannot satisfy the frozen source-distance policy")
    payload = {
        key: value
        for key, value in dict(policy_calibration).items()
        if key not in {"schema_version", "policy_sha256", "applied_alpha_grid"}
    }
    payload.update(
        {
            "schema_version": REAL_ANCHORED_CONTRACT_DOSE_CALIBRATION_SCHEMA,
            "applied_alpha_grid": [float(value) for value in alphas],
            "dose_policy_sha256": str(policy_calibration["policy_sha256"]),
            "dose_reference_sha256": _canonical_hash(dict(evidence)),
            "background_id": str(evidence.get("background_id", "")),
            "target_future_used_for_contract_mapping": False,
        }
    )
    payload["policy_sha256"] = _canonical_hash(payload)
    validate_dose_calibration(payload, capability_id=capability_id)
    return payload


def dose_calibration_from_policy(
    qualification_policy: Mapping[str, Any],
    capability_id: str,
    *,
    require_available: bool = True,
) -> dict[str, Any]:
    """Read and validate one frozen capability dose calibration."""

    capabilities = qualification_policy.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ValueError("qualification policy has no capability cells")
    cell = capabilities.get(str(capability_id))
    if not isinstance(cell, Mapping):
        raise ValueError("qualification policy has no capability dose cell")
    calibration = cell.get("dose_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("qualification policy capability has no dose calibration")
    validate_dose_calibration(calibration, capability_id=str(capability_id))
    if require_available and calibration.get("status") != "available":
        raise ValueError("capability dose calibration is unavailable")
    return dict(calibration)


def build_dose_policy_summary(
    capability_policies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind all per-capability calibrations without duplicating their payloads."""

    hashes: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for capability_id, cell in sorted(capability_policies.items()):
        calibration = cell.get("dose_calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError("capability policy has no dose calibration")
        validate_dose_calibration(calibration, capability_id=capability_id)
        hashes[capability_id] = str(calibration["policy_sha256"])
        statuses[capability_id] = str(calibration["status"])
    payload: dict[str, Any] = {
        "schema_version": REAL_ANCHORED_DOSE_POLICY_SCHEMA,
        "strength_grid": list(REAL_ANCHORED_CANONICAL_STRENGTH_GRID),
        "mapping_policy": (
            "reference_frozen_contract_specific_history_only_solver_v2"
        ),
        "reference_quantile": {
            "probability": REAL_ANCHORED_REFERENCE_DOSE_QUANTILE,
            "method": REAL_ANCHORED_REFERENCE_QUANTILE_METHOD,
        },
        "minimum_acceptance_fraction": (
            REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        ),
        "minimum_reference_evidence_count": (
            REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT
        ),
        "panel_channel_target_fraction": (
            REAL_ANCHORED_PANEL_CHANNEL_TARGET_FRACTION
        ),
        "maximum_history_macro_separation": (
            REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION
        ),
        "maximum_future_macro_separation": (
            REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION
        ),
        "maximum_affected_channel_separation": (
            REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION
        ),
        "capability_policy_sha256": hashes,
        "capability_status": statuses,
        "evaluation_origins_used_for_mapping": False,
        "synthetic_anti_copy_reused_for_real_anchors": False,
        "treatment_source_distance_minimum": float(
            REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
        ),
        "baseline_member_exempt": True,
        "adjacent_distance_role": "diagnostic_only",
    }
    payload["dose_policy_sha256"] = _canonical_hash(payload)
    return payload


def validate_dose_policy_summary(
    summary: Mapping[str, Any],
    capability_policies: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = build_dose_policy_summary(capability_policies)
    if dict(summary) != expected:
        raise ValueError("qualification dose-policy summary mismatch")


def standardized_channel_separations(
    delta: np.ndarray,
    *,
    context_length: int,
    scale_by_channel: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-L168 and H48 per-channel RMS paired distances."""

    values = np.asarray(delta, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if (
        values.ndim != 2
        or not np.isfinite(values).all()
        or int(context_length) < REAL_ANCHORED_FIXED_HISTORY_LENGTH
        or values.shape[0] != int(context_length) + REAL_ANCHORED_HORIZON
    ):
        raise ValueError("paired separation delta has invalid shape or values")
    if scale_by_channel is None:
        scales = np.ones(values.shape[1], dtype=float)
    else:
        scales = np.asarray(scale_by_channel, dtype=float)
        if (
            scales.shape != (values.shape[1],)
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
        ):
            raise ValueError("paired separation scales are invalid")
    normalized = values / scales[None, :]
    history = normalized[
        int(context_length) - REAL_ANCHORED_FIXED_HISTORY_LENGTH : int(context_length)
    ]
    future = normalized[int(context_length) :]
    return (
        np.sqrt(np.mean(history**2, axis=0)),
        np.sqrt(np.mean(future**2, axis=0)),
    )


def paired_minimum_separation_gate(
    delta: np.ndarray,
    *,
    context_length: int,
    dose_index: int,
    dose_calibration: Mapping[str, Any],
    affected_channel_indices: Sequence[int],
    scale_by_channel: Sequence[float] | None = None,
    previous_delta: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the real-anchored treatment-to-baseline separation band.

    ``delta`` is the treatment minus its paired authentic baseline.  The
    function is treatment/pair-level; callers must not run it on the repeated
    alpha-one baseline member, whose zero distance is intentional.
    """

    validate_dose_calibration(dose_calibration)
    if dose_calibration.get("status") != "available":
        raise ValueError("paired separation requires an available dose mapping")
    index = int(dose_index)
    if not 1 <= index <= len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID):
        raise ValueError("paired separation dose index is out of range")
    history_by_channel, future_by_channel = standardized_channel_separations(
        delta,
        context_length=context_length,
        scale_by_channel=scale_by_channel,
    )
    affected = [int(value) for value in affected_channel_indices]
    if (
        not affected
        or len(affected) != len(set(affected))
        or min(affected) < 0
        or max(affected) >= history_by_channel.size
    ):
        raise ValueError("paired separation affected channels are invalid")
    history_values = history_by_channel[affected]
    future_values = future_by_channel[affected]
    designed_history_target = float(
        dose_calibration["history_target_grid"][index - 1]
    )
    designed_future_target = float(
        dose_calibration["future_target_grid"][index - 1]
    )
    history_target = (
        REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        * designed_history_target
    )
    future_target = (
        REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        * designed_future_target
    )
    history_macro = float(np.mean(history_values))
    future_macro = float(np.mean(future_values))
    channel_history_target = (
        REAL_ANCHORED_PANEL_CHANNEL_TARGET_FRACTION * history_target
    )
    channel_future_target = (
        REAL_ANCHORED_PANEL_CHANNEL_TARGET_FRACTION * future_target
    )
    channel_passed = (
        (history_values + _NUMERICAL_TOLERANCE >= channel_history_target)
        & (future_values + _NUMERICAL_TOLERANCE >= channel_future_target)
    )
    minimum_channel_count = int(math.ceil(len(affected) / 2.0))
    macro_passed = bool(
        history_macro + _NUMERICAL_TOLERANCE >= history_target
        and future_macro + _NUMERICAL_TOLERANCE >= future_target
    )
    coverage_passed = bool(int(np.sum(channel_passed)) >= minimum_channel_count)
    maximum_history_macro = float(
        dose_calibration["maximum_history_macro_separation"]
    )
    maximum_future_macro = float(
        dose_calibration["maximum_future_macro_separation"]
    )
    maximum_channel = float(
        dose_calibration["maximum_affected_channel_separation"]
    )
    macro_upper_bound_passed = bool(
        history_macro <= maximum_history_macro + _NUMERICAL_TOLERANCE
        and future_macro <= maximum_future_macro + _NUMERICAL_TOLERANCE
    )
    channel_upper_bound_passed = bool(
        np.all(
            history_values
            <= maximum_channel + _NUMERICAL_TOLERANCE
        )
        and np.all(
            future_values
            <= maximum_channel + _NUMERICAL_TOLERANCE
        )
    )
    local_augmentation_budget_passed = bool(
        macro_upper_bound_passed and channel_upper_bound_passed
    )
    if previous_delta is None:
        adjacent_history_values = history_values.copy()
        adjacent_future_values = future_values.copy()
        designed_adjacent_history_target = designed_history_target
        designed_adjacent_future_target = designed_future_target
        previous_index = 0
    else:
        if index <= 1:
            raise ValueError(
                "paired separation previous_delta is only valid after dose one"
            )
        previous_history, previous_future = standardized_channel_separations(
            np.asarray(delta, dtype=float) - np.asarray(previous_delta, dtype=float),
            context_length=context_length,
            scale_by_channel=scale_by_channel,
        )
        adjacent_history_values = previous_history[affected]
        adjacent_future_values = previous_future[affected]
        designed_adjacent_history_target = designed_history_target - float(
            dose_calibration["history_target_grid"][index - 2]
        )
        designed_adjacent_future_target = designed_future_target - float(
            dose_calibration["future_target_grid"][index - 2]
        )
        previous_index = index - 1
    adjacent_history_minimum = (
        REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        * designed_adjacent_history_target
    )
    adjacent_future_minimum = (
        REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        * designed_adjacent_future_target
    )
    adjacent_history_macro = float(np.mean(adjacent_history_values))
    adjacent_future_macro = float(np.mean(adjacent_future_values))
    adjacent_channel_history_target = (
        REAL_ANCHORED_PANEL_CHANNEL_TARGET_FRACTION
        * adjacent_history_minimum
    )
    adjacent_channel_future_target = (
        REAL_ANCHORED_PANEL_CHANNEL_TARGET_FRACTION
        * adjacent_future_minimum
    )
    adjacent_channel_passed = (
        (
            adjacent_history_values + _NUMERICAL_TOLERANCE
            >= adjacent_channel_history_target
        )
        & (
            adjacent_future_values + _NUMERICAL_TOLERANCE
            >= adjacent_channel_future_target
        )
    )
    adjacent_macro_passed = bool(
        adjacent_history_macro + _NUMERICAL_TOLERANCE
        >= adjacent_history_minimum
        and adjacent_future_macro + _NUMERICAL_TOLERANCE
        >= adjacent_future_minimum
    )
    adjacent_coverage_passed = bool(
        int(np.sum(adjacent_channel_passed)) >= minimum_channel_count
    )
    adjacent_accepted = bool(
        adjacent_macro_passed and adjacent_coverage_passed
    )
    # Adjacent-dose separation is a legibility diagnostic.  The requested
    # contamination control is treatment-to-authentic-source only.
    accepted = bool(
        macro_passed and coverage_passed and local_augmentation_budget_passed
    )
    reason_code = None
    if not accepted:
        if not local_augmentation_budget_passed:
            reason_code = "real_anchor_treatment_exceeds_local_augmentation_budget"
        else:
            reason_code = "real_anchor_treatment_too_close_to_authentic_source"
    return {
        "schema_version": REAL_ANCHORED_PAIRED_SEPARATION_SCHEMA,
        "status": "passed" if accepted else "failed",
        "accepted": accepted,
        "reason_code": reason_code,
        "dose_index": index,
        "canonical_strength": float(
            dose_calibration["strength_grid"][index - 1]
        ),
        "applied_alpha": float(
            dose_calibration["applied_alpha_grid"][index - 1]
        ),
        "history_window_length": REAL_ANCHORED_FIXED_HISTORY_LENGTH,
        "horizon": REAL_ANCHORED_HORIZON,
        "history_macro_separation": history_macro,
        "future_macro_separation": future_macro,
        "minimum_history_macro_separation": history_target,
        "minimum_future_macro_separation": future_target,
        "designed_history_target": designed_history_target,
        "designed_future_target": designed_future_target,
        "minimum_acceptance_fraction": (
            REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        ),
        "affected_channel_indices": affected,
        "history_separation_by_affected_channel": history_values.tolist(),
        "future_separation_by_affected_channel": future_values.tolist(),
        "channel_target_fraction": (
            REAL_ANCHORED_PANEL_CHANNEL_TARGET_FRACTION
        ),
        "minimum_passing_channel_count": minimum_channel_count,
        "passing_channel_count": int(np.sum(channel_passed)),
        "passing_affected_channel_indices": [
            affected[offset]
            for offset, passed in enumerate(channel_passed.tolist())
            if passed
        ],
        "macro_passed": macro_passed,
        "channel_coverage_passed": coverage_passed,
        "maximum_history_macro_separation": maximum_history_macro,
        "maximum_future_macro_separation": maximum_future_macro,
        "maximum_affected_channel_separation": maximum_channel,
        "macro_upper_bound_passed": macro_upper_bound_passed,
        "channel_upper_bound_passed": channel_upper_bound_passed,
        "local_augmentation_budget_passed": (
            local_augmentation_budget_passed
        ),
        "adjacent_status": (
            "passed" if adjacent_accepted else "diagnostic_below_target"
        ),
        "adjacent_accepted": adjacent_accepted,
        "previous_dose_index": previous_index,
        "adjacent_history_macro_separation": adjacent_history_macro,
        "adjacent_future_macro_separation": adjacent_future_macro,
        "designed_adjacent_history_target": (
            designed_adjacent_history_target
        ),
        "designed_adjacent_future_target": designed_adjacent_future_target,
        "minimum_adjacent_history_macro_separation": (
            adjacent_history_minimum
        ),
        "minimum_adjacent_future_macro_separation": (
            adjacent_future_minimum
        ),
        "adjacent_history_separation_by_affected_channel": (
            adjacent_history_values.tolist()
        ),
        "adjacent_future_separation_by_affected_channel": (
            adjacent_future_values.tolist()
        ),
        "adjacent_passing_channel_count": int(
            np.sum(adjacent_channel_passed)
        ),
        "adjacent_macro_passed": adjacent_macro_passed,
        "adjacent_channel_coverage_passed": adjacent_coverage_passed,
        "dose_calibration_policy_sha256": str(
            dose_calibration.get(
                "dose_policy_sha256",
                dose_calibration["policy_sha256"],
            )
        ),
        "contract_dose_calibration_sha256": str(
            dose_calibration["policy_sha256"]
        ),
        "anti_copy_semantics": (
            "treatment_only_distance_from_authentic_source"
        ),
        "baseline_member_policy": "exact_authentic_source_exempt",
        "adjacent_distance_role": "diagnostic_only",
        "gate_semantics": (
            "treatment_source_distance_not_synthetic_loo_dcr_nndr"
        ),
        "target_future_used_for_gate": False,
    }
