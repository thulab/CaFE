#!/usr/bin/env python3
"""Lightweight dataset-local realism audits for CaFE samples.

The gate intentionally has two narrow responsibilities:

* report where a generated primary feature falls relative to the real anchors;
  this is deliberately diagnostic rather than a sample acceptance gate; and
* reject histories that are unusually close to a real calibration anchor under
  a leave-one-out DCR/NNDR rule.

It does not re-extract synthetic features and it does not attempt to make a
synthetic trajectory look globally similar to a real trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "cafe.realism_gate.v1"
HISTORY_LENGTH = 168
MINIMUM_ANCHOR_COUNT = 12
DISTANCE_EPSILON = 1e-12
REAL_FEATURE_SCOPE = "dataset_real_generator_overlap_reference"
LEGACY_REAL_FEATURE_SCOPE = "dataset_real_feature_range"


@dataclass(frozen=True)
class FeatureSupportPolicy:
    capability_id: str
    target_feature: str | None
    source_scope: str | None
    enforced: bool
    reason_code: str | None
    finite_anchor_count: int
    real_minimum: float | None
    real_maximum: float | None
    full_span: float | None
    padding_fraction: float
    lower_bound: float | None
    upper_bound: float | None

    def summary(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "target_feature": self.target_feature,
            "source_scope": self.source_scope,
            "enforced": self.enforced,
            "reason_code": self.reason_code,
            "finite_anchor_count": self.finite_anchor_count,
            "real_minimum": self.real_minimum,
            "real_maximum": self.real_maximum,
            "full_span": self.full_span,
            "padding_fraction": self.padding_fraction,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class NearDistancePolicy:
    requested_enabled: bool
    enforced: bool
    reason_code: str | None
    history_length: int
    anchor_count: int
    pooled_mean: float | None
    pooled_std: float | None
    d1_p01: float | None
    nndr_p01: float | None
    normalized_anchor_histories: np.ndarray | None

    def summary(self) -> dict[str, Any]:
        """Return metadata only; never expose the reference matrix."""

        return {
            "requested_enabled": self.requested_enabled,
            "enforced": self.enforced,
            "reason_code": self.reason_code,
            "history_length": self.history_length,
            "anchor_count": self.anchor_count,
            "pooled_mean": self.pooled_mean,
            "pooled_std": self.pooled_std,
            "d1_p01": self.d1_p01,
            "nndr_p01": self.nndr_p01,
            "distance": "pooled_z_normalized_rms",
            "calibration": "all_real_anchors_leave_one_out",
            "channel_risk_rule": (
                "d1<=loo_d1_p01 AND nndr<=loo_nndr_p01"
            ),
            "sample_risk_rule": (
                "univariate_channel_risk_or_multivariate_majority_vote"
            ),
        }


@dataclass(frozen=True)
class RealismGateContext:
    schema_version: str
    feature_policies: dict[str, FeatureSupportPolicy]
    single_feature_policy: FeatureSupportPolicy | None
    near_distance_policy: NearDistancePolicy
    policy_summary: dict[str, Any]


def build_realism_gate_context(
    anchors: Iterable[Mapping[str, Any]],
    real_anchor_masters: Iterable[Mapping[str, Any]],
    capability_calibration: Mapping[str, Any],
    near_distance_enabled: bool,
    feature_padding_fraction: float = 0.1,
) -> RealismGateContext:
    """Build a pickle-safe, reusable realism-gate context.

    Large reference histories live only in ``near_distance_policy``.  The
    ``policy_summary`` field is safe to write into a manifest or report.
    Padding is applied on each side of the raw anchor range, so the default
    expands its total width to 1.2 times the observed span.
    """

    padding = float(feature_padding_fraction)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError(
            "feature_padding_fraction must be finite and non-negative"
        )
    anchor_rows = list(anchors)
    calibration_rows, single_record = _capability_calibration_rows(
        capability_calibration
    )
    feature_policies = {
        capability_id: _build_feature_policy(
            capability_id,
            record,
            anchor_rows,
            padding,
        )
        for capability_id, record in calibration_rows.items()
    }
    single_feature_policy = (
        _build_feature_policy(
            "__single__",
            single_record,
            anchor_rows,
            padding,
        )
        if single_record is not None
        else None
    )
    near_policy = _build_near_distance_policy(
        list(real_anchor_masters),
        enabled=bool(near_distance_enabled),
    )
    policy_summary = {
        "schema_version": SCHEMA_VERSION,
        "real_feature_alignment_semantics": (
            "diagnostic_only_never_rejects_a_finite_generated_sample"
        ),
        "feature_support": {
            capability_id: policy.summary()
            for capability_id, policy in sorted(feature_policies.items())
        },
        "single_feature_support": (
            single_feature_policy.summary()
            if single_feature_policy is not None
            else None
        ),
        "near_distance": near_policy.summary(),
    }
    return RealismGateContext(
        schema_version=SCHEMA_VERSION,
        feature_policies=feature_policies,
        single_feature_policy=single_feature_policy,
        near_distance_policy=near_policy,
        policy_summary=policy_summary,
    )


def evaluate_sample(
    sample: Mapping[str, Any],
    context: RealismGateContext,
) -> dict[str, Any]:
    """Evaluate one generated sample without re-extracting any feature."""

    target, validation_failure = _validated_sample_target(sample)
    feature_result = _evaluate_feature_support(sample, context)
    target_match_result = _evaluate_intensity_target_match(
        sample,
        context,
    )
    if validation_failure is not None:
        near_result = {
            "enforced": bool(context.near_distance_policy.enforced),
            "accepted": False,
            "status": "failed",
            "reason_code": validation_failure,
            "channels": [],
            "risk_channel_indices": [],
        }
        failure_codes = [validation_failure]
        if not feature_result["accepted"]:
            failure_codes.append(str(feature_result["failure_code"]))
        if not target_match_result["accepted"]:
            failure_codes.append(
                str(target_match_result["failure_code"])
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_support": feature_result,
            "intensity_target_match": target_match_result,
            "near_distance": near_result,
            "accepted": False,
            "failure_codes": _deduplicate(failure_codes),
        }

    assert target is not None
    near_result = _evaluate_near_distance(
        target,
        context_length=int(sample["context_length"]),
        policy=context.near_distance_policy,
    )
    failure_codes: list[str] = []
    if not feature_result["accepted"]:
        failure_codes.append(str(feature_result["failure_code"]))
    if not target_match_result["accepted"]:
        failure_codes.append(
            str(target_match_result["failure_code"])
        )
    if not near_result["accepted"]:
        failure_codes.append("near_distance_copy_risk")
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_support": feature_result,
        "intensity_target_match": target_match_result,
        "near_distance": near_result,
        "accepted": not failure_codes,
        "failure_codes": failure_codes,
    }


def _capability_calibration_rows(
    calibration: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any] | None]:
    if not isinstance(calibration, Mapping):
        raise ValueError("capability_calibration must be a mapping")
    wrapped = calibration.get("capabilities")
    if isinstance(wrapped, Mapping):
        return _mapping_records(wrapped), None
    if "intensity_calibration_scope" in calibration:
        return {}, calibration
    records = _mapping_records(calibration)
    if records:
        return records, None
    return {}, None


def _mapping_records(
    values: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(capability_id): record
        for capability_id, record in values.items()
        if isinstance(record, Mapping)
        and (
            "intensity_calibration_scope" in record
            or "target_feature" in record
        )
    }


def _build_feature_policy(
    capability_id: str,
    calibration: Mapping[str, Any],
    anchors: list[Mapping[str, Any]],
    padding_fraction: float,
) -> FeatureSupportPolicy:
    target_feature_raw = calibration.get("target_feature")
    target_feature = (
        str(target_feature_raw) if target_feature_raw is not None else None
    )
    scope = calibration.get("intensity_calibration_scope")
    real_feature_calibration = calibration.get("real_feature_calibration")
    real_reference_available = bool(
        isinstance(real_feature_calibration, Mapping)
        and real_feature_calibration.get("usable")
    ) or scope in {REAL_FEATURE_SCOPE, LEGACY_REAL_FEATURE_SCOPE}
    if not real_reference_available:
        return _feature_not_enforced(
            capability_id,
            target_feature,
            "real_feature_reference_unavailable",
            padding_fraction,
        )
    if not target_feature:
        return _feature_not_enforced(
            capability_id,
            None,
            "target_feature_missing",
            padding_fraction,
        )
    values: list[float] = []
    source_scope = (
        str(real_feature_calibration.get("scope"))
        if isinstance(real_feature_calibration, Mapping)
        and real_feature_calibration.get("scope") is not None
        else "real_univariate"
    )
    feature_row_key = {
        "real_univariate": "features",
        "real_native_multivariate": "native_multivariate_features",
        "real_declared_hierarchy": "declared_hierarchy_features",
        "real_hierarchy_children": "hierarchy_children_features",
        "real_known_future_covariates": (
            "known_future_covariate_features"
        ),
    }.get(source_scope)
    if feature_row_key is None:
        return _feature_not_enforced(
            capability_id,
            target_feature,
            "unsupported_real_feature_scope",
            padding_fraction,
        )
    for anchor in anchors:
        features = anchor.get(feature_row_key)
        if not isinstance(features, Mapping):
            continue
        value = _finite_float(features.get(target_feature))
        if value is not None:
            values.append(value)
    if len(values) < MINIMUM_ANCHOR_COUNT:
        return _feature_not_enforced(
            capability_id,
            target_feature,
            "insufficient_finite_anchor_features",
            padding_fraction,
            finite_anchor_count=len(values),
        )
    minimum = float(min(values))
    maximum = float(max(values))
    span = maximum - minimum
    padding = padding_fraction * span
    return FeatureSupportPolicy(
        capability_id=capability_id,
        target_feature=target_feature,
        source_scope=source_scope,
        enforced=True,
        reason_code=None,
        finite_anchor_count=len(values),
        real_minimum=minimum,
        real_maximum=maximum,
        full_span=span,
        padding_fraction=padding_fraction,
        lower_bound=minimum - padding,
        upper_bound=maximum + padding,
    )


def _feature_not_enforced(
    capability_id: str,
    target_feature: str | None,
    reason_code: str,
    padding_fraction: float,
    *,
    finite_anchor_count: int = 0,
) -> FeatureSupportPolicy:
    return FeatureSupportPolicy(
        capability_id=capability_id,
        target_feature=target_feature,
        source_scope=None,
        enforced=False,
        reason_code=reason_code,
        finite_anchor_count=finite_anchor_count,
        real_minimum=None,
        real_maximum=None,
        full_span=None,
        padding_fraction=padding_fraction,
        lower_bound=None,
        upper_bound=None,
    )


def _build_near_distance_policy(
    real_anchor_masters: list[Mapping[str, Any]],
    *,
    enabled: bool,
) -> NearDistancePolicy:
    if not enabled:
        return _near_not_enforced(
            enabled=False,
            reason_code="disabled_by_policy",
            anchor_count=len(real_anchor_masters),
        )
    if len(real_anchor_masters) < MINIMUM_ANCHOR_COUNT:
        return _near_not_enforced(
            enabled=True,
            reason_code="insufficient_real_anchor_masters",
            anchor_count=len(real_anchor_masters),
        )

    histories = np.stack(
        [_real_anchor_history(row) for row in real_anchor_masters],
        axis=0,
    ).astype(np.float32, copy=False)
    pooled_mean = float(np.mean(histories, dtype=np.float64))
    pooled_std = float(np.std(histories, dtype=np.float64))
    if not math.isfinite(pooled_std) or pooled_std <= DISTANCE_EPSILON:
        return _near_not_enforced(
            enabled=True,
            reason_code="degenerate_real_anchor_scale",
            anchor_count=len(real_anchor_masters),
            pooled_mean=pooled_mean,
            pooled_std=pooled_std,
        )
    normalized = np.asarray(
        (histories - np.float32(pooled_mean)) / np.float32(pooled_std),
        dtype=np.float32,
    )
    pairwise = _rms_distances(normalized[:, None, :], normalized[None, :, :])
    np.fill_diagonal(pairwise, np.inf)
    nearest_two = np.partition(pairwise, kth=1, axis=1)[:, :2]
    nearest_two.sort(axis=1)
    loo_d1 = nearest_two[:, 0]
    loo_d2 = nearest_two[:, 1]
    loo_nndr = loo_d1 / np.maximum(loo_d2, DISTANCE_EPSILON)
    d1_p01 = float(np.quantile(loo_d1, 0.01))
    nndr_p01 = float(np.quantile(loo_nndr, 0.01))
    if not math.isfinite(d1_p01) or not math.isfinite(nndr_p01):
        raise ValueError("real-anchor LOO thresholds must be finite")
    normalized.setflags(write=False)
    return NearDistancePolicy(
        requested_enabled=True,
        enforced=True,
        reason_code=None,
        history_length=HISTORY_LENGTH,
        anchor_count=len(real_anchor_masters),
        pooled_mean=pooled_mean,
        pooled_std=pooled_std,
        d1_p01=d1_p01,
        nndr_p01=nndr_p01,
        normalized_anchor_histories=normalized,
    )


def _near_not_enforced(
    *,
    enabled: bool,
    reason_code: str,
    anchor_count: int,
    pooled_mean: float | None = None,
    pooled_std: float | None = None,
) -> NearDistancePolicy:
    return NearDistancePolicy(
        requested_enabled=enabled,
        enforced=False,
        reason_code=reason_code,
        history_length=HISTORY_LENGTH,
        anchor_count=anchor_count,
        pooled_mean=pooled_mean,
        pooled_std=pooled_std,
        d1_p01=None,
        nndr_p01=None,
        normalized_anchor_histories=None,
    )


def _real_anchor_history(master: Mapping[str, Any]) -> np.ndarray:
    try:
        values = np.asarray(master["target"], dtype=np.float32)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("real anchor master has an invalid target") from error
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(
            "real anchor master target must have exactly one channel"
        )
    if not np.isfinite(values).all():
        raise ValueError("real anchor master target must be finite")
    context_length = _positive_int(
        master.get("context_length"),
        label="real anchor master context_length",
    )
    if context_length < HISTORY_LENGTH or context_length > values.shape[0]:
        raise ValueError(
            "real anchor master must contain a complete L168 history"
        )
    return np.asarray(
        values[
            context_length - HISTORY_LENGTH : context_length,
            0,
        ],
        dtype=np.float32,
    )


def _evaluate_feature_support(
    sample: Mapping[str, Any],
    context: RealismGateContext,
) -> dict[str, Any]:
    capability_id = str(sample.get("capability_id", ""))
    policy = context.feature_policies.get(capability_id)
    if policy is None:
        policy = context.single_feature_policy
    if policy is None:
        return {
            "capability_id": capability_id,
            "target_feature": sample.get("target_feature"),
            "enforced": False,
            "accepted": True,
            "status": "not_enforced",
            "reason_code": "capability_calibration_missing",
            "failure_code": None,
        }
    if not policy.enforced:
        return {
            **policy.summary(),
            "accepted": True,
            "status": "not_enforced",
            "failure_code": None,
        }
    value = _finite_float(sample.get("target_feature_value"))
    if value is None:
        return {
            **policy.summary(),
            "value": None,
            "accepted": False,
            "status": "failed",
            "reason_code": "target_feature_value_missing_or_nonfinite",
            "failure_code": "feature_value_missing_or_nonfinite",
        }
    assert policy.lower_bound is not None and policy.upper_bound is not None
    accepted = policy.lower_bound <= value <= policy.upper_bound
    return {
        **policy.summary(),
        "hard_gate_enforced": False,
        "diagnostic_only": True,
        "value": value,
        "within_reference_support": accepted,
        "accepted": True,
        "status": (
            "diagnostic_inside_reference"
            if accepted
            else "diagnostic_outside_reference"
        ),
        "reason_code": (
            None if accepted else "outside_padded_real_anchor_minmax"
        ),
        "failure_code": None,
    }


def _evaluate_intensity_target_match(
    sample: Mapping[str, Any],
    context: RealismGateContext,
) -> dict[str, Any]:
    capability_id = str(sample.get("capability_id", ""))
    policy = context.feature_policies.get(capability_id)
    if policy is None:
        policy = context.single_feature_policy
    actual = _finite_float(sample.get("target_feature_value"))
    expected = _finite_float(
        sample.get("intensity_target_feature_value")
    )
    if actual is None or expected is None:
        return {
            "capability_id": capability_id,
            "target_feature": (
                policy.target_feature if policy is not None else None
            ),
            "enforced": True,
            "accepted": False,
            "status": "failed",
            "actual_value": actual,
            "target_value": expected,
            "reason_code": "family_target_metadata_missing",
            "failure_code": "family_target_metadata_missing_or_nonfinite",
        }
    if policy is None or not policy.enforced:
        return {
            "capability_id": capability_id,
            "target_feature": (
                policy.target_feature if policy is not None else None
            ),
            "enforced": False,
            "accepted": True,
            "status": "not_enforced",
            "actual_value": actual,
            "target_value": expected,
            "absolute_error": abs(actual - expected),
            "reason_code": (
                "capability_calibration_missing"
                if policy is None
                else "real_feature_reference_unavailable"
            ),
            "failure_code": None,
        }
    tolerance = 0.10 * float(policy.full_span or 0.0)
    absolute_error = abs(actual - expected)
    within_tolerance = absolute_error <= tolerance + 1e-12
    return {
        "capability_id": capability_id,
        "target_feature": policy.target_feature,
        "enforced": False,
        "hard_gate_enforced": False,
        "diagnostic_only": True,
        "accepted": True,
        "status": (
            "diagnostic_within_reference_tolerance"
            if within_tolerance
            else "diagnostic_outside_reference_tolerance"
        ),
        "actual_value": actual,
        "target_value": expected,
        "absolute_error": absolute_error,
        "tolerance": tolerance,
        "tolerance_policy": (
            "ten_percent_of_real_anchor_minmax_span_diagnostic_only"
        ),
        "within_reference_tolerance": within_tolerance,
        "reason_code": (
            None
            if within_tolerance
            else "family_mean_target_error_exceeds_reference_tolerance"
        ),
        "failure_code": None,
    }


def _validated_sample_target(
    sample: Mapping[str, Any],
) -> tuple[np.ndarray | None, str | None]:
    try:
        target = np.asarray(sample["target"], dtype=np.float32)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "invalid_sample_target_shape"
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[1] < 1:
        return None, "invalid_sample_target_shape"
    if not np.isfinite(target).all():
        return None, "nonfinite_sample_target"
    try:
        context_length = int(sample["context_length"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "invalid_sample_context_length"
    if (
        context_length < HISTORY_LENGTH
        or context_length > target.shape[0]
    ):
        return None, "invalid_sample_context_length"
    if "target_dim" in sample:
        try:
            if int(sample["target_dim"]) != target.shape[1]:
                return None, "invalid_sample_target_shape"
        except (TypeError, ValueError, OverflowError):
            return None, "invalid_sample_target_shape"
    if "horizon" in sample:
        try:
            if context_length + int(sample["horizon"]) != target.shape[0]:
                return None, "invalid_sample_target_shape"
        except (TypeError, ValueError, OverflowError):
            return None, "invalid_sample_target_shape"
    return target, None


def _evaluate_near_distance(
    target: np.ndarray,
    *,
    context_length: int,
    policy: NearDistancePolicy,
) -> dict[str, Any]:
    if not policy.enforced:
        return {
            **policy.summary(),
            "accepted": True,
            "status": "not_enforced",
            "channels": [],
            "risk_channel_indices": [],
        }
    references = policy.normalized_anchor_histories
    assert references is not None
    assert policy.pooled_mean is not None and policy.pooled_std is not None
    assert policy.d1_p01 is not None and policy.nndr_p01 is not None
    history = target[
        context_length - HISTORY_LENGTH : context_length,
        :,
    ]
    normalized = (
        history - np.float32(policy.pooled_mean)
    ) / np.float32(policy.pooled_std)
    channel_results: list[dict[str, Any]] = []
    risk_channels: list[int] = []
    # A channel loop is intentional; all anchor comparisons remain vectorized.
    for channel_index in range(normalized.shape[1]):
        distances = _rms_distances(
            references,
            normalized[:, channel_index][None, :],
        ).reshape(-1)
        nearest = np.partition(distances, kth=1)[:2]
        nearest.sort()
        d1 = float(nearest[0])
        d2 = float(nearest[1])
        nndr = d1 / max(d2, DISTANCE_EPSILON)
        risk = bool(d1 <= policy.d1_p01 and nndr <= policy.nndr_p01)
        if risk:
            risk_channels.append(channel_index)
        channel_results.append(
            {
                "channel_index": channel_index,
                "d1": d1,
                "d2": d2,
                "nndr": nndr,
                "risk": risk,
            }
        )
    target_dim = int(normalized.shape[1])
    minimum_risk_channels = (
        1 if target_dim == 1 else int(math.ceil(target_dim / 2.0))
    )
    sample_copy_risk = len(risk_channels) >= minimum_risk_channels
    accepted = not sample_copy_risk
    return {
        **policy.summary(),
        "accepted": accepted,
        "status": "passed" if accepted else "failed",
        "reason_code": None if accepted else "near_real_anchor_copy_risk",
        "channels": channel_results,
        "risk_channel_indices": risk_channels,
        "risk_channel_count": len(risk_channels),
        "minimum_risk_channels_for_rejection": minimum_risk_channels,
        "multivariate_multiple_comparison_policy": (
            "not_applicable"
            if target_dim == 1
            else "majority_vote_over_univariate_anchor_comparisons"
        ),
    }


def _rms_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.asarray(left, dtype=np.float32) - np.asarray(
        right,
        dtype=np.float32,
    )
    return np.sqrt(
        np.mean(difference * difference, axis=-1, dtype=np.float64)
    )


def _positive_int(value: Any, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
