from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_v2_feature_gate_artifact.json"
SCHEMA_VERSION = "synthetic_feature_support_gate.v1"
CLEAN_DETERMINISTIC_EXCLUDED_CONTROLS = frozenset(
    {
        "noise_ratio",
        "outlier_rate",
        "spike_rate",
        "diff_spike_rate",
        "volatility_shift_strength",
        "covariate_residual_outlier_rate",
        "covariate_residual_spike_rate",
    }
)


def evaluate_feature_support_gate(
    *,
    capability_id: str,
    features: dict[str, float],
    profile_ids: tuple[str, ...],
    context_length: int,
    horizon: int,
    target_dim: int,
    artifact: dict[str, Any] | None = None,
    evaluation_mode: str = "standard",
) -> dict[str, Any]:
    if evaluation_mode not in {"standard", "clean_deterministic"}:
        raise ValueError(
            "evaluation_mode must be standard or clean_deterministic"
        )
    source = artifact if artifact is not None else load_feature_gate_artifact()
    if not source:
        return not_enforced(
            "artifact_missing",
            capability_id,
            profile_ids,
            context_length,
            horizon,
            target_dim,
        )

    buckets = matching_calibrated_buckets(
        capability_id=capability_id,
        profile_ids=profile_ids,
        context_length=context_length,
        horizon=horizon,
        target_dim=target_dim,
        artifact=source,
    )
    if not buckets:
        return not_enforced(
            "no_matching_calibrated_bucket",
            capability_id,
            profile_ids,
            context_length,
            horizon,
            target_dim,
        )

    bucket_results = [
        evaluate_bucket(
            capability_id,
            features,
            bucket,
            evaluation_mode=evaluation_mode,
        )
        for bucket in buckets
    ]
    accepted_results = [result for result in bucket_results if result["accepted"]]
    candidates = accepted_results or bucket_results
    matched = min(
        candidates,
        key=lambda result: (
            float(result["normalized_score"])
            if result.get("normalized_score") is not None
            else float("inf")
        ),
    )
    accepted = bool(accepted_results)
    failed_features = sorted(
        {
            feature
            for result in bucket_results
            for feature in result.get("missing_features", [])
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_schema_version": source.get("schema_version"),
        "artifact_created_at": source.get("created_at"),
        "accepted": accepted,
        "enforced": True,
        "status": "passed" if accepted else "outside_real_control_support",
        "capability_id": capability_id,
        "evaluation_mode": evaluation_mode,
        "profile_ids": list(profile_ids),
        "evaluated_bucket_count": len(bucket_results),
        "matched_profile_id": matched["profile_id"],
        "support_method": matched["support_method"],
        "score": matched["score"],
        "threshold": matched["threshold"],
        "calibration_coverage": matched["coverage"],
        "normalized_score": matched["normalized_score"],
        "control_features": matched["control_features"],
        "target_percentile_diagnostics": matched["target_percentile_diagnostics"],
        "failed_features": failed_features,
        "bucket_results": bucket_results,
    }


@lru_cache(maxsize=1)
def load_feature_gate_artifact() -> dict[str, Any] | None:
    if not ARTIFACT_PATH.exists():
        return None
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def matching_calibrated_buckets(
    *,
    capability_id: str,
    profile_ids: tuple[str, ...],
    context_length: int,
    horizon: int,
    target_dim: int,
    artifact: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source = artifact if artifact is not None else load_feature_gate_artifact()
    if not source:
        return []
    buckets = source.get("buckets", {})
    return [
        bucket
        for profile_id in profile_ids
        for bucket in [buckets.get(profile_id)]
        if bucket is not None
        and capability_id in bucket.get("capabilities", {})
        and int(bucket.get("context_length", -1)) == int(context_length)
        and int(bucket.get("horizon", -1)) == int(horizon)
        and int(bucket.get("target_dim", -1)) == int(target_dim)
    ]


def evaluate_bucket(
    capability_id: str,
    features: dict[str, float],
    bucket: dict[str, Any],
    *,
    evaluation_mode: str = "standard",
) -> dict[str, Any]:
    config = bucket["capabilities"][capability_id]
    original_support = config["control_support"]
    support = (
        project_clean_deterministic_support(original_support)
        if evaluation_mode == "clean_deterministic"
        else original_support
    )
    names = tuple(str(name) for name in support["feature_names"])
    excluded_names = sorted(
        set(original_support["feature_names"]) - set(names)
    )
    missing = [
        name
        for name in names
        if name not in features or not np.isfinite(features[name])
    ]
    threshold = float(support["threshold"])
    if not names:
        score = 0.0
        accepted = True
        values = []
    elif missing:
        score = float("inf")
        accepted = False
        values: list[float] = []
    else:
        values = [float(features[name]) for name in names]
        score = robust_mahalanobis_score(np.asarray(values, dtype=float), support)
        accepted = bool(score <= threshold)

    marginal_quantiles = support.get("marginal_quantiles", {})
    control_diagnostics = {
        name: {
            "value": float(features[name]),
            "p01": float(marginal_quantiles.get(name, {}).get("p01", float("nan"))),
            "p50": float(marginal_quantiles.get(name, {}).get("p50", float("nan"))),
            "p99": float(marginal_quantiles.get(name, {}).get("p99", float("nan"))),
        }
        for name in names
        if name in features and np.isfinite(features[name])
    }
    target_diagnostics = target_percentile_diagnostics(
        features,
        config.get("target_reference", {}),
    )
    normalized = score / max(threshold, 1e-12)
    return {
        "profile_id": bucket["profile_id"],
        "accepted": accepted,
        "status": "passed" if accepted else ("missing_features" if missing else "outside_support"),
        "support_method": str(support.get("method", "robust_mahalanobis")),
        "evaluation_mode": evaluation_mode,
        "excluded_control_features": excluded_names,
        "threshold_calibration": (
            "projected_from_standard_support_requires_v8_recalibration"
            if excluded_names
            else "artifact_native"
        ),
        "score": round_float(score),
        "threshold": round_float(threshold),
        "normalized_score": round_float(normalized),
        "control_features": control_diagnostics,
        "target_percentile_diagnostics": target_diagnostics,
        "missing_features": missing,
        "reference_count": int(support.get("reference_count", 0)),
        "calibration_count": int(support.get("calibration_count", 0)),
        "coverage": float(support.get("coverage", 0.0)),
    }


def project_clean_deterministic_support(
    support: dict[str, Any],
) -> dict[str, Any]:
    """Project a legacy real-data gate away from stochastic tail controls.

    This projection is useful for a v8 pilot audit.  Its threshold remains the
    legacy conformal threshold and must be rebuilt from the original split
    before use as a formal acceptance gate.
    """

    original_names = [str(name) for name in support["feature_names"]]
    keep = [
        index
        for index, name in enumerate(original_names)
        if name not in CLEAN_DETERMINISTIC_EXCLUDED_CONTROLS
    ]
    projected = dict(support)
    projected["feature_names"] = [original_names[index] for index in keep]
    for key in ("feature_center", "feature_scale", "robust_location_z"):
        values = np.asarray(support.get(key, []), dtype=float)
        projected[key] = values[keep].tolist() if keep else []
    if keep:
        precision = np.asarray(support["precision"], dtype=float)
        covariance = np.linalg.pinv(precision)
        projected_covariance = covariance[np.ix_(keep, keep)]
        projected["precision"] = np.linalg.pinv(
            projected_covariance
        ).tolist()
    else:
        projected["precision"] = []
    marginal = support.get("marginal_quantiles", {})
    projected["marginal_quantiles"] = {
        name: marginal[name]
        for name in projected["feature_names"]
        if name in marginal
    }
    return projected


def robust_mahalanobis_score(values: np.ndarray, support: dict[str, Any]) -> float:
    center = np.asarray(support["feature_center"], dtype=float)
    scale = np.asarray(support["feature_scale"], dtype=float)
    location = np.asarray(support["robust_location_z"], dtype=float)
    precision = np.asarray(support["precision"], dtype=float)
    safe_scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (values - center) / safe_scale
    delta = standardized - location
    squared = float(delta @ precision @ delta)
    dimension = max(1, values.size)
    return float(np.sqrt(max(0.0, squared) / dimension))


def target_percentile_diagnostics(
    features: dict[str, float],
    target_reference: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for name, reference in target_reference.items():
        value = features.get(name)
        if value is None or not np.isfinite(value):
            continue
        quantiles = {
            key: float(item)
            for key, item in reference.get("quantiles", {}).items()
            if key.startswith("p") and np.isfinite(item)
        }
        diagnostics[name] = {
            "value": float(value),
            "direction": str(reference.get("direction", "increase")),
            "approx_real_percentile": round_float(approximate_percentile(float(value), quantiles)),
            "quantiles": quantiles,
        }
    return diagnostics


def approximate_percentile(value: float, quantiles: dict[str, float]) -> float:
    points = sorted(
        (float(key[1:]) / 100.0, float(item))
        for key, item in quantiles.items()
        if key[1:].isdigit()
    )
    if not points:
        return float("nan")
    percentile_values = np.asarray([point[0] for point in points], dtype=float)
    feature_values = np.asarray([point[1] for point in points], dtype=float)
    order = np.argsort(feature_values, kind="stable")
    feature_values = feature_values[order]
    percentile_values = percentile_values[order]
    unique_values, inverse = np.unique(feature_values, return_inverse=True)
    unique_percentiles = np.asarray(
        [float(np.mean(percentile_values[inverse == index])) for index in range(unique_values.size)],
        dtype=float,
    )
    if unique_values.size == 1:
        return float(unique_percentiles[0])
    return float(
        np.interp(
            value,
            unique_values,
            unique_percentiles,
            left=0.0,
            right=1.0,
        )
    )


def not_enforced(
    reason: str,
    capability_id: str,
    profile_ids: tuple[str, ...],
    context_length: int,
    horizon: int,
    target_dim: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_schema_version": None,
        "artifact_created_at": None,
        "accepted": False,
        "enforced": False,
        "status": reason,
        "capability_id": capability_id,
        "profile_ids": list(profile_ids),
        "evaluated_bucket_count": 0,
        "matched_profile_id": None,
        "support_method": None,
        "score": None,
        "threshold": None,
        "calibration_coverage": None,
        "normalized_score": None,
        "control_features": {},
        "target_percentile_diagnostics": {},
        "failed_features": [],
        "bucket_results": [],
        "context_length": int(context_length),
        "horizon": int(horizon),
        "target_dim": int(target_dim),
    }


def round_float(value: float, digits: int = 8) -> float | None:
    if not np.isfinite(value):
        return None
    return float(round(float(value), digits))
