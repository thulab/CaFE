from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_v2_near_distance_artifact.json"
SCHEMA_VERSION = "synthetic_near_distance_gate.v2"
ARTIFACT_SCHEMA_VERSION = "synthetic_v2_near_distance_online.v2"


def evaluate_near_distance_gate(
    *,
    target: np.ndarray,
    features: dict[str, float],
    profile_ids: tuple[str, ...],
    context_length: int,
    horizon: int,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = artifact if artifact is not None else load_near_distance_artifact()
    if not source:
        return not_enforced("artifact_missing", profile_ids, context_length, horizon)
    if source.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        return not_enforced("artifact_schema_mismatch", profile_ids, context_length, horizon)

    try:
        target_array = np.asarray(target, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return not_enforced("invalid_query", profile_ids, context_length, horizon)
    if target_array.ndim == 1:
        target_array = target_array[:, None]
    if (
        target_array.ndim != 2
        or target_array.shape[0] != int(context_length) + int(horizon)
        or target_array.shape[1] < 1
        or not np.isfinite(target_array).all()
    ):
        target_dim = int(target_array.shape[1]) if target_array.ndim == 2 else None
        return not_enforced(
            "invalid_query",
            profile_ids,
            context_length,
            horizon,
            target_dim=target_dim,
        )
    target_dim = int(target_array.shape[1])
    candidate_buckets = matching_calibrated_buckets(
        profile_ids=profile_ids,
        context_length=context_length,
        horizon=horizon,
        target_dim=target_dim,
        artifact=source,
    )
    if not candidate_buckets:
        return not_enforced("no_matching_calibrated_bucket", profile_ids, context_length, horizon, target_dim=target_dim)

    bucket_results = [
        evaluate_bucket(target_array, features, bucket)
        for bucket in candidate_buckets
    ]
    incompatible = [result for result in bucket_results if not result["calibration_compatible"]]
    if incompatible:
        return {
            "schema_version": SCHEMA_VERSION,
            "accepted": False,
            "enforced": False,
            "status": "artifact_schema_mismatch",
            "profile_ids": list(profile_ids),
            "evaluated_bucket_count": len(bucket_results),
            "strict_risk": False,
            "combined_risk": False,
            "bucket_results": bucket_results,
        }
    strict_risk = any(result["strict_risk"] for result in bucket_results)
    combined_risk = any(result["combined_risk"] for result in bucket_results)
    accepted = not (strict_risk or combined_risk)
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted": accepted,
        "enforced": True,
        "status": "passed" if accepted else "failed",
        "profile_ids": list(profile_ids),
        "evaluated_bucket_count": len(bucket_results),
        "strict_risk": strict_risk,
        "combined_risk": combined_risk,
        "bucket_results": bucket_results,
    }


@lru_cache(maxsize=1)
def load_near_distance_artifact() -> dict[str, Any] | None:
    if not ARTIFACT_PATH.exists():
        return None
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def matching_calibrated_buckets(
    *,
    profile_ids: tuple[str, ...],
    context_length: int,
    horizon: int,
    target_dim: int,
    artifact: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source = artifact if artifact is not None else load_near_distance_artifact()
    if not source or source.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        return []
    buckets = source.get("buckets", {})
    return [
        bucket
        for profile_id in profile_ids
        for bucket in [buckets.get(profile_id)]
        if bucket is not None
        and int(bucket.get("context_length", -1)) == int(context_length)
        and int(bucket.get("horizon", -1)) == int(horizon)
        and int(bucket.get("target_dim", -1)) == int(target_dim)
    ]


def not_enforced(
    reason: str,
    profile_ids: tuple[str, ...],
    context_length: int,
    horizon: int,
    *,
    target_dim: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "accepted": False,
        "enforced": False,
        "status": reason,
        "profile_ids": list(profile_ids),
        "evaluated_bucket_count": 0,
        "strict_risk": False,
        "combined_risk": False,
        "bucket_results": [],
        "context_length": int(context_length),
        "horizon": int(horizon),
    }
    if target_dim is not None:
        result["target_dim"] = int(target_dim)
    return result


def evaluate_bucket(target: np.ndarray, features: dict[str, float], bucket: dict[str, Any]) -> dict[str, Any]:
    required_artifact_keys = {
        "context_length",
        "horizon",
        "target_dim",
        "feature_names",
        "feature_center",
        "feature_scale",
        "reference_raw",
        "reference_context_raw",
        "reference_features_z",
        "thresholds",
    }
    missing_artifact_keys = sorted(required_artifact_keys - set(bucket))
    feature_names = tuple(str(name) for name in bucket.get("feature_names", ()))
    missing_feature_names = [
        name
        for name in feature_names
        if name not in features or not _is_finite_number(features[name])
    ]
    if missing_artifact_keys or missing_feature_names:
        return incompatible_bucket_result(
            bucket,
            missing_artifact_keys=missing_artifact_keys,
            missing_feature_names=missing_feature_names,
        )

    required_thresholds = {
        "raw_mae_p01",
        "raw_mae_p05",
        "raw_l2_p01",
        "raw_l2_p05",
        "feature_l2_p01",
        "raw_mae_nndr_p01",
        "context_raw_mae_p01",
        "context_raw_mae_p05",
        "context_raw_l2_p01",
        "context_raw_l2_p05",
        "context_raw_mae_nndr_p01",
    }
    raw_thresholds = bucket["thresholds"]
    if not isinstance(raw_thresholds, dict):
        return incompatible_bucket_result(bucket, artifact_errors=["thresholds must be an object"])
    missing_thresholds = sorted(required_thresholds - set(raw_thresholds))
    if missing_thresholds:
        return incompatible_bucket_result(bucket, missing_thresholds=missing_thresholds)

    raw = flatten_raw(target)
    context_length = int(bucket["context_length"])
    context_raw = flatten_raw(target[:context_length])
    try:
        reference_raw = np.asarray(bucket["reference_raw"], dtype=float)
        reference_context_raw = np.asarray(bucket["reference_context_raw"], dtype=float)
        feature_center = np.asarray(bucket["feature_center"], dtype=float)
        feature_scale = np.asarray(bucket["feature_scale"], dtype=float)
        reference_features_z = np.asarray(bucket["reference_features_z"], dtype=float)
        thresholds = {name: float(raw_thresholds[name]) for name in required_thresholds}
    except (TypeError, ValueError, OverflowError) as error:
        return incompatible_bucket_result(bucket, artifact_errors=[f"non-numeric calibration data: {error}"])

    reference_count = reference_raw.shape[0] if reference_raw.ndim == 2 else 0
    artifact_errors: list[str] = []
    if not feature_names:
        artifact_errors.append("feature_names must not be empty")
    if reference_raw.ndim != 2 or reference_raw.shape != (reference_count, raw.size) or reference_count < 2:
        artifact_errors.append("reference_raw has an incompatible shape")
    if reference_context_raw.shape != (reference_count, context_raw.size):
        artifact_errors.append("reference_context_raw has an incompatible shape")
    if feature_center.shape != (len(feature_names),) or feature_scale.shape != (len(feature_names),):
        artifact_errors.append("feature center/scale has an incompatible shape")
    if reference_features_z.shape != (reference_count, len(feature_names)):
        artifact_errors.append("reference_features_z has an incompatible shape")
    calibration_arrays = (
        reference_raw,
        reference_context_raw,
        feature_center,
        feature_scale,
        reference_features_z,
    )
    if any(not np.isfinite(array).all() for array in calibration_arrays):
        artifact_errors.append("calibration arrays must contain only finite values")
    if feature_scale.size and np.any(feature_scale <= 0.0):
        artifact_errors.append("feature_scale must be positive")
    if any(not np.isfinite(value) or value < 0.0 for value in thresholds.values()):
        artifact_errors.append("thresholds must be finite and non-negative")
    if artifact_errors:
        return incompatible_bucket_result(bucket, artifact_errors=artifact_errors)

    raw_mae = nearest_distances(raw[None, :], reference_raw, metric="mae")
    raw_l2 = nearest_distances(raw[None, :], reference_raw, metric="l2")
    context_raw_mae = nearest_distances(context_raw[None, :], reference_context_raw, metric="mae")
    context_raw_l2 = nearest_distances(context_raw[None, :], reference_context_raw, metric="l2")

    query_features_z = feature_vector_z(features, feature_names, feature_center, feature_scale)
    feature_l2 = nearest_distances(query_features_z[None, :], reference_features_z, metric="l2")
    full_strict_risk = bool(
        raw_mae["d1"][0] <= thresholds["raw_mae_p01"]
        and raw_l2["d1"][0] <= thresholds["raw_l2_p01"]
    )
    context_strict_risk = bool(
        context_raw_mae["d1"][0] <= thresholds["context_raw_mae_p01"]
        and context_raw_l2["d1"][0] <= thresholds["context_raw_l2_p01"]
    )
    full_combined_risk = bool(
        raw_mae["d1"][0] <= thresholds["raw_mae_p05"]
        and raw_l2["d1"][0] <= thresholds["raw_l2_p05"]
        and (
            feature_l2["d1"][0] <= thresholds["feature_l2_p01"]
            or raw_mae["nndr"][0] <= thresholds["raw_mae_nndr_p01"]
        )
    )
    context_combined_risk = bool(
        context_raw_mae["d1"][0] <= thresholds["context_raw_mae_p05"]
        and context_raw_l2["d1"][0] <= thresholds["context_raw_l2_p05"]
        and context_raw_mae["nndr"][0] <= thresholds["context_raw_mae_nndr_p01"]
    )
    strict_risk = full_strict_risk or context_strict_risk
    combined_risk = full_combined_risk or context_combined_risk
    return {
        "profile_id": bucket["profile_id"],
        "calibration_compatible": True,
        "strict_risk": strict_risk,
        "combined_risk": combined_risk,
        "full_strict_risk": full_strict_risk,
        "context_strict_risk": context_strict_risk,
        "full_combined_risk": full_combined_risk,
        "context_combined_risk": context_combined_risk,
        "raw_mae_d1": round_float(raw_mae["d1"][0]),
        "raw_l2_d1": round_float(raw_l2["d1"][0]),
        "context_raw_mae_d1": round_float(context_raw_mae["d1"][0]),
        "context_raw_l2_d1": round_float(context_raw_l2["d1"][0]),
        "feature_l2_d1": round_float(feature_l2["d1"][0]),
        "raw_mae_nndr": round_float(raw_mae["nndr"][0]),
        "context_raw_mae_nndr": round_float(context_raw_mae["nndr"][0]),
        "thresholds": {
            "raw_mae_p01": round_float(thresholds["raw_mae_p01"]),
            "raw_mae_p05": round_float(thresholds["raw_mae_p05"]),
            "raw_l2_p01": round_float(thresholds["raw_l2_p01"]),
            "raw_l2_p05": round_float(thresholds["raw_l2_p05"]),
            "feature_l2_p01": round_float(thresholds["feature_l2_p01"]),
            "raw_mae_nndr_p01": round_float(thresholds["raw_mae_nndr_p01"]),
            "context_raw_mae_p01": round_float(thresholds["context_raw_mae_p01"]),
            "context_raw_mae_p05": round_float(thresholds["context_raw_mae_p05"]),
            "context_raw_l2_p01": round_float(thresholds["context_raw_l2_p01"]),
            "context_raw_l2_p05": round_float(thresholds["context_raw_l2_p05"]),
            "context_raw_mae_nndr_p01": round_float(thresholds["context_raw_mae_nndr_p01"]),
        },
    }


def flatten_raw(target: np.ndarray) -> np.ndarray:
    arr = np.asarray(target, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.reshape(-1)


def feature_vector_z(
    features: dict[str, float],
    feature_names: tuple[str, ...],
    feature_center: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    if not feature_names:
        return np.zeros(1, dtype=float)
    values = np.empty(len(feature_names), dtype=float)
    for index, name in enumerate(feature_names):
        value = features.get(name)
        values[index] = float(value) if _is_finite_number(value) else feature_center[index]
    scale = np.where(feature_scale > 1e-9, feature_scale, 1.0)
    return (values - feature_center) / scale


def nearest_distances(query: np.ndarray, reference: np.ndarray, *, metric: str) -> dict[str, np.ndarray]:
    if reference.shape[0] == 0:
        raise ValueError("reference must contain at least one row")
    diff = query[:, None, :] - reference[None, :, :]
    if metric == "mae":
        distances = np.mean(np.abs(diff), axis=2)
    elif metric == "l2":
        distances = np.sqrt(np.mean(diff * diff, axis=2))
    else:
        raise ValueError(f"unknown metric: {metric}")
    kth = min(1, reference.shape[0] - 1)
    part = np.partition(distances, kth=kth, axis=1)
    d1 = part[:, 0]
    d2 = part[:, 1] if reference.shape[0] > 1 else part[:, 0]
    return {"d1": d1, "d2": d2, "nndr": d1 / np.maximum(d2, 1e-9)}


def round_float(value: float, digits: int = 8) -> float:
    return float(round(float(value), digits))


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def incompatible_bucket_result(bucket: dict[str, Any], **details: Any) -> dict[str, Any]:
    return {
        "profile_id": bucket.get("profile_id"),
        "calibration_compatible": False,
        "strict_risk": False,
        "combined_risk": False,
        "missing_artifact_keys": [],
        "missing_feature_names": [],
        **details,
    }
