from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_v2_near_distance_artifact.json"
SCHEMA_VERSION = "synthetic_near_distance_gate.v1"


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

    target_array = np.asarray(target, dtype=float)
    if target_array.ndim == 1:
        target_array = target_array[:, None]
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
    if not source:
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
    raw = flatten_raw(target)
    reference_raw = np.asarray(bucket["reference_raw"], dtype=float)
    raw_mae = nearest_distances(raw[None, :], reference_raw, metric="mae")
    raw_l2 = nearest_distances(raw[None, :], reference_raw, metric="l2")

    feature_names = tuple(str(name) for name in bucket.get("feature_names", ()))
    feature_center = np.asarray(bucket.get("feature_center", []), dtype=float)
    feature_scale = np.asarray(bucket.get("feature_scale", []), dtype=float)
    query_features_z = feature_vector_z(features, feature_names, feature_center, feature_scale)
    reference_features_z = np.asarray(bucket["reference_features_z"], dtype=float)
    feature_l2 = nearest_distances(query_features_z[None, :], reference_features_z, metric="l2")

    thresholds = bucket["thresholds"]
    strict_risk = bool(raw_mae["d1"][0] <= thresholds["raw_mae_p01"] and raw_l2["d1"][0] <= thresholds["raw_l2_p01"])
    combined_risk = bool(
        raw_mae["d1"][0] <= thresholds["raw_mae_p05"]
        and raw_l2["d1"][0] <= thresholds["raw_l2_p05"]
        and (
            feature_l2["d1"][0] <= thresholds["feature_l2_p01"]
            or raw_mae["nndr"][0] <= thresholds["raw_mae_nndr_p01"]
        )
    )
    return {
        "profile_id": bucket["profile_id"],
        "strict_risk": strict_risk,
        "combined_risk": combined_risk,
        "raw_mae_d1": round_float(raw_mae["d1"][0]),
        "raw_l2_d1": round_float(raw_l2["d1"][0]),
        "feature_l2_d1": round_float(feature_l2["d1"][0]),
        "raw_mae_nndr": round_float(raw_mae["nndr"][0]),
        "thresholds": {
            "raw_mae_p01": round_float(thresholds["raw_mae_p01"]),
            "raw_mae_p05": round_float(thresholds["raw_mae_p05"]),
            "raw_l2_p01": round_float(thresholds["raw_l2_p01"]),
            "raw_l2_p05": round_float(thresholds["raw_l2_p05"]),
            "feature_l2_p01": round_float(thresholds["feature_l2_p01"]),
            "raw_mae_nndr_p01": round_float(thresholds["raw_mae_nndr_p01"]),
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
        values[index] = float(value) if value is not None and np.isfinite(value) else feature_center[index]
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
