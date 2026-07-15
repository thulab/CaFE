from __future__ import annotations

import numpy as np

from app.services.synthetic_near_distance_gate import evaluate_near_distance_gate


def artifact_with_reference() -> dict:
    return {
        "schema_version": "synthetic_v2_near_distance_online.v2",
        "buckets": {
            "unit_bucket": {
                "profile_id": "unit_bucket",
                "context_length": 2,
                "horizon": 0,
                "target_dim": 1,
                "feature_names": ["trend_strength"],
                "feature_center": [0.0],
                "feature_scale": [1.0],
                "reference_raw": [[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]],
                "reference_context_raw": [[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]],
                "reference_features_z": [[0.0], [2.0], [10.0]],
                "thresholds": {
                    "raw_mae_p01": 0.01,
                    "raw_mae_p05": 0.05,
                    "raw_l2_p01": 0.01,
                    "raw_l2_p05": 0.05,
                    "feature_l2_p01": 0.01,
                    "feature_l2_p05": 0.05,
                    "raw_mae_nndr_p01": 0.05,
                    "raw_mae_nndr_p05": 0.10,
                    "context_raw_mae_p01": 0.01,
                    "context_raw_mae_p05": 0.05,
                    "context_raw_l2_p01": 0.01,
                    "context_raw_l2_p05": 0.05,
                    "context_raw_mae_nndr_p01": 0.05,
                    "context_raw_mae_nndr_p05": 0.10,
                },
            }
        },
    }


def test_near_distance_gate_rejects_exact_reference_copy():
    result = evaluate_near_distance_gate(
        target=np.asarray([[2.0], [2.0]]),
        features={"trend_strength": 2.0},
        profile_ids=("unit_bucket",),
        context_length=2,
        horizon=0,
        artifact=artifact_with_reference(),
    )

    assert result["enforced"] is True
    assert result["accepted"] is False
    assert result["strict_risk"] is True
    assert result["combined_risk"] is True


def test_near_distance_gate_passes_far_sample_and_fails_missing_bucket():
    artifact = artifact_with_reference()
    far = evaluate_near_distance_gate(
        target=np.asarray([[50.0], [50.0]]),
        features={"trend_strength": 50.0},
        profile_ids=("unit_bucket",),
        context_length=2,
        horizon=0,
        artifact=artifact,
    )
    missing = evaluate_near_distance_gate(
        target=np.asarray([[50.0], [50.0]]),
        features={"trend_strength": 50.0},
        profile_ids=("missing",),
        context_length=2,
        horizon=0,
        artifact=artifact,
    )

    assert far["accepted"] is True
    assert far["strict_risk"] is False
    assert missing["accepted"] is False
    assert missing["enforced"] is False
    assert missing["status"] == "no_matching_calibrated_bucket"


def test_near_distance_gate_rejects_exact_context_with_replaced_future():
    artifact = artifact_with_reference()
    bucket = artifact["buckets"]["unit_bucket"]
    bucket["horizon"] = 2
    bucket["reference_raw"] = [
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 2.0],
        [10.0, 10.0, 10.0, 10.0],
    ]

    result = evaluate_near_distance_gate(
        target=np.asarray([[2.0], [2.0], [50.0], [50.0]]),
        features={"trend_strength": 50.0},
        profile_ids=("unit_bucket",),
        context_length=2,
        horizon=2,
        artifact=artifact,
    )

    bucket_result = result["bucket_results"][0]
    assert result["accepted"] is False
    assert bucket_result["full_strict_risk"] is False
    assert bucket_result["context_strict_risk"] is True


def test_near_distance_gate_fails_closed_on_stale_feature_schema():
    result = evaluate_near_distance_gate(
        target=np.asarray([[2.0], [2.0]]),
        features={},
        profile_ids=("unit_bucket",),
        context_length=2,
        horizon=0,
        artifact=artifact_with_reference(),
    )

    assert result["accepted"] is False
    assert result["enforced"] is False
    assert result["status"] == "artifact_schema_mismatch"
    assert result["bucket_results"][0]["missing_feature_names"] == ["trend_strength"]


def test_near_distance_gate_fails_closed_on_wrong_artifact_version():
    artifact = artifact_with_reference()
    artifact["schema_version"] = "synthetic_v2_near_distance_online.v1"

    result = evaluate_near_distance_gate(
        target=np.asarray([[2.0], [2.0]]),
        features={"trend_strength": 2.0},
        profile_ids=("unit_bucket",),
        context_length=2,
        horizon=0,
        artifact=artifact,
    )

    assert result["accepted"] is False
    assert result["enforced"] is False
    assert result["status"] == "artifact_schema_mismatch"


def test_near_distance_gate_fails_closed_on_non_finite_query():
    result = evaluate_near_distance_gate(
        target=np.asarray([[2.0], [np.nan]]),
        features={"trend_strength": 2.0},
        profile_ids=("unit_bucket",),
        context_length=2,
        horizon=0,
        artifact=artifact_with_reference(),
    )

    assert result["accepted"] is False
    assert result["enforced"] is False
    assert result["status"] == "invalid_query"


def test_near_distance_gate_fails_closed_on_malformed_reference_shape():
    artifact = artifact_with_reference()
    artifact["buckets"]["unit_bucket"]["reference_context_raw"] = [[0.0], [2.0], [10.0]]

    result = evaluate_near_distance_gate(
        target=np.asarray([[2.0], [2.0]]),
        features={"trend_strength": 2.0},
        profile_ids=("unit_bucket",),
        context_length=2,
        horizon=0,
        artifact=artifact,
    )

    assert result["accepted"] is False
    assert result["enforced"] is False
    assert result["status"] == "artifact_schema_mismatch"
    assert "reference_context_raw has an incompatible shape" in result["bucket_results"][0]["artifact_errors"]
