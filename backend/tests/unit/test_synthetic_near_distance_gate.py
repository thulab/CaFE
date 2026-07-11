from __future__ import annotations

import numpy as np

from app.services.synthetic_near_distance_gate import evaluate_near_distance_gate


def artifact_with_reference() -> dict:
    return {
        "schema_version": "synthetic_v2_near_distance_online.v1",
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
