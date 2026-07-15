from __future__ import annotations

import numpy as np

from app.services.synthetic_feature_gate import evaluate_feature_support_gate


def artifact_with_joint_support() -> dict:
    capability = {
        "control_support": {
            "method": "shrunk_robust_mahalanobis",
            "feature_names": ["noise_ratio", "spike_rate"],
            "feature_center": [0.0, 0.0],
            "feature_scale": [1.0, 1.0],
            "robust_location_z": [0.0, 0.0],
            "precision": [[1.0, 0.8], [0.8, 1.0]],
            "threshold": 0.75,
            "coverage": 0.99,
            "reference_count": 80,
            "calibration_count": 20,
            "marginal_quantiles": {
                "noise_ratio": {"p01": -1.0, "p50": 0.0, "p99": 1.0},
                "spike_rate": {"p01": -1.0, "p50": 0.0, "p99": 1.0},
            },
        },
        "target_reference": {
            "trend_strength": {
                "direction": "increase",
                "quantiles": {
                    "p01": 0.0,
                    "p10": 0.1,
                    "p50": 0.5,
                    "p90": 0.9,
                    "p99": 1.0,
                },
            }
        },
    }
    return {
        "schema_version": "synthetic_v2_feature_gate_online.v1",
        "buckets": {
            "unit_bucket": {
                "profile_id": "unit_bucket",
                "context_length": 168,
                "horizon": 24,
                "target_dim": 1,
                "capabilities": {"trend": capability},
            }
        },
    }


def test_joint_support_rejects_a_combination_that_passes_marginal_bounds():
    result = evaluate_feature_support_gate(
        capability_id="trend",
        features={"noise_ratio": 0.7, "spike_rate": 0.7, "trend_strength": 0.6},
        profile_ids=("unit_bucket",),
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact_with_joint_support(),
    )

    assert result["enforced"] is True
    assert result["accepted"] is False
    assert result["status"] == "outside_real_control_support"
    assert result["score"] > result["threshold"]
    assert result["matched_profile_id"] == "unit_bucket"
    assert 0.5 < result["target_percentile_diagnostics"]["trend_strength"]["approx_real_percentile"] < 0.9


def test_joint_support_accepts_an_in_support_vector_and_fails_closed_on_missing_features():
    artifact = artifact_with_joint_support()
    accepted = evaluate_feature_support_gate(
        capability_id="trend",
        features={"noise_ratio": 0.1, "spike_rate": -0.1, "trend_strength": 0.2},
        profile_ids=("unit_bucket",),
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    )
    missing = evaluate_feature_support_gate(
        capability_id="trend",
        features={"noise_ratio": 0.1, "trend_strength": 0.2},
        profile_ids=("unit_bucket",),
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    )

    assert accepted["accepted"] is True
    assert accepted["normalized_score"] < 1.0
    assert missing["accepted"] is False
    assert missing["failed_features"] == ["spike_rate"]


def test_feature_gate_requires_an_exact_calibrated_bucket():
    result = evaluate_feature_support_gate(
        capability_id="trend",
        features={"noise_ratio": 0.0, "spike_rate": 0.0},
        profile_ids=("unit_bucket",),
        context_length=2048,
        horizon=24,
        target_dim=1,
        artifact=artifact_with_joint_support(),
    )

    assert result["accepted"] is False
    assert result["enforced"] is False
    assert result["status"] == "no_matching_calibrated_bucket"


def test_target_percentile_is_diagnostic_not_an_acceptance_condition():
    result = evaluate_feature_support_gate(
        capability_id="trend",
        features={"noise_ratio": 0.0, "spike_rate": 0.0, "trend_strength": 10.0},
        profile_ids=("unit_bucket",),
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact_with_joint_support(),
    )

    assert result["accepted"] is True
    assert np.isclose(
        result["target_percentile_diagnostics"]["trend_strength"]["approx_real_percentile"],
        1.0,
    )
