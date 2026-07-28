from __future__ import annotations

import pickle

import numpy as np

from cafe.validation import realism


def anchor_rows(count: int = 12) -> list[dict]:
    return [
        {
            "anchor_id": f"anchor-{index}",
            "features": {"trend_feature": float(index)},
        }
        for index in range(count)
    ]


def real_masters(count: int = 12) -> list[dict]:
    time = np.arange(realism.HISTORY_LENGTH, dtype=float)
    return [
        {
            "sample_id": f"real-{index}",
            "context_length": realism.HISTORY_LENGTH,
            "target_dim": 1,
            "target": (
                np.sin(time / 11.0 + index * 0.21)
                + 0.035 * index * np.cos(time / 7.0)
            )[:, None].tolist(),
        }
        for index in range(count)
    ]


def calibration(scope: str = realism.REAL_FEATURE_SCOPE) -> dict:
    return {
        "schema_version": "test",
        "capabilities": {
            "trend": {
                "target_feature": "trend_feature",
                "intensity_calibration_scope": scope,
            }
        },
    }


def sample_from_history(
    history: np.ndarray,
    *,
    feature_value: float = 5.0,
) -> dict:
    values = np.asarray(history, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    prefix = np.zeros((168, values.shape[1]), dtype=float)
    future = np.zeros((48, values.shape[1]), dtype=float)
    target = np.concatenate([prefix, values, future], axis=0)
    return {
        "sample_id": "synthetic",
        "capability_id": "trend",
        "context_length": 336,
        "horizon": 48,
        "target_dim": values.shape[1],
        "target_feature": "trend_feature",
        "target_feature_value": feature_value,
        "intensity_target_feature_value": feature_value,
        "intensity_lambda": 0.5,
        "target": target.tolist(),
    }


def test_exact_real_anchor_history_fails_loo_dcr_nndr_gate():
    masters = real_masters()
    context = realism.build_realism_gate_context(
        anchor_rows(),
        masters,
        calibration(),
        near_distance_enabled=True,
    )
    exact = np.asarray(masters[3]["target"], dtype=float)[:, 0]
    result = realism.evaluate_sample(sample_from_history(exact), context)

    assert context.near_distance_policy.enforced
    assert result["accepted"] is False
    assert result["near_distance"]["accepted"] is False
    assert result["near_distance"]["risk_channel_indices"] == [0]
    assert result["near_distance"]["channels"][0]["d1"] == 0.0
    assert result["near_distance"]["channels"][0]["nndr"] == 0.0
    assert "near_distance_copy_risk" in result["failure_codes"]


def test_far_history_passes_and_each_channel_is_checked_vectorially():
    context = realism.build_realism_gate_context(
        anchor_rows(),
        real_masters(),
        calibration(),
        near_distance_enabled=True,
    )
    time = np.arange(realism.HISTORY_LENGTH, dtype=float)
    far = np.column_stack(
        [
            50.0 + 0.7 * time,
            -80.0 + 0.4 * time + np.sin(time),
        ]
    )
    result = realism.evaluate_sample(sample_from_history(far), context)

    assert result["accepted"] is True
    assert result["near_distance"]["accepted"] is True
    assert len(result["near_distance"]["channels"]) == 2
    assert not result["near_distance"]["risk_channel_indices"]


def test_multivariate_near_distance_uses_sample_level_majority_vote():
    masters = real_masters()
    context = realism.build_realism_gate_context(
        anchor_rows(),
        masters,
        calibration(),
        near_distance_enabled=True,
    )
    exact = np.asarray(masters[3]["target"], dtype=float)[:, 0]
    time = np.arange(realism.HISTORY_LENGTH, dtype=float)
    one_risky = np.column_stack(
        [exact, 50.0 + time, -80.0 + 0.4 * time]
    )
    all_risky = np.column_stack([exact, exact, exact])

    accepted = realism.evaluate_sample(
        sample_from_history(one_risky),
        context,
    )
    rejected = realism.evaluate_sample(
        sample_from_history(all_risky),
        context,
    )

    assert accepted["near_distance"]["risk_channel_indices"] == [0]
    assert accepted["near_distance"]["accepted"] is True
    assert (
        accepted["near_distance"]["minimum_risk_channels_for_rejection"]
        == 2
    )
    assert rejected["near_distance"]["accepted"] is False
    assert rejected["near_distance"]["risk_channel_indices"] == [0, 1, 2]


def test_feature_support_is_diagnostic_outside_raw_anchor_padding():
    context = realism.build_realism_gate_context(
        anchor_rows(),
        real_masters(),
        calibration(),
        near_distance_enabled=False,
        feature_padding_fraction=0.5,
    )
    policy = context.feature_policies["trend"]

    assert policy.real_minimum == 0.0
    assert policy.real_maximum == 11.0
    assert policy.lower_bound == -5.5
    assert policy.upper_bound == 16.5
    inside = realism.evaluate_sample(
        sample_from_history(np.arange(168), feature_value=16.5),
        context,
    )
    outside = realism.evaluate_sample(
        sample_from_history(np.arange(168), feature_value=16.5001),
        context,
    )
    assert inside["feature_support"]["accepted"] is True
    assert inside["feature_support"]["within_reference_support"] is True
    assert outside["feature_support"]["accepted"] is True
    assert outside["feature_support"]["within_reference_support"] is False
    assert outside["accepted"] is True
    assert outside["failure_codes"] == []


def test_default_feature_support_expands_total_anchor_span_by_1_2():
    context = realism.build_realism_gate_context(
        anchor_rows(),
        real_masters(),
        calibration(),
        near_distance_enabled=False,
    )
    policy = context.feature_policies["trend"]

    assert policy.padding_fraction == 0.1
    assert policy.lower_bound == -1.1
    assert policy.upper_bound == 12.1


def test_disabled_near_gate_is_not_enforced_and_context_is_pickle_safe():
    context = realism.build_realism_gate_context(
        anchor_rows(),
        real_masters(),
        calibration(scope="generator_structural_relative_grid"),
        near_distance_enabled=False,
    )
    restored = pickle.loads(pickle.dumps(context))
    result = realism.evaluate_sample(
        sample_from_history(np.arange(168)),
        restored,
    )

    assert restored.near_distance_policy.normalized_anchor_histories is None
    assert restored.policy_summary["near_distance"]["requested_enabled"] is False
    assert "normalized_anchor_histories" not in restored.policy_summary[
        "near_distance"
    ]
    assert result["feature_support"]["status"] == "not_enforced"
    assert result["near_distance"]["status"] == "not_enforced"
    assert result["near_distance"]["accepted"] is True
    assert result["accepted"] is True


def test_malformed_or_nonfinite_sample_target_hard_fails():
    context = realism.build_realism_gate_context(
        anchor_rows(),
        real_masters(),
        calibration(),
        near_distance_enabled=True,
    )
    malformed = sample_from_history(np.arange(168))
    malformed["target"] = [[0.0]]
    malformed_result = realism.evaluate_sample(malformed, context)
    assert malformed_result["accepted"] is False
    assert "invalid_sample_context_length" in malformed_result[
        "failure_codes"
    ]

    nonfinite = sample_from_history(np.arange(168))
    nonfinite["target"][0][0] = float("nan")
    nonfinite_result = realism.evaluate_sample(nonfinite, context)
    assert nonfinite_result["accepted"] is False
    assert "nonfinite_sample_target" in nonfinite_result["failure_codes"]
