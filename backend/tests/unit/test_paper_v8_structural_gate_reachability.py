from __future__ import annotations

import numpy as np

import app.services.synthetic_v8_feature_gate as feature_gate


def test_hierarchy_calibration_gate_requires_exact_parent_sum() -> None:
    children = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 24),
            np.linspace(0.5, -0.5, 24),
        ]
    )
    coherent = np.column_stack([np.sum(children, axis=1), children])
    broken = coherent.copy()
    broken[-1, 0] += 0.01

    accepted = feature_gate.structural_calibration_member_gate(
        "hierarchical_coherence",
        coherent,
        context_length=16,
    )
    rejected = feature_gate.structural_calibration_member_gate(
        "hierarchical_coherence",
        broken,
        context_length=16,
    )

    assert accepted["accepted"] is True
    assert accepted["near_distance_evaluated"] is False
    assert accepted["gate_scope"] == "generator_structural_hard_gate_only"
    assert rejected["accepted"] is False


def test_covariate_calibration_gate_requires_valid_counterfactual_pair() -> None:
    context = 16
    length = 24
    history = np.linspace(-0.5, 0.5, context)
    first_target = np.concatenate([history, np.zeros(length - context)])
    second_target = np.concatenate([history, np.ones(length - context)])
    shared_past = np.linspace(-1.0, 1.0, context)
    first_covariate = np.concatenate(
        [shared_past, np.zeros(length - context)]
    )
    second_covariate = np.concatenate(
        [shared_past, np.ones(length - context)]
    )

    result = feature_gate.structural_calibration_member_gate(
        "covariate_response",
        first_target[:, None],
        context_length=context,
        second_target=second_target[:, None],
        first_covariates=first_covariate[:, None],
        second_covariates=second_covariate[:, None],
    )

    assert result["accepted"] is True
    assert result["target_history_max_abs_difference"] == 0.0
    assert result["past_covariate_max_abs_difference"] == 0.0
    assert result["future_covariate_max_abs_difference"] == 1.0
    assert result["target_future_max_abs_difference"] == 1.0


def test_counterfactual_calibration_gate_fails_closed_when_member_missing() -> None:
    result = feature_gate.structural_calibration_member_gate(
        "common_factor",
        np.zeros((24, 3)),
        context_length=16,
        metadata={},
    )

    assert result["accepted"] is False
    assert result["reason"] == "malformed_structural_calibration_member"
    assert result["near_distance_evaluated"] is False


def test_common_factor_calibration_gate_delegates_to_formal_gate(
    monkeypatch,
) -> None:
    captured = {}

    def fake_gate(
        first_target,
        second_target,
        *,
        context_length,
        metadata,
        enforced,
    ):
        captured.update(
            {
                "first_shape": first_target.shape,
                "second_shape": second_target.shape,
                "context_length": context_length,
                "metadata": metadata,
                "enforced": enforced,
            }
        )
        return {
            "schema_version": "fake_common_gate.v1",
            "accepted": True,
            "enforced": enforced,
        }

    monkeypatch.setattr(
        feature_gate,
        "common_factor_identifiability_gate",
        fake_gate,
    )
    result = feature_gate.structural_calibration_member_gate(
        "common_factor",
        np.zeros((24, 3)),
        context_length=16,
        metadata={"protected_target_index": 1},
        second_target=np.ones((24, 3)),
    )

    assert result["accepted"] is True
    assert result["underlying_gate_schema_version"] == "fake_common_gate.v1"
    assert captured == {
        "first_shape": (24, 3),
        "second_shape": (24, 3),
        "context_length": 16,
        "metadata": {"protected_target_index": 1},
        "enforced": True,
    }


def _path_gate(capability_id: str, accepted: bool) -> dict[str, object]:
    return {
        "capability_id": capability_id,
        "calibration_reachability": True,
        "near_distance_evaluated": False,
        "accepted": accepted,
    }


def test_structural_reachability_requires_every_path() -> None:
    all_passing = feature_gate.summarize_structural_calibration_reachability(
        "cross_series_dependence",
        family_role="primary",
        lambda_value=0.72,
        path_gates=[
            _path_gate("cross_series_dependence", accepted=True)
            for index in range(5)
        ],
        expected_path_count=5,
    )
    one_failing = feature_gate.summarize_structural_calibration_reachability(
        "cross_series_dependence",
        family_role="primary",
        lambda_value=0.72,
        path_gates=[
            _path_gate("cross_series_dependence", accepted=index < 4)
            for index in range(5)
        ],
        expected_path_count=5,
    )

    assert all_passing["accepted"] is True
    assert all_passing["minimum_pass_fraction"] == 1.0
    assert all_passing["required_pass_count"] == 5
    assert one_failing["accepted"] is False
    assert one_failing["reason_codes"] == [
        "selected_i5_structural_gate_unreachable"
    ]


def test_exact_structural_reachability_fails_closed_on_missing_or_failed_path() -> None:
    result = feature_gate.summarize_structural_calibration_reachability(
        "hierarchical_coherence",
        family_role="secondary",
        lambda_value=0.65,
        path_gates=[
            _path_gate("hierarchical_coherence", True),
            _path_gate("hierarchical_coherence", False),
        ],
        expected_path_count=3,
    )

    assert result["accepted"] is False
    assert result["minimum_pass_fraction"] == 1.0
    assert result["missing_path_count"] == 1
    assert result["reason_codes"] == [
        "structural_gate_qualification_paths_missing",
        "selected_i5_structural_gate_unreachable",
    ]
    assert result["near_distance_evaluated"] is False
