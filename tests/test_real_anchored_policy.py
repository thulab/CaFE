from cafe.generation.real_anchored_policy import (
    MINIMUM_FORMAL_PANEL_DIMENSION,
    REAL_ANCHORED_FORMAL_CAPABILITIES,
    REAL_ANCHORED_QUALIFICATION_ONLY_CAPABILITIES,
    STRUCTURAL_INPUT_ABLATION_CAPABILITIES,
    protocol_decisions,
)


def test_v3_decisions_freeze_user_selected_protocol() -> None:
    decisions = protocol_decisions()

    assert decisions["time_varying_seasonality_basis"].endswith(
        "constrained_am_v1"
    )
    assert decisions["nonlinear_future_innovation_main"].startswith(
        "zero_future_innovation"
    )
    assert REAL_ANCHORED_QUALIFICATION_ONLY_CAPABILITIES == (
        "hierarchical_coherence",
    )
    assert "hierarchical_coherence" not in REAL_ANCHORED_FORMAL_CAPABILITIES
    assert MINIMUM_FORMAL_PANEL_DIMENSION == 3
    assert STRUCTURAL_INPUT_ABLATION_CAPABILITIES == (
        "common_factor",
        "cross_series_dependence",
    )
    assert "not_score_weighted" in decisions["structural_input_ablation"]
    assert "never_evaluation_origins" in decisions[
        "qualification_threshold_source"
    ]
