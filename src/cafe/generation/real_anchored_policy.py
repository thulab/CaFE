"""Frozen v3 scientific decisions for the real-anchored benchmark track."""

from __future__ import annotations


REAL_ANCHORED_PROTOCOL_SCHEMA = "cafe.real_anchored_protocol.v3"
REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA = (
    "cafe.real_anchored_qualification_policy.v1"
)

REAL_ANCHORED_FORMAL_CAPABILITIES = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
    "common_factor",
    "cross_series_dependence",
    "covariate_response",
)
REAL_ANCHORED_QUALIFICATION_ONLY_CAPABILITIES = (
    "hierarchical_coherence",
)

TIME_VARYING_SEASONALITY_BASIS_POLICY = (
    "carrier_phase_locked_symmetric_constrained_am_v1"
)
NONLINEAR_FUTURE_INNOVATION_MAIN_POLICY = (
    "zero_future_innovation_paired_rollout_v1"
)
NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY = (
    "history_residual_replay_qualification_only_v1"
)
HIERARCHY_FORMAL_RANK_POLICY = (
    "qualification_only_until_nonnegative_raw_support_policy_is_frozen"
)
MINIMUM_FORMAL_PANEL_DIMENSION = 3
MINIMUM_FORMAL_BACKGROUND_COUNT = 4
MINIMUM_TWO_CHANNEL_SENSITIVITY_BACKGROUND_COUNT = 2
TWO_CHANNEL_PANEL_POLICY = "sensitivity_only_never_formal_rank"
STRUCTURAL_INPUT_ABLATION_CAPABILITIES = (
    "common_factor",
    "cross_series_dependence",
)
STRUCTURAL_INPUT_ABLATION_POLICY = (
    "mandatory_attribution_component_reported_separately_not_score_weighted_v1"
)
QUALIFICATION_THRESHOLD_SOURCE_POLICY = (
    "independent_source_time_disjoint_reference_bank_never_evaluation_origins_v1"
)


def protocol_decisions() -> dict[str, object]:
    """Return the complete JSON-safe decision payload frozen in contracts."""

    return {
        "schema_version": REAL_ANCHORED_PROTOCOL_SCHEMA,
        "formal_capabilities": list(REAL_ANCHORED_FORMAL_CAPABILITIES),
        "qualification_only_capabilities": list(
            REAL_ANCHORED_QUALIFICATION_ONLY_CAPABILITIES
        ),
        "time_varying_seasonality_basis": (
            TIME_VARYING_SEASONALITY_BASIS_POLICY
        ),
        "nonlinear_future_innovation_main": (
            NONLINEAR_FUTURE_INNOVATION_MAIN_POLICY
        ),
        "nonlinear_future_innovation_sensitivity": (
            NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
        ),
        "hierarchy_formal_rank": HIERARCHY_FORMAL_RANK_POLICY,
        "minimum_formal_panel_dimension": MINIMUM_FORMAL_PANEL_DIMENSION,
        "minimum_formal_background_count": MINIMUM_FORMAL_BACKGROUND_COUNT,
        "minimum_two_channel_sensitivity_background_count": (
            MINIMUM_TWO_CHANNEL_SENSITIVITY_BACKGROUND_COUNT
        ),
        "two_channel_panel_policy": TWO_CHANNEL_PANEL_POLICY,
        "structural_input_ablation_capabilities": list(
            STRUCTURAL_INPUT_ABLATION_CAPABILITIES
        ),
        "structural_input_ablation": STRUCTURAL_INPUT_ABLATION_POLICY,
        "qualification_threshold_source": (
            QUALIFICATION_THRESHOLD_SOURCE_POLICY
        ),
    }
