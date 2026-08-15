"""Frozen v5 scientific decisions for the real-anchored benchmark track."""

from __future__ import annotations


REAL_ANCHORED_PROTOCOL_SCHEMA = "cafe.real_anchored_protocol.v5"
REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA = (
    "cafe.real_anchored_qualification_policy.v3"
)

REAL_ANCHORED_CANONICAL_STRENGTH_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM = 0.10
REAL_ANCHORED_ADDITIVE_HISTORY_TARGET_MAXIMUM = 0.50
REAL_ANCHORED_ADDITIVE_FUTURE_TARGET_MAXIMUM = 0.15
REAL_ANCHORED_NONLINEAR_HISTORY_TARGET_MAXIMUM = 0.20
REAL_ANCHORED_NONLINEAR_FUTURE_TARGET_MAXIMUM = 0.10
REAL_ANCHORED_REFERENCE_DOSE_QUANTILE = 0.75
REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION = 1.00
REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION = 1.0
REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION = 1.0
REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION = 1.5
REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT = 3
REAL_ANCHORED_ADDITIVE_MAXIMUM_GAIN = 20.0
REAL_ANCHORED_ADDITIVE_MAXIMUM_ALPHA = 21.0
REAL_ANCHORED_NONLINEAR_MAXIMUM_ALPHA = 3.0
REAL_ANCHORED_NONLINEAR_REFERENCE_ALPHA_STEP = 0.005

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
REAL_ANCHORED_NONLINEAR_DOSE_CAPABILITIES = (
    "nonlinear_persistence",
)
REAL_ANCHORED_ADDITIVE_DOSE_CAPABILITIES = tuple(
    capability_id
    for capability_id in (
        *REAL_ANCHORED_FORMAL_CAPABILITIES,
        *REAL_ANCHORED_QUALIFICATION_ONLY_CAPABILITIES,
    )
    if capability_id not in REAL_ANCHORED_NONLINEAR_DOSE_CAPABILITIES
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
        "dose_design": {
            "policy": (
                "reference_frozen_contract_specific_treatment_source_"
                "distance_v2"
            ),
            "strength_grid": list(REAL_ANCHORED_CANONICAL_STRENGTH_GRID),
            "additive_history_target_maximum": (
                REAL_ANCHORED_ADDITIVE_HISTORY_TARGET_MAXIMUM
            ),
            "additive_future_target_maximum": (
                REAL_ANCHORED_ADDITIVE_FUTURE_TARGET_MAXIMUM
            ),
            "nonlinear_history_target_maximum": (
                REAL_ANCHORED_NONLINEAR_HISTORY_TARGET_MAXIMUM
            ),
            "nonlinear_future_target_maximum": (
                REAL_ANCHORED_NONLINEAR_FUTURE_TARGET_MAXIMUM
            ),
            "reference_quantile": REAL_ANCHORED_REFERENCE_DOSE_QUANTILE,
            "minimum_reference_evidence_count": (
                REAL_ANCHORED_MINIMUM_REFERENCE_DOSE_EVIDENCE_COUNT
            ),
            "treatment_source_distance_minimum": (
                REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
            ),
            "minimum_separation_acceptance_fraction": (
                REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
            ),
            "maximum_history_macro_separation": (
                REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION
            ),
            "maximum_future_macro_separation": (
                REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION
            ),
            "maximum_affected_channel_separation": (
                REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION
            ),
            "additive_maximum_gain": (
                REAL_ANCHORED_ADDITIVE_MAXIMUM_GAIN
            ),
            "additive_maximum_alpha": (
                REAL_ANCHORED_ADDITIVE_MAXIMUM_ALPHA
            ),
            "nonlinear_maximum_alpha": (
                REAL_ANCHORED_NONLINEAR_MAXIMUM_ALPHA
            ),
            "nonlinear_reference_alpha_step": (
                REAL_ANCHORED_NONLINEAR_REFERENCE_ALPHA_STEP
            ),
            "alpha_semantics": (
                "controlled_component_multiplier_not_cross_capability_strength"
            ),
            "canonical_strength_semantics": (
                "cross_capability_dose_index_with_frozen_standardized_targets"
            ),
            "real_anchored_anti_copy_policy": (
                "treatment_only_source_distance_baseline_exact_and_exempt"
            ),
            "real_anchored_proximity_policy": (
                "fixed_l168_treatment_vs_own_source_hard_gate"
            ),
        },
    }
