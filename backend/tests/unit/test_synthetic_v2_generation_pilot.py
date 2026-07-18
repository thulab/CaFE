from __future__ import annotations

import numpy as np

from app.services.synthetic_generation_service import (
    ACCEPTANCE_PROFILE_BY_CAPABILITY,
    CAPABILITIES_BY_ID,
    CONTROL_FEATURES_BY_CAPABILITY,
    INTENSITY_FEATURE_DIRECTIONS,
    _generate_accepted_sample_values,
    _resolve_seasonality,
    _seed_for,
)
from app.services.synthetic_feature_gate import load_feature_gate_artifact


def test_single_period_noise_ratio_is_not_a_control_for_modulated_seasonality():
    for capability_id in ("multi_seasonal", "time_varying_seasonality"):
        controls = CONTROL_FEATURES_BY_CAPABILITY[capability_id]
        assert "noise_ratio" not in controls
        assert "outlier_rate" in controls


def test_trend_pilot_features_are_monotonic_by_intensity_and_inside_joint_support():
    summaries = []
    for intensity in range(1, 6):
        generated = [
            _generate_accepted_sample_values(
                "trend",
                192,
                168,
                1,
                24,
                intensity,
                _seed_for(321, "trend", sample_index),
            )
            for sample_index in range(96)
        ]
        rows = [item[3] for item in generated]
        gates = [item[1]["acceptance"]["validation"]["feature_gate"] for item in generated]
        summaries.append(
            {
                "trend_strength": float(np.mean([row["trend_strength"] for row in rows])),
                "slope_abs": float(np.mean([row["slope_abs"] for row in rows])),
                "curvature_abs": float(np.mean([row["curvature_abs"] for row in rows])),
                "max_slope_abs": float(np.max([row["slope_abs"] for row in rows])),
                "max_noise_ratio": float(np.max([row["noise_ratio"] for row in rows])),
            }
        )
        assert all(gate["accepted"] for gate in gates)
        assert all(gate["score"] <= gate["threshold"] for gate in gates)

    for feature in INTENSITY_FEATURE_DIRECTIONS["trend"]:
        values = [summary[feature] for summary in summaries]
        assert values == sorted(values)
    assert "curvature_abs" not in INTENSITY_FEATURE_DIRECTIONS["trend"]


def test_multi_seasonal_canonical_intensity_has_paired_dose_response():
    seasonal_naive_mae = []
    multi_period_scores = []
    for intensity in range(1, 6):
        errors = []
        scores = []
        for sample_index in range(96):
            values, latent_params, _, features = _generate_accepted_sample_values(
                "multi_seasonal",
                192,
                168,
                1,
                24,
                intensity,
                _seed_for(321, "multi_seasonal", sample_index),
            )
            history = values[:168, 0]
            actual = values[168:, 0]
            errors.append(float(np.mean(np.abs(actual - history[-24:]))))
            scores.append(float(features["multi_period_score"]))
            assert latent_params["intensity"] == intensity
            assert latent_params["acceptance"]["accepted"] is True
            assert latent_params["acceptance"]["validation"]["schema_version"] == "synthetic_post_generation_validation.v4"
            assert "multi_period_score" in latent_params["acceptance"]["validation"]["target_features"]
            assert latent_params["acceptance"]["validation"]["feature_gate"]["accepted"] is True
            assert latent_params["acceptance"]["validation"]["near_distance_gate"]["accepted"] is True
            assert latent_params["acceptance"]["validation"]["predictability_gate"]["accepted"] is True
            assert latent_params["acceptance"]["validation"]["novelty_check"] == "online_dcr_nndr_gate"
            feature_gate = latent_params["acceptance"]["validation"]["feature_gate"]
            assert feature_gate["score"] <= feature_gate["threshold"]
            assert set(feature_gate["control_features"]) == set(CONTROL_FEATURES_BY_CAPABILITY["multi_seasonal"])
        seasonal_naive_mae.append(float(np.mean(errors)))
        multi_period_scores.append(float(np.mean(scores)))

    assert multi_period_scores == sorted(multi_period_scores)
    assert multi_period_scores[-1] > multi_period_scores[0]
    assert seasonal_naive_mae[-1] > seasonal_naive_mae[0]


def test_all_capabilities_have_real_only_joint_support_calibrations():
    artifact = load_feature_gate_artifact()
    assert artifact is not None
    calibrated_capabilities = {
        capability_id
        for bucket in artifact["buckets"].values()
        for capability_id in bucket["capabilities"]
    }
    assert calibrated_capabilities == set(CAPABILITIES_BY_ID)
    assert artifact["config"]["coverage"] == 0.95
    assert artifact["config"]["target_features"].startswith("diagnostic")
    assert ACCEPTANCE_PROFILE_BY_CAPABILITY["covariate_response"] == "known_future_covariate_envelope_v1"
    assert ACCEPTANCE_PROFILE_BY_CAPABILITY["hierarchical_coherence"] == "m5_hierarchy_envelope_365ctx_28h"


def test_profile_resolved_seasonality_ignores_requested_season_length():
    trend = _resolve_seasonality("trend", requested_frequency="h", seed=1)
    hierarchy = _resolve_seasonality("hierarchical_coherence", requested_frequency="h", seed=1)
    covariate_hourly = _resolve_seasonality("covariate_response", requested_frequency="h", seed=1)
    covariate_daily = _resolve_seasonality("covariate_response", requested_frequency="d", seed=1)
    covariate_unclear = _resolve_seasonality("covariate_response", requested_frequency="15min", seed=1)

    assert trend.season_length == 24
    assert trend.source == "profile_bucket"
    assert hierarchy.season_length == 7
    assert hierarchy.source == "profile_bucket"
    assert covariate_hourly.season_length == 24
    assert covariate_daily.season_length == 7
    assert covariate_unclear.source == "significant_period_sample"
    assert set(covariate_unclear.candidate_periods) == {7, 24}


def test_all_capabilities_return_accepted_samples_after_resampling():
    for capability_id, capability in CAPABILITIES_BY_ID.items():
        target_dim = 3 if capability.target_dim_mode == "multi" else 1
        context_length = 365 if capability_id == "hierarchical_coherence" else 168
        horizon = 28 if capability_id == "hierarchical_coherence" else 24
        season_length = 7 if capability_id == "hierarchical_coherence" else 24

        _, latent_params, _, features = _generate_accepted_sample_values(
            capability_id,
            context_length + horizon,
            context_length,
            target_dim,
            season_length,
            3,
            _seed_for(123, capability_id, 0),
        )

        acceptance = latent_params["acceptance"]
        assert acceptance["accepted"] is True
        assert acceptance["profile"] is not None
        assert acceptance["profile_selection_stage"] == "pre_generation"
        assert 1 <= acceptance["attempts"] <= 32
        assert acceptance["failed_gates"] == []
        assert not acceptance["failed_features"]
        feature_gate = acceptance["validation"]["feature_gate"]
        assert feature_gate["accepted"] is True
        assert feature_gate["enforced"] is True
        assert feature_gate["matched_profile_id"] == acceptance["profile"]
        assert feature_gate["score"] <= feature_gate["threshold"]
        assert acceptance["validation"]["near_distance_gate"]["accepted"] is True
        assert set(CONTROL_FEATURES_BY_CAPABILITY[capability_id]).issubset(features)
