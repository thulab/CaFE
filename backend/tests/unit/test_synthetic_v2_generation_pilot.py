from __future__ import annotations

import numpy as np

from app.services.synthetic_generation_service import (
    ACCEPTANCE_PROFILE_BY_CAPABILITY,
    CAPABILITIES_BY_ID,
    PILOT_ACCEPTANCE_CAPS,
    _accept_synthetic_features,
    _generate_accepted_sample_values,
    _resolve_seasonality,
    _seed_for,
)


def test_trend_pilot_features_are_monotonic_by_intensity_and_capped():
    summaries = []
    for intensity in range(1, 6):
        rows = [
            _generate_accepted_sample_values(
                "trend",
                192,
                168,
                1,
                24,
                intensity,
                _seed_for(321, "trend", intensity * 1000 + sample_index),
            )[3]
            for sample_index in range(96)
        ]
        summaries.append(
            {
                "trend_strength": float(np.mean([row["trend_strength"] for row in rows])),
                "slope_abs": float(np.mean([row["slope_abs"] for row in rows])),
                "curvature_abs": float(np.mean([row["curvature_abs"] for row in rows])),
                "max_slope_abs": float(np.max([row["slope_abs"] for row in rows])),
                "max_noise_ratio": float(np.max([row["noise_ratio"] for row in rows])),
            }
        )

    for feature in ("trend_strength", "slope_abs", "curvature_abs"):
        values = [summary[feature] for summary in summaries]
        assert values == sorted(values)
    assert summaries[-1]["max_slope_abs"] <= PILOT_ACCEPTANCE_CAPS["trend"]["slope_abs"] + 1e-6
    assert summaries[-1]["max_noise_ratio"] <= PILOT_ACCEPTANCE_CAPS["trend"]["noise_ratio"]


def test_multi_seasonal_intensity_degrades_single_period_seasonal_naive():
    seasonal_naive_mae = []
    for intensity in range(1, 6):
        errors = []
        for sample_index in range(96):
            values, latent_params, _, features = _generate_accepted_sample_values(
                "multi_seasonal",
                192,
                168,
                1,
                24,
                intensity,
                _seed_for(321, "multi_seasonal", intensity * 1000 + sample_index),
            )
            history = values[:168, 0]
            actual = values[168:, 0]
            errors.append(float(np.mean(np.abs(actual - history[-24:]))))
            assert latent_params["intensity"] == intensity
            assert latent_params["acceptance"]["accepted"] is True
            assert latent_params["acceptance"]["validation"]["schema_version"] == "synthetic_post_generation_validation.v1"
            assert "multi_period_score" in latent_params["acceptance"]["validation"]["target_features"]
            assert latent_params["acceptance"]["validation"]["feature_gate"]["accepted"] is True
            assert latent_params["acceptance"]["validation"]["near_distance_gate"]["accepted"] is True
            assert latent_params["acceptance"]["validation"]["novelty_check"] == "online_dcr_nndr_gate"
            assert features["noise_ratio"] <= PILOT_ACCEPTANCE_CAPS["multi_seasonal"]["noise_ratio"]
        seasonal_naive_mae.append(float(np.mean(errors)))

    assert seasonal_naive_mae == sorted(seasonal_naive_mae)
    assert seasonal_naive_mae[-1] > seasonal_naive_mae[0] * 4


def test_all_capabilities_have_hard_acceptance_rules():
    assert set(PILOT_ACCEPTANCE_CAPS) == set(CAPABILITIES_BY_ID)
    assert "change_point_shift_energy" in PILOT_ACCEPTANCE_CAPS["regime_switching"]
    assert "pca_top1_explained" in PILOT_ACCEPTANCE_CAPS["common_factor"]
    assert "event_lift_abs" in PILOT_ACCEPTANCE_CAPS["covariate_response"]
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


def test_hard_acceptance_rejects_new_capability_feature_over_cap():
    cap = PILOT_ACCEPTANCE_CAPS["regime_switching"]["level_shift_strength"]

    accepted, failed = _accept_synthetic_features("regime_switching", {"level_shift_strength": cap + 0.01})

    assert accepted is False
    assert failed == ["level_shift_strength"]


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
        assert acceptance["failed_gates"] == []
        assert not acceptance["failed_features"]
        assert acceptance["validation"]["feature_gate"]["accepted"] is True
        assert acceptance["validation"]["near_distance_gate"]["accepted"] is True
        assert set(PILOT_ACCEPTANCE_CAPS[capability_id]).intersection(features)
