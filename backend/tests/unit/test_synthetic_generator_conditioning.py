from __future__ import annotations

import numpy as np

from app.services.synthetic_generation_service import _generate_sample_values
from app.services.synthetic_generator_conditioning import (
    load_generator_conditioning_artifact,
    matching_generator_profiles,
    resolve_generator_conditioning,
    select_balanced_profile_id,
)


def conditioning_artifact() -> dict:
    capability = {
        "parameters": {"structure_scale": 0.75},
        "intensity_lambdas": [0.0, 0.1, 0.3, 0.6, 1.0],
        "canonical_reference_percentile_levels": [0.1, 0.3, 0.5, 0.7, 0.9],
        "canonical_target_feature": "trend_strength",
        "canonical_target_values": [0.0, 0.1, 0.2, 0.4, 0.7],
        "calibrated_realized_strengths": [0.01, 0.11, 0.21, 0.39, 0.69],
        "local_real_percentiles_at_canonical_targets": [0.05, 0.2, 0.4, 0.75, 0.98],
        "local_real_target_quantiles": {
            "trend_strength": [0.02, 0.12, 0.25, 0.38, 0.55]
        },
        "canonical_calibration": {
            "status": "supported",
            "max_normalized_error": 0.02,
        },
        "calibration_method": "unit-test",
    }
    return {
        "schema_version": "unit-conditioning.v2",
        "created_at": "2026-07-15T00:00:00+00:00",
        "canonical_intensity": {
            "scale_id": "unit-scale-v1",
            "scale_fingerprint": "0123456789abcdef",
            "capabilities": {
                "trend": {
                    "primary_feature": "trend_strength",
                    "target_values": [0.0, 0.1, 0.2, 0.4, 0.7],
                }
            }
        },
        "profiles": {
            "profile_a": {
                "profile_id": "profile_a",
                "context_length": 168,
                "horizon": 24,
                "target_dim": 1,
                "season_length": 24,
                "frequency": "h",
                "nuisance_parameters": {"noise_scale_multiplier": 1.5},
                "capabilities": {"trend": capability},
            },
            "profile_b": {
                "profile_id": "profile_b",
                "context_length": 168,
                "horizon": 24,
                "target_dim": 1,
                "season_length": 24,
                "frequency": "h",
                "nuisance_parameters": {"noise_scale_multiplier": 0.5},
                "capabilities": {"trend": capability},
            },
        },
    }


def test_conditioning_requires_an_exact_task_window_and_merges_parameters():
    artifact = conditioning_artifact()
    matches = matching_generator_profiles(
        capability_id="trend",
        profile_ids=("profile_a", "profile_b"),
        context_length=168,
        horizon=24,
        target_dim=1,
        frequency="hourly",
        artifact=artifact,
    )
    missing = matching_generator_profiles(
        capability_id="trend",
        profile_ids=("profile_a",),
        context_length=168,
        horizon=48,
        target_dim=1,
        artifact=artifact,
    )
    conditioning = resolve_generator_conditioning(
        capability_id="trend",
        profile_id="profile_a",
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    )

    assert [profile["profile_id"] for profile in matches] == ["profile_a", "profile_b"]
    assert missing == []
    assert conditioning is not None
    assert conditioning.parameters == {
        "noise_scale_multiplier": 1.5,
        "structure_scale": 0.75,
    }
    assert conditioning.lambda_for(3) == 0.3
    assert conditioning.metadata(3)["profile_id"] == "profile_a"
    assert conditioning.metadata(3)["canonical_target_strength"] == 0.2
    assert conditioning.metadata(3)["calibrated_profile_median_strength"] == 0.21
    assert conditioning.metadata(3)["local_real_percentile"] == 0.4
    assert conditioning.metadata(3)["canonical_scale_id"] == "unit-scale-v1"
    assert conditioning.metadata(3)["canonical_scale_fingerprint"] == "0123456789abcdef"


def test_balanced_profile_selection_is_deterministic_and_exactly_balanced():
    profile_ids = ("profile_a", "profile_b", "profile_c")
    first = [
        select_balanced_profile_id(
            profile_ids,
            capability_id="trend",
            seed=17,
            sample_index=index,
        )
        for index in range(9)
    ]
    second = [
        select_balanced_profile_id(
            tuple(reversed(profile_ids)),
            capability_id="trend",
            seed=17,
            sample_index=index,
        )
        for index in range(9)
    ]

    assert first == second
    assert {profile_id: first.count(profile_id) for profile_id in profile_ids} == {
        "profile_a": 3,
        "profile_b": 3,
        "profile_c": 3,
    }


def test_profile_conditioned_generators_preserve_horizon_prefixes():
    artifact = load_generator_conditioning_artifact()
    assert artifact is not None
    cases = {
        "trend": "traffic_hourly_daily_168ctx",
        "multi_seasonal": "m4_hourly_daily_168ctx",
        "time_varying_seasonality": "electricity_hourly_daily_168ctx",
        "regime_switching": "traffic_hourly_daily_168ctx",
        "nonlinear_persistence": "traffic_hourly_daily_168ctx",
        "predictable_intermittency": "m4_hourly_daily_168ctx",
        "common_factor": "traffic_hourly_panel_168ctx",
        "hierarchical_coherence": "m5_daily_hierarchy_365ctx_28h",
        "covariate_response": "m5_daily_covariate_365ctx_28h",
    }
    for capability_id, profile_id in cases.items():
        profile = artifact["profiles"][profile_id]
        context_length = int(profile["context_length"])
        horizon = int(profile["horizon"])
        target_dim = int(profile["target_dim"])
        season_length = int(profile["season_length"])
        conditioning = resolve_generator_conditioning(
            capability_id=capability_id,
            profile_id=profile_id,
            context_length=context_length,
            horizon=horizon,
            target_dim=target_dim,
            artifact=artifact,
        )
        assert conditioning is not None
        short, _, short_covariates = _generate_sample_values(
            capability_id,
            context_length + horizon,
            context_length,
            target_dim,
            season_length,
            3,
            np.random.default_rng(91),
            generator_conditioning=conditioning,
        )
        long, _, long_covariates = _generate_sample_values(
            capability_id,
            context_length + horizon + season_length,
            context_length,
            target_dim,
            season_length,
            3,
            np.random.default_rng(91),
            generator_conditioning=conditioning,
        )

        assert np.allclose(short, long[: len(short)])
        if short_covariates is not None:
            assert long_covariates is not None
            assert np.allclose(short_covariates, long_covariates[: len(short_covariates)])


def test_committed_artifact_uses_one_canonical_strength_curve_per_capability():
    artifact = load_generator_conditioning_artifact()
    assert artifact is not None
    assert artifact["schema_version"] == "synthetic_v2_generator_conditioning_artifact.v2"
    assert artifact["canonical_intensity"]["scale_id"]
    assert len(artifact["canonical_intensity"]["scale_fingerprint"]) == 16
    assert artifact["config"]["canonical_scale_id"] == artifact["canonical_intensity"]["scale_id"]

    canonical = artifact["canonical_intensity"]["capabilities"]
    observed: dict[str, set[tuple[float, ...]]] = {}
    for profile in artifact["profiles"].values():
        for capability_id, capability in profile["capabilities"].items():
            observed.setdefault(capability_id, set()).add(
                tuple(capability["canonical_target_values"])
            )
            assert capability["canonical_calibration"]["status"] == "supported"
            assert capability["canonical_calibration"]["fit_sample_count"] >= 64
            assert capability["canonical_calibration"]["validation_sample_count"] >= 64
            assert capability["canonical_calibration"]["validation_seed_is_independent"] is True
            assert len(capability["local_real_percentiles_at_canonical_targets"]) == 5

    assert all(len(curves) == 1 for curves in observed.values())
    assert {
        capability_id: next(iter(curves))
        for capability_id, curves in observed.items()
    } == {
        capability_id: tuple(definition["target_values"])
        for capability_id, definition in canonical.items()
    }
