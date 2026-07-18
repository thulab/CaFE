from __future__ import annotations

import copy

import numpy as np

from app.services.synthetic_generation_service import _generate_sample_values
from app.services.synthetic_generator_conditioning import (
    ARTIFACT_SCHEMA_VERSION,
    INTENSITY_POLICY_ID,
    REAL_BOUNDED_INTENSITY_POLICY_ID,
    GeneratorConditioning,
    matching_generator_profiles,
    resolve_generator_conditioning,
    select_balanced_profile_id,
)


def conditioning_artifact() -> dict:
    capability = {
        "parameters": {"structure_scale": 0.75},
        "intensity_lambdas": [0.0, 0.1, 0.3, 0.6, 1.0],
        "target_percentile_levels": [0.1, 0.3, 0.5, 0.7, 0.9],
        "target_feature": "trend_strength",
        "target_values": [0.02, 0.12, 0.25, 0.38, 0.55],
        "calibrated_realized_strengths": [0.01, 0.11, 0.21, 0.39, 0.69],
        "calibration": {
            "status": "supported",
            "max_normalized_error": 0.02,
        },
        "calibration_method": "unit-test",
    }
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": "2026-07-15T00:00:00+00:00",
        "intensity_policy": {
            "policy_id": INTENSITY_POLICY_ID,
            "percentile_levels": [0.1, 0.3, 0.5, 0.7, 0.9],
            "definition": "five relative strength quantiles calibrated independently per dataset",
        },
        "profiles": {
            "profile_a": {
                "profile_id": "profile_a",
                "dataset_id": "dataset_a",
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
                "dataset_id": "dataset_b",
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
    assert conditioning.metadata(3)["dataset_id"] == "dataset_a"
    assert conditioning.metadata(3)["target_percentile_level"] == 0.5
    assert conditioning.metadata(3)["target_relative_level"] == 0.5
    assert conditioning.metadata(3)["target_level_semantics"] == "empirical_quantile"
    assert conditioning.metadata(3)["target_strength"] == 0.25
    assert conditioning.metadata(3)["calibrated_expected_strength"] == 0.21
    assert conditioning.metadata(3)["intensity_policy_id"] == INTENSITY_POLICY_ID
    assert "canonical_scale_id" not in conditioning.metadata(3)


def test_real_bounded_policy_exposes_relative_level_semantics():
    artifact = conditioning_artifact()
    artifact["intensity_policy"] = {
        "policy_id": REAL_BOUNDED_INTENSITY_POLICY_ID,
        "percentile_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
        "relative_dose_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
        "definition": "dataset-local real-bounded generator-feasible levels",
        "real_tolerance": {
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "upper_multiplier": 1.2,
        },
    }
    for profile in artifact["profiles"].values():
        profile["capabilities"]["trend"]["target_percentile_levels"] = [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ]

    conditioning = resolve_generator_conditioning(
        capability_id="trend",
        profile_id="profile_a",
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    )

    assert conditioning is not None
    metadata = conditioning.metadata(4)
    assert metadata["target_relative_level"] == 0.75
    assert metadata["target_percentile_level"] == 0.75
    assert metadata["target_level_semantics"] == "relative_position"
    assert "real-bounded generator-feasible" in metadata["intensity_semantics"]

    artifact["intensity_policy"]["real_tolerance"].pop("upper_multiplier")
    assert (
        resolve_generator_conditioning(
            capability_id="trend",
            profile_id="profile_a",
            context_length=168,
            horizon=24,
            target_dim=1,
            artifact=artifact,
        )
        is None
    )


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


def test_dataset_profiles_keep_independent_target_strength_curves():
    artifact = conditioning_artifact()
    artifact["profiles"]["profile_b"]["capabilities"]["trend"] = copy.deepcopy(
        artifact["profiles"]["profile_b"]["capabilities"]["trend"]
    )
    artifact["profiles"]["profile_b"]["capabilities"]["trend"]["target_values"] = [
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]

    first = resolve_generator_conditioning(
        capability_id="trend",
        profile_id="profile_a",
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    )
    second = resolve_generator_conditioning(
        capability_id="trend",
        profile_id="profile_b",
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    )

    assert first is not None
    assert second is not None
    assert first.target_percentile_levels == second.target_percentile_levels
    assert first.target_values != second.target_values
    assert first.dataset_id == "dataset_a"
    assert second.dataset_id == "dataset_b"


def test_legacy_canonical_artifact_fails_closed():
    artifact = conditioning_artifact()
    artifact["schema_version"] = "synthetic_v2_generator_conditioning_artifact.v3"
    artifact["canonical_intensity"] = {"scale_id": "legacy-global-scale"}

    assert matching_generator_profiles(
        capability_id="trend",
        profile_ids=("profile_a",),
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    ) == []
    assert resolve_generator_conditioning(
        capability_id="trend",
        profile_id="profile_a",
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    ) is None


def test_profile_percentile_levels_must_match_dataset_local_policy():
    artifact = conditioning_artifact()
    artifact["profiles"]["profile_a"]["capabilities"]["trend"]["target_percentile_levels"] = [
        0.05,
        0.25,
        0.5,
        0.75,
        0.95,
    ]

    assert resolve_generator_conditioning(
        capability_id="trend",
        profile_id="profile_a",
        context_length=168,
        horizon=24,
        target_dim=1,
        artifact=artifact,
    ) is None


def test_profile_conditioned_generators_preserve_horizon_prefixes():
    cases = {
        "trend": 1,
        "multi_seasonal": 1,
        "time_varying_seasonality": 1,
        "regime_switching": 1,
        "nonlinear_persistence": 1,
        "predictable_intermittency": 1,
        "common_factor": 3,
        "hierarchical_coherence": 3,
        "covariate_response": 1,
    }
    context_length = 168
    horizon = 24
    season_length = 24
    for capability_id, target_dim in cases.items():
        conditioning = GeneratorConditioning(
            profile_id="dataset_a__task__L168_H24",
            dataset_id="dataset_a",
            capability_id=capability_id,
            context_length=context_length,
            horizon=horizon,
            target_dim=target_dim,
            season_length=season_length,
            frequency="h",
            parameters={"structure_scale": 1.0},
            intensity_lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
            target_percentile_levels=(0.1, 0.3, 0.5, 0.7, 0.9),
            target_feature="unit_feature",
            target_values=(0.1, 0.2, 0.3, 0.4, 0.5),
            calibrated_realized_strengths=(0.1, 0.2, 0.3, 0.4, 0.5),
            calibration_max_normalized_error=0.01,
            intensity_policy_id=INTENSITY_POLICY_ID,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            artifact_created_at="2026-07-18T00:00:00+00:00",
            calibration_method="unit-test",
        )
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
