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
    assert conditioning.metadata(3)["calibrated_profile_expected_strength"] == 0.21
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
    assert artifact["schema_version"] == "synthetic_v2_generator_conditioning_artifact.v3"
    assert artifact["canonical_intensity"]["scale_id"] == (
        "synthetic-v2-paper-v2-shortcut-resistant-2026-07-18"
    )
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
            expected_bank_count = (
                4
                if capability_id
                in {"nonlinear_persistence", "covariate_response"}
                else 2
            )
            assert (
                capability["canonical_calibration"]["fit_seed_bank_count"]
                == expected_bank_count
            )
            assert capability["canonical_calibration"]["validation_sample_count"] >= 256
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
    assert all(
        all(right > left for left, right in zip(row["target_values"], row["target_values"][1:]))
        for row in canonical.values()
    )
    assert canonical["regime_switching"]["primary_feature"] == (
        "regime_clock_history_incremental_r2"
    )
    assert canonical["regime_switching"]["target_values"][0] == 0.1
    assert canonical["regime_switching"]["target_resolution"]["method"] == (
        "qualification_boundary_to_q90_linear_grid"
    )
    assert canonical["nonlinear_persistence"]["primary_feature"] == (
        "nonlinear_conditional_gain"
    )
    assert canonical["nonlinear_persistence"]["target_values"][0] == 0.0
    assert canonical["nonlinear_persistence"]["target_resolution"]["method"] == (
        "adjusted_r2_null_to_q90_linear_grid"
    )
    for profile in artifact["profiles"].values():
        nonlinear = profile["capabilities"].get("nonlinear_persistence")
        if nonlinear is not None:
            assert nonlinear["canonical_calibration"]["fit_sample_count"] >= 512
    assert artifact["config"]["canonical_reference_profile_ids_by_capability"][
        "regime_switching"
    ] == [
        "uci_hydraulic_eps1_420ctx_60h",
        "skchange_hvac_unit0_504ctx_144h",
    ]
    assert len(artifact["config"]["online_conditioning_profile_ids"]) == 8
    assert artifact["config"]["research_only_conditioning_profile_ids"] == [
        "electricity_hourly_daily_2048ctx_24h"
    ]
    assert all(
        artifact["profiles"][profile_id]["conditioning_role"] == "paper_v2_online"
        for profile_id in artifact["config"]["online_conditioning_profile_ids"]
    )
    assert artifact["profiles"]["electricity_hourly_daily_2048ctx_24h"][
        "conditioning_role"
    ] == "research_only_pending_near_distance_gate"
    qualification = artifact["canonical_intensity"]["reference_qualification"]
    assert qualification["uci_hydraulic_eps1_420ctx_60h"]["regime_switching"][
        "qualified_window_count"
    ] >= 30
    assert qualification["skchange_hvac_unit0_504ctx_144h"]["regime_switching"][
        "qualified_window_count"
    ] >= 30
