from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_calibration_strata_are_nonoverlapping_and_use_floor_capacity():
    common = load_script("paper_v8_pipeline_common")

    strata = common.nonoverlapping_strata(2 * 504 + 37)

    assert len(strata) == 2
    assert all(upper >= lower for lower, upper in strata)
    assert strata[0][1] + 504 <= strata[1][0]


def test_direct_anchor_summary_does_not_compress_feature_values():
    common = load_script("paper_v8_pipeline_common")
    anchor = {"acf1": 0.93, "slope_abs": 0.17}

    summary = common.anchor_summary(anchor)

    assert summary == {
        "acf1": {"p50": 0.93},
        "slope_abs": {"p50": 0.17},
    }


def test_v8_registry_freezes_nineteen_l504_eligible_datasets():
    common = load_script("paper_v8_pipeline_common")

    assert len(common.DATASET_REGISTRY) == 19
    assert {
        "gift_bizitobs_application",
        "gift_bizitobs_service",
        "gift_hierarchical_sales_d",
        "gift_m4_hourly",
        "gift_us_births_d",
        "gift_saugeenday_d",
        "gift_temperature_rain_d",
    }.issubset(common.DATASET_REGISTRY)


def test_10s_period_policy_separates_calendar_feature_and_mase_periods():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.CONTEXT_LENGTH, dtype=float)
    history = np.sin(2.0 * np.pi * time / 120.0)

    policy = common.calibration_period_policy("10S", history)

    assert policy["calendar_season_length"] == 8640
    assert policy["calendar_season_feature_observable"] is False
    assert 110 <= policy["raw_profile_dominant_period"] <= 130
    assert policy["feature_period"] == round(policy["raw_profile_dominant_period"])
    assert policy["mase_period"] == 1


def test_hourly_period_policy_keeps_calendar_period_for_features_and_mase():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.CONTEXT_LENGTH, dtype=float)
    history = np.sin(2.0 * np.pi * time / 24.0)

    policy = common.calibration_period_policy("H", history)

    assert policy["calendar_season_length"] == 24
    assert policy["calendar_season_feature_observable"] is True
    assert policy["feature_period"] == 24
    assert policy["mase_period"] == 24


def test_slow_profile_period_is_clipped_per_mechanism_not_to_lag_one():
    common = load_script("paper_v8_pipeline_common")
    dataset = common.resolve_dataset("gift_bizitobs_application")
    parameters = {"profile_dominant_period": 252.0}

    def metadata(capability_id: str, seed: int = 41):
        conditioning = common.build_conditioning(
            dataset,
            capability_id=capability_id,
            frequency="10S",
            season_length=168,
            parameters=parameters,
        )
        return common.generate_deterministic_sample(
            capability_id,
            common.MASTER_LENGTH,
            common.CONTEXT_LENGTH,
            common.TARGET_DIM_BY_CAPABILITY[capability_id],
            conditioning.season_length,
            5,
            np.random.default_rng(seed),
            conditioning=conditioning,
        )[1]

    multi = metadata("multi_seasonal")
    time_varying = metadata("time_varying_seasonality")
    regime = metadata("regime_switching")
    nonlinear = metadata("nonlinear_persistence")
    intermittent = metadata("predictable_intermittency")
    covariate = metadata("covariate_response")

    assert multi["effective_primary_period"] == 84
    assert max(multi["periods"]) <= 168
    assert 8 <= time_varying["primary_period"] <= 126
    assert all(12 <= value <= 84 for value in regime["dwell_pattern"])
    assert 4 <= nonlinear["seasonal_lag"] <= 48
    assert 2 <= nonlinear["nonlinear_lag"] <= 32
    assert 8 <= intermittent["event_period"] <= 126
    assert all(8 <= value <= 126 for value in intermittent["pulse_interval_pattern"])
    assert 2 <= covariate["event_width"] <= 6


def test_sensitivity_seed_selection_is_prefix_stable():
    generation = load_script("generate_paper_v8_samples")

    first = generation.selected_sensitivity_seeds(
        "gift_electricity_h",
        list(range(1)),
        4,
    )
    expanded = generation.selected_sensitivity_seeds(
        "gift_electricity_h",
        list(range(16)),
        4,
    )

    assert first == expanded.intersection(range(1))
    assert first == set()


def test_multi_dataset_pipeline_accepts_explicit_dataset_list():
    pipeline = load_script("run_paper_v8_pipeline")
    args = SimpleNamespace(
        dataset_id=None,
        dataset_ids=[
            "gift_electricity_h",
            "gift_bizitobs_application",
        ],
    )

    assert pipeline.requested_dataset_ids(args) == [
        "gift_electricity_h",
        "gift_bizitobs_application",
    ]


def test_experiment_manifest_is_identity_scoped_and_immutable(tmp_path):
    pipeline = load_script("run_paper_v8_pipeline")
    protocol = {
        "schema_version": "paper_v8_experiment_protocol.v1",
        "dataset_ids": ["gift_electricity_h"],
        "seed_start": 0,
        "seed_count": 64,
    }
    protocol_sha256 = pipeline.v8.json_sha256(protocol)
    experiment_id = pipeline.default_experiment_id(
        protocol_sha256,
        now=datetime(2026, 7, 24, 12, 30, tzinfo=timezone.utc),
    )

    experiment_root, manifest = pipeline.initialize_experiment(
        storage_root=tmp_path,
        experiment_id=experiment_id,
        protocol=protocol,
        endpoints=["http://service-a"],
    )
    second_root, second_manifest = pipeline.initialize_experiment(
        storage_root=tmp_path,
        experiment_id=experiment_id,
        protocol=protocol,
        endpoints=["http://different-runtime-service"],
    )

    assert experiment_id.startswith("v8_")
    assert experiment_id.endswith("_20260724T123000Z")
    assert experiment_root == second_root
    assert manifest == second_manifest
    assert manifest["protocol_sha256"] == protocol_sha256
    assert (experiment_root / "experiment_manifest.json").is_file()

    with pytest.raises(ValueError, match="does not match"):
        pipeline.initialize_experiment(
            storage_root=tmp_path,
            experiment_id=experiment_id,
            protocol={**protocol, "seed_count": 32},
            endpoints=["http://service-a"],
        )


def test_pre_inference_execution_policy_upgrade_is_explicit_and_audited(
    tmp_path,
):
    pipeline = load_script("run_paper_v8_pipeline")
    protocol = {
        "schema_version": "paper_v8_experiment_protocol.v1",
        "dataset_ids": ["gift_electricity_h"],
        "seed_start": 0,
        "seed_count": 64,
        "model_execution_config": {"Chronos-2": {"http_concurrency": 32}},
        "model_scheduling_policy": {"policy_id": "legacy"},
    }
    experiment_root, original = pipeline.initialize_experiment(
        storage_root=tmp_path,
        experiment_id="experiment",
        protocol=protocol,
        endpoints=["http://service-a"],
    )
    upgraded_protocol = {
        **protocol,
        "model_execution_config": {"Chronos-2": {"http_concurrency": 384}},
        "model_scheduling_policy": {"policy_id": "all-services-per-model"},
    }

    _root, upgraded = pipeline.initialize_experiment(
        storage_root=tmp_path,
        experiment_id="experiment",
        protocol=upgraded_protocol,
        endpoints=["http://service-a"],
        allow_inference_execution_upgrade=True,
    )

    assert upgraded["protocol"] == upgraded_protocol
    assert upgraded["protocol_history"][0]["protocol"] == protocol
    assert (
        upgraded["protocol_history"][0]["protocol_sha256"]
        == original["protocol_sha256"]
    )

    inference_file = (
        experiment_root
        / "gift_electricity_h"
        / "03_inference"
        / "seed"
        / "predictions.jsonl"
    )
    inference_file.parent.mkdir(parents=True)
    inference_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="after inference artifacts"):
        pipeline.initialize_experiment(
            storage_root=tmp_path,
            experiment_id="experiment",
            protocol={
                **upgraded_protocol,
                "model_execution_config": {"Chronos-2": {"http_concurrency": 512}},
            },
            endpoints=["http://service-a"],
            allow_inference_execution_upgrade=True,
        )


def test_execution_policy_upgrade_allows_matching_preparation_only_run(
    tmp_path,
):
    pipeline = load_script("run_paper_v8_pipeline")
    protocol = {
        "schema_version": "paper_v8_experiment_protocol.v1",
        "dataset_ids": ["gift_electricity_h"],
        "model_execution_config": {"Chronos-2": {"http_concurrency": 32}},
        "model_scheduling_policy": {"policy_id": "legacy"},
    }
    experiment_root, original = pipeline.initialize_experiment(
        storage_root=tmp_path,
        experiment_id="experiment",
        protocol=protocol,
        endpoints=["http://service-a"],
    )
    pipeline.write_pipeline_status(
        experiment_root,
        experiment_id="experiment",
        protocol_sha256=original["protocol_sha256"],
        state="running",
        start_at="calibration",
        stop_after="validation",
        completed=[],
        active_dataset_id="gift_electricity_h",
        active_step="generation",
    )
    upgraded_protocol = {
        **protocol,
        "model_execution_config": {"Chronos-2": {"http_concurrency": 384}},
        "model_scheduling_policy": {"policy_id": "all-services-per-model"},
    }

    _root, upgraded = pipeline.initialize_experiment(
        storage_root=tmp_path,
        experiment_id="experiment",
        protocol=upgraded_protocol,
        endpoints=["http://service-a"],
        allow_inference_execution_upgrade=True,
    )

    assert upgraded["protocol"] == upgraded_protocol
    assert (
        upgraded["protocol_history"][0]["concurrent_preparation_status"]["active_step"]
        == "generation"
    )


def test_execution_policy_upgrade_rejects_active_run_that_can_infer(tmp_path):
    pipeline = load_script("run_paper_v8_pipeline")
    protocol = {
        "schema_version": "paper_v8_experiment_protocol.v1",
        "dataset_ids": ["gift_electricity_h"],
        "model_execution_config": {"Chronos-2": {"http_concurrency": 32}},
        "model_scheduling_policy": {"policy_id": "legacy"},
    }
    experiment_root, original = pipeline.initialize_experiment(
        storage_root=tmp_path,
        experiment_id="experiment",
        protocol=protocol,
        endpoints=["http://service-a"],
    )
    pipeline.write_pipeline_status(
        experiment_root,
        experiment_id="experiment",
        protocol_sha256=original["protocol_sha256"],
        state="running",
        start_at="calibration",
        stop_after="analysis",
        completed=[],
        active_dataset_id="gift_electricity_h",
        active_step="generation",
    )

    with pytest.raises(ValueError, match="may enter inference"):
        pipeline.initialize_experiment(
            storage_root=tmp_path,
            experiment_id="experiment",
            protocol={
                **protocol,
                "model_execution_config": {"Chronos-2": {"http_concurrency": 384}},
            },
            endpoints=["http://service-a"],
            allow_inference_execution_upgrade=True,
        )


def test_response_support_detects_sustained_foldback_without_magic_bound():
    common = load_script("paper_v8_pipeline_common")
    grid = np.linspace(0.0, 1.0, 11)
    response = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.62, 0.55, 0.58])

    index, audit = common.stable_monotone_support(grid, response)

    assert index == 7
    assert audit["foldback_detected"] is True
    assert audit["effective_lambda_support"] == pytest.approx([0.0, 0.7])


def test_v8_nonlinear_gain_searches_relative_lags_and_tracks_strength():
    common = load_script("paper_v8_pipeline_common")

    def rational_delay_series(gain: float) -> np.ndarray:
        rng = np.random.default_rng(9)
        values = np.zeros(common.MASTER_LENGTH, dtype=float)
        values[:24] = rng.normal(0.0, 0.6, size=24)
        time = np.arange(common.MASTER_LENGTH, dtype=float)
        forcing = np.sin(2.0 * np.pi * time / 24.0 + 0.7) + 0.4 * np.sin(
            2.0 * np.pi * time / 47.0 + 1.2
        )
        for index in range(24, common.MASTER_LENGTH):
            delayed = values[index - 4]
            values[index] = (
                0.52 * values[index - 1]
                + 0.14 * values[index - 24]
                + gain * delayed / (1.0 + delayed * delayed)
                + 0.18 * forcing[index]
            )
        return values

    lower, lower_lag, candidates = common.v8_nonlinear_conditional_gain(
        rational_delay_series(0.18),
        24,
    )
    upper, upper_lag, repeated_candidates = common.v8_nonlinear_conditional_gain(
        rational_delay_series(0.50),
        24,
    )

    assert candidates == (4, 5, 6, 8, 12)
    assert repeated_candidates == candidates
    assert lower_lag == 4
    assert upper_lag == 4
    assert upper > lower > 0.0
    actual_lower = common.v8_nonlinear_actual_lag_gain(
        rational_delay_series(0.18),
        24,
        4,
    )
    actual_upper = common.v8_nonlinear_actual_lag_gain(
        rational_delay_series(0.50),
        24,
        4,
    )
    assert actual_upper > actual_lower > 0.0


def test_nonlinear_response_uses_construction_dose_not_observable_proxy(
    monkeypatch,
):
    common = load_script("paper_v8_pipeline_common")

    def calibration_member(
        dataset,
        anchor,
        *,
        capability_id,
        family_role,
        lambda_value,
        calibration_seed_index,
    ):
        del dataset, anchor, capability_id, family_role
        return {
            "nonlinear_strength": lambda_value,
            # Deliberately folded: this observable diagnostic must not
            # truncate or reverse the construction-dose support.
            "nonlinear_conditional_gain": (
                lambda_value
                if calibration_seed_index >= 2 or lambda_value <= 0.2
                else 0.2 - 2.0 * (lambda_value - 0.2)
            ),
        }, {}

    monkeypatch.setattr(
        common,
        "generate_calibration_member",
        calibration_member,
    )

    grid, response, audit = common.monotone_response_curve(
        common.resolve_dataset("gift_electricity_h"),
        [{} for _ in range(20)],
        capability_id="nonlinear_persistence",
        family_role="primary",
        calibration_seed_count=20,
    )

    assert grid == pytest.approx(np.linspace(0.0, 1.0, 21))
    assert response == pytest.approx(grid)
    assert audit["effective_lambda_support"] == pytest.approx([0.0, 1.0])
    assert "path_support_quantile" not in audit
    assert audit["split_half_diagnostic"]["triggers_path_expansion"] is False
    assert audit["split_half_diagnostic"]["first_half_path_count"] == 10
    assert audit["split_half_diagnostic"]["second_half_path_count"] == 10


def test_compressed_nonlinear_secondary_match_uses_relative_grid(
    monkeypatch,
):
    common = load_script("paper_v8_pipeline_common")

    def response_curve(
        dataset,
        anchors,
        *,
        capability_id,
        family_role,
        calibration_seed_count,
    ):
        del dataset, anchors, capability_id, calibration_seed_count
        if family_role == "primary":
            grid = np.asarray([0.0, 0.3])
            response = np.asarray([0.0, 1.0])
        else:
            grid = np.asarray([0.0, 0.5])
            response = np.asarray([0.0, 10.0])
        return (
            grid,
            response,
            {
                "effective_lambda_support": [
                    float(grid[0]),
                    float(grid[-1]),
                ]
            },
        )

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    monkeypatch.setattr(
        common,
        "selected_lambda_mean_response",
        lambda *args, selected_lambdas, **kwargs: tuple(selected_lambdas),
    )
    anchors = [
        {"features": {"nonlinear_conditional_gain": value}}
        for value in np.linspace(0.0, 1.0, 20)
    ]

    calibration = common.calibrate_capabilities(
        common.resolve_dataset("gift_electricity_h"),
        anchors,
        calibration_seed_count=32,
        capability_ids=["nonlinear_persistence"],
    )
    nonlinear = calibration["capabilities"]["nonlinear_persistence"]

    assert nonlinear["response_calibration_seed_count"] == 32
    assert nonlinear["secondary"]["calibration_status"] == (
        "nonlinear_secondary_compressed_match_fixed_relative_grid_used"
    )
    assert nonlinear["secondary"]["selected_lambdas"] == pytest.approx(
        np.linspace(0.0, 0.5, 5)
    )


def test_response_paths_expand_only_after_hard_failure(monkeypatch):
    common = load_script("paper_v8_pipeline_common")
    calls = []

    def response_curve(
        dataset,
        anchors,
        *,
        capability_id,
        family_role,
        calibration_seed_count,
    ):
        del dataset, anchors, capability_id
        calls.append((family_role, calibration_seed_count))
        if calibration_seed_count == 32:
            grid = np.asarray([0.0])
            response = np.asarray([0.2])
        else:
            grid = np.asarray([0.0, 1.0])
            response = np.asarray([0.0, 1.0])
        return (
            grid,
            response,
            {
                "effective_lambda_support": [
                    float(grid[0]),
                    float(grid[-1]),
                ]
            },
        )

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    monkeypatch.setattr(
        common,
        "selected_lambda_mean_response",
        lambda *args, selected_lambdas, **kwargs: tuple(selected_lambdas),
    )
    anchors = [
        {"features": {"curvature_abs": value}} for value in np.linspace(0.2, 0.8, 20)
    ]

    calibration = common.calibrate_capabilities(
        common.resolve_dataset("gift_electricity_h"),
        anchors,
        calibration_seed_count=32,
        maximum_calibration_seed_count=96,
        capability_ids=["trend"],
    )
    trend = calibration["capabilities"]["trend"]

    assert calls == [
        ("primary", 32),
        ("secondary", 32),
        ("primary", 64),
        ("secondary", 64),
    ]
    assert trend["response_calibration_seed_count"] == 64
    assert trend["response_calibration_path_policy"] == {
        "policy": (
            "formal_generation_seed_bank_" "fixed_base_hard_failure_only_expansion_v2"
        ),
        "path_sampling": {
            "anchor": "formal_logical_seed_hash_v1",
            "rng": "formal_generation_path_v1",
            "seed_start": 0,
        },
        "initial_path_count": 32,
        "maximum_path_count": 96,
        "attempted_path_counts": [32, 64],
        "selected_path_count": 64,
        "expanded": True,
        "hard_failure_attempts": [
            {
                "path_count": 32,
                "reasons": {
                    "primary": [
                        "lambda_support_collapsed",
                    ],
                    "secondary": [
                        "lambda_support_collapsed",
                    ],
                },
            }
        ],
        "split_half_diagnostics_trigger_expansion": False,
    }


def test_intermittency_measured_dose_comes_from_generator_metadata():
    common = load_script("paper_v8_pipeline_common")
    dataset = common.resolve_dataset("gift_electricity_h")
    conditioning = common.build_conditioning(
        dataset,
        capability_id="predictable_intermittency",
        frequency="H",
        season_length=24,
    )
    target, metadata, covariates = common.generate_deterministic_sample(
        "predictable_intermittency",
        common.MASTER_LENGTH,
        common.CONTEXT_LENGTH,
        1,
        24,
        5,
        np.random.default_rng(41),
        conditioning=conditioning,
    )
    target, covariates = common.standardize_generated_sample(
        "predictable_intermittency",
        target,
        covariates,
        metadata=metadata,
    )

    features = common.measured_features(
        "predictable_intermittency",
        target,
        covariates,
        season_length=24,
        metadata=metadata,
    )

    assert (
        common.PRIMARY_TARGET_FEATURE["predictable_intermittency"]
        == "event_effect_energy_share"
    )
    assert features["event_effect_energy_share"] == pytest.approx(
        metadata["event_effect_energy_share"]
    )


def test_master_views_share_exact_future_and_l504_mase_scale():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.MASTER_LENGTH, dtype=float)
    target = (np.sin(2 * np.pi * time / 24.0) + 0.002 * time)[:, None]
    scale, scale_by_target = common.mase_scales(target, season_length=24)
    master = {
        "sample_id": "v8__demo",
        "master_sample_id": "v8__demo",
        "counterfactual_pair_id": None,
        "capability_id": "trend",
        "context_length": common.CONTEXT_LENGTH,
        "horizon": common.HORIZON,
        "target_dim": 1,
        "covariate_dim": 0,
        "target": target.tolist(),
        "covariates": None,
        "generation_metadata": {},
        "mase_scale": scale,
        "mase_scale_by_target": scale_by_target,
        "future_sha256": "future",
    }

    views = [
        common.master_view(master, context) for context in common.VIEW_CONTEXT_LENGTHS
    ]

    futures = [
        np.asarray(view["target"], dtype=float)[view["context_length"] :]
        for view in views
    ]
    assert all(np.array_equal(futures[0], future) for future in futures[1:])
    assert {view["mase_scale"] for view in views} == {scale}
    assert all(
        view["view_standardization_policy"]
        == "slice_exact_l504_standardized_master_without_restandardization"
        for view in views
    )


@pytest.mark.parametrize(
    ("capability_id", "target_dim"),
    (("common_factor", 5), ("cross_series_dependence", 3)),
)
def test_multivariate_input_ablation_keeps_future_and_marginal_scale(
    capability_id,
    target_dim,
):
    common = load_script("paper_v8_pipeline_common")
    dataset = common.resolve_dataset("gift_electricity_h")
    conditioning = common.build_conditioning(
        dataset,
        capability_id=capability_id,
        frequency="H",
        season_length=24,
    )

    def sample(seed):
        target, metadata, covariates = common.generate_deterministic_sample(
            capability_id,
            common.MASTER_LENGTH,
            common.CONTEXT_LENGTH,
            target_dim,
            24,
            5,
            np.random.default_rng(seed),
            conditioning=conditioning,
        )
        target, covariates = common.standardize_generated_sample(
            capability_id,
            target,
            covariates,
            metadata=metadata,
        )
        scale, scale_by_target = common.mase_scales(
            target,
            season_length=24,
        )
        return {
            "schema_version": "test",
            "sample_id": f"sample-{seed}",
            "master_sample_id": f"sample-{seed}",
            "dataset_id": dataset.dataset_id,
            "capability_id": capability_id,
            "generator_family_role": "primary",
            "intensity": 5,
            "context_length": common.CONTEXT_LENGTH,
            "horizon": common.HORIZON,
            "target_dim": target_dim,
            "covariate_dim": 0,
            "covariates": None,
            "counterfactual_pair_id": None,
            "counterfactual_member": None,
            "generation_metadata": metadata,
            "target": target.tolist(),
            "mase_scale": scale,
            "mase_scale_by_target": scale_by_target,
            "future_sha256": "future",
        }

    clean = sample(101)
    donor = sample(103)
    ablated = common.multivariate_input_ablation_sample(clean, donor)
    clean_target = np.asarray(clean["target"])
    ablated_target = np.asarray(ablated["target"])
    metadata = ablated["input_ablation_metadata"]
    channels = metadata["replaced_channels"]
    start, stop = metadata["replaced_history_slice"]

    assert np.array_equal(
        clean_target[common.CONTEXT_LENGTH :],
        ablated_target[common.CONTEXT_LENGTH :],
    )
    assert not np.array_equal(
        clean_target[start:stop, channels],
        ablated_target[start:stop, channels],
    )
    assert np.mean(
        clean_target[start:stop, channels],
        axis=0,
    ) == pytest.approx(np.mean(ablated_target[start:stop, channels], axis=0))
    assert np.std(
        clean_target[start:stop, channels],
        axis=0,
    ) == pytest.approx(np.std(ablated_target[start:stop, channels], axis=0))


@pytest.mark.parametrize(
    ("capability_id", "metadata", "field", "expected"),
    [
        (
            "regime_switching",
            {"cut_points": [480, 510]},
            "cut_points",
            [72, 102],
        ),
        (
            "predictable_intermittency",
            {"pulse_centers": [480, 520]},
            "pulse_centers",
            [72, 112],
        ),
    ],
)
def test_short_views_shift_indexed_generation_metadata(
    capability_id,
    metadata,
    field,
    expected,
):
    common = load_script("paper_v8_pipeline_common")
    target = np.arange(common.MASTER_LENGTH, dtype=float)[:, None]
    scale, scale_by_target = common.mase_scales(target, season_length=24)
    master = {
        "sample_id": f"v8__{capability_id}",
        "counterfactual_pair_id": None,
        "capability_id": capability_id,
        "context_length": common.CONTEXT_LENGTH,
        "horizon": common.HORIZON,
        "target_dim": 1,
        "covariate_dim": 0,
        "target": target.tolist(),
        "covariates": None,
        "generation_metadata": metadata,
        "mase_scale": scale,
        "mase_scale_by_target": scale_by_target,
        "future_sha256": "future",
    }

    view = common.master_view(master, 96)

    assert view["generation_metadata"][field] == expected


def test_oracle_context_uses_one_context_for_both_pair_members():
    analysis = load_script("analyze_paper_v8")
    rows = []
    for context, member_mase in (
        (96, (0.5, 1.5)),
        (168, (0.7, 0.7)),
        (336, (0.9, 0.9)),
        (504, (1.0, 1.0)),
    ):
        for member, mase in enumerate(member_mase):
            rows.append(
                {
                    "model_id": "demo",
                    "master_sample_id": f"member-{member}",
                    "master_counterfactual_pair_id": "pair",
                    "counterfactual_member": member,
                    "context_length": context,
                    "metrics": {"mase": mase},
                }
            )

    selected, pair_context = analysis.selected_context_rows(rows)

    oracle = [row for row in selected if row["context_policy"] == "oracle_context"]
    assert pair_context[("demo", "pair")] == 168
    assert len(oracle) == 2
    assert {row["context_length"] for row in oracle} == {168}


def test_oracle_context_reuses_clean_parent_context_for_input_ablation():
    analysis = load_script("analyze_paper_v8")
    rows = []
    for context, clean_mase, ablated_mase in (
        (96, 0.8, 0.1),
        (168, 0.4, 0.9),
        (336, 0.6, 0.3),
        (504, 0.7, 0.2),
    ):
        rows.extend(
            [
                {
                    "model_id": "demo",
                    "master_sample_id": "clean",
                    "master_counterfactual_pair_id": None,
                    "clean_master_sample_id": None,
                    "context_length": context,
                    "metrics": {"mase": clean_mase},
                },
                {
                    "model_id": "demo",
                    "master_sample_id": "ablated",
                    "master_counterfactual_pair_id": None,
                    "clean_master_sample_id": "clean",
                    "context_length": context,
                    "metrics": {"mase": ablated_mase},
                },
            ]
        )

    selected, _ = analysis.selected_context_rows(rows)
    oracle = [row for row in selected if row["context_policy"] == "oracle_context"]

    assert len(oracle) == 2
    assert {row["context_length"] for row in oracle} == {168}


def test_split_bank_requires_two_batches_for_stability_statistics():
    analysis = load_script("analyze_paper_v8")
    rows = []
    for seed in range(64):
        batch_scale = 1.0 if seed < 32 else 2.0
        for model_id, model_scale in (("a", 1.0), ("b", 2.0)):
            for policy in ("fixed_l504", "oracle_context"):
                rows.append(
                    {
                        "dataset_id": "dataset",
                        "context_policy": policy,
                        "evaluation_table": "main",
                        "generator_family_role": "primary",
                        "capability_id": "trend",
                        "model_id": model_id,
                        "seed_index": seed,
                        "intensity": 5,
                        "metrics": {
                            "mase": batch_scale * model_scale,
                            "trend_slope_relative_abs_error": (
                                batch_scale * model_scale
                            ),
                        },
                    }
                )

    split = analysis.split_bank(
        rows,
        [],
        models=["a", "b"],
        seed_start=0,
        seed_count=64,
    )
    by_key = {
        (
            row["batch_size"],
            row["context_policy"],
            row["capability_id"],
            row["score_kind"],
        ): row
        for row in split
    }
    two_batches = by_key[(32, "fixed_l504", "trend", "accuracy")]
    one_batch = by_key[(64, "fixed_l504", "trend", "accuracy")]

    assert two_batches["mean_kendall_tau_b"] == pytest.approx(1.0)
    assert two_batches["top1_consistency"] == pytest.approx(1.0)
    assert two_batches["mean_top3_overlap"] == pytest.approx(1.0)
    assert two_batches["mean_pairwise_relative_score_difference"] == pytest.approx(
        2.0 / 3.0
    )
    assert one_batch["mean_kendall_tau_b"] is None
    assert one_batch["top1_consistency"] is None
    assert one_batch["mean_top3_overlap"] is None
    assert one_batch["mean_pairwise_relative_score_difference"] is None


def test_matched_comparison_excludes_unmatched_clean_seeds_and_intensities():
    analysis = load_script("analyze_paper_v8")

    def row(
        *,
        seed,
        intensity,
        mase,
        family="primary",
        table="main",
    ):
        return {
            "dataset_id": "dataset",
            "context_policy": "fixed_l504",
            "evaluation_table": table,
            "generator_family_role": family,
            "capability_id": "trend",
            "model_id": "model",
            "seed_index": seed,
            "intensity": intensity,
            "metrics": {
                "mase": mase,
                "trend_slope_relative_abs_error": mase,
            },
        }

    comparisons = analysis.matched_comparison_rows(
        [
            row(seed=0, intensity=1, mase=100.0),
            row(seed=1, intensity=5, mase=1.0),
            row(
                seed=1,
                intensity=5,
                mase=2.0,
                family="secondary",
            ),
        ],
        [],
    )

    secondary = next(
        item for item in comparisons if item["comparison_id"] == "secondary_family"
    )
    assert secondary["matched_seed_count"] == 1
    assert secondary["matched_intensities"] == [5]
    assert secondary["control_accuracy_score"] == pytest.approx(1.0)
    assert secondary["treatment_accuracy_score"] == pytest.approx(2.0)
    assert secondary["accuracy_relative_delta"] == pytest.approx(1.0)


def test_input_ablation_comparison_uses_unchanged_focal_channel_metric():
    analysis = load_script("analyze_paper_v8")

    def row(table, mase, protected):
        return {
            "dataset_id": "dataset",
            "context_policy": "fixed_l504",
            "evaluation_table": table,
            "generator_family_role": "primary",
            "capability_id": "common_factor",
            "model_id": "model",
            "seed_index": 0,
            "intensity": 5,
            "metrics": {
                "mase": mase,
                "protected_target_nmae": protected,
                "common_component_nmae": protected,
            },
        }

    comparisons = analysis.matched_comparison_rows(
        [
            row("main", mase=0.2, protected=0.4),
            row(
                "multivariate_input_ablation",
                mase=5.0,
                protected=0.5,
            ),
        ],
        [],
    )
    ablation = next(
        item
        for item in comparisons
        if item["comparison_id"] == "multivariate_input_ablation"
    )

    assert ablation["accuracy_metric"] == "protected_target_nmae"
    assert ablation["control_accuracy_score"] == pytest.approx(0.4)
    assert ablation["treatment_accuracy_score"] == pytest.approx(0.5)
    assert ablation["accuracy_delta"] == pytest.approx(0.1)


def test_inference_prediction_keeps_only_analysis_inputs():
    inference = load_script("run_paper_v8_inference")
    target = np.arange(12, dtype=float)[:, None]
    sample = {
        "sample_id": "view",
        "master_sample_id": "master",
        "dataset_id": "dataset",
        "config_id": "config",
        "profile_id": "profile",
        "capability_id": "trend",
        "generator_family_role": "primary",
        "generator_family_id": "family",
        "evaluation_table": "main",
        "intensity": 1,
        "seed_index": 0,
        "counterfactual_pair_id": None,
        "counterfactual_member": None,
        "context_length": 8,
        "horizon": 4,
        "target_dim": 1,
        "covariate_dim": 0,
        "mase_scale": 2.0,
        "target": target.tolist(),
        "future_sha256": "future",
    }
    forecast = np.zeros((4, 1), dtype=float)

    row = inference.prediction_row(
        "model",
        "foundation",
        sample,
        forecast,
    )

    assert row == {
        "schema_version": "paper_v8_inference_prediction.v2",
        "model_id": "model",
        "sample_id": "view",
        "forecast": forecast.tolist(),
    }


def test_bulk_request_preserves_multivariate_and_covariate_axes():
    inference = load_script("run_paper_v8_inference")
    children = []
    for offset in (0, 100):
        children.append(
            {
                "context_length": 4,
                "horizon": 2,
                "target_dim": 3,
                "covariate_dim": 2,
                "target": (
                    np.arange(18, dtype=np.float32).reshape(6, 3) + offset
                ).tolist(),
                "covariates": (
                    np.arange(12, dtype=np.float32).reshape(6, 2) + offset
                ).tolist(),
            }
        )

    content, target_shape, horizon = inference._bulk_request_content(
        "tabpfn-ts3",
        children,
    )
    payload = inference.msgpack.unpackb(content, raw=False)

    assert target_shape == (2, 3, 4)
    assert horizon == 2
    assert payload["shape"] == [2, 3, 4]
    assert payload["history_covariates_shape"] == [2, 2, 4]
    assert payload["future_covariates_shape"] == [2, 2, 2]
    targets = np.frombuffer(payload["targets"], dtype=np.float32).reshape(target_shape)
    history = np.frombuffer(
        payload["history_covariates"],
        dtype=np.float32,
    ).reshape(2, 2, 4)
    future = np.frombuffer(
        payload["future_covariates"],
        dtype=np.float32,
    ).reshape(2, 2, 2)
    assert targets[1, 2, 3] == 111
    assert history[1, 1, 3] == 107
    assert future[1, 1, 1] == 111


def test_model_phase_is_partitioned_across_all_services(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    task_path = tmp_path / "tasks.jsonl"
    common.write_jsonl(
        task_path,
        ({"sample_id": f"sample-{index}"} for index in range(30)),
    )
    model_id = "Chronos-2"
    services = [
        (
            f"http://service-{index}",
            {model_id: {"model_id": model_id}},
        )
        for index in range(3)
    ]

    work, manifest = inference.plan_model_phase(
        model_id,
        services,
        task_path=task_path,
        inference_dir=tmp_path / "inference",
    )

    assert manifest["model_id"] == model_id
    assert manifest["part_count"] == 3
    assert sum(part["row_count"] for part in manifest["parts"]) == 30
    assert sorted(item.part_index for items in work.values() for item in items) == [
        0,
        1,
        2,
    ]
    assert all(len(items) == 1 for items in work.values())
    assert all(item.model_id == model_id for items in work.values() for item in items)

    resumed_work, resumed_manifest = inference.plan_model_phase(
        model_id,
        services[:2],
        task_path=task_path,
        inference_dir=tmp_path / "inference",
    )
    assert resumed_manifest["part_count"] == 2
    assert sorted(
        item.part_index for items in resumed_work.values() for item in items
    ) == [0, 1]
    assert sorted(len(items) for items in resumed_work.values()) == [1, 1]


def test_eight_card_endpoint_does_not_infer_performance_from_devices():
    inference = load_script("run_paper_v8_inference")
    endpoint = "http://timecho89:10810"

    profiles = inference.build_endpoint_profiles(
        [endpoint],
        default_devices="0,1",
        endpoint_presets=[],
        endpoint_devices=[f"{endpoint}=0,1,2,3,4,5,6,7"],
        endpoint_capacities=[],
        endpoint_concurrency_scales=[],
        endpoint_model_capacities=[],
        endpoint_model_concurrencies=[],
    )

    assert profiles[endpoint].devices == "0,1,2,3,4,5,6,7"
    assert profiles[endpoint].capacity_units == 1
    assert profiles[endpoint].concurrency_scale == 1.0
    assert profiles[endpoint].capacity_for("Timer-3.5") == 1
    assert profiles[endpoint].http_concurrency_for("Timer-3.5", 512) == 512


def test_endpoint_profile_accepts_per_model_measured_overrides():
    inference = load_script("run_paper_v8_inference")
    endpoint = "http://timecho89:10810"

    profiles = inference.build_endpoint_profiles(
        [endpoint],
        default_devices="0,1",
        endpoint_presets=[],
        endpoint_devices=[f"{endpoint}=0,1,2,3,4,5,6,7"],
        endpoint_capacities=[],
        endpoint_concurrency_scales=[],
        endpoint_model_capacities=[
            f"{endpoint}|Timer-3.5=2",
            f"{endpoint}|Chronos-2=1",
        ],
        endpoint_model_concurrencies=[
            f"{endpoint}|Timer-3.5=3072",
            f"{endpoint}|Chronos-2=3072",
        ],
    )
    profile = profiles[endpoint]

    assert profile.capacity_for("Timer-3.5") == 2
    assert profile.capacity_for("Chronos-2") == 1
    assert profile.capacity_for("moirai2") == 1
    assert profile.http_concurrency_for("Timer-3.5", 512) == 3072
    assert profile.http_concurrency_for("Chronos-2", 384) == 3072
    assert profile.http_concurrency_for("moirai2", 384) == 384


def test_eight_card_preset_uses_measured_bulk_profiles():
    inference = load_script("run_paper_v8_inference")
    endpoint = "http://timecho89:10810"

    profiles = inference.build_endpoint_profiles(
        [endpoint],
        default_devices="0,1",
        endpoint_presets=[f"{endpoint}={inference.RTX5090X8_H48_B1_PRESET}"],
        endpoint_devices=[],
        endpoint_capacities=[],
        endpoint_concurrency_scales=[],
        endpoint_model_capacities=[],
        endpoint_model_concurrencies=[],
    )
    profile = profiles[endpoint]

    assert profile.devices == "0,1,2,3,4,5,6,7"
    assert profile.capacity_for("tirex2") == 3
    assert profile.capacity_for("tabpfn-ts3") == 4
    assert profile.capacity_for("Timer-3.5") == 2.4
    assert profile.http_concurrency_for("tirex2", 32) == 8
    assert profile.http_concurrency_for("tabpfn-ts3", 32) == 64


def test_default_topology_contains_three_dual_card_services_and_eight_card():
    inference = load_script("run_paper_v8_inference")

    assert inference.DEFAULT_ENDPOINTS == (
        "http://127.0.0.1:10810",
        "http://192.168.99.17:10811",
        "http://192.168.99.18:10810",
        "http://192.168.99.89:10810",
    )
    assert inference.DEFAULT_ENDPOINT_PRESETS == (
        "http://192.168.99.89:10810=rtx5090x8-h48-b1-v1",
    )


def test_model_phase_honors_explicit_per_model_capacity(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    model_id = "Chronos-2"
    eight_card_endpoint = "http://timecho89:10810"
    endpoints = [
        "http://127.0.0.1:10810",
        "http://192.168.99.17:10811",
        "http://192.168.99.18:10810",
        eight_card_endpoint,
    ]
    services = [
        (endpoint, {model_id: {"model_id": model_id}}) for endpoint in endpoints
    ]
    profiles = inference.build_endpoint_profiles(
        endpoints,
        default_devices="0,1",
        endpoint_presets=[],
        endpoint_devices=[f"{eight_card_endpoint}=0,1,2,3,4,5,6,7"],
        endpoint_capacities=[],
        endpoint_concurrency_scales=[],
        endpoint_model_capacities=[f"{eight_card_endpoint}|{model_id}=4"],
        endpoint_model_concurrencies=[],
    )
    task_path = tmp_path / "tasks.jsonl"
    common.write_jsonl(
        task_path,
        ({"sample_id": f"sample-{index}"} for index in range(70)),
    )

    work, manifest = inference.plan_model_phase(
        model_id,
        services,
        task_path=task_path,
        inference_dir=tmp_path / "inference",
        endpoint_profiles=profiles,
    )

    assert manifest["part_count"] == 4
    assert manifest["part_weights"] == [1, 1, 1, 4]
    assert len(work[eight_card_endpoint]) == 1
    assert all(work[endpoint] for endpoint in endpoints)


def test_single_eight_card_service_normalizes_to_one_part(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    endpoint = "http://timecho89:10810"
    model_id = "Chronos-2"
    profiles = inference.build_endpoint_profiles(
        [endpoint],
        default_devices="0,1",
        endpoint_presets=[],
        endpoint_devices=[f"{endpoint}=0,1,2,3,4,5,6,7"],
        endpoint_capacities=[],
        endpoint_concurrency_scales=[],
        endpoint_model_capacities=[],
        endpoint_model_concurrencies=[],
    )
    task_path = tmp_path / "tasks.jsonl"
    common.write_jsonl(
        task_path,
        ({"sample_id": f"sample-{index}"} for index in range(20)),
    )

    work, manifest = inference.plan_model_phase(
        model_id,
        [(endpoint, {model_id: {"model_id": model_id}})],
        task_path=task_path,
        inference_dir=tmp_path / "inference",
        endpoint_profiles=profiles,
    )

    assert manifest["part_count"] == 1
    assert len(work[endpoint]) == 1


def test_every_model_phase_uses_all_compatible_services(tmp_path):
    inference = load_script("run_paper_v8_inference")
    model_ids = [
        "Chronos-2",
        "toto2.0",
        "timesfm2.5",
        "tabpfn-ts3",
        "tirex2",
        "moirai2",
        "Timer-3.5",
    ]
    services = [
        (
            endpoint,
            {model_id: {"model_id": model_id} for model_id in model_ids},
        )
        for endpoint in (
            "http://127.0.0.1:10810",
            "http://192.168.99.17:10811",
            "http://192.168.99.18:10810",
        )
    ]
    task_path = tmp_path / "tasks.jsonl"
    load_script("paper_v8_pipeline_common").write_jsonl(
        task_path,
        ({"sample_id": f"sample-{index}"} for index in range(30)),
    )

    for model_id in model_ids:
        work, manifest = inference.plan_model_phase(
            model_id,
            services,
            task_path=task_path,
            inference_dir=tmp_path / "inference",
        )
        assert manifest["part_count"] == 3
        assert sorted(work) == sorted(endpoint for endpoint, _ in services)
        assert all(len(items) == 1 for items in work.values())


def test_model_shards_refresh_when_source_task_changes(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    task_path = tmp_path / "tasks.jsonl"
    inference_dir = tmp_path / "inference"
    common.write_jsonl(
        task_path,
        ({"sample_id": f"old-{index}"} for index in range(6)),
    )
    first = inference.prepare_model_task_shards(
        task_path,
        model_id="timesfm2.5",
        part_count=2,
        inference_dir=inference_dir,
    )

    common.write_jsonl(
        task_path,
        ({"sample_id": f"new-{index}"} for index in range(8)),
    )
    second = inference.prepare_model_task_shards(
        task_path,
        model_id="timesfm2.5",
        part_count=2,
        inference_dir=inference_dir,
    )

    assert first["source_task_sha256"] != second["source_task_sha256"]
    assert second["source_task_row_count"] == 8
    assert sum(part["row_count"] for part in second["parts"]) == 8
    assert {
        row["sample_id"]
        for part in second["parts"]
        for row in common.iter_jsonl(Path(part["path"]))
    } == {f"new-{index}" for index in range(8)}


def test_model_predictions_are_merged_only_after_complete_coverage(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    model_id = "timesfm2.5"
    task_path = tmp_path / "tasks.jsonl"
    common.write_jsonl(
        task_path,
        ({"sample_id": f"sample-{index}"} for index in range(12)),
    )
    manifest = inference.prepare_model_task_shards(
        task_path,
        model_id=model_id,
        part_count=2,
        inference_dir=tmp_path,
    )
    for part in manifest["parts"]:
        part_index = int(part["part_index"])
        sample_ids = [row["sample_id"] for row in common.iter_jsonl(Path(part["path"]))]
        part_root = inference.model_part_root(
            tmp_path,
            model_id,
            part_index,
        )
        common.write_jsonl(
            inference.prediction_path_for(part_root, model_id),
            (
                {
                    "model_id": model_id,
                    "sample_id": sample_id,
                    "forecast": [[0.0]],
                }
                for sample_id in sample_ids
            ),
        )

    assert inference.consolidate_model_predictions(tmp_path, manifest)

    canonical = list(
        common.iter_jsonl(
            inference.prediction_path_for(
                inference.model_root(tmp_path, model_id),
                model_id,
            )
        )
    )
    assert [row["sample_id"] for row in canonical] == [
        f"sample-{index}"
        for index in sorted(range(12), key=lambda value: f"sample-{value}")
    ]
    assert not (
        tmp_path / "model_task_shards" / inference.engine.safe_filename(model_id)
    ).exists()
    assert not list(
        (inference.model_root(tmp_path, model_id) / "parts").glob(
            "part_*/predictions/*.jsonl"
        )
    )
    statuses = inference.aggregate_model_statuses(
        [model_id],
        [
            {
                "model_id": model_id,
                "status": "complete",
                "endpoint": "service-a",
                "native_view_count": 2,
            },
            {
                "model_id": model_id,
                "status": "complete",
                "endpoint": "service-b",
                "native_view_count": 2,
            },
        ],
        inference_dir=tmp_path,
        expected_view_count=12,
    )
    assert statuses[0]["status"] == "complete"
    assert statuses[0]["native_view_count"] == 4
    assert statuses[0]["endpoints"] == ["service-a", "service-b"]
