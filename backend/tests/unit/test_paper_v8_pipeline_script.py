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
    assert policy["feature_period"] == round(
        policy["raw_profile_dominant_period"]
    )
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
    assert all(
        8 <= value <= 126
        for value in intermittent["pulse_interval_pattern"]
    )
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
    assert (
        experiment_root / "experiment_manifest.json"
    ).is_file()

    with pytest.raises(ValueError, match="does not match"):
        pipeline.initialize_experiment(
            storage_root=tmp_path,
            experiment_id=experiment_id,
            protocol={**protocol, "seed_count": 32},
            endpoints=["http://service-a"],
        )


def test_response_support_detects_sustained_foldback_without_magic_bound():
    common = load_script("paper_v8_pipeline_common")
    grid = np.linspace(0.0, 1.0, 11)
    response = np.asarray(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.62, 0.55, 0.58]
    )

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
        forcing = (
            np.sin(2.0 * np.pi * time / 24.0 + 0.7)
            + 0.4 * np.sin(2.0 * np.pi * time / 47.0 + 1.2)
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

    lower, lower_lag, candidates = (
        common.v8_nonlinear_conditional_gain(
            rational_delay_series(0.18),
            24,
        )
    )
    upper, upper_lag, repeated_candidates = (
        common.v8_nonlinear_conditional_gain(
            rational_delay_series(0.50),
            24,
        )
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


def test_nonlinear_response_support_uses_lower_pathwise_quantile(
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
        if calibration_seed_index < 2 and lambda_value > 0.2:
            value = 0.2 - 2.0 * (lambda_value - 0.2)
        else:
            value = lambda_value
        return {"nonlinear_conditional_gain": value}, {}

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

    assert grid[-1] == pytest.approx(0.2)
    assert response[-1] > response[0]
    assert audit["path_support_quantile"] == pytest.approx(0.10)
    assert audit["path_support_quantile_lambda"] == pytest.approx(0.2)
    assert audit["effective_lambda_support"] == pytest.approx([0.0, 0.2])
    assert audit["split_half_diagnostic"][
        "triggers_path_expansion"
    ] is False
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
        return grid, response, {
            "effective_lambda_support": [
                float(grid[0]),
                float(grid[-1]),
            ]
        }

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    anchors = [
        {"features": {"nonlinear_conditional_gain": value}}
        for value in np.linspace(0.0, 1.0, 20)
    ]

    calibration = common.calibrate_capabilities(
        common.resolve_dataset("gift_electricity_h"),
        anchors,
        calibration_seed_count=12,
        nonlinear_calibration_seed_count=64,
        capability_ids=["nonlinear_persistence"],
    )
    nonlinear = calibration["capabilities"]["nonlinear_persistence"]

    assert nonlinear["response_calibration_seed_count"] == 64
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
        return grid, response, {
            "effective_lambda_support": [
                float(grid[0]),
                float(grid[-1]),
            ]
        }

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    anchors = [
        {"features": {"curvature_abs": value}}
        for value in np.linspace(0.2, 0.8, 20)
    ]

    calibration = common.calibrate_capabilities(
        common.resolve_dataset("gift_electricity_h"),
        anchors,
        calibration_seed_count=32,
        maximum_calibration_seed_count=96,
        nonlinear_calibration_seed_count=64,
        maximum_nonlinear_calibration_seed_count=128,
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
            "formal_generation_seed_bank_"
            "fixed_base_hard_failure_only_expansion_v2"
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
    target = (
        np.sin(2 * np.pi * time / 24.0) + 0.002 * time
    )[:, None]
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
        common.master_view(master, context)
        for context in common.VIEW_CONTEXT_LENGTHS
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
        target, metadata, covariates = (
            common.generate_deterministic_sample(
                capability_id,
                common.MASTER_LENGTH,
                common.CONTEXT_LENGTH,
                target_dim,
                24,
                5,
                np.random.default_rng(seed),
                conditioning=conditioning,
            )
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
    ) == pytest.approx(
        np.mean(ablated_target[start:stop, channels], axis=0)
    )
    assert np.std(
        clean_target[start:stop, channels],
        axis=0,
    ) == pytest.approx(
        np.std(ablated_target[start:stop, channels], axis=0)
    )


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

    oracle = [
        row for row in selected if row["context_policy"] == "oracle_context"
    ]
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
    oracle = [
        row for row in selected if row["context_policy"] == "oracle_context"
    ]

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
    assert two_batches[
        "mean_pairwise_relative_score_difference"
    ] == pytest.approx(2.0 / 3.0)
    assert one_batch["mean_kendall_tau_b"] is None
    assert one_batch["top1_consistency"] is None
    assert one_batch["mean_top3_overlap"] is None
    assert one_batch[
        "mean_pairwise_relative_score_difference"
    ] is None


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
        item
        for item in comparisons
        if item["comparison_id"] == "secondary_family"
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


def test_inference_prediction_uses_frozen_mase_scale():
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

    expected_mae = float(np.mean(np.arange(8, 12)))
    assert row["metrics"]["mae"] == pytest.approx(expected_mae)
    assert row["metrics"]["mase"] == pytest.approx(expected_mae / 2.0)


def test_tail_model_is_partitioned_across_idle_services(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    task_path = tmp_path / "tasks.jsonl"
    common.write_jsonl(
        task_path,
        ({"sample_id": f"sample-{index}"} for index in range(30)),
    )
    model_ids = ["Chronos-2", "toto2.0", "tirex2", "timesfm2.5"]
    services = [
        (
            f"http://service-{index}",
            {model_id: {"model_id": model_id} for model_id in model_ids},
        )
        for index in range(3)
    ]

    work, assignments, manifest = inference.plan_inference_work(
        model_ids,
        services,
        task_path=task_path,
        inference_dir=tmp_path / "inference",
        enable_tail_sharding=True,
    )

    assert manifest is not None
    tail_model = manifest["model_id"]
    assert tail_model in model_ids
    assert manifest["part_count"] == 3
    assert sum(part["row_count"] for part in manifest["parts"]) == 30
    assert sorted(
        item.tail_part_index
        for items in work.values()
        for item in items
        if item.model_id == tail_model
    ) == [0, 1, 2]
    assert all(
        any(item.model_id == tail_model for item in items)
        for items in work.values()
    )
    assert sum(
        model_id == tail_model
        for model_ids_for_endpoint in assignments.values()
        for model_id in model_ids_for_endpoint
    ) == 1


def test_slow_models_end_distinct_service_queues():
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

    assignments = inference.assign_models(model_ids, services)

    slow_locations = {
        model_id: endpoint
        for endpoint, queue in assignments.items()
        for model_id in inference.SLOW_TAIL_MODELS
        if model_id in queue
    }
    assert len(set(slow_locations.values())) == 3
    assert {
        queue[-1] for queue in assignments.values()
    } == set(inference.SLOW_TAIL_MODELS)
    assert all(
        queue[0] not in inference.SLOW_TAIL_MODELS
        for queue in assignments.values()
    )


def test_tail_shards_refresh_when_source_task_changes(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    task_path = tmp_path / "tasks.jsonl"
    inference_dir = tmp_path / "inference"
    common.write_jsonl(
        task_path,
        ({"sample_id": f"old-{index}"} for index in range(6)),
    )
    first = inference.prepare_tail_task_shards(
        task_path,
        model_id="timesfm2.5",
        part_count=2,
        inference_dir=inference_dir,
    )

    common.write_jsonl(
        task_path,
        ({"sample_id": f"new-{index}"} for index in range(8)),
    )
    second = inference.prepare_tail_task_shards(
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


def test_tail_predictions_are_merged_only_after_complete_coverage(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    model_id = "timesfm2.5"
    model_root = tmp_path / "model_shards" / "timesfm2_5"
    manifest = {
        "model_id": model_id,
        "source_task_row_count": 4,
        "parts": [
            {"part_index": 0, "row_count": 2},
            {"part_index": 1, "row_count": 2},
        ],
    }
    for part_index, sample_ids in enumerate(
        (("sample-0", "sample-2"), ("sample-1", "sample-3"))
    ):
        part_root = (
            model_root / "tail_parts" / f"part_{part_index:03d}"
        )
        common.write_jsonl(
            inference.prediction_path_for(part_root, model_id),
            (
                {"model_id": model_id, "sample_id": sample_id}
                for sample_id in sample_ids
            ),
        )

    inference.consolidate_tail_predictions(tmp_path, manifest)

    canonical = list(
        common.iter_jsonl(
            inference.prediction_path_for(model_root, model_id)
        )
    )
    assert [row["sample_id"] for row in canonical] == [
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
    ]
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
        expected_view_count=4,
    )
    assert statuses[0]["status"] == "complete"
    assert statuses[0]["native_view_count"] == 4
    assert statuses[0]["endpoints"] == ["service-a", "service-b"]
