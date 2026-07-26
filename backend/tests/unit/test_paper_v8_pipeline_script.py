from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
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

    window = common.REAL_FORECAST_MASTER_LENGTH
    strata = common.nonoverlapping_strata(2 * window + 37)

    assert len(strata) == 2
    assert all(upper >= lower for lower, upper in strata)
    assert strata[0][1] + window <= strata[1][0]


def test_direct_anchor_summary_does_not_compress_feature_values():
    common = load_script("paper_v8_pipeline_common")
    anchor = {"acf1": 0.93, "slope_abs": 0.17}

    summary = common.anchor_summary(anchor)

    assert summary == {
        "acf1": {"p50": 0.93},
        "slope_abs": {"p50": 0.17},
    }


def test_v8_registry_includes_forecastable_l168_datasets():
    common = load_script("paper_v8_pipeline_common")

    assert len(common.DATASET_REGISTRY) == 21
    assert {
        "gift_bizitobs_application",
        "gift_bizitobs_service",
        "gift_restaurant_d",
        "gift_hierarchical_sales_d",
        "gift_m4_hourly",
        "gift_us_births_d",
        "gift_saugeenday_d",
        "gift_temperature_rain_d",
        "m5_daily",
    }.issubset(common.DATASET_REGISTRY)
    assert common.resolve_dataset("m5_daily").real_data_adapter == "m5_csv"


def test_v8_window_contract_separates_real_calibration_and_synthetic_master():
    common = load_script("paper_v8_pipeline_common")

    assert common.REAL_CALIBRATION_CONTEXT_LENGTH == 168
    assert common.CONTEXT_LENGTH == 336
    assert common.HORIZON == 48
    assert common.REAL_FORECAST_MASTER_LENGTH == 216
    assert common.MASTER_LENGTH == 384
    assert common.VIEW_CONTEXT_LENGTHS == (96, 168, 336)
    assert common.FIXED_CONTEXT_LENGTH == 168


def test_v8_features_use_local_trend_and_history_only():
    common = load_script("paper_v8_pipeline_common")
    time = np.linspace(-1.0, 1.0, common.MASTER_LENGTH)
    target = (0.4 * time + 0.8 * time**2)[:, None]
    changed_future = target.copy()
    changed_future[common.CONTEXT_LENGTH :] += 1000.0

    first = common.measured_features(
        "trend",
        target,
        None,
        season_length=24,
    )
    second = common.measured_features(
        "trend",
        changed_future,
        None,
        season_length=24,
    )

    assert common.PRIMARY_TARGET_FEATURE["trend"] == (
        "local_polynomial_energy_share_w96"
    )
    assert first == second
    assert first["local_polynomial_energy_share_w96"] > 0.0
    assert first["v8_feature_history_length"] == common.CONTEXT_LENGTH


def test_v8_multi_period_feature_accumulates_noncarrier_spectral_energy():
    features = load_script("paper_v8_features")
    time = np.arange(336, dtype=float)
    carrier = np.sin(2.0 * np.pi * time / 24.0)
    weak_multi = carrier + 0.15 * np.sin(2.0 * np.pi * time / 16.0)
    strong_multi = (
        carrier
        + 0.55 * np.sin(2.0 * np.pi * time / 16.0)
        + 0.45 * np.sin(2.0 * np.pi * time / 40.0)
    )

    carrier_score = features.v8_feature_vector(
        carrier,
        season_length=24,
    )["multi_period_score"]
    weak_score = features.v8_feature_vector(
        weak_multi,
        season_length=24,
    )["multi_period_score"]
    strong_score = features.v8_feature_vector(
        strong_multi,
        season_length=24,
    )["multi_period_score"]

    assert carrier_score < weak_score < strong_score


def test_v8_multi_period_feature_rejects_quadratic_trend():
    features = load_script("paper_v8_features")
    time = np.linspace(-1.0, 1.0, 336)
    trend = 0.7 * time + 1.8 * time**2

    score = features.v8_feature_vector(
        trend,
        season_length=24,
    )["multi_period_score"]

    assert score < 1e-6


def test_v8_local_curvature_removes_carriers_without_hiding_trend_dose():
    features = load_script("paper_v8_features")
    observations = 336
    index = np.arange(observations, dtype=float)
    local_time = np.clip(
        (index - (observations - 96)) / 95.0,
        0.0,
        None,
    )
    trend_low = 0.01 * index + 0.1 * local_time**2
    trend_high = 0.01 * index + 0.8 * local_time**2
    multi_low = (
        np.sin(2.0 * np.pi * index / 24.0)
        + 0.1 * np.sin(2.0 * np.pi * index / 16.0)
        + 0.08 * np.sin(2.0 * np.pi * index / 40.0)
    )
    multi_high = (
        np.sin(2.0 * np.pi * index / 24.0)
        + 0.6 * np.sin(2.0 * np.pi * index / 16.0)
        + 0.48 * np.sin(2.0 * np.pi * index / 40.0)
    )
    modulated_low = (
        1.0 + 0.05 * np.sin(2.0 * np.pi * index / 48.0)
    ) * np.sin(2.0 * np.pi * index / 16.0)
    modulated_high = (
        1.0 + 0.6 * np.sin(2.0 * np.pi * index / 48.0)
    ) * np.sin(2.0 * np.pi * index / 16.0)

    def curvature(values, period):
        return features.v8_feature_vector(
            values,
            season_length=period,
        )["local_curvature_abs_w96"]

    trend_delta = abs(
        curvature(trend_high, 24) - curvature(trend_low, 24)
    )
    multi_delta = abs(
        curvature(multi_high, 24) - curvature(multi_low, 24)
    )
    modulation_delta = abs(
        curvature(modulated_high, 16)
        - curvature(modulated_low, 16)
    )

    assert curvature(trend_high, 24) > curvature(trend_low, 24)
    assert multi_delta < 0.5 * trend_delta
    assert modulation_delta < 0.5 * trend_delta


def test_v8_transition_sparsity_requires_material_unexplained_changes():
    features = load_script("paper_v8_features")
    observations = 336
    index = np.arange(observations, dtype=float)
    carrier = np.sin(2.0 * np.pi * index / 24.0)
    regime_state = np.where((index // 24) % 2 == 0, -1.0, 1.0)
    regime_low = carrier + 0.05 * regime_state
    regime_high = carrier + 0.6 * regime_state
    multi_low = (
        carrier
        + 0.1 * np.sin(2.0 * np.pi * index / 16.0)
        + 0.08 * np.sin(2.0 * np.pi * index / 40.0)
    )
    multi_high = (
        carrier
        + 0.6 * np.sin(2.0 * np.pi * index / 16.0)
        + 0.48 * np.sin(2.0 * np.pi * index / 40.0)
    )
    modulated_low = (
        1.0 + 0.05 * np.sin(2.0 * np.pi * index / 48.0)
    ) * np.sin(2.0 * np.pi * index / 16.0)
    modulated_high = (
        1.0 + 0.6 * np.sin(2.0 * np.pi * index / 48.0)
    ) * np.sin(2.0 * np.pi * index / 16.0)

    def sparsity(values, period):
        return features.v8_feature_vector(
            values,
            season_length=period,
        )["regime_sparse_transition_score"]

    regime_delta = abs(
        sparsity(regime_high, 24) - sparsity(regime_low, 24)
    )
    multi_delta = abs(
        sparsity(multi_high, 24) - sparsity(multi_low, 24)
    )
    modulation_delta = abs(
        sparsity(modulated_high, 16)
        - sparsity(modulated_low, 16)
    )

    assert sparsity(regime_high, 24) > sparsity(regime_low, 24)
    assert multi_delta < 0.5 * regime_delta
    assert modulation_delta < 0.5 * regime_delta


def test_calibration_anchor_is_forecastable_and_preserves_native_panel(
    monkeypatch,
    tmp_path,
):
    common = load_script("paper_v8_pipeline_common")
    dataset = common.resolve_dataset("gift_ett1_h")
    time = np.arange(500, dtype=float)
    panel = np.vstack(
        [
            np.sin(2.0 * np.pi * time / 24.0),
            np.cos(2.0 * np.pi * time / 24.0),
            np.sin(2.0 * np.pi * (time - 3.0) / 24.0),
        ]
    )
    monkeypatch.setattr(
        common,
        "load_real_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(
            frequency="H",
            records=(
                common.RealSeriesRecord(
                    item_id="panel",
                    values=panel,
                ),
            ),
            asset_files=(),
            adapter_id="gift_arrow",
            metadata={},
        ),
    )

    anchors, metadata = common.build_calibration_anchors(
        dataset,
        gift_eval_dir=tmp_path,
        maximum_anchors=2,
    )

    assert len(anchors) == 2
    assert metadata["feature_profiles"]["native_multivariate"]
    assert not metadata["feature_profiles"]["hierarchy_children"]
    for anchor in anchors:
        master = anchor["real_forecast_master"]
        target = np.asarray(master["target"], dtype=float)
        assert target.shape == (216, 1)
        assert master["context_length"] == 168
        assert master["horizon"] == 48
        assert np.mean(target[:168]) == pytest.approx(0.0, abs=1e-12)
        assert np.std(target[:168]) == pytest.approx(1.0)
        assert anchor["native_multivariate_features"][
            "pca_top1_explained"
        ] > 0.0
        assert anchor["feature_provenance"]["pca_top1_explained"] == (
            "real_native_multivariate_history_l168"
        )
        assert not anchor["hierarchy_children_features"]
        assert not anchor["feature_provenance_by_scope"][
            "real_hierarchy_children"
        ]


def test_m5_adapter_extracts_all_declared_structural_scopes(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    day_count = 500
    days = [f"d_{index}" for index in range(1, day_count + 1)]
    dates = pd.date_range("2011-01-29", periods=day_count, freq="D")
    calendar = pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": 11101 + np.arange(day_count) // 7,
            "weekday": dates.day_name(),
            "wday": dates.dayofweek + 1,
            "month": dates.month,
            "year": dates.year,
            "d": days,
            "event_name_1": [
                "event" if index % 31 == 0 else None
                for index in range(day_count)
            ],
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": (np.arange(day_count) % 10 == 0).astype(int),
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )
    calendar.to_csv(tmp_path / "calendar.csv", index=False)

    rows = []
    time = np.arange(day_count, dtype=float)
    for index in range(10):
        department = "HOBBIES_1" if index < 5 else "HOBBIES_2"
        values = np.maximum(
            0.0,
            np.round(
                3.0
                + 0.3 * index
                + 1.5 * np.sin(2.0 * np.pi * (time - index) / 7.0)
                + 0.4 * np.sin(2.0 * np.pi * time / (11.0 + index))
                + (time % (13 + index) == 0),
            ),
        ).astype(int)
        row = {
            "id": f"item_{index}_CA_1_evaluation",
            "item_id": f"item_{index}",
            "dept_id": department,
            "cat_id": "HOBBIES",
            "store_id": "CA_1",
            "state_id": "CA",
        }
        row.update(dict(zip(days, values, strict=True)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        tmp_path / "sales_train_evaluation.csv",
        index=False,
    )

    dataset = common.resolve_dataset("m5_daily")
    anchors, metadata = common.build_calibration_anchors(
        dataset,
        source_root=tmp_path,
        maximum_anchors=2,
    )

    assert len(anchors) == 2
    assert metadata["real_data_adapter"] == "m5_csv"
    assert metadata["adapter_metadata"]["known_future_covariate_columns"]
    for anchor in anchors:
        assert anchor["native_multivariate_features"]
        assert anchor["known_future_covariate_features"]
        assert anchor["declared_hierarchy_features"]
        assert anchor["hierarchy_children_features"]
        assert anchor["hierarchy_children_features"][
            "hierarchy_residual_mean_abs"
        ] == pytest.approx(0.0, abs=1e-12)
        assert anchor["covariate_column_names"] == [
            "day_of_week_sin",
            "day_of_week_cos",
            "event_count",
            "snap",
        ]


def test_parameter_mapping_records_real_and_fallback_sources():
    common = load_script("paper_v8_pipeline_common")
    anchor = {
        "features": {"acf1": 0.8},
        "native_multivariate_features": {"factor_score_acf1": 0.7},
        "hierarchy_children_features": {"hierarchy_aggregate_acf1": 0.9},
        "known_future_covariate_features": {"covariate_incremental_r2": 0.2},
        "feature_support": {
            "acf1": {"usable": True},
            "factor_score_acf1": {"usable": True},
            "hierarchy_aggregate_acf1": {"usable": True},
            "covariate_incremental_r2": {"usable": True},
        },
    }
    mappings = common.parameter_mapping_provenance(
        [
            {"parameter": "profile_acf1", "source_feature": "acf1"},
            {
                "parameter": "factor_persistence",
                "source_feature": "factor_score_acf1",
            },
        ],
        anchor,
        capability_id="common_factor",
    )

    assert mappings[0]["source_status"] == "real_univariate"
    assert mappings[0]["fallback_used"] is False
    assert mappings[1]["source_status"] == "real_native_multivariate"
    assert mappings[1]["fallback_used"] is False

    hierarchy_mapping = common.parameter_mapping_provenance(
        [
            {
                "parameter": "aggregate_persistence",
                "source_feature": "hierarchy_aggregate_acf1",
            }
        ],
        anchor,
        capability_id="hierarchical_coherence",
    )
    assert hierarchy_mapping[0]["source_status"] == "real_hierarchy_children"
    covariate_mapping = common.parameter_mapping_provenance(
        [
            {
                "parameter": "covariate_effect_scale",
                "source_feature": "covariate_incremental_r2",
            }
        ],
        anchor,
        capability_id="covariate_response",
    )
    assert covariate_mapping[0]["source_status"] == (
        "real_known_future_covariates"
    )

    unavailable = common.parameter_mapping_provenance(
        [
            {
                "parameter": "factor_persistence",
                "source_feature": "factor_score_acf1",
            }
        ],
        {**anchor, "feature_support": {"factor_score_acf1": {"usable": False}}},
        capability_id="common_factor",
    )
    assert unavailable[0]["source_status"] == "protocol_fallback"
    assert "factor_score_acf1" in unavailable[0]["fallback_reason"]


def test_10s_period_policy_separates_calendar_feature_and_mase_periods():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.REAL_CALIBRATION_CONTEXT_LENGTH, dtype=float)
    history = np.sin(2.0 * np.pi * time / 42.0)

    policy = common.calibration_period_policy("10S", history)

    assert policy["calendar_season_length"] == 8640
    assert policy["calendar_season_feature_observable"] is False
    assert 38 <= policy["raw_profile_dominant_period"] <= 46
    assert policy["feature_period"] == round(policy["raw_profile_dominant_period"])
    assert policy["mase_period"] == 1


def test_hourly_period_policy_keeps_calendar_period_for_features_and_mase():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.REAL_CALIBRATION_CONTEXT_LENGTH, dtype=float)
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

    assert 24 <= multi["effective_primary_period"] <= 48
    assert max(multi["periods"]) <= 48
    assert 8 <= time_varying["primary_period"] <= 84
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


def test_generation_retries_a_numerically_invalid_candidate(monkeypatch):
    generation = load_script("generate_paper_v8_samples")
    dataset = SimpleNamespace(dataset_id="test_dataset")
    anchor = {"anchor_id": "anchor-0"}
    attempted: list[int] = []

    monkeypatch.setattr(
        generation.v8,
        "anchor_for_seed",
        lambda *args, **kwargs: anchor,
    )

    def fake_clean_seed_bundle(*args, **kwargs):
        attempt = int(kwargs["generation_attempt"])
        attempted.append(attempt)
        if attempt == 0:
            raise ValueError("collapsed candidate")
        return [
            {
                "sample_id": "accepted-sample",
                "capability_id": "trend",
                "evaluation_table": "main",
                "parameter_sampling": {
                    "path_seed": generation.generation_path_seed(
                        dataset.dataset_id,
                        "trend",
                        0,
                        attempt,
                    )
                },
            }
        ]

    monkeypatch.setattr(
        generation,
        "clean_seed_bundle",
        fake_clean_seed_bundle,
    )
    monkeypatch.setattr(
        generation.realism,
        "evaluate_sample",
        lambda *args, **kwargs: {
            "accepted": True,
            "failure_codes": [],
        },
    )
    audits: list[dict] = []

    rows = list(
        generation.iter_clean_samples(
            dataset,
            [anchor],
            {"capabilities": {"trend": {}}},
            capability_ids=("trend",),
            seed_indexes=[0],
            sensitivity_seeds=set(),
            gate_context=object(),
            max_generation_attempts=2,
            attempt_audits=audits,
        )
    )

    assert attempted == [0, 1]
    assert rows[0]["generation_attempt"] == 1
    assert audits[0]["selected_attempt"] == 1
    assert audits[0]["attempts"][0]["failed_samples"][0][
        "failure_codes"
    ] == ["candidate_generation_error"]


def test_generation_skips_capabilities_without_real_calibrated_grid():
    generation = load_script("generate_paper_v8_samples")
    calibration = {
        "capabilities": {
            "trend": {
                "available_for_generation": True,
                "availability_status": "available",
                "unavailable_reason_codes": [],
                "intensity_calibration_scope": (
                    "dataset_real_generator_overlap_reference"
                ),
            },
            "nonlinear_persistence": {
                "available_for_generation": False,
                "availability_status": "unavailable",
                "unavailable_reason_codes": [
                    "insufficient_finite_real_anchor_features"
                ],
                "intensity_calibration_scope": (
                    "dataset_real_calibration_unavailable"
                ),
            },
        }
    }

    available, unavailable = generation.resolve_generation_capabilities(
        calibration,
        ("trend", "nonlinear_persistence"),
    )

    assert available == ("trend",)
    assert unavailable == [
        {
            "capability_id": "nonlinear_persistence",
            "availability_status": "unavailable",
            "reason_codes": [
                "insufficient_finite_real_anchor_features"
            ],
            "intensity_calibration_scope": (
                "dataset_real_calibration_unavailable"
            ),
        }
    ]


def test_real_intensity_sources_are_capability_specific():
    common = load_script("paper_v8_pipeline_common")
    anchors = [
        {
            "features": {
                "local_polynomial_energy_share_w96": float(index),
            },
            "native_multivariate_features": {
                "pca_top1_explained": float(index),
            },
            "hierarchy_children_features": {
                "hierarchy_child_heterogeneity": float(index),
            },
            "known_future_covariate_features": {
                "covariate_incremental_r2": float(index),
            },
        }
        for index in range(12)
    ]

    expected_scopes = {
        "trend": "real_univariate",
        "common_factor": "real_native_multivariate",
        "hierarchical_coherence": "real_hierarchy_children",
        "covariate_response": "real_known_future_covariates",
    }
    for capability_id, expected_scope in expected_scopes.items():
        summary, audit = common.real_intensity_feature_summary(
            anchors,
            capability_id=capability_id,
        )
        assert summary is not None
        assert audit["usable"] is True
        assert audit["scope"] == expected_scope

    summary, audit = common.real_intensity_feature_summary(
        anchors,
        capability_id="cross_series_dependence",
    )
    assert summary is None
    assert audit["scope"] == "real_native_multivariate"
    assert audit["reason_code"] == (
        "insufficient_finite_real_anchor_features"
    )


def test_covariate_features_require_explicit_known_future_declaration():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.REAL_CALIBRATION_CONTEXT_LENGTH, dtype=float)
    target = np.sin(time / 7.0)
    covariates = np.column_stack([np.cos(time / 11.0)])
    arguments = {
        "target_values": target,
        "covariates": covariates,
        "channel_index": 0,
        "start": 0,
        "feature_period": 24,
        "minimum_observed_fraction": 0.5,
    }

    assert common._known_future_covariate_features(
        **arguments,
        covariate_kind=None,
    ) == {}
    declared = common._known_future_covariate_features(
        **arguments,
        covariate_kind="known_future",
    )
    assert "covariate_incremental_r2" in declared


def test_parallel_calibration_merge_rebuilds_availability_summary():
    calibration_script = load_script("calibrate_paper_v8")
    results = {
        "trend": {
            "schema_version": "test",
            "available_capabilities": ["trend"],
            "unavailable_capabilities": {},
            "capabilities": {
                "trend": {
                    "available_for_generation": True,
                    "unavailable_reason_codes": [],
                }
            },
        },
        "nonlinear_persistence": {
            "schema_version": "test",
            "available_capabilities": [],
            "unavailable_capabilities": {
                "nonlinear_persistence": ["missing_real_feature"]
            },
            "capabilities": {
                "nonlinear_persistence": {
                    "available_for_generation": False,
                    "unavailable_reason_codes": ["missing_real_feature"],
                }
            },
        },
    }

    merged = calibration_script.merge_capability_calibrations(
        results,
        ("trend", "nonlinear_persistence"),
    )

    assert merged["available_capabilities"] == ["trend"]
    assert merged["unavailable_capabilities"] == {
        "nonlinear_persistence": ["missing_real_feature"]
    }


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


def test_pipeline_routes_multi_dataset_inference_through_model_major_controller(
    tmp_path,
):
    pipeline = load_script("run_paper_v8_pipeline")
    args = SimpleNamespace(
        seed_start=0,
        seed_count=64,
        models=["Chronos-2", "toto2.0"],
        endpoints=[
            "http://127.0.0.1:10810",
            "http://192.168.99.89:10810",
        ],
        devices="0,1",
        endpoint_preset=[
            "http://192.168.99.89:10810=rtx5090x8-h48-b1-v1"
        ],
        endpoint_devices=[],
        endpoint_capacity=[],
        endpoint_concurrency_scale=[],
        endpoint_model_capacity=[],
        endpoint_model_concurrency=[],
        inference_preprocess_workers=16,
        resume_inference=True,
    )

    arguments = pipeline.model_major_inference_arguments(
        args,
        ["gift_electricity_h", "gift_solar_h"],
        experiment_root=tmp_path,
    )

    assert "--dataset-id" not in arguments
    dataset_option = arguments.index("--dataset-ids")
    assert arguments[dataset_option + 1 : dataset_option + 3] == [
        "gift_electricity_h",
        "gift_solar_h",
    ]
    preprocess_option = arguments.index("--preprocess-workers")
    assert arguments[preprocess_option + 1] == "16"
    assert "--resume" in arguments
    assert (
        "http://192.168.99.89:10810=rtx5090x8-h48-b1-v1"
        in arguments
    )


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


def test_nonlinear_response_uses_observable_proxy_not_construction_dose(
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
        qualification_path_index,
    ):
        del dataset, anchor, capability_id, family_role
        return {
            # Deliberately reversed: generator metadata must not define the
            # real-data intensity coordinate.
            "nonlinear_strength": 1.0 - lambda_value,
            "nonlinear_conditional_gain": lambda_value,
            "nonlinear_conditional_effect_size": lambda_value,
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


def test_real_nonlinear_observable_dose_enters_generator_qualification(
    monkeypatch,
):
    common = load_script("paper_v8_pipeline_common")
    response_called = False

    def response_curve(
        dataset,
        anchors,
        *,
        capability_id,
        family_role,
        calibration_seed_count,
    ):
        nonlocal response_called
        response_called = True
        del dataset, anchors, capability_id
        del family_role
        grid = np.asarray([0.0, 1.0])
        response = np.asarray([0.0, 1.0])
        return (
            grid,
            response,
            {
                "effective_lambda_support": [
                    float(grid[0]),
                    float(grid[-1]),
                ],
                "raw_lambda_grid": grid.tolist(),
                "per_path_raw_response_curves": [
                    response.tolist()
                    for _ in range(calibration_seed_count)
                ],
            },
        )

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    anchors = [
        {"features": {"nonlinear_conditional_effect_size": value}}
        for value in np.linspace(0.0, 1.0, 20)
    ]

    calibration = common.calibrate_capabilities(
        common.resolve_dataset("gift_electricity_h"),
        anchors,
        calibration_seed_count=32,
        capability_ids=["nonlinear_persistence"],
    )
    nonlinear = calibration["capabilities"]["nonlinear_persistence"]

    assert response_called is True
    assert nonlinear["available_for_generation"] is True
    assert nonlinear["qualification_path_count"] == 32
    assert nonlinear["unavailable_reason_codes"] == []
    assert nonlinear["primary"] is not None
    assert calibration["available_capabilities"] == [
        "nonlinear_persistence"
    ]
    assert calibration["unavailable_capabilities"] == {}


def test_secondary_family_cannot_fall_back_when_real_grid_is_compressed(
    monkeypatch,
):
    common = load_script("paper_v8_pipeline_common")

    def response_curve(*args, family_role, **kwargs):
        del args, kwargs
        grid = np.linspace(0.0, 1.0, 21)
        response = grid if family_role == "primary" else 10.0 * grid
        return grid, response, {
            "effective_lambda_support": [0.0, 1.0],
            "raw_lambda_grid": grid.tolist(),
            "per_path_raw_response_curves": [response.tolist()] * 32,
        }

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    anchors = [
        {
            "features": {
                "local_polynomial_energy_share_w96": float(value)
            }
        }
        for value in np.linspace(0.0, 1.0, 20)
    ]

    calibration = common.calibrate_capabilities(
        common.resolve_dataset("gift_electricity_h"),
        anchors,
        calibration_seed_count=32,
        capability_ids=["trend"],
    )
    trend = calibration["capabilities"]["trend"]

    assert trend["available_for_generation"] is False
    assert trend["unavailable_reason_codes"] == [
        "real_reference_maps_to_insufficient_secondary_lambda_span"
    ]
    assert trend["secondary"]["selected_lambdas"] != pytest.approx(
        np.linspace(0.0, 1.0, 5)
    )


def test_structural_gate_unreachable_marks_calibration_cell_unavailable(
    monkeypatch,
):
    common = load_script("paper_v8_pipeline_common")

    def response_curve(*args, **kwargs):
        del args, kwargs
        grid = np.linspace(0.0, 1.0, 21)
        return grid, grid, {
            "effective_lambda_support": [0.0, 1.0],
            "raw_lambda_grid": grid.tolist(),
            "per_path_raw_response_curves": [grid.tolist()] * 32,
        }

    observed: dict[str, object] = {}

    def reachability(*args, **kwargs):
        del args
        observed.update(kwargs)
        return {
            "accepted": False,
            "reason_codes": ["selected_i5_structural_gate_unreachable"],
            "near_distance_evaluated": False,
        }

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    monkeypatch.setattr(
        common,
        "structural_calibration_reachability",
        reachability,
    )
    anchors = [
        {
            "native_multivariate_features": {
                "cross_series_incremental_r2": float(value)
            },
            "features": {},
        }
        for value in np.linspace(0.2, 0.8, 20)
    ]

    calibration = common.calibrate_capabilities(
        common.resolve_dataset("gift_ett1_h"),
        anchors,
        calibration_seed_count=32,
        capability_ids=["cross_series_dependence"],
    )
    record = calibration["capabilities"]["cross_series_dependence"]

    assert observed["family_role"] == "primary"
    assert observed["calibration_seed_count"] == 32
    assert observed["lambda_value"] == pytest.approx(
        record["primary"]["selected_lambdas"][-1]
    )
    assert record["available_for_generation"] is False
    assert record["unavailable_reason_codes"] == [
        "selected_i5_structural_gate_unreachable"
    ]
    assert record["structural_gate_reachability"]["accepted"] is False
    assert calibration["available_capabilities"] == []


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
                ],
                "raw_lambda_grid": grid.tolist(),
                "per_path_raw_response_curves": [
                    response.tolist()
                    for _ in range(calibration_seed_count)
                ],
            },
        )

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    anchors = [
        {
            "features": {
                "local_polynomial_energy_share_w96": value
            }
        }
        for value in np.linspace(0.2, 0.8, 20)
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
    assert trend["qualification_path_count"] == 64
    assert trend["qualification_path_policy"] == {
        "policy": (
            "independent_family_response_qualification_bank_"
            "fixed_base_hard_failure_only_expansion_v1"
        ),
        "path_sampling": {
            "anchor": "independent_qualification_anchor_hash_v1",
            "rng": "independent_qualification_path_v1",
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


def test_family_mean_inverse_uses_real_generator_overlap_without_seed_inverse(
    monkeypatch,
):
    common = load_script("paper_v8_pipeline_common")

    def response_curve(*args, family_role, **kwargs):
        del args, kwargs, family_role
        grid = np.linspace(0.0, 1.0, 21)
        return grid, grid, {
            "effective_lambda_support": [0.0, 1.0],
            "raw_lambda_grid": grid.tolist(),
            "per_path_raw_response_curves": [grid.tolist()] * 32,
        }

    monkeypatch.setattr(common, "monotone_response_curve", response_curve)
    anchors = [
        {
            "features": {
                "local_polynomial_energy_share_w96": float(value)
            }
        }
        for value in np.linspace(0.2, 0.8, 20)
    ]
    calibration = common.calibrate_capabilities(
        common.resolve_dataset("gift_ett1_h"),
        anchors,
        calibration_seed_count=32,
        capability_ids=["trend"],
    )
    trend = calibration["capabilities"]["trend"]

    assert trend["intensity_calibration_scope"] == (
        "dataset_real_generator_overlap_reference"
    )
    assert trend["real_alignment_reference"]["formal_seed_inverse"] is False
    assert trend["real_alignment_reference"][
        "sample_level_alignment_enforced"
    ] is False
    assert trend["primary"]["selected_lambdas"] == pytest.approx(
        trend["primary"]["selected_target_values"]
    )


def test_intermittency_measured_dose_comes_from_history_feature():
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
        == "event_positive_residual_energy_share"
    )
    direct = common.v8_feature_vector(
        target[: common.CONTEXT_LENGTH],
        int(metadata["event_period"]),
    )
    assert features["event_positive_residual_energy_share"] == pytest.approx(
        direct["event_positive_residual_energy_share"]
    )
    assert "event_effect_energy_share" in features


def test_master_views_share_exact_future_and_l336_mase_scale():
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
        == "slice_exact_l336_standardized_master_without_restandardization"
        for view in views
    )


def test_exact_seasonal_mase_scale_uses_recorded_lag_one_fallback():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.MASTER_LENGTH, dtype=float)
    target = np.sin(2.0 * np.pi * time / 24.0)[:, None]

    policy = common.mase_scale_policy(target, season_length=24)
    scale, scale_by_target = common.mase_scales(
        target,
        season_length=24,
    )

    expected = float(
        np.mean(np.abs(np.diff(target[: common.CONTEXT_LENGTH, 0])))
    )
    assert policy["requested_period"] == 24
    assert policy["effective_period_by_target"] == [1]
    assert policy["fallback_target_indices"] == [0]
    assert policy["scale"] == pytest.approx(expected)
    assert scale == pytest.approx(expected)
    assert scale_by_target == pytest.approx([expected])


def test_robustness_sample_uses_clean_l336_mase_scale_by_target():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.MASTER_LENGTH, dtype=float)
    target = np.column_stack(
        (
            0.01 * time,
            np.sin(2.0 * np.pi * time / 24.0),
        )
    )
    scale, scale_by_target = common.mase_scales(
        target,
        season_length=24,
    )
    clean = {
        "schema_version": "test",
        "sample_id": "v8__noise_scale",
        "master_sample_id": "v8__noise_scale",
        "paired_group_id": "v8__noise_scale",
        "counterfactual_pair_id": None,
        "capability_id": "trend",
        "target": target.tolist(),
        "covariates": None,
        "mase_scale": scale,
        "mase_scale_by_target": scale_by_target,
    }

    robust = common.robustness_sample(clean)
    observed = np.asarray(robust["target"], dtype=float)
    applied_noise_std = np.std(
        observed[: common.CONTEXT_LENGTH]
        - target[: common.CONTEXT_LENGTH],
        axis=0,
    )
    metadata = robust["observation_noise_metadata"]

    assert robust["schema_version"] == (
        "paper_v8_robustness_master_sample.v2"
    )
    assert robust["observation_noise_scale_policy"] == (
        "ratio_times_clean_l336_mase_denominator_by_target"
    )
    assert metadata["noise_scale_source"] == (
        "clean_l336_mase_denominator_by_target"
    )
    assert metadata["requested_noise_scale_by_target"] == pytest.approx(
        scale_by_target
    )
    assert applied_noise_std == pytest.approx(
        common.ROBUSTNESS_NOISE_RATIO * np.asarray(scale_by_target),
        rel=0.15,
    )
    assert np.array_equal(
        observed[common.CONTEXT_LENGTH :],
        target[common.CONTEXT_LENGTH :],
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
            {"cut_points": [312, 342]},
            "cut_points",
            [72, 102],
        ),
        (
            "predictable_intermittency",
            {"pulse_centers": [312, 352]},
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


def test_experiment_capability_rows_keep_fixed_and_oracle_separate():
    analysis = load_script("analyze_paper_v8")
    rows = []
    for dataset_id, offset in (("first", 0.0), ("second", 0.2)):
        for policy, model_values in (
            ("fixed_l168", {"a": (1.0, 1), "b": (2.0, 2)}),
            ("oracle_context", {"a": (0.8, 2), "b": (0.7, 1)}),
        ):
            for model_id, (accuracy, rank) in model_values.items():
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "context_policy": policy,
                        "evaluation_table": "main",
                        "generator_family_role": "primary",
                        "capability_id": "trend",
                        "model_id": model_id,
                        "accuracy_score": accuracy + offset,
                        "history_std_normalized_mae": accuracy + offset,
                        "accuracy_rank": rank,
                        "mechanism_score": accuracy + offset + 0.5,
                        "mechanism_rank": rank,
                    }
                )

    fixed = analysis.experiment_capability_rows(
        rows,
        context_policy="fixed_l168",
        dataset_ids=["first", "second"],
        models=["a", "b"],
        capabilities=["trend"],
    )
    oracle = analysis.experiment_capability_rows(
        rows,
        context_policy="oracle_context",
        dataset_ids=["first", "second"],
        models=["a", "b"],
        capabilities=["trend"],
    )

    fixed_by_model = {row["model_id"]: row for row in fixed}
    oracle_by_model = {row["model_id"]: row for row in oracle}
    assert fixed_by_model["a"]["accuracy_rank"] == 1
    assert fixed_by_model["a"]["macro_mean_accuracy_score"] == pytest.approx(
        1.1
    )
    assert oracle_by_model["b"]["accuracy_rank"] == 1
    assert oracle_by_model["b"]["macro_mean_accuracy_score"] == pytest.approx(
        0.8
    )
    assert {row["context_policy"] for row in fixed} == {"fixed_l168"}
    assert {row["context_policy"] for row in oracle} == {"oracle_context"}


def test_experiment_capability_rows_reject_incomplete_policy_coverage():
    analysis = load_script("analyze_paper_v8")

    with pytest.raises(ValueError, match="coverage mismatch"):
        analysis.experiment_capability_rows(
            [],
            context_policy="oracle_context",
            dataset_ids=["dataset"],
            models=["model"],
            capabilities=["trend"],
        )


def test_experiment_capability_rows_use_capability_specific_dataset_support():
    analysis = load_script("analyze_paper_v8")
    rows = []
    for dataset_id in ("first", "second"):
        for model_id, rank in (("a", 1), ("b", 2)):
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "context_policy": "fixed_l168",
                    "evaluation_table": "main",
                    "generator_family_role": "primary",
                    "capability_id": "trend",
                    "model_id": model_id,
                    "accuracy_score": float(rank),
                    "history_std_normalized_mae": float(rank),
                    "accuracy_rank": rank,
                    "mechanism_score": float(rank),
                    "mechanism_rank": rank,
                }
            )
    for model_id, rank in (("a", 2), ("b", 1)):
        rows.append(
            {
                "dataset_id": "second",
                "context_policy": "fixed_l168",
                "evaluation_table": "main",
                "generator_family_role": "primary",
                "capability_id": "common_factor",
                "model_id": model_id,
                "accuracy_score": float(rank),
                "history_std_normalized_mae": float(rank),
                "accuracy_rank": rank,
                "mechanism_score": float(rank),
                "mechanism_rank": rank,
            }
        )

    result = analysis.experiment_capability_rows(
        rows,
        context_policy="fixed_l168",
        dataset_ids=["first", "second"],
        models=["a", "b"],
        capabilities=["trend", "common_factor"],
        capability_dataset_ids={
            "trend": ["first", "second"],
            "common_factor": ["second"],
        },
    )

    common_factor = [
        row for row in result if row["capability_id"] == "common_factor"
    ]
    assert {row["dataset_count"] for row in common_factor} == {1}
    assert {tuple(row["dataset_ids"]) for row in common_factor} == {
        ("second",)
    }


def test_split_bank_requires_two_batches_for_stability_statistics():
    analysis = load_script("analyze_paper_v8")
    rows = []
    for seed in range(64):
        batch_scale = 1.0 if seed < 32 else 2.0
        for model_id, model_scale in (("a", 1.0), ("b", 2.0)):
            for policy in ("fixed_l168", "oracle_context"):
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
    two_batches = by_key[(32, "fixed_l168", "trend", "accuracy")]
    one_batch = by_key[(64, "fixed_l168", "trend", "accuracy")]

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
            "context_policy": "fixed_l168",
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
            "context_policy": "fixed_l168",
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


def test_cross_effect_audit_scores_metadata_declared_active_prefix():
    analysis = load_script("analyze_paper_v8")
    context = 8
    horizon = 4
    first_target = np.zeros((context + horizon, 3))
    second_target = first_target.copy()
    second_target[context : context + 2, 1:] = 1.0
    first = {
        "dataset_id": "dataset",
        "capability_id": "cross_series_dependence",
        "generator_family_role": "primary",
        "evaluation_table": "strict_counterfactual_audit",
        "intensity": 5,
        "seed_index": 0,
        "context_length": context,
        "horizon": horizon,
        "target_dim": 3,
        "target": first_target.tolist(),
        "master_counterfactual_pair_id": "pair",
        "generation_metadata": {
            "responder_indices": [1, 2],
            "counterfactual_effect_forecast_steps": 2,
        },
    }
    second = {**first, "target": second_target.tolist()}
    first_forecast = np.zeros((horizon, 3))
    second_forecast = np.zeros((horizon, 3))
    second_forecast[:2, 1:] = 1.0
    second_forecast[2:, 1:] = 5.0

    row = analysis.effect_row(
        first,
        first_forecast,
        second,
        second_forecast,
        model_id="model",
    )

    assert row["active_prefix_steps"] == 2
    assert row["active_prefix_source"].endswith(
        "counterfactual_effect_forecast_steps"
    )
    assert row["active_effect_nrmse"] == pytest.approx(0.0)
    assert row["counterfactual_effect_nrmse"] > 1.0
    assert row["zero_tail_leakage_nrmse"] > 1.0


def test_cross_effect_audit_keeps_legacy_full_horizon_fallback():
    analysis = load_script("analyze_paper_v8")
    sample = {
        "dataset_id": "dataset",
        "capability_id": "cross_series_dependence",
        "generator_family_role": "primary",
        "evaluation_table": "strict_counterfactual_audit",
        "intensity": 5,
        "seed_index": 0,
        "context_length": 4,
        "horizon": 2,
        "target_dim": 2,
        "target": np.zeros((6, 2)).tolist(),
        "master_counterfactual_pair_id": "pair",
        "generation_metadata": {"responder_indices": [1]},
    }
    second = np.zeros((6, 2))
    second[4:, 1] = 1.0
    row = analysis.effect_row(
        sample,
        np.zeros((2, 2)),
        {**sample, "target": second.tolist()},
        np.column_stack([np.zeros(2), np.ones(2)]),
        model_id="model",
    )

    assert row["active_prefix_steps"] == 2
    assert row["active_prefix_source"] == "legacy_full_horizon_fallback"
    assert row["active_effect_nrmse"] == pytest.approx(
        row["counterfactual_effect_nrmse"]
    )


def test_common_structured_assessment_requires_advantage_and_strict_recovery():
    analysis = load_script("analyze_paper_v8")

    def metric(model, table, value, seed):
        metrics = {
            "common_component_nmae": value,
            "factor_trajectory_correlation": 0.9,
        }
        return {
            "capability_id": "common_factor",
            "model_id": model,
            "context_length": 168,
            "evaluation_table": table,
            "seed_index": seed,
            "metrics": metrics,
            "input_adaptation": {
                "structured_baseline": {"fallback_used": False}
            },
        }

    metrics = []
    effects = []
    for seed in range(3):
        metrics.extend(
            [
                metric("diagonal_ar", "main", 1.0, seed),
                metric("dynamic_factor_var", "main", 0.95, seed),
                metric(
                    "dynamic_factor_var",
                    "multivariate_input_ablation",
                    1.2,
                    seed,
                ),
            ]
        )
        effects.append(
            {
                "capability_id": "common_factor",
                "model_id": "dynamic_factor_var",
                "context_length": 168,
                "counterfactual_effect_nrmse": 0.2,
                "effect_correlation": 0.9,
                "effect_amplitude_ratio": 1.0,
            }
        )
    row = next(
        item
        for item in analysis._structured_context_curve(
            metrics,
            effects,
            dataset_id="dataset",
        )
        if item["capability_id"] == "common_factor"
        and item["context_length"] == 168
    )

    assert not row["structured_positive_control_passed"]
    assert "no_10pct_advantage_over_diagonal_ar" in row["failure_codes"]
    assert row["strict_effect_passed"] is True
    assert row["strict_effect_assessment"] == (
        "evaluated_as_blind_shared_fit_hard_gate"
    )

    for metric_row in metrics:
        if (
            metric_row["model_id"] == "dynamic_factor_var"
            and metric_row["evaluation_table"] == "main"
        ):
            metric_row["metrics"]["common_component_nmae"] = 0.8
    for effect in effects:
        effect.update(
            {
                "counterfactual_effect_nrmse": 1.0,
                "effect_correlation": 0.0,
                "effect_amplitude_ratio": 0.0,
            }
        )
    strict_failure = next(
        item
        for item in analysis._structured_context_curve(
            metrics,
            effects,
            dataset_id="dataset",
        )
        if item["capability_id"] == "common_factor"
        and item["context_length"] == 168
    )
    assert not strict_failure["structured_positive_control_passed"]
    assert "strict_counterfactual_recovery_below_threshold" in (
        strict_failure["failure_codes"]
    )


def test_multivariate_utilization_audit_marks_independent_adapter_reference():
    analysis = load_script("analyze_paper_v8")

    def main_row(model, target_mode):
        return {
            "dataset_id": "dataset",
            "context_policy": "fixed_l168",
            "evaluation_table": "main",
            "generator_family_role": "primary",
            "capability_id": "common_factor",
            "model_id": model,
            "seed_index": 0,
            "intensity": 5,
            "input_adaptation": {"target_mode": target_mode},
        }

    comparisons = [
        {
            "comparison_id": "multivariate_input_ablation",
            "dataset_id": "dataset",
            "context_policy": "fixed_l168",
            "capability_id": "common_factor",
            "model_id": "independent",
            "accuracy_metric": "protected_target_nmae",
            "accuracy_relative_delta": 0.0,
            "matched_seed_count": 1,
        }
    ]
    rows = analysis.multivariate_utilization_audit_rows(
        [
            main_row("independent", "independent_univariate"),
            main_row("native", "native_multivariate"),
        ],
        [],
        comparisons,
        models=["independent", "native"],
    )
    by_model = {row["model_id"]: row for row in rows}

    assert by_model["independent"]["audit_role"] == (
        "independent_univariate_reference"
    )
    assert not by_model["independent"][
        "eligible_for_multivariate_utilization_claim"
    ]
    assert by_model["native"]["audit_role"] == "multivariate_model"
    assert by_model["native"][
        "eligible_for_multivariate_utilization_claim"
    ]
    assert all(row["audit_has_no_ranking"] for row in rows)


def test_inference_tasks_include_real_anchors_as_separate_auxiliary_table(
    monkeypatch,
    tmp_path,
):
    inference = load_script("run_paper_v8_inference")
    source_records = []
    for name in ("clean", "robustness", "input_ablations"):
        path = tmp_path / f"{name}.jsonl"
        inference.v8.write_jsonl(path, ())
        source_records.append(inference.v8.file_record(path))
    synthetic_task = {
        "sample_id": "synthetic__L168",
        "context_length": 168,
        "horizon": 48,
        "target_dim": 1,
        "covariate_dim": 0,
        "target": np.zeros((216, 1)).tolist(),
        "covariates": None,
        "frequency": "1h",
    }
    monkeypatch.setattr(
        inference.v8,
        "iter_master_views",
        lambda _masters: iter([synthetic_task]),
    )
    real_master_path = tmp_path / "real_anchor_masters.jsonl"
    real_master = {
        **synthetic_task,
        "sample_id": "v8real__anchor",
        "anchor_id": "anchor",
    }
    inference.v8.write_jsonl(real_master_path, [real_master])
    generation_manifest = {
        "config_sha256": "generation",
        "files": {
            key: record
            for key, record in zip(
                ("clean", "robustness", "input_ablations"),
                source_records,
                strict=True,
            )
        },
    }
    calibration_bundle = {
        "bundle_content_sha256": "calibration",
        "files": {
            "real_anchor_masters": inference.v8.file_record(
                real_master_path
            )
        },
    }

    task_path, manifest = inference.prepare_view_tasks(
        generation_manifest,
        inference_dir=tmp_path / "inference",
        calibration_bundle=calibration_bundle,
    )
    tasks = list(inference.v8.iter_jsonl(task_path))

    assert manifest["synthetic_view_count"] == 1
    assert manifest["real_anchor_view_count"] == 1
    assert manifest["view_count"] == 2
    assert [row["sample_id"] for row in tasks] == [
        "synthetic__L168",
        "v8real__anchor",
    ]
    assert tasks[1]["evaluation_table"] == "real_anchor_forecast"
    assert tasks[1]["context_policy"] == "fixed_l168"

    canonical_path = tmp_path / "predictions.jsonl"
    inference.v8.write_jsonl(
        canonical_path,
        [
            {"sample_id": row["sample_id"], "forecast": []}
            for row in tasks
        ],
    )
    output_path = tmp_path / "real_predictions.jsonl"
    count = inference.write_real_anchor_prediction_subset(
        canonical_path,
        real_anchor_task_path=Path(
            manifest["task_components"]["real_anchors"]["path"]
        ),
        output_path=output_path,
    )
    assert count == 1
    assert [
        row["sample_id"]
        for row in inference.v8.iter_jsonl(output_path)
    ] == ["v8real__anchor"]


def test_formal_inference_requires_a_valid_real_anchor_source(tmp_path):
    inference = load_script("run_paper_v8_inference")

    with pytest.raises(ValueError, match="requires calibration_bundle"):
        inference.formal_real_anchor_source_record(None)
    with pytest.raises(ValueError, match="missing files.real_anchor_masters"):
        inference.formal_real_anchor_source_record({"files": {}})

    path = tmp_path / "real_anchor_masters.jsonl"
    inference.v8.write_jsonl(path, [{"sample_id": "v8real__anchor"}])
    record = inference.v8.file_record(path)
    assert inference.formal_real_anchor_source_record(
        {"files": {"real_anchor_masters": record}}
    ) == record

    path.write_text('{"sample_id":"changed"}\\n', encoding="utf-8")
    with pytest.raises(ValueError, match="byte-size mismatch|hash mismatch"):
        inference.formal_real_anchor_source_record(
            {"files": {"real_anchor_masters": record}}
        )


def test_prepare_view_tasks_allows_explicit_no_anchor_compatibility(
    monkeypatch,
    tmp_path,
):
    inference = load_script("run_paper_v8_inference")
    source_records = {}
    for name in ("clean", "robustness", "input_ablations"):
        path = tmp_path / f"{name}.jsonl"
        inference.v8.write_jsonl(path, ())
        source_records[name] = inference.v8.file_record(path)
    monkeypatch.setattr(
        inference.v8,
        "iter_master_views",
        lambda _masters: iter(()),
    )

    _, manifest = inference.prepare_view_tasks(
        {
            "config_sha256": "generation",
            "files": source_records,
        },
        inference_dir=tmp_path / "inference",
        calibration_bundle=None,
    )

    assert manifest["real_anchor_view_count"] == 0
    assert inference.validate_inference_task_manifest_files(manifest).is_file()


def test_resume_task_validation_checks_combined_and_both_components(
    tmp_path,
):
    inference = load_script("run_paper_v8_inference")
    root = tmp_path / "inference"
    synthetic_path = root / "synthetic.jsonl"
    real_path = root / "real.jsonl"
    combined_path = root / "combined.jsonl"
    synthetic = {"sample_id": "synthetic"}
    real = {"sample_id": "v8real__anchor"}
    inference.v8.write_jsonl(synthetic_path, [synthetic])
    inference.v8.write_jsonl(real_path, [real])
    inference.v8.write_jsonl(combined_path, [synthetic, real])

    def record(path, rows):
        return {
            **inference.v8.file_record(path),
            "row_count": rows,
        }

    manifest = {
        "schema_version": "paper_v8_inference_task_manifest.v2",
        "synthetic_view_count": 1,
        "real_anchor_view_count": 1,
        "view_count": 2,
        "task_components": {
            "synthetic": record(synthetic_path, 1),
            "real_anchors": record(real_path, 1),
        },
        "task_file": record(combined_path, 2),
    }
    assert (
        inference.validate_inference_task_manifest_files(manifest)
        == combined_path
    )

    synthetic_path.write_text(
        '{"sample_id":"corrupted"}\\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="component synthetic.*mismatch"):
        inference.validate_inference_task_manifest_files(manifest)


def test_cached_prediction_reuse_requires_exact_unique_task_ids(tmp_path):
    inference = load_script("run_paper_v8_inference")
    model_id = "Chronos-2"
    canonical_path = inference.prediction_path_for(
        inference.model_root(tmp_path, model_id),
        model_id,
    )
    inference.v8.write_jsonl(
        canonical_path,
        [
            {
                "model_id": model_id,
                "sample_id": "a",
                "forecast": [[0.0]],
            },
            {
                "model_id": model_id,
                "sample_id": "b",
                "forecast": [[0.0]],
            },
        ],
    )
    record = {
        "model_id": model_id,
        "row_count": 2,
        **inference.v8.file_record(canonical_path),
    }
    inference.v8.write_json(
        tmp_path / "inference_manifest.json",
        {
            "schema_version": "paper_v8_inference_manifest.v3",
            "complete": True,
            "statuses": [
                {
                    "model_id": model_id,
                    "status": "complete",
                    "succeeded_original_view_count": 2,
                }
            ],
            "predictions": {"files": [record]},
        },
    )

    assert model_id in inference.cached_complete_model_records(
        tmp_path,
        expected_sample_ids={"a", "b"},
    )
    assert not inference.cached_complete_model_records(
        tmp_path,
        expected_sample_ids={"a", "c"},
    )

    inference.v8.write_jsonl(
        canonical_path,
        [
            {"model_id": model_id, "sample_id": "a", "forecast": [[0.0]]},
            {"model_id": model_id, "sample_id": "a", "forecast": [[0.0]]},
        ],
    )
    with pytest.raises(ValueError, match="duplicate canonical prediction"):
        inference.canonical_prediction_sample_ids(
            canonical_path,
            model_id=model_id,
        )


def test_analysis_validates_synthetic_component_and_rejects_real_table(
    tmp_path,
):
    analyze = load_script("analyze_paper_v8")
    inference_dir = tmp_path / "inference"
    synthetic_path = inference_dir / "synthetic.jsonl"
    rows = [
        {"sample_id": "main", "evaluation_table": "main"},
        {
            "sample_id": "robustness",
            "evaluation_table": "observation_noise_robustness",
        },
    ]
    analyze.v8.write_jsonl(synthetic_path, rows)

    def write_manifests():
        component = {
            **analyze.v8.file_record(synthetic_path),
            "row_count": len(rows),
        }
        task_manifest_path = inference_dir / "task_manifest.json"
        analyze.v8.write_json(
            task_manifest_path,
            {
                "schema_version": "paper_v8_inference_task_manifest.v2",
                "synthetic_view_count": len(rows),
                "task_components": {"synthetic": component},
            },
        )
        return {
            "task_manifest_sha256": analyze.v8.file_sha256(
                task_manifest_path
            )
        }

    inference_manifest = write_manifests()
    path, _ = analyze.validated_synthetic_task_path(
        inference_dir,
        inference_manifest,
    )
    assert path == synthetic_path

    with pytest.raises(ValueError, match="task manifest hash mismatch"):
        analyze.validated_synthetic_task_path(
            inference_dir,
            {"task_manifest_sha256": "stale"},
        )

    rows[:] = [
        {
            "sample_id": "v8real__anchor",
            "evaluation_table": "real_anchor_forecast",
        }
    ]
    analyze.v8.write_jsonl(synthetic_path, rows)
    inference_manifest = write_manifests()
    with pytest.raises(ValueError, match="non-synthetic evaluation table"):
        analyze.validated_synthetic_task_path(
            inference_dir,
            inference_manifest,
        )


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


def test_compaction_probe_skips_final_format_and_detects_legacy_rows(tmp_path):
    inference = load_script("run_paper_v8_inference")
    path = tmp_path / "predictions.jsonl"
    compact = {
        "schema_version": "paper_v8_inference_prediction.v2",
        "model_id": "model",
        "sample_id": "view",
        "forecast": [[1.0]],
        "input_adaptation": {"target_mode": "native_univariate"},
    }
    path.write_text(inference.json.dumps(compact) + "\n", encoding="utf-8")

    assert inference.canonical_prediction_file_is_compact(path)

    legacy = {**compact, "request_seconds": 1.25}
    path.write_text(inference.json.dumps(legacy) + "\n", encoding="utf-8")

    assert not inference.canonical_prediction_file_is_compact(path)


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
        "model",
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
    assert profile.capacity_for("TimePFN") == 4
    assert profile.capacity_for("Timer-3.5") == 2.4
    assert profile.http_concurrency_for("tirex2", 32) == 8
    assert profile.http_concurrency_for("TimePFN", 32) == 8


def test_timepfn_profile_prioritizes_fast_loading_and_large_bulk_requests():
    inference = load_script("run_paper_v8_inference")

    assert inference.MODEL_EXECUTION_CONFIG["TimePFN"] == {
        "replicas_per_device": 1,
        "http_concurrency": 2,
        "task_batch_size": 512,
        "transport": "msgpack_bulk",
    }
    assert inference.MODEL_MAJOR_DATASET_PARALLELISM["TimePFN"] == 4
    assert inference.DEFAULT_MODELS[-1] == "toto2.0"
    assert "TimePFN" not in inference.DEFAULT_MODELS
    assert "tabpfn-ts3" not in inference.DEFAULT_MODELS


def test_formal_models_use_bounded_model_major_dataset_parallelism():
    inference = load_script("run_paper_v8_inference")

    assert {
        model_id: inference.MODEL_MAJOR_DATASET_PARALLELISM[model_id]
        for model_id in inference.DEFAULT_MODELS
    } == {
        "Chronos-2": 4,
        "timesfm2.5": 2,
        "tirex2": 2,
        "moirai2": 2,
        "Timer-3.5": 2,
        "toto2.0": 4,
    }


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


def test_model_load_waits_for_concurrent_loading_state(monkeypatch):
    inference = load_script("run_paper_v8_inference")
    client = object.__new__(inference.TimerServiceClient)
    expected = {
        "model_id": "model",
        "status": "loaded",
        "endpoints": [
            {"device": "cuda:0", "worker_pid": 10},
            {"device": "cuda:1", "worker_pid": 11},
        ],
    }
    states = iter(
        [
            {
                "model_id": "model",
                "status": "loading",
                "devices": ["cuda:0", "cuda:1"],
                "endpoints": [],
            },
            expected,
        ]
    )
    posts = []
    monkeypatch.setattr(client, "_loaded_state", lambda _model_id: next(states))
    monkeypatch.setattr(
        client,
        "_post",
        lambda path, body, **kwargs: posts.append((path, body, kwargs)) or {},
    )
    monkeypatch.setattr(inference.time, "sleep", lambda _seconds: None)

    _seconds, state = client.ensure_loaded(
        "model",
        devices="0,1",
        replicas_per_device=1,
        timeout_seconds=60,
    )

    assert state == expected
    assert posts == []


def test_model_load_treats_concurrent_409_as_in_progress(monkeypatch):
    inference = load_script("run_paper_v8_inference")
    client = object.__new__(inference.TimerServiceClient)
    expected = {
        "model_id": "model",
        "status": "loaded",
        "endpoints": [
            {"device": "cuda:0", "worker_pid": 10},
            {"device": "cuda:1", "worker_pid": 11},
        ],
    }
    states = iter(
        [
            None,
            {
                "model_id": "model",
                "status": "loading",
                "devices": ["cuda:0", "cuda:1"],
                "endpoints": [],
            },
            expected,
        ]
    )
    posts = []

    def conflicting_post(path, body, **kwargs):
        posts.append((path, body, kwargs))
        raise RuntimeError("returned 409: model is already loading")

    monkeypatch.setattr(client, "_loaded_state", lambda _model_id: next(states))
    monkeypatch.setattr(client, "_post", conflicting_post)
    monkeypatch.setattr(inference.time, "sleep", lambda _seconds: None)

    _seconds, state = client.ensure_loaded(
        "model",
        devices="0,1",
        replicas_per_device=1,
        timeout_seconds=60,
    )

    assert state == expected
    assert len(posts) == 1


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
        "TimePFN",
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
        tmp_path / "model_task_shards" / inference.safe_filename(model_id)
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
