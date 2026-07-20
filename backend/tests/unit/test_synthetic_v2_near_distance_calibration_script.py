from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run_synthetic_v2_near_distance_calibration.py"


def load_calibration_module():
    repo_root = SCRIPT_PATH.parents[1]
    for path in (repo_root / "backend", repo_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("run_synthetic_v2_near_distance_calibration", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nearest_distances_reports_exact_copy_as_zero():
    module = load_calibration_module()
    reference = np.asarray([[0.0, 1.0], [2.0, 3.0], [10.0, 11.0]])
    query = np.asarray([[2.0, 3.0]])

    distances = module.nearest_distances(query, reference, metric="mae")

    assert distances["d1"][0] == 0.0
    assert distances["d2"][0] > 0.0
    assert distances["nndr"][0] == 0.0


def test_positive_tail_threshold_does_not_collapse_on_cross_group_duplicate():
    module = load_calibration_module()

    assert module.positive_lower_tail_quantile(np.asarray([0.0, 0.0, 0.2, 0.4]), 0.01) > 0.0


def test_positive_tail_threshold_fails_closed_when_all_distances_are_zero():
    module = load_calibration_module()

    with np.testing.assert_raises_regex(ValueError, "all-zero"):
        module.positive_lower_tail_quantile(np.asarray([0.0, 0.0]), 0.01)


def test_online_artifact_preserves_small_positive_feature_scales():
    module = load_calibration_module()
    spec = module.BucketSpec("unit", "tsf_univariate", "unit.zip", 8, 4, 1, 4)
    reference_rows = [
        {
            "raw": np.full(12, float(index)),
            "context_raw": np.full(8, float(index)),
            "features": {"spike_rate": index * 3e-7},
        }
        for index in range(3)
    ]
    thresholds = {
        name: 0.1
        for name in (
            "raw_mae_p01",
            "raw_mae_p05",
            "raw_l2_p01",
            "raw_l2_p05",
            "feature_l2_p01",
            "feature_l2_p05",
            "raw_mae_nndr_p01",
            "raw_mae_nndr_p05",
            "context_raw_mae_p01",
            "context_raw_mae_p05",
            "context_raw_l2_p01",
            "context_raw_l2_p05",
            "context_raw_mae_nndr_p01",
            "context_raw_mae_nndr_p05",
        )
    }

    bucket = module.online_artifact_bucket(
        spec,
        reference_rows,
        thresholds=thresholds,
    )

    assert bucket["feature_names"] == ["spike_rate"]
    assert bucket["feature_scale"][0] > 0.0
    assert bucket["reference_covariates"] is None
    assert bucket["reference_group_ids"] == [None, None, None]
    assert bucket["reference_window_starts"] == [None, None, None]


def test_online_artifact_preserves_reconstructable_structured_references():
    module = load_calibration_module()
    spec = module.BucketSpec(
        "structured",
        "fixture",
        "fixture.npz",
        4,
        2,
        1,
        4,
        target_dim=1,
        covariate_dim=2,
        frequency="h",
        target_column_names=("load",),
        covariate_column_names=("calendar_a", "calendar_b"),
    )
    reference_rows = [
        {
            "raw": np.full(6, float(index)),
            "context_raw": np.full(4, float(index)),
            "covariates": np.full((6, 2), float(index)),
            "features": {"trend_strength": float(index)},
            "group_id": f"group-{index}",
            "window_start": index * 10,
            "source_segment_id": index,
            "processed_npz_sha256": "npz-hash",
            "processed_metadata_sha256": "metadata-hash",
            "source_provenance": {"source_record_url": "https://example.test"},
            "forecast_release_id": f"release-{index}",
            "forecast_release_valid_start": f"2026-01-0{index + 1}",
            "forecast_release_valid_end": f"2026-01-0{index + 2}",
            "forecast_window_valid_start": f"2026-01-0{index + 1}",
            "forecast_stitching": "forbidden",
            "issue_time_semantics": "fixture as-of",
            "issue_time": f"issue-{index}",
        }
        for index in range(3)
    ]
    thresholds = {
        name: 0.1
        for name in (
            "raw_mae_p01",
            "raw_mae_p05",
            "raw_l2_p01",
            "raw_l2_p05",
            "feature_l2_p01",
            "feature_l2_p05",
            "raw_mae_nndr_p01",
            "raw_mae_nndr_p05",
            "context_raw_mae_p01",
            "context_raw_mae_p05",
            "context_raw_l2_p01",
            "context_raw_l2_p05",
            "context_raw_mae_nndr_p01",
            "context_raw_mae_nndr_p05",
        )
    }

    bucket = module.online_artifact_bucket(
        spec,
        reference_rows,
        thresholds=thresholds,
    )

    assert np.asarray(bucket["reference_covariates"]).shape == (3, 6, 2)
    assert bucket["reference_group_ids"] == ["group-0", "group-1", "group-2"]
    assert bucket["reference_window_starts"] == [0, 10, 20]
    assert bucket["reference_segment_ids"] == [0, 1, 2]
    assert bucket["reference_forecast_release_ids"] == [
        "release-0",
        "release-1",
        "release-2",
    ]
    assert bucket["reference_forecast_stitching"] == ["forbidden"] * 3
    assert bucket["reference_issue_times"] == ["issue-0", "issue-1", "issue-2"]
    assert bucket["target_column_names"] == ["load"]
    assert bucket["covariate_column_names"] == [
        "calendar_a",
        "calendar_b",
    ]
    assert bucket["frequency"] == "h"
    assert bucket["provenance"]["processed_npz_sha256"] == "npz-hash"


def test_swiss_covariate_window_never_stitches_benchmark_future():
    module = load_calibration_module()
    row_count = 80
    cube = np.empty((row_count, 6, 24), dtype=float)
    for row in range(row_count):
        for variable in range(6):
            cube[row, variable] = (
                1000.0 * row + 10.0 * variable + np.arange(24)
            )
    timestamps = np.arange(
        np.datetime64("2026-01-01T00:00"),
        np.datetime64("2026-01-02T16:00"),
        np.timedelta64(30, "m"),
    )
    arrays = {
        "timestamps": timestamps,
        "nwp_asof_timestamps": timestamps,
        "nwp_valid_timestamps": (
            timestamps[:, None] + np.arange(24, dtype="timedelta64[h]")
        ),
        "nwp_forecasts": cube,
    }
    spec = module.BucketSpec(
        "swiss",
        "paper_v7_swiss",
        "swiss.npz",
        4,
        6,
        1,
        48,
        covariate_dim=6,
    )

    covariates, audit = module.paper_v7_swiss_strict_covariate_window(
        arrays,
        start=2,
        spec=spec,
    )

    origin = 2 + spec.context_length - 1
    expected_future = np.repeat(cube[origin + 1].T, 2, axis=0)[:6]
    assert covariates.shape == (10, 6)
    assert np.array_equal(covariates[4:], expected_future)
    assert audit["forecast_release_id"].endswith(
        module.timestamp_iso_utc(timestamps[origin + 1])
    )
    assert audit["benchmark_covariate_vintage_count"] == 1
    assert "forbidden_for_benchmark_H48" in audit["forecast_stitching"]


def test_gefcom2012_loader_never_crosses_segment_gaps(monkeypatch, tmp_path):
    module = load_calibration_module()
    timestamps = np.concatenate(
        [
            np.arange(
                np.datetime64("2026-01-01T00:00"),
                np.datetime64("2026-01-01T08:00"),
                np.timedelta64(1, "h"),
            ),
            np.arange(
                np.datetime64("2026-01-03T00:00"),
                np.datetime64("2026-01-03T08:00"),
                np.timedelta64(1, "h"),
            ),
        ]
    )
    zones = np.column_stack(
        [np.arange(len(timestamps), dtype=float) + zone for zone in range(20)]
    )
    arrays = {
        "timestamps": timestamps,
        "segment_ids": np.asarray([0] * 8 + [1] * 8, dtype=np.int32),
        "zones": zones,
        "total": zones.sum(axis=1),
        "canonical_hierarchy": np.column_stack(
            [zones.sum(axis=1), zones[:, :10].sum(axis=1), zones[:, 10:].sum(axis=1)]
        ),
        "calendar_covariates": np.zeros((len(timestamps), 6)),
    }
    metadata = {
        "zone_columns": [f"zone_{zone}" for zone in range(1, 21)],
        "canonical_hierarchy_columns": [
            "total",
            "sum_zones_1_10",
            "sum_zones_11_20",
        ],
        "derived_total_column": "total",
        "frequency": "h",
        "calendar_covariate_columns": list(
            module.PAPER_V7_GEFCOM2012_CALENDAR_COLUMNS
        ),
        "source_record_url": "https://example.test",
        "source_files": [],
        "_processed_npz_sha256": "npz-hash",
        "_processed_metadata_sha256": "metadata-hash",
    }
    monkeypatch.setattr(
        module,
        "read_paper_v7_gefcom2012_processed",
        lambda _path: (arrays, metadata),
    )
    spec = module.BucketSpec(
        "gefcom-factor",
        "paper_v7_gefcom2012",
        "gefcom2012.npz",
        4,
        2,
        1,
        24,
        target_dim=3,
    )
    path = tmp_path / "gefcom2012.npz"
    path.touch()

    rows = module.load_paper_v7_gefcom2012_rows(spec, path, max_windows=100)

    assert len(rows) == 6
    assert {row["source_segment_id"] for row in rows} == {0, 1}
    assert {row["window_start"] for row in rows} == {0, 1, 2, 8, 9, 10}
    assert {row["group_id"] for row in rows} == {
        "gefcom2012-load:common_factor"
    }


@pytest.mark.skipif(
    not (
        SCRIPT_PATH.parents[1]
        / "runtime/research/v7-p0-data/processed/swiss_hierarchical_demand.npz"
    ).is_file(),
    reason="Paper v7 P0 processed assets are not present",
)
def test_paper_v7_real_processed_assets_load_all_six_views():
    module = load_calibration_module()
    data_dir = SCRIPT_PATH.parents[1] / "runtime/research/v7-p0-data/processed"
    cases = [
        (
            module.BucketSpec(
                "swiss-factor",
                "paper_v7_swiss",
                "swiss_hierarchical_demand.npz",
                504,
                96,
                48,
                48,
                target_dim=3,
            ),
            data_dir / "swiss_hierarchical_demand.npz",
        ),
        (
            module.BucketSpec(
                "swiss-hierarchy",
                "paper_v7_swiss",
                "swiss_hierarchical_demand.npz",
                504,
                96,
                48,
                48,
                target_dim=3,
                hierarchy="additive_first",
            ),
            data_dir / "swiss_hierarchical_demand.npz",
        ),
        (
            module.BucketSpec(
                "swiss-covariate",
                "paper_v7_swiss",
                "swiss_hierarchical_demand.npz",
                504,
                96,
                48,
                48,
                covariate_dim=6,
                known_future_covariates=module.PAPER_V7_SWISS_NWP_COLUMNS,
            ),
            data_dir / "swiss_hierarchical_demand.npz",
        ),
        (
            module.BucketSpec(
                "gefcom-factor",
                "paper_v7_gefcom2012",
                "gefcom2012_load.npz",
                504,
                96,
                48,
                24,
                target_dim=3,
            ),
            data_dir / "gefcom2012_load.npz",
        ),
        (
            module.BucketSpec(
                "gefcom-hierarchy",
                "paper_v7_gefcom2012",
                "gefcom2012_load.npz",
                504,
                96,
                48,
                24,
                target_dim=3,
                hierarchy="additive_first",
            ),
            data_dir / "gefcom2012_load.npz",
        ),
        (
            module.BucketSpec(
                "gefcom-covariate",
                "paper_v7_gefcom2012",
                "gefcom2012_load.npz",
                504,
                96,
                48,
                24,
                covariate_dim=6,
                known_future_covariates=(
                    module.PAPER_V7_GEFCOM2012_CALENDAR_COLUMNS
                ),
            ),
            data_dir / "gefcom2012_load.npz",
        ),
    ]

    for spec, path in cases:
        rows = module.load_real_bucket(spec, path, max_windows=4)
        assert len(rows) == 4
        assert rows[0]["target"].shape == (
            spec.context_length + spec.horizon,
            spec.target_dim,
        )
        if spec.covariate_dim:
            assert rows[0]["covariates"].shape == (
                spec.context_length + spec.horizon,
                spec.covariate_dim,
            )
        assert rows[0]["processed_npz_sha256"]


def test_evaluate_risk_flags_exact_copy_and_spares_far_sample():
    module = load_calibration_module()
    train = [
        {"raw": np.asarray([0.0, 0.0]), "context_raw": np.asarray([0.0]), "features": {"trend_strength": 0.1}},
        {"raw": np.asarray([1.0, 1.0]), "context_raw": np.asarray([1.0]), "features": {"trend_strength": 0.2}},
        {"raw": np.asarray([2.0, 2.0]), "context_raw": np.asarray([2.0]), "features": {"trend_strength": 0.3}},
    ]
    thresholds = {
        "raw_mae_p01": 0.01,
        "raw_mae_p05": 0.05,
        "raw_l2_p01": 0.01,
        "raw_l2_p05": 0.05,
        "feature_l2_p01": 0.01,
        "raw_mae_nndr_p01": 0.05,
        "context_raw_mae_p01": 0.01,
        "context_raw_mae_p05": 0.05,
        "context_raw_l2_p01": 0.01,
        "context_raw_l2_p05": 0.05,
        "context_raw_mae_nndr_p01": 0.05,
    }

    exact = module.evaluate_risk(
        [train[1]],
        train,
        ("trend_strength",),
        np.asarray([0.2]),
        np.asarray([0.1]),
        thresholds,
    )
    far = module.evaluate_risk(
        [{"raw": np.asarray([100.0, 100.0]), "context_raw": np.asarray([100.0]), "features": {"trend_strength": 10.0}}],
        train,
        ("trend_strength",),
        np.asarray([0.2]),
        np.asarray([0.1]),
        thresholds,
    )

    assert exact["strict_risk_rate"] == 1.0
    assert exact["combined_risk_rate"] == 1.0
    assert far["strict_risk_rate"] == 0.0
    assert far["combined_risk_rate"] == 0.0


def test_render_report_uses_relative_summary_path(tmp_path):
    module = load_calibration_module()
    summary = {
        "config": {
            "max_windows_per_bucket": 10,
            "splits": 1,
            "synthetic_count": 1,
            "jitter_scale": 0.02,
            "strict_rule": "strict",
            "combined_rule": "combined",
        },
        "buckets": [
            {
                "profile_id": "unit_bucket",
                "real_window_count": 10,
                "threshold_stability": {
                    "raw_mae_p01": {"mean": 0.1, "cv": 0.0},
                    "raw_l2_p01": {"mean": 0.2, "cv": 0.0},
                    "feature_l2_p01": {"mean": 0.3, "cv": 0.0},
                    "raw_mae_nndr_p01": {"mean": 0.4, "cv": 0.0},
                },
                "control_summary": {
                    label: {
                        "combined_risk_rate": {"mean": 0.0},
                        "strict_risk_rate": {"mean": 0.0},
                    }
                    for label in ("real_holdout", "exact_copy", "affine_copy", "context_copy", "jitter_copy", "normal_synthetic")
                },
            }
        ],
        "overall": {
            "exact_copy_strict_risk_min": 1.0,
            "jitter_copy_combined_risk_min": 1.0,
            "affine_copy_strict_risk_min": 1.0,
            "context_copy_strict_risk_min": 1.0,
            "normal_synthetic_combined_risk_max": 0.0,
        },
    }

    report = module.render_report(summary, output_dir=tmp_path)

    assert "Synthetic v2 Near-Distance Calibration" in report
    assert "summary.json" in report


def split_row(index: int, *, group_id: str, start: int) -> dict:
    return {"id": index, "group_id": group_id, "window_start": start}


def test_group_split_never_shares_a_source_series():
    module = load_calibration_module()
    spec = module.BucketSpec("unit", "tsf_univariate", "unit.zip", 8, 4, 4, 4)
    rows = [
        split_row(group * 10 + offset, group_id=f"series:{group}", start=offset * 4)
        for group in range(10)
        for offset in range(10)
    ]

    reference, holdout, summary = module.split_rows_leakage_safe(rows, spec, seed=123)

    assert summary["policy"] == "group"
    assert {row["group_id"] for row in reference}.isdisjoint({row["group_id"] for row in holdout})


def test_single_series_split_applies_context_plus_horizon_embargo():
    module = load_calibration_module()
    spec = module.BucketSpec("unit", "gefcom2014_load", "unit.zip", 8, 4, 1, 4)
    rows = [split_row(index, group_id="series:one", start=index * 4) for index in range(100)]

    reference, holdout, summary = module.split_rows_leakage_safe(rows, spec, seed=123)

    assert summary["policy"] == "temporal_embargo"
    assert summary["embargo_steps"] == 12
    assert max(row["window_start"] for row in reference) + 12 <= min(row["window_start"] for row in holdout)


def test_single_series_split_fails_closed_without_window_positions():
    module = load_calibration_module()
    spec = module.BucketSpec("unit", "gefcom2014_load", "unit.zip", 8, 4, 1, 4)
    rows = [{"group_id": "series:one"} for _ in range(30)]

    with np.testing.assert_raises_regex(ValueError, "window_start metadata"):
        module.split_rows_leakage_safe(rows, spec, seed=123)


def test_gift_native_panel_keeps_canonical_and_sensitivity_dimensions(
    monkeypatch,
    tmp_path,
):
    module = load_calibration_module()
    values = np.vstack(
        [
            np.sin(np.arange(220, dtype=float) / 7.0 + channel)
            for channel in range(7)
        ]
    )
    monkeypatch.setattr(
        module,
        "read_gift_arrow_targets",
        lambda _path: ("h", [("item", values)]),
    )
    monkeypatch.setattr(
        module,
        "truncate_gift_eval_official_test_tail",
        lambda _frequency, records: (96, records),
    )
    spec = module.BucketSpec(
        "ett1",
        "gift_panel",
        "ett1/H",
        24,
        8,
        8,
        24,
        target_dim=3,
        native_target_dim=7,
        sensitivity_target_dims=(7,),
        target_selection_policy="first three canonical",
    )
    path = tmp_path / "ett1"
    path.mkdir()

    rows = module.load_real_bucket(spec, path, max_windows=4)

    assert rows
    assert rows[0]["target"].shape == (32, 3)
    assert rows[0]["native_target_dim"] == 7
    assert rows[0]["canonical_target_dim"] == 3
    assert rows[0]["target_channel_indices"] == [0, 1, 2]
    assert rows[0]["sensitivity_target_dims"] == [7]


def test_m5_covariate_loader_excludes_price_and_records_provenance(
    monkeypatch,
    tmp_path,
):
    module = load_calibration_module()
    days = [f"d_{index}" for index in range(1, 41)]
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "d": days,
            "event_name_1": [
                "event" if index % 7 == 0 else np.nan for index in range(40)
            ],
            "event_name_2": [np.nan] * 40,
            "snap_CA": [index % 3 == 0 for index in range(40)],
        }
    )
    sales = pd.DataFrame(
        [
            {
                "id": "item",
                "item_id": "item",
                "store_id": "CA_1",
                "dept_id": "dept",
                "state_id": "CA",
                **{
                    day: 1.0 + index % 7 + (3.0 if index % 7 == 0 else 0.0)
                    for index, day in enumerate(days)
                },
            }
        ]
    )
    monkeypatch.setattr(
        module,
        "read_m5_calendar_and_sales",
        lambda _path: (calendar, sales, days),
    )
    spec = module.BucketSpec(
        "m5",
        "m5_covariate",
        "m5.zip",
        14,
        4,
        4,
        7,
        covariate_dim=4,
        known_future_covariates=module.M5_KNOWN_FUTURE_COVARIATES,
        covariate_provenance=module.M5_COVARIATE_PROVENANCE,
    )
    path = tmp_path / "m5.zip"
    path.touch()

    rows = module.load_real_bucket(spec, path, max_windows=4)

    assert rows
    assert rows[0]["covariates"].shape == (18, 4)
    assert rows[0]["known_future_covariates"] == [
        "day_of_week_sin",
        "day_of_week_cos",
        "event_count",
        "snap",
    ]
    assert "sell_price" not in rows[0]["known_future_covariates"]
    assert "excluded" in rows[0]["covariate_provenance"]


def test_m5_sibling_loader_uses_three_distinct_leaf_series(
    monkeypatch,
    tmp_path,
):
    module = load_calibration_module()
    days = [f"d_{index}" for index in range(1, 41)]
    sales = pd.DataFrame(
        [
            {
                "id": f"item-{item}",
                "item_id": f"item-{item}",
                "store_id": "CA_1",
                "dept_id": "dept",
                **{
                    day: float(1 + item + (index + item) % 7)
                    for index, day in enumerate(days)
                },
            }
            for item in range(6)
        ]
    )
    monkeypatch.setattr(
        module,
        "read_m5_calendar_and_sales",
        lambda _path: (pd.DataFrame(), sales, days),
    )
    spec = module.BucketSpec(
        "m5-panel",
        "m5_sibling_panel",
        "m5.zip",
        14,
        4,
        4,
        7,
        target_dim=3,
    )
    path = tmp_path / "m5.zip"
    path.touch()

    rows = module.load_real_bucket(spec, path, max_windows=8)

    assert rows
    assert rows[0]["target"].shape == (18, 3)
    assert len(set(rows[0]["leaf_item_ids"])) == 3
    assert "aggregate" not in rows[0]["panel_semantics"]


def test_gefcom_wind_covariate_loader_preserves_one_release(
    monkeypatch,
    tmp_path,
):
    module = load_calibration_module()
    history_times = pd.date_range("2026-01-01", periods=8, freq="h")
    future_times = pd.date_range("2026-01-01 08:00", periods=6, freq="h")
    history = {}
    future = {}
    for zone in range(1, 11):
        history[zone] = pd.DataFrame(
            {
                "TIMESTAMP": history_times,
                "TARGETVAR": zone + np.sin(np.arange(8)),
                "U10": np.arange(8) + zone,
                "V10": np.arange(8) + zone + 10,
                "U100": np.arange(8) + zone + 20,
                "V100": np.arange(8) + zone + 30,
            }
        )
        future[zone] = pd.DataFrame(
            {
                "TIMESTAMP": future_times,
                "U10": np.arange(6) + zone + 100,
                "V10": np.arange(6) + zone + 110,
                "U100": np.arange(6) + zone + 120,
                "V100": np.arange(6) + zone + 130,
                "TARGETVAR": zone + np.cos(np.arange(6)),
            }
        )
    release = {
        "task": 1,
        "release_id": "release-1",
        "valid_start": future_times[0],
        "valid_end": future_times[-1],
        "history": history,
        "future": future,
        "available_future_steps": 6,
        "covariate_provenance": module.GEFCOM2014_WIND_COVARIATE_PROVENANCE,
    }
    monkeypatch.setattr(
        module,
        "read_gefcom2014_wind_forecast_releases",
        lambda _path, minimum_future_steps: [release],
    )
    covariate_names = tuple(
        f"target_{target}_{column}"
        for target in range(3)
        for column in module.GEFCOM2014_WIND_NWP_COLUMNS
    )
    spec = module.BucketSpec(
        "wind-cov",
        "gefcom2014_wind_covariate",
        "wind.zip",
        4,
        2,
        2,
        24,
        target_dim=3,
        covariate_dim=12,
        native_target_dim=10,
        sensitivity_target_dims=(10,),
        known_future_covariates=covariate_names,
        covariate_provenance=module.GEFCOM2014_WIND_COVARIATE_PROVENANCE,
    )
    path = tmp_path / "wind.zip"
    path.touch()

    rows = module.load_real_bucket(spec, path, max_windows=4)

    assert rows
    assert rows[0]["target"].shape == (6, 3)
    assert rows[0]["covariates"].shape == (6, 12)
    assert rows[0]["forecast_release_id"] == "release-1"
    assert rows[0]["forecast_stitching"] == "forbidden"
    assert rows[0]["forecast_available_future_steps"] == 6
    for row in rows:
        offset = int(
            (
                pd.Timestamp(row["forecast_window_valid_start"])
                - future_times[0]
            )
            / pd.Timedelta(hours=1)
        )
        assert row["forecast_available_future_steps"] == 6 - offset
        assert row["forecast_window_future_steps"] == spec.horizon
        assert row["issue_time"] == "official-task-release:release-1"
        assert "does not publish a wall-clock" in row["issue_time_semantics"]
    assert rows[0]["native_target_dim"] == 10
