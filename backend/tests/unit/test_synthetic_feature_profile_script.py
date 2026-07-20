from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "synthetic_feature_profile.py"


def load_profiler_module():
    spec = importlib.util.spec_from_file_location("synthetic_feature_profile", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profile_csv_summarizes_window_features_and_target_caps(tmp_path):
    profiler = load_profiler_module()
    csv_path = tmp_path / "hourly.csv"
    start = datetime(2026, 1, 1)
    rows = []
    for index in range(96):
        seasonal = 2.0 * math.sin(2 * math.pi * index / 12)
        trend = 0.08 * index
        rows.append({"time": start + timedelta(hours=index), "target": trend + seasonal})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    profile = profiler.profile_csv(
        csv_path,
        context_length=24,
        horizon=12,
        stride=12,
        season_length=12,
        domain="energy",
        target_features=["trend_strength", "seasonal_strength", "slope_abs"],
        target_max_multiplier=1.5,
    )

    assert profile["schema_version"] == "synthetic_feature_profile.v1"
    assert profile["bucket"]["frequency"] == "h"
    assert profile["bucket"]["domain"] == "energy"
    assert profile["window_count"] == 6
    assert profile["target_columns"] == ["target"]
    assert profile["features"]["trend_strength"]["p50"] > 0
    assert profile["features"]["seasonal_strength"]["p50"] > 0
    cap = profile["target_feature_caps"]["slope_abs"]
    assert cap["multiplier"] == 1.5
    assert cap["max_allowed"] == cap["basis_value"] * 1.5
    assert profile["target_feature_caps"]["trend_strength"]["max_allowed"] <= 1.0
    assert profile["target_feature_caps"]["seasonal_strength"]["max_allowed"] <= 1.0


def test_feature_vector_reports_multitarget_correlation():
    profiler = load_profiler_module()
    base = np.linspace(0, 10, 60)
    values = np.column_stack([base, base * 2 + 1])

    features = profiler.feature_vector(values, season_length=12)

    assert features["avg_abs_target_corr"] > 0.99
    assert features["pca_top1_explained"] > 0.99
    assert features["effective_factor_rank"] < 1.1
    assert features["trend_strength"] > 0.9


def test_feature_vector_reports_spec_univariate_structure_features():
    profiler = load_profiler_module()
    t = np.arange(120, dtype=float)
    values = (
        np.sin(2 * math.pi * t / 24)
        + 0.4 * np.sin(2 * math.pi * t / 12)
        + np.where(t >= 72, 1.5, 0.0)
    )
    values[::17] += 6.0

    features = profiler.feature_vector(values, season_length=24)

    for feature in (
        "multi_period_score",
        "seasonal_drift_score",
        "change_point_shift_energy",
        "level_shift_strength",
        "volatility_shift_strength",
        "nonlinear_lag1_gain",
        "burst_rate",
    ):
        assert feature in features
    assert features["multi_period_score"] > 0
    assert features["level_shift_strength"] > 0
    assert features["burst_rate"] > 0


def test_read_uci_hydraulic_sensor_cycles_preserves_cycle_order_and_downsamples(tmp_path):
    profiler = load_profiler_module()
    archive_path = tmp_path / "hydraulic.zip"
    matrix = np.arange(240, dtype=float).reshape(2, 120)
    payload = "\n".join(" ".join(str(value) for value in row) for row in matrix)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/EPS1.txt", payload)

    values = profiler.read_uci_hydraulic_sensor_cycles(archive_path, sensor="EPS1")

    assert values.shape == (120,)
    assert np.allclose(values[:3], [0.5, 2.5, 4.5])
    assert np.allclose(values[60:63], [120.5, 122.5, 124.5])


def test_read_skchange_hvac_series_regularizes_sparse_missing_timestamps(tmp_path):
    profiler = load_profiler_module()
    csv_path = tmp_path / "data.csv"
    times = pd.date_range("2026-01-01", periods=200, freq="10min", tz="UTC")
    frame = pd.DataFrame(
        {
            "time": times.delete(10),
            "vibration": np.delete(np.arange(200, dtype=float), 10),
            "unit_id": 0,
        }
    )
    frame.to_csv(csv_path, index=False)

    values = profiler.read_skchange_hvac_series(csv_path, unit_id=0)

    assert values.shape == (200,)
    assert values[10] == 10.0


def test_gift_eval_short_term_tail_matches_frozen_protocol_and_is_removed():
    profiler = load_profiler_module()
    records = [
        ("short", np.arange(800, dtype=float)),
        ("long", np.arange(1_000, dtype=float)),
    ]

    holdout, truncated = profiler.truncate_gift_eval_official_test_tail("h", records)

    # ceil(10% * 800 / 48) == 2 rolling windows.
    assert holdout == 96
    assert truncated[0][1].shape == (704,)
    assert truncated[1][1].shape == (904,)
    assert profiler.gift_eval_short_term_test_holdout_steps(
        "M",
        [("monthly", np.arange(120, dtype=float))],
    ) == 12


def test_profile_csv_can_extract_covariate_and_hierarchy_features(tmp_path):
    profiler = load_profiler_module()
    csv_path = tmp_path / "covariate-hierarchy.csv"
    start = datetime(2026, 1, 1)
    rows = []
    for index in range(96):
        event = 1.0 if index % 24 >= 18 else 0.0
        weather = math.sin(2 * math.pi * index / 24)
        child_a = 0.5 * weather + 1.2 * event
        child_b = 0.3 * math.cos(2 * math.pi * index / 24) + 0.8 * event
        rows.append(
            {
                "time": start + timedelta(hours=index),
                "parent": child_a + child_b,
                "child_a": child_a,
                "child_b": child_b,
                "weather": weather,
                "event": event,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    profile = profiler.profile_csv(
        csv_path,
        context_length=48,
        horizon=12,
        stride=12,
        season_length=24,
        target_columns=["parent", "child_a", "child_b"],
        covariate_columns=["weather", "event"],
        hierarchy="additive_first",
    )

    assert profile["bucket"]["target_dim"] == 3
    assert profile["bucket"]["covariate_dim"] == 2
    assert profile["features"]["future_abs_covariate_target_corr"]["p50"] > 0
    assert profile["features"]["event_lift_abs"]["p50"] > 0
    assert profile["features"]["hierarchy_residual_mean_abs"]["p95"] < 1e-9


def test_profile_m5_zip_can_extract_covariate_features(tmp_path):
    profiler = load_profiler_module()
    zip_path = tmp_path / "m5.zip"
    days = [f"d_{index}" for index in range(1, 101)]
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=100, freq="D").strftime("%Y-%m-%d"),
            "wm_yr_wk": [1 + index // 7 for index in range(100)],
            "d": days,
            "event_name_1": ["Promo" if index % 14 == 0 else np.nan for index in range(100)],
            "event_name_2": [np.nan] * 100,
            "snap_CA": [1 if index % 10 < 3 else 0 for index in range(100)],
            "snap_TX": [0] * 100,
            "snap_WI": [0] * 100,
        }
    )
    sales_rows = []
    prices = []
    for item_index in range(4):
        row = {
            "id": f"ITEM_{item_index}_CA_1_validation",
            "item_id": f"ITEM_{item_index}",
            "dept_id": "HOBBIES_1",
            "cat_id": "HOBBIES",
            "store_id": "CA_1",
            "state_id": "CA",
        }
        for day_index, day in enumerate(days):
            row[day] = 2 + item_index + (3 if day_index % 14 == 0 else 0) + (1 if day_index % 10 < 3 else 0)
        sales_rows.append(row)
        for week in sorted(calendar["wm_yr_wk"].unique()):
            prices.append({"store_id": "CA_1", "item_id": f"ITEM_{item_index}", "wm_yr_wk": week, "sell_price": 1.0 + item_index + week * 0.01})
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("calendar.csv", calendar.to_csv(index=False))
        archive.writestr("sales_train_validation.csv", pd.DataFrame(sales_rows).to_csv(index=False))
        archive.writestr("sell_prices.csv", pd.DataFrame(prices).to_csv(index=False))

    profile = profiler.profile_m5_covariate(
        zip_path,
        context_length=56,
        horizon=14,
        stride=14,
        max_windows=8,
        max_series=4,
        season_length=7,
    )

    assert profile["bucket"]["frequency"] == "d"
    assert profile["bucket"]["covariate_dim"] == 4
    assert profile["covariate_columns"] == [
        "day_of_week_sin",
        "day_of_week_cos",
        "event_count",
        "snap",
    ]
    assert "sell_price" not in profile["covariate_columns"]
    assert "price_change" not in profile["covariate_columns"]
    assert "fixed before" in profile["covariate_provenance"]
    assert profile["features"]["future_abs_covariate_target_corr"]["p50"] > 0
    assert profile["features"]["event_lift_abs"]["p50"] > 0


def test_m5_sibling_panels_are_disjoint_leaf_targets() -> None:
    profiler = load_profiler_module()
    days = [f"d_{index}" for index in range(1, 9)]
    rows = []
    for item_index in range(7):
        row = {
            "id": f"ITEM_{item_index}_CA_1_validation",
            "item_id": f"ITEM_{item_index}",
            "dept_id": "HOBBIES_1",
            "store_id": "CA_1",
        }
        row.update(
            {
                day: float(item_index + day_index + 1)
                for day_index, day in enumerate(days)
            }
        )
        rows.append(row)

    panels = profiler.m5_sibling_leaf_panels(
        pd.DataFrame(rows),
        days,
        target_dim=3,
    )

    assert len(panels) == 2
    first_ids = set(panels[0][1])
    second_ids = set(panels[1][1])
    assert first_ids.isdisjoint(second_ids)
    assert panels[0][2].shape == (len(days), 3)
    assert panels[1][2].shape == (len(days), 3)
    assert not np.allclose(
        panels[0][2][:, 0] + panels[0][2][:, 1],
        panels[0][2][:, 2],
    )


def test_profile_m5_zip_can_extract_hierarchy_features(tmp_path):
    profiler = load_profiler_module()
    zip_path = tmp_path / "m5-hierarchy.zip"
    days = [f"d_{index}" for index in range(1, 101)]
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=100, freq="D").strftime("%Y-%m-%d"),
            "wm_yr_wk": [1 + index // 7 for index in range(100)],
            "d": days,
            "event_name_1": [np.nan] * 100,
            "event_name_2": [np.nan] * 100,
            "snap_CA": [0] * 100,
            "snap_TX": [0] * 100,
            "snap_WI": [0] * 100,
        }
    )
    sales_rows = []
    for dept_index, dept in enumerate(["HOBBIES_1", "HOBBIES_2"]):
        for item_index in range(2):
            row = {
                "id": f"{dept}_{item_index}_CA_1_validation",
                "item_id": f"{dept}_{item_index}",
                "dept_id": dept,
                "cat_id": "HOBBIES",
                "store_id": "CA_1",
                "state_id": "CA",
            }
            for day_index, day in enumerate(days):
                row[day] = dept_index + item_index + day_index % 5
            sales_rows.append(row)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("calendar.csv", calendar.to_csv(index=False))
        archive.writestr("sales_train_validation.csv", pd.DataFrame(sales_rows).to_csv(index=False))
        archive.writestr("sell_prices.csv", pd.DataFrame(columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(index=False))

    profile = profiler.profile_m5_hierarchy(
        zip_path,
        context_length=56,
        horizon=14,
        stride=14,
        max_windows=8,
        max_groups=4,
        season_length=7,
    )

    assert profile["bucket"]["target_dim"] == 3
    assert profile["features"]["hierarchy_residual_mean_abs"]["p95"] < 1e-9
    assert profile["features"]["avg_abs_target_corr"]["p50"] > 0


def test_profile_gefcom2014_load_zip_can_extract_temperature_covariates(tmp_path):
    profiler = load_profiler_module()
    outer_path = tmp_path / "GEFCom2014.zip"
    rows = []
    for index in range(240):
        rows.append(
            {
                "ZONEID": 1,
                "TIMESTAMP": f"1012026 {index % 24}:00",
                "LOAD": 100 + 10 * math.sin(2 * math.pi * index / 24),
                "w1": 50 + 8 * math.sin(2 * math.pi * index / 24),
                "w2": 45 + 6 * math.cos(2 * math.pi * index / 24),
            }
        )
    nested = tmp_path / "GEFCom2014-L_V2.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("Load/Task 1/L1-train.csv", pd.DataFrame(rows).to_csv(index=False))
    with zipfile.ZipFile(outer_path, "w") as archive:
        archive.write(nested, "GEFCom2014 Data/GEFCom2014-L_V2.zip")

    profile = profiler.profile_gefcom2014_load(
        outer_path,
        context_length=72,
        horizon=24,
        stride=24,
        max_windows=8,
        season_length=24,
    )

    assert profile["bucket"]["frequency"] == "h"
    assert profile["bucket"]["covariate_dim"] == 2
    assert profile["features"]["future_abs_covariate_target_corr"]["p50"] > 0.5


def _wind_nested_zone_zip(
    *,
    task: int,
    timestamps: pd.DatetimeIndex,
    predictors_only: bool,
) -> bytes:
    payload = io.BytesIO()
    prefix = (
        f"TaskExpVars{task}_W_Zone1_10"
        if predictors_only
        else f"Task{task}_W_Zone1_10"
    )
    with zipfile.ZipFile(payload, "w") as archive:
        for zone_id in range(1, 11):
            index = np.arange(len(timestamps), dtype=float)
            frame = pd.DataFrame(
                {
                    "ZONEID": zone_id,
                    "TIMESTAMP": timestamps.strftime("%Y%m%d %H:%M"),
                    "U10": 1_000.0 + zone_id + index if predictors_only else zone_id + index,
                    "V10": 2_000.0 + zone_id + index if predictors_only else 2 * zone_id + index,
                    "U100": 3_000.0 + zone_id + index if predictors_only else 3 * zone_id + index,
                    "V100": 4_000.0 + zone_id + index if predictors_only else 4 * zone_id + index,
                }
            )
            if not predictors_only:
                frame.insert(
                    2,
                    "TARGETVAR",
                    zone_id + np.sin(index / 2.0),
                )
            archive.writestr(
                f"{prefix}/{prefix.rsplit('_Zone1_10', 1)[0]}_Zone{zone_id}.csv",
                frame.to_csv(index=False),
            )
    return payload.getvalue()


def make_wind_track_fixture(tmp_path, *, predictor_steps: int = 6) -> Path:
    start = pd.Timestamp("2026-01-01 01:00")
    history = pd.date_range(start, periods=12, freq="h")
    predictor = pd.date_range(
        history[-1] + pd.Timedelta(hours=1),
        periods=predictor_steps,
        freq="h",
    )
    wind_payload = io.BytesIO()
    with zipfile.ZipFile(wind_payload, "w") as wind:
        wind.writestr(
            "Wind/Task 1/Task1_W_Zone1_10.zip",
            _wind_nested_zone_zip(
                task=1,
                timestamps=history,
                predictors_only=False,
            ),
        )
        wind.writestr(
            "Wind/Task 1/TaskExpVars1_W_Zone1_10.zip",
            _wind_nested_zone_zip(
                task=1,
                timestamps=predictor,
                predictors_only=True,
            ),
        )
        wind.writestr(
            "Wind/Task 2/Task2_W_Zone1_10.zip",
            _wind_nested_zone_zip(
                task=2,
                timestamps=history.append(predictor),
                predictors_only=False,
            ),
        )
    outer_path = tmp_path / "GEFCom2014.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.writestr(
            "GEFCom2014 Data/GEFCom2014-W_V2.zip",
            wind_payload.getvalue(),
        )
    return outer_path


def test_gefcom_wind_release_uses_task_predictors_and_next_observed_targets(
    tmp_path,
) -> None:
    profiler = load_profiler_module()
    path = make_wind_track_fixture(tmp_path, predictor_steps=6)

    releases = profiler.read_gefcom2014_wind_forecast_releases(
        path,
        minimum_future_steps=4,
    )

    assert len(releases) == 1
    release = releases[0]
    assert release["release_id"] == "GEFCom2014-W-TaskExpVars-1"
    assert release["available_future_steps"] == 6
    assert release["future"][1]["U10"].iloc[0] == 1_001.0
    assert release["future"][1]["TARGETVAR"].iloc[0] == pytest.approx(
        1.0 + math.sin(6.0)
    )
    assert "never stitched" in release["covariate_provenance"]


def test_gefcom_wind_release_fails_closed_when_horizon_is_incomplete(
    tmp_path,
) -> None:
    profiler = load_profiler_module()
    path = make_wind_track_fixture(tmp_path, predictor_steps=3)

    with pytest.raises(ValueError, match="fails closed"):
        profiler.read_gefcom2014_wind_forecast_releases(
            path,
            minimum_future_steps=4,
        )


def test_profile_tsf_panel_summarizes_multitarget_features(tmp_path):
    profiler = load_profiler_module()
    t = np.arange(96, dtype=float)
    series_lines = []
    for series_index in range(4):
        values = np.sin(2 * np.pi * t / 24 + series_index * 0.1) + series_index * 0.05
        series_lines.append("series_{}:{}".format(series_index, ",".join(str(value) for value in values)))
    tsf_text = "\n".join(
        [
            "@attribute series_name string",
            "@frequency hourly",
            "@horizon 12",
            "@missing false",
            "@equallength true",
            "@data",
            *series_lines,
        ]
    )
    zip_path = tmp_path / "panel.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("panel.tsf", tsf_text)

    profile = profiler.profile_tsf_panel(
        zip_path,
        context_length=48,
        horizon=12,
        stride=12,
        max_windows=4,
        season_length=24,
        target_dim=3,
        domain="traffic",
        dataset_name="panel smoke",
        target_features=["pca_top1_explained", "avg_abs_target_corr"],
    )

    assert profile["bucket"]["target_dim"] == 3
    assert profile["window_count"] == 4
    assert profile["features"]["pca_top1_explained"]["p50"] > 0.95
    assert profile["features"]["avg_abs_target_corr"]["p50"] > 0.95
    assert profile["target_feature_caps"]["pca_top1_explained"]["max_allowed"] <= 1.0


def test_profile_tsf_zip_reads_monash_style_series(tmp_path):
    profiler = load_profiler_module()
    values = ",".join(str(index + math.sin(index / 2)) for index in range(60))
    tsf_text = "\n".join(
        [
            "@attribute series_name string",
            "@frequency hourly",
            "@horizon 6",
            "@missing false",
            "@equallength true",
            "@data",
            f"series_1:{values}",
        ]
    )
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("sample.tsf", tsf_text)

    profile = profiler.profile_input(
        zip_path,
        context_length=24,
        horizon=6,
        season_length=12,
        input_format="auto",
        target_max_multiplier=2.0,
    )

    assert profile["bucket"]["frequency"] == "h"
    assert profile["series_count"] == 1
    assert profile["used_series_count"] == 1
    assert profile["window_count"] > 0
    assert profile["features"]["trend_strength"]["p50"] > 0


def test_read_tsf_series_records_preserves_start_timestamp(tmp_path):
    profiler = load_profiler_module()
    values = ",".join(str(index) for index in range(48))
    tsf_text = "\n".join(
        [
            "@attribute series_name string",
            "@attribute start_timestamp date",
            "@frequency hourly",
            "@horizon 6",
            "@missing false",
            "@equallength true",
            "@data",
            f"series_1:2026-01-02 03-00-00:{values}",
        ]
    )
    zip_path = tmp_path / "timestamped.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("timestamped.tsf", tsf_text)

    metadata, records = profiler.read_tsf_series_records(zip_path)
    _legacy_metadata, legacy_series = profiler.read_tsf_series(zip_path)

    assert metadata["frequency"] == "hourly"
    assert records[0].series_id == "series_1"
    assert records[0].attributes["start_timestamp"] == "2026-01-02 03-00-00"
    assert np.array_equal(records[0].values, legacy_series[0][1])


def test_profile_tsf_zip_reads_cp1252_metadata(tmp_path):
    profiler = load_profiler_module()
    values = ",".join(str(index) for index in range(48))
    tsf_text = "\n".join(
        [
            "@attribute series_name string",
            "@frequency hourly",
            "@horizon 6",
            "@missing false",
            "@equallength true",
            "# source uses windows punctuation: \u2013",
            "@data",
            f"series_1:{values}",
        ]
    )
    zip_path = tmp_path / "sample-cp1252.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("sample.tsf", tsf_text.encode("cp1252"))

    profile = profiler.profile_input(
        zip_path,
        context_length=24,
        horizon=6,
        season_length=12,
        input_format="auto",
    )

    assert profile["series_count"] == 1
    assert profile["window_count"] > 0


def test_profile_tsf_max_windows_is_global(tmp_path):
    profiler = load_profiler_module()
    series_lines = []
    for series_index in range(3):
        values = ",".join(str(index + series_index) for index in range(80))
        series_lines.append(f"series_{series_index}:{values}")
    tsf_text = "\n".join(
        [
            "@attribute series_name string",
            "@frequency hourly",
            "@horizon 6",
            "@missing false",
            "@equallength true",
            "@data",
            *series_lines,
        ]
    )
    zip_path = tmp_path / "multi-series.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("sample.tsf", tsf_text)

    profile = profiler.profile_input(
        zip_path,
        context_length=24,
        horizon=6,
        stride=6,
        max_windows=5,
        season_length=12,
        input_format="auto",
    )

    assert profile["candidate_window_count"] == 5
    assert profile["window_count"] == 5


def test_paper_v7_processed_reader_verifies_numeric_npz_and_hash(tmp_path):
    profiler = load_profiler_module()
    path = tmp_path / "fixture.npz"
    np.savez_compressed(
        path,
        timestamps=np.asarray(
            ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
            dtype="datetime64[ns]",
        ),
        values=np.asarray([[1.0], [2.0]], dtype=float),
    )
    metadata = {
        "dataset_id": "fixture",
        "output_npz": {"sha256": profiler.file_sha256(path)},
    }
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    arrays, loaded_metadata = profiler.read_paper_v7_processed_npz(
        path,
        expected_dataset_id="fixture",
        required_arrays=("timestamps", "values"),
    )

    assert arrays["values"].shape == (2, 1)
    assert not arrays["values"].dtype.hasobject
    assert loaded_metadata["_processed_npz_sha256"] == profiler.file_sha256(
        path
    )


def test_paper_v7_gefcom_reader_enforces_hourly_segments(tmp_path):
    profiler = load_profiler_module()
    path = tmp_path / "gefcom2012_load.npz"
    timestamps = np.asarray(
        [
            "2026-01-01T00:00:00",
            "2026-01-01T01:00:00",
            "2026-01-03T00:00:00",
            "2026-01-03T01:00:00",
        ],
        dtype="datetime64[ns]",
    )
    zones = np.arange(80, dtype=float).reshape(4, 20)

    def write_fixture(segment_ids: np.ndarray) -> None:
        np.savez_compressed(
            path,
            timestamps=timestamps,
            segment_ids=segment_ids,
            zones=zones,
            total=zones.sum(axis=1),
            canonical_hierarchy=np.column_stack(
                [
                    zones.sum(axis=1),
                    zones[:, :10].sum(axis=1),
                    zones[:, 10:].sum(axis=1),
                ]
            ),
            calendar_covariates=np.zeros((4, 6)),
        )
        metadata = {
            "dataset_id": "gefcom2012_load",
            "calendar_covariate_columns": list(
                profiler.PAPER_V7_GEFCOM2012_CALENDAR_COLUMNS
            ),
            "output_npz": {"sha256": profiler.file_sha256(path)},
        }
        path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

    write_fixture(np.asarray([0, 0, 1, 1], dtype=np.int32))
    arrays, _metadata = profiler.read_paper_v7_gefcom2012_processed(path)
    assert arrays["segment_ids"].tolist() == [0, 0, 1, 1]

    write_fixture(np.asarray([0, 0, 0, 0], dtype=np.int32))
    with pytest.raises(ValueError, match="hourly within each segment"):
        profiler.read_paper_v7_gefcom2012_processed(path)
