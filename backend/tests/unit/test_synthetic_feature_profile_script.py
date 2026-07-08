from __future__ import annotations

import importlib.util
import math
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


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
    assert profile["features"]["future_abs_covariate_target_corr"]["p50"] > 0
    assert profile["features"]["event_lift_abs"]["p50"] > 0


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
