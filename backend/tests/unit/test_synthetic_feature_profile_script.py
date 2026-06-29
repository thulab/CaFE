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
    assert features["trend_strength"] > 0.9


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
