from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


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
