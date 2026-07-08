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


def test_evaluate_risk_flags_exact_copy_and_spares_far_sample():
    module = load_calibration_module()
    train = [
        {"raw": np.asarray([0.0, 0.0]), "features": {"trend_strength": 0.1}},
        {"raw": np.asarray([1.0, 1.0]), "features": {"trend_strength": 0.2}},
        {"raw": np.asarray([2.0, 2.0]), "features": {"trend_strength": 0.3}},
    ]
    thresholds = {
        "raw_mae_p01": 0.01,
        "raw_mae_p05": 0.05,
        "raw_l2_p01": 0.01,
        "raw_l2_p05": 0.05,
        "feature_l2_p01": 0.01,
        "raw_mae_nndr_p01": 0.05,
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
        [{"raw": np.asarray([100.0, 100.0]), "features": {"trend_strength": 10.0}}],
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
                    for label in ("real_holdout", "exact_copy", "jitter_copy", "normal_synthetic")
                },
            }
        ],
        "overall": {
            "exact_copy_strict_risk_min": 1.0,
            "jitter_copy_combined_risk_min": 1.0,
            "normal_synthetic_combined_risk_max": 0.0,
        },
    }

    report = module.render_report(summary, output_dir=tmp_path)

    assert "Synthetic v2 Near-Distance Calibration" in report
    assert "summary.json" in report
