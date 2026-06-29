from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run_synthetic_v2_generator_experiment.py"


def load_experiment_module():
    repo_root = SCRIPT_PATH.parents[1]
    for path in (repo_root / "backend", repo_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("run_synthetic_v2_generator_experiment", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_acceptance_checks_and_report_rendering():
    module = load_experiment_module()
    rows = []
    for group_id in ("legacy_trend", "v2_trend", "v2_multi_seasonal"):
        for difficulty in range(1, 6):
            rows.append(
                {
                    "group_id": group_id,
                    "capability_id": group_id,
                    "difficulty": difficulty,
                    "sample_count": 2,
                    "features": {
                        "trend_strength": difficulty * 0.1,
                        "seasonal_strength": 0.8,
                        "slope_abs": 0.2 if group_id != "legacy_trend" else 0.8,
                        "curvature_abs": 0.1,
                        "noise_ratio": 0.1,
                    },
                    "baselines": {
                        "naive_mase": 1.0,
                        "seasonal_naive_mase": 1.0,
                        "seasonal_naive_mae": difficulty * 0.1,
                    },
                }
            )
    rows.append(
        {
            "group_id": "real_m4_hourly",
            "capability_id": "real_anchor",
            "difficulty": None,
            "sample_count": 2,
            "features": {"trend_strength": 0.2, "seasonal_strength": 0.9, "slope_abs": 0.1, "curvature_abs": 0.1, "noise_ratio": 0.1},
            "baselines": {"naive_mase": 1.0, "seasonal_naive_mase": 1.0, "seasonal_naive_mae": 0.2},
        }
    )
    results = {
        "sample_count": 2,
        "rows": rows,
        "checks": module.acceptance_checks(rows),
    }

    report = module.render_report(results, m4_path=Path("runtime/m4.zip"), output_dir=Path("runtime/out"))

    assert results["checks"]["v2_trend_strength_monotonic"] is True
    assert results["checks"]["legacy_trend_slope_mean_within_cap"] is False
    assert "Synthetic v2 Generator Experiment" in report
    assert "v2_multi_seasonal_naive_mae_growth" not in report
    assert "增长倍数" in report
