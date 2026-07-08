from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run_synthetic_v2_profile_smoke.py"


def load_smoke_module():
    scripts_dir = SCRIPT_PATH.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("run_synthetic_v2_profile_smoke", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_report_summarizes_profiles():
    module = load_smoke_module()
    profiles = {
        spec.profile_id: {
            "window_count": 10,
            "used_series_count": 2,
            "features": {
                "trend_strength": {"p50": 0.25, "p95": 0.75},
                "seasonal_strength": {"p50": 0.5, "p95": 0.9},
                "slope_abs": {"p95": 0.3},
                "curvature_abs": {"p95": 0.2},
                "noise_ratio": {"p95": 0.1},
            },
            "target_feature_caps": {
                "trend_strength": {"max_allowed": 1.0},
                "seasonal_strength": {"max_allowed": 1.0},
                "slope_abs": {"max_allowed": 0.45},
                "curvature_abs": {"max_allowed": 0.3},
            },
        }
        for spec in module.PROFILE_SPECS
    }

    report = module.render_report(profiles, output_dir=Path("runtime/out"), data_dir=Path("runtime/data"))

    assert "Synthetic v2 真实数据 Profile 烟测" in report
    assert "us_births_weekly" in report
    assert "electricity_hourly_daily_168ctx" in report
    assert "traffic_hourly_panel_168ctx" in report
    assert "Spec 主特征覆盖" in report
    assert "change_point_shift_energy" in report
    assert "future_abs_covariate_target_corr" in report
    assert "0.25/0.75/1" in report
    assert "python3 scripts/run_synthetic_v2_profile_smoke.py" in report
