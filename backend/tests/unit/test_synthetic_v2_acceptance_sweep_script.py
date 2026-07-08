from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run_synthetic_v2_acceptance_sweep.py"


def load_sweep_module():
    repo_root = SCRIPT_PATH.parents[1]
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    spec = importlib.util.spec_from_file_location("run_synthetic_v2_acceptance_sweep", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_cap_sets_sweeps_profile_and_event_multipliers():
    module = load_sweep_module()

    cap_sets = module.build_cap_sets()

    assert "current" in cap_sets
    assert "profile_m1_5_event5" in cap_sets
    assert "profile_m1_5_event2" in cap_sets
    assert cap_sets["profile_m1_5_event5"]["caps"]["covariate_response"]["event_lift_abs"] > cap_sets["profile_m1_5_event2"]["caps"]["covariate_response"]["event_lift_abs"]
    assert cap_sets["profile_m2_5_event5"]["caps"]["trend"]["trend_strength"] >= cap_sets["profile_m1_event5"]["caps"]["trend"]["trend_strength"]


def test_evaluate_cell_reports_attempts_and_failed_features():
    module = load_sweep_module()
    sample_attempts = [
        [
            {"attempt": 1, "features": {"trend_strength": 2.0, "noise_ratio": 0.1}},
            {"attempt": 2, "features": {"trend_strength": 0.5, "noise_ratio": 0.1}},
        ],
        [
            {"attempt": 1, "features": {"trend_strength": 2.0, "noise_ratio": 2.0}},
            {"attempt": 2, "features": {"trend_strength": 2.0, "noise_ratio": 2.0}},
        ],
    ]

    cell = module.evaluate_cell(
        "unit",
        "trend",
        1,
        sample_attempts,
        caps={"trend": {"trend_strength": 1.0, "noise_ratio": 1.0}},
        mins={},
    )

    assert cell["acceptance_rate"] == 0.5
    assert cell["mean_attempts_accepted"] == 2.0
    assert cell["first_attempt_failed_features"]["trend_strength"] == 2
    assert cell["terminal_failed_features"]["noise_ratio"] == 1


def test_recommend_strategy_prefers_smaller_profile_multiplier():
    module = load_sweep_module()
    template = {
        "min_acceptance_rate": 1.0,
        "cells_below_0_95": 0,
        "max_mean_attempts_accepted": 1.0,
    }

    recommendation = module.recommend_strategy(
        {
            "profile_m2_5_event5": template,
            "profile_m2_event5": {**template, "max_mean_attempts_accepted": 1.1},
        }
    )

    assert recommendation["strategy_id"] == "profile_m2_event5"
