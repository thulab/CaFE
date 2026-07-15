from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "build_synthetic_v2_generator_conditioning_artifact.py"


def load_calibration_module():
    repo_root = SCRIPT_PATH.parents[1]
    for path in (repo_root / "backend", repo_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "build_synthetic_v2_generator_conditioning_artifact",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def feature_summary(*, noise: float, seasonal: float, trend: float, spike: float) -> dict:
    return {
        "noise_ratio": {"p50": noise, "iqr": 0.1},
        "seasonal_strength": {"p50": seasonal, "iqr": 0.1},
        "trend_strength": {"p50": trend, "iqr": 0.1},
        "spike_rate": {"p50": spike, "iqr": 0.02},
    }


def test_profile_nuisance_changes_with_real_distribution():
    module = load_calibration_module()
    smooth = module.derive_profile_nuisance(
        feature_summary(noise=0.08, seasonal=0.92, trend=0.03, spike=0.0),
        168,
        24,
    )
    noisy = module.derive_profile_nuisance(
        feature_summary(noise=0.30, seasonal=0.70, trend=0.10, spike=0.07),
        168,
        24,
    )

    assert noisy["noise_scale_multiplier"] > smooth["noise_scale_multiplier"]
    assert noisy["seasonal_amplitude_multiplier"] < smooth["seasonal_amplitude_multiplier"]
    assert noisy["noise_degrees_of_freedom"] > 2
    assert smooth["noise_degrees_of_freedom"] == 0


def test_intensity_lambda_projection_is_strict_and_bounded():
    module = load_calibration_module()
    values = module.strictly_monotone_lambdas([0.0, 0.0, 0.9, 1.0, 1.0])

    assert all(0.0 <= value <= 1.0 for value in values)
    assert all(right - left >= 0.049 for left, right in zip(values, values[1:]))
