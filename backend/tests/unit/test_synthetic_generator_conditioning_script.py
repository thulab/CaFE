from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
    assert all(right - left >= 0.009 for left, right in zip(values, values[1:]))


def test_inverse_calibration_interpolates_a_monotone_feature_curve():
    module = load_calibration_module()

    values = module.invert_monotone_feature_curve(
        {0.0: 0.1, 0.5: 0.3, 1.0: 0.7},
        [0.1, 0.2, 0.3, 0.5, 0.8],
    )

    assert np.allclose(values, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_inverse_calibration_projects_noisy_grid_to_monotone_support():
    module = load_calibration_module()

    values = module.invert_monotone_feature_curve(
        {0.0: 0.1, 0.5: 0.4, 0.75: 0.35, 1.0: 0.8},
        [0.6],
    )

    assert np.allclose(values, [0.875])


def test_canonical_targets_use_one_curve_per_profile_not_bucket_row_counts():
    module = load_calibration_module()
    module.CANONICAL_REFERENCE_PROFILE_IDS = ("profile_low", "profile_high")

    def profile_input(profile_id: str, curve: list[float], repeated_values: list[float]):
        return module.ProfileCalibrationInput(
            spec=SimpleNamespace(
                profile_id=profile_id,
                synthetic_capabilities=("trend",),
            ),
            parameter_window_count=len(repeated_values),
            split_summary={},
            real_feature_summary={},
            profile_nuisance={},
            local_target_quantiles={"trend": {"trend_strength": curve}},
            primary_values={"trend": np.asarray(repeated_values, dtype=float)},
        )

    definitions = module.derive_canonical_target_definitions(
        {
            "profile_low": profile_input(
                "profile_low",
                [0.0, 0.1, 0.2, 0.3, 0.4],
                [0.0] * 1_000,
            ),
            "profile_high": profile_input(
                "profile_high",
                [0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0] * 10,
            ),
        }
    )

    assert definitions["trend"]["target_values"] == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert definitions["trend"]["profile_weighting"] == "equal"


def test_empirical_percentiles_expose_when_canonical_strength_is_outside_local_support():
    module = load_calibration_module()

    percentiles = module.empirical_percentiles(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        [0.05, 0.2, 0.35, 0.5, 0.9],
    )

    assert percentiles == [0.0, 0.5, 0.75, 1.0, 1.0]


def test_canonical_scale_fingerprint_changes_with_reference_or_target_definition():
    module = load_calibration_module()
    definitions = {
        "trend": {
            "primary_feature": "trend_strength",
            "target_values": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    }
    first = module.canonical_scale_fingerprint(definitions)
    definitions["trend"]["target_values"][-1] = 0.6
    second = module.canonical_scale_fingerprint(definitions)

    assert len(first) == 16
    assert first != second


def test_research_window_profiles_do_not_reweight_the_canonical_scale():
    module = load_calibration_module()

    assert "electricity_hourly_daily_168ctx" in module.CANONICAL_REFERENCE_PROFILE_IDS
    assert "electricity_hourly_daily_2048ctx_24h" not in module.CANONICAL_REFERENCE_PROFILE_IDS
    assert "electricity_hourly_daily_2048ctx_24h" in module.CONDITIONING_PROFILE_IDS
