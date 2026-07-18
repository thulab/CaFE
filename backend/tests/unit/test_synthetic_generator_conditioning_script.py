from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[3]
    / "scripts"
    / "build_synthetic_v2_generator_conditioning_artifact.py"
)


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


def test_profile_nuisance_changes_with_dataset_distribution():
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
    assert noisy["seasonal_amplitude_multiplier"] < smooth[
        "seasonal_amplitude_multiplier"
    ]
    assert noisy["noise_degrees_of_freedom"] > 2
    assert smooth["noise_degrees_of_freedom"] == 0


def test_dataset_local_targets_require_five_ordered_separated_levels():
    module = load_calibration_module()

    supported = module.local_target_support([0.10, 0.20, 0.35, 0.55, 0.90])
    repeated = module.local_target_support([0.10, 0.10, 0.35, 0.55, 0.90])
    flat = module.local_target_support([0.20] * 5)

    assert supported["status"] == "supported"
    assert repeated["reason"] == "insufficient_local_intensity_spacing"
    assert flat["reason"] == "insufficient_local_target_range"


def test_structurally_unsupported_capability_is_recorded_without_calibration():
    module = load_calibration_module()

    result = module.local_target_support(
        [],
        structural_status={
            "status": "unsupported",
            "reason": "variable_structure_not_supported",
            "detail": "requires multiple target variables",
        },
    )

    assert result == {
        "status": "unsupported",
        "reason": "variable_structure_not_supported",
        "detail": "requires multiple target variables",
    }


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


def test_capability_calibration_uses_dataset_local_targets_and_cross_fit():
    module = load_calibration_module()
    module.structure_scale_grid = lambda _capability_id: (0.1, 0.2, 0.4)
    module.mean_feature_over_seed_banks = lambda **kwargs: (
        kwargs["parameters"]["structure_scale"] * kwargs["intensity_lambda"]
    )
    module.simulate_feature_means = lambda **kwargs: {
        kwargs["feature_names"][0]: (
            kwargs["parameters"]["structure_scale"]
            * kwargs["intensity_lambda"]
        )
    }

    parameters, lambdas, summary = module.calibrate_capability_conditioning(
        spec=SimpleNamespace(
            profile_id="test",
            context_length=168,
            horizon=24,
            target_dim=1,
            season_length=24,
        ),
        capability_id="trend",
        profile_nuisance={},
        real_feature_summary={},
        target_values=[0.06, 0.12, 0.18, 0.24, 0.30],
        sample_count=8,
        seed=7,
    )

    assert np.isclose(parameters["structure_scale"], 0.3)
    assert np.allclose(lambdas, [0.2, 0.4, 0.6, 0.8, 1.0])
    assert summary["target_values"] == [0.06, 0.12, 0.18, 0.24, 0.3]
    assert summary["status"] == "supported"
    assert summary["fit_seed_bank_count"] == 2
    assert summary["fit_sample_count"] == 16
    assert summary["validation_sample_count"] == 256


def test_high_variance_capability_uses_larger_seed_bank():
    module = load_calibration_module()
    observed_sample_counts: list[int] = []
    module.structure_scale_grid = lambda _capability_id: (0.1, 0.2)

    def fake_mean(**kwargs):
        observed_sample_counts.append(kwargs["sample_count"])
        return kwargs["parameters"]["structure_scale"] * kwargs["intensity_lambda"]

    module.mean_feature_over_seed_banks = fake_mean
    module.simulate_feature_means = lambda **kwargs: {
        kwargs["feature_names"][0]: (
            kwargs["parameters"]["structure_scale"]
            * kwargs["intensity_lambda"]
        )
    }

    _, _, summary = module.calibrate_capability_conditioning(
        spec=SimpleNamespace(
            profile_id="test",
            context_length=168,
            horizon=24,
            target_dim=1,
            season_length=24,
        ),
        capability_id="nonlinear_persistence",
        profile_nuisance={},
        real_feature_summary={},
        target_values=[0.02, 0.04, 0.06, 0.08, 0.10],
        sample_count=8,
        seed=7,
    )

    assert set(observed_sample_counts) == {128}
    assert summary["fit_sample_count"] == 512
    assert summary["fit_seed_bank_count"] == 4
    assert summary["validation_sample_count"] == 1024


def test_profile_identity_is_dataset_local_and_has_no_family_or_canonical_scale():
    module = load_calibration_module()

    assert module.DATASET_ID_BY_PROFILE_ID["m4_hourly_daily_168ctx"] == "m4_hourly"
    assert module.reference_percentile_levels("regime_switching") == (
        0.10,
        0.30,
        0.50,
        0.70,
        0.90,
    )
    assert not hasattr(module, "CANONICAL_REFERENCE_PROFILE_IDS")
    assert not hasattr(module, "CANONICAL_SCALE_ID")
