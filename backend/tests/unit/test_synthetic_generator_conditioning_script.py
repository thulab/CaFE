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


def test_canonical_resolution_preserves_endpoints_and_separates_near_duplicate_levels():
    module = load_calibration_module()

    resolved = module.enforce_target_resolution(
        np.asarray([0.0, 0.30, 0.32, 0.35, 1.0])
    )

    assert np.allclose(resolved, [0.0, 0.30, 0.40, 0.50, 1.0])
    assert np.all(np.diff(resolved) >= 0.10 - 1e-12)


def test_capability_calibration_interpolates_structure_scale_and_records_cross_fit():
    module = load_calibration_module()
    module.structure_scale_grid = lambda _capability_id: (0.1, 0.2, 0.4)
    module.mean_feature_over_seed_banks = lambda **kwargs: (
        kwargs["parameters"]["structure_scale"] * kwargs["intensity_lambda"]
    )
    module.simulate_feature_means = lambda **kwargs: {
        kwargs["feature_names"][0]: (
            kwargs["parameters"]["structure_scale"] * kwargs["intensity_lambda"]
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
        canonical_target_values=[0.06, 0.12, 0.18, 0.24, 0.30],
        sample_count=8,
        seed=7,
    )

    assert np.isclose(parameters["structure_scale"], 0.3)
    assert np.allclose(lambdas, [0.2, 0.4, 0.6, 0.8, 1.0])
    assert summary["status"] == "supported"
    assert summary["fit_seed_bank_count"] == 2
    assert summary["fit_sample_count"] == 16
    assert summary["validation_sample_count"] == 256


def test_nonlinear_calibration_uses_a_larger_seed_bank_for_stable_inverse_fit():
    module = load_calibration_module()
    observed_sample_counts: list[int] = []
    module.structure_scale_grid = lambda _capability_id: (0.1, 0.2)

    def fake_mean(**kwargs):
        observed_sample_counts.append(kwargs["sample_count"])
        return (
            kwargs["parameters"]["structure_scale"]
            * kwargs["intensity_lambda"]
        )

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
        canonical_target_values=[0.02, 0.04, 0.06, 0.08, 0.10],
        sample_count=8,
        seed=7,
    )

    assert set(observed_sample_counts) == {128}
    assert summary["fit_sample_count"] == 512
    assert summary["fit_samples_per_seed_bank"] == 128
    assert summary["fit_seed_bank_count"] == 4
    assert summary["validation_sample_count"] == 1024


def test_canonical_targets_use_one_curve_per_profile_not_bucket_row_counts():
    module = load_calibration_module()
    module.CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY = {
        "trend": ("profile_low", "profile_middle", "profile_high"),
    }

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
            capability_parameter_counts={"trend": len(repeated_values)},
            capability_qualification_summaries={},
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
            "profile_middle": profile_input(
                "profile_middle",
                [0.3, 0.4, 0.5, 0.6, 0.7],
                [0.5],
            ),
        }
    )

    assert definitions["trend"]["target_values"] == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert definitions["trend"]["profile_weighting"] == "equal"


def test_regime_targets_span_from_predictability_boundary_to_real_q90():
    module = load_calibration_module()
    module.CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY = {
        "regime_switching": ("profile_a", "profile_b"),
    }

    def profile_input(profile_id: str, curve: list[float]):
        return module.ProfileCalibrationInput(
            spec=SimpleNamespace(
                profile_id=profile_id,
                synthetic_capabilities=("regime_switching",),
            ),
            parameter_window_count=40,
            split_summary={},
            real_feature_summary={},
            profile_nuisance={},
            local_target_quantiles={
                "regime_switching": {
                    "regime_clock_history_incremental_r2": curve,
                },
            },
            primary_values={
                "regime_switching": np.asarray(curve, dtype=float),
            },
            capability_parameter_counts={"regime_switching": 40},
            capability_qualification_summaries={},
        )

    definitions = module.derive_canonical_target_definitions(
        {
            "profile_a": profile_input(
                "profile_a",
                [0.25, 0.27, 0.28, 0.29, 0.30],
            ),
            "profile_b": profile_input(
                "profile_b",
                [0.27, 0.28, 0.29, 0.30, 0.32],
            ),
        }
    )

    regime = definitions["regime_switching"]
    assert np.allclose(regime["target_values"], np.linspace(0.10, 0.31, 5))
    assert (
        regime["target_resolution"]["method"]
        == "qualification_boundary_to_q90_linear_grid"
    )


def test_nonlinear_targets_span_from_adjusted_r2_null_to_real_q90():
    module = load_calibration_module()
    module.CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY = {
        "nonlinear_persistence": ("profile_a", "profile_b", "profile_c"),
    }

    def profile_input(profile_id: str, curve: list[float]):
        return module.ProfileCalibrationInput(
            spec=SimpleNamespace(
                profile_id=profile_id,
                synthetic_capabilities=("nonlinear_persistence",),
            ),
            parameter_window_count=40,
            split_summary={},
            real_feature_summary={},
            profile_nuisance={},
            local_target_quantiles={
                "nonlinear_persistence": {
                    "nonlinear_conditional_gain": curve,
                },
            },
            primary_values={
                "nonlinear_persistence": np.asarray(curve, dtype=float),
            },
            capability_parameter_counts={"nonlinear_persistence": 40},
            capability_qualification_summaries={},
        )

    definitions = module.derive_canonical_target_definitions(
        {
            "profile_a": profile_input(
                "profile_a",
                [0.0001, 0.0005, 0.001, 0.003, 0.005],
            ),
            "profile_b": profile_input(
                "profile_b",
                [0.0002, 0.0006, 0.002, 0.004, 0.007],
            ),
            "profile_c": profile_input(
                "profile_c",
                [0.0003, 0.0007, 0.003, 0.005, 0.009],
            ),
        }
    )

    nonlinear = definitions["nonlinear_persistence"]
    assert np.allclose(
        nonlinear["target_values"],
        np.linspace(0.0, 0.007, 5),
    )
    assert (
        nonlinear["target_resolution"]["method"]
        == "adjusted_r2_null_to_q90_linear_grid"
    )


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
    assert module.RESEARCH_ONLY_CONDITIONING_PROFILE_IDS == (
        "electricity_hourly_daily_2048ctx_24h",
    )
    assert "electricity_hourly_daily_2048ctx_24h" not in module.ONLINE_CONDITIONING_PROFILE_IDS
    assert set(module.ONLINE_CONDITIONING_PROFILE_IDS).isdisjoint(
        module.RESEARCH_ONLY_CONDITIONING_PROFILE_IDS
    )


def test_regime_scale_uses_only_qualified_specialist_reference_profiles():
    module = load_calibration_module()

    assert module.CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY["regime_switching"] == (
        "uci_hydraulic_eps1_420ctx_60h",
        "skchange_hvac_unit0_504ctx_144h",
    )
    assert (
        "m4_hourly_daily_168ctx"
        not in module.CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY["regime_switching"]
    )
    assert (
        "m4_hourly_daily_168ctx"
        in module.CANONICAL_REFERENCE_PROFILE_IDS_BY_CAPABILITY["trend"]
    )
