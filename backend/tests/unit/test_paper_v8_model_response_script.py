from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[3] / "scripts" / "run_paper_v8_model_response.py"
)


def load_response_module():
    spec = importlib.util.spec_from_file_location(
        "run_paper_v8_model_response_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_for(
    capability_id: str,
    target: np.ndarray,
    *,
    context_length: int,
    season_length: int = 4,
    generation_metadata: dict | None = None,
) -> dict:
    return {
        "capability_id": capability_id,
        "target": target.tolist(),
        "context_length": context_length,
        "horizon": target.shape[0] - context_length,
        "season_length": season_length,
        "generation_metadata": generation_metadata or {},
    }


def test_common_factor_and_hierarchy_metrics_are_ideal_for_exact_forecasts():
    response = load_response_module()
    time = np.linspace(-2.0, 2.0, 36)
    factor = np.column_stack([time, -1.5 * time, 0.7 * time])
    factor_sample = sample_for(
        "common_factor",
        factor,
        context_length=24,
        generation_metadata={"protected_target_index": 0},
    )

    factor_metrics = response.prediction_metrics(
        factor_sample,
        factor[24:],
    )

    assert factor_metrics["factor_loading_cosine"] == pytest.approx(1.0)
    assert factor_metrics["factor_trajectory_correlation"] == pytest.approx(1.0)
    assert factor_metrics["factor_score_nrmse"] == pytest.approx(0.0)
    assert factor_metrics["common_component_nmae"] == pytest.approx(0.0)
    assert factor_metrics["protected_target_mae"] == pytest.approx(0.0)
    assert factor_metrics["protected_target_nmae"] == pytest.approx(0.0)

    child_a = np.sin(np.arange(36) / 3)
    child_b = np.cos(np.arange(36) / 4)
    hierarchy = np.column_stack(
        [child_a + child_b, child_a, child_b]
    )
    hierarchy_sample = sample_for(
        "hierarchical_coherence",
        hierarchy,
        context_length=24,
    )

    hierarchy_metrics = response.prediction_metrics(
        hierarchy_sample,
        hierarchy[24:],
    )

    assert hierarchy_metrics["coherence_nmae"] == pytest.approx(0.0)
    assert hierarchy_metrics["aggregation_correlation"] == pytest.approx(1.0)
    assert hierarchy_metrics["child_contrast_correlation"] == pytest.approx(1.0)
    assert hierarchy_metrics["child_contrast_nmae"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("capability_id", "generation_metadata", "expected_metrics"),
    [
        (
            "trend",
            {},
            {
                "trend_slope_relative_abs_error": 0.0,
                "trend_curvature_relative_abs_error": 0.0,
                "trend_direction_accuracy": 1.0,
                "trend_curvature_component_nrmse": 0.0,
                "trend_curvature_sign_accuracy": 1.0,
                "trend_curvature_magnitude_ratio": 1.0,
            },
        ),
        (
            "multi_seasonal",
            {"periods": [6.0, 9.0]},
            {
                "seasonal_spectral_amplitude_relative_error": 0.0,
                "seasonal_spectral_phase_amplitude_alignment": 1.0,
            },
        ),
        (
            "time_varying_seasonality",
            {"primary_period": 8.0},
            {
                "modulation_envelope_nmae": 0.0,
                "modulation_phase_alignment": 1.0,
                "instantaneous_frequency_nmae": 0.0,
            },
        ),
        (
            "nonlinear_persistence",
            {
                "nonlinear_lag": 3,
                "seasonal_lag": 8,
                "nonlinear_transform": "signed_rational_quadratic",
            },
            {
                "nonlinear_recurrence_residual_nrmse": 0.0,
                "nonlinear_delayed_response_nrmse": 0.0,
                "nonlinear_delayed_response_correlation": 1.0,
            },
        ),
    ],
)
def test_univariate_mechanism_metrics_are_ideal_for_exact_forecast(
    capability_id,
    generation_metadata,
    expected_metrics,
):
    response = load_response_module()
    time = np.arange(48, dtype=float)
    target = (
        0.002 * time * time
        + np.sin(2.0 * np.pi * time / 8.0)
        + 0.3 * np.sin(2.0 * np.pi * time / 13.0)
    )[:, None]
    sample = sample_for(
        capability_id,
        target,
        context_length=32,
        season_length=8,
        generation_metadata=generation_metadata,
    )

    metrics = response.prediction_metrics(sample, target[32:])

    for name, expected in expected_metrics.items():
        assert metrics[name] == pytest.approx(expected, abs=1e-9)


def test_trend_curvature_component_reports_sign_and_magnitude_separately():
    response = load_response_module()
    time = np.linspace(-1.0, 1.0, 48)
    truth = (time * time + 0.2 * time)[:, None]
    forecast = (-0.5 * time * time + 0.2 * time)[:, None]

    metrics = response.trend_recovery_metrics(truth, forecast)

    assert metrics["trend_curvature_component_nrmse"] == pytest.approx(1.5)
    assert metrics["trend_curvature_sign_accuracy"] == pytest.approx(0.0)
    assert metrics["trend_curvature_magnitude_ratio"] == pytest.approx(0.5)


def test_regime_and_event_metrics_are_ideal_for_exact_forecast():
    response = load_response_module()
    context = 20
    horizon = 12
    regime = np.zeros((context + horizon, 1), dtype=float)
    regime[23:28] = 2.0
    regime[28:] = -1.0
    regime_sample = sample_for(
        "regime_switching",
        regime,
        context_length=context,
        generation_metadata={"cut_points": [23, 28]},
    )
    regime_metrics = response.prediction_metrics(
        regime_sample,
        regime[context:],
    )

    assert regime_metrics["regime_jump_nmae"] == pytest.approx(0.0)
    assert regime_metrics["regime_jump_amplitude_ratio"] == pytest.approx(1.0)
    assert regime_metrics["regime_jump_sign_accuracy"] == pytest.approx(1.0)

    event = np.sin(np.arange(context + horizon) / 5.0)[:, None] * 0.05
    event[23, 0] += 2.0
    event[29, 0] += 1.5
    event_sample = sample_for(
        "predictable_intermittency",
        event,
        context_length=context,
        generation_metadata={
            "pulse_centers": [23, 29],
            "pulse_width": 1.0,
        },
    )
    event_metrics = response.prediction_metrics(
        event_sample,
        event[context:],
    )

    assert event_metrics["event_peak_timing_widths"] == pytest.approx(0.0)
    assert event_metrics["event_peak_amplitude_nmae"] == pytest.approx(0.0)
    assert event_metrics["event_window_nmae"] == pytest.approx(0.0)
    assert event_metrics["background_window_nmae"] == pytest.approx(0.0)


def test_event_metrics_omit_background_when_event_windows_cover_future():
    response = load_response_module()
    context = 20
    horizon = 4
    event = np.sin(np.arange(context + horizon) / 5.0)[:, None]
    sample = sample_for(
        "predictable_intermittency",
        event,
        context_length=context,
        generation_metadata={
            "pulse_centers": [21, 23],
            "pulse_width": 1.0,
        },
    )

    metrics = response.prediction_metrics(sample, event[context:])

    assert metrics["event_window_nmae"] == pytest.approx(0.0)
    assert "background_window_nmae" not in metrics


def test_cross_series_metric_scores_only_driver_covered_responder_steps():
    response = load_response_module()
    target = np.column_stack(
        [
            np.arange(12, dtype=float),
            np.sin(np.arange(12, dtype=float)),
            np.cos(np.arange(12, dtype=float)),
        ]
    )
    sample = sample_for(
        "cross_series_dependence",
        target,
        context_length=8,
        generation_metadata={
            "responder_indices": [1, 2],
            "history_covered_forecast_steps": 2,
        },
    )
    forecast = target[8:].copy()
    forecast[2:, 1:] += 10.0

    metrics = response.prediction_metrics(sample, forecast)

    assert metrics["driver_covered_responder_mae"] == pytest.approx(0.0)
    assert metrics["driver_covered_responder_correlation"] == pytest.approx(1.0)
    assert metrics["responder_mae"] == pytest.approx(5.0)
    assert metrics["history_covered_forecast_steps"] == 2


def test_cross_lag_linear_probe_recovers_history_identifiable_effect():
    response = load_response_module()
    context = 24
    horizon = 4
    delay = 4
    driver = np.sin(np.arange(context + horizon, dtype=float) / 2.7)
    target = np.zeros((context + horizon, 3), dtype=float)
    target[:, 0] = driver
    target[delay:, 1] = 1.5 + 2.0 * driver[:-delay]
    target[delay:, 2] = -0.5 - 0.75 * driver[:-delay]
    sample = sample_for(
        "cross_series_dependence",
        target,
        context_length=context,
        generation_metadata={
            "cross_lag_steps": delay,
            "driver_index": 0,
            "responder_indices": [1, 2],
        },
    )

    forecast = response.baseline_forecast(
        sample,
        "cross_lag_linear_probe",
    )

    assert forecast[:, 1:] == pytest.approx(target[context:, 1:])


def test_master_expansion_creates_common_context_views_of_one_cross_series_dgp():
    response = load_response_module()
    master_context = response.v8_common.CONTEXT_LENGTH
    horizon = response.v8_common.HORIZON
    target, metadata, _ = response.v8_pilot.generate_deterministic_sample(
        "cross_series_dependence",
        master_context + horizon,
        master_context,
        3,
        24,
        5,
        np.random.default_rng(71),
        counterfactual_variant=0,
    )
    target, _ = (
            response.v8_pilot.standardize_cross_series_counterfactual_member(
                target,
                context_length=master_context,
                metadata=metadata,
            )
    )
    master = {
        "sample_id": "master",
        "master_sample_id": "master",
        "paired_group_id": "group",
        "counterfactual_pair_id": "pair",
        "capability_id": "cross_series_dependence",
        "context_length": master_context,
        "horizon": horizon,
        "target_dim": 3,
        "target_feature": "cross_series_incremental_r2",
        "target": target.tolist(),
        "covariates": None,
        "generation_metadata": metadata,
    }

    views = response.expand_master_samples(
        [master],
        context_lengths=response.v8_common.VIEW_CONTEXT_LENGTHS,
    )

    assert [row["context_length"] for row in views] == list(
        response.v8_common.VIEW_CONTEXT_LENGTHS
    )
    assert all(
        np.asarray(row["target"]).shape
        == (row["context_length"] + 48, 3)
        for row in views
    )
    assert all(row["source_master_sample_id"] == "master" for row in views)
    delay = int(metadata["cross_lag_steps"])
    assert all(
        row["generation_metadata"]["counterfactual_driver_slice"]
        == [row["context_length"] - delay, row["context_length"]]
        for row in views
    )


def test_context_profile_reports_diagnostic_best_without_pooling_views():
    response = load_response_module()
    predictions = []
    for context_length, mase in ((96, 0.8), (168, 0.6), (336, 0.7)):
        predictions.append(
            {
                "model_id": "model",
                "variant": "native",
                "evaluation_table": "main",
                "capability_id": "trend",
                "context_length": context_length,
                "metrics": {
                    "mase": mase,
                    "future_curve_correlation": 0.5,
                    "flat_forecast": 0.0,
                },
            }
        )

    profile = response.context_performance_profiles(predictions)

    assert profile["diagnostic_preferred_context_by_model"]["model"][
        "context_length"
    ] == 168
    assert {
        row["context_length"]
        for row in profile["profiles"]
        if row["capability_id"] == "__overall__"
    } == {96, 168, 336}


def test_master_context_audit_does_not_count_suffix_views_as_samples():
    response = load_response_module()
    predictions = []
    samples = []
    for context_length in (96, 168):
        sample_id = f"master::L{context_length}"
        samples.append(
            {
                "sample_id": sample_id,
                "evaluation_table": "main",
                "capability_id": "trend",
                "generator_family_role": "primary",
                "counterfactual_pair_id": None,
            }
        )
        for model_id, mase in (
            ("seasonal_naive", 1.0),
            ("Chronos-2", 0.5),
        ):
            predictions.append(
                {
                    "model_id": model_id,
                    "variant": "native",
                    "sample_id": sample_id,
                    "master_sample_id": "master",
                    "evaluation_table": "main",
                    "capability_id": "trend",
                    "generator_family_role": "primary",
                    "intensity": 5,
                    "context_length": context_length,
                    "metrics": {
                        "mae": mase,
                        "mase": mase,
                        "future_curve_correlation": 1.0,
                        "flat_forecast": 0.0,
                    },
                }
            )

    audit = response.master_context_audit(predictions, samples)

    assert audit["context_length"] == 168
    assert audit["distinct_master_sample_count"] == 1
    aggregate = next(
        row
        for row in audit["aggregates"]
        if row["model_id"] == "Chronos-2"
    )
    assert aggregate["sample_count"] == 1
    assert aggregate["metrics"]["mase"] == pytest.approx(0.5)


def cross_prediction(
    *,
    variant: str,
    sample_id: str,
    intensity: int,
    loss: float,
    correlation: float,
) -> dict:
    return {
        "model_id": "Chronos-2",
        "variant": variant,
        "sample_id": sample_id,
        "evaluation_table": "main",
        "capability_id": "cross_series_dependence",
        "generator_family_role": "primary",
        "intensity": intensity,
        "metrics": {
            "driver_covered_responder_mae": loss,
            "driver_covered_responder_correlation": correlation,
        },
    }


def test_cross_series_audit_uses_ratio_of_mean_losses_and_splits_by_intensity():
    response = load_response_module()
    predictions = [
        cross_prediction(
            variant="native",
            sample_id="i1",
            intensity=1,
            loss=1.0,
            correlation=0.8,
        ),
        cross_prediction(
            variant="forced_independent_targets",
            sample_id="i1",
            intensity=1,
            loss=2.0,
            correlation=0.6,
        ),
        cross_prediction(
            variant="native",
            sample_id="i5",
            intensity=5,
            loss=9.0,
            correlation=0.9,
        ),
        cross_prediction(
            variant="forced_independent_targets",
            sample_id="i5",
            intensity=5,
            loss=10.0,
            correlation=0.7,
        ),
    ]

    paired = response.paired_variant_audits(predictions)
    cross = paired[0]
    assert cross["native_relative_mean_loss_gain"] == pytest.approx(
        (6.0 - 5.0) / 6.0
    )
    intensity_rows = response.cross_series_dependence_audits(predictions)
    by_intensity = {row["intensity"]: row for row in intensity_rows}
    assert by_intensity[1]["native_relative_mean_loss_gain"] == pytest.approx(0.5)
    assert by_intensity[5]["native_relative_mean_loss_gain"] == pytest.approx(0.1)


def test_cross_series_counterfactual_audit_detects_driver_response() -> None:
    response = load_response_module()
    history = np.column_stack(
        [
            np.arange(6, dtype=float),
            np.sin(np.arange(6, dtype=float)),
            np.cos(np.arange(6, dtype=float)),
        ]
    )
    first_target = np.vstack(
        [history, [[0.0, -1.0, 1.0], [0.0, -2.0, 2.0]]]
    )
    second_target = np.vstack(
        [history, [[0.0, 1.0, -1.0], [0.0, 2.0, -2.0]]]
    )
    samples = []
    for member, target in enumerate((first_target, second_target)):
        samples.append(
            {
                "sample_id": f"member-{member}",
                "evaluation_table": "main",
                "capability_id": "cross_series_dependence",
                "generator_family_role": "primary",
                "intensity": 5,
                "context_length": 6,
                "target": target.tolist(),
                "counterfactual_pair_id": "pair-0",
                "counterfactual_member": member,
                "generation_metadata": {"responder_indices": [1, 2]},
            }
        )
    predictions = []
    for member, target in enumerate((first_target, second_target)):
        truth = target[6:]
        predictions.extend(
            [
                {
                    "sample_id": f"member-{member}",
                    "model_id": "joint",
                    "variant": "native",
                    "forecast": truth.tolist(),
                    "metrics": {"responder_mae": 0.0},
                },
                {
                    "sample_id": f"member-{member}",
                    "model_id": "joint",
                    "variant": "forced_independent_targets",
                    "forecast": np.zeros_like(truth).tolist(),
                    "metrics": {"responder_mae": 1.5},
                },
            ]
        )

    rows = response.cross_series_counterfactual_audits(
        predictions,
        samples,
    )
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["native"]["mean_effect_nrmse"] == pytest.approx(0.0)
    assert by_variant["native"]["mean_effect_correlation"] == pytest.approx(1.0)
    assert by_variant["native"]["mean_effect_amplitude_ratio"] == pytest.approx(1.0)
    assert by_variant["native"]["mean_effect_signed_projection"] == pytest.approx(1.0)
    assert by_variant["native"][
        "mean_responder_history_max_abs_difference"
    ] == pytest.approx(0.0)
    assert by_variant["forced_independent_targets"][
        "mean_effect_amplitude_ratio"
    ] == pytest.approx(0.0)


def test_common_factor_counterfactual_audit_requires_joint_response() -> None:
    response = load_response_module()
    generated = []
    for member in (0, 1):
        target, metadata, _ = (
            response.v8_pilot.generate_deterministic_sample(
                "common_factor",
                552,
                504,
                3,
                24,
                5,
                np.random.default_rng(91),
                counterfactual_variant=member,
            )
        )
        target, _ = (
            response.v8_pilot.standardize_common_factor_counterfactual_member(
                target,
                context_length=504,
                metadata=metadata,
            )
        )
        generated.append((target, metadata))

    samples = []
    predictions = []
    for member, (target, metadata) in enumerate(generated):
        sample_id = f"common-member-{member}"
        sample = {
            "sample_id": sample_id,
            "evaluation_table": "main",
            "capability_id": "common_factor",
            "generator_family_role": "primary",
            "intensity": 5,
            "context_length": 504,
            "horizon": 48,
            "season_length": 24,
            "target": target.tolist(),
            "counterfactual_pair_id": "common-pair-0",
            "counterfactual_member": member,
            "generation_metadata": metadata,
        }
        samples.append(sample)
        probe = response.baseline_forecast(
            sample,
            "common_factor_joint_probe",
        )
        protected = int(metadata["protected_target_index"])
        split = np.repeat(
            target[503:504],
            48,
            axis=0,
        )
        predictions.extend(
            [
                {
                    "sample_id": sample_id,
                    "model_id": "joint-probe",
                    "variant": "native",
                    "forecast": probe.tolist(),
                    "metrics": {
                        "protected_target_mae": float(
                            np.mean(
                                np.abs(
                                    probe[:, protected]
                                    - target[504:, protected]
                                )
                            )
                        )
                    },
                },
                {
                    "sample_id": sample_id,
                    "model_id": "joint-probe",
                    "variant": "forced_independent_targets",
                    "forecast": split.tolist(),
                    "metrics": {
                        "protected_target_mae": float(
                            np.mean(
                                np.abs(
                                    split[:, protected]
                                    - target[504:, protected]
                                )
                            )
                        )
                    },
                },
            ]
        )

    rows = response.common_factor_counterfactual_audits(
        predictions,
        samples,
    )
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["native"]["mean_effect_nrmse"] < 0.15
    assert by_variant["native"]["mean_effect_correlation"] > 0.95
    assert by_variant["native"]["mean_effect_amplitude_ratio"] == pytest.approx(
        1.0,
        abs=0.15,
    )
    assert by_variant["native"][
        "mean_protected_history_max_abs_difference"
    ] == pytest.approx(0.0)
    assert by_variant["forced_independent_targets"][
        "mean_effect_amplitude_ratio"
    ] == pytest.approx(0.0)


def test_covariate_effect_audit_recovers_known_future_response() -> None:
    response = load_response_module()
    context = 4
    weather = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, -0.5, -1.0])
    event = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    covariates = np.column_stack([weather, event])
    weights = {"weather": [2.0], "event": [-0.5]}
    target = (
        weather[:, None] * np.asarray(weights["weather"])[None, :]
        + event[:, None] * np.asarray(weights["event"])[None, :]
    )
    sample = {
        "sample_id": "covariate-sample",
        "evaluation_table": "main",
        "capability_id": "covariate_response",
        "generator_family_role": "primary",
        "intensity": 5,
        "context_length": context,
        "horizon": 4,
        "covariates": covariates.tolist(),
        "generation_metadata": {
            "response_law": "instantaneous_linear",
            "weather_effect_by_target": weights["weather"],
            "event_effect_by_target": weights["event"],
        },
    }
    truth_effect = target[context:]
    predictions = [
        {
            "sample_id": sample["sample_id"],
            "model_id": "joint",
            "variant": "native",
            "forecast": truth_effect.tolist(),
            "input_adaptation": {"covariate_mode": "native"},
        },
        {
            "sample_id": sample["sample_id"],
            "model_id": "joint",
            "variant": "covariates_ablated",
            "forecast": np.zeros_like(truth_effect).tolist(),
            "input_adaptation": {"covariate_mode": "none"},
        },
    ]

    rows = response.covariate_effect_audits(predictions, [sample])

    assert len(rows) == 1
    assert rows[0]["mean_effect_nrmse"] == pytest.approx(0.0)
    assert rows[0]["mean_effect_correlation"] == pytest.approx(1.0)
    assert rows[0]["mean_effect_amplitude_ratio"] == pytest.approx(1.0)
    assert rows[0]["mean_effect_signed_projection"] == pytest.approx(1.0)
    assert rows[0]["native_covariate_modes"] == ["native"]


def test_covariate_counterfactual_audit_requires_future_covariate_response():
    response = load_response_module()
    context = 4
    samples = []
    predictions = []
    for member, future_weather in enumerate(
        (
            np.asarray([1.0, 0.5, -0.5, -1.0]),
            np.asarray([-1.0, -0.5, 0.5, 1.0]),
        )
    ):
        weather = np.concatenate([np.zeros(context), future_weather])
        event = np.zeros_like(weather)
        target = (2.0 * weather)[:, None]
        sample_id = f"covariate-member-{member}"
        samples.append(
            {
                "sample_id": sample_id,
                "evaluation_table": "main",
                "capability_id": "covariate_response",
                "generator_family_role": "primary",
                "intensity": 5,
                "context_length": context,
                "horizon": 4,
                "target": target.tolist(),
                "covariates": np.column_stack([weather, event]).tolist(),
                "counterfactual_pair_id": "covariate-pair-0",
                "counterfactual_member": member,
            }
        )
        predictions.extend(
            [
                {
                    "sample_id": sample_id,
                    "model_id": "joint",
                    "variant": "native",
                    "forecast": target[context:].tolist(),
                    "input_adaptation": {"covariate_mode": "native"},
                },
                {
                    "sample_id": sample_id,
                    "model_id": "joint",
                    "variant": "covariates_ablated",
                    "forecast": np.zeros((4, 1)).tolist(),
                    "input_adaptation": {"covariate_mode": "none"},
                },
            ]
        )

    rows = response.covariate_counterfactual_audits(
        predictions,
        samples,
    )
    by_variant = {row["variant"]: row for row in rows}

    assert by_variant["native"]["mean_effect_nrmse"] == pytest.approx(0.0)
    assert by_variant["native"]["mean_effect_correlation"] == pytest.approx(1.0)
    assert by_variant["native"]["mean_effect_amplitude_ratio"] == pytest.approx(
        1.0
    )
    assert by_variant["native"]["mean_effect_signed_projection"] == pytest.approx(
        1.0
    )
    assert by_variant["native"][
        "mean_target_history_max_abs_difference"
    ] == pytest.approx(0.0)
    assert by_variant["native"][
        "mean_past_covariate_max_abs_difference"
    ] == pytest.approx(0.0)
    assert by_variant["covariates_ablated"][
        "mean_effect_amplitude_ratio"
    ] == pytest.approx(0.0)
