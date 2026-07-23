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
    factor_sample = sample_for("common_factor", factor, context_length=24)

    factor_metrics = response.prediction_metrics(
        factor_sample,
        factor[24:],
    )

    assert factor_metrics["factor_loading_cosine"] == pytest.approx(1.0)
    assert factor_metrics["factor_trajectory_correlation"] == pytest.approx(1.0)
    assert factor_metrics["factor_score_nrmse"] == pytest.approx(0.0)
    assert factor_metrics["common_component_nmae"] == pytest.approx(0.0)

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
