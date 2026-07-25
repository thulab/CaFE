from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).parents[3] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample(
    capability_id: str,
    target: np.ndarray,
    *,
    context: int,
    horizon: int,
) -> dict:
    return {
        "capability_id": capability_id,
        "generator_family_role": "primary",
        "intensity": 5,
        "evaluation_table": "main",
        "context_length": context,
        "horizon": horizon,
        "target": target.tolist(),
    }


def test_ridge_var_recovers_history_covered_lag_better_than_diagonal_ar():
    baselines = load_script("paper_v8_structured_baselines")
    rng = np.random.default_rng(7)
    context = 128
    horizon = 8
    driver = rng.normal(size=context + horizon)
    delayed = np.concatenate([np.zeros(horizon), driver[:-horizon]])
    target = np.column_stack([driver, delayed, -0.8 * delayed])
    task = sample(
        "cross_series_dependence",
        target,
        context=context,
        horizon=horizon,
    )

    diagonal = baselines.forecast(task, "diagonal_ar")
    vector = baselines.forecast(task, "ridge_var")
    truth = target[context:, 1:]
    diagonal_mae = float(np.mean(np.abs(diagonal.forecast[:, 1:] - truth)))
    vector_mae = float(np.mean(np.abs(vector.forecast[:, 1:] - truth)))

    assert vector.diagnostics["selected_lag"] == horizon
    assert not vector.diagnostics["fallback_used"]
    assert vector_mae < 0.05 * diagonal_mae


def test_structured_forecast_is_history_only_and_deterministic():
    baselines = load_script("paper_v8_structured_baselines")
    rng = np.random.default_rng(11)
    context = 96
    horizon = 12
    history = rng.normal(size=(context, 3))
    first = np.vstack([history, np.zeros((horizon, 3))])
    second = np.vstack([history, np.full((horizon, 3), 999.0)])
    first_task = sample(
        "cross_series_dependence",
        first,
        context=context,
        horizon=horizon,
    )
    second_task = sample(
        "cross_series_dependence",
        second,
        context=context,
        horizon=horizon,
    )

    first_result = baselines.forecast(first_task, "ridge_var")
    repeated = baselines.forecast(first_task, "ridge_var")
    changed_future = baselines.forecast(second_task, "ridge_var")

    np.testing.assert_allclose(first_result.forecast, repeated.forecast)
    np.testing.assert_allclose(first_result.forecast, changed_future.forecast)
    assert first_result.diagnostics["history_only"] is True


def test_rank1_dfm_reports_factor_fit_and_reconstructs_shared_signal():
    baselines = load_script("paper_v8_structured_baselines")
    rng = np.random.default_rng(13)
    context = 128
    horizon = 8
    time = np.arange(context + horizon, dtype=float)
    factor = np.sin(0.2 * time) + 0.5 * np.sin(0.07 * time)
    loadings = np.asarray([1.0, -1.0, 0.7, -0.5, 1.2])
    target = factor[:, None] * loadings[None, :]
    target += 0.02 * rng.normal(size=target.shape)
    task = sample(
        "common_factor",
        target,
        context=context,
        horizon=horizon,
    )

    diagonal = baselines.forecast(task, "diagonal_ar")
    dfm = baselines.forecast(task, "dynamic_factor_var")
    truth = target[context:]

    assert dfm.diagnostics["selected_factor_rank"] in {1, 2}
    assert dfm.diagnostics["history_factor_share"] > 0.95
    assert not dfm.diagnostics["fallback_used"]
    assert np.mean(np.abs(dfm.forecast - truth)) < np.mean(
        np.abs(diagonal.forecast - truth)
    )


def test_scope_is_primary_i5_and_rejects_other_samples():
    baselines = load_script("paper_v8_structured_baselines")
    target = np.arange(108, dtype=float)[:, None]
    task = sample(
        "common_factor",
        target,
        context=96,
        horizon=12,
    )

    assert baselines.is_structured_sample(task)
    assert not baselines.is_structured_sample({**task, "intensity": 4})
    assert not baselines.is_structured_sample(
        {**task, "generator_family_role": "secondary"}
    )


def test_shared_pair_ardl_recovers_active_effect_without_tail_leakage():
    baselines = load_script("paper_v8_structured_baselines")
    rng = np.random.default_rng(17)
    context = 128
    horizon = 12
    lag = 8
    base_driver = rng.normal(size=context + horizon)
    drivers = []
    for sign in (-1.0, 1.0):
        driver = base_driver.copy()
        driver[context - lag : context] += sign * np.linspace(
            -1.0,
            1.0,
            lag,
        )
        driver[context:] = base_driver[context:]
        drivers.append(driver)
    targets = []
    for driver in drivers:
        shifted = np.concatenate([np.zeros(lag), driver[:-lag]])
        targets.append(np.column_stack([driver, shifted, -0.8 * shifted]))
    samples = [
        {
            **sample(
                "cross_series_dependence",
                target,
                context=context,
                horizon=horizon,
            ),
            "evaluation_table": "strict_counterfactual_audit",
        }
        for target in targets
    ]

    first, second = baselines.forecast_cross_counterfactual_pair(*samples)
    truth_effect = targets[1][context:, 1:] - targets[0][context:, 1:]
    forecast_effect = second.forecast[:, 1:] - first.forecast[:, 1:]

    assert first.diagnostics["selected_source_channel"] == 0
    assert first.diagnostics["selected_lag"] == lag
    np.testing.assert_allclose(forecast_effect[lag:], 0.0, atol=1e-10)
    assert np.corrcoef(
        truth_effect[:lag].ravel(),
        forecast_effect[:lag].ravel(),
    )[0, 1] > 0.99
