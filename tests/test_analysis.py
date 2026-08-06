from __future__ import annotations

import numpy as np

from cafe.analysis import diagnostics, structured


def load_script(name: str):
    assert name == "cafe.structured_baselines"
    return structured


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


def test_intermittent_recovery_metrics_handle_pulse_width() -> None:
    context = 96
    horizon = 12
    target = np.zeros((context + horizon, 1), dtype=float)
    target[context + 5, 0] = 2.0
    forecast = target[context:].copy()
    task = {
        **sample(
            "predictable_intermittency",
            target,
            context=context,
            horizon=horizon,
        ),
        "generation_metadata": {
            "pulse_centers": [context + 5],
            "pulse_width": 1.5,
        },
    }

    result = diagnostics.intermittent_recovery_metrics(
        task,
        target,
        forecast,
    )

    assert result["event_peak_timing_widths"] == 0.0
    assert result["event_window_nmae"] == 0.0


def test_ridge_var_recovers_history_covered_lag_better_than_diagonal_ar():
    baselines = load_script("cafe.structured_baselines")
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
    baselines = load_script("cafe.structured_baselines")
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
    baselines = load_script("cafe.structured_baselines")
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
    baselines = load_script("cafe.structured_baselines")
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


def test_shared_pair_ardl_recovers_history_propagated_full_horizon_effect():
    baselines = load_script("cafe.structured_baselines")
    rng = np.random.default_rng(17)
    context = 128
    horizon = 12
    lag = 8
    driver_persistence = 0.72
    innovations = rng.normal(size=context + horizon)
    base_driver = np.empty(context + horizon)
    base_driver[0] = innovations[0]
    for index in range(1, len(base_driver)):
        base_driver[index] = (
            driver_persistence * base_driver[index - 1]
            + innovations[index]
        )
    drivers = []
    for sign in (-1.0, 1.0):
        driver = base_driver.copy()
        state = 0.0
        for index in range(context - lag, context + horizon):
            shock = sign * 0.4 if index < context else 0.0
            state = driver_persistence * state + shock
            driver[index] += state
        drivers.append(driver)
    targets = []
    for driver in drivers:
        responses = np.zeros((context + horizon, 2))
        for index in range(1, context + horizon):
            source = driver[max(0, index - lag)]
            responses[index, 0] = 0.20 * responses[index - 1, 0] + source
            responses[index, 1] = (
                0.10 * responses[index - 1, 1] - 0.8 * source
            )
        targets.append(np.column_stack([driver, responses]))
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
    assert np.sqrt(np.mean(truth_effect[-4:] ** 2)) > 0.01
    assert np.corrcoef(
        truth_effect.ravel(),
        forecast_effect.ravel(),
    )[0, 1] > 0.95
    assert (
        np.sqrt(np.mean((forecast_effect - truth_effect) ** 2))
        / np.sqrt(np.mean(truth_effect**2))
    ) < 0.25


def test_common_shared_pair_dfm_is_blind_and_propagates_latent_state():
    baselines = load_script("cafe.structured_baselines")
    rng = np.random.default_rng(23)
    context = 128
    horizon = 12
    invariant_stop = 112
    factor = np.empty(context + horizon)
    factor[0] = rng.normal()
    for index in range(1, len(factor)):
        factor[index] = 0.92 * factor[index - 1] + rng.normal(scale=0.08)
    loadings = np.asarray([0.8, -1.0, 0.7, -0.6, 1.2])
    base = factor[:, None] * loadings[None, :]
    base += rng.normal(scale=0.02, size=base.shape)

    targets = []
    future_effects = []
    for sign in (-1.0, 1.0):
        target = base.copy()
        latent_effect = np.zeros(context + horizon)
        latent_effect[invariant_stop] = sign
        for index in range(invariant_stop + 1, len(latent_effect)):
            latent_effect[index] = 0.92 * latent_effect[index - 1]
        target[invariant_stop:context, 1:] += (
            latent_effect[invariant_stop:context, None]
            * loadings[None, 1:]
        )
        target[context:] += (
            latent_effect[context:, None] * loadings[None, :]
        )
        targets.append(target)
        future_effects.append(latent_effect[context:])
    samples = [
        {
            **sample(
                "common_factor",
                target,
                context=context,
                horizon=horizon,
            ),
            "evaluation_table": "strict_counterfactual_audit",
            "generation_metadata": {"hidden_codebook": "must_not_be_used"},
        }
        for target in targets
    ]

    first, second = baselines.forecast_common_counterfactual_pair(*samples)
    forecast_effect = second.forecast[:, 0] - first.forecast[:, 0]

    assert not first.diagnostics["fallback_used"]
    assert first.diagnostics["pair_invariant_history_stop"] == invariant_stop
    assert first.diagnostics["pair_invariant_channel_indices"] == [0]
    assert first.diagnostics["generator_metadata_used_for_fitting"] is False
    assert first.diagnostics["counterfactual_residual_forecast_shared"] is True
    assert np.sqrt(np.mean(forecast_effect**2)) > 0.01
    assert np.corrcoef(
        future_effects[1] - future_effects[0],
        forecast_effect,
    )[0, 1] > 0.8
