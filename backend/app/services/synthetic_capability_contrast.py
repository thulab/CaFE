from __future__ import annotations

from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "synthetic_capability_contrast.v1"
NUMERIC_EPSILON = 1e-9
NONLINEAR_RECURSIVE_SHRINKAGE = 0.5


def capability_contrast_forecasts(
    *,
    capability_id: str,
    history: np.ndarray,
    horizon: int,
    season_length: int,
    latent_params: dict[str, Any],
    covariates: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build blind and capability-aware forecasts without reading future targets.

    Most aware forecasts estimate their mechanism from history.  Trend uses
    frozen polynomial coefficients, while regime and intermittency use only
    historical event annotations from construction metadata.  No branch
    receives future target values; known-future covariates are allowed only
    for ``covariate_response``.
    """

    values = _as_target_matrix(history)
    forecast_horizon = int(horizon)
    if forecast_horizon <= 0:
        raise ValueError("horizon must be positive")
    season = max(4, int(season_length))
    blind, blind_method = _generic_blind_forecast(
        values,
        forecast_horizon,
        season,
    )
    aware_method = capability_id
    aware_information_set = "history_estimated_mechanism"
    if capability_id == "trend":
        slope = np.asarray(
            latent_params.get("slope_by_target", ()),
            dtype=float,
        )
        curvature = np.asarray(
            latent_params.get("curvature_by_target", ()),
            dtype=float,
        )
        if slope.shape == (values.shape[1],) and curvature.shape == (
            values.shape[1],
        ):
            time = (
                np.arange(len(values) + forecast_horizon, dtype=float)
                - max(0, len(values) - 1)
            ) / season
            component = (
                time[:, None] * slope[None, :]
                + time[:, None] ** 2 * curvature[None, :]
            )
            residual = values - component[: len(values)]
            aware = (
                component[len(values) :]
                + _short_ar_forecast(residual, forecast_horizon)
            )
            aware_method = "construction_polynomial_continuation"
            aware_information_set = "history_plus_frozen_construction_metadata"
        else:
            design = _polynomial_design(
                len(values) + forecast_horizon,
                len(values),
                1,
            )
            aware = _design_plus_residual_forecast(
                values,
                design,
                forecast_horizon,
            )
            aware_method = "estimated_linear_continuation"
    elif capability_id == "multi_seasonal":
        periods = _estimate_harmonic_periods_from_history(
            values,
            fallback_period=season,
            component_count=3,
        )
        if not periods:
            periods = [float(season)]
        design = _harmonic_design(
            len(values) + forecast_horizon,
            len(values),
            periods,
            include_trend=True,
        )
        aware = _design_plus_residual_forecast(values, design, forecast_horizon)
        aware_method = "history_spectral_multi_harmonic_continuation"
    elif capability_id == "time_varying_seasonality":
        modulation_period = _estimate_modulation_period_from_history(
            values,
            carrier_period=season,
        )
        periods = _time_varying_sideband_periods(
            carrier_period=season,
            modulation_period=modulation_period,
        )
        design = _harmonic_design(
            len(values) + forecast_horizon,
            len(values),
            periods,
            include_trend=True,
        )
        aware = _design_plus_residual_forecast(values, design, forecast_horizon)
        aware_method = "history_estimated_sideband_continuation"
    elif capability_id == "regime_switching":
        aware_information_set = "history_plus_historical_event_annotations"
        cut_points = _extrapolated_regime_cut_points(
            len(values),
            forecast_horizon,
            latent_params.get("cut_points", ()),
        )
        state = _alternating_state(
            len(values) + forecast_horizon,
            cut_points,
        )
        design = np.column_stack(
            [
                _polynomial_design(
                    len(values) + forecast_horizon,
                    len(values),
                    1,
                ),
                state,
            ]
        )
        aware = _design_plus_residual_forecast(values, design, forecast_horizon)
        aware_method = "history_inferred_explicit_duration_clock"
    elif capability_id == "nonlinear_persistence":
        nonlinear_forecast = _nonlinear_forecast(
            values,
            forecast_horizon,
            season,
            latent_params,
        )
        # Recursive nonlinear forecasts have higher multi-step variance than
        # the matched generic forecast.  Apply one frozen, profile-independent
        # shrinkage coefficient to the capability-specific correction.  This
        # keeps the aware information set nested over the blind prediction and
        # is never tuned from the scored future.
        aware = blind + NONLINEAR_RECURSIVE_SHRINKAGE * (
            nonlinear_forecast - blind
        )
        aware_method = "shrunken_nonlinear_multi_lag_recurrence"
    elif capability_id == "predictable_intermittency":
        aware_information_set = "history_plus_historical_event_annotations"
        pulse_width = float(latent_params.get("pulse_width", 1.0))
        pulse_centers = _extrapolated_pulse_centers(
            len(values),
            forecast_horizon,
            latent_params.get("pulse_centers", ()),
            future_padding=int(max(1, np.ceil(4 * pulse_width))),
        )
        pulse = _pulse_component(
            len(values) + forecast_horizon,
            pulse_centers,
            pulse_width,
        )
        design = np.column_stack(
            [
                _polynomial_design(
                    len(values) + forecast_horizon,
                    len(values),
                    1,
                ),
                pulse,
            ]
        )
        aware = _design_plus_residual_forecast(values, design, forecast_horizon)
        aware_method = "history_estimated_nonuniform_pulse_clock"
    elif capability_id == "common_factor":
        aware = _common_factor_forecast(values, forecast_horizon)
        aware_method = "history_estimated_rank1_dynamic_factor"
    elif capability_id == "hierarchical_coherence":
        aware = _hierarchical_history_forecast(
            values,
            forecast_horizon,
            season,
        )
        aware_method = "history_estimated_aggregate_plus_local_contrasts"
    elif capability_id == "covariate_response":
        if covariates is None:
            raise ValueError("covariate contrast requires known-future covariates")
        known = np.asarray(covariates, dtype=float)
        required_length = len(values) + forecast_horizon
        if known.ndim != 2 or len(known) < required_length:
            raise ValueError("known-future covariates do not cover the forecast horizon")
        design = np.column_stack(
            [
                _polynomial_design(required_length, len(values), 1),
                known[:required_length],
            ]
        )
        aware = _design_plus_residual_forecast(
            values,
            design,
            forecast_horizon,
        )
        aware_method = "history_estimated_known_future_covariate_regression"
    else:
        raise ValueError(f"unknown synthetic capability: {capability_id}")

    return {
        "blind": blind,
        "aware": aware,
        "blind_method": blind_method,
        "aware_method": aware_method,
        "future_target_used_for_forecast": False,
        "aware_information_set": (
            aware_information_set
            + (
                "_plus_known_future_covariates"
                if capability_id == "covariate_response"
                else ""
            )
        ),
        "known_future_covariates_used": capability_id == "covariate_response",
    }


def evaluate_capability_contrast(
    *,
    capability_id: str,
    target: np.ndarray,
    context_length: int,
    season_length: int,
    intensity: int,
    latent_params: dict[str, Any],
    covariates: np.ndarray | None = None,
    evaluation_scale: str = "caller_input",
) -> dict[str, Any]:
    values = _as_target_matrix(target)
    context = int(context_length)
    if not 0 < context < len(values):
        raise ValueError("context_length must split history and future")
    forecasts = capability_contrast_forecasts(
        capability_id=capability_id,
        history=values[:context],
        horizon=len(values) - context,
        season_length=season_length,
        latent_params=latent_params,
        covariates=covariates,
    )
    actual = values[context:]
    blind = np.asarray(forecasts["blind"], dtype=float)
    aware = np.asarray(forecasts["aware"], dtype=float)
    blind_mae = float(np.mean(np.abs(actual - blind)))
    aware_mae = float(np.mean(np.abs(actual - aware)))
    blind_structure = _structure_error(capability_id, blind)
    aware_structure = _structure_error(capability_id, aware)
    structure_weight = 1.0 if capability_id == "hierarchical_coherence" else 0.0
    blind_loss = blind_mae + structure_weight * blind_structure
    aware_loss = aware_mae + structure_weight * aware_structure
    relative_gain = float(
        (blind_loss - aware_loss) / max(blind_loss, NUMERIC_EPSILON)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "capability_id": capability_id,
        "intensity": int(intensity),
        "protocol": "construction_aware_vs_capability_blind",
        "evaluation_scale": str(evaluation_scale),
        "selection_role": "diagnostic_only",
        "enforced_online": False,
        "future_target_used_for_forecast": False,
        "aware_information_set": forecasts["aware_information_set"],
        "known_future_covariates_used": bool(
            forecasts["known_future_covariates_used"]
        ),
        "blind_method": forecasts["blind_method"],
        "aware_method": forecasts["aware_method"],
        "blind_mae": blind_mae,
        "aware_mae": aware_mae,
        "blind_structure_error": blind_structure,
        "aware_structure_error": aware_structure,
        "blind_composite_loss": blind_loss,
        "aware_composite_loss": aware_loss,
        "relative_loss_gain": relative_gain,
        "aware_wins": bool(aware_loss < blind_loss),
    }


def summarize_capability_contrasts(
    results: Iterable[dict[str, Any]],
    *,
    minimum_sample_count: int = 24,
    minimum_win_rate: float = 0.60,
    minimum_mean_relative_gain: float = 0.02,
) -> dict[str, Any]:
    rows = list(results)
    if not rows:
        raise ValueError("at least one capability contrast result is required")
    capability_ids = {str(row["capability_id"]) for row in rows}
    intensities = {int(row["intensity"]) for row in rows}
    if len(capability_ids) != 1 or len(intensities) != 1:
        raise ValueError("a contrast summary must contain one capability and intensity")
    blind_losses = np.asarray(
        [float(row["blind_composite_loss"]) for row in rows],
        dtype=float,
    )
    aware_losses = np.asarray(
        [float(row["aware_composite_loss"]) for row in rows],
        dtype=float,
    )
    paired_improvements = blind_losses - aware_losses
    wins = np.asarray([bool(row["aware_wins"]) for row in rows], dtype=float)
    blind_loss_mean = float(np.mean(blind_losses))
    aware_loss_mean = float(np.mean(aware_losses))
    mean_gain = float(
        np.mean(paired_improvements)
        / max(blind_loss_mean, NUMERIC_EPSILON)
    )
    gain_se = float(
        np.std(paired_improvements, ddof=1)
        / np.sqrt(len(paired_improvements))
        / max(blind_loss_mean, NUMERIC_EPSILON)
        if len(paired_improvements) > 1
        else 0.0
    )
    gain_lcb = float(mean_gain - 1.645 * gain_se)
    win_rate = float(np.mean(wins))
    enough_samples = len(rows) >= int(minimum_sample_count)
    passed = bool(
        enough_samples
        and win_rate >= float(minimum_win_rate)
        and gain_lcb >= float(minimum_mean_relative_gain)
    )
    return {
        "schema_version": "synthetic_capability_contrast_summary.v1",
        "capability_id": next(iter(capability_ids)),
        "intensity": next(iter(intensities)),
        "sample_count": len(rows),
        "aggregation": "paired_difference_of_mean_composite_loss",
        "blind_composite_loss_mean": blind_loss_mean,
        "aware_composite_loss_mean": aware_loss_mean,
        "mean_relative_loss_gain": mean_gain,
        "relative_loss_gain_lcb_95_one_sided": gain_lcb,
        "aware_win_rate": win_rate,
        "criteria": {
            "minimum_sample_count": int(minimum_sample_count),
            "minimum_win_rate": float(minimum_win_rate),
            "minimum_mean_relative_gain": float(minimum_mean_relative_gain),
        },
        "passed": passed,
    }


def _as_target_matrix(values: np.ndarray) -> np.ndarray:
    target = np.asarray(values, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or not np.isfinite(target).all():
        raise ValueError("target must be a finite one- or two-dimensional array")
    return target


def _generic_blind_forecast(
    history: np.ndarray,
    horizon: int,
    season_length: int,
) -> tuple[np.ndarray, str]:
    values = _as_target_matrix(history)
    validation_horizon = min(
        int(horizon),
        max(6, min(int(season_length), len(values) // 4)),
    )
    train = values[:-validation_horizon]
    actual = values[-validation_horizon:]
    candidates = {
        "last": lambda frame, steps: np.repeat(frame[-1:], steps, axis=0),
        "seasonal_naive": lambda frame, steps: _seasonal_naive(
            frame,
            steps,
            season_length,
        ),
        "linear_drift": _linear_drift,
        "channelwise_linear_ar": lambda frame, steps: _linear_ar_forecast(
            frame,
            steps,
            season_length,
        ),
    }
    scores: dict[str, float] = {}
    for name, forecaster in candidates.items():
        try:
            prediction = forecaster(train, validation_horizon)
            scores[name] = float(np.mean(np.abs(actual - prediction)))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            scores[name] = float("inf")
    selected = min(scores, key=lambda name: (scores[name], name))
    return candidates[selected](values, int(horizon)), selected


def _seasonal_naive(
    history: np.ndarray,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    period = min(max(1, int(season_length)), len(history))
    pattern = history[-period:]
    return np.vstack([pattern[index % period] for index in range(horizon)])


def _linear_drift(history: np.ndarray, horizon: int) -> np.ndarray:
    values = _as_target_matrix(history)
    fit_length = min(len(values), max(24, len(values) // 2))
    train = values[-fit_length:]
    time = np.arange(fit_length, dtype=float)
    design = np.column_stack([np.ones(fit_length), time])
    coefficient = _ridge_coefficients(design, train)
    future_time = np.arange(fit_length, fit_length + horizon, dtype=float)
    return np.column_stack([np.ones(horizon), future_time]) @ coefficient


def _linear_ar_forecast(
    history: np.ndarray,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    values = _as_target_matrix(history)
    lags = tuple(
        sorted(
            {
                1,
                2,
                max(3, int(season_length) // 2),
                max(4, int(season_length)),
            }
        )
    )
    if len(values) <= max(lags) + 8:
        return np.repeat(values[-1:], horizon, axis=0)
    output = np.array(values, copy=True)
    coefficients: list[np.ndarray] = []
    start = max(lags)
    design = np.column_stack(
        [
            np.ones(len(values) - start),
            *[
                values[start - lag : len(values) - lag]
                for lag in lags
            ],
        ]
    )
    for channel in range(values.shape[1]):
        channel_columns = np.column_stack(
            [
                design[:, 0],
                *[
                    values[
                        start - lag : len(values) - lag,
                        channel,
                    ]
                    for lag in lags
                ],
            ]
        )
        coefficients.append(
            _ridge_coefficients(
                channel_columns,
                values[start:, channel],
            ).reshape(-1)
        )
    predictions: list[np.ndarray] = []
    for _ in range(horizon):
        row = []
        for channel, coefficient in enumerate(coefficients):
            features = np.asarray(
                [1.0, *[output[-lag, channel] for lag in lags]],
                dtype=float,
            )
            row.append(float(features @ coefficient))
        next_value = np.asarray(row, dtype=float)
        predictions.append(next_value)
        output = np.vstack([output, next_value])
    return np.asarray(predictions, dtype=float)


def _design_plus_residual_forecast(
    history: np.ndarray,
    design: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = _as_target_matrix(history)
    if len(design) < len(values) + horizon:
        raise ValueError("design does not cover the forecast horizon")
    coefficient = _ridge_coefficients(design[: len(values)], values)
    fitted = design[: len(values)] @ coefficient
    base_future = design[len(values) : len(values) + horizon] @ coefficient
    residual = values - fitted
    if len(residual) < 12:
        return base_future
    residual_future = _short_ar_forecast(residual, horizon)
    return base_future + residual_future


def _short_ar_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    values = _as_target_matrix(history)
    if len(values) < 8:
        return np.zeros((horizon, values.shape[1]))
    design = np.column_stack(
        [
            np.ones(len(values) - 2),
            values[1:-1],
            values[:-2],
        ]
    )
    output = np.array(values, copy=True)
    coefficients = [
        _ridge_coefficients(
            np.column_stack(
                [
                    design[:, 0],
                    values[1:-1, channel],
                    values[:-2, channel],
                ]
            ),
            values[2:, channel],
        ).reshape(-1)
        for channel in range(values.shape[1])
    ]
    predictions: list[np.ndarray] = []
    for _ in range(horizon):
        next_value = np.asarray(
            [
                float(
                    np.asarray(
                        [1.0, output[-1, channel], output[-2, channel]]
                    )
                    @ coefficient
                )
                for channel, coefficient in enumerate(coefficients)
            ]
        )
        predictions.append(next_value)
        output = np.vstack([output, next_value])
    return np.asarray(predictions)


def _polynomial_design(length: int, origin: int, degree: int) -> np.ndarray:
    time = (np.arange(length, dtype=float) - max(0, origin - 1)) / max(
        1,
        origin,
    )
    return np.column_stack(
        [np.ones(length), *[time**power for power in range(1, degree + 1)]]
    )


def _estimate_harmonic_periods_from_history(
    history: np.ndarray,
    *,
    fallback_period: int,
    component_count: int,
) -> list[float]:
    """Estimate separated integer harmonic periods without latent periods."""
    values = _as_target_matrix(history)
    sample_count = len(values)
    maximum_period = sample_count // 2
    if sample_count < 16 or maximum_period < 4 or component_count <= 0:
        return [float(max(4, int(fallback_period)))]

    time = np.arange(sample_count, dtype=float)
    trend_design = np.column_stack([np.ones(sample_count), time])
    residual = values - trend_design @ _ridge_coefficients(
        trend_design,
        values,
    )
    candidate_periods = range(4, maximum_period + 1)
    minimum_frequency_gap = 1.0 / sample_count
    selected: list[float] = []
    for _ in range(int(component_count)):
        best: tuple[float, int, np.ndarray] | None = None
        for period in candidate_periods:
            frequency = 1.0 / period
            if any(
                abs(frequency - 1.0 / existing) < minimum_frequency_gap
                for existing in selected
            ):
                continue
            phase = 2 * np.pi * time / period
            basis = np.column_stack([np.sin(phase), np.cos(phase)])
            fitted = basis @ _ridge_coefficients(basis, residual)
            score = float(np.mean(fitted**2))
            candidate = (score, int(period), fitted)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or best[0] <= NUMERIC_EPSILON:
            break
        _, period, fitted = best
        selected.append(float(period))
        residual = residual - fitted

    if selected:
        return selected
    fallback = min(max(4, int(fallback_period)), maximum_period)
    return [float(fallback)]


def _time_varying_modulation_period_candidates(
    history_length: int,
    carrier_period: int,
) -> list[float]:
    primary_period = max(4, int(carrier_period))
    return [
        float(period)
        for period in sorted(
            {
                max(primary_period + 1, int(round(primary_period * ratio)))
                for ratio in (2.0, 2.5, 3.0, 3.5)
                if 2 * int(round(primary_period * ratio)) <= history_length
            }
        )
    ]


def _time_varying_sideband_periods(
    *,
    carrier_period: int,
    modulation_period: float,
) -> list[float]:
    carrier_frequency = 1.0 / max(4, int(carrier_period))
    modulation_frequency = 1.0 / max(float(modulation_period), 2.0)
    frequencies = {
        carrier_frequency,
        0.5 * carrier_frequency,
    }
    for harmonic in (1, 2):
        frequencies.add(carrier_frequency + harmonic * modulation_frequency)
        difference = carrier_frequency - harmonic * modulation_frequency
        if difference > NUMERIC_EPSILON:
            frequencies.add(difference)
    return [
        1.0 / frequency
        for frequency in sorted(frequencies)
        if frequency > NUMERIC_EPSILON
    ]


def _estimate_modulation_period_from_history(
    history: np.ndarray,
    *,
    carrier_period: int,
) -> float:
    """Select a modulation clock by its carrier-sideband energy in history."""
    values = _as_target_matrix(history)
    sample_count = len(values)
    primary_period = max(4, int(carrier_period))
    candidates = _time_varying_modulation_period_candidates(
        sample_count,
        primary_period,
    )
    if not candidates:
        return float(
            max(
                primary_period + 1,
                sample_count // 2,
            )
        )

    baseline_periods = [float(primary_period), float(2 * primary_period)]
    baseline = _harmonic_design(
        sample_count,
        sample_count,
        baseline_periods,
        include_trend=True,
    )
    residual = values - baseline @ _ridge_coefficients(
        baseline,
        values,
    )
    candidate_scores: list[tuple[float, float]] = []
    baseline_frequencies = {
        1.0 / primary_period,
        1.0 / (2 * primary_period),
    }
    for modulation_period in candidates:
        sideband_periods = [
            period
            for period in _time_varying_sideband_periods(
                carrier_period=primary_period,
                modulation_period=modulation_period,
            )
            if all(
                abs(1.0 / period - frequency) > 1.0 / sample_count
                for frequency in baseline_frequencies
            )
        ]
        if not sideband_periods:
            continue
        sidebands = _harmonic_design(
            sample_count,
            sample_count,
            sideband_periods,
            include_trend=False,
        )
        fitted = sidebands @ _ridge_coefficients(
            sidebands,
            residual,
        )
        score = float(np.mean(fitted**2))
        candidate_scores.append((score, modulation_period))

    if not candidate_scores:
        return candidates[0]
    return max(
        candidate_scores,
        key=lambda candidate: (candidate[0], -candidate[1]),
    )[1]


def _harmonic_design(
    length: int,
    origin: int,
    periods: list[float],
    *,
    include_trend: bool,
) -> np.ndarray:
    time = np.arange(length, dtype=float)
    columns = [np.ones(length)]
    if include_trend:
        columns.append((time - max(0, origin - 1)) / max(1, origin))
    for period in sorted({round(float(value), 8) for value in periods if value >= 2.0}):
        phase = 2 * np.pi * time / period
        columns.extend([np.sin(phase), np.cos(phase)])
    return np.column_stack(columns)


def _alternating_state(length: int, cut_points: Iterable[int]) -> np.ndarray:
    state = np.ones(length, dtype=float)
    sign = 1.0
    start = 0
    for point in sorted(
        int(value) for value in cut_points if 0 < int(value) < length
    ):
        state[start:point] = sign
        sign *= -1.0
        start = point
    state[start:] = sign
    return state


def _extrapolated_regime_cut_points(
    history_length: int,
    horizon: int,
    cut_points: Iterable[int],
) -> list[int]:
    """Infer and continue a duration motif using historical cuts only."""
    observed = sorted(
        {
            int(value)
            for value in cut_points
            if 0 < int(value) < int(history_length)
        }
    )
    intervals = [
        int(right - left)
        for left, right in zip(observed, observed[1:])
        if right > left
    ]
    pattern: tuple[int, ...] = ()
    for candidate_length in range(
        1,
        min(4, len(intervals) // 2) + 1,
    ):
        candidate = tuple(intervals[:candidate_length])
        if all(
            value == candidate[index % candidate_length]
            for index, value in enumerate(intervals)
        ):
            pattern = candidate
            break
    if not observed or not pattern:
        return observed

    end = int(history_length) + int(horizon)
    cursor = observed[-1]
    pattern_index = len(intervals) % len(pattern)
    while cursor + pattern[pattern_index] < end:
        cursor += pattern[pattern_index]
        pattern_index = (pattern_index + 1) % len(pattern)
        if cursor >= history_length:
            observed.append(cursor)
    return observed


def _extrapolated_pulse_centers(
    history_length: int,
    horizon: int,
    centers: Iterable[int],
    *,
    future_padding: int = 0,
) -> list[int]:
    """Estimate a short repeating motif from historical pulses and extrapolate."""
    observed = sorted(
        {
            int(value)
            for value in centers
            if 0 <= int(value) < int(history_length)
        }
    )
    if len(observed) < 2:
        return observed

    intervals = np.asarray(
        [right - left for left, right in zip(observed, observed[1:])],
        dtype=float,
    )
    maximum_motif_length = min(5, len(intervals) // 2)
    candidates: list[tuple[float, int, tuple[int, ...]]] = []
    for motif_length in range(2, maximum_motif_length + 1):
        pattern = tuple(
            max(
                1,
                int(round(float(np.median(intervals[offset::motif_length])))),
            )
            for offset in range(motif_length)
        )
        reconstructed = np.asarray(
            [pattern[index % motif_length] for index in range(len(intervals))],
            dtype=float,
        )
        mismatch = float(np.mean(np.abs(intervals - reconstructed)))
        candidates.append((mismatch, motif_length, pattern))

    if candidates:
        _, motif_length, pattern = min(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        next_index = len(intervals) % motif_length
    else:
        pattern = (max(1, int(round(float(np.median(intervals))))),)
        next_index = 0

    end = (
        int(history_length)
        + int(horizon)
        + max(0, int(future_padding))
    )
    cursor = observed[-1]
    while cursor + pattern[next_index] < end:
        cursor += pattern[next_index]
        next_index = (next_index + 1) % len(pattern)
        if cursor >= history_length:
            observed.append(cursor)
    return observed


def _pulse_component(
    length: int,
    centers: Iterable[int],
    width: float,
) -> np.ndarray:
    time = np.arange(length, dtype=float)
    pulse = np.zeros(length, dtype=float)
    safe_width = max(float(width), 1e-6)
    support_radius = int(max(1, np.ceil(4 * safe_width)))
    for center in centers:
        distance = np.abs(time - float(center))
        pulse += np.where(
            distance <= support_radius,
            np.exp(-0.5 * (distance / safe_width) ** 2),
            0.0,
        )
    return pulse


def _nonlinear_forecast(
    history: np.ndarray,
    horizon: int,
    season_length: int,
    latent_params: dict[str, Any],
) -> np.ndarray:
    """Fit a bounded nonlinear recurrence using history-only validation."""

    values = _as_target_matrix(history)
    del latent_params
    seasonal_lag = max(4, int(season_length))
    nonlinear_lags = sorted(
        {
            max(2, seasonal_lag // 3),
            max(2, seasonal_lag // 2),
            max(2, (2 * seasonal_lag) // 3),
        }
    )
    start = max(2, seasonal_lag, *nonlinear_lags)
    if len(values) <= start + 12:
        return _linear_ar_forecast(values, horizon, season_length)

    transform_families = (
        "shifted_sine_squared",
        "shifted_tanh",
    )
    frequencies = (0.4, 0.5, 0.6, 0.7)
    offsets = (-0.6, -0.3, 0.3, 0.6)
    validation_length = min(
        max(12, int(horizon)),
        max(12, (len(values) - start) // 3),
    )
    split = max(start + 8, len(values) - validation_length)
    selected_specs: list[tuple[int, str, float, float]] = []
    coefficients: list[np.ndarray] = []

    for channel in range(values.shape[1]):
        target = values[start:, channel]
        lag1 = values[start - 1 : -1, channel]
        lag2 = values[start - 2 : len(values) - 2, channel]
        lag_seasonal = values[
            start - seasonal_lag : len(values) - seasonal_lag,
            channel,
        ]
        best: tuple[
            float,
            tuple[int, str, float, float],
            np.ndarray,
        ] | None = None
        for nonlinear_lag in nonlinear_lags:
            lag_nonlinear = values[
                start - nonlinear_lag : len(values) - nonlinear_lag,
                channel,
            ]
            for family in transform_families:
                for frequency in frequencies:
                    for offset in offsets:
                        nonlinear_feature = _contrast_nonlinear_response(
                            lag_nonlinear,
                            family=family,
                            frequency=frequency,
                            offset=offset,
                        )
                        design = np.column_stack(
                            [
                                np.ones(len(target)),
                                lag1,
                                lag2,
                                lag_seasonal,
                                lag_nonlinear,
                                nonlinear_feature,
                            ]
                        )
                        fit_rows = split - start
                        coefficient = _ridge_coefficients(
                            design[:fit_rows],
                            target[:fit_rows],
                        ).reshape(-1)
                        error = float(
                            np.mean(
                                (
                                    target[fit_rows:]
                                    - design[fit_rows:] @ coefficient
                                )
                                ** 2
                            )
                        )
                        spec = (
                            nonlinear_lag,
                            family,
                            frequency,
                            offset,
                        )
                        candidate = (error, spec, coefficient)
                        if best is None or candidate[0] < best[0]:
                            best = candidate
        assert best is not None
        _, spec, _ = best
        nonlinear_lag, family, frequency, offset = spec
        lag_nonlinear = values[
            start - nonlinear_lag : len(values) - nonlinear_lag,
            channel,
        ]
        full_design = np.column_stack(
            [
                np.ones(len(target)),
                lag1,
                lag2,
                lag_seasonal,
                lag_nonlinear,
                _contrast_nonlinear_response(
                    lag_nonlinear,
                    family=family,
                    frequency=frequency,
                    offset=offset,
                ),
            ]
        )
        selected_specs.append(spec)
        coefficients.append(
            _ridge_coefficients(full_design, target).reshape(-1)
        )

    output = np.array(values, copy=True)
    medians = np.median(values, axis=0)
    scales = np.maximum(np.std(values, axis=0), 1e-3)
    predictions: list[np.ndarray] = []
    for _ in range(horizon):
        row: list[float] = []
        for channel, (spec, coefficient) in enumerate(
            zip(selected_specs, coefficients)
        ):
            nonlinear_lag, family, frequency, offset = spec
            nonlinear_value = output[-nonlinear_lag, channel]
            features = np.asarray(
                [
                    1.0,
                    output[-1, channel],
                    output[-2, channel],
                    output[-seasonal_lag, channel],
                    nonlinear_value,
                    float(
                        _contrast_nonlinear_response(
                            np.asarray([nonlinear_value]),
                            family=family,
                            frequency=frequency,
                            offset=offset,
                        )[0]
                    ),
                ]
            )
            prediction = float(features @ coefficient)
            row.append(
                float(
                    np.clip(
                        prediction,
                        medians[channel] - 6.0 * scales[channel],
                        medians[channel] + 6.0 * scales[channel],
                    )
                )
            )
        next_value = np.asarray(row)
        predictions.append(next_value)
        output = np.vstack([output, next_value])
    return np.asarray(predictions)


def _contrast_nonlinear_response(
    values: np.ndarray,
    *,
    family: str,
    frequency: float,
    offset: float,
) -> np.ndarray:
    argument = float(frequency) * np.asarray(values, dtype=float) + float(
        offset
    )
    if family == "shifted_sine_squared":
        return np.sin(argument) ** 2 - np.sin(float(offset)) ** 2
    if family == "shifted_tanh":
        return np.tanh(argument) - np.tanh(float(offset))
    raise ValueError(f"unsupported nonlinear transform family: {family}")


def _common_factor_forecast(
    history: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Estimate rank-1 factor/loadings from history and continue fitted laws."""
    values = _as_target_matrix(history)
    center = np.mean(values, axis=0, keepdims=True)
    centered = values - center
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    loading = right[0]
    factor = centered @ loading
    factor_forecast = _short_ar_forecast(factor[:, None], horizon)[:, 0]
    residual = centered - factor[:, None] * loading[None, :]
    residual_forecast = _short_ar_forecast(residual, horizon)
    return (
        center
        + factor_forecast[:, None] * loading[None, :]
        + residual_forecast
    )


def _hierarchical_history_forecast(
    history: np.ndarray,
    horizon: int,
    season_length: int,
) -> np.ndarray:
    """Forecast aggregate and history-estimated zero-sum child contrasts."""
    values = _as_target_matrix(history)
    if values.shape[1] < 3:
        raise ValueError(
            "hierarchical contrast requires a parent and at least two children"
        )
    children = values[:, 1:]
    child_count = children.shape[1]
    aggregate_history = np.sum(children, axis=1, keepdims=True)
    aggregate_forecast = _linear_ar_forecast(
        aggregate_history,
        horizon,
        season_length,
    )
    deviations = children - aggregate_history / child_count

    try:
        _, _, right = np.linalg.svd(
            deviations,
            full_matrices=False,
        )
        contrast_rank = max(1, child_count - 1)
        loadings = right[:contrast_rank].T
        scores = deviations @ loadings
        score_forecast = _linear_ar_forecast(
            scores,
            horizon,
            season_length,
        )
        deviation_forecast = score_forecast @ loadings.T
    except np.linalg.LinAlgError:
        deviation_forecast = _linear_ar_forecast(
            deviations,
            horizon,
            season_length,
        )

    deviation_forecast -= np.mean(
        deviation_forecast,
        axis=1,
        keepdims=True,
    )
    child_forecast = aggregate_forecast / child_count + deviation_forecast
    parent_forecast = np.sum(child_forecast, axis=1, keepdims=True)
    return np.column_stack([parent_forecast, child_forecast])


def _ridge_coefficients(
    design: np.ndarray,
    response: np.ndarray,
    *,
    penalty: float = 1e-5,
) -> np.ndarray:
    matrix = np.asarray(design, dtype=float)
    target = np.asarray(response, dtype=float)
    gram = matrix.T @ matrix
    regularizer = float(penalty) * np.eye(gram.shape[0])
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        gram + regularizer,
        matrix.T @ target,
    )


def _structure_error(capability_id: str, forecast: np.ndarray) -> float:
    values = _as_target_matrix(forecast)
    if capability_id != "hierarchical_coherence" or values.shape[1] < 3:
        return 0.0
    return float(
        np.mean(
            np.abs(values[:, 0] - np.sum(values[:, 1:], axis=1))
        )
    )
