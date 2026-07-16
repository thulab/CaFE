from __future__ import annotations

from typing import Any

import numpy as np


REGIME_HISTORY_INCREMENTAL_R2_MIN = 0.10
REGIME_FUTURE_MSE_GAIN_MIN = 0.10
REGIME_AMPLITUDE_RATIO_MIN = 0.30
REGIME_HISTORY_LEVEL_SHIFT_RATIO_MIN = 1.10
REGIME_FUTURE_LEVEL_SHIFT_RATIO_MIN = 1.15
REGIME_HISTORY_DIRECTION_CONSISTENCY_MIN = 0.80
REGIME_HISTORY_STATE_COVERAGE_MIN = 0.55
REGIME_FUTURE_STATE_COVERAGE_MIN = 0.55
REGIME_CONTEXT_ABSOLUTE_SKEW_MAX = 2.0


def regime_clock_features(
    target: np.ndarray,
    *,
    context_length: int,
    season_length: int,
) -> dict[str, Any]:
    """Audit a predictable recurring two-state clock on a complete window.

    Candidate period/duty/phase values are selected using history only. Qualification
    then requires the selected clock to improve prediction on the untouched
    future segment, preventing a single drift or one-off change point from being
    mistaken for recurring regime switching.
    """

    values = np.asarray(target, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or context_length <= 0 or context_length >= len(values):
        raise ValueError("regime qualification requires [time, target] with a non-empty future")
    if not np.isfinite(values).all():
        raise ValueError("regime qualification does not accept missing target values")

    per_channel = [
        _regime_clock_channel_features(
            values[:, channel],
            context_length=context_length,
            season_length=season_length,
        )
        for channel in range(values.shape[1])
    ]
    history_r2 = float(np.median([item["history_incremental_r2"] for item in per_channel]))
    future_gain = float(np.median([item["future_mse_gain"] for item in per_channel]))
    amplitude_ratio = float(np.median([item["amplitude_ratio"] for item in per_channel]))
    history_level_shift_ratio = float(
        np.median([item["history_level_shift_ratio"] for item in per_channel])
    )
    future_level_shift_ratio = float(
        np.median([item["future_level_shift_ratio"] for item in per_channel])
    )
    history_direction_consistency = float(
        np.median([item["history_direction_consistency"] for item in per_channel])
    )
    future_direction_consistency = float(
        np.median([item["future_direction_consistency"] for item in per_channel])
    )
    history_state_coverage = float(
        np.median([item["history_state_coverage"] for item in per_channel])
    )
    future_state_coverage = float(
        np.median([item["future_state_coverage"] for item in per_channel])
    )
    context_absolute_skew = float(
        np.median([item["context_absolute_skew"] for item in per_channel])
    )
    qualified = (
        history_r2 >= REGIME_HISTORY_INCREMENTAL_R2_MIN
        and future_gain >= REGIME_FUTURE_MSE_GAIN_MIN
        and amplitude_ratio >= REGIME_AMPLITUDE_RATIO_MIN
        and history_level_shift_ratio >= REGIME_HISTORY_LEVEL_SHIFT_RATIO_MIN
        and future_level_shift_ratio >= REGIME_FUTURE_LEVEL_SHIFT_RATIO_MIN
        and history_direction_consistency >= REGIME_HISTORY_DIRECTION_CONSISTENCY_MIN
        and future_direction_consistency == 1.0
        and history_state_coverage >= REGIME_HISTORY_STATE_COVERAGE_MIN
        and future_state_coverage >= REGIME_FUTURE_STATE_COVERAGE_MIN
        and context_absolute_skew <= REGIME_CONTEXT_ABSOLUTE_SKEW_MAX
        and all(item["historical_switch_count"] >= 2 for item in per_channel)
        and all(item["future_switch_count"] >= 1 for item in per_channel)
    )
    return {
        "qualified": bool(qualified),
        "history_incremental_r2": history_r2,
        "future_mse_gain": future_gain,
        "amplitude_ratio": amplitude_ratio,
        "history_level_shift_ratio": history_level_shift_ratio,
        "future_level_shift_ratio": future_level_shift_ratio,
        "history_direction_consistency": history_direction_consistency,
        "future_direction_consistency": future_direction_consistency,
        "history_state_coverage": history_state_coverage,
        "future_state_coverage": future_state_coverage,
        "context_absolute_skew": context_absolute_skew,
        "selected_dwell_length": int(round(np.median([item["dwell_length"] for item in per_channel]))),
        "selected_period": int(round(np.median([item["period"] for item in per_channel]))),
        "selected_duty_fraction": float(
            np.median([item["duty_fraction"] for item in per_channel])
        ),
        "historical_switch_count": int(min(item["historical_switch_count"] for item in per_channel)),
        "future_switch_count": int(min(item["future_switch_count"] for item in per_channel)),
        "thresholds": {
            "history_incremental_r2_min": REGIME_HISTORY_INCREMENTAL_R2_MIN,
            "future_mse_gain_min": REGIME_FUTURE_MSE_GAIN_MIN,
            "amplitude_ratio_min": REGIME_AMPLITUDE_RATIO_MIN,
            "history_level_shift_ratio_min": REGIME_HISTORY_LEVEL_SHIFT_RATIO_MIN,
            "future_level_shift_ratio_min": REGIME_FUTURE_LEVEL_SHIFT_RATIO_MIN,
            "history_direction_consistency_min": REGIME_HISTORY_DIRECTION_CONSISTENCY_MIN,
            "history_state_coverage_min": REGIME_HISTORY_STATE_COVERAGE_MIN,
            "future_state_coverage_min": REGIME_FUTURE_STATE_COVERAGE_MIN,
            "context_absolute_skew_max": REGIME_CONTEXT_ABSOLUTE_SKEW_MAX,
        },
        "channels": per_channel,
    }


def _regime_clock_channel_features(
    values: np.ndarray,
    *,
    context_length: int,
    season_length: int,
) -> dict[str, float | int]:
    length = len(values)
    history = values[:context_length]
    future = values[context_length:]
    best: dict[str, Any] | None = None
    for period in _candidate_periods(context_length, season_length):
        baseline = _baseline_design(length, season_length, clock_period=period)
        baseline_history = baseline[:context_length]
        baseline_coef = np.linalg.lstsq(baseline_history, history, rcond=None)[0]
        baseline_history_prediction = baseline_history @ baseline_coef
        baseline_sse = float(np.sum((history - baseline_history_prediction) ** 2))
        baseline_scale = max(baseline_sse, 1e-9)
        for duty_fraction in _candidate_duty_fractions():
            active_length = int(round(period * duty_fraction))
            inactive_length = period - active_length
            dwell_length = min(active_length, inactive_length)
            phase_step = max(1, period // 96)
            for phase in range(0, period, phase_step):
                state = _periodic_state(length, period, active_length, phase)
                historical_switch_count = int(np.sum(np.diff(state[:context_length]) != 0.0))
                future_switch_count = int(np.sum(np.diff(state[context_length - 1 :]) != 0.0))
                if historical_switch_count < 2 or future_switch_count < 1:
                    continue
                full_history_design = np.column_stack([baseline_history, state[:context_length]])
                full_coef = np.linalg.lstsq(full_history_design, history, rcond=None)[0]
                full_sse = float(np.sum((history - full_history_design @ full_coef) ** 2))
                history_incremental_r2 = max(0.0, (baseline_sse - full_sse) / baseline_scale)
                candidate = {
                    "history_incremental_r2": history_incremental_r2,
                    "period": period,
                    "duty_fraction": active_length / period,
                    "dwell_length": dwell_length,
                    "phase": phase,
                    "state": state,
                    "coef": full_coef,
                    "historical_switch_count": historical_switch_count,
                    "future_switch_count": future_switch_count,
                    "baseline": baseline,
                    "baseline_coef": baseline_coef,
                    "baseline_history_prediction": baseline_history_prediction,
                }
                if best is None or (
                    history_incremental_r2,
                    -abs(active_length / period - 0.5),
                    -period,
                    -phase,
                ) > (
                    best["history_incremental_r2"],
                    -abs(best["duty_fraction"] - 0.5),
                    -best["period"],
                    -best["phase"],
                ):
                    best = candidate
    if best is None:
        return {
            "history_incremental_r2": 0.0,
            "future_mse_gain": -1.0,
            "amplitude_ratio": 0.0,
            "history_level_shift_ratio": 0.0,
            "future_level_shift_ratio": 0.0,
            "history_direction_consistency": 0.0,
            "future_direction_consistency": 0.0,
            "history_state_coverage": 0.0,
            "future_state_coverage": 0.0,
            "context_absolute_skew": _absolute_skew(history),
            "dwell_length": 0,
            "period": 0,
            "duty_fraction": 0.0,
            "phase": 0,
            "historical_switch_count": 0,
            "future_switch_count": 0,
        }

    baseline = best["baseline"]
    baseline_coef = best["baseline_coef"]
    baseline_history_prediction = best["baseline_history_prediction"]
    baseline_future_prediction = baseline[context_length:] @ baseline_coef
    full_future_design = np.column_stack(
        [baseline[context_length:], best["state"][context_length:]]
    )
    full_future_prediction = full_future_design @ best["coef"]
    baseline_future_mse = float(np.mean((future - baseline_future_prediction) ** 2))
    full_future_mse = float(np.mean((future - full_future_prediction) ** 2))
    future_mse_gain = (baseline_future_mse - full_future_mse) / max(
        baseline_future_mse,
        1e-9,
    )
    amplitude_ratio = abs(float(best["coef"][-1])) / max(
        float(np.std(history - baseline_history_prediction)),
        1e-9,
    )
    transition_features = _persistent_transition_features(
        values,
        state=best["state"],
        state_coefficient=float(best["coef"][-1]),
        dwell_length=int(best["dwell_length"]),
        context_length=context_length,
    )
    state_coverage = _state_coverage_features(
        values,
        state=best["state"],
        state_coefficient=float(best["coef"][-1]),
        baseline_prediction=baseline @ best["coef"][:-1],
        context_length=context_length,
    )
    return {
        "history_incremental_r2": float(best["history_incremental_r2"]),
        "future_mse_gain": float(future_mse_gain),
        "amplitude_ratio": float(amplitude_ratio),
        **transition_features,
        **state_coverage,
        "dwell_length": int(best["dwell_length"]),
        "period": int(best["period"]),
        "duty_fraction": float(best["duty_fraction"]),
        "phase": int(best["phase"]),
        "historical_switch_count": int(best["historical_switch_count"]),
        "future_switch_count": int(best["future_switch_count"]),
        "context_absolute_skew": _absolute_skew(history),
    }


def _candidate_periods(context_length: int, season_length: int) -> tuple[int, ...]:
    season = max(4, int(season_length))
    expected_dwell = max(4, min(2 * season, max(4, context_length // 3)))
    candidates = {season, 2 * season, 2 * expected_dwell}
    weekly_period = 7 * season
    if 2 * weekly_period <= context_length:
        candidates.add(weekly_period)
    return tuple(sorted(period for period in candidates if period <= context_length))


def _candidate_duty_fractions() -> tuple[float, ...]:
    return (0.25, 1.0 / 3.0, 0.50, 2.0 / 3.0, 0.75)


def _periodic_state(
    length: int,
    period: int,
    active_length: int,
    phase: int,
) -> np.ndarray:
    position = np.mod(np.arange(length, dtype=int) - int(phase), int(period))
    return np.where(position < int(active_length), 1.0, -1.0)


def _persistent_transition_features(
    values: np.ndarray,
    *,
    state: np.ndarray,
    state_coefficient: float,
    dwell_length: int,
    context_length: int,
) -> dict[str, float]:
    plateau_width = max(2, min(8, dwell_length // 4))
    transition_margin = max(1, min(3, dwell_length // 8))
    required_radius = plateau_width + transition_margin
    switch_indices = np.flatnonzero(np.diff(state) != 0.0) + 1
    history_indices = switch_indices[switch_indices < context_length]
    future_indices = switch_indices[switch_indices >= context_length]

    def plateau_shift(index: int) -> float:
        before = values[
            index - transition_margin - plateau_width : index - transition_margin
        ]
        after = values[
            index + transition_margin : index + transition_margin + plateau_width
        ]
        return float(np.median(after) - np.median(before))

    valid_centers = np.arange(required_radius, len(values) - required_radius + 1)
    local_shifts = np.asarray(
        [plateau_shift(int(index)) for index in valid_centers],
        dtype=float,
    )
    reference_mask = np.ones(len(valid_centers), dtype=bool)
    for index in switch_indices:
        reference_mask &= np.abs(valid_centers - index) > transition_margin
    reference = np.abs(local_shifts[reference_mask])
    if not len(reference):
        reference = np.abs(np.diff(values))
    reference_scale = max(float(np.quantile(reference, 0.75)), 1e-9)

    def summarize(indices: np.ndarray) -> tuple[float, float]:
        indices = indices[
            (indices >= required_radius) & (indices <= len(values) - required_radius)
        ]
        if not len(indices):
            return 0.0, 0.0
        shifts = np.asarray([plateau_shift(int(index)) for index in indices], dtype=float)
        expected = np.asarray(
            [state_coefficient * (state[index] - state[index - 1]) for index in indices],
            dtype=float,
        )
        ratio = float(np.median(np.abs(shifts))) / reference_scale
        consistency = float(np.mean(np.sign(shifts) == np.sign(expected)))
        return ratio, consistency

    history_ratio, history_consistency = summarize(history_indices)
    future_ratio, future_consistency = summarize(future_indices)
    return {
        "history_level_shift_ratio": float(history_ratio),
        "future_level_shift_ratio": float(future_ratio),
        "history_direction_consistency": float(history_consistency),
        "future_direction_consistency": float(future_consistency),
    }


def _state_coverage_features(
    values: np.ndarray,
    *,
    state: np.ndarray,
    state_coefficient: float,
    baseline_prediction: np.ndarray,
    context_length: int,
) -> dict[str, float]:
    residual = values - baseline_prediction
    signed_residual = residual * np.sign(state_coefficient * state)
    minimum_margin = 0.25 * abs(state_coefficient)

    def coverage(segment: slice) -> float:
        segment_state = state[segment]
        segment_residual = signed_residual[segment]
        state_coverages = [
            float(np.mean(segment_residual[segment_state == state_value] >= minimum_margin))
            for state_value in (-1.0, 1.0)
            if np.any(segment_state == state_value)
        ]
        return min(state_coverages, default=0.0)

    return {
        "history_state_coverage": coverage(slice(0, context_length)),
        "future_state_coverage": coverage(slice(context_length, None)),
    }


def _absolute_skew(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    scale = float(np.std(centered))
    if scale <= 1e-9:
        return 0.0
    return abs(float(np.mean((centered / scale) ** 3)))


def _baseline_design(
    length: int,
    season_length: int,
    *,
    clock_period: int,
) -> np.ndarray:
    time = np.arange(length, dtype=float)
    normalized_time = (time - (length - 1) / 2.0) / max(length - 1, 1)
    period = max(4, int(season_length))
    columns = [np.ones(length), normalized_time]
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * time / period
        columns.extend([np.sin(angle), np.cos(angle)])
    for harmonic in (1, 2, 3, 4, 5):
        angle = 2.0 * np.pi * harmonic * time / max(4, int(clock_period))
        columns.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(columns)
