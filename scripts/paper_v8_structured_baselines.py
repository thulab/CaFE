#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STRUCTURED_BASELINE_SCHEMA_VERSION = "paper_v8_structured_baseline.v1"
STRUCTURED_CAPABILITIES = frozenset(
    {"common_factor", "cross_series_dependence"}
)
STRUCTURED_EVALUATION_TABLES = frozenset(
    {"main", "multivariate_input_ablation", "strict_counterfactual_audit"}
)
RIDGE_ALPHA_CANDIDATES = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
BASE_LAG_CANDIDATES = (1, 2, 4, 6, 8, 12, 16, 24, 32)
DFM_ALPHA_CANDIDATES = (0.0, 0.01, 0.1, 1.0)
DFM_BASE_LAG_CANDIDATES = (1, 4, 8, 12, 24, 32)
VALIDATION_FRACTION = 0.25
MIN_TRAINING_ROWS = 8
MIN_VALIDATION_ROWS = 8
STANDARDIZED_FORECAST_LIMIT = 20.0
VAR_STABILITY_RADIUS = 1.01


@dataclass(frozen=True)
class StructuredForecast:
    forecast: np.ndarray
    diagnostics: dict[str, Any]


def is_structured_sample(sample: dict[str, Any]) -> bool:
    return (
        str(sample.get("capability_id")) in STRUCTURED_CAPABILITIES
        and str(sample.get("generator_family_role")) == "primary"
        and int(sample.get("intensity", -1)) == 5
        and str(sample.get("evaluation_table", "main"))
        in STRUCTURED_EVALUATION_TABLES
    )


def baseline_ids_for(capability_id: str) -> tuple[str, ...]:
    if capability_id == "common_factor":
        return ("diagonal_ar", "dynamic_factor_var")
    if capability_id == "cross_series_dependence":
        return ("diagonal_ar", "ridge_var")
    return ()


def lag_candidates(context_length: int, horizon: int) -> tuple[int, ...]:
    maximum = max(1, context_length // 2)
    values = {
        int(value)
        for value in (*BASE_LAG_CANDIDATES, horizon)
        if 1 <= int(value) <= maximum
    }
    return tuple(sorted(values))


def dfm_lag_candidates(
    context_length: int,
    horizon: int,
) -> tuple[int, ...]:
    maximum = max(1, context_length // 2)
    return tuple(
        sorted(
            {
                int(value)
                for value in (*DFM_BASE_LAG_CANDIDATES, horizon)
                if 1 <= int(value) <= maximum
            }
        )
    )


def _standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return center, scale


def _lag_design(values: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or len(values) <= lag:
        raise ValueError("lagged regression requires a two-dimensional history")
    design = np.vstack(
        [
            np.concatenate(
                [values[index - offset] for offset in range(1, lag + 1)]
            )
            for index in range(lag, len(values))
        ]
    )
    return design, values[lag:]


def _ridge(
    design: np.ndarray,
    response: np.ndarray,
    alpha: float,
) -> np.ndarray:
    augmented = np.column_stack([np.ones(len(design)), design])
    penalty = np.eye(augmented.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    system = augmented.T @ augmented + (
        float(alpha) * max(len(design), 1) * penalty
    )
    right = augmented.T @ response
    try:
        return np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, right, rcond=None)[0]


def _fit_coefficients(
    values: np.ndarray,
    *,
    lag: int,
    alpha: float,
    diagonal: bool,
) -> np.ndarray | list[np.ndarray]:
    if diagonal:
        coefficients: list[np.ndarray] = []
        for channel in range(values.shape[1]):
            design, target = _lag_design(values[:, channel : channel + 1], lag)
            coefficients.append(_ridge(design, target, alpha).reshape(-1))
        return coefficients
    design, target = _lag_design(values, lag)
    return _ridge(design, target, alpha)


def _predict_one(
    history: np.ndarray,
    coefficients: np.ndarray | list[np.ndarray],
    *,
    lag: int,
    diagonal: bool,
) -> np.ndarray:
    if diagonal:
        assert isinstance(coefficients, list)
        return np.asarray(
            [
                float(
                    np.concatenate(
                        [
                            np.ones(1),
                            np.asarray(
                                [
                                    history[-offset, channel]
                                    for offset in range(1, lag + 1)
                                ]
                            ),
                        ]
                    )
                    @ coefficients[channel]
                )
                for channel in range(history.shape[1])
            ],
            dtype=float,
        )
    assert isinstance(coefficients, np.ndarray)
    features = np.concatenate(
        [history[-offset] for offset in range(1, lag + 1)]
    )
    return np.concatenate([np.ones(1), features]) @ coefficients


def _recursive_forecast(
    history: np.ndarray,
    coefficients: np.ndarray | list[np.ndarray],
    *,
    lag: int,
    diagonal: bool,
    horizon: int,
) -> np.ndarray:
    extended = np.asarray(history, dtype=float).copy()
    forecast = []
    for _ in range(horizon):
        next_value = _predict_one(
            extended,
            coefficients,
            lag=lag,
            diagonal=diagonal,
        )
        forecast.append(next_value)
        extended = np.vstack([extended, next_value])
    return np.asarray(forecast, dtype=float)


def _stable_var_coefficients(
    coefficients: np.ndarray,
    *,
    dimension: int,
    lag: int,
) -> tuple[np.ndarray, float, float, float]:
    def radius(scale: float) -> float:
        companion = np.zeros(
            (dimension * lag, dimension * lag),
            dtype=float,
        )
        for offset in range(lag):
            block = coefficients[
                1 + offset * dimension : 1 + (offset + 1) * dimension
            ]
            companion[:dimension, offset * dimension : (offset + 1) * dimension] = (
                scale * block.T
            )
        if lag > 1:
            companion[dimension:, :-dimension] = np.eye(
                dimension * (lag - 1)
            )
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    before = radius(1.0)
    if before <= VAR_STABILITY_RADIUS:
        return coefficients, before, before, 1.0
    lower, upper = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        if radius(middle) <= VAR_STABILITY_RADIUS:
            lower = middle
        else:
            upper = middle
    result = coefficients.copy()
    result[1:] *= lower
    return result, before, radius(lower), lower


def _stable_diagonal_coefficients(
    coefficients: list[np.ndarray],
    *,
    lag: int,
) -> tuple[list[np.ndarray], float, float, float]:
    output = []
    before = []
    after = []
    scales = []
    for values in coefficients:
        stable, radius_before, radius_after, scale = (
            _stable_var_coefficients(
                values[:, None],
                dimension=1,
                lag=lag,
            )
        )
        output.append(stable[:, 0])
        before.append(radius_before)
        after.append(radius_after)
        scales.append(scale)
    return (
        output,
        max(before, default=0.0),
        max(after, default=0.0),
        min(scales, default=1.0),
    )


def _hub_design(
    values: np.ndarray,
    *,
    source: int,
    destination: int,
    cross_lag: int,
    own_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = max(cross_lag, own_lag)
    rows = []
    for index in range(start, len(values)):
        features = (
            [
                values[index - offset, destination]
                for offset in range(1, own_lag + 1)
            ]
            if destination == source
            else [values[index - cross_lag, source]]
        )
        rows.append(features)
    return np.asarray(rows, dtype=float), values[start:, destination]


def _fit_hub_coefficients(
    values: np.ndarray,
    *,
    source: int,
    cross_lag: int,
    own_lag: int,
    alpha: float,
) -> list[np.ndarray]:
    output = []
    for destination in range(values.shape[1]):
        design, target = _hub_design(
            values,
            source=source,
            destination=destination,
            cross_lag=cross_lag,
            own_lag=own_lag,
        )
        output.append(_ridge(design, target[:, None], alpha).reshape(-1))
    return output


def _predict_hub_one(
    history: np.ndarray,
    coefficients: list[np.ndarray],
    *,
    source: int,
    cross_lag: int,
    own_lag: int,
) -> np.ndarray:
    predictions = []
    for destination in range(history.shape[1]):
        features = (
            [
                history[-offset, destination]
                for offset in range(1, own_lag + 1)
            ]
            if destination == source
            else [history[-cross_lag, source]]
        )
        predictions.append(
            float(
                np.concatenate([np.ones(1), np.asarray(features)])
                @ coefficients[destination]
            )
        )
    return np.asarray(predictions, dtype=float)


def _recursive_hub_forecast(
    history: np.ndarray,
    coefficients: list[np.ndarray],
    *,
    source: int,
    cross_lag: int,
    own_lag: int,
    horizon: int,
) -> np.ndarray:
    extended = np.asarray(history, dtype=float).copy()
    output = []
    for _ in range(horizon):
        next_value = _predict_hub_one(
            extended,
            coefficients,
            source=source,
            cross_lag=cross_lag,
            own_lag=own_lag,
        )
        output.append(next_value)
        extended = np.vstack([extended, next_value])
    return np.asarray(output, dtype=float)


def _hub_chronological_validation(
    history: np.ndarray,
    *,
    horizon: int,
) -> tuple[int, int, int, float, float, float, int]:
    context = len(history)
    validation_rows = max(
        MIN_VALIDATION_ROWS,
        int(round(VALIDATION_FRACTION * context)),
    )
    validation_rows = min(validation_rows, context - MIN_TRAINING_ROWS - 1)
    split = context - validation_rows
    center, scale = _standardizer(history[:split])
    standardized = (history - center) / scale
    own_lag = min(4, max(1, split // 8))
    edge_candidates: list[tuple[float, int, int]] = []
    for cross_lag in lag_candidates(context, horizon):
        paired_length = context - cross_lag
        edge_split = int(
            np.clip(round(0.70 * paired_length), 8, paired_length - 4)
        )
        for source in range(history.shape[1]):
            holdout_r2 = []
            for destination in range(history.shape[1]):
                if destination == source:
                    continue
                source_values = standardized[:-cross_lag, source]
                response_values = standardized[cross_lag:, destination]
                design = np.column_stack(
                    [np.ones(edge_split), source_values[:edge_split]]
                )
                coefficients = np.linalg.lstsq(
                    design,
                    response_values[:edge_split],
                    rcond=None,
                )[0]
                truth = response_values[edge_split:]
                prediction = (
                    coefficients[0]
                    + coefficients[1] * source_values[edge_split:]
                )
                denominator = float(
                    np.sum((truth - float(np.mean(truth))) ** 2)
                )
                holdout_r2.append(
                    (
                        1.0
                        - float(np.sum((truth - prediction) ** 2))
                        / denominator
                    )
                    if denominator > 1e-12
                    else -np.inf
                )
            score = float(np.mean(holdout_r2))
            if np.isfinite(score):
                edge_candidates.append((score, source, cross_lag))
    if not edge_candidates:
        raise ValueError("no valid blind source-lag edge candidate")
    discovery_r2, source, cross_lag = max(
        edge_candidates,
        key=lambda row: (row[0], -row[2], -row[1]),
    )
    effective_rows = split - max(cross_lag, own_lag)
    if effective_rows < MIN_TRAINING_ROWS:
        raise ValueError("discovered edge leaves too few training rows")
    destinations = [
        index for index in range(history.shape[1]) if index != source
    ]
    candidates: list[tuple[float, float]] = []
    for alpha in RIDGE_ALPHA_CANDIDATES:
        coefficients = _fit_hub_coefficients(
            standardized[:split],
            source=source,
            cross_lag=cross_lag,
            own_lag=own_lag,
            alpha=alpha,
        )
        predictions = np.vstack(
            [
                _predict_hub_one(
                    standardized[:index],
                    coefficients,
                    source=source,
                    cross_lag=cross_lag,
                    own_lag=own_lag,
                )
                for index in range(split, context)
            ]
        )
        destination_loss = float(
            np.mean(
                np.abs(
                    standardized[split:, destinations]
                    - predictions[:, destinations]
                )
            )
        )
        if np.isfinite(destination_loss):
            candidates.append((destination_loss, float(alpha)))
    if not candidates:
        raise ValueError("no valid sparse hub VAR candidate")
    loss, alpha = min(
        candidates,
        key=lambda row: (row[0], -row[1]),
    )
    return (
        source,
        cross_lag,
        own_lag,
        alpha,
        loss,
        discovery_r2,
        effective_rows,
    )


def _chronological_validation(
    history: np.ndarray,
    *,
    diagonal: bool,
    horizon: int,
) -> tuple[int, float, float, int]:
    context = len(history)
    validation_rows = max(
        MIN_VALIDATION_ROWS,
        int(round(VALIDATION_FRACTION * context)),
    )
    validation_rows = min(validation_rows, context - MIN_TRAINING_ROWS - 1)
    split = context - validation_rows
    candidates: list[tuple[float, int, float, int]] = []
    for lag in lag_candidates(context, horizon):
        effective_rows = split - lag
        if effective_rows < MIN_TRAINING_ROWS:
            continue
        center, scale = _standardizer(history[:split])
        standardized = (history - center) / scale
        for alpha in RIDGE_ALPHA_CANDIDATES:
            coefficients = _fit_coefficients(
                standardized[:split],
                lag=lag,
                alpha=alpha,
                diagonal=diagonal,
            )
            predictions = np.vstack(
                [
                    _predict_one(
                        standardized[:index],
                        coefficients,
                        lag=lag,
                        diagonal=diagonal,
                    )
                    for index in range(split, context)
                ]
            )
            loss = float(
                np.mean(np.abs(standardized[split:] - predictions))
            )
            if np.isfinite(loss):
                candidates.append((loss, lag, float(alpha), effective_rows))
    if not candidates:
        raise ValueError("no valid chronological ridge candidate")
    loss, lag, alpha, effective_rows = min(
        candidates,
        key=lambda row: (row[0], row[1], -row[2]),
    )
    return lag, alpha, loss, effective_rows


def _validate_forecast(
    forecast: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
) -> str | None:
    if not np.isfinite(forecast).all():
        return "non_finite_forecast"
    standardized = (forecast - center) / scale
    if float(np.max(np.abs(standardized))) > STANDARDIZED_FORECAST_LIMIT:
        return "standardized_forecast_limit_exceeded"
    return None


def _fallback(
    history: np.ndarray,
    horizon: int,
    *,
    model_id: str,
    reason: str,
    diagnostics: dict[str, Any],
) -> StructuredForecast:
    return StructuredForecast(
        forecast=np.repeat(history[-1:], horizon, axis=0),
        diagnostics={
            **diagnostics,
            "schema_version": STRUCTURED_BASELINE_SCHEMA_VERSION,
            "model_id": model_id,
            "fit_status": "fallback",
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_model": "last_value",
        },
    )


def _ar_or_var_forecast(
    history: np.ndarray,
    horizon: int,
    *,
    model_id: str,
    diagonal: bool,
) -> StructuredForecast:
    diagnostics: dict[str, Any] = {
        "history_only": True,
        "context_length": int(len(history)),
        "horizon": int(horizon),
        "target_dim": int(history.shape[1]),
        "validation_policy": "final_25pct_chronological_one_step",
        "lag_candidates": list(lag_candidates(len(history), horizon)),
        "ridge_alpha_candidates": list(RIDGE_ALPHA_CANDIDATES),
        "standardized_forecast_limit": STANDARDIZED_FORECAST_LIMIT,
    }
    try:
        lag, alpha, validation_mae, training_rows = (
            _chronological_validation(
                history,
                diagonal=diagonal,
                horizon=horizon,
            )
        )
        center, scale = _standardizer(history)
        standardized = (history - center) / scale
        coefficients = _fit_coefficients(
            standardized,
            lag=lag,
            alpha=alpha,
            diagonal=diagonal,
        )
        standardized_forecast = _recursive_forecast(
            standardized,
            coefficients,
            lag=lag,
            diagonal=diagonal,
            horizon=horizon,
        )
        forecast = center + standardized_forecast * scale
        failure = _validate_forecast(
            forecast,
            center=center,
            scale=scale,
        )
        diagnostics.update(
            {
                "selected_lag": int(lag),
                "selected_ridge_alpha": float(alpha),
                "validation_normalized_mae": float(validation_mae),
                "validation_training_rows": int(training_rows),
            }
        )
        if failure is not None:
            return _fallback(
                history,
                horizon,
                model_id=model_id,
                reason=failure,
                diagnostics=diagnostics,
            )
        return StructuredForecast(
            forecast=forecast,
            diagnostics={
                **diagnostics,
                "schema_version": STRUCTURED_BASELINE_SCHEMA_VERSION,
                "model_id": model_id,
                "fit_status": "ok",
                "fallback_used": False,
                "fallback_reason": None,
            },
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return _fallback(
            history,
            horizon,
            model_id=model_id,
            reason=f"{type(error).__name__}:{error}",
            diagnostics=diagnostics,
        )


def _hub_var_forecast(
    history: np.ndarray,
    horizon: int,
) -> StructuredForecast:
    model_id = "ridge_var"
    diagnostics: dict[str, Any] = {
        "history_only": True,
        "context_length": int(len(history)),
        "horizon": int(horizon),
        "target_dim": int(history.shape[1]),
        "model_structure": "blind_single_source_sparse_lag_hub_var",
        "validation_policy": (
            "final_25pct_chronological_one_step_max_gain_over_diagonal_ar"
        ),
        "lag_candidates": list(lag_candidates(len(history), horizon)),
        "ridge_alpha_candidates": list(RIDGE_ALPHA_CANDIDATES),
        "standardized_forecast_limit": STANDARDIZED_FORECAST_LIMIT,
    }
    try:
        (
            source,
            cross_lag,
            own_lag,
            alpha,
            validation_mae,
            discovery_r2,
            training_rows,
        ) = _hub_chronological_validation(history, horizon=horizon)
        center, scale = _standardizer(history)
        standardized = (history - center) / scale
        coefficients = _fit_hub_coefficients(
            standardized,
            source=source,
            cross_lag=cross_lag,
            own_lag=own_lag,
            alpha=alpha,
        )
        standardized_forecast = _recursive_hub_forecast(
            standardized,
            coefficients,
            source=source,
            cross_lag=cross_lag,
            own_lag=own_lag,
            horizon=horizon,
        )
        forecast = center + standardized_forecast * scale
        failure = _validate_forecast(
            forecast,
            center=center,
            scale=scale,
        )
        diagnostics.update(
            {
                "selected_source_channel": int(source),
                "selected_lag": int(cross_lag),
                "selected_own_lag": int(own_lag),
                "selected_ridge_alpha": float(alpha),
                "blind_edge_mean_holdout_r2": float(discovery_r2),
                "validation_normalized_mae": float(validation_mae),
                "validation_training_rows": int(training_rows),
            }
        )
        if failure is not None:
            return _fallback(
                history,
                horizon,
                model_id=model_id,
                reason=failure,
                diagnostics=diagnostics,
            )
        return StructuredForecast(
            forecast=forecast,
            diagnostics={
                **diagnostics,
                "schema_version": STRUCTURED_BASELINE_SCHEMA_VERSION,
                "model_id": model_id,
                "fit_status": "ok",
                "fallback_used": False,
                "fallback_reason": None,
            },
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return _fallback(
            history,
            horizon,
            model_id=model_id,
            reason=f"{type(error).__name__}:{error}",
            diagnostics=diagnostics,
        )


def forecast_cross_counterfactual_pair(
    first_sample: dict[str, Any],
    second_sample: dict[str, Any],
) -> tuple[StructuredForecast, StructuredForecast]:
    """Fit one blind ARDL on pair-invariant history and apply it to both paths."""

    first_target = np.asarray(first_sample["target"], dtype=float)
    second_target = np.asarray(second_sample["target"], dtype=float)
    context = int(first_sample["context_length"])
    horizon = int(first_sample["horizon"])
    if (
        first_target.shape != second_target.shape
        or context != int(second_sample["context_length"])
        or horizon != int(second_sample["horizon"])
    ):
        raise ValueError("counterfactual members must have matching shapes")
    first_history = first_target[:context]
    second_history = second_target[:context]
    difference = np.max(
        np.abs(first_history - second_history),
        axis=1,
    )
    changed = np.flatnonzero(difference > 1e-10)
    invariant_stop = int(changed[0]) if changed.size else context
    diagnostics: dict[str, Any] = {
        "history_only": True,
        "paired_shared_fit": True,
        "pair_invariant_history_stop": invariant_stop,
        "context_length": context,
        "horizon": horizon,
        "target_dim": int(first_history.shape[1]),
        "model_structure": "blind_shared_fit_counterfactual_ardl",
        "generator_metadata_used_for_fitting": False,
        "validation_policy": "blind_source_lag_holdout_r2_then_ridge",
        "lag_candidates": list(lag_candidates(invariant_stop, horizon)),
        "ridge_alpha_candidates": list(RIDGE_ALPHA_CANDIDATES),
    }
    try:
        invariant_history = first_history[:invariant_stop]
        (
            source,
            cross_lag,
            own_lag,
            alpha,
            validation_mae,
            discovery_r2,
            training_rows,
        ) = _hub_chronological_validation(
            invariant_history,
            horizon=horizon,
        )
        center, scale = _standardizer(invariant_history)
        standardized_invariant = (invariant_history - center) / scale
        coefficients = _fit_hub_coefficients(
            standardized_invariant,
            source=source,
            cross_lag=cross_lag,
            own_lag=own_lag,
            alpha=alpha,
        )
        source_extended = standardized_invariant[:, source : source + 1]
        source_coefficients = [coefficients[source]]
        shared_source_future = _recursive_forecast(
            source_extended,
            source_coefficients,
            lag=own_lag,
            diagonal=True,
            horizon=horizon,
        )[:, 0]

        forecasts = []
        for history in (first_history, second_history):
            standardized_history = (history - center) / scale
            standardized_forecast = np.empty(
                (horizon, history.shape[1]),
                dtype=float,
            )
            standardized_forecast[:, source] = shared_source_future
            for destination in range(history.shape[1]):
                if destination == source:
                    continue
                coefficient = coefficients[destination]
                for step in range(horizon):
                    source_time = context - cross_lag + step
                    source_value = (
                        standardized_history[source_time, source]
                        if source_time < context
                        else shared_source_future[source_time - context]
                    )
                    standardized_forecast[step, destination] = (
                        coefficient[0] + coefficient[1] * source_value
                    )
            forecasts.append(center + standardized_forecast * scale)
        common_diagnostics = {
            **diagnostics,
            "schema_version": STRUCTURED_BASELINE_SCHEMA_VERSION,
            "model_id": "ridge_var",
            "fit_status": "ok",
            "fallback_used": False,
            "fallback_reason": None,
            "selected_source_channel": int(source),
            "selected_lag": int(cross_lag),
            "selected_own_lag": int(own_lag),
            "selected_ridge_alpha": float(alpha),
            "validation_normalized_mae": float(validation_mae),
            "blind_edge_mean_holdout_r2": float(discovery_r2),
            "validation_training_rows": int(training_rows),
            "counterfactual_active_prefix_steps": int(
                min(cross_lag, horizon)
            ),
            "counterfactual_tail_driver_forecast_shared": True,
        }
        return (
            StructuredForecast(forecasts[0], common_diagnostics),
            StructuredForecast(forecasts[1], common_diagnostics),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        reason = f"{type(error).__name__}:{error}"
        first_result = _fallback(
            first_history,
            horizon,
            model_id="ridge_var",
            reason=reason,
            diagnostics=diagnostics,
        )
        second_result = _fallback(
            second_history,
            horizon,
            model_id="ridge_var",
            reason=reason,
            diagnostics=diagnostics,
        )
        return first_result, second_result


def _dfm_validation(
    history: np.ndarray,
    *,
    horizon: int,
) -> tuple[int, int, float, float, int]:
    context = len(history)
    validation_rows = max(
        MIN_VALIDATION_ROWS,
        int(round(VALIDATION_FRACTION * context)),
    )
    validation_rows = min(validation_rows, context - MIN_TRAINING_ROWS - 1)
    split = context - validation_rows
    candidates: list[tuple[float, int, int, float, int]] = []
    center, scale = _standardizer(history[:split])
    standardized = (history - center) / scale
    _, _, right = np.linalg.svd(
        standardized[:split],
        full_matrices=False,
    )
    maximum_rank = min(2, history.shape[1] - 1)
    for rank in range(1, maximum_rank + 1):
        loading = right[:rank]
        factor = standardized @ loading.T
        residual = standardized - factor @ loading
        for lag in dfm_lag_candidates(context, horizon):
            effective_rows = split - lag
            if effective_rows < MIN_TRAINING_ROWS:
                continue
            for alpha in DFM_ALPHA_CANDIDATES:
                factor_coefficients = _fit_coefficients(
                    factor[:split],
                    lag=lag,
                    alpha=alpha,
                    diagonal=False,
                )
                residual_coefficients = _fit_coefficients(
                    residual[:split],
                    lag=lag,
                    alpha=alpha,
                    diagonal=True,
                )
                factor_predictions = np.vstack(
                    [
                        _predict_one(
                            factor[:index],
                            factor_coefficients,
                            lag=lag,
                            diagonal=False,
                        )
                        for index in range(split, context)
                    ]
                )
                residual_predictions = np.vstack(
                    [
                        _predict_one(
                            residual[:index],
                            residual_coefficients,
                            lag=lag,
                            diagonal=True,
                        )
                        for index in range(split, context)
                    ]
                )
                predictions = factor_predictions @ loading + residual_predictions
                loss = float(
                    np.mean(np.abs(standardized[split:] - predictions))
                )
                if np.isfinite(loss):
                    candidates.append(
                        (loss, rank, lag, float(alpha), effective_rows)
                    )
    if not candidates:
        raise ValueError("no valid chronological DFM candidate")
    loss, rank, lag, alpha, effective_rows = min(
        candidates,
        key=lambda row: (row[0], row[1], row[2], -row[3]),
    )
    return rank, lag, alpha, loss, effective_rows


def _dfm_forecast(history: np.ndarray, horizon: int) -> StructuredForecast:
    model_id = "dynamic_factor_var"
    diagnostics: dict[str, Any] = {
        "history_only": True,
        "context_length": int(len(history)),
        "horizon": int(horizon),
        "target_dim": int(history.shape[1]),
        "factor_rank_candidates": [1, 2],
        "factor_dynamics": "ridge_var",
        "residual_model": "matched_diagonal_ar",
        "validation_policy": "final_25pct_chronological_one_step",
        "lag_candidates": list(dfm_lag_candidates(len(history), horizon)),
        "ridge_alpha_candidates": list(DFM_ALPHA_CANDIDATES),
        "standardized_forecast_limit": STANDARDIZED_FORECAST_LIMIT,
    }
    try:
        rank, lag, alpha, validation_mae, training_rows = _dfm_validation(
            history,
            horizon=horizon,
        )
        center, scale = _standardizer(history)
        standardized = (history - center) / scale
        _, singular, right = np.linalg.svd(
            standardized,
            full_matrices=False,
        )
        loading = right[:rank]
        factor = standardized @ loading.T
        residual = standardized - factor @ loading
        factor_coefficients = _fit_coefficients(
            factor,
            lag=lag,
            alpha=alpha,
            diagonal=False,
        )
        assert isinstance(factor_coefficients, np.ndarray)
        (
            factor_coefficients,
            factor_radius_before,
            factor_radius_after,
            factor_stability_scale,
        ) = _stable_var_coefficients(
            factor_coefficients,
            dimension=rank,
            lag=lag,
        )
        residual_coefficients = _fit_coefficients(
            residual,
            lag=lag,
            alpha=alpha,
            diagonal=True,
        )
        assert isinstance(residual_coefficients, list)
        (
            residual_coefficients,
            residual_radius_before,
            residual_radius_after,
            residual_stability_scale,
        ) = _stable_diagonal_coefficients(
            residual_coefficients,
            lag=lag,
        )
        factor_forecast = _recursive_forecast(
            factor,
            factor_coefficients,
            lag=lag,
            diagonal=False,
            horizon=horizon,
        )
        residual_forecast = _recursive_forecast(
            residual,
            residual_coefficients,
            lag=lag,
            diagonal=True,
            horizon=horizon,
        )
        standardized_forecast = factor_forecast @ loading + residual_forecast
        forecast = center + standardized_forecast * scale
        total_energy = float(np.sum(singular * singular))
        failure = _validate_forecast(
            forecast,
            center=center,
            scale=scale,
        )
        diagnostics.update(
            {
                "selected_lag": int(lag),
                "selected_factor_rank": int(rank),
                "selected_ridge_alpha": float(alpha),
                "validation_normalized_mae": float(validation_mae),
                "validation_training_rows": int(training_rows),
                "history_factor_share": float(
                    np.sum(singular[:rank] ** 2) / max(total_energy, 1e-12)
                ),
                "factor_var_spectral_radius_before": factor_radius_before,
                "factor_var_spectral_radius_after": factor_radius_after,
                "factor_var_stability_scale": factor_stability_scale,
                "residual_ar_max_spectral_radius_before": (
                    residual_radius_before
                ),
                "residual_ar_max_spectral_radius_after": (
                    residual_radius_after
                ),
                "residual_ar_min_stability_scale": residual_stability_scale,
            }
        )
        if failure is not None:
            return _fallback(
                history,
                horizon,
                model_id=model_id,
                reason=failure,
                diagnostics=diagnostics,
            )
        return StructuredForecast(
            forecast=forecast,
            diagnostics={
                **diagnostics,
                "schema_version": STRUCTURED_BASELINE_SCHEMA_VERSION,
                "model_id": model_id,
                "fit_status": "ok",
                "fallback_used": False,
                "fallback_reason": None,
            },
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return _fallback(
            history,
            horizon,
            model_id=model_id,
            reason=f"{type(error).__name__}:{error}",
            diagnostics=diagnostics,
        )


def _pair_invariant_prefix_stop(
    first_history: np.ndarray,
    second_history: np.ndarray,
) -> tuple[int, list[int]]:
    if first_history.shape != second_history.shape:
        raise ValueError("counterfactual pair histories must have equal shape")
    reference_scale = max(
        float(np.max(np.abs(first_history))),
        float(np.max(np.abs(second_history))),
        1.0,
    )
    tolerance = 1e-9 * reference_scale
    absolute_difference = np.abs(second_history - first_history)
    invariant_channels = np.flatnonzero(
        np.max(absolute_difference, axis=0) <= tolerance
    ).astype(int)
    changed_rows = np.flatnonzero(
        np.max(absolute_difference, axis=1) > tolerance
    )
    if not changed_rows.size:
        raise ValueError("counterfactual pair has no changed history")
    invariant_stop = int(changed_rows[0])
    if invariant_stop < MIN_TRAINING_ROWS * 2:
        raise ValueError("counterfactual invariant prefix is too short")
    if not invariant_channels.size:
        raise ValueError("counterfactual pair has no invariant channel")
    return invariant_stop, invariant_channels.tolist()


def forecast_common_counterfactual_pair(
    first_sample: dict[str, Any],
    second_sample: dict[str, Any],
) -> tuple[StructuredForecast, StructuredForecast]:
    """Blind shared-fit DFM forecast for a common-factor strict pair.

    The common invariant prefix determines scaling, loadings, dynamics, and
    residual model. Each member's changed history tail only updates its latent
    factor state. A shared residual forecast prevents member-specific
    idiosyncratic fits from cancelling the latent-state effect under audit.
    """

    model_id = "dynamic_factor_var"
    first_target = np.asarray(first_sample["target"], dtype=float)
    second_target = np.asarray(second_sample["target"], dtype=float)
    context = int(first_sample["context_length"])
    horizon = int(first_sample["horizon"])
    if (
        int(second_sample["context_length"]) != context
        or int(second_sample["horizon"]) != horizon
    ):
        raise ValueError("counterfactual pair view shapes do not match")
    first_history = first_target[:context]
    second_history = second_target[:context]
    diagnostics: dict[str, Any] = {
        "history_only": True,
        "context_length": context,
        "horizon": horizon,
        "target_dim": int(first_history.shape[1]),
        "model_structure": "blind_shared_fit_dynamic_factor_var",
        "generator_metadata_used_for_fitting": False,
        "paired_members_share_fit": True,
        "factor_rank_candidates": [1, 2],
        "factor_dynamics": "ridge_var",
        "residual_model": "shared_matched_diagonal_ar",
        "validation_policy": "final_25pct_chronological_one_step",
        "ridge_alpha_candidates": list(DFM_ALPHA_CANDIDATES),
        "standardized_forecast_limit": STANDARDIZED_FORECAST_LIMIT,
    }
    try:
        invariant_stop, invariant_channels = _pair_invariant_prefix_stop(
            first_history,
            second_history,
        )
        shared_prefix = first_history[:invariant_stop]
        rank, lag, alpha, validation_mae, training_rows = _dfm_validation(
            shared_prefix,
            horizon=horizon,
        )
        center, scale = _standardizer(shared_prefix)
        standardized_prefix = (shared_prefix - center) / scale
        _, singular, right = np.linalg.svd(
            standardized_prefix,
            full_matrices=False,
        )
        loading = right[:rank]
        prefix_factor = standardized_prefix @ loading.T
        prefix_residual = standardized_prefix - prefix_factor @ loading

        factor_coefficients = _fit_coefficients(
            prefix_factor,
            lag=lag,
            alpha=alpha,
            diagonal=False,
        )
        assert isinstance(factor_coefficients, np.ndarray)
        (
            factor_coefficients,
            factor_radius_before,
            factor_radius_after,
            factor_stability_scale,
        ) = _stable_var_coefficients(
            factor_coefficients,
            dimension=rank,
            lag=lag,
        )
        residual_coefficients = _fit_coefficients(
            prefix_residual,
            lag=lag,
            alpha=alpha,
            diagonal=True,
        )
        assert isinstance(residual_coefficients, list)
        (
            residual_coefficients,
            residual_radius_before,
            residual_radius_after,
            residual_stability_scale,
        ) = _stable_diagonal_coefficients(
            residual_coefficients,
            lag=lag,
        )

        standardized_histories = [
            (history - center) / scale
            for history in (first_history, second_history)
        ]
        state_channels = [
            index
            for index in range(first_history.shape[1])
            if index not in invariant_channels
        ]
        if len(state_channels) < rank:
            raise ValueError(
                "too few changed channels to filter the shared state"
            )
        state_loading = loading[:, state_channels]
        state_loading_gram = state_loading @ state_loading.T
        state_loading_inverse = np.linalg.pinv(state_loading_gram)
        factor_histories = [
            (
                history[:, state_channels]
                @ state_loading.T
                @ state_loading_inverse
            )
            for history in standardized_histories
        ]
        residual_histories = [
            history - factor @ loading
            for history, factor in zip(
                standardized_histories,
                factor_histories,
                strict=True,
            )
        ]
        shared_residual_history = np.mean(
            np.stack(residual_histories),
            axis=0,
        )
        shared_residual_forecast = _recursive_forecast(
            shared_residual_history,
            residual_coefficients,
            lag=lag,
            diagonal=True,
            horizon=horizon,
        )
        forecasts: list[np.ndarray] = []
        for factor_history in factor_histories:
            factor_forecast = _recursive_forecast(
                factor_history,
                factor_coefficients,
                lag=lag,
                diagonal=False,
                horizon=horizon,
            )
            standardized_forecast = (
                factor_forecast @ loading + shared_residual_forecast
            )
            forecast = center + standardized_forecast * scale
            failure = _validate_forecast(
                forecast,
                center=center,
                scale=scale,
            )
            if failure is not None:
                raise FloatingPointError(failure)
            forecasts.append(forecast)

        total_energy = float(np.sum(singular * singular))
        common_diagnostics = {
            **diagnostics,
            "schema_version": STRUCTURED_BASELINE_SCHEMA_VERSION,
            "model_id": model_id,
            "fit_status": "ok",
            "fallback_used": False,
            "fallback_reason": None,
            "pair_invariant_history_stop": invariant_stop,
            "pair_invariant_channel_indices": invariant_channels,
            "state_filter_channel_indices": state_channels,
            "selected_lag": int(lag),
            "selected_factor_rank": int(rank),
            "selected_ridge_alpha": float(alpha),
            "validation_normalized_mae": float(validation_mae),
            "validation_training_rows": int(training_rows),
            "history_factor_share": float(
                np.sum(singular[:rank] ** 2) / max(total_energy, 1e-12)
            ),
            "factor_var_spectral_radius_before": factor_radius_before,
            "factor_var_spectral_radius_after": factor_radius_after,
            "factor_var_stability_scale": factor_stability_scale,
            "residual_ar_max_spectral_radius_before": residual_radius_before,
            "residual_ar_max_spectral_radius_after": residual_radius_after,
            "residual_ar_min_stability_scale": residual_stability_scale,
            "counterfactual_residual_forecast_shared": True,
        }
        return (
            StructuredForecast(forecasts[0], common_diagnostics),
            StructuredForecast(forecasts[1], common_diagnostics),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        reason = f"{type(error).__name__}:{error}"
        first_result = _fallback(
            first_history,
            horizon,
            model_id=model_id,
            reason=reason,
            diagnostics=diagnostics,
        )
        second_result = _fallback(
            second_history,
            horizon,
            model_id=model_id,
            reason=reason,
            diagnostics=diagnostics,
        )
        return first_result, second_result


def forecast(
    sample: dict[str, Any],
    model_id: str,
) -> StructuredForecast:
    if not is_structured_sample(sample):
        raise ValueError("sample is outside the structured positive-control scope")
    capability = str(sample["capability_id"])
    if model_id not in baseline_ids_for(capability):
        raise ValueError(
            f"{model_id} is unsupported for structured capability {capability}"
        )
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    horizon = int(sample["horizon"])
    history = target[:context]
    if model_id == "diagonal_ar":
        return _ar_or_var_forecast(
            history,
            horizon,
            model_id=model_id,
            diagonal=True,
        )
    if model_id == "ridge_var":
        return _hub_var_forecast(history, horizon)
    return _dfm_forecast(history, horizon)
