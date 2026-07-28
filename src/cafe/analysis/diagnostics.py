from __future__ import annotations

from typing import Any

import numpy as np

from cafe.analysis.metrics import compute_sample_metrics


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel().copy()
    right = np.asarray(right, dtype=float).ravel().copy()
    if left.size < 3 or left.size != right.size:
        return 0.0
    left -= float(np.mean(left))
    right -= float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def top_factor_share(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[1] < 2:
        return 1.0
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    variance = singular * singular
    total = float(np.sum(variance))
    return float(variance[0] / total) if total > 1e-12 else 0.0


def leading_factor_decomposition(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    try:
        left, singular, right = np.linalg.svd(
            centered,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return (
            np.zeros(values.shape[0]),
            np.zeros(values.shape[1]),
            np.zeros_like(values),
            0.0,
        )
    if not singular.size or float(np.sum(singular * singular)) <= 1e-12:
        return (
            np.zeros(values.shape[0]),
            np.zeros(values.shape[1]),
            np.zeros_like(values),
            0.0,
        )
    score = left[:, 0] * singular[0]
    loading = right[0]
    common = score[:, None] * loading[None, :]
    share = float(
        singular[0] ** 2 / np.sum(singular * singular)
    )
    return score, loading, common, share


def common_factor_recovery_metrics(
    truth: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    truth_score, truth_loading, truth_common, truth_share = (
        leading_factor_decomposition(truth)
    )
    forecast_score, forecast_loading, forecast_common, forecast_share = (
        leading_factor_decomposition(forecast)
    )
    loading_dot = float(np.dot(truth_loading, forecast_loading))
    sign = 1.0 if loading_dot >= 0 else -1.0
    forecast_score = sign * forecast_score
    forecast_loading = sign * forecast_loading
    forecast_common = forecast_score[:, None] * forecast_loading[None, :]
    truth_score_std = float(np.std(truth_score))
    truth_scale = float(np.mean(np.std(truth, axis=0)))
    return {
        "truth_factor_share": truth_share,
        "forecast_factor_share": forecast_share,
        "factor_share_abs_error": abs(truth_share - forecast_share),
        "factor_loading_cosine": abs(loading_dot),
        "factor_trajectory_correlation": safe_corr(
            truth_score,
            forecast_score,
        ),
        "factor_score_nrmse": float(
            np.sqrt(np.mean((truth_score - forecast_score) ** 2))
            / max(truth_score_std, 1e-12)
        ),
        "common_component_nmae": float(
            np.mean(np.abs(truth_common - forecast_common))
            / max(truth_scale, 1e-12)
        ),
    }


def child_heterogeneity(values: np.ndarray) -> float:
    if values.shape[1] < 3:
        return 0.0
    return float(np.mean(np.std(values[:, 1:], axis=1)))


def hierarchy_recovery_metrics(
    truth: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    forecast_children_sum = np.sum(forecast[:, 1:], axis=1)
    residual = forecast[:, 0] - forecast_children_sum
    parent_scale = float(np.std(truth[:, 0]))
    truth_contrast = truth[:, 1:] - np.mean(
        truth[:, 1:],
        axis=1,
        keepdims=True,
    )
    forecast_contrast = forecast[:, 1:] - np.mean(
        forecast[:, 1:],
        axis=1,
        keepdims=True,
    )
    contrast_scale = float(np.mean(np.std(truth_contrast, axis=0)))
    aggregation_ratio = float(
        np.std(forecast_children_sum)
        / max(float(np.std(forecast[:, 0])), 1e-12)
    )
    truth_heterogeneity = child_heterogeneity(truth)
    forecast_heterogeneity = child_heterogeneity(forecast)
    return {
        "coherence_mae": float(np.mean(np.abs(residual))),
        "coherence_nmae": float(
            np.mean(np.abs(residual)) / max(parent_scale, 1e-12)
        ),
        "aggregation_correlation": safe_corr(
            forecast[:, 0],
            forecast_children_sum,
        ),
        "aggregation_scale_abs_log_error": abs(
            math.log(max(aggregation_ratio, 1e-12))
        ),
        "truth_child_heterogeneity": truth_heterogeneity,
        "forecast_child_heterogeneity": forecast_heterogeneity,
        "child_heterogeneity_abs_error": abs(
            truth_heterogeneity - forecast_heterogeneity
        ),
        "child_contrast_correlation": safe_corr(
            truth_contrast,
            forecast_contrast,
        ),
        "child_contrast_nmae": float(
            np.mean(np.abs(truth_contrast - forecast_contrast))
            / max(contrast_scale, 1e-12)
        ),
    }


def cross_series_recovery_metrics(
    sample: dict[str, Any],
    truth: np.ndarray,
    forecast: np.ndarray,
    history: np.ndarray,
) -> dict[str, float]:
    metadata = sample.get("generation_metadata", {})
    responders = [
        int(index)
        for index in metadata.get(
            "responder_indices",
            list(range(1, truth.shape[1])),
        )
        if 0 <= int(index) < truth.shape[1]
    ]
    if not responders:
        responders = list(range(truth.shape[1]))
    covered = min(
        truth.shape[0],
        max(
            1,
            int(
                metadata.get(
                    "history_covered_forecast_steps",
                    truth.shape[0],
                )
            ),
        ),
    )
    responder_truth = truth[:, responders]
    responder_forecast = forecast[:, responders]
    responder_scale = float(
        np.mean(np.std(history[:, responders], axis=0))
    )
    covered_truth = responder_truth[:covered]
    covered_forecast = responder_forecast[:covered]
    return {
        "responder_mae": float(
            np.mean(np.abs(responder_truth - responder_forecast))
        ),
        "responder_normalized_mae": float(
            np.mean(np.abs(responder_truth - responder_forecast))
            / max(responder_scale, 1e-12)
        ),
        "driver_covered_responder_mae": float(
            np.mean(np.abs(covered_truth - covered_forecast))
        ),
        "driver_covered_responder_correlation": safe_corr(
            covered_truth,
            covered_forecast,
        ),
        "history_covered_forecast_steps": float(covered),
    }


def covariate_future_corr(
    forecast: np.ndarray,
    sample: dict[str, Any],
) -> float:
    covariates = sample.get("covariates")
    if covariates is None:
        return 0.0
    context = int(sample["context_length"])
    future = np.asarray(covariates, dtype=float)[context:]
    scores = [
        abs(safe_corr(future[:, covariate], forecast[:, target]))
        for covariate in range(future.shape[1])
        for target in range(forecast.shape[1])
    ]
    return float(np.mean(scores)) if scores else 0.0


def _mean_relative_coefficient_error(
    truth: np.ndarray,
    forecast: np.ndarray,
    *,
    coefficient_index: int,
) -> float:
    time = np.linspace(-1.0, 1.0, truth.shape[0])
    errors = []
    for target in range(truth.shape[1]):
        truth_coefficients = np.polyfit(time, truth[:, target], deg=2)
        forecast_coefficients = np.polyfit(
            time,
            forecast[:, target],
            deg=2,
        )
        truth_value = float(truth_coefficients[coefficient_index])
        forecast_value = float(forecast_coefficients[coefficient_index])
        errors.append(
            abs(forecast_value - truth_value)
            / max(abs(truth_value), 1e-8)
        )
    return float(np.mean(errors))


def trend_recovery_metrics(
    truth: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    time = np.linspace(-1.0, 1.0, truth.shape[0])
    centered_quadratic_basis = time * time - float(np.mean(time * time))
    truth_slopes = np.asarray(
        [np.polyfit(time, truth[:, target], deg=1)[0] for target in range(truth.shape[1])]
    )
    forecast_slopes = np.asarray(
        [
            np.polyfit(time, forecast[:, target], deg=1)[0]
            for target in range(forecast.shape[1])
        ]
    )
    truth_curvatures = np.asarray(
        [
            np.polyfit(time, truth[:, target], deg=2)[0]
            for target in range(truth.shape[1])
        ]
    )
    forecast_curvatures = np.asarray(
        [
            np.polyfit(time, forecast[:, target], deg=2)[0]
            for target in range(forecast.shape[1])
        ]
    )
    truth_curvature_component = (
        centered_quadratic_basis[:, None] * truth_curvatures[None, :]
    )
    forecast_curvature_component = (
        centered_quadratic_basis[:, None] * forecast_curvatures[None, :]
    )
    truth_curvature_rms = float(
        np.sqrt(np.mean(truth_curvature_component**2))
    )
    forecast_curvature_rms = float(
        np.sqrt(np.mean(forecast_curvature_component**2))
    )
    return {
        "trend_slope_relative_abs_error": float(
            np.mean(
                np.abs(forecast_slopes - truth_slopes)
                / np.maximum(np.abs(truth_slopes), 1e-8)
            )
        ),
        "trend_curvature_relative_abs_error": (
            _mean_relative_coefficient_error(
                truth,
                forecast,
                coefficient_index=0,
            )
        ),
        "trend_direction_accuracy": float(
            np.mean(np.sign(truth_slopes) == np.sign(forecast_slopes))
        ),
        "trend_curvature_component_nrmse": float(
            np.sqrt(
                np.mean(
                    (
                        forecast_curvature_component
                        - truth_curvature_component
                    )
                    ** 2
                )
            )
            / max(truth_curvature_rms, 1e-8)
        ),
        "trend_curvature_sign_accuracy": float(
            np.mean(
                np.sign(truth_curvatures)
                == np.sign(forecast_curvatures)
            )
        ),
        "trend_curvature_magnitude_ratio": (
            forecast_curvature_rms / max(truth_curvature_rms, 1e-8)
        ),
    }


def _detrend(values: np.ndarray) -> np.ndarray:
    time = np.linspace(-1.0, 1.0, values.shape[0])
    design = np.column_stack([np.ones(values.shape[0]), time])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def multi_seasonal_recovery_metrics(
    sample: dict[str, Any],
    truth: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    periods = [
        float(value)
        for value in sample.get("generation_metadata", {}).get("periods", [])
        if float(value) > 1.0
    ]
    if not periods:
        return {}
    time = np.arange(truth.shape[0], dtype=float)
    truth_centered = _detrend(truth)
    forecast_centered = _detrend(forecast)
    truth_coefficients = []
    forecast_coefficients = []
    for period in periods:
        basis = np.exp(-2j * np.pi * time / period)[:, None]
        truth_coefficients.append(
            np.sum(truth_centered * basis, axis=0)
            / max(truth.shape[0], 1)
        )
        forecast_coefficients.append(
            np.sum(forecast_centered * basis, axis=0)
            / max(forecast.shape[0], 1)
        )
    truth_vector = np.asarray(truth_coefficients).ravel()
    forecast_vector = np.asarray(forecast_coefficients).ravel()
    denominator = float(
        np.linalg.norm(truth_vector) * np.linalg.norm(forecast_vector)
    )
    complex_alignment = (
        float(
            np.real(
                np.vdot(truth_vector, forecast_vector)
            )
            / denominator
        )
        if denominator > 1e-12
        else 0.0
    )
    return {
        "seasonal_spectral_amplitude_relative_error": float(
            np.sum(np.abs(np.abs(forecast_vector) - np.abs(truth_vector)))
            / max(float(np.sum(np.abs(truth_vector))), 1e-12)
        ),
        "seasonal_spectral_phase_amplitude_alignment": complex_alignment,
    }


def _analytic_signal(values: np.ndarray) -> np.ndarray:
    """Return the Hilbert analytic signal using only NumPy FFT primitives."""

    length = values.shape[0]
    spectrum = np.fft.fft(values, axis=0)
    multiplier = np.zeros(length, dtype=float)
    if length % 2 == 0:
        multiplier[0] = 1.0
        multiplier[length // 2] = 1.0
        multiplier[1 : length // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (length + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier[:, None], axis=0)


def time_varying_seasonality_recovery_metrics(
    sample: dict[str, Any],
    truth: np.ndarray,
    forecast: np.ndarray,
    history: np.ndarray,
) -> dict[str, float]:
    metadata = sample.get("generation_metadata", {})
    primary_period = max(
        float(
            metadata.get(
                "primary_period",
                sample.get("feature_period", sample.get("season_length", 24)),
            )
        ),
        2.0,
    )
    tail_length = min(
        history.shape[0],
        max(int(round(6.0 * primary_period)), 4 * truth.shape[0]),
    )
    truth_path = np.vstack([history[-tail_length:], truth])
    forecast_path = np.vstack([history[-tail_length:], forecast])
    truth_analytic = _analytic_signal(truth_path)
    forecast_analytic = _analytic_signal(forecast_path)
    truth_amplitude = np.abs(truth_analytic)[-truth.shape[0] :]
    forecast_amplitude = np.abs(forecast_analytic)[-truth.shape[0] :]
    truth_phase = np.unwrap(np.angle(truth_analytic), axis=0)[-truth.shape[0] :]
    forecast_phase = np.unwrap(np.angle(forecast_analytic), axis=0)[
        -truth.shape[0] :
    ]
    trim = max(2, min(6, truth.shape[0] // 8))
    valid = slice(0, truth.shape[0] - trim)
    truth_frequency = np.diff(truth_phase, axis=0)
    forecast_frequency = np.diff(forecast_phase, axis=0)
    base_frequency = 2.0 * np.pi / primary_period
    return {
        "modulation_envelope_nmae": float(
            np.mean(
                np.abs(
                    forecast_amplitude[valid] - truth_amplitude[valid]
                )
            )
            / max(float(np.mean(truth_amplitude[valid])), 1e-12)
        ),
        "modulation_phase_alignment": float(
            np.mean(
                np.cos(forecast_phase[valid] - truth_phase[valid])
            )
        ),
        "instantaneous_frequency_nmae": float(
            np.mean(
                np.abs(
                    forecast_frequency[: truth.shape[0] - trim - 1]
                    - truth_frequency[: truth.shape[0] - trim - 1]
                )
            )
            / max(base_frequency, 1e-12)
        ),
    }


def regime_recovery_metrics(
    sample: dict[str, Any],
    target: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    context = int(sample["context_length"])
    horizon = int(sample["horizon"])
    boundaries = [
        int(value)
        for value in sample.get("generation_metadata", {}).get(
            "cut_points",
            [],
        )
        if context <= int(value) < context + horizon
    ]
    if not boundaries:
        return {}
    forecast_path = np.vstack([target[:context], forecast])
    truth_jumps = np.asarray(
        [target[index] - target[index - 1] for index in boundaries]
    )
    forecast_jumps = np.asarray(
        [
            forecast_path[index] - forecast_path[index - 1]
            for index in boundaries
        ]
    )
    return {
        "regime_boundary_count": float(len(boundaries)),
        "regime_jump_nmae": float(
            np.mean(np.abs(forecast_jumps - truth_jumps))
            / max(float(np.mean(np.abs(truth_jumps))), 1e-12)
        ),
        "regime_jump_amplitude_ratio": float(
            np.mean(np.abs(forecast_jumps))
            / max(float(np.mean(np.abs(truth_jumps))), 1e-12)
        ),
        "regime_jump_sign_accuracy": float(
            np.mean(np.sign(forecast_jumps) == np.sign(truth_jumps))
        ),
    }


def nonlinear_recovery_metrics(
    sample: dict[str, Any],
    target: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    metadata = sample.get("generation_metadata", {})
    context = int(sample["context_length"])
    lag = int(metadata.get("nonlinear_lag", 1))
    seasonal_lag = int(
        metadata.get(
            "seasonal_lag",
            sample.get("feature_period", sample.get("season_length", 24)),
        )
    )
    transform = str(
        metadata.get(
            "nonlinear_transform",
            "signed_rational_quadratic",
        )
    )
    persistence = float(
        metadata.get(
            "persistence_weight",
            0.58
            if transform
            in {"shifted_tanh", "signed_rational_quadratic"}
            else 0.52,
        )
    )
    seasonal_weight = float(
        metadata.get(
            "seasonal_weight",
            0.10
            if transform
            in {"shifted_tanh", "signed_rational_quadratic"}
            else 0.14,
        )
    )
    if transform == "signed_rational_quadratic":
        def response(values: np.ndarray) -> np.ndarray:
            return values * np.abs(values) / (1.0 + values * values)

    elif transform == "signed_softsign_quadratic":
        def response(values: np.ndarray) -> np.ndarray:
            return values * np.abs(values) / (1.0 + np.abs(values))

    elif transform == "shifted_tanh":
        response_slope = float(
            metadata.get("nonlinear_response_slope", 1.35)
        )
        response_shift = float(
            metadata.get("nonlinear_response_shift", 0.55)
        )

        def response(values: np.ndarray) -> np.ndarray:
            return (
                np.tanh(response_slope * values + response_shift)
                - np.tanh(response_shift)
            )

    else:
        def response(values: np.ndarray) -> np.ndarray:
            return values / (1.0 + values * values)

    forecast_path = np.vstack([target[:context], forecast])
    indexes = np.arange(context, context + forecast.shape[0])
    truth_residual = (
        target[indexes]
        - persistence * target[indexes - 1]
        - seasonal_weight * target[indexes - seasonal_lag]
    )
    forecast_residual = (
        forecast_path[indexes]
        - persistence * forecast_path[indexes - 1]
        - seasonal_weight * forecast_path[indexes - seasonal_lag]
    )
    delayed_indexes = indexes[indexes >= context + lag]
    truth_response = response(target[delayed_indexes - lag])
    forecast_response = response(
        forecast_path[delayed_indexes - lag]
    )
    return {
        "nonlinear_recurrence_residual_nrmse": float(
            np.sqrt(np.mean((forecast_residual - truth_residual) ** 2))
            / max(float(np.std(truth_residual)), 1e-12)
        ),
        "nonlinear_delayed_response_nrmse": float(
            np.sqrt(np.mean((forecast_response - truth_response) ** 2))
            / max(float(np.std(truth_response)), 1e-12)
        ),
        "nonlinear_delayed_response_correlation": safe_corr(
            truth_response,
            forecast_response,
        ),
    }


def intermittent_recovery_metrics(
    sample: dict[str, Any],
    target: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, float]:
    metadata = sample.get("generation_metadata", {})
    context = int(sample["context_length"])
    horizon = int(sample["horizon"])
    centers = [
        int(value)
        for value in metadata.get("pulse_centers", [])
        if context <= int(value) < context + horizon
    ]
    if not centers:
        return {}
    width = max(float(metadata.get("pulse_width", 1.0)), 1.0)
    radius = max(2, int(math.ceil(2.0 * width)))
    timing_errors = []
    amplitude_errors = []
    mask = np.zeros(horizon, dtype=bool)
    for center in centers:
        local_center = center - context
        start = max(0, local_center - radius)
        stop = min(horizon, local_center + radius + 1)
        mask[start:stop] = True
        for target_index in range(target.shape[1]):
            truth_window = target[context + start : context + stop, target_index]
            forecast_window = forecast[start:stop, target_index]
            truth_peak = int(np.argmax(truth_window))
            forecast_peak = int(np.argmax(forecast_window))
            timing_errors.append(abs(forecast_peak - truth_peak) / width)
            amplitude_errors.append(
                abs(
                    float(forecast_window[forecast_peak])
                    - float(truth_window[truth_peak])
                )
            )
    history_scale = float(
        np.mean(np.std(target[:context], axis=0))
    )
    event_truth = target[context:][mask]
    event_forecast = forecast[mask]
    background_truth = target[context:][~mask]
    background_forecast = forecast[~mask]
    metrics = {
        "event_peak_timing_widths": float(np.mean(timing_errors)),
        "event_peak_amplitude_nmae": float(
            np.mean(amplitude_errors) / max(history_scale, 1e-12)
        ),
        "event_window_nmae": float(
            np.mean(np.abs(event_forecast - event_truth))
            / max(history_scale, 1e-12)
        ),
    }
    if background_truth.size:
        metrics["background_window_nmae"] = float(
            np.mean(np.abs(background_forecast - background_truth))
            / max(history_scale, 1e-12)
        )
    return metrics


def prediction_metrics(
    sample: dict[str, Any],
    forecast: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    truth = target[context:]
    history = target[:context]
    metrics = compute_sample_metrics(
        truth.tolist(),
        forecast.tolist(),
        history.tolist(),
        seasonal_period=int(
            sample.get("mase_period", sample.get("season_length", 1))
        ),
    )
    history_scale = float(np.mean(np.std(history, axis=0)))
    truth_std = float(np.mean(np.std(truth, axis=0)))
    forecast_std = float(np.mean(np.std(forecast, axis=0)))
    per_target_corr = [
        safe_corr(truth[:, index], forecast[:, index])
        for index in range(truth.shape[1])
    ]
    output = {
        name: float(value)
        for name, value in metrics.items()
        if value is not None and np.isfinite(value)
    }
    output.update(
        {
            "normalized_mae_history_std": float(
                np.mean(np.abs(truth - forecast)) / max(history_scale, 1e-12)
            ),
            "forecast_to_truth_std_ratio": forecast_std / max(truth_std, 1e-12),
            "future_curve_correlation": float(np.mean(per_target_corr)),
            "flat_forecast": float(
                forecast_std < max(0.05 * truth_std, 1e-4)
            ),
            "truth_future_std": truth_std,
            "forecast_future_std": forecast_std,
        }
    )
    capability_id = sample["capability_id"]
    if capability_id == "trend":
        output.update(trend_recovery_metrics(truth, forecast))
    elif capability_id == "multi_seasonal":
        output.update(
            multi_seasonal_recovery_metrics(sample, truth, forecast)
        )
    elif capability_id == "time_varying_seasonality":
        output.update(
            time_varying_seasonality_recovery_metrics(
                sample,
                truth,
                forecast,
                history,
            )
        )
    elif capability_id == "regime_switching":
        output.update(regime_recovery_metrics(sample, target, forecast))
    elif capability_id == "nonlinear_persistence":
        output.update(
            nonlinear_recovery_metrics(sample, target, forecast)
        )
    elif capability_id == "predictable_intermittency":
        output.update(
            intermittent_recovery_metrics(sample, target, forecast)
        )
    elif capability_id == "common_factor":
        output.update(
            common_factor_recovery_metrics(truth, forecast)
        )
        protected = int(
            sample["generation_metadata"]["protected_target_index"]
        )
        protected_mae = float(
            np.mean(
                np.abs(
                    truth[:, protected] - forecast[:, protected]
                )
            )
        )
        protected_scale = float(np.std(history[:, protected]))
        output.update(
            {
                "protected_target_mae": protected_mae,
                "protected_target_nmae": (
                    protected_mae / max(protected_scale, 1e-12)
                ),
            }
        )
    elif capability_id == "hierarchical_coherence":
        output.update(
            hierarchy_recovery_metrics(truth, forecast)
        )
    elif capability_id == "cross_series_dependence":
        output.update(
            cross_series_recovery_metrics(
                sample,
                truth,
                forecast,
                history,
            )
        )
    elif capability_id == "covariate_response":
        output["truth_future_covariate_corr"] = covariate_future_corr(truth, sample)
        output["forecast_future_covariate_corr"] = covariate_future_corr(
            forecast,
            sample,
        )
        output["future_covariate_corr_abs_error"] = abs(
            output["truth_future_covariate_corr"]
            - output["forecast_future_covariate_corr"]
        )
    return output
