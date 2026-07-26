from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from app.services.synthetic_generator_conditioning import GeneratorConditioning


GENERATOR_VERSION = "capts-paper-v8-family-calibrated-v6"
FamilyRole = Literal["primary", "secondary"]

BACKGROUND_PERIOD_RANGE = (8.0, 168.0)
SHORTEST_SUPPORTED_CONTEXT = 96
TREND_LOCAL_EVIDENCE_WINDOW = 96
TREND_DESIGN_HORIZON = 48
MULTI_SEASONAL_PRIMARY_PERIOD_RANGE = (8.0, 32.0)
MULTI_SEASONAL_COMPONENT_PERIOD_RANGE = (4.0, 48.0)
TIME_VARYING_CARRIER_PERIOD_RANGE = (8.0, 32.0)
TIME_VARYING_MODULATION_PERIOD_RANGE = (24.0, 96.0)
INTERMITTENT_EVENT_PERIOD_RANGE = (8, 126)
REGIME_DWELL_RANGE = (12, 84)
NONLINEAR_SEASONAL_LAG_RANGE = (4, 48)
NONLINEAR_LAG_RANGE = (2, 32)
COVARIATE_EVENT_WIDTH_RANGE = (2, 6)

COMMON_FACTOR_MIN_EXCESS_PCA_SHARE = 0.02
COMMON_FACTOR_MAX_EFFECT_NRMSE = 0.35
COMMON_FACTOR_MIN_EFFECT_CORRELATION = 0.95
COMMON_FACTOR_EFFECT_AMPLITUDE_RANGE = (0.70, 1.30)
CROSS_SERIES_MIN_INCREMENTAL_HOLDOUT_GAIN = 0.0025
CROSS_SERIES_MAX_EFFECT_NRMSE = 0.15
CROSS_SERIES_MIN_EFFECT_CORRELATION = 0.95
CROSS_SERIES_EFFECT_AMPLITUDE_RANGE = (0.80, 1.20)


PRIMARY_FAMILY_BY_CAPABILITY: dict[str, str] = {
    "trend": "sample_specific_polynomial",
    "multi_seasonal": "sample_specific_fourier_basis",
    "time_varying_seasonality": "modulated_oscillator",
    "regime_switching": "deterministic_duration_motif",
    "nonlinear_persistence": "centered_rational_quadratic_recurrence",
    "predictable_intermittency": "deterministic_gaussian_event_clock",
    "common_factor": "dense_dynamic_factor_with_shared_state_evidence",
    "hierarchical_coherence": "aggregate_contrast_linear_state_space",
    "cross_series_dependence": "dense_delayed_linear_scm",
    "covariate_response": "known_future_linear_response",
}

SECONDARY_FAMILY_BY_CAPABILITY: dict[str, str] = {
    "trend": "sample_specific_cubic_trend",
    "multi_seasonal": "periodic_spline_motif",
    "time_varying_seasonality": "chirped_triangular_modulation",
    "regime_switching": "thresholded_quasiperiodic_oscillator_regime",
    "nonlinear_persistence": "centered_tanh_quadratic_recurrence",
    "predictable_intermittency": "deterministic_raised_cosine_event_clock",
    "common_factor": "dense_spline_factor_with_shared_state_evidence",
    "hierarchical_coherence": "aggregate_contrast_periodic_spline",
    "cross_series_dependence": "dense_delayed_nonlinear_scm",
    "covariate_response": "known_future_distributed_nonlinear_response",
}


def add_observation_noise_to_history(
    clean_target: np.ndarray,
    *,
    context_length: int,
    noise_ratio: float,
    rng: np.random.Generator,
    noise_scale_by_target: np.ndarray | None = None,
    noise_scale_source: str | None = None,
    preserve_additive_hierarchy: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Corrupt only the visible history while retaining a clean future target.

    The returned array is deliberately shaped like an ordinary benchmark
    target: callers expose its noisy prefix to the model and score its untouched
    suffix.  Callers may provide a positive per-target scale so that the noise
    dose is tied to an evaluation scale rather than the variance of a
    nonstationary level.  For additive hierarchies noise is sampled on children
    and the parent prefix is recomputed, avoiding a second coherence
    intervention.
    """

    values = np.asarray(clean_target, dtype=float)
    if values.ndim != 2:
        raise ValueError("clean_target must be a two-dimensional array")
    if not 1 <= context_length < values.shape[0]:
        raise ValueError("context_length must split history and future")
    if not math.isfinite(noise_ratio) or noise_ratio < 0:
        raise ValueError("noise_ratio must be finite and non-negative")
    result = values.copy()
    history = values[:context_length]
    if noise_scale_by_target is None:
        requested_scales = np.std(history, axis=0)
        requested_scales = np.where(
            requested_scales > 1e-12,
            requested_scales,
            1.0,
        )
        resolved_scale_source = (
            noise_scale_source or "clean_history_standard_deviation"
        )
    else:
        requested_scales = np.asarray(
            noise_scale_by_target,
            dtype=float,
        )
        if requested_scales.shape != (values.shape[1],):
            raise ValueError(
                "noise_scale_by_target must have one value per target"
            )
        if (
            not np.isfinite(requested_scales).all()
            or np.any(requested_scales <= 1e-12)
        ):
            raise ValueError(
                "noise_scale_by_target must be finite and strictly positive"
            )
        resolved_scale_source = (
            noise_scale_source or "caller_provided_per_target_scale"
        )
    effective_scales = requested_scales.copy()
    if preserve_additive_hierarchy:
        if values.shape[1] < 3:
            raise ValueError("additive hierarchy requires parent plus children")
        effective_scales[0] = float(
            np.sqrt(np.sum(np.square(requested_scales[1:])))
        )
        child_noise = rng.normal(
            size=(context_length, values.shape[1] - 1),
        ) * (noise_ratio * requested_scales[1:])[None, :]
        result[:context_length, 1:] += child_noise
        result[:context_length, 0] = np.sum(
            result[:context_length, 1:],
            axis=1,
        )
        applied_noise = result[:context_length] - history
    else:
        applied_noise = rng.normal(size=history.shape) * (
            noise_ratio * requested_scales
        )[None, :]
        result[:context_length] += applied_noise
    applied_noise_std = np.std(applied_noise, axis=0)
    realized_ratio_by_target = (
        applied_noise_std / effective_scales
    )
    metadata: dict[str, Any] = {
        "noise_scale_source": resolved_scale_source,
        "requested_noise_to_scale_ratio": float(noise_ratio),
        "realized_noise_to_scale_ratio": float(
            np.mean(realized_ratio_by_target)
        ),
        "realized_noise_to_scale_ratio_by_target": [
            float(value) for value in realized_ratio_by_target
        ],
        "requested_noise_scale_by_target": [
            float(value) for value in requested_scales
        ],
        "effective_noise_scale_by_target": [
            float(value) for value in effective_scales
        ],
        "applied_noise_std_by_target": [
            float(value) for value in applied_noise_std
        ],
        "additive_hierarchy_parent_scale_policy": (
            "root_sum_square_of_child_scales"
            if preserve_additive_hierarchy
            else None
        ),
        "future_noise_max_abs": float(
            np.max(np.abs(result[context_length:] - values[context_length:]))
        ),
    }
    if noise_scale_by_target is None:
        metadata["requested_noise_to_history_std_ratio"] = float(
            noise_ratio
        )
        metadata["realized_noise_to_history_std_ratio"] = float(
            np.mean(realized_ratio_by_target)
        )
    return result, metadata


REQUIRED_REAL_FEATURES_BY_CAPABILITY: dict[str, tuple[str, ...]] = {
    "trend": ("trend_strength", "slope_abs", "curvature_abs", "acf1"),
    "multi_seasonal": (
        "multi_period_score",
        "dominant_period",
        "spectral_concentration",
    ),
    "time_varying_seasonality": (
        "seasonal_amplitude_modulation",
        "seasonal_phase_variation",
        "dominant_period",
    ),
    "regime_switching": (
        "change_point_shift_energy",
        "level_shift_strength",
        "regime_sparse_transition_score",
        "acf1",
    ),
    "nonlinear_persistence": (
        "acf1",
        "dominant_period",
        "spectral_concentration",
    ),
    "predictable_intermittency": (
        "spike_rate",
        "intermittency_clock_incremental_r2",
        "dominant_period",
    ),
    "common_factor": (
        "pca_top1_explained",
        "effective_factor_rank",
        "factor_score_acf1",
        "factor_residual_acf1",
    ),
    "hierarchical_coherence": (
        "hierarchy_child_heterogeneity",
        "hierarchy_aggregate_acf1",
        "hierarchy_contrast_acf1",
        "hierarchy_aggregate_seasonal_acf",
        "hierarchy_contrast_seasonal_acf",
        "hierarchy_contrast_to_aggregate_std_ratio",
        "hierarchy_aggregate_contrast_abs_corr",
    ),
    "cross_series_dependence": (
        "cross_series_incremental_r2",
        "lead_lag_peak_abs",
        "lead_lag_peak_lag_abs",
        "avg_abs_target_corr",
    ),
    "covariate_response": (
        "covariate_incremental_r2",
        "event_lift_abs",
        "covariate_residual_acf_abs_mean",
        "acf1",
    ),
}


@dataclass(frozen=True)
class ParameterMapping:
    parameter: str
    source_feature: str
    source_value: float
    mapped_value: float
    transform: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "source_feature": self.source_feature,
            "source_value": float(self.source_value),
            "mapped_value": float(self.mapped_value),
            "transform": self.transform,
        }


def _median(
    summary: dict[str, dict[str, float]],
    feature: str,
    default: float,
) -> float:
    row = summary.get(feature, {})
    value = float(row.get("p50", default))
    return value if math.isfinite(value) else float(default)


def derive_deterministic_parameters(
    capability_id: str,
    real_feature_summary: dict[str, dict[str, float]],
    *,
    season_length: int,
    context_length: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Map descriptive real-window features to deterministic generator knobs.

    The mapping deliberately records every source feature.  It does not claim
    that the mapped family is the real DGP; it only constrains a controlled
    surrogate to the empirical support of one dataset profile.
    """

    mappings: list[ParameterMapping] = []
    parameters: dict[str, float] = {}

    def add(
        parameter: str,
        source: str,
        default: float,
        lower: float,
        upper: float,
        transform: str = "clip",
        value_fn=lambda value: value,
    ) -> None:
        source_value = _median(real_feature_summary, source, default)
        mapped = float(np.clip(value_fn(source_value), lower, upper))
        parameters[parameter] = mapped
        mappings.append(
            ParameterMapping(
                parameter,
                source,
                source_value,
                mapped,
                transform,
            )
        )

    add("profile_acf1", "acf1", 0.75, -0.2, 0.995)
    add("profile_seasonal_acf", "seasonal_acf", 0.5, -0.5, 0.995)
    add(
        "profile_dominant_period",
        "dominant_period",
        float(season_length),
        4.0,
        max(4.0, context_length / 2.0),
    )
    add(
        "profile_spectral_concentration",
        "spectral_concentration",
        0.25,
        0.02,
        0.95,
    )

    if capability_id == "trend":
        add("trend_strength_target", "trend_strength", 0.15, 0.02, 0.95)
        add("trend_slope_scale", "slope_abs", 0.2, 0.02, 1.5)
        slope = max(abs(_median(real_feature_summary, "slope_abs", 0.2)), 1e-6)
        curvature = abs(_median(real_feature_summary, "curvature_abs", 0.03))
        value = float(np.clip(curvature / slope, 0.01, 0.35))
        parameters["trend_curvature_ratio"] = value
        mappings.append(
            ParameterMapping(
                "trend_curvature_ratio",
                "curvature_abs/slope_abs",
                curvature / slope,
                value,
                "ratio_then_clip",
            )
        )
    elif capability_id == "multi_seasonal":
        add(
            "secondary_component_ratio",
            "multi_period_score",
            0.2,
            0.12,
            0.85,
            "sqrt_then_clip",
            lambda value: math.sqrt(max(value, 0.0)),
        )
    elif capability_id == "time_varying_seasonality":
        add(
            "modulation_depth_scale",
            "seasonal_amplitude_modulation",
            0.25,
            0.08,
            0.85,
            "identity_then_clip",
        )
        add(
            "phase_variation_scale",
            "seasonal_phase_variation",
            0.08,
            0.01,
            0.8,
        )
    elif capability_id == "regime_switching":
        add(
            "regime_level_scale",
            "level_shift_strength",
            0.8,
            0.2,
            2.5,
        )
        add(
            "regime_dwell_scale",
            "change_point_shift_energy",
            0.8,
            0.55,
            1.8,
            "inverse_sqrt_then_clip",
            lambda value: 1.0 / math.sqrt(max(value, 0.05)),
        )
        add(
            "regime_sparse_scale",
            "regime_sparse_transition_score",
            0.25,
            0.1,
            1.0,
        )
    elif capability_id == "nonlinear_persistence":
        # Nonlinear strength is a controlled synthetic mechanism dose.  The
        # observable adjusted-R2 proxy is retained for diagnostics, but must
        # not feed back into the coefficient it is supposed to measure.
        parameters["nonlinear_gain_scale"] = 2.1
        mappings.append(
            ParameterMapping(
                "nonlinear_gain_scale",
                "synthetic_protocol_constant",
                2.1,
                2.1,
                "fixed_mechanism_dose_scale",
            )
        )
        parameters["nonlinear_lag_scale"] = 1.0 / 3.0
        mappings.append(
            ParameterMapping(
                "nonlinear_lag_scale",
                "synthetic_protocol_constant",
                1.0 / 3.0,
                1.0 / 3.0,
                "fixed_fraction_of_profile_period",
            )
        )
    elif capability_id == "predictable_intermittency":
        add(
            "event_width_scale",
            "spike_rate",
            0.02,
            0.035,
            0.18,
            "inverse_linear_then_clip",
            lambda value: 0.16 - 1.5 * max(value, 0.0),
        )
        add(
            "event_base_scale",
            "intermittency_clock_incremental_r2",
            0.1,
            0.03,
            0.5,
            "sqrt_then_clip",
            lambda value: math.sqrt(max(value, 0.0)),
        )
    elif capability_id == "common_factor":
        add("factor_persistence", "factor_score_acf1", 0.9, 0.2, 0.995)
        add("local_persistence", "factor_residual_acf1", 0.7, 0.1, 0.99)
        add(
            "shared_variance_target",
            "pca_top1_explained",
            0.75,
            0.35,
            0.995,
        )
        add(
            "local_mode_spread",
            "effective_factor_rank",
            1.8,
            0.15,
            0.9,
            "rank_excess_then_clip",
            lambda value: max(value - 1.0, 0.0),
        )
    elif capability_id == "hierarchical_coherence":
        add(
            "aggregate_persistence",
            "hierarchy_aggregate_acf1",
            0.9,
            0.2,
            0.995,
        )
        add(
            "contrast_persistence",
            "hierarchy_contrast_acf1",
            0.8,
            0.2,
            0.995,
        )
        add(
            "aggregate_seasonal_memory",
            "hierarchy_aggregate_seasonal_acf",
            0.5,
            -0.25,
            0.995,
        )
        add(
            "contrast_seasonal_memory",
            "hierarchy_contrast_seasonal_acf",
            0.4,
            -0.25,
            0.995,
        )
        add(
            "contrast_to_aggregate_ratio",
            "hierarchy_contrast_to_aggregate_std_ratio",
            0.3,
            0.05,
            2.0,
        )
        add(
            "aggregate_contrast_abs_corr",
            "hierarchy_aggregate_contrast_abs_corr",
            0.3,
            0.0,
            0.95,
        )
        add(
            "hierarchy_heterogeneity_scale",
            "hierarchy_child_heterogeneity",
            0.2,
            0.05,
            1.5,
        )
    elif capability_id == "cross_series_dependence":
        add(
            "cross_dependence_scale",
            "cross_series_incremental_r2",
            0.1,
            0.35,
            1.5,
            "sqrt_then_clip",
            lambda value: 1.8 * math.sqrt(max(value, 0.0)),
        )
        add(
            "cross_lag_steps",
            "lead_lag_peak_lag_abs",
            float(season_length),
            8.0,
            24.0,
            "round_then_clip",
            lambda value: float(round(value)),
        )
        add(
            "cross_lag_alignment",
            "lead_lag_peak_abs",
            0.5,
            0.2,
            0.995,
        )
        add(
            "cross_channel_background_ratio",
            "avg_abs_target_corr",
            0.4,
            0.15,
            0.8,
        )
    elif capability_id == "covariate_response":
        add(
            "covariate_effect_scale",
            "covariate_incremental_r2",
            0.1,
            0.15,
            1.5,
            "sqrt_then_clip",
            lambda value: math.sqrt(max(value, 0.0)),
        )
        add(
            "covariate_explained_scale",
            "covariate_incremental_r2",
            0.1,
            0.25,
            1.5,
            "sqrt_then_clip",
            lambda value: 2.0 * math.sqrt(max(value, 0.0)),
        )
        add(
            "event_effect_ratio",
            "event_lift_abs",
            0.8,
            0.3,
            2.0,
        )
        add(
            "covariate_residual_memory_target",
            "covariate_residual_acf_abs_mean",
            0.42,
            0.40,
            0.44,
            "compress_to_all_suffix_real_support_intersection",
            lambda value: (
                0.40
                + 0.04
                * float(
                    np.clip(
                        (value - 0.25) / (0.78 - 0.25),
                        0.0,
                        1.0,
                    )
                )
            ),
        )

    return parameters, [mapping.as_dict() for mapping in mappings]


def _parameter(
    conditioning: GeneratorConditioning | None,
    name: str,
    default: float,
) -> float:
    if conditioning is None:
        return float(default)
    return float(conditioning.parameters.get(name, default))


def _profile_period(
    conditioning: GeneratorConditioning | None,
    fallback: float,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Resolve an observable generator time scale from a real-window peak.

    ``profile_dominant_period`` is a descriptive spectral peak rather than a
    literal calendar season.  Every mechanism therefore clips it to a range
    that is identifiable inside the shortest supported L96/H48 view.
    """

    raw = _parameter(conditioning, "profile_dominant_period", fallback)
    effective = float(np.clip(raw, lower, upper))
    return float(raw), effective


def _lambda(
    conditioning: GeneratorConditioning | None,
    intensity: int,
) -> float:
    if conditioning is None:
        return (int(intensity) - 1) / 4.0
    return float(conditioning.lambda_for(intensity))


def _standardize_history(
    values: np.ndarray,
    context_length: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    history = matrix[:context_length]
    center = np.mean(history, axis=0, keepdims=True)
    scale = np.std(history, axis=0, keepdims=True)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return amplitude * (matrix - center) / scale


def standardize_cross_series_counterfactual_member(
    values: np.ndarray,
    *,
    context_length: int,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use pair-invariant statistics for a cross-series counterfactual member.

    Responder histories are exactly invariant across pair members, so ordinary
    context standardization already gives them shared statistics.  The driver
    intervention lies in the final ``delay`` history steps; using the full
    driver context separately for each member would turn that localized
    intervention into a global affine difference.  Standardizing the driver
    from its invariant prefix avoids that counterfactual confound.
    """

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("cross-series target must be a two-dimensional array")
    if not 1 <= context_length < matrix.shape[0]:
        raise ValueError("context_length must split history and future")
    driver = int(metadata["driver_index"])
    driver_slice = metadata.get("counterfactual_driver_slice")
    if (
        not isinstance(driver_slice, list)
        or len(driver_slice) != 2
        or int(driver_slice[1]) != context_length
    ):
        raise ValueError(
            "counterfactual_driver_slice must end at the context boundary"
        )
    invariant_stop = int(driver_slice[0])
    if invariant_stop < 8:
        raise ValueError(
            "counterfactual driver invariant prefix is too short to standardize"
        )

    standardized = _standardize_history(matrix, context_length)
    driver_reference = matrix[:invariant_stop, driver]
    driver_center = float(np.mean(driver_reference))
    driver_scale = float(np.std(driver_reference))
    if driver_scale <= 1e-9:
        driver_scale = 1.0
    standardized[:, driver] = (
        matrix[:, driver] - driver_center
    ) / driver_scale
    return standardized, {
        "policy": "pair_shared_driver_invariant_prefix",
        "driver_reference_slice": [0, invariant_stop],
        "driver_center": driver_center,
        "driver_scale": driver_scale,
        "responder_reference_slice": [0, context_length],
    }


def standardize_common_factor_counterfactual_member(
    values: np.ndarray,
    *,
    context_length: int,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the pair-invariant prefix to normalize every channel.

    Pair members differ only in the auxiliary shared-state evidence suffix and
    the resulting future.  Computing normalization statistics before that
    suffix prevents the intervention from changing the affine scale seen by a
    model.
    """

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("common-factor target must be two-dimensional")
    if not 1 <= context_length < matrix.shape[0]:
        raise ValueError("context_length must split history and future")
    code_slice = metadata.get(
        "shared_state_evidence_slice",
        metadata.get("final_code_slice"),
    )
    if (
        not isinstance(code_slice, list)
        or len(code_slice) != 2
        or int(code_slice[1]) != context_length
    ):
        raise ValueError(
            "shared-state evidence slice must end at the context boundary"
        )
    invariant_stop = int(code_slice[0])
    if invariant_stop < 16:
        raise ValueError(
            "common-factor invariant prefix is too short to standardize"
        )
    reference = matrix[:invariant_stop]
    center = np.mean(reference, axis=0)
    global_scale = float(
        np.sqrt(np.mean(np.var(reference, axis=0)))
    )
    if global_scale <= 1e-9:
        global_scale = 1.0
    standardized = (matrix - center[None, :]) / global_scale
    input_loadings = np.asarray(
        metadata.get(
            "standardized_response_loadings",
            metadata["response_loadings"],
        ),
        dtype=float,
    )
    standardized_loadings = input_loadings / global_scale
    metadata["standardized_response_loadings"] = (
        standardized_loadings.tolist()
    )
    return standardized, {
        "policy": "pair_shared_global_scale_pre_state_evidence_prefix",
        "reference_slice": [0, invariant_stop],
        "center_by_target": center.tolist(),
        "global_scale": global_scale,
        "scale_by_target": [global_scale] * matrix.shape[1],
    }


def _fit_affine_map(
    source: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        [np.ones(source.shape[0], dtype=float), source]
    )
    ridge = np.eye(design.shape[1], dtype=float) * 1e-8
    ridge[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + ridge,
        design.T @ response,
    )


def _chronological_affine_holdout_r2(
    source: np.ndarray,
    response: np.ndarray,
) -> float:
    source = np.asarray(source, dtype=float)
    response = np.asarray(response, dtype=float)
    if source.ndim == 1:
        source = source[:, None]
    if response.ndim == 1:
        response = response[:, None]
    if source.shape[0] != response.shape[0] or source.shape[0] < 24:
        return -math.inf
    split = int(np.clip(round(0.70 * source.shape[0]), 16, source.shape[0] - 8))
    coefficients = _fit_affine_map(source[:split], response[:split])
    design = np.column_stack(
        [np.ones(source.shape[0] - split), source[split:]]
    )
    forecasts = design @ coefficients
    truth = response[split:]
    denominator = float(
        np.sum((truth - np.mean(truth, axis=0)) ** 2)
    )
    if denominator <= 1e-12:
        return -math.inf
    return 1.0 - float(np.sum((truth - forecasts) ** 2)) / denominator


def _blind_seasonal_state_forecast(
    history: np.ndarray,
    *,
    horizon: int,
    selected_lag: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Blindly select a seasonal state recurrence on a chronological holdout."""

    values = np.asarray(history, dtype=float).ravel()
    validation_rows = max(12, int(round(0.25 * values.size)))
    validation_rows = min(validation_rows, values.size - 24)
    split = values.size - validation_rows
    candidates: list[tuple[float, int]] = []
    candidate_lags = (
        [int(selected_lag)]
        if selected_lag is not None
        else list(range(1, min(48, split // 2) + 1))
    )
    for lag in candidate_lags:
        if lag < 1 or lag > split // 2:
            continue
        truth = values[split:]
        validation_history = values[:split].tolist()
        prediction_values = []
        for _ in range(validation_rows):
            next_value = float(validation_history[-lag])
            prediction_values.append(next_value)
            validation_history.append(next_value)
        prediction = np.asarray(prediction_values, dtype=float)
        loss = float(np.mean(np.abs(truth - prediction)))
        if math.isfinite(loss):
            candidates.append((loss, lag))
    if not candidates:
        raise ValueError("no valid blind shared-state recurrence")
    validation_mae, fitted_lag = min(
        candidates,
        key=lambda row: (row[0], row[1]),
    )
    extended = values.tolist()
    for _ in range(horizon):
        extended.append(float(extended[-fitted_lag]))
    return (
        np.asarray(extended[-horizon:], dtype=float),
        int(fitted_lag),
        float(validation_mae),
    )


def _blind_common_factor_pair_forecast(
    first: np.ndarray,
    second: np.ndarray,
    *,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Forecast a pair using only its shared prefix and observed panel values."""

    first_history = np.asarray(first[:context_length], dtype=float)
    second_history = np.asarray(second[:context_length], dtype=float)
    channel_difference = np.max(
        np.abs(second_history - first_history),
        axis=0,
    )
    invariant_channels = np.flatnonzero(channel_difference <= 1e-10)
    if invariant_channels.size != 1:
        raise ValueError(
            "blind common-factor pair must expose one invariant channel"
        )
    protected = int(invariant_channels[0])
    row_difference = np.max(
        np.abs(second_history - first_history),
        axis=1,
    )
    changed_rows = np.flatnonzero(row_difference > 1e-10)
    if changed_rows.size == 0:
        raise ValueError("common-factor pair has no auxiliary evidence change")
    invariant_stop = int(changed_rows[0])
    if invariant_stop < 48:
        raise ValueError("common-factor invariant prefix is too short")

    reference = first_history[:invariant_stop]
    center = np.mean(reference, axis=0)
    scale = np.std(reference, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized_reference = (reference - center) / scale
    _, singular_values, right = np.linalg.svd(
        standardized_reference,
        full_matrices=False,
    )
    loading = right[0]
    auxiliary = np.asarray(
        [index for index in range(first.shape[1]) if index != protected],
        dtype=int,
    )
    auxiliary_loading_energy = float(
        np.dot(loading[auxiliary], loading[auxiliary])
    )
    if auxiliary_loading_energy <= 1e-8:
        raise ValueError("shared state is not observable from auxiliaries")

    reference_factor = (
        standardized_reference[:, auxiliary] @ loading[auxiliary]
        / auxiliary_loading_energy
    )
    _, selected_state_lag, reference_validation_loss = (
        _blind_seasonal_state_forecast(
            reference_factor,
            horizon=first.shape[0] - context_length,
        )
    )
    forecasts = []
    selected_lags = []
    validation_losses = []
    for history in (first_history, second_history):
        standardized = (history - center) / scale
        factor = (
            standardized[:, auxiliary] @ loading[auxiliary]
            / auxiliary_loading_energy
        )
        factor_forecast, selected_lag, validation_loss = (
            _blind_seasonal_state_forecast(
                factor,
                horizon=first.shape[0] - context_length,
                selected_lag=selected_state_lag,
            )
        )
        forecasts.append(
            center[None, :]
            + factor_forecast[:, None]
            * loading[None, :]
            * scale[None, :]
        )
        selected_lags.append(selected_lag)
        validation_losses.append(validation_loss)

    total_energy = float(np.sum(singular_values * singular_values))
    return forecasts[0], forecasts[1], {
        "protected_target_index": protected,
        "pair_invariant_history_stop": invariant_stop,
        "selected_state_lags": selected_lags,
        "shared_prefix_selected_state_lag": selected_state_lag,
        "shared_prefix_state_validation_mae": reference_validation_loss,
        "state_validation_mae": validation_losses,
        "history_factor_share": float(
            singular_values[0] ** 2 / max(total_energy, 1e-12)
        ),
        "loading": loading.tolist(),
        "generator_metadata_used_for_fitting": False,
        "history_only_fitting": True,
    }


def common_factor_joint_positive_control(
    values: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Compatibility probe using a blind rank-one state forecast.

    The formal identifiability gate does not call this helper.  It remains for
    the legacy diagnostic probe model, but no longer consumes a hidden episode
    layout or codebook.
    """

    matrix = np.asarray(values, dtype=float)
    evidence_slice = metadata.get(
        "shared_state_evidence_slice",
        metadata.get("final_code_slice"),
    )
    if not isinstance(evidence_slice, list) or len(evidence_slice) != 2:
        raise ValueError("common-factor evidence slice is missing")
    invariant_stop, context_length = (
        int(value) for value in evidence_slice
    )
    if not 48 <= invariant_stop < context_length < matrix.shape[0]:
        raise ValueError("invalid common-factor evidence slice")
    reference = matrix[:invariant_stop]
    center = np.mean(reference, axis=0)
    scale = np.std(reference, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized_reference = (reference - center) / scale
    _, _, right = np.linalg.svd(
        standardized_reference,
        full_matrices=False,
    )
    loading = right[0]
    protected = int(metadata["protected_target_index"])
    auxiliary = np.asarray(
        [
            index
            for index in range(matrix.shape[1])
            if index != protected
        ],
        dtype=int,
    )
    loading_energy = float(
        np.dot(loading[auxiliary], loading[auxiliary])
    )
    standardized_history = (
        matrix[:context_length] - center
    ) / scale
    factor_history = (
        standardized_history[:, auxiliary] @ loading[auxiliary]
        / max(loading_energy, 1e-12)
    )
    factor_forecast, _, _ = _blind_seasonal_state_forecast(
        factor_history,
        horizon=matrix.shape[0] - context_length,
        selected_lag=int(metadata["shared_state_period"]),
    )
    return (
        factor_forecast[:, None]
        * loading[None, :]
        * scale[None, :]
    )


def common_factor_identifiability_gate(
    first_target: np.ndarray,
    second_target: np.ndarray,
    *,
    context_length: int,
    metadata: dict[str, Any],
    enforced: bool,
) -> dict[str, Any]:
    """Check the real-aligned factor coordinate and paired construction.

    A fixed predictive-R2 floor is not a valid common-factor requirement:
    weak but genuine real panels can have a top-component share well below
    0.50.  The main gate therefore uses the same observable as calibration,
    relative to its finite-panel isotropic floor.  The stricter blind
    counterfactual forecast remains a diagnostic positive control and is
    exercised separately at a strong protocol dose.
    """

    first = np.asarray(first_target, dtype=float)
    second = np.asarray(second_target, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("common-factor pair must be equal-shaped matrices")
    if not 1 <= context_length < first.shape[0]:
        raise ValueError("context_length must split history and future")
    declared_protected = int(metadata["protected_target_index"])
    evidence_slice = metadata.get(
        "shared_state_evidence_slice",
        metadata.get("final_code_slice"),
    )
    if not isinstance(evidence_slice, list) or len(evidence_slice) != 2:
        raise ValueError("common-factor evidence slice is missing")
    invariant_stop = int(evidence_slice[0])
    history = first[:invariant_stop]
    protected = declared_protected
    auxiliary = [
        index for index in range(first.shape[1]) if index != protected
    ]
    history_center = np.mean(history, axis=0)
    history_scale = np.std(history, axis=0)
    history_scale = np.where(history_scale > 1e-9, history_scale, 1.0)
    standardized_history = (
        history - history_center[None, :]
    ) / history_scale[None, :]
    _, _, history_right = np.linalg.svd(
        standardized_history,
        full_matrices=False,
    )
    history_loading = history_right[0]
    auxiliary_loading_energy = float(
        np.dot(
            history_loading[auxiliary],
            history_loading[auxiliary],
        )
    )
    joint_factor_score = (
        standardized_history[:, auxiliary] @ history_loading[auxiliary]
        / max(auxiliary_loading_energy, 1e-12)
    )
    joint_r2 = _chronological_affine_holdout_r2(
        joint_factor_score,
        standardized_history[:, protected],
    )
    single_r2 = [
        _chronological_affine_holdout_r2(
            standardized_history[:, index],
            standardized_history[:, protected],
        )
        for index in auxiliary
    ]
    best_single_r2 = float(np.max(single_r2))
    joint_gain = float(joint_r2 - best_single_r2)
    singular_values = np.linalg.svd(
        standardized_history,
        compute_uv=False,
    )
    total_singular_energy = float(
        np.sum(singular_values * singular_values)
    )
    observable_factor_share = float(
        singular_values[0] ** 2 / max(total_singular_energy, 1e-12)
    )
    isotropic_factor_share = 1.0 / float(first.shape[1])
    minimum_factor_share = min(
        1.0,
        isotropic_factor_share + COMMON_FACTOR_MIN_EXCESS_PCA_SHARE,
    )
    observability_passed = observable_factor_share >= minimum_factor_share

    protected_history_difference = float(
        np.max(
            np.abs(
                second[:context_length, protected]
                - first[:context_length, protected]
            )
        )
    )
    truth_effect = (
        second[context_length:, protected]
        - first[context_length:, protected]
    )
    first_forecast, second_forecast, blind_diagnostics = (
        _blind_common_factor_pair_forecast(
            first,
            second,
            context_length=context_length,
        )
    )
    if int(blind_diagnostics["protected_target_index"]) != protected:
        raise ValueError(
            "blind invariant channel does not match declared protected target"
        )
    forecast_effect = (
        second_forecast[:, protected] - first_forecast[:, protected]
    )
    truth_rms = float(np.sqrt(np.mean(truth_effect * truth_effect)))
    forecast_rms = float(
        np.sqrt(np.mean(forecast_effect * forecast_effect))
    )
    effect_nrmse = float(
        np.sqrt(np.mean((forecast_effect - truth_effect) ** 2))
        / max(truth_rms, 1e-12)
    )
    effect_correlation = _safe_flat_correlation(
        truth_effect,
        forecast_effect,
    )
    effect_amplitude_ratio = forecast_rms / max(truth_rms, 1e-12)
    amplitude_lower, amplitude_upper = (
        COMMON_FACTOR_EFFECT_AMPLITUDE_RANGE
    )
    counterfactual_passed = (
        protected_history_difference <= 1e-10
        and truth_rms > 1e-4
        and effect_nrmse <= COMMON_FACTOR_MAX_EFFECT_NRMSE
        and effect_correlation >= COMMON_FACTOR_MIN_EFFECT_CORRELATION
        and amplitude_lower <= effect_amplitude_ratio <= amplitude_upper
    )
    construction_passed = (
        protected_history_difference <= 1e-10
        and truth_rms > 1e-4
        and len(auxiliary) >= 2
    )
    accepted = observability_passed and construction_passed
    return {
        "schema_version": "common_factor_identifiability_gate.v3",
        "enforced": bool(enforced),
        "accepted": bool(accepted),
        "latent_state_dimension": 1,
        "generator_metadata_used_for_fitting": False,
        "positive_control_model": (
            "blind_rank1_panel_filter_with_seasonal_state_recurrence"
        ),
        "joint_holdout_r2": float(joint_r2),
        "single_channel_holdout_r2": [
            float(value) for value in single_r2
        ],
        "best_single_channel_holdout_r2": best_single_r2,
        "joint_minus_best_single_holdout_r2": joint_gain,
        "joint_holdout_r2_is_diagnostic_only": True,
        "observable_factor_share": observable_factor_share,
        "isotropic_factor_share": isotropic_factor_share,
        "minimum_excess_pca_share": COMMON_FACTOR_MIN_EXCESS_PCA_SHARE,
        "minimum_observable_factor_share": minimum_factor_share,
        "single_channel_holdout_is_diagnostic_only": True,
        "joint_observability_passed": bool(observability_passed),
        "protected_target_index": protected,
        "protected_history_max_abs_difference": (
            protected_history_difference
        ),
        "truth_effect_rms": truth_rms,
        "positive_control_effect_nrmse": effect_nrmse,
        "positive_control_effect_correlation": effect_correlation,
        "positive_control_effect_amplitude_ratio": effect_amplitude_ratio,
        "positive_control_max_effect_nrmse": (
            COMMON_FACTOR_MAX_EFFECT_NRMSE
        ),
        "positive_control_min_effect_correlation": (
            COMMON_FACTOR_MIN_EFFECT_CORRELATION
        ),
        "positive_control_effect_amplitude_range": [
            amplitude_lower,
            amplitude_upper,
        ],
        "counterfactual_passed": bool(counterfactual_passed),
        "counterfactual_positive_control_is_diagnostic_at_selected_dose": True,
        "separate_strong_dose_positive_control_required": True,
        "paired_construction_passed": bool(construction_passed),
        "blind_fit_diagnostics": blind_diagnostics,
    }


def _chronological_linear_holdout(
    source: np.ndarray,
    response: np.ndarray,
) -> tuple[float, float, float]:
    source = np.asarray(source, dtype=float).ravel()
    response = np.asarray(response, dtype=float).ravel()
    if source.size != response.size or source.size < 12:
        return -math.inf, 0.0, 0.0
    split = int(np.clip(round(0.70 * source.size), 8, source.size - 4))
    design = np.column_stack(
        [np.ones(split, dtype=float), source[:split]]
    )
    intercept, slope = np.linalg.lstsq(
        design,
        response[:split],
        rcond=None,
    )[0]
    truth = response[split:]
    forecast = intercept + slope * source[split:]
    denominator = float(np.sum((truth - float(np.mean(truth))) ** 2))
    if denominator <= 1e-12:
        return -math.inf, float(intercept), float(slope)
    r2 = 1.0 - float(np.sum((truth - forecast) ** 2)) / denominator
    return float(r2), float(intercept), float(slope)


def _chronological_incremental_holdout_gain(
    source: np.ndarray,
    response: np.ndarray,
    *,
    source_lag: int,
    own_order: int = 12,
) -> float:
    """Held-out gain from one lagged source beyond response self-history."""

    source_values = np.asarray(source, dtype=float).ravel()
    response_values = np.asarray(response, dtype=float).ravel()
    if source_values.size != response_values.size:
        return -math.inf
    start = max(int(source_lag), int(own_order))
    sample_count = response_values.size - start
    if sample_count < 36:
        return -math.inf
    target = response_values[start:]
    own = np.column_stack(
        [
            response_values[start - lag : response_values.size - lag]
            for lag in range(1, own_order + 1)
        ]
    )
    source_column = source_values[
        start - source_lag : source_values.size - source_lag
    ]
    split = int(np.clip(round(0.70 * sample_count), 24, sample_count - 12))
    baseline_design = np.column_stack(
        [np.ones(sample_count, dtype=float), own]
    )
    full_design = np.column_stack([baseline_design, source_column])
    ridge = np.eye(full_design.shape[1], dtype=float) * 1e-6
    ridge[0, 0] = 0.0

    def prediction(design: np.ndarray) -> np.ndarray:
        penalty = ridge[: design.shape[1], : design.shape[1]]
        coefficients = np.linalg.solve(
            design[:split].T @ design[:split] + penalty,
            design[:split].T @ target[:split],
        )
        return design[split:] @ coefficients

    truth = target[split:]
    baseline_error = float(
        np.sum((truth - prediction(baseline_design)) ** 2)
    )
    if baseline_error <= 1e-12:
        return -math.inf
    full_error = float(np.sum((truth - prediction(full_design)) ** 2))
    return float((baseline_error - full_error) / baseline_error)


def _safe_flat_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel().copy()
    right = np.asarray(right, dtype=float).ravel().copy()
    if left.size < 3 or left.size != right.size:
        return 0.0
    left -= float(np.mean(left))
    right -= float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return (
        float(np.dot(left, right) / denominator)
        if denominator > 1e-12
        else 0.0
    )


def cross_series_identifiability_gate(
    first_target: np.ndarray,
    second_target: np.ndarray,
    *,
    context_length: int,
    metadata: dict[str, Any],
    enforced: bool,
) -> dict[str, Any]:
    """Audit whether a history-only learner can recover the intended SCM.

    The blind search receives neither the driver identity nor the lag.  It
    selects the source/lag pair that jointly predicts every other channel on a
    chronological holdout.  The positive control then uses the declared
    protocol roles only after discovery, estimates signed response coefficients
    from history, and checks the paired future effect.
    """

    first = np.asarray(first_target, dtype=float)
    second = np.asarray(second_target, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("counterfactual targets must be equal-shaped matrices")
    if not 1 <= context_length < first.shape[0]:
        raise ValueError("context_length must split history and future")
    driver = int(metadata["driver_index"])
    responders = [int(value) for value in metadata["responder_indices"]]
    delay = int(metadata["cross_lag_steps"])
    horizon = first.shape[0] - context_length
    max_lag = min(24, context_length - 36)
    if not 1 <= delay <= max_lag:
        return {
            "schema_version": "cross_series_identifiability_gate.v2",
            "enforced": bool(enforced),
            "accepted": False,
            "reason": "declared_lag_outside_blind_search_support",
            "declared_driver": driver,
            "declared_lag": delay,
            "blind_max_lag": max_lag,
        }

    history = first[:context_length]
    blind_candidates: list[tuple[float, int, int, list[float]]] = []
    for candidate_source in range(history.shape[1]):
        destinations = [
            index
            for index in range(history.shape[1])
            if index != candidate_source
        ]
        for candidate_lag in range(1, max_lag + 1):
            holdout_scores = [
                _chronological_incremental_holdout_gain(
                    history[:, candidate_source],
                    history[:, destination],
                    source_lag=candidate_lag,
                )
                for destination in destinations
            ]
            finite_scores = [
                value for value in holdout_scores if math.isfinite(value)
            ]
            score = (
                float(np.mean(finite_scores))
                if len(finite_scores) == len(holdout_scores)
                else -math.inf
            )
            blind_candidates.append(
                (
                    score,
                    candidate_source,
                    candidate_lag,
                    holdout_scores,
                )
            )
    blind_candidates.sort(key=lambda row: row[0], reverse=True)
    best_score, best_source, best_lag, best_destination_scores = (
        blind_candidates[0]
    )
    blind_passed = best_source == driver and abs(best_lag - delay) <= 2

    declared_holdout_r2 = [
        _chronological_incremental_holdout_gain(
            history[:, driver],
            history[:, responder],
            source_lag=delay,
        )
        for responder in responders
    ]
    minimum_holdout_r2 = float(np.min(declared_holdout_r2))
    aggregate_holdout_gain = float(
        np.mean(np.clip(declared_holdout_r2, 0.0, 1.0))
    )
    holdout_passed = (
        aggregate_holdout_gain
        >= CROSS_SERIES_MIN_INCREMENTAL_HOLDOUT_GAIN
    )

    training_driver = history[: context_length - delay, driver]
    design = np.column_stack(
        [np.ones(training_driver.size, dtype=float), training_driver]
    )
    first_forecast = np.empty((horizon, len(responders)), dtype=float)
    second_forecast = np.empty_like(first_forecast)
    fitted_slopes: list[float] = []
    for column, responder in enumerate(responders):
        training_response = history[delay:context_length, responder]
        intercept, slope = np.linalg.lstsq(
            design,
            training_response,
            rcond=None,
        )[0]
        fitted_slopes.append(float(slope))
        first_driver = first[
            context_length - delay : context_length - delay + horizon,
            driver,
        ]
        second_driver = second[
            context_length - delay : context_length - delay + horizon,
            driver,
        ]
        first_forecast[:, column] = intercept + slope * first_driver
        second_forecast[:, column] = intercept + slope * second_driver

    truth_effect = (
        second[context_length:, responders]
        - first[context_length:, responders]
    )
    forecast_effect = second_forecast - first_forecast
    truth_rms = float(np.sqrt(np.mean(truth_effect * truth_effect)))
    forecast_rms = float(
        np.sqrt(np.mean(forecast_effect * forecast_effect))
    )
    effect_nrmse = float(
        np.sqrt(np.mean((forecast_effect - truth_effect) ** 2))
        / max(truth_rms, 1e-12)
    )
    effect_correlation = _safe_flat_correlation(
        truth_effect,
        forecast_effect,
    )
    effect_amplitude_ratio = forecast_rms / max(truth_rms, 1e-12)
    amplitude_lower, amplitude_upper = CROSS_SERIES_EFFECT_AMPLITUDE_RANGE
    positive_control_passed = (
        effect_nrmse <= CROSS_SERIES_MAX_EFFECT_NRMSE
        and effect_correlation >= CROSS_SERIES_MIN_EFFECT_CORRELATION
        and amplitude_lower <= effect_amplitude_ratio <= amplitude_upper
    )
    accepted = holdout_passed and positive_control_passed
    return {
        "schema_version": "cross_series_identifiability_gate.v2",
        "enforced": bool(enforced),
        "accepted": bool(accepted),
        "declared_driver": driver,
        "declared_lag": delay,
        "blind_max_lag": max_lag,
        "blind_best_driver": int(best_source),
        "blind_best_lag": int(best_lag),
        "blind_best_mean_incremental_holdout_gain": float(best_score),
        "blind_best_destination_incremental_holdout_gain": [
            float(value) for value in best_destination_scores
        ],
        "blind_driver_lag_passed": bool(blind_passed),
        "blind_driver_lag_is_diagnostic_at_selected_dose": True,
        "separate_strong_dose_graph_recovery_required": True,
        "declared_responder_incremental_holdout_gain": [
            float(value) for value in declared_holdout_r2
        ],
        "minimum_declared_incremental_holdout_gain": minimum_holdout_r2,
        "aggregate_declared_incremental_holdout_gain": (
            aggregate_holdout_gain
        ),
        "responder_aggregation_policy": (
            "mean_positive_incremental_gain_matching_public_coordinate"
        ),
        "minimum_incremental_holdout_gain_threshold": (
            CROSS_SERIES_MIN_INCREMENTAL_HOLDOUT_GAIN
        ),
        "incremental_history_holdout_passed": bool(holdout_passed),
        "positive_control_fitted_slopes": fitted_slopes,
        "positive_control_effect_nrmse": effect_nrmse,
        "positive_control_effect_correlation": effect_correlation,
        "positive_control_effect_amplitude_ratio": effect_amplitude_ratio,
        "positive_control_max_effect_nrmse": CROSS_SERIES_MAX_EFFECT_NRMSE,
        "positive_control_min_effect_correlation": (
            CROSS_SERIES_MIN_EFFECT_CORRELATION
        ),
        "positive_control_effect_amplitude_range": [
            amplitude_lower,
            amplitude_upper,
        ],
        "positive_control_passed": bool(positive_control_passed),
    }


def _lds_signal(
    length: int,
    context_length: int,
    rng: np.random.Generator,
    *,
    period: float,
    persistence: float,
    spectral_concentration: float = 0.5,
    mode_count: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Autonomous deterministic rotations with sample-specific modes."""

    time = np.arange(length, dtype=float)
    value = np.zeros(length, dtype=float)
    modes: list[dict[str, float]] = []
    persistence = float(np.clip(persistence, 0.0, 0.995))
    spectral_concentration = float(np.clip(spectral_concentration, 0.02, 0.95))
    envelope_depth = 0.04 + 0.16 * (1.0 - persistence)
    for index in range(mode_count):
        ratio = float(rng.uniform(0.78, 1.25) * (1.0 + 0.55 * index))
        selected_period = float(
            np.clip(
                period * ratio,
                BACKGROUND_PERIOD_RANGE[0],
                min(BACKGROUND_PERIOD_RANGE[1], context_length / 3.0),
            )
        )
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        secondary_scale = 1.0 if index == 0 else 1.15 - spectral_concentration
        amplitude = float(
            secondary_scale * (0.65**index) * rng.uniform(0.8, 1.2)
        )
        angle = 2.0 * np.pi / selected_period
        state = np.asarray([math.cos(phase), math.sin(phase)], dtype=float)
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=float,
        )
        component = np.empty(length, dtype=float)
        for step in range(length):
            component[step] = state[0]
            state = rotation @ state
        slow_period = max(selected_period * (4.5 + index), selected_period + 1.0)
        envelope = 1.0 + envelope_depth * np.cos(
            2.0 * np.pi * time / slow_period + phase / 2.0
        )
        value += amplitude * envelope * component
        modes.append(
            {
                "period": selected_period,
                "amplitude": amplitude,
                "phase": phase,
                "rotation_angle": angle,
            }
        )
    normalized = _standardize_history(value, context_length)[:, 0]
    return normalized, {
        "law": "autonomous_sample_specific_rotation_lds",
        "modes": modes,
        "persistence_calibration": persistence,
        "spectral_concentration_calibration": spectral_concentration,
        "future_process_noise_scale": 0.0,
    }


def _spline_motif_signal(
    length: int,
    context_length: int,
    rng: np.random.Generator,
    *,
    period: float,
    spectral_concentration: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    knot_count = 9
    knots = rng.normal(0.0, 1.0, size=knot_count)
    knots -= float(np.mean(knots))
    knots = np.concatenate([knots, knots[:1]])
    phase = float(rng.uniform(0.0, period))
    time = np.arange(length, dtype=float)
    position = ((time + phase) % period) / period * knot_count
    lower = np.floor(position).astype(int)
    fraction = position - lower
    values = (1.0 - fraction) * knots[lower] + fraction * knots[lower + 1]
    concentration = float(np.clip(spectral_concentration, 0.02, 0.95))
    sinusoid = np.sin(2.0 * np.pi * (time + phase) / period)
    values = concentration * sinusoid + (1.0 - concentration) * values
    slow_period = max(period * float(rng.uniform(4.5, 7.5)), period + 1.0)
    values *= 1.0 + 0.12 * np.sin(2.0 * np.pi * time / slow_period + phase)
    normalized = _standardize_history(values, context_length)[:, 0]
    return normalized, {
        "law": "sample_specific_periodic_spline_motif",
        "period": float(period),
        "knot_count": knot_count,
        "phase": phase,
        "spectral_concentration_calibration": concentration,
        "future_process_noise_scale": 0.0,
    }


def _calibrated_signal(
    length: int,
    context_length: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None,
    family_role: FamilyRole,
    *,
    persistence_name: str = "profile_acf1",
    seasonal_memory_name: str = "profile_seasonal_acf",
    period_multiplier: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_period, period = _profile_period(
        conditioning,
        24.0,
        lower=BACKGROUND_PERIOD_RANGE[0],
        upper=min(BACKGROUND_PERIOD_RANGE[1], context_length / 3.0),
    )
    period = float(
        np.clip(
            period * period_multiplier,
            BACKGROUND_PERIOD_RANGE[0],
            min(BACKGROUND_PERIOD_RANGE[1], context_length / 3.0),
        )
    )
    persistence = _parameter(conditioning, persistence_name, 0.85)
    seasonal_memory = _parameter(conditioning, seasonal_memory_name, 0.65)
    spectral_concentration = _parameter(
        conditioning,
        "profile_spectral_concentration",
        0.25,
    )
    effective_memory = float(
        np.clip(0.65 * persistence + 0.35 * seasonal_memory, 0.0, 0.995)
    )
    if family_role == "primary":
        values, metadata = _lds_signal(
            length,
            context_length,
            rng,
            period=period,
            persistence=effective_memory,
            spectral_concentration=spectral_concentration,
        )
    else:
        values, metadata = _spline_motif_signal(
            length,
            context_length,
            rng,
            period=period,
            spectral_concentration=spectral_concentration,
        )
    metadata["persistence_calibration"] = persistence
    metadata["seasonal_memory_calibration"] = seasonal_memory
    metadata["effective_memory_calibration"] = effective_memory
    metadata["raw_profile_dominant_period"] = raw_period
    metadata["effective_background_period"] = period
    metadata["background_period_bounds"] = [
        BACKGROUND_PERIOD_RANGE[0],
        min(BACKGROUND_PERIOD_RANGE[1], context_length / 3.0),
    ]
    return values, metadata


def _metadata(
    capability_id: str,
    family_role: FamilyRole,
    target: np.ndarray,
    detail: dict[str, Any],
) -> dict[str, Any]:
    family_id = (
        PRIMARY_FAMILY_BY_CAPABILITY[capability_id]
        if family_role == "primary"
        else SECONDARY_FAMILY_BY_CAPABILITY[capability_id]
    )
    return {
        "generator_version": GENERATOR_VERSION,
        "generator_family_role": family_role,
        "generator_family_id": family_id,
        "clean_latent_is_target": True,
        "future_process_noise_scale": 0.0,
        "observation_noise_scale": 0.0,
        "clean_latent_sha256": hashlib.sha256(
            np.ascontiguousarray(target).tobytes()
        ).hexdigest(),
        "predictability": {
            "construction_validated": True,
            "future_is_deterministic_given_history_parameters": True,
            "future_only_randomness": False,
        },
        **detail,
    }


def generate_deterministic_sample(
    capability_id: str,
    length: int,
    context_length: int,
    target_dim: int,
    season_length: int,
    intensity: int,
    rng: np.random.Generator,
    *,
    conditioning: GeneratorConditioning | None = None,
    family_role: FamilyRole = "primary",
    counterfactual_variant: int = 0,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    if family_role not in ("primary", "secondary"):
        raise ValueError("family_role must be primary or secondary")
    if capability_id == "trend":
        return _trend(length, context_length, target_dim, season_length, intensity, rng, conditioning, family_role)
    if capability_id == "multi_seasonal":
        return _multi_seasonal(length, context_length, target_dim, season_length, intensity, rng, conditioning, family_role)
    if capability_id == "time_varying_seasonality":
        return _time_varying(length, context_length, target_dim, season_length, intensity, rng, conditioning, family_role)
    if capability_id == "regime_switching":
        return _regime(length, context_length, target_dim, season_length, intensity, rng, conditioning, family_role)
    if capability_id == "nonlinear_persistence":
        return _nonlinear(length, context_length, target_dim, season_length, intensity, rng, conditioning, family_role)
    if capability_id == "predictable_intermittency":
        return _intermittent(length, context_length, target_dim, season_length, intensity, rng, conditioning, family_role)
    if capability_id == "common_factor":
        return _common_factor(
            length,
            context_length,
            target_dim,
            season_length,
            intensity,
            rng,
            conditioning,
            family_role,
            counterfactual_variant,
        )
    if capability_id == "hierarchical_coherence":
        return _hierarchy(length, context_length, target_dim, season_length, intensity, rng, conditioning, family_role)
    if capability_id == "cross_series_dependence":
        return _cross_series_dependence(
            length,
            context_length,
            target_dim,
            season_length,
            intensity,
            rng,
            conditioning,
            family_role,
            counterfactual_variant,
        )
    if capability_id == "covariate_response":
        return _covariate(
            length,
            context_length,
            target_dim,
            season_length,
            intensity,
            rng,
            conditioning,
            family_role,
            counterfactual_variant,
        )
    raise ValueError(f"unknown capability: {capability_id}")


def _trend(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    evidence_window = min(TREND_LOCAL_EVIDENCE_WINDOW, context)
    join_index = context - evidence_window
    coordinate = (
        np.arange(length, dtype=float) - float(join_index)
    ) / max(4, evidence_window)
    direction = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    slope_jitter = rng.uniform(0.75, 1.25, size=dim)
    # Direction still varies by seed, but the polynomial bends with the local
    # tangent.  Randomly bending against it caused some fixed realizations to
    # fold before the shared real q90 target and made the inverse undefined.
    curvature_sign = -np.ones(dim, dtype=float)
    profile_strength = _parameter(cond, "trend_strength_target", 0.15)
    slope_scale = _parameter(cond, "trend_slope_scale", 0.2)
    strength = (
        float(np.clip(_parameter(cond, "structure_scale", 1.0), 0.1, 2.0))
        * (0.03 + 0.24 * lam)
        * (0.5 + 2.0 * slope_scale)
        * (0.5 + profile_strength)
    )
    ratio = float(
        np.clip(
            _parameter(cond, "trend_curvature_ratio", 0.06),
            0.01,
            0.35,
        )
    )
    dose = float(np.clip(lam, 0.0, 1.0))
    if family == "primary":
        polynomial_degree = 2
        # The real-window curvature ratio remains useful provenance and
        # background metadata, but must not multiply the controlled dose a
        # second time.  A shared cap gives every fixed mechanism realization
        # enough support for the same dataset-level feature targets; the
        # per-realization inverse later absorbs the residual sign/jitter
        # variation.
        ratio_cap = 0.99
        basis_name = "c1_local_quadratic_with_linear_tangent_history"
    else:
        polynomial_degree = 3
        # At the end of H48, v=1.5.  A negative cubic coefficient therefore
        # retains at least 5.5% of the tangent slope even at the maximum cap:
        # 1 - 3 * 0.14 * 1.5**2 = 0.055.
        ratio_cap = 1.32
        basis_name = "c1_local_cubic_with_linear_tangent_history"
    effective_ratio = ratio_cap * (0.01 + 0.99 * dose)
    polynomial_coefficients = (
        curvature_sign * effective_ratio * slope_jitter
    )

    # The recent W96 evidence window and H48 forecast share one polynomial.
    # Earlier history is its tangent at v=0.  Values beyond the benchmark
    # horizon use the tangent at v=1+H/W, retaining prefix invariance for audit
    # calls that request a path longer than the formal H48 sample.
    design_horizon = min(
        TREND_DESIGN_HORIZON,
        max(1, length - context),
    )
    design_stop_coordinate = 1.0 + design_horizon / evidence_window
    basis = slope_jitter[None, :] * coordinate[:, None]
    local_mask = (
        (coordinate >= 0.0)
        & (coordinate <= design_stop_coordinate)
    )
    basis[local_mask] += (
        polynomial_coefficients[None, :]
        * coordinate[local_mask, None] ** polynomial_degree
    )
    post_mask = coordinate > design_stop_coordinate
    if np.any(post_mask):
        stop_value = (
            slope_jitter * design_stop_coordinate
            + polynomial_coefficients
            * design_stop_coordinate**polynomial_degree
        )
        stop_derivative = (
            slope_jitter
            + polynomial_degree
            * polynomial_coefficients
            * design_stop_coordinate ** (polynomial_degree - 1)
        )
        basis[post_mask] = (
            stop_value[None, :]
            + (
                coordinate[post_mask, None]
                - design_stop_coordinate
            )
            * stop_derivative[None, :]
        )

    endpoint_derivative_ratio = (
        1.0
        + polynomial_degree
        * curvature_sign
        * effective_ratio
        * design_stop_coordinate ** (polynomial_degree - 1)
    )
    minimum_derivative_ratio = np.minimum(
        1.0,
        endpoint_derivative_ratio,
    )
    target = strength * basis * direction[None, :]
    detail = {
        "trend_basis": basis_name,
        "trend_prehistory_law": "linear_tangent_at_local_join",
        "trend_postforecast_law": "linear_tangent_at_design_horizon",
        "trend_continuity_order": 1,
        "trend_local_evidence_window": evidence_window,
        "trend_join_index": join_index,
        "trend_join_coordinate": 0.0,
        "trend_design_horizon": design_horizon,
        "trend_design_stop_index": context + design_horizon,
        "trend_design_stop_coordinate": design_stop_coordinate,
        "trend_local_polynomial_degree": polynomial_degree,
        "trend_strength_parameter": strength,
        "curvature_ratio": ratio,
        "curvature_ratio_role": "anchor_nuisance_provenance_only",
        "trend_polynomial_ratio_cap": ratio_cap,
        "effective_curvature_ratio": effective_ratio,
        "effective_polynomial_ratio": effective_ratio,
        "local_polynomial_coefficient_by_target": (
            polynomial_coefficients.tolist()
        ),
        "minimum_tangent_derivative_ratio_by_target": (
            minimum_derivative_ratio.tolist()
        ),
        "design_endpoint_derivative_ratio_by_target": (
            endpoint_derivative_ratio.tolist()
        ),
        "slope_reversal_inside_design_window": bool(
            np.any(minimum_derivative_ratio <= 0.0)
        ),
        "direction_by_target": direction.tolist(),
        "slope_jitter_by_target": slope_jitter.tolist(),
        "curvature_sign_by_target": curvature_sign.tolist(),
        "deterministic_texture": None,
    }
    return target, _metadata("trend", family, target, detail), None


def _multi_seasonal(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    time = np.arange(length, dtype=float)
    evidence_window = min(SHORTEST_SUPPORTED_CONTEXT, context)
    primary_period_upper = min(
        MULTI_SEASONAL_PRIMARY_PERIOD_RANGE[1],
        evidence_window / 3.0,
    )
    component_period_upper = min(
        MULTI_SEASONAL_COMPONENT_PERIOD_RANGE[1],
        evidence_window / 2.0,
    )
    raw_period, primary_period = _profile_period(
        cond,
        float(season),
        lower=MULTI_SEASONAL_PRIMARY_PERIOD_RANGE[0],
        upper=primary_period_upper,
    )
    ratio = _parameter(cond, "secondary_component_ratio", 0.45)
    target = np.zeros((length, dim), dtype=float)
    periods: list[float] = []
    if family == "primary":
        candidate_ratios = np.asarray(
            [
                rng.uniform(0.52, 0.88),
                rng.uniform(1.25, 2.35),
            ],
            dtype=float,
        )
        periods = [
            primary_period,
            *np.clip(
                primary_period * candidate_ratios,
                MULTI_SEASONAL_COMPONENT_PERIOD_RANGE[0],
                component_period_upper,
            ).tolist(),
        ]
        controlled_ratio = 0.03 + 6.0 * float(np.clip(lam, 0.0, 1.0))
        amplitudes = [1.0, controlled_ratio, 0.6 * controlled_ratio]
        for period, amplitude in zip(periods, amplitudes, strict=True):
            phase = rng.uniform(0.0, 2.0 * np.pi, size=dim)
            amplitude_jitter = rng.uniform(0.85, 1.15, size=dim)
            target += (
                amplitude
                * amplitude_jitter[None, :]
                * np.sin(
                    2.0 * np.pi * time[:, None] / period
                    + phase[None, :]
                )
            )
        law = "sample_specific_resolved_fourier_components"
    else:
        concentration = _parameter(
            cond,
            "profile_spectral_concentration",
            0.25,
        )
        base_concentration = max(concentration, 0.90)
        motif, motif_meta = _spline_motif_signal(
            length,
            context,
            rng,
            period=primary_period,
            spectral_concentration=base_concentration,
        )
        second, second_meta = _spline_motif_signal(
            length,
            context,
            rng,
            period=float(
                np.clip(
                    primary_period * float(rng.uniform(1.35, 2.35)),
                    MULTI_SEASONAL_COMPONENT_PERIOD_RANGE[0],
                    component_period_upper,
                )
            ),
            spectral_concentration=concentration,
        )
        # Keep the strongest controlled component on an exact context DFT bin
        # in a short, resolved band. A free period leaks energy across bins and
        # can make an otherwise strong component look weak after the real
        # anchor's carrier bin is excluded by the feature definition.
        candidate_cycles = np.asarray([48, 56, 64], dtype=int)
        eligible_cycles = candidate_cycles[
            (context / candidate_cycles)
            >= MULTI_SEASONAL_COMPONENT_PERIOD_RANGE[0]
        ]
        if eligible_cycles.size == 0:
            eligible_cycles = np.asarray(
                [
                    max(
                        2,
                        int(
                            np.floor(
                                context
                                / MULTI_SEASONAL_COMPONENT_PERIOD_RANGE[0]
                            )
                        ),
                    )
                ],
                dtype=int,
            )
        third_cycles = int(rng.choice(eligible_cycles))
        third_period = float(context / third_cycles)
        third_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        third = np.sin(
            2.0 * np.pi * time / third_period + third_phase
        )
        load = rng.uniform(0.85, 1.15, size=dim)
        controlled_ratio = 0.03 + 10.0 * float(np.clip(lam, 0.0, 1.0))
        target = (
            motif[:, None] * load[None, :]
            + controlled_ratio * second[:, None]
            + 2.20 * controlled_ratio * third[:, None]
        )
        periods = [
            float(motif_meta["period"]),
            float(second_meta["period"]),
            third_period,
        ]
        law = "sample_specific_periodic_spline_superposition"
    detail = {
        "periods": periods,
        "component_ratio": ratio,
        "component_ratio_role": "anchor_nuisance_provenance_only",
        "controlled_component_ratio": controlled_ratio,
        "mechanism_law": law,
        "raw_profile_dominant_period": raw_period,
        "effective_primary_period": primary_period,
        "minimum_supported_context_length": SHORTEST_SUPPORTED_CONTEXT,
        "period_evidence_window": evidence_window,
        "primary_period_bounds": [
            MULTI_SEASONAL_PRIMARY_PERIOD_RANGE[0],
            primary_period_upper,
        ],
        "component_period_bounds": [
            MULTI_SEASONAL_COMPONENT_PERIOD_RANGE[0],
            component_period_upper,
        ],
        "cycles_in_shortest_evidence_window": [
            float(evidence_window / max(period, 1e-12))
            for period in periods
        ],
        "controlled_component_cycles_per_context": (
            third_cycles if family != "primary" else None
        ),
    }
    return target, _metadata("multi_seasonal", family, target, detail), None


def _time_varying(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    time = np.arange(length, dtype=float)
    evidence_window = min(SHORTEST_SUPPORTED_CONTEXT, context)
    carrier_period_upper = min(
        TIME_VARYING_CARRIER_PERIOD_RANGE[1],
        evidence_window / 3.0,
    )
    raw_period, base_period = _profile_period(
        cond,
        float(season),
        lower=TIME_VARYING_CARRIER_PERIOD_RANGE[0],
        upper=carrier_period_upper,
    )
    period = float(
        round(
            np.clip(
                base_period * rng.uniform(0.85, 1.15),
                TIME_VARYING_CARRIER_PERIOD_RANGE[0],
                carrier_period_upper,
            )
        )
    )
    modulation_period_lower = max(
        TIME_VARYING_MODULATION_PERIOD_RANGE[0],
        2.0 * period,
    )
    modulation_period_upper = min(
        TIME_VARYING_MODULATION_PERIOD_RANGE[1],
        float(evidence_window),
    )
    modulation_period = float(
        np.clip(
            period * rng.uniform(2.2, 4.0),
            modulation_period_lower,
            modulation_period_upper,
        )
    )
    carrier_phase = rng.uniform(0.0, 2.0 * np.pi, size=dim)
    modulation_phase = rng.uniform(0.0, 2.0 * np.pi, size=dim)
    anchor_depth = _parameter(cond, "modulation_depth_scale", 0.35)
    depth = 1.80 * float(np.clip(lam, 0.0, 1.0))
    if family == "secondary":
        depth *= 1.10
    phase_scale = _parameter(cond, "phase_variation_scale", 0.08)
    angle = 2.0 * np.pi * time[:, None] / modulation_period + modulation_phase[None, :]
    if family == "primary":
        harmonic_weight = float(rng.uniform(0.10, 0.30))
        harmonic_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        modulation = np.sin(angle) + harmonic_weight * np.sin(
            2.0 * angle + harmonic_phase
        )
        phase = (0.04 + phase_scale) * depth * modulation
        law = "smooth_amplitude_and_phase_modulated_oscillator"
    else:
        harmonic_weight = 0.0
        harmonic_phase = 0.0
        fractional = ((angle / (2.0 * np.pi)) % 1.0)
        modulation = 1.0 - 4.0 * np.abs(fractional - 0.5)
        chirp_power = float(rng.uniform(1.7, 2.4))
        phase = (
            depth
            * (0.10 + phase_scale)
            * (time[:, None] / max(context, 1)) ** chirp_power
        )
        law = "triangular_amplitude_modulation_with_chirp"
    target = (1.0 + depth * modulation) * np.sin(
        2.0 * np.pi * time[:, None] / period + carrier_phase[None, :] + phase
    )
    detail = {
        "primary_period": period,
        "modulation_period": modulation_period,
        "modulation_depth": depth,
        "anchor_modulation_depth_scale": anchor_depth,
        "anchor_modulation_depth_role": "nuisance_provenance_only",
        "modulation_harmonic_weight": harmonic_weight,
        "modulation_harmonic_phase": harmonic_phase,
        "mechanism_law": law,
        "raw_profile_dominant_period": raw_period,
        "effective_carrier_base_period": base_period,
        "minimum_supported_context_length": SHORTEST_SUPPORTED_CONTEXT,
        "period_evidence_window": evidence_window,
        "carrier_period_bounds": [
            TIME_VARYING_CARRIER_PERIOD_RANGE[0],
            carrier_period_upper,
        ],
        "modulation_period_bounds": [
            modulation_period_lower,
            modulation_period_upper,
        ],
        "carrier_cycles_in_shortest_evidence_window": float(
            evidence_window / period
        ),
        "modulation_cycles_in_shortest_evidence_window": float(
            evidence_window / modulation_period
        ),
    }
    return target, _metadata("time_varying_seasonality", family, target, detail), None


def _duration_schedule(
    length: int,
    context: int,
    pattern: list[int],
    anchor_offset: int,
    initial_state: float,
) -> tuple[np.ndarray, list[int]]:
    if not pattern or any(duration < 2 for duration in pattern):
        raise ValueError("duration pattern must contain positive durations")
    anchor = context + anchor_offset
    cuts = [anchor]
    cursor = anchor
    index = -1
    while cursor - pattern[index % len(pattern)] > 0:
        cursor -= pattern[index % len(pattern)]
        cuts.append(cursor)
        index -= 1
    cursor = anchor
    index = 0
    while cursor + pattern[index % len(pattern)] < length:
        cursor += pattern[index % len(pattern)]
        cuts.append(cursor)
        index += 1
    cuts = sorted(cut for cut in cuts if 0 < cut < length)
    state = np.ones(length, dtype=float)
    for segment, (start, end) in enumerate(
        zip([0, *cuts], [*cuts, length], strict=True)
    ):
        state[start:end] = initial_state * (-1.0 if segment % 2 else 1.0)
    return state, cuts


def _sample_duration_motif(
    reference_period: float,
    scale: float,
    rng: np.random.Generator,
) -> tuple[list[int], int]:
    base = int(
        round(
            np.clip(
                0.65 * max(4.0, reference_period) * scale,
                REGIME_DWELL_RANGE[0],
                REGIME_DWELL_RANGE[1],
            )
        )
    )
    motif_length = int(rng.integers(5, 8))
    # Cover a real range of short/long dwell times in every realization.
    # Independent uniforms frequently rounded to an all-equal motif, turning
    # the task into an ordinary square-wave seasonality that the feature
    # extractor correctly removed.  A shuffled stratified motif retains seed
    # diversity while making the regime mechanism non-degenerate.
    offsets = np.linspace(-0.38, 0.38, motif_length)
    rng.shuffle(offsets)
    multipliers = 1.0 + offsets + rng.uniform(
        -0.04,
        0.04,
        size=motif_length,
    )
    pattern = np.clip(
        np.rint(base * multipliers).astype(int),
        REGIME_DWELL_RANGE[0],
        REGIME_DWELL_RANGE[1],
    )
    return pattern.tolist(), base


def _regime(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    raw_period, reference_period = _profile_period(
        cond,
        float(season),
        lower=BACKGROUND_PERIOD_RANGE[0],
        upper=min(BACKGROUND_PERIOD_RANGE[1], context / 3.0),
    )
    # Generate a suffix beyond the requested horizon so that smooth
    # transitions near the right boundary do not depend on requested length.
    pattern, base_dwell = _sample_duration_motif(
        reference_period,
        _parameter(cond, "regime_dwell_scale", 1.0),
        rng,
    )
    schedule_length = length + 8 * max(pattern)
    anchor_offset = int(
        rng.integers(2, max(3, min(int(round(np.median(pattern))), 32)))
    )
    initial_state = float(rng.choice(np.asarray([-1.0, 1.0])))
    state, cuts = _duration_schedule(
        schedule_length,
        context,
        pattern,
        anchor_offset,
        initial_state,
    )
    if family == "secondary":
        time = np.arange(schedule_length, dtype=float)
        clock_period = max(
            8.0,
            rng.uniform(1.7, 2.4) * float(np.median(pattern)),
        )
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        oscillator = np.sin(2.0 * np.pi * time / clock_period + phase)
        secondary_ratio = float(rng.uniform(math.sqrt(1.5), math.sqrt(2.5)))
        secondary_weight = float(rng.uniform(0.25, 0.45))
        oscillator += secondary_weight * np.sin(
            2.0 * np.pi * time / (clock_period * secondary_ratio)
            + rng.uniform(0.25, 0.75) * phase
        )
        state = np.where(oscillator >= 0.0, 1.0, -1.0)
        cuts = (
            np.flatnonzero(np.diff(state) != 0.0) + 1
        ).astype(int).tolist()
        transition = "hard_thresholded_quasiperiodic_oscillator"
    else:
        transition = "hard"
    level_amplitude_pattern = np.linspace(
        0.60,
        1.40,
        len(pattern),
    )
    rng.shuffle(level_amplitude_pattern)
    for segment, (start, end) in enumerate(
        zip([0, *cuts], [*cuts, schedule_length], strict=True)
    ):
        state[start:end] *= level_amplitude_pattern[
            segment % len(level_amplitude_pattern)
        ]
    anchor_level_scale = _parameter(cond, "regime_level_scale", 0.8)
    anchor_sparse_scale = _parameter(cond, "regime_sparse_scale", 0.25)
    strength = 0.10 * float(np.clip(lam, 0.0, 1.0))
    if family == "secondary":
        strength *= 2.0
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    time = np.arange(length, dtype=float)
    texture_primary_period = 8.0
    texture_period_ratio = math.sqrt(2.0)
    texture_secondary_period = (
        texture_primary_period * texture_period_ratio
    )
    texture_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    texture_secondary_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    texture = (
        np.sin(
            2.0 * np.pi * time / texture_primary_period
            + texture_phase
        )
        + 0.25
        * np.sin(
            2.0 * np.pi * time / texture_secondary_period
            + texture_secondary_phase
        )
    )
    texture_meta = {
        "family": "smooth_incommensurate_two_tone_regime_background",
        "law": "deterministic_quasiperiodic_two_tone",
        "primary_period": texture_primary_period,
        "primary_phase": texture_phase,
        "secondary_period": texture_secondary_period,
        "secondary_phase": texture_secondary_phase,
        "period_ratio": texture_period_ratio,
        "secondary_weight": 0.25,
        "future_process_noise_scale": 0.0,
    }
    state = state[:length]
    target = strength * state[:, None] * signs[None, :] + 0.04 * texture[:, None]
    detail = {
        "cut_points": [cut for cut in cuts if cut < length],
        "dwell_pattern": pattern,
        "level_amplitude_pattern": level_amplitude_pattern.tolist(),
        "dwell_length": int(round(np.median(pattern))),
        "dwell_base": base_dwell,
        "dwell_bounds": list(REGIME_DWELL_RANGE),
        "dwell_anchor_offset": anchor_offset,
        "initial_regime_state": initial_state,
        "regime_strength": strength,
        "anchor_regime_level_scale": anchor_level_scale,
        "anchor_regime_sparse_scale": anchor_sparse_scale,
        "anchor_regime_strength_role": "nuisance_provenance_only",
        "transition": transition,
        "deterministic_texture": texture_meta,
        "raw_profile_dominant_period": raw_period,
        "effective_regime_reference_period": reference_period,
    }
    return target, _metadata("regime_switching", family, target, detail), None


def _nonlinear(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    raw_period, reference_period = _profile_period(
        cond,
        float(season),
        lower=NONLINEAR_SEASONAL_LAG_RANGE[0],
        upper=NONLINEAR_SEASONAL_LAG_RANGE[1],
    )
    seasonal_lag = int(round(reference_period))
    calibrated_lag = max(
        NONLINEAR_LAG_RANGE[0],
        min(
            NONLINEAR_LAG_RANGE[1],
            int(
                round(
                    seasonal_lag
                    * _parameter(
                        cond,
                        "nonlinear_lag_scale",
                        1.0 / 3.0,
                    )
                )
            ),
        ),
    )
    lag = int(
        np.clip(
            round(calibrated_lag * rng.uniform(0.80, 1.20)),
            NONLINEAR_LAG_RANGE[0],
            NONLINEAR_LAG_RANGE[1],
        )
    )
    burn = max(256, 8 * seasonal_lag)
    total = burn + length
    state = np.zeros((total, dim), dtype=float)
    state[:seasonal_lag] = rng.normal(0.0, 0.65, size=(seasonal_lag, dim))
    forcing, forcing_meta = _calibrated_signal(
        total,
        burn + context,
        rng,
        cond,
        "primary",
        period_multiplier=1.25,
    )
    gain = _parameter(cond, "nonlinear_gain_scale", 2.1) * (
        0.08 + 0.72 * lam
    )
    effective_gain = gain
    if family == "primary":
        persistence_weight = 0.58
        seasonal_weight = 0.10
        forcing_weight = 0.18
        transform = "centered_rational_quadratic"

        def nonlinear_response(values):
            squared = values * values
            return squared / (1.0 + squared) - 0.35

    else:
        persistence_weight = 0.52
        seasonal_weight = 0.14
        forcing_weight = 0.18
        transform = "centered_tanh_quadratic"

        def nonlinear_response(values):
            transformed = np.tanh(values)
            return transformed * transformed - 0.30

    clipped_state_value_count = 0
    generated_state_value_count = 0
    for index in range(seasonal_lag, total):
        delayed = state[index - lag]
        response = nonlinear_response(delayed)
        next_value = (
            persistence_weight * state[index - 1]
            + seasonal_weight * state[index - seasonal_lag]
            + effective_gain * response
            + forcing_weight * forcing[index]
        )
        clipped_state_value_count += int(
            np.count_nonzero(np.abs(next_value) > 5.0)
        )
        generated_state_value_count += int(np.size(next_value))
        state[index] = np.clip(next_value, -5.0, 5.0)

    audit_start = max(burn, seasonal_lag, lag)
    audit_stop = burn + context
    delayed_history = state[
        audit_start - lag : audit_stop - lag
    ].reshape(-1)
    response_history = nonlinear_response(delayed_history)
    response_design = np.column_stack(
        [np.ones_like(delayed_history), delayed_history]
    )
    response_linear_fit = response_design @ np.linalg.lstsq(
        response_design,
        response_history,
        rcond=None,
    )[0]
    response_variance = float(np.var(response_history))
    response_curvature_fraction = float(
        np.mean((response_history - response_linear_fit) ** 2)
        / max(response_variance, 1e-12)
    )
    raw_recurrence_residual = (
        state[audit_start:audit_stop]
        - persistence_weight * state[audit_start - 1 : audit_stop - 1]
        - seasonal_weight
        * state[
            audit_start - seasonal_lag : audit_stop - seasonal_lag
        ]
    )
    nonlinear_effect = effective_gain * response_history
    effect_to_residual_std_ratio = float(
        np.std(nonlinear_effect)
        / max(float(np.std(raw_recurrence_residual)), 1e-12)
    )

    target = _standardize_history(state[burn:], context)
    detail = {
        "nonlinear_transform": transform,
        "nonlinear_lag": lag,
        "calibrated_nonlinear_lag": calibrated_lag,
        "seasonal_lag": seasonal_lag,
        "seasonal_lag_bounds": list(NONLINEAR_SEASONAL_LAG_RANGE),
        "nonlinear_lag_bounds": list(NONLINEAR_LAG_RANGE),
        "raw_profile_dominant_period": raw_period,
        "effective_nonlinear_reference_period": reference_period,
        "nonlinear_strength": effective_gain,
        "unscaled_nonlinear_strength": gain,
        "persistence_weight": persistence_weight,
        "seasonal_weight": seasonal_weight,
        "forcing_weight": forcing_weight,
        "nonlinear_response_order": 2,
        "nonlinear_response_curvature_fraction": (
            response_curvature_fraction
        ),
        "nonlinear_effect_to_recurrence_residual_std_ratio": (
            effect_to_residual_std_ratio
        ),
        "state_clip_fraction": float(
            clipped_state_value_count
            / max(generated_state_value_count, 1)
        ),
        "state_clip_value_count": clipped_state_value_count,
        "burn_in_steps": burn,
        "recurrence_amplitude": 1.0,
        "deterministic_forcing": forcing_meta,
    }
    return target, _metadata("nonlinear_persistence", family, target, detail), None


def _event_clock(
    length: int,
    context: int,
    pattern: list[int],
    anchor_offset: int,
) -> list[int]:
    anchor = context + anchor_offset
    centers = [anchor]
    cursor = anchor
    index = -1
    while cursor - pattern[index % len(pattern)] >= 0:
        cursor -= pattern[index % len(pattern)]
        centers.append(cursor)
        index -= 1
    cursor = anchor
    index = 0
    while cursor + pattern[index % len(pattern)] < length:
        cursor += pattern[index % len(pattern)]
        centers.append(cursor)
        index += 1
    return sorted(centers)


def _intermittent(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    raw_period, _effective_period = _profile_period(
        cond,
        float(season),
        lower=float(INTERMITTENT_EVENT_PERIOD_RANGE[0]),
        upper=float(INTERMITTENT_EVENT_PERIOD_RANGE[1]),
    )
    # The event-clock feature is parameterized by the public feature period,
    # so the actuator must repeat on that same clock.  The real-data dominant
    # spectral period remains nuisance provenance only.
    event_period = int(np.clip(round(season), 4, 56))
    # The public observable fits a three-period event clock.  Preserve
    # irregularity within the motif, but make the three intervals sum exactly
    # to that observable clock instead of drawing an unrelated 3--6 interval
    # grammar that the real-data coordinate cannot recover.
    if family == "primary":
        proportions = np.asarray([0.75, 1.00, 1.25], dtype=float)
    else:
        proportions = np.asarray([0.65, 1.15, 1.20], dtype=float)
    interval_pattern_array = np.maximum(
        2,
        np.rint(event_period * proportions).astype(int),
    )
    interval_pattern_array[-1] += (
        3 * event_period - int(np.sum(interval_pattern_array))
    )
    interval_pattern = interval_pattern_array.tolist()
    anchor_offset = int(
        rng.integers(3, max(4, min(event_period, 40)))
    )
    # Include event centers beyond the requested end because smooth pulse tails
    # are part of the same infinite deterministic clock.
    centers = _event_clock(
        length + 8 * event_period,
        context,
        interval_pattern,
        anchor_offset,
    )
    width = max(
        0.75,
        event_period * _parameter(cond, "event_width_scale", 0.12),
    )
    time = np.arange(length, dtype=float)
    pulse = np.zeros(length, dtype=float)
    for center in centers:
        distance = np.abs(time - center)
        if family == "primary":
            pulse += np.exp(-0.5 * (distance / width) ** 2)
            shape = "gaussian"
        else:
            support = max(2.0 * width, 1.0)
            compact = distance <= support
            pulse[compact] += 0.5 * (
                1.0 + np.cos(np.pi * distance[compact] / support)
            )
            shape = "compact_raised_cosine"
    strength = (
        float(np.clip(_parameter(cond, "structure_scale", 1.0), 0.1, 2.0))
        * _parameter(cond, "event_base_scale", 0.2)
        * (0.02 + 4.00 * lam)
    )
    loading = rng.uniform(0.9, 1.1, size=dim)
    texture_period = event_period * math.sqrt(2.0)
    texture_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    texture = np.sin(
        2.0 * np.pi * time / texture_period + texture_phase
    )
    texture_meta = {
        "law": "incommensurate_deterministic_sinusoid",
        "period": texture_period,
        "phase": texture_phase,
        "future_process_noise_scale": 0.0,
    }
    event_component = (
        strength * pulse[:, None] * loading[None, :]
    )
    texture_component = 0.03 * texture[:, None]
    target = event_component + texture_component
    event_energy = float(
        np.mean(np.square(event_component[:context]))
    )
    texture_energy = float(
        np.mean(np.square(texture_component[:context]))
    )
    event_effect_energy_share = float(
        event_energy / max(event_energy + texture_energy, 1e-12)
    )
    detail = {
        "pulse_centers": [center for center in centers if center < length],
        "pulse_interval_pattern": interval_pattern,
        "event_period": event_period,
        "event_period_bounds": [4, 56],
        "event_period_source": "public_feature_period",
        "raw_profile_dominant_period": raw_period,
        "pulse_anchor_offset": anchor_offset,
        "pulse_width": width,
        "pulse_shape": shape,
        "pulse_strength": strength,
        "event_component_history_energy": event_energy,
        "background_component_history_energy": texture_energy,
        "event_effect_energy_share": event_effect_energy_share,
        "event_dose_semantics": (
            "history_event_component_energy_divided_by_event_plus_"
            "deterministic_texture_energy"
        ),
        "deterministic_texture": texture_meta,
    }
    return target, _metadata("predictable_intermittency", family, target, detail), None


def _orthonormal_factor_basis(
    width: int,
    rng: np.random.Generator,
    family: FamilyRole,
    *,
    factor_persistence: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if width < 8:
        raise ValueError("common-factor response width must be at least eight")
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    time = (np.arange(width, dtype=float) + 0.5) / width
    persistence = float(np.clip(factor_persistence, -0.2, 0.995))
    base_cycles = float(
        np.clip(1.55 - 0.80 * persistence, 0.65, 1.70)
    )
    if family == "primary":
        second_cycle_ratio = float(rng.uniform(1.35, 1.85))
        first = np.sin(
            2.0 * np.pi * base_cycles * time + phase
        )
        second = np.cos(
            2.0
            * np.pi
            * base_cycles
            * second_cycle_ratio
            * time
            - 0.5 * phase
        )
        law = "sample_specific_two_mode_fourier_basis"
    else:
        first_center = float(rng.uniform(0.28, 0.42))
        second_center = float(rng.uniform(0.62, 0.78))
        persistence_width_scale = float(
            np.clip(0.75 + 0.35 * persistence, 0.68, 1.10)
        )
        first_width = float(
            rng.uniform(0.16, 0.25) * persistence_width_scale
        )
        second_width = float(
            rng.uniform(0.16, 0.25) * persistence_width_scale
        )
        second_cycle_ratio = None

        def compact_cosine(center: float, radius: float) -> np.ndarray:
            distance = np.abs(time - center)
            values = np.zeros(width, dtype=float)
            active = distance <= radius
            values[active] = 0.5 * (
                1.0 + np.cos(np.pi * distance[active] / radius)
            )
            values -= float(np.mean(values))
            return values

        first = compact_cosine(first_center, first_width)
        second = compact_cosine(second_center, second_width)
        law = "sample_specific_compact_spline_pulse_basis"
    first -= float(np.mean(first))
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second -= float(np.mean(second))
    second -= float(np.dot(second, first)) * first
    second /= max(float(np.linalg.norm(second)), 1e-12)
    basis = math.sqrt(width) * np.column_stack([first, second])
    return basis, {
        "law": law,
        "phase": phase,
        "factor_persistence_parameter": persistence,
        "base_cycles_over_horizon": base_cycles,
        "second_cycle_ratio": second_cycle_ratio,
        "basis_gram": (basis.T @ basis / width).tolist(),
    }


def _common_factor(
    length,
    context,
    dim,
    season,
    intensity,
    rng,
    cond,
    family,
    counterfactual_variant,
):
    """Generate a dense low-rank state observed through multiple channels.

    The ordinary path is a conventional dynamic-factor model: every channel is
    a noisy projection of one shared state plus an idiosyncratic local mode.
    The strict pair changes the amplitude of that same state in the final L48
    auxiliary observations and throughout the future.  The protected channel's
    history is left unchanged, so its future can only be distinguished by
    filtering the shared state from the other channels.

    There is deliberately no privileged source channel, directional edge, lag,
    codebook, or generator-private episode grammar.  A blind history-only
    dynamic-factor/state-space learner can infer the loading vector and state
    dynamics from the ordinary prefix.
    """

    if dim < 3:
        raise ValueError("common_factor requires at least three targets")
    variant = int(counterfactual_variant)
    if variant not in {0, 1}:
        raise ValueError("counterfactual_variant must be 0 or 1")
    lam = _lambda(cond, intensity)
    horizon = min(48, max(1, length - context))
    factor_persistence = _parameter(
        cond,
        "factor_persistence",
        0.80,
    )
    protected_target = int(rng.integers(0, dim))

    sign_pattern = np.where(np.arange(dim) % 2 == 0, 1.0, -1.0)
    sign_pattern = sign_pattern[rng.permutation(dim)]
    response_loadings = sign_pattern * rng.uniform(0.80, 1.20, size=dim)
    response_loadings /= math.sqrt(
        float(np.mean(response_loadings * response_loadings))
    )
    target_shared = float(
        np.clip(
            _parameter(cond, "shared_variance_target", 0.8),
            0.35,
            0.995,
        )
    )

    local_spread = _parameter(cond, "local_mode_spread", 0.45)
    local_period_multipliers = [
        (1.20 + local_spread * (index + 1))
        * float(rng.uniform(0.85, 1.15))
        for index in range(dim)
    ]
    local = np.column_stack(
        [
            _calibrated_signal(
                length,
                context,
                rng,
                cond,
                family,
                persistence_name="local_persistence",
                period_multiplier=local_period_multipliers[index],
            )[0]
            for index in range(dim)
        ]
    )
    # Independent observation texture makes state filtering genuinely useful:
    # one channel is a noisy sensor, whereas the shared rank-one projection is
    # reinforced across the panel.  The history texture is drawn from the
    # paired seed and removed at the forecast boundary; the scored future is a
    # clean deterministic latent continuation shared by both pair members.
    local_texture_seed = int(rng.integers(0, 2**63 - 1))
    local_texture_rng = np.random.default_rng(local_texture_seed)
    local_texture = local_texture_rng.normal(size=(length, dim))
    local_texture[1:] = (
        0.20 * local_texture[:-1] + local_texture[1:]
    )
    local_texture /= np.maximum(
        np.std(local_texture[:context], axis=0),
        1e-9,
    )[None, :]
    local_texture[context:] = 0.0
    local = 0.35 * local + 0.65 * local_texture
    # Keep the I1 common-factor baseline genuinely weak.  Independently drawn
    # smooth local modes can still be highly correlated on a finite L336
    # history, which made some paths start above the real q10 target before
    # any common factor was injected.  History-fitted Gram-Schmidt removes
    # only that accidental shared subspace and applies the same deterministic
    # projection to the future.
    for channel in range(1, dim):
        design = local[:context, :channel]
        coefficients = np.linalg.lstsq(
            design,
            local[:context, channel],
            rcond=None,
        )[0]
        local[:, channel] -= local[:, :channel] @ coefficients
    local_scale = np.std(local[:context], axis=0)
    local /= np.maximum(local_scale, 1e-9)[None, :]

    # The calibrated smooth process supplies real-data nuisance texture.  A
    # plainly periodic state supplies a blind-learnable continuation that is
    # present in the ordinary prefix, the evidence suffix, and the future.
    dense_factor, dense_factor_meta = _calibrated_signal(
        length,
        context,
        rng,
        cond,
        family,
        persistence_name="factor_persistence",
        period_multiplier=float(rng.uniform(0.85, 1.15)),
    )
    state_period = int(rng.choice(np.asarray([12, 16, 24, 32])))
    state_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    state_time = np.arange(length, dtype=float)
    observable_state = (
        np.sin(2.0 * np.pi * state_time / state_period + state_phase)
        + 0.28
        * np.sin(
            4.0 * np.pi * state_time / state_period
            - 0.35 * state_phase
        )
    )
    observable_state -= float(np.mean(observable_state[:context]))
    observable_state /= max(
        float(np.std(observable_state[:context])),
        1e-9,
    )
    dense_factor -= float(np.mean(dense_factor[:context]))
    dense_factor /= max(float(np.std(dense_factor[:context])), 1e-9)
    shared_state = 0.20 * dense_factor + 0.80 * observable_state
    shared_state /= max(float(np.std(shared_state[:context])), 1e-9)

    local_strength = float(
        np.clip(
            0.72 + 0.12 * (0.65 - target_shared),
            0.62,
            0.82,
        )
    )
    dense_factor_strength = float(0.02 + 1.60 * lam)
    target = (
        local_strength * local
        + dense_factor_strength
        * shared_state[:, None]
        * response_loadings[None, :]
    )

    evidence_width = min(48, context - 16)
    evidence_start = context - evidence_width
    perturbation_scale = float(
        (0.20 + 0.85 * lam) * rng.uniform(0.90, 1.10)
    )
    member_sign = -0.5 if variant == 0 else 0.5
    auxiliary_targets = [
        index for index in range(dim) if index != protected_target
    ]
    target[evidence_start:context, auxiliary_targets] += (
        member_sign
        * perturbation_scale
        * observable_state[evidence_start:context, None]
        * response_loadings[None, auxiliary_targets]
    )
    future_stop = min(length, context + horizon)
    target[context:future_stop] += (
        member_sign
        * perturbation_scale
        * observable_state[context:future_stop, None]
        * response_loadings[None, :]
    )

    detail = {
        "factor_rank": 1,
        "latent_state_dimension": 1,
        "factor_persistence": factor_persistence,
        "dense_factor_process": dense_factor_meta,
        "dense_factor_strength": dense_factor_strength,
        "shared_state_period": state_period,
        "shared_state_phase": state_phase,
        "shared_state_dense_process_weight": 0.20,
        "shared_state_periodic_process_weight": 0.80,
        "response_loadings": response_loadings.tolist(),
        "loadings": response_loadings.tolist(),
        "loading_rms": float(
            np.sqrt(np.mean(response_loadings**2))
        ),
        "minimum_supported_context_length": (
            SHORTEST_SUPPORTED_CONTEXT
        ),
        # Retain the generic slice key used by pair-shared normalization.  It
        # now denotes the observable-state evidence suffix, not a hidden code.
        "final_code_slice": [evidence_start, context],
        "shared_state_evidence_slice": [evidence_start, context],
        "shared_state_evidence_width": evidence_width,
        "protected_target_index": protected_target,
        "auxiliary_target_indices": auxiliary_targets,
        "counterfactual_variant": variant,
        "counterfactual_perturbation_scale": perturbation_scale,
        "counterfactual_protected_history_invariant": True,
        "counterfactual_future_is_shared_state_determined": True,
        "joint_observability_law": (
            "dense_rank1_state_filtered_from_symmetric_multichannel_observations"
        ),
        "local_amplitude": local_strength,
        "local_period_multipliers": local_period_multipliers,
        "local_observation_texture_weight": 0.65,
        "local_smooth_nuisance_weight": 0.35,
        "local_observation_texture_ar1": 0.20,
        "local_observation_texture_seed": local_texture_seed,
        "local_observation_texture_history_only": True,
        "future_local_observation_texture_scale": 0.0,
        "local_nuisance_path_pair_invariant": True,
        "local_factor_loading_orthogonalized": True,
        "main_task_is_dense_dynamic_factor": True,
        "directional_driver_present": False,
        "channel_specific_lag_present": False,
        "generator_private_codebook_present": False,
        "strict_counterfactual_role": "i5_subset_audit",
        "future_only_shock_count": 0,
    }
    return target, _metadata("common_factor", family, target, detail), None


def _hierarchy(length, context, dim, season, intensity, rng, cond, family):
    if dim < 3:
        raise ValueError("hierarchical_coherence requires parent and children")
    child_count = dim - 1
    lam = _lambda(cond, intensity)
    aggregate, aggregate_meta = _calibrated_signal(
        length,
        context,
        rng,
        cond,
        family,
        persistence_name="aggregate_persistence",
        seasonal_memory_name="aggregate_seasonal_memory",
        period_multiplier=float(rng.uniform(0.85, 1.15)),
    )
    contrast_rank = child_count - 1
    contrast_period_multipliers = [
        (1.55 + 0.35 * index) * float(rng.uniform(0.85, 1.15))
        for index in range(contrast_rank)
    ]
    independent = np.column_stack(
        [
            _calibrated_signal(
                length,
                context,
                rng,
                cond,
                family,
                persistence_name="contrast_persistence",
                seasonal_memory_name="contrast_seasonal_memory",
                period_multiplier=contrast_period_multipliers[index],
            )[0]
            for index in range(contrast_rank)
        ]
    )
    corr = _parameter(cond, "aggregate_contrast_abs_corr", 0.3)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=contrast_rank)
    scores = math.sqrt(max(0.0, 1.0 - corr * corr)) * independent
    scores += corr * aggregate[:, None] * signs[None, :]
    scores = _standardize_history(scores, context)
    basis = np.zeros((child_count, contrast_rank), dtype=float)
    for column in range(contrast_rank):
        prefix = column + 1
        denominator = math.sqrt(prefix * (prefix + 1))
        basis[:prefix, column] = 1.0 / denominator
        basis[prefix, column] = -prefix / denominator
    rotation_raw = rng.normal(size=(contrast_rank, contrast_rank))
    rotation, triangular = np.linalg.qr(rotation_raw)
    diagonal_sign = np.sign(np.diag(triangular))
    diagonal_sign[diagonal_sign == 0.0] = 1.0
    rotation *= diagonal_sign[None, :]
    basis = basis @ rotation
    child_permutation = rng.permutation(child_count)
    basis = basis[child_permutation]
    ratio = _parameter(cond, "contrast_to_aggregate_ratio", 0.3)
    structure_scale = float(np.clip(_parameter(cond, "structure_scale", 1.0), 0.1, 2.0))
    calibrated_heterogeneity = _parameter(
        cond,
        "hierarchy_heterogeneity_scale",
        0.2,
    )
    heterogeneity = (
        1.5
        * math.sqrt(max(ratio * calibrated_heterogeneity, 0.0))
        * structure_scale
        * (0.12 + 0.88 * lam)
    )
    deviations = heterogeneity * scores @ basis.T
    deviations -= np.mean(deviations, axis=1, keepdims=True)
    aggregate_weights = rng.dirichlet(
        np.full(child_count, rng.uniform(1.2, 3.0))
    )
    children = aggregate[:, None] * aggregate_weights[None, :] + deviations
    parent = np.sum(children, axis=1, keepdims=True)
    target = np.column_stack([parent, children])
    detail = {
        "hierarchy": "target_0=sum(target_1:)",
        "child_count": child_count,
        "local_contrast_rank": contrast_rank,
        "aggregate_share_by_child": aggregate_weights.tolist(),
        "heterogeneity_strength": heterogeneity,
        "aggregate_process": aggregate_meta,
        "contrast_correlation_target": corr,
        "local_contrast_loadings": basis.tolist(),
        "contrast_period_multipliers": contrast_period_multipliers,
        "contrast_basis_rotation": rotation.tolist(),
        "child_permutation": child_permutation.tolist(),
        "coherence_residual_mean_abs": float(
            np.mean(np.abs(parent[:, 0] - np.sum(children, axis=1)))
        ),
        "future_only_shock_count": 0,
    }
    return target, _metadata("hierarchical_coherence", family, target, detail), None


def _dense_driver_excitation(
    width: int,
    rng: np.random.Generator,
    *,
    knot_spacing: int,
) -> np.ndarray:
    """Draw a smooth, dense, zero-mean excitation on a fixed interval."""

    if width < 8:
        raise ValueError("dense driver excitation requires width >= 8")
    knot_count = max(5, int(math.ceil(width / knot_spacing)) + 2)
    knots = rng.normal(size=knot_count)
    knot_x = np.linspace(0.0, width - 1.0, knot_count)
    values = np.interp(np.arange(width, dtype=float), knot_x, knots)
    # A short symmetric smoother removes piecewise-linear corners without
    # erasing the broadband lag signature used by the blind edge search.
    values = np.convolve(
        np.pad(values, (2, 2), mode="reflect"),
        np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0,
        mode="valid",
    )
    values -= float(np.mean(values))
    values /= max(float(np.std(values)), 1e-12)
    return values


def _counterfactual_dense_driver(
    background: np.ndarray,
    context: int,
    delay: int,
    rng: np.random.Generator,
    *,
    variant: int,
    family: FamilyRole,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Add continuous teaching excitation and one in-support swappable block."""

    length = len(background)
    driver = 0.55 * np.asarray(background, dtype=float).copy()
    teaching_span = context - delay
    if teaching_span < 16:
        raise ValueError(
            "cross-series lag leaves too little history for dense teaching"
        )
    knot_spacing = int(
        rng.choice(
            np.asarray([4, 5, 6])
            if family == "primary"
            else np.asarray([6, 7, 8])
        )
    )
    history_excitation, history_excitation_meta = (
        _stationary_fourier_nuisance(
            context,
            context,
            rng,
            family=family,
        )
    )
    excitation_scale = float(rng.uniform(0.75, 1.05))
    driver[:context] += excitation_scale * history_excitation
    # Keep one pair-invariant reference for constructing responder nuisance
    # paths.  Regressing nuisance paths on the realized counterfactual driver
    # would make the supposedly shared responder history differ across the two
    # paired members.
    nuisance_reference = driver.copy()

    # Construct two dense alternatives from the same generator and force both
    # to share mean, variance and smooth endpoints.  The intervention is thus
    # an ordinary in-support driver path, not a sign-flipped isolated anomaly.
    alternative_a = _dense_driver_excitation(
        delay,
        rng,
        knot_spacing=knot_spacing,
    )
    alternative_b = _dense_driver_excitation(
        delay,
        rng,
        knot_spacing=knot_spacing,
    )
    taper_width = max(4, min(12, delay // 6))
    taper = np.ones(delay, dtype=float)
    edge = np.sin(
        0.5
        * np.pi
        * (np.arange(taper_width, dtype=float) + 1.0)
        / (taper_width + 1.0)
    ) ** 2
    taper[:taper_width] = edge
    taper[-taper_width:] = edge[::-1]
    alternative_a *= taper
    alternative_b *= taper
    for values in (alternative_a, alternative_b):
        values -= float(np.mean(values))
        values /= max(float(np.std(values)), 1e-12)
    alternatives = (
        excitation_scale * alternative_a,
        excitation_scale * alternative_b,
    )
    start = context - delay
    # Replace only the excitation component.  Background and driver future are
    # identical across members.  Because the block ends exactly at the context
    # boundary, every responder-history value still depends on the shared
    # driver prefix; only the declared future effect prefix differs.
    driver[start:context] = (
        0.55 * np.asarray(background[start:context], dtype=float)
        + alternatives[variant]
    )
    difference = alternatives[1] - alternatives[0]
    return driver, nuisance_reference, {
        "counterfactual_variant": variant,
        "counterfactual_driver_slice": [start, context],
        "counterfactual_alternative_rms": float(
            np.sqrt(np.mean(difference * difference))
        ),
        "driver_background_family": "real_feature_calibrated_continuous_signal",
        "driver_excitation_family": (
            "sample_specific_stationary_fourier_nuisance"
        ),
        "driver_excitation": history_excitation_meta,
        "driver_excitation_knot_spacing": knot_spacing,
        "driver_excitation_scale": excitation_scale,
        "dense_teaching_fraction": float(teaching_span / context),
        "historical_teaching_span": teaching_span,
        "counterfactual_path_taper_width": taper_width,
        "counterfactual_path_mean_by_member": [
            float(np.mean(values)) for values in alternatives
        ],
        "counterfactual_path_std_by_member": [
            float(np.std(values)) for values in alternatives
        ],
        "counterfactual_path_is_dense": True,
        "counterfactual_path_is_in_support": True,
    }


def _bidirectional_nuisance_lag_design(
    backgrounds: list[np.ndarray],
    *,
    maximum_delay: int,
    reference_stop: int,
) -> np.ndarray:
    """Build a prefix-invariant zero/lead/lag nuisance design.

    The formal task has a fixed 48-step forecast.  Lead columns are therefore
    capped at that fixed design boundary instead of reading an arbitrarily
    longer suffix supplied only for a prefix-invariance check.
    """

    columns: list[np.ndarray] = []
    for background in backgrounds:
        values = np.asarray(background, dtype=float)
        columns.append(values)
        for lag in range(1, maximum_delay + 1):
            columns.append(
                np.concatenate(
                    [
                        np.full(lag, values[0], dtype=float),
                        values[:-lag],
                    ]
                )
            )
            lead_prefix = np.concatenate(
                [
                    values[lag:reference_stop],
                    np.full(
                        lag,
                        values[reference_stop - 1],
                        dtype=float,
                    ),
                ]
            )
            columns.append(
                np.concatenate(
                    [
                        lead_prefix,
                        np.full(
                            len(values) - reference_stop,
                            lead_prefix[-1],
                            dtype=float,
                        ),
                    ]
                )
            )
    return np.column_stack(columns)


def _stationary_fourier_nuisance(
    length: int,
    context: int,
    rng: np.random.Generator,
    *,
    family: FamilyRole,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw a deterministic nuisance with stable train/holdout statistics."""

    time = np.arange(length, dtype=float)
    mode_count = 16 if family == "primary" else 20
    values = np.zeros(length, dtype=float)
    modes: list[dict[str, float]] = []
    minimum_cycles = max(14, int(math.ceil(context / 24.0)))
    maximum_cycles = max(
        minimum_cycles + mode_count - 1,
        int(math.floor(context / 4.8)),
    )
    cycle_counts = np.rint(
        np.linspace(minimum_cycles, maximum_cycles, mode_count)
    ).astype(int)
    for mode, cycle_count in enumerate(cycle_counts):
        period = float(context / cycle_count)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amplitude = float(
            rng.uniform(0.8, 1.2) / math.sqrt(mode_count)
        )
        values += amplitude * np.sin(
            2.0 * np.pi * time / period + phase
        )
        modes.append(
            {
                "period": period,
                "cycles_per_context": int(cycle_count),
                "phase": phase,
                "amplitude": amplitude,
            }
        )
    values -= float(np.mean(values[:context]))
    values /= max(float(np.std(values[:context])), 1e-12)
    return values, {
        "law": "sample_specific_stationary_fourier_nuisance",
        "mode_count": mode_count,
        "modes": modes,
        "history_normalization": "shared_full_context_mean_std",
        "future_process_noise_scale": 0.0,
    }


def _balance_aligned_teaching_blocks(
    local: np.ndarray,
    driver: np.ndarray,
    *,
    delay: int,
    context: int,
) -> np.ndarray:
    """Make lag-aligned nuisance stable across chronological train/holdout."""

    balanced = np.asarray(local, dtype=float).copy()
    aligned_size = context - delay
    split = int(
        np.clip(round(0.70 * aligned_size), 8, aligned_size - 4)
    )
    aligned_driver = np.asarray(driver[:aligned_size], dtype=float)
    for lower, upper in ((0, split), (split, aligned_size)):
        source = aligned_driver[lower:upper]
        response_slice = slice(delay + lower, delay + upper)
        nuisance = balanced[response_slice]
        design = np.column_stack(
            [np.ones(source.size, dtype=float), source]
        )
        coefficients = np.linalg.lstsq(
            design,
            nuisance,
            rcond=None,
        )[0]
        residual = nuisance - design @ coefficients
        residual_scale = max(float(np.std(residual)), 1e-12)
        source_scale = max(float(np.std(source)), 1e-12)
        balanced[response_slice] = (
            residual * (source_scale / residual_scale)
        )
    return balanced


def _cross_series_dependence(
    length,
    context,
    dim,
    season,
    intensity,
    rng,
    cond,
    family,
    counterfactual_variant,
):
    if dim < 3:
        raise ValueError(
            "cross_series_dependence requires one driver and at least two responders"
        )
    lam = _lambda(cond, intensity)
    variant = int(counterfactual_variant)
    if variant not in (0, 1):
        raise ValueError("counterfactual_variant must be 0 or 1")
    stream_seeds = rng.integers(0, 2**63 - 1, size=2)
    driver_rng = np.random.default_rng(int(stream_seeds[0]))
    responder_rng = np.random.default_rng(int(stream_seeds[1]))
    requested_delay = int(
        round(_parameter(cond, "cross_lag_steps", float(season)))
    )
    # V8 has a fixed H=48 protocol.  The lag itself is capped at 24 so a blind
    # history-only search still has at least 72 aligned observations inside the
    # shortest L96 view.  The paired intervention consequently affects the
    # first ``delay`` forecast points (plus the two-tap tail in the secondary
    # family); the remainder is an explicit unaffected control region.
    horizon = min(48, max(1, length - context))
    shortest_view = min(context, 96)
    minimum_invariant_driver_prefix = min(16, shortest_view // 4)
    minimum_delay = 8
    maximum_delay = min(24, shortest_view // 4)
    delay = int(
        np.clip(
            requested_delay,
            minimum_delay,
            maximum_delay,
        )
    )
    lag_step = 1
    lag_candidates = np.arange(
        minimum_delay,
        maximum_delay + 1,
        lag_step,
        dtype=int,
    )
    driver_background, background_meta = _calibrated_signal(
        length,
        context,
        driver_rng,
        cond,
        family,
        period_multiplier=4.0,
    )
    driver, nuisance_driver, driver_meta = _counterfactual_dense_driver(
        driver_background,
        context,
        delay,
        driver_rng,
        variant=variant,
        family=family,
    )
    shifted = np.empty(length, dtype=float)
    shifted[:delay] = driver[0]
    shifted[delay:] = driver[:-delay]
    if family == "primary":
        response_source = shifted
        response_law = "single_linear_cross_lag"
        counterfactual_effect_steps = min(delay, horizon)
    else:
        shifted_1 = np.concatenate([[shifted[0]], shifted[:-1]])
        shifted_2 = np.concatenate([[shifted[0], shifted[0]], shifted[:-2]])
        filtered = 0.88 * shifted + 0.08 * shifted_1 + 0.04 * shifted_2
        response_source = np.tanh(0.35 * filtered) / 0.35
        response_law = "distributed_lag_saturating_response"
        counterfactual_effect_steps = min(delay + 2, horizon)
    dependence_scale = _parameter(cond, "cross_dependence_scale", 0.65)
    alignment = _parameter(cond, "cross_lag_alignment", 0.6)
    gain = dependence_scale * (1.50 * lam) * (
        0.65 + 0.35 * alignment
    )
    if family == "secondary":
        gain *= 2.10
    calibrated_background_ratio = _parameter(
        cond,
        "cross_channel_background_ratio",
        0.4,
    )
    # Intensity isolates the directed transfer gain.  Holding the responder
    # nuisance fixed prevents high doses from becoming easier twice: once
    # through a stronger edge and again through disappearing background.
    background_ratio = calibrated_background_ratio
    target = np.empty((length, dim), dtype=float)
    target[:, 0] = driver
    responder_signs = (
        np.where(
            np.arange(dim - 1, dtype=int) % 2 == 0,
            1.0,
            -1.0,
        )
        if family == "primary"
        else responder_rng.choice(
            np.asarray([-1.0, 1.0]),
            size=dim - 1,
        )
    )
    responder_gains = gain * responder_rng.uniform(
        0.85,
        1.15,
        size=dim - 1,
    )
    nuisance_backgrounds = [nuisance_driver]
    responder_backgrounds: list[dict[str, Any]] = []
    for responder in range(dim - 1):
        local, local_meta = _stationary_fourier_nuisance(
            length,
            context,
            responder_rng,
            family=family,
        )
        nuisance_lag_design = _bidirectional_nuisance_lag_design(
            nuisance_backgrounds,
            maximum_delay=maximum_delay,
            reference_stop=min(length, context + horizon),
        )
        nuisance_coefficients = np.linalg.lstsq(
            nuisance_lag_design[:context],
            local[:context],
            rcond=None,
        )[0]
        local = local - nuisance_lag_design @ nuisance_coefficients
        local = _balance_aligned_teaching_blocks(
            local,
            nuisance_driver,
            delay=delay,
            context=context,
        )
        local /= max(float(np.std(local[:context])), 1e-9)
        nuisance_backgrounds.append(local)
        target[:, responder + 1] = (
            background_ratio * local
            + responder_signs[responder]
            * responder_gains[responder]
            * response_source
        )
        responder_backgrounds.append(local_meta)
    detail = {
        "driver_index": 0,
        "responder_indices": list(range(1, dim)),
        "requested_cross_lag_steps": requested_delay,
        "cross_lag_steps": delay,
        "cross_lag_candidate_steps": lag_candidates.tolist(),
        "cross_lag_sampling_policy": (
            "real_anchor_lag_clipped_to_l96_identifiable_range"
        ),
        "cross_lag_step": lag_step,
        "minimum_supported_context_length": shortest_view,
        "minimum_invariant_driver_prefix": (
            minimum_invariant_driver_prefix
        ),
        "history_covered_forecast_steps": counterfactual_effect_steps,
        "counterfactual_effect_forecast_steps": (
            counterfactual_effect_steps
        ),
        "counterfactual_effect_future_slice": [
            context,
            context + counterfactual_effect_steps,
        ],
        "counterfactual_unaffected_future_slice": [
            context + counterfactual_effect_steps,
            context + horizon,
        ],
        "response_law": response_law,
        "dependence_gain": float(gain),
        "calibrated_background_ratio": float(calibrated_background_ratio),
        "effective_background_ratio": float(background_ratio),
        "background_ratio_intensity_policy": (
            "fixed_calibrated_nuisance_across_intensity"
        ),
        "responder_gains": responder_gains.tolist(),
        "responder_signs": responder_signs.tolist(),
        "driver_background": background_meta,
        "responder_backgrounds": responder_backgrounds,
        "counterfactual_responder_history_invariant": True,
        "responder_nuisance_reference_policy": (
            "pair_invariant_pre_counterfactual_driver"
        ),
        "responder_nuisance_orthogonalization": (
            "zero_and_bidirectional_lags_1_through_maximum_delay"
        ),
        "responder_teaching_block_balance": (
            "lag_aligned_70_30_blocks_zero_mean_orthogonal_"
            "and_driver_variance_matched"
        ),
        "counterfactual_future_is_driver_determined": True,
        **driver_meta,
        "future_only_shock_count": 0,
    }
    return (
        target,
        _metadata("cross_series_dependence", family, target, detail),
        None,
    )


def _mean_abs_acf(values: np.ndarray, max_lag: int = 10) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return 0.0
    scores = [
        abs(
            float(
                np.dot(centered[:-lag], centered[lag:])
                / denominator
            )
        )
        for lag in range(1, min(max_lag, len(centered) - 1) + 1)
    ]
    return float(np.mean(scores)) if scores else 0.0


def _covariate_residual_motif(
    length: int,
    context: int,
    season: int,
    rng: np.random.Generator,
    conditioning: GeneratorConditioning | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a sample-specific, forecastable nuisance residual.

    The target ACF is calibrated into the support shared by all formal suffix
    views.  Candidate motifs are drawn before generation and the closest one
    is selected using history-only periodic statistics; no future values are
    inspected and no stochastic innovation is added.
    """

    period = int(
        np.clip(
            round(
                _parameter(
                    conditioning,
                    "profile_dominant_period",
                    float(season),
                )
            ),
            12,
            max(12, min(48, context // 4)),
        )
    )
    target_memory = float(
        np.clip(
            _parameter(
                conditioning,
                "covariate_residual_memory_target",
                0.42,
            ),
            0.40,
            0.44,
        )
    )
    candidates: list[
        tuple[float, np.ndarray, int, int, float]
    ] = []
    for _ in range(32):
        motif = rng.normal(0.0, 1.0, size=period)
        smoothing_passes = int(rng.integers(3, 13))
        for _ in range(smoothing_passes):
            motif = (
                np.roll(motif, 1)
                + 2.0 * motif
                + np.roll(motif, -1)
            ) / 4.0
        motif -= float(np.mean(motif))
        motif_scale = float(np.std(motif))
        if motif_scale <= 1e-9:
            continue
        motif /= motif_scale
        phase = int(rng.integers(0, period))
        history = motif[
            (np.arange(context, dtype=int) + phase) % period
        ]
        realized_memory = _mean_abs_acf(history)
        candidates.append(
            (
                abs(realized_memory - target_memory),
                motif,
                phase,
                smoothing_passes,
                realized_memory,
            )
        )
    if not candidates:
        raise ValueError("failed to construct covariate residual motif")
    (
        selection_error,
        selected_motif,
        selected_phase,
        selected_smoothing_passes,
        selected_memory,
    ) = min(candidates, key=lambda row: row[0])
    values = selected_motif[
        (np.arange(length, dtype=int) + selected_phase) % period
    ]
    return values, {
        "law": "sample_specific_circular_filtered_periodic_motif",
        "period": period,
        "phase": selected_phase,
        "smoothing_passes": selected_smoothing_passes,
        "candidate_count": len(candidates),
        "target_residual_acf_abs_mean": target_memory,
        "realized_history_residual_acf_abs_mean": selected_memory,
        "selection_absolute_error": selection_error,
        "motif_sha256": hashlib.sha256(
            np.ascontiguousarray(selected_motif).tobytes()
        ).hexdigest(),
        "future_process_noise_scale": 0.0,
    }


def _covariate(
    length,
    context,
    dim,
    season,
    intensity,
    rng,
    cond,
    family,
    counterfactual_variant,
):
    lam = _lambda(cond, intensity)
    variant = int(counterfactual_variant)
    if variant not in (0, 1):
        raise ValueError("counterfactual_variant must be 0 or 1")
    # Family sensitivity must isolate the response law.  Reusing the primary
    # driver process makes weather, events, nuisance baseline and random signs
    # identical for a matched primary/secondary seed; otherwise the secondary
    # spline driver also changes the MASE denominator and confounds the audit.
    weather, weather_meta = _calibrated_signal(
        length,
        context,
        rng,
        cond,
        "primary",
    )
    base_weather = weather.copy()
    weather_transform = str(
        rng.choice(
            np.asarray(
                [
                    "future_scaled_reflection",
                    "future_smooth_offset",
                    "future_amplitude_expansion",
                ]
            )
        )
    )
    transform_scale = float(rng.uniform(0.75, 1.35))
    transform_sign = float(rng.choice(np.asarray([-1.0, 1.0])))
    history_weather_scale = max(
        float(np.std(weather[:context])),
        1e-6,
    )
    offset_amplitude = (
        transform_sign
        * history_weather_scale
        * float(rng.uniform(1.40, 2.00))
    )
    if variant == 1:
        # Change only the known future continuation. Both branches share
        # identical past covariates and target history, while the alternative
        # is selected from several smooth, in-support nuisance transforms.
        anchor = float(weather[context - 1])
        future_deviation = weather[context:] - anchor
        if weather_transform == "future_scaled_reflection":
            weather[context:] = anchor - transform_scale * future_deviation
        elif weather_transform == "future_smooth_offset":
            progress = np.linspace(
                0.0,
                1.0,
                len(weather) - context,
                endpoint=True,
            )
            weather[context:] += offset_amplitude * np.sin(
                np.pi * progress
            )
        else:
            expansion = 1.55 + 0.65 * transform_scale
            weather[context:] = anchor + expansion * future_deviation
    event = np.zeros(length, dtype=float)
    width = int(
        rng.integers(
            COVARIATE_EVENT_WIDTH_RANGE[0],
            COVARIATE_EVENT_WIDTH_RANGE[1] + 1,
        )
    )
    historical_fractions = np.asarray([0.18, 0.48, 0.76])
    historical_fractions += rng.uniform(-0.07, 0.07, size=3)
    historical = sorted(
        int(round(context * fraction))
        for fraction in np.clip(historical_fractions, 0.08, 0.90)
    )
    design_horizon = min(48, max(1, length - context))
    candidate_offsets = np.arange(
        2,
        max(3, design_horizon - width),
        dtype=int,
    )
    if candidate_offsets.size >= 2:
        selected_offsets = np.sort(
            rng.choice(candidate_offsets, size=2, replace=False)
        )
        future_offsets = (
            int(selected_offsets[0]),
            int(selected_offsets[1]),
        )
    else:
        future_offsets = (1, max(1, design_horizon - width))
    future_start = context + future_offsets[variant]
    starts = [*historical, future_start]
    for start in starts:
        event[start : min(length, start + width)] = 1.0
    covariates = np.column_stack([weather, event])
    baseline, baseline_meta = _covariate_residual_motif(
        length,
        context,
        season,
        rng,
        cond,
    )
    effect = (
        _parameter(cond, "covariate_effect_scale", 0.55)
        * _parameter(cond, "covariate_explained_scale", 0.65)
        * (0.12 + 1.08 * lam)
    )
    event_ratio = _parameter(cond, "event_effect_ratio", 0.9)
    weather_sign = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    event_sign = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    primary_reference_response = (
        weather[:, None] * weather_sign[None, :]
        + event_ratio * event[:, None] * event_sign[None, :]
    )
    if family == "primary":
        response = primary_reference_response
        response_law = "instantaneous_linear"
        raw_response_history_mean = np.mean(response[:context], axis=0)
        raw_response_history_std = np.std(response[:context], axis=0)
        response_normalization = "primary_reference_unchanged"
    else:
        weather_center = float(np.mean(weather[:context]))
        weather_scale = max(float(np.std(weather[:context])), 1e-6)
        standardized_weather = (weather - weather_center) / weather_scale
        lagged = np.concatenate(
            [[standardized_weather[0]], standardized_weather[:-1]]
        )
        weather_response = (
            0.60 * standardized_weather
            + 0.25 * np.tanh(standardized_weather)
            + 0.15 * np.tanh(lagged)
        )
        event_response = np.convolve(
            event,
            np.asarray([0.50, 0.30, 0.20]),
            mode="full",
        )[:length]
        raw_response = (
            weather_response[:, None] * weather_sign[None, :]
            + event_ratio
            * event_response[:, None]
            * event_sign[None, :]
        )
        raw_response_history_mean = np.mean(
            raw_response[:context],
            axis=0,
        )
        raw_response_history_std = np.std(
            raw_response[:context],
            axis=0,
        )
        raw_response_history_std = np.where(
            raw_response_history_std > 1e-9,
            raw_response_history_std,
            1.0,
        )
        reference_mean = np.mean(
            primary_reference_response[:context],
            axis=0,
        )
        reference_std = np.std(
            primary_reference_response[:context],
            axis=0,
        )
        response = (
            (raw_response - raw_response_history_mean[None, :])
            / raw_response_history_std[None, :]
            * reference_std[None, :]
            + reference_mean[None, :]
        )
        response_law = "semilinear_saturating_distributed_lag"
        response_normalization = (
            "affine_match_primary_reference_history_mean_and_std"
        )

    # Family identity changes the response law, not the calibrated dose.
    # Primary remains numerically unchanged; secondary is affine matched to
    # the same-seed primary history response.  Counterfactual members share
    # history, so they use exactly the same matching statistics.
    response_history_mean = np.mean(response[:context], axis=0)
    response_history_std = np.std(response[:context], axis=0)
    nuisance_component = 0.18 * baseline[:, None]
    covariate_component = effect * response
    target = nuisance_component + covariate_component
    nuisance_variance = float(np.var(nuisance_component[:context, 0]))
    covariate_variance_by_target = np.var(
        covariate_component[:context],
        axis=0,
    )
    effect_variance_share_by_target = (
        covariate_variance_by_target
        / np.maximum(
            covariate_variance_by_target + nuisance_variance,
            1e-12,
        )
    )
    detail = {
        "weather_process": weather_meta,
        "driver_process_matching": (
            "identical_across_primary_secondary_for_matched_seed"
        ),
        "baseline_process": baseline_meta,
        "effect_strength": effect,
        "weather_effect_by_target": (effect * weather_sign).tolist(),
        "event_effect_by_target": (effect * event_ratio * event_sign).tolist(),
        "event_starts": starts,
        "historical_event_fractions": historical_fractions.tolist(),
        "event_width": width,
        "event_width_bounds": list(COVARIATE_EVENT_WIDTH_RANGE),
        "future_event_start": future_start,
        "response_law": response_law,
        "raw_response_history_mean_by_target": (
            raw_response_history_mean.tolist()
        ),
        "raw_response_history_std_by_target": (
            raw_response_history_std.tolist()
        ),
        "response_history_mean_by_target": response_history_mean.tolist(),
        "response_history_std_by_target": response_history_std.tolist(),
        "response_normalization": response_normalization,
        "nuisance_history_variance": nuisance_variance,
        "covariate_effect_history_variance_by_target": (
            covariate_variance_by_target.tolist()
        ),
        "covariate_effect_variance_share_by_target": (
            effect_variance_share_by_target.tolist()
        ),
        "covariate_effect_variance_share": float(
            np.mean(effect_variance_share_by_target)
        ),
        "counterfactual_variant": variant,
        "counterfactual_covariate_future_slice": [context, length],
        "counterfactual_target_history_invariant": True,
        "counterfactual_past_covariates_invariant": True,
        "counterfactual_future_is_covariate_determined": True,
        "counterfactual_weather_transform": (
            "base_continuation" if variant == 0 else weather_transform
        ),
        "counterfactual_weather_transform_selected": weather_transform,
        "counterfactual_weather_transform_scale": transform_scale,
        "counterfactual_weather_offset_amplitude": offset_amplitude,
        "counterfactual_event_start_options": [
            context + offset for offset in future_offsets
        ],
        "counterfactual_weather_future_rms": float(
            np.sqrt(
                np.mean(
                    (weather[context:] - base_weather[context:]) ** 2
                )
            )
        ),
    }
    return target, _metadata("covariate_response", family, target, detail), covariates
