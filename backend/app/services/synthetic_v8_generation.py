from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from app.services.synthetic_generator_conditioning import GeneratorConditioning


GENERATOR_VERSION = "capts-paper-v8-dynamic-nuisance-audit"
FamilyRole = Literal["primary", "secondary"]


PRIMARY_FAMILY_BY_CAPABILITY: dict[str, str] = {
    "trend": "sample_specific_polynomial",
    "multi_seasonal": "sample_specific_fourier_basis",
    "time_varying_seasonality": "modulated_oscillator",
    "regime_switching": "deterministic_duration_motif",
    "nonlinear_persistence": "bounded_tanh_recurrence",
    "predictable_intermittency": "deterministic_gaussian_event_clock",
    "common_factor": "latent_factor_linear_state_space",
    "hierarchical_coherence": "aggregate_contrast_linear_state_space",
    "cross_series_dependence": "counterfactual_event_driven_linear_scm",
    "covariate_response": "known_future_linear_response",
}

SECONDARY_FAMILY_BY_CAPABILITY: dict[str, str] = {
    "trend": "sample_specific_cubic_trend",
    "multi_seasonal": "periodic_spline_motif",
    "time_varying_seasonality": "chirped_triangular_modulation",
    "regime_switching": "thresholded_quasiperiodic_oscillator_regime",
    "nonlinear_persistence": "rational_delay_recurrence",
    "predictable_intermittency": "deterministic_raised_cosine_event_clock",
    "common_factor": "latent_factor_periodic_spline",
    "hierarchical_coherence": "aggregate_contrast_periodic_spline",
    "cross_series_dependence": "counterfactual_event_driven_nonlinear_scm",
    "covariate_response": "known_future_distributed_nonlinear_response",
}


def add_observation_noise_to_history(
    clean_target: np.ndarray,
    *,
    context_length: int,
    noise_ratio: float,
    rng: np.random.Generator,
    preserve_additive_hierarchy: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Corrupt only the visible history while retaining a clean future target.

    The returned array is deliberately shaped like an ordinary benchmark
    target: callers expose its noisy prefix to the model and score its untouched
    suffix.  For additive hierarchies noise is sampled on children and the
    parent prefix is recomputed, avoiding a second coherence intervention.
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
    scales = np.std(history, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    if preserve_additive_hierarchy:
        if values.shape[1] < 3:
            raise ValueError("additive hierarchy requires parent plus children")
        child_noise = rng.normal(
            size=(context_length, values.shape[1] - 1),
        ) * (noise_ratio * scales[1:])[None, :]
        result[:context_length, 1:] += child_noise
        result[:context_length, 0] = np.sum(
            result[:context_length, 1:],
            axis=1,
        )
        applied_noise = result[:context_length] - history
    else:
        applied_noise = rng.normal(size=history.shape) * (
            noise_ratio * scales
        )[None, :]
        result[:context_length] += applied_noise
    realized_ratio = float(
        np.mean(np.std(applied_noise, axis=0) / np.maximum(scales, 1e-12))
    )
    return result, {
        "requested_noise_to_history_std_ratio": float(noise_ratio),
        "realized_noise_to_history_std_ratio": realized_ratio,
        "future_noise_max_abs": float(
            np.max(np.abs(result[context_length:] - values[context_length:]))
        ),
    }


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
        "nonlinear_conditional_gain",
        "nonlinear_multi_lag_gain",
        "acf1",
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
        "future_abs_covariate_target_corr",
        "event_lift_abs",
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
        add(
            "nonlinear_gain_scale",
            "nonlinear_conditional_gain",
            0.01,
            0.35,
            1.0,
            "log_compression_then_clip",
            lambda value: 0.35 + 0.65 * min(1.0, max(value, 0.0) / 0.03),
        )
        add(
            "nonlinear_lag_scale",
            "nonlinear_multi_lag_gain",
            0.08,
            0.22,
            0.70,
            "linear_compression_then_clip",
            lambda value: 0.22 + 2.4 * max(value, 0.0),
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
            12.0,
            48.0,
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
            "future_abs_covariate_target_corr",
            0.3,
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

    return parameters, [mapping.as_dict() for mapping in mappings]


def _parameter(
    conditioning: GeneratorConditioning | None,
    name: str,
    default: float,
) -> float:
    if conditioning is None:
        return float(default)
    return float(conditioning.parameters.get(name, default))


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
        selected_period = float(np.clip(period * ratio, 4.0, context_length / 2.0))
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
    period = _parameter(
        conditioning,
        "profile_dominant_period",
        24.0,
    ) * period_multiplier
    period = float(np.clip(period, 4.0, context_length / 2.0))
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
        return _lds_signal(
            length,
            context_length,
            rng,
            period=period,
            persistence=effective_memory,
            spectral_concentration=spectral_concentration,
        )
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
        return _common_factor(length, context_length, target_dim, season_length, intensity, rng, conditioning, family_role)
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
    # One context spans one normalized unit.  This keeps polynomial families
    # numerically comparable and makes the forecast an actual extrapolation
    # instead of letting a cubic term explode over a long history window.
    x = (np.arange(length, dtype=float) - (context - 1)) / max(4, context - 1)
    direction = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    slope_jitter = rng.uniform(0.75, 1.25, size=dim)
    curvature_sign = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    profile_strength = _parameter(cond, "trend_strength_target", 0.15)
    slope_scale = _parameter(cond, "trend_slope_scale", 0.2)
    strength = (
        float(np.clip(_parameter(cond, "structure_scale", 1.0), 0.1, 2.0))
        * (0.03 + 0.24 * lam)
        * (0.5 + 2.0 * slope_scale)
        * (0.5 + profile_strength)
    )
    ratio = _parameter(cond, "trend_curvature_ratio", 0.06)
    effective_ratio = ratio * (0.10 + 0.90 * lam)
    if family == "primary":
        basis = (
            slope_jitter[None, :] * x[:, None]
            + curvature_sign[None, :] * effective_ratio * x[:, None] ** 2
        )
        basis_name = "centered_quadratic"
    else:
        basis = (
            slope_jitter[None, :] * x[:, None]
            + curvature_sign[None, :]
            * 1.50
            * effective_ratio
            * x[:, None] ** 3
        )
        basis_name = "centered_cubic"
    target = strength * basis * direction[None, :]
    detail = {
        "trend_basis": basis_name,
        "trend_strength_parameter": strength,
        "curvature_ratio": ratio,
        "effective_curvature_ratio": effective_ratio,
        "direction_by_target": direction.tolist(),
        "slope_jitter_by_target": slope_jitter.tolist(),
        "curvature_sign_by_target": curvature_sign.tolist(),
        "deterministic_texture": None,
    }
    return target, _metadata("trend", family, target, detail), None


def _multi_seasonal(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    time = np.arange(length, dtype=float)
    primary_period = float(
        np.clip(
            _parameter(cond, "profile_dominant_period", float(season)),
            4.0,
            context / 2.0,
        )
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
        periods = [primary_period, *(primary_period * candidate_ratios).tolist()]
        amplitudes = [1.0, ratio * (0.15 + 0.85 * lam), 0.6 * ratio * (0.15 + 0.85 * lam)]
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
            period=primary_period * float(rng.uniform(1.35, 2.35)),
            spectral_concentration=concentration,
        )
        load = rng.uniform(0.85, 1.15, size=dim)
        target = (
            motif[:, None] * load[None, :]
            + ratio * (0.10 + 1.60 * lam) * second[:, None]
        )
        periods = [float(motif_meta["period"]), float(second_meta["period"])]
        law = "sample_specific_periodic_spline_superposition"
    detail = {"periods": periods, "component_ratio": ratio, "mechanism_law": law}
    return target, _metadata("multi_seasonal", family, target, detail), None


def _time_varying(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    time = np.arange(length, dtype=float)
    period = float(max(4.0, season * rng.uniform(0.85, 1.15)))
    modulation_period = float(np.clip(period * rng.uniform(2.2, 4.0), 2 * period, context / 2))
    carrier_phase = rng.uniform(0.0, 2.0 * np.pi, size=dim)
    modulation_phase = rng.uniform(0.0, 2.0 * np.pi, size=dim)
    depth = _parameter(cond, "modulation_depth_scale", 0.35) * (0.10 + 0.90 * lam)
    if family == "secondary":
        depth *= 1.30
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
        "modulation_harmonic_weight": harmonic_weight,
        "modulation_harmonic_phase": harmonic_phase,
        "mechanism_law": law,
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
    season: int,
    scale: float,
    rng: np.random.Generator,
) -> list[int]:
    base = max(4, int(round(0.65 * max(4, season) * scale)))
    motif_length = int(rng.integers(3, 7))
    multipliers = rng.uniform(0.65, 1.45, size=motif_length)
    pattern = np.maximum(4, np.rint(base * multipliers).astype(int))
    return pattern.tolist()


def _regime(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    # Generate a suffix beyond the requested horizon so that smooth
    # transitions near the right boundary do not depend on requested length.
    schedule_length = length + 8 * max(4, season)
    pattern = _sample_duration_motif(
        season,
        _parameter(cond, "regime_dwell_scale", 1.0),
        rng,
    )
    anchor_offset = int(rng.integers(2, max(3, min(max(4, season), 32))))
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
    strength = (
        _parameter(cond, "regime_level_scale", 0.8)
        * _parameter(cond, "regime_sparse_scale", 0.25)
        * (0.01 + 0.08 * lam)
    )
    if family == "secondary":
        strength *= 2.0
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    texture, texture_meta = _calibrated_signal(length, context, rng, cond, family, period_multiplier=2.3)
    state = state[:length]
    target = strength * state[:, None] * signs[None, :] + 0.04 * texture[:, None]
    detail = {
        "cut_points": [cut for cut in cuts if cut < length],
        "dwell_pattern": pattern,
        "dwell_length": int(round(np.median(pattern))),
        "dwell_anchor_offset": anchor_offset,
        "initial_regime_state": initial_state,
        "regime_strength": strength,
        "transition": transition,
        "deterministic_texture": texture_meta,
    }
    return target, _metadata("regime_switching", family, target, detail), None


def _nonlinear(length, context, dim, season, intensity, rng, cond, family):
    lam = _lambda(cond, intensity)
    calibrated_lag = max(
        2,
        int(round(season * _parameter(cond, "nonlinear_lag_scale", 1.0 / 3.0))),
    )
    lag = max(2, int(round(calibrated_lag * rng.uniform(0.80, 1.20))))
    seasonal_lag = max(4, season)
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
    gain = _parameter(cond, "nonlinear_gain_scale", 0.7) * (0.08 + 0.72 * lam)
    effective_gain = gain if family == "primary" else 3.0 * gain
    if family == "primary":
        persistence_weight = 0.58
        seasonal_weight = 0.10
        forcing_weight = 0.18
        response_slope = 1.35
        response_shift = 0.55
    else:
        persistence_weight = 0.52
        seasonal_weight = 0.14
        forcing_weight = 0.18
        response_slope = 1.0
        response_shift = 0.0
    for index in range(seasonal_lag, total):
        delayed = state[index - lag]
        if family == "primary":
            response = (
                np.tanh(response_slope * delayed + response_shift)
                - np.tanh(response_shift)
            )
            next_value = (
                persistence_weight * state[index - 1]
                + seasonal_weight * state[index - seasonal_lag]
                + effective_gain * response
                + forcing_weight * forcing[index]
            )
            transform = "shifted_tanh"
        else:
            response = delayed / (1.0 + delayed * delayed)
            next_value = (
                persistence_weight * state[index - 1]
                + seasonal_weight * state[index - seasonal_lag]
                + effective_gain * response
                + forcing_weight * forcing[index]
            )
            transform = "rational_delay"
        state[index] = np.clip(next_value, -5.0, 5.0)
    target = _standardize_history(state[burn:], context)
    detail = {
        "nonlinear_transform": transform,
        "nonlinear_lag": lag,
        "calibrated_nonlinear_lag": calibrated_lag,
        "seasonal_lag": seasonal_lag,
        "nonlinear_strength": effective_gain,
        "unscaled_nonlinear_strength": gain,
        "persistence_weight": persistence_weight,
        "seasonal_weight": seasonal_weight,
        "forcing_weight": forcing_weight,
        "nonlinear_response_slope": response_slope,
        "nonlinear_response_shift": response_shift,
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
    event_period = max(
        4,
        int(round(_parameter(cond, "profile_dominant_period", float(season)))),
    )
    motif_length = int(rng.integers(3, 7))
    interval_pattern = np.maximum(
        4,
        np.rint(
            event_period
            * rng.uniform(0.70, 1.35, size=motif_length)
        ).astype(int),
    ).tolist()
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
        * (0.02 + 1.20 * lam)
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
    target = strength * pulse[:, None] * loading[None, :] + 0.03 * texture[:, None]
    detail = {
        "pulse_centers": [center for center in centers if center < length],
        "pulse_interval_pattern": interval_pattern,
        "pulse_anchor_offset": anchor_offset,
        "pulse_width": width,
        "pulse_shape": shape,
        "pulse_strength": strength,
        "deterministic_texture": texture_meta,
    }
    return target, _metadata("predictable_intermittency", family, target, detail), None


def _common_factor(length, context, dim, season, intensity, rng, cond, family):
    if dim < 3:
        raise ValueError("common_factor requires at least three targets")
    lam = _lambda(cond, intensity)
    shared, shared_meta = _calibrated_signal(
        length,
        context,
        rng,
        cond,
        family,
        persistence_name="factor_persistence",
        period_multiplier=float(rng.uniform(0.85, 1.15)),
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
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    loadings = signs * rng.uniform(0.75, 1.25, size=dim)
    loadings /= math.sqrt(float(np.mean(loadings * loadings)))
    target_shared = _parameter(cond, "shared_variance_target", 0.8)
    shared_strength = (0.15 + 1.35 * lam) * math.sqrt(target_shared / max(1.0 - target_shared, 0.05))
    local_strength = float(rng.uniform(0.55, 0.75))
    target = shared_strength * shared[:, None] * loadings[None, :] + local_strength * local
    detail = {
        "factor_rank": 1,
        "shared_factor_strength": shared_strength,
        "shared_factor_process": shared_meta,
        "loadings": loadings.tolist(),
        "loading_rms": float(np.sqrt(np.mean(loadings**2))),
        "local_amplitude": local_strength,
        "local_period_multipliers": local_period_multipliers,
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


def _compact_cross_event(
    time: np.ndarray,
    *,
    center: float,
    width: float,
    family: FamilyRole,
) -> np.ndarray:
    distance = (time - center) / max(width, 1.0)
    if family == "primary":
        return np.where(
            np.abs(distance) <= 3.0,
            np.exp(-0.5 * distance * distance),
            0.0,
        )
    return np.where(
        np.abs(distance) <= 2.0,
        np.maximum(1.0 - 0.5 * np.abs(distance), 0.0),
        0.0,
    )


def _counterfactual_realistic_driver(
    background: np.ndarray,
    context: int,
    delay: int,
    rng: np.random.Generator,
    *,
    variant: int,
    family: FamilyRole,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add broad in-support events and one swappable observed intervention."""

    length = len(background)
    time = np.arange(length, dtype=float)
    driver = 0.8 * np.asarray(background, dtype=float).copy()
    width = float(rng.uniform(4.0, 7.0))
    support_multiplier = 3.0 if family == "primary" else 2.0
    observable_stop = context - delay - int(
        math.ceil(support_multiplier * width)
    )
    historical_event_count = int(rng.integers(3, 6))
    historical_fractions = np.linspace(
        0.15,
        0.85,
        historical_event_count,
    )
    historical_fractions += rng.uniform(
        -0.055,
        0.055,
        size=historical_event_count,
    )
    historical_fractions = np.clip(historical_fractions, 0.08, 0.92)
    historical_centers = sorted(
        int(round(observable_stop * fraction))
        for fraction in historical_fractions
    )
    historical_amplitudes = (
        rng.uniform(0.55, 1.05, size=len(historical_centers))
        * rng.choice(
            np.asarray([-1.0, 1.0]),
            size=len(historical_centers),
        )
    )
    for center, amplitude in zip(
        historical_centers,
        historical_amplitudes,
        strict=True,
    ):
        driver += amplitude * _compact_cross_event(
            time,
            center=float(center),
            width=width,
            family=family,
        )

    design_horizon = min(48, max(1, length - context))
    response_margin = support_multiplier * width
    response_lower = min(
        max(2.0, response_margin),
        max(2.0, design_horizon / 2.0),
    )
    response_upper = max(
        response_lower,
        design_horizon - response_margin,
    )
    response_center_offset = float(
        rng.uniform(response_lower, response_upper)
        if response_upper > response_lower + 1e-9
        else response_lower
    )
    intervention_center = float(
        context - delay + response_center_offset
    )
    intervention_amplitude = float(rng.uniform(0.9, 1.3))
    intervention_sign = float(rng.choice(np.asarray([-1.0, 1.0])))
    alternative_amplitudes = (
        intervention_sign * intervention_amplitude,
        -intervention_sign * intervention_amplitude,
    )
    intervention = _compact_cross_event(
        time,
        center=intervention_center,
        width=width,
        family=family,
    )
    # Enforce exact responder-history invariance. Values before context-delay
    # would already have propagated into the visible responder prefix.
    intervention[: context - delay] = 0.0
    driver += alternative_amplitudes[variant] * intervention
    return driver, {
        "counterfactual_variant": variant,
        "counterfactual_driver_slice": [context - delay, context],
        "counterfactual_alternative_rms": float(
            2.0
            * intervention_amplitude
            * np.sqrt(np.mean(intervention[:context] ** 2))
        ),
        "driver_background_family": "real_feature_calibrated_continuous_signal",
        "historical_event_centers": historical_centers,
        "historical_event_fractions": historical_fractions.tolist(),
        "historical_event_amplitudes": historical_amplitudes.tolist(),
        "counterfactual_event_center": intervention_center,
        "counterfactual_response_center_offset": response_center_offset,
        "counterfactual_event_width": width,
        "counterfactual_event_amplitudes": list(alternative_amplitudes),
        "counterfactual_event_shape": (
            "compact_gaussian"
            if family == "primary"
            else "compact_triangular"
        ),
    }


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
    # Candidate lags are >= the fixed H=48 protocol and aligned to the 32-step
    # tokenization used by Toto 2.0 and TiRex2. The real-data lag extraction is
    # retained as a calibrated requested value, while sample-level nuisance
    # variation avoids a single public lag template and sub-patch confounding.
    lag_candidates = np.arange(64, context // 3 + 1, 32, dtype=int)
    if lag_candidates.size == 0:
        lag_candidates = np.asarray(
            [max(48, min(context // 3, context - 1))],
            dtype=int,
        )
    delay = int(driver_rng.choice(lag_candidates))
    driver_background, background_meta = _calibrated_signal(
        length,
        context,
        driver_rng,
        cond,
        family,
        period_multiplier=4.0,
    )
    driver, driver_meta = _counterfactual_realistic_driver(
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
    else:
        shifted_1 = np.concatenate([[shifted[0]], shifted[:-1]])
        shifted_2 = np.concatenate([[shifted[0], shifted[0]], shifted[:-2]])
        filtered = 0.60 * shifted + 0.28 * shifted_1 + 0.12 * shifted_2
        response_source = np.tanh(1.25 * filtered)
        response_law = "distributed_lag_saturating_response"
    dependence_scale = _parameter(cond, "cross_dependence_scale", 0.65)
    alignment = _parameter(cond, "cross_lag_alignment", 0.6)
    gain = dependence_scale * (0.12 + 1.38 * lam) * (
        0.65 + 0.35 * alignment
    )
    if family == "secondary":
        gain *= 1.45
    calibrated_background_ratio = _parameter(
        cond,
        "cross_channel_background_ratio",
        0.4,
    )
    background_ratio = calibrated_background_ratio * (1.0 - 0.92 * lam)
    target = np.empty((length, dim), dtype=float)
    target[:, 0] = driver
    responder_signs = (
        np.ones(dim - 1, dtype=float)
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
    responder_backgrounds: list[dict[str, Any]] = []
    for responder in range(dim - 1):
        local, local_meta = _calibrated_signal(
            length,
            context,
            responder_rng,
            cond,
            family,
            period_multiplier=1.35 + 0.25 * responder,
        )
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
        "cross_lag_sampling_policy": "uniform_patch_aligned_lag_at_least_horizon",
        "history_covered_forecast_steps": min(delay, length - context),
        "response_law": response_law,
        "dependence_gain": float(gain),
        "calibrated_background_ratio": float(calibrated_background_ratio),
        "effective_background_ratio": float(background_ratio),
        "responder_gains": responder_gains.tolist(),
        "responder_signs": responder_signs.tolist(),
        "driver_background": background_meta,
        "responder_backgrounds": responder_backgrounds,
        "counterfactual_responder_history_invariant": True,
        "counterfactual_future_is_driver_determined": True,
        **driver_meta,
        "future_only_shock_count": 0,
    }
    return (
        target,
        _metadata("cross_series_dependence", family, target, detail),
        None,
    )


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
    weather, weather_meta = _calibrated_signal(length, context, rng, cond, family)
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
    width = int(rng.integers(2, max(3, min(7, season // 3 + 1))))
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
    baseline, baseline_meta = _calibrated_signal(
        length,
        context,
        rng,
        cond,
        family,
        period_multiplier=1.8,
    )
    effect = (
        _parameter(cond, "covariate_effect_scale", 0.55)
        * _parameter(cond, "covariate_explained_scale", 0.65)
        * (0.12 + 1.08 * lam)
    )
    event_ratio = _parameter(cond, "event_effect_ratio", 0.9)
    weather_sign = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    event_sign = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    if family == "primary":
        response = weather[:, None] * weather_sign[None, :] + event_ratio * event[:, None] * event_sign[None, :]
        response_law = "instantaneous_linear"
    else:
        effect *= 1.7
        lagged = np.concatenate([[weather[0]], weather[:-1]])
        weather_response = np.tanh(weather) + 0.35 * np.tanh(lagged)
        event_response = np.convolve(event, np.asarray([0.6, 0.3, 0.1]), mode="full")[:length]
        response = weather_response[:, None] * weather_sign[None, :] + event_ratio * event_response[:, None] * event_sign[None, :]
        response_law = "distributed_lag_saturating"
    target = 0.18 * baseline[:, None] + effect * response
    detail = {
        "weather_process": weather_meta,
        "baseline_process": baseline_meta,
        "effect_strength": effect,
        "weather_effect_by_target": (effect * weather_sign).tolist(),
        "event_effect_by_target": (effect * event_ratio * event_sign).tolist(),
        "event_starts": starts,
        "historical_event_fractions": historical_fractions.tolist(),
        "event_width": width,
        "future_event_start": future_start,
        "response_law": response_law,
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
