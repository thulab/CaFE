from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from app.services.synthetic_capability_contrast import (
    capability_contrast_forecasts,
)


SCHEMA_VERSION = "synthetic_mechanism_fidelity.v1"
EPSILON = 1e-9


def evaluate_mechanism_fidelity(
    *,
    capability_id: str,
    history: np.ndarray,
    target_future: np.ndarray,
    forecast: np.ndarray,
    season_length: int,
    latent_params: dict[str, Any],
    intensity: int,
    forecast_start_index: int | None = None,
    covariates: np.ndarray | None = None,
    counterfactual_forecast: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score whether a forecast preserves one synthetic capability mechanism.

    The construction metadata and future targets are evaluation-only
    information. They are never passed to the forecasting model. The score
    describes mechanism-aligned output behavior; it is not evidence that the
    model internally identified a causal data-generating mechanism.
    """

    history_values = _as_matrix(history, name="history")
    actual = _as_matrix(target_future, name="target_future")
    predicted = _as_matrix(forecast, name="forecast")
    if actual.shape != predicted.shape:
        raise ValueError("forecast and target_future must have the same shape")
    if history_values.shape[1] != actual.shape[1]:
        raise ValueError("history and future target dimensions must match")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("mechanism scoring requires finite target and forecast")
    if not 1 <= int(intensity) <= 5:
        raise ValueError("intensity must be between 1 and 5")

    origin = (
        len(history_values)
        if forecast_start_index is None
        else int(forecast_start_index)
    )
    scorer = _CAPABILITY_SCORERS.get(capability_id)
    if scorer is None:
        raise ValueError(f"unknown synthetic capability: {capability_id}")
    details = scorer(
        history=history_values,
        actual=actual,
        predicted=predicted,
        season_length=max(2, int(season_length)),
        latent=latent_params,
        origin=origin,
        covariates=covariates,
        counterfactual_forecast=counterfactual_forecast,
    )

    component_scores = {
        str(key): _bounded_score(float(value))
        for key, value in details.pop("component_scores").items()
    }
    if not component_scores:
        raise ValueError("a mechanism scorer must return component scores")
    mechanism_score = _geometric_mean(tuple(component_scores.values()))
    formal_eligible = bool(details.pop("formal_score_eligible", True))
    unsupported_reason = details.pop("unsupported_reason", None)
    if unsupported_reason is not None:
        formal_eligible = False

    return {
        "schema_version": SCHEMA_VERSION,
        "capability_id": capability_id,
        "intensity": int(intensity),
        "interpretation": "mechanism_aligned_forecast_behavior",
        "causal_mechanism_claim": False,
        "future_target_used_by_evaluator_only": True,
        "construction_metadata_used_by_evaluator_only": True,
        "formal_score_eligible": formal_eligible,
        "unsupported_reason": unsupported_reason,
        "mechanism_fidelity_score": mechanism_score,
        "detection_score": component_scores.get("detection"),
        "timing_score": component_scores.get("timing"),
        "magnitude_score": component_scores.get("magnitude"),
        "selectivity_score": component_scores.get("selectivity"),
        "component_scores": component_scores,
        "truth_mechanism_strength": _finite_nonnegative(
            details.pop("truth_mechanism_strength")
        ),
        "forecast_mechanism_strength": _finite_nonnegative(
            details.pop("forecast_mechanism_strength")
        ),
        "point_mae": float(np.mean(np.abs(predicted - actual))),
        "diagnostics": _json_finite(details),
    }


def capability_score(
    *,
    mechanism_fidelity_score: float,
    model_point_loss: float,
    blind_point_loss: float,
) -> float:
    """Apply a point-accuracy safety gate to a mechanism fidelity score."""

    mechanism = _bounded_score(float(mechanism_fidelity_score))
    model_loss = float(model_point_loss)
    blind_loss = float(blind_point_loss)
    if not math.isfinite(model_loss) or model_loss <= 0:
        raise ValueError("model_point_loss must be finite and positive")
    if not math.isfinite(blind_loss) or blind_loss <= 0:
        raise ValueError("blind_point_loss must be finite and positive")
    return float(mechanism * min(1.0, blind_loss / model_loss))


def _score_trend(**kwargs: Any) -> dict[str, Any]:
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    horizon = len(actual)
    time = np.linspace(-1.0, 1.0, horizon)
    nuisance = np.ones((horizon, 1), dtype=float)
    mechanism = np.column_stack([time, time**2])
    truth_component, truth_coefficients, truth_r2 = _partial_component(
        actual,
        nuisance,
        mechanism,
    )
    pred_component, pred_coefficients, pred_r2 = _partial_component(
        predicted,
        nuisance,
        mechanism,
    )
    truth_displacement = truth_component[-1] - truth_component[0]
    pred_displacement = pred_component[-1] - pred_component[0]
    direction = _sign_agreement(pred_displacement, truth_displacement)
    timing = _positive_correlation(pred_component, truth_component)
    magnitude = _ratio_score(_rms(pred_component), _rms(truth_component))
    selectivity = _coefficient_alignment(
        pred_coefficients,
        truth_coefficients,
    )
    return {
        "component_scores": {
            "detection": direction,
            "timing": timing,
            "magnitude": magnitude,
            "selectivity": selectivity,
        },
        "truth_mechanism_strength": _rms(truth_component),
        "forecast_mechanism_strength": _rms(pred_component),
        "truth_incremental_r2": truth_r2,
        "forecast_incremental_r2": pred_r2,
        "trend_direction_accuracy": direction,
        "coefficient_alignment": selectivity,
    }


def _score_multi_seasonal(**kwargs: Any) -> dict[str, Any]:
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    latent = kwargs["latent"]
    periods = [int(value) for value in latent.get("periods", ())]
    if not periods:
        return _unsupported("missing_period_metadata")
    horizon = len(actual)
    time = np.arange(horizon, dtype=float)
    nuisance = np.column_stack(
        [np.ones(horizon, dtype=float), np.linspace(-1.0, 1.0, horizon)]
    )
    harmonic_columns: list[np.ndarray] = []
    for period in periods:
        angle = 2.0 * np.pi * time / max(1, period)
        harmonic_columns.extend([np.sin(angle), np.cos(angle)])
    mechanism = np.column_stack(harmonic_columns)
    design = np.column_stack([nuisance, mechanism])
    if np.linalg.cond(design) > 1e10:
        return _unsupported("future_harmonic_design_not_identifiable")
    truth_component, truth_coefficients, truth_r2 = _partial_component(
        actual,
        nuisance,
        mechanism,
    )
    pred_component, pred_coefficients, pred_r2 = _partial_component(
        predicted,
        nuisance,
        mechanism,
    )

    truth_amplitudes = _harmonic_amplitudes(truth_coefficients)
    pred_amplitudes = _harmonic_amplitudes(pred_coefficients)
    detection_scores: list[float] = []
    phase_scores: list[float] = []
    magnitude_scores: list[float] = []
    for index in range(len(periods)):
        truth_pair = truth_coefficients[2 * index : 2 * index + 2]
        pred_pair = pred_coefficients[2 * index : 2 * index + 2]
        truth_amp = float(np.mean(truth_amplitudes[index]))
        pred_amp = float(np.mean(pred_amplitudes[index]))
        detection_scores.append(min(1.0, pred_amp / max(truth_amp, EPSILON)))
        magnitude_scores.append(_ratio_score(pred_amp, truth_amp))
        phase_scores.append(_phasor_alignment(pred_pair, truth_pair))

    roles = [
        str(item.get("role", ""))
        for item in latent.get("period_components", ())
    ]
    additional_indices = [
        index for index, role in enumerate(roles) if role.startswith("additional")
    ]
    if not additional_indices:
        additional_indices = list(range(1, len(periods)))
    strength_indices = additional_indices or list(range(len(periods)))
    truth_strength = float(
        np.mean(
            [
                np.mean(truth_amplitudes[index])
                for index in strength_indices
            ]
        )
    )
    pred_strength = float(
        np.mean(
            [
                np.mean(pred_amplitudes[index])
                for index in strength_indices
            ]
        )
    )
    selectivity = _ratio_score(pred_r2, truth_r2)
    return {
        "component_scores": {
            "detection": float(np.mean(detection_scores)),
            "timing": float(np.mean(phase_scores)),
            "magnitude": float(np.mean(magnitude_scores)),
            "selectivity": selectivity,
        },
        "truth_mechanism_strength": truth_strength,
        "forecast_mechanism_strength": pred_strength,
        "period_count": len(periods),
        "additional_period_count": len(strength_indices),
        "period_detection_mean": float(np.mean(detection_scores)),
        "phase_alignment_mean": float(np.mean(phase_scores)),
        "truth_incremental_r2": truth_r2,
        "forecast_incremental_r2": pred_r2,
        "harmonic_path_alignment": _positive_correlation(
            pred_component,
            truth_component,
        ),
    }


def _score_time_varying_seasonality(**kwargs: Any) -> dict[str, Any]:
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    latent = kwargs["latent"]
    origin = kwargs["origin"]
    horizon, target_dim = actual.shape
    primary_period = int(latent.get("primary_period", 0))
    modulation_period = int(latent.get("modulation_period", 0))
    if primary_period <= 0 or modulation_period <= 0:
        return _unsupported("missing_modulation_metadata")
    carrier_phase = _metadata_vector(
        latent,
        "carrier_phase_by_target",
        target_dim,
    )
    modulation_phase = _metadata_vector(
        latent,
        "modulation_phase_by_target",
        target_dim,
    )
    time = np.arange(origin, origin + horizon, dtype=float)
    modulation_angle = (
        2 * np.pi * time[:, None] / modulation_period
        + modulation_phase[None, :]
    )
    second_ratio = float(latent.get("modulation_second_harmonic_ratio", 0.0))
    second_phase = float(latent.get("modulation_second_harmonic_phase", 0.0))
    modulation = (
        np.sin(modulation_angle)
        + second_ratio * np.sin(2 * modulation_angle + second_phase)
    ) / (1.0 + second_ratio)
    base_angle = (
        2 * np.pi * time[:, None] / primary_period
        + carrier_phase[None, :]
    )
    base_carrier = np.sin(base_angle)
    amplitude_depth = float(latent.get("amplitude_depth", 0.0))
    phase_depth = float(latent.get("phase_modulation_depth_cycles", 0.0))
    modulated = (1.0 + amplitude_depth * modulation) * np.sin(
        base_angle + 2 * np.pi * phase_depth * modulation
    )
    delta = modulated - base_carrier
    nuisance = np.column_stack(
        [
            np.ones(horizon, dtype=float),
            np.linspace(-1.0, 1.0, horizon),
            base_carrier,
        ]
    )
    truth_component, truth_coefficients, truth_r2 = _channel_basis_component(
        actual,
        nuisance,
        delta,
    )
    pred_component, pred_coefficients, pred_r2 = _channel_basis_component(
        predicted,
        nuisance,
        delta,
    )
    return {
        "component_scores": {
            "detection": _sign_agreement(
                pred_coefficients,
                truth_coefficients,
            ),
            "timing": _positive_correlation(
                pred_component,
                truth_component,
            ),
            "magnitude": _ratio_score(
                _rms(pred_component),
                _rms(truth_component),
            ),
            "selectivity": _ratio_score(pred_r2, truth_r2),
        },
        "truth_mechanism_strength": _rms(truth_component),
        "forecast_mechanism_strength": _rms(pred_component),
        "truth_incremental_r2": truth_r2,
        "forecast_incremental_r2": pred_r2,
        "modulation_path_alignment": _positive_correlation(
            pred_component,
            truth_component,
        ),
        "modulation_period": modulation_period,
    }


def _score_regime_switching(**kwargs: Any) -> dict[str, Any]:
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    latent = kwargs["latent"]
    origin = kwargs["origin"]
    horizon = len(actual)
    cut_points = sorted(int(value) for value in latent.get("cut_points", ()))
    future_cuts = [
        value for value in cut_points if origin < value < origin + horizon
    ]
    if not future_cuts:
        return _unsupported("no_future_regime_switch")
    boundaries = [0, *cut_points, origin, origin + horizon]
    boundaries = sorted(set(value for value in boundaries if value <= origin + horizon))
    segment = sum(value <= origin for value in cut_points)
    initial_sign = float(latent.get("initial_state_sign", 1.0))
    state = np.empty(horizon, dtype=float)
    for offset in range(horizon):
        switches = sum(value <= origin + offset for value in cut_points)
        state[offset] = initial_sign * (-1.0 if switches % 2 else 1.0)
    del boundaries, segment
    nuisance = np.column_stack(
        [np.ones(horizon, dtype=float), np.linspace(-1.0, 1.0, horizon)]
    )
    mechanism = state[:, None]
    truth_component, truth_coefficients, truth_r2 = _partial_component(
        actual,
        nuisance,
        mechanism,
    )
    pred_component, pred_coefficients, pred_r2 = _partial_component(
        predicted,
        nuisance,
        mechanism,
    )
    direction = _switch_direction_accuracy(
        actual,
        predicted,
        future_cuts,
        origin,
    )
    timing = _event_timing_score(
        predicted,
        event_offsets=[value - origin for value in future_cuts],
        tolerance=max(1, int(round(float(latent.get("dwell_length", 4))) / 8)),
    )
    return {
        "component_scores": {
            "detection": direction,
            "timing": timing,
            "magnitude": _ratio_score(
                _rms(pred_component),
                _rms(truth_component),
            ),
            "selectivity": _ratio_score(pred_r2, truth_r2),
        },
        "truth_mechanism_strength": _rms(truth_component),
        "forecast_mechanism_strength": _rms(pred_component),
        "future_switch_count": len(future_cuts),
        "switch_direction_accuracy": direction,
        "switch_timing_score": timing,
        "state_path_alignment": _positive_correlation(
            pred_component,
            truth_component,
        ),
        "truth_incremental_r2": truth_r2,
        "forecast_incremental_r2": pred_r2,
        "state_coefficient_alignment": _coefficient_alignment(
            pred_coefficients,
            truth_coefficients,
        ),
    }


def _score_nonlinear_persistence(**kwargs: Any) -> dict[str, Any]:
    history = kwargs["history"]
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    season_length = kwargs["season_length"]
    latent = kwargs["latent"]
    contrast = capability_contrast_forecasts(
        capability_id="nonlinear_persistence",
        history=history,
        horizon=len(actual),
        season_length=season_length,
        latent_params=latent,
    )
    blind = np.asarray(contrast["blind"], dtype=float)
    aware = np.asarray(contrast["aware"], dtype=float)
    basis = aware - blind
    if _rms(basis) <= 1e-8:
        return _unsupported("history_implies_negligible_nonlinear_contrast")
    truth_delta = actual - blind
    pred_delta = predicted - blind
    truth_projection, truth_coefficients = _project_onto_basis(
        truth_delta,
        basis,
    )
    pred_projection, pred_coefficients = _project_onto_basis(
        pred_delta,
        basis,
    )
    truth_share = _energy_share(truth_projection, truth_delta)
    pred_share = _energy_share(pred_projection, pred_delta)
    truth_variation = float(np.std(actual))
    pred_variation = float(np.std(predicted))
    nontrivial_dynamics = min(
        1.0,
        pred_variation / max(0.10 * truth_variation, EPSILON),
    )
    return {
        "component_scores": {
            "detection": _sign_agreement(
                pred_coefficients,
                truth_coefficients,
            ),
            "timing": _positive_correlation(
                pred_projection,
                truth_projection,
            ),
            "magnitude": _ratio_score(
                _rms(pred_projection),
                _rms(truth_projection),
            ),
            "selectivity": min(
                _ratio_score(pred_share, truth_share),
                nontrivial_dynamics,
            ),
        },
        "truth_mechanism_strength": _rms(truth_projection),
        "forecast_mechanism_strength": _rms(pred_projection),
        "nonlinear_contrast_rms": _rms(basis),
        "truth_contrast_energy_share": truth_share,
        "forecast_contrast_energy_share": pred_share,
        "nontrivial_forecast_dynamics": nontrivial_dynamics,
        "aware_method": contrast["aware_method"],
    }


def _score_predictable_intermittency(**kwargs: Any) -> dict[str, Any]:
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    latent = kwargs["latent"]
    origin = kwargs["origin"]
    horizon = len(actual)
    centers = [
        int(value)
        for value in latent.get("pulse_centers", ())
        if origin <= int(value) < origin + horizon
    ]
    if not centers:
        return _unsupported("no_future_pulse")
    width = max(0.25, float(latent.get("pulse_width", 1.0)))
    support = max(1, int(latent.get("pulse_support_radius", math.ceil(4 * width))))
    time = np.arange(origin, origin + horizon, dtype=float)
    pulse = np.zeros(horizon, dtype=float)
    for center in centers:
        distance = np.abs(time - center)
        pulse += np.where(
            distance <= support,
            np.exp(-0.5 * (distance / width) ** 2),
            0.0,
        )
    nuisance = np.column_stack(
        [np.ones(horizon, dtype=float), np.linspace(-1.0, 1.0, horizon)]
    )
    mechanism = pulse[:, None]
    truth_component, truth_coefficients, truth_r2 = _partial_component(
        actual,
        nuisance,
        mechanism,
    )
    pred_component, pred_coefficients, pred_r2 = _partial_component(
        predicted,
        nuisance,
        mechanism,
    )
    event_offsets = [center - origin for center in centers]
    timing = _event_peak_timing_score(
        actual,
        predicted,
        event_offsets,
        tolerance=max(1, int(math.ceil(2 * width))),
    )
    detection = _sign_agreement(pred_coefficients, truth_coefficients)
    return {
        "component_scores": {
            "detection": detection,
            "timing": timing,
            "magnitude": _ratio_score(
                _rms(pred_component),
                _rms(truth_component),
            ),
            "selectivity": _ratio_score(pred_r2, truth_r2),
        },
        "truth_mechanism_strength": _rms(truth_component),
        "forecast_mechanism_strength": _rms(pred_component),
        "future_pulse_count": len(centers),
        "pulse_detection_direction": detection,
        "pulse_timing_score": timing,
        "truth_incremental_r2": truth_r2,
        "forecast_incremental_r2": pred_r2,
    }


def _score_common_factor(**kwargs: Any) -> dict[str, Any]:
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    if actual.shape[1] < 3:
        return _unsupported("common_factor_requires_at_least_three_targets")
    truth = actual - actual.mean(axis=0, keepdims=True)
    pred = predicted - predicted.mean(axis=0, keepdims=True)
    truth_u, truth_s, truth_vt = np.linalg.svd(truth, full_matrices=False)
    pred_u, pred_s, pred_vt = np.linalg.svd(pred, full_matrices=False)
    truth_strength = float(truth_s[0] / math.sqrt(max(1, len(actual))))
    pred_strength = float(pred_s[0] / math.sqrt(max(1, len(actual))))
    if truth_strength <= EPSILON:
        return _unsupported("future_common_factor_not_identifiable")
    pred_total = float(np.sum(pred_s**2))
    truth_total = float(np.sum(truth_s**2))
    pred_share = float(pred_s[0] ** 2 / pred_total) if pred_total > EPSILON else 0.0
    truth_share = float(truth_s[0] ** 2 / max(truth_total, EPSILON))
    loading = float(abs(np.dot(pred_vt[0], truth_vt[0])))
    factor_path = float(abs(_raw_correlation(pred_u[:, 0], truth_u[:, 0])))
    return {
        "component_scores": {
            "detection": loading,
            "timing": factor_path,
            "magnitude": _ratio_score(pred_strength, truth_strength),
            "selectivity": _ratio_score(pred_share, truth_share),
        },
        "truth_mechanism_strength": truth_strength,
        "forecast_mechanism_strength": pred_strength,
        "loading_subspace_alignment": loading,
        "factor_path_alignment": factor_path,
        "truth_rank1_variance_share": truth_share,
        "forecast_rank1_variance_share": pred_share,
    }


def _score_hierarchical_coherence(**kwargs: Any) -> dict[str, Any]:
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    if actual.shape[1] < 3:
        return _unsupported("hierarchy_requires_parent_and_two_children")
    truth_children = actual[:, 1:]
    pred_children = predicted[:, 1:]
    truth_contrast = truth_children - truth_children.mean(axis=1, keepdims=True)
    pred_contrast = pred_children - pred_children.mean(axis=1, keepdims=True)
    truth_strength = _rms(truth_contrast)
    pred_strength = _rms(pred_contrast)
    truth_parent_scale = max(
        float(np.mean(np.abs(actual[:, 0]))),
        float(np.mean(np.abs(truth_children))),
        EPSILON,
    )
    coherence_error = float(
        np.mean(np.abs(predicted[:, 0] - pred_children.sum(axis=1)))
        / truth_parent_scale
    )
    coherence_score = float(math.exp(-coherence_error))
    contrast_alignment = _positive_correlation(
        np.diff(pred_contrast, axis=0),
        np.diff(truth_contrast, axis=0),
    )
    detection = (
        min(1.0, pred_strength / max(truth_strength, EPSILON))
        if truth_strength > EPSILON
        else 0.0
    )
    return {
        "component_scores": {
            "detection": detection,
            "timing": contrast_alignment,
            "magnitude": _ratio_score(pred_strength, truth_strength),
            "selectivity": coherence_score,
        },
        "truth_mechanism_strength": truth_strength,
        "forecast_mechanism_strength": pred_strength,
        "coherence_normalized_mae": coherence_error,
        "coherence_score": coherence_score,
        "child_contrast_path_alignment": contrast_alignment,
        "zero_forecast_guard": "child_heterogeneity_magnitude",
    }


def _score_covariate_response(**kwargs: Any) -> dict[str, Any]:
    actual = kwargs["actual"]
    predicted = kwargs["predicted"]
    history = kwargs["history"]
    covariates = kwargs["covariates"]
    counterfactual = kwargs["counterfactual_forecast"]
    if covariates is None:
        return _unsupported("covariate_values_are_required")
    known = np.asarray(covariates, dtype=float)
    required = len(history) + len(actual)
    if known.ndim != 2 or len(known) < required or known.shape[1] < 1:
        return _unsupported("known_covariates_do_not_cover_forecast")
    future_covariates = known[len(history) : required]
    horizon = len(actual)
    nuisance = np.column_stack(
        [np.ones(horizon, dtype=float), np.linspace(-1.0, 1.0, horizon)]
    )
    truth_component, truth_coefficients, truth_r2 = _partial_component(
        actual,
        nuisance,
        future_covariates,
    )
    if counterfactual is None:
        pred_component, pred_coefficients, pred_r2 = _partial_component(
            predicted,
            nuisance,
            future_covariates,
        )
        formal_eligible = False
        evaluation_mode = "observational_future_covariate_projection"
    else:
        counterfactual_values = _as_matrix(
            counterfactual,
            name="counterfactual_forecast",
        )
        if counterfactual_values.shape != predicted.shape:
            raise ValueError(
                "counterfactual_forecast and forecast must have the same shape"
            )
        effect = predicted - counterfactual_values
        pred_component, pred_coefficients, pred_r2 = _partial_component(
            effect,
            np.ones((horizon, 1), dtype=float),
            future_covariates,
        )
        formal_eligible = True
        evaluation_mode = "paired_future_covariate_ablation"
    return {
        "component_scores": {
            "detection": _sign_agreement(
                pred_coefficients,
                truth_coefficients,
            ),
            "timing": _positive_correlation(
                pred_component,
                truth_component,
            ),
            "magnitude": _ratio_score(
                _rms(pred_component),
                _rms(truth_component),
            ),
            "selectivity": _ratio_score(pred_r2, truth_r2),
        },
        "truth_mechanism_strength": _rms(truth_component),
        "forecast_mechanism_strength": _rms(pred_component),
        "formal_score_eligible": formal_eligible,
        "evaluation_mode": evaluation_mode,
        "counterfactual_prediction_available": counterfactual is not None,
        "truth_incremental_r2": truth_r2,
        "forecast_incremental_r2": pred_r2,
        "response_coefficient_alignment": _coefficient_alignment(
            pred_coefficients,
            truth_coefficients,
        ),
    }


def _partial_component(
    values: np.ndarray,
    nuisance: np.ndarray,
    mechanism: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    target = _as_matrix(values, name="values")
    nuisance_design = np.asarray(nuisance, dtype=float)
    mechanism_design = np.asarray(mechanism, dtype=float)
    if mechanism_design.ndim == 1:
        mechanism_design = mechanism_design[:, None]
    design = np.column_stack([nuisance_design, mechanism_design])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    nuisance_coefficients = np.linalg.lstsq(
        nuisance_design,
        target,
        rcond=None,
    )[0]
    mechanism_coefficients = coefficients[len(nuisance_design.T) :]
    component = mechanism_design @ mechanism_coefficients
    full_residual = target - design @ coefficients
    blind_residual = target - nuisance_design @ nuisance_coefficients
    blind_energy = float(np.sum(blind_residual**2))
    full_energy = float(np.sum(full_residual**2))
    incremental_r2 = (
        max(0.0, min(1.0, 1.0 - full_energy / blind_energy))
        if blind_energy > EPSILON
        else 0.0
    )
    return component, mechanism_coefficients, incremental_r2


def _channel_basis_component(
    values: np.ndarray,
    nuisance: np.ndarray,
    mechanism: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    target = _as_matrix(values, name="values")
    nuisance_matrix = np.asarray(nuisance, dtype=float)
    mechanism_matrix = _as_matrix(mechanism, name="mechanism")
    if target.shape != mechanism_matrix.shape:
        raise ValueError("channel-specific mechanism must match target shape")
    components = np.zeros_like(target)
    coefficients: list[float] = []
    blind_energy = 0.0
    full_energy = 0.0
    for channel in range(target.shape[1]):
        channel_nuisance = np.column_stack(
            [
                nuisance_matrix[:, :2],
                nuisance_matrix[:, 2 + channel],
            ]
        )
        component, coefficient, _ = _partial_component(
            target[:, channel],
            channel_nuisance,
            mechanism_matrix[:, channel],
        )
        components[:, channel] = component[:, 0]
        coefficients.append(float(coefficient[0, 0]))
        blind_fit = channel_nuisance @ np.linalg.lstsq(
            channel_nuisance,
            target[:, channel],
            rcond=None,
        )[0]
        full_design = np.column_stack(
            [channel_nuisance, mechanism_matrix[:, channel]]
        )
        full_fit = full_design @ np.linalg.lstsq(
            full_design,
            target[:, channel],
            rcond=None,
        )[0]
        blind_energy += float(np.sum((target[:, channel] - blind_fit) ** 2))
        full_energy += float(np.sum((target[:, channel] - full_fit) ** 2))
    incremental_r2 = (
        max(0.0, min(1.0, 1.0 - full_energy / blind_energy))
        if blind_energy > EPSILON
        else 0.0
    )
    return components, np.asarray(coefficients), incremental_r2


def _project_onto_basis(
    values: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = _as_matrix(values, name="values")
    mechanism = _as_matrix(basis, name="basis")
    if target.shape != mechanism.shape:
        raise ValueError("basis and values must have the same shape")
    coefficients = np.zeros(target.shape[1], dtype=float)
    projection = np.zeros_like(target)
    for channel in range(target.shape[1]):
        denominator = float(np.dot(mechanism[:, channel], mechanism[:, channel]))
        if denominator <= EPSILON:
            continue
        coefficients[channel] = float(
            np.dot(target[:, channel], mechanism[:, channel]) / denominator
        )
        projection[:, channel] = coefficients[channel] * mechanism[:, channel]
    return projection, coefficients


def _harmonic_amplitudes(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 2 or values.shape[0] % 2:
        raise ValueError("harmonic coefficients must contain sine/cosine pairs")
    return np.sqrt(values[0::2] ** 2 + values[1::2] ** 2)


def _phasor_alignment(predicted: np.ndarray, actual: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=float)
    truth = np.asarray(actual, dtype=float)
    scores: list[float] = []
    for channel in range(truth.shape[1]):
        pred_pair = pred[:, channel]
        truth_pair = truth[:, channel]
        denominator = float(np.linalg.norm(pred_pair) * np.linalg.norm(truth_pair))
        if denominator <= EPSILON:
            scores.append(1.0 if np.allclose(pred_pair, truth_pair) else 0.0)
            continue
        cosine = float(np.dot(pred_pair, truth_pair) / denominator)
        scores.append((1.0 + np.clip(cosine, -1.0, 1.0)) / 2.0)
    return float(np.mean(scores))


def _switch_direction_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    cut_points: list[int],
    origin: int,
) -> float:
    scores: list[float] = []
    horizon = len(actual)
    for cut in cut_points:
        offset = cut - origin
        radius = max(1, min(3, offset, horizon - offset))
        if radius <= 0:
            continue
        truth_jump = np.mean(actual[offset : offset + radius], axis=0) - np.mean(
            actual[offset - radius : offset],
            axis=0,
        )
        pred_jump = np.mean(
            predicted[offset : offset + radius],
            axis=0,
        ) - np.mean(predicted[offset - radius : offset], axis=0)
        scores.append(_sign_agreement(pred_jump, truth_jump))
    return float(np.mean(scores)) if scores else 0.0


def _event_timing_score(
    predicted: np.ndarray,
    *,
    event_offsets: list[int],
    tolerance: int,
) -> float:
    values = _as_matrix(predicted, name="predicted")
    differences = np.mean(np.abs(np.diff(values, axis=0)), axis=1)
    if not len(differences) or float(np.max(differences)) <= EPSILON:
        return 0.0
    scores: list[float] = []
    boundaries = [0, *event_offsets, len(values)]
    for index, offset in enumerate(event_offsets):
        left = (boundaries[index] + offset) // 2
        right = (offset + boundaries[index + 2]) // 2
        start = max(0, left - 1)
        stop = min(len(differences), max(start + 1, right))
        local = differences[start:stop]
        if not len(local) or float(np.max(local)) <= EPSILON:
            scores.append(0.0)
            continue
        predicted_boundary = start + int(np.argmax(local)) + 1
        distance = abs(predicted_boundary - offset)
        scores.append(float(math.exp(-distance / max(1, tolerance))))
    return float(np.mean(scores)) if scores else 0.0


def _event_peak_timing_score(
    actual: np.ndarray,
    predicted: np.ndarray,
    event_offsets: list[int],
    *,
    tolerance: int,
) -> float:
    truth = _as_matrix(actual, name="actual")
    pred = _as_matrix(predicted, name="predicted")
    pred_centered = pred - np.median(pred, axis=0, keepdims=True)
    truth_centered = truth - np.median(truth, axis=0, keepdims=True)
    scores: list[float] = []
    for offset in event_offsets:
        start = max(0, offset - 2 * tolerance)
        stop = min(len(pred), offset + 2 * tolerance + 1)
        if start >= stop:
            continue
        channel_scores: list[float] = []
        for channel in range(pred.shape[1]):
            sign = 1.0 if truth_centered[offset, channel] >= 0 else -1.0
            local = sign * pred_centered[start:stop, channel]
            if float(np.max(local)) <= EPSILON:
                channel_scores.append(0.0)
                continue
            peak = start + int(np.argmax(local))
            channel_scores.append(
                float(math.exp(-abs(peak - offset) / max(1, tolerance)))
            )
        scores.append(float(np.mean(channel_scores)))
    return float(np.mean(scores)) if scores else 0.0


def _metadata_vector(
    metadata: dict[str, Any],
    key: str,
    size: int,
) -> np.ndarray:
    values = np.asarray(metadata.get(key, ()), dtype=float)
    if values.shape != (size,):
        raise ValueError(f"{key} must contain one value per target")
    return values


def _ratio_score(predicted: float, actual: float) -> float:
    left = max(0.0, float(predicted))
    right = max(0.0, float(actual))
    if left <= EPSILON or right <= EPSILON:
        return 1.0 if left <= EPSILON and right <= EPSILON else 0.0
    return float(math.exp(-abs(math.log(left / right))))


def _coefficient_alignment(predicted: np.ndarray, actual: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    truth = np.asarray(actual, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(pred) * np.linalg.norm(truth))
    if denominator <= EPSILON:
        return 1.0 if np.allclose(pred, truth) else 0.0
    return float(max(0.0, np.dot(pred, truth) / denominator))


def _sign_agreement(predicted: np.ndarray, actual: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    truth = np.asarray(actual, dtype=float).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError("sign comparison requires equal shapes")
    informative = np.abs(truth) > EPSILON
    if not np.any(informative):
        return 1.0 if np.all(np.abs(pred) <= EPSILON) else 0.0
    pred_values = pred[informative]
    truth_values = truth[informative]
    return float(np.mean(np.sign(pred_values) == np.sign(truth_values)))


def _positive_correlation(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(max(0.0, _raw_correlation(predicted, actual)))


def _raw_correlation(predicted: np.ndarray, actual: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    truth = np.asarray(actual, dtype=float).reshape(-1)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denominator = float(np.linalg.norm(pred) * np.linalg.norm(truth))
    if denominator <= EPSILON:
        return 1.0 if np.allclose(pred, truth) else 0.0
    return float(np.clip(np.dot(pred, truth) / denominator, -1.0, 1.0))


def _energy_share(component: np.ndarray, values: np.ndarray) -> float:
    denominator = float(np.sum(np.asarray(values, dtype=float) ** 2))
    if denominator <= EPSILON:
        return 0.0
    return float(
        np.clip(
            np.sum(np.asarray(component, dtype=float) ** 2) / denominator,
            0.0,
            1.0,
        )
    )


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def _geometric_mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("geometric mean requires at least one value")
    bounded = np.asarray([_bounded_score(value) for value in values], dtype=float)
    if np.any(bounded <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(bounded))))


def _bounded_score(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("mechanism component scores must be finite")
    return float(np.clip(value, 0.0, 1.0))


def _finite_nonnegative(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("mechanism strength must be finite and non-negative")
    return number


def _json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _unsupported(reason: str) -> dict[str, Any]:
    return {
        "component_scores": {
            "detection": 0.0,
            "timing": 0.0,
            "magnitude": 0.0,
            "selectivity": 0.0,
        },
        "truth_mechanism_strength": 0.0,
        "forecast_mechanism_strength": 0.0,
        "formal_score_eligible": False,
        "unsupported_reason": reason,
    }


def _as_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty one- or two-dimensional array")
    return matrix


_CAPABILITY_SCORERS: dict[str, Callable[..., dict[str, Any]]] = {
    "trend": _score_trend,
    "multi_seasonal": _score_multi_seasonal,
    "time_varying_seasonality": _score_time_varying_seasonality,
    "regime_switching": _score_regime_switching,
    "nonlinear_persistence": _score_nonlinear_persistence,
    "predictable_intermittency": _score_predictable_intermittency,
    "common_factor": _score_common_factor,
    "hierarchical_coherence": _score_hierarchical_coherence,
    "covariate_response": _score_covariate_response,
}
