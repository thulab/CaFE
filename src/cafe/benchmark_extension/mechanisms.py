from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.gift_eval import GiftEvalInstance


MECHANISM_SCHEMA = "cafe.native_path_mechanism.v6"
CAPABILITY_IDS = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
    "common_factor",
    "hierarchical_coherence",
    "cross_series_dependence",
    "covariate_impulse_response",
)
GENERATABLE_CAPABILITY_IDS = tuple(
    capability
    for capability in CAPABILITY_IDS
    if capability != "hierarchical_coherence"
)
CAPABILITY_LEVELS = (1, 2, 3, 4, 5)
SOURCE_DISTANCE_MINIMUM_MACRO = 0.10
SOURCE_DISTANCE_MAXIMUM_MACRO = 2.0
SOURCE_DISTANCE_MAXIMUM_CHANNEL = 3.0
SOURCE_DISTANCE_MODEL_MAX_CONTEXTS = {
    "tirex2": 2048,
    "Timer-4.0": 8192,
    "Chronos-2": 8192,
    "Timer-3.5": 11520,
    "timesfm2.5": 15360,
    "moirai2": 16384,
    "toto2.0": 16384,
}
# Kept as a compatibility alias for callers that only need the lower bound.
SOURCE_DISTANCE_THRESHOLD = SOURCE_DISTANCE_MINIMUM_MACRO
MECHANISM_EFFECT_MINIMUM_MASE_RMS = 0.05
TVS_ENVELOPE_ACTIVE_AMPLITUDE_FRACTION = 0.25
TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION = 0.25
TVS_MINIMUM_INCREMENTAL_R2 = 0.01
COMMON_FACTOR_MINIMUM_HARMONIC_SHARE = 0.05
COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO = 0.50
STRENGTH_INTERVALS = (
    (0.10, 0.14),
    (0.16, 0.20),
    (0.22, 0.28),
    (0.30, 0.38),
    (0.42, 0.55),
)
REGIME_RECENCY_INTERVALS = (
    (0.20, 0.32),
    (0.38, 0.50),
    (0.56, 0.66),
    (0.72, 0.82),
    (0.87, 0.94),
)
INTERMITTENCY_GAP_INTERVALS = (
    (0.10, 0.18),
    (0.22, 0.30),
    (0.34, 0.44),
    (0.50, 0.64),
    (0.72, 0.92),
)


@dataclass(frozen=True)
class CapabilityTreatment:
    level: int
    history_delta: np.ndarray
    future_delta: np.ndarray
    affected_target_indices: tuple[int, ...]
    controlled_coordinate: str
    coordinate_interval: tuple[float, float]
    sampled_coordinate: float
    applied_component_gain: float
    metadata: dict[str, Any]
    source_distance_gate: dict[str, Any]
    horizon_support_gate: dict[str, Any] | None
    history_covariate_delta: np.ndarray
    future_covariate_delta: np.ndarray


@dataclass(frozen=True)
class CapabilityGroup:
    capability_id: str
    available: bool
    reason: str | None
    treatments: tuple[CapabilityTreatment, ...]
    group_metadata: dict[str, Any]


@dataclass(frozen=True)
class _UnitTreatment:
    history_delta: np.ndarray
    future_delta: np.ndarray
    affected: tuple[int, ...]
    coordinate_name: str
    coordinate_interval: tuple[float, float]
    sampled_coordinate: float
    metadata: dict[str, Any]
    history_covariate_delta: np.ndarray | None = None
    future_covariate_delta: np.ndarray | None = None


def _rng(*parts: object, augmentation_seed: int) -> np.random.Generator:
    return np.random.default_rng(
        protocol.stable_seed(*parts, base=int(augmentation_seed))
    )


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _scale_by_target(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    scales = np.std(values, axis=0)
    differences = np.diff(values, axis=0)
    fallback = (
        np.std(differences, axis=0)
        if differences.shape[0]
        else np.ones(values.shape[1])
    )
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, fallback)
    return np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)


def _season_length(frequency: str) -> int:
    raw = str(frequency)
    if raw.endswith(("H", "h")):
        return 24
    if raw.endswith("D"):
        return 7
    if raw.endswith(("M", "ME")):
        return 12
    if raw.endswith("W") or raw.startswith("W-"):
        return 52
    return 1


def mase_scale_by_target(history: np.ndarray, frequency: str) -> np.ndarray:
    """Return the authentic-history scale used by treatment MASE."""

    values = np.asarray(history, dtype=float)
    period = _season_length(frequency)
    lag = min(max(1, period), max(1, values.shape[0] - 1))
    differences = np.abs(values[lag:] - values[:-lag])
    scales = (
        np.mean(differences, axis=0)
        if differences.size
        else np.ones(values.shape[1])
    )
    fallback = np.mean(np.abs(np.diff(values, axis=0)), axis=0)
    scales = np.where(
        np.isfinite(scales) & (scales > 1e-8), scales, fallback
    )
    return np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)


def mechanism_effect_signal(
    future_delta: np.ndarray,
    observed_mask: np.ndarray,
    scale_by_target: np.ndarray,
    affected: tuple[int, ...] | list[int],
) -> tuple[float, float, int]:
    """Measure scored future effect in raw and MASE-standardized units."""

    delta = np.asarray(future_delta, dtype=float)
    mask = np.asarray(observed_mask, dtype=bool)
    scales = np.asarray(scale_by_target, dtype=float)
    assessed = np.zeros_like(mask, dtype=bool)
    indices = [int(value) for value in affected]
    assessed[:, indices] = mask[:, indices]
    count = int(np.count_nonzero(assessed))
    if count == 0:
        return 0.0, 0.0, 0
    raw = delta[assessed]
    standardized = (delta / scales[None, :])[assessed]
    return (
        float(np.sqrt(np.mean(np.square(raw)))),
        float(np.sqrt(np.mean(np.square(standardized)))),
        count,
    )


def _distance_gate(
    history_delta: np.ndarray,
    history: np.ndarray,
    affected: tuple[int, ...],
) -> dict[str, Any]:
    delta = np.asarray(history_delta, dtype=float)
    scales = _scale_by_target(history)
    history_length = int(history.shape[0])
    model_ids_by_context: dict[int, list[str]] = {}
    for model_id, maximum in SOURCE_DISTANCE_MODEL_MAX_CONTEXTS.items():
        context = min(history_length, int(maximum))
        model_ids_by_context.setdefault(context, []).append(model_id)
    by_context: list[dict[str, Any]] = []
    for context in sorted(model_ids_by_context):
        standardized = delta[-context:, affected] / scales[list(affected)]
        channel = np.sqrt(np.mean(np.square(standardized), axis=0))
        macro = float(np.mean(channel))
        by_context.append(
            {
                "context_length": int(context),
                "model_ids": sorted(model_ids_by_context[context]),
                "macro_normalized_rms": macro,
                "channel_normalized_rms": channel.tolist(),
            }
        )
    full_standardized = delta[:, affected] / scales[list(affected)]
    full_channels = np.sqrt(np.mean(np.square(full_standardized), axis=0))
    full_macro = float(np.mean(full_channels))
    minimum_macro = min(row["macro_normalized_rms"] for row in by_context)
    maximum_macro = max(row["macro_normalized_rms"] for row in by_context)
    maximum_channel = max(
        max(row["channel_normalized_rms"], default=0.0) for row in by_context
    )
    below_minimum = minimum_macro < SOURCE_DISTANCE_MINIMUM_MACRO - 1e-12
    above_macro_maximum = maximum_macro > SOURCE_DISTANCE_MAXIMUM_MACRO + 1e-12
    above_channel_maximum = (
        maximum_channel > SOURCE_DISTANCE_MAXIMUM_CHANNEL + 1e-12
    )
    accepted = not (below_minimum or above_macro_maximum or above_channel_maximum)
    reason = None
    if below_minimum:
        reason = "below_minimum_model_context_macro_distance"
    elif above_macro_maximum:
        reason = "above_maximum_model_context_macro_distance"
    elif above_channel_maximum:
        reason = "above_maximum_model_context_channel_distance"
    return {
        "schema_version": "cafe.treatment_source_distance_gate.v3",
        "metric": "source_frozen_scale_actual_model_context_normalized_rms",
        "scope": "treatment_history_vs_authentic_official_history",
        "treatment_only": True,
        "strength_reference": "full_official_history_macro_normalized_rms",
        "full_history_context_length": history_length,
        "full_history_macro_normalized_rms": full_macro,
        "full_history_channel_normalized_rms": full_channels.tolist(),
        "model_max_contexts": dict(SOURCE_DISTANCE_MODEL_MAX_CONTEXTS),
        "evaluated_model_contexts": sorted(model_ids_by_context),
        "minimum_required_macro_distance": SOURCE_DISTANCE_MINIMUM_MACRO,
        "maximum_allowed_macro_distance": SOURCE_DISTANCE_MAXIMUM_MACRO,
        "maximum_allowed_channel_distance": SOURCE_DISTANCE_MAXIMUM_CHANNEL,
        "minimum_observed_macro_distance": minimum_macro,
        "maximum_observed_macro_distance": maximum_macro,
        "maximum_observed_channel_distance": maximum_channel,
        "by_model_context": by_context,
        "accepted": accepted,
        "reason": reason,
    }


def _full_history_unit_distance(
    delta: np.ndarray,
    history: np.ndarray,
    affected: tuple[int, ...],
) -> float:
    values = np.asarray(delta, dtype=float)
    scales = _scale_by_target(history)[list(affected)]
    standardized = values[:, affected] / scales
    channel = np.sqrt(np.mean(np.square(standardized), axis=0))
    return float(np.mean(channel))


def _linear_extrapolation(values: np.ndarray, horizon: int) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    length, dimension = series.shape
    window = min(length, max(16, min(256, length // 2)))
    x = np.arange(window, dtype=float)
    future_x = np.arange(window, window + horizon, dtype=float)
    output = np.empty((horizon, dimension), dtype=float)
    for channel in range(dimension):
        coefficients = np.polyfit(x, series[-window:, channel], 1)
        output[:, channel] = np.polyval(coefficients, future_x)
    return output


def _harmonic_component(
    values: np.ndarray,
    frequency_index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(values, dtype=float)
    length = series.shape[0]
    t = np.arange(length, dtype=float)
    all_t = np.arange(length + horizon, dtype=float)
    omega = 2.0 * math.pi * float(frequency_index) / float(length)
    design = np.column_stack((np.sin(omega * t), np.cos(omega * t)))
    extended = np.column_stack(
        (np.sin(omega * all_t), np.cos(omega * all_t))
    )
    coefficients, *_ = np.linalg.lstsq(design, series, rcond=None)
    component = extended @ coefficients
    return component[:length], component[length:]


def _harmonic_signal(
    *,
    length: int,
    horizon: int,
    frequency_index: int,
    sin_coefficient: float,
    cos_coefficient: float,
) -> np.ndarray:
    all_t = np.arange(length + horizon, dtype=float)
    omega = 2.0 * math.pi * float(frequency_index) / float(length)
    return (
        float(sin_coefficient) * np.sin(omega * all_t)
        + float(cos_coefficient) * np.cos(omega * all_t)
    )


def _harmonic_coefficients(
    values: np.ndarray,
    frequency_index: int,
) -> tuple[float, float]:
    series = np.asarray(values, dtype=float)
    t = np.arange(series.size, dtype=float)
    omega = 2.0 * math.pi * float(frequency_index) / float(series.size)
    design = np.column_stack((np.sin(omega * t), np.cos(omega * t)))
    coefficients = np.linalg.lstsq(design, series, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def _dominant_frequency_indexes(series: np.ndarray) -> list[int]:
    values = np.asarray(series, dtype=float)
    length = values.size
    if length < 24:
        return []
    t = np.linspace(-1.0, 1.0, length)
    design = np.column_stack((np.ones(length), t))
    detrended = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    spectrum = np.abs(np.fft.rfft(detrended * np.hanning(length))) ** 2
    indexes = np.arange(spectrum.size)
    valid = (indexes >= 2) & (indexes <= length // 4)
    ranked = np.argsort(np.where(valid, spectrum, -np.inf))[::-1]
    return [int(index) for index in ranked[:16] if np.isfinite(spectrum[index])]


def _trend_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    if length < 16:
        raise ValueError("history_too_short_for_trend")
    scale = _scale_by_target(history)
    x = np.linspace(0.0, 1.0, length)
    directions = np.zeros(dimension, dtype=float)
    stable: list[int] = []
    for channel in range(dimension):
        normalized = (history[:, channel] - np.mean(history[:, channel])) / scale[channel]
        full = float(np.polyfit(x, normalized, 1)[0])
        midpoint = length // 2
        first = float(np.polyfit(x[:midpoint], normalized[:midpoint], 1)[0])
        second = float(np.polyfit(x[midpoint:], normalized[midpoint:], 1)[0])
        if (
            abs(full) >= 0.05
            and math.copysign(1.0, first) == math.copysign(1.0, full)
            and math.copysign(1.0, second) == math.copysign(1.0, full)
        ):
            directions[channel] = math.copysign(1.0, full)
            stable.append(channel)
    if not stable:
        raise ValueError("history_trend_direction_not_stable")
    extended = np.arange(length + instance.prediction_length, dtype=float)
    extended /= max(1, length - 1)
    base = np.zeros((extended.size, dimension), dtype=float)
    base[:, stable] = (
        extended[:, None]
        * directions[stable][None, :]
        * scale[stable][None, :]
    )
    unit_distance = _full_history_unit_distance(base[:length], history, tuple(stable))
    if unit_distance <= 1e-10:
        raise ValueError("trend_component_not_visible")
    units: list[_UnitTreatment] = []
    for level, interval in zip(CAPABILITY_LEVELS, STRENGTH_INTERVALS, strict=True):
        draw = float(
            _rng(
                instance.official_instance_id,
                "trend",
                level,
                augmentation_seed=augmentation_seed,
            ).uniform(*interval)
        )
        gain = draw / unit_distance
        units.append(
            _UnitTreatment(
                history_delta=base[:length] * gain,
                future_delta=base[length:] * gain,
                affected=tuple(stable),
                coordinate_name="full_history_macro_normalized_rms",
                coordinate_interval=interval,
                sampled_coordinate=draw,
                metadata={
                    "trend_type": "whole_history_linear",
                    "direction_source": "history_only_stable_robust_slope_sign",
                    "direction_by_target": directions.tolist(),
                    "unit_component_distance": unit_distance,
                    "physical_linear_gain": gain,
                },
            )
        )
    return units, {"stable_trend_target_indices": stable}


def _secondary_seasonal_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    history_component = np.zeros_like(history)
    future_component = np.zeros_like(instance.future)
    affected: list[int] = []
    resolved: dict[str, Any] = {}
    for channel in range(dimension):
        ranked = _dominant_frequency_indexes(history[:, channel])
        if not ranked:
            continue
        carrier = ranked[0]
        secondary = next(
            (
                index
                for index in ranked[1:]
                if abs(index - carrier) > 1
                and all(abs(index - harmonic * carrier) > 1 for harmonic in range(2, 9))
            ),
            None,
        )
        if secondary is None:
            continue
        component_h, component_f = _harmonic_component(
            history[:, channel], secondary, instance.prediction_length
        )
        if np.std(component_h) < 0.01 * _scale_by_target(history)[channel]:
            continue
        history_component[:, channel] = component_h
        future_component[:, channel] = component_f
        affected.append(channel)
        resolved[str(channel)] = {
            "carrier_period": float(length / carrier),
            "secondary_period": float(length / secondary),
        }
    if not affected:
        raise ValueError("independent_secondary_seasonality_not_resolved")
    return _strength_scaled_units(
        instance,
        augmentation_seed,
        "multi_seasonal",
        history_component,
        future_component,
        tuple(affected),
        metadata={
            "component": "history_fitted_independent_secondary_harmonic",
            "resolved_periods_by_target": resolved,
        },
    )


def _time_varying_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    component_h = np.zeros_like(history)
    component_f = np.zeros_like(instance.future)
    affected: list[int] = []
    details: dict[str, Any] = {}
    t = np.arange(length, dtype=float)
    centered_t = np.linspace(-1.0, 1.0, length)
    for channel in range(dimension):
        ranked = _dominant_frequency_indexes(history[:, channel])
        if not ranked:
            continue
        carrier_index = ranked[0]
        maximum_modulation_index = min(16, (carrier_index - 1) // 2)
        if maximum_modulation_index < 2:
            continue
        series = np.asarray(history[:, channel], dtype=float)
        trend_design = np.column_stack((np.ones(length), centered_t))
        detrended = series - trend_design @ np.linalg.lstsq(
            trend_design, series, rcond=None
        )[0]
        carrier_sin, carrier_cos = _harmonic_coefficients(
            detrended, carrier_index
        )
        carrier = _harmonic_signal(
            length=length,
            horizon=instance.prediction_length,
            frequency_index=carrier_index,
            sin_coefficient=carrier_sin,
            cos_coefficient=carrier_cos,
        )
        carrier_history = carrier[:length]
        residual = detrended - carrier_history
        baseline_error = float(np.mean(np.square(residual)))
        if baseline_error <= 1e-12:
            continue
        best: tuple[float, int, float, float, np.ndarray] | None = None
        for modulation_index in range(2, maximum_modulation_index + 1):
            omega = 2.0 * math.pi * float(modulation_index) / float(length)
            design = np.column_stack(
                (
                    carrier_history * np.sin(omega * t),
                    carrier_history * np.cos(omega * t),
                )
            )
            coefficients = np.linalg.lstsq(design, residual, rcond=None)[0]
            fitted = design @ coefficients
            incremental_r2 = 1.0 - float(
                np.mean(np.square(residual - fitted))
            ) / baseline_error
            candidate = (
                incremental_r2,
                modulation_index,
                float(coefficients[0]),
                float(coefficients[1]),
                fitted,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None or best[0] < TVS_MINIMUM_INCREMENTAL_R2:
            continue
        incremental_r2, slower, envelope_sin, envelope_cos, fitted = best
        envelope = _harmonic_signal(
            length=length,
            horizon=instance.prediction_length,
            frequency_index=slower,
            sin_coefficient=envelope_sin,
            cos_coefficient=envelope_cos,
        )
        combined = carrier * envelope
        if np.std(fitted) < 0.01 * _scale_by_target(history)[channel]:
            continue
        component_h[:, channel] = combined[:length]
        component_f[:, channel] = combined[length:]
        affected.append(channel)
        details[str(channel)] = {
            "carrier_period": float(length / carrier_index),
            "modulation_period": float(length / slower),
            "carrier_frequency_index": int(carrier_index),
            "modulation_frequency_index": int(slower),
            "carrier_sin_coefficient": carrier_sin,
            "carrier_cos_coefficient": carrier_cos,
            "envelope_sin_coefficient": envelope_sin,
            "envelope_cos_coefficient": envelope_cos,
            "envelope_amplitude": float(
                math.hypot(envelope_sin, envelope_cos)
            ),
            "am_incremental_r2": float(incremental_r2),
        }
    if not affected:
        raise ValueError("constrained_am_envelope_not_resolved")
    return _strength_scaled_units(
        instance,
        augmentation_seed,
        "time_varying_seasonality",
        component_h,
        component_f,
        tuple(affected),
        metadata={
            "component": "history_fitted_constrained_am_carrier_envelope",
            "resolved_periods_by_target": details,
        },
    )


def _strength_scaled_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
    capability_id: str,
    history_component: np.ndarray,
    future_component: np.ndarray,
    affected: tuple[int, ...],
    *,
    metadata: dict[str, Any],
    history_covariate_component: np.ndarray | None = None,
    future_covariate_component: np.ndarray | None = None,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    unit_distance = _full_history_unit_distance(
        history_component, instance.history, affected
    )
    if unit_distance <= 1e-10:
        raise ValueError("controlled_component_not_visible")
    units: list[_UnitTreatment] = []
    for level, interval in zip(CAPABILITY_LEVELS, STRENGTH_INTERVALS, strict=True):
        draw = float(
            _rng(
                instance.official_instance_id,
                capability_id,
                level,
                augmentation_seed=augmentation_seed,
            ).uniform(*interval)
        )
        gain = draw / unit_distance
        units.append(
            _UnitTreatment(
                history_delta=history_component * gain,
                future_delta=future_component * gain,
                affected=affected,
                coordinate_name="full_history_macro_normalized_rms",
                coordinate_interval=interval,
                sampled_coordinate=draw,
                metadata={
                    **metadata,
                    "unit_component_distance": unit_distance,
                    "physical_component_gain": gain,
                },
                history_covariate_delta=(
                    None
                    if history_covariate_component is None
                    else np.asarray(history_covariate_component * gain, dtype=float)
                ),
                future_covariate_delta=(
                    None
                    if future_covariate_component is None
                    else np.asarray(future_covariate_component * gain, dtype=float)
                ),
            )
        )
    return units, {"unit_component_distance": unit_distance}


def _regime_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    if length < 40:
        raise ValueError("history_too_short_for_five_regime_locations")
    scales = _scale_by_target(history)
    signs = _rng(
        instance.official_instance_id,
        "regime_switching",
        "shared-sign",
        augmentation_seed=augmentation_seed,
    ).choice(np.asarray([-1.0, 1.0]), size=dimension)
    units: list[_UnitTreatment] = []
    for level, interval in zip(
        CAPABILITY_LEVELS, REGIME_RECENCY_INTERVALS, strict=True
    ):
        recency = float(
            _rng(
                instance.official_instance_id,
                "regime_switching",
                level,
                augmentation_seed=augmentation_seed,
            ).uniform(*interval)
        )
        join = int(np.clip(round(recency * length), 8, length - 8))
        history_delta = np.zeros_like(history)
        history_delta[join:] = signs * scales
        future_delta = np.tile(signs * scales, (instance.prediction_length, 1))
        units.append(
            _UnitTreatment(
                history_delta=history_delta,
                future_delta=future_delta,
                affected=tuple(range(dimension)),
                coordinate_name="change_location_fraction_of_history",
                coordinate_interval=interval,
                sampled_coordinate=recency,
                metadata={
                    "change_index": join,
                    "post_change_history_length": length - join,
                    "shared_amplitude_before_distance_adjustment": scales.tolist(),
                    "direction_by_target": signs.tolist(),
                },
            )
        )
    return _shared_amplitude_units(instance, units, "regime_switching")


def _intermittency_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    if length < 24:
        raise ValueError("history_too_short_for_predictable_intermittency")
    scales = _scale_by_target(history)
    maximum_gap = min(instance.prediction_length, max(3, (length - 1) // 3))
    if maximum_gap < 8:
        raise ValueError("history_or_horizon_too_short_for_five_sparse_levels")
    width = int(
        _rng(
            instance.official_instance_id,
            "predictable_intermittency",
            "shared-width",
            augmentation_seed=augmentation_seed,
        ).integers(1, min(4, maximum_gap))
    )
    units: list[_UnitTreatment] = []
    total = length + instance.prediction_length
    for level, interval in zip(
        CAPABILITY_LEVELS, INTERMITTENCY_GAP_INTERVALS, strict=True
    ):
        fraction = float(
            _rng(
                instance.official_instance_id,
                "predictable_intermittency",
                level,
                augmentation_seed=augmentation_seed,
            ).uniform(*interval)
        )
        gap = int(np.clip(round(fraction * maximum_gap), width + 1, maximum_gap))
        phase_rng = _rng(
            instance.official_instance_id,
            "predictable_intermittency",
            level,
            "phase",
            augmentation_seed=augmentation_seed,
        )
        lag: int | None = None
        centers: list[int] = []
        phase_candidates = phase_rng.permutation(
            min(gap, instance.prediction_length)
        )
        for candidate in phase_candidates:
            candidate_lag = int(candidate)
            last_history_center = length - 1 - candidate_lag
            candidate_centers = list(
                range(last_history_center, -1, -gap)
            )[::-1]
            candidate_centers.extend(
                range(last_history_center + gap, total, gap)
            )
            if sum(center < length for center in candidate_centers) < 3:
                continue
            future_indices = {
                center + offset - length
                for center in candidate_centers
                for offset in range(width)
                if length <= center + offset < total
            }
            if not future_indices:
                continue
            if not np.any(
                instance.future_observed_mask[sorted(future_indices)]
            ):
                continue
            lag = candidate_lag
            centers = candidate_centers
            break
        if lag is None:
            raise ValueError("no_observed_scheduled_future_event")
        delta = np.zeros((total, dimension), dtype=float)
        for center in centers:
            for offset in range(width):
                index = center + offset
                if 0 <= index < total:
                    delta[index] += scales
        units.append(
            _UnitTreatment(
                history_delta=delta[:length],
                future_delta=delta[length:],
                affected=tuple(range(dimension)),
                coordinate_name="event_gap_fraction_of_maximum_legal_gap",
                coordinate_interval=interval,
                sampled_coordinate=fraction,
                metadata={
                    "event_gap": gap,
                    "pulse_width": width,
                    "history_event_count": sum(center < length for center in centers),
                    "future_event_count": sum(length <= center < total for center in centers),
                    "event_centers": centers,
                    "phase_candidate_count": int(len(phase_candidates)),
                    "future_event_observed_mask_required": True,
                    "positive_event_amplitude_before_distance_adjustment": scales.tolist(),
                },
            )
        )
    return _shared_amplitude_units(
        instance,
        units,
        "predictable_intermittency",
        minimum_future_effect_mase_rms=MECHANISM_EFFECT_MINIMUM_MASE_RMS,
    )


def _shared_amplitude_units(
    instance: GiftEvalInstance,
    units: list[_UnitTreatment],
    capability_id: str,
    *,
    minimum_future_effect_mase_rms: float | None = None,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    minimum = min(
        _full_history_unit_distance(unit.history_delta, instance.history, unit.affected)
        for unit in units
    )
    if minimum <= 1e-10:
        raise ValueError(f"{capability_id}_weakest_level_not_visible")
    history_gain = SOURCE_DISTANCE_MINIMUM_MACRO / minimum
    future_gain = 0.0
    future_signals: list[float] = []
    if minimum_future_effect_mase_rms is not None:
        mase_scales = mase_scale_by_target(instance.history, instance.frequency)
        future_signals = [
            mechanism_effect_signal(
                unit.future_delta,
                instance.future_observed_mask,
                mase_scales,
                unit.affected,
            )[1]
            for unit in units
        ]
        minimum_future_signal = min(future_signals)
        if minimum_future_signal <= 1e-12:
            raise ValueError("future_mechanism_effect_not_observable")
        future_gain = (
            float(minimum_future_effect_mase_rms) / minimum_future_signal
        )
    shared_gain = max(1.0, history_gain, future_gain)
    if minimum_future_effect_mase_rms is not None:
        shared_gain = float(np.nextafter(shared_gain, math.inf))
    scaled = [
        _UnitTreatment(
            history_delta=unit.history_delta * shared_gain,
            future_delta=unit.future_delta * shared_gain,
            affected=unit.affected,
            coordinate_name=unit.coordinate_name,
            coordinate_interval=unit.coordinate_interval,
            sampled_coordinate=unit.sampled_coordinate,
            metadata={
                **unit.metadata,
                "shared_amplitude_adjustment": shared_gain,
                "amplitude_policy": (
                    "one_gain_shared_by_all_five_levels_to_meet_treatment_"
                    "source_distance_and_future_effect_visibility"
                    if future_signals
                    else "one_history_only_gain_shared_by_all_five_levels_to_"
                    "meet_treatment_source_distance"
                ),
                "future_effect_mase_rms_before_adjustment": (
                    future_signals[index] if future_signals else None
                ),
                "future_effect_mase_rms_after_adjustment": (
                    future_signals[index] * shared_gain
                    if future_signals
                    else None
                ),
                "minimum_future_effect_mase_rms": (
                    minimum_future_effect_mase_rms
                ),
            },
        )
        for index, unit in enumerate(units)
    ]
    return scaled, {
        "shared_amplitude_adjustment": shared_gain,
        "weakest_pre_adjustment_distance": minimum,
        "minimum_future_effect_mase_rms": minimum_future_effect_mase_rms,
        "minimum_future_effect_mase_rms_after_adjustment": (
            min(future_signals) * shared_gain if future_signals else None
        ),
        "future_effect_signal_uses_target_values": False,
        "future_effect_signal_uses_observed_mask": (
            minimum_future_effect_mase_rms is not None
        ),
    }


def _nonlinear_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    if length < 48:
        raise ValueError("history_too_short_for_nonlinear_persistence")
    scales = _scale_by_target(history)
    component_h = np.zeros_like(history)
    component_f = np.zeros_like(instance.future)
    affected: list[int] = []
    diagnostics: dict[str, Any] = {}
    for channel in range(dimension):
        z = (history[:, channel] - np.mean(history[:, channel])) / scales[channel]
        previous = z[:-1]
        response = z[1:]
        quadratic = np.square(previous) - float(np.mean(np.square(previous)))
        linear_design = np.column_stack((np.ones(previous.size), previous))
        nonlinear_design = np.column_stack((linear_design, quadratic))
        linear_fit = linear_design @ np.linalg.lstsq(linear_design, response, rcond=None)[0]
        coefficients = np.linalg.lstsq(nonlinear_design, response, rcond=None)[0]
        nonlinear_fit = nonlinear_design @ coefficients
        baseline_error = float(np.mean(np.square(response - linear_fit)))
        gain = (
            0.0
            if baseline_error <= 1e-12
            else 1.0 - float(np.mean(np.square(response - nonlinear_fit))) / baseline_error
        )
        beta = float(coefficients[-1])
        if gain < 0.005 or abs(beta) < 1e-4:
            continue
        history_component = np.zeros(length, dtype=float)
        history_component[1:] = scales[channel] * beta * quadratic
        component_h[:, channel] = history_component
        state = float(z[-1])
        mean_square = float(np.mean(np.square(previous)))
        for step in range(instance.prediction_length):
            q = state * state - mean_square
            component_f[step, channel] = scales[channel] * beta * q
            state = float(coefficients[0] + coefficients[1] * state)
            state = float(np.clip(state, -8.0, 8.0))
        affected.append(channel)
        diagnostics[str(channel)] = {
            "in_sample_incremental_r2": gain,
            "quadratic_coefficient": beta,
            "future_innovation_policy": "zero_innovation_history_state_rollout",
        }
    if not affected:
        raise ValueError("nonlinear_incremental_predictive_gain_not_resolved")
    return _strength_scaled_units(
        instance,
        augmentation_seed,
        "nonlinear_persistence",
        component_h,
        component_f,
        tuple(affected),
        metadata={
            "component": "history_fitted_quadratic_lag_state",
            "diagnostics_by_target": diagnostics,
            "target_future_used_for_delta": False,
        },
    )


def _common_factor_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    if dimension < 3 or length < 24:
        raise ValueError("common_factor_requires_native_panel_d_ge_3")
    means = np.mean(history, axis=0)
    scales = _scale_by_target(history)
    z = (history - means) / scales
    _u, singular, vt = np.linalg.svd(z, full_matrices=False)
    share = float(singular[0] ** 2 / np.sum(np.square(singular)))
    if share < 1.0 / dimension + 0.02:
        raise ValueError("top_common_factor_share_too_small")
    loading = vt[0]
    affected = tuple(
        int(index)
        for index in np.flatnonzero(np.abs(loading) >= 0.25 * np.max(np.abs(loading)))
    )
    if len(affected) < min(3, dimension):
        raise ValueError("too_few_nondegenerate_common_factor_loadings")
    factor = z @ loading
    factor_variance = float(np.var(factor))
    if factor_variance <= 1e-12:
        raise ValueError("common_factor_latent_variance_too_small")
    best: tuple[float, int, float, float, np.ndarray] | None = None
    for frequency_index in _dominant_frequency_indexes(factor):
        sin_coefficient, cos_coefficient = _harmonic_coefficients(
            factor, frequency_index
        )
        carrier = _harmonic_signal(
            length=length,
            horizon=instance.prediction_length,
            frequency_index=frequency_index,
            sin_coefficient=sin_coefficient,
            cos_coefficient=cos_coefficient,
        )
        fitted_share = max(
            0.0,
            1.0
            - float(np.mean(np.square(factor - carrier[:length])))
            / factor_variance,
        )
        candidate = (
            fitted_share,
            frequency_index,
            sin_coefficient,
            cos_coefficient,
            carrier,
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None or best[0] < COMMON_FACTOR_MINIMUM_HARMONIC_SHARE:
        raise ValueError("common_factor_stable_latent_carrier_not_resolved")
    fitted_share, frequency_index, sin_coefficient, cos_coefficient, carrier = best
    history_component = (
        carrier[:length, None] * loading[None, :] * scales[None, :]
    )
    future_component = (
        carrier[length:, None] * loading[None, :] * scales[None, :]
    )
    return _strength_scaled_units(
        instance,
        augmentation_seed,
        "common_factor",
        history_component,
        future_component,
        affected,
        metadata={
            "component": "history_pca_loading_with_stable_latent_harmonic",
            "top1_explained_share": share,
            "loading": loading.tolist(),
            "latent_carrier_frequency_index": int(frequency_index),
            "latent_carrier_period": float(length / frequency_index),
            "latent_carrier_sin_coefficient": float(sin_coefficient),
            "latent_carrier_cos_coefficient": float(cos_coefficient),
            "latent_carrier_history_explained_share": float(fitted_share),
            "future_continuation": "analytic_constant_amplitude_harmonic",
        },
    )


def _cross_series_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    if dimension < 2 or length < 48:
        raise ValueError("cross_series_requires_native_panel_d_ge_2")
    scales = _scale_by_target(history)
    z = (history - np.mean(history, axis=0)) / scales
    best: tuple[float, int, int, int, np.ndarray] | None = None
    maximum_lag = min(24, max(1, length // 8))
    for driver in range(dimension):
        for responder in range(dimension):
            if responder == driver:
                continue
            for lag in range(1, maximum_lag + 1):
                y = z[lag:, responder]
                own = z[lag - 1 : -1, responder]
                source = z[:-lag, driver]
                own_design = np.column_stack((np.ones(y.size), own))
                full_design = np.column_stack((own_design, source))
                own_fit = own_design @ np.linalg.lstsq(own_design, y, rcond=None)[0]
                coefficients = np.linalg.lstsq(full_design, y, rcond=None)[0]
                full_fit = full_design @ coefficients
                baseline_error = float(np.mean(np.square(y - own_fit)))
                incremental = (
                    0.0
                    if baseline_error <= 1e-12
                    else 1.0 - float(np.mean(np.square(y - full_fit))) / baseline_error
                )
                if best is None or incremental > best[0]:
                    best = (incremental, driver, responder, lag, coefficients)
    if best is None or best[0] < 0.0025:
        raise ValueError("directed_incremental_predictive_gain_too_small")
    incremental, driver, responder, lag, coefficients = best
    beta = float(coefficients[-1])
    component_h = np.zeros_like(history)
    component_h[lag:, responder] = scales[responder] * beta * z[:-lag, driver]
    driver_series = z[:, driver : driver + 1]
    driver_future = _linear_extrapolation(driver_series, instance.prediction_length)[:, 0]
    extended_driver = np.concatenate((z[:, driver], driver_future))
    component_f = np.zeros_like(instance.future)
    for step in range(instance.prediction_length):
        source_index = length + step - lag
        component_f[step, responder] = (
            scales[responder] * beta * extended_driver[source_index]
        )
    return _strength_scaled_units(
        instance,
        augmentation_seed,
        "cross_series_dependence",
        component_h,
        component_f,
        (responder,),
        metadata={
            "component": "directed_linear_predictive_transfer",
            "driver_target_index": driver,
            "responder_target_index": responder,
            "lag": lag,
            "incremental_r2": incremental,
            "transfer_coefficient": beta,
            "claim_scope": "predictive_not_causal",
            "target_future_used_for_delta": False,
        },
    )


def _covariate_impulse_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    covariates = instance.history_covariates
    length = int(history.shape[0])
    horizon = int(instance.prediction_length)
    if covariates.shape[1] == 0:
        raise ValueError("no_native_dynamic_covariate")
    period = max(8, 2 * horizon)
    if length < 2 * period:
        raise ValueError("insufficient_history_for_repeated_covariate_impulses")
    covariate_scales = np.sqrt(
        np.mean(
            np.square(covariates - np.mean(covariates, axis=0)),
            axis=0,
        )
    )
    legal = np.flatnonzero(
        np.isfinite(covariate_scales) & (covariate_scales > 1e-8)
    )
    if legal.size == 0:
        raise ValueError("native_covariates_have_no_nonconstant_channel")
    covariate = int(legal[np.argmax(covariate_scales[legal])])
    scales = _scale_by_target(history)
    target_index = int(np.argmax(scales))
    total = length + horizon
    impulse = np.zeros(total, dtype=float)
    historical_centers: list[int] = []
    center = length - 1
    while center >= 0:
        impulse[center] = 1.0
        historical_centers.append(center)
        center -= period
    historical_centers.sort()
    future_centers: list[int] = []
    if instance.future_covariate_visible[covariate]:
        future_center = length + max(0, horizon // 2)
        if future_center < total:
            impulse[future_center] = 1.0
            future_centers.append(future_center)
    kernel_index = np.arange(2 * horizon + 1, dtype=float)
    kernel = np.exp(-math.log(2.0) * kernel_index / float(max(1, horizon)))
    mase_scales = mase_scale_by_target(instance.history, instance.frequency)
    minimum_coordinate = float(STRENGTH_INTERVALS[0][0])
    terminal_amplitude = 1.0
    constructed_minimum_future_effect = 0.0
    response = np.zeros(total, dtype=float)
    history_component = np.zeros_like(history)
    future_component = np.zeros_like(instance.future)
    for _attempt in range(24):
        impulse[length - 1] = terminal_amplitude
        response = np.convolve(impulse, kernel, mode="full")[:total]
        history_component.fill(0.0)
        future_component.fill(0.0)
        history_component[:, target_index] = (
            scales[target_index] * response[:length]
        )
        future_component[:, target_index] = (
            scales[target_index] * response[length:]
        )
        unit_distance = _full_history_unit_distance(
            history_component, instance.history, (target_index,)
        )
        unit_future_effect = mechanism_effect_signal(
            future_component,
            instance.future_observed_mask,
            mase_scales,
            (target_index,),
        )[1]
        if unit_distance > 1e-12:
            constructed_minimum_future_effect = (
                minimum_coordinate * unit_future_effect / unit_distance
            )
        if (
            constructed_minimum_future_effect
            >= MECHANISM_EFFECT_MINIMUM_MASE_RMS
        ):
            break
        terminal_amplitude *= 2.0
    else:
        raise ValueError("cannot_construct_scoreable_covariate_response_tail")
    history_covariate_component = np.zeros_like(instance.history_covariates)
    future_covariate_component = np.zeros_like(instance.future_covariates)
    history_covariate_component[:, covariate] = (
        covariate_scales[covariate] * impulse[:length]
    )
    if instance.future_covariate_visible[covariate]:
        future_covariate_component[:, covariate] = (
            covariate_scales[covariate] * impulse[length:]
        )
    history_covariate_component[length - 1, covariate] = (
        covariate_scales[covariate] * terminal_amplitude
    )
    future_rms = float(np.sqrt(np.mean(np.square(response[length:]))))
    if future_rms <= 1e-8:
        raise ValueError("constructed_impulse_response_has_zero_future_energy")
    return _strength_scaled_units(
        instance,
        augmentation_seed,
        "covariate_impulse_response",
        history_component,
        future_component,
        (target_index,),
        metadata={
            "component": "native_covariate_fixed_causal_impulse_response",
            "eligible_target_index": target_index,
            "covariate_index": covariate,
            "covariate_name": instance.covariate_column_names[covariate],
            "covariate_availability": instance.covariate_availability[covariate],
            "historical_impulse_centers": historical_centers,
            "future_impulse_centers": future_centers,
            "terminal_impulse_amplitude": terminal_amplitude,
            "impulse_period": period,
            "kernel": "exponential_half_life_equal_to_forecast_horizon",
            "kernel_length": int(kernel.size),
            "kernel_half_life": horizon,
            "covariate_unit_scale": float(covariate_scales[covariate]),
            "unit_future_response_rms": future_rms,
            "constructed_minimum_future_effect_mase_rms": (
                constructed_minimum_future_effect
            ),
            "minimum_required_future_effect_mase_rms": (
                MECHANISM_EFFECT_MINIMUM_MASE_RMS
            ),
            "future_covariate_path_visible_to_model": bool(
                instance.future_covariate_visible[covariate]
            ),
            "target_future_used_for_delta": False,
        },
        history_covariate_component=history_covariate_component,
        future_covariate_component=future_covariate_component,
    )


_BUILDERS: dict[
    str,
    Callable[[GiftEvalInstance, int], tuple[list[_UnitTreatment], dict[str, Any]]],
] = {
    "trend": _trend_units,
    "multi_seasonal": _secondary_seasonal_units,
    "time_varying_seasonality": _time_varying_units,
    "regime_switching": _regime_units,
    "nonlinear_persistence": _nonlinear_units,
    "predictable_intermittency": _intermittency_units,
    "common_factor": _common_factor_units,
    "cross_series_dependence": _cross_series_units,
    "covariate_impulse_response": _covariate_impulse_units,
}


def _horizon_support_gate(
    instance: GiftEvalInstance,
    capability_id: str,
    unit: _UnitTreatment,
) -> dict[str, Any] | None:
    if capability_id == "time_varying_seasonality":
        details = unit.metadata["resolved_periods_by_target"]
        future_t = np.arange(
            instance.context_length,
            instance.context_length + instance.prediction_length,
            dtype=float,
        )
        by_target: dict[str, Any] = {}
        active_fractions: list[float] = []
        for channel in unit.affected:
            row = details[str(channel)]
            modulation_index = int(row["modulation_frequency_index"])
            omega = (
                2.0
                * math.pi
                * float(modulation_index)
                / float(instance.context_length)
            )
            envelope = (
                float(row["envelope_sin_coefficient"])
                * np.sin(omega * future_t)
                + float(row["envelope_cos_coefficient"])
                * np.cos(omega * future_t)
            )
            amplitude = float(row["envelope_amplitude"])
            observed = np.asarray(
                instance.future_observed_mask[:, channel], dtype=bool
            )
            active = (
                np.abs(envelope)
                >= TVS_ENVELOPE_ACTIVE_AMPLITUDE_FRACTION * amplitude
            )
            fraction = (
                float(np.mean(active[observed])) if np.any(observed) else 0.0
            )
            active_fractions.append(fraction)
            by_target[str(channel)] = {
                "observed_future_count": int(np.count_nonzero(observed)),
                "active_future_count": int(np.count_nonzero(active & observed)),
                "active_fraction": fraction,
            }
        minimum = min(active_fractions, default=0.0)
        accepted = minimum >= TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION - 1e-12
        return {
            "schema_version": "cafe.capability_horizon_support_gate.v1",
            "capability_id": capability_id,
            "metric": "future_envelope_active_fraction_by_affected_target",
            "horizon_partition": "whole_forecast_horizon",
            "active_amplitude_fraction": (
                TVS_ENVELOPE_ACTIVE_AMPLITUDE_FRACTION
            ),
            "minimum_required_active_fraction": (
                TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION
            ),
            "minimum_observed_active_fraction": minimum,
            "by_target": by_target,
            "target_future_values_used": False,
            "accepted": accepted,
            "reason": None if accepted else "future_envelope_coverage_too_small",
        }
    if capability_id == "common_factor":
        scales = _scale_by_target(instance.history)[list(unit.affected)]
        standardized = unit.future_delta[:, unit.affected] / scales[None, :]
        sections = np.array_split(
            np.arange(instance.prediction_length, dtype=int), 3
        )
        section_rms: list[float] = []
        section_names = ("head", "middle", "tail")
        by_section: dict[str, Any] = {}
        for name, indexes in zip(section_names, sections, strict=True):
            channel_rms: list[float] = []
            for position, channel in enumerate(unit.affected):
                observed = instance.future_observed_mask[indexes, channel]
                values = standardized[indexes, position][observed]
                if values.size:
                    channel_rms.append(
                        float(np.sqrt(np.mean(np.square(values))))
                    )
            macro = float(np.mean(channel_rms)) if channel_rms else 0.0
            section_rms.append(macro)
            by_section[name] = {
                "time_count": int(indexes.size),
                "observed_target_count": len(channel_rms),
                "macro_normalized_rms": macro,
            }
        head, _middle, tail = section_rms
        ratio = tail / max(head, 1e-12)
        accepted = (
            head > 0.0
            and tail > 0.0
            and ratio >= COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO - 1e-12
        )
        return {
            "schema_version": "cafe.capability_horizon_support_gate.v1",
            "capability_id": capability_id,
            "metric": "common_factor_tail_to_head_macro_normalized_rms_ratio",
            "horizon_partition": "three_equal_relative_sections",
            "minimum_required_tail_head_ratio": (
                COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO
            ),
            "observed_tail_head_ratio": ratio,
            "by_section": by_section,
            "target_future_values_used": False,
            "accepted": accepted,
            "reason": None if accepted else "common_factor_tail_support_too_small",
        }
    return None


def build_capability_group(
    instance: GiftEvalInstance,
    capability_id: str,
    *,
    augmentation_seed: int,
) -> CapabilityGroup:
    if capability_id not in CAPABILITY_IDS:
        raise ValueError(f"unknown capability {capability_id!r}")
    if capability_id == "hierarchical_coherence":
        return CapabilityGroup(
            capability_id=capability_id,
            available=False,
            reason="qualification_only_no_generation",
            treatments=(),
            group_metadata={"ranking_eligible": False},
        )
    builder = _BUILDERS[capability_id]
    try:
        units, group_metadata = builder(instance, int(augmentation_seed))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return CapabilityGroup(
            capability_id=capability_id,
            available=False,
            reason=str(error),
            treatments=(),
            group_metadata={},
        )
    if len(units) != len(CAPABILITY_LEVELS):
        raise RuntimeError(f"{capability_id} did not produce five levels")
    treatments: list[CapabilityTreatment] = []
    for level, unit in zip(CAPABILITY_LEVELS, units, strict=True):
        gate = _distance_gate(
            unit.history_delta,
            instance.history,
            unit.affected,
        )
        if not gate["accepted"]:
            return CapabilityGroup(
                capability_id=capability_id,
                available=False,
                reason=f"level_{level}_{gate['reason']}",
                treatments=(),
                group_metadata={
                    **group_metadata,
                    "failed_level": level,
                    "failed_source_distance_gate": gate,
                },
            )
        horizon_support_gate = _horizon_support_gate(
            instance, capability_id, unit
        )
        if (
            horizon_support_gate is not None
            and not horizon_support_gate["accepted"]
        ):
            return CapabilityGroup(
                capability_id=capability_id,
                available=False,
                reason=(
                    f"level_{level}_{horizon_support_gate['reason']}"
                ),
                treatments=(),
                group_metadata={
                    **group_metadata,
                    "failed_level": level,
                    "failed_horizon_support_gate": horizon_support_gate,
                },
            )
        treatments.append(
            CapabilityTreatment(
                level=level,
                history_delta=unit.history_delta,
                future_delta=unit.future_delta,
                affected_target_indices=unit.affected,
                controlled_coordinate=unit.coordinate_name,
                coordinate_interval=unit.coordinate_interval,
                sampled_coordinate=unit.sampled_coordinate,
                applied_component_gain=float(
                    unit.metadata.get(
                        "physical_component_gain",
                        unit.metadata.get(
                            "physical_linear_gain",
                            unit.metadata.get("shared_amplitude_adjustment", 1.0),
                        ),
                    )
                ),
                metadata=unit.metadata,
                source_distance_gate=gate,
                horizon_support_gate=horizon_support_gate,
                history_covariate_delta=(
                    np.zeros_like(instance.history_covariates)
                    if unit.history_covariate_delta is None
                    else np.asarray(unit.history_covariate_delta, dtype=float)
                ),
                future_covariate_delta=(
                    np.zeros_like(instance.future_covariates)
                    if unit.future_covariate_delta is None
                    else np.asarray(unit.future_covariate_delta, dtype=float)
                ),
            )
        )
    parameter_payload = [
        {
            "level": treatment.level,
            "coordinate": treatment.sampled_coordinate,
            "interval": list(treatment.coordinate_interval),
            "gain": treatment.applied_component_gain,
            "history_delta_sha256": _array_sha256(treatment.history_delta),
            "future_delta_sha256": _array_sha256(treatment.future_delta),
            "history_covariate_delta_sha256": _array_sha256(
                treatment.history_covariate_delta
            ),
            "future_covariate_delta_sha256": _array_sha256(
                treatment.future_covariate_delta
            ),
        }
        for treatment in treatments
    ]
    return CapabilityGroup(
        capability_id=capability_id,
        available=True,
        reason=None,
        treatments=tuple(treatments),
        group_metadata={
            **group_metadata,
            "schema_version": MECHANISM_SCHEMA,
            "augmentation_seed": int(augmentation_seed),
            "parameter_draw_sha256": protocol.json_sha256(parameter_payload),
            "target_future_used_for_fit_or_parameter_draw": False,
            "source_distance_policy": (
                "full_history_strength_actual_model_context_bounds_v3"
            ),
        },
    )


def replay_treatment_deltas(
    instance: GiftEvalInstance,
    contracts: list[dict[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Apply frozen treatment contracts without repeating capability selection.

    Generation has already selected channels, periods, joins, lags, loadings and
    parameter draws.  Inference only reconstructs the corresponding component
    from the authentic source path and applies the stored physical gain.
    """

    if not contracts:
        return {}
    capability_ids = {str(row["capability_id"]) for row in contracts}
    if len(capability_ids) != 1:
        raise ValueError("treatment replay group mixes capabilities")
    capability_id = capability_ids.pop()
    history = np.asarray(instance.history, dtype=float)
    horizon = int(instance.prediction_length)
    length, dimension = history.shape
    scales = _scale_by_target(history)
    shared_components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def shared_component(
        row: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nonlocal shared_components
        if shared_components is not None:
            return shared_components
        metadata = dict(row["mechanism_metadata"])
        component_h = np.zeros_like(history)
        component_f = np.zeros_like(instance.future)
        covariate_h = np.zeros_like(instance.history_covariates)
        covariate_f = np.zeros_like(instance.future_covariates)
        if capability_id == "trend":
            directions = np.asarray(metadata["direction_by_target"], dtype=float)
            extended = np.arange(length + horizon, dtype=float)
            extended /= max(1, length - 1)
            affected = [int(value) for value in row["affected_target_indices"]]
            component = np.zeros((length + horizon, dimension), dtype=float)
            component[:, affected] = (
                extended[:, None]
                * directions[affected][None, :]
                * scales[affected][None, :]
            )
            component_h, component_f = component[:length], component[length:]
        elif capability_id == "multi_seasonal":
            details = metadata["resolved_periods_by_target"]
            for raw_channel, periods in details.items():
                channel = int(raw_channel)
                frequency_index = int(
                    round(length / float(periods["secondary_period"]))
                )
                fitted_h, fitted_f = _harmonic_component(
                    history[:, channel], frequency_index, horizon
                )
                component_h[:, channel] = fitted_h
                component_f[:, channel] = fitted_f
        elif capability_id == "time_varying_seasonality":
            details = metadata["resolved_periods_by_target"]
            for raw_channel, periods in details.items():
                channel = int(raw_channel)
                carrier = _harmonic_signal(
                    length=length,
                    horizon=horizon,
                    frequency_index=int(periods["carrier_frequency_index"]),
                    sin_coefficient=float(
                        periods["carrier_sin_coefficient"]
                    ),
                    cos_coefficient=float(
                        periods["carrier_cos_coefficient"]
                    ),
                )
                envelope = _harmonic_signal(
                    length=length,
                    horizon=horizon,
                    frequency_index=int(
                        periods["modulation_frequency_index"]
                    ),
                    sin_coefficient=float(
                        periods["envelope_sin_coefficient"]
                    ),
                    cos_coefficient=float(
                        periods["envelope_cos_coefficient"]
                    ),
                )
                combined = carrier * envelope
                component_h[:, channel] = combined[:length]
                component_f[:, channel] = combined[length:]
        elif capability_id == "nonlinear_persistence":
            diagnostics = metadata["diagnostics_by_target"]
            for raw_channel, diagnostic in diagnostics.items():
                channel = int(raw_channel)
                z = (history[:, channel] - np.mean(history[:, channel])) / scales[
                    channel
                ]
                previous = z[:-1]
                response = z[1:]
                quadratic = np.square(previous) - float(
                    np.mean(np.square(previous))
                )
                design = np.column_stack(
                    (np.ones(previous.size), previous, quadratic)
                )
                coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
                beta = float(diagnostic["quadratic_coefficient"])
                component_h[1:, channel] = scales[channel] * beta * quadratic
                state = float(z[-1])
                mean_square = float(np.mean(np.square(previous)))
                for step in range(horizon):
                    q = state * state - mean_square
                    component_f[step, channel] = scales[channel] * beta * q
                    state = float(coefficients[0] + coefficients[1] * state)
                    state = float(np.clip(state, -8.0, 8.0))
        elif capability_id == "common_factor":
            loading = np.asarray(metadata["loading"], dtype=float)
            carrier = _harmonic_signal(
                length=length,
                horizon=horizon,
                frequency_index=int(
                    metadata["latent_carrier_frequency_index"]
                ),
                sin_coefficient=float(
                    metadata["latent_carrier_sin_coefficient"]
                ),
                cos_coefficient=float(
                    metadata["latent_carrier_cos_coefficient"]
                ),
            )
            component = carrier[:, None] * loading[None, :] * scales[None, :]
            component_h, component_f = component[:length], component[length:]
        elif capability_id == "cross_series_dependence":
            driver = int(metadata["driver_target_index"])
            responder = int(metadata["responder_target_index"])
            lag = int(metadata["lag"])
            beta = float(metadata["transfer_coefficient"])
            z = (history - np.mean(history, axis=0)) / scales
            component_h[lag:, responder] = (
                scales[responder] * beta * z[:-lag, driver]
            )
            driver_future = _linear_extrapolation(
                z[:, driver : driver + 1], horizon
            )[:, 0]
            extended_driver = np.concatenate((z[:, driver], driver_future))
            for step in range(horizon):
                source_index = length + step - lag
                component_f[step, responder] = (
                    scales[responder] * beta * extended_driver[source_index]
                )
        elif capability_id == "covariate_impulse_response":
            target = int(metadata["eligible_target_index"])
            covariate = int(metadata["covariate_index"])
            impulse = np.zeros(length + horizon, dtype=float)
            for center in metadata["historical_impulse_centers"]:
                impulse[int(center)] = 1.0
            impulse[length - 1] = float(metadata["terminal_impulse_amplitude"])
            for center in metadata["future_impulse_centers"]:
                impulse[int(center)] = 1.0
            kernel_index = np.arange(int(metadata["kernel_length"]), dtype=float)
            half_life = float(metadata["kernel_half_life"])
            kernel = np.exp(-math.log(2.0) * kernel_index / half_life)
            response = np.convolve(impulse, kernel, mode="full")[: length + horizon]
            component_h[:, target] = scales[target] * response[:length]
            component_f[:, target] = scales[target] * response[length:]
            covariate_scale = float(metadata["covariate_unit_scale"])
            covariate_h[:, covariate] = covariate_scale * impulse[:length]
            if bool(metadata["future_covariate_path_visible_to_model"]):
                covariate_f[:, covariate] = covariate_scale * impulse[length:]
        else:
            raise ValueError(f"capability {capability_id!r} has level-specific replay")
        shared_components = component_h, component_f, covariate_h, covariate_f
        return shared_components

    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in contracts:
        metadata = dict(row["mechanism_metadata"])
        if capability_id == "regime_switching":
            component_h = np.zeros_like(history)
            component_f = np.zeros_like(instance.future)
            covariate_h = np.zeros_like(instance.history_covariates)
            covariate_f = np.zeros_like(instance.future_covariates)
            join = int(metadata["change_index"])
            amplitude = np.asarray(
                metadata["shared_amplitude_before_distance_adjustment"],
                dtype=float,
            )
            direction = np.asarray(metadata["direction_by_target"], dtype=float)
            component_h[join:] = direction * amplitude
            component_f[:] = direction * amplitude
        elif capability_id == "predictable_intermittency":
            combined = np.zeros((length + horizon, dimension), dtype=float)
            amplitude = np.asarray(
                metadata["positive_event_amplitude_before_distance_adjustment"],
                dtype=float,
            )
            width = int(metadata["pulse_width"])
            for center in metadata["event_centers"]:
                for offset in range(width):
                    index = int(center) + offset
                    if 0 <= index < combined.shape[0]:
                        combined[index] += amplitude
            component_h, component_f = combined[:length], combined[length:]
            covariate_h = np.zeros_like(instance.history_covariates)
            covariate_f = np.zeros_like(instance.future_covariates)
        else:
            component_h, component_f, covariate_h, covariate_f = shared_component(row)
        gain = float(row["applied_component_gain"])
        output[str(row["sample_id"])] = (
            np.asarray(component_h * gain, dtype=float),
            np.asarray(component_f * gain, dtype=float),
            np.asarray(covariate_h * gain, dtype=float),
            np.asarray(covariate_f * gain, dtype=float),
        )
    return output
