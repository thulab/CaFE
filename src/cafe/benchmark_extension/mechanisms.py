from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.gift_eval import GiftEvalInstance


MECHANISM_SCHEMA = "cafe.native_path_mechanism.v10"
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
SOURCE_DISTANCE_MODEL_MAX_CONTEXTS_BY_TERM = {
    "short": SOURCE_DISTANCE_MODEL_MAX_CONTEXTS,
    "medium": {
        model_id: maximum
        for model_id, maximum in SOURCE_DISTANCE_MODEL_MAX_CONTEXTS.items()
        if model_id != "tirex2"
    },
    "long": {
        model_id: maximum
        for model_id, maximum in SOURCE_DISTANCE_MODEL_MAX_CONTEXTS.items()
        if model_id != "tirex2"
    },
}


def source_distance_model_max_contexts(term: str) -> dict[str, int]:
    try:
        values = SOURCE_DISTANCE_MODEL_MAX_CONTEXTS_BY_TERM[str(term)]
    except KeyError as error:
        raise ValueError(f"unsupported GIFT-Eval term {term!r}") from error
    return dict(values)
# Kept as a compatibility alias for callers that only need the lower bound.
SOURCE_DISTANCE_THRESHOLD = SOURCE_DISTANCE_MINIMUM_MACRO
MECHANISM_EFFECT_MINIMUM_MASE_RMS = 0.05
STRICT_FUTURE_EFFECT_CAPABILITIES = frozenset(
    {
        "nonlinear_persistence",
        "predictable_intermittency",
        "covariate_impulse_response",
    }
)
TVS_ENVELOPE_ACTIVE_AMPLITUDE_FRACTION = 0.25
TVS_ENVELOPE_MINIMUM_ACTIVE_FRACTION = 0.25
TVS_MINIMUM_INCREMENTAL_R2 = 0.01
COMMON_FACTOR_MINIMUM_HARMONIC_SHARE = 0.05
COMMON_FACTOR_MINIMUM_TAIL_HEAD_RMS_RATIO = 0.50
MULTI_SEASONAL_MAXIMUM_ADDITIONAL_PERIODS = 5
MULTI_SEASONAL_REAL_ANCHOR_CANDIDATE_COUNT = 3
MULTI_SEASONAL_COMPONENT_VISIBILITY = 0.05
MULTI_SEASONAL_SPLIT_PHASE_COSINE_MINIMUM = 0.50
MULTI_SEASONAL_SPLIT_AMPLITUDE_RATIO_MINIMUM = 0.50
MULTI_SEASONAL_MAXIMUM_HARMONIC_MULTIPLE = 8
MULTI_SEASONAL_MINIMUM_PERIOD = 4.0
MULTI_SEASONAL_MINIMUM_HISTORY_CYCLES = 3.0
MULTI_SEASONAL_MINIMUM_FUTURE_CYCLE_FRACTION = 0.50
MULTI_SEASONAL_MINIMUM_FREQUENCY_SEPARATION_CYCLES = 0.50
MULTI_SEASONAL_HARMONIC_RELATIVE_TOLERANCE = 0.05
MULTI_SEASONAL_PERIOD_CANDIDATE_COUNT = 512
MULTI_SEASONAL_SHARED_DISTANCE_INTERVAL = (0.22, 0.28)
NONLINEAR_MINIMUM_HISTORY = 96
NONLINEAR_HOLDOUT_FRACTION = 0.25
NONLINEAR_MINIMUM_HOLDOUT_SIZE = 24
NONLINEAR_MINIMUM_HOLDOUT_R2_GAIN = 0.005
NONLINEAR_MINIMUM_COEFFICIENT_ABS = 0.01
NONLINEAR_ORDINARY_STATE_MAXIMUM_ABS = 0.75
NONLINEAR_EXTREME_STATE_MINIMUM_ABS = 1.50
NONLINEAR_MINIMUM_TRAIN_ORDINARY_COUNT = 8
NONLINEAR_MINIMUM_TRAIN_EXTREME_COUNT = 4
NONLINEAR_MINIMUM_HOLDOUT_ORDINARY_COUNT = 4
NONLINEAR_MINIMUM_HOLDOUT_EXTREME_COUNT = 2
NONLINEAR_MINIMUM_HALF_COEFFICIENT_RATIO = 0.10
NONLINEAR_MINIMUM_MULTISTEP_HOLDOUT_R2_GAIN = 0.0
NONLINEAR_MULTISTEP_AUDIT_ORIGIN_COUNT = 4
NONLINEAR_STABILITY_LIMIT = 0.98
NONLINEAR_STATE_ABSOLUTE_LIMIT = 8.0
NONLINEAR_FUTURE_INNOVATION_PATH_COUNT = 128
NONLINEAR_FUTURE_INNOVATION_MINIMUM_BLOCK_LENGTH = 4
NONLINEAR_MINIMUM_FUTURE_PROFILE_RANGE = 0.10
NONLINEAR_MAXIMUM_FUTURE_PEAK_FRACTION = 0.50
NONLINEAR_MAXIMUM_TAIL_TO_PEAK_RATIO = 0.90
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
NONLINEAR_PERSISTENCE_INTERVALS = (
    (0.20, 0.26),
    (0.30, 0.38),
    (0.42, 0.50),
    (0.56, 0.66),
    (0.72, 0.84),
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
    *,
    model_max_contexts: dict[str, int] | None = None,
) -> dict[str, Any]:
    delta = np.asarray(history_delta, dtype=float)
    scales = _scale_by_target(history)
    history_length = int(history.shape[0])
    model_ids_by_context: dict[int, list[str]] = {}
    configured_contexts = dict(
        SOURCE_DISTANCE_MODEL_MAX_CONTEXTS
        if model_max_contexts is None
        else model_max_contexts
    )
    if not configured_contexts:
        raise ValueError("source distance requires at least one model context")
    for model_id, maximum in configured_contexts.items():
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
        "model_max_contexts": configured_contexts,
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
    valid_indexes = indexes[valid]
    ranked = valid_indexes[np.argsort(spectrum[valid_indexes])[::-1]]
    return [int(index) for index in ranked[:16]]


def _independent_seasonal_period(
    left: float,
    right: float,
    context_length: int,
) -> bool:
    left_frequency = 1.0 / float(left)
    right_frequency = 1.0 / float(right)
    separation = abs(left_frequency - right_frequency) * context_length
    if separation < MULTI_SEASONAL_MINIMUM_FREQUENCY_SEPARATION_CYCLES:
        return False
    ratio = max(left_frequency, right_frequency) / min(
        left_frequency, right_frequency
    )
    for multiple in range(2, MULTI_SEASONAL_MAXIMUM_HARMONIC_MULTIPLE + 1):
        if (
            abs(ratio - multiple) / multiple
            <= MULTI_SEASONAL_HARMONIC_RELATIVE_TOLERANCE
        ):
            return False
    return True


def _segmented_harmonic_fit(
    values: np.ndarray,
    frequency_index: int,
    start: int,
    stop: int,
) -> tuple[np.ndarray, float]:
    series = np.asarray(values, dtype=float)
    segment = series[start:stop]
    if segment.size < 8:
        return np.zeros(2, dtype=float), 0.0
    local_t = np.linspace(-1.0, 1.0, segment.size)
    trend = np.column_stack((np.ones(segment.size), local_t))
    detrended = segment - trend @ np.linalg.lstsq(
        trend, segment, rcond=None
    )[0]
    absolute_t = np.arange(start, stop, dtype=float)
    omega = 2.0 * math.pi * float(frequency_index) / float(series.size)
    harmonic = np.column_stack(
        (np.sin(omega * absolute_t), np.cos(omega * absolute_t))
    )
    coefficients = np.linalg.lstsq(harmonic, detrended, rcond=None)[0]
    fitted = harmonic @ coefficients
    return np.asarray(coefficients, dtype=float), float(np.std(fitted))


def _harmonic_split_stability(
    values: np.ndarray,
    frequency_index: int,
    scale: float,
) -> dict[str, Any]:
    series = np.asarray(values, dtype=float)
    midpoint = series.size // 2
    first, first_std = _segmented_harmonic_fit(
        series, frequency_index, 0, midpoint
    )
    second, second_std = _segmented_harmonic_fit(
        series, frequency_index, midpoint, series.size
    )
    first_amplitude = float(np.linalg.norm(first))
    second_amplitude = float(np.linalg.norm(second))
    denominator = first_amplitude * second_amplitude
    phase_cosine = (
        float(np.dot(first, second) / denominator)
        if denominator > 1e-12
        else -1.0
    )
    amplitude_ratio = min(first_amplitude, second_amplitude) / max(
        first_amplitude, second_amplitude, 1e-12
    )
    normalized_stds = (first_std / scale, second_std / scale)
    accepted = (
        min(normalized_stds) >= MULTI_SEASONAL_COMPONENT_VISIBILITY - 1e-12
        and phase_cosine
        >= MULTI_SEASONAL_SPLIT_PHASE_COSINE_MINIMUM - 1e-12
        and amplitude_ratio
        >= MULTI_SEASONAL_SPLIT_AMPLITUDE_RATIO_MINIMUM - 1e-12
    )
    return {
        "first_half_normalized_std": float(normalized_stds[0]),
        "second_half_normalized_std": float(normalized_stds[1]),
        "coefficient_phase_cosine": phase_cosine,
        "coefficient_amplitude_ratio": amplitude_ratio,
        "minimum_required_half_normalized_std": (
            MULTI_SEASONAL_COMPONENT_VISIBILITY
        ),
        "minimum_required_phase_cosine": (
            MULTI_SEASONAL_SPLIT_PHASE_COSINE_MINIMUM
        ),
        "minimum_required_amplitude_ratio": (
            MULTI_SEASONAL_SPLIT_AMPLITUDE_RATIO_MINIMUM
        ),
        "accepted": accepted,
    }


def _continuous_harmonic_signal(
    *,
    length: int,
    horizon: int,
    period: float,
    sin_coefficient: float,
    cos_coefficient: float,
) -> np.ndarray:
    all_t = np.arange(length + horizon, dtype=float)
    omega = 2.0 * math.pi / float(period)
    return (
        float(sin_coefficient) * np.sin(omega * all_t)
        + float(cos_coefficient) * np.cos(omega * all_t)
    )


def _normalized_continuous_harmonic(
    *,
    length: int,
    horizon: int,
    period: float,
    sin_coefficient: float,
    cos_coefficient: float,
    target_scale: float,
) -> tuple[np.ndarray, float, float]:
    signal = _continuous_harmonic_signal(
        length=length,
        horizon=horizon,
        period=period,
        sin_coefficient=sin_coefficient,
        cos_coefficient=cos_coefficient,
    )
    history_std = float(np.std(signal[:length]))
    if history_std <= 1e-12:
        raise ValueError("multi_seasonal_component_has_zero_history_energy")
    adjustment = float(target_scale) / history_std
    adjusted_sin = float(sin_coefficient) * adjustment
    adjusted_cos = float(cos_coefficient) * adjustment
    normalized = _continuous_harmonic_signal(
        length=length,
        horizon=horizon,
        period=period,
        sin_coefficient=adjusted_sin,
        cos_coefficient=adjusted_cos,
    )
    return (
        normalized,
        adjusted_sin,
        adjusted_cos,
    )


def _protocol_generated_periods(
    instance: GiftEvalInstance,
    channel: int,
    *,
    shortest_context: int,
    existing_periods: tuple[float, ...],
    required_count: int,
    augmentation_seed: int,
) -> tuple[list[float], tuple[float, float]]:
    minimum = MULTI_SEASONAL_MINIMUM_PERIOD
    maximum = min(
        shortest_context / MULTI_SEASONAL_MINIMUM_HISTORY_CYCLES,
        instance.prediction_length
        / MULTI_SEASONAL_MINIMUM_FUTURE_CYCLE_FRACTION,
    )
    if maximum <= minimum + 1e-12:
        raise ValueError("history_or_horizon_too_short_for_artificial_period_pool")
    candidates = np.geomspace(
        minimum,
        maximum,
        MULTI_SEASONAL_PERIOD_CANDIDATE_COUNT,
    )
    order = _rng(
        instance.official_instance_id,
        "multi_seasonal",
        "period_pool",
        channel,
        augmentation_seed=augmentation_seed,
    ).permutation(candidates.size)
    selected = list(existing_periods)
    generated: list[float] = []
    for raw_index in order:
        period = float(candidates[int(raw_index)])
        if not all(
            _independent_seasonal_period(
                period, previous, shortest_context
            )
            for previous in selected
        ):
            continue
        selected.append(period)
        generated.append(period)
        if len(generated) == required_count:
            return generated, (minimum, maximum)
    raise ValueError("artificial_independent_period_pool_exhausted")


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


def _multi_seasonal_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = instance.history
    length, dimension = history.shape
    scales = _scale_by_target(history)
    model_max_contexts = source_distance_model_max_contexts(instance.term)
    shortest_context = min(length, min(model_max_contexts.values()))
    components_by_target: dict[int, list[dict[str, Any]]] = {}
    period_bounds_by_target: dict[str, list[float]] = {}
    anchor_search_by_target: dict[str, dict[str, Any]] = {}
    for channel in range(dimension):
        ranked = _dominant_frequency_indexes(history[:, channel])
        components: list[dict[str, Any]] = []
        anchor_period: float | None = None
        anchor_search: dict[str, Any] = {
            "maximum_candidate_count": (
                MULTI_SEASONAL_REAL_ANCHOR_CANDIDATE_COUNT
            ),
            "attempts": [],
            "accepted_rank": None,
            "accepted_frequency_index": None,
            "fallback_reason": None,
        }
        for rank, carrier_index in enumerate(
            ranked[:MULTI_SEASONAL_REAL_ANCHOR_CANDIDATE_COUNT], start=1
        ):
            candidate_period = float(length / carrier_index)
            maximum_period = min(
                shortest_context / MULTI_SEASONAL_MINIMUM_HISTORY_CYCLES,
                instance.prediction_length
                / MULTI_SEASONAL_MINIMUM_FUTURE_CYCLE_FRACTION,
            )
            rejection_reasons: list[str] = []
            if not (
                MULTI_SEASONAL_MINIMUM_PERIOD
                <= candidate_period
                <= maximum_period
            ):
                rejection_reasons.append("period_outside_supported_range")
                anchor_search["attempts"].append(
                    {
                        "rank": rank,
                        "frequency_index": int(carrier_index),
                        "period": candidate_period,
                        "rejection_reasons": rejection_reasons,
                    }
                )
                continue
            fitted_h, _fitted_f = _harmonic_component(
                history[:, channel], carrier_index, instance.prediction_length
            )
            normalized_std = float(np.std(fitted_h) / scales[channel])
            if normalized_std < MULTI_SEASONAL_COMPONENT_VISIBILITY:
                rejection_reasons.append("full_history_component_too_weak")
            stability = _harmonic_split_stability(
                history[:, channel], carrier_index, float(scales[channel])
            )
            if not stability["accepted"]:
                rejection_reasons.append("split_half_stability_failed")
            anchor_search["attempts"].append(
                {
                    "rank": rank,
                    "frequency_index": int(carrier_index),
                    "period": candidate_period,
                    "full_history_normalized_std": normalized_std,
                    "rejection_reasons": rejection_reasons,
                }
            )
            if rejection_reasons:
                continue
            raw_sin, raw_cos = _harmonic_coefficients(
                history[:, channel], carrier_index
            )
            signal, sin_coefficient, cos_coefficient = (
                _normalized_continuous_harmonic(
                    length=length,
                    horizon=instance.prediction_length,
                    period=candidate_period,
                    sin_coefficient=raw_sin,
                    cos_coefficient=raw_cos,
                    target_scale=float(scales[channel]),
                )
            )
            anchor_period = candidate_period
            anchor_search["accepted_rank"] = rank
            anchor_search["accepted_frequency_index"] = int(carrier_index)
            components.append(
                {
                    "role": "anchor",
                    "source": "history_top3_stable_harmonic",
                    "period": candidate_period,
                    "frequency_index": int(carrier_index),
                    "history_candidate_rank": rank,
                    "sin_coefficient": sin_coefficient,
                    "cos_coefficient": cos_coefficient,
                    "history_component": signal[:length],
                    "future_component": signal[length:],
                    "history_normalized_std_before_aggregate_gain": 1.0,
                    "history_fit_normalized_std": normalized_std,
                    "history_split_stability": stability,
                }
            )
            break

        if anchor_period is None:
            anchor_search["fallback_reason"] = (
                "no_ranked_history_frequency"
                if not ranked
                else "top3_history_anchor_candidates_rejected"
            )
        anchor_search_by_target[str(channel)] = anchor_search

        required_generated = MULTI_SEASONAL_MAXIMUM_ADDITIONAL_PERIODS
        if anchor_period is None:
            required_generated += 1
        generated_periods, bounds = _protocol_generated_periods(
            instance,
            channel,
            shortest_context=shortest_context,
            existing_periods=(
                () if anchor_period is None else (anchor_period,)
            ),
            required_count=required_generated,
            augmentation_seed=augmentation_seed,
        )
        period_bounds_by_target[str(channel)] = list(bounds)
        for generated_index, period in enumerate(generated_periods):
            role = (
                "anchor"
                if anchor_period is None and generated_index == 0
                else "additional"
            )
            phase = float(
                _rng(
                    instance.official_instance_id,
                    "multi_seasonal",
                    "phase",
                    channel,
                    generated_index,
                    augmentation_seed=augmentation_seed,
                ).uniform(0.0, 2.0 * math.pi)
            )
            raw_sin = math.sqrt(2.0) * scales[channel] * math.cos(phase)
            raw_cos = math.sqrt(2.0) * scales[channel] * math.sin(phase)
            signal, sin_coefficient, cos_coefficient = (
                _normalized_continuous_harmonic(
                    length=length,
                    horizon=instance.prediction_length,
                    period=period,
                    sin_coefficient=float(raw_sin),
                    cos_coefficient=float(raw_cos),
                    target_scale=float(scales[channel]),
                )
            )
            components.append(
                {
                    "role": role,
                    "source": "protocol_generated",
                    "period": period,
                    "frequency_index": None,
                    "phase": phase,
                    "sin_coefficient": sin_coefficient,
                    "cos_coefficient": cos_coefficient,
                    "history_component": signal[:length],
                    "future_component": signal[length:],
                    "history_normalized_std_before_aggregate_gain": 1.0,
                }
            )
            if role == "anchor":
                anchor_period = period
        if len(components) != MULTI_SEASONAL_MAXIMUM_ADDITIONAL_PERIODS + 1:
            raise ValueError("multi_seasonal_component_count_mismatch")
        components_by_target[channel] = components

    affected = tuple(range(dimension))

    shared_distance = float(
        _rng(
            instance.official_instance_id,
            "multi_seasonal",
            "shared_distance",
            augmentation_seed=augmentation_seed,
        ).uniform(*MULTI_SEASONAL_SHARED_DISTANCE_INTERVAL)
    )
    units: list[_UnitTreatment] = []
    for level in CAPABILITY_LEVELS:
        history_component = np.zeros_like(history)
        future_component = np.zeros_like(instance.future)
        resolved: dict[str, Any] = {}
        for channel in affected:
            selected = components_by_target[channel][: level + 1]
            for row in selected:
                history_component[:, channel] += row["history_component"]
                future_component[:, channel] += row["future_component"]
            resolved[str(channel)] = {
                "anchor_source": str(selected[0]["source"]),
                "history_anchor_search": anchor_search_by_target[str(channel)],
                "period_bounds": period_bounds_by_target[str(channel)],
                "components": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"history_component", "future_component"}
                    }
                    for row in selected
                ],
            }
        unit_distance = _full_history_unit_distance(
            history_component, history, affected
        )
        if unit_distance <= 1e-10:
            raise ValueError("multi_seasonal_aggregate_component_not_visible")
        gain = shared_distance / unit_distance
        units.append(
            _UnitTreatment(
                history_delta=history_component * gain,
                future_delta=future_component * gain,
                affected=affected,
                coordinate_name="additional_independent_period_count",
                coordinate_interval=(float(level), float(level)),
                sampled_coordinate=float(level),
                metadata={
                    "component": (
                        "hybrid_anchor_protocol_generated_nested_independent_"
                        "periods"
                    ),
                    "resolved_periods_by_target": resolved,
                    "additional_independent_period_count": int(level),
                    "total_controlled_period_count": int(level + 1),
                    "history_anchor_component_visibility_threshold": (
                        MULTI_SEASONAL_COMPONENT_VISIBILITY
                    ),
                    "shared_full_history_macro_normalized_rms": shared_distance,
                    "shared_distance_interval": list(
                        MULTI_SEASONAL_SHARED_DISTANCE_INTERVAL
                    ),
                    "amplitude_policy": (
                        "each_nested_level_normalized_to_one_shared_full_history_"
                        "macro_rms"
                    ),
                    "unit_component_distance": unit_distance,
                    "physical_component_gain": gain,
                },
            )
        )
    return units, {
        "eligible_target_indices": list(affected),
        "shared_full_history_macro_normalized_rms": shared_distance,
        "maximum_additional_periods": (
            MULTI_SEASONAL_MAXIMUM_ADDITIONAL_PERIODS
        ),
        "shortest_evaluated_model_context": shortest_context,
        "period_policy": (
            "stable_history_anchor_else_protocol_anchor_plus_protocol_generated_"
            "independent_additional_periods"
        ),
    }


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


def _nonlinear_state_response(values: np.ndarray | float) -> np.ndarray | float:
    """Return a signed bounded-quadratic persistence response.

    Dividing by ``1 + |z|`` makes the additional effective persistence tend to
    a finite coefficient while retaining a second-order response near zero.
    """

    array = np.asarray(values, dtype=float)
    response = array * np.abs(array) / (1.0 + np.abs(array))
    return float(response) if array.ndim == 0 else response


def _nonlinear_fit(values: np.ndarray, *, nonlinear: bool) -> np.ndarray:
    previous = np.asarray(values[:-1], dtype=float)
    response = np.asarray(values[1:], dtype=float)
    columns = [np.ones(previous.size), previous]
    if nonlinear:
        columns.append(np.asarray(_nonlinear_state_response(previous)))
    return np.linalg.lstsq(np.column_stack(columns), response, rcond=None)[0]


def _nonlinear_innovation_bootstrap_paths(
    innovations: np.ndarray,
    *,
    horizon: int,
    path_count: int,
    seed: int,
    innovation_pool: str = "linear_skeleton_full_history_residuals",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw centered circular moving-block paths from historical innovations."""

    pool = np.asarray(innovations, dtype=float).reshape(-1)
    if pool.size < 2:
        raise ValueError("nonlinear_innovation_pool_too_short")
    if horizon < 1 or path_count < 2:
        raise ValueError("nonlinear_innovation_bootstrap_shape_invalid")
    centered = pool - float(np.mean(pool))
    block_length = min(
        centered.size,
        max(
            NONLINEAR_FUTURE_INNOVATION_MINIMUM_BLOCK_LENGTH,
            int(math.ceil(math.sqrt(horizon))),
        ),
    )
    block_count = int(math.ceil(horizon / block_length))
    starts = np.random.default_rng(int(seed)).integers(
        0,
        centered.size,
        size=(int(path_count), block_count),
    )
    offsets = np.arange(block_length, dtype=int)
    indexes = (starts[:, :, None] + offsets[None, None, :]) % centered.size
    paths = centered[indexes.reshape(int(path_count), -1)[:, :horizon]]
    # Finite bootstrap ensembles otherwise carry a small horizon-specific mean
    # shock.  Removing it keeps the estimand focused on persistence while
    # retaining each sampled path's local block structure.
    paths = paths - np.mean(paths, axis=0, keepdims=True)
    return paths, {
        "schema_version": "cafe.nonlinear_innovation_bootstrap.v1",
        "method": "centered_circular_moving_block_bootstrap",
        "innovation_pool": innovation_pool,
        "path_count": int(path_count),
        "block_length": int(block_length),
        "seed": int(seed),
        "aggregation": "paired_path_mean",
        "shared_across_linear_and_nonlinear_branches": True,
        "ensemble_centered_at_each_horizon_step": True,
        "target_future_values_used": False,
    }


def _nonlinear_multistep_holdout_audit(
    values: np.ndarray,
    *,
    split: int,
    prediction_horizon: int,
    linear_train: np.ndarray,
    nonlinear_train: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    holdout_size = values.size - 1 - split
    audit_horizon = min(
        max(1, int(prediction_horizon)),
        max(4, holdout_size // 2),
    )
    latest_origin = values.size - 1 - audit_horizon
    if latest_origin < split:
        return {
            "accepted": False,
            "reason": "insufficient_multistep_holdout_support",
        }
    origin_count = min(
        NONLINEAR_MULTISTEP_AUDIT_ORIGIN_COUNT,
        latest_origin - split + 1,
    )
    origins = np.unique(
        np.linspace(split, latest_origin, num=origin_count, dtype=int)
    )
    train_previous = values[:split]
    train_response = values[1 : split + 1]
    innovations = train_response - (
        float(linear_train[0]) + float(linear_train[1]) * train_previous
    )
    paths, bootstrap = _nonlinear_innovation_bootstrap_paths(
        innovations,
        horizon=audit_horizon,
        path_count=NONLINEAR_FUTURE_INNOVATION_PATH_COUNT,
        seed=seed,
        innovation_pool="linear_skeleton_training_prefix_residuals",
    )
    linear_forecasts: list[np.ndarray] = []
    nonlinear_forecasts: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    for origin in origins:
        linear_state = np.full(paths.shape[0], values[origin], dtype=float)
        nonlinear_state = linear_state.copy()
        linear_mean = np.empty(audit_horizon, dtype=float)
        nonlinear_mean = np.empty(audit_horizon, dtype=float)
        for step in range(audit_horizon):
            innovation = paths[:, step]
            linear_state = (
                float(linear_train[0])
                + float(linear_train[1]) * linear_state
                + innovation
            )
            nonlinear_state = (
                float(nonlinear_train[0])
                + float(nonlinear_train[1]) * nonlinear_state
                + float(nonlinear_train[2])
                * np.asarray(_nonlinear_state_response(nonlinear_state))
                + innovation
            )
            linear_mean[step] = float(np.mean(linear_state))
            nonlinear_mean[step] = float(np.mean(nonlinear_state))
        linear_forecasts.append(linear_mean)
        nonlinear_forecasts.append(nonlinear_mean)
        observations.append(values[origin + 1 : origin + 1 + audit_horizon])
    actual = np.concatenate(observations)
    linear = np.concatenate(linear_forecasts)
    nonlinear = np.concatenate(nonlinear_forecasts)
    linear_mse = float(np.mean(np.square(actual - linear)))
    nonlinear_mse = float(np.mean(np.square(actual - nonlinear)))
    gain = (
        0.0
        if linear_mse <= 1e-12
        else 1.0 - nonlinear_mse / linear_mse
    )
    accepted = gain > NONLINEAR_MINIMUM_MULTISTEP_HOLDOUT_R2_GAIN + 1e-12
    return {
        "accepted": accepted,
        "reason": None if accepted else "nonlinear_does_not_beat_linear_multistep",
        "origin_indices": [int(value) for value in origins],
        "origin_count": int(origins.size),
        "forecast_horizon": int(audit_horizon),
        "linear_mse": linear_mse,
        "nonlinear_mse": nonlinear_mse,
        "incremental_r2": gain,
        "minimum_required_incremental_r2": (
            NONLINEAR_MINIMUM_MULTISTEP_HOLDOUT_R2_GAIN
        ),
        "innovation_bootstrap": bootstrap,
    }


def _nonlinear_channel_audit(
    z: np.ndarray,
    *,
    prediction_horizon: int,
    multistep_seed: int,
) -> dict[str, Any]:
    """Audit one standardized channel using one blocked holdout split.

    The function deliberately performs a constant number of tiny regressions;
    it does not search lags, thresholds, or hyperparameters.
    """

    values = np.asarray(z, dtype=float)
    transition_count = values.size - 1
    holdout_size = max(
        NONLINEAR_MINIMUM_HOLDOUT_SIZE,
        int(round(NONLINEAR_HOLDOUT_FRACTION * transition_count)),
    )
    split = transition_count - holdout_size
    audit: dict[str, Any] = {
        "schema_version": "cafe.nonlinear_identifiability_channel.v2",
        "transition_count": transition_count,
        "training_transition_count": split,
        "holdout_transition_count": holdout_size,
        "holdout_policy": "single_blocked_suffix_without_parameter_search",
        "accepted": False,
        "reason": None,
    }
    if split < 32 or holdout_size < NONLINEAR_MINIMUM_HOLDOUT_SIZE:
        audit["reason"] = "insufficient_train_holdout_transitions"
        return audit

    previous = values[:-1]
    response = values[1:]
    train_previous = previous[:split]
    train_response = response[:split]
    holdout_previous = previous[split:]
    holdout_response = response[split:]
    support = {
        "train_ordinary_count": int(
            np.count_nonzero(
                np.abs(train_previous) <= NONLINEAR_ORDINARY_STATE_MAXIMUM_ABS
            )
        ),
        "train_extreme_count": int(
            np.count_nonzero(
                np.abs(train_previous) >= NONLINEAR_EXTREME_STATE_MINIMUM_ABS
            )
        ),
        "holdout_ordinary_count": int(
            np.count_nonzero(
                np.abs(holdout_previous) <= NONLINEAR_ORDINARY_STATE_MAXIMUM_ABS
            )
        ),
        "holdout_extreme_count": int(
            np.count_nonzero(
                np.abs(holdout_previous) >= NONLINEAR_EXTREME_STATE_MINIMUM_ABS
            )
        ),
    }
    audit["state_support"] = support
    if (
        support["train_ordinary_count"] < NONLINEAR_MINIMUM_TRAIN_ORDINARY_COUNT
        or support["train_extreme_count"] < NONLINEAR_MINIMUM_TRAIN_EXTREME_COUNT
        or support["holdout_ordinary_count"]
        < NONLINEAR_MINIMUM_HOLDOUT_ORDINARY_COUNT
        or support["holdout_extreme_count"]
        < NONLINEAR_MINIMUM_HOLDOUT_EXTREME_COUNT
    ):
        audit["reason"] = "ordinary_and_extreme_state_support_insufficient"
        return audit

    train_values = values[: split + 1]
    linear_train = _nonlinear_fit(train_values, nonlinear=False)
    nonlinear_train = _nonlinear_fit(train_values, nonlinear=True)
    linear_holdout = (
        linear_train[0] + linear_train[1] * holdout_previous
    )
    nonlinear_holdout = (
        nonlinear_train[0]
        + nonlinear_train[1] * holdout_previous
        + nonlinear_train[2]
        * np.asarray(_nonlinear_state_response(holdout_previous))
    )
    linear_mse = float(np.mean(np.square(holdout_response - linear_holdout)))
    nonlinear_mse = float(
        np.mean(np.square(holdout_response - nonlinear_holdout))
    )
    holdout_gain = (
        0.0
        if linear_mse <= 1e-12
        else 1.0 - nonlinear_mse / linear_mse
    )
    midpoint = values.size // 2
    first_coefficients = _nonlinear_fit(values[: midpoint + 1], nonlinear=True)
    second_coefficients = _nonlinear_fit(values[midpoint:], nonlinear=True)
    full_nonlinear = _nonlinear_fit(values, nonlinear=True)
    full_linear = _nonlinear_fit(values, nonlinear=False)
    multistep = _nonlinear_multistep_holdout_audit(
        values,
        split=split,
        prediction_horizon=prediction_horizon,
        linear_train=linear_train,
        nonlinear_train=nonlinear_train,
        seed=multistep_seed,
    )
    coefficient_values = np.asarray(
        [
            nonlinear_train[2],
            first_coefficients[2],
            second_coefficients[2],
            full_nonlinear[2],
        ],
        dtype=float,
    )
    nonzero = np.abs(coefficient_values) >= NONLINEAR_MINIMUM_COEFFICIENT_ABS
    sign_stable = bool(
        np.all(nonzero)
        and np.all(np.sign(coefficient_values) == np.sign(coefficient_values[-1]))
    )
    half_ratio = float(
        min(abs(first_coefficients[2]), abs(second_coefficients[2]))
        / max(abs(first_coefficients[2]), abs(second_coefficients[2]), 1e-12)
    )
    linear_persistence = float(full_linear[1])
    direction = float(np.sign(full_nonlinear[2]))
    stability_headroom = float(
        NONLINEAR_STABILITY_LIMIT - direction * linear_persistence
    )
    audit.update(
        {
            "linear_holdout_mse": linear_mse,
            "nonlinear_holdout_mse": nonlinear_mse,
            "holdout_incremental_r2": holdout_gain,
            "minimum_required_holdout_incremental_r2": (
                NONLINEAR_MINIMUM_HOLDOUT_R2_GAIN
            ),
            "training_nonlinear_coefficient": float(nonlinear_train[2]),
            "first_half_nonlinear_coefficient": float(first_coefficients[2]),
            "second_half_nonlinear_coefficient": float(second_coefficients[2]),
            "full_history_nonlinear_coefficient": float(full_nonlinear[2]),
            "minimum_required_coefficient_abs": (
                NONLINEAR_MINIMUM_COEFFICIENT_ABS
            ),
            "coefficient_sign_stable": sign_stable,
            "half_coefficient_magnitude_ratio": half_ratio,
            "minimum_required_half_coefficient_ratio": (
                NONLINEAR_MINIMUM_HALF_COEFFICIENT_RATIO
            ),
            "linear_intercept": float(full_linear[0]),
            "linear_persistence_coefficient": linear_persistence,
            "nonlinear_direction": direction,
            "stability_limit": NONLINEAR_STABILITY_LIMIT,
            "stability_headroom": stability_headroom,
            "multistep_holdout": multistep,
        }
    )
    if holdout_gain < NONLINEAR_MINIMUM_HOLDOUT_R2_GAIN:
        audit["reason"] = "nonlinear_structure_does_not_beat_linear_ar_on_holdout"
    elif not sign_stable:
        audit["reason"] = "nonlinear_coefficient_direction_not_stable"
    elif half_ratio < NONLINEAR_MINIMUM_HALF_COEFFICIENT_RATIO:
        audit["reason"] = "nonlinear_coefficient_magnitude_not_stable"
    elif abs(linear_persistence) >= NONLINEAR_STABILITY_LIMIT:
        audit["reason"] = "linear_skeleton_not_stable"
    elif stability_headroom <= 0.0:
        audit["reason"] = "no_stability_headroom_in_detected_direction"
    elif multistep.get("accepted") is not True:
        audit["reason"] = "nonlinear_structure_does_not_beat_linear_multistep"
    else:
        audit["accepted"] = True
    return audit


def _state_dependent_persistence_delta_batch(
    z: np.ndarray,
    *,
    scale: float,
    horizon: int,
    linear_intercept: float,
    linear_persistence: float,
    nonlinear_coefficients: np.ndarray,
    future_innovation_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Apply all doses and marginalize paired future innovation paths."""

    source = np.asarray(z, dtype=float)
    coefficients = np.asarray(nonlinear_coefficients, dtype=float).reshape(-1)
    treated = np.empty((coefficients.size, source.size), dtype=float)
    treated[:, 0] = source[0]
    innovations = source[1:] - (
        linear_intercept + linear_persistence * source[:-1]
    )
    for index in range(1, source.size):
        previous = treated[:, index - 1]
        treated[:, index] = (
            linear_intercept
            + linear_persistence * previous
            + coefficients * np.asarray(_nonlinear_state_response(previous))
            + innovations[index - 1]
        )
    if np.max(np.abs(treated)) > NONLINEAR_STATE_ABSOLUTE_LIMIT:
        raise ValueError("nonlinear_treated_history_exceeds_state_limit")

    history_delta = float(scale) * (treated - source[None, :])
    future_innovations = np.asarray(future_innovation_paths, dtype=float)
    if future_innovations.ndim != 2 or future_innovations.shape[1] != horizon:
        raise ValueError("nonlinear_future_innovation_paths_shape_invalid")
    path_count = future_innovations.shape[0]
    future_delta = np.empty((coefficients.size, horizon), dtype=float)
    linear_state = np.full(path_count, source[-1], dtype=float)
    nonlinear_state = np.broadcast_to(
        treated[:, -1, None], (coefficients.size, path_count)
    ).copy()
    maximum_linear_future_state = float(np.max(np.abs(linear_state)))
    maximum_nonlinear_future_state = np.max(
        np.abs(nonlinear_state), axis=1
    )
    half_path_count = path_count // 2
    half_ensemble_max_abs_difference = np.zeros(coefficients.size, dtype=float)
    for step in range(horizon):
        innovation = future_innovations[:, step]
        linear_state = (
            linear_intercept + linear_persistence * linear_state + innovation
        )
        nonlinear_state = (
            linear_intercept
            + linear_persistence * nonlinear_state
            + coefficients[:, None]
            * np.asarray(_nonlinear_state_response(nonlinear_state))
            + innovation[None, :]
        )
        if (
            np.max(np.abs(linear_state)) > NONLINEAR_STATE_ABSOLUTE_LIMIT
            or np.max(np.abs(nonlinear_state)) > NONLINEAR_STATE_ABSOLUTE_LIMIT
        ):
            raise ValueError("nonlinear_future_rollout_exceeds_state_limit")
        maximum_linear_future_state = max(
            maximum_linear_future_state, float(np.max(np.abs(linear_state)))
        )
        maximum_nonlinear_future_state = np.maximum(
            maximum_nonlinear_future_state,
            np.max(np.abs(nonlinear_state), axis=1),
        )
        paired_delta = nonlinear_state - linear_state[None, :]
        future_delta[:, step] = float(scale) * np.mean(paired_delta, axis=1)
        if half_path_count:
            first_half = np.mean(paired_delta[:, :half_path_count], axis=1)
            second_half = np.mean(paired_delta[:, half_path_count:], axis=1)
            half_ensemble_max_abs_difference = np.maximum(
                half_ensemble_max_abs_difference,
                float(scale) * np.abs(first_half - second_half),
            )
    maximum_history = np.max(np.abs(treated), axis=1)
    diagnostics = [
        {
            "maximum_treated_history_state_abs": float(maximum_history[index]),
            "maximum_linear_future_state_abs": float(
                maximum_linear_future_state
            ),
            "maximum_nonlinear_future_state_abs": float(
                maximum_nonlinear_future_state[index]
            ),
            "future_innovation_path_count": int(path_count),
            "future_effect_half_ensemble_max_abs_difference": float(
                half_ensemble_max_abs_difference[index]
            ),
        }
        for index in range(coefficients.size)
    ]
    return history_delta, future_delta, diagnostics


def _state_dependent_persistence_deltas(
    z: np.ndarray,
    *,
    scale: float,
    horizon: int,
    linear_intercept: float,
    linear_persistence: float,
    nonlinear_coefficient: float,
    future_innovation_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Scalar compatibility wrapper around the batched recurrence."""

    history, future, diagnostics = _state_dependent_persistence_delta_batch(
        z,
        scale=scale,
        horizon=horizon,
        linear_intercept=linear_intercept,
        linear_persistence=linear_persistence,
        nonlinear_coefficients=np.asarray([nonlinear_coefficient], dtype=float),
        future_innovation_paths=future_innovation_paths,
    )
    return history[0], future[0], diagnostics[0]


def _nonlinear_units(
    instance: GiftEvalInstance,
    augmentation_seed: int,
) -> tuple[list[_UnitTreatment], dict[str, Any]]:
    history = np.asarray(instance.history, dtype=float)
    length, dimension = history.shape
    if length < NONLINEAR_MINIMUM_HISTORY:
        raise ValueError("history_too_short_for_state_dependent_persistence")
    scales = _scale_by_target(history)
    # Use the same per-channel reduction order as compact replay so the dense
    # generation hashes remain bit-for-bit reproducible.
    standardized = np.empty_like(history)
    for channel in range(dimension):
        standardized[:, channel] = (
            history[:, channel] - np.mean(history[:, channel])
        ) / scales[channel]
    diagnostics: dict[str, Any] = {}
    affected: list[int] = []
    for channel in range(dimension):
        multistep_seed = protocol.stable_seed(
            instance.official_instance_id,
            "nonlinear_persistence",
            "multistep_holdout",
            channel,
            base=int(augmentation_seed),
        )
        audit = _nonlinear_channel_audit(
            standardized[:, channel],
            prediction_horizon=instance.prediction_length,
            multistep_seed=multistep_seed,
        )
        diagnostics[str(channel)] = audit
        if audit["accepted"]:
            affected.append(channel)
    if not affected:
        raise ValueError("state_dependent_persistence_not_identifiable_on_holdout")

    headroom_fractions = [
        float(
            _rng(
                instance.official_instance_id,
                "nonlinear_persistence",
                level,
                augmentation_seed=augmentation_seed,
            ).uniform(*interval)
        )
        for level, interval in zip(
            CAPABILITY_LEVELS, NONLINEAR_PERSISTENCE_INTERVALS, strict=True
        )
    ]
    history_deltas = [np.zeros_like(history) for _ in CAPABILITY_LEVELS]
    future_deltas = [
        np.zeros_like(instance.future) for _ in CAPABILITY_LEVELS
    ]
    level_diagnostics: list[dict[str, Any]] = [
        {} for _ in CAPABILITY_LEVELS
    ]
    future_bootstrap_by_target: dict[str, Any] = {}
    for channel in affected:
        audit = diagnostics[str(channel)]
        direction = float(audit["nonlinear_direction"])
        coefficients = np.asarray(
            [
                direction
                * headroom_fraction
                * float(audit["stability_headroom"])
                for headroom_fraction in headroom_fractions
            ],
            dtype=float,
        )
        effective_extreme_persistence = (
            float(audit["linear_persistence_coefficient"]) + coefficients
        )
        if np.any(
            np.abs(effective_extreme_persistence)
            >= NONLINEAR_STABILITY_LIMIT
        ):
            raise ValueError("nonlinear_level_exceeds_stability_limit")
        source = standardized[:, channel]
        full_innovations = source[1:] - (
            float(audit["linear_intercept"])
            + float(audit["linear_persistence_coefficient"]) * source[:-1]
        )
        bootstrap_seed = protocol.stable_seed(
            instance.official_instance_id,
            "nonlinear_persistence",
            "future_innovation_bootstrap",
            channel,
            base=int(augmentation_seed),
        )
        future_innovation_paths, bootstrap = (
            _nonlinear_innovation_bootstrap_paths(
                full_innovations,
                horizon=instance.prediction_length,
                path_count=NONLINEAR_FUTURE_INNOVATION_PATH_COUNT,
                seed=bootstrap_seed,
            )
        )
        future_bootstrap_by_target[str(channel)] = bootstrap
        channel_history, channel_future, rollouts = (
            _state_dependent_persistence_delta_batch(
                source,
                scale=float(scales[channel]),
                horizon=instance.prediction_length,
                linear_intercept=float(audit["linear_intercept"]),
                linear_persistence=float(
                    audit["linear_persistence_coefficient"]
                ),
                nonlinear_coefficients=coefficients,
                future_innovation_paths=future_innovation_paths,
            )
        )
        for index, _level in enumerate(CAPABILITY_LEVELS):
            history_deltas[index][:, channel] = channel_history[index]
            future_deltas[index][:, channel] = channel_future[index]
            level_diagnostics[index][str(channel)] = {
                "nonlinear_persistence_coefficient": float(
                    coefficients[index]
                ),
                "effective_extreme_persistence_limit": float(
                    effective_extreme_persistence[index]
                ),
                **rollouts[index],
            }

    units: list[_UnitTreatment] = []
    full_distances: list[float] = []
    for index, (level, interval, headroom_fraction) in enumerate(
        zip(
            CAPABILITY_LEVELS,
            NONLINEAR_PERSISTENCE_INTERVALS,
            headroom_fractions,
            strict=True,
        )
    ):
        history_delta = history_deltas[index]
        future_delta = future_deltas[index]
        for channel in affected:
            effective = level_diagnostics[index][str(channel)][
                "effective_extreme_persistence_limit"
            ]
            if abs(effective) >= NONLINEAR_STABILITY_LIMIT:
                raise ValueError("nonlinear_level_exceeds_stability_limit")
        distance = _full_history_unit_distance(
            history_delta, history, tuple(affected)
        )
        full_distances.append(distance)
        units.append(
            _UnitTreatment(
                history_delta=history_delta,
                future_delta=future_delta,
                affected=tuple(affected),
                coordinate_name="stable_persistence_headroom_fraction",
                coordinate_interval=interval,
                sampled_coordinate=headroom_fraction,
                metadata={
                    "component": (
                        "same_innovation_state_dependent_persistence_recurrence"
                    ),
                    "state_response": "z_abs_z_over_one_plus_abs_z",
                    "future_innovation_policy": (
                        "history_innovation_marginalized_shared_path_mean"
                    ),
                    "future_innovation_bootstrap_by_target": (
                        future_bootstrap_by_target
                    ),
                    "headroom_fraction": headroom_fraction,
                    "physical_component_gain": 1.0,
                    "identifiability_by_target": diagnostics,
                    "level_diagnostics_by_target": level_diagnostics[index],
                    "full_history_macro_normalized_rms": distance,
                    "target_future_used_for_delta": False,
                },
            )
        )
    if any(
        current <= previous + 1e-12
        for previous, current in zip(full_distances, full_distances[1:])
    ):
        raise ValueError("nonlinear_treatment_distance_not_strictly_monotone")
    return units, {
        "nonlinear_identifiability_gate": {
            "schema_version": "cafe.nonlinear_identifiability_gate.v2",
            "method": (
                "blocked_suffix_one_step_and_rolling_multistep_"
                "nonlinear_vs_linear_ar1"
            ),
            "state_response": "z_abs_z_over_one_plus_abs_z",
            "minimum_history_length": NONLINEAR_MINIMUM_HISTORY,
            "affected_target_indices": affected,
            "diagnostics_by_target": diagnostics,
            "target_future_values_used": False,
            "accepted": True,
            "reason": None,
        },
        "dose_policy": (
            "ordered_fraction_of_detected_direction_stability_headroom"
        ),
        "future_estimand": (
            "paired_conditional_mean_over_shared_bootstrapped_innovations"
        ),
        "full_history_distance_by_level": full_distances,
    }


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
    "multi_seasonal": _multi_seasonal_units,
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
    if capability_id == "nonlinear_persistence":
        scales = _scale_by_target(instance.history)[list(unit.affected)]
        standardized = unit.future_delta[:, unit.affected] / scales[None, :]
        profile = np.zeros(instance.prediction_length, dtype=float)
        for step in range(instance.prediction_length):
            observed = instance.future_observed_mask[step, list(unit.affected)]
            values = standardized[step, observed]
            profile[step] = (
                float(np.sqrt(np.mean(np.square(values))))
                if values.size
                else float("nan")
            )
        valid_indexes = np.flatnonzero(np.isfinite(profile))
        if valid_indexes.size:
            valid_profile = profile[valid_indexes]
            peak_position = int(np.argmax(valid_profile))
            peak_index = int(valid_indexes[peak_position])
            peak = float(valid_profile[peak_position])
            minimum = float(np.min(valid_profile))
            relative_range = (peak - minimum) / max(peak, 1e-12)
            tail_count = max(1, valid_profile.size // 4)
            tail_rms = float(
                np.sqrt(np.mean(np.square(valid_profile[-tail_count:])))
            )
            tail_peak_ratio = tail_rms / max(peak, 1e-12)
            half_threshold = 0.5 * peak
            crossings = valid_indexes[
                (valid_indexes >= peak_index) & (profile[valid_indexes] <= half_threshold)
            ]
            truth_half_life = (
                int(crossings[0] - peak_index) if crossings.size else None
            )
        else:
            peak_index = -1
            peak = 0.0
            relative_range = 0.0
            tail_rms = 0.0
            tail_peak_ratio = None
            truth_half_life = None
        latest_peak = int(
            math.floor(
                NONLINEAR_MAXIMUM_FUTURE_PEAK_FRACTION
                * max(0, instance.prediction_length - 1)
            )
        )
        accepted = (
            peak > 0.0
            and peak_index <= latest_peak
            and relative_range >= NONLINEAR_MINIMUM_FUTURE_PROFILE_RANGE - 1e-12
            and tail_peak_ratio is not None
            and tail_peak_ratio <= NONLINEAR_MAXIMUM_TAIL_TO_PEAK_RATIO + 1e-12
        )
        reason = None
        if peak <= 0.0:
            reason = "nonlinear_future_effect_profile_empty"
        elif peak_index > latest_peak:
            reason = "nonlinear_future_effect_peaks_too_late"
        elif relative_range < NONLINEAR_MINIMUM_FUTURE_PROFILE_RANGE - 1e-12:
            reason = "nonlinear_future_effect_profile_too_flat"
        elif (
            tail_peak_ratio is not None
            and tail_peak_ratio > NONLINEAR_MAXIMUM_TAIL_TO_PEAK_RATIO + 1e-12
        ):
            reason = "nonlinear_future_effect_does_not_decay"
        return {
            "schema_version": "cafe.capability_horizon_support_gate.v1",
            "capability_id": capability_id,
            "metric": "innovation_marginalized_future_effect_decay_profile",
            "horizon_partition": "whole_forecast_horizon",
            "minimum_required_relative_range": (
                NONLINEAR_MINIMUM_FUTURE_PROFILE_RANGE
            ),
            "maximum_allowed_peak_fraction": (
                NONLINEAR_MAXIMUM_FUTURE_PEAK_FRACTION
            ),
            "maximum_allowed_tail_peak_ratio": (
                NONLINEAR_MAXIMUM_TAIL_TO_PEAK_RATIO
            ),
            "observed_relative_range": relative_range,
            "observed_peak_index": peak_index,
            "observed_peak_history_scale": peak,
            "observed_tail_rms_history_scale": tail_rms,
            "observed_tail_peak_ratio": tail_peak_ratio,
            "observed_profile_count": int(valid_indexes.size),
            "truth_effect_half_life_from_peak": truth_half_life,
            "truth_effect_half_life_censored": truth_half_life is None,
            "target_future_values_used": False,
            "accepted": accepted,
            "reason": reason,
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
        model_max_contexts = source_distance_model_max_contexts(instance.term)
        gate = _distance_gate(
            unit.history_delta,
            instance.history,
            unit.affected,
            model_max_contexts=model_max_contexts,
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
        if capability_id in STRICT_FUTURE_EFFECT_CAPABILITIES:
            _raw, signal, observed_count = mechanism_effect_signal(
                unit.future_delta,
                instance.future_observed_mask,
                mase_scale_by_target(instance.history, instance.frequency),
                unit.affected,
            )
            if (
                observed_count <= 0
                or signal < MECHANISM_EFFECT_MINIMUM_MASE_RMS - 1e-12
            ):
                return CapabilityGroup(
                    capability_id=capability_id,
                    available=False,
                    reason=f"level_{level}_future_effect_not_scoreable",
                    treatments=(),
                    group_metadata={
                        **group_metadata,
                        "failed_level": level,
                        "failed_future_effect_mase_rms": signal,
                        "minimum_required_future_effect_mase_rms": (
                            MECHANISM_EFFECT_MINIMUM_MASE_RMS
                        ),
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
    if capability_id == "nonlinear_persistence":
        history_components = [np.zeros_like(history) for _ in contracts]
        future_components = [
            np.zeros_like(instance.future) for _ in contracts
        ]
        affected = [
            int(value) for value in contracts[0]["affected_target_indices"]
        ]
        for channel in affected:
            first_metadata = contracts[0]["mechanism_metadata"]
            audit = first_metadata["identifiability_by_target"][str(channel)]
            bootstrap = first_metadata[
                "future_innovation_bootstrap_by_target"
            ][str(channel)]
            coefficients = np.asarray(
                [
                    row["mechanism_metadata"][
                        "level_diagnostics_by_target"
                    ][str(channel)]["nonlinear_persistence_coefficient"]
                    for row in contracts
                ],
                dtype=float,
            )
            z = (
                history[:, channel] - np.mean(history[:, channel])
            ) / scales[channel]
            innovations = z[1:] - (
                float(audit["linear_intercept"])
                + float(audit["linear_persistence_coefficient"]) * z[:-1]
            )
            future_innovation_paths, _bootstrap_replay = (
                _nonlinear_innovation_bootstrap_paths(
                    innovations,
                    horizon=horizon,
                    path_count=int(bootstrap["path_count"]),
                    seed=int(bootstrap["seed"]),
                )
            )
            channel_history, channel_future, _rollout = (
                _state_dependent_persistence_delta_batch(
                    z,
                    scale=float(scales[channel]),
                    horizon=horizon,
                    linear_intercept=float(audit["linear_intercept"]),
                    linear_persistence=float(
                        audit["linear_persistence_coefficient"]
                    ),
                    nonlinear_coefficients=coefficients,
                    future_innovation_paths=future_innovation_paths,
                )
            )
            for index in range(len(contracts)):
                history_components[index][:, channel] = channel_history[index]
                future_components[index][:, channel] = channel_future[index]
        for index, row in enumerate(contracts):
            gain = float(row["applied_component_gain"])
            output[str(row["sample_id"])] = (
                np.asarray(history_components[index] * gain, dtype=float),
                np.asarray(future_components[index] * gain, dtype=float),
                np.zeros_like(instance.history_covariates),
                np.zeros_like(instance.future_covariates),
            )
        return output
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
        elif capability_id == "multi_seasonal":
            component_h = np.zeros_like(history)
            component_f = np.zeros_like(instance.future)
            covariate_h = np.zeros_like(instance.history_covariates)
            covariate_f = np.zeros_like(instance.future_covariates)
            details = metadata["resolved_periods_by_target"]
            for raw_channel, periods in details.items():
                channel = int(raw_channel)
                for component in periods["components"]:
                    signal = _continuous_harmonic_signal(
                        length=length,
                        horizon=horizon,
                        period=float(component["period"]),
                        sin_coefficient=float(component["sin_coefficient"]),
                        cos_coefficient=float(component["cos_coefficient"]),
                    )
                    component_h[:, channel] += signal[:length]
                    component_f[:, channel] += signal[length:]
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


def replay_treatment_deltas_for_history_suffix(
    instance: GiftEvalInstance,
    contracts: list[dict[str, Any]],
    *,
    history_start: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Replay a frozen treatment while materializing only a history suffix.

    The frozen mechanism is still defined on the complete official history.
    ``history_start`` only changes which already-defined treatment values are
    materialized for a model request.  Pointwise/analytic mechanisms avoid
    allocating the discarded prefix; stateful and structural fallbacks replay
    the full contract and slice the result to preserve exact semantics.
    """

    if not contracts:
        return {}
    full_history = np.asarray(instance.history, dtype=float)
    full_length, dimension = full_history.shape
    start = int(history_start)
    if start < 0 or start > full_length:
        raise ValueError("history suffix start is outside the official history")
    if start == 0:
        return replay_treatment_deltas(instance, contracts)

    capability_ids = {str(row["capability_id"]) for row in contracts}
    if len(capability_ids) != 1:
        raise ValueError("treatment replay group mixes capabilities")
    capability_id = capability_ids.pop()
    optimized = {
        "trend",
        "time_varying_seasonality",
        "regime_switching",
        "predictable_intermittency",
        "multi_seasonal",
    }
    if capability_id not in optimized:
        return {
            sample_id: (history[start:], future, history_covariates[start:], future_covariates)
            for sample_id, (
                history,
                future,
                history_covariates,
                future_covariates,
            ) in replay_treatment_deltas(instance, contracts).items()
        }

    horizon = int(instance.prediction_length)
    visible_length = full_length - start
    scales = _scale_by_target(full_history)

    def empty_components() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((visible_length, dimension), dtype=float),
            np.zeros_like(instance.future, dtype=float),
            np.zeros((visible_length, instance.history_covariates.shape[1]), dtype=float),
            np.zeros_like(instance.future_covariates, dtype=float),
        )

    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in contracts:
        metadata = dict(row["mechanism_metadata"])
        component_h, component_f, covariate_h, covariate_f = empty_components()
        if capability_id == "trend":
            directions = np.asarray(metadata["direction_by_target"], dtype=float)
            affected = [int(value) for value in row["affected_target_indices"]]
            absolute_time = np.arange(start, full_length + horizon, dtype=float)
            absolute_time /= max(1, full_length - 1)
            component = np.zeros((visible_length + horizon, dimension), dtype=float)
            component[:, affected] = (
                absolute_time[:, None]
                * directions[affected][None, :]
                * scales[affected][None, :]
            )
            component_h = component[:visible_length]
            component_f = component[visible_length:]
        elif capability_id == "time_varying_seasonality":
            absolute_time = np.arange(start, full_length + horizon, dtype=float)
            for raw_channel, periods in metadata["resolved_periods_by_target"].items():
                channel = int(raw_channel)
                carrier_omega = (
                    2.0
                    * math.pi
                    * float(periods["carrier_frequency_index"])
                    / float(full_length)
                )
                envelope_omega = (
                    2.0
                    * math.pi
                    * float(periods["modulation_frequency_index"])
                    / float(full_length)
                )
                carrier = (
                    float(periods["carrier_sin_coefficient"])
                    * np.sin(carrier_omega * absolute_time)
                    + float(periods["carrier_cos_coefficient"])
                    * np.cos(carrier_omega * absolute_time)
                )
                envelope = (
                    float(periods["envelope_sin_coefficient"])
                    * np.sin(envelope_omega * absolute_time)
                    + float(periods["envelope_cos_coefficient"])
                    * np.cos(envelope_omega * absolute_time)
                )
                combined = carrier * envelope
                component_h[:, channel] = combined[:visible_length]
                component_f[:, channel] = combined[visible_length:]
        elif capability_id == "regime_switching":
            join = int(metadata["change_index"])
            amplitude = np.asarray(
                metadata["shared_amplitude_before_distance_adjustment"],
                dtype=float,
            )
            direction = np.asarray(metadata["direction_by_target"], dtype=float)
            visible_indexes = np.arange(start, full_length)
            component_h[visible_indexes >= join] = direction * amplitude
            component_f[:] = direction * amplitude
        elif capability_id == "predictable_intermittency":
            combined = np.zeros((visible_length + horizon, dimension), dtype=float)
            amplitude = np.asarray(
                metadata["positive_event_amplitude_before_distance_adjustment"],
                dtype=float,
            )
            width = int(metadata["pulse_width"])
            for center in metadata["event_centers"]:
                for offset in range(width):
                    relative_index = int(center) + offset - start
                    if 0 <= relative_index < combined.shape[0]:
                        combined[relative_index] += amplitude
            component_h = combined[:visible_length]
            component_f = combined[visible_length:]
        elif capability_id == "multi_seasonal":
            absolute_time = np.arange(start, full_length + horizon, dtype=float)
            for raw_channel, periods in metadata["resolved_periods_by_target"].items():
                channel = int(raw_channel)
                for component in periods["components"]:
                    omega = 2.0 * math.pi / float(component["period"])
                    signal = (
                        float(component["sin_coefficient"])
                        * np.sin(omega * absolute_time)
                        + float(component["cos_coefficient"])
                        * np.cos(omega * absolute_time)
                    )
                    component_h[:, channel] += signal[:visible_length]
                    component_f[:, channel] += signal[visible_length:]
        gain = float(row["applied_component_gain"])
        output[str(row["sample_id"])] = (
            np.asarray(component_h * gain, dtype=float),
            np.asarray(component_f * gain, dtype=float),
            np.asarray(covariate_h * gain, dtype=float),
            np.asarray(covariate_f * gain, dtype=float),
        )
    return output
