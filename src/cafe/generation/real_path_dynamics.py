"""History-only real-path contracts for event clocks and nonlinear dynamics.

The four decomposition capabilities in :mod:`cafe.generation.anchored` have a
fixed additive component.  Predictable intermittency is also additive, but its
component is a sparse clock learned from real peaks.  Nonlinear persistence is
different: changing its gain changes downstream states recursively.  Keeping
the two mechanisms here prevents the additive decomposition contract from
silently pretending that a dynamic intervention is linear in ``alpha``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from cafe.generation.real_anchored_dose import (
    additive_dose_reference,
    dose_calibration_from_policy,
    nonlinear_dose_reference,
    paired_minimum_separation_gate,
    resolve_contract_dose_calibration,
)
from cafe.generation.real_anchored_policy import (
    NONLINEAR_FUTURE_INNOVATION_MAIN_POLICY,
    NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY,
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
    REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
)


REAL_PATH_DYNAMIC_CONTRACT_SCHEMA = "cafe.real_path_dynamic_contract.v2"
LEGACY_REAL_PATH_DYNAMIC_CONTRACT_SCHEMA = (
    "cafe.real_path_dynamic_contract.v1"
)
REAL_PATH_DYNAMIC_CAPABILITIES = frozenset(
    {"nonlinear_persistence", "predictable_intermittency"}
)
NONLINEAR_ALPHA_GRID = (1.2, 1.4, 1.6, 1.8, 2.0)
NONLINEAR_REFERENCE_ALPHA_GRID = tuple(
    round(1.0 + 0.005 * index, 3) for index in range(401)
)
_CONTEXT_LENGTH = 504
_VISIBLE_CONTEXT_LENGTH = 336
_FIXED_CONTEXT_LENGTH = 168
_HORIZON = 48
_VISIBLE_START = _CONTEXT_LENGTH - _VISIBLE_CONTEXT_LENGTH
_FIXED_START = _CONTEXT_LENGTH - _FIXED_CONTEXT_LENGTH
_MINIMUM_COMPONENT_RMS_RATIO = 0.01


def _ridge_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    *,
    penalty: float = 1e-4,
) -> np.ndarray:
    matrix = np.asarray(design, dtype=float)
    response = np.asarray(target, dtype=float)
    regularizer = np.eye(matrix.shape[1], dtype=float) * float(penalty)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(
        matrix.T @ matrix + regularizer,
        matrix.T @ response,
    )


def _strict_alpha_grid(
    values: Sequence[float],
    *,
    maximum: float,
    label: str,
) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if (
        not grid
        or any(
            not math.isfinite(value) or not 1.0 < value <= float(maximum)
            for value in grid
        )
        or any(
            right <= left
            for left, right in zip(grid, grid[1:], strict=False)
        )
    ):
        raise ValueError(
            f"{label} must be a finite, strictly increasing alpha grid "
            f"inside (1, {float(maximum):g}]"
        )
    return grid

DEFAULT_DYNAMIC_QUALIFICATION_THRESHOLDS: dict[str, dict[str, float | int]] = {
    "predictable_intermittency": {
        "minimum_peak_z": 1.0,
        "minimum_holdout_clock_r2": 0.10,
        "minimum_timing_f1": 0.60,
        "maximum_timing_error_widths": 1.0,
        "minimum_positive_pulse_fraction": 0.80,
        "minimum_amplitude_to_off_event_scale": 2.0,
        "maximum_event_duty_cycle": 0.25,
        "minimum_training_event_count": 6,
        "minimum_future_event_count": 1,
        "minimum_component_rms_ratio": 0.01,
    },
    "nonlinear_persistence": {
        "minimum_median_blocked_holdout_gain": 0.01,
        "minimum_positive_fold_fraction": 2.0 / 3.0,
        # Recursive stability is still strict (<1).  A 0.995 margin avoids
        # discarding slowly mean-reverting real states merely because the
        # older 0.98 convenience bound was overly conservative.
        "maximum_linear_spectral_radius": 0.995,
        "minimum_component_rms_ratio": 0.01,
        "maximum_latent_support_multiplier": 4.0,
        "minimum_absolute_safe_bound": 8.0,
    },
}


def default_dynamic_qualification_policy() -> dict[str, Any]:
    """Return JSON-safe thresholds used until a reference-bank fit replaces them."""

    thresholds = {
        capability_id: dict(values)
        for capability_id, values in DEFAULT_DYNAMIC_QUALIFICATION_THRESHOLDS.items()
    }
    payload: dict[str, Any] = {
        "schema_version": REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
        "qualification_policy_id": "cafe.real_anchored.dynamic.reference.v1",
        "threshold_source": QUALIFICATION_THRESHOLD_SOURCE_POLICY,
        "qualification_thresholds": thresholds,
        "threshold_derivation": (
            "frozen_protocol_defaults_pending_or_replaced_by_disjoint_reference_bank"
        ),
    }
    payload["qualification_policy_sha256"] = _payload_sha256(payload)
    return payload


def _capability_qualification(
    capability_id: str,
    qualification_policy: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    policy = (
        default_dynamic_qualification_policy()
        if qualification_policy is None
        else dict(qualification_policy)
    )
    if policy.get("schema_version") != REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA:
        raise ValueError("unsupported real-anchored qualification policy schema")
    threshold_source = policy.get(
        "threshold_source_policy",
        policy.get("threshold_source"),
    )
    if threshold_source != QUALIFICATION_THRESHOLD_SOURCE_POLICY:
        raise ValueError(
            "dynamic qualification thresholds must come from the independent "
            "source-time-disjoint reference bank"
        )
    frozen_capabilities = policy.get("capabilities")
    if isinstance(frozen_capabilities, Mapping):
        capability_policy = frozen_capabilities.get(capability_id)
        if not isinstance(capability_policy, Mapping):
            raise ValueError(
                f"frozen qualification policy has no {capability_id} cell"
            )
        policy_id = capability_policy.get("qualification_policy_id")
        raw = capability_policy.get("qualification_thresholds")
    else:
        capability_policy = policy
        policy_id = policy.get("qualification_policy_id")
        all_thresholds = policy.get("qualification_thresholds")
        if not isinstance(all_thresholds, Mapping):
            raise ValueError("qualification policy is missing threshold mappings")
        raw = all_thresholds.get(capability_id)
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("qualification policy requires a stable non-empty id")
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"qualification policy has no thresholds for {capability_id}"
        )
    defaults = DEFAULT_DYNAMIC_QUALIFICATION_THRESHOLDS[capability_id]
    thresholds = {
        name: raw.get(name, value)
        for name, value in defaults.items()
    }
    for name, value in thresholds.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"qualification threshold {name!r} must be finite")
    provenance = {
        "qualification_policy_id": policy_id,
        "qualification_policy_sha256": policy.get(
            "qualification_policy_sha256",
            _payload_sha256(policy),
        ),
        "qualification_threshold_source": str(threshold_source),
        "qualification_thresholds": dict(thresholds),
    }
    if isinstance(frozen_capabilities, Mapping):
        provenance["dose_calibration"] = dose_calibration_from_policy(
            policy,
            capability_id,
            require_available=False,
        )
    return provenance, thresholds


def dynamic_qualification_provenance(
    capability_id: str,
    qualification_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose the frozen row-level policy fields without fitting a contract."""

    provenance, _thresholds = _capability_qualification(
        capability_id,
        qualification_policy,
    )
    return provenance


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _float_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _reference_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(b"cafe.real_path.reference.float64.v1\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finalize_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["capability_contract_sha256"] = _payload_sha256(result)
    return result


def _unavailable(
    base: Mapping[str, Any],
    reason: str,
    detail: str,
    **metrics: Any,
) -> dict[str, Any]:
    return _finalize_contract(
        {
            **base,
            "available": False,
            "unavailable_reason": reason,
            "unavailable_detail": detail,
            **metrics,
            "model": None,
        }
    )


def _validate_history(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.shape != (_CONTEXT_LENGTH,):
        raise ValueError("real-path dynamic history must be one finite L504 path")
    if not np.isfinite(values).all():
        raise ValueError("real-path dynamic history must be finite")
    return values


def _robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    if mad > 1e-9:
        return mad
    standard_deviation = float(np.std(values))
    return standard_deviation if standard_deviation > 1e-9 else 1.0


def _nuisance_design(
    length: int,
    *,
    training_length: int,
    periods: Sequence[float],
) -> np.ndarray:
    time = np.arange(length, dtype=float)
    coordinate = (time - 0.5 * (training_length - 1)) / max(
        training_length - 1,
        1,
    )
    columns = [np.ones(length), coordinate, coordinate**2]
    for period in periods:
        if not math.isfinite(float(period)) or float(period) < 2.0:
            continue
        phase = 2.0 * np.pi * time / float(period)
        columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(columns)


def _robust_nuisance_residual(
    history: np.ndarray,
    *,
    periods: Sequence[float],
    training_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a Huber-weighted smooth nuisance on the pre-validation history."""

    design = _nuisance_design(
        history.size,
        training_length=training_length,
        periods=periods,
    )
    train_design = design[:training_length]
    train_target = history[:training_length]
    coefficients, *_ = np.linalg.lstsq(train_design, train_target, rcond=None)
    weights = np.ones(training_length, dtype=float)
    for _iteration in range(12):
        residual = train_target - train_design @ coefficients
        scale = _robust_scale(residual)
        standardized = np.abs(residual) / max(1.5 * scale, 1e-12)
        weights = np.ones(training_length, dtype=float)
        downweighted = standardized > 1.0
        weights[downweighted] = 1.0 / standardized[downweighted]
        weighted_design = train_design * np.sqrt(weights)[:, None]
        weighted_target = train_target * np.sqrt(weights)
        updated, *_ = np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )
        if np.allclose(updated, coefficients, rtol=1e-10, atol=1e-12):
            coefficients = updated
            break
        coefficients = updated
    fitted = design @ coefficients
    return history - fitted, {
        "law": "huber_quadratic_plus_history_resolved_harmonics_v1",
        "training_length": training_length,
        "periods": [float(value) for value in periods],
        "coefficients": coefficients.tolist(),
        "downweighted_fraction": float(np.mean(weights < 0.999)),
    }


def _reference_fields(
    reference_history: np.ndarray,
    *,
    mase_period: int,
    mase_scale: float,
    mase_effective_period: int,
    mase_scale_source: str,
) -> dict[str, Any]:
    reference = np.asarray(reference_history, dtype=float)
    if reference.shape != (_VISIBLE_CONTEXT_LENGTH,):
        raise ValueError("dynamic reference history must be the trailing L336")
    if not np.isfinite(reference).all():
        raise ValueError("dynamic reference history must be finite")
    scale = float(np.std(reference))
    if scale <= 1e-12:
        raise ValueError("dynamic reference history has no usable scale")
    if not 1 <= int(mase_period) < reference.size:
        raise ValueError("dynamic MASE period must lie inside L336")
    if not math.isfinite(float(mase_scale)) or float(mase_scale) <= 0.0:
        raise ValueError("dynamic MASE scale must be finite and positive")
    return {
        "reference_start": _VISIBLE_START,
        "reference_length": _VISIBLE_CONTEXT_LENGTH,
        "reference_history_sha256": _reference_sha256(reference[:, None]),
        "normalization_mean_by_target": [float(np.mean(reference))],
        "normalization_scale_by_target": [scale],
        "normalization_policy": "baseline_history_shared_by_pair_v1",
        "mase_period": int(mase_period),
        "mase_scale_by_target": [float(mase_scale)],
        "mase_effective_period_by_target": [int(mase_effective_period)],
        "mase_scale_source_by_target": [str(mase_scale_source)],
        "mase_reference_policy": "baseline_history_shared_by_pair_v1",
    }


def _base_contract(
    history: np.ndarray,
    *,
    capability_id: str,
    reference_fields: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
        "capability_id": capability_id,
        "history_only": True,
        "context_length": _CONTEXT_LENGTH,
        "visible_context_length": _VISIBLE_CONTEXT_LENGTH,
        "horizon": _HORIZON,
        "source_history_sha256": _float_array_sha256(history[:, None]),
        "minimum_component_rms_ratio": _MINIMUM_COMPONENT_RMS_RATIO,
        "minimum_future_component_rms_ratio": _MINIMUM_COMPONENT_RMS_RATIO,
        **dict(reference_fields),
    }


def _detect_positive_peaks(
    residual: np.ndarray,
    *,
    start: int,
    stop: int,
    center: float,
    scale: float,
    minimum_distance: int,
    minimum_peak_z: float,
) -> np.ndarray:
    threshold = center + float(minimum_peak_z) * scale
    candidates = [
        index
        for index in range(max(start, 1), min(stop, residual.size - 1))
        if residual[index] >= threshold
        and residual[index] >= residual[index - 1]
        and residual[index] > residual[index + 1]
    ]
    ordered = sorted(candidates, key=lambda index: (-residual[index], index))
    selected: list[int] = []
    for index in ordered:
        if all(abs(index - prior) >= minimum_distance for prior in selected):
            selected.append(index)
    return np.asarray(sorted(selected), dtype=int)


def _peak_half_width(residual: np.ndarray, center: int, floor: float) -> float:
    height = max(float(residual[center] - floor), 0.0)
    if height <= 0.0:
        return 1.0
    threshold = floor + 0.5 * height
    left = center
    right = center
    while left > 0 and center - left < 24 and residual[left - 1] >= threshold:
        left -= 1
    while (
        right + 1 < residual.size
        and right - center < 24
        and residual[right + 1] >= threshold
    ):
        right += 1
    return max(1.0, 0.5 * (right - left + 1))


def _clock_centers(
    *,
    length: int,
    anchor: int,
    interval_pattern: Sequence[int],
) -> list[tuple[int, int]]:
    pattern = tuple(int(value) for value in interval_pattern)
    centers: list[tuple[int, int]] = [(int(anchor), 0)]
    cursor = int(anchor)
    phase = 0
    while cursor < length:
        cursor += pattern[phase]
        phase = (phase + 1) % len(pattern)
        if cursor < length:
            centers.append((cursor, phase))
    cursor = int(anchor)
    phase = 0
    backwards: list[tuple[int, int]] = []
    while cursor >= 0:
        previous_phase = (phase - 1) % len(pattern)
        cursor -= pattern[previous_phase]
        phase = previous_phase
        if cursor >= 0:
            backwards.append((cursor, phase))
    return sorted((*backwards, *centers))


def _match_centers(
    predicted: Sequence[int],
    observed: Sequence[int],
    *,
    tolerance: float,
) -> tuple[int, float]:
    remaining = set(int(value) for value in observed)
    errors: list[float] = []
    for center in predicted:
        if not remaining:
            break
        nearest = min(remaining, key=lambda value: (abs(value - center), value))
        error = abs(nearest - center)
        if error <= tolerance:
            errors.append(float(error))
            remaining.remove(nearest)
    return len(errors), float(np.median(errors)) if errors else 1.0e12


def _event_component(
    *,
    length: int,
    centers: Sequence[tuple[int, int]],
    templates: Sequence[Sequence[float]],
    amplitudes: Sequence[float],
) -> np.ndarray:
    output = np.zeros(length, dtype=float)
    for center, phase in centers:
        template = np.asarray(templates[phase], dtype=float)
        radius = template.size // 2
        lower = max(0, center - radius)
        upper = min(length, center + radius + 1)
        template_lower = lower - (center - radius)
        template_upper = template_lower + (upper - lower)
        output[lower:upper] += (
            float(amplitudes[phase])
            * template[template_lower:template_upper]
        )
    return output


def _fit_clock_candidate(
    residual: np.ndarray,
    train_peaks: np.ndarray,
    validation_peaks: np.ndarray,
    *,
    motif_length: int,
    training_length: int,
    minimum_training_event_count: int,
) -> dict[str, Any] | None:
    if train_peaks.size < max(minimum_training_event_count, 3 * motif_length):
        return None
    intervals = np.diff(train_peaks)
    if intervals.size < 2 * motif_length:
        return None
    pattern = tuple(
        max(
            2,
            int(round(float(np.median(intervals[index::motif_length])))),
        )
        for index in range(motif_length)
    )
    median_interval = float(np.median(pattern))
    widths = np.asarray(
        [
            _peak_half_width(
                residual,
                int(center),
                float(np.median(residual[:training_length])),
            )
            for center in train_peaks
        ],
        dtype=float,
    )
    pulse_width = float(np.median(widths))
    if pulse_width > 0.20 * median_interval:
        return None
    radius = max(2, int(math.ceil(2.0 * pulse_width)))
    anchor = int(train_peaks[0])
    schedule = _clock_centers(
        length=residual.size + _HORIZON,
        anchor=anchor,
        interval_pattern=pattern,
    )
    training_schedule = [
        (center, phase)
        for center, phase in schedule
        if 0 <= center < training_length
    ]
    tolerance = max(1.0, pulse_width)
    phase_windows: list[list[np.ndarray]] = [
        [] for _index in range(motif_length)
    ]
    phase_amplitudes: list[list[float]] = [
        [] for _index in range(motif_length)
    ]
    for center, phase in training_schedule:
        nearest = int(
            min(train_peaks, key=lambda value: (abs(int(value) - center), int(value)))
        )
        if abs(nearest - center) > tolerance:
            continue
        if nearest - radius < 0 or nearest + radius >= training_length:
            continue
        amplitude = max(float(residual[nearest]), 0.0)
        if amplitude <= 1e-12:
            continue
        window = np.clip(
            residual[nearest - radius : nearest + radius + 1],
            0.0,
            None,
        )
        phase_windows[phase].append(window / amplitude)
        phase_amplitudes[phase].append(amplitude)
    if any(not windows for windows in phase_windows):
        return None
    templates: list[list[float]] = []
    amplitudes: list[float] = []
    for windows, values in zip(phase_windows, phase_amplitudes, strict=True):
        template = np.median(np.vstack(windows), axis=0)
        maximum = float(np.max(template))
        if maximum <= 1e-12:
            return None
        templates.append((template / maximum).tolist())
        amplitudes.append(float(np.median(values)) * maximum)
    component = _event_component(
        length=residual.size + _HORIZON,
        centers=schedule,
        templates=templates,
        amplitudes=amplitudes,
    )
    validation_predicted = [
        center
        for center, _phase in schedule
        if training_length <= center < residual.size
    ]
    matched, median_timing_error = _match_centers(
        validation_predicted,
        validation_peaks,
        tolerance=tolerance,
    )
    denominator = len(validation_predicted) + int(validation_peaks.size)
    timing_f1 = 2.0 * matched / denominator if denominator else 0.0
    validation = residual[training_length:]
    prediction = component[training_length : residual.size]
    null_sse = float(np.sum(validation**2))
    full_sse = float(np.sum((validation - prediction) ** 2))
    holdout_r2 = (
        1.0 - full_sse / null_sse if null_sse > 1e-12 else -1.0e12
    )
    return {
        "motif_length": motif_length,
        "anchor": anchor,
        "interval_pattern": list(pattern),
        "pulse_width": pulse_width,
        "template_radius": radius,
        "templates": templates,
        "amplitudes": amplitudes,
        "centers": [[int(center), int(phase)] for center, phase in schedule],
        "component": component,
        "holdout_clock_r2": holdout_r2,
        "timing_f1": timing_f1,
        "median_timing_error": median_timing_error,
        "training_event_count": int(train_peaks.size),
        "validation_event_count": int(validation_peaks.size),
        "validation_predicted_event_count": len(validation_predicted),
    }


def fit_predictable_intermittency_contract(
    history: np.ndarray,
    *,
    background_id: str | None = None,
    carrier_period: float,
    secondary_periods: Sequence[float],
    reference_history: np.ndarray,
    mase_period: int,
    mase_scale: float,
    mase_effective_period: int,
    mase_scale_source: str,
    qualification_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a sparse recurrent event clock and an empirical positive template."""

    values = _validate_history(history)
    references = _reference_fields(
        reference_history,
        mase_period=mase_period,
        mase_scale=mase_scale,
        mase_effective_period=mase_effective_period,
        mase_scale_source=mase_scale_source,
    )
    base = _base_contract(
        values,
        capability_id="predictable_intermittency",
        reference_fields=references,
    )
    qualification, thresholds = _capability_qualification(
        "predictable_intermittency",
        qualification_policy,
    )
    base.update(qualification)
    component_rms_ratio = float(thresholds["minimum_component_rms_ratio"])
    base.update(
        {
            "minimum_component_rms_ratio": component_rms_ratio,
            "minimum_visible_component_rms_ratio": component_rms_ratio,
            "minimum_future_component_rms_ratio": component_rms_ratio,
        }
    )
    training_length = 336
    periods = (float(carrier_period), *map(float, secondary_periods))
    residual, nuisance = _robust_nuisance_residual(
        values,
        periods=periods,
        training_length=training_length,
    )
    center = float(np.median(residual[:training_length]))
    scale = _robust_scale(residual[:training_length])
    minimum_distance = max(2, min(8, int(round(carrier_period / 8.0))))
    train_peaks = _detect_positive_peaks(
        residual,
        start=0,
        stop=training_length,
        center=center,
        scale=scale,
        minimum_distance=minimum_distance,
        minimum_peak_z=float(thresholds["minimum_peak_z"]),
    )
    validation_peaks = _detect_positive_peaks(
        residual,
        start=training_length,
        stop=values.size,
        center=center,
        scale=scale,
        minimum_distance=minimum_distance,
        minimum_peak_z=float(thresholds["minimum_peak_z"]),
    )
    off_event_mask = np.ones(training_length, dtype=bool)
    exclusion_radius = max(2, int(round(carrier_period / 8.0)))
    for peak in train_peaks:
        off_event_mask[
            max(0, int(peak) - exclusion_radius) : min(
                training_length,
                int(peak) + exclusion_radius + 1,
            )
        ] = False
    off_event_scale = (
        _robust_scale(residual[:training_length][off_event_mask])
        if np.count_nonzero(off_event_mask) >= 24
        else scale
    )
    candidates = [
        candidate
        for motif_length in (1, 2, 3)
        if (
            candidate := _fit_clock_candidate(
                residual,
                train_peaks,
                validation_peaks,
                motif_length=motif_length,
                training_length=training_length,
                minimum_training_event_count=int(
                    thresholds["minimum_training_event_count"]
                ),
            )
        )
        is not None
    ]
    if not candidates:
        return _unavailable(
            base,
            "predictable_event_clock_not_identifiable",
            "no one-to-three interval motif had enough history events",
            detected_training_event_count=int(train_peaks.size),
            detected_validation_event_count=int(validation_peaks.size),
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["holdout_clock_r2"]),
            float(row["timing_f1"]),
            -int(row["motif_length"]),
        ),
    )
    component = np.asarray(selected.pop("component"), dtype=float)
    schedule = [tuple(map(int, row)) for row in selected["centers"]]
    model_visible_event_count = sum(
        _VISIBLE_START <= center < _CONTEXT_LENGTH
        for center, _phase in schedule
    )
    fixed_context_event_count = sum(
        _FIXED_START <= center < _CONTEXT_LENGTH
        for center, _phase in schedule
    )
    future_event_count = sum(
        _CONTEXT_LENGTH <= center < _CONTEXT_LENGTH + _HORIZON
        for center, _phase in schedule
    )
    history_schedule = [
        (event_center, phase)
        for event_center, phase in schedule
        if 0 <= event_center < _CONTEXT_LENGTH
    ]
    duty_cycle = float(
        min(
            1.0,
            len(history_schedule)
            * 2.0
            * float(selected["pulse_width"])
            / _CONTEXT_LENGTH,
        )
    )
    training_clock_centers = np.asarray(
        [
            event_center
            for event_center, _phase in schedule
            if 0 <= event_center < training_length
        ],
        dtype=int,
    )
    positive_fraction = float(
        np.mean(residual[training_clock_centers] > center)
        if training_clock_centers.size
        else 0.0
    )
    amplitude_signal_ratio = float(
        np.median(residual[train_peaks] - center)
        / max(off_event_scale, 1e-12)
        if train_peaks.size
        else 0.0
    )
    history_rms = float(np.sqrt(np.mean(component[:_CONTEXT_LENGTH] ** 2)))
    model_visible_rms = float(
        np.sqrt(np.mean(component[_VISIBLE_START:_CONTEXT_LENGTH] ** 2))
    )
    fixed_context_rms = float(
        np.sqrt(np.mean(component[_FIXED_START:_CONTEXT_LENGTH] ** 2))
    )
    future_rms = float(np.sqrt(np.mean(component[_CONTEXT_LENGTH:] ** 2)))
    reference_scale = float(references["normalization_scale_by_target"][0])
    threshold = component_rms_ratio * reference_scale
    gates = {
        "holdout_clock_r2": float(selected["holdout_clock_r2"]),
        "timing_f1": float(selected["timing_f1"]),
        "median_timing_error": float(selected["median_timing_error"]),
        "positive_pulse_fraction": positive_fraction,
        "amplitude_to_off_event_scale": amplitude_signal_ratio,
        "event_duty_cycle": duty_cycle,
        "model_visible_event_count": model_visible_event_count,
        "fixed_context_event_count": fixed_context_event_count,
        "future_event_count": future_event_count,
        "controlled_component_history_rms": history_rms,
        "controlled_component_visible_history_rms": fixed_context_rms,
        "controlled_component_visible_context_length": _FIXED_CONTEXT_LENGTH,
        "controlled_component_model_history_rms": model_visible_rms,
        "controlled_component_future_rms": future_rms,
        "minimum_history_component_rms": threshold,
        "minimum_visible_history_component_rms": threshold,
        "minimum_future_component_rms": threshold,
        "future_component_horizon": _HORIZON,
        "future_component_source": "analytic_history_fitted_event_clock",
    }
    dose_reference = (
        additive_dose_reference(
            capability_id="predictable_intermittency",
            background_id=(
                str(background_id)
                if background_id is not None
                else str(base["source_history_sha256"])
            ),
            unit_gain_history_separation=(
                fixed_context_rms / reference_scale
            ),
            unit_gain_future_separation=future_rms / reference_scale,
            affected_channel_indices=(0,),
        )
        if min(fixed_context_rms, future_rms) > 0.0
        else None
    )
    dose_calibration = qualification.get("dose_calibration")
    dose_mapping_failed = False
    paired_separation_gates: list[dict[str, Any]] | None = None
    if (
        isinstance(dose_calibration, Mapping)
        and dose_calibration.get("status") == "available"
        and isinstance(dose_reference, Mapping)
    ):
        try:
            dose_calibration = resolve_contract_dose_calibration(
                dose_calibration,
                dose_reference,
            )
        except ValueError:
            dose_mapping_failed = True
    if (
        isinstance(dose_calibration, Mapping)
        and dose_calibration.get("status") == "available"
        and not dose_mapping_failed
    ):
        component_delta = component.copy()
        paired_separation_gates = []
        previous_delta: np.ndarray | None = None
        for dose_index, alpha in enumerate(
            dose_calibration["applied_alpha_grid"],
            start=1,
        ):
            current_delta = (float(alpha) - 1.0) * component_delta
            paired_separation_gates.append(
                paired_minimum_separation_gate(
                    current_delta,
                    context_length=_CONTEXT_LENGTH,
                    dose_index=dose_index,
                    dose_calibration=dose_calibration,
                    affected_channel_indices=(0,),
                    scale_by_channel=(reference_scale,),
                    previous_delta=previous_delta,
                )
            )
            previous_delta = current_delta
    failures: list[str] = []
    if float(selected["holdout_clock_r2"]) < float(
        thresholds["minimum_holdout_clock_r2"]
    ):
        failures.append("event_clock_holdout_r2_too_weak")
    if float(selected["timing_f1"]) < float(thresholds["minimum_timing_f1"]):
        failures.append("event_clock_timing_f1_too_weak")
    if int(selected["training_event_count"]) < int(
        thresholds["minimum_training_event_count"]
    ):
        failures.append("insufficient_training_event_repetitions")
    if float(selected["median_timing_error"]) > (
        float(thresholds["maximum_timing_error_widths"])
        * float(selected["pulse_width"])
    ):
        failures.append("event_clock_timing_error_too_large")
    if positive_fraction < float(thresholds["minimum_positive_pulse_fraction"]):
        failures.append("event_polarity_not_stably_positive")
    if amplitude_signal_ratio < float(
        thresholds["minimum_amplitude_to_off_event_scale"]
    ):
        failures.append("event_amplitude_too_weak")
    if duty_cycle > float(thresholds["maximum_event_duty_cycle"]):
        failures.append("event_component_not_intermittent")
    if model_visible_event_count < max(
        3,
        2 * int(selected["motif_length"]) + 1,
    ):
        failures.append("insufficient_visible_event_repetitions")
    if future_event_count < int(thresholds["minimum_future_event_count"]):
        failures.append("no_history_clock_event_in_h48")
    if min(history_rms, fixed_context_rms, future_rms) <= threshold:
        failures.append("event_component_rms_too_weak")
    if isinstance(dose_calibration, Mapping):
        if dose_calibration.get("status") != "available":
            failures.append("dose_calibration_unavailable")
        elif dose_mapping_failed:
            failures.append("contract_source_distance_mapping_unavailable")
        elif paired_separation_gates is None or not all(
            bool(gate["accepted"]) for gate in paired_separation_gates
        ):
            failures.append("paired_minimum_separation_gate_failed")
    if failures:
        return _unavailable(
            base,
            failures[0],
            ",".join(failures),
            **gates,
            dose_design_reference=dose_reference,
            dose_calibration=dose_calibration,
            paired_minimum_separation_gate=paired_separation_gates,
            clock_qualification=selected,
        )
    model = {
        "law": "history_fitted_recurrent_empirical_positive_pulse_clock_v1",
        "dose_response_law": "additive_linear_in_alpha_minus_one",
        "nuisance": nuisance,
        "clock": selected,
        "event_component": component.tolist(),
    }
    model["model_sha256"] = _payload_sha256(model)
    return _finalize_contract(
        {
            **base,
            "available": True,
            "unavailable_reason": None,
            "unavailable_detail": None,
            "controlled_component_rms": history_rms,
            "dose_design_reference": dose_reference,
            "dose_calibration": dose_calibration,
            "paired_minimum_separation_gate": paired_separation_gates,
            **gates,
            "model": model,
        }
    )


def _lag_set(period: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {1, 2, 3}
            | {
                int(np.clip(round(period * fraction), 2, 32))
                for fraction in (1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2)
            }
        )
    )


def _bounded_basis_parameters(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    scale = float(q75 - q25)
    if scale <= 1e-9:
        scale = max(float(np.std(values)), 1.0)
    return median, scale


def _bounded_raw_basis(
    values: np.ndarray,
    *,
    median: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.clip((np.asarray(values) - median) / scale, -3.0, 3.0)
    raw = np.column_stack(
        (
            q * q / (1.0 + q * q),
            q * q * q / (1.0 + np.abs(q) ** 3),
        )
    )
    return q, raw


def _dynamic_design(
    latent: np.ndarray,
    indexes: np.ndarray,
    *,
    period: int,
    nonlinear_lag: int,
    basis_median: float,
    basis_scale: float,
    residualization: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lags = tuple(dict.fromkeys((1, int(period), int(nonlinear_lag))))
    linear = np.column_stack(
        [np.ones(indexes.size)]
        + [latent[indexes - lag] for lag in lags]
    )
    delayed = latent[indexes - nonlinear_lag]
    q, raw = _bounded_raw_basis(
        delayed,
        median=basis_median,
        scale=basis_scale,
    )
    affine = np.column_stack((np.ones(q.size), q))
    if residualization is None:
        residualization, *_ = np.linalg.lstsq(affine, raw, rcond=None)
    nonlinear = raw - affine @ residualization
    return linear, nonlinear, residualization


def _blocked_lag_candidate(
    history: np.ndarray,
    *,
    period: int,
    nonlinear_lag: int,
    nuisance_periods: Sequence[float],
) -> dict[str, Any]:
    gains: list[float] = []
    for training_stop, validation_stop in ((252, 336), (336, 420), (420, 504)):
        # The smooth nuisance is re-fit on each fold's training prefix.  The
        # validation segment is transformed only by those frozen coefficients;
        # fitting one nuisance on all L504 here would leak the held-out fold
        # into lag selection even though the final contract is history-only.
        latent, _nuisance = _robust_nuisance_residual(
            history[:validation_stop],
            periods=nuisance_periods,
            training_length=training_stop,
        )
        start = max(period, nonlinear_lag, 1)
        train_indexes = np.arange(start, training_stop)
        validation_indexes = np.arange(training_stop, validation_stop)
        delayed_train = latent[train_indexes - nonlinear_lag]
        median, scale = _bounded_basis_parameters(delayed_train)
        train_linear, train_nonlinear, residualization = _dynamic_design(
            latent,
            train_indexes,
            period=period,
            nonlinear_lag=nonlinear_lag,
            basis_median=median,
            basis_scale=scale,
            residualization=None,
        )
        validation_linear, validation_nonlinear, _ = _dynamic_design(
            latent,
            validation_indexes,
            period=period,
            nonlinear_lag=nonlinear_lag,
            basis_median=median,
            basis_scale=scale,
            residualization=residualization,
        )
        null_coefficients, *_ = np.linalg.lstsq(
            train_linear,
            latent[train_indexes],
            rcond=None,
        )
        full_coefficients, *_ = np.linalg.lstsq(
            np.column_stack((train_linear, train_nonlinear)),
            latent[train_indexes],
            rcond=None,
        )
        null_error = latent[validation_indexes] - validation_linear @ null_coefficients
        full_error = latent[validation_indexes] - np.column_stack(
            (validation_linear, validation_nonlinear)
        ) @ full_coefficients
        null_sse = float(np.sum(null_error**2))
        full_sse = float(np.sum(full_error**2))
        gains.append(
            1.0 - full_sse / null_sse if null_sse > 1e-12 else -1.0e12
        )
    return {
        "nonlinear_lag": nonlinear_lag,
        "blocked_holdout_gains": gains,
        "median_blocked_holdout_gain": float(np.median(gains)),
        "positive_fold_fraction": float(np.mean(np.asarray(gains) > 0.0)),
    }


def _linear_spectral_radius(
    coefficients: np.ndarray,
    lags: Sequence[int],
) -> float:
    maximum_lag = max(lags)
    companion = np.zeros((maximum_lag, maximum_lag), dtype=float)
    for coefficient, lag in zip(coefficients, lags, strict=True):
        companion[0, lag - 1] += float(coefficient)
    if maximum_lag > 1:
        companion[1:, :-1] = np.eye(maximum_lag - 1)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _nonlinear_response(value: float, model: Mapping[str, Any]) -> float:
    median = float(model["basis_median"])
    scale = float(model["basis_scale"])
    q = float(np.clip((value - median) / scale, -3.0, 3.0))
    raw = np.asarray(
        [q * q / (1.0 + q * q), q**3 / (1.0 + abs(q) ** 3)],
        dtype=float,
    )
    residualization = np.asarray(model["basis_residualization"], dtype=float)
    phi = raw - np.asarray([1.0, q]) @ residualization
    return float(phi @ np.asarray(model["nonlinear_coefficients"], dtype=float))


def _recurrence_value(
    path: np.ndarray,
    index: int,
    *,
    alpha: float,
    model: Mapping[str, Any],
) -> float:
    lags = tuple(int(value) for value in model["linear_lags"])
    linear = float(model["intercept"])
    for coefficient, lag in zip(
        model["linear_coefficients"],
        lags,
        strict=True,
    ):
        linear += float(coefficient) * float(path[index - lag])
    nonlinear_lag = int(model["nonlinear_lag"])
    return linear + alpha * _nonlinear_response(
        float(path[index - nonlinear_lag]),
        model,
    )


def _dynamic_effect(
    latent_history: np.ndarray,
    innovations: np.ndarray,
    *,
    alpha: float,
    model: Mapping[str, Any],
    future_innovations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Alpha one is the authentic source by definition.  Replaying the fitted
    # recurrence plus abducted innovations can differ from the stored latent
    # history by floating-point roundoff, especially after a small frozen
    # normalization scale.  Do not let that numerical reconstruction error
    # masquerade as an intervention in the reference dose curve.
    if float(alpha) == 1.0:
        return (
            np.zeros_like(latent_history, dtype=float),
            np.zeros(_HORIZON, dtype=float),
            latent_history.copy(),
        )
    treatment_history = latent_history.copy()
    for index in range(_VISIBLE_START, _CONTEXT_LENGTH):
        treatment_history[index] = _recurrence_value(
            treatment_history,
            index,
            alpha=alpha,
            model=model,
        ) + innovations[index]
    baseline_rollout = np.concatenate((latent_history, np.zeros(_HORIZON)))
    treatment_rollout = np.concatenate((treatment_history, np.zeros(_HORIZON)))
    future_path = (
        np.zeros(_HORIZON, dtype=float)
        if future_innovations is None
        else np.asarray(future_innovations, dtype=float)
    )
    for offset in range(_HORIZON):
        index = _CONTEXT_LENGTH + offset
        innovation = float(future_path[offset])
        baseline_rollout[index] = _recurrence_value(
            baseline_rollout,
            index,
            alpha=1.0,
            model=model,
        ) + innovation
        treatment_rollout[index] = _recurrence_value(
            treatment_rollout,
            index,
            alpha=alpha,
            model=model,
        ) + innovation
    history_delta = treatment_history - latent_history
    future_delta = (
        treatment_rollout[_CONTEXT_LENGTH:]
        - baseline_rollout[_CONTEXT_LENGTH:]
    )
    return history_delta, future_delta, treatment_history


def _nonlinear_curve_row(
    latent_history: np.ndarray,
    innovations: np.ndarray,
    *,
    alpha: float,
    model: Mapping[str, Any],
    safe_bound: float,
) -> dict[str, Any]:
    history_delta, future_delta, treatment_history = _dynamic_effect(
        latent_history,
        innovations,
        alpha=float(alpha),
        model=model,
    )
    visible_history = history_delta[_VISIBLE_START:]
    fixed_history = history_delta[_FIXED_START:]
    effect = np.concatenate((visible_history, future_delta))
    safe = bool(
        np.isfinite(effect).all()
        and np.isfinite(treatment_history).all()
        and np.max(np.abs(treatment_history)) <= float(safe_bound)
    )
    return {
        "alpha": float(alpha),
        "intervention_rms": float(np.sqrt(np.mean(effect**2))),
        "visible_history_effect_rms": float(
            np.sqrt(np.mean(visible_history**2))
        ),
        "fixed_context_effect_rms": float(
            np.sqrt(np.mean(fixed_history**2))
        ),
        "full_history_effect_rms": float(
            np.sqrt(np.mean(history_delta**2))
        ),
        "future_effect_rms": float(np.sqrt(np.mean(future_delta**2))),
        "safe": safe,
    }


def _nonlinear_dose_reference(
    latent_history: np.ndarray,
    innovations: np.ndarray,
    *,
    background_id: str,
    model: Mapping[str, Any],
    safe_bound: float,
    normalization_scale: float,
) -> tuple[dict[str, Any], dict[float, dict[str, Any]]]:
    rows = [
        _nonlinear_curve_row(
            latent_history,
            innovations,
            alpha=alpha,
            model=model,
            safe_bound=safe_bound,
        )
        for alpha in NONLINEAR_REFERENCE_ALPHA_GRID
    ]
    history_values = [
        float(row["fixed_context_effect_rms"]) for row in rows
    ]
    future_values = [float(row["future_effect_rms"]) for row in rows]
    tolerance = 1e-12
    history_monotone = all(
        right + tolerance >= left
        for left, right in zip(
            history_values,
            history_values[1:],
            strict=False,
        )
    )
    future_monotone = all(
        right + tolerance >= left
        for left, right in zip(
            future_values,
            future_values[1:],
            strict=False,
        )
    )
    all_safe = all(bool(row["safe"]) for row in rows)
    monotone_prefix_count = 1
    for index in range(1, len(rows)):
        if (
            not bool(rows[index]["safe"])
            or float(rows[index]["fixed_context_effect_rms"]) + tolerance
            < float(rows[index - 1]["fixed_context_effect_rms"])
            or float(rows[index]["future_effect_rms"]) + tolerance
            < float(rows[index - 1]["future_effect_rms"])
        ):
            break
        monotone_prefix_count = index + 1
    public_curve = [
        {
            "alpha": float(row["alpha"]),
            "history_separation": float(
                row["fixed_context_effect_rms"]
            ),
            "future_separation": float(row["future_effect_rms"]),
            "safe": bool(row["safe"]),
        }
        for row in rows
    ]
    standardized_curve = [
        {
            **row,
            "history_separation": (
                float(row["history_separation"]) / normalization_scale
            ),
            "future_separation": (
                float(row["future_separation"]) / normalization_scale
            ),
        }
        for row in public_curve
    ]
    evidence = nonlinear_dose_reference(
        background_id=background_id,
        zero_innovation_curve=standardized_curve,
        monotone=bool(history_monotone and future_monotone),
    )
    evidence.update(
        {
            "candidate_alpha_start": 1.0,
            "candidate_alpha_stop": 3.0,
            "candidate_alpha_step": 0.005,
            "history_separation_monotone": history_monotone,
            "future_separation_monotone": future_monotone,
            "all_candidates_safe": all_safe,
            "monotone_safe_prefix_count": monotone_prefix_count,
            "monotone_safe_prefix_alpha_max": float(
                rows[monotone_prefix_count - 1]["alpha"]
            ),
        }
    )
    return evidence, {
        float(row["alpha"]): row for row in rows
    }


def fit_nonlinear_persistence_contract(
    history: np.ndarray,
    *,
    background_id: str | None = None,
    carrier_period: float,
    secondary_periods: Sequence[float],
    reference_history: np.ndarray,
    mase_period: int,
    mase_scale: float,
    mase_effective_period: int,
    mase_scale_source: str,
    qualification_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a bounded nonlinear recurrence with honest blocked lag selection."""

    values = _validate_history(history)
    references = _reference_fields(
        reference_history,
        mase_period=mase_period,
        mase_scale=mase_scale,
        mase_effective_period=mase_effective_period,
        mase_scale_source=mase_scale_source,
    )
    base = _base_contract(
        values,
        capability_id="nonlinear_persistence",
        reference_fields=references,
    )
    qualification, thresholds = _capability_qualification(
        "nonlinear_persistence",
        qualification_policy,
    )
    base.update(qualification)
    component_rms_ratio = float(thresholds["minimum_component_rms_ratio"])
    base.update(
        {
            "minimum_component_rms_ratio": component_rms_ratio,
            "minimum_visible_component_rms_ratio": component_rms_ratio,
            "minimum_future_component_rms_ratio": component_rms_ratio,
        }
    )
    period = int(np.clip(round(float(carrier_period)), 4, 168))
    nuisance, nuisance_metadata = _robust_nuisance_residual(
        values,
        periods=(float(carrier_period), *map(float, secondary_periods)),
        training_length=_CONTEXT_LENGTH,
    )
    latent = nuisance
    candidates = [
        _blocked_lag_candidate(
            values,
            period=period,
            nonlinear_lag=lag,
            nuisance_periods=(
                float(carrier_period),
                *map(float, secondary_periods),
            ),
        )
        for lag in _lag_set(period)
    ]
    ranked_candidates = sorted(
        candidates,
        key=lambda row: (
            float(row["median_blocked_holdout_gain"]),
            float(row["positive_fold_fraction"]),
            -int(row["nonlinear_lag"]),
        ),
        reverse=True,
    )
    qualified_candidates = [
        row
        for row in ranked_candidates
        if float(row["median_blocked_holdout_gain"])
        >= float(thresholds["minimum_median_blocked_holdout_gain"])
        and float(row["positive_fold_fraction"])
        >= float(thresholds["minimum_positive_fold_fraction"])
    ]
    if not qualified_candidates:
        return _unavailable(
            base,
            "nonlinear_blocked_holdout_gain_too_weak",
            "no nonlinear lag passed median gain and fold-sign gates",
            nonlinear_lag_resolution=ranked_candidates[0],
            nonlinear_lag_candidates=candidates,
        )
    selected: Mapping[str, Any] | None = None
    selected_fit: tuple[
        int, np.ndarray, tuple[int, ...], dict[str, Any], float
    ] | None = None
    candidate_stability: list[dict[str, Any]] = []
    for candidate in qualified_candidates:
        candidate_lag = int(candidate["nonlinear_lag"])
        candidate_start = max(period, candidate_lag, 1)
        candidate_indexes = np.arange(candidate_start, _CONTEXT_LENGTH)
        delayed = latent[candidate_indexes - candidate_lag]
        basis_median, basis_scale = _bounded_basis_parameters(delayed)
        linear, nonlinear, residualization = _dynamic_design(
            latent,
            candidate_indexes,
            period=period,
            nonlinear_lag=candidate_lag,
            basis_median=basis_median,
            basis_scale=basis_scale,
            residualization=None,
        )
        coefficients = _ridge_coefficients(
            np.column_stack((linear, nonlinear)),
            latent[candidate_indexes],
        )
        candidate_lags = tuple(dict.fromkeys((1, period, candidate_lag)))
        linear_count = linear.shape[1]
        candidate_model: dict[str, Any] = {
            "law": "bounded_residualized_nonlinear_autoregression_v2",
            "dose_response_law": "dynamic_recursive_nonproportional",
            "period": period,
            "nonlinear_lag": candidate_lag,
            "linear_lags": list(candidate_lags),
            "intercept": float(coefficients[0]),
            "linear_coefficients": coefficients[1:linear_count].tolist(),
            "basis_median": basis_median,
            "basis_scale": basis_scale,
            "basis_clip": 3.0,
            "basis_functions": [
                "q^2/(1+q^2)_residualized_against_[1,q]",
                "q^3/(1+abs(q)^3)_residualized_against_[1,q]",
            ],
            "basis_residualization": residualization.tolist(),
            "nonlinear_coefficients": coefficients[linear_count:].tolist(),
            "nuisance": nuisance_metadata,
            "future_innovation_policy": NONLINEAR_FUTURE_INNOVATION_MAIN_POLICY,
            "history_innovation_policy": "shared_observed_one_step_innovations",
            "intervention_start": _VISIBLE_START,
            "qualified_alpha_max": 3.0,
        }
        radius = _linear_spectral_radius(
            np.asarray(candidate_model["linear_coefficients"]),
            candidate_lags,
        )
        candidate_stability.append(
            {"nonlinear_lag": candidate_lag, "linear_spectral_radius": radius}
        )
        if radius < float(thresholds["maximum_linear_spectral_radius"]):
            selected = candidate
            selected_fit = (
                candidate_lag,
                candidate_indexes,
                candidate_lags,
                candidate_model,
                radius,
            )
            break
    if selected is None or selected_fit is None:
        return _unavailable(
            base,
            "nonlinear_linear_recurrence_unstable",
            "all holdout-qualified nonlinear lags had unstable linear state",
            nonlinear_lag_resolution=qualified_candidates[0],
            nonlinear_lag_candidates=candidates,
            nonlinear_candidate_stability=candidate_stability,
        )
    lag, indexes, lags, model, spectral_radius = selected_fit
    innovations = np.zeros(_CONTEXT_LENGTH, dtype=float)
    for index in indexes:
        innovations[index] = latent[index] - _recurrence_value(
            latent,
            int(index),
            alpha=1.0,
            model=model,
        )
    nonlinear_component = np.asarray(
        [
            _nonlinear_response(float(latent[index - lag]), model)
            for index in indexes
        ],
        dtype=float,
    )
    nonlinear_component_rms = float(np.sqrt(np.mean(nonlinear_component**2)))
    history_scale = float(references["normalization_scale_by_target"][0])
    threshold = component_rms_ratio * history_scale
    safe_bound = max(
        float(thresholds["minimum_absolute_safe_bound"]),
        float(thresholds["maximum_latent_support_multiplier"])
        * float(np.max(np.abs(latent))),
    )
    dose_reference, reference_curve = _nonlinear_dose_reference(
        latent,
        innovations,
        background_id=(
            str(background_id)
            if background_id is not None
            else str(base["source_history_sha256"])
        ),
        model=model,
        safe_bound=safe_bound,
        normalization_scale=history_scale,
    )
    dose_calibration = qualification.get("dose_calibration")
    dose_mapping_failed = False
    if (
        isinstance(dose_calibration, Mapping)
        and dose_calibration.get("status") == "available"
    ):
        try:
            dose_calibration = resolve_contract_dose_calibration(
                dose_calibration,
                dose_reference,
            )
        except ValueError:
            dose_mapping_failed = True
    applied_alpha_grid = (
        _strict_alpha_grid(
            dose_calibration.get("applied_alpha_grid", ()),
            maximum=3.0,
            label="frozen nonlinear applied_alpha_grid",
        )
        if isinstance(dose_calibration, Mapping)
        and dose_calibration.get("status") == "available"
        and not dose_mapping_failed
        else NONLINEAR_ALPHA_GRID
    )
    main_effects = [
        dict(
            reference_curve.get(float(alpha))
            or _nonlinear_curve_row(
                latent,
                innovations,
                alpha=alpha,
                model=model,
                safe_bound=safe_bound,
            )
        )
        for alpha in applied_alpha_grid
    ]
    paired_separation_gates: list[dict[str, Any]] | None = None
    if (
        isinstance(dose_calibration, Mapping)
        and dose_calibration.get("status") == "available"
        and not dose_mapping_failed
    ):
        paired_separation_gates = []
        previous_delta: np.ndarray | None = None
        for dose_index, alpha in enumerate(applied_alpha_grid, start=1):
            history_delta, future_delta, _treatment = _dynamic_effect(
                latent,
                innovations,
                alpha=alpha,
                model=model,
            )
            current_delta = np.concatenate((history_delta, future_delta))
            paired_separation_gates.append(
                paired_minimum_separation_gate(
                    current_delta,
                    context_length=_CONTEXT_LENGTH,
                    dose_index=dose_index,
                    dose_calibration=dose_calibration,
                    affected_channel_indices=(0,),
                    scale_by_channel=(history_scale,),
                    previous_delta=previous_delta,
                )
            )
            previous_delta = current_delta
    rms_values = [row["intervention_rms"] for row in main_effects]
    dose_monotone = all(
        right > left
        for left, right in zip(rms_values, rms_values[1:], strict=False)
    )
    replay = innovations[_CONTEXT_LENGTH - _HORIZON :].copy()
    maximum_applied_alpha = max(applied_alpha_grid)
    replay_history_delta, replay_future_delta, _ = _dynamic_effect(
        latent,
        innovations,
        alpha=maximum_applied_alpha,
        model=model,
        future_innovations=replay,
    )
    replay_rms = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    (replay_history_delta[_VISIBLE_START:], replay_future_delta)
                )
                ** 2
            )
        )
    )
    if abs(maximum_applied_alpha - 2.0) <= 1e-12:
        replay_alpha2_rms = replay_rms
    else:
        replay_alpha2_history, replay_alpha2_future, _ = _dynamic_effect(
            latent,
            innovations,
            alpha=2.0,
            model=model,
            future_innovations=replay,
        )
        replay_alpha2_rms = float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        (
                            replay_alpha2_history[_VISIBLE_START:],
                            replay_alpha2_future,
                        )
                    )
                    ** 2
                )
            )
        )
    maximum_effect = main_effects[-1]
    metrics = {
        "nonlinear_lag_resolution": selected,
        "nonlinear_lag_candidates": candidates,
        "nonlinear_candidate_stability": candidate_stability,
        "linear_spectral_radius": spectral_radius,
        "controlled_component_rms": nonlinear_component_rms,
        "controlled_component_history_rms": float(
            maximum_effect["full_history_effect_rms"]
        ),
        "controlled_component_visible_history_rms": float(
            maximum_effect["fixed_context_effect_rms"]
        ),
        "controlled_component_visible_context_length": _FIXED_CONTEXT_LENGTH,
        "controlled_component_model_history_rms": float(
            maximum_effect["visible_history_effect_rms"]
        ),
        "controlled_component_future_rms": float(
            maximum_effect["future_effect_rms"]
        ),
        "minimum_history_component_rms": threshold,
        "minimum_visible_history_component_rms": threshold,
        "minimum_future_component_rms": threshold,
        "future_component_horizon": _HORIZON,
        "future_component_source": "paired_zero_innovation_dynamic_rollout",
        "dose_response_qualification": main_effects,
        "dose_rms_strictly_increasing": dose_monotone,
        "dose_design_reference": dose_reference,
        "dose_calibration": dose_calibration,
        "paired_minimum_separation_gate": paired_separation_gates,
        "history_residual_replay_sensitivity_alpha2_rms": (
            replay_alpha2_rms
        ),
        "history_residual_replay_sensitivity_max_dose_rms": replay_rms,
        "history_residual_replay_sensitivity_applied_alpha": (
            maximum_applied_alpha
        ),
        "history_residual_replay_policy": (
            NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
        ),
    }
    failures: list[str] = []
    if spectral_radius >= float(thresholds["maximum_linear_spectral_radius"]):
        failures.append("nonlinear_linear_recurrence_unstable")
    if nonlinear_component_rms <= threshold:
        failures.append("nonlinear_component_too_weak")
    if float(maximum_effect["full_history_effect_rms"]) <= threshold:
        failures.append("nonlinear_history_effect_too_weak")
    if float(maximum_effect["fixed_context_effect_rms"]) <= threshold:
        failures.append("nonlinear_visible_history_effect_too_weak")
    if float(maximum_effect["future_effect_rms"]) <= threshold:
        failures.append("nonlinear_zero_innovation_future_effect_too_weak")
    if not dose_monotone:
        failures.append("nonlinear_dynamic_dose_rms_not_monotone")
    if float(dose_reference["monotone_safe_prefix_alpha_max"]) <= 1.0:
        failures.append("nonlinear_no_safe_monotone_dose_interval")
    if isinstance(dose_calibration, Mapping):
        if dose_calibration.get("status") != "available":
            failures.append("dose_calibration_unavailable")
        elif dose_mapping_failed:
            failures.append("contract_source_distance_mapping_unavailable")
        elif paired_separation_gates is None or not all(
            bool(gate["accepted"]) for gate in paired_separation_gates
        ):
            failures.append("paired_minimum_separation_gate_failed")
    if failures:
        return _unavailable(
            base,
            failures[0],
            ",".join(failures),
            **metrics,
        )
    model.update(
        {
            "latent_history": latent.tolist(),
            "history_innovations": innovations.tolist(),
            "history_innovation_sha256": _float_array_sha256(innovations),
            "zero_future_innovation_sha256": _float_array_sha256(
                np.zeros(_HORIZON)
            ),
            "safe_abs_bound": safe_bound,
        }
    )
    model["model_sha256"] = _payload_sha256(model)
    return _finalize_contract(
        {
            **base,
            "available": True,
            "unavailable_reason": None,
            "unavailable_detail": None,
            **metrics,
            "model": model,
        }
    )


def fit_real_path_dynamic_contract(
    history: np.ndarray,
    *,
    capability_id: str,
    background_id: str | None = None,
    carrier_period: float,
    secondary_periods: Sequence[float],
    reference_history: np.ndarray,
    mase_period: int,
    mase_scale: float,
    mase_effective_period: int,
    mase_scale_source: str,
    qualification_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if capability_id == "predictable_intermittency":
        return fit_predictable_intermittency_contract(
            history,
            background_id=background_id,
            carrier_period=carrier_period,
            secondary_periods=secondary_periods,
            reference_history=reference_history,
            mase_period=mase_period,
            mase_scale=mase_scale,
            mase_effective_period=mase_effective_period,
            mase_scale_source=mase_scale_source,
            qualification_policy=qualification_policy,
        )
    if capability_id == "nonlinear_persistence":
        return fit_nonlinear_persistence_contract(
            history,
            background_id=background_id,
            carrier_period=carrier_period,
            secondary_periods=secondary_periods,
            reference_history=reference_history,
            mase_period=mase_period,
            mase_scale=mase_scale,
            mase_effective_period=mase_effective_period,
            mase_scale_source=mase_scale_source,
            qualification_policy=qualification_policy,
        )
    raise ValueError(f"unsupported real-path dynamic capability {capability_id!r}")


def validate_real_path_dynamic_contract(contract: Mapping[str, Any]) -> None:
    payload = dict(contract)
    expected = payload.pop("capability_contract_sha256", None)
    if not isinstance(expected, str) or expected != _payload_sha256(payload):
        raise ValueError("real-path dynamic capability contract hash mismatch")
    if contract.get("schema") not in {
        REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
        LEGACY_REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
    }:
        raise ValueError("unsupported real-path dynamic capability schema")
    if contract.get("capability_id") not in REAL_PATH_DYNAMIC_CAPABILITIES:
        raise ValueError("unsupported real-path dynamic capability id")
    if contract.get("available") is not True:
        return
    model = contract.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("available real-path dynamic contract has no model")
    model_payload = dict(model)
    model_hash = model_payload.pop("model_sha256", None)
    if not isinstance(model_hash, str) or model_hash != _payload_sha256(model_payload):
        raise ValueError("real-path dynamic model hash mismatch")


def _shared_metadata(
    contract: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        "capability_id": str(contract["capability_id"]),
        "contract_sha256": str(model["model_sha256"]),
        "capability_contract_sha256": str(
            contract["capability_contract_sha256"]
        ),
        "source_history_sha256": str(contract["source_history_sha256"]),
        "decomposition_history_sha256": str(contract["source_history_sha256"]),
        "normalization_mean_by_target": list(
            contract["normalization_mean_by_target"]
        ),
        "normalization_scale_by_target": list(
            contract["normalization_scale_by_target"]
        ),
        "normalization_policy": "baseline_history_shared_by_pair_v1",
        "mase_scale_by_target": list(contract["mase_scale_by_target"]),
        "mase_scale": float(contract["mase_scale_by_target"][0]),
        "mase_period": int(contract["mase_period"]),
        "mase_effective_period_by_target": list(
            contract["mase_effective_period_by_target"]
        ),
        "mase_scale_source_by_target": list(
            contract["mase_scale_source_by_target"]
        ),
        "mase_reference_policy": "baseline_history_shared_by_pair_v1",
        "reference_start": int(contract["reference_start"]),
        "reference_length": int(contract["reference_length"]),
        "reference_history_sha256": str(contract["reference_history_sha256"]),
        "reference_history_policy": (
            "unmodified_fit_history_suffix_shared_by_pair_v1"
        ),
        "carrier_fixed": True,
        "dose_response_law": str(model["dose_response_law"]),
    }
    if isinstance(contract.get("dose_calibration"), Mapping):
        metadata["dose_calibration"] = dict(contract["dose_calibration"])
    if isinstance(contract.get("dose_design_reference"), Mapping):
        metadata["dose_design_reference"] = dict(
            contract["dose_design_reference"]
        )
    return metadata


def _dose_gate_for_alpha(
    contract: Mapping[str, Any],
    *,
    alpha: float,
) -> dict[str, Any] | None:
    calibration = contract.get("dose_calibration")
    if not isinstance(calibration, Mapping) or alpha == 1.0:
        return None
    if calibration.get("status") != "available":
        raise ValueError("dynamic contract has no available dose calibration")
    selected_index = next(
        (
            index
            for index, candidate in enumerate(
                calibration["applied_alpha_grid"],
                start=1,
            )
            if abs(float(candidate) - alpha) <= 1e-12
        ),
        None,
    )
    if selected_index is None:
        raise ValueError("dynamic alpha is not on the frozen capability grid")
    gates = contract.get("paired_minimum_separation_gate")
    if not isinstance(gates, list) or len(gates) != len(
        calibration["applied_alpha_grid"]
    ):
        raise ValueError("dynamic contract lacks paired separation gates")
    gate = gates[selected_index - 1]
    if not isinstance(gate, Mapping) or gate.get("accepted") is not True:
        raise ValueError("dynamic dose failed paired minimum separation")
    return dict(gate)


def apply_real_path_dynamic_contract(
    full_baseline: np.ndarray,
    contract: Mapping[str, Any],
    *,
    alpha: float,
    context_length: int | None = None,
    future_innovation_policy: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply an event-clock or recursive nonlinear contract in baseline units."""

    validate_real_path_dynamic_contract(contract)
    if contract.get("available") is not True:
        raise ValueError("cannot apply an unavailable real-path dynamic contract")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("dynamic alpha must be finite and non-negative")
    if context_length is not None and int(context_length) != _CONTEXT_LENGTH:
        raise ValueError("dynamic context length does not match L504 contract")
    baseline = np.asarray(full_baseline, dtype=float)
    was_2d = baseline.ndim == 2
    if was_2d and baseline.shape[1] == 1:
        baseline = baseline[:, 0]
    if baseline.shape != (_CONTEXT_LENGTH + _HORIZON,):
        raise ValueError("dynamic full baseline must be L504+H48")
    if not np.isfinite(baseline).all():
        raise ValueError("dynamic full baseline must be finite")
    if _float_array_sha256(baseline[:_CONTEXT_LENGTH, None]) != contract.get(
        "source_history_sha256"
    ):
        raise ValueError("dynamic baseline source history hash mismatch")
    model = contract["model"]
    capability_id = str(contract["capability_id"])
    paired_gate = _dose_gate_for_alpha(contract, alpha=alpha)
    requested_future_policy = (
        str(model.get("future_innovation_policy"))
        if future_innovation_policy is None
        else str(future_innovation_policy)
    )
    replay_policy = NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
    innovations = (
        np.asarray(model["history_innovations"], dtype=float)
        if capability_id == "nonlinear_persistence"
        else np.empty(0, dtype=float)
    )
    if capability_id != "nonlinear_persistence" and (
        future_innovation_policy is not None
    ):
        raise ValueError(
            "future innovation sensitivity is nonlinear-persistence only"
        )
    if capability_id == "nonlinear_persistence" and requested_future_policy not in {
        str(model["future_innovation_policy"]),
        replay_policy,
    }:
        raise ValueError("unsupported nonlinear future innovation policy")
    if alpha == 1.0:
        delta = np.zeros_like(baseline)
    elif capability_id == "predictable_intermittency":
        component = np.asarray(model["event_component"], dtype=float)
        if component.shape != baseline.shape:
            raise ValueError("event-clock component has the wrong length")
        delta = (alpha - 1.0) * component
    else:
        if alpha > float(model["qualified_alpha_max"]) + 1e-12:
            raise ValueError("nonlinear alpha exceeds the qualified contract range")
        latent = np.asarray(model["latent_history"], dtype=float)
        replay_innovations = (
            np.asarray(innovations[-_HORIZON:], dtype=float)
            if requested_future_policy == replay_policy
            else None
        )
        history_delta, future_delta, treatment_history = _dynamic_effect(
            latent,
            innovations,
            alpha=alpha,
            model=model,
            future_innovations=replay_innovations,
        )
        if np.max(np.abs(treatment_history)) > float(model["safe_abs_bound"]):
            raise ValueError("nonlinear treatment left the qualified support")
        delta = np.concatenate((history_delta, future_delta))
    augmented = baseline + delta
    if paired_gate is not None:
        gate_index = int(paired_gate["dose_index"])
        previous_delta: np.ndarray | None = None
        if gate_index > 1:
            previous_alpha = float(
                contract["dose_calibration"]["applied_alpha_grid"][
                    gate_index - 2
                ]
            )
            if capability_id == "predictable_intermittency":
                previous_delta = (previous_alpha - 1.0) * np.asarray(
                    model["event_component"],
                    dtype=float,
                )
            else:
                previous_history_delta, previous_future_delta, _previous = (
                    _dynamic_effect(
                        latent,
                        innovations,
                        alpha=previous_alpha,
                        model=model,
                        future_innovations=replay_innovations,
                    )
                )
                previous_delta = np.concatenate(
                    (previous_history_delta, previous_future_delta)
                )
        paired_gate = paired_minimum_separation_gate(
            delta,
            context_length=_CONTEXT_LENGTH,
            dose_index=gate_index,
            dose_calibration=contract["dose_calibration"],
            affected_channel_indices=(0,),
            scale_by_channel=contract["normalization_scale_by_target"],
            previous_delta=previous_delta,
        )
        if paired_gate.get("accepted") is not True:
            raise ValueError("dynamic dose failed paired minimum separation")
    visible_delta = delta[_VISIBLE_START:]
    metadata = {
        **_shared_metadata(contract, model),
        "alpha": alpha,
        "intervention_rms": float(np.sqrt(np.mean(visible_delta**2))),
        "visible_history_effect_rms": float(
            np.sqrt(np.mean(delta[_VISIBLE_START:_CONTEXT_LENGTH] ** 2))
        ),
        "fixed_context_effect_rms": float(
            np.sqrt(np.mean(delta[_FIXED_START:_CONTEXT_LENGTH] ** 2))
        ),
        "future_effect_rms": float(
            np.sqrt(np.mean(delta[_CONTEXT_LENGTH:] ** 2))
        ),
        "output_units": "baseline_raw_units",
        "controlled_component": (
            "predictable_recurrent_positive_event_clock"
            if capability_id == "predictable_intermittency"
            else "bounded_nonlinear_autoregressive_gain"
        ),
        "secondary_fixed": True,
        "trend_nonlinearity_fixed": True,
        "amplitude_modulation_fixed": True,
        "regime_level_shift_fixed": True,
        "future_component_source": contract["future_component_source"],
        "paired_minimum_separation_gate": paired_gate,
    }
    if capability_id == "predictable_intermittency":
        clock = model["clock"]
        metadata.update(
            {
                "pulse_centers": [
                    int(center) - _VISIBLE_START
                    for center, _phase in clock["centers"]
                    if _VISIBLE_START
                    <= int(center)
                    < _CONTEXT_LENGTH + _HORIZON
                ],
                "pulse_width": float(clock["pulse_width"]),
                "pulse_interval_pattern": list(clock["interval_pattern"]),
                "pulse_shape": "history_empirical_positive_template",
                "event_clock_holdout_r2": float(clock["holdout_clock_r2"]),
                "event_clock_timing_f1": float(clock["timing_f1"]),
            }
        )
    else:
        metadata.update(
            {
                "nonlinear_lag": int(model["nonlinear_lag"]),
                "seasonal_lag": int(model["period"]),
                "nonlinear_transform": "bounded_residualized_quadratic_cubic",
                "dynamic_contract_replay_verified": True,
                "history_innovation_policy": model["history_innovation_policy"],
                "history_innovation_sha256": model[
                    "history_innovation_sha256"
                ],
                "future_innovation_policy": requested_future_policy,
                "future_innovation_sha256": (
                    _float_array_sha256(
                        np.asarray(innovations[-_HORIZON:], dtype=float)
                    )
                    if requested_future_policy == replay_policy
                    else model["zero_future_innovation_sha256"]
                ),
                "history_residual_replay_sensitivity_alpha2_rms": contract[
                    "history_residual_replay_sensitivity_alpha2_rms"
                ],
                "history_residual_replay_sensitivity_max_dose_rms": contract[
                    "history_residual_replay_sensitivity_max_dose_rms"
                ],
                "history_residual_replay_sensitivity_applied_alpha": contract[
                    "history_residual_replay_sensitivity_applied_alpha"
                ],
                "history_residual_replay_policy": contract[
                    "history_residual_replay_policy"
                ],
                "dose_response_qualification": contract[
                    "dose_response_qualification"
                ],
                "linear_spectral_radius": contract[
                    "linear_spectral_radius"
                ],
                "nonlinear_lag_resolution": contract[
                    "nonlinear_lag_resolution"
                ],
            }
        )
    result = augmented[:, None] if was_2d else augmented
    return result, metadata
