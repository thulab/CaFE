"""Real-path anchored counterfactual construction.

The deterministic synthetic track remains unchanged.  This module builds a
separate benchmark track whose nuisance path is an observed real series and
whose only deterministic part is the declared intervention delta.  All
decomposition fits are history-only and all pair members reuse the baseline
L336 normalization and MASE reference.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from cafe import protocol
from cafe.generation.anchored import (
    AnchoredDecompositionContract,
    apply_real_anchored_contract,
    fit_real_anchored_contract,
)


REAL_ANCHORED_GENERATOR_VERSION = "cafe.real_anchored_generator.v1"
REAL_ANCHORED_BACKGROUND_SCHEMA = (
    "cafe.real_anchored_background_master.v1"
)
REAL_ANCHORED_MASTER_SCHEMA = (
    "cafe.real_anchored_counterfactual_master.v1"
)
REAL_ANCHORED_AVAILABILITY_SCHEMA = (
    "cafe.real_anchored_availability.v1"
)
REAL_ANCHORED_SUPPORTED_CAPABILITIES = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
)
REAL_ANCHORED_ALPHAS = (1.2, 1.4, 1.6, 1.8, 2.0)
REAL_ANCHORED_MINIMUM_CYCLES = 3.0
REAL_ANCHORED_MINIMUM_COMPONENT_RMS_RATIO = 0.01
REAL_ANCHORED_MINIMUM_FUTURE_COMPONENT_RMS_RATIO = 0.01
REAL_ANCHORED_MINIMUM_ELIGIBLE_BACKGROUNDS = 4
REAL_ANCHORED_FIT_WINDOW = protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH


def array_sha256(values: np.ndarray) -> str:
    """Hash a finite numeric array using one public canonical representation."""

    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(b"cafe.real_anchored.array.float64.v1\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _history_1d(background: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(background["_decomposition_history"], dtype=float)
    if values.ndim != 1 or values.shape[0] != (
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
    ):
        raise ValueError("real-anchored decomposition history must be L504")
    if not np.isfinite(values).all():
        raise ValueError("real-anchored decomposition history must be finite")
    return values


def _nonlinear_detrend(values: np.ndarray) -> np.ndarray:
    time = np.linspace(-1.0, 1.0, values.size)
    design = np.column_stack((np.ones(values.size), time, time**2))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _spectral_candidates(
    history: np.ndarray,
    *,
    minimum_period: float = 4.0,
    minimum_cycles: float = REAL_ANCHORED_MINIMUM_CYCLES,
) -> list[dict[str, float]]:
    """Return deterministic history-only Fourier-bin candidates by power."""

    values = _nonlinear_detrend(np.asarray(history, dtype=float))
    length = values.size
    tapered = values * np.hanning(length)
    spectrum = np.fft.rfft(tapered)
    power = np.abs(spectrum) ** 2
    maximum_period = length / float(minimum_cycles)
    candidates: list[dict[str, float]] = []
    total_power = float(np.sum(power[1:]))
    for index in range(1, power.size):
        period = length / float(index)
        if period < minimum_period or period > maximum_period:
            continue
        candidates.append(
            {
                "frequency_bin": float(index),
                "period": float(period),
                "power": float(power[index]),
                "power_share": (
                    float(power[index] / total_power)
                    if total_power > 0.0
                    else 0.0
                ),
            }
        )
    return sorted(
        candidates,
        key=lambda row: (-row["power"], row["frequency_bin"]),
    )


def _frequency_collision(
    candidate_period: float,
    reference_period: float,
    *,
    history_length: int,
) -> bool:
    resolution = 1.5 / float(history_length)
    candidate_frequency = 1.0 / float(candidate_period)
    reference_frequency = 1.0 / float(reference_period)
    # A shorter candidate at an integer multiple of the carrier frequency is
    # merely a carrier harmonic (P8/P6 for a P24 carrier), not a second
    # seasonality.  A longer subharmonic such as P168 beside P24 remains a
    # distinct frequency and is intentionally retained.
    return any(
        abs(candidate_frequency - harmonic * reference_frequency)
        <= resolution
        for harmonic in range(1, 9)
        if harmonic * reference_frequency <= 0.5 + resolution
    )


def resolve_history_periods(
    history: np.ndarray,
    *,
    declared_carrier_period: float,
    maximum_secondary_periods: int = 2,
    minimum_secondary_power_share: float = 0.01,
) -> dict[str, Any]:
    """Resolve one fixed carrier and identifiable non-carrier components.

    The carrier comes from the history-only calibration period policy when it
    has at least three cycles in L504.  Otherwise the strongest admissible
    spectral peak is used.  Secondary peaks must be separated from the carrier
    and its first harmonic, and each must explain at least one percent of the
    tapered detrended spectrum.
    """

    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size < 12 or not np.isfinite(values).all():
        raise ValueError("period resolution requires one finite history")
    candidates = _spectral_candidates(values)
    maximum_period = values.size / REAL_ANCHORED_MINIMUM_CYCLES
    carrier = float(declared_carrier_period)
    carrier_source = "calibration_feature_period"
    if not 2.0 <= carrier <= maximum_period:
        if not candidates:
            raise ValueError("no carrier period has three complete cycles")
        carrier = float(candidates[0]["period"])
        carrier_source = "history_spectral_peak_fallback"
    secondary: list[float] = []
    selected_rows: list[dict[str, float]] = []
    for row in candidates:
        period = float(row["period"])
        if float(row["power_share"]) < minimum_secondary_power_share:
            continue
        if _frequency_collision(
            period,
            carrier,
            history_length=values.size,
        ):
            continue
        if any(
            _frequency_collision(
                period,
                prior,
                history_length=values.size,
            )
            for prior in secondary
        ):
            continue
        secondary.append(period)
        selected_rows.append(dict(row))
        if len(secondary) >= int(maximum_secondary_periods):
            break
    return {
        "schema_version": "cafe.real_anchored_period_resolution.v1",
        "history_only": True,
        "history_length": int(values.size),
        "minimum_cycles": REAL_ANCHORED_MINIMUM_CYCLES,
        "carrier_period": carrier,
        "carrier_source": carrier_source,
        "secondary_periods": secondary,
        "secondary_peaks": selected_rows,
        "minimum_secondary_power_share": minimum_secondary_power_share,
        "candidate_count": len(candidates),
    }


def resolve_modulation_period(
    history: np.ndarray,
    *,
    carrier_period: float,
    minimum_amplitude_cv: float = 0.05,
    minimum_envelope_power_share: float = 0.10,
) -> dict[str, Any]:
    """Resolve a slow carrier-amplitude modulation from history only.

    Carrier amplitude is estimated independently in non-overlapping carrier
    cycles.  A modulation is admitted only when the carrier itself is visible,
    its amplitude varies materially, and one envelope Fourier bin contains a
    meaningful share of the detrended amplitude power.  At least three
    modulation cycles must fit L504.
    """

    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("modulation resolution requires one finite history")
    period_steps = int(round(float(carrier_period)))
    if period_steps < 4:
        return {
            "available": False,
            "unavailable_reason": "carrier_period_too_short_for_envelope",
            "modulation_period": None,
        }
    cycle_count = values.size // period_steps
    if cycle_count < 7:
        return {
            "available": False,
            "unavailable_reason": "insufficient_carrier_cycles_for_modulation",
            "modulation_period": None,
            "carrier_cycle_count": cycle_count,
        }
    detrended = _nonlinear_detrend(values)
    start = values.size - cycle_count * period_steps
    amplitudes: list[float] = []
    for cycle_index in range(cycle_count):
        lower = start + cycle_index * period_steps
        upper = lower + period_steps
        time = np.arange(lower, upper, dtype=float)
        phase = 2.0 * np.pi * time / float(carrier_period)
        design = np.column_stack(
            (np.ones(period_steps), np.sin(phase), np.cos(phase))
        )
        coefficients, *_ = np.linalg.lstsq(
            design,
            detrended[lower:upper],
            rcond=None,
        )
        amplitudes.append(float(np.hypot(coefficients[1], coefficients[2])))
    envelope = np.asarray(amplitudes, dtype=float)
    carrier_amplitude = float(np.mean(envelope))
    history_scale = max(float(np.std(detrended)), 1e-12)
    carrier_strength = carrier_amplitude / history_scale
    amplitude_cv = float(np.std(envelope) / max(carrier_amplitude, 1e-12))
    envelope_time = np.linspace(-1.0, 1.0, cycle_count)
    envelope_design = np.column_stack(
        (np.ones(cycle_count), envelope_time)
    )
    envelope_coefficients, *_ = np.linalg.lstsq(
        envelope_design,
        envelope,
        rcond=None,
    )
    envelope_residual = envelope - envelope_design @ envelope_coefficients
    envelope_power = np.abs(
        np.fft.rfft(envelope_residual * np.hanning(cycle_count))
    ) ** 2
    admissible_bins = [
        index
        for index in range(3, envelope_power.size)
        if cycle_count / float(index) > 1.0
    ]
    total_power = float(np.sum(envelope_power[1:]))
    if not admissible_bins or total_power <= 1e-12:
        return {
            "available": False,
            "unavailable_reason": "amplitude_envelope_spectrum_degenerate",
            "modulation_period": None,
            "carrier_cycle_count": cycle_count,
            "carrier_strength": carrier_strength,
            "amplitude_cv": amplitude_cv,
        }
    selected_bin = max(
        admissible_bins,
        key=lambda index: (float(envelope_power[index]), -index),
    )
    power_share = float(envelope_power[selected_bin] / total_power)
    modulation_period = float(
        period_steps * cycle_count / float(selected_bin)
    )
    reason: str | None = None
    if carrier_strength < 0.10:
        reason = "carrier_component_too_weak_for_modulation"
    elif amplitude_cv < minimum_amplitude_cv:
        reason = "carrier_amplitude_variation_too_weak"
    elif power_share < minimum_envelope_power_share:
        reason = "amplitude_envelope_peak_too_diffuse"
    elif modulation_period <= float(carrier_period):
        reason = "modulation_not_slower_than_carrier"
    elif values.size / modulation_period < REAL_ANCHORED_MINIMUM_CYCLES:
        reason = "insufficient_complete_modulation_cycles"
    return {
        "schema_version": "cafe.real_anchored_modulation_resolution.v1",
        "history_only": True,
        "available": reason is None,
        "unavailable_reason": reason,
        "modulation_period": modulation_period if reason is None else None,
        "carrier_period": float(carrier_period),
        "carrier_period_steps": period_steps,
        "carrier_cycle_count": cycle_count,
        "carrier_strength": carrier_strength,
        "amplitude_cv": amplitude_cv,
        "envelope_frequency_bin": selected_bin,
        "envelope_power_share": power_share,
        "minimum_amplitude_cv": minimum_amplitude_cv,
        "minimum_envelope_power_share": minimum_envelope_power_share,
        "minimum_cycles": REAL_ANCHORED_MINIMUM_CYCLES,
    }


def _nuisance_residual(
    history: np.ndarray,
    *,
    carrier_period: float,
    secondary_periods: Sequence[float],
) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    time = np.linspace(-1.0, 1.0, values.size)
    columns = [np.ones(values.size), time, time**2]
    absolute_time = np.arange(values.size, dtype=float)
    for period in (float(carrier_period), *map(float, secondary_periods)):
        phase = 2.0 * np.pi * absolute_time / period
        columns.extend((np.sin(phase), np.cos(phase)))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def resolve_regime_joinpoint(
    history: np.ndarray,
    *,
    carrier_period: float,
    secondary_periods: Sequence[float],
    visible_context_length: int = protocol.FIXED_CONTEXT_LENGTH,
    minimum_segment_length: int = 24,
    local_comparison_length: int = 72,
    minimum_standardized_jump: float = 0.35,
    minimum_local_sse_reduction: float = 0.05,
) -> dict[str, Any]:
    """Detect one observable history regime level shift without future data."""

    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("regime resolution requires one finite history")
    residual = _nuisance_residual(
        values,
        carrier_period=carrier_period,
        secondary_periods=secondary_periods,
    )
    scale = max(float(np.std(residual)), 1e-12)
    candidate_start = max(
        minimum_segment_length,
        values.size - int(visible_context_length) + minimum_segment_length,
    )
    candidate_stop = values.size - minimum_segment_length
    candidates: list[dict[str, float | int]] = []
    for join_index in range(candidate_start, candidate_stop + 1):
        lower = max(0, join_index - local_comparison_length)
        upper = min(values.size, join_index + local_comparison_length)
        left = residual[lower:join_index]
        right = residual[join_index:upper]
        if (
            left.size < minimum_segment_length
            or right.size < minimum_segment_length
        ):
            continue
        combined = np.concatenate((left, right))
        null_sse = float(np.sum((combined - np.mean(combined)) ** 2))
        alternative_sse = float(
            np.sum((left - np.mean(left)) ** 2)
            + np.sum((right - np.mean(right)) ** 2)
        )
        reduction = (
            max(0.0, 1.0 - alternative_sse / null_sse)
            if null_sse > 1e-12
            else 0.0
        )
        jump = float(np.mean(right) - np.mean(left))
        standardized_jump = abs(jump) / scale
        score = standardized_jump * math.sqrt(
            left.size * right.size / float(left.size + right.size)
        )
        candidates.append(
            {
                "join_index": join_index,
                "jump": jump,
                "standardized_jump": standardized_jump,
                "local_sse_reduction": reduction,
                "selection_score": score,
            }
        )
    if not candidates:
        return {
            "available": False,
            "unavailable_reason": "no_regime_joinpoint_candidate",
            "regime_join_index": None,
        }
    selected = max(
        candidates,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["local_sse_reduction"]),
            -int(row["join_index"]),
        ),
    )
    reason: str | None = None
    if float(selected["standardized_jump"]) < minimum_standardized_jump:
        reason = "regime_level_shift_too_weak"
    elif (
        float(selected["local_sse_reduction"])
        < minimum_local_sse_reduction
    ):
        reason = "regime_joinpoint_sse_reduction_too_weak"
    return {
        "schema_version": "cafe.real_anchored_regime_resolution.v1",
        "history_only": True,
        "available": reason is None,
        "unavailable_reason": reason,
        "regime_join_index": (
            int(selected["join_index"]) if reason is None else None
        ),
        "candidate_range": [candidate_start, candidate_stop],
        "candidate_count": len(candidates),
        "visible_context_length": int(visible_context_length),
        "minimum_segment_length": minimum_segment_length,
        "local_comparison_length": local_comparison_length,
        "minimum_standardized_jump": minimum_standardized_jump,
        "minimum_local_sse_reduction": minimum_local_sse_reduction,
        **selected,
    }


def fit_background_capability_contracts(
    backgrounds: Sequence[dict[str, Any]],
    *,
    capability_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit per-background contracts and freeze explicit availability rows."""

    requested = tuple(dict.fromkeys(str(value) for value in capability_ids))
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for background in backgrounds:
        history = _history_1d(background)
        try:
            period_resolution = resolve_history_periods(
                history,
                declared_carrier_period=float(background["feature_period"]),
            )
        except ValueError as error:
            period_resolution = None
            period_error = str(error)
        else:
            period_error = None
        if period_resolution is None:
            modulation_resolution = None
            regime_resolution = None
        else:
            carrier_period = float(period_resolution["carrier_period"])
            secondary_periods = tuple(
                float(value)
                for value in period_resolution["secondary_periods"]
            )
            modulation_resolution = resolve_modulation_period(
                history,
                carrier_period=carrier_period,
            )
            regime_resolution = resolve_regime_joinpoint(
                history,
                carrier_period=carrier_period,
                secondary_periods=secondary_periods,
            )
        for capability_id in requested:
            base = {
                "schema_version": (
                    "cafe.real_anchored_background_capability.v2"
                ),
                "dataset_id": str(background["dataset_id"]),
                "background_id": str(background["background_id"]),
                "capability_id": capability_id,
                "benchmark_track": "real_anchored_counterfactual",
                "source_history_sha256": str(
                    background["decomposition_history_sha256"]
                ),
                "period_resolution": period_resolution,
                "modulation_resolution": modulation_resolution,
                "regime_resolution": regime_resolution,
            }
            if capability_id not in REAL_ANCHORED_SUPPORTED_CAPABILITIES:
                reason = "real_univariate_transform_not_implemented"
                row = {
                    **base,
                    "available": False,
                    "unavailable_reason": reason,
                    "unavailable_detail": None,
                    "contract": None,
                }
            elif period_resolution is None:
                reason = "carrier_period_not_identifiable"
                row = {
                    **base,
                    "available": False,
                    "unavailable_reason": reason,
                    "unavailable_detail": period_error,
                    "contract": None,
                }
            elif (
                capability_id == "multi_seasonal"
                and not period_resolution["secondary_periods"]
            ):
                reason = "secondary_period_not_identifiable"
                row = {
                    **base,
                    "available": False,
                    "unavailable_reason": reason,
                    "unavailable_detail": (
                        "no non-carrier spectral peak passed separation and "
                        "power gates"
                    ),
                    "contract": None,
                }
            elif (
                capability_id == "time_varying_seasonality"
                and (
                    modulation_resolution is None
                    or modulation_resolution.get("available") is not True
                )
            ):
                reason = (
                    "modulation_period_not_identifiable"
                    if modulation_resolution is None
                    else str(
                        modulation_resolution.get(
                            "unavailable_reason",
                            "modulation_period_not_identifiable",
                        )
                    )
                )
                row = {
                    **base,
                    "available": False,
                    "unavailable_reason": reason,
                    "unavailable_detail": (
                        "carrier amplitude envelope failed the history-only "
                        "strength, concentration, or cycle gate"
                    ),
                    "contract": None,
                }
            elif (
                capability_id == "regime_switching"
                and (
                    regime_resolution is None
                    or regime_resolution.get("available") is not True
                )
            ):
                reason = (
                    "regime_joinpoint_not_identifiable"
                    if regime_resolution is None
                    else str(
                        regime_resolution.get(
                            "unavailable_reason",
                            "regime_joinpoint_not_identifiable",
                        )
                    )
                )
                row = {
                    **base,
                    "available": False,
                    "unavailable_reason": reason,
                    "unavailable_detail": (
                        "no recent history-only level shift passed the jump "
                        "and local SSE-reduction gates"
                    ),
                    "contract": None,
                }
            else:
                fitted = fit_real_anchored_contract(
                    history,
                    capability_id=capability_id,
                    carrier_period=float(
                        period_resolution["carrier_period"]
                    ),
                    secondary_periods=(
                        tuple(period_resolution["secondary_periods"])
                    ),
                    horizon=protocol.HORIZON,
                    fit_window=REAL_ANCHORED_FIT_WINDOW,
                    trend_window=96,
                    trend_degree=2,
                    harmonics_per_period=1,
                    modulation_period=(
                        None
                        if capability_id != "time_varying_seasonality"
                        or modulation_resolution is None
                        or modulation_resolution.get("available") is not True
                        else float(
                            modulation_resolution["modulation_period"]
                        )
                    ),
                    regime_join_index=(
                        None
                        if capability_id != "regime_switching"
                        or regime_resolution is None
                        or regime_resolution.get("available") is not True
                        else int(regime_resolution["regime_join_index"])
                    ),
                    minimum_regime_segment_length=24,
                    minimum_cycles=REAL_ANCHORED_MINIMUM_CYCLES,
                    mase_period=int(background["mase_period"]),
                    minimum_component_rms_ratio=(
                        REAL_ANCHORED_MINIMUM_COMPONENT_RMS_RATIO
                    ),
                    minimum_future_component_rms_ratio=(
                        REAL_ANCHORED_MINIMUM_FUTURE_COMPONENT_RMS_RATIO
                    ),
                    reference_history=history[
                        -protocol.REAL_ANCHORED_CONTEXT_LENGTH:
                    ],
                )
                reason = fitted.get("unavailable_reason")
                row = {
                    **base,
                    "available": bool(fitted["available"]),
                    "unavailable_reason": reason,
                    "unavailable_detail": fitted.get(
                        "unavailable_detail"
                    ),
                    "controlled_component_rms": fitted.get(
                        "controlled_component_rms"
                    ),
                    "controlled_component_history_rms": fitted.get(
                        "controlled_component_history_rms"
                    ),
                    "controlled_component_future_rms": fitted.get(
                        "controlled_component_future_rms"
                    ),
                    "minimum_history_component_rms": fitted.get(
                        "minimum_history_component_rms"
                    ),
                    "minimum_future_component_rms": fitted.get(
                        "minimum_future_component_rms"
                    ),
                    "minimum_component_rms_ratio": fitted.get(
                        "minimum_component_rms_ratio"
                    ),
                    "minimum_future_component_rms_ratio": fitted.get(
                        "minimum_future_component_rms_ratio"
                    ),
                    "future_component_horizon": fitted.get(
                        "future_component_horizon"
                    ),
                    "future_component_source": fitted.get(
                        "future_component_source"
                    ),
                    "contract": fitted,
                }
            if not row["available"]:
                reason_counts[str(row["unavailable_reason"])] += 1
            rows.append(row)
    summary = build_availability(
        rows,
        requested_capability_ids=requested,
        minimum_eligible_backgrounds=(
            REAL_ANCHORED_MINIMUM_ELIGIBLE_BACKGROUNDS
        ),
    )
    summary["unavailable_background_reason_counts"] = dict(
        sorted(reason_counts.items())
    )
    return rows, summary


def build_availability(
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    requested_capability_ids: Iterable[str],
    minimum_eligible_backgrounds: int,
) -> dict[str, Any]:
    """Build an explicit dataset x capability eligibility artifact."""

    if minimum_eligible_backgrounds < 1:
        raise ValueError("minimum eligible backgrounds must be positive")
    requested = tuple(dict.fromkeys(str(value) for value in requested_capability_ids))
    dataset_ids = sorted(
        {str(row["dataset_id"]) for row in contract_rows}
    )
    dataset_id = dataset_ids[0] if len(dataset_ids) == 1 else None
    cells: list[dict[str, Any]] = []
    for capability_id in requested:
        selected = [
            row
            for row in contract_rows
            if str(row["capability_id"]) == capability_id
        ]
        eligible_ids = sorted(
            str(row["background_id"])
            for row in selected
            if row.get("available") is True
        )
        background_reasons = sorted(
            {
                str(row.get("unavailable_reason"))
                for row in selected
                if row.get("available") is not True
                and row.get("unavailable_reason") is not None
            }
        )
        status = (
            "available"
            if len(eligible_ids) >= minimum_eligible_backgrounds
            else "unavailable"
        )
        reasons: list[str] = []
        if status == "unavailable":
            reasons = sorted(
                {
                    *background_reasons,
                    "insufficient_eligible_backgrounds",
                }
            )
        gate_rows = [
            row
            for row in selected
            if all(
                isinstance(row.get(field), (int, float))
                and math.isfinite(float(row[field]))
                for field in (
                    "controlled_component_history_rms",
                    "controlled_component_future_rms",
                    "minimum_history_component_rms",
                    "minimum_future_component_rms",
                )
            )
        ]

        def metric_range(field: str) -> dict[str, float] | None:
            values = [float(row[field]) for row in gate_rows]
            if not values:
                return None
            return {
                "minimum": min(values),
                "maximum": max(values),
            }

        history_ratios = sorted(
            {
                float(row["minimum_component_rms_ratio"])
                for row in gate_rows
                if isinstance(
                    row.get("minimum_component_rms_ratio"),
                    (int, float),
                )
                and math.isfinite(
                    float(row["minimum_component_rms_ratio"])
                )
            }
        )
        future_ratios = sorted(
            {
                float(row["minimum_future_component_rms_ratio"])
                for row in gate_rows
                if isinstance(
                    row.get("minimum_future_component_rms_ratio"),
                    (int, float),
                )
                and math.isfinite(
                    float(row["minimum_future_component_rms_ratio"])
                )
            }
        )
        cells.append(
            {
                "dataset_id": dataset_id,
                "capability_id": capability_id,
                "requested": True,
                "status": status,
                "reason_codes": reasons,
                "background_unavailable_reason_codes": background_reasons,
                "requested_background_count": len(selected),
                "eligible_background_count": len(eligible_ids),
                "minimum_eligible_background_count": (
                    minimum_eligible_backgrounds
                ),
                "eligible_background_ids_sha256": protocol.json_sha256(
                    eligible_ids
                ),
                "supported_context_lengths": [
                    protocol.FIXED_CONTEXT_LENGTH
                ],
                "dose_parameter": "alpha",
                "supported_dose_values": list(REAL_ANCHORED_ALPHAS),
                "controlled_component_rms_gate": {
                    "history_source": "history_fitted_component_l504",
                    "future_source": (
                        "analytic_history_fitted_component_extension"
                    ),
                    "future_horizon": protocol.HORIZON,
                    "evaluated_background_count": len(gate_rows),
                    "history_minimum_rms_ratios": history_ratios,
                    "future_minimum_rms_ratios": future_ratios,
                    "history_rms_range": metric_range(
                        "controlled_component_history_rms"
                    ),
                    "future_rms_range": metric_range(
                        "controlled_component_future_rms"
                    ),
                    "history_threshold_range": metric_range(
                        "minimum_history_component_rms"
                    ),
                    "future_threshold_range": metric_range(
                        "minimum_future_component_rms"
                    ),
                },
            }
        )
    return {
        "schema_version": REAL_ANCHORED_AVAILABILITY_SCHEMA,
        "benchmark_track": "real_anchored_counterfactual",
        "dataset_id": dataset_id,
        "minimum_eligible_backgrounds": minimum_eligible_backgrounds,
        "cells": cells,
    }


def available_capabilities(
    availability: Mapping[str, Any],
) -> tuple[str, ...]:
    return tuple(
        str(cell["capability_id"])
        for cell in availability.get("cells", [])
        if cell.get("status") == "available"
    )


def validate_availability_contract(
    availability: Mapping[str, Any],
    contract_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Verify that persisted cell availability exactly matches anchor rows."""

    cells = availability.get("cells")
    if not isinstance(cells, list):
        raise ValueError("real-anchored availability is missing cells")
    requested = tuple(str(cell["capability_id"]) for cell in cells)
    minimum = int(availability.get("minimum_eligible_backgrounds", 0))
    recomputed = build_availability(
        contract_rows,
        requested_capability_ids=requested,
        minimum_eligible_backgrounds=minimum,
    )
    observed_cells = [dict(cell) for cell in cells]
    expected_cells = [dict(cell) for cell in recomputed["cells"]]
    dataset_id = availability.get("dataset_id")
    for cell in expected_cells:
        cell["dataset_id"] = dataset_id
    if observed_cells != expected_cells:
        raise ValueError(
            "real-anchored availability cells disagree with contracts"
        )
    if availability.get("schema_version") != REAL_ANCHORED_AVAILABILITY_SCHEMA:
        raise ValueError("unsupported real-anchored availability schema")
    if availability.get("benchmark_track") != (
        "real_anchored_counterfactual"
    ):
        raise ValueError("real-anchored availability lost track identity")


def public_background(background: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the private fit payload while keeping the required L168 prefix."""

    result = dict(background)
    history = np.asarray(result.pop("_decomposition_history"), dtype=float)
    prefix_length = (
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        - protocol.REAL_ANCHORED_CONTEXT_LENGTH
    )
    prefix = history[:prefix_length]
    result["decomposition_prefix"] = prefix.tolist()
    result["decomposition_prefix_sha256"] = array_sha256(prefix)
    result["benchmark_track"] = "real_accuracy"
    return result


def reconstruct_source_baseline(
    background: Mapping[str, Any],
) -> np.ndarray:
    prefix = np.asarray(background["decomposition_prefix"], dtype=float)
    visible = np.asarray(background["target"], dtype=float)
    if visible.ndim == 2 and visible.shape[1] == 1:
        visible = visible[:, 0]
    expected_prefix = (
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        - protocol.REAL_ANCHORED_CONTEXT_LENGTH
    )
    if prefix.shape != (expected_prefix,):
        raise ValueError("real-anchored background prefix has the wrong shape")
    if visible.shape != (protocol.REAL_ANCHORED_MASTER_LENGTH,):
        raise ValueError("real-anchored background target has the wrong shape")
    if array_sha256(prefix) != background["decomposition_prefix_sha256"]:
        raise ValueError("real-anchored background prefix hash mismatch")
    return np.concatenate((prefix, visible))


def _target_hash(values: np.ndarray) -> str:
    return protocol.target_and_covariate_sha256(values[:, None], None)


def _sample_row(
    *,
    background: Mapping[str, Any],
    contract_row: Mapping[str, Any],
    seed_index: int,
    dose_index: int,
    pair_member: int,
    alpha: float,
    visible_target: np.ndarray,
    visible_delta: np.ndarray,
    baseline_visible: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    capability_id = str(contract_row["capability_id"])
    dataset_token = protocol.safe_id(str(background["dataset_id"]))
    pair_id = (
        f"cafe_real_cf__{dataset_token}__{capability_id}__"
        f"a{dose_index}__seed{seed_index:06d}"
    )
    sample_id = f"{pair_id}__m{pair_member}"
    baseline_id = f"{pair_id}__m0"
    target_2d = np.asarray(visible_target, dtype=float)[:, None]
    delta_2d = np.asarray(visible_delta, dtype=float)[:, None]
    context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
    return {
        "schema_version": REAL_ANCHORED_MASTER_SCHEMA,
        "feature_schema_version": protocol.FEATURE_SCHEMA_VERSION,
        "benchmark_track": "real_anchored_counterfactual",
        "sample_id": sample_id,
        "master_sample_id": sample_id,
        "baseline_sample_id": baseline_id,
        "paired_group_id": (
            f"cafe_real_cf__{dataset_token}__{capability_id}__"
            f"seed{seed_index:06d}"
        ),
        "counterfactual_pair_id": pair_id,
        "counterfactual_member": pair_member,
        "dataset_id": str(background["dataset_id"]),
        "config_id": str(background["config_id"]),
        "task_id": str(background["task_view_id"]),
        "task_view_id": str(background["task_view_id"]),
        "profile_id": f"real_anchored_{capability_id}_v1",
        "anchor_id": str(background["background_id"]),
        "background_id": str(background["background_id"]),
        "anchor_provenance": {
            key: background[key]
            for key in (
                "item_id",
                "series_id",
                "channel_id",
                "decomposition_start",
                "context_start",
                "forecast_origin",
                "history_sha256",
            )
        },
        "generator_version": REAL_ANCHORED_GENERATOR_VERSION,
        "generator_family_role": "real_anchored",
        "generator_family_id": f"real_anchored_{capability_id}_v1",
        "capability_id": capability_id,
        "intensity": dose_index,
        "intensity_lambda": alpha,
        "dose_index": dose_index,
        "dose_parameter": "alpha",
        "dose_value": alpha,
        "baseline_dose_value": 1.0,
        "seed_index": seed_index,
        "sample_index": seed_index,
        "context_length": context,
        "horizon": protocol.HORIZON,
        "target_dim": 1,
        "covariate_dim": 0,
        "covariate_column_names": [],
        "frequency": str(background["frequency"]),
        "season_length": int(background["season_length"]),
        "calendar_season_length": int(background["calendar_season_length"]),
        "calendar_season_feature_observable": bool(
            background["calendar_season_feature_observable"]
        ),
        "calendar_cycles_in_calibration_history": float(
            background["calendar_cycles_in_calibration_history"]
        ),
        "feature_period": int(background["feature_period"]),
        "feature_period_source": str(background["feature_period_source"]),
        "hierarchy": None,
        "target_feature": "real_anchored_intervention_rms",
        "target_feature_value": float(metadata["intervention_rms"]),
        "intensity_target_feature_value": float(
            metadata["intervention_rms"]
        ),
        "synthetic_target_feature_reference": (
            protocol.PRIMARY_TARGET_FEATURE[capability_id]
        ),
        "intensity_calibration": {
            "policy": "physical_component_amplitude_alpha_grid_v1",
            "scope": "real_anchored_history_only_decomposition",
            "formal_seed_inverse": False,
            "sample_level_target_gate": True,
            "selected_alphas": list(REAL_ANCHORED_ALPHAS),
        },
        "realized_features": {},
        "sampled_generator_parameters": {
            "alpha": alpha,
            "controlled_component": metadata["controlled_component"],
        },
        "parameter_mapping": {},
        "parameter_sampling": {
            "policy": "real_background_contract_deterministic_selection_v1",
            "background_id": str(background["background_id"]),
            "contract_sha256": str(metadata["contract_sha256"]),
        },
        "generation_metadata": dict(metadata),
        "evaluation_table": "main",
        "input_history_semantics": (
            "observed_real_history_plus_declared_intervention"
        ),
        "scoring_target_semantics": (
            "held_out_real_future_plus_history_fitted_intervention"
        ),
        "observation_noise_scale": 0.0,
        "future_process_noise_scale": None,
        "mase_period": int(background["mase_period"]),
        "mase_period_source": str(background["mase_period_source"]),
        "mase_scale": float(background["mase_scale"]),
        "mase_scale_by_target": list(background["mase_scale_by_target"]),
        "mase_scale_effective_period_by_target": list(
            background["mase_scale_effective_period_by_target"]
        ),
        "mase_scale_fallback_target_indices": list(
            background["mase_scale_fallback_target_indices"]
        ),
        "mase_scale_policy": str(background["mase_scale_policy"]),
        "mase_scale_source": "shared_unmodified_real_l336_history",
        "shared_standardization": dict(background["standardization"]),
        "baseline_history_sha256": array_sha256(
            baseline_visible[:context]
        ),
        "baseline_future_sha256": array_sha256(
            baseline_visible[context:]
        ),
        "baseline_target_sha256": array_sha256(baseline_visible),
        "intervention_delta_sha256": array_sha256(delta_2d),
        "target_sha256": _target_hash(target_2d),
        "future_sha256": array_sha256(target_2d[context:]),
        "anti_copy_gate": {
            "status": "not_applicable",
            "reason_code": "intentional_real_anchor_counterfactual",
        },
        "target": target_2d.tolist(),
        "covariates": None,
    }


def real_anchored_assignments(
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    capability_ids: Iterable[str],
    seed_indexes: Iterable[int],
) -> dict[str, list[tuple[int, Mapping[str, Any]]]]:
    """Assign unique authentic backgrounds across the full experiment.

    Synthetic seed identities are retained in sample IDs, but they do not
    create new independent real observations.  Each capability therefore uses
    a frozen permutation of eligible backgrounds without replacement. Global
    seed ordinals beyond the available-background count produce no assignment.
    """

    seeds = tuple(int(value) for value in seed_indexes)
    if len(seeds) != len(set(seeds)) or any(value < 0 for value in seeds):
        raise ValueError("real-anchored seed indexes must be unique/non-negative")
    requested = tuple(dict.fromkeys(str(value) for value in capability_ids))
    assignments: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for capability_id in requested:
        eligible = [
            row
            for row in contract_rows
            if str(row["capability_id"]) == capability_id
            and row.get("available") is True
        ]
        ordered = sorted(
            eligible,
            key=lambda row: (
                protocol.stable_seed(
                    str(row["dataset_id"]),
                    capability_id,
                    str(row["background_id"]),
                    "real-anchored-background-permutation",
                    base=protocol.REAL_ANCHORED_SAMPLE_SEED,
                ),
                str(row["background_id"]),
            ),
        )
        background_ids = [str(row["background_id"]) for row in ordered]
        if len(background_ids) != len(set(background_ids)):
            raise ValueError(
                "real-anchored contracts contain duplicate eligible "
                f"backgrounds for {capability_id}"
            )
        # Seed indexes are global experiment ordinals, not independent draws.
        # Direct indexing makes the mapping invariant to shard boundaries:
        # seed 16 receives the same (unique) background whether it is generated
        # in [0, 32) or in a separate [16, 32) shard.  Indexes beyond the
        # authentic background bank are deliberately unavailable rather than
        # recycling observations under fresh synthetic IDs.
        assignments[capability_id] = [
            (seed_index, ordered[seed_index])
            for seed_index in seeds
            if seed_index < len(ordered)
        ]
    return assignments


def iter_real_anchored_samples(
    backgrounds: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    capability_ids: Iterable[str],
    seed_indexes: Iterable[int],
    alphas: Sequence[float] = REAL_ANCHORED_ALPHAS,
) -> Iterator[dict[str, Any]]:
    """Yield exact baseline/treatment pairs for every requested physical dose."""

    alpha_values = tuple(float(value) for value in alphas)
    if not alpha_values or any(
        not math.isfinite(value) or value <= 1.0 for value in alpha_values
    ):
        raise ValueError("real-anchored alpha values must be finite and > 1")
    if any(
        right <= left
        for left, right in zip(alpha_values, alpha_values[1:])
    ):
        raise ValueError("real-anchored alpha values must increase strictly")
    by_background = {
        str(background["background_id"]): background
        for background in backgrounds
    }
    assignments = real_anchored_assignments(
        contract_rows,
        capability_ids=capability_ids,
        seed_indexes=seed_indexes,
    )
    for capability_id, capability_assignments in assignments.items():
        for seed_index, contract_row in capability_assignments:
            background = by_background[str(contract_row["background_id"])]
            source_baseline = reconstruct_source_baseline(background)
            visible_start = (
                protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                - protocol.REAL_ANCHORED_CONTEXT_LENGTH
            )
            baseline_visible = source_baseline[visible_start:]
            for dose_index, treatment_alpha in enumerate(alpha_values, start=1):
                for pair_member, alpha in ((0, 1.0), (1, treatment_alpha)):
                    augmented, metadata = apply_real_anchored_contract(
                        source_baseline,
                        contract_row["contract"],
                        alpha=alpha,
                        context_length=(
                            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                        ),
                    )
                    augmented_array = np.asarray(augmented, dtype=float)
                    if augmented_array.ndim == 2:
                        augmented_array = augmented_array[:, 0]
                    visible_target = augmented_array[visible_start:]
                    visible_delta = visible_target - baseline_visible
                    yield _sample_row(
                        background=background,
                        contract_row=contract_row,
                        seed_index=int(seed_index),
                        dose_index=dose_index,
                        pair_member=pair_member,
                        alpha=alpha,
                        visible_target=visible_target,
                        visible_delta=visible_delta,
                        baseline_visible=baseline_visible,
                        metadata=metadata,
                    )


def validate_contract_integrity(contract_row: Mapping[str, Any]) -> None:
    """Raise when a persisted available contract cannot be reconstructed."""

    if contract_row.get("available") is not True:
        return
    capability = contract_row.get("contract")
    if not isinstance(capability, Mapping):
        raise ValueError("available real-anchored row has no contract")
    decomposition = capability.get("decomposition_contract")
    if not isinstance(decomposition, Mapping):
        raise ValueError("available real-anchored row has no decomposition")
    restored = AnchoredDecompositionContract.from_dict(decomposition)
    if restored.contract_sha256 != decomposition.get("contract_sha256"):
        raise ValueError("real-anchored decomposition hash mismatch")
