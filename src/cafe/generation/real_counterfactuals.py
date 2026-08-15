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
    ANCHORED_CONTRACT_SCHEMA,
    AnchoredDecompositionContract,
    apply_real_anchored_contract,
    fit_real_anchored_contract,
)
from cafe.generation.real_anchored_policy import (
    MINIMUM_FORMAL_BACKGROUND_COUNT,
    NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY,
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
    REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
    TIME_VARYING_SEASONALITY_BASIS_POLICY,
)
from cafe.generation.real_anchored_dose import (
    additive_dose_reference,
    dose_calibration_from_policy,
    dose_targets,
    paired_minimum_separation_gate,
    resolve_contract_dose_calibration,
    validate_dose_calibration,
)
from cafe.generation.real_path_dynamics import (
    LEGACY_REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
    REAL_PATH_DYNAMIC_CAPABILITIES,
    REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
    apply_real_path_dynamic_contract,
    dynamic_qualification_provenance,
    fit_real_path_dynamic_contract,
    validate_real_path_dynamic_contract,
)


REAL_ANCHORED_GENERATOR_VERSION = "cafe.real_anchored_generator.v5"
REAL_ANCHORED_BACKGROUND_SCHEMA = (
    "cafe.real_anchored_background_master.v1"
)
REAL_ANCHORED_MASTER_SCHEMA = (
    "cafe.real_anchored_counterfactual_master.v3"
)
REAL_ANCHORED_AVAILABILITY_SCHEMA = (
    "cafe.real_anchored_availability.v3"
)
REAL_ANCHORED_SUPPORTED_CAPABILITIES = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
)
# Legacy compatibility declaration only. Formal v5 generation reads each
# contract-specific history-only ``applied_alpha_grid`` from its contract row.
REAL_ANCHORED_ALPHAS = (1.2, 1.4, 1.6, 1.8, 2.0)
REAL_ANCHORED_MINIMUM_CYCLES = 3.0
REAL_ANCHORED_MINIMUM_COMPONENT_RMS_RATIO = 0.01
REAL_ANCHORED_MINIMUM_VISIBLE_COMPONENT_RMS_RATIO = 0.01
REAL_ANCHORED_MINIMUM_FUTURE_COMPONENT_RMS_RATIO = 0.01
REAL_ANCHORED_MINIMUM_ELIGIBLE_BACKGROUNDS = (
    MINIMUM_FORMAL_BACKGROUND_COUNT
)
REAL_ANCHORED_FIT_WINDOW = protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
REAL_ANCHORED_VISIBLE_CONTEXT_LENGTH = protocol.FIXED_CONTEXT_LENGTH

REAL_ANCHORED_MAXIMUM_SECONDARY_PERIODS = 2
REAL_ANCHORED_MINIMUM_SECONDARY_POWER_SHARE = 0.01
REAL_ANCHORED_MINIMUM_CARRIER_RMS_RATIO = 0.10
REAL_ANCHORED_MINIMUM_CARRIER_POWER_SHARE = 0.01
REAL_ANCHORED_MINIMUM_AMPLITUDE_CV = 0.05
REAL_ANCHORED_MINIMUM_ENVELOPE_POWER_SHARE = 0.10
REAL_ANCHORED_MINIMUM_MODULATION_CARRIER_STRENGTH = 0.10
REAL_ANCHORED_REGIME_MINIMUM_SEGMENT_LENGTH = 24
REAL_ANCHORED_REGIME_LOCAL_COMPARISON_LENGTH = 72
REAL_ANCHORED_REGIME_MINIMUM_STANDARDIZED_JUMP = 0.35
REAL_ANCHORED_REGIME_MINIMUM_LOCAL_SSE_REDUCTION = 0.05
REAL_ANCHORED_REGIME_MINIMUM_STEP_OVER_RAMP_ADVANTAGE = 0.01
REAL_ANCHORED_REGIME_STABILITY_SCORE_FRACTION = 0.95
REAL_ANCHORED_REGIME_MAXIMUM_JOIN_STABILITY_WIDTH = 12
REAL_ANCHORED_FOUR_CAPABILITY_QUALIFICATION_POLICY_ID = (
    "cafe.real_anchored.decomposition_four.reference.v3"
)


DEFAULT_FOUR_CAPABILITY_QUALIFICATION_THRESHOLDS: dict[
    str, dict[str, float | int]
] = {
    "trend": {},
    "multi_seasonal": {},
    "time_varying_seasonality": {},
    "regime_switching": {},
}

_SHARED_FOUR_CAPABILITY_THRESHOLDS: dict[str, float | int] = {
    "minimum_cycles": REAL_ANCHORED_MINIMUM_CYCLES,
    "minimum_carrier_rms_ratio": REAL_ANCHORED_MINIMUM_CARRIER_RMS_RATIO,
    "minimum_carrier_power_share": (
        REAL_ANCHORED_MINIMUM_CARRIER_POWER_SHARE
    ),
    "maximum_secondary_periods": REAL_ANCHORED_MAXIMUM_SECONDARY_PERIODS,
    "minimum_secondary_power_share": (
        REAL_ANCHORED_MINIMUM_SECONDARY_POWER_SHARE
    ),
    "minimum_amplitude_cv": REAL_ANCHORED_MINIMUM_AMPLITUDE_CV,
    "minimum_envelope_power_share": (
        REAL_ANCHORED_MINIMUM_ENVELOPE_POWER_SHARE
    ),
    "minimum_modulation_carrier_strength": (
        REAL_ANCHORED_MINIMUM_MODULATION_CARRIER_STRENGTH
    ),
    "minimum_regime_segment_length": (
        REAL_ANCHORED_REGIME_MINIMUM_SEGMENT_LENGTH
    ),
    "regime_local_comparison_length": (
        REAL_ANCHORED_REGIME_LOCAL_COMPARISON_LENGTH
    ),
    "minimum_standardized_jump": (
        REAL_ANCHORED_REGIME_MINIMUM_STANDARDIZED_JUMP
    ),
    "minimum_local_sse_reduction": (
        REAL_ANCHORED_REGIME_MINIMUM_LOCAL_SSE_REDUCTION
    ),
    "minimum_step_over_ramp_advantage": (
        REAL_ANCHORED_REGIME_MINIMUM_STEP_OVER_RAMP_ADVANTAGE
    ),
    "regime_stability_score_fraction": (
        REAL_ANCHORED_REGIME_STABILITY_SCORE_FRACTION
    ),
    "maximum_join_stability_width": (
        REAL_ANCHORED_REGIME_MAXIMUM_JOIN_STABILITY_WIDTH
    ),
    "minimum_component_rms_ratio": (
        REAL_ANCHORED_MINIMUM_COMPONENT_RMS_RATIO
    ),
    "minimum_visible_component_rms_ratio": (
        REAL_ANCHORED_MINIMUM_VISIBLE_COMPONENT_RMS_RATIO
    ),
    "minimum_future_component_rms_ratio": (
        REAL_ANCHORED_MINIMUM_FUTURE_COMPONENT_RMS_RATIO
    ),
    "visible_context_length": REAL_ANCHORED_VISIBLE_CONTEXT_LENGTH,
    "fit_window": REAL_ANCHORED_FIT_WINDOW,
    "trend_window": 96,
    "trend_degree": 2,
    "harmonics_per_period": 1,
}


def default_four_capability_qualification_policy() -> dict[str, Any]:
    """Return the predeclared four-capability qualification thresholds."""

    thresholds = {
        capability_id: {
            **_SHARED_FOUR_CAPABILITY_THRESHOLDS,
            **specific,
        }
        for capability_id, specific in (
            DEFAULT_FOUR_CAPABILITY_QUALIFICATION_THRESHOLDS.items()
        )
    }
    payload: dict[str, Any] = {
        "schema_version": REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
        "qualification_policy_id": (
            REAL_ANCHORED_FOUR_CAPABILITY_QUALIFICATION_POLICY_ID
        ),
        "threshold_source": QUALIFICATION_THRESHOLD_SOURCE_POLICY,
        "qualification_thresholds": thresholds,
        "threshold_derivation": (
            "frozen_protocol_defaults_pending_or_replaced_by_"
            "disjoint_reference_bank"
        ),
    }
    payload["qualification_policy_sha256"] = protocol.json_sha256(payload)
    return payload


def _four_capability_qualification(
    capability_id: str,
    qualification_policy: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    policy = (
        default_four_capability_qualification_policy()
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
            "qualification thresholds must come from the independent "
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
        raw_thresholds = capability_policy.get("qualification_thresholds")
    else:
        policy_id = policy.get("qualification_policy_id")
        all_thresholds = policy.get("qualification_thresholds")
        if not isinstance(all_thresholds, Mapping):
            raise ValueError("qualification policy is missing threshold mappings")
        raw_thresholds = all_thresholds.get(capability_id)
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("qualification policy requires a stable non-empty id")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError(
            f"qualification policy has no thresholds for {capability_id}"
        )
    defaults = {
        **_SHARED_FOUR_CAPABILITY_THRESHOLDS,
        **DEFAULT_FOUR_CAPABILITY_QUALIFICATION_THRESHOLDS[capability_id],
    }
    thresholds: dict[str, float | int] = {}
    for name, default in defaults.items():
        value = raw_thresholds.get(name, default)
        if not isinstance(value, (int, float)) or not math.isfinite(
            float(value)
        ):
            raise ValueError(
                f"qualification threshold {name!r} must be finite"
            )
        thresholds[name] = value
    provenance = {
        "qualification_policy_id": policy_id,
        "qualification_policy_sha256": policy.get(
            "qualification_policy_sha256",
            protocol.json_sha256(policy),
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


def _period_visibility(
    history: np.ndarray,
    *,
    period: float,
) -> dict[str, float]:
    """Measure one declared Fourier component against detrended history."""

    values = _nonlinear_detrend(np.asarray(history, dtype=float))
    time = np.arange(values.size, dtype=float)
    phase = 2.0 * np.pi * time / float(period)
    design = np.column_stack((np.sin(phase), np.cos(phase)))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    component = design @ coefficients
    history_rms = max(float(np.sqrt(np.mean(values**2))), 1e-12)
    component_rms = float(np.sqrt(np.mean(component**2)))
    total_energy = float(np.sum(values**2))
    power_share = (
        float(np.clip(np.sum(component**2) / total_energy, 0.0, 1.0))
        if total_energy > 1e-24
        else 0.0
    )
    return {
        "carrier_rms": component_rms,
        "detrended_history_rms": history_rms,
        "carrier_rms_ratio": component_rms / history_rms,
        "carrier_power_share": power_share,
    }


def resolve_history_periods(
    history: np.ndarray,
    *,
    declared_carrier_period: float,
    maximum_secondary_periods: int = REAL_ANCHORED_MAXIMUM_SECONDARY_PERIODS,
    minimum_secondary_power_share: float = (
        REAL_ANCHORED_MINIMUM_SECONDARY_POWER_SHARE
    ),
    minimum_carrier_rms_ratio: float = (
        REAL_ANCHORED_MINIMUM_CARRIER_RMS_RATIO
    ),
    minimum_carrier_power_share: float = (
        REAL_ANCHORED_MINIMUM_CARRIER_POWER_SHARE
    ),
    minimum_cycles: float = REAL_ANCHORED_MINIMUM_CYCLES,
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
    minimum_cycles = float(minimum_cycles)
    if not math.isfinite(minimum_cycles) or minimum_cycles <= 0.0:
        raise ValueError("minimum_cycles must be finite and positive")
    candidates = _spectral_candidates(values, minimum_cycles=minimum_cycles)
    maximum_period = values.size / minimum_cycles
    maximum_secondary_periods = int(maximum_secondary_periods)
    if maximum_secondary_periods < 0:
        raise ValueError("maximum_secondary_periods must be non-negative")
    for name, threshold in (
        ("minimum_secondary_power_share", minimum_secondary_power_share),
        ("minimum_carrier_rms_ratio", minimum_carrier_rms_ratio),
        ("minimum_carrier_power_share", minimum_carrier_power_share),
    ):
        if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    declared = float(declared_carrier_period)
    declared_visibility: dict[str, float] | None = None
    if 2.0 <= declared <= maximum_period:
        declared_visibility = _period_visibility(values, period=declared)
    declared_visible = bool(
        declared_visibility is not None
        and declared_visibility["carrier_rms_ratio"]
        >= float(minimum_carrier_rms_ratio)
        and declared_visibility["carrier_power_share"]
        >= float(minimum_carrier_power_share)
    )
    if declared_visible:
        carrier = declared
        carrier_source = "visible_calibration_feature_period"
        carrier_visibility = dict(declared_visibility or {})
    else:
        carrier = float("nan")
        carrier_visibility = {}
        carrier_source = "history_spectral_peak_fallback"
        for candidate in candidates:
            candidate_period = float(candidate["period"])
            visibility = _period_visibility(
                values,
                period=candidate_period,
            )
            if (
                visibility["carrier_rms_ratio"]
                >= float(minimum_carrier_rms_ratio)
                and visibility["carrier_power_share"]
                >= float(minimum_carrier_power_share)
            ):
                carrier = candidate_period
                carrier_visibility = visibility
                break
        if not math.isfinite(carrier):
            raise ValueError(
                "no carrier period passed the frozen visibility gates"
            )
    secondary: list[float] = []
    selected_rows: list[dict[str, float]] = []
    secondary_candidate_pool: list[dict[str, float]] = []
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
        secondary_candidate_pool.append(dict(row))
        if len(secondary) >= maximum_secondary_periods:
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
    return {
        "schema_version": "cafe.real_anchored_period_resolution.v2",
        "history_only": True,
        "history_length": int(values.size),
        "minimum_cycles": minimum_cycles,
        "carrier_period": carrier,
        "carrier_source": carrier_source,
        "declared_carrier_period": declared,
        "declared_carrier_visibility": declared_visibility,
        "carrier_visibility_passed": True,
        **carrier_visibility,
        "secondary_periods": secondary,
        "secondary_peaks": selected_rows,
        "secondary_candidate_pool": secondary_candidate_pool,
        "minimum_secondary_power_share": minimum_secondary_power_share,
        "maximum_secondary_periods": maximum_secondary_periods,
        "minimum_carrier_rms_ratio": minimum_carrier_rms_ratio,
        "minimum_carrier_power_share": minimum_carrier_power_share,
        "candidate_count": len(candidates),
    }


def _remove_trend_and_secondary(
    history: np.ndarray,
    *,
    secondary_periods: Sequence[float],
) -> np.ndarray:
    """Remove non-carrier nuisance while leaving carrier/AM signal intact."""

    values = np.asarray(history, dtype=float)
    scaled_time = np.linspace(-1.0, 1.0, values.size)
    absolute_time = np.arange(values.size, dtype=float)
    columns = [np.ones(values.size), scaled_time, scaled_time**2]
    for period in map(float, secondary_periods):
        phase = 2.0 * np.pi * absolute_time / period
        columns.extend((np.sin(phase), np.cos(phase)))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    # Keep the carrier-bearing residual, including its slow AM envelope.
    return values - design @ coefficients


def resolve_modulation_period(
    history: np.ndarray,
    *,
    carrier_period: float,
    secondary_periods: Sequence[float] = (),
    minimum_amplitude_cv: float = REAL_ANCHORED_MINIMUM_AMPLITUDE_CV,
    minimum_envelope_power_share: float = (
        REAL_ANCHORED_MINIMUM_ENVELOPE_POWER_SHARE
    ),
    minimum_carrier_strength: float = (
        REAL_ANCHORED_MINIMUM_MODULATION_CARRIER_STRENGTH
    ),
    minimum_cycles: float = REAL_ANCHORED_MINIMUM_CYCLES,
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
    for name, threshold in (
        ("minimum_amplitude_cv", minimum_amplitude_cv),
        ("minimum_envelope_power_share", minimum_envelope_power_share),
        ("minimum_carrier_strength", minimum_carrier_strength),
    ):
        if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    minimum_cycles = float(minimum_cycles)
    if not math.isfinite(minimum_cycles) or minimum_cycles <= 0.0:
        raise ValueError("minimum_cycles must be finite and positive")
    base = {
        "schema_version": "cafe.real_anchored_modulation_resolution.v2",
        "history_only": True,
        "carrier_period": float(carrier_period),
        "secondary_periods_fixed_as_nuisance": [
            float(value) for value in secondary_periods
        ],
        "modulation_basis": TIME_VARYING_SEASONALITY_BASIS_POLICY,
        "minimum_amplitude_cv": minimum_amplitude_cv,
        "minimum_envelope_power_share": minimum_envelope_power_share,
        "minimum_carrier_strength": minimum_carrier_strength,
        "minimum_cycles": minimum_cycles,
    }
    period_steps = int(round(float(carrier_period)))
    if period_steps < 4:
        return {
            **base,
            "available": False,
            "unavailable_reason": "carrier_period_too_short_for_envelope",
            "modulation_period": None,
        }
    cycle_count = values.size // period_steps
    minimum_cycle_observations = 2 * int(math.ceil(minimum_cycles)) + 1
    if cycle_count < minimum_cycle_observations:
        return {
            **base,
            "available": False,
            "unavailable_reason": "insufficient_carrier_cycles_for_modulation",
            "modulation_period": None,
            "carrier_cycle_count": cycle_count,
        }
    detrended = _remove_trend_and_secondary(
        values,
        secondary_periods=secondary_periods,
    )
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
        for index in range(int(math.ceil(minimum_cycles)), envelope_power.size)
        if cycle_count / float(index) > 1.0
    ]
    total_power = float(np.sum(envelope_power[1:]))
    if not admissible_bins or total_power <= 1e-12:
        return {
            **base,
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
    if carrier_strength < minimum_carrier_strength:
        reason = "carrier_component_too_weak_for_modulation"
    elif amplitude_cv < minimum_amplitude_cv:
        reason = "carrier_amplitude_variation_too_weak"
    elif power_share < minimum_envelope_power_share:
        reason = "amplitude_envelope_peak_too_diffuse"
    elif modulation_period <= float(carrier_period):
        reason = "modulation_not_slower_than_carrier"
    elif values.size / modulation_period < minimum_cycles:
        reason = "insufficient_complete_modulation_cycles"
    return {
        **base,
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
    }


def _modulation_sideband_collision(
    period: float,
    *,
    carrier_period: float,
    modulation_period: float,
    history_length: int,
    harmonics_per_period: int = 1,
) -> bool:
    frequency = 1.0 / float(period)
    carrier_frequency = 1.0 / float(carrier_period)
    modulation_frequency = 1.0 / float(modulation_period)
    resolution = 1.5 / float(history_length)
    return any(
        abs(
            frequency
            - (harmonic * carrier_frequency + sign * modulation_frequency)
        )
        <= resolution
        for harmonic in range(1, int(harmonics_per_period) + 1)
        for sign in (-1.0, 1.0)
    )


def _resolve_spectral_component_ownership(
    period_resolution: Mapping[str, Any],
    modulation_resolution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assign AM sideband bins to AM before choosing secondary periods."""

    resolved = dict(period_resolution)
    carrier_period = float(resolved["carrier_period"])
    history_length = int(resolved["history_length"])
    maximum_secondary = int(resolved["maximum_secondary_periods"])
    modulation_period = (
        float(modulation_resolution["modulation_period"])
        if modulation_resolution.get("available") is True
        else None
    )
    pool = [
        dict(row)
        for row in resolved.get(
            "secondary_candidate_pool",
            resolved.get("secondary_peaks", []),
        )
    ]
    selected_periods: list[float] = []
    selected_rows: list[dict[str, float]] = []
    assigned_sidebands: list[dict[str, float]] = []
    for row in pool:
        period = float(row["period"])
        if (
            modulation_period is not None
            and _modulation_sideband_collision(
                period,
                carrier_period=carrier_period,
                modulation_period=modulation_period,
                history_length=history_length,
            )
        ):
            assigned_sidebands.append(dict(row))
            continue
        if len(selected_periods) >= maximum_secondary:
            continue
        if any(
            _frequency_collision(
                period,
                prior,
                history_length=history_length,
            )
            for prior in selected_periods
        ):
            continue
        selected_periods.append(period)
        selected_rows.append(dict(row))
    resolved["secondary_periods"] = selected_periods
    resolved["secondary_peaks"] = selected_rows
    resolved["spectral_component_ownership"] = (
        "carrier_then_symmetric_am_then_independent_secondary_v1"
    )
    resolved["am_sideband_owned_peaks"] = assigned_sidebands
    ownership = {
        "schema_version": "cafe.real_anchored_component_ownership.v1",
        "history_only": True,
        "policy": "shared_background_joint_design_v1",
        "priority": [
            "carrier_and_harmonics",
            "symmetric_carrier_amplitude_modulation",
            "independent_secondary_periods",
            "regime_level_shift",
            "trend_nonlinearity",
            "residual",
        ],
        "carrier_period": carrier_period,
        "modulation_period": modulation_period,
        "secondary_periods": selected_periods,
        "am_sideband_owned_peak_count": len(assigned_sidebands),
        "modulation_basis": TIME_VARYING_SEASONALITY_BASIS_POLICY,
    }
    return resolved, ownership


def _nuisance_residual(
    history: np.ndarray,
    *,
    carrier_period: float,
    secondary_periods: Sequence[float],
    modulation_period: float | None = None,
) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    time = np.linspace(-1.0, 1.0, values.size)
    columns = [np.ones(values.size), time, time**2]
    absolute_time = np.arange(values.size, dtype=float)
    for period in (float(carrier_period), *map(float, secondary_periods)):
        phase = 2.0 * np.pi * absolute_time / period
        columns.extend((np.sin(phase), np.cos(phase)))
    preliminary_design = np.column_stack(columns)
    preliminary_coefficients, *_ = np.linalg.lstsq(
        preliminary_design,
        values,
        rcond=None,
    )
    if modulation_period is not None:
        # The carrier occupies columns 3/4 after the quadratic trend.  Freeze
        # that history-only phase and remove only its constrained AM product
        # subspace; do not introduce four free sidebands.
        carrier_phase = math.atan2(
            float(preliminary_coefficients[4]),
            float(preliminary_coefficients[3]),
        )
        carrier_wave = np.sin(
            2.0 * np.pi * absolute_time / float(carrier_period)
            + carrier_phase
        )
        slow_phase = (
            2.0 * np.pi * absolute_time / float(modulation_period)
        )
        columns.extend(
            (
                carrier_wave * np.cos(slow_phase),
                carrier_wave * np.sin(slow_phase),
            )
        )
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def resolve_regime_joinpoint(
    history: np.ndarray,
    *,
    carrier_period: float,
    secondary_periods: Sequence[float],
    modulation_period: float | None = None,
    visible_context_length: int = REAL_ANCHORED_VISIBLE_CONTEXT_LENGTH,
    minimum_segment_length: int = (
        REAL_ANCHORED_REGIME_MINIMUM_SEGMENT_LENGTH
    ),
    local_comparison_length: int = (
        REAL_ANCHORED_REGIME_LOCAL_COMPARISON_LENGTH
    ),
    minimum_standardized_jump: float = (
        REAL_ANCHORED_REGIME_MINIMUM_STANDARDIZED_JUMP
    ),
    minimum_local_sse_reduction: float = (
        REAL_ANCHORED_REGIME_MINIMUM_LOCAL_SSE_REDUCTION
    ),
    minimum_step_over_ramp_advantage: float = (
        REAL_ANCHORED_REGIME_MINIMUM_STEP_OVER_RAMP_ADVANTAGE
    ),
    stability_score_fraction: float = (
        REAL_ANCHORED_REGIME_STABILITY_SCORE_FRACTION
    ),
    maximum_join_stability_width: int = (
        REAL_ANCHORED_REGIME_MAXIMUM_JOIN_STABILITY_WIDTH
    ),
) -> dict[str, Any]:
    """Detect a recent abrupt, stable level shift without future data."""

    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("regime resolution requires one finite history")
    visible_context_length = int(visible_context_length)
    minimum_segment_length = int(minimum_segment_length)
    local_comparison_length = int(local_comparison_length)
    maximum_join_stability_width = int(maximum_join_stability_width)
    if not 2 * minimum_segment_length <= visible_context_length <= values.size:
        raise ValueError(
            "visible_context_length must hold two complete regime segments"
        )
    if local_comparison_length < minimum_segment_length:
        raise ValueError(
            "local_comparison_length must cover one minimum segment"
        )
    if maximum_join_stability_width < 0:
        raise ValueError("maximum_join_stability_width must be non-negative")
    for name, threshold in (
        ("minimum_standardized_jump", minimum_standardized_jump),
        ("minimum_local_sse_reduction", minimum_local_sse_reduction),
        (
            "minimum_step_over_ramp_advantage",
            minimum_step_over_ramp_advantage,
        ),
    ):
        if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(float(stability_score_fraction)) or not (
        0.0 < float(stability_score_fraction) <= 1.0
    ):
        raise ValueError("stability_score_fraction must lie in (0, 1]")
    residual = _nuisance_residual(
        values,
        carrier_period=carrier_period,
        secondary_periods=secondary_periods,
        modulation_period=modulation_period,
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
        local_time = np.arange(lower, upper, dtype=float) - float(join_index)
        ramp_design = np.column_stack(
            (
                np.ones(combined.size),
                local_time,
                np.maximum(local_time, 0.0),
            )
        )
        ramp_coefficients, *_ = np.linalg.lstsq(
            ramp_design,
            combined,
            rcond=None,
        )
        ramp_sse = float(
            np.sum((combined - ramp_design @ ramp_coefficients) ** 2)
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
                "step_sse": alternative_sse,
                "continuous_ramp_sse": ramp_sse,
                "step_over_ramp_sse_advantage": (
                    (ramp_sse - alternative_sse) / null_sse
                    if null_sse > 1e-12
                    else 0.0
                ),
                "selection_score": score,
            }
        )
    if not candidates:
        return {
            "schema_version": "cafe.real_anchored_regime_resolution.v2",
            "history_only": True,
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
    near_optimal = [
        int(row["join_index"])
        for row in candidates
        if float(row["selection_score"])
        >= float(stability_score_fraction)
        * float(selected["selection_score"])
    ]
    join_stability_width = max(near_optimal) - min(near_optimal)
    reason: str | None = None
    if float(selected["standardized_jump"]) < minimum_standardized_jump:
        reason = "regime_level_shift_too_weak"
    elif (
        float(selected["local_sse_reduction"])
        < minimum_local_sse_reduction
    ):
        reason = "regime_joinpoint_sse_reduction_too_weak"
    elif (
        float(selected["step_over_ramp_sse_advantage"])
        < minimum_step_over_ramp_advantage
    ):
        reason = "continuous_ramp_preferred_over_level_step"
    elif join_stability_width > maximum_join_stability_width:
        reason = "regime_joinpoint_not_locally_stable"
    return {
        "schema_version": "cafe.real_anchored_regime_resolution.v2",
        "history_only": True,
        "available": reason is None,
        "unavailable_reason": reason,
        "regime_join_index": (
            int(selected["join_index"]) if reason is None else None
        ),
        "candidate_range": [candidate_start, candidate_stop],
        "candidate_count": len(candidates),
        "visible_context_length": int(visible_context_length),
        "modulation_period_fixed_as_nuisance": modulation_period,
        "minimum_segment_length": minimum_segment_length,
        "local_comparison_length": local_comparison_length,
        "minimum_standardized_jump": minimum_standardized_jump,
        "minimum_local_sse_reduction": minimum_local_sse_reduction,
        "minimum_step_over_ramp_advantage": (
            minimum_step_over_ramp_advantage
        ),
        "stability_score_fraction": stability_score_fraction,
        "maximum_join_stability_width": maximum_join_stability_width,
        "near_optimal_join_range": [
            min(near_optimal),
            max(near_optimal),
        ],
        "join_stability_width": join_stability_width,
        **selected,
    }


def _resolve_background_structural_components(
    history: np.ndarray,
    *,
    declared_carrier_period: float,
    thresholds: Mapping[str, float | int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve one shared ownership/decomposition state for a background."""

    raw_period_resolution = resolve_history_periods(
        history,
        declared_carrier_period=declared_carrier_period,
        maximum_secondary_periods=int(
            thresholds["maximum_secondary_periods"]
        ),
        minimum_secondary_power_share=float(
            thresholds["minimum_secondary_power_share"]
        ),
        minimum_carrier_rms_ratio=float(
            thresholds["minimum_carrier_rms_ratio"]
        ),
        minimum_carrier_power_share=float(
            thresholds["minimum_carrier_power_share"]
        ),
        minimum_cycles=float(thresholds["minimum_cycles"]),
    )
    carrier_period = float(raw_period_resolution["carrier_period"])
    modulation_resolution = resolve_modulation_period(
        history,
        carrier_period=carrier_period,
        secondary_periods=(),
        minimum_amplitude_cv=float(thresholds["minimum_amplitude_cv"]),
        minimum_envelope_power_share=float(
            thresholds["minimum_envelope_power_share"]
        ),
        minimum_carrier_strength=float(
            thresholds["minimum_modulation_carrier_strength"]
        ),
        minimum_cycles=float(thresholds["minimum_cycles"]),
    )
    initial_modulation_resolution = dict(modulation_resolution)
    period_resolution: dict[str, Any]
    ownership: dict[str, Any]
    ownership_stable = False
    for iteration in range(1, 6):
        period_resolution, ownership = _resolve_spectral_component_ownership(
            raw_period_resolution,
            modulation_resolution,
        )
        secondary_periods = tuple(
            float(value)
            for value in period_resolution["secondary_periods"]
        )
        refined_modulation = resolve_modulation_period(
            history,
            carrier_period=carrier_period,
            secondary_periods=secondary_periods,
            minimum_amplitude_cv=float(thresholds["minimum_amplitude_cv"]),
            minimum_envelope_power_share=float(
                thresholds["minimum_envelope_power_share"]
            ),
            minimum_carrier_strength=float(
                thresholds["minimum_modulation_carrier_strength"]
            ),
            minimum_cycles=float(thresholds["minimum_cycles"]),
        )
        refined_periods, refined_ownership = (
            _resolve_spectral_component_ownership(
                raw_period_resolution,
                refined_modulation,
            )
        )
        if tuple(refined_periods["secondary_periods"]) == tuple(
            period_resolution["secondary_periods"]
        ):
            period_resolution = refined_periods
            ownership = refined_ownership
            modulation_resolution = refined_modulation
            ownership_stable = True
            break
        modulation_resolution = refined_modulation
    if not ownership_stable:
        modulation_resolution = {
            **modulation_resolution,
            "available": False,
            "unavailable_reason": "spectral_component_ownership_not_stable",
            "modulation_period": None,
        }
        period_resolution, ownership = _resolve_spectral_component_ownership(
            raw_period_resolution,
            modulation_resolution,
        )
    ownership["fixed_point_iterations"] = iteration
    ownership["fixed_point_converged"] = ownership_stable
    modulation_resolution["initial_history_only_resolution"] = (
        initial_modulation_resolution
    )
    modulation_period = (
        float(modulation_resolution["modulation_period"])
        if modulation_resolution.get("available") is True
        else None
    )
    regime_resolution = resolve_regime_joinpoint(
        history,
        carrier_period=carrier_period,
        secondary_periods=tuple(period_resolution["secondary_periods"]),
        modulation_period=modulation_period,
        visible_context_length=int(thresholds["visible_context_length"]),
        minimum_segment_length=int(
            thresholds["minimum_regime_segment_length"]
        ),
        local_comparison_length=int(
            thresholds["regime_local_comparison_length"]
        ),
        minimum_standardized_jump=float(
            thresholds["minimum_standardized_jump"]
        ),
        minimum_local_sse_reduction=float(
            thresholds["minimum_local_sse_reduction"]
        ),
        minimum_step_over_ramp_advantage=float(
            thresholds["minimum_step_over_ramp_advantage"]
        ),
        stability_score_fraction=float(
            thresholds["regime_stability_score_fraction"]
        ),
        maximum_join_stability_width=int(
            thresholds["maximum_join_stability_width"]
        ),
    )
    return (
        period_resolution,
        modulation_resolution,
        regime_resolution,
        ownership,
    )


def _additive_dose_artifacts(
    *,
    background_id: str,
    capability_id: str,
    history: np.ndarray,
    fitted: Mapping[str, Any],
    dose_calibration: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]] | None,
    str | None,
]:
    """Build history-only dose evidence and replay frozen evaluation doses."""

    history_rms = fitted.get("controlled_component_visible_history_rms")
    future_rms = fitted.get("controlled_component_future_rms")
    raw_decomposition = fitted.get("decomposition_contract")
    if not isinstance(raw_decomposition, Mapping):
        return None, dose_calibration, None, None
    raw_scales = raw_decomposition.get("normalization_scale_by_target")
    if not isinstance(raw_scales, list) or len(raw_scales) != 1:
        raise ValueError("additive contract has invalid normalization scales")
    normalization_scale = float(raw_scales[0])
    evidence = (
        additive_dose_reference(
            capability_id=capability_id,
            background_id=background_id,
            unit_gain_history_separation=(
                float(history_rms) / normalization_scale
            ),
            unit_gain_future_separation=(
                float(future_rms) / normalization_scale
            ),
            affected_channel_indices=(0,),
        )
        if (
            isinstance(history_rms, (int, float))
            and isinstance(future_rms, (int, float))
            and float(history_rms) > 0.0
            and float(future_rms) > 0.0
        )
        else None
    )
    if dose_calibration is None or fitted.get("available") is not True:
        return evidence, dose_calibration, None, None
    if dose_calibration.get("status") != "available":
        return evidence, dose_calibration, None, "dose_calibration_unavailable"
    if evidence is None:
        return evidence, dose_calibration, None, "dose_reference_unavailable"
    try:
        resolved_calibration = resolve_contract_dose_calibration(
            dose_calibration,
            evidence,
        )
    except ValueError:
        return (
            evidence,
            dose_calibration,
            None,
            "contract_source_distance_mapping_unavailable",
        )

    source_baseline = np.concatenate(
        (np.asarray(history, dtype=float), np.zeros(protocol.HORIZON))
    )
    gates: list[dict[str, Any]] = []
    previous_delta: np.ndarray | None = None
    for dose_index, alpha in enumerate(
        resolved_calibration["applied_alpha_grid"],
        start=1,
    ):
        augmented, _metadata = apply_real_anchored_contract(
            source_baseline,
            fitted,
            alpha=float(alpha),
            context_length=(
                protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
            ),
        )
        delta = np.asarray(augmented, dtype=float).reshape(-1) - source_baseline
        gates.append(
            paired_minimum_separation_gate(
                delta,
                context_length=(
                    protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                ),
                dose_index=dose_index,
                dose_calibration=resolved_calibration,
                affected_channel_indices=(0,),
                scale_by_channel=(normalization_scale,),
                previous_delta=previous_delta,
            )
        )
        previous_delta = delta
    if not all(bool(gate["accepted"]) for gate in gates):
        return (
            evidence,
            resolved_calibration,
            gates,
            "treatment_source_distance_gate_failed",
        )
    return evidence, resolved_calibration, gates, None


def fit_background_capability_contracts(
    backgrounds: Sequence[dict[str, Any]],
    *,
    capability_ids: Iterable[str],
    qualification_policy: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit per-background contracts and freeze explicit availability rows."""

    requested = tuple(dict.fromkeys(str(value) for value in capability_ids))
    four_capabilities = tuple(
        capability_id
        for capability_id in requested
        if capability_id in DEFAULT_FOUR_CAPABILITY_QUALIFICATION_THRESHOLDS
    )
    four_qualification = {
        capability_id: _four_capability_qualification(
            capability_id,
            qualification_policy,
        )
        for capability_id in four_capabilities
    }
    if four_qualification:
        first_thresholds = dict(
            next(iter(four_qualification.values()))[1]
        )
        structural_threshold_names = (
            "minimum_cycles",
            "minimum_carrier_rms_ratio",
            "minimum_carrier_power_share",
            "maximum_secondary_periods",
            "minimum_secondary_power_share",
            "minimum_amplitude_cv",
            "minimum_envelope_power_share",
            "minimum_modulation_carrier_strength",
            "visible_context_length",
            "minimum_regime_segment_length",
            "regime_local_comparison_length",
            "minimum_standardized_jump",
            "minimum_local_sse_reduction",
            "minimum_step_over_ramp_advantage",
            "regime_stability_score_fraction",
            "maximum_join_stability_width",
            "fit_window",
            "trend_window",
            "trend_degree",
            "harmonics_per_period",
        )
        for _provenance, thresholds in four_qualification.values():
            if any(
                float(thresholds[name]) != float(first_thresholds[name])
                for name in structural_threshold_names
            ):
                raise ValueError(
                    "four-capability policies disagree on shared structural "
                    "decomposition thresholds"
                )
        structural_thresholds = first_thresholds
    else:
        structural_thresholds = {
            **_SHARED_FOUR_CAPABILITY_THRESHOLDS,
        }
    dynamic_qualification = {
        capability_id: dynamic_qualification_provenance(
            capability_id,
            qualification_policy,
        )
        for capability_id in requested
        if capability_id in REAL_PATH_DYNAMIC_CAPABILITIES
    }
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for background in backgrounds:
        history = _history_1d(background)
        try:
            (
                period_resolution,
                modulation_resolution,
                regime_resolution,
                component_ownership,
            ) = _resolve_background_structural_components(
                history,
                declared_carrier_period=float(background["feature_period"]),
                thresholds=structural_thresholds,
            )
        except ValueError as error:
            period_resolution = None
            modulation_resolution = None
            regime_resolution = None
            component_ownership = None
            period_error = str(error)
        else:
            period_error = None
        for capability_id in requested:
            if capability_id in four_qualification:
                qualification, thresholds = four_qualification[capability_id]
            elif capability_id in dynamic_qualification:
                qualification = dynamic_qualification[capability_id]
                thresholds = dict(
                    qualification["qualification_thresholds"]
                )
            else:
                qualification = {
                    "qualification_policy_id": (
                        "cafe.real_anchored.unsupported.no_thresholds.v1"
                    ),
                    "qualification_policy_sha256": None,
                    "qualification_threshold_source": (
                        QUALIFICATION_THRESHOLD_SOURCE_POLICY
                    ),
                    "qualification_thresholds": {},
                }
                thresholds = {}
            base = {
                "schema_version": (
                    "cafe.real_anchored_background_capability.v4"
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
                "component_ownership": component_ownership,
                "shared_decomposition_policy": (
                    "one_background_one_joint_design_all_four_capabilities_v1"
                ),
                "modulation_basis": TIME_VARYING_SEASONALITY_BASIS_POLICY,
                **qualification,
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
                        "local SSE, step-vs-ramp, and join-stability gates"
                    ),
                    "contract": None,
                }
            else:
                if capability_id in REAL_PATH_DYNAMIC_CAPABILITIES:
                    effective_period = int(
                        background["mase_scale_effective_period_by_target"][0]
                    )
                    fitted = fit_real_path_dynamic_contract(
                        history,
                        capability_id=capability_id,
                        background_id=str(background["background_id"]),
                        carrier_period=float(
                            period_resolution["carrier_period"]
                        ),
                        secondary_periods=tuple(
                            ()
                            if capability_id == "predictable_intermittency"
                            else period_resolution["secondary_periods"]
                        ),
                        reference_history=history[
                            -protocol.REAL_ANCHORED_CONTEXT_LENGTH:
                        ],
                        mase_period=int(background["mase_period"]),
                        mase_scale=float(background["mase_scale"]),
                        mase_effective_period=effective_period,
                        mase_scale_source=(
                            "seasonal_history"
                            if effective_period == int(background["mase_period"])
                            else (
                                "lag_one_history_fallback"
                                if effective_period == 1
                                else "normalization_scale_constant_fallback"
                            )
                        ),
                        qualification_policy=qualification_policy,
                    )
                else:
                    modulation_period = (
                        float(modulation_resolution["modulation_period"])
                        if modulation_resolution is not None
                        and modulation_resolution.get("available") is True
                        else None
                    )
                    regime_join_index = (
                        int(regime_resolution["regime_join_index"])
                        if regime_resolution is not None
                        and regime_resolution.get("available") is True
                        else None
                    )
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
                        fit_window=int(thresholds["fit_window"]),
                        trend_window=int(thresholds["trend_window"]),
                        trend_degree=int(thresholds["trend_degree"]),
                        harmonics_per_period=int(
                            thresholds["harmonics_per_period"]
                        ),
                        # All detected nuisance is shared across all four
                        # capability contracts.  Only the intervention slice
                        # changes; the joint fit and ownership do not.
                        modulation_period=modulation_period,
                        regime_join_index=regime_join_index,
                        minimum_regime_segment_length=int(
                            thresholds["minimum_regime_segment_length"]
                        ),
                        minimum_cycles=float(thresholds["minimum_cycles"]),
                        mase_period=int(background["mase_period"]),
                        minimum_component_rms_ratio=(
                            float(thresholds["minimum_component_rms_ratio"])
                        ),
                        minimum_visible_component_rms_ratio=(
                            float(
                                thresholds[
                                    "minimum_visible_component_rms_ratio"
                                ]
                            )
                        ),
                        visible_context_length=int(
                            thresholds["visible_context_length"]
                        ),
                        minimum_future_component_rms_ratio=(
                            float(
                                thresholds[
                                    "minimum_future_component_rms_ratio"
                                ]
                            )
                        ),
                        reference_history=history[
                            -protocol.REAL_ANCHORED_CONTEXT_LENGTH:
                        ],
                        qualification_policy_id=str(
                            qualification["qualification_policy_id"]
                        ),
                        qualification_policy_sha256=(
                            None
                            if qualification.get(
                                "qualification_policy_sha256"
                            )
                            is None
                            else str(
                                qualification[
                                    "qualification_policy_sha256"
                                ]
                            )
                        ),
                        qualification_threshold_source=str(
                            qualification[
                                "qualification_threshold_source"
                            ]
                        ),
                        qualification_thresholds=thresholds,
                    )
                dose_calibration = qualification.get("dose_calibration")
                if capability_id in REAL_PATH_DYNAMIC_CAPABILITIES:
                    dose_calibration = fitted.get(
                        "dose_calibration",
                        dose_calibration,
                    )
                    dose_reference = fitted.get("dose_design_reference")
                    paired_separation_gates = fitted.get(
                        "paired_minimum_separation_gate"
                    )
                    dose_failure = None
                else:
                    (
                        dose_reference,
                        dose_calibration,
                        paired_separation_gates,
                        dose_failure,
                    ) = _additive_dose_artifacts(
                        background_id=str(background["background_id"]),
                        capability_id=capability_id,
                        history=history,
                        fitted=fitted,
                        dose_calibration=(
                            dose_calibration
                            if isinstance(dose_calibration, Mapping)
                            else None
                        ),
                    )
                fitted_available = bool(fitted["available"])
                available = bool(fitted_available and dose_failure is None)
                reason = (
                    fitted.get("unavailable_reason")
                    if not fitted_available
                    else dose_failure
                )
                row = {
                    **base,
                    "available": available,
                    "unavailable_reason": reason,
                    "unavailable_detail": (
                        fitted.get("unavailable_detail")
                        if not fitted_available
                        else dose_failure
                    ),
                    "dose_design_reference": dose_reference,
                    "dose_calibration": dose_calibration,
                    "paired_minimum_separation_gate": (
                        paired_separation_gates
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
                    "controlled_component_visible_history_rms": fitted.get(
                        "controlled_component_visible_history_rms"
                    ),
                    "controlled_component_visible_context_length": fitted.get(
                        "controlled_component_visible_context_length"
                    ),
                    "minimum_history_component_rms": fitted.get(
                        "minimum_history_component_rms"
                    ),
                    "minimum_future_component_rms": fitted.get(
                        "minimum_future_component_rms"
                    ),
                    "minimum_visible_history_component_rms": fitted.get(
                        "minimum_visible_history_component_rms"
                    ),
                    "minimum_component_rms_ratio": fitted.get(
                        "minimum_component_rms_ratio"
                    ),
                    "minimum_future_component_rms_ratio": fitted.get(
                        "minimum_future_component_rms_ratio"
                    ),
                    "minimum_visible_component_rms_ratio": fitted.get(
                        "minimum_visible_component_rms_ratio"
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
        try:
            capability_strength_grid = list(
                dose_targets(capability_id)["strength_grid"]
            )
        except ValueError:
            capability_strength_grid = []
        selected = [
            row
            for row in contract_rows
            if str(row["capability_id"]) == capability_id
        ]
        available_dose_calibrations = [
            row["dose_calibration"]
            for row in selected
            if isinstance(row.get("dose_calibration"), Mapping)
            and row["dose_calibration"].get("status") == "available"
            and row.get("available") is True
            and len(row["dose_calibration"].get("applied_alpha_grid", ()))
            == len(capability_strength_grid)
        ]
        dose_policy_hashes = {
            str(
                calibration.get(
                    "dose_policy_sha256",
                    calibration.get("policy_sha256", ""),
                )
            )
            for calibration in available_dose_calibrations
        }
        if len(dose_policy_hashes) > 1:
            raise ValueError(
                f"{capability_id} rows disagree on frozen dose calibration"
            )
        applied_grids = [
            [float(value) for value in calibration["applied_alpha_grid"]]
            for calibration in available_dose_calibrations
        ]
        supported_applied_alphas = (
            applied_grids[0]
            if applied_grids
            and all(grid == applied_grids[0] for grid in applied_grids)
            else []
        )
        applied_alpha_ranges = [
            {
                "minimum": min(grid[index] for grid in applied_grids),
                "maximum": max(grid[index] for grid in applied_grids),
            }
            for index in range(len(capability_strength_grid))
        ] if applied_grids else []
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
            values = [
                float(row[field])
                for row in gate_rows
                if isinstance(row.get(field), (int, float))
                and math.isfinite(float(row[field]))
            ]
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
        visible_ratios = sorted(
            {
                float(row["minimum_visible_component_rms_ratio"])
                for row in gate_rows
                if isinstance(
                    row.get("minimum_visible_component_rms_ratio"),
                    (int, float),
                )
                and math.isfinite(
                    float(row["minimum_visible_component_rms_ratio"])
                )
            }
        )
        visible_context_lengths = sorted(
            {
                int(row["controlled_component_visible_context_length"])
                for row in gate_rows
                if isinstance(
                    row.get("controlled_component_visible_context_length"),
                    int,
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
                "dose_parameter": "canonical_strength_lambda",
                "supported_dose_values": capability_strength_grid,
                "supported_applied_alpha_values": supported_applied_alphas,
                "applied_alpha_range_by_dose_level": applied_alpha_ranges,
                "applied_alpha_scope": "contract_specific_history_only",
                "controlled_component_rms_gate": {
                    "history_source": "history_fitted_component_l504",
                    "visible_history_source": (
                        "history_fitted_component_trailing_l168"
                    ),
                    "future_source": (
                        "analytic_history_fitted_component_extension"
                    ),
                    "future_horizon": protocol.HORIZON,
                    "evaluated_background_count": len(gate_rows),
                    "history_minimum_rms_ratios": history_ratios,
                    "visible_history_minimum_rms_ratios": visible_ratios,
                    "visible_context_lengths": visible_context_lengths,
                    "future_minimum_rms_ratios": future_ratios,
                    "history_rms_range": metric_range(
                        "controlled_component_history_rms"
                    ),
                    "future_rms_range": metric_range(
                        "controlled_component_future_rms"
                    ),
                    "visible_history_rms_range": metric_range(
                        "controlled_component_visible_history_rms"
                    ),
                    "history_threshold_range": metric_range(
                        "minimum_history_component_rms"
                    ),
                    "future_threshold_range": metric_range(
                        "minimum_future_component_rms"
                    ),
                    "visible_history_threshold_range": metric_range(
                        "minimum_visible_history_component_rms"
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
    schema_version = availability.get("schema_version")
    if (
        schema_version == REAL_ANCHORED_AVAILABILITY_SCHEMA
        and minimum != REAL_ANCHORED_MINIMUM_ELIGIBLE_BACKGROUNDS
    ):
        raise ValueError("real-anchored availability changed the formal-N gate")
    recomputed = build_availability(
        contract_rows,
        requested_capability_ids=requested,
        minimum_eligible_backgrounds=minimum,
    )
    blocked_capabilities = {
        str(value)
        for value in availability.get(
            "qualification_blocked_capabilities",
            [],
        )
    }
    for cell in recomputed["cells"]:
        if str(cell["capability_id"]) in blocked_capabilities:
            cell["reason_codes"] = sorted(
                {
                    *cell["reason_codes"],
                    "independent_reference_bank_unavailable",
                }
            )
    if schema_version == "cafe.real_anchored_availability.v1":
        # Pipeline v2 froze the same eligibility decision before the
        # fixed-L168 visibility audit was added.  Preserve that immutable
        # upstream contract by projecting the v2 diagnostic payload back to
        # its exact v1 shape; generation may consume it but never reinterpret
        # it as a v3 qualification result.
        for cell in recomputed["cells"]:
            gate = cell.get("controlled_component_rms_gate")
            if isinstance(gate, dict):
                for field in (
                    "visible_history_source",
                    "visible_history_minimum_rms_ratios",
                    "visible_context_lengths",
                    "visible_history_rms_range",
                    "visible_history_threshold_range",
                ):
                    gate.pop(field, None)
    elif schema_version != REAL_ANCHORED_AVAILABILITY_SCHEMA:
        raise ValueError("unsupported real-anchored availability schema")
    observed_cells = [dict(cell) for cell in cells]
    expected_cells = [dict(cell) for cell in recomputed["cells"]]
    dataset_id = availability.get("dataset_id")
    for cell in expected_cells:
        cell["dataset_id"] = dataset_id
    if observed_cells != expected_cells:
        raise ValueError(
            "real-anchored availability cells disagree with contracts"
        )
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
    canonical_strength: float,
    treatment_alpha: float,
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
    dose_calibration = contract_row.get("dose_calibration")
    if not isinstance(dose_calibration, Mapping):
        raise ValueError("generation row has no frozen dose calibration")
    gates = contract_row.get("paired_minimum_separation_gate")
    if not isinstance(gates, list) or len(gates) != len(
        dose_calibration["strength_grid"]
    ):
        raise ValueError("generation row has no paired separation gates")
    treatment_gate = metadata.get("paired_minimum_separation_gate")
    if treatment_gate is None:
        treatment_gate = gates[dose_index - 1]
    if not isinstance(treatment_gate, Mapping) or (
        treatment_gate.get("accepted") is not True
    ):
        raise ValueError("generation row failed paired minimum separation")
    row_gate = (
        {
            "status": "not_applicable",
            "accepted": None,
            "reason_code": "repeated_authentic_baseline_member",
            "dose_index": dose_index,
            "paired_treatment_gate_status": "passed",
            "dose_calibration_policy_sha256": dose_calibration.get(
                "dose_policy_sha256",
                dose_calibration["policy_sha256"],
            ),
        }
        if pair_member == 0
        else dict(treatment_gate)
    )
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
        "intensity_lambda": (
            0.0 if pair_member == 0 else canonical_strength
        ),
        "dose_index": dose_index,
        "dose_parameter": "canonical_strength_lambda",
        "dose_value": 0.0 if pair_member == 0 else canonical_strength,
        "baseline_dose_value": 0.0,
        "paired_treatment_strength": canonical_strength,
        "physical_dose_parameter": (
            "controlled_component_multiplier_alpha"
        ),
        "applied_alpha": alpha,
        "paired_treatment_applied_alpha": treatment_alpha,
        "dose_calibration_policy_sha256": dose_calibration.get(
            "dose_policy_sha256",
            dose_calibration["policy_sha256"],
        ),
        "contract_dose_calibration_sha256": dose_calibration[
            "policy_sha256"
        ],
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
            "policy": (
                "reference_frozen_contract_specific_history_solver_v2"
            ),
            "scope": "real_anchored_history_only_decomposition",
            "formal_seed_inverse": False,
            "sample_level_target_gate": True,
            "canonical_strength_grid": list(
                dose_calibration["strength_grid"]
            ),
            "selected_alphas": list(
                dose_calibration["applied_alpha_grid"]
            ),
            "history_target_grid": list(
                dose_calibration["history_target_grid"]
            ),
            "future_target_grid": list(
                dose_calibration["future_target_grid"]
            ),
            "dose_calibration_policy_sha256": dose_calibration.get(
                "dose_policy_sha256",
                dose_calibration["policy_sha256"],
            ),
        },
        "dose_calibration": dict(dose_calibration),
        "paired_minimum_separation_gate": row_gate,
        "realized_features": {},
        "sampled_generator_parameters": {
            "alpha": alpha,
            "canonical_strength": (
                0.0 if pair_member == 0 else canonical_strength
            ),
            "controlled_component": metadata["controlled_component"],
        },
        "parameter_mapping": {},
        "parameter_sampling": {
            "policy": "real_background_contract_deterministic_selection_v1",
            "background_id": str(background["background_id"]),
            "contract_sha256": str(metadata["contract_sha256"]),
        },
        "generation_metadata": dict(metadata),
        "evaluation_table": "real_anchored_counterfactual",
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


def _row_dose_grid(
    contract_row: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    capability_id = str(contract_row["capability_id"])
    calibration = contract_row.get("dose_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError(
            f"{capability_id} contract row has no frozen dose calibration"
        )
    validate_dose_calibration(calibration, capability_id=capability_id)
    if calibration.get("status") != "available":
        raise ValueError(
            f"{capability_id} contract row has unavailable dose calibration"
        )
    strengths = tuple(float(value) for value in calibration["strength_grid"])
    alphas = tuple(float(value) for value in calibration["applied_alpha_grid"])
    return tuple(zip(strengths, alphas, strict=True))


def iter_real_anchored_samples(
    backgrounds: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    capability_ids: Iterable[str],
    seed_indexes: Iterable[int],
    alphas: Sequence[float] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield pairs on each capability's reference-frozen physical grid."""

    if alphas is not None and tuple(float(value) for value in alphas) != (
        REAL_ANCHORED_ALPHAS
    ):
        raise ValueError(
            "global alpha overrides are unsupported; each capability uses "
            "its frozen applied_alpha_grid"
        )
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
            for dose_index, (
                canonical_strength,
                treatment_alpha,
            ) in enumerate(_row_dose_grid(contract_row), start=1):
                for pair_member, alpha in ((0, 1.0), (1, treatment_alpha)):
                    capability_contract = contract_row["contract"]
                    apply_contract = (
                        apply_real_path_dynamic_contract
                        if capability_contract.get("schema")
                        in {
                            REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
                            LEGACY_REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
                        }
                        else apply_real_anchored_contract
                    )
                    augmented, metadata = apply_contract(
                        source_baseline,
                        capability_contract,
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
                        canonical_strength=canonical_strength,
                        treatment_alpha=treatment_alpha,
                        alpha=alpha,
                        visible_target=visible_target,
                        visible_delta=visible_delta,
                        baseline_visible=baseline_visible,
                        metadata=metadata,
                    )


def iter_nonlinear_replay_sensitivity_samples(
    backgrounds: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    seed_indexes: Iterable[int],
    alphas: Sequence[float] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the history-residual-replay nonlinear auxiliary track."""

    if alphas is not None and tuple(float(value) for value in alphas) != (
        REAL_ANCHORED_ALPHAS
    ):
        raise ValueError(
            "nonlinear replay uses its capability-frozen applied_alpha_grid"
        )
    by_background = {
        str(background["background_id"]): background
        for background in backgrounds
    }
    assignments = real_anchored_assignments(
        contract_rows,
        capability_ids=("nonlinear_persistence",),
        seed_indexes=seed_indexes,
    )
    for seed_index, contract_row in assignments.get(
        "nonlinear_persistence", []
    ):
        background = by_background[str(contract_row["background_id"])]
        source_baseline = reconstruct_source_baseline(background)
        visible_start = (
            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
            - protocol.REAL_ANCHORED_CONTEXT_LENGTH
        )
        baseline_visible = source_baseline[visible_start:]
        capability_contract = contract_row["contract"]
        if capability_contract.get("schema") not in {
            REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
            LEGACY_REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
        }:
            raise ValueError("nonlinear replay requires a dynamic contract")
        prepared: list[
            tuple[int, float, float, int, float, np.ndarray, dict[str, Any]]
        ] = []
        replay_group_eligible = True
        for dose_index, (
            canonical_strength,
            treatment_alpha,
        ) in enumerate(_row_dose_grid(contract_row), start=1):
            for pair_member, alpha in ((0, 1.0), (1, treatment_alpha)):
                try:
                    augmented, metadata = apply_real_path_dynamic_contract(
                        source_baseline,
                        capability_contract,
                        alpha=alpha,
                        context_length=(
                            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                        ),
                        future_innovation_policy=(
                            NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
                        ),
                    )
                except ValueError as error:
                    if (
                        pair_member == 1
                        and str(error)
                        == "dynamic dose failed paired minimum separation"
                    ):
                        replay_group_eligible = False
                        break
                    raise
                augmented_array = np.asarray(augmented, dtype=float)
                if augmented_array.ndim == 2:
                    augmented_array = augmented_array[:, 0]
                visible_target = augmented_array[visible_start:]
                prepared.append(
                    (
                        dose_index,
                        canonical_strength,
                        treatment_alpha,
                        pair_member,
                        alpha,
                        visible_target,
                        metadata,
                    )
                )
            if not replay_group_eligible:
                break
        if not replay_group_eligible:
            continue
        for (
            dose_index,
            canonical_strength,
            treatment_alpha,
            pair_member,
            alpha,
            visible_target,
            metadata,
        ) in prepared:
            visible_delta = visible_target - baseline_visible
            row = _sample_row(
                background=background,
                contract_row=contract_row,
                seed_index=int(seed_index),
                dose_index=dose_index,
                pair_member=pair_member,
                canonical_strength=canonical_strength,
                treatment_alpha=treatment_alpha,
                alpha=alpha,
                visible_target=visible_target,
                visible_delta=visible_delta,
                baseline_visible=baseline_visible,
                metadata=metadata,
            )
            main_pair_id = str(row["counterfactual_pair_id"])
            main_group_id = str(row["paired_group_id"])
            main_sample_id = str(row["sample_id"])
            row["evaluation_table"] = (
                "real_anchored_nonlinear_replay_sensitivity"
            )
            row["sample_id"] = f"{main_sample_id}__nonlinear_replay"
            row["master_sample_id"] = row["sample_id"]
            row["counterfactual_pair_id"] = (
                f"{main_pair_id}__nonlinear_replay"
            )
            row["paired_group_id"] = (
                f"{main_group_id}__nonlinear_replay"
            )
            row["baseline_sample_id"] = (
                f"{main_pair_id}__m0__nonlinear_replay"
            )
            row["sensitivity_source_sample_id"] = main_sample_id
            row["sensitivity_source_pair_id"] = main_pair_id
            row["sensitivity_source_paired_group_id"] = main_group_id
            row["excluded_from_primary_score"] = True
            row["generation_metadata"][
                "sensitivity_role"
            ] = "history_residual_replay_auxiliary"
            yield row


def validate_contract_integrity(contract_row: Mapping[str, Any]) -> None:
    """Raise when a persisted available contract cannot be reconstructed."""

    if contract_row.get("available") is not True:
        return
    capability = contract_row.get("contract")
    if not isinstance(capability, Mapping):
        raise ValueError("available real-anchored row has no contract")
    if contract_row.get("schema_version") == (
        "cafe.real_anchored_background_capability.v4"
    ):
        capability_id = str(contract_row.get("capability_id", ""))
        calibration = contract_row.get("dose_calibration")
        if calibration is not None:
            if not isinstance(calibration, Mapping):
                raise ValueError("v4 dose calibration must be a mapping")
            validate_dose_calibration(
                calibration,
                capability_id=capability_id,
            )
            if calibration.get("status") != "available":
                raise ValueError(
                    "available v4 row has unavailable dose calibration"
                )
            gates = contract_row.get("paired_minimum_separation_gate")
            if (
                not isinstance(gates, list)
                or len(gates) != len(calibration["strength_grid"])
                or any(
                    not isinstance(gate, Mapping)
                    or gate.get("accepted") is not True
                    for gate in gates
                )
            ):
                raise ValueError(
                    "available v4 row failed treatment-source separation gates"
                )
        evidence = contract_row.get("dose_design_reference")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("capability_id") != capability_id
            or evidence.get("background_id")
            != contract_row.get("background_id")
        ):
            raise ValueError("v4 real-anchored row has invalid dose evidence")
    if capability.get("schema") in {
        REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
        LEGACY_REAL_PATH_DYNAMIC_CONTRACT_SCHEMA,
    }:
        validate_real_path_dynamic_contract(capability)
        if contract_row.get("qualification_policy_id") != capability.get(
            "qualification_policy_id"
        ):
            raise ValueError("dynamic row/contract qualification policy mismatch")
        if contract_row.get("qualification_thresholds") != capability.get(
            "qualification_thresholds"
        ):
            raise ValueError("dynamic row/contract thresholds mismatch")
        if (
            contract_row.get("schema_version")
            == "cafe.real_anchored_background_capability.v4"
            and contract_row.get("dose_calibration") is not None
            and contract_row.get("dose_calibration")
            != capability.get("dose_calibration")
        ):
            raise ValueError("dynamic row/contract dose calibration mismatch")
        if (
            contract_row.get("dose_design_reference")
            != capability.get("dose_design_reference")
            or contract_row.get("paired_minimum_separation_gate")
            != capability.get("paired_minimum_separation_gate")
        ):
            raise ValueError("dynamic row/contract dose evidence mismatch")
        return
    decomposition = capability.get("decomposition_contract")
    if not isinstance(decomposition, Mapping):
        raise ValueError("available real-anchored row has no decomposition")
    restored = AnchoredDecompositionContract.from_dict(decomposition)
    if restored.contract_sha256 != decomposition.get("contract_sha256"):
        raise ValueError("real-anchored decomposition hash mismatch")
    if restored.schema == ANCHORED_CONTRACT_SCHEMA and (
        contract_row.get("qualification_policy_id")
        != capability.get("qualification_policy_id")
    ):
        raise ValueError("anchored row/contract qualification policy mismatch")
    if restored.schema == ANCHORED_CONTRACT_SCHEMA and (
        contract_row.get("qualification_thresholds")
        != capability.get("qualification_thresholds")
    ):
        raise ValueError("anchored row/contract thresholds mismatch")
    if (
        restored.schema == ANCHORED_CONTRACT_SCHEMA
        and restored.modulation_basis
        != TIME_VARYING_SEASONALITY_BASIS_POLICY
    ):
        raise ValueError("anchored contract lost constrained AM basis policy")
