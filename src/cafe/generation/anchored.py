"""History-only real-anchored counterfactual interventions.

This module intentionally depends only on NumPy (plus the Python standard
library).  It fits a joint local-polynomial and Fourier model to the observed
history, analytically extends the fitted components into the forecast horizon,
and applies paired interventions to the original real path.  The real path is
never reconstructed from the fit: at dose one it is therefore exactly the
input path, including its residuals and held-out future innovations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cafe.generation.real_anchored_policy import (
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
    REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA,
    TIME_VARYING_SEASONALITY_BASIS_POLICY,
)


ANCHORED_CONTRACT_SCHEMA = "cafe.real_anchored_decomposition.v3"
LEGACY_ANCHORED_CONTRACT_SCHEMA = "cafe.real_anchored_decomposition.v2"
REAL_ANCHORED_CAPABILITY_CONTRACT_SCHEMA = (
    "cafe.real_anchored_capability_contract.v3"
)
LEGACY_REAL_ANCHORED_CAPABILITY_CONTRACT_SCHEMA = (
    "cafe.real_anchored_capability_contract.v2"
)
ANCHORED_DECOMPOSITION_METHOD = (
    "joint_structural_fourier_constrained_am_lstsq_v3"
)
LEGACY_ANCHORED_DECOMPOSITION_METHOD = (
    "joint_structural_fourier_lstsq_v2"
)
ANCHORED_EXTENSION_METHOD = "analytic_absolute_time_basis_v1"
CONSTRAINED_AM_BASIS = TIME_VARYING_SEASONALITY_BASIS_POLICY
LEGACY_FREE_SIDEBAND_BASIS = "free_fourier_sidebands_v2"

_SUPPORTED_CAPABILITIES = frozenset(
    {
        "multi_seasonal",
        "regime_switching",
        "time_varying_seasonality",
        "trend",
    }
)
_MINIMUM_SCALE = 1e-12


def _as_2d(values: np.ndarray, *, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=float)
    was_1d = array.ndim == 1
    if was_1d:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [time] or [time, target]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=float), was_1d


def _readonly(values: np.ndarray, *, was_1d: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if was_1d:
        result = result[:, 0]
    result.setflags(write=False)
    return result


def _readonly_int(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=int).copy()
    result.setflags(write=False)
    return result


def _history_sha256(history: np.ndarray) -> str:
    canonical = np.ascontiguousarray(history, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"cafe.real_anchor.history.float64.v1\0")
    digest.update(json.dumps(list(canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _source_values_sha256(values: np.ndarray) -> str:
    """Match the repository's canonical raw float64 source-array hash."""

    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _finalize_capability_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(value)
    result["capability_contract_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _verify_capability_contract(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    expected = payload.pop("capability_contract_sha256", None)
    if not isinstance(expected, str):
        raise ValueError("capability contract has no integrity hash")
    observed = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if expected != observed:
        raise ValueError("capability contract integrity hash mismatch")


def _float_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _coefficient_tuple(values: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(_float_tuple(row) for row in values)


def _period_label(period: float) -> str:
    return format(float(period), ".12g")


def _modulation_sidebands(
    *,
    carrier_period: float,
    modulation_period: float | None,
    harmonics_per_period: int,
    modulation_basis: str,
) -> list[dict[str, float | int | str]]:
    if modulation_period is None:
        return []
    carrier_frequency = 1.0 / carrier_period
    modulation_frequency = 1.0 / modulation_period
    result: list[dict[str, float | int | str]] = []
    for harmonic in range(1, harmonics_per_period + 1):
        for side, frequency in (
            ("lower", harmonic * carrier_frequency - modulation_frequency),
            ("upper", harmonic * carrier_frequency + modulation_frequency),
        ):
            row: dict[str, float | int | str] = {
                "carrier_harmonic": harmonic,
                "side": side,
                "frequency": frequency,
                "period": 1.0 / frequency,
            }
            if modulation_basis == CONSTRAINED_AM_BASIS:
                row.update(
                    {
                        "amplitude_constraint": (
                            "equal_magnitude_symmetric_pair"
                        ),
                        "phase_constraint": (
                            "carrier_product_slow_envelope"
                        ),
                    }
                )
            result.append(row)
    return result


def _feature_names(
    *,
    trend_degree: int,
    carrier_period: float,
    secondary_periods: tuple[float, ...],
    harmonics_per_period: int,
    modulation_period: float | None,
    modulation_basis: str,
    regime_join_index: int | None,
) -> tuple[str, ...]:
    names = ["level", "local_linear_trend"]
    names.extend(
        f"local_trend_power_{power}"
        for power in range(2, trend_degree + 1)
    )
    for role, periods in (
        ("carrier", (carrier_period,)),
        ("secondary", secondary_periods),
    ):
        for period_index, period in enumerate(periods):
            label = _period_label(period)
            for harmonic in range(1, harmonics_per_period + 1):
                prefix = f"{role}_{period_index}_p{label}_h{harmonic}"
                names.extend((f"{prefix}_sin", f"{prefix}_cos"))
    if modulation_period is not None:
        label = _period_label(modulation_period)
        for harmonic in range(1, harmonics_per_period + 1):
            prefix = f"carrier_h{harmonic}_mod_p{label}"
            if modulation_basis == CONSTRAINED_AM_BASIS:
                names.extend(
                    (
                        f"{prefix}_envelope_cos",
                        f"{prefix}_envelope_sin",
                    )
                )
            elif modulation_basis == LEGACY_FREE_SIDEBAND_BASIS:
                names.extend(
                    (
                        f"{prefix}_lower_sideband_sin",
                        f"{prefix}_lower_sideband_cos",
                        f"{prefix}_upper_sideband_sin",
                        f"{prefix}_upper_sideband_cos",
                    )
                )
            else:
                raise ValueError(
                    f"unsupported modulation basis {modulation_basis!r}"
                )
    if regime_join_index is not None:
        names.append(f"regime_level_step_at_{regime_join_index}")
    return tuple(names)


def _basis_matrix(
    length: int,
    *,
    fit_start: int,
    fit_window: int,
    trend_start: int,
    trend_window: int,
    trend_degree: int,
    carrier_period: float,
    secondary_periods: tuple[float, ...],
    harmonics_per_period: int,
    modulation_period: float | None,
    modulation_basis: str,
    modulation_carrier_phases: tuple[float, ...],
    regime_join_index: int | None,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("component extension length must be positive")
    time = np.arange(length, dtype=float)
    # The joint regression may use a long decomposition history (L504), while
    # the controlled trend coordinate remains local (normally W96).  The
    # nonlinear basis is zero in both value and first derivative at its join,
    # so scaling it preserves the earlier level and tangent exactly.
    linear_coordinate = np.maximum(
        (time - float(fit_start)) / float(fit_window),
        0.0,
    )
    nonlinear_coordinate = np.maximum(
        (time - float(trend_start)) / float(trend_window),
        0.0,
    )
    columns = [np.ones(length, dtype=float), linear_coordinate]
    columns.extend(
        nonlinear_coordinate**power
        for power in range(2, trend_degree + 1)
    )
    for period in (carrier_period, *secondary_periods):
        for harmonic in range(1, harmonics_per_period + 1):
            phase = 2.0 * np.pi * harmonic * time / period
            columns.extend((np.sin(phase), np.cos(phase)))
    if modulation_period is not None:
        modulation_frequency = 1.0 / modulation_period
        carrier_frequency = 1.0 / carrier_period
        for harmonic in range(1, harmonics_per_period + 1):
            harmonic_frequency = harmonic * carrier_frequency
            if modulation_basis == CONSTRAINED_AM_BASIS:
                if len(modulation_carrier_phases) != harmonics_per_period:
                    raise ValueError(
                        "constrained AM requires one frozen carrier phase "
                        "per harmonic"
                    )
                carrier_wave = np.sin(
                    2.0 * np.pi * harmonic_frequency * time
                    + float(modulation_carrier_phases[harmonic - 1])
                )
                slow_phase = 2.0 * np.pi * modulation_frequency * time
                columns.extend(
                    (
                        carrier_wave * np.cos(slow_phase),
                        carrier_wave * np.sin(slow_phase),
                    )
                )
            elif modulation_basis == LEGACY_FREE_SIDEBAND_BASIS:
                for sideband_frequency in (
                    harmonic_frequency - modulation_frequency,
                    harmonic_frequency + modulation_frequency,
                ):
                    phase = 2.0 * np.pi * sideband_frequency * time
                    columns.extend((np.sin(phase), np.cos(phase)))
            else:
                raise ValueError(
                    f"unsupported modulation basis {modulation_basis!r}"
                )
    if regime_join_index is not None:
        columns.append((time >= float(regime_join_index)).astype(float))
    return np.column_stack(columns)


def _coefficient_slices(
    *,
    trend_degree: int,
    secondary_period_count: int,
    harmonics_per_period: int,
    has_modulation: bool,
    modulation_basis: str,
    has_regime: bool,
) -> tuple[
    slice,
    slice,
    slice,
    tuple[slice, ...],
    slice,
    slice,
]:
    linear_stop = 2
    trend_stop = trend_degree + 1
    width = 2 * harmonics_per_period
    carrier = slice(trend_stop, trend_stop + width)
    secondary = tuple(
        slice(
            carrier.stop + index * width,
            carrier.stop + (index + 1) * width,
        )
        for index in range(secondary_period_count)
    )
    next_index = carrier.stop + secondary_period_count * width
    if has_modulation:
        modulation_width = (
            2 * harmonics_per_period
            if modulation_basis == CONSTRAINED_AM_BASIS
            else 4 * harmonics_per_period
        )
    else:
        modulation_width = 0
    modulation = slice(next_index, next_index + modulation_width)
    regime_width = 1 if has_regime else 0
    regime = slice(modulation.stop, modulation.stop + regime_width)
    return (
        slice(0, linear_stop),
        slice(linear_stop, trend_stop),
        carrier,
        secondary,
        modulation,
        regime,
    )


def _validate_frequencies(
    *,
    carrier_period: float,
    secondary_periods: tuple[float, ...],
    harmonics_per_period: int,
    modulation_period: float | None,
) -> None:
    frequencies: list[tuple[str, float]] = []
    for role, period in (
        ("carrier", carrier_period),
        *(
            (f"secondary[{index}]", period)
            for index, period in enumerate(secondary_periods)
        ),
    ):
        if not math.isfinite(period) or period < 2.0:
            raise ValueError(f"{role} period must be finite and at least 2")
        for harmonic in range(1, harmonics_per_period + 1):
            frequency = harmonic / period
            if frequency > 0.5 + 1e-12:
                raise ValueError(
                    f"{role} harmonic {harmonic} exceeds the Nyquist frequency"
                )
            frequencies.append((f"{role}:h{harmonic}", frequency))
    if modulation_period is not None:
        if (
            not math.isfinite(modulation_period)
            or modulation_period <= carrier_period
        ):
            raise ValueError(
                "modulation_period must be finite and longer than carrier_period"
            )
        modulation_frequency = 1.0 / modulation_period
        carrier_frequency = 1.0 / carrier_period
        for harmonic in range(1, harmonics_per_period + 1):
            harmonic_frequency = harmonic * carrier_frequency
            for side, frequency in (
                ("lower", harmonic_frequency - modulation_frequency),
                ("upper", harmonic_frequency + modulation_frequency),
            ):
                if not 0.0 < frequency <= 0.5 + 1e-12:
                    raise ValueError(
                        f"carrier harmonic {harmonic} {side} sideband is "
                        "outside resolvable frequencies"
                    )
                frequencies.append(
                    (f"carrier:h{harmonic}:{side}_sideband", frequency)
                )
    for index, (left_name, left) in enumerate(frequencies):
        for right_name, right in frequencies[index + 1 :]:
            tolerance = 1e-10 * max(1.0, abs(left), abs(right))
            if abs(left - right) <= tolerance:
                raise ValueError(
                    "harmonic frequencies must be identifiable; "
                    f"{left_name} collides with {right_name}"
                )


def _mase_reference(
    history: np.ndarray,
    *,
    mase_period: int,
    normalization_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not 1 <= mase_period < history.shape[0]:
        raise ValueError("mase_period must be defined inside the history")
    seasonal = np.mean(
        np.abs(history[mase_period:] - history[:-mase_period]),
        axis=0,
    )
    lag_one = np.mean(np.abs(np.diff(history, axis=0)), axis=0)
    scales = seasonal.copy()
    effective_periods = np.full(history.shape[1], mase_period, dtype=int)
    sources: list[str] = ["seasonal_history"] * history.shape[1]
    for target_index in range(history.shape[1]):
        if not math.isfinite(float(scales[target_index])):
            raise ValueError("history produced a non-finite MASE reference")
        if scales[target_index] > _MINIMUM_SCALE:
            continue
        if lag_one[target_index] > _MINIMUM_SCALE:
            scales[target_index] = lag_one[target_index]
            effective_periods[target_index] = 1
            sources[target_index] = "lag_one_history_fallback"
        else:
            scales[target_index] = max(
                float(normalization_scale[target_index]),
                1.0,
            )
            effective_periods[target_index] = 0
            sources[target_index] = "normalization_scale_constant_fallback"
    return scales, effective_periods, tuple(sources)


@dataclass(frozen=True)
class AnchoredComponents:
    """Analytically extended fitted components in raw target units."""

    level_and_linear_trend: np.ndarray
    trend_nonlinearity: np.ndarray
    carrier: np.ndarray
    secondary: np.ndarray
    secondary_by_period: tuple[np.ndarray, ...]
    amplitude_modulation: np.ndarray
    regime_level_shift: np.ndarray
    fitted: np.ndarray


@dataclass(frozen=True)
class AnchoredCounterfactualMember:
    """One dose member using references frozen by the baseline history."""

    capability_id: str
    alpha: float
    values: np.ndarray
    normalized_values: np.ndarray
    intervention: np.ndarray
    normalization_mean: np.ndarray
    normalization_scale: np.ndarray
    mase_period: int
    mase_scale_by_target: np.ndarray
    mase_effective_period_by_target: np.ndarray
    mase_scale_source_by_target: tuple[str, ...]
    reference_start: int
    reference_length: int
    reference_history_sha256: str
    contract_sha256: str

    @property
    def intervention_rms(self) -> float:
        return float(np.sqrt(np.mean(np.asarray(self.intervention) ** 2)))

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe provenance shared by inference/evaluation rows."""

        return {
            "capability_id": self.capability_id,
            "alpha": self.alpha,
            "contract_sha256": self.contract_sha256,
            "intervention_rms": self.intervention_rms,
            "normalization_mean_by_target": (
                np.asarray(self.normalization_mean).tolist()
            ),
            "normalization_scale_by_target": (
                np.asarray(self.normalization_scale).tolist()
            ),
            "normalization_policy": "baseline_history_shared_by_pair_v1",
            "mase_scale_by_target": (
                np.asarray(self.mase_scale_by_target).tolist()
            ),
            "mase_scale": float(np.mean(self.mase_scale_by_target)),
            "mase_period": self.mase_period,
            "mase_effective_period_by_target": (
                np.asarray(self.mase_effective_period_by_target).tolist()
            ),
            "mase_scale_source_by_target": list(
                self.mase_scale_source_by_target
            ),
            "mase_reference_policy": "baseline_history_shared_by_pair_v1",
            "reference_start": self.reference_start,
            "reference_length": self.reference_length,
            "reference_history_sha256": self.reference_history_sha256,
            "reference_history_policy": (
                "unmodified_fit_history_suffix_shared_by_pair_v1"
            ),
        }


@dataclass(frozen=True)
class AnchoredDecompositionContract:
    """Immutable, JSON-serializable fit and reference contract."""

    context_length: int
    horizon: int
    target_dim: int
    fit_window: int
    fit_start: int
    trend_window: int
    trend_start: int
    trend_degree: int
    carrier_period: float
    secondary_periods: tuple[float, ...]
    harmonics_per_period: int
    modulation_period: float | None
    modulation_basis: str
    modulation_carrier_phases_by_target: tuple[tuple[float, ...], ...]
    regime_join_index: int | None
    minimum_regime_segment_length: int
    minimum_cycles: float
    feature_names: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    design_rank: int
    design_condition_number: float
    fit_rmse_by_target: tuple[float, ...]
    reference_start: int
    reference_length: int
    reference_history_sha256: str
    normalization_mean_by_target: tuple[float, ...]
    normalization_scale_by_target: tuple[float, ...]
    mase_period: int
    mase_scale_by_target: tuple[float, ...]
    mase_effective_period_by_target: tuple[int, ...]
    mase_scale_source_by_target: tuple[str, ...]
    history_sha256: str
    schema: str = ANCHORED_CONTRACT_SCHEMA
    decomposition_method: str = ANCHORED_DECOMPOSITION_METHOD
    extension_method: str = ANCHORED_EXTENSION_METHOD

    def _payload(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "decomposition_method": self.decomposition_method,
            "extension_method": self.extension_method,
            "history_only": True,
            "context_length": self.context_length,
            "horizon": self.horizon,
            "target_dim": self.target_dim,
            "fit_window": self.fit_window,
            "fit_start": self.fit_start,
            "trend_window": self.trend_window,
            "trend_start": self.trend_start,
            "trend_degree": self.trend_degree,
            "carrier_period": self.carrier_period,
            "secondary_periods": list(self.secondary_periods),
            "harmonics_per_period": self.harmonics_per_period,
            "modulation_period": self.modulation_period,
            "modulation_sidebands": _modulation_sidebands(
                carrier_period=self.carrier_period,
                modulation_period=self.modulation_period,
                harmonics_per_period=self.harmonics_per_period,
                modulation_basis=self.modulation_basis,
            ),
            "modulation_extension": (
                (
                    "bounded_symmetric_carrier_amplitude_modulation"
                    if self.modulation_basis == CONSTRAINED_AM_BASIS
                    else "bounded_stationary_carrier_sidebands"
                )
                if self.modulation_period is not None
                else None
            ),
            "regime_join_index": self.regime_join_index,
            "minimum_regime_segment_length": (
                self.minimum_regime_segment_length
            ),
            "regime_extension": (
                "constant_post_join_level"
                if self.regime_join_index is not None
                else None
            ),
            "minimum_cycles": self.minimum_cycles,
            "feature_names": list(self.feature_names),
            "coefficients": [list(row) for row in self.coefficients],
            "design_rank": self.design_rank,
            "design_condition_number": self.design_condition_number,
            "fit_rmse_by_target": list(self.fit_rmse_by_target),
            "reference_start": self.reference_start,
            "reference_length": self.reference_length,
            "reference_history_sha256": self.reference_history_sha256,
            "reference_history_policy": (
                "unmodified_fit_history_suffix_shared_by_pair_v1"
            ),
            "normalization_mean_by_target": list(
                self.normalization_mean_by_target
            ),
            "normalization_scale_by_target": list(
                self.normalization_scale_by_target
            ),
            "normalization_policy": "baseline_history_shared_by_pair_v1",
            "mase_period": self.mase_period,
            "mase_scale_by_target": list(self.mase_scale_by_target),
            "mase_effective_period_by_target": list(
                self.mase_effective_period_by_target
            ),
            "mase_scale_source_by_target": list(
                self.mase_scale_source_by_target
            ),
            "mase_reference_policy": "baseline_history_shared_by_pair_v1",
            "history_sha256": self.history_sha256,
            "interventions": {
                "multi_seasonal": {
                    "law": "x_alpha=x+(alpha-1)*secondary",
                    "fixed_components": [
                        "level_and_linear_trend",
                        "trend_nonlinearity",
                        "carrier",
                        "amplitude_modulation",
                        "regime_level_shift",
                        "real_residual_and_future_innovations",
                    ],
                },
                "trend": {
                    "law": "x_alpha=x+(alpha-1)*trend_nonlinearity",
                    "fixed_components": [
                        "level_and_linear_trend",
                        "carrier",
                        "secondary",
                        "amplitude_modulation",
                        "regime_level_shift",
                        "real_residual_and_future_innovations",
                    ],
                },
                "time_varying_seasonality": {
                    "law": (
                        (
                            "x_alpha=x+(alpha-1)*"
                            "carrier_times_slow_amplitude_envelope"
                        )
                        if self.modulation_basis == CONSTRAINED_AM_BASIS
                        else (
                            "x_alpha=x+(alpha-1)*"
                            "carrier_modulation_sidebands"
                        )
                    ),
                    "fixed_components": [
                        "level_and_linear_trend",
                        "trend_nonlinearity",
                        "carrier",
                        "secondary",
                        "regime_level_shift",
                        "real_residual_and_future_innovations",
                    ],
                },
                "regime_switching": {
                    "law": "x_alpha=x+(alpha-1)*regime_level_shift",
                    "fixed_components": [
                        "level_and_linear_trend",
                        "trend_nonlinearity",
                        "carrier",
                        "secondary",
                        "amplitude_modulation",
                        "real_residual_and_future_innovations",
                    ],
                },
            },
        }
        if self.schema == ANCHORED_CONTRACT_SCHEMA:
            payload["modulation_basis"] = self.modulation_basis
            payload["modulation_carrier_phases_by_target"] = [
                list(row)
                for row in self.modulation_carrier_phases_by_target
            ]
            payload["spectral_component_ownership"] = (
                "shared_background_joint_design_v1"
            )
        return payload

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._payload()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe contract with an integrity hash."""

        payload = self._payload()
        payload["contract_sha256"] = self.contract_sha256
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> AnchoredDecompositionContract:
        """Restore and integrity-check a contract produced by :meth:`to_dict`."""

        payload = dict(value)
        expected_hash = payload.pop("contract_sha256", None)
        if not isinstance(expected_hash, str):
            raise ValueError("real-anchored contract has no integrity hash")
        schema = str(payload.get("schema"))
        if schema not in {
            ANCHORED_CONTRACT_SCHEMA,
            LEGACY_ANCHORED_CONTRACT_SCHEMA,
        }:
            raise ValueError("unsupported real-anchored contract schema")
        if payload.get("history_only") is not True:
            raise ValueError("real-anchored decomposition must be history-only")
        expected_method = (
            ANCHORED_DECOMPOSITION_METHOD
            if schema == ANCHORED_CONTRACT_SCHEMA
            else LEGACY_ANCHORED_DECOMPOSITION_METHOD
        )
        if payload.get("decomposition_method") != expected_method:
            raise ValueError("unsupported real-anchored decomposition method")
        if payload.get("extension_method") != ANCHORED_EXTENSION_METHOD:
            raise ValueError("unsupported real-anchored extension method")
        contract = cls(
            context_length=int(payload["context_length"]),
            horizon=int(payload["horizon"]),
            target_dim=int(payload["target_dim"]),
            fit_window=int(payload["fit_window"]),
            fit_start=int(payload["fit_start"]),
            trend_window=int(payload["trend_window"]),
            trend_start=int(payload["trend_start"]),
            trend_degree=int(payload["trend_degree"]),
            carrier_period=float(payload["carrier_period"]),
            secondary_periods=_float_tuple(payload["secondary_periods"]),
            harmonics_per_period=int(payload["harmonics_per_period"]),
            modulation_period=(
                None
                if payload["modulation_period"] is None
                else float(payload["modulation_period"])
            ),
            modulation_basis=(
                str(payload["modulation_basis"])
                if schema == ANCHORED_CONTRACT_SCHEMA
                else LEGACY_FREE_SIDEBAND_BASIS
            ),
            modulation_carrier_phases_by_target=(
                tuple(
                    _float_tuple(row)
                    for row in payload[
                        "modulation_carrier_phases_by_target"
                    ]
                )
                if schema == ANCHORED_CONTRACT_SCHEMA
                else tuple()
            ),
            regime_join_index=(
                None
                if payload["regime_join_index"] is None
                else int(payload["regime_join_index"])
            ),
            minimum_regime_segment_length=int(
                payload["minimum_regime_segment_length"]
            ),
            minimum_cycles=float(payload["minimum_cycles"]),
            feature_names=tuple(str(name) for name in payload["feature_names"]),
            coefficients=_coefficient_tuple(payload["coefficients"]),
            design_rank=int(payload["design_rank"]),
            design_condition_number=float(
                payload["design_condition_number"]
            ),
            fit_rmse_by_target=_float_tuple(payload["fit_rmse_by_target"]),
            reference_start=int(payload["reference_start"]),
            reference_length=int(payload["reference_length"]),
            reference_history_sha256=str(
                payload["reference_history_sha256"]
            ),
            normalization_mean_by_target=_float_tuple(
                payload["normalization_mean_by_target"]
            ),
            normalization_scale_by_target=_float_tuple(
                payload["normalization_scale_by_target"]
            ),
            mase_period=int(payload["mase_period"]),
            mase_scale_by_target=_float_tuple(
                payload["mase_scale_by_target"]
            ),
            mase_effective_period_by_target=_int_tuple(
                payload["mase_effective_period_by_target"]
            ),
            mase_scale_source_by_target=tuple(
                str(source)
                for source in payload["mase_scale_source_by_target"]
            ),
            history_sha256=str(payload["history_sha256"]),
            schema=schema,
            decomposition_method=str(payload["decomposition_method"]),
            extension_method=str(payload["extension_method"]),
        )
        contract._validate()
        if payload != contract._payload():
            raise ValueError(
                "serialized real-anchored contract does not match its schema"
            )
        if expected_hash != contract.contract_sha256:
            raise ValueError("real-anchored contract integrity hash mismatch")
        return contract

    def _validate(self) -> None:
        if self.schema not in {
            ANCHORED_CONTRACT_SCHEMA,
            LEGACY_ANCHORED_CONTRACT_SCHEMA,
        }:
            raise ValueError("unsupported real-anchored contract schema")
        expected_method = (
            ANCHORED_DECOMPOSITION_METHOD
            if self.schema == ANCHORED_CONTRACT_SCHEMA
            else LEGACY_ANCHORED_DECOMPOSITION_METHOD
        )
        if self.decomposition_method != expected_method:
            raise ValueError("unsupported real-anchored decomposition method")
        if self.extension_method != ANCHORED_EXTENSION_METHOD:
            raise ValueError("unsupported real-anchored extension method")
        if self.context_length < 3:
            raise ValueError("context_length must be at least 3")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.target_dim < 1:
            raise ValueError("target_dim must be positive")
        if not 3 <= self.fit_window <= self.context_length:
            raise ValueError("fit_window must lie inside the history")
        if self.fit_start != self.context_length - self.fit_window:
            raise ValueError("fit_start does not match the local fit window")
        if not 3 <= self.trend_window <= self.fit_window:
            raise ValueError("trend_window must lie inside the fit window")
        if self.trend_start != self.context_length - self.trend_window:
            raise ValueError("trend_start does not match the local trend window")
        if not 0 <= self.reference_start < self.context_length:
            raise ValueError("reference_start must lie inside the fit history")
        if self.reference_length < 3:
            raise ValueError("reference_length must be at least 3")
        if self.reference_start + self.reference_length != self.context_length:
            raise ValueError(
                "normalization/MASE reference must be a fit-history suffix"
            )
        if self.trend_degree not in (2, 3):
            raise ValueError("trend_degree must be 2 or 3")
        if self.harmonics_per_period < 1:
            raise ValueError("harmonics_per_period must be positive")
        expected_modulation_basis = (
            CONSTRAINED_AM_BASIS
            if self.schema == ANCHORED_CONTRACT_SCHEMA
            else LEGACY_FREE_SIDEBAND_BASIS
        )
        if self.modulation_basis != expected_modulation_basis:
            raise ValueError(
                "modulation basis does not match decomposition schema"
            )
        if self.schema == ANCHORED_CONTRACT_SCHEMA:
            expected_phase_shape = (
                (self.target_dim, self.harmonics_per_period)
                if self.modulation_period is not None
                else (0,)
            )
            observed_phases = np.asarray(
                self.modulation_carrier_phases_by_target,
                dtype=float,
            )
            if observed_phases.shape != expected_phase_shape:
                raise ValueError(
                    "modulation carrier phases do not match target/harmonic "
                    "shape"
                )
            if not np.isfinite(observed_phases).all():
                raise ValueError("modulation carrier phases must be finite")
        elif self.modulation_carrier_phases_by_target:
            raise ValueError(
                "legacy free-sideband contracts cannot carry AM phases"
            )
        if self.minimum_regime_segment_length < 2:
            raise ValueError(
                "minimum_regime_segment_length must be at least 2"
            )
        if not math.isfinite(self.minimum_cycles) or self.minimum_cycles <= 0:
            raise ValueError("minimum_cycles must be finite and positive")
        _validate_frequencies(
            carrier_period=self.carrier_period,
            secondary_periods=self.secondary_periods,
            harmonics_per_period=self.harmonics_per_period,
            modulation_period=self.modulation_period,
        )
        for period in (self.carrier_period, *self.secondary_periods):
            if self.fit_window / period + 1e-12 < self.minimum_cycles:
                raise ValueError(
                    "each period must have the contracted minimum number "
                    "of cycles in the fit window"
                )
        if self.modulation_period is not None and (
            self.fit_window / self.modulation_period + 1e-12
            < self.minimum_cycles
        ):
            raise ValueError(
                "modulation_period must have the contracted minimum number "
                "of cycles in the fit window"
            )
        if self.regime_join_index is not None:
            pre_length = self.regime_join_index - self.fit_start
            post_length = self.context_length - self.regime_join_index
            if (
                pre_length < self.minimum_regime_segment_length
                or post_length < self.minimum_regime_segment_length
            ):
                raise ValueError(
                    "regime joinpoint must leave the contracted history-only "
                    "segment length on both sides"
                )
        expected_names = _feature_names(
            trend_degree=self.trend_degree,
            carrier_period=self.carrier_period,
            secondary_periods=self.secondary_periods,
            harmonics_per_period=self.harmonics_per_period,
            modulation_period=self.modulation_period,
            modulation_basis=self.modulation_basis,
            regime_join_index=self.regime_join_index,
        )
        if self.feature_names != expected_names:
            raise ValueError("feature_names do not match the basis contract")
        if self.fit_window < len(expected_names) + 2:
            raise ValueError(
                "fit_window must exceed the joint decomposition feature count by 2"
            )
        if self.design_rank != len(expected_names):
            raise ValueError("design_rank does not match the joint basis")
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.shape != (len(expected_names), self.target_dim):
            raise ValueError("coefficient matrix does not match contract shape")
        reference_vectors = (
            self.fit_rmse_by_target,
            self.normalization_mean_by_target,
            self.normalization_scale_by_target,
            self.mase_scale_by_target,
            self.mase_effective_period_by_target,
            self.mase_scale_source_by_target,
        )
        if any(len(vector) != self.target_dim for vector in reference_vectors):
            raise ValueError("target reference vectors do not match target_dim")
        numeric = np.asarray(
            [
                *coefficients.ravel().tolist(),
                self.design_condition_number,
                *self.fit_rmse_by_target,
                *self.normalization_mean_by_target,
                *self.normalization_scale_by_target,
                *self.mase_scale_by_target,
            ],
            dtype=float,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("contract contains non-finite numeric values")
        if not 1.0 <= self.design_condition_number <= 1e12:
            raise ValueError("design condition number is outside policy")
        if any(value < 0 for value in self.fit_rmse_by_target):
            raise ValueError("fit RMSE values must be non-negative")
        if any(value <= 0 for value in self.normalization_scale_by_target):
            raise ValueError("normalization scales must be positive")
        if any(value <= 0 for value in self.mase_scale_by_target):
            raise ValueError("MASE scales must be positive")
        if not 1 <= self.mase_period < self.reference_length:
            raise ValueError("mase_period must lie inside reference_history")
        if any(
            value not in (0, 1, self.mase_period)
            for value in self.mase_effective_period_by_target
        ):
            raise ValueError("effective MASE periods are inconsistent")
        if not _is_sha256(self.history_sha256):
            raise ValueError("history_sha256 is malformed")
        if not _is_sha256(self.reference_history_sha256):
            raise ValueError("reference_history_sha256 is malformed")

    def components(self, length: int | None = None) -> AnchoredComponents:
        """Extend every fitted component without observing future values."""

        total_length = (
            self.context_length + self.horizon
            if length is None
            else int(length)
        )
        if not self.context_length <= total_length <= (
            self.context_length + self.horizon
        ):
            raise ValueError(
                "component length must span the history and lie inside the "
                "contracted forecast horizon"
            )
        designs = tuple(
            _basis_matrix(
                total_length,
                fit_start=self.fit_start,
                fit_window=self.fit_window,
                trend_start=self.trend_start,
                trend_window=self.trend_window,
                trend_degree=self.trend_degree,
                carrier_period=self.carrier_period,
                secondary_periods=self.secondary_periods,
                harmonics_per_period=self.harmonics_per_period,
                modulation_period=self.modulation_period,
                modulation_basis=self.modulation_basis,
                modulation_carrier_phases=(
                    self.modulation_carrier_phases_by_target[target_index]
                    if self.modulation_carrier_phases_by_target
                    else tuple()
                ),
                regime_join_index=self.regime_join_index,
            )
            for target_index in range(self.target_dim)
        )
        coefficients = np.asarray(self.coefficients, dtype=float)
        (
            linear_slice,
            nonlinear_slice,
            carrier_slice,
            secondary_slices,
            modulation_slice,
            regime_slice,
        ) = _coefficient_slices(
            trend_degree=self.trend_degree,
            secondary_period_count=len(self.secondary_periods),
            harmonics_per_period=self.harmonics_per_period,
            has_modulation=self.modulation_period is not None,
            modulation_basis=self.modulation_basis,
            has_regime=self.regime_join_index is not None,
        )
        def evaluate(component_slice: slice) -> np.ndarray:
            return np.column_stack(
                [
                    design[:, component_slice]
                    @ coefficients[component_slice, target_index]
                    for target_index, design in enumerate(designs)
                ]
            )

        level_and_linear = evaluate(linear_slice)
        trend_nonlinearity = evaluate(nonlinear_slice)
        carrier = evaluate(carrier_slice)
        secondary_by_period = tuple(
            evaluate(component_slice)
            for component_slice in secondary_slices
        )
        secondary = (
            np.sum(np.stack(secondary_by_period, axis=0), axis=0)
            if secondary_by_period
            else np.zeros_like(carrier)
        )
        amplitude_modulation = (
            evaluate(modulation_slice)
            if self.modulation_period is not None
            else np.zeros_like(carrier)
        )
        regime_level_shift = (
            evaluate(regime_slice)
            if self.regime_join_index is not None
            else np.zeros_like(carrier)
        )
        fitted = (
            level_and_linear
            + trend_nonlinearity
            + carrier
            + secondary
            + amplitude_modulation
            + regime_level_shift
        )
        return AnchoredComponents(
            level_and_linear_trend=_readonly(level_and_linear),
            trend_nonlinearity=_readonly(trend_nonlinearity),
            carrier=_readonly(carrier),
            secondary=_readonly(secondary),
            secondary_by_period=tuple(
                _readonly(component) for component in secondary_by_period
            ),
            amplitude_modulation=_readonly(amplitude_modulation),
            regime_level_shift=_readonly(regime_level_shift),
            fitted=_readonly(fitted),
        )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize with the baseline history reference frozen in this fit."""

        array, was_1d = _as_2d(values, name="values")
        if array.shape[1] != self.target_dim:
            raise ValueError("values target dimension does not match contract")
        mean = np.asarray(self.normalization_mean_by_target, dtype=float)
        scale = np.asarray(self.normalization_scale_by_target, dtype=float)
        return _readonly((array - mean[None, :]) / scale[None, :], was_1d=was_1d)

    def intervention_component(
        self,
        capability_id: str,
        *,
        length: int | None = None,
    ) -> np.ndarray:
        """Return the sole component controlled by a capability dose."""

        if capability_id not in _SUPPORTED_CAPABILITIES:
            raise ValueError(
                f"unsupported anchored capability {capability_id!r}; "
                f"expected one of {sorted(_SUPPORTED_CAPABILITIES)}"
            )
        if capability_id == "multi_seasonal" and not self.secondary_periods:
            raise ValueError(
                "multi_seasonal requires at least one secondary period"
            )
        if (
            capability_id == "time_varying_seasonality"
            and self.modulation_period is None
        ):
            raise ValueError(
                "time_varying_seasonality requires modulation_period"
            )
        if capability_id == "regime_switching" and (
            self.regime_join_index is None
        ):
            raise ValueError("regime_switching requires regime_join_index")
        components = self.components(length)
        if capability_id == "multi_seasonal":
            return components.secondary
        if capability_id == "trend":
            return components.trend_nonlinearity
        if capability_id == "time_varying_seasonality":
            return components.amplitude_modulation
        return components.regime_level_shift


def fit_anchored_decomposition(
    baseline: np.ndarray,
    *,
    context_length: int,
    horizon: int,
    carrier_period: float,
    secondary_periods: tuple[float, ...] | list[float],
    fit_window: int | None = None,
    trend_window: int | None = None,
    trend_degree: int = 2,
    harmonics_per_period: int = 1,
    modulation_period: float | None = None,
    regime_join_index: int | None = None,
    minimum_regime_segment_length: int = 24,
    minimum_cycles: float = 2.0,
    mase_period: int | None = None,
    reference_history: np.ndarray | None = None,
) -> AnchoredDecompositionContract:
    """Fit a deterministic contract using only ``baseline[:context_length]``.

    ``baseline`` may contain the held-out future to make pipeline integration
    convenient, but no value at or after ``context_length`` is read.  Fourier
    phases use absolute integer time, so the fitted carrier, secondary, and
    modulation-sideband components have a deterministic analytic continuation
    through ``horizon``.  A supplied history joinpoint adds only a bounded
    level step whose forecast extension is constant, never a fitted future
    regime transition.
    If ``reference_history`` is supplied, it must exactly equal a suffix of
    the fit history.  Normalization and MASE are then frozen from that suffix,
    which permits an L504 decomposition while exposing only its final L336 to
    a forecasting model.
    """

    values = np.asarray(baseline, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("baseline must have shape [time] or [time, target]")
    context_length = int(context_length)
    horizon = int(horizon)
    if context_length < 3 or values.shape[0] < context_length:
        raise ValueError("baseline does not contain the requested history")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    history = np.ascontiguousarray(values[:context_length], dtype=float)
    if not np.isfinite(history).all():
        raise ValueError("baseline history must contain only finite values")
    if reference_history is None:
        reference = history
    else:
        reference, _ = _as_2d(
            reference_history,
            name="reference_history",
        )
        if reference.shape[1] != history.shape[1]:
            raise ValueError(
                "reference_history target dimension does not match history"
            )
        if not 3 <= reference.shape[0] <= context_length:
            raise ValueError(
                "reference_history length must lie inside the fit history"
            )
        if not np.array_equal(history[-reference.shape[0] :], reference):
            raise ValueError(
                "reference_history must exactly match the fit-history suffix"
            )
    reference_start = context_length - reference.shape[0]
    secondary = tuple(float(period) for period in secondary_periods)
    window = min(96, context_length) if fit_window is None else int(fit_window)
    fit_start = context_length - window
    local_trend_window = (
        min(96, window) if trend_window is None else int(trend_window)
    )
    trend_start = context_length - local_trend_window
    carrier_period = float(carrier_period)
    harmonics_per_period = int(harmonics_per_period)
    modulation_period = (
        None if modulation_period is None else float(modulation_period)
    )
    regime_join_index = (
        None if regime_join_index is None else int(regime_join_index)
    )
    minimum_regime_segment_length = int(minimum_regime_segment_length)
    minimum_cycles = float(minimum_cycles)
    modulation_basis = CONSTRAINED_AM_BASIS
    modulation_carrier_phases_by_target: tuple[tuple[float, ...], ...]
    if modulation_period is None:
        modulation_carrier_phases_by_target = tuple()
    else:
        # Freeze one carrier phase per target/harmonic before introducing the
        # slow envelope.  The final AM columns are products of that carrier
        # waveform and sin/cos envelope coordinates.  This is a two-degree-
        # of-freedom symmetric AM subspace, rather than four unrelated Fourier
        # sidebands that can silently encode phase modulation.
        preliminary_design = _basis_matrix(
            context_length,
            fit_start=fit_start,
            fit_window=window,
            trend_start=trend_start,
            trend_window=local_trend_window,
            trend_degree=int(trend_degree),
            carrier_period=carrier_period,
            secondary_periods=secondary,
            harmonics_per_period=harmonics_per_period,
            modulation_period=None,
            modulation_basis=modulation_basis,
            modulation_carrier_phases=tuple(),
            regime_join_index=regime_join_index,
        )[fit_start:context_length]
        preliminary_coefficients, *_ = np.linalg.lstsq(
            preliminary_design,
            history[fit_start:context_length],
            rcond=None,
        )
        _, _, preliminary_carrier_slice, _, _, _ = _coefficient_slices(
            trend_degree=int(trend_degree),
            secondary_period_count=len(secondary),
            harmonics_per_period=harmonics_per_period,
            has_modulation=False,
            modulation_basis=modulation_basis,
            has_regime=regime_join_index is not None,
        )
        phase_rows: list[tuple[float, ...]] = []
        for target_index in range(history.shape[1]):
            target_phases: list[float] = []
            for harmonic_index in range(harmonics_per_period):
                offset = preliminary_carrier_slice.start + 2 * harmonic_index
                sine_coefficient = float(
                    preliminary_coefficients[offset, target_index]
                )
                cosine_coefficient = float(
                    preliminary_coefficients[offset + 1, target_index]
                )
                target_phases.append(
                    float(
                        math.atan2(
                            cosine_coefficient,
                            sine_coefficient,
                        )
                    )
                )
            phase_rows.append(tuple(target_phases))
        modulation_carrier_phases_by_target = tuple(phase_rows)
    feature_names = _feature_names(
        trend_degree=trend_degree,
        carrier_period=carrier_period,
        secondary_periods=secondary,
        harmonics_per_period=harmonics_per_period,
        modulation_period=modulation_period,
        modulation_basis=modulation_basis,
        regime_join_index=regime_join_index,
    )
    provisional = AnchoredDecompositionContract(
        context_length=context_length,
        horizon=horizon,
        target_dim=history.shape[1],
        fit_window=window,
        fit_start=fit_start,
        trend_window=local_trend_window,
        trend_start=trend_start,
        trend_degree=int(trend_degree),
        carrier_period=carrier_period,
        secondary_periods=secondary,
        harmonics_per_period=harmonics_per_period,
        modulation_period=modulation_period,
        modulation_basis=modulation_basis,
        modulation_carrier_phases_by_target=(
            modulation_carrier_phases_by_target
        ),
        regime_join_index=regime_join_index,
        minimum_regime_segment_length=minimum_regime_segment_length,
        minimum_cycles=minimum_cycles,
        feature_names=feature_names,
        coefficients=tuple(
            tuple(0.0 for _ in range(history.shape[1]))
            for _ in feature_names
        ),
        design_rank=len(feature_names),
        design_condition_number=1.0,
        fit_rmse_by_target=tuple(0.0 for _ in range(history.shape[1])),
        reference_start=reference_start,
        reference_length=reference.shape[0],
        reference_history_sha256=_history_sha256(reference),
        normalization_mean_by_target=tuple(
            float(value) for value in np.mean(reference, axis=0)
        ),
        normalization_scale_by_target=tuple(
            1.0 for _ in range(history.shape[1])
        ),
        mase_period=1,
        mase_scale_by_target=tuple(1.0 for _ in range(history.shape[1])),
        mase_effective_period_by_target=tuple(
            1 for _ in range(history.shape[1])
        ),
        mase_scale_source_by_target=tuple(
            "provisional" for _ in range(history.shape[1])
        ),
        history_sha256=_history_sha256(history),
    )
    provisional._validate()

    designs = tuple(
        _basis_matrix(
            context_length,
            fit_start=fit_start,
            fit_window=window,
            trend_start=trend_start,
            trend_window=local_trend_window,
            trend_degree=int(trend_degree),
            carrier_period=carrier_period,
            secondary_periods=secondary,
            harmonics_per_period=harmonics_per_period,
            modulation_period=modulation_period,
            modulation_basis=modulation_basis,
            modulation_carrier_phases=(
                modulation_carrier_phases_by_target[target_index]
                if modulation_carrier_phases_by_target
                else tuple()
            ),
            regime_join_index=regime_join_index,
        )[fit_start:context_length]
        for target_index in range(history.shape[1])
    )
    if window < designs[0].shape[1] + 2:
        raise ValueError(
            "fit_window must exceed the joint decomposition feature count by 2"
        )
    coefficient_columns: list[np.ndarray] = []
    ranks: list[int] = []
    condition_numbers: list[float] = []
    fit_rmse_values: list[float] = []
    for target_index, design in enumerate(designs):
        target = history[fit_start:context_length, target_index]
        target_coefficients, _, design_rank, singular_values = (
            np.linalg.lstsq(design, target, rcond=None)
        )
        if int(design_rank) != design.shape[1]:
            raise ValueError("joint decomposition design is rank deficient")
        condition_numbers.append(
            float(singular_values[0] / singular_values[-1])
        )
        fitted_target = design @ target_coefficients
        fit_rmse_values.append(
            float(np.sqrt(np.mean((target - fitted_target) ** 2)))
        )
        ranks.append(int(design_rank))
        coefficient_columns.append(target_coefficients)
    coefficients = np.column_stack(coefficient_columns)
    condition_number = max(condition_numbers)
    if not math.isfinite(condition_number) or condition_number > 1e12:
        raise ValueError("joint decomposition design is ill-conditioned")
    design_rank = min(ranks)
    fit_rmse = np.asarray(fit_rmse_values, dtype=float)

    normalization_mean = np.mean(reference, axis=0)
    normalization_scale = np.std(reference, axis=0)
    normalization_scale = np.where(
        normalization_scale > 1e-6,
        normalization_scale,
        1.0,
    )
    resolved_mase_period = (
        int(round(carrier_period))
        if mase_period is None
        else int(mase_period)
    )
    mase_scale, effective_period, mase_source = _mase_reference(
        reference,
        mase_period=resolved_mase_period,
        normalization_scale=normalization_scale,
    )
    contract = AnchoredDecompositionContract(
        context_length=context_length,
        horizon=horizon,
        target_dim=history.shape[1],
        fit_window=window,
        fit_start=fit_start,
        trend_window=local_trend_window,
        trend_start=trend_start,
        trend_degree=int(trend_degree),
        carrier_period=carrier_period,
        secondary_periods=secondary,
        harmonics_per_period=harmonics_per_period,
        modulation_period=modulation_period,
        modulation_basis=modulation_basis,
        modulation_carrier_phases_by_target=(
            modulation_carrier_phases_by_target
        ),
        regime_join_index=regime_join_index,
        minimum_regime_segment_length=minimum_regime_segment_length,
        minimum_cycles=minimum_cycles,
        feature_names=feature_names,
        coefficients=tuple(
            tuple(float(value) for value in row) for row in coefficients
        ),
        design_rank=int(design_rank),
        design_condition_number=condition_number,
        fit_rmse_by_target=tuple(float(value) for value in fit_rmse),
        reference_start=reference_start,
        reference_length=reference.shape[0],
        reference_history_sha256=_history_sha256(reference),
        normalization_mean_by_target=tuple(
            float(value) for value in normalization_mean
        ),
        normalization_scale_by_target=tuple(
            float(value) for value in normalization_scale
        ),
        mase_period=resolved_mase_period,
        mase_scale_by_target=tuple(float(value) for value in mase_scale),
        mase_effective_period_by_target=tuple(
            int(value) for value in effective_period
        ),
        mase_scale_source_by_target=mase_source,
        history_sha256=_history_sha256(history),
    )
    contract._validate()
    return contract


def apply_anchored_counterfactual(
    baseline: np.ndarray,
    contract: AnchoredDecompositionContract,
    *,
    capability_id: str,
    alpha: float,
) -> AnchoredCounterfactualMember:
    """Apply a paired intervention while retaining the complete real path.

    The input history must match the history that froze ``contract``.  Its
    held-out future remains untouched except for the analytically extended
    controlled component.  Every member is normalized with, and evaluated
    against, the same baseline-history references stored in the contract.
    """

    values, was_1d = _as_2d(baseline, name="baseline")
    if values.shape[1] != contract.target_dim:
        raise ValueError("baseline target dimension does not match contract")
    if not contract.context_length <= values.shape[0] <= (
        contract.context_length + contract.horizon
    ):
        raise ValueError(
            "baseline must contain the history and no more than the "
            "contracted forecast horizon"
        )
    if _history_sha256(values[: contract.context_length]) != (
        contract.history_sha256
    ):
        raise ValueError("baseline history does not match the frozen contract")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    component = np.asarray(
        contract.intervention_component(
            capability_id,
            length=values.shape[0],
        )
    )
    if alpha == 1.0:
        intervention = np.zeros_like(values)
        augmented = values.copy()
    else:
        intervention = (alpha - 1.0) * component
        augmented = values + intervention
    mean = np.asarray(contract.normalization_mean_by_target, dtype=float)
    scale = np.asarray(contract.normalization_scale_by_target, dtype=float)
    normalized = (augmented - mean[None, :]) / scale[None, :]
    return AnchoredCounterfactualMember(
        capability_id=capability_id,
        alpha=alpha,
        values=_readonly(augmented, was_1d=was_1d),
        normalized_values=_readonly(normalized, was_1d=was_1d),
        intervention=_readonly(intervention, was_1d=was_1d),
        normalization_mean=_readonly(mean[None, :])[0],
        normalization_scale=_readonly(scale[None, :])[0],
        mase_period=contract.mase_period,
        mase_scale_by_target=_readonly(
            np.asarray(contract.mase_scale_by_target, dtype=float)[None, :]
        )[0],
        mase_effective_period_by_target=_readonly_int(
            np.asarray(contract.mase_effective_period_by_target, dtype=int)
        ),
        mase_scale_source_by_target=(
            contract.mase_scale_source_by_target
        ),
        reference_start=contract.reference_start,
        reference_length=contract.reference_length,
        reference_history_sha256=contract.reference_history_sha256,
        contract_sha256=contract.contract_sha256,
    )


def anchored_pair_delta(
    contract: AnchoredDecompositionContract,
    *,
    capability_id: str,
    alpha_from: float,
    alpha_to: float,
    length: int | None = None,
) -> np.ndarray:
    """Return the exact declared effect between two paired dose members."""

    alpha_from = float(alpha_from)
    alpha_to = float(alpha_to)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (alpha_from, alpha_to)
    ):
        raise ValueError("pair alphas must be finite and non-negative")
    component = contract.intervention_component(
        capability_id,
        length=length,
    )
    return _readonly((alpha_to - alpha_from) * np.asarray(component))


def fit_real_anchored_contract(
    history: np.ndarray,
    *,
    capability_id: str,
    carrier_period: float,
    secondary_periods: tuple[float, ...] | list[float] = (),
    horizon: int = 48,
    fit_window: int | None = None,
    trend_window: int | None = None,
    trend_degree: int = 2,
    harmonics_per_period: int = 1,
    modulation_period: float | None = None,
    regime_join_index: int | None = None,
    minimum_regime_segment_length: int = 24,
    minimum_cycles: float = 2.0,
    mase_period: int | None = None,
    minimum_component_rms_ratio: float = 1e-8,
    minimum_visible_component_rms_ratio: float | None = None,
    visible_context_length: int = 168,
    minimum_future_component_rms_ratio: float | None = None,
    reference_history: np.ndarray | None = None,
    qualification_policy_id: str = (
        REAL_ANCHORED_QUALIFICATION_POLICY_SCHEMA
    ),
    qualification_policy_sha256: str | None = None,
    qualification_threshold_source: str = (
        QUALIFICATION_THRESHOLD_SOURCE_POLICY
    ),
    qualification_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze a JSON-safe capability contract for one real history.

    Unlike :func:`fit_anchored_decomposition`, this calibration-facing wrapper
    records expected data limitations as ``available=False`` instead of
    aborting a batch.  Unexpected input types and non-finite histories remain
    errors because no trustworthy source hash can be frozen for them.
    """

    values, _ = _as_2d(history, name="history")
    capability_id = str(capability_id)
    minimum_component_rms_ratio = float(minimum_component_rms_ratio)
    minimum_visible_component_rms_ratio = (
        minimum_component_rms_ratio
        if minimum_visible_component_rms_ratio is None
        else float(minimum_visible_component_rms_ratio)
    )
    visible_context_length = int(visible_context_length)
    minimum_future_component_rms_ratio = (
        minimum_component_rms_ratio
        if minimum_future_component_rms_ratio is None
        else float(minimum_future_component_rms_ratio)
    )
    if (
        not math.isfinite(minimum_component_rms_ratio)
        or minimum_component_rms_ratio < 0.0
    ):
        raise ValueError(
            "minimum_component_rms_ratio must be finite and non-negative"
        )
    if (
        not math.isfinite(minimum_visible_component_rms_ratio)
        or minimum_visible_component_rms_ratio < 0.0
    ):
        raise ValueError(
            "minimum_visible_component_rms_ratio must be finite and "
            "non-negative"
        )
    if not 3 <= visible_context_length <= values.shape[0]:
        raise ValueError(
            "visible_context_length must lie inside the fit history"
        )
    if (
        not math.isfinite(minimum_future_component_rms_ratio)
        or minimum_future_component_rms_ratio < 0.0
    ):
        raise ValueError(
            "minimum_future_component_rms_ratio must be finite and "
            "non-negative"
        )
    if not isinstance(qualification_policy_id, str) or not (
        qualification_policy_id
    ):
        raise ValueError("qualification_policy_id must be a non-empty string")
    if qualification_threshold_source != QUALIFICATION_THRESHOLD_SOURCE_POLICY:
        raise ValueError("unsupported qualification threshold source policy")
    local_qualification_thresholds = {
        "minimum_component_rms_ratio": minimum_component_rms_ratio,
        "minimum_visible_component_rms_ratio": (
            minimum_visible_component_rms_ratio
        ),
        "visible_context_length": visible_context_length,
        "minimum_future_component_rms_ratio": (
            minimum_future_component_rms_ratio
        ),
    }
    if qualification_thresholds is None:
        frozen_qualification_thresholds = local_qualification_thresholds
    else:
        frozen_qualification_thresholds = dict(qualification_thresholds)
        for name, expected in local_qualification_thresholds.items():
            observed = frozen_qualification_thresholds.get(name)
            if not isinstance(observed, (int, float)) or float(observed) != (
                float(expected)
            ):
                raise ValueError(
                    f"qualification threshold {name!r} disagrees with fit args"
                )
    source_hash = _source_values_sha256(values)
    base = {
        "schema": REAL_ANCHORED_CAPABILITY_CONTRACT_SCHEMA,
        "capability_id": capability_id,
        "source_history_sha256": source_hash,
        "qualification_policy_id": qualification_policy_id,
        "qualification_policy_sha256": qualification_policy_sha256,
        "qualification_threshold_source": qualification_threshold_source,
        "qualification_thresholds": frozen_qualification_thresholds,
        "minimum_component_rms_ratio": minimum_component_rms_ratio,
        "minimum_visible_component_rms_ratio": (
            minimum_visible_component_rms_ratio
        ),
        "visible_context_length": visible_context_length,
        "minimum_future_component_rms_ratio": (
            minimum_future_component_rms_ratio
        ),
    }
    if capability_id not in _SUPPORTED_CAPABILITIES:
        return _finalize_capability_contract(
            {
                **base,
                "available": False,
                "unavailable_reason": "unsupported_capability",
                "unavailable_detail": (
                    f"expected one of {sorted(_SUPPORTED_CAPABILITIES)}"
                ),
                "decomposition_contract": None,
            }
        )
    if capability_id == "multi_seasonal" and not secondary_periods:
        return _finalize_capability_contract(
            {
                **base,
                "available": False,
                "unavailable_reason": "secondary_periods_required",
                "unavailable_detail": (
                    "multi_seasonal needs at least one resolved secondary period"
                ),
                "decomposition_contract": None,
            }
        )
    if (
        capability_id == "time_varying_seasonality"
        and modulation_period is None
    ):
        return _finalize_capability_contract(
            {
                **base,
                "available": False,
                "unavailable_reason": "modulation_period_required",
                "unavailable_detail": (
                    "time_varying_seasonality needs a history-resolved "
                    "modulation period"
                ),
                "decomposition_contract": None,
            }
        )
    if capability_id == "regime_switching" and regime_join_index is None:
        return _finalize_capability_contract(
            {
                **base,
                "available": False,
                "unavailable_reason": "regime_join_index_required",
                "unavailable_detail": (
                    "regime_switching needs a history-detected joinpoint"
                ),
                "decomposition_contract": None,
            }
        )
    try:
        decomposition = fit_anchored_decomposition(
            values,
            context_length=values.shape[0],
            horizon=horizon,
            carrier_period=carrier_period,
            secondary_periods=secondary_periods,
            fit_window=fit_window,
            trend_window=trend_window,
            trend_degree=trend_degree,
            harmonics_per_period=harmonics_per_period,
            modulation_period=modulation_period,
            regime_join_index=regime_join_index,
            minimum_regime_segment_length=(
                minimum_regime_segment_length
            ),
            minimum_cycles=minimum_cycles,
            mase_period=mase_period,
            reference_history=reference_history,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return _finalize_capability_contract(
            {
                **base,
                "available": False,
                "unavailable_reason": "decomposition_unavailable",
                "unavailable_detail": str(error),
                "decomposition_contract": None,
            }
        )
    history_component = decomposition.intervention_component(
        capability_id,
        length=decomposition.context_length,
    )
    extended_component = np.asarray(
        decomposition.intervention_component(
            capability_id,
            length=decomposition.context_length + decomposition.horizon,
        ),
        dtype=float,
    )
    future_component = extended_component[decomposition.context_length :]
    visible_component = np.asarray(history_component, dtype=float)[
        -visible_context_length:
    ]
    component_rms = float(
        np.sqrt(np.mean(np.asarray(history_component, dtype=float) ** 2))
    )
    visible_component_rms = float(
        np.sqrt(np.mean(visible_component**2))
    )
    future_component_rms = float(
        np.sqrt(np.mean(future_component**2))
    )
    reference_scale = float(
        np.mean(decomposition.normalization_scale_by_target)
    )
    threshold = float(minimum_component_rms_ratio) * reference_scale
    visible_threshold = (
        float(minimum_visible_component_rms_ratio) * reference_scale
    )
    future_threshold = (
        float(minimum_future_component_rms_ratio) * reference_scale
    )
    gate_metrics = {
        "controlled_component_rms": component_rms,
        "controlled_component_history_rms": component_rms,
        "controlled_component_visible_history_rms": visible_component_rms,
        "controlled_component_visible_context_length": (
            visible_context_length
        ),
        "controlled_component_future_rms": future_component_rms,
        "minimum_component_rms": threshold,
        "minimum_history_component_rms": threshold,
        "minimum_visible_history_component_rms": visible_threshold,
        "minimum_future_component_rms": future_threshold,
        "future_component_horizon": decomposition.horizon,
        "future_component_source": (
            "analytic_history_fitted_component_extension"
        ),
    }
    if not all(
        math.isfinite(value)
        for value in (
            component_rms,
            visible_component_rms,
            future_component_rms,
            threshold,
            visible_threshold,
            future_threshold,
        )
    ):
        raise ValueError(
            "controlled component RMS gates must be finite and non-negative"
        )
    if component_rms <= threshold:
        return _finalize_capability_contract(
            {
                **base,
                "available": False,
                "unavailable_reason": "controlled_component_too_weak",
                "unavailable_detail": (
                    f"component_rms={component_rms:.12g}, "
                    f"threshold={threshold:.12g}"
                ),
                **gate_metrics,
                "decomposition_contract": decomposition.to_dict(),
                "decomposition_history_sha256": (
                    decomposition.history_sha256
                ),
            }
        )
    if visible_component_rms <= visible_threshold:
        return _finalize_capability_contract(
            {
                **base,
                "available": False,
                "unavailable_reason": (
                    "controlled_visible_component_too_weak"
                ),
                "unavailable_detail": (
                    f"visible_component_rms={visible_component_rms:.12g}, "
                    f"threshold={visible_threshold:.12g}"
                ),
                **gate_metrics,
                "decomposition_contract": decomposition.to_dict(),
                "decomposition_history_sha256": (
                    decomposition.history_sha256
                ),
            }
        )
    if future_component_rms <= future_threshold:
        return _finalize_capability_contract(
            {
                **base,
                "available": False,
                "unavailable_reason": "controlled_future_component_too_weak",
                "unavailable_detail": (
                    f"future_component_rms={future_component_rms:.12g}, "
                    f"threshold={future_threshold:.12g}"
                ),
                **gate_metrics,
                "decomposition_contract": decomposition.to_dict(),
                "decomposition_history_sha256": (
                    decomposition.history_sha256
                ),
            }
        )
    return _finalize_capability_contract(
        {
            **base,
            "available": True,
            "unavailable_reason": None,
            "unavailable_detail": None,
            **gate_metrics,
            "decomposition_contract": decomposition.to_dict(),
            "decomposition_history_sha256": decomposition.history_sha256,
        }
    )


def apply_real_anchored_contract(
    full_baseline: np.ndarray,
    contract: Mapping[str, Any],
    *,
    alpha: float,
    context_length: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a calibration wrapper contract and return raw units plus metadata."""

    _verify_capability_contract(contract)
    capability_schema = str(contract.get("schema"))
    if capability_schema not in {
        REAL_ANCHORED_CAPABILITY_CONTRACT_SCHEMA,
        LEGACY_REAL_ANCHORED_CAPABILITY_CONTRACT_SCHEMA,
    }:
        raise ValueError("unsupported real-anchored capability contract schema")
    if contract.get("available") is not True:
        raise ValueError(
            "real-anchored capability is unavailable: "
            f"{contract.get('unavailable_reason', 'unknown')}"
        )
    raw_decomposition = contract.get("decomposition_contract")
    if not isinstance(raw_decomposition, Mapping):
        raise ValueError("capability contract has no decomposition contract")
    decomposition = AnchoredDecompositionContract.from_dict(
        raw_decomposition
    )
    if context_length is not None and int(context_length) != (
        decomposition.context_length
    ):
        raise ValueError("context_length does not match the frozen contract")
    if contract.get("decomposition_history_sha256") != (
        decomposition.history_sha256
    ):
        raise ValueError("capability and decomposition history hashes differ")
    baseline, _ = _as_2d(full_baseline, name="full_baseline")
    source_hash = _source_values_sha256(
        baseline[: decomposition.context_length]
    )
    if contract.get("source_history_sha256") != source_hash:
        raise ValueError("full baseline source history hash mismatch")
    capability_id = str(contract["capability_id"])
    member = apply_anchored_counterfactual(
        full_baseline,
        decomposition,
        capability_id=capability_id,
        alpha=alpha,
    )
    metadata = member.metadata()
    metadata.update(
        {
            "schema": capability_schema,
            "available": True,
            "capability_contract_sha256": contract[
                "capability_contract_sha256"
            ],
            "source_history_sha256": source_hash,
            "qualification_policy_id": contract.get(
                "qualification_policy_id"
            ),
            "qualification_policy_sha256": contract.get(
                "qualification_policy_sha256"
            ),
            "qualification_threshold_source": contract.get(
                "qualification_threshold_source"
            ),
            "qualification_thresholds": contract.get(
                "qualification_thresholds"
            ),
            "decomposition_history_sha256": (
                decomposition.history_sha256
            ),
            "output_units": "baseline_raw_units",
            "controlled_component": {
                "multi_seasonal": "secondary_harmonic_sum",
                "trend": "local_trend_nonlinearity",
                "time_varying_seasonality": (
                    (
                        "carrier_phase_locked_symmetric_"
                        "amplitude_modulation"
                    )
                    if decomposition.modulation_basis
                    == CONSTRAINED_AM_BASIS
                    else "carrier_amplitude_modulation_sidebands"
                ),
                "regime_switching": "history_joinpoint_level_shift",
            }[capability_id],
            "carrier_fixed": True,
            "secondary_fixed": capability_id != "multi_seasonal",
            "trend_nonlinearity_fixed": capability_id != "trend",
            "amplitude_modulation_fixed": (
                capability_id != "time_varying_seasonality"
            ),
            "regime_level_shift_fixed": (
                capability_id != "regime_switching"
            ),
            "modulation_period": decomposition.modulation_period,
            "modulation_basis": decomposition.modulation_basis,
            "modulation_extension": (
                (
                    "bounded_symmetric_carrier_amplitude_modulation"
                    if decomposition.modulation_basis
                    == CONSTRAINED_AM_BASIS
                    else "bounded_stationary_carrier_sidebands"
                )
                if decomposition.modulation_period is not None
                else None
            ),
            "regime_join_index": decomposition.regime_join_index,
            "regime_extension": (
                "constant_post_join_level"
                if decomposition.regime_join_index is not None
                else None
            ),
        }
    )
    return member.values, metadata
