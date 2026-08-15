"""Structural real-path counterfactuals fitted from authentic panels.

The univariate real-anchored track deliberately expands native records into
individual channels.  That representation is invalid for structural
capabilities: common factors and directed transfer require a synchronized
panel, hierarchy requires a declared aggregation relation, and covariate
response requires an explicitly known-future covariate path.  This module
therefore owns a separate, fail-closed structural background and contract.

Every fitted parameter and every H48 component continuation uses only the
L504 target history.  The sole exception is a covariate path declared by the
source adapter as known-future; it is an inference input, never a target.  The
observed target future remains a paired nuisance realization and is never
used to tune a contract or an availability threshold.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from cafe import protocol
from cafe.data.imputation import impute_observed_window
from cafe.data.real import RealDatasetBundle, RealSeriesRecord, load_real_dataset
from cafe.generation.normalization import standardize_hierarchy_by_context
from cafe.generation.real_anchored_dose import (
    REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA,
    additive_dose_reference,
    dose_calibration_from_policy,
    paired_minimum_separation_gate,
    resolve_contract_dose_calibration,
    standardized_channel_separations,
    validate_dose_calibration,
)
from cafe.generation.real_counterfactuals import array_sha256
from cafe.generation.real_anchored_policy import (
    HIERARCHY_FORMAL_RANK_POLICY,
    MINIMUM_FORMAL_BACKGROUND_COUNT,
    MINIMUM_FORMAL_PANEL_DIMENSION,
    MINIMUM_TWO_CHANNEL_SENSITIVITY_BACKGROUND_COUNT,
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
    STRUCTURAL_INPUT_ABLATION_POLICY,
    TWO_CHANNEL_PANEL_POLICY,
)


STRUCTURAL_BACKGROUND_SCHEMA = "cafe.structural_real_background.v1"
STRUCTURAL_BACKGROUND_BANK_SCHEMA = "cafe.structural_real_background_bank.v1"
STRUCTURAL_CONTRACT_SCHEMA = "cafe.structural_real_contract.v2"
STRUCTURAL_CAPABILITY_ROW_SCHEMA = "cafe.structural_real_capability_row.v2"
STRUCTURAL_MASTER_SCHEMA = "cafe.structural_real_counterfactual_master.v2"
STRUCTURAL_ABLATION_SCHEMA = "cafe.structural_input_ablation.v1"
STRUCTURAL_AVAILABILITY_SCHEMA = "cafe.structural_real_availability.v3"
STRUCTURAL_DONOR_COMMITMENT_SCHEMA = (
    "cafe.structural_donor_commitment_manifest.v2"
)
STRUCTURAL_DONOR_COMMITMENT_ENTRY_SCHEMA = (
    "cafe.structural_donor_commitment_entry.v2"
)
STRUCTURAL_DONOR_COMMITMENT_POLICY = (
    "calibration_frozen_structural_l336_per_channel_dose_commitment_v2"
)
LEGACY_STRUCTURAL_CONTRACT_SCHEMA = "cafe.structural_real_contract.v1"
LEGACY_STRUCTURAL_AVAILABILITY_SCHEMA = "cafe.structural_real_availability.v2"
STRUCTURAL_REFERENCE_BANK_ID = "cafe.structural_reference_bank.2026-08-v1"

STRUCTURAL_CAPABILITIES = (
    "common_factor",
    "cross_series_dependence",
    "covariate_response",
    "hierarchical_coherence",
)
STRUCTURAL_ALPHAS = (1.2, 1.4, 1.6, 1.8, 2.0)
FORMAL_PANEL_MINIMUM_DIMENSION = MINIMUM_FORMAL_PANEL_DIMENSION
SENSITIVITY_PANEL_DIMENSION = 2
MINIMUM_SENSITIVITY_BACKGROUND_COUNT = (
    MINIMUM_TWO_CHANNEL_SENSITIVITY_BACKGROUND_COUNT
)
MINIMUM_COMPONENT_RMS_RATIO = 0.01


# These gates are protocol constants selected before inspecting evaluation
# origins.  A contract records both the value and its source.  Dataset-specific
# q10/q90 values, realized pass rates, and held-out targets cannot update them.
STRUCTURAL_REFERENCE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "common_min_excess_pca_share": {
        "value": 0.02,
        "source": (
            f"{STRUCTURAL_REFERENCE_BANK_ID}:common_factor_identifiability"
        ),
    },
    "common_min_loading_stability_cosine": {
        "value": 0.75,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:split_loading_stability",
    },
    "common_min_loading_relative_magnitude": {
        "value": 0.25,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:dense_loading_support",
    },
    "common_min_one_step_holdout_r2": {
        "value": 0.0,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:factor_forecastability",
    },
    "cross_min_corrected_incremental_r2": {
        "value": 0.0025,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:directed_transfer_gain",
    },
    "cross_min_fold_driver_agreement": {
        "value": 0.5,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:edge_stability",
    },
    "cross_max_fold_lag_deviation": {
        "value": 2.0,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:edge_stability",
    },
    "cross_max_lag": {
        "value": 24,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:l96_identifiable_lag_bank",
    },
    "covariate_min_excess_incremental_r2": {
        "value": 0.0025,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:known_future_gain",
    },
    "covariate_min_coefficient_stability_cosine": {
        "value": 0.50,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:response_stability",
    },
    "covariate_fixed_null_shift": {
        "value": 53,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:time_shift_null",
    },
    "hierarchy_min_contrast_holdout_r2": {
        "value": 0.0,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:contrast_forecastability",
    },
    "component_minimum_rms_ratio": {
        "value": MINIMUM_COMPONENT_RMS_RATIO,
        "source": f"{STRUCTURAL_REFERENCE_BANK_ID}:effect_numerical_stability",
    },
}


def structural_threshold_contract() -> dict[str, Any]:
    """Return an immutable-by-convention copy of the predeclared gate bank."""

    return {
        "reference_bank_id": STRUCTURAL_REFERENCE_BANK_ID,
        "selection_policy": QUALIFICATION_THRESHOLD_SOURCE_POLICY,
        "evaluation_origin_adaptation_allowed": False,
        "thresholds": copy.deepcopy(STRUCTURAL_REFERENCE_THRESHOLDS),
    }


def _threshold(name: str) -> float:
    return float(STRUCTURAL_REFERENCE_THRESHOLDS[name]["value"])


_QUALIFICATION_THRESHOLD_NAMES = {
    "common_factor": (
        "common_min_excess_pca_share",
        "common_min_loading_stability_cosine",
        "common_min_loading_relative_magnitude",
        "common_min_one_step_holdout_r2",
        "component_minimum_rms_ratio",
    ),
    "cross_series_dependence": (
        "cross_min_corrected_incremental_r2",
        "cross_min_fold_driver_agreement",
        "cross_max_fold_lag_deviation",
        "cross_max_lag",
        "component_minimum_rms_ratio",
    ),
    "covariate_response": (
        "covariate_min_excess_incremental_r2",
        "covariate_min_coefficient_stability_cosine",
        "covariate_fixed_null_shift",
        "component_minimum_rms_ratio",
    ),
    "hierarchical_coherence": (
        "hierarchy_min_contrast_holdout_r2",
        "component_minimum_rms_ratio",
    ),
}


def _qualification_thresholds(capability_id: str) -> dict[str, float]:
    return {
        name: _threshold(name)
        for name in _QUALIFICATION_THRESHOLD_NAMES[capability_id]
    }


def _qualification_policy_id(capability_id: str) -> str:
    return f"cafe.structural.{capability_id}.qualification.reference_bank.v1"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _array_sha256(values: np.ndarray, *, domain: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(f"cafe.{domain}.float64.v1\0".encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = {
        str(key): value
        for key, value in contract.items()
        if key != "contract_sha256"
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _time_by_target(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError("structural target must have shape [time] or [target,time]")
    return array.T


def _impute_history_matrix(
    values: np.ndarray,
    *,
    minimum_observed_fraction: float,
) -> tuple[np.ndarray | None, list[float]]:
    matrix = np.asarray(values, dtype=float)
    imputed: list[np.ndarray] = []
    fractions: list[float] = []
    for column in range(matrix.shape[1]):
        result, observed = impute_observed_window(
            matrix[:, column],
            minimum_observed_fraction=minimum_observed_fraction,
        )
        if result is None:
            return None, fractions
        imputed.append(np.asarray(result, dtype=float))
        fractions.append(float(observed))
    return np.column_stack(imputed), fractions


def _normalize_panel(
    history: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = history[-protocol.REAL_ANCHORED_CONTEXT_LENGTH :]
    center = np.mean(reference, axis=0)
    scale = np.std(reference, axis=0)
    if not np.isfinite(scale).all() or np.any(scale <= 1e-10):
        raise ValueError("structural panel contains an uninformative target")
    return (
        (history - center[None, :]) / scale[None, :],
        (future - center[None, :]) / scale[None, :],
        center,
        scale,
    )


def _normalize_known_future_covariates(
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(values, dtype=float)
    reference = matrix[
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
    ]
    normalized = matrix.copy()
    centers: list[float] = []
    scales: list[float] = []
    kinds: list[str] = []
    for column in range(matrix.shape[1]):
        history_column = reference[:, column]
        unique = set(np.unique(history_column).tolist())
        if unique.issubset({0.0, 1.0}):
            center = 0.0
            scale = 1.0
            kind = "binary_passthrough"
        else:
            center = float(np.mean(history_column))
            scale = float(np.std(history_column))
            if not math.isfinite(scale) or scale <= 1e-10:
                # A constant known-future column carries no estimable response.
                scale = 1.0
                kind = "constant_continuous"
            else:
                kind = "continuous_history_zscore"
            normalized[:, column] = (matrix[:, column] - center) / scale
        centers.append(center)
        scales.append(scale)
        kinds.append(kind)
    return normalized, {
        "scope": "shared_unmodified_real_l336_history",
        "center_by_covariate": centers,
        "scale_by_covariate": scales,
        "kind_by_covariate": kinds,
        "future_statistics_used": False,
    }


def _hierarchy_window(
    record: RealSeriesRecord,
    *,
    start: int,
    training_length: int,
    minimum_observed_fraction: float,
) -> tuple[dict[str, Any] | None, str | None]:
    if record.hierarchy_values is None or record.hierarchy_kind is None:
        return None, None
    native = np.asarray(record.hierarchy_values, dtype=float)
    if native.ndim != 2 or native.shape[1] < training_length:
        return None, "hierarchy_not_time_aligned_with_target"
    native = native[:, :training_length]
    raw = native[
        :,
        start : start + protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
    ].T
    if raw.shape[0] != protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH:
        return None, "hierarchy_window_too_short"
    history_raw, fractions = _impute_history_matrix(
        raw[: protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH],
        minimum_observed_fraction=minimum_observed_fraction,
    )
    if history_raw is None:
        return None, "hierarchy_history_missing"
    future_raw = raw[protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH :]
    if not np.isfinite(future_raw).all():
        return None, "hierarchy_future_missing"
    if record.hierarchy_kind == "children_only_additive":
        if history_raw.shape[1] < 2:
            return None, "hierarchy_requires_two_children"
        history_children = history_raw
        future_children = future_raw
        history_nodes = np.column_stack(
            [np.sum(history_children, axis=1), history_children]
        )
        future_nodes = np.column_stack(
            [np.sum(future_children, axis=1), future_children]
        )
        parent_source = "constructed_exact_sum_of_declared_children"
    elif record.hierarchy_kind == "additive_first":
        if history_raw.shape[1] < 3:
            return None, "hierarchy_requires_parent_and_two_children"
        history_nodes = history_raw
        future_nodes = future_raw
        parent_source = "observed_declared_parent"
    else:
        return None, "unsupported_hierarchy_kind"
    children = list(range(1, history_nodes.shape[1]))
    if (
        record.hierarchy_kind == "children_only_additive"
        and len(record.channel_ids) == len(children)
    ):
        node_ids = ["constructed_parent_sum", *record.channel_ids]
    else:
        node_ids = [f"hierarchy_node_{index}" for index in range(history_nodes.shape[1])]
    history_residual = history_nodes[:, 0] - np.sum(
        history_nodes[:, children], axis=1
    )
    future_residual = future_nodes[:, 0] - np.sum(
        future_nodes[:, children], axis=1
    )
    tolerance = 1e-8 * max(float(np.std(history_nodes[:, 0])), 1.0)
    if (
        float(np.max(np.abs(history_residual))) > tolerance
        or float(np.max(np.abs(future_residual))) > tolerance
    ):
        return None, "declared_hierarchy_is_not_additively_coherent"
    nodes = np.vstack([history_nodes, future_nodes])
    standardized = standardize_hierarchy_by_context(
        nodes[
            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
            - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
        ],
        protocol.REAL_ANCHORED_CONTEXT_LENGTH,
    )
    # Recover the shared scale/centers explicitly for component conversion and
    # raw-domain support audits.
    reference = history_nodes[-protocol.REAL_ANCHORED_CONTEXT_LENGTH :]
    center = np.mean(reference, axis=0)
    shared_scale = float(np.std(reference[:, 0]))
    if shared_scale <= 1e-6:
        shared_scale = float(np.mean(np.std(reference, axis=0)))
    if shared_scale <= 1e-6:
        return None, "hierarchy_uninformative"
    standardized_full = (nodes - center[None, :]) / shared_scale
    return {
        "kind": record.hierarchy_kind,
        "parent_source": parent_source,
        "node_count": int(nodes.shape[1]),
        "parent_index": 0,
        "child_indices": children,
        "node_ids": node_ids,
        "source_channel_ids": list(record.channel_ids),
        "aggregation_law": "node_0=sum(nodes_1_to_end)",
        "fit_observed_fraction_by_source_node": fractions,
        "standardization": {
            "scope": "hierarchy_shared_l336_parent_scale",
            "center_by_node": center.tolist(),
            "shared_scale": shared_scale,
        },
        "history_coherence_max_abs": float(
            np.max(np.abs(history_residual))
        ),
        "future_coherence_max_abs": float(np.max(np.abs(future_residual))),
        "raw_minimum_by_node": np.min(nodes, axis=0).tolist(),
        "raw_source_window_sha256": _array_sha256(
            nodes,
            domain="structural_hierarchy_raw_source",
        ),
        "target": standardized_full[
            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
            - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
        ].tolist(),
        "target_sha256": _array_sha256(
            standardized,
            domain="structural_hierarchy_visible",
        ),
        "_decomposition_history": standardized_full[
            : protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        ].tolist(),
        "_raw_source_window": nodes.tolist(),
    }, None


def build_structural_real_anchored_backgrounds(
    dataset: protocol.DatasetSpec,
    *,
    source_root: Path,
    maximum_backgrounds: int,
    sample_seed: int = protocol.REAL_ANCHORED_SAMPLE_SEED,
    minimum_observed_fraction: float = 0.90,
    real_bundle: RealDatasetBundle | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample synchronized authentic L504+H48 structural backgrounds.

    Sampling is per native record, never per expanded channel.  All attached
    covariates and hierarchy nodes use the identical origin and official-tail
    exclusion as the target panel.
    """

    if maximum_backgrounds < 1:
        raise ValueError("maximum_backgrounds must be positive")
    asset_path = source_root / dataset.asset_name
    if real_bundle is None:
        real_bundle = load_real_dataset(dataset.real_data_adapter, asset_path)
    source_records = [(record.item_id, record.values) for record in real_bundle.records]
    training_records, official_holdout = protocol.training_records_for_anchor_sampling(
        dataset,
        real_bundle.frequency,
        source_records,
    )
    training_by_item = {
        str(item_id): np.asarray(values, dtype=float)
        for item_id, values in training_records
    }
    record_by_item = {record.item_id: record for record in real_bundle.records}
    candidates: list[tuple[str, int, int]] = []
    for item_id, values in training_records:
        length = int(np.asarray(values).shape[-1])
        for lower, upper in protocol.nonoverlapping_strata(
            length,
            window_length=protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
        ):
            candidates.append((str(item_id), lower, upper))
    rng = np.random.default_rng(
        protocol.stable_seed(dataset.dataset_id, sample_seed, base=sample_seed)
    )
    order = rng.permutation(len(candidates)) if candidates else np.asarray([], dtype=int)
    target_count = min(int(maximum_backgrounds), len(candidates))
    backgrounds: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for candidate_index in order:
        item_id, lower, upper = candidates[int(candidate_index)]
        start = int(rng.integers(lower, upper + 1)) if upper > lower else int(lower)
        native = training_by_item[item_id]
        target_window = _time_by_target(
            native[..., start : start + protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH]
        )
        if target_window.shape[0] != protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH:
            reasons["target_window_too_short"] += 1
            continue
        fit_history, observed_fractions = _impute_history_matrix(
            target_window[: protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH],
            minimum_observed_fraction=max(0.90, minimum_observed_fraction),
        )
        if fit_history is None:
            reasons["target_history_missing"] += 1
            continue
        future = target_window[protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH :]
        if not np.isfinite(future).all():
            reasons["target_future_missing"] += 1
            continue
        try:
            normalized_history, normalized_future, center, scale = _normalize_panel(
                fit_history,
                future,
            )
        except ValueError:
            reasons["target_uninformative"] += 1
            continue
        visible_target = np.vstack(
            [
                normalized_history[-protocol.REAL_ANCHORED_CONTEXT_LENGTH :],
                normalized_future,
            ]
        )
        feature_history = normalized_history[
            -protocol.REAL_CALIBRATION_CONTEXT_LENGTH :, 0
        ]
        period_policy = protocol.calibration_period_policy(
            real_bundle.frequency,
            feature_history,
        )
        mase_policy = protocol.mase_scale_policy(
            visible_target,
            season_length=int(period_policy["mase_period"]),
        )
        training_length = int(native.shape[-1])
        record = record_by_item[item_id]

        covariate_payload: dict[str, Any] | None = None
        covariate_reason: str | None = None
        if record.covariates is not None:
            covariates = np.asarray(record.covariates, dtype=float)
            if record.covariate_kind != "known_future":
                covariate_reason = "covariates_not_declared_known_future"
            elif covariates.ndim != 2 or covariates.shape[0] < training_length:
                covariate_reason = "covariates_not_time_aligned_with_target"
            else:
                covariate_window = covariates[
                    :training_length
                ][start : start + protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH]
                if (
                    covariate_window.shape[0]
                    != protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
                    or not np.isfinite(covariate_window).all()
                ):
                    covariate_reason = "known_future_covariate_window_incomplete"
                else:
                    normalized_covariates, covariate_normalization = (
                        _normalize_known_future_covariates(covariate_window)
                    )
                    covariate_payload = {
                        "kind": "known_future",
                        "column_names": (
                            list(record.covariate_names)
                            if len(record.covariate_names)
                            == normalized_covariates.shape[1]
                            else [
                                f"covariate_{index}"
                                for index in range(
                                    normalized_covariates.shape[1]
                                )
                            ]
                        ),
                        "normalization": covariate_normalization,
                        "target": normalized_covariates[
                            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                            - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
                        ].tolist(),
                        "target_sha256": _array_sha256(
                            normalized_covariates[
                                protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                                - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
                            ],
                            domain="structural_known_future_visible",
                        ),
                        "raw_source_window_sha256": _array_sha256(
                            covariate_window,
                            domain="structural_known_future_raw_source",
                        ),
                        "_source_window": normalized_covariates.tolist(),
                    }

        hierarchy_payload, hierarchy_reason = _hierarchy_window(
            record,
            start=start,
            training_length=training_length,
            minimum_observed_fraction=max(0.90, minimum_observed_fraction),
        )
        origin = start + protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        target_dim = int(visible_target.shape[1])
        if target_dim >= FORMAL_PANEL_MINIMUM_DIMENSION:
            panel_role = "formal_main_candidate"
        elif target_dim == SENSITIVITY_PANEL_DIMENSION:
            panel_role = "d2_sensitivity_only"
        else:
            panel_role = "not_a_panel"
        background_id = (
            f"{protocol.safe_id(dataset.dataset_id)}__{protocol.safe_id(item_id)}"
            f"__panel__o{origin}"
        )
        backgrounds.append(
            {
                "schema_version": STRUCTURAL_BACKGROUND_SCHEMA,
                "background_id": background_id,
                "dataset_id": dataset.dataset_id,
                "config_id": dataset.config_id,
                "task_view_id": dataset.task_view_id,
                "item_id": item_id,
                "structural_group_id": record.structural_group_id,
                "channel_ids": (
                    list(record.channel_ids)
                    if record.channel_ids
                    else [f"target_{index}" for index in range(target_dim)]
                ),
                "decomposition_start": start,
                "context_start": (
                    origin - protocol.REAL_ANCHORED_CONTEXT_LENGTH
                ),
                "forecast_origin": origin,
                "decomposition_context_length": (
                    protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                ),
                "context_length": protocol.REAL_ANCHORED_CONTEXT_LENGTH,
                "horizon": protocol.HORIZON,
                "target_dim": target_dim,
                "panel_contract": {
                    "synchronized_native_record": True,
                    "dimension": target_dim,
                    "formal_minimum_dimension": FORMAL_PANEL_MINIMUM_DIMENSION,
                    "formal_main_eligible": bool(
                        target_dim >= FORMAL_PANEL_MINIMUM_DIMENSION
                    ),
                    "sensitivity_only": bool(
                        target_dim == SENSITIVITY_PANEL_DIMENSION
                    ),
                    "role": panel_role,
                    "two_channel_policy": TWO_CHANNEL_PANEL_POLICY,
                    "independent_items_were_combined": False,
                },
                "target": visible_target.tolist(),
                "target_sha256": _array_sha256(
                    visible_target,
                    domain="structural_visible_target",
                ),
                "raw_source_target_window_sha256": _array_sha256(
                    target_window,
                    domain="structural_raw_source_target",
                ),
                "decomposition_history_sha256": _array_sha256(
                    normalized_history,
                    domain="structural_decomposition_history",
                ),
                "future_sha256": _array_sha256(
                    normalized_future,
                    domain="structural_real_future",
                ),
                "target_standardization": {
                    "scope": "shared_unmodified_real_l336_history",
                    "center_by_target": center.tolist(),
                    "scale_by_target": scale.tolist(),
                    "member_specific": False,
                },
                "fit_observed_fraction_by_target": observed_fractions,
                "future_observed_fraction": 1.0,
                "frequency": real_bundle.frequency,
                "season_length": int(period_policy["calendar_season_length"]),
                **period_policy,
                "mase_period": int(period_policy["mase_period"]),
                "mase_scale": float(mase_policy["scale"]),
                "mase_scale_by_target": list(mase_policy["scale_by_target"]),
                "mase_scale_effective_period_by_target": list(
                    mase_policy["effective_period_by_target"]
                ),
                "mase_scale_fallback_target_indices": list(
                    mase_policy["fallback_target_indices"]
                ),
                "mase_scale_policy": str(mase_policy["policy"]),
                "known_future_covariates": covariate_payload,
                "known_future_covariate_unavailable_reason": covariate_reason,
                "hierarchy": hierarchy_payload,
                "hierarchy_unavailable_reason": hierarchy_reason,
                "threshold_contract": structural_threshold_contract(),
                "official_holdout": dict(official_holdout),
                "target_future_used_for_contract_fit": False,
                "_decomposition_target": normalized_history.tolist(),
            }
        )
        if len(backgrounds) >= target_count:
            break
    return backgrounds, {
        "schema_version": STRUCTURAL_BACKGROUND_BANK_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "official_holdout": official_holdout,
        "source_window_length": protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
        "decomposition_context_length": (
            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        ),
        "model_context_length": protocol.REAL_ANCHORED_CONTEXT_LENGTH,
        "horizon": protocol.HORIZON,
        "sampling_unit": "native_synchronized_record_window",
        "expanded_channel_sampling_prohibited": True,
        "candidate_count": len(candidates),
        "accepted_background_count": len(backgrounds),
        "requested_background_limit": int(maximum_backgrounds),
        "rejection_reason_counts": dict(sorted(reasons.items())),
        "sample_seed": int(sample_seed),
        "source_asset_files": [str(path) for path in real_bundle.asset_files],
        "threshold_contract": structural_threshold_contract(),
    }


def public_structural_background(background: Mapping[str, Any]) -> dict[str, Any]:
    """Remove calibration-only arrays while preserving structural semantics."""

    result = {
        str(key): copy.deepcopy(value)
        for key, value in background.items()
        if not str(key).startswith("_")
    }
    covariates = result.get("known_future_covariates")
    if isinstance(covariates, dict):
        covariates.pop("_source_window", None)
    hierarchy = result.get("hierarchy")
    if isinstance(hierarchy, dict):
        hierarchy.pop("_decomposition_history", None)
        hierarchy.pop("_raw_source_window", None)
    return result


def _ridge_coefficients(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=float) * 1e-4
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )


def _safe_r2(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((actual - prediction) ** 2) / denominator)


def _fit_ar1(values: np.ndarray) -> tuple[float, float]:
    series = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones(series.size - 1), series[:-1]])
    coefficients = _ridge_coefficients(design, series[1:])
    return float(coefficients[0]), float(np.clip(coefficients[1], -0.98, 0.98))


def _ar1_forecast(values: np.ndarray, horizon: int) -> tuple[np.ndarray, float, float]:
    intercept, persistence = _fit_ar1(values)
    forecast = np.empty(horizon, dtype=float)
    state = float(values[-1])
    for step in range(horizon):
        state = intercept + persistence * state
        forecast[step] = state
    return forecast, intercept, persistence


def _ar1_one_step_holdout_r2(values: np.ndarray) -> float:
    series = np.asarray(values, dtype=float)
    split = max(48, int(math.floor(0.75 * series.size)))
    intercept, persistence = _fit_ar1(series[:split])
    actual = series[split:]
    prediction = intercept + persistence * series[split - 1 : -1]
    return _safe_r2(actual, prediction)


def _fit_ar_lags(values: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    lag_values = tuple(sorted({int(value) for value in lags}))
    start = max(lag_values)
    indexes = np.arange(start, series.size)
    design = np.column_stack(
        [np.ones(indexes.size)]
        + [series[indexes - lag] for lag in lag_values]
    )
    return _ridge_coefficients(design, series[indexes])


def _ar_lag_one_step_r2(values: np.ndarray, lags: Sequence[int]) -> float:
    series = np.asarray(values, dtype=float)
    lag_values = tuple(sorted({int(value) for value in lags}))
    split = max(max(lag_values) + 24, int(math.floor(0.75 * series.size)))
    coefficients = _fit_ar_lags(series[:split], lag_values)
    indexes = np.arange(split, series.size)
    design = np.column_stack(
        [np.ones(indexes.size)]
        + [series[indexes - lag] for lag in lag_values]
    )
    return _safe_r2(series[indexes], design @ coefficients)


def _selected_state_forecast(
    values: np.ndarray,
    horizon: int,
    *,
    period: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a history-only stable AR/seasonal state extension."""

    series = np.asarray(values, dtype=float)
    seasonal = int(np.clip(period, 2, min(168, series.size // 3)))
    candidates = tuple(
        dict.fromkeys(
            (
                (1,),
                (1, 2),
                (seasonal,),
                tuple(sorted({1, seasonal})),
                tuple(sorted({1, 2, seasonal})),
            )
        )
    )
    scored = [
        {
            "lags": list(lags),
            "one_step_holdout_r2": _ar_lag_one_step_r2(series, lags),
        }
        for lags in candidates
    ]
    selected = max(
        scored,
        key=lambda row: (
            float(row["one_step_holdout_r2"]),
            -len(row["lags"]),
        ),
    )
    lags = tuple(int(value) for value in selected["lags"])
    coefficients = _fit_ar_lags(series, lags)
    extended = series.astype(float).tolist()
    center = float(np.median(series))
    scale = max(float(np.std(series)), 1e-6)
    lower = center - 6.0 * scale
    upper = center + 6.0 * scale
    clipped = 0
    for _step in range(int(horizon)):
        state = float(coefficients[0]) + sum(
            float(coefficient) * float(extended[-lag])
            for lag, coefficient in zip(lags, coefficients[1:], strict=True)
        )
        bounded = float(np.clip(state, lower, upper))
        clipped += int(bounded != state)
        extended.append(bounded)
    forecast = np.asarray(extended[-int(horizon):], dtype=float)
    metadata = {
        "policy": "history_only_holdout_selected_bounded_ar_seasonal_v1",
        "candidate_scores": scored,
        "selected_lags": list(lags),
        "selected_one_step_holdout_r2": float(
            selected["one_step_holdout_r2"]
        ),
        "coefficients": coefficients.tolist(),
        "seasonal_period": seasonal,
        "forecast_clip_bound_standard_deviations": 6.0,
        "forecast_clipped_step_count": clipped,
        "target_future_used": False,
    }
    return forecast, metadata


def _principal_loading(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(values, dtype=float)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _left, singular, right = np.linalg.svd(centered, full_matrices=False)
    loading = right[0].copy()
    pivot = int(np.argmax(np.abs(loading)))
    if loading[pivot] < 0.0:
        loading *= -1.0
    scores = centered @ loading
    total = float(np.sum(singular * singular))
    share = float(singular[0] ** 2 / max(total, 1e-12))
    return loading, scores, share


def _component_gate(
    component: np.ndarray,
    *,
    history_length: int,
) -> dict[str, Any]:
    values = np.asarray(component, dtype=float)
    history_rms = float(
        np.sqrt(
            np.mean(
                values[
                    history_length - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
                    history_length
                ]
                ** 2
            )
        )
    )
    future_rms = float(np.sqrt(np.mean(values[history_length:] ** 2)))
    minimum = _threshold("component_minimum_rms_ratio")
    return {
        "history_rms": history_rms,
        "future_rms": future_rms,
        "minimum_rms": minimum,
        "history_passed": bool(history_rms >= minimum),
        "future_passed": bool(future_rms >= minimum),
        "passed": bool(history_rms >= minimum and future_rms >= minimum),
        "reference_scale": "per_target_standardized_real_l336_unit_scale",
    }


def _structural_affected_channel_indices(
    capability_id: str,
    background: Mapping[str, Any],
    fit_diagnostics: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return the channels whose controlled component defines dose strength."""

    if capability_id == "common_factor":
        raw = fit_diagnostics.get("nondegenerate_loading_indices", ())
    elif capability_id == "cross_series_dependence":
        raw = fit_diagnostics.get("responders", ())
    elif capability_id == "covariate_response":
        raw = fit_diagnostics.get("eligible_target_indices", ())
    elif capability_id == "hierarchical_coherence":
        hierarchy = background.get("hierarchy")
        raw = (
            ()
            if not isinstance(hierarchy, Mapping)
            else hierarchy.get("child_indices", ())
        )
    else:
        raise ValueError(f"unsupported structural dose capability {capability_id}")
    affected = tuple(int(value) for value in raw)
    if len(affected) != len(set(affected)) or any(value < 0 for value in affected):
        raise ValueError("structural dose channels must be unique/non-negative")
    return affected


def _structural_dose_reference(
    background: Mapping[str, Any],
    *,
    capability_id: str,
    component: np.ndarray,
    fit_diagnostics: Mapping[str, Any],
    evidence_role: str | None,
) -> dict[str, Any] | None:
    """Build reference-only standardized L168/H48 unit-gain evidence."""

    if evidence_role is None:
        return None
    affected = _structural_affected_channel_indices(
        capability_id,
        background,
        fit_diagnostics,
    )
    if not affected:
        return None
    history_by_channel, future_by_channel = standardized_channel_separations(
        np.asarray(component, dtype=float),
        context_length=protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH,
    )
    if max(affected) >= history_by_channel.size:
        raise ValueError("structural dose channel exceeds component dimension")
    history = float(np.mean(history_by_channel[list(affected)]))
    future = float(np.mean(future_by_channel[list(affected)]))
    if min(history, future) <= 0.0:
        return None
    return additive_dose_reference(
        capability_id=capability_id,
        background_id=str(background["background_id"]),
        unit_gain_history_separation=history,
        unit_gain_future_separation=future,
        affected_channel_indices=affected,
        known_future_covariate_path_used=(
            capability_id == "covariate_response"
        ),
        evidence_role=evidence_role,
    )


def _base_contract(
    background: Mapping[str, Any],
    *,
    capability_id: str,
    component: np.ndarray,
    fit_diagnostics: Mapping[str, Any],
    formal_main_eligible: bool,
    sensitivity_eligible: bool,
    mandatory_input_ablation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    component_values = np.asarray(component, dtype=float)
    qualification_only_eligible = bool(
        capability_id == "hierarchical_coherence"
        and fit_diagnostics.get("qualification_passed") is True
    )
    evidence_role = (
        "formal"
        if formal_main_eligible
        else "sensitivity"
        if sensitivity_eligible
        else "qualification_only"
        if qualification_only_eligible
        else None
    )
    dose_reference = _structural_dose_reference(
        background,
        capability_id=capability_id,
        component=component_values,
        fit_diagnostics=fit_diagnostics,
        evidence_role=evidence_role,
    )
    contract: dict[str, Any] = {
        "schema_version": STRUCTURAL_CONTRACT_SCHEMA,
        "capability_id": capability_id,
        "background_id": str(background["background_id"]),
        "source_history_sha256": str(background["decomposition_history_sha256"]),
        "source_target_dim": int(background["target_dim"]),
        "component_dim": int(component_values.shape[1]),
        "fit_context_length": protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH,
        "model_context_length": protocol.REAL_ANCHORED_CONTEXT_LENGTH,
        "horizon": protocol.HORIZON,
        "intervention_law": "X_alpha=X+(alpha-1)*M_hat",
        # Reference fitting measures unit-gain response only.  The formal grid
        # is selected later from the independent reference bank.
        "alpha_grid": [],
        "dose_grid_status": "reference_mapping_pending",
        "dose_design_reference": dose_reference,
        "component": component_values.tolist(),
        "component_sha256": _array_sha256(
            component_values,
            domain=f"structural_{capability_id}_component",
        ),
        "component_gate": _component_gate(
            component_values,
            history_length=protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH,
        ),
        "fit_diagnostics": copy.deepcopy(dict(fit_diagnostics)),
        "threshold_contract": structural_threshold_contract(),
        "qualification_thresholds": _qualification_thresholds(capability_id),
        "qualification_policy_id": _qualification_policy_id(capability_id),
        "qualification_threshold_source": (
            QUALIFICATION_THRESHOLD_SOURCE_POLICY
        ),
        "thresholds_adapted_from_evaluation_origins": False,
        "target_future_used_for_fit": False,
        "formal_main_eligible": bool(formal_main_eligible),
        "sensitivity_eligible": bool(sensitivity_eligible),
        "generation_eligible": bool(formal_main_eligible),
        "ranking_eligible": bool(formal_main_eligible),
        "mandatory_input_ablation": (
            None
            if mandatory_input_ablation is None
            else copy.deepcopy(dict(mandatory_input_ablation))
        ),
    }
    contract["contract_sha256"] = _contract_sha256(contract)
    return contract


def fit_common_factor_contract(background: Mapping[str, Any]) -> dict[str, Any]:
    history = np.asarray(background["_decomposition_target"], dtype=float)
    dimension = history.shape[1]
    if dimension < SENSITIVITY_PANEL_DIMENSION:
        raise ValueError("common factor requires a synchronized panel")
    loading, scores, share = _principal_loading(history)
    first_loading, _first_scores, _first_share = _principal_loading(
        history[: history.shape[0] // 2]
    )
    second_loading, _second_scores, _second_share = _principal_loading(
        history[history.shape[0] // 2 :]
    )
    stability = float(abs(np.dot(first_loading, second_loading)))
    relative = np.abs(loading) / max(float(np.max(np.abs(loading))), 1e-12)
    nondegenerate = np.flatnonzero(
        relative >= _threshold("common_min_loading_relative_magnitude")
    )
    factor_future, factor_forecast = _selected_state_forecast(
        scores,
        protocol.HORIZON,
        period=int(round(float(background.get("season_length", 24)))),
    )
    component = np.vstack(
        [
            scores[:, None] * loading[None, :],
            factor_future[:, None] * loading[None, :],
        ]
    )
    holdout_r2 = _ar1_one_step_holdout_r2(scores)
    minimum_share = 1.0 / float(dimension) + _threshold(
        "common_min_excess_pca_share"
    )
    protected = int(nondegenerate[0]) if nondegenerate.size else 0
    gate = {
        "top_factor_share": share,
        "isotropic_share": 1.0 / float(dimension),
        "minimum_top_factor_share": minimum_share,
        "loading_split_cosine": stability,
        "minimum_loading_split_cosine": _threshold(
            "common_min_loading_stability_cosine"
        ),
        "nondegenerate_loading_indices": nondegenerate.tolist(),
        "minimum_nondegenerate_loading_count": 3,
        "factor_one_step_holdout_r2": holdout_r2,
        "minimum_factor_one_step_holdout_r2": _threshold(
            "common_min_one_step_holdout_r2"
        ),
        "factor_future_forecast": factor_forecast,
        "loadings": loading.tolist(),
        "protected_target_index": protected,
    }
    panel = background["panel_contract"]
    observable_passed = bool(
        share >= minimum_share
        and stability >= _threshold("common_min_loading_stability_cosine")
        and nondegenerate.size >= min(3, dimension)
        and holdout_r2 >= _threshold("common_min_one_step_holdout_r2")
    )
    component_passed = bool(_component_gate(component, history_length=504)["passed"])
    formal = bool(panel["formal_main_eligible"] and observable_passed and component_passed)
    sensitivity = bool(
        panel["sensitivity_only"] and observable_passed and component_passed
    )
    gate["observable_passed"] = observable_passed
    return _base_contract(
        background,
        capability_id="common_factor",
        component=component,
        fit_diagnostics=gate,
        formal_main_eligible=formal,
        sensitivity_eligible=sensitivity,
        mandatory_input_ablation={
            "required": True,
            "evaluation_table": "real_anchored_input_ablation",
            "assessed_target_indices": [protected],
            "ablated_input_indices": [
                index for index in range(dimension) if index != protected
            ],
            "replacement_policy": (
                "distinct_frozen_background_donor_affine_matched_on_l336"
            ),
            "target_future_unchanged": True,
            "excluded_from_primary_score": True,
            "reported_as_separate_attribution_audit": True,
            "attribution_policy": STRUCTURAL_INPUT_ABLATION_POLICY,
        },
    )


def _incremental_gain(
    matrix: np.ndarray,
    *,
    source: int,
    destination: int,
    lag: int,
) -> float:
    values = np.asarray(matrix, dtype=float)
    start = max(1, int(lag))
    response = values[start:, destination]
    own = values[start - 1 : -1, destination]
    driver = values[start - lag : values.shape[0] - lag, source]
    split = max(36, int(math.floor(0.70 * response.size)))
    if response.size - split < 12:
        return 0.0
    own_design = np.column_stack([np.ones(response.size), own])
    full_design = np.column_stack([np.ones(response.size), own, driver])
    own_coefficients = _ridge_coefficients(
        own_design[:split], response[:split]
    )
    full_coefficients = _ridge_coefficients(
        full_design[:split], response[:split]
    )
    actual = response[split:]
    own_error = float(
        np.sum((actual - own_design[split:] @ own_coefficients) ** 2)
    )
    if own_error <= 1e-12:
        return 0.0
    full_error = float(
        np.sum((actual - full_design[split:] @ full_coefficients) ** 2)
    )
    return float(np.clip((own_error - full_error) / own_error, 0.0, 1.0))


def _select_directed_edge(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    max_lag = int(_threshold("cross_max_lag"))
    minimum_gain = float(_threshold("cross_min_corrected_incremental_r2"))
    minimum_responder_count = min(2, values.shape[1] - 1)
    candidates: list[dict[str, Any]] = []
    for source in range(values.shape[1]):
        destinations = [index for index in range(values.shape[1]) if index != source]
        for lag in range(1, max_lag + 1):
            forward = [
                _incremental_gain(
                    values,
                    source=source,
                    destination=destination,
                    lag=lag,
                )
                for destination in destinations
            ]
            reverse = [
                _incremental_gain(
                    values[::-1],
                    source=source,
                    destination=destination,
                    lag=lag,
                )
                for destination in destinations
            ]
            corrected = [
                max(float(left) - float(right), 0.0)
                if math.isfinite(left) and math.isfinite(right)
                else 0.0
                for left, right in zip(forward, reverse, strict=True)
            ]
            eligible_offsets = [
                offset
                for offset, gain in enumerate(corrected)
                if float(gain) >= minimum_gain
            ]
            eligible_responders = [
                destinations[offset] for offset in eligible_offsets
            ]
            eligible_gains = [corrected[offset] for offset in eligible_offsets]
            score = (
                float(np.median(eligible_gains))
                if len(eligible_gains) >= minimum_responder_count
                else 0.0
            )
            candidates.append(
                {
                    "score": score,
                    "source": source,
                    "lag": lag,
                    "responders": eligible_responders,
                    "eligible_corrected_gains": eligible_gains,
                    "corrected_all_destinations": corrected,
                    "forward_all_destinations": forward,
                    "all_destinations": destinations,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["score"]),
            len(row["responders"]),
            -int(row["lag"]),
            -int(row["source"]),
        ),
    )
    return {
        "source": int(selected["source"]),
        "lag": int(selected["lag"]),
        "responders": [int(value) for value in selected["responders"]],
        "minimum_responder_count": minimum_responder_count,
        "eligible_responder_count": len(selected["responders"]),
        "minimum_corrected_incremental_r2": float(selected["score"]),
        "corrected_incremental_r2_by_responder": list(
            selected["eligible_corrected_gains"]
        ),
        "all_destination_indices": list(selected["all_destinations"]),
        "corrected_incremental_r2_by_all_destination": list(
            selected["corrected_all_destinations"]
        ),
        "forward_incremental_r2_by_all_destination": list(
            selected["forward_all_destinations"]
        ),
        "responder_selection_policy": (
            "at_least_two_threshold_passing_responders_for_d_ge_3_v1"
        ),
    }


def fit_cross_series_contract(background: Mapping[str, Any]) -> dict[str, Any]:
    history = np.asarray(background["_decomposition_target"], dtype=float)
    dimension = history.shape[1]
    if dimension < SENSITIVITY_PANEL_DIMENSION:
        raise ValueError("cross-series dependence requires a synchronized panel")
    selected = _select_directed_edge(history)
    source = int(selected["source"])
    lag = int(selected["lag"])
    responders = [int(value) for value in selected["responders"]]
    fold_edge_evidence: list[dict[str, Any]] = []
    for fold in (history[:336], history[-336:]):
        corrected: list[float] = []
        for responder in responders:
            forward = _incremental_gain(
                fold,
                source=source,
                destination=responder,
                lag=lag,
            )
            reverse = _incremental_gain(
                fold[::-1],
                source=source,
                destination=responder,
                lag=lag,
            )
            corrected.append(max(float(forward) - float(reverse), 0.0))
        passing = [
            gain
            for gain in corrected
            if gain >= _threshold("cross_min_corrected_incremental_r2")
        ]
        fold_edge_evidence.append(
            {
                "source": source,
                "lag": lag,
                "corrected_incremental_r2_by_full_selected_responder": corrected,
                "passing_responder_count": len(passing),
                "minimum_required_responder_count": int(
                    selected["minimum_responder_count"]
                ),
                "median_passing_corrected_incremental_r2": (
                    float(np.median(passing)) if passing else 0.0
                ),
            }
        )
    fixed_edge_fold_passed = all(
        int(row["passing_responder_count"])
        >= int(row["minimum_required_responder_count"])
        and float(row["median_passing_corrected_incremental_r2"])
        >= _threshold("cross_min_corrected_incremental_r2")
        for row in fold_edge_evidence
    )
    driver_agreement = 1.0 if fixed_edge_fold_passed else 0.0
    fold_lag_deviation = (
        0.0
        if fixed_edge_fold_passed
        else float(_threshold("cross_max_lag") + 1.0)
    )
    source_center = float(np.mean(history[:, source]))
    source_values = history[:, source] - source_center
    source_future, source_forecast = _selected_state_forecast(
        source_values,
        protocol.HORIZON,
        period=int(round(float(background.get("season_length", 24)))),
    )
    source_extended = np.concatenate([source_values, source_future])
    component = np.zeros(
        (
            protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
            dimension,
        ),
        dtype=float,
    )
    response_parameters: list[dict[str, float]] = []
    for responder in responders:
        start = max(1, lag)
        response = history[start:, responder]
        design = np.column_stack(
            [
                np.ones(response.size),
                history[start - 1 : -1, responder],
                source_values[
                    start - lag : history.shape[0] - lag
                ],
            ]
        )
        coefficients = _ridge_coefficients(design, response)
        persistence = float(np.clip(coefficients[1], -0.98, 0.98))
        gain = float(coefficients[2])
        response_parameters.append(
            {
                "responder_index": responder,
                "response_persistence": persistence,
                "transfer_gain": gain,
            }
        )
        for time in range(1, component.shape[0]):
            source_index = time - lag
            source_term = (
                source_extended[source_index] if source_index >= 0 else 0.0
            )
            component[time, responder] = (
                persistence * component[time - 1, responder]
                + gain * source_term
            )
    edge_passed = bool(
        float(selected["minimum_corrected_incremental_r2"])
        >= _threshold("cross_min_corrected_incremental_r2")
        and int(selected["eligible_responder_count"])
        >= int(selected["minimum_responder_count"])
        and fixed_edge_fold_passed
    )
    component_passed = bool(_component_gate(component, history_length=504)["passed"])
    panel = background["panel_contract"]
    formal = bool(panel["formal_main_eligible"] and edge_passed and component_passed)
    sensitivity = bool(
        panel["sensitivity_only"] and edge_passed and component_passed
    )
    diagnostics = {
        **selected,
        "fold_selections": fold_edge_evidence,
        "fixed_full_edge_fold_validation_passed": fixed_edge_fold_passed,
        "fold_validation_policy": (
            "full_history_selected_edge_replayed_without_reselection_in_halves_v2"
        ),
        "fold_driver_agreement": driver_agreement,
        "fold_lag_deviation": fold_lag_deviation,
        "source_center": source_center,
        "source_future_forecast": source_forecast,
        "response_parameters": response_parameters,
        "edge_passed": edge_passed,
        "interpretation": "directed_predictive_transfer_not_causal_scm",
        "causal_identification_claimed": False,
    }
    return _base_contract(
        background,
        capability_id="cross_series_dependence",
        component=component,
        fit_diagnostics=diagnostics,
        formal_main_eligible=formal,
        sensitivity_eligible=sensitivity,
        mandatory_input_ablation={
            "required": True,
            "evaluation_table": "real_anchored_input_ablation",
            "assessed_target_indices": responders,
            "ablated_input_indices": [source],
            "replacement_policy": (
                "distinct_frozen_background_donor_affine_matched_on_l336"
            ),
            "target_future_unchanged": True,
            "excluded_from_primary_score": True,
            "reported_as_separate_attribution_audit": True,
            "attribution_policy": STRUCTURAL_INPUT_ABLATION_POLICY,
        },
    )


def _covariate_design(
    target: np.ndarray,
    covariates: np.ndarray,
    *,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    maximum_lag = 2
    time = np.arange(maximum_lag, target.size, dtype=float)
    phase = 2.0 * np.pi * time / max(int(period), 2)
    base = np.column_stack(
        [
            np.ones(time.size),
            target[maximum_lag - 1 : -1],
            (time - np.mean(time)) / max(float(np.std(time)), 1.0),
            np.sin(phase),
            np.cos(phase),
        ]
    )
    response_columns = [
        covariates[
            maximum_lag - lag : covariates.shape[0] - lag
            if lag > 0
            else covariates.shape[0],
            column,
        ]
        for column in range(covariates.shape[1])
        for lag in range(maximum_lag + 1)
    ]
    response = np.column_stack(response_columns)
    return base, response, target[maximum_lag:], maximum_lag


def _design_holdout_gain(
    base: np.ndarray,
    response: np.ndarray,
    target: np.ndarray,
) -> float:
    split = max(48, int(math.floor(0.70 * target.size)))
    if target.size - split < 12:
        return 0.0
    full = np.column_stack([base, response])
    base_coefficients = _ridge_coefficients(base[:split], target[:split])
    full_coefficients = _ridge_coefficients(full[:split], target[:split])
    actual = target[split:]
    base_error = float(
        np.sum((actual - base[split:] @ base_coefficients) ** 2)
    )
    if base_error <= 1e-12:
        return 0.0
    full_error = float(
        np.sum((actual - full[split:] @ full_coefficients) ** 2)
    )
    return float(np.clip((base_error - full_error) / base_error, 0.0, 1.0))


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def fit_covariate_response_contract(background: Mapping[str, Any]) -> dict[str, Any]:
    payload = background.get("known_future_covariates")
    if not isinstance(payload, Mapping) or payload.get("kind") != "known_future":
        raise ValueError("covariate response requires declared known-future covariates")
    target_history = np.asarray(background["_decomposition_target"], dtype=float)
    covariate_path = np.asarray(payload["_source_window"], dtype=float)
    covariate_history = covariate_path[
        : protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
    ]
    period = int(background["feature_period"])
    component = np.zeros_like(target_history, shape=(552, target_history.shape[1]))
    target_diagnostics: list[dict[str, Any]] = []
    eligible_targets: list[int] = []
    for target_index in range(target_history.shape[1]):
        series = target_history[:, target_index]
        base, response, outcome, maximum_lag = _covariate_design(
            series,
            covariate_history,
            period=period,
        )
        gain = _design_holdout_gain(base, response, outcome)
        null_shift = int(_threshold("covariate_fixed_null_shift"))
        shifted_covariates = np.roll(covariate_history, null_shift, axis=0)
        _null_base, null_response, null_outcome, _null_lag = _covariate_design(
            series,
            shifted_covariates,
            period=period,
        )
        null_gain = _design_holdout_gain(base, null_response, null_outcome)
        excess_gain = float(gain - null_gain)
        full_design = np.column_stack([base, response])
        coefficients = _ridge_coefficients(full_design, outcome)
        beta = coefficients[base.shape[1] :]
        split = target_history.shape[0] // 2
        half_betas: list[np.ndarray] = []
        for subset in (slice(0, split), slice(split, None)):
            subset_target = series[subset]
            subset_covariates = covariate_history[subset]
            half_base, half_response, half_outcome, _ = _covariate_design(
                subset_target,
                subset_covariates,
                period=period,
            )
            half_full = np.column_stack([half_base, half_response])
            half_coefficients = _ridge_coefficients(half_full, half_outcome)
            half_betas.append(half_coefficients[half_base.shape[1] :])
        stability = _safe_cosine(half_betas[0], half_betas[1])
        passed = bool(
            excess_gain >= _threshold("covariate_min_excess_incremental_r2")
            and stability
            >= _threshold("covariate_min_coefficient_stability_cosine")
            and float(np.linalg.norm(beta)) > 1e-8
        )
        if passed:
            eligible_targets.append(target_index)
        persistence = float(np.clip(coefficients[1], -0.98, 0.98))
        covariate_response = np.zeros(552, dtype=float)
        beta_matrix = beta.reshape(covariate_history.shape[1], maximum_lag + 1)
        for time in range(1, covariate_response.size):
            response_value = 0.0
            for column in range(covariate_path.shape[1]):
                for lag in range(maximum_lag + 1):
                    source_index = time - lag
                    if source_index >= 0:
                        response_value += (
                            beta_matrix[column, lag]
                            * covariate_path[source_index, column]
                        )
            covariate_response[time] = (
                persistence * covariate_response[time - 1]
                + response_value
            )
        if passed:
            component[:, target_index] = covariate_response
        target_diagnostics.append(
            {
                "target_index": target_index,
                "incremental_holdout_r2": gain,
                "fixed_shift_null_incremental_r2": null_gain,
                "excess_incremental_r2": excess_gain,
                "coefficient_split_cosine": stability,
                "response_persistence": persistence,
                "covariate_lags": list(range(maximum_lag + 1)),
                "beta_by_covariate_and_lag": beta_matrix.tolist(),
                "passed": passed,
            }
        )
    component_passed = bool(_component_gate(component, history_length=504)["passed"])
    formal = bool(eligible_targets and component_passed)
    return _base_contract(
        background,
        capability_id="covariate_response",
        component=component,
        fit_diagnostics={
            "covariate_names": list(payload["column_names"]),
            "eligible_target_indices": eligible_targets,
            "minimum_eligible_target_count": 1,
            "target_diagnostics": target_diagnostics,
            "known_future_path_used_for_h48_component": True,
            "target_future_used_for_h48_component": False,
            "interpretation": (
                "known_future_conditional_predictive_response_not_causal_lift"
            ),
            "causal_identification_claimed": False,
        },
        formal_main_eligible=formal,
        sensitivity_eligible=False,
    )


def _hierarchy_raw_negativity_audit(
    background: Mapping[str, Any],
    component: np.ndarray,
    alphas: Sequence[float],
) -> dict[str, Any]:
    payload = background.get("hierarchy")
    if not isinstance(payload, Mapping):
        raise ValueError("hierarchy negativity audit requires hierarchy payload")
    raw_path = np.asarray(payload["_raw_source_window"], dtype=float)
    children = [int(value) for value in payload["child_indices"]]
    raw_component = (
        np.asarray(component, dtype=float)
        * float(payload["standardization"]["shared_scale"])
    )
    negativity: dict[str, Any] = {}
    for alpha in alphas:
        alpha_value = float(alpha)
        augmented = raw_path.copy()
        augmented[:, children] += (
            (alpha_value - 1.0) * raw_component[:, children]
        )
        counts = np.sum(augmented[:, children] < 0.0, axis=0)
        negativity[str(alpha_value)] = {
            "negative_value_count_by_child": counts.astype(int).tolist(),
            "total_negative_value_count": int(np.sum(counts)),
            "minimum_augmented_child_value": float(
                np.min(augmented[:, children])
            ),
        }
    return negativity


def fit_hierarchy_qualification_contract(
    background: Mapping[str, Any],
) -> dict[str, Any]:
    payload = background.get("hierarchy")
    if not isinstance(payload, Mapping):
        raise ValueError("hierarchy qualification requires a declared hierarchy")
    history = np.asarray(payload["_decomposition_history"], dtype=float)
    children = [int(value) for value in payload["child_indices"]]
    parent = history[:, int(payload["parent_index"])]
    child_history = history[:, children]
    denominator = max(float(np.dot(parent, parent)), 1e-12)
    weights = parent @ child_history / denominator
    weights += (1.0 - float(np.sum(weights))) / float(weights.size)
    contrast = child_history - parent[:, None] * weights[None, :]
    contrast -= np.mean(contrast, axis=1, keepdims=True)
    future_contrast = np.empty((protocol.HORIZON, contrast.shape[1]), dtype=float)
    holdout_r2: list[float] = []
    ar_parameters: list[dict[str, float]] = []
    for column in range(contrast.shape[1] - 1):
        forecast, intercept, persistence = _ar1_forecast(
            contrast[:, column], protocol.HORIZON
        )
        future_contrast[:, column] = forecast
        holdout_r2.append(_ar1_one_step_holdout_r2(contrast[:, column]))
        ar_parameters.append(
            {"intercept": intercept, "persistence": persistence}
        )
    future_contrast[:, -1] = -np.sum(future_contrast[:, :-1], axis=1)
    contrast_full = np.vstack([contrast, future_contrast])
    component = np.zeros((552, history.shape[1]), dtype=float)
    component[:, children] = contrast_full
    coherence_max_abs = float(
        np.max(np.abs(np.sum(component[:, children], axis=1)))
    )
    minimum_holdout = _threshold("hierarchy_min_contrast_holdout_r2")
    qualification_passed = bool(
        coherence_max_abs <= 1e-10
        and holdout_r2
        and float(np.mean(holdout_r2)) >= minimum_holdout
        and _component_gate(component, history_length=504)["passed"]
    )
    negativity = _hierarchy_raw_negativity_audit(
        background,
        component,
        STRUCTURAL_ALPHAS,
    )
    contract = _base_contract(
        background,
        capability_id="hierarchical_coherence",
        component=component,
        fit_diagnostics={
            "allocation_weights": weights.tolist(),
            "contrast_ar_parameters": ar_parameters,
            "contrast_one_step_holdout_r2": holdout_r2,
            "minimum_mean_contrast_holdout_r2": minimum_holdout,
            "zero_sum_component_max_abs": coherence_max_abs,
            "qualification_passed": qualification_passed,
            "raw_negativity_audit_by_alpha": negativity,
            "raw_negativity_uses_real_future_only_as_post_fit_audit": True,
            "raw_negativity_affects_fit_or_thresholds": False,
            "interpretation": "forecastable_zero_sum_hierarchical_contrast",
        },
        formal_main_eligible=False,
        sensitivity_eligible=False,
    )
    contract["qualification_only"] = True
    contract["generation_eligible"] = False
    contract["ranking_eligible"] = False
    contract["generation_prohibition_reason"] = (
        "hierarchy_raw_support_policy_unresolved_qualification_only"
    )
    contract["hierarchy_formal_rank_policy"] = HIERARCHY_FORMAL_RANK_POLICY
    contract["contract_sha256"] = _contract_sha256(contract)
    return contract


def _attach_frozen_structural_dose_design(
    background: Mapping[str, Any],
    contract: Mapping[str, Any],
    frozen_qualification_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay one reference-frozen grid and fail closed on weak separation."""

    result = copy.deepcopy(dict(contract))
    capability_id = str(result["capability_id"])
    calibration = dose_calibration_from_policy(
        frozen_qualification_policy,
        capability_id,
        require_available=False,
    )
    evidence = result.get("dose_design_reference")
    gates: list[dict[str, Any]] = []
    failure_reason: str | None = None
    if calibration.get("status") != "available":
        failure_reason = "reference_dose_calibration_unavailable"
    elif not isinstance(evidence, Mapping):
        failure_reason = "evaluation_dose_design_reference_unavailable"
    else:
        try:
            calibration = resolve_contract_dose_calibration(
                calibration,
                evidence,
            )
        except ValueError:
            failure_reason = "contract_source_distance_mapping_unavailable"
    result["dose_calibration"] = copy.deepcopy(calibration)
    if failure_reason is None:
        affected = [
            int(value) for value in evidence["affected_channel_indices"]
        ]
        component = np.asarray(result["component"], dtype=float)[
            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
            - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
        ]
        previous_delta: np.ndarray | None = None
        for dose_index, alpha in enumerate(
            calibration["applied_alpha_grid"],
            start=1,
        ):
            delta = (float(alpha) - 1.0) * component
            gates.append(
                paired_minimum_separation_gate(
                    delta,
                    context_length=protocol.REAL_ANCHORED_CONTEXT_LENGTH,
                    dose_index=dose_index,
                    dose_calibration=calibration,
                    affected_channel_indices=affected,
                    previous_delta=previous_delta,
                )
            )
            previous_delta = delta
        if not all(bool(gate["accepted"]) for gate in gates):
            failure_reason = "paired_minimum_separation_gate_failed"
    passed = bool(failure_reason is None)
    adjacent_passed = bool(
        gates and all(gate["adjacent_accepted"] is True for gate in gates)
    )
    result["paired_minimum_separation_gate"] = gates
    result["paired_minimum_separation_qualification"] = {
        "status": "passed" if passed else "failed",
        "accepted": passed,
        "reason_code": failure_reason,
        "all_levels_passed": passed,
        "adjacent_minimum_separation_passed": adjacent_passed,
        "adjacent_separation_derivation": (
            "explicit_previous_treatment_delta_with_level_1_baseline"
        ),
        "evaluated_level_count": len(gates),
        "dose_calibration_policy_sha256": str(
            calibration.get(
                "dose_policy_sha256",
                calibration["policy_sha256"],
            )
        ),
        "contract_dose_calibration_sha256": str(
            calibration["policy_sha256"]
        ),
        "target_future_used": False,
        "anti_copy_semantics": (
            "treatment_only_distance_from_authentic_source"
        ),
    }
    result["dose_pairing_eligible"] = passed
    if calibration.get("status") == "available":
        result["dose_grid_status"] = "reference_frozen_available"
        result["canonical_strength_grid"] = list(
            calibration["strength_grid"]
        )
        result["applied_alpha_grid"] = list(
            calibration["applied_alpha_grid"]
        )
        # ``alpha_grid`` is retained as a compatibility alias but is no longer
        # the cross-capability strength coordinate.
        result["alpha_grid"] = list(calibration["applied_alpha_grid"])
        if capability_id == "hierarchical_coherence":
            diagnostics = copy.deepcopy(dict(result["fit_diagnostics"]))
            diagnostics["raw_negativity_audit_by_alpha"] = (
                _hierarchy_raw_negativity_audit(
                    background,
                    np.asarray(result["component"], dtype=float),
                    tuple(float(value) for value in result["applied_alpha_grid"]),
                )
            )
            diagnostics["raw_negativity_alpha_grid_source"] = (
                "reference_frozen_dose_calibration"
            )
            result["fit_diagnostics"] = diagnostics
    else:
        result["dose_grid_status"] = "reference_frozen_unavailable"
        result["canonical_strength_grid"] = list(
            calibration["strength_grid"]
        )
        result["applied_alpha_grid"] = []
        result["alpha_grid"] = []
    if capability_id != "hierarchical_coherence" and not passed:
        result["formal_main_eligible"] = False
        result["sensitivity_eligible"] = False
    if capability_id != "hierarchical_coherence":
        result["generation_eligible"] = bool(
            result.get("formal_main_eligible")
        )
        result["ranking_eligible"] = bool(result.get("formal_main_eligible"))
    result["contract_sha256"] = _contract_sha256(result)
    return result


def fit_structural_capability_contracts(
    backgrounds: Sequence[Mapping[str, Any]],
    *,
    capability_ids: Iterable[str] = STRUCTURAL_CAPABILITIES,
    frozen_qualification_policy: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = tuple(dict.fromkeys(str(value) for value in capability_ids))
    unknown = sorted(set(requested) - set(STRUCTURAL_CAPABILITIES))
    if unknown:
        raise ValueError(f"unsupported structural capabilities: {unknown}")
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    fitters = {
        "common_factor": fit_common_factor_contract,
        "cross_series_dependence": fit_cross_series_contract,
        "covariate_response": fit_covariate_response_contract,
        "hierarchical_coherence": fit_hierarchy_qualification_contract,
    }
    for background in backgrounds:
        for capability_id in requested:
            reason: str | None = None
            try:
                contract = fitters[capability_id](background)
            except (ValueError, np.linalg.LinAlgError) as error:
                contract = None
                reason = str(error)
            if contract is not None and frozen_qualification_policy is not None:
                contract = _attach_frozen_structural_dose_design(
                    background,
                    contract,
                    frozen_qualification_policy,
                )
            formal = bool(
                contract is not None and contract.get("formal_main_eligible") is True
            )
            sensitivity = bool(
                contract is not None and contract.get("sensitivity_eligible") is True
            )
            qualification = bool(
                contract is not None
                and capability_id == "hierarchical_coherence"
                and contract["fit_diagnostics"].get("qualification_passed") is True
            )
            if capability_id == "hierarchical_coherence":
                reason = (
                    "qualification_only_generation_and_ranking_prohibited"
                    if contract is not None
                    else reason
                )
            elif contract is not None and not formal:
                dose_qualification = contract.get(
                    "paired_minimum_separation_qualification"
                )
                if (
                    isinstance(dose_qualification, Mapping)
                    and dose_qualification.get("accepted") is False
                ):
                    reason = str(dose_qualification["reason_code"])
                elif sensitivity:
                    reason = "panel_d2_sensitivity_only"
                else:
                    reason = "fixed_reference_bank_structural_gates_failed"
            if reason is not None:
                reason_counts[reason] += 1
            row = {
                "schema_version": STRUCTURAL_CAPABILITY_ROW_SCHEMA,
                "dataset_id": str(background["dataset_id"]),
                "background_id": str(background["background_id"]),
                "background_bank_role": background.get(
                    "background_bank_role"
                ),
                "capability_id": capability_id,
                "benchmark_track": "real_anchored_counterfactual",
                # ``available`` is the field consumed by the generic reference
                # bank freezer. Hierarchy availability means qualification
                # evidence only and never implies generation/ranking eligibility.
                "available": bool(
                    qualification
                    if capability_id == "hierarchical_coherence"
                    else formal
                ),
                "formal_main_available": formal,
                "sensitivity_available": sensitivity,
                "qualification_available": qualification,
                "generation_eligible": bool(
                    formal and capability_id != "hierarchical_coherence"
                ),
                "ranking_eligible": bool(
                    formal and capability_id != "hierarchical_coherence"
                ),
                "unavailable_reason": reason,
                "qualification_thresholds": _qualification_thresholds(
                    capability_id
                ),
                "qualification_policy_id": _qualification_policy_id(
                    capability_id
                ),
                "qualification_threshold_source": (
                    QUALIFICATION_THRESHOLD_SOURCE_POLICY
                ),
                "frozen_qualification_policy_sha256": None,
                "dose_design_reference": (
                    None
                    if contract is None
                    else copy.deepcopy(contract.get("dose_design_reference"))
                ),
                "dose_calibration": (
                    None
                    if contract is None
                    else copy.deepcopy(contract.get("dose_calibration"))
                ),
                "paired_minimum_separation_gate": (
                    []
                    if contract is None
                    else copy.deepcopy(
                        contract.get("paired_minimum_separation_gate", [])
                    )
                ),
                "paired_minimum_separation_qualification": (
                    None
                    if contract is None
                    else copy.deepcopy(
                        contract.get(
                            "paired_minimum_separation_qualification"
                        )
                    )
                ),
                "contract": contract,
            }
            if frozen_qualification_policy is not None:
                capabilities = frozen_qualification_policy.get("capabilities")
                if not isinstance(capabilities, Mapping):
                    raise ValueError(
                        "frozen qualification policy has no capabilities"
                    )
                frozen = capabilities.get(capability_id)
                if not isinstance(frozen, Mapping):
                    raise ValueError(
                        "frozen qualification policy lacks structural capability "
                        f"{capability_id}"
                    )
                if _canonical_json(row["qualification_thresholds"]) != (
                    _canonical_json(
                        dict(frozen.get("qualification_thresholds", {}))
                    )
                ):
                    raise ValueError(
                        "evaluation structural thresholds differ from the "
                        f"reference bank for {capability_id}"
                    )
                if row["qualification_policy_id"] != frozen.get(
                    "qualification_policy_id"
                ):
                    raise ValueError(
                        "evaluation structural policy ID differs from the "
                        f"reference bank for {capability_id}"
                    )
                row["frozen_qualification_policy_sha256"] = (
                    frozen_qualification_policy.get(
                        "qualification_policy_sha256"
                    )
                )
            rows.append(row)
    formal_counts = {
        capability_id: sum(
            row["capability_id"] == capability_id
            and row["formal_main_available"]
            for row in rows
        )
        for capability_id in requested
    }
    sensitivity_counts = {
        capability_id: sum(
            row["capability_id"] == capability_id
            and row["sensitivity_available"]
            for row in rows
        )
        for capability_id in requested
    }
    qualification_counts = {
        capability_id: sum(
            row["capability_id"] == capability_id
            and row["qualification_available"]
            for row in rows
        )
        for capability_id in requested
    }
    cells: list[dict[str, Any]] = []
    for capability_id in requested:
        eligible_background_ids = sorted(
            str(row["background_id"])
            for row in rows
            if row["capability_id"] == capability_id
            and row["generation_eligible"] is True
        )
        sensitivity_background_ids = sorted(
            str(row["background_id"])
            for row in rows
            if row["capability_id"] == capability_id
            and row["sensitivity_available"] is True
        )
        if capability_id in {"common_factor", "cross_series_dependence"}:
            sensitivity_status = (
                "available"
                if len(sensitivity_background_ids)
                >= MINIMUM_SENSITIVITY_BACKGROUND_COUNT
                else "unavailable"
            )
            sensitivity_reason_codes = (
                []
                if sensitivity_status == "available"
                else ["insufficient_d2_sensitivity_backgrounds"]
            )
        else:
            sensitivity_status = "not_applicable"
            sensitivity_reason_codes = []
        if (
            capability_id == "hierarchical_coherence"
            and qualification_counts[capability_id] > 0
        ):
            status = "qualification_only"
            reason_codes = [
                "hierarchy_generation_and_ranking_prohibited"
            ]
        elif capability_id == "hierarchical_coherence":
            status = "unavailable"
            reason_codes = ["no_eligible_qualification_backgrounds"]
        elif (
            len(eligible_background_ids)
            >= MINIMUM_FORMAL_BACKGROUND_COUNT
        ):
            status = "available"
            reason_codes = []
        else:
            status = "unavailable"
            reason_codes = ["insufficient_eligible_backgrounds"]
        cells.append(
            {
                "capability_id": capability_id,
                "status": status,
                "reason_codes": reason_codes,
                "formal_background_count": formal_counts[capability_id],
                "sensitivity_background_count": sensitivity_counts[
                    capability_id
                ],
                "qualification_background_count": qualification_counts[
                    capability_id
                ],
                "minimum_formal_background_count": (
                    MINIMUM_FORMAL_BACKGROUND_COUNT
                ),
                "eligible_background_ids_sha256": protocol.json_sha256(
                    eligible_background_ids
                ),
                "sensitivity_status": sensitivity_status,
                "sensitivity_reason_codes": sensitivity_reason_codes,
                "minimum_sensitivity_background_count": (
                    MINIMUM_SENSITIVITY_BACKGROUND_COUNT
                ),
                "sensitivity_background_ids_sha256": protocol.json_sha256(
                    sensitivity_background_ids
                ),
                "sensitivity_generation_eligible": (
                    sensitivity_status == "available"
                ),
                "generation_eligible": status == "available",
                "ranking_eligible": status == "available",
            }
        )
    return rows, {
        "schema_version": STRUCTURAL_AVAILABILITY_SCHEMA,
        "requested_capability_ids": list(requested),
        "minimum_formal_background_count": (
            MINIMUM_FORMAL_BACKGROUND_COUNT
        ),
        "minimum_sensitivity_background_count": (
            MINIMUM_SENSITIVITY_BACKGROUND_COUNT
        ),
        "formal_background_count_by_capability": formal_counts,
        "sensitivity_background_count_by_capability": sensitivity_counts,
        "qualification_background_count_by_capability": qualification_counts,
        "cells": cells,
        "hierarchy_generation_and_ranking_prohibited": True,
        "threshold_contract": structural_threshold_contract(),
        "qualification_threshold_source": (
            QUALIFICATION_THRESHOLD_SOURCE_POLICY
        ),
        "frozen_qualification_policy_sha256": (
            None
            if frozen_qualification_policy is None
            else frozen_qualification_policy.get(
                "qualification_policy_sha256"
            )
        ),
        "unavailable_reason_counts": dict(sorted(reason_counts.items())),
    }


def available_structural_capabilities(
    availability: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return only dataset cells that clear the frozen formal-N gate."""

    cells = availability.get("cells")
    if isinstance(cells, list):
        return tuple(
            str(cell["capability_id"])
            for cell in cells
            if cell.get("status") == "available"
            and cell.get("generation_eligible") is True
        )
    # Compatibility for short-lived v1 qualification artifacts: do not trust
    # per-background eligibility alone; reapply the same protocol-wide N gate.
    counts = availability.get("formal_background_count_by_capability", {})
    if not isinstance(counts, Mapping):
        return ()
    return tuple(
        str(capability_id)
        for capability_id, count in counts.items()
        if str(capability_id) != "hierarchical_coherence"
        and int(count) >= MINIMUM_FORMAL_BACKGROUND_COUNT
    )


def available_structural_sensitivity_capabilities(
    availability: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return D=2 panel cells with enough distinct donors for sensitivity."""

    cells = availability.get("cells")
    if not isinstance(cells, list):
        return ()
    return tuple(
        str(cell["capability_id"])
        for cell in cells
        if cell.get("sensitivity_status") == "available"
        and cell.get("sensitivity_generation_eligible") is True
    )


def validate_structural_availability(
    availability: Mapping[str, Any],
    contract_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed when aggregate structural eligibility was altered."""

    if not contract_rows and availability.get("cells") == []:
        return
    requested = availability.get("requested_capability_ids")
    if not isinstance(requested, list):
        raise ValueError("structural availability lacks requested capabilities")
    formal_counts = {
        capability_id: sum(
            str(row.get("capability_id")) == capability_id
            and row.get("formal_main_available") is True
            for row in contract_rows
        )
        for capability_id in map(str, requested)
    }
    observed_counts = availability.get(
        "formal_background_count_by_capability"
    )
    if observed_counts != formal_counts:
        raise ValueError(
            "structural availability formal counts disagree with contracts"
        )
    if int(
        availability.get(
            "minimum_formal_background_count",
            MINIMUM_FORMAL_BACKGROUND_COUNT,
        )
    ) != MINIMUM_FORMAL_BACKGROUND_COUNT:
        raise ValueError("structural availability changed the formal-N gate")
    if int(
        availability.get(
            "minimum_sensitivity_background_count",
            MINIMUM_SENSITIVITY_BACKGROUND_COUNT,
        )
    ) != MINIMUM_SENSITIVITY_BACKGROUND_COUNT:
        raise ValueError(
            "structural availability changed the D2 sensitivity-N gate"
        )
    cells = availability.get("cells")
    if cells is None and availability.get("schema_version") == (
        "cafe.structural_real_availability.v1"
    ):
        return
    if availability.get("schema_version") not in {
        STRUCTURAL_AVAILABILITY_SCHEMA,
        LEGACY_STRUCTURAL_AVAILABILITY_SCHEMA,
    }:
        raise ValueError("unsupported structural availability schema")
    if not isinstance(cells, list):
        raise ValueError("structural availability lacks cells")
    cell_by_capability = {
        str(cell.get("capability_id")): cell
        for cell in cells
        if isinstance(cell, Mapping)
    }
    blocked_capabilities = {
        str(value)
        for value in availability.get(
            "qualification_blocked_capabilities",
            [],
        )
    }
    if set(cell_by_capability) != set(formal_counts):
        raise ValueError("structural availability cell coverage mismatch")
    for capability_id, formal_count in formal_counts.items():
        selected = [
            row
            for row in contract_rows
            if str(row.get("capability_id")) == capability_id
        ]
        sensitivity_count = sum(
            row.get("sensitivity_available") is True for row in selected
        )
        qualification_count = sum(
            row.get("qualification_available") is True for row in selected
        )
        eligible_ids = sorted(
            str(row["background_id"])
            for row in selected
            if row.get("generation_eligible") is True
        )
        sensitivity_ids = sorted(
            str(row["background_id"])
            for row in selected
            if row.get("sensitivity_available") is True
        )
        if capability_id in {"common_factor", "cross_series_dependence"}:
            sensitivity_status = (
                "available"
                if sensitivity_count >= MINIMUM_SENSITIVITY_BACKGROUND_COUNT
                else "unavailable"
            )
            sensitivity_reasons = (
                []
                if sensitivity_status == "available"
                else ["insufficient_d2_sensitivity_backgrounds"]
            )
        else:
            sensitivity_status = "not_applicable"
            sensitivity_reasons = []
        if (
            capability_id == "hierarchical_coherence"
            and qualification_count > 0
        ):
            expected_status = "qualification_only"
            expected_reasons = [
                "hierarchy_generation_and_ranking_prohibited"
            ]
        elif capability_id == "hierarchical_coherence":
            expected_status = "unavailable"
            expected_reasons = ["no_eligible_qualification_backgrounds"]
        elif (
            formal_count >= MINIMUM_FORMAL_BACKGROUND_COUNT
        ):
            expected_status = "available"
            expected_reasons = []
        else:
            expected_status = "unavailable"
            expected_reasons = ["insufficient_eligible_backgrounds"]
        if capability_id in blocked_capabilities:
            expected_status = "unavailable"
            expected_reasons = sorted(
                {
                    *expected_reasons,
                    "independent_reference_bank_unavailable",
                }
            )
        cell = cell_by_capability[capability_id]
        expected_fields = {
            "status": expected_status,
            "reason_codes": expected_reasons,
            "formal_background_count": formal_count,
            "sensitivity_background_count": sensitivity_count,
            "qualification_background_count": qualification_count,
            "minimum_formal_background_count": (
                MINIMUM_FORMAL_BACKGROUND_COUNT
            ),
            "eligible_background_ids_sha256": protocol.json_sha256(
                eligible_ids
            ),
            "sensitivity_status": sensitivity_status,
            "sensitivity_reason_codes": sensitivity_reasons,
            "minimum_sensitivity_background_count": (
                MINIMUM_SENSITIVITY_BACKGROUND_COUNT
            ),
            "sensitivity_background_ids_sha256": protocol.json_sha256(
                sensitivity_ids
            ),
            "sensitivity_generation_eligible": (
                sensitivity_status == "available"
            ),
            "generation_eligible": expected_status == "available",
            "ranking_eligible": expected_status == "available",
        }
        if any(cell.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(
                "structural availability cells disagree with contracts"
            )


def validate_structural_contract(
    contract: Mapping[str, Any],
    background: Mapping[str, Any],
) -> None:
    if contract.get("schema_version") not in {
        STRUCTURAL_CONTRACT_SCHEMA,
        LEGACY_STRUCTURAL_CONTRACT_SCHEMA,
    }:
        raise ValueError("unsupported structural contract schema")
    if contract.get("background_id") != background.get("background_id"):
        raise ValueError("structural contract/background identity mismatch")
    if contract.get("source_history_sha256") != background.get(
        "decomposition_history_sha256"
    ):
        raise ValueError("structural source history hash mismatch")
    if contract.get("thresholds_adapted_from_evaluation_origins") is not False:
        raise ValueError("structural thresholds may not adapt to evaluation origins")
    if contract.get("contract_sha256") != _contract_sha256(contract):
        raise ValueError("structural contract integrity hash mismatch")
    component = np.asarray(contract["component"], dtype=float)
    expected_shape = (
        protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
        int(contract["component_dim"]),
    )
    if component.shape != expected_shape or not np.isfinite(component).all():
        raise ValueError("structural component shape/finite contract failed")
    if contract.get("component_sha256") != _array_sha256(
        component,
        domain=f"structural_{contract['capability_id']}_component",
    ):
        raise ValueError("structural component hash mismatch")
    if contract["capability_id"] in {
        "common_factor",
        "cross_series_dependence",
    }:
        ablation = contract.get("mandatory_input_ablation")
        if not isinstance(ablation, Mapping) or ablation.get("required") is not True:
            raise ValueError("panel structural contract requires input ablation")
    evidence = contract.get("dose_design_reference")
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise ValueError("structural dose reference evidence is malformed")
        expected_evidence = _structural_dose_reference(
            background,
            capability_id=str(contract["capability_id"]),
            component=component,
            fit_diagnostics=contract["fit_diagnostics"],
            evidence_role=str(evidence.get("evidence_role", "")),
        )
        if expected_evidence != dict(evidence):
            raise ValueError(
                "structural dose reference disagrees with fitted component"
            )
    calibration = contract.get("dose_calibration")
    if calibration is None:
        if contract.get("schema_version") == STRUCTURAL_CONTRACT_SCHEMA:
            if (
                contract.get("alpha_grid") != []
                or contract.get("dose_grid_status")
                != "reference_mapping_pending"
            ):
                raise ValueError("reference structural dose mapping was altered")
        return
    if not isinstance(calibration, Mapping):
        raise ValueError("structural dose calibration is malformed")
    capability_id = str(contract["capability_id"])
    validate_dose_calibration(calibration, capability_id=capability_id)
    expected_strengths = list(calibration["strength_grid"])
    expected_alphas = list(calibration["applied_alpha_grid"])
    if contract.get("canonical_strength_grid") != expected_strengths:
        raise ValueError("structural canonical strength grid changed")
    if contract.get("applied_alpha_grid") != expected_alphas:
        raise ValueError("structural applied alpha grid changed")
    if contract.get("alpha_grid") != expected_alphas:
        raise ValueError("structural alpha compatibility grid changed")
    qualification = contract.get("paired_minimum_separation_qualification")
    gates = contract.get("paired_minimum_separation_gate")
    if not isinstance(qualification, Mapping) or not isinstance(gates, list):
        raise ValueError("structural paired separation audit is missing")
    contract_mapping_unavailable = False
    if (
        calibration.get("status") == "available"
        and calibration.get("schema_version")
        == REAL_ANCHORED_DOSE_CALIBRATION_SCHEMA
        and isinstance(evidence, Mapping)
    ):
        try:
            resolve_contract_dose_calibration(calibration, evidence)
        except ValueError:
            contract_mapping_unavailable = True
        else:
            raise ValueError(
                "resolvable structural dose mapping was stored as unavailable"
            )
    expected_gates: list[dict[str, Any]] = []
    if calibration.get("status") == "available" and isinstance(evidence, Mapping):
        visible_component = component[
            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
            - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
        ]
        affected = [
            int(value) for value in evidence["affected_channel_indices"]
        ]
        previous_delta: np.ndarray | None = None
        for dose_index, alpha in enumerate(expected_alphas, start=1):
            delta = (float(alpha) - 1.0) * visible_component
            expected_gates.append(
                paired_minimum_separation_gate(
                    delta,
                    context_length=protocol.REAL_ANCHORED_CONTEXT_LENGTH,
                    dose_index=dose_index,
                    dose_calibration=calibration,
                    affected_channel_indices=affected,
                    previous_delta=previous_delta,
                )
            )
            previous_delta = delta
    if gates != expected_gates:
        raise ValueError("structural paired separation gates changed")
    expected_accepted = bool(
        calibration.get("status") == "available"
        and isinstance(evidence, Mapping)
        and expected_gates
        and all(bool(gate["accepted"]) for gate in expected_gates)
    )
    expected_reason = (
        "reference_dose_calibration_unavailable"
        if calibration.get("status") != "available"
        else "evaluation_dose_design_reference_unavailable"
        if not isinstance(evidence, Mapping)
        else "contract_source_distance_mapping_unavailable"
        if contract_mapping_unavailable
        else "paired_minimum_separation_gate_failed"
        if not expected_accepted
        else None
    )
    if (
        qualification.get("accepted") is not expected_accepted
        or qualification.get("status")
        != ("passed" if expected_accepted else "failed")
        or qualification.get("reason_code") != expected_reason
        or qualification.get("all_levels_passed") is not expected_accepted
        or qualification.get("evaluated_level_count") != len(expected_gates)
        or contract.get("dose_pairing_eligible") is not expected_accepted
        or qualification.get("adjacent_minimum_separation_passed")
        is not bool(
            expected_gates
            and all(
                gate["adjacent_accepted"] is True for gate in expected_gates
            )
        )
            or qualification.get("dose_calibration_policy_sha256")
            != calibration.get(
                "dose_policy_sha256",
                calibration.get("policy_sha256"),
            )
            or qualification.get("contract_dose_calibration_sha256")
            != calibration.get("policy_sha256")
        or qualification.get("target_future_used") is not False
            or qualification.get("anti_copy_semantics")
            != "treatment_only_distance_from_authentic_source"
    ):
        raise ValueError("structural paired separation qualification changed")
    expected_grid_status = (
        "reference_frozen_available"
        if calibration.get("status") == "available"
        else "reference_frozen_unavailable"
    )
    if contract.get("dose_grid_status") != expected_grid_status:
        raise ValueError("structural dose grid status changed")
    if capability_id != "hierarchical_coherence" and not expected_accepted:
        if (
            contract.get("formal_main_eligible") is not False
            or contract.get("sensitivity_eligible") is not False
            or contract.get("generation_eligible") is not False
            or contract.get("ranking_eligible") is not False
        ):
            raise ValueError("failed structural dose gate did not fail closed")
    if capability_id != "hierarchical_coherence" and expected_accepted:
        expected_generation = bool(contract.get("formal_main_eligible"))
        if (
            contract.get("generation_eligible") is not expected_generation
            or contract.get("ranking_eligible") is not expected_generation
        ):
            raise ValueError("structural generation eligibility changed")


def apply_structural_contract(
    background: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    alpha: float,
    allow_sensitivity: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    validate_structural_contract(contract, background)
    capability_id = str(contract["capability_id"])
    if capability_id == "hierarchical_coherence":
        raise ValueError(
            "hierarchical coherence is qualification-only and cannot generate"
        )
    if not contract.get("formal_main_eligible"):
        if not (allow_sensitivity and contract.get("sensitivity_eligible")):
            raise ValueError("structural contract is not eligible for this track")
    alpha_value = float(alpha)
    calibrated_grid = contract.get("applied_alpha_grid")
    allowed_alphas = (
        tuple(float(value) for value in calibrated_grid)
        if isinstance(calibrated_grid, list)
        else STRUCTURAL_ALPHAS
    )
    if alpha_value != 1.0 and not any(
        math.isclose(alpha_value, allowed, rel_tol=0.0, abs_tol=1e-12)
        for allowed in allowed_alphas
    ):
        raise ValueError("alpha is outside the frozen structural dose grid")
    baseline = np.asarray(background["target"], dtype=float)
    component = np.asarray(contract["component"], dtype=float)[
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
    ]
    delta = (alpha_value - 1.0) * component
    augmented = baseline.copy() if alpha_value == 1.0 else baseline + delta
    diagnostics = contract["fit_diagnostics"]
    role_metadata: dict[str, Any] = {}
    if capability_id == "common_factor":
        role_metadata = {
            "protected_target_index": int(
                diagnostics["protected_target_index"]
            ),
            "response_loadings": list(diagnostics["loadings"]),
        }
    elif capability_id == "cross_series_dependence":
        role_metadata = {
            "driver_index": int(diagnostics["source"]),
            "responder_indices": [
                int(value) for value in diagnostics["responders"]
            ],
            "cross_lag_steps": int(diagnostics["lag"]),
        }
    elif capability_id == "covariate_response":
        role_metadata = {
            "eligible_target_indices": [
                int(value)
                for value in diagnostics["eligible_target_indices"]
            ],
        }
    return augmented, {
        "capability_id": capability_id,
        "alpha": alpha_value,
        "controlled_component": (
            "rank1_shared_component"
            if capability_id == "common_factor"
            else "linear_directed_predictive_transfer"
            if capability_id == "cross_series_dependence"
            else "known_future_covariate_predictive_response"
        ),
        "truth_delta": delta.tolist(),
        "truth_delta_sha256": _array_sha256(
            delta,
            domain=f"structural_{capability_id}_truth_delta",
        ),
        "contract_sha256": str(contract["contract_sha256"]),
        "dose_calibration_policy_sha256": (
            None
            if not isinstance(contract.get("dose_calibration"), Mapping)
            else str(
                contract["dose_calibration"].get(
                    "dose_policy_sha256",
                    contract["dose_calibration"]["policy_sha256"],
                )
            )
        ),
        "contract_dose_calibration_sha256": (
            None
            if not isinstance(contract.get("dose_calibration"), Mapping)
            else str(contract["dose_calibration"]["policy_sha256"])
        ),
        "target_future_used_for_delta": False,
        "known_future_covariate_path_used_for_delta": bool(
            capability_id == "covariate_response"
        ),
        "mandatory_input_ablation": copy.deepcopy(
            contract.get("mandatory_input_ablation")
        ),
        **role_metadata,
    }


def iter_structural_real_anchored_samples(
    backgrounds: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    alphas: Iterable[float] = STRUCTURAL_ALPHAS,
    sensitivity: bool = False,
    seed_indexes: Iterable[int] | None = None,
) -> Iterator[dict[str, Any]]:
    legacy_alphas = tuple(float(value) for value in alphas)
    if (
        not legacy_alphas
        or any(
            not math.isfinite(value) or value <= 1.0
            for value in legacy_alphas
        )
        or len(legacy_alphas) != len(set(legacy_alphas))
    ):
        raise ValueError("structural alpha grid must be finite/unique/above one")
    background_by_id = {
        str(background["background_id"]): background for background in backgrounds
    }
    raw_seeds = (
        None
        if seed_indexes is None
        else tuple(int(value) for value in seed_indexes)
    )
    requested_seeds = (
        None if raw_seeds is None else tuple(dict.fromkeys(raw_seeds))
    )
    if requested_seeds is not None and (
        any(value < 0 for value in requested_seeds)
        or len(requested_seeds) != len(raw_seeds or ())
    ):
        raise ValueError("structural seed indexes must be unique/non-negative")
    eligible_rows: list[Mapping[str, Any]] = []
    for row in contract_rows:
        contract = row.get("contract")
        if not isinstance(contract, Mapping):
            continue
        eligible = (
            bool(row.get("sensitivity_available"))
            if sensitivity
            else bool(row.get("generation_eligible"))
        )
        if not eligible:
            continue
        eligible_rows.append(row)

    assigned_rows: list[tuple[int, Mapping[str, Any]]] = []
    by_capability: dict[str, list[Mapping[str, Any]]] = {}
    for row in eligible_rows:
        capability_id = str(row["capability_id"])
        by_capability.setdefault(capability_id, []).append(row)
    for capability_id, rows in sorted(by_capability.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                protocol.stable_seed(
                    str(row["dataset_id"]),
                    capability_id,
                    str(row["background_id"]),
                    "structural-real-anchored-background-permutation",
                    base=protocol.REAL_ANCHORED_SAMPLE_SEED,
                ),
                str(row["background_id"]),
            ),
        )
        ids = [str(row["background_id"]) for row in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "structural contracts contain duplicate eligible backgrounds"
            )
        if requested_seeds is None:
            assigned_rows.extend(
                (
                    int(
                        hashlib.sha256(background_id.encode("utf-8")).hexdigest()[
                            :12
                        ],
                        16,
                    ),
                    row,
                )
                for background_id, row in zip(ids, ordered)
            )
        else:
            assigned_rows.extend(
                (seed_index, ordered[seed_index])
                for seed_index in requested_seeds
                if seed_index < len(ordered)
            )

    for seed_index, row in assigned_rows:
        contract = row["contract"]
        background = background_by_id[str(row["background_id"])]
        validate_structural_contract(contract, background)
        calibration = contract.get("dose_calibration")
        stored_gates = contract.get("paired_minimum_separation_gate", [])
        if isinstance(calibration, Mapping):
            validate_dose_calibration(
                calibration,
                capability_id=str(contract["capability_id"]),
            )
            if (
                calibration.get("status") != "available"
                or contract.get("dose_pairing_eligible") is not True
            ):
                raise ValueError(
                    "eligible structural row lacks an accepted frozen dose grid"
                )
            evidence = contract["dose_design_reference"]
            visible_component = np.asarray(contract["component"], dtype=float)[
                protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
                - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
            ]
            replayed_gates: list[dict[str, Any]] = []
            previous_delta: np.ndarray | None = None
            for replay_index, replay_alpha in enumerate(
                calibration["applied_alpha_grid"],
                start=1,
            ):
                replay_delta = (
                    float(replay_alpha) - 1.0
                ) * visible_component
                replayed_gates.append(
                    paired_minimum_separation_gate(
                        replay_delta,
                        context_length=protocol.REAL_ANCHORED_CONTEXT_LENGTH,
                        dose_index=replay_index,
                        dose_calibration=calibration,
                        affected_channel_indices=evidence[
                            "affected_channel_indices"
                        ],
                        previous_delta=previous_delta,
                    )
                )
                previous_delta = replay_delta
            if replayed_gates != stored_gates:
                raise ValueError("structural paired gate replay changed")
            dose_plan = tuple(
                (
                    dose_index,
                    float(canonical_strength),
                    float(alpha),
                    copy.deepcopy(stored_gates[dose_index - 1]),
                )
                for dose_index, (canonical_strength, alpha) in enumerate(
                    zip(
                        calibration["strength_grid"],
                        calibration["applied_alpha_grid"],
                    ),
                    start=1,
                )
            )
        else:
            dose_plan = tuple(
                (
                    dose_index,
                    float(alpha - 1.0),
                    float(alpha),
                    None,
                )
                for dose_index, alpha in enumerate(legacy_alphas, start=1)
            )
        for dose_index, canonical_strength, alpha, pair_gate in dose_plan:
            pair_id = (
                f"cafe_structural_cf__{protocol.safe_id(str(background['dataset_id']))}"
                f"__{contract['capability_id']}__{protocol.safe_id(str(background['background_id']))}"
                f"__a{dose_index}"
            )
            paired_group_id = (
                f"cafe_structural_cf__{protocol.safe_id(str(background['dataset_id']))}"
                f"__{contract['capability_id']}__{protocol.safe_id(str(background['background_id']))}"
            )
            for member, member_alpha in ((0, 1.0), (1, alpha)):
                member_strength = 0.0 if member == 0 else canonical_strength
                target, metadata = apply_structural_contract(
                    background,
                    contract,
                    alpha=member_alpha,
                    allow_sensitivity=sensitivity,
                )
                metadata = {
                    **metadata,
                    "dose_index": dose_index,
                    "canonical_strength": member_strength,
                    "paired_treatment_strength": canonical_strength,
                    "applied_alpha": member_alpha,
                    "paired_treatment_applied_alpha": alpha,
                    "paired_minimum_separation_gate": copy.deepcopy(pair_gate),
                    "adjacent_minimum_separation_passed": (
                        None
                        if not isinstance(calibration, Mapping)
                        else bool(
                            contract[
                                "paired_minimum_separation_qualification"
                            ]["adjacent_minimum_separation_passed"]
                        )
                    ),
                }
                covariate_payload = background.get("known_future_covariates")
                covariates = (
                    None
                    if not isinstance(covariate_payload, Mapping)
                    else np.asarray(covariate_payload["target"], dtype=float)
                )
                intervention_rms = float(
                    np.sqrt(
                        np.mean(
                            np.asarray(metadata["truth_delta"], dtype=float)
                            ** 2
                        )
                    )
                )
                row_gate = (
                    None
                    if pair_gate is None
                    else {
                        "status": "not_applicable",
                        "accepted": None,
                        "reason_code": "repeated_authentic_baseline_member",
                        "dose_index": dose_index,
                        "paired_treatment_gate_status": "passed",
                        "dose_calibration_policy_sha256": calibration.get(
                            "dose_policy_sha256",
                            calibration["policy_sha256"],
                        ),
                    }
                    if member == 0
                    else copy.deepcopy(pair_gate)
                )
                yield {
                    "schema_version": STRUCTURAL_MASTER_SCHEMA,
                    "benchmark_track": "real_anchored_counterfactual",
                    "evaluation_table": (
                        "real_anchored_structural_sensitivity"
                        if sensitivity
                        else "real_anchored_counterfactual"
                    ),
                    "sample_id": f"{pair_id}__m{member}",
                    "master_sample_id": f"{pair_id}__m{member}",
                    "baseline_sample_id": f"{pair_id}__m0",
                    "counterfactual_pair_id": pair_id,
                    "paired_group_id": paired_group_id,
                    "counterfactual_member": member,
                    "dataset_id": str(background["dataset_id"]),
                    "config_id": str(background["config_id"]),
                    "task_id": str(background["task_view_id"]),
                    "task_view_id": str(background["task_view_id"]),
                    "profile_id": (
                        f"real_anchored_structural_{contract['capability_id']}_v2"
                    ),
                    "background_id": str(background["background_id"]),
                    "anchor_id": str(background["background_id"]),
                    "source_structural_background_target_sha256": str(
                        background["target_sha256"]
                    ),
                    "capability_id": str(contract["capability_id"]),
                    "generator_version": "cafe.structural_real_generator.v2",
                    "generator_family_role": "real_anchored_structural",
                    "generator_family_id": (
                        f"real_anchored_structural_{contract['capability_id']}_v2"
                    ),
                    "intensity": dose_index,
                    "intensity_lambda": member_strength,
                    "dose_index": dose_index,
                    "dose_parameter": "canonical_strength_lambda",
                    "physical_dose_parameter": (
                        "controlled_component_multiplier_alpha"
                    ),
                    "dose_value": member_strength,
                    "baseline_dose_value": 0.0,
                    "canonical_strength": member_strength,
                    "paired_treatment_strength": canonical_strength,
                    "applied_alpha": member_alpha,
                    "paired_treatment_applied_alpha": alpha,
                    "paired_minimum_separation_gate": row_gate,
                    "adjacent_minimum_separation_passed": metadata[
                        "adjacent_minimum_separation_passed"
                    ],
                    "dose_calibration_policy_sha256": (
                        None
                        if not isinstance(calibration, Mapping)
                        else str(
                            calibration.get(
                                "dose_policy_sha256",
                                calibration["policy_sha256"],
                            )
                        )
                    ),
                    "contract_dose_calibration_sha256": (
                        None
                        if not isinstance(calibration, Mapping)
                        else str(calibration["policy_sha256"])
                    ),
                    "seed_index": seed_index,
                    "sample_index": seed_index,
                    "context_length": protocol.REAL_ANCHORED_CONTEXT_LENGTH,
                    "horizon": protocol.HORIZON,
                    "target_dim": int(background["target_dim"]),
                    "covariate_dim": (
                        0 if covariates is None else int(covariates.shape[1])
                    ),
                    "frequency": str(background["frequency"]),
                    "season_length": int(background["season_length"]),
                    "calendar_season_length": int(
                        background["calendar_season_length"]
                    ),
                    "feature_period": int(background["feature_period"]),
                    "target": target.tolist(),
                    "target_sha256": protocol.target_and_covariate_sha256(
                        target,
                        covariates,
                    ),
                    "covariates": (
                        None if covariates is None else covariates.tolist()
                    ),
                    "covariate_column_names": (
                        []
                        if not isinstance(covariate_payload, Mapping)
                        else list(covariate_payload["column_names"])
                    ),
                    "hierarchy": None,
                    "generation_metadata": metadata,
                    "target_feature": "real_anchored_intervention_rms",
                    "target_feature_value": intervention_rms,
                    "intensity_target_feature_value": intervention_rms,
                    "sampled_generator_parameters": {
                        "alpha": member_alpha,
                        "canonical_strength": member_strength,
                        "controlled_component": metadata[
                            "controlled_component"
                        ],
                    },
                    "parameter_sampling": {
                        "policy": (
                            "authentic_structural_background_contract_v2"
                        ),
                        "background_id": str(background["background_id"]),
                        "contract_sha256": str(contract["contract_sha256"]),
                        "dose_calibration_policy_sha256": (
                            None
                            if not isinstance(calibration, Mapping)
                            else str(
                                calibration.get(
                                    "dose_policy_sha256",
                                    calibration["policy_sha256"],
                                )
                            )
                        ),
                    },
                    "intensity_calibration": {
                        "policy": (
                            "reference_q75_capability_specific_alpha_grid_v1"
                            if isinstance(calibration, Mapping)
                            else "legacy_physical_component_alpha_grid_v1"
                        ),
                        "scope": (
                            "structural_real_history_only_contract"
                        ),
                        "canonical_strength_grid": [
                            float(plan[1]) for plan in dose_plan
                        ],
                        "selected_alphas": [
                            float(plan[2]) for plan in dose_plan
                        ],
                        "applied_alpha_grid": [
                            float(plan[2]) for plan in dose_plan
                        ],
                        "history_target_grid": (
                            []
                            if not isinstance(calibration, Mapping)
                            else list(calibration["history_target_grid"])
                        ),
                        "future_target_grid": (
                            []
                            if not isinstance(calibration, Mapping)
                            else list(calibration["future_target_grid"])
                        ),
                        "dose_calibration_policy_sha256": (
                            None
                            if not isinstance(calibration, Mapping)
                            else str(
                                calibration.get(
                                    "dose_policy_sha256",
                                    calibration["policy_sha256"],
                                )
                            )
                        ),
                        "contract_dose_calibration_sha256": (
                            None
                            if not isinstance(calibration, Mapping)
                            else str(calibration["policy_sha256"])
                        ),
                    },
                    "dose_calibration": (
                        None
                        if not isinstance(calibration, Mapping)
                        else copy.deepcopy(dict(calibration))
                    ),
                    "shared_standardization": copy.deepcopy(
                        background["target_standardization"]
                    ),
                    "mase_period": int(background["mase_period"]),
                    "mase_scale": float(background["mase_scale"]),
                    "mase_scale_by_target": list(
                        background["mase_scale_by_target"]
                    ),
                    "mase_scale_effective_period_by_target": list(
                        background["mase_scale_effective_period_by_target"]
                    ),
                    "mase_scale_fallback_target_indices": list(
                        background["mase_scale_fallback_target_indices"]
                    ),
                    "mase_scale_policy": str(background["mase_scale_policy"]),
                    "mase_scale_source": (
                        "shared_unmodified_real_l336_history"
                    ),
                    "future_sha256": array_sha256(
                        target[protocol.REAL_ANCHORED_CONTEXT_LENGTH :]
                    ),
                    "anti_copy_gate": {
                        "status": "not_applicable",
                        "reason_code": "intentional_real_anchor_counterfactual",
                    },
                    "mandatory_input_ablation": copy.deepcopy(
                        contract.get("mandatory_input_ablation")
                    ),
                    "input_ablation_status": (
                        "required_not_yet_materialized"
                        if contract.get("mandatory_input_ablation") is not None
                        else "not_applicable"
                    ),
                    "excluded_from_univariate_real_anchored_rank": True,
                }


def _structural_donor_commitment_entry(
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit one exact model-visible donor history to its upstream identity."""

    target = np.asarray(sample["target"], dtype=float)
    context = int(sample["context_length"])
    target_dim = int(sample["target_dim"])
    if (
        target.ndim != 2
        or target.shape[0] != context + int(sample["horizon"])
        or target.shape[1] != target_dim
        or not np.isfinite(target).all()
    ):
        raise ValueError("structural donor commitment target is invalid")
    history = target[:context]
    metadata = sample.get("generation_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("structural donor sample lacks generation metadata")
    entry: dict[str, Any] = {
        "schema_version": STRUCTURAL_DONOR_COMMITMENT_ENTRY_SCHEMA,
        "sample_id": str(sample["sample_id"]),
        "dataset_id": str(sample["dataset_id"]),
        "background_id": str(sample["background_id"]),
        "capability_id": str(sample["capability_id"]),
        "dose_index": int(sample["dose_index"]),
        "dose_value": float(sample["dose_value"]),
        "intensity_lambda": float(sample["intensity_lambda"]),
        "canonical_strength": float(sample["canonical_strength"]),
        "paired_treatment_strength": float(
            sample["paired_treatment_strength"]
        ),
        "applied_alpha": float(sample["applied_alpha"]),
        "paired_treatment_applied_alpha": float(
            sample["paired_treatment_applied_alpha"]
        ),
        "dose_calibration_policy_sha256": sample.get(
            "dose_calibration_policy_sha256"
        ),
        "paired_minimum_separation_gate_sha256": (
            None
            if not isinstance(
                sample.get("paired_minimum_separation_gate"), Mapping
            )
            else protocol.json_sha256(
                sample["paired_minimum_separation_gate"]
            )
        ),
        "counterfactual_member": int(sample["counterfactual_member"]),
        "evaluation_table": str(sample["evaluation_table"]),
        "seed_index": int(sample["seed_index"]),
        "target_dim": target_dim,
        "context_length": context,
        "source_contract_sha256": str(metadata["contract_sha256"]),
        "source_structural_background_target_sha256": str(
            sample["source_structural_background_target_sha256"]
        ),
        "visible_history_sha256": array_sha256(history),
        "visible_history_by_channel_sha256": {
            str(channel): array_sha256(history[:, channel])
            for channel in range(target_dim)
        },
    }
    entry["entry_sha256"] = protocol.json_sha256(entry)
    return entry


def build_structural_donor_commitment_manifest(
    backgrounds: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
) -> dict[str, Any]:
    """Freeze every common/cross donor history from calibration artifacts.

    The returned root is useful only when its file record is bound by the
    immutable calibration bundle.  Generation and validation must therefore
    compare this manifest with that upstream bundle rather than trusting a
    row-local copy or a self-hash.
    """

    public_backgrounds = [
        public_structural_background(background) for background in backgrounds
    ]
    selected_contracts = [
        row
        for row in contract_rows
        if row.get("capability_id") in {
            "common_factor",
            "cross_series_dependence",
        }
    ]
    dose_hashes_by_capability: dict[str, set[str]] = {}
    for row in selected_contracts:
        contract = row.get("contract")
        if not isinstance(contract, Mapping):
            continue
        calibration = contract.get("dose_calibration")
        if not isinstance(calibration, Mapping):
            continue
        dose_hashes_by_capability.setdefault(
            str(row["capability_id"]),
            set(),
        ).add(
            str(
                calibration.get(
                    "dose_policy_sha256",
                    calibration["policy_sha256"],
                )
            )
        )
    if any(len(values) != 1 for values in dose_hashes_by_capability.values()):
        raise ValueError("structural donor rows disagree on frozen dose mapping")
    dose_hash_by_capability = {
        capability_id: next(iter(values))
        for capability_id, values in sorted(dose_hashes_by_capability.items())
    }
    samples = (
        sample
        for sensitivity in (False, True)
        for sample in iter_structural_real_anchored_samples(
            public_backgrounds,
            selected_contracts,
            seed_indexes=range(len(public_backgrounds)),
            sensitivity=sensitivity,
        )
    )
    entries = sorted(
        (_structural_donor_commitment_entry(sample) for sample in samples),
        key=lambda entry: str(entry["sample_id"]),
    )
    sample_ids = [str(entry["sample_id"]) for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("structural donor commitment sample IDs are not unique")
    payload: dict[str, Any] = {
        "schema_version": STRUCTURAL_DONOR_COMMITMENT_SCHEMA,
        "commitment_policy": STRUCTURAL_DONOR_COMMITMENT_POLICY,
        "dataset_id": str(dataset_id),
        "context_length": protocol.REAL_ANCHORED_CONTEXT_LENGTH,
        "source_structural_background_bank_sha256": protocol.json_sha256(
            public_backgrounds
        ),
        "source_structural_contract_bank_sha256": protocol.json_sha256(
            list(contract_rows)
        ),
        "dose_calibration_policy_sha256_by_capability": (
            dose_hash_by_capability
        ),
        "entry_count": len(entries),
        "eligible_donor_sample_ids_sha256": protocol.json_sha256(sample_ids),
        "entries_sha256": protocol.json_sha256(entries),
        "entries": entries,
    }
    payload["commitment_root_sha256"] = protocol.json_sha256(payload)
    return payload


def validate_structural_donor_commitment_manifest(
    manifest: Mapping[str, Any],
    backgrounds: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
) -> None:
    """Bind a donor commitment manifest to its calibration banks."""

    if manifest.get("schema_version") != STRUCTURAL_DONOR_COMMITMENT_SCHEMA:
        raise ValueError("unsupported structural donor commitment schema")
    if manifest.get("commitment_policy") != STRUCTURAL_DONOR_COMMITMENT_POLICY:
        raise ValueError("structural donor commitment policy changed")
    self_payload = dict(manifest)
    observed_root = self_payload.pop("commitment_root_sha256", None)
    if observed_root != protocol.json_sha256(self_payload):
        raise ValueError("structural donor commitment root self-hash mismatch")
    expected = build_structural_donor_commitment_manifest(
        backgrounds,
        contract_rows,
        dataset_id=dataset_id,
    )
    if dict(manifest) != expected:
        raise ValueError(
            "structural donor commitments disagree with calibration banks"
        )


def _committed_donor_entry(
    donor: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve and verify one donor against an already trusted manifest."""

    payload = dict(manifest)
    observed_root = payload.pop("commitment_root_sha256", None)
    if observed_root != protocol.json_sha256(payload):
        raise ValueError("structural donor commitment root self-hash mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("structural donor commitment entries are missing")
    by_sample_id = {
        str(entry.get("sample_id")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if len(by_sample_id) != len(entries):
        raise ValueError("structural donor commitment entries are ambiguous")
    donor_sample_id = str(donor.get("sample_id", ""))
    entry = by_sample_id.get(donor_sample_id)
    if not isinstance(entry, Mapping):
        raise ValueError("structural donor is absent from calibration commitment")
    entry_payload = dict(entry)
    entry_hash = entry_payload.pop("entry_sha256", None)
    if entry_hash != protocol.json_sha256(entry_payload):
        raise ValueError("structural donor commitment entry hash mismatch")
    metadata = donor.get("generation_metadata")
    identity_matches = bool(
        isinstance(metadata, Mapping)
        and entry.get("dataset_id") == donor.get("dataset_id")
        and entry.get("background_id") == donor.get("background_id")
        and entry.get("capability_id") == donor.get("capability_id")
        and entry.get("dose_index") == donor.get("dose_index")
        and entry.get("dose_value") == donor.get("dose_value")
        and entry.get("intensity_lambda") == donor.get("intensity_lambda")
        and entry.get("canonical_strength") == donor.get("canonical_strength")
        and entry.get("paired_treatment_strength")
        == donor.get("paired_treatment_strength")
        and entry.get("applied_alpha") == donor.get("applied_alpha")
        and entry.get("paired_treatment_applied_alpha")
        == donor.get("paired_treatment_applied_alpha")
        and entry.get("dose_calibration_policy_sha256")
        == donor.get("dose_calibration_policy_sha256")
        and entry.get("paired_minimum_separation_gate_sha256")
        == (
            None
            if not isinstance(
                donor.get("paired_minimum_separation_gate"), Mapping
            )
            else protocol.json_sha256(
                donor["paired_minimum_separation_gate"]
            )
        )
        and entry.get("counterfactual_member")
        == donor.get("counterfactual_member")
        and entry.get("evaluation_table") == donor.get("evaluation_table")
        and entry.get("seed_index") == donor.get("seed_index")
        and entry.get("target_dim") == donor.get("target_dim")
        and entry.get("context_length") == donor.get("context_length")
        and entry.get("source_contract_sha256")
        == metadata.get("contract_sha256")
        and entry.get("source_structural_background_target_sha256")
        == donor.get("source_structural_background_target_sha256")
    )
    if not identity_matches:
        raise ValueError("structural donor identity differs from commitment")
    target = np.asarray(donor.get("target"), dtype=float)
    context = int(donor["context_length"])
    target_dim = int(donor["target_dim"])
    if target.shape != (context + int(donor["horizon"]), target_dim):
        raise ValueError("structural donor target shape differs from commitment")
    history = target[:context]
    committed_by_channel = entry.get("visible_history_by_channel_sha256")
    history_matches = bool(
        entry.get("visible_history_sha256") == array_sha256(history)
        and isinstance(committed_by_channel, Mapping)
        and committed_by_channel
        == {
            str(channel): array_sha256(history[:, channel])
            for channel in range(target_dim)
        }
    )
    if not history_matches:
        raise ValueError("structural donor history differs from commitment")
    return entry


def build_matched_input_ablation_task(
    sample: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    donor_commitment_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize the mandatory matched-donor panel attribution task.

    Only model-visible input channels named by the frozen ablation contract are
    replaced.  The assessed target histories and every scored future value stay
    bit-identical to the main sample.  This row is a separate audit and cannot
    enter the primary capability score.
    """

    if sample.get("schema_version") != STRUCTURAL_MASTER_SCHEMA:
        raise ValueError("main sample has an unsupported structural schema")
    if donor.get("schema_version") != STRUCTURAL_MASTER_SCHEMA:
        raise ValueError("donor has an unsupported structural schema")
    committed_entry = (
        None
        if donor_commitment_manifest is None
        else _committed_donor_entry(donor, donor_commitment_manifest)
    )
    if sample.get("background_id") == donor.get("background_id"):
        raise ValueError("input ablation donor must be a distinct background")
    for key in (
        "capability_id",
        "dose_index",
        "counterfactual_member",
        "target_dim",
        "canonical_strength",
        "paired_treatment_strength",
        "dose_calibration_policy_sha256",
    ):
        if sample.get(key) != donor.get(key):
            raise ValueError(f"input ablation donor mismatch for {key}")
    contract = sample.get("mandatory_input_ablation")
    if not isinstance(contract, Mapping) or contract.get("required") is not True:
        raise ValueError("sample does not require a structural input ablation")
    target = np.asarray(sample["target"], dtype=float)
    donor_target = np.asarray(donor["target"], dtype=float)
    if target.shape != donor_target.shape:
        raise ValueError("input ablation donor target shape mismatch")
    context = int(sample["context_length"])
    assessed = [int(value) for value in contract["assessed_target_indices"]]
    ablated = [int(value) for value in contract["ablated_input_indices"]]
    result_target = target.copy()
    affine: dict[str, Any] = {}
    for channel in ablated:
        source = donor_target[:context, channel]
        destination = target[:context, channel]
        source_center = float(np.mean(source))
        source_scale = max(float(np.std(source)), 1e-9)
        destination_center = float(np.mean(destination))
        destination_scale = max(float(np.std(destination)), 1e-9)
        result_target[:context, channel] = (
            (source - source_center) / source_scale * destination_scale
            + destination_center
        )
        affine[str(channel)] = {
            "donor_center": source_center,
            "donor_scale": source_scale,
            "recipient_center": destination_center,
            "recipient_scale": destination_scale,
        }
    if assessed:
        np.testing.assert_array_equal(
            result_target[:context, assessed],
            target[:context, assessed],
        )
    np.testing.assert_array_equal(result_target[context:], target[context:])
    result = copy.deepcopy(dict(sample))
    result["schema_version"] = STRUCTURAL_MASTER_SCHEMA
    result["input_ablation_schema_version"] = STRUCTURAL_ABLATION_SCHEMA
    result["sample_id"] = f"{sample['sample_id']}__input_ablation"
    result["master_sample_id"] = result["sample_id"]
    source_evaluation_table = str(
        sample.get("evaluation_table", "real_anchored_counterfactual")
    )
    result["evaluation_table"] = (
        "real_anchored_structural_sensitivity_input_ablation"
        if source_evaluation_table == "real_anchored_structural_sensitivity"
        else "real_anchored_input_ablation"
    )
    result["input_ablation_source_sample_id"] = str(sample["sample_id"])
    result["input_ablation_source_pair_id"] = str(
        sample["counterfactual_pair_id"]
    )
    result["input_ablation_source_paired_group_id"] = str(
        sample["paired_group_id"]
    )
    result["counterfactual_pair_id"] = (
        f"{sample['counterfactual_pair_id']}__input_ablation"
    )
    result["paired_group_id"] = (
        f"{sample['paired_group_id']}__input_ablation"
    )
    result["baseline_sample_id"] = (
        f"{sample['baseline_sample_id']}__input_ablation"
    )
    result["donor_sample_id"] = str(donor["sample_id"])
    result["donor_background_id"] = str(donor["background_id"])
    result["donor_seed_index"] = int(donor["seed_index"])
    if committed_entry is not None and donor_commitment_manifest is not None:
        result["structural_donor_commitment_root_sha256"] = str(
            donor_commitment_manifest["commitment_root_sha256"]
        )
        result["donor_structural_commitment_entry_sha256"] = str(
            committed_entry["entry_sha256"]
        )
    result["target"] = result_target.tolist()
    result_covariates = (
        None
        if result.get("covariates") is None
        else np.asarray(result["covariates"], dtype=float)
    )
    result["target_sha256"] = protocol.target_and_covariate_sha256(
        result_target,
        result_covariates,
    )
    result["input_ablation_status"] = "materialized"
    result["input_ablation_metadata"] = {
        "assessed_target_indices": assessed,
        "ablated_input_indices": ablated,
        "replacement_policy": contract["replacement_policy"],
        "affine_match_by_channel": affine,
        "assessed_target_history_unchanged": True,
        "scored_future_unchanged": True,
        "excluded_from_primary_score": True,
        "reported_as_separate_attribution_audit": True,
        "same_real_anchored_counterfactual_artifact_required": True,
        "donor_visible_history_by_channel": {
            str(channel): donor_target[:context, channel].tolist()
            for channel in ablated
        },
        "donor_visible_history_sha256": array_sha256(
            donor_target[:context, ablated]
        ),
        "ablated_visible_history_sha256": array_sha256(
            result_target[:context, ablated]
        ),
        "donor_selection_policy": (
            "global_eligible_background_successor_shard_invariant_v1"
        ),
    }
    if committed_entry is not None and donor_commitment_manifest is not None:
        result["input_ablation_metadata"]["donor_upstream_commitment"] = {
            "manifest_schema_version": str(
                donor_commitment_manifest["schema_version"]
            ),
            "commitment_policy": str(
                donor_commitment_manifest["commitment_policy"]
            ),
            "commitment_root_sha256": str(
                donor_commitment_manifest["commitment_root_sha256"]
            ),
            "entries_sha256": str(
                donor_commitment_manifest["entries_sha256"]
            ),
            "entry_sha256": str(committed_entry["entry_sha256"]),
            "source_structural_background_bank_sha256": str(
                donor_commitment_manifest[
                    "source_structural_background_bank_sha256"
                ]
            ),
            "source_structural_contract_bank_sha256": str(
                donor_commitment_manifest[
                    "source_structural_contract_bank_sha256"
                ]
            ),
        }
    result["excluded_from_primary_score"] = True
    return result


def iter_mandatory_structural_input_ablation_tasks(
    samples: Sequence[Mapping[str, Any]],
    *,
    donor_samples: Sequence[Mapping[str, Any]] | None = None,
    donor_commitment_manifest: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Materialize exactly one matched ablation row per required main row.

    Donors are selected by a deterministic cyclic permutation within identical
    dataset/capability/dose/member/dimension cells. ``donor_samples`` may contain
    the complete eligible-bank population while ``samples`` contains only one
    seed shard; this makes donor identity independent of shard boundaries. A
    cell with fewer than two authentic backgrounds fails closed. Formal stage
    generation also supplies the calibration-frozen ``donor_commitment_manifest``;
    direct callers may omit it only for backward-compatible in-memory fixtures.
    Consequently every common/cross main pair yields one corresponding
    two-member ablation pair.
    """

    def group_rows(
        source_rows: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
        output: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for sample in source_rows:
            contract = sample.get("mandatory_input_ablation")
            if (
                not isinstance(contract, Mapping)
                or contract.get("required") is not True
            ):
                continue
            if sample.get("evaluation_table") not in {
                "real_anchored_counterfactual",
                "real_anchored_structural_sensitivity",
            }:
                raise ValueError(
                    "mandatory ablation source must be a main structural row"
                )
            key = (
                str(sample["dataset_id"]),
                str(sample["capability_id"]),
                int(sample["dose_index"]),
                int(sample["counterfactual_member"]),
                int(sample["target_dim"]),
            )
            output.setdefault(key, []).append(sample)
        return output

    source_groups = group_rows(samples)
    donor_groups = group_rows(
        samples if donor_samples is None else donor_samples
    )
    for key in sorted(source_groups):
        donor_rows = sorted(
            donor_groups.get(key, []),
            key=lambda row: (
                int(row["seed_index"]),
                str(row["background_id"]),
                str(row["sample_id"]),
            ),
        )
        donor_background_ids = [
            str(row["background_id"]) for row in donor_rows
        ]
        if len(set(donor_background_ids)) < 2:
            raise ValueError(
                "mandatory structural input ablation requires two distinct "
                f"backgrounds in cell {key}"
            )
        donor_by_background = {
            str(row["background_id"]): index
            for index, row in enumerate(donor_rows)
        }
        for sample in sorted(
            source_groups[key],
            key=lambda row: (
                int(row["seed_index"]),
                str(row["sample_id"]),
            ),
        ):
            background_id = str(sample["background_id"])
            if background_id not in donor_by_background:
                raise ValueError(
                    "structural ablation source is absent from donor population"
                )
            donor = donor_rows[
                (donor_by_background[background_id] + 1) % len(donor_rows)
            ]
            if donor.get("background_id") == sample.get("background_id"):
                raise ValueError("structural ablation donor selection recycled source")
            yield build_matched_input_ablation_task(
                sample,
                donor,
                donor_commitment_manifest=donor_commitment_manifest,
            )
