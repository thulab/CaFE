#!/usr/bin/env python3
"""Build the Paper v8 clean-deterministic ten-capability pilot.

This is intentionally a test artifact builder.  It reuses one available v7
dataset split per capability, but it does not mutate the frozen v7 generator
or its production artifact.  The formal v8 calibration must rebuild the
three-way real-window split after this pilot has validated the feature and
family choices.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.services.synthetic_feature_gate import (  # noqa: E402
    CLEAN_DETERMINISTIC_EXCLUDED_CONTROLS,
    evaluate_feature_support_gate,
)
from app.services.synthetic_generation_service import (  # noqa: E402
    _normalize_covariates,
    _realized_features,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    GeneratorConditioning,
    resolve_generator_conditioning,
)
from app.services.synthetic_v8_generation import (  # noqa: E402
    GENERATOR_VERSION,
    PRIMARY_FAMILY_BY_CAPABILITY,
    REQUIRED_REAL_FEATURES_BY_CAPABILITY,
    SECONDARY_FAMILY_BY_CAPABILITY,
    add_observation_noise_to_history,
    cross_series_identifiability_gate,
    derive_deterministic_parameters,
    generate_deterministic_sample,
    standardize_cross_series_counterfactual_member,
)
from synthetic_feature_profile import (  # noqa: E402
    feature_vector,
    summarize_feature_rows,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime" / "paper_exp" / "v8_test"
V7_SUITE_DIR = REPO_ROOT / "runtime" / "paper_exp" / "v7" / "01_nine_capability_suite"
V7_REAL_SAMPLES = (
    REPO_ROOT
    / "runtime"
    / "paper_exp"
    / "v7"
    / "02_real_source_window_suite"
    / "real_source_samples.jsonl"
)
CONTEXT_LENGTH = 504
HORIZON = 48
INTENSITIES = (1, 2, 3, 4, 5)
PRIMARY_TARGET_FEATURE = {
    "trend": "curvature_abs",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "regime_sparse_transition_score",
    "nonlinear_persistence": "nonlinear_conditional_gain",
    "predictable_intermittency": "intermittency_clock_incremental_r2",
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    "cross_series_dependence": "cross_series_incremental_r2",
    "covariate_response": "covariate_incremental_r2",
}
SELECTED_DATASET = {
    "trend": ("gift_electricity_h", "univariate"),
    "multi_seasonal": ("gift_electricity_h", "univariate"),
    "time_varying_seasonality": ("gift_electricity_h", "univariate"),
    "regime_switching": ("gift_electricity_h", "univariate"),
    "nonlinear_persistence": ("gift_electricity_h", "univariate"),
    "predictable_intermittency": ("gift_electricity_h", "univariate"),
    "common_factor": ("electricity_hourly_panel", "common_factor"),
    "hierarchical_coherence": ("gefcom2012_load", "hierarchy"),
    "cross_series_dependence": (
        "electricity_hourly_panel",
        "common_factor",
    ),
    "covariate_response": ("gefcom2012_load", "covariate"),
}
CONDITIONING_SOURCE_CAPABILITY = {
    "cross_series_dependence": "common_factor",
}
PARAMETER_SEED_BASE = 2026072300
PATH_SEED_BASE = 2026072400
ROBUSTNESS_NOISE_RATIO = 0.15
ROBUSTNESS_SEED_BASE = 2026072500
COUNTERFACTUAL_CAPABILITIES = frozenset(
    {"cross_series_dependence", "covariate_response"}
)
NUISANCE_FINGERPRINT_FIELDS = {
    "trend": (
        "slope_jitter_by_target",
        "curvature_sign_by_target",
    ),
    "multi_seasonal": ("periods",),
    "time_varying_seasonality": (
        "primary_period",
        "modulation_period",
        "modulation_harmonic_weight",
        "modulation_harmonic_phase",
    ),
    "regime_switching": (
        "dwell_pattern",
        "dwell_anchor_offset",
        "initial_regime_state",
    ),
    "nonlinear_persistence": (
        "nonlinear_lag",
        "deterministic_forcing",
    ),
    "predictable_intermittency": (
        "pulse_interval_pattern",
        "pulse_anchor_offset",
        "deterministic_texture",
    ),
    "common_factor": (
        "loadings",
        "local_period_multipliers",
        "shared_factor_process",
    ),
    "hierarchical_coherence": (
        "aggregate_share_by_child",
        "child_permutation",
        "contrast_period_multipliers",
        "local_contrast_loadings",
    ),
    "cross_series_dependence": (
        "cross_lag_steps",
        "historical_event_centers",
        "counterfactual_response_center_offset",
        "responder_gains",
    ),
    "covariate_response": (
        "counterfactual_weather_transform_selected",
        "counterfactual_event_start_options",
        "event_width",
        "weather_effect_by_target",
        "event_effect_by_target",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-seeds", type=int, default=32)
    parser.add_argument("--secondary-seeds", type=int, default=8)
    parser.add_argument("--calibration-seeds", type=int, default=12)
    parser.add_argument("--inference-seeds", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def selected_rows(
    rows: list[dict[str, Any]],
    capability_id: str,
) -> list[dict[str, Any]]:
    dataset_id, task_id = SELECTED_DATASET[capability_id]
    matched = [
        row
        for row in rows
        if row["dataset_id"] == dataset_id
        and row["task_view_id"].endswith(f"::{task_id}")
        and int(row["context_length"]) == CONTEXT_LENGTH
        and int(row["horizon"]) == HORIZON
    ]
    if not matched:
        raise ValueError(f"no real rows for {capability_id}/{dataset_id}/{task_id}")
    return matched


def real_feature_rows(
    rows: list[dict[str, Any]],
    *,
    history_only: bool,
    capability_id: str,
) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for row in rows:
        stop = int(row["context_length"]) if history_only else None
        target = np.asarray(row["target"], dtype=float)[:stop]
        covariates = (
            None
            if row.get("covariates") is None
            else np.asarray(row["covariates"], dtype=float)[:stop]
        )
        output.append(
            feature_vector(
                target,
                int(row["season_length"]),
                covariates=covariates,
                context_length=min(CONTEXT_LENGTH, len(target)),
                hierarchy=row.get("hierarchy"),
                include_cross_series_predictability=(
                    capability_id == "cross_series_dependence"
                ),
            )
        )
    return output


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    names = sorted({name for row in rows for name in row})
    return summarize_feature_rows(rows, names)


def sampled_summary(
    summary: dict[str, dict[str, float]],
    rng: np.random.Generator,
    *,
    feature_rows: list[dict[str, float]],
    anchor_index: int,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Sample marginal values from one empirical-copula anchor window.

    Each feature keeps the selected real window's empirical rank, then maps
    that rank into the central p25-p75 support used by the pilot. This retains
    cross-feature dependence without copying the real window's raw scale or
    exposing tail outliers.
    """

    if not feature_rows:
        raise ValueError("feature_rows must not be empty")
    normalized_anchor = int(anchor_index) % len(feature_rows)
    anchor = feature_rows[normalized_anchor]
    fallback_quantile = float(rng.uniform(0.25, 0.75))
    sampled = deepcopy(summary)
    sampled_quantiles: dict[str, float] = {}
    fallback_features: list[str] = []
    for feature, values in sampled.items():
        p25 = float(values.get("p25", values.get("p50", 0.0)))
        p50 = float(values.get("p50", p25))
        p75 = float(values.get("p75", p50))
        finite_values = np.asarray(
            [
                float(row[feature])
                for row in feature_rows
                if feature in row and math.isfinite(float(row[feature]))
            ],
            dtype=float,
        )
        anchor_value = float(anchor.get(feature, math.nan))
        if finite_values.size and math.isfinite(anchor_value):
            lower_count = float(np.sum(finite_values < anchor_value))
            equal_count = float(np.sum(finite_values == anchor_value))
            empirical_rank = (
                lower_count + 0.5 * equal_count
            ) / finite_values.size
            quantile = float(0.25 + 0.5 * empirical_rank)
        else:
            quantile = fallback_quantile
            fallback_features.append(feature)
        values["p50"] = float(
            np.interp(quantile, [0.25, 0.5, 0.75], [p25, p50, p75])
        )
        values["sampled_quantile"] = quantile
        sampled_quantiles[feature] = quantile
    return sampled, {
        "policy": "empirical_copula_anchor_rank_mapped_to_p25_p75",
        "source_window_index": normalized_anchor,
        "source_window_count": len(feature_rows),
        "sampled_quantiles": sampled_quantiles,
        "fallback_features": sorted(fallback_features),
    }


def sample_parameters(
    capability_id: str,
    summary: dict[str, dict[str, float]],
    *,
    feature_rows: list[dict[str, float]],
    season_length: int,
    sample_index: int,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(PARAMETER_SEED_BASE + 1009 * sample_index)
    anchor_stride = 17
    while math.gcd(anchor_stride, len(feature_rows)) != 1:
        anchor_stride += 2
    anchor_index = (
        PARAMETER_SEED_BASE % len(feature_rows)
        + anchor_stride * sample_index
    ) % len(feature_rows)
    sampled, sampling_metadata = sampled_summary(
        summary,
        rng,
        feature_rows=feature_rows,
        anchor_index=anchor_index,
    )
    sampling_metadata["anchor_stride"] = anchor_stride
    parameters, mappings = derive_deterministic_parameters(
        capability_id,
        sampled,
        season_length=season_length,
        context_length=CONTEXT_LENGTH,
    )
    return parameters, mappings, sampling_metadata


def v8_conditioning(
    base: GeneratorConditioning,
    *,
    parameters: dict[str, float],
    lambdas: tuple[float, ...],
) -> GeneratorConditioning:
    # Legacy v7 structure scales and lambda inversions are generator-family
    # specific.  Carrying them into v8 would silently reintroduce the old DGP.
    return dataclasses.replace(
        base,
        parameters=dict(parameters),
        intensity_lambdas=tuple(float(value) for value in lambdas),
        calibration_method="v8_clean_deterministic_pilot_inverse_mapping",
        artifact_generator_version=GENERATOR_VERSION,
    )


def standardize_sample(
    capability_id: str,
    target: np.ndarray,
    covariates: np.ndarray | None,
    *,
    metadata: dict[str, Any] | None = None,
    context_length: int = CONTEXT_LENGTH,
) -> tuple[np.ndarray, np.ndarray | None]:
    if capability_id == "cross_series_dependence":
        if metadata is None:
            raise ValueError(
                "cross-series standardization requires generation metadata"
            )
        target, normalization = (
            standardize_cross_series_counterfactual_member(
                target,
                context_length=context_length,
                metadata=metadata,
            )
        )
        metadata["counterfactual_standardization"] = normalization
    elif capability_id == "hierarchical_coherence":
        target = _standardize_hierarchy_by_context(target, context_length)
    else:
        target = _standardize_by_context(target, context_length)
    if covariates is not None:
        covariates = _normalize_covariates(covariates, context_length)
    return target, covariates


def attach_cross_series_identifiability_gates(
    rows: list[dict[str, Any]],
) -> None:
    pairs: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = row.get("counterfactual_pair_id")
        member = row.get("counterfactual_member")
        if pair_id is not None and member is not None:
            pairs[str(pair_id)][int(member)] = row
    for pair_id, members in pairs.items():
        if set(members) != {0, 1}:
            raise ValueError(f"incomplete cross-series pair: {pair_id}")
        first = members[0]
        second = members[1]
        enforced = (
            first["generator_family_role"] == "primary"
            and int(first["intensity"]) == 5
        )
        gate = cross_series_identifiability_gate(
            np.asarray(first["target"], dtype=float),
            np.asarray(second["target"], dtype=float),
            context_length=int(first["context_length"]),
            metadata=first["generation_metadata"],
            enforced=enforced,
        )
        first["generation_metadata"]["identifiability_gate"] = deepcopy(gate)
        second["generation_metadata"]["identifiability_gate"] = deepcopy(gate)
        if enforced and not gate["accepted"]:
            raise ValueError(
                "cross-series identifiability gate failed for "
                f"{pair_id}: {gate}"
            )


def measured_features(
    capability_id: str,
    target: np.ndarray,
    covariates: np.ndarray | None,
    season_length: int,
) -> dict[str, float]:
    hierarchy = (
        "additive_first"
        if capability_id == "hierarchical_coherence"
        else None
    )
    result = _realized_features(
        target,
        covariates,
        season_length,
        CONTEXT_LENGTH,
    )
    result.update(
        feature_vector(
            target,
            season_length,
            covariates=covariates,
            context_length=CONTEXT_LENGTH,
            hierarchy=hierarchy,
            include_cross_series_predictability=(
                capability_id == "cross_series_dependence"
            ),
        )
    )
    return result


def generate_one(
    capability_id: str,
    *,
    family_role: str,
    intensity: int,
    sample_index: int,
    base_conditioning: GeneratorConditioning,
    parameter_summary: dict[str, dict[str, float]],
    parameter_feature_rows: list[dict[str, float]],
    intensity_lambdas: tuple[float, ...],
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    dict[str, Any],
    dict[str, float],
    dict[str, float],
    list[dict[str, Any]],
    dict[str, Any],
]:
    generation_index = (
        sample_index // 2
        if capability_id in COUNTERFACTUAL_CAPABILITIES
        else sample_index
    )
    parameters, mappings, sampling_metadata = sample_parameters(
        capability_id,
        parameter_summary,
        feature_rows=parameter_feature_rows,
        season_length=base_conditioning.season_length,
        sample_index=generation_index,
    )
    conditioning = v8_conditioning(
        base_conditioning,
        parameters=parameters,
        lambdas=intensity_lambdas,
    )
    path_seed = PATH_SEED_BASE + 10007 * generation_index
    target, metadata, covariates = generate_deterministic_sample(
        capability_id,
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        base_conditioning.target_dim,
        base_conditioning.season_length,
        intensity,
        np.random.default_rng(path_seed),
        conditioning=conditioning,
        family_role=family_role,
        counterfactual_variant=(
            sample_index % 2
            if capability_id in COUNTERFACTUAL_CAPABILITIES
            else 0
        ),
    )
    target, covariates = standardize_sample(
        capability_id,
        target,
        covariates,
        metadata=metadata,
    )
    features = measured_features(
        capability_id,
        target,
        covariates,
        base_conditioning.season_length,
    )
    return (
        target,
        covariates,
        metadata,
        features,
        parameters,
        mappings,
        sampling_metadata,
    )


def response_curve(
    capability_id: str,
    *,
    family_role: str,
    base_conditioning: GeneratorConditioning,
    parameter_summary: dict[str, dict[str, float]],
    parameter_feature_rows: list[dict[str, float]],
    seed_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature = PRIMARY_TARGET_FEATURE[capability_id]
    lambda_grid = np.linspace(0.0, 1.0, 21)
    means: list[float] = []
    for lambda_value in lambda_grid:
        values: list[float] = []
        lambdas = (float(lambda_value),) * 5
        for sample_index in range(seed_count):
            _, _, _, features, _, _, _ = generate_one(
                capability_id,
                family_role=family_role,
                intensity=1,
                sample_index=sample_index,
                base_conditioning=base_conditioning,
                parameter_summary=parameter_summary,
                parameter_feature_rows=parameter_feature_rows,
                intensity_lambdas=lambdas,
            )
            values.append(float(features[feature]))
        means.append(float(np.mean(values)))
    return lambda_grid, np.maximum.accumulate(np.asarray(means, dtype=float))


def response_at_selected_lambdas(
    capability_id: str,
    *,
    family_role: str,
    base_conditioning: GeneratorConditioning,
    parameter_summary: dict[str, dict[str, float]],
    parameter_feature_rows: list[dict[str, float]],
    intensity_lambdas: tuple[float, ...],
    seed_count: int,
) -> list[float]:
    feature = PRIMARY_TARGET_FEATURE[capability_id]
    means: list[float] = []
    for intensity in INTENSITIES:
        values = []
        for sample_index in range(seed_count):
            _, _, _, features, _, _, _ = generate_one(
                capability_id,
                family_role=family_role,
                intensity=intensity,
                sample_index=sample_index,
                base_conditioning=base_conditioning,
                parameter_summary=parameter_summary,
                parameter_feature_rows=parameter_feature_rows,
                intensity_lambdas=intensity_lambdas,
            )
            values.append(float(features[feature]))
        means.append(float(np.mean(values)))
    return means


def calibrated_lambdas(
    response_lambdas: np.ndarray,
    response_values: np.ndarray,
    target_reference: dict[str, float],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    real_lower = float(target_reference.get("p10", target_reference["p05"]))
    real_upper = float(target_reference.get("p90", target_reference["p95"]))
    generator_lower = float(response_values[0])
    generator_upper = float(response_values[-1])
    lower = max(real_lower, generator_lower)
    upper = min(real_upper, generator_upper)
    status = "real_generator_intersection"
    if upper <= lower + 1e-9:
        lower, upper = generator_lower, generator_upper
        status = "no_real_generator_intersection_generator_range_used"
    targets = np.linspace(lower, upper, 5)
    unique_values, unique_indexes = np.unique(response_values, return_index=True)
    unique_lambdas = response_lambdas[unique_indexes]
    if unique_values.size == 1:
        lambdas = np.linspace(0.0, 1.0, 5)
        status = "flat_response_default_lambdas"
    else:
        lambdas = np.interp(targets, unique_values, unique_lambdas)
    return tuple(float(value) for value in lambdas), {
        "status": status,
        "real_interval_p10_p90": [real_lower, real_upper],
        "generator_interval": [generator_lower, generator_upper],
        "selected_target_values": targets.tolist(),
        "selected_lambdas": lambdas.tolist(),
        "lambda_grid": response_lambdas.tolist(),
        "response_curve": response_values.tolist(),
    }


def matched_family_lambdas(
    response_lambdas: np.ndarray,
    response_values: np.ndarray,
    target_values: list[float],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    unique_values, unique_indexes = np.unique(response_values, return_index=True)
    unique_lambdas = response_lambdas[unique_indexes]
    if unique_values.size == 1:
        lambdas = np.linspace(0.0, 1.0, 5)
        status = "flat_secondary_response"
    else:
        lambdas = np.interp(target_values, unique_values, unique_lambdas)
        status = (
            "matched_primary_target_values"
            if min(target_values) >= unique_values[0] - 1e-9
            and max(target_values) <= unique_values[-1] + 1e-9
            else "partially_outside_secondary_response_clipped"
        )
    return tuple(float(value) for value in lambdas), {
        "status": status,
        "target_values_from_primary": list(target_values),
        "selected_lambdas": lambdas.tolist(),
        "lambda_grid": response_lambdas.tolist(),
        "response_curve": response_values.tolist(),
        "secondary_response_interval": [
            float(unique_values[0]),
            float(unique_values[-1]),
        ],
    }


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def parameter_combination_audit(
    capability_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["generator_family_role"] == "primary"
        and int(row["intensity"]) == 5
    ]
    anchors = {
        int(row["parameter_sampling"]["source_window_index"])
        for row in selected
    }
    parameter_names = sorted(
        {
            name
            for row in selected
            for name in row["sampled_generator_parameters"]
        }
    )
    vectors = {
        tuple(
            round(
                float(row["sampled_generator_parameters"].get(name, math.nan)),
                12,
            )
            for name in parameter_names
        )
        for row in selected
    }
    expected_independent_draws = (
        len(selected) // 2
        if capability_id in COUNTERFACTUAL_CAPABILITIES
        else len(selected)
    )
    unique_by_parameter = {
        name: len(
            {
                round(
                    float(row["sampled_generator_parameters"][name]),
                    12,
                )
                for row in selected
                if name in row["sampled_generator_parameters"]
            }
        )
        for name in parameter_names
    }
    fallback_features = sorted(
        {
            feature
            for row in selected
            for feature in row["parameter_sampling"]["fallback_features"]
        }
    )
    return {
        "policy": "empirical_copula_anchor_rank_mapped_to_p25_p75",
        "source_window_count": (
            int(selected[0]["parameter_sampling"]["source_window_count"])
            if selected
            else 0
        ),
        "expected_independent_draw_count": expected_independent_draws,
        "distinct_anchor_count": len(anchors),
        "unique_parameter_vector_count": len(vectors),
        "distinct_anchor_coverage": (
            len(anchors) / max(expected_independent_draws, 1)
        ),
        "unique_parameter_vector_coverage": (
            len(vectors) / max(expected_independent_draws, 1)
        ),
        "unique_value_count_by_parameter": unique_by_parameter,
        "fallback_features": fallback_features,
    }


def nuisance_combination_audit(
    capability_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["generator_family_role"] == "primary"
        and int(row["intensity"]) == 5
        and (
            capability_id not in COUNTERFACTUAL_CAPABILITIES
            or int(row["counterfactual_member"]) == 0
        )
    ]
    fields = NUISANCE_FINGERPRINT_FIELDS[capability_id]
    missing_fields = sorted(
        {
            field
            for row in selected
            for field in fields
            if field not in row["generation_metadata"]
        }
    )
    fingerprints = {
        json.dumps(
            {
                field: row["generation_metadata"].get(field)
                for field in fields
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in selected
    }
    return {
        "fingerprint_fields": list(fields),
        "expected_independent_path_count": len(selected),
        "unique_nuisance_fingerprint_count": len(fingerprints),
        "unique_nuisance_fingerprint_coverage": (
            len(fingerprints) / max(len(selected), 1)
        ),
        "missing_fingerprint_fields": missing_fields,
    }


def sample_row(
    capability_id: str,
    *,
    family_role: str,
    intensity: int,
    sample_index: int,
    base: GeneratorConditioning,
    target: np.ndarray,
    covariates: np.ndarray | None,
    metadata: dict[str, Any],
    features: dict[str, float],
    parameters: dict[str, float],
    mappings: list[dict[str, Any]],
    parameter_sampling: dict[str, Any],
    inference_selected: bool,
) -> dict[str, Any]:
    sample_id = (
        f"v8test__{capability_id}__{family_role}__"
        f"i{intensity}__s{sample_index:03d}"
    )
    covariate_dim = 0 if covariates is None else int(covariates.shape[1])
    counterfactual_pair_id = (
        f"v8test__{capability_id}__{family_role}__i{intensity}__"
        f"pair{sample_index // 2:03d}"
        if capability_id in COUNTERFACTUAL_CAPABILITIES
        else None
    )
    return {
        "schema_version": "paper_v8_deterministic_test_sample.v1",
        "sample_id": sample_id,
        "master_sample_id": sample_id,
        "paired_group_id": (
            counterfactual_pair_id
            if counterfactual_pair_id is not None
            else f"v8test__{capability_id}__{family_role}__s{sample_index:03d}"
        ),
        "counterfactual_pair_id": counterfactual_pair_id,
        "counterfactual_member": (
            int(metadata["counterfactual_variant"])
            if counterfactual_pair_id is not None
            else None
        ),
        "capability_id": capability_id,
        "dataset_id": base.dataset_id,
        "task_id": SELECTED_DATASET[capability_id][1],
        "task_view_id": f"{base.dataset_id}::{SELECTED_DATASET[capability_id][1]}",
        "profile_id": base.profile_id,
        "generator_version": GENERATOR_VERSION,
        "generator_family_role": family_role,
        "generator_family_id": metadata["generator_family_id"],
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "target_dim": int(target.shape[1]),
        "covariate_dim": covariate_dim,
        "covariate_column_names": [
            "weather_driver",
            "known_event",
        ][:covariate_dim],
        "frequency": base.frequency,
        "season_length": base.season_length,
        "hierarchy": (
            "additive_first"
            if capability_id == "hierarchical_coherence"
            else None
        ),
        "intensity": intensity,
        "sample_index": sample_index,
        "target_feature": PRIMARY_TARGET_FEATURE[capability_id],
        "target_feature_value": float(
            features[PRIMARY_TARGET_FEATURE[capability_id]]
        ),
        "realized_features": {
            name: float(value)
            for name, value in features.items()
            if math.isfinite(value)
        },
        "sampled_generator_parameters": parameters,
        "parameter_mapping": mappings,
        "parameter_sampling": parameter_sampling,
        "generation_metadata": metadata,
        "construction_validated": True,
        "evaluation_table": "main",
        "input_history_semantics": "clean_latent",
        "scoring_target_semantics": "clean_latent_future",
        "clean_latent_is_target": True,
        "observation_noise_scale": 0.0,
        "future_process_noise_scale": 0.0,
        "target_sha256": sha256_array(target),
        "future_sha256": sha256_array(target[CONTEXT_LENGTH:]),
        "target": target.tolist(),
        "covariates": None if covariates is None else covariates.tolist(),
        "inference_selected": inference_selected,
    }


def robustness_sample_row(clean_row: dict[str, Any]) -> dict[str, Any]:
    """Build a paired noisy-history sample scored on the clean latent future."""

    clean_target = np.asarray(clean_row["target"], dtype=float)
    digest = hashlib.sha256(clean_row["sample_id"].encode("utf-8")).digest()
    seed_offset = int.from_bytes(digest[:8], "big") % (2**32 - 1)
    observed_target, noise_metadata = add_observation_noise_to_history(
        clean_target,
        context_length=CONTEXT_LENGTH,
        noise_ratio=ROBUSTNESS_NOISE_RATIO,
        rng=np.random.default_rng(ROBUSTNESS_SEED_BASE + seed_offset),
        preserve_additive_hierarchy=(
            clean_row["capability_id"] == "hierarchical_coherence"
        ),
    )
    result = deepcopy(clean_row)
    result["schema_version"] = "paper_v8_observation_robustness_sample.v1"
    result["sample_id"] = clean_row["sample_id"] + "__obsnoise15"
    result["master_sample_id"] = clean_row["sample_id"]
    result["paired_group_id"] = clean_row["sample_id"]
    result["evaluation_table"] = "observation_noise_robustness"
    result["input_history_semantics"] = "noisy_observation"
    result["scoring_target_semantics"] = "clean_latent_future"
    result["clean_latent_is_target"] = False
    result["clean_latent_is_scoring_future"] = True
    result["observation_noise_scale"] = ROBUSTNESS_NOISE_RATIO
    result["observation_noise_metadata"] = noise_metadata
    result["clean_latent_sha256"] = sha256_array(clean_target)
    result["target_sha256"] = sha256_array(observed_target)
    result["future_sha256"] = sha256_array(
        observed_target[CONTEXT_LENGTH:]
    )
    result["target"] = observed_target.tolist()
    result["inference_selected"] = True
    return result


def validate_prefix(
    capability_id: str,
    *,
    family_role: str,
    intensity: int,
    sample_index: int,
    base: GeneratorConditioning,
    parameter_summary: dict[str, dict[str, float]],
    parameter_feature_rows: list[dict[str, float]],
    intensity_lambdas: tuple[float, ...],
) -> float:
    generation_index = (
        sample_index // 2
        if capability_id in COUNTERFACTUAL_CAPABILITIES
        else sample_index
    )
    parameters, _, _ = sample_parameters(
        capability_id,
        parameter_summary,
        feature_rows=parameter_feature_rows,
        season_length=base.season_length,
        sample_index=generation_index,
    )
    conditioning = v8_conditioning(
        base,
        parameters=parameters,
        lambdas=intensity_lambdas,
    )
    seed = PATH_SEED_BASE + 10007 * generation_index
    short, _, _ = generate_deterministic_sample(
        capability_id,
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        base.target_dim,
        base.season_length,
        intensity,
        np.random.default_rng(seed),
        conditioning=conditioning,
        family_role=family_role,
        counterfactual_variant=(
            sample_index % 2
            if capability_id in COUNTERFACTUAL_CAPABILITIES
            else 0
        ),
    )
    long, _, _ = generate_deterministic_sample(
        capability_id,
        CONTEXT_LENGTH + 2 * HORIZON,
        CONTEXT_LENGTH,
        base.target_dim,
        base.season_length,
        intensity,
        np.random.default_rng(seed),
        conditioning=conditioning,
        family_role=family_role,
        counterfactual_variant=(
            sample_index % 2
            if capability_id in COUNTERFACTUAL_CAPABILITIES
            else 0
        ),
    )
    return float(np.max(np.abs(short - long[: len(short)])))


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Paper v8 clean-deterministic pilot",
        "",
        f"- Generator: `{summary['generator_version']}`",
        f"- Protocol: L={CONTEXT_LENGTH}, H={HORIZON}; main table has no process or observation noise.",
        "- Scope: one available calibration dataset per capability; v8 test only.",
        "",
        "## Capability audit",
        "",
        "| capability | dataset | extraction | mapping | joint params | nuisance paths | primary / secondary | dose | future std/history std | clean gate |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for capability_id, row in summary["capabilities"].items():
        lines.append(
            "| {cap} | {dataset} | {finite}/{required} | {mapped}/{required} | "
            "{combos}/{expected} | {paths}/{expected_paths} | "
            "`{primary}` / `{secondary}` | "
            "{dose} | {ratio:.3f} | {gate} |".format(
                cap=capability_id,
                dataset=row["dataset_id"],
                finite=row["required_features_fully_finite"],
                required=row["required_feature_count"],
                mapped=row["required_features_mapped"],
                combos=row["parameter_combination_audit"][
                    "unique_parameter_vector_count"
                ],
                expected=row["parameter_combination_audit"][
                    "expected_independent_draw_count"
                ],
                paths=row["nuisance_combination_audit"][
                    "unique_nuisance_fingerprint_count"
                ],
                expected_paths=row["nuisance_combination_audit"][
                    "expected_independent_path_count"
                ],
                primary=row["primary_family"],
                secondary=row["secondary_family"],
                dose="pass" if row["primary_dose_monotone"] else "FAIL",
                ratio=row["median_future_to_history_std_ratio"],
                gate=(
                    "not calibrated"
                    if not row["clean_gate_enforced"]
                    else (
                    f"{row['clean_gate_acceptance_rate']:.1%}*"
                    if row["clean_gate_projection_requires_recalibration"]
                    else f"{row['clean_gate_acceptance_rate']:.1%}"
                    )
                ),
            )
        )
    lines.extend(
        [
            "",
            "`*` clean-gate scores are projected from the v7 support artifact. "
            "The stochastic tail controls were removed, but the conformal threshold "
            "must be rebuilt before a formal v8 run.",
            "",
            "`joint params` counts unique I5 parameter vectors over independent "
            "draws. Each draw uses one real-window empirical-copula anchor and maps "
            "its joint feature ranks into the central p25-p75 support. `nuisance "
            "paths` independently fingerprints the mechanism-specific motif, phase, "
            "lag, loading or counterfactual branch selected before path generation.",
            "",
            "## Gate audit",
            "",
            "The clean main table excludes these stochastic/tail controls: `"
            + "`, `".join(sorted(CLEAN_DETERMINISTIC_EXCLUDED_CONTROLS))
            + "`. Structural controls such as hierarchy residual and residual "
            "autocorrelation remain active.",
            "",
            "## Secondary-family sensitivity",
            "",
            "Secondary families are inverse-matched to the primary family's target "
            "feature levels before I3/I5 comparison; the remaining difference is "
            "therefore family shape rather than an intentional dose change.",
            "",
            "| capability | calibration | I3 relative feature difference | I5 relative feature difference |",
            "|---|---|---:|---:|",
        ]
    )
    for capability_id, row in summary["capabilities"].items():
        lines.append(
            f"| {capability_id} | {row['secondary_intensity_calibration_status']} | "
            f"{row['family_sensitivity_at_i3_i5']['3']['relative_absolute_difference']:.1%} | "
            f"{row['family_sensitivity_at_i3_i5']['5']['relative_absolute_difference']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Inference",
            "",
            "`inference_samples.jsonl` contains the preregistered small model-response "
            "subset (primary I1/I3/I5 and secondary I3/I5). Run the companion "
            "inference/analyzer before treating model behavior as validated.",
            "",
            "`robustness_samples.jsonl` is paired to primary I3/I5 samples. It adds "
            f"fixed Gaussian observation noise at {ROBUSTNESS_NOISE_RATIO:.0%} of each "
            "history channel's standard deviation, changes history only, and retains "
            "the exact clean latent future as the scoring target. Additive hierarchy "
            "noise is sampled on children and the parent is recomputed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if min(
        args.primary_seeds,
        args.secondary_seeds,
        args.calibration_seeds,
        args.inference_seeds,
    ) <= 0:
        raise ValueError("all seed counts must be positive")
    if any(
        count % 2
        for count in (
            args.primary_seeds,
            args.secondary_seeds,
            args.calibration_seeds,
            args.inference_seeds,
        )
    ):
        raise ValueError(
            "seed counts must be even so cross-series and covariate "
            "counterfactual pairs are complete"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    real_rows = read_jsonl(V7_REAL_SAMPLES)
    conditioning_artifact = read_json(
        V7_SUITE_DIR / "generator_conditioning_artifact.json"
    )
    feature_gate_artifact = read_json(V7_SUITE_DIR / "feature_gate_artifact.json")

    samples: list[dict[str, Any]] = []
    extraction_artifact: dict[str, Any] = {
        "schema_version": "paper_v8_real_feature_extraction_test.v1",
        "capabilities": {},
    }
    mapping_artifact: dict[str, Any] = {
        "schema_version": "paper_v8_parameter_mapping_test.v1",
        "capabilities": {},
    }
    summary: dict[str, Any] = {
        "schema_version": "paper_v8_deterministic_pilot_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "robustness_table_generated": True,
        "robustness_observation_noise_ratio": ROBUSTNESS_NOISE_RATIO,
        "capabilities": {},
    }

    for capability_id in PRIMARY_FAMILY_BY_CAPABILITY:
        rows = selected_rows(real_rows, capability_id)
        parameter_rows = real_feature_rows(
            rows,
            history_only=True,
            capability_id=capability_id,
        )
        target_rows = real_feature_rows(
            rows,
            history_only=False,
            capability_id=capability_id,
        )
        parameter_summary = summarize(parameter_rows)
        target_summary = summarize(target_rows)
        required = REQUIRED_REAL_FEATURES_BY_CAPABILITY[capability_id]
        finite_counts = {
            name: sum(
                name in row and math.isfinite(row[name])
                for row in parameter_rows
            )
            for name in required
        }
        extraction_artifact["capabilities"][capability_id] = {
            "dataset_id": rows[0]["dataset_id"],
            "task_view_id": rows[0]["task_view_id"],
            "profile_id": rows[0]["profile_id"],
            "window_count": len(rows),
            "history_only_parameter_extraction": True,
            "required_features": list(required),
            "finite_counts": finite_counts,
            "parameter_feature_summary": parameter_summary,
            "full_window_target_reference": target_summary.get(
                PRIMARY_TARGET_FEATURE[capability_id],
                {},
            ),
        }

        source_capability_id = CONDITIONING_SOURCE_CAPABILITY.get(
            capability_id,
            capability_id,
        )
        base = resolve_generator_conditioning(
            capability_id=source_capability_id,
            profile_id=rows[0]["profile_id"],
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            target_dim=int(rows[0]["target_dim"]),
            artifact=conditioning_artifact,
        )
        if base is None:
            raise ValueError(f"missing conditioning profile for {capability_id}")
        if source_capability_id != capability_id:
            base = dataclasses.replace(
                base,
                capability_id=capability_id,
                target_feature=PRIMARY_TARGET_FEATURE[capability_id],
            )
        base_parameters, base_mappings = derive_deterministic_parameters(
            capability_id,
            parameter_summary,
            season_length=base.season_length,
            context_length=CONTEXT_LENGTH,
        )
        mapped_sources = {str(row["source_feature"]) for row in base_mappings}
        mapped_required = [
            name
            for name in required
            if name in mapped_sources
            or any(name in source.split("/") for source in mapped_sources)
        ]

        response_lambdas, response_values = response_curve(
            capability_id,
            family_role="primary",
            base_conditioning=base,
            parameter_summary=parameter_summary,
            parameter_feature_rows=parameter_rows,
            seed_count=args.calibration_seeds,
        )
        target_reference = target_summary[PRIMARY_TARGET_FEATURE[capability_id]]
        intensity_lambdas, calibration = calibrated_lambdas(
            response_lambdas,
            response_values,
            target_reference,
        )
        paired_primary_target_values = response_at_selected_lambdas(
            capability_id,
            family_role="primary",
            base_conditioning=base,
            parameter_summary=parameter_summary,
            parameter_feature_rows=parameter_rows,
            intensity_lambdas=intensity_lambdas,
            seed_count=args.secondary_seeds,
        )
        secondary_response_lambdas, secondary_response_values = response_curve(
            capability_id,
            family_role="secondary",
            base_conditioning=base,
            parameter_summary=parameter_summary,
            parameter_feature_rows=parameter_rows,
            seed_count=args.secondary_seeds,
        )
        secondary_intensity_lambdas, secondary_calibration = matched_family_lambdas(
            secondary_response_lambdas,
            secondary_response_values,
            paired_primary_target_values,
        )
        mapping_artifact["capabilities"][capability_id] = {
            "dataset_id": base.dataset_id,
            "profile_id": base.profile_id,
            "base_parameters": base_parameters,
            "base_parameter_mapping": base_mappings,
            "sample_parameter_policy": (
                "one deterministic empirical-copula anchor window per seed; "
                "each feature keeps that window's marginal rank mapped into "
                "its p25-p75 interval; paired across intensity; "
                "cross-series and covariate-response counterfactual members "
                "additionally share every parameter and path draw except the "
                "designated observed driver block or known-future covariate branch"
            ),
            "legacy_v7_structure_parameters_reused": False,
            "intensity_calibration": calibration,
            "secondary_family_intensity_calibration": secondary_calibration,
            "paired_seed_primary_target_values": paired_primary_target_values,
        }

        capability_samples: list[dict[str, Any]] = []
        for family_role, intensities, seed_count in (
            ("primary", INTENSITIES, args.primary_seeds),
            ("secondary", (3, 5), args.secondary_seeds),
        ):
            for intensity in intensities:
                for sample_index in range(seed_count):
                    (
                        target,
                        covariates,
                        metadata,
                        features,
                        parameters,
                        mappings,
                        parameter_sampling,
                    ) = generate_one(
                        capability_id,
                        family_role=family_role,
                        intensity=intensity,
                        sample_index=sample_index,
                        base_conditioning=base,
                        parameter_summary=parameter_summary,
                        parameter_feature_rows=parameter_rows,
                        intensity_lambdas=(
                            intensity_lambdas
                            if family_role == "primary"
                            else secondary_intensity_lambdas
                        ),
                    )
                    inference_selected = (
                        sample_index < args.inference_seeds
                        and (
                            (family_role == "primary" and intensity in {1, 3, 5})
                            or family_role == "secondary"
                        )
                    )
                    row = sample_row(
                        capability_id,
                        family_role=family_role,
                        intensity=intensity,
                        sample_index=sample_index,
                        base=base,
                        target=target,
                        covariates=covariates,
                        metadata=metadata,
                        features=features,
                        parameters=parameters,
                        mappings=mappings,
                        parameter_sampling=parameter_sampling,
                        inference_selected=inference_selected,
                    )
                    capability_samples.append(row)
                    samples.append(row)

        if capability_id == "cross_series_dependence":
            attach_cross_series_identifiability_gates(capability_samples)

        primary_means = []
        for intensity in INTENSITIES:
            values = [
                float(row["target_feature_value"])
                for row in capability_samples
                if row["generator_family_role"] == "primary"
                and int(row["intensity"]) == intensity
            ]
            primary_means.append(float(np.mean(values)))
        secondary_means = {
            str(intensity): float(
                np.mean(
                    [
                        float(row["target_feature_value"])
                        for row in capability_samples
                        if row["generator_family_role"] == "secondary"
                        and int(row["intensity"]) == intensity
                    ]
                )
            )
            for intensity in (3, 5)
        }
        primary_sensitivity_means = {
            str(intensity): float(
                np.mean(
                    [
                        float(row["target_feature_value"])
                        for row in capability_samples
                        if row["generator_family_role"] == "primary"
                        and int(row["intensity"]) == intensity
                        and int(row["sample_index"]) < args.secondary_seeds
                    ]
                )
            )
            for intensity in (3, 5)
        }
        clean_gates = [
            evaluate_feature_support_gate(
                capability_id=capability_id,
                features=row["realized_features"],
                profile_ids=(base.profile_id,),
                context_length=CONTEXT_LENGTH,
                horizon=HORIZON,
                target_dim=base.target_dim,
                artifact=feature_gate_artifact,
                evaluation_mode="clean_deterministic",
            )
            for row in capability_samples
            if row["generator_family_role"] == "primary"
        ]
        standard_gates = [
            evaluate_feature_support_gate(
                capability_id=capability_id,
                features=row["realized_features"],
                profile_ids=(base.profile_id,),
                context_length=CONTEXT_LENGTH,
                horizon=HORIZON,
                target_dim=base.target_dim,
                artifact=feature_gate_artifact,
                evaluation_mode="standard",
            )
            for row in capability_samples
            if row["generator_family_role"] == "primary"
        ]
        future_ratios = []
        for row in capability_samples:
            target = np.asarray(row["target"], dtype=float)
            history_std = float(np.mean(np.std(target[:CONTEXT_LENGTH], axis=0)))
            future_std = float(np.mean(np.std(target[CONTEXT_LENGTH:], axis=0)))
            future_ratios.append(future_std / max(history_std, 1e-12))
        prefix_errors = [
            validate_prefix(
                capability_id,
                family_role=family_role,
                intensity=intensity,
                sample_index=0,
                base=base,
                parameter_summary=parameter_summary,
                parameter_feature_rows=parameter_rows,
                intensity_lambdas=(
                    intensity_lambdas
                    if family_role == "primary"
                    else secondary_intensity_lambdas
                ),
            )
            for family_role, intensity in (("primary", 3), ("secondary", 5))
        ]
        hierarchy_residual = max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(row["target"])[:, 0]
                            - np.sum(np.asarray(row["target"])[:, 1:], axis=1)
                        )
                    )
                )
                for row in capability_samples
            ),
            default=0.0,
        ) if capability_id == "hierarchical_coherence" else 0.0
        parameter_audit = parameter_combination_audit(
            capability_id,
            capability_samples,
        )
        nuisance_audit = nuisance_combination_audit(
            capability_id,
            capability_samples,
        )
        mapping_artifact["capabilities"][capability_id][
            "parameter_combination_audit"
        ] = parameter_audit
        mapping_artifact["capabilities"][capability_id][
            "nuisance_combination_audit"
        ] = nuisance_audit
        summary["capabilities"][capability_id] = {
            "dataset_id": base.dataset_id,
            "task_view_id": rows[0]["task_view_id"],
            "profile_id": base.profile_id,
            "real_window_count": len(rows),
            "required_feature_count": len(required),
            "required_features_fully_finite": sum(
                count == len(rows) for count in finite_counts.values()
            ),
            "required_features_mapped": len(mapped_required),
            "unmapped_required_features": sorted(set(required) - set(mapped_required)),
            "primary_family": PRIMARY_FAMILY_BY_CAPABILITY[capability_id],
            "secondary_family": SECONDARY_FAMILY_BY_CAPABILITY[capability_id],
            "primary_target_feature": PRIMARY_TARGET_FEATURE[capability_id],
            "primary_target_feature_means": primary_means,
            "secondary_target_feature_means": secondary_means,
            "primary_target_feature_means_on_sensitivity_seeds": (
                primary_sensitivity_means
            ),
            "family_sensitivity_at_i3_i5": {
                str(intensity): {
                    "absolute_difference": float(
                        secondary_means[str(intensity)]
                        - primary_sensitivity_means[str(intensity)]
                    ),
                    "relative_absolute_difference": float(
                        abs(
                            secondary_means[str(intensity)]
                            - primary_sensitivity_means[str(intensity)]
                        )
                        / max(
                            abs(primary_sensitivity_means[str(intensity)]),
                            1e-12,
                        )
                    ),
                }
                for intensity in (3, 5)
            },
            "primary_dose_monotone": all(
                right >= left - 1e-9
                for left, right in zip(primary_means, primary_means[1:])
            ),
            "selected_intensity_lambdas": list(intensity_lambdas),
            "intensity_calibration_status": calibration["status"],
            "secondary_selected_intensity_lambdas": list(
                secondary_intensity_lambdas
            ),
            "secondary_intensity_calibration_status": secondary_calibration[
                "status"
            ],
            "primary_sample_count": args.primary_seeds * 5,
            "secondary_sensitivity_sample_count": args.secondary_seeds * 2,
            "median_future_to_history_std_ratio": float(np.median(future_ratios)),
            "minimum_future_to_history_std_ratio": float(np.min(future_ratios)),
            "max_prefix_invariance_error": max(prefix_errors),
            "max_hierarchy_residual": hierarchy_residual,
            "parameter_combination_audit": parameter_audit,
            "nuisance_combination_audit": nuisance_audit,
            "standard_gate_acceptance_rate": float(
                np.mean([row["accepted"] for row in standard_gates])
            ),
            "clean_gate_acceptance_rate": float(
                np.mean([row["accepted"] for row in clean_gates])
            ),
            "standard_gate_enforced": all(
                row["enforced"] for row in standard_gates
            ),
            "clean_gate_enforced": all(
                row["enforced"] for row in clean_gates
            ),
            "clean_gate_projection_requires_recalibration": any(
                any(
                    bucket.get("threshold_calibration")
                    == "projected_from_standard_support_requires_v8_recalibration"
                    for bucket in row.get("bucket_results", [])
                )
                for row in clean_gates
            ),
        }
        if capability_id == "cross_series_dependence":
            identifiability_gates = [
                row["generation_metadata"]["identifiability_gate"]
                for row in capability_samples
            ]
            enforced_identifiability_gates = [
                gate for gate in identifiability_gates if gate["enforced"]
            ]
            identifiability_summary = {
                "schema_version": "cross_series_identifiability_summary.v1",
                "pair_count": len(identifiability_gates) // 2,
                "enforced_pair_count": len(enforced_identifiability_gates) // 2,
                "enforced_acceptance_rate": float(
                    np.mean(
                        [
                            gate["accepted"]
                            for gate in enforced_identifiability_gates
                        ]
                    )
                ),
                "minimum_enforced_history_holdout_r2": float(
                    np.min(
                        [
                            gate["minimum_declared_holdout_r2"]
                            for gate in enforced_identifiability_gates
                        ]
                    )
                ),
                "maximum_enforced_positive_control_effect_nrmse": float(
                    np.max(
                        [
                            gate["positive_control_effect_nrmse"]
                            for gate in enforced_identifiability_gates
                        ]
                    )
                ),
                "minimum_enforced_positive_control_effect_correlation": float(
                    np.min(
                        [
                            gate["positive_control_effect_correlation"]
                            for gate in enforced_identifiability_gates
                        ]
                    )
                ),
            }
            summary["capabilities"][capability_id][
                "identifiability_gate"
            ] = identifiability_summary
            mapping_artifact["capabilities"][capability_id][
                "identifiability_gate"
            ] = identifiability_summary

    inference_samples = [row for row in samples if row["inference_selected"]]
    robustness_samples = [
        robustness_sample_row(row)
        for row in samples
        if row["generator_family_role"] == "primary"
        and int(row["intensity"]) in {3, 5}
        and int(row["sample_index"]) < args.inference_seeds
    ]
    summary["robustness_sample_count"] = len(robustness_samples)
    summary["robustness_by_capability"] = {
        capability_id: {
            "sample_count": sum(
                row["capability_id"] == capability_id
                for row in robustness_samples
            ),
            "mean_realized_noise_to_history_std_ratio": float(
                np.mean(
                    [
                        row["observation_noise_metadata"][
                            "realized_noise_to_history_std_ratio"
                        ]
                        for row in robustness_samples
                        if row["capability_id"] == capability_id
                    ]
                )
            ),
            "max_future_noise_abs": max(
                row["observation_noise_metadata"]["future_noise_max_abs"]
                for row in robustness_samples
                if row["capability_id"] == capability_id
            ),
        }
        for capability_id in PRIMARY_FAMILY_BY_CAPABILITY
    }
    write_jsonl(output_dir / "samples.jsonl", samples)
    write_jsonl(output_dir / "inference_samples.jsonl", inference_samples)
    write_jsonl(output_dir / "robustness_samples.jsonl", robustness_samples)
    write_json(output_dir / "real_feature_extraction.json", extraction_artifact)
    write_json(output_dir / "parameter_mapping.json", mapping_artifact)
    write_json(output_dir / "generation_summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        report_markdown(summary),
        encoding="utf-8",
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "paper_v8_deterministic_pilot_manifest.v1",
            "created_at": summary["created_at"],
            "generator_version": GENERATOR_VERSION,
            "files": [
                "samples.jsonl",
                "inference_samples.jsonl",
                "robustness_samples.jsonl",
                "real_feature_extraction.json",
                "parameter_mapping.json",
                "generation_summary.json",
                "REPORT.md",
            ],
            "sample_count": len(samples),
            "inference_sample_count": len(inference_samples),
            "robustness_sample_count": len(robustness_samples),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_count": len(samples),
                "inference_sample_count": len(inference_samples),
                "robustness_sample_count": len(robustness_samples),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
