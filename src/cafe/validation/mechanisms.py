from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from cafe.generation.families import (
    common_factor_identifiability_gate,
    cross_series_identifiability_gate,
)


SCHEMA_VERSION = "cafe.feature_gate.v1"
MINIMUM_NONLINEAR_ACTIVITY_PAIRED_POSITIVE_FRACTION = 0.50
COUNTERFACTUAL_CAPABILITIES = frozenset(
    {
        "common_factor",
        "cross_series_dependence",
        "covariate_response",
    }
)
STRUCTURAL_CAPABILITIES = frozenset(
    {
        "common_factor",
        "hierarchical_coherence",
        "cross_series_dependence",
        "covariate_response",
    }
)
MINIMUM_CALIBRATION_REACHABILITY_FRACTION = {
    "common_factor": 1.00,
    "hierarchical_coherence": 1.00,
    "cross_series_dependence": 1.00,
    "covariate_response": 1.00,
}
PRIMARY_FEATURE_BY_CAPABILITY = {
    "trend": "local_polynomial_energy_share_w96",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "regime_sparse_transition_score",
    "nonlinear_persistence": "nonlinear_conditional_effect_size",
    "predictable_intermittency": (
        "event_positive_residual_energy_share"
    ),
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    "cross_series_dependence": "cross_series_incremental_r2",
    "covariate_response": "covariate_incremental_r2",
}
SELECTIVITY_EXCEPTIONS = (
    {
        "intervention_capability": "multi_seasonal",
        "feature": "seasonal_amplitude_modulation",
        "reason": (
            "stationary symmetric spectral sidebands are algebraically "
            "equivalent to sinusoidal amplitude modulation"
        ),
    },
    {
        "intervention_capability": "time_varying_seasonality",
        "feature": "multi_period_score",
        "reason": (
            "sinusoidal amplitude modulation necessarily creates stationary "
            "Fourier sidebands over a finite observation window"
        ),
    },
)


def sample_content_sha256(sample: dict[str, Any]) -> str:
    target = np.asarray(sample["target"], dtype="<f8")
    digest = hashlib.sha256(target.tobytes())
    covariates = sample.get("covariates")
    if covariates is not None:
        digest.update(np.asarray(covariates, dtype="<f8").tobytes())
    return digest.hexdigest()


def basic_sample_checks(sample: dict[str, Any]) -> dict[str, Any]:
    target = np.asarray(sample.get("target"), dtype=float)
    context = int(sample.get("context_length", -1))
    horizon = int(sample.get("horizon", -1))
    target_dim = int(sample.get("target_dim", -1))
    expected_shape = (context + horizon, target_dim)
    covariates = sample.get("covariates")
    covariate_dim = int(sample.get("covariate_dim", 0))
    covariate_array = (
        None if covariates is None else np.asarray(covariates, dtype=float)
    )
    intensity_calibration = sample.get("intensity_calibration")
    covariate_shape_valid = (
        covariate_array is None
        if covariate_dim == 0
        else (
            covariate_array is not None
            and covariate_array.shape == (context + horizon, covariate_dim)
        )
    )
    checks = {
        "target_shape_valid": target.shape == expected_shape,
        "target_finite": bool(np.isfinite(target).all()),
        "covariate_shape_valid": bool(covariate_shape_valid),
        "covariate_finite": bool(
            covariate_array is None or np.isfinite(covariate_array).all()
        ),
        "mase_scale_valid": bool(
            math.isfinite(float(sample.get("mase_scale", math.nan)))
            and float(sample.get("mase_scale", 0.0)) > 0.0
        ),
        "mase_period_valid": bool(
            1
            <= int(
                sample.get(
                    "mase_period",
                    sample.get("season_length", 0),
                )
            )
            < context
        ),
        "target_feature_value_finite": bool(
            math.isfinite(
                float(sample.get("target_feature_value", math.nan))
            )
        ),
        "intensity_lambda_finite": bool(
            math.isfinite(float(sample.get("intensity_lambda", math.nan)))
        ),
        "intensity_grid_real_calibrated": bool(
            isinstance(intensity_calibration, dict)
            and intensity_calibration.get("scope")
            == "dataset_real_generator_overlap_reference"
        ),
        "target_hash_matches": (
            str(sample.get("target_sha256")) == sample_content_sha256(sample)
            if sample.get("target_sha256")
            else True
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted": all(checks.values()),
        "checks": checks,
        "expected_target_shape": list(expected_shape),
    }


def hierarchy_checks(sample: dict[str, Any], *, atol: float = 1e-10) -> dict[str, Any]:
    target = np.asarray(sample["target"], dtype=float)
    residual = target[:, 0] - np.sum(target[:, 1:], axis=1)
    maximum = float(np.max(np.abs(residual)))
    return {
        "maximum_additive_residual": maximum,
        "accepted": maximum <= atol,
    }


def covariate_pair_checks(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    atol: float = 1e-10,
) -> dict[str, Any]:
    context = int(first["context_length"])
    first_target = np.asarray(first["target"], dtype=float)
    second_target = np.asarray(second["target"], dtype=float)
    first_covariates = np.asarray(first["covariates"], dtype=float)
    second_covariates = np.asarray(second["covariates"], dtype=float)
    values = {
        "target_history_max_abs_difference": float(
            np.max(np.abs(first_target[:context] - second_target[:context]))
        ),
        "past_covariate_max_abs_difference": float(
            np.max(
                np.abs(
                    first_covariates[:context] - second_covariates[:context]
                )
            )
        ),
        "future_covariate_max_abs_difference": float(
            np.max(
                np.abs(
                    first_covariates[context:] - second_covariates[context:]
                )
            )
        ),
        "target_future_max_abs_difference": float(
            np.max(
                np.abs(first_target[context:] - second_target[context:])
            )
        ),
    }
    values["accepted"] = bool(
        values["target_history_max_abs_difference"] <= atol
        and values["past_covariate_max_abs_difference"] <= atol
        and values["future_covariate_max_abs_difference"] > atol
        and values["target_future_max_abs_difference"] > atol
    )
    return values


def structural_calibration_member_gate(
    capability_id: str,
    first_target: np.ndarray,
    *,
    context_length: int,
    metadata: Mapping[str, Any] | None = None,
    second_target: np.ndarray | None = None,
    first_covariates: np.ndarray | None = None,
    second_covariates: np.ndarray | None = None,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Evaluate one calibration path using only structural construction gates.

    This is deliberately separate from dataset-local feature support and
    near-distance checks.  Calibration asks whether a selected generator dose
    can satisfy the same structural hard gate later enforced during formal
    generation; similarity to a real trajectory is irrelevant to that
    reachability question.

    Counterfactual capabilities require both members.  Hierarchy requires one
    parent-first additive target.  Malformed or incomplete inputs fail closed
    and are returned as records so a qualification bank remains auditable.
    """

    result_prefix = {
        "schema_version": "structural_calibration_member_gate.v1",
        "capability_id": str(capability_id),
        "enforced": True,
        "calibration_reachability": True,
        "gate_scope": "generator_structural_hard_gate_only",
        "near_distance_evaluated": False,
    }
    if capability_id not in STRUCTURAL_CAPABILITIES:
        return {
            **result_prefix,
            "accepted": False,
            "reason": "unsupported_structural_capability",
        }
    try:
        first = np.asarray(first_target, dtype=float)
        if first.ndim != 2 or not np.isfinite(first).all():
            raise ValueError("first target must be a finite matrix")
        if not 1 <= int(context_length) < first.shape[0]:
            raise ValueError("context_length must split history and future")

        if capability_id == "hierarchical_coherence":
            if first.shape[1] < 3:
                raise ValueError(
                    "hierarchy requires parent and at least two children"
                )
            gate = hierarchy_checks({"target": first}, atol=atol)
        else:
            if second_target is None:
                raise ValueError("counterfactual second target is missing")
            second = np.asarray(second_target, dtype=float)
            if (
                second.shape != first.shape
                or not np.isfinite(second).all()
            ):
                raise ValueError(
                    "counterfactual targets must be finite equal-shaped matrices"
                )
            if capability_id == "common_factor":
                gate = common_factor_identifiability_gate(
                    first,
                    second,
                    context_length=int(context_length),
                    metadata=dict(metadata or {}),
                    enforced=True,
                )
            elif capability_id == "cross_series_dependence":
                gate = cross_series_identifiability_gate(
                    first,
                    second,
                    context_length=int(context_length),
                    metadata=dict(metadata or {}),
                    enforced=True,
                )
            else:
                if first_covariates is None or second_covariates is None:
                    raise ValueError(
                        "covariate counterfactual arrays are missing"
                    )
                first_covariate_array = np.asarray(
                    first_covariates,
                    dtype=float,
                )
                second_covariate_array = np.asarray(
                    second_covariates,
                    dtype=float,
                )
                expected_covariate_rows = first.shape[0]
                if (
                    first_covariate_array.ndim != 2
                    or second_covariate_array.shape
                    != first_covariate_array.shape
                    or first_covariate_array.shape[0]
                    != expected_covariate_rows
                    or not np.isfinite(first_covariate_array).all()
                    or not np.isfinite(second_covariate_array).all()
                ):
                    raise ValueError(
                        "covariates must be finite equal-shaped matrices "
                        "aligned with the targets"
                    )
                gate = covariate_pair_checks(
                    {
                        "context_length": int(context_length),
                        "target": first,
                        "covariates": first_covariate_array,
                    },
                    {
                        "context_length": int(context_length),
                        "target": second,
                        "covariates": second_covariate_array,
                    },
                    atol=atol,
                )
                gate["enforced"] = True
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return {
            **result_prefix,
            "accepted": False,
            "reason": "malformed_structural_calibration_member",
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        **gate,
        **result_prefix,
        "underlying_gate_schema_version": gate.get("schema_version"),
    }


def summarize_structural_calibration_reachability(
    capability_id: str,
    *,
    family_role: str,
    lambda_value: float,
    path_gates: Iterable[Mapping[str, Any]],
    expected_path_count: int,
    minimum_pass_fraction: float | None = None,
) -> dict[str, Any]:
    """Fail closed unless the selected I5 dose is structurally reachable.

    Every structural qualification path must pass.  Formal generation fails
    closed for every requested seed, so accepting a partially reachable
    qualification bank would allow a cell to pass calibration and then
    exhaust its fixed candidate budget during generation.  The caller is
    responsible for evaluating the exact selected I5 lambda, rather than a
    nearby raw-grid value.
    """

    if capability_id not in STRUCTURAL_CAPABILITIES:
        raise ValueError(
            f"unsupported structural capability: {capability_id}"
        )
    if family_role not in {"primary", "secondary"}:
        raise ValueError("family_role must be primary or secondary")
    if (
        expected_path_count < 1
        or not math.isfinite(float(lambda_value))
        or not 0.0 <= float(lambda_value) <= 1.0
    ):
        raise ValueError(
            "expected path count and selected lambda must be valid"
        )
    required_fraction = float(
        MINIMUM_CALIBRATION_REACHABILITY_FRACTION[capability_id]
        if minimum_pass_fraction is None
        else minimum_pass_fraction
    )
    if (
        not math.isfinite(required_fraction)
        or not 0.0 < required_fraction <= 1.0
    ):
        raise ValueError("minimum pass fraction must be in (0, 1]")

    rows = [dict(row) for row in path_gates]
    invalid_result_count = sum(
        row.get("capability_id") != capability_id
        or row.get("calibration_reachability") is not True
        or row.get("near_distance_evaluated") is not False
        or not isinstance(row.get("accepted"), bool)
        for row in rows
    )
    accepted_path_count = sum(
        row.get("accepted") is True for row in rows
    )
    missing_path_count = max(0, expected_path_count - len(rows))
    unexpected_path_count = max(0, len(rows) - expected_path_count)
    required_pass_count = int(
        math.ceil(required_fraction * expected_path_count - 1e-12)
    )
    reason_codes: list[str] = []
    if missing_path_count:
        reason_codes.append("structural_gate_qualification_paths_missing")
    if unexpected_path_count:
        reason_codes.append("structural_gate_qualification_paths_unexpected")
    if invalid_result_count:
        reason_codes.append("structural_gate_qualification_results_invalid")
    if accepted_path_count < required_pass_count:
        reason_codes.append("selected_i5_structural_gate_unreachable")
    accepted = not reason_codes
    return {
        "schema_version": "structural_calibration_reachability.v1",
        "capability_id": capability_id,
        "family_role": family_role,
        "selected_intensity": 5,
        "selected_lambda": float(lambda_value),
        "qualification_scope": "selected_i5_exact_lambda",
        "gate_scope": "generator_structural_hard_gate_only",
        "near_distance_evaluated": False,
        "expected_path_count": int(expected_path_count),
        "observed_path_count": len(rows),
        "missing_path_count": missing_path_count,
        "unexpected_path_count": unexpected_path_count,
        "invalid_result_count": invalid_result_count,
        "accepted_path_count": accepted_path_count,
        "required_pass_count": required_pass_count,
        "pass_fraction": (
            float(accepted_path_count / expected_path_count)
        ),
        "minimum_pass_fraction": required_fraction,
        "accepted": accepted,
        "reason_codes": reason_codes,
        "path_gates": rows,
    }


def covariate_family_match_checks(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    dose_atol: float = 1e-8,
    mase_scale_relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Check that covariate families share the calibrated target and nuisance.

    Primary and secondary use separate inverse response curves, so their
    lambda, internal coefficient, and per-seed realized proxy need not be
    identical.  The paired contract is the same real-derived reference target,
    covariate path, and baseline motif.
    """

    primary_covariates = np.asarray(primary["covariates"], dtype=float)
    secondary_covariates = np.asarray(secondary["covariates"], dtype=float)
    primary_metadata = primary["generation_metadata"]
    secondary_metadata = secondary["generation_metadata"]
    primary_scale = float(primary["mase_scale"])
    primary_reference = float(
        primary.get("intensity_target_feature_value", math.nan)
    )
    secondary_reference = float(
        secondary.get("intensity_target_feature_value", math.nan)
    )
    scale_relative_difference = abs(
        float(secondary["mase_scale"]) - primary_scale
    ) / max(abs(primary_scale), 1e-12)
    values = {
        "primary_sample_id": str(primary["sample_id"]),
        "secondary_sample_id": str(secondary["sample_id"]),
        "covariate_max_abs_difference": float(
            np.max(np.abs(primary_covariates - secondary_covariates))
        ),
        "reference_target_absolute_difference": abs(
            primary_reference - secondary_reference
        ),
        "realized_feature_absolute_difference": abs(
            float(primary["target_feature_value"])
            - float(secondary["target_feature_value"])
        ),
        "effect_strength_absolute_difference": abs(
            float(primary_metadata["effect_strength"])
            - float(secondary_metadata["effect_strength"])
        ),
        "baseline_motif_matches": bool(
            primary_metadata["baseline_process"]["motif_sha256"]
            == secondary_metadata["baseline_process"]["motif_sha256"]
        ),
        "mase_scale_relative_difference": scale_relative_difference,
        "mase_scale_relative_tolerance": (
            mase_scale_relative_tolerance
        ),
        "family_specific_inverse_allowed": True,
    }
    values["accepted"] = bool(
        values["covariate_max_abs_difference"] <= dose_atol
        and math.isfinite(primary_reference)
        and math.isfinite(secondary_reference)
        and values["reference_target_absolute_difference"] <= dose_atol
        and values["baseline_motif_matches"]
        and scale_relative_difference <= mase_scale_relative_tolerance
    )
    return values


def nonlinear_mechanism_response_checks(
    samples: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate injected nonlinear dose without misusing exact-lag R².

    The ordinary dose gate and this construction gate both follow the
    generator-recorded nonlinear coefficient, which must increase in aggregate
    and within every paired seed. ``nonlinear_conditional_effect_size`` is the
    history-only inverse coordinate shared with real windows; the underlying
    ``nonlinear_conditional_gain`` remains an observable diagnostic.  The
    adjusted-R² gain at the exact causal lag is likewise diagnostic: recursive
    feedback can spread dependence over correlated lags and make either
    conditional proxy fall while the injected coefficient grows.
    """

    strength_groups: dict[
        tuple[str, str], dict[int, list[tuple[int, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    actual_lag_groups: dict[
        tuple[str, str], dict[int, list[tuple[int, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    observable_groups: dict[
        tuple[str, str], dict[int, list[tuple[int, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    activity_groups: dict[
        tuple[str, str], dict[int, list[tuple[int, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    clip_fractions: dict[tuple[str, str], list[float]] = defaultdict(list)
    expected_counts: dict[
        tuple[str, str], Counter[int]
    ] = defaultdict(Counter)
    expected_groups: set[tuple[str, str]] = set()
    for row in samples:
        if (
            row.get("capability_id") != "nonlinear_persistence"
            or row.get("evaluation_table", "main") != "main"
        ):
            continue
        key = (
            str(row["dataset_id"]),
            str(row["generator_family_role"]),
        )
        expected_groups.add(key)
        expected_counts[key][int(row["intensity"])] += 1
        strength = float(
            row.get("generation_metadata", {}).get(
                "nonlinear_strength",
                math.nan,
            )
        )
        if math.isfinite(strength):
            strength_groups[key][int(row["intensity"])].append(
                (int(row["seed_index"]), strength)
            )
        actual_lag_gain = float(
            row.get("realized_features", {}).get(
                "nonlinear_actual_lag_gain",
                math.nan,
            )
        )
        if math.isfinite(actual_lag_gain):
            actual_lag_groups[key][int(row["intensity"])].append(
                (int(row["seed_index"]), actual_lag_gain)
            )
        observable_gain = float(
            row.get("realized_features", {}).get(
                "nonlinear_conditional_gain",
                math.nan,
            )
        )
        if math.isfinite(observable_gain):
            observable_groups[key][int(row["intensity"])].append(
                (int(row["seed_index"]), observable_gain)
            )
        activity = float(
            row.get("generation_metadata", {}).get(
                "nonlinear_effect_to_recurrence_residual_std_ratio",
                math.nan,
            )
        )
        if math.isfinite(activity):
            activity_groups[key][int(row["intensity"])].append(
                (int(row["seed_index"]), activity)
            )
        clip_fraction = float(
            row.get("generation_metadata", {}).get(
                "state_clip_fraction",
                math.nan,
            )
        )
        if math.isfinite(clip_fraction):
            clip_fractions[key].append(clip_fraction)

    results: list[dict[str, Any]] = []
    for key in sorted(expected_groups):
        strengths_by_intensity = strength_groups[key]
        strength_means = {
            intensity: float(
                np.mean([value for _seed, value in values])
            )
            for intensity, values in sorted(
                strengths_by_intensity.items()
            )
            if values
        }
        ordered_strengths = [
            strength_means[index] for index in sorted(strength_means)
        ]
        missing_strength_count = sum(
            expected_count
            - len(strengths_by_intensity.get(intensity, ()))
            for intensity, expected_count
            in expected_counts[key].items()
        )
        strength_seed_values = {
            intensity: {seed: value for seed, value in values}
            for intensity, values in strengths_by_intensity.items()
        }
        strength_paired_deltas: list[float] = []
        if len(strength_seed_values) >= 2:
            lower_intensity = min(strength_seed_values)
            upper_intensity = max(strength_seed_values)
            shared_seeds = sorted(
                set(strength_seed_values[lower_intensity])
                & set(strength_seed_values[upper_intensity])
            )
            strength_paired_deltas = [
                strength_seed_values[upper_intensity][seed]
                - strength_seed_values[lower_intensity][seed]
                for seed in shared_seeds
            ]

        actual_by_intensity = actual_lag_groups[key]
        actual_means = {
            intensity: float(
                np.mean([value for _seed, value in values])
            )
            for intensity, values in sorted(actual_by_intensity.items())
            if values
        }
        missing_actual_lag_gain_count = sum(
            expected_count
            - len(actual_by_intensity.get(intensity, ()))
            for intensity, expected_count
            in expected_counts[key].items()
        )
        actual_seed_values = {
            intensity: {seed: value for seed, value in values}
            for intensity, values in actual_by_intensity.items()
        }
        actual_paired_deltas: list[float] = []
        if len(actual_seed_values) >= 2:
            lower_intensity = min(actual_seed_values)
            upper_intensity = max(actual_seed_values)
            shared_seeds = sorted(
                set(actual_seed_values[lower_intensity])
                & set(actual_seed_values[upper_intensity])
            )
            actual_paired_deltas = [
                actual_seed_values[upper_intensity][seed]
                - actual_seed_values[lower_intensity][seed]
                for seed in shared_seeds
            ]

        observable_by_intensity = observable_groups[key]
        observable_means = {
            intensity: float(
                np.mean([value for _seed, value in values])
            )
            for intensity, values in sorted(
                observable_by_intensity.items()
            )
            if values
        }
        missing_observable_count = sum(
            expected_count
            - len(observable_by_intensity.get(intensity, ()))
            for intensity, expected_count
            in expected_counts[key].items()
        )
        observable_seed_values = {
            intensity: {seed: value for seed, value in values}
            for intensity, values in observable_by_intensity.items()
        }
        observable_paired_deltas: list[float] = []
        if len(observable_seed_values) >= 2:
            lower_intensity = min(observable_seed_values)
            upper_intensity = max(observable_seed_values)
            shared_seeds = sorted(
                set(observable_seed_values[lower_intensity])
                & set(observable_seed_values[upper_intensity])
            )
            observable_paired_deltas = [
                observable_seed_values[upper_intensity][seed]
                - observable_seed_values[lower_intensity][seed]
                for seed in shared_seeds
            ]

        activity_by_intensity = activity_groups[key]
        activity_means = {
            intensity: float(
                np.mean([value for _seed, value in values])
            )
            for intensity, values in sorted(
                activity_by_intensity.items()
            )
            if values
        }
        ordered_activity = [
            activity_means[index] for index in sorted(activity_means)
        ]
        missing_activity_count = sum(
            expected_count
            - len(activity_by_intensity.get(intensity, ()))
            for intensity, expected_count
            in expected_counts[key].items()
        )
        activity_seed_values = {
            intensity: {seed: value for seed, value in values}
            for intensity, values in activity_by_intensity.items()
        }
        activity_paired_deltas: list[float] = []
        if len(activity_seed_values) >= 2:
            lower_intensity = min(activity_seed_values)
            upper_intensity = max(activity_seed_values)
            shared_seeds = sorted(
                set(activity_seed_values[lower_intensity])
                & set(activity_seed_values[upper_intensity])
            )
            activity_paired_deltas = [
                activity_seed_values[upper_intensity][seed]
                - activity_seed_values[lower_intensity][seed]
                for seed in shared_seeds
            ]
        expected_total = sum(expected_counts[key].values())
        clip_values = clip_fractions[key]
        activity_positive_fraction = (
            float(
                np.mean(
                    np.asarray(activity_paired_deltas) > 0.0
                )
            )
            if activity_paired_deltas
            else None
        )
        activity_median_delta = (
            float(np.median(activity_paired_deltas))
            if activity_paired_deltas
            else None
        )
        dynamic_activity_accepted = bool(
            missing_activity_count == 0
            and len(ordered_activity) >= 2
            and ordered_activity[-1] > ordered_activity[0] + 1e-12
            and bool(activity_paired_deltas)
            and activity_positive_fraction is not None
            and activity_positive_fraction
            > MINIMUM_NONLINEAR_ACTIVITY_PAIRED_POSITIVE_FRACTION
            and activity_median_delta is not None
            and activity_median_delta > 1e-12
            and len(clip_values) == expected_total
            and max(clip_values, default=math.inf) <= 1e-12
        )
        accepted = bool(
            missing_strength_count == 0
            and len(ordered_strengths) >= 2
            and max(ordered_strengths) - min(ordered_strengths) > 1e-9
            and all(
                right >= left - 1e-8
                for left, right in zip(
                    ordered_strengths,
                    ordered_strengths[1:],
                )
            )
            and bool(strength_paired_deltas)
            and all(delta > 1e-12 for delta in strength_paired_deltas)
            and missing_actual_lag_gain_count == 0
            and bool(actual_means)
            and max(actual_means.values()) > 1e-9
            and dynamic_activity_accepted
        )
        results.append(
            {
                "dataset_id": key[0],
                "family_role": key[1],
                "feature": "nonlinear_strength",
                "missing_feature_count": missing_strength_count,
                "mean_feature_by_intensity": {
                    str(name): value
                    for name, value in strength_means.items()
                },
                "paired_low_high_count": len(strength_paired_deltas),
                "paired_low_high_positive_fraction": (
                    float(
                        np.mean(
                            np.asarray(strength_paired_deltas) > 0.0
                        )
                    )
                    if strength_paired_deltas
                    else None
                ),
                "paired_low_high_median_delta": (
                    float(np.median(strength_paired_deltas))
                    if strength_paired_deltas
                    else None
                ),
                "actual_lag_gain_diagnostic": {
                    "feature": "nonlinear_actual_lag_gain",
                    "monotonicity_enforced": False,
                    "missing_feature_count": (
                        missing_actual_lag_gain_count
                    ),
                    "mean_feature_by_intensity": {
                        str(name): value
                        for name, value in actual_means.items()
                    },
                    "paired_low_high_count": len(actual_paired_deltas),
                    "paired_low_high_positive_fraction": (
                        float(
                            np.mean(
                                np.asarray(actual_paired_deltas) > 0.0
                            )
                        )
                        if actual_paired_deltas
                        else None
                    ),
                    "paired_low_high_median_delta": (
                        float(np.median(actual_paired_deltas))
                        if actual_paired_deltas
                        else None
                    ),
                },
                "observable_proxy_diagnostic": {
                    "feature": "nonlinear_conditional_gain",
                    "monotonicity_enforced": False,
                    "missing_feature_count": missing_observable_count,
                    "mean_feature_by_intensity": {
                        str(name): value
                        for name, value in observable_means.items()
                    },
                    "paired_low_high_count": len(
                        observable_paired_deltas
                    ),
                    "paired_low_high_positive_fraction": (
                        float(
                            np.mean(
                                np.asarray(observable_paired_deltas) > 0.0
                            )
                        )
                        if observable_paired_deltas
                        else None
                    ),
                    "paired_low_high_median_delta": (
                        float(np.median(observable_paired_deltas))
                        if observable_paired_deltas
                        else None
                    ),
                },
                "dynamic_activity_gate": {
                    "feature": (
                        "nonlinear_effect_to_recurrence_residual_std_ratio"
                    ),
                    "missing_feature_count": missing_activity_count,
                    "mean_feature_by_intensity": {
                        str(name): value
                        for name, value in activity_means.items()
                    },
                    "paired_low_high_count": len(activity_paired_deltas),
                    "paired_low_high_positive_fraction": (
                        activity_positive_fraction
                    ),
                    "minimum_paired_positive_fraction": (
                        MINIMUM_NONLINEAR_ACTIVITY_PAIRED_POSITIVE_FRACTION
                    ),
                    "strict_majority_required": True,
                    "intermediate_intensity_means_are_diagnostic_only": True,
                    "paired_low_high_median_delta": activity_median_delta,
                    "state_clip_missing_count": (
                        expected_total - len(clip_values)
                    ),
                    "maximum_state_clip_fraction": (
                        max(clip_values) if clip_values else None
                    ),
                    "accepted": dynamic_activity_accepted,
                },
                "accepted": accepted,
            }
        )
    return results


def paired_off_target_selectivity_matrix(
    samples: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize paired intensity effects on every capability coordinate.

    This is intentionally diagnostic.  Each column is normalized by that
    feature's own median paired low/high response under its owning
    intervention.  Thus the diagonal is one when identifiable, and each
    off-diagonal cell directly answers how large the cross-response is relative
    to the feature's intended dose response.

    Multi-period sidebands and sinusoidal amplitude modulation are a known
    Fourier equivalence.  Those cells remain visible in the matrix but are
    explicitly excluded from the maximum off-target summary.
    """

    rows = [
        row
        for row in samples
        if row.get("evaluation_table", "main") == "main"
        and row.get("counterfactual_member") in (None, 0)
    ]
    feature_names = tuple(PRIMARY_FEATURE_BY_CAPABILITY.values())
    scopes: set[tuple[str, str]] = set()
    by_path: dict[
        tuple[str, str, str, int], dict[int, dict[str, float]]
    ] = defaultdict(dict)
    for row in rows:
        scope = (
            str(row["dataset_id"]),
            str(row["generator_family_role"]),
        )
        scopes.add(scope)
        features = dict(row.get("realized_features") or {})
        target_name = PRIMARY_FEATURE_BY_CAPABILITY.get(
            str(row.get("capability_id"))
        )
        if target_name and row.get("target_feature_value") is not None:
            features.setdefault(
                target_name,
                float(row["target_feature_value"]),
            )
        finite = {
            name: float(features[name])
            for name in feature_names
            if name in features and math.isfinite(float(features[name]))
        }
        by_path[
            (
                scope[0],
                scope[1],
                str(row["capability_id"]),
                int(row["seed_index"]),
            )
        ][int(row["intensity"])] = finite

    results: list[dict[str, Any]] = []
    exception_pairs = {
        (
            str(row["intervention_capability"]),
            str(row["feature"]),
        )
        for row in SELECTIVITY_EXCEPTIONS
    }
    for scope in sorted(scopes):
        raw_deltas: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        paired_counts: dict[str, int] = {}
        for capability_id in PRIMARY_FEATURE_BY_CAPABILITY:
            path_count = 0
            for key, intensities in by_path.items():
                if key[:2] != scope or key[2] != capability_id:
                    continue
                if len(intensities) < 2:
                    continue
                lower = intensities[min(intensities)]
                upper = intensities[max(intensities)]
                shared = set(lower) & set(upper)
                if not shared:
                    continue
                path_count += 1
                for name in shared:
                    raw_deltas[capability_id][name].append(
                        abs(float(upper[name]) - float(lower[name]))
                    )
            paired_counts[capability_id] = path_count

        normalizers: dict[str, float | None] = {}
        normalizer_counts: dict[str, int] = {}
        for capability_id, feature_name in (
            PRIMARY_FEATURE_BY_CAPABILITY.items()
        ):
            values = raw_deltas[capability_id].get(feature_name, [])
            normalizer_counts[feature_name] = len(values)
            span = float(np.median(values)) if values else math.nan
            normalizers[feature_name] = (
                span if math.isfinite(span) and span > 1e-12 else None
            )

        matrix: dict[str, dict[str, float | None]] = {}
        selectivity: dict[str, dict[str, Any]] = {}
        for capability_id, on_target in PRIMARY_FEATURE_BY_CAPABILITY.items():
            matrix[capability_id] = {
                name: (
                    float(np.median(raw_deltas[capability_id][name]))
                    / float(normalizers[name])
                    if raw_deltas[capability_id].get(name)
                    and normalizers.get(name) is not None
                    else None
                )
                for name in feature_names
            }
            on_value = matrix[capability_id].get(on_target)
            off_values = [
                float(value)
                for name, value in matrix[capability_id].items()
                if name != on_target
                and value is not None
                and (capability_id, name) not in exception_pairs
            ]
            maximum_off = max(off_values) if off_values else None
            maximum_off_feature = next(
                (
                    name
                    for name, value in matrix[capability_id].items()
                    if name != on_target
                    and value is not None
                    and (capability_id, name) not in exception_pairs
                    and float(value) == maximum_off
                ),
                None,
            )
            excluded_features = [
                name
                for name, value in matrix[capability_id].items()
                if value is not None
                and (capability_id, name) in exception_pairs
            ]
            selectivity[capability_id] = {
                "on_target_normalized_delta": on_value,
                "maximum_off_target_normalized_delta": maximum_off,
                "maximum_nonexception_off_target_normalized_delta": maximum_off,
                "maximum_nonexception_off_target_feature": (
                    maximum_off_feature
                ),
                "excluded_off_target_features": excluded_features,
                "on_to_max_off_target_ratio": (
                    None
                    if on_value is None or maximum_off is None
                    else float(on_value) / max(maximum_off, 1e-12)
                ),
            }
        results.append(
            {
                "dataset_id": scope[0],
                "family_role": scope[1],
                "diagnostic_only": True,
                "blocking": False,
                "normalization": (
                    "feature_owner_intervention_median_abs_paired_low_high_delta"
                ),
                "feature_by_capability": dict(
                    PRIMARY_FEATURE_BY_CAPABILITY
                ),
                "feature_own_intervention_span": normalizers,
                "feature_own_intervention_paired_seed_count": normalizer_counts,
                "paired_seed_count_by_intervention": paired_counts,
                "normalized_absolute_delta_matrix": matrix,
                "selectivity_summary": selectivity,
                "selectivity_exceptions": [
                    dict(row) for row in SELECTIVITY_EXCEPTIONS
                ],
            }
        )
    return results


def validate_sample_collection(
    samples: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(samples)
    basic_results = {
        str(row["sample_id"]): basic_sample_checks(row) for row in rows
    }
    content_hashes: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        content_hashes[sample_content_sha256(row)].append(str(row["sample_id"]))
    row_by_id = {str(row["sample_id"]): row for row in rows}
    duplicate_groups = []
    for identifiers in content_hashes.values():
        if len(identifiers) <= 1:
            continue
        independent_seed_groups = {
            (
                str(row_by_id[identifier].get("dataset_id")),
                str(row_by_id[identifier].get("capability_id")),
                str(row_by_id[identifier].get("generator_family_role")),
                int(row_by_id[identifier].get("intensity", -1)),
                int(row_by_id[identifier].get("seed_index", -1)),
            )
            for identifier in identifiers
        }
        if len(independent_seed_groups) > 1:
            duplicate_groups.append(identifiers)

    intensity_groups: dict[
        tuple[str, str, str], dict[int, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("evaluation_table", "main") != "main":
            continue
        feature = float(row.get("target_feature_value", math.nan))
        if math.isfinite(feature):
            intensity_groups[
                (
                    str(row["dataset_id"]),
                    str(row["capability_id"]),
                    str(row["generator_family_role"]),
                )
            ][int(row["intensity"])].append(feature)
    dose_results: list[dict[str, Any]] = []
    for key, values_by_intensity in sorted(intensity_groups.items()):
        means = {
            intensity: float(np.mean(values))
            for intensity, values in sorted(values_by_intensity.items())
            if values
        }
        ordered = [means[index] for index in sorted(means)]
        accepted = bool(
            len(ordered) >= 2
            and max(ordered) - min(ordered) > 1e-9
            and all(
                right >= left - 1e-8
                for left, right in zip(ordered, ordered[1:])
            )
        )
        dose_results.append(
            {
                "dataset_id": key[0],
                "capability_id": key[1],
                "family_role": key[2],
                "mean_feature_by_intensity": {
                    str(name): value for name, value in means.items()
                },
                "accepted": accepted,
            }
        )

    pair_groups: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = row.get("counterfactual_pair_id")
        member = row.get("counterfactual_member")
        if pair_id is not None and member is not None:
            pair_groups[str(pair_id)][int(member)] = row
    structural_results: list[dict[str, Any]] = []
    for row in rows:
        if row["capability_id"] == "hierarchical_coherence":
            result = hierarchy_checks(row)
            structural_results.append(
                {
                    "sample_id": row["sample_id"],
                    "capability_id": row["capability_id"],
                    **result,
                }
            )
    for pair_id, members in sorted(pair_groups.items()):
        if set(members) != {0, 1}:
            structural_results.append(
                {
                    "pair_id": pair_id,
                    "accepted": False,
                    "error": "incomplete_counterfactual_pair",
                }
            )
            continue
        first, second = members[0], members[1]
        capability = str(first["capability_id"])
        enforced = (
            first["generator_family_role"] == "primary"
            and int(first["intensity"]) == 5
        )
        if capability == "common_factor":
            result = common_factor_identifiability_gate(
                np.asarray(first["target"], dtype=float),
                np.asarray(second["target"], dtype=float),
                context_length=int(first["context_length"]),
                metadata=first["generation_metadata"],
                enforced=enforced,
            )
        elif capability == "cross_series_dependence":
            result = cross_series_identifiability_gate(
                np.asarray(first["target"], dtype=float),
                np.asarray(second["target"], dtype=float),
                context_length=int(first["context_length"]),
                metadata=first["generation_metadata"],
                enforced=enforced,
            )
        elif capability == "covariate_response":
            result = covariate_pair_checks(first, second)
            result["enforced"] = enforced
        else:
            continue
        structural_results.append(
            {
                "pair_id": pair_id,
                "capability_id": capability,
                **result,
            }
        )

    covariate_primary_by_match = {
        (
            str(row["dataset_id"]),
            int(row["seed_index"]),
            int(row["intensity"]),
            row.get("counterfactual_member"),
        ): row
        for row in rows
        if row["capability_id"] == "covariate_response"
        and row["generator_family_role"] == "primary"
        and row.get("evaluation_table", "main") == "main"
    }
    matched_family_results: list[dict[str, Any]] = []
    for secondary in rows:
        if (
            secondary["capability_id"] != "covariate_response"
            or secondary["generator_family_role"] != "secondary"
            or secondary.get("evaluation_table", "main") != "main"
        ):
            continue
        key = (
            str(secondary["dataset_id"]),
            int(secondary["seed_index"]),
            int(secondary["intensity"]),
            secondary.get("counterfactual_member"),
        )
        primary = covariate_primary_by_match.get(key)
        if primary is None:
            matched_family_results.append(
                {
                    "secondary_sample_id": secondary["sample_id"],
                    "accepted": False,
                    "error": "matched_primary_sample_missing",
                }
            )
            continue
        matched_family_results.append(
            covariate_family_match_checks(primary, secondary)
        )

    nonlinear_mechanism_results = nonlinear_mechanism_response_checks(
        rows
    )
    off_target_selectivity = paired_off_target_selectivity_matrix(rows)
    accepted = bool(
        all(result["accepted"] for result in basic_results.values())
        and not duplicate_groups
        and all(result["accepted"] for result in dose_results)
        and all(
            result["accepted"]
            for result in nonlinear_mechanism_results
        )
        and all(
            result.get("accepted", False)
            for result in structural_results
            if result.get("enforced", True)
        )
        and all(
            result["accepted"] for result in matched_family_results
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted": accepted,
        "sample_count": len(rows),
        "basic_failure_count": sum(
            not result["accepted"] for result in basic_results.values()
        ),
        "duplicate_groups": duplicate_groups,
        "dose_response": dose_results,
        "nonlinear_mechanism_response": nonlinear_mechanism_results,
        "structural_results": structural_results,
        "matched_family_results": matched_family_results,
        "off_target_selectivity": off_target_selectivity,
    }
