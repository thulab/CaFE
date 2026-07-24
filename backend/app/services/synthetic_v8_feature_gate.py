from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from app.services.synthetic_v8_generation import (
    common_factor_identifiability_gate,
    cross_series_identifiability_gate,
)


SCHEMA_VERSION = "synthetic_v8_feature_gate.v3"
COUNTERFACTUAL_CAPABILITIES = frozenset(
    {
        "common_factor",
        "cross_series_dependence",
        "covariate_response",
    }
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


def covariate_family_match_checks(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    dose_atol: float = 1e-8,
    mase_scale_relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Check that covariate secondary changes the response law in isolation."""

    primary_covariates = np.asarray(primary["covariates"], dtype=float)
    secondary_covariates = np.asarray(secondary["covariates"], dtype=float)
    primary_metadata = primary["generation_metadata"]
    secondary_metadata = secondary["generation_metadata"]
    primary_scale = float(primary["mase_scale"])
    scale_relative_difference = abs(
        float(secondary["mase_scale"]) - primary_scale
    ) / max(abs(primary_scale), 1e-12)
    values = {
        "primary_sample_id": str(primary["sample_id"]),
        "secondary_sample_id": str(secondary["sample_id"]),
        "covariate_max_abs_difference": float(
            np.max(np.abs(primary_covariates - secondary_covariates))
        ),
        "dose_absolute_difference": abs(
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
    }
    values["accepted"] = bool(
        values["covariate_max_abs_difference"] <= dose_atol
        and values["dose_absolute_difference"] <= dose_atol
        and values["effect_strength_absolute_difference"] <= dose_atol
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
    and within every paired seed. ``nonlinear_conditional_gain`` remains an
    observable lag-search diagnostic shared with real windows, but it is not
    used as an inverse coordinate or assumed monotone.  The adjusted-R² gain at
    the exact causal lag is likewise diagnostic: recursive feedback can spread
    dependence over correlated lags and make either conditional proxy fall
    while the injected coefficient grows.
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
        dynamic_activity_accepted = bool(
            missing_activity_count == 0
            and len(ordered_activity) >= 2
            and all(
                right > left + 1e-12
                for left, right in zip(
                    ordered_activity,
                    ordered_activity[1:],
                )
            )
            and bool(activity_paired_deltas)
            and all(delta > 1e-12 for delta in activity_paired_deltas)
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
                        float(
                            np.mean(
                                np.asarray(activity_paired_deltas) > 0.0
                            )
                        )
                        if activity_paired_deltas
                        else None
                    ),
                    "paired_low_high_median_delta": (
                        float(np.median(activity_paired_deltas))
                        if activity_paired_deltas
                        else None
                    ),
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
    }
