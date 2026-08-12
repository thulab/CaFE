#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cafe import protocol
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_MASTER_SCHEMA,
    array_sha256,
)
from cafe.validation.mechanisms import (
    basic_sample_checks,
    validate_sample_collection,
)


DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a formal CaFE generated sample shard."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    return parser.parse_args()


def validate_manifest_file(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if protocol.file_sha256(path) != record["sha256"]:
        raise ValueError(f"manifest hash mismatch: {path}")


def robustness_checks(
    clean_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_by_id = {row["sample_id"]: row for row in clean_rows}
    failures: list[dict[str, Any]] = []
    for row in robustness_rows:
        basic = basic_sample_checks(row)
        clean_id = str(row["clean_master_sample_id"])
        clean = clean_by_id.get(clean_id)
        checks = {
            "basic": basic["accepted"],
            "clean_parent_exists": clean is not None,
        }
        if clean is not None:
            context = int(row["context_length"])
            observed = np.asarray(row["target"], dtype=float)
            latent = np.asarray(clean["target"], dtype=float)
            checks.update(
                {
                    "future_exact": bool(
                        np.array_equal(observed[context:], latent[context:])
                    ),
                    "history_changed": bool(
                        not np.array_equal(
                            observed[:context],
                            latent[:context],
                        )
                    ),
                    "mase_scale_reused": bool(
                        float(row["mase_scale"]) == float(clean["mase_scale"])
                    ),
                }
            )
        if not all(checks.values()):
            failures.append(
                {"sample_id": row["sample_id"], "checks": checks}
            )
    return {
        "accepted": not failures,
        "sample_count": len(robustness_rows),
        "failures": failures,
    }


def input_ablation_checks(
    clean_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_by_id = {row["sample_id"]: row for row in clean_rows}
    failures: list[dict[str, Any]] = []
    for row in ablation_rows:
        basic = basic_sample_checks(row)
        clean = clean_by_id.get(str(row["clean_master_sample_id"]))
        checks = {
            "basic": basic["accepted"],
            "clean_parent_exists": clean is not None,
        }
        if clean is not None:
            context = int(row["context_length"])
            observed = np.asarray(row["target"], dtype=float)
            latent = np.asarray(clean["target"], dtype=float)
            metadata = row["input_ablation_metadata"]
            donor = clean_by_id.get(str(metadata["donor_sample_id"]))
            channels = [
                int(value) for value in metadata["replaced_channels"]
            ]
            start, stop = (
                int(value)
                for value in metadata["replaced_history_slice"]
            )
            untouched_channels = [
                index
                for index in range(int(row["target_dim"]))
                if index not in channels
            ]
            expected_scope = (
                "replaced_segment"
                if str(row["capability_id"]) == "common_factor"
                else "pair_invariant_driver_prefix"
            )
            segment_match_declared = bool(
                metadata.get("affine_matched_mean_and_std", False)
            )
            expected_segment: np.ndarray | None = None
            if donor is not None:
                donor_target = np.asarray(donor["target"], dtype=float)
                expected_segment = np.empty(
                    (stop - start, len(channels)),
                    dtype=float,
                )
                for column, channel in enumerate(channels):
                    if expected_scope == "replaced_segment":
                        donor_values = donor_target[start:stop, channel]
                        reference_values = latent[start:stop, channel]
                        donor_center = float(np.mean(donor_values))
                        donor_scale = float(np.std(donor_values))
                        reference_center = float(np.mean(reference_values))
                        reference_scale = float(np.std(reference_values))
                        if donor_scale <= 1e-12:
                            expected_segment[:, column] = reference_center
                        else:
                            expected_segment[:, column] = (
                                (donor_values - donor_center)
                                * reference_scale
                                / donor_scale
                                + reference_center
                            )
                    else:
                        donor_context = (
                            donor_target.shape[0] - int(donor["horizon"])
                        )
                        donor_start = donor_context - (stop - start)
                        donor_reference = donor_target[
                            :donor_start,
                            channel,
                        ]
                        clean_reference = latent[:start, channel]
                        expected_segment[:, column] = (
                            (
                                donor_target[
                                    donor_start:donor_context,
                                    channel,
                                ]
                                - float(np.mean(donor_reference))
                            )
                            * max(float(np.std(clean_reference)), 1e-12)
                            / max(float(np.std(donor_reference)), 1e-12)
                            + float(np.mean(clean_reference))
                        )
            checks.update(
                {
                    "future_exact": bool(
                        np.array_equal(observed[context:], latent[context:])
                    ),
                    "replaced_history_changed": bool(
                        not np.array_equal(
                            observed[start:stop, channels],
                            latent[start:stop, channels],
                        )
                    ),
                    "untouched_channels_exact": bool(
                        not untouched_channels
                        or np.array_equal(
                            observed[:context, untouched_channels],
                            latent[:context, untouched_channels],
                        )
                    ),
                    "donor_parent_exists": donor is not None,
                    "affine_reference_scope_valid": bool(
                        metadata.get("affine_reference_scope")
                        == expected_scope
                    ),
                    "segment_match_declaration_valid": bool(
                        segment_match_declared
                        == (expected_scope == "replaced_segment")
                    ),
                    "declared_segment_mean_matched": bool(
                        not segment_match_declared
                        or np.allclose(
                            np.mean(
                                observed[start:stop, channels],
                                axis=0,
                            ),
                            np.mean(
                                latent[start:stop, channels],
                                axis=0,
                            ),
                            atol=1e-10,
                            rtol=1e-10,
                        )
                    ),
                    "declared_segment_std_matched": bool(
                        not segment_match_declared
                        or np.allclose(
                            np.std(
                                observed[start:stop, channels],
                                axis=0,
                            ),
                            np.std(
                                latent[start:stop, channels],
                                axis=0,
                            ),
                            atol=1e-10,
                            rtol=1e-10,
                        )
                    ),
                    "declared_affine_transform_exact": bool(
                        expected_segment is not None
                        and np.allclose(
                            observed[start:stop, channels],
                            expected_segment,
                            atol=1e-12,
                            rtol=1e-12,
                        )
                    ),
                    "mase_scale_reused": bool(
                        float(row["mase_scale"]) == float(clean["mase_scale"])
                    ),
                }
            )
        if not all(checks.values()):
            failures.append(
                {"sample_id": row["sample_id"], "checks": checks}
            )
    return {
        "accepted": not failures,
        "sample_count": len(ablation_rows),
        "failures": failures,
    }


def mase_scale_audit(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "policy": (
                "not_applicable_no_deterministic_synthetic_samples"
            ),
            "mase_period_counts": {},
            "effective_period_by_target_counts": {},
            "fallback_sample_count": 0,
            "fallback_target_count": 0,
            "mase_scale_to_history_std": None,
            "by_capability": {},
        }
    by_capability: dict[str, list[float]] = {}
    period_counts: Counter[int] = Counter()
    effective_period_counts: Counter[int] = Counter()
    fallback_sample_count = 0
    fallback_target_count = 0
    ratios: list[float] = []
    for row in rows:
        target = np.asarray(row["target"], dtype=float)
        context = int(row["context_length"])
        history_scale = float(
            np.mean(np.std(target[:context], axis=0))
        )
        ratio = float(row["mase_scale"]) / max(history_scale, 1e-12)
        ratios.append(ratio)
        by_capability.setdefault(
            str(row["capability_id"]),
            [],
        ).append(ratio)
        period_counts[int(row["mase_period"])] += 1
        effective_periods = row.get(
            "mase_scale_effective_period_by_target",
            [int(row["mase_period"])] * int(row["target_dim"]),
        )
        effective_period_counts.update(
            int(period) for period in effective_periods
        )
        fallback_indices = row.get(
            "mase_scale_fallback_target_indices",
            [],
        )
        fallback_sample_count += int(bool(fallback_indices))
        fallback_target_count += len(fallback_indices)

    def summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=float)
        return {
            "minimum": float(np.min(array)),
            "p05": float(np.quantile(array, 0.05)),
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95)),
            "maximum": float(np.max(array)),
        }

    return {
        "sample_count": len(rows),
        "policy": (
            "seasonal_lag_with_per_target_lag1_degeneracy_fallback_v1; "
            "no denominator floor; companion inference metric="
            "history_std_normalized_mae"
        ),
        "mase_period_counts": {
            str(period): count
            for period, count in sorted(period_counts.items())
        },
        "effective_period_by_target_counts": {
            str(period): count
            for period, count in sorted(effective_period_counts.items())
        },
        "fallback_sample_count": fallback_sample_count,
        "fallback_target_count": fallback_target_count,
        "mase_scale_to_history_std": summary(ratios),
        "by_capability": {
            capability_id: summary(values)
            for capability_id, values in sorted(by_capability.items())
        },
    }


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same_finite_float(*values: Any) -> bool:
    return bool(
        values
        and all(_finite_float(value) for value in values)
        and all(float(value) == float(values[0]) for value in values[1:])
    )


def _exact_shared(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> bool:
    if not rows:
        return True
    first = rows[0]
    return all(
        row.get(field) == first.get(field)
        for row in rows[1:]
        for field in fields
    )


def _metadata_exact_shared(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> bool:
    if not rows:
        return True
    metadata = [row.get("generation_metadata") for row in rows]
    if not all(isinstance(value, dict) for value in metadata):
        return False
    first = metadata[0]
    return all(
        value.get(field) == first.get(field)
        for value in metadata[1:]
        for field in fields
    )


def real_anchored_counterfactual_checks(
    rows: list[dict[str, Any]],
    *,
    expected_row_count: int | None = None,
) -> dict[str, Any]:
    """Validate the intentional-real-anchor counterfactual component.

    Repeated alpha=1 baselines across physical doses are intentional here.
    Consequently duplicate *content* is not an anti-copy failure, while IDs,
    exact pair deltas, frozen references, and the monotone dose response are
    all hard requirements.
    """

    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    master_ids = [str(row.get("master_sample_id", "")) for row in rows]
    duplicate_sample_ids = sorted(
        identifier
        for identifier, count in Counter(sample_ids).items()
        if identifier and count > 1
    )
    duplicate_master_ids = sorted(
        identifier
        for identifier, count in Counter(master_ids).items()
        if identifier and count > 1
    )
    row_failures: list[dict[str, Any]] = []
    targets: dict[int, np.ndarray] = {}
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_length = (
        protocol.REAL_ANCHORED_CONTEXT_LENGTH + protocol.HORIZON
    )
    reference_start = (
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        - protocol.REAL_ANCHORED_CONTEXT_LENGTH
    )

    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        pair_id = str(row.get("counterfactual_pair_id", ""))
        group_id = str(row.get("paired_group_id", ""))
        member = row.get("counterfactual_member")
        metadata = row.get("generation_metadata")
        parameter_sampling = row.get("parameter_sampling")
        sampled_parameters = row.get("sampled_generator_parameters")
        calibration = row.get("intensity_calibration")
        standardization = row.get("shared_standardization")
        anti_copy = row.get("anti_copy_gate")
        try:
            target = np.asarray(row.get("target"), dtype=float)
        except (TypeError, ValueError):
            target = np.empty((0, 0), dtype=float)
        target_valid = bool(
            target.shape == (expected_length, 1)
            and np.isfinite(target).all()
        )
        if target_valid:
            targets[id(row)] = target

        member_valid = isinstance(member, int) and member in (0, 1)
        dose_index = row.get("dose_index")
        dose_index_valid = bool(
            isinstance(dose_index, int) and dose_index >= 1
        )
        alpha_values = (
            calibration.get("selected_alphas")
            if isinstance(calibration, dict)
            else None
        )
        alpha_grid_valid = bool(
            isinstance(alpha_values, list)
            and alpha_values
            and all(
                _finite_float(value) and float(value) > 1.0
                for value in alpha_values
            )
            and all(
                float(right) > float(left)
                for left, right in zip(alpha_values, alpha_values[1:])
            )
        )
        alpha = row.get("dose_value")
        exposed_alphas = [
            alpha,
            row.get("intensity_lambda"),
            metadata.get("alpha") if isinstance(metadata, dict) else None,
            (
                sampled_parameters.get("alpha")
                if isinstance(sampled_parameters, dict)
                else None
            ),
        ]
        expected_member_alpha = bool(
            member_valid
            and _finite_float(alpha)
            and (
                float(alpha) == 1.0
                if member == 0
                else float(alpha) > 1.0
            )
        )

        normalization_mean = (
            metadata.get("normalization_mean_by_target")
            if isinstance(metadata, dict)
            else None
        )
        normalization_scale = (
            metadata.get("normalization_scale_by_target")
            if isinstance(metadata, dict)
            else None
        )
        mase_by_target = row.get("mase_scale_by_target")
        effective_mase_periods = row.get(
            "mase_scale_effective_period_by_target"
        )
        mase_period = row.get("mase_period")
        metadata_mase = (
            metadata.get("mase_scale_by_target")
            if isinstance(metadata, dict)
            else None
        )
        standardization_valid = bool(
            isinstance(standardization, dict)
            and standardization.get("scope")
            == "shared_unmodified_real_l336_history"
            and _finite_float(standardization.get("location"))
            and _finite_float(standardization.get("scale"))
            and float(standardization["scale"]) > 0.0
            and isinstance(normalization_mean, list)
            and len(normalization_mean) == 1
            and _finite_float(normalization_mean[0])
            and isinstance(normalization_scale, list)
            and len(normalization_scale) == 1
            and _finite_float(normalization_scale[0])
            and float(normalization_scale[0]) > 0.0
            and metadata.get("normalization_policy")
            == "baseline_history_shared_by_pair_v1"
            and metadata.get("reference_history_policy")
            == "unmodified_fit_history_suffix_shared_by_pair_v1"
            and metadata.get("reference_start") == reference_start
            and metadata.get("reference_length")
            == protocol.REAL_ANCHORED_CONTEXT_LENGTH
            and _is_sha256(metadata.get("reference_history_sha256"))
        )
        mase_valid = bool(
            _finite_float(row.get("mase_scale"))
            and float(row["mase_scale"]) > 0.0
            and isinstance(mase_by_target, list)
            and len(mase_by_target) == 1
            and _same_finite_float(row["mase_scale"], mase_by_target[0])
            and isinstance(metadata_mase, list)
            and metadata_mase == mase_by_target
            and isinstance(mase_period, int)
            and 1 <= mase_period < protocol.REAL_ANCHORED_CONTEXT_LENGTH
            and isinstance(effective_mase_periods, list)
            and len(effective_mase_periods) == 1
            and effective_mase_periods[0] in (0, 1, mase_period)
            and _same_finite_float(
                row.get("mase_scale"),
                metadata.get("mase_scale")
                if isinstance(metadata, dict)
                else None,
            )
            and row.get("mase_period")
            == (
                metadata.get("mase_period")
                if isinstance(metadata, dict)
                else None
            )
            and row.get("mase_scale_effective_period_by_target")
            == (
                metadata.get("mase_effective_period_by_target")
                if isinstance(metadata, dict)
                else None
            )
            and metadata.get("mase_reference_policy")
            == "baseline_history_shared_by_pair_v1"
            and isinstance(metadata.get("mase_scale_source_by_target"), list)
            and len(metadata["mase_scale_source_by_target"]) == 1
            and row.get("mase_scale_source")
            == "shared_unmodified_real_l336_history"
        )
        background_id = str(row.get("background_id", ""))
        contract_hash = (
            metadata.get("contract_sha256")
            if isinstance(metadata, dict)
            else None
        )
        contract_valid = bool(
            background_id
            and str(row.get("anchor_id", "")) == background_id
            and isinstance(parameter_sampling, dict)
            and parameter_sampling.get("background_id") == background_id
            and parameter_sampling.get("contract_sha256") == contract_hash
            and _is_sha256(contract_hash)
            and _is_sha256(
                metadata.get("capability_contract_sha256")
                if isinstance(metadata, dict)
                else None
            )
            and _is_sha256(metadata.get("source_history_sha256"))
            and _is_sha256(
                metadata.get("decomposition_history_sha256")
            )
            and metadata.get("capability_id") == row.get("capability_id")
        )
        target_hash_matches = bool(
            target_valid
            and row.get("target_sha256")
            == protocol.target_and_covariate_sha256(target, None)
        )
        future_hash_matches = bool(
            target_valid
            and row.get("future_sha256")
            == array_sha256(
                target[protocol.REAL_ANCHORED_CONTEXT_LENGTH :]
            )
        )
        checks = {
            "schema_valid": row.get("schema_version")
            == REAL_ANCHORED_MASTER_SCHEMA,
            "track_valid": bool(
                row.get("benchmark_track")
                == "real_anchored_counterfactual"
                and row.get("generator_family_role") == "real_anchored"
            ),
            "sample_id_valid": bool(
                sample_id
                and member_valid
                and sample_id == f"{pair_id}__m{member}"
                and row.get("master_sample_id") == sample_id
            ),
            "pair_id_valid": bool(pair_id and group_id),
            "baseline_sample_id_valid": bool(
                pair_id
                and row.get("baseline_sample_id") == f"{pair_id}__m0"
            ),
            "member_valid": member_valid,
            "fixed_shape_valid": bool(
                row.get("context_length")
                == protocol.REAL_ANCHORED_CONTEXT_LENGTH
                and row.get("horizon") == protocol.HORIZON
                and row.get("target_dim") == 1
                and row.get("covariate_dim") == 0
                and row.get("covariates") is None
                and target_valid
            ),
            "target_hash_matches": target_hash_matches,
            "future_hash_matches": future_hash_matches,
            "baseline_hashes_well_formed": all(
                _is_sha256(row.get(field))
                for field in (
                    "baseline_history_sha256",
                    "baseline_future_sha256",
                    "baseline_target_sha256",
                    "intervention_delta_sha256",
                )
            ),
            "dose_index_valid": bool(
                dose_index_valid
                and row.get("intensity") == dose_index
                and row.get("dose_parameter") == "alpha"
                and _same_finite_float(row.get("baseline_dose_value"), 1.0)
            ),
            "alpha_grid_valid": alpha_grid_valid,
            "alpha_exposure_consistent": bool(
                _same_finite_float(*exposed_alphas)
                and expected_member_alpha
            ),
            "target_feature_matches_metadata": bool(
                isinstance(metadata, dict)
                and row.get("target_feature")
                == "real_anchored_intervention_rms"
                and _same_finite_float(
                    row.get("target_feature_value"),
                    row.get("intensity_target_feature_value"),
                    metadata.get("intervention_rms"),
                )
                and float(row.get("target_feature_value", -1.0)) >= 0.0
            ),
            "normalization_reference_valid": standardization_valid,
            "mase_reference_valid": mase_valid,
            "contract_and_background_valid": contract_valid,
            "anti_copy_explicitly_not_applicable": anti_copy
            == {
                "status": "not_applicable",
                "reason_code": "intentional_real_anchor_counterfactual",
            },
        }
        if not all(checks.values()):
            row_failures.append(
                {
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "checks": checks,
                }
            )
        by_pair[pair_id].append(row)
        by_group[group_id].append(row)

    pair_failures: list[dict[str, Any]] = []
    complete_pairs: dict[str, dict[str, Any]] = {}
    pair_shared_fields = (
        "counterfactual_pair_id",
        "paired_group_id",
        "baseline_sample_id",
        "dataset_id",
        "capability_id",
        "seed_index",
        "dose_index",
        "intensity",
        "background_id",
        "anchor_id",
        "shared_standardization",
        "mase_period",
        "mase_scale",
        "mase_scale_by_target",
        "mase_scale_effective_period_by_target",
        "mase_scale_fallback_target_indices",
        "mase_scale_policy",
        "mase_scale_source",
        "baseline_history_sha256",
        "baseline_future_sha256",
        "baseline_target_sha256",
        "parameter_sampling",
        "intensity_calibration",
    )
    metadata_shared_fields = (
        "capability_id",
        "contract_sha256",
        "capability_contract_sha256",
        "source_history_sha256",
        "decomposition_history_sha256",
        "normalization_mean_by_target",
        "normalization_scale_by_target",
        "normalization_policy",
        "mase_scale_by_target",
        "mase_scale",
        "mase_period",
        "mase_effective_period_by_target",
        "mase_scale_source_by_target",
        "mase_reference_policy",
        "reference_start",
        "reference_length",
        "reference_history_sha256",
        "reference_history_policy",
        "controlled_component",
        "carrier_fixed",
    )
    for pair_id, pair_rows in sorted(by_pair.items()):
        members = [row.get("counterfactual_member") for row in pair_rows]
        pair_complete = bool(
            len(pair_rows) == 2
            and members.count(0) == 1
            and members.count(1) == 1
        )
        checks: dict[str, bool] = {
            "exactly_two_members": pair_complete,
            "pair_fields_shared": _exact_shared(pair_rows, pair_shared_fields),
            "metadata_references_shared": _metadata_exact_shared(
                pair_rows,
                metadata_shared_fields,
            ),
        }
        if pair_complete:
            baseline = next(
                row
                for row in pair_rows
                if row["counterfactual_member"] == 0
            )
            treatment = next(
                row
                for row in pair_rows
                if row["counterfactual_member"] == 1
            )
            baseline_target = targets.get(id(baseline))
            treatment_target = targets.get(id(treatment))
            arrays_valid = bool(
                baseline_target is not None and treatment_target is not None
            )
            if arrays_valid:
                assert baseline_target is not None
                assert treatment_target is not None
                baseline_1d = baseline_target[:, 0]
                delta = treatment_target - baseline_target
                zero_delta = np.zeros_like(baseline_target)
                delta_rms = float(np.sqrt(np.mean(delta**2)))
                checks.update(
                    {
                        "baseline_member_exact": bool(
                            _same_finite_float(
                                baseline.get("dose_value"),
                                1.0,
                            )
                            and isinstance(
                                baseline.get("generation_metadata"),
                                dict,
                            )
                            and _same_finite_float(
                                baseline["generation_metadata"].get(
                                    "intervention_rms"
                                ),
                                0.0,
                            )
                            and _same_finite_float(
                                baseline.get("target_feature_value"),
                                0.0,
                            )
                            and baseline.get("intervention_delta_sha256")
                            == array_sha256(zero_delta)
                        ),
                        "baseline_history_hash_exact": baseline.get(
                            "baseline_history_sha256"
                        )
                        == array_sha256(
                            baseline_1d[
                                : protocol.REAL_ANCHORED_CONTEXT_LENGTH
                            ]
                        ),
                        "baseline_future_hash_exact": baseline.get(
                            "baseline_future_sha256"
                        )
                        == array_sha256(
                            baseline_1d[
                                protocol.REAL_ANCHORED_CONTEXT_LENGTH :
                            ]
                        ),
                        "baseline_target_hash_exact": baseline.get(
                            "baseline_target_sha256"
                        )
                        == array_sha256(baseline_1d),
                        "treatment_baseline_hashes_exact": bool(
                            treatment.get("baseline_history_sha256")
                            == baseline.get("baseline_history_sha256")
                            and treatment.get("baseline_future_sha256")
                            == baseline.get("baseline_future_sha256")
                            and treatment.get("baseline_target_sha256")
                            == baseline.get("baseline_target_sha256")
                        ),
                        "treatment_delta_hash_exact": bool(
                            treatment.get("intervention_delta_sha256")
                            == array_sha256(delta)
                        ),
                        "treatment_delta_nonzero": bool(delta_rms > 0.0),
                    }
                )
                complete_pairs[pair_id] = {
                    "baseline": baseline,
                    "treatment": treatment,
                    "baseline_target": baseline_target,
                    "delta": delta,
                    "delta_rms": delta_rms,
                }
            else:
                checks["pair_targets_valid"] = False
        if not all(checks.values()):
            pair_failures.append(
                {
                    "counterfactual_pair_id": pair_id,
                    "checks": checks,
                }
            )

    group_failures: list[dict[str, Any]] = []
    group_shared_fields = (
        "paired_group_id",
        "dataset_id",
        "capability_id",
        "seed_index",
        "background_id",
        "anchor_id",
        "shared_standardization",
        "mase_period",
        "mase_scale",
        "mase_scale_by_target",
        "mase_scale_effective_period_by_target",
        "mase_scale_fallback_target_indices",
        "mase_scale_policy",
        "mase_scale_source",
        "baseline_history_sha256",
        "baseline_future_sha256",
        "baseline_target_sha256",
        "parameter_sampling",
        "intensity_calibration",
    )
    for group_id, group_rows in sorted(by_group.items()):
        group_pairs = [
            pair
            for pair in complete_pairs.values()
            if pair["baseline"].get("paired_group_id") == group_id
        ]
        try:
            ordered_pairs = sorted(
                group_pairs,
                key=lambda pair: int(pair["treatment"]["dose_index"]),
            )
            dose_indexes = [
                int(pair["treatment"]["dose_index"])
                for pair in ordered_pairs
            ]
            treatment_alphas = [
                float(pair["treatment"]["dose_value"])
                for pair in ordered_pairs
            ]
            delta_rms = [float(pair["delta_rms"]) for pair in ordered_pairs]
            metadata_rms = [
                float(
                    pair["treatment"]["generation_metadata"][
                        "intervention_rms"
                    ]
                )
                for pair in ordered_pairs
            ]
            selected_alphas = [
                float(value)
                for value in group_rows[0]["intensity_calibration"][
                    "selected_alphas"
                ]
            ]
        except (KeyError, TypeError, ValueError, IndexError):
            ordered_pairs = []
            dose_indexes = []
            treatment_alphas = []
            delta_rms = []
            metadata_rms = []
            selected_alphas = []
        baseline_targets = [
            np.asarray(pair["baseline_target"], dtype=float)
            for pair in ordered_pairs
        ]
        scaled_deltas = [
            np.asarray(pair["delta"], dtype=float) / (alpha - 1.0)
            for pair, alpha in zip(ordered_pairs, treatment_alphas)
            if alpha > 1.0
        ]
        scaled_metadata_rms = [
            rms / (alpha - 1.0)
            for rms, alpha in zip(metadata_rms, treatment_alphas)
            if alpha > 1.0
        ]
        checks = {
            "group_id_nonempty": bool(group_id),
            "group_references_shared": bool(
                _exact_shared(group_rows, group_shared_fields)
                and _metadata_exact_shared(
                    group_rows,
                    metadata_shared_fields,
                )
            ),
            "complete_pair_count_matches_grid": bool(
                selected_alphas
                and len(ordered_pairs) == len(selected_alphas)
                and len(group_rows) == 2 * len(selected_alphas)
            ),
            "dose_indexes_match_grid": dose_indexes
            == list(range(1, len(selected_alphas) + 1)),
            "treatment_alphas_match_grid": treatment_alphas
            == selected_alphas,
            "duplicate_baselines_are_exact": bool(
                baseline_targets
                and all(
                    np.array_equal(target, baseline_targets[0])
                    for target in baseline_targets[1:]
                )
            ),
            "delta_rms_strictly_increases": bool(
                delta_rms
                and all(
                    right > left
                    for left, right in zip(delta_rms, delta_rms[1:])
                )
            ),
            "metadata_rms_strictly_increases": bool(
                metadata_rms
                and all(
                    right > left
                    for left, right in zip(metadata_rms, metadata_rms[1:])
                )
            ),
            "delta_is_linear_in_alpha_minus_one": bool(
                scaled_deltas
                and len(scaled_deltas) == len(ordered_pairs)
                and all(
                    np.allclose(
                        delta,
                        scaled_deltas[0],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                    for delta in scaled_deltas[1:]
                )
            ),
            "metadata_rms_is_linear_in_alpha_minus_one": bool(
                scaled_metadata_rms
                and len(scaled_metadata_rms) == len(ordered_pairs)
                and all(
                    math.isclose(
                        value,
                        scaled_metadata_rms[0],
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    )
                    for value in scaled_metadata_rms[1:]
                )
            ),
        }
        if not all(checks.values()):
            group_failures.append(
                {
                    "paired_group_id": group_id,
                    "dose_indexes": dose_indexes,
                    "treatment_alphas": treatment_alphas,
                    "delta_rms": delta_rms,
                    "checks": checks,
                }
            )

    background_groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    seed_backgrounds: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for group_id, group_rows in by_group.items():
        if not group_rows:
            continue
        first = group_rows[0]
        try:
            dataset_id = str(first["dataset_id"])
            capability_id = str(first["capability_id"])
            background_id = str(first["background_id"])
            seed_index = int(first["seed_index"])
        except (KeyError, TypeError, ValueError):
            continue
        background_groups[
            (dataset_id, capability_id, background_id)
        ].add(group_id)
        seed_backgrounds[(dataset_id, capability_id, seed_index)].add(
            background_id
        )
    repeated_background_failures = [
        {
            "dataset_id": key[0],
            "capability_id": key[1],
            "background_id": key[2],
            "paired_group_ids": sorted(group_ids),
        }
        for key, group_ids in sorted(background_groups.items())
        if len(group_ids) > 1
    ]
    seed_assignment_failures = [
        {
            "dataset_id": key[0],
            "capability_id": key[1],
            "seed_index": key[2],
            "background_ids": sorted(background_ids),
        }
        for key, background_ids in sorted(seed_backgrounds.items())
        if len(background_ids) > 1
    ]

    manifest_row_count_matches = bool(
        expected_row_count is None or expected_row_count == len(rows)
    )
    accepted = bool(
        manifest_row_count_matches
        and not duplicate_sample_ids
        and not duplicate_master_ids
        and not row_failures
        and not pair_failures
        and not group_failures
        and not repeated_background_failures
        and not seed_assignment_failures
    )
    return {
        "schema_version": "cafe.real_anchored_validation.v1",
        "status": "evaluated",
        "accepted": accepted,
        "sample_count": len(rows),
        "pair_count": len(by_pair),
        "paired_group_count": len(by_group),
        "expected_manifest_row_count": expected_row_count,
        "manifest_row_count_matches": manifest_row_count_matches,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_master_sample_ids": duplicate_master_ids,
        "row_failures": row_failures,
        "pair_failures": pair_failures,
        "paired_group_failures": group_failures,
        "repeated_background_failures": repeated_background_failures,
        "seed_assignment_failures": seed_assignment_failures,
        "effective_background_count_by_capability": {
            capability_id: len(
                {
                    background_id
                    for (_dataset_id, candidate_capability, background_id) in (
                        background_groups
                    )
                    if candidate_capability == capability_id
                }
            )
            for capability_id in sorted(
                {
                    key[1]
                    for key in background_groups
                }
            )
        },
        "background_sampling_policy": (
            "unique_real_background_per_dataset_capability"
        ),
        "intentional_duplicate_baseline_policy": (
            "allowed_within_paired_group_across_alpha_doses_if_exact"
        ),
        "anti_copy_policy": (
            "not_applicable_intentional_real_anchor_counterfactual"
        ),
    }


def main() -> int:
    args = parse_args()
    dataset = protocol.resolve_dataset(args.dataset_id)
    generation_dir = (
        args.output_root.resolve() / dataset.dataset_id / "02_generation"
    )
    shard_name = (
        f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    )
    manifest_path = generation_dir / f"manifest__{shard_name}.json"
    manifest = protocol.read_json(manifest_path)
    for record in manifest["files"].values():
        validate_manifest_file(record)
    clean_rows = list(
        protocol.iter_jsonl(Path(manifest["files"]["clean"]["path"]))
    )
    robustness_rows = list(
        protocol.iter_jsonl(Path(manifest["files"]["robustness"]["path"]))
    )
    ablation_rows = list(
        protocol.iter_jsonl(Path(manifest["files"]["input_ablations"]["path"]))
    )
    real_anchored_record = manifest["files"].get(
        "real_anchored_counterfactuals"
    )
    if real_anchored_record is None:
        real_anchored_rows: list[dict[str, Any]] = []
        real_anchored_validation = {
            **real_anchored_counterfactual_checks(real_anchored_rows),
            "status": "not_present",
        }
    else:
        real_anchored_rows = list(
            protocol.iter_jsonl(Path(real_anchored_record["path"]))
        )
        raw_expected_real_anchored_count = real_anchored_record.get(
            "row_count"
        )
        expected_real_anchored_count = (
            raw_expected_real_anchored_count
            if isinstance(raw_expected_real_anchored_count, int)
            else None
        )
        real_anchored_validation = real_anchored_counterfactual_checks(
            real_anchored_rows,
            expected_row_count=expected_real_anchored_count,
        )
        if expected_real_anchored_count is None:
            real_anchored_validation.update(
                {
                    "accepted": False,
                    "expected_manifest_row_count": (
                        raw_expected_real_anchored_count
                    ),
                    "manifest_row_count_matches": False,
                    "manifest_row_count_declared_as_integer": False,
                }
            )
        else:
            real_anchored_validation[
                "manifest_row_count_declared_as_integer"
            ] = True
    identifiers = [str(row["sample_id"]) for row in clean_rows]
    duplicate_ids = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    clean_validation = validate_sample_collection(clean_rows)
    robust_validation = robustness_checks(clean_rows, robustness_rows)
    ablation_validation = input_ablation_checks(
        clean_rows,
        ablation_rows,
    )
    accepted = bool(
        not duplicate_ids
        and clean_validation["accepted"]
        and robust_validation["accepted"]
        and ablation_validation["accepted"]
        and real_anchored_validation["accepted"]
    )
    report = {
        "schema_version": "cafe.generation_validation.v2",
        "created_at": protocol.utc_now(),
        "dataset_id": dataset.dataset_id,
        "generation_manifest": str(manifest_path),
        "generation_manifest_sha256": protocol.file_sha256(manifest_path),
        "clean_sample_count": len(clean_rows),
        "robustness_sample_count": len(robustness_rows),
        "input_ablation_sample_count": len(ablation_rows),
        "real_anchored_sample_count": len(real_anchored_rows),
        "duplicate_sample_ids": duplicate_ids,
        "clean_validation": clean_validation,
        "robustness_validation": robust_validation,
        "input_ablation_validation": ablation_validation,
        "real_anchored_validation": real_anchored_validation,
        "mase_scale_audit": mase_scale_audit(clean_rows),
        "accepted": accepted,
    }
    report_path = generation_dir / f"validation__{shard_name}.json"
    protocol.write_json(report_path, report)
    if not accepted:
        raise ValueError(f"cafe generation validation failed: {report_path}")
    print(
        protocol.canonical_json(
            {
                "accepted": True,
                "clean_sample_count": len(clean_rows),
                "robustness_sample_count": len(robustness_rows),
                "input_ablation_sample_count": len(ablation_rows),
                "real_anchored_sample_count": len(real_anchored_rows),
                "report": str(report_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
