#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cafe import protocol
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_BACKGROUND_SCHEMA,
    REAL_ANCHORED_MASTER_SCHEMA,
    REAL_ANCHORED_SUPPORTED_CAPABILITIES,
    array_sha256,
    available_capabilities as available_real_anchored_capabilities,
    iter_nonlinear_replay_sensitivity_samples,
    iter_real_anchored_samples,
    validate_availability_contract,
    validate_contract_integrity,
)
from cafe.generation.real_anchored_policy import (
    NONLINEAR_FUTURE_INNOVATION_MAIN_POLICY,
    NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY,
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
    REAL_ANCHORED_CANONICAL_STRENGTH_GRID,
)
from cafe.generation.real_anchored_dose import (
    paired_minimum_separation_gate,
    validate_dose_calibration,
)
from cafe.generation.reference_bank import (
    validate_evaluation_qualification_policy,
    validate_real_anchored_reference_chain,
)
from cafe.generation.structural_real_counterfactuals import (
    FORMAL_PANEL_MINIMUM_DIMENSION,
    STRUCTURAL_ABLATION_SCHEMA,
    STRUCTURAL_BACKGROUND_SCHEMA,
    STRUCTURAL_CAPABILITY_ROW_SCHEMA,
    STRUCTURAL_CAPABILITIES,
    STRUCTURAL_DONOR_COMMITMENT_ENTRY_SCHEMA,
    STRUCTURAL_DONOR_COMMITMENT_POLICY,
    STRUCTURAL_DONOR_COMMITMENT_SCHEMA,
    STRUCTURAL_MASTER_SCHEMA,
    available_structural_capabilities,
    available_structural_sensitivity_capabilities,
    _array_sha256 as structural_array_sha256,
    iter_structural_real_anchored_samples,
    validate_structural_availability,
    validate_structural_contract,
    validate_structural_donor_commitment_manifest,
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


def _validated_row_dose_calibration(
    row: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a self-hash-valid v4 dose mapping, or ``None`` on failure."""

    calibration = row.get("dose_calibration")
    if not isinstance(calibration, dict):
        return None
    try:
        validate_dose_calibration(
            calibration,
            capability_id=str(row.get("capability_id", "")),
        )
    except (TypeError, ValueError):
        return None
    if calibration.get("status") != "available":
        return None
    return calibration


def _v4_dose_row_checks(row: dict[str, Any]) -> dict[str, bool]:
    """Validate canonical lambda versus capability-specific physical alpha."""

    calibration = _validated_row_dose_calibration(row)
    member = row.get("counterfactual_member")
    dose_index = row.get("dose_index")
    valid_index = bool(
        calibration is not None
        and isinstance(dose_index, int)
        and 1 <= dose_index <= len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
    )
    if valid_index:
        assert calibration is not None
        assert isinstance(dose_index, int)
        strength = float(calibration["strength_grid"][dose_index - 1])
        treatment_alpha = float(
            calibration["applied_alpha_grid"][dose_index - 1]
        )
    else:
        strength = math.nan
        treatment_alpha = math.nan
    member_strength = 0.0 if member == 0 else strength
    member_alpha = 1.0 if member == 0 else treatment_alpha
    intensity_calibration = row.get("intensity_calibration")
    sampled = row.get("sampled_generator_parameters")
    metadata = row.get("generation_metadata")
    calibration_hash = (
        None
        if calibration is None
        else calibration.get(
            "dose_policy_sha256",
            calibration.get("policy_sha256"),
        )
    )
    exposed_hashes = [
        value
        for value in (
            row.get("dose_calibration_policy_sha256"),
            (
                intensity_calibration.get("dose_calibration_policy_sha256")
                if isinstance(intensity_calibration, dict)
                else None
            ),
            (
                row.get("parameter_sampling", {}).get(
                    "dose_calibration_policy_sha256"
                )
                if isinstance(row.get("parameter_sampling"), dict)
                else None
            ),
            (
                metadata.get("dose_calibration_policy_sha256")
                if isinstance(metadata, dict)
                else None
            ),
        )
        if value is not None
    ]
    gate = row.get("paired_minimum_separation_gate")
    gate_declaration_valid = bool(
        isinstance(gate, dict)
        and (
            (
                member == 0
                and gate.get("status") == "not_applicable"
                and gate.get("accepted") is None
                and gate.get("reason_code")
                == "repeated_authentic_baseline_member"
                and gate.get("paired_treatment_gate_status") == "passed"
            )
            or (
                member == 1
                and gate.get("status") == "passed"
                and gate.get("accepted") is True
                and gate.get("reason_code") is None
            )
        )
        and gate.get("dose_index") == dose_index
        and gate.get("dose_calibration_policy_sha256") == calibration_hash
    )
    return {
        "dose_calibration_self_hash_valid": calibration is not None,
        "dose_index_valid": bool(
            valid_index and row.get("intensity") == dose_index
        ),
        "canonical_strength_grid_exact": bool(
            calibration is not None
            and calibration.get("strength_grid")
            == list(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
        ),
        "intensity_calibration_mapping_exact": bool(
            calibration is not None
            and isinstance(intensity_calibration, dict)
            and intensity_calibration.get("canonical_strength_grid")
            == calibration.get("strength_grid")
            and intensity_calibration.get("selected_alphas")
            == calibration.get("applied_alpha_grid")
            and intensity_calibration.get(
                "dose_calibration_policy_sha256"
            )
            == calibration_hash
        ),
        "dose_semantics_valid": bool(
            valid_index
            and member in (0, 1)
            and row.get("dose_parameter") == "canonical_strength_lambda"
            and row.get("physical_dose_parameter")
            == "controlled_component_multiplier_alpha"
            and _same_finite_float(row.get("baseline_dose_value"), 0.0)
            and _same_finite_float(row.get("dose_value"), member_strength)
            and _same_finite_float(
                row.get("intensity_lambda"), member_strength
            )
            and _same_finite_float(
                row.get("paired_treatment_strength"), strength
            )
            and _same_finite_float(row.get("applied_alpha"), member_alpha)
            and _same_finite_float(
                row.get("paired_treatment_applied_alpha"), treatment_alpha
            )
            and isinstance(sampled, dict)
            and _same_finite_float(sampled.get("alpha"), member_alpha)
            and (
                sampled.get("canonical_strength") is None
                or _same_finite_float(
                    sampled.get("canonical_strength"), member_strength
                )
            )
            and isinstance(metadata, dict)
            and _same_finite_float(metadata.get("alpha"), member_alpha)
            and (
                metadata.get("canonical_strength") is None
                or _same_finite_float(
                    metadata.get("canonical_strength"), member_strength
                )
            )
        ),
        "dose_calibration_hash_exposure_exact": bool(
            _is_sha256(calibration_hash)
            and exposed_hashes
            and all(value == calibration_hash for value in exposed_hashes)
        ),
        "paired_separation_gate_declared": gate_declaration_valid,
    }


def _affected_channel_indices_from_row(
    row: dict[str, Any],
    *,
    target_dim: int,
) -> list[int] | None:
    """Reconstruct the capability-declared channel mask used by the gate."""

    capability_id = str(row.get("capability_id", ""))
    metadata = row.get("generation_metadata")
    if not isinstance(metadata, dict) or target_dim < 1:
        return None
    if row.get("generator_family_role") == "real_anchored":
        return [0] if target_dim == 1 else None
    raw: Any
    if capability_id == "common_factor":
        loadings = metadata.get("response_loadings")
        if not isinstance(loadings, list) or len(loadings) != target_dim:
            return None
        try:
            absolute = np.abs(np.asarray(loadings, dtype=float))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(absolute).all() or float(np.max(absolute)) <= 0.0:
            return None
        raw = np.flatnonzero(
            absolute / float(np.max(absolute)) >= 0.25
        ).tolist()
    elif capability_id == "cross_series_dependence":
        raw = metadata.get("responder_indices")
    elif capability_id == "covariate_response":
        raw = metadata.get("eligible_target_indices")
    else:
        return None
    if not isinstance(raw, list):
        return None
    try:
        affected = [int(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if (
        not affected
        or len(affected) != len(set(affected))
        or min(affected) < 0
        or max(affected) >= target_dim
    ):
        return None
    return affected


def _numerically_equal_json(left: Any, right: Any) -> bool:
    """Compare persisted numerical evidence after JSON float round-tripping."""

    if isinstance(left, dict) and isinstance(right, dict):
        return bool(
            set(left) == set(right)
            and all(_numerically_equal_json(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return bool(
            len(left) == len(right)
            and all(
                _numerically_equal_json(a, b)
                for a, b in zip(left, right, strict=True)
            )
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return bool(
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and math.isclose(
                float(left),
                float(right),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    return left == right


def _paired_gate_replay_checks(
    ordered_pairs: list[dict[str, Any]],
) -> dict[str, bool]:
    """Replay absolute and adjacent L168/H48 gates from observed pair deltas."""

    if len(ordered_pairs) != len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID):
        return {
            "paired_minimum_separation_gate_exact": False,
            "absolute_minimum_separation_passed": False,
            "adjacent_minimum_separation_passed": False,
            "affected_channel_mask_exact": False,
        }
    previous_delta: np.ndarray | None = None
    exact = True
    absolute = True
    adjacent = True
    masks = True
    for pair in ordered_pairs:
        treatment = pair["treatment"]
        baseline = pair["baseline"]
        delta = np.asarray(pair["delta"], dtype=float)
        calibration = _validated_row_dose_calibration(treatment)
        baseline_calibration = _validated_row_dose_calibration(baseline)
        affected = _affected_channel_indices_from_row(
            treatment,
            target_dim=delta.shape[1],
        )
        treatment_metadata = treatment.get("generation_metadata")
        scale_by_channel = (
            treatment_metadata.get("normalization_scale_by_target")
            if treatment.get("generator_family_role") == "real_anchored"
            and isinstance(treatment_metadata, dict)
            else None
        )
        if (
            calibration is None
            or baseline_calibration != calibration
            or affected is None
        ):
            exact = absolute = adjacent = masks = False
            previous_delta = delta
            continue
        try:
            expected = paired_minimum_separation_gate(
                delta,
                context_length=int(treatment["context_length"]),
                dose_index=int(treatment["dose_index"]),
                dose_calibration=calibration,
                affected_channel_indices=affected,
                scale_by_channel=scale_by_channel,
                previous_delta=previous_delta,
            )
        except (KeyError, TypeError, ValueError):
            exact = absolute = adjacent = masks = False
            previous_delta = delta
            continue
        stored = treatment.get("paired_minimum_separation_gate")
        metadata = treatment.get("generation_metadata")
        metadata_gate = (
            metadata.get("paired_minimum_separation_gate")
            if isinstance(metadata, dict)
            else None
        )
        exact = bool(
            exact
            and _numerically_equal_json(stored, expected)
            and (
                metadata_gate is None
                or _numerically_equal_json(metadata_gate, expected)
            )
        )
        absolute = bool(
            absolute
            and expected.get("macro_passed") is True
            and expected.get("channel_coverage_passed") is True
            and expected.get("local_augmentation_budget_passed") is True
        )
        adjacent = bool(adjacent and expected.get("adjacent_accepted") is True)
        masks = bool(
            masks and expected.get("affected_channel_indices") == affected
        )
        previous_delta = delta
    return {
        "paired_minimum_separation_gate_exact": exact,
        "absolute_minimum_separation_passed": absolute,
        "adjacent_minimum_separation_passed": adjacent,
        "affected_channel_mask_exact": masks,
    }


def _paired_group_checks_pass(checks: dict[str, bool]) -> bool:
    """Require source-distance and budget gates; keep adjacency diagnostic."""

    return all(
        value
        for name, value in checks.items()
        if name != "adjacent_minimum_separation_passed"
    )


def _validate_reference_bank_chain(
    bundle_files: dict[str, Any],
    qualification_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = {
        "real_anchored_backgrounds",
        "structural_real_anchored_backgrounds",
        "real_anchored_reference_backgrounds",
        "structural_real_anchored_reference_backgrounds",
        "real_anchored_reference_contracts",
        "real_anchored_bank_split_audit",
    }
    missing = sorted(required - set(bundle_files))
    if missing:
        raise ValueError(
            "v4 calibration bundle lacks reference-bank evidence: "
            + ", ".join(missing)
        )

    def rows(key: str) -> list[dict[str, Any]]:
        return list(protocol.iter_jsonl(Path(bundle_files[key]["path"])))

    univariate_evaluation_backgrounds = rows("real_anchored_backgrounds")
    structural_evaluation_backgrounds = rows(
        "structural_real_anchored_backgrounds"
    )
    validate_real_anchored_reference_chain(
        [
            *univariate_evaluation_backgrounds,
            *structural_evaluation_backgrounds,
        ],
        [
            *rows("real_anchored_reference_backgrounds"),
            *rows("structural_real_anchored_reference_backgrounds"),
        ],
        protocol.read_json(
            Path(bundle_files["real_anchored_bank_split_audit"]["path"])
        ),
        qualification_policy,
        reference_contract_rows=rows("real_anchored_reference_contracts"),
    )
    return univariate_evaluation_backgrounds, structural_evaluation_backgrounds


def _validated_structural_donor_commitments(
    manifest: dict[str, Any],
    *,
    bundle: dict[str, Any],
    expected_bundle_hash: str,
    dataset_id: str,
) -> dict[str, dict[str, Any]]:
    """Load the calibration-bound donor sidecar and return trusted entries."""

    files = manifest.get("files")
    bundle_files = bundle.get("files")
    config = manifest.get("config")
    if not isinstance(files, dict) or not isinstance(bundle_files, dict):
        raise ValueError("structural donor commitment file maps are invalid")
    generation_record = files.get("structural_donor_commitments")
    calibration_record = bundle_files.get(
        "structural_real_anchored_donor_commitments"
    )
    if not isinstance(generation_record, dict) or not isinstance(
        calibration_record,
        dict,
    ):
        raise ValueError("structural donor commitment artifact is missing")
    for field in ("path", "sha256", "bytes"):
        if generation_record.get(field) != calibration_record.get(field):
            raise ValueError(
                "generation donor commitment is not the calibration artifact"
            )
    if generation_record.get("source_calibration_bundle_sha256") != (
        expected_bundle_hash
    ):
        raise ValueError("donor commitment lost calibration bundle binding")
    validate_manifest_file(generation_record)
    sidecar = protocol.read_json(Path(generation_record["path"]))
    if sidecar.get("schema_version") != STRUCTURAL_DONOR_COMMITMENT_SCHEMA:
        raise ValueError("structural donor commitment schema is invalid")
    if sidecar.get("commitment_policy") != STRUCTURAL_DONOR_COMMITMENT_POLICY:
        raise ValueError("structural donor commitment policy changed")
    if sidecar.get("dataset_id") != dataset_id:
        raise ValueError("structural donor commitment dataset mismatch")
    root_payload = dict(sidecar)
    observed_root = root_payload.pop("commitment_root_sha256", None)
    if observed_root != protocol.json_sha256(root_payload):
        raise ValueError("structural donor commitment root mismatch")
    entries = sidecar.get("entries")
    if not isinstance(entries, list):
        raise ValueError("structural donor commitment entries are missing")
    if sidecar.get("entry_count") != len(entries):
        raise ValueError("structural donor commitment entry count mismatch")
    if sidecar.get("entries_sha256") != protocol.json_sha256(entries):
        raise ValueError("structural donor commitment entries hash mismatch")
    sample_ids = sorted(str(entry.get("sample_id", "")) for entry in entries)
    if (
        not all(sample_ids)
        or len(sample_ids) != len(set(sample_ids))
        or sidecar.get("eligible_donor_sample_ids_sha256")
        != protocol.json_sha256(sample_ids)
    ):
        raise ValueError("structural donor commitment sample IDs are invalid")
    trusted: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("schema_version") != (
            STRUCTURAL_DONOR_COMMITMENT_ENTRY_SCHEMA
        ):
            raise ValueError("structural donor commitment entry is invalid")
        entry_payload = dict(entry)
        entry_hash = entry_payload.pop("entry_sha256", None)
        if entry_hash != protocol.json_sha256(entry_payload):
            raise ValueError("structural donor commitment entry hash mismatch")
        channel_hashes = entry.get("visible_history_by_channel_sha256")
        if (
            not _is_sha256(entry.get("visible_history_sha256"))
            or not isinstance(channel_hashes, dict)
            or set(channel_hashes)
            != {str(index) for index in range(int(entry.get("target_dim", -1)))}
            or not all(_is_sha256(value) for value in channel_hashes.values())
        ):
            raise ValueError("structural donor history commitment is invalid")
        trusted[str(entry["sample_id"])] = entry

    if not isinstance(config, dict):
        raise ValueError("generation config is missing for donor commitment")
    real_config = config.get("real_anchored_counterfactual")
    declaration = (
        real_config.get("structural_donor_commitment")
        if isinstance(real_config, dict)
        else None
    )
    expected_declaration = {
        "schema_version": STRUCTURAL_DONOR_COMMITMENT_SCHEMA,
        "commitment_policy": STRUCTURAL_DONOR_COMMITMENT_POLICY,
        "commitment_root_sha256": observed_root,
        "source_calibration_bundle_sha256": expected_bundle_hash,
        "source_file_sha256": calibration_record.get("sha256"),
    }
    if declaration != expected_declaration:
        raise ValueError("generation config donor commitment binding mismatch")
    if generation_record.get("commitment_root_sha256") != observed_root:
        raise ValueError("generation manifest donor commitment root mismatch")

    background_record = bundle_files.get(
        "structural_real_anchored_backgrounds"
    )
    contract_record = bundle_files.get("structural_real_anchored_contracts")
    if not isinstance(background_record, dict) or not isinstance(
        contract_record,
        dict,
    ):
        raise ValueError(
            "calibration bundle lacks structural donor source banks"
        )
    validate_structural_donor_commitment_manifest(
        sidecar,
        list(protocol.iter_jsonl(Path(background_record["path"]))),
        list(protocol.iter_jsonl(Path(contract_record["path"]))),
        dataset_id=dataset_id,
    )
    return trusted


def _current_v4_row_replay_evidence(
    *,
    bundle_files: dict[str, Any],
    config: dict[str, Any],
    dataset_id: str,
    seed_indexes: list[int],
) -> dict[str, Any]:
    """Recreate every declared v4 main row from bundle-bound calibration."""

    required = {
        "real_anchored_backgrounds",
        "real_anchored_contracts",
        "real_anchored_availability",
        "structural_real_anchored_backgrounds",
        "structural_real_anchored_contracts",
        "structural_real_anchored_availability",
    }
    missing = sorted(required - set(bundle_files))
    if missing:
        raise ValueError(
            "v4 calibration bundle lacks row replay evidence: "
            + ", ".join(missing)
        )

    real_backgrounds = list(
        protocol.iter_jsonl(
            Path(bundle_files["real_anchored_backgrounds"]["path"])
        )
    )
    real_contracts = list(
        protocol.iter_jsonl(
            Path(bundle_files["real_anchored_contracts"]["path"])
        )
    )
    real_availability = protocol.read_json(
        Path(bundle_files["real_anchored_availability"]["path"])
    )
    structural_backgrounds = list(
        protocol.iter_jsonl(
            Path(
                bundle_files["structural_real_anchored_backgrounds"]["path"]
            )
        )
    )
    structural_contracts = list(
        protocol.iter_jsonl(
            Path(bundle_files["structural_real_anchored_contracts"]["path"])
        )
    )
    structural_availability = protocol.read_json(
        Path(bundle_files["structural_real_anchored_availability"]["path"])
    )

    def unique_backgrounds(
        rows: list[dict[str, Any]],
        *,
        schema: str,
        label: str,
    ) -> dict[str, dict[str, Any]]:
        mapped: dict[str, dict[str, Any]] = {}
        for row in rows:
            background_id = str(row.get("background_id", ""))
            if (
                row.get("schema_version") != schema
                or row.get("dataset_id") != dataset_id
                or not background_id
                or background_id in mapped
            ):
                raise ValueError(f"{label} background bank identity is invalid")
            mapped[background_id] = row
        return mapped

    real_by_id = unique_backgrounds(
        real_backgrounds,
        schema=REAL_ANCHORED_BACKGROUND_SCHEMA,
        label="univariate",
    )
    structural_by_id = unique_backgrounds(
        structural_backgrounds,
        schema=STRUCTURAL_BACKGROUND_SCHEMA,
        label="structural",
    )

    def raw_float64_sha256(values: np.ndarray) -> str:
        return hashlib.sha256(
            np.asarray(values, dtype="<f8").tobytes(order="C")
        ).hexdigest()

    for background in real_backgrounds:
        try:
            target = np.asarray(background["target"], dtype=float)
            prefix = np.asarray(
                background["decomposition_prefix"],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "univariate replay background payload is invalid"
            ) from error
        if target.shape == (protocol.REAL_ANCHORED_MASTER_LENGTH, 1):
            target_1d = target[:, 0]
        elif target.shape == (protocol.REAL_ANCHORED_MASTER_LENGTH,):
            target_1d = target
        else:
            raise ValueError("univariate replay background target shape is invalid")
        prefix_length = (
            protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
            - protocol.REAL_ANCHORED_CONTEXT_LENGTH
        )
        if (
            prefix.shape != (prefix_length,)
            or not np.isfinite(prefix).all()
            or not np.isfinite(target_1d).all()
            or background.get("decomposition_prefix_sha256")
            != array_sha256(prefix)
        ):
            raise ValueError("univariate replay background prefix is invalid")
        decomposition_history = np.concatenate(
            (
                prefix,
                target_1d[: protocol.REAL_ANCHORED_CONTEXT_LENGTH],
            )
        )
        expected_hashes = {
            "decomposition_history_sha256": raw_float64_sha256(
                decomposition_history
            ),
            "history_sha256": raw_float64_sha256(
                target_1d[: protocol.REAL_ANCHORED_CONTEXT_LENGTH]
            ),
            "future_sha256": raw_float64_sha256(
                target_1d[protocol.REAL_ANCHORED_CONTEXT_LENGTH :]
            ),
            "target_sha256": raw_float64_sha256(target_1d),
        }
        if any(
            background.get(field) != expected
            for field, expected in expected_hashes.items()
        ):
            raise ValueError("univariate replay background hash is invalid")

    for background in structural_backgrounds:
        try:
            target = np.asarray(background["target"], dtype=float)
            target_dim = int(background["target_dim"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "structural replay background payload is invalid"
            ) from error
        expected_shape = (
            protocol.REAL_ANCHORED_MASTER_LENGTH,
            target_dim,
        )
        if (
            target_dim < 1
            or target.shape != expected_shape
            or not np.isfinite(target).all()
            or background.get("target_sha256")
            != structural_array_sha256(
                target,
                domain="structural_visible_target",
            )
            or background.get("future_sha256")
            != structural_array_sha256(
                target[protocol.REAL_ANCHORED_CONTEXT_LENGTH :],
                domain="structural_real_future",
            )
        ):
            raise ValueError("structural replay background target is invalid")
        covariates = background.get("known_future_covariates")
        if isinstance(covariates, dict):
            try:
                covariate_target = np.asarray(
                    covariates["target"],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "structural replay covariate payload is invalid"
                ) from error
            if (
                covariates.get("kind") != "known_future"
                or covariate_target.ndim != 2
                or covariate_target.shape[0]
                != protocol.REAL_ANCHORED_MASTER_LENGTH
                or covariate_target.shape[1] < 1
                or not np.isfinite(covariate_target).all()
                or covariates.get("target_sha256")
                != structural_array_sha256(
                    covariate_target,
                    domain="structural_known_future_visible",
                )
            ):
                raise ValueError(
                    "structural replay covariate payload is invalid"
                )

    real_contract_identities: set[tuple[str, str]] = set()
    for row in real_contracts:
        background_id = str(row.get("background_id", ""))
        capability_id = str(row.get("capability_id", ""))
        background = real_by_id.get(background_id)
        identity = (capability_id, background_id)
        if (
            background is None
            or row.get("schema_version")
            != "cafe.real_anchored_background_capability.v4"
            or row.get("dataset_id") != dataset_id
            or capability_id not in REAL_ANCHORED_SUPPORTED_CAPABILITIES
            or identity in real_contract_identities
            or row.get("source_history_sha256")
            != background.get("decomposition_history_sha256")
        ):
            raise ValueError("univariate contract/background binding is invalid")
        real_contract_identities.add(identity)
        validate_contract_integrity(row)

    structural_contract_identities: set[tuple[str, str]] = set()
    for row in structural_contracts:
        background_id = str(row.get("background_id", ""))
        capability_id = str(row.get("capability_id", ""))
        background = structural_by_id.get(background_id)
        contract = row.get("contract")
        identity = (capability_id, background_id)
        if (
            background is None
            or row.get("schema_version") != STRUCTURAL_CAPABILITY_ROW_SCHEMA
            or row.get("dataset_id") != dataset_id
            or capability_id not in STRUCTURAL_CAPABILITIES
            or identity in structural_contract_identities
            or not isinstance(contract, dict)
            or contract.get("capability_id") != capability_id
            or contract.get("background_id") != background_id
        ):
            if row.get("generation_eligible") is True:
                raise ValueError(
                    "eligible structural contract/background binding is invalid"
                )
            continue
        structural_contract_identities.add(identity)
        validate_structural_contract(contract, background)
    validate_availability_contract(real_availability, real_contracts)
    validate_structural_availability(
        structural_availability,
        structural_contracts,
    )

    requested = config.get("requested_capabilities")
    real_config = config.get("real_anchored_counterfactual")
    if (
        not isinstance(requested, list)
        or not all(isinstance(value, str) for value in requested)
        or len(requested) != len(set(requested))
        or not isinstance(real_config, dict)
    ):
        raise ValueError("v4 generation capability request is invalid")
    available_real = tuple(
        capability_id
        for capability_id in available_real_anchored_capabilities(
            real_availability
        )
        if capability_id in requested
    )
    available_structural = tuple(
        capability_id
        for capability_id in requested
        if capability_id in available_structural_capabilities(
            structural_availability
        )
        and capability_id != "hierarchical_coherence"
    )
    sensitivity_fields = (
        "structural_sensitivity_capabilities",
        "structural_sensitivity_main_count",
        "structural_sensitivity_input_ablation_count",
        "nonlinear_replay_sensitivity_count",
    )
    if not all(
        field in real_config for field in sensitivity_fields
    ):
        raise ValueError(
            "v4 auxiliary sensitivity declaration is incomplete"
        )
    raw_sensitivity_capabilities = real_config.get(
        "structural_sensitivity_capabilities",
        [],
    )
    if (
        not isinstance(raw_sensitivity_capabilities, list)
        or not all(
            isinstance(value, str)
            for value in raw_sensitivity_capabilities
        )
        or len(raw_sensitivity_capabilities)
        != len(set(raw_sensitivity_capabilities))
    ):
        raise ValueError("v4 structural sensitivity capabilities are invalid")
    sensitivity_capabilities = tuple(raw_sensitivity_capabilities)
    available_sensitivity = set(
        available_structural_sensitivity_capabilities(
            structural_availability
        )
    )
    expected_sensitivity_capabilities = tuple(
        capability_id
        for capability_id in requested
        if capability_id in {"common_factor", "cross_series_dependence"}
        and capability_id in available_sensitivity
    )
    sensitivity_available = {
        str(row.get("capability_id"))
        for row in structural_contracts
        if row.get("sensitivity_available") is True
    }
    if sensitivity_capabilities != expected_sensitivity_capabilities or any(
        capability_id not in {"common_factor", "cross_series_dependence"}
        or capability_id not in requested
        or capability_id not in available_sensitivity
        or capability_id not in sensitivity_available
        for capability_id in sensitivity_capabilities
    ):
        raise ValueError(
            "v4 structural sensitivity capability is not contract-eligible"
        )
    if real_config.get("calibrated_available_capabilities") != list(
        available_real
    ):
        raise ValueError("v4 calibrated capability declaration is not replayable")

    real_expected = list(
        iter_real_anchored_samples(
            real_backgrounds,
            real_contracts,
            capability_ids=available_real,
            seed_indexes=seed_indexes,
        )
    )
    nonlinear_replay_expected = (
        list(
            iter_nonlinear_replay_sensitivity_samples(
                real_backgrounds,
                real_contracts,
                seed_indexes=seed_indexes,
            )
        )
        if "nonlinear_persistence" in available_real
        else []
    )
    structural_expected = list(
        iter_structural_real_anchored_samples(
            structural_backgrounds,
            [
                row
                for row in structural_contracts
                if row.get("capability_id") in available_structural
            ],
            seed_indexes=seed_indexes,
        )
    )
    structural_sensitivity_expected = list(
        iter_structural_real_anchored_samples(
            structural_backgrounds,
            [
                row
                for row in structural_contracts
                if row.get("capability_id") in sensitivity_capabilities
                and row.get("sensitivity_available") is True
            ],
            sensitivity=True,
            seed_indexes=seed_indexes,
        )
    )
    for row in structural_sensitivity_expected:
        row["excluded_from_primary_score"] = True
    generated_real = tuple(
        capability_id
        for capability_id in available_real
        if any(
            row.get("capability_id") == capability_id
            for row in real_expected
        )
    )
    generated_structural = tuple(
        capability_id
        for capability_id in available_structural
        if any(
            row.get("capability_id") == capability_id
            for row in structural_expected
        )
    )
    expected_generated = [*generated_real, *generated_structural]
    if real_config.get("generated_capabilities") != expected_generated:
        raise ValueError("v4 generated capability declaration is not replayable")
    if real_config.get("structural_main_count") != len(structural_expected):
        raise ValueError("v4 structural main count is not replayable")
    if real_config.get("nonlinear_replay_sensitivity_count", 0) != len(
        nonlinear_replay_expected
    ):
        raise ValueError("v4 nonlinear replay count is not replayable")
    if (
        real_config.get("structural_sensitivity_main_count")
        != len(structural_sensitivity_expected)
        or real_config.get("structural_sensitivity_input_ablation_count")
        != len(structural_sensitivity_expected)
    ):
        raise ValueError("v4 structural sensitivity count is not replayable")

    def by_sample_id(
        rows: list[dict[str, Any]],
        *,
        label: str,
    ) -> dict[str, dict[str, Any]]:
        mapped = {str(row.get("sample_id", "")): row for row in rows}
        if "" in mapped or len(mapped) != len(rows):
            raise ValueError(f"{label} replay produced duplicate sample IDs")
        return mapped

    return {
        "schema_version": "cafe.real_anchored_row_replay_evidence.v2",
        "dataset_id": dataset_id,
        "seed_indexes": list(seed_indexes),
        "univariate_expected_rows": by_sample_id(
            real_expected,
            label="univariate",
        ),
        "nonlinear_replay_expected_rows": by_sample_id(
            nonlinear_replay_expected,
            label="nonlinear replay sensitivity",
        ),
        "structural_expected_rows": by_sample_id(
            structural_expected,
            label="structural",
        ),
        "structural_sensitivity_expected_rows": by_sample_id(
            structural_sensitivity_expected,
            label="structural sensitivity",
        ),
    }


def validate_generation_manifest_contract(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    calibration_dir: Path,
    dataset_id: str,
    seed_start: int,
    seed_count: int,
    replay_evidence_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate generation identity and current upstream bindings before reads."""

    if replay_evidence_out is not None:
        replay_evidence_out.clear()
    schema = manifest.get("schema_version")
    schema_pairs = {
        "cafe.generation_manifest.v1": "cafe.generation_config.v1",
        "cafe.generation_manifest.v2": "cafe.generation_config.v2",
        "cafe.generation_manifest.v3": "cafe.generation_config.v3",
        "cafe.generation_manifest.v4": "cafe.generation_config.v4",
        "cafe.generation_manifest.v5": "cafe.generation_config.v5",
    }
    if schema not in schema_pairs:
        raise ValueError(f"unsupported generation manifest schema: {schema!r}")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("generation manifest config must be an object")
    if config.get("schema_version") != schema_pairs[schema]:
        raise ValueError(
            "generation config schema does not match manifest schema"
        )
    if manifest.get("config_sha256") != protocol.json_sha256(config):
        raise ValueError("generation manifest config hash mismatch")

    expected_seed_indexes = list(range(seed_start, seed_start + seed_count))
    identity = {
        "dataset_id": dataset_id,
        "seed_start": seed_start,
        "seed_count": seed_count,
    }
    for field, expected in identity.items():
        if config.get(field) != expected:
            raise ValueError(
                f"generation config {field} disagrees with validation CLI"
            )
        if field in manifest and manifest[field] != expected:
            raise ValueError(
                f"generation manifest {field} disagrees with validation CLI"
            )
    if config.get("seed_indexes") != expected_seed_indexes:
        raise ValueError("generation config seed indexes disagree with shard")
    expected_name = (
        f"manifest__seed_{seed_start:06d}_"
        f"{seed_start + seed_count:06d}.json"
    )
    if manifest_path.name != expected_name:
        raise ValueError("generation manifest filename disagrees with shard")
    if not _is_sha256(config.get("calibration_bundle_sha256")):
        raise ValueError("generation config calibration bundle hash is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("generation manifest files must be an object")
    required_base_files = {"clean", "robustness", "input_ablations"}
    missing_base = sorted(required_base_files - set(files))
    if missing_base:
        raise ValueError(
            "generation manifest is missing required files: "
            + ", ".join(missing_base)
        )
    if schema in {
        "cafe.generation_manifest.v1",
        "cafe.generation_manifest.v2",
    }:
        return config

    # Pre-v5 manifests remain readable only with an empty real-anchor
    # component. Their earlier dose semantics are never regenerated or ranked.
    if schema in {
        "cafe.generation_manifest.v3",
        "cafe.generation_manifest.v4",
    }:
        real_config = config.get("real_anchored_counterfactual")
        real_record = files.get("real_anchored_counterfactuals")
        generated = (
            real_config.get("generated_capabilities")
            if isinstance(real_config, dict)
            else None
        )
        counts = (
            [
                real_config.get(field, 0)
                for field in (
                    "structural_main_count",
                    "structural_input_ablation_count",
                    "structural_sensitivity_main_count",
                    "structural_sensitivity_input_ablation_count",
                    "nonlinear_replay_sensitivity_count",
                )
            ]
            if isinstance(real_config, dict)
            else []
        )
        if (
            not isinstance(real_config, dict)
            or generated != []
            or any(value != 0 for value in counts)
            or (
                real_record is not None
                and (
                    not isinstance(real_record, dict)
                    or real_record.get("row_count") != 0
                )
            )
            or real_config.get("included_in_synthetic_ranking") is not False
        ):
            raise ValueError(
                "pre-v5 real-anchored rows are legacy and cannot enter v5 "
                "validation"
            )
        return config

    required_v4_files = {
        "real_anchored_counterfactuals",
        "real_anchored_availability",
        "structural_real_anchored_availability",
    }
    missing_v4 = sorted(required_v4_files - set(files))
    if missing_v4:
        raise ValueError(
            "v4 generation manifest is missing bound artifacts: "
            + ", ".join(missing_v4)
        )
    real_config = config.get("real_anchored_counterfactual")
    if not isinstance(real_config, dict):
        raise ValueError("v4 generation config lacks real-anchored contract")
    upstream_pipeline_schema = real_config.get(
        "upstream_real_anchored_protocol"
    )
    supported_upstream_schemas = {
        "cafe.pipeline.v1",
        "cafe.pipeline.v2",
        "cafe.pipeline.v3",
        "cafe.pipeline.v4",
        protocol.SCHEMA_VERSION,
    }
    if upstream_pipeline_schema not in supported_upstream_schemas:
        raise ValueError("v4 generation config has unsupported upstream protocol")
    if real_config.get("included_in_synthetic_ranking") is not False:
        raise ValueError("real-anchored track entered the synthetic ranking")
    if real_config.get("formal_panel_minimum_dimension") != 3:
        raise ValueError("v4 generation config changed the formal panel minimum")
    if real_config.get("hierarchy_policy") != (
        "qualification_only_zero_generation_rows"
    ):
        raise ValueError("v4 generation config changed hierarchy policy")
    if real_config.get("paired_minimum_separation") != (
        "mandatory_treatment_source_l168_distance_with_budget_v1"
    ):
        raise ValueError(
            "v4 generation config changed paired minimum-separation policy"
        )

    bundle_path = calibration_dir / "calibration_bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    bundle = protocol.read_json(bundle_path)
    expected_bundle_schema = {
        "cafe.pipeline.v1": "cafe.calibration_bundle.v1",
        "cafe.pipeline.v2": "cafe.calibration_bundle.v2",
        "cafe.pipeline.v3": "cafe.calibration_bundle.v3",
        "cafe.pipeline.v4": "cafe.calibration_bundle.v4",
        protocol.SCHEMA_VERSION: "cafe.calibration_bundle.v5",
    }[upstream_pipeline_schema]
    if bundle.get("schema_version") != expected_bundle_schema:
        raise ValueError(
            "calibration bundle schema does not match generation upstream"
        )
    if bundle.get("pipeline_schema_version") != upstream_pipeline_schema:
        raise ValueError(
            "calibration bundle pipeline schema does not match generation upstream"
        )
    try:
        expected_bundle_hash = protocol.json_sha256(
            {
                "dataset": bundle["dataset"],
                "source": bundle["source"],
                "files": bundle["files"],
                "generator_version": bundle["generator_version"],
            }
        )
    except KeyError as error:
        raise ValueError(
            "calibration bundle lacks content-hash fields"
        ) from error
    if bundle.get("bundle_content_sha256") != expected_bundle_hash:
        raise ValueError("calibration bundle content hash mismatch")
    if config.get("calibration_bundle_sha256") != expected_bundle_hash:
        raise ValueError("generation config is not bound to calibration bundle")
    bundle_dataset = bundle.get("dataset")
    if not isinstance(bundle_dataset, dict) or bundle_dataset.get(
        "dataset_id"
    ) != dataset_id:
        raise ValueError("calibration bundle dataset binding mismatch")

    bundle_files = bundle.get("files")
    if not isinstance(bundle_files, dict):
        raise ValueError("calibration bundle files must be an object")
    for record in bundle_files.values():
        if not isinstance(record, dict):
            raise ValueError("calibration bundle has an invalid file record")
        validate_manifest_file(record)

    for key in (
        "real_anchored_availability",
        "structural_real_anchored_availability",
    ):
        record = files.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"generation manifest has invalid {key} record")
        validate_manifest_file(record)
    real_availability = protocol.read_json(
        Path(files["real_anchored_availability"]["path"])
    )
    structural_availability = protocol.read_json(
        Path(files["structural_real_anchored_availability"]["path"])
    )
    for name, availability in (
        ("real", real_availability),
        ("structural", structural_availability),
    ):
        if availability.get("source_calibration_bundle_sha256") != (
            expected_bundle_hash
        ):
            raise ValueError(
                f"{name} availability lost calibration bundle binding"
            )
        if availability.get("requested_seed_indexes") != expected_seed_indexes:
            raise ValueError(f"{name} availability seed binding mismatch")
    generated_capabilities = real_config.get("generated_capabilities")
    if not isinstance(generated_capabilities, list) or (
        real_availability.get("generated_capabilities")
        != generated_capabilities
    ):
        raise ValueError("real availability capability binding mismatch")
    if real_config.get("nonlinear_replay_sensitivity_count", 0) != (
        real_availability.get(
            "generated_nonlinear_replay_sensitivity_count",
            0,
        )
    ):
        raise ValueError("nonlinear replay availability binding mismatch")
    structural_generated = structural_availability.get(
        "generated_capabilities"
    )
    if not isinstance(structural_generated, list) or not set(
        structural_generated
    ).issubset(set(generated_capabilities)):
        raise ValueError("structural availability capability binding mismatch")
    real_record = files["real_anchored_counterfactuals"]
    real_row_count = real_record.get("row_count")
    if not isinstance(real_row_count, int) or real_row_count < 0:
        raise ValueError("v4 real-anchored row count is invalid")
    if real_availability.get("generated_master_count") != real_row_count:
        raise ValueError("real availability row-count binding mismatch")
    if real_config.get("structural_main_count") != (
        structural_availability.get("generated_main_master_count")
    ):
        raise ValueError("structural main count binding mismatch")
    if real_config.get("structural_input_ablation_count") != (
        structural_availability.get("generated_input_ablation_master_count")
    ):
        raise ValueError("structural ablation count binding mismatch")
    sensitivity_binding_fields = {
        "structural_sensitivity_capabilities": (
            "generated_sensitivity_capabilities"
        ),
        "structural_sensitivity_main_count": (
            "generated_sensitivity_main_master_count"
        ),
        "structural_sensitivity_input_ablation_count": (
            "generated_sensitivity_input_ablation_master_count"
        ),
    }
    if any(field in real_config for field in sensitivity_binding_fields):
        if any(
            real_config.get(config_field)
            != structural_availability.get(availability_field)
            for config_field, availability_field in (
                sensitivity_binding_fields.items()
            )
        ):
            raise ValueError(
                "structural sensitivity availability binding mismatch"
            )
    if real_availability.get("hierarchical_coherence_generation_count") != 0 or (
        structural_availability.get("hierarchical_coherence_generation_count")
        != 0
    ):
        raise ValueError("hierarchy rows were declared in v4 availability")

    if upstream_pipeline_schema in {
        "cafe.pipeline.v1",
        "cafe.pipeline.v2",
        "cafe.pipeline.v3",
    }:
        if files.get("structural_donor_commitments") is not None:
            raise ValueError("legacy generation declared donor commitments")
        if real_config.get("structural_donor_commitment") is not None:
            raise ValueError("legacy generation config declared donor commitments")
        if real_config.get("qualification_policy_sha256") is not None:
            raise ValueError("legacy generation config declared a v3 qualification")
        if real_config.get("qualification_threshold_source") is not None:
            raise ValueError(
                "legacy generation config declared a v3 threshold source"
            )
        if real_config.get("legacy_upstream_component_policy") != (
            "validated_but_not_regenerated_or_ranked_as_v3"
        ):
            raise ValueError("legacy generation component policy changed")
        if generated_capabilities != [] or structural_generated != []:
            raise ValueError("legacy upstream generated real-anchored capabilities")
        if real_row_count != 0 or real_availability.get(
            "generated_master_count"
        ) != 0:
            raise ValueError("legacy upstream generated real-anchored rows")
        if real_config.get("structural_main_count") != 0 or real_config.get(
            "structural_input_ablation_count"
        ) != 0:
            raise ValueError("legacy upstream generated structural rows")
        if structural_availability.get("generated_main_master_count") != 0 or (
            structural_availability.get(
                "generated_input_ablation_master_count"
            )
            != 0
        ):
            raise ValueError("legacy structural availability declared rows")
        if structural_availability.get(
            "frozen_qualification_policy_sha256"
        ) is not None:
            raise ValueError(
                "legacy structural availability declared a v3 qualification"
            )
        return config

    qualification_hash = real_config.get("qualification_policy_sha256")
    if not _is_sha256(qualification_hash):
        raise ValueError("v4 generation config qualification hash is invalid")
    if real_config.get("qualification_threshold_source") != (
        QUALIFICATION_THRESHOLD_SOURCE_POLICY
    ):
        raise ValueError("v4 generation qualification threshold source changed")
    if real_config.get("legacy_upstream_component_policy") is not None:
        raise ValueError("v4 generation config unexpectedly declares legacy mode")
    policy_record = bundle_files.get("real_anchored_qualification_policy")
    if not isinstance(policy_record, dict):
        raise ValueError("calibration bundle lacks qualification policy file")
    qualification_policy = protocol.read_json(Path(policy_record["path"]))
    if qualification_policy.get("qualification_policy_sha256") != (
        qualification_hash
    ):
        raise ValueError("generation config qualification policy mismatch")
    if bundle.get("real_anchored_qualification_policy_sha256") != (
        qualification_hash
    ):
        raise ValueError("calibration bundle qualification policy mismatch")
    if structural_availability.get(
        "frozen_qualification_policy_sha256"
    ) != qualification_hash:
        raise ValueError("structural availability qualification binding mismatch")
    capability_cells = qualification_policy.get("capabilities")
    dose_policy = qualification_policy.get("dose_policy")
    if not isinstance(capability_cells, dict) or not isinstance(
        dose_policy, dict
    ):
        raise ValueError("v4 qualification policy lacks frozen dose mappings")
    expected_dose_hashes: dict[str, str] = {}
    expected_alpha_grids: dict[str, list[float]] = {}
    for capability_id, cell in sorted(capability_cells.items()):
        calibration = (
            cell.get("dose_calibration")
            if isinstance(cell, dict)
            else None
        )
        if not isinstance(calibration, dict):
            raise ValueError("v4 qualification capability lacks dose mapping")
        validate_dose_calibration(
            calibration,
            capability_id=str(capability_id),
        )
        if calibration.get("status") == "available":
            expected_dose_hashes[str(capability_id)] = str(
                calibration["policy_sha256"]
            )
            expected_alpha_grids[str(capability_id)] = [
                float(value)
                for value in calibration["applied_alpha_grid"]
            ]
    if (
        real_config.get("dose_parameter") != "canonical_strength_lambda"
        or real_config.get("canonical_strength_grid")
        != list(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
        or real_config.get("dose_policy_sha256")
        != dose_policy.get("dose_policy_sha256")
        or real_config.get("dose_calibration_sha256_by_capability")
        != expected_dose_hashes
        or real_config.get("applied_alpha_grid_by_capability")
        != expected_alpha_grids
        or real_config.get("applied_alpha_scope")
        != "contract_specific_history_only"
    ):
        raise ValueError("v4 generation dose-policy binding mismatch")
    _validate_reference_bank_chain(bundle_files, qualification_policy)
    evaluation_contracts = [
        *protocol.iter_jsonl(
            Path(bundle_files["real_anchored_contracts"]["path"])
        ),
        *protocol.iter_jsonl(
            Path(
                bundle_files["structural_real_anchored_contracts"]["path"]
            )
        ),
    ]
    generated_dose_capabilities = set(
        real_config.get("generated_capabilities", [])
    ) | set(real_config.get("structural_sensitivity_capabilities", []))
    resolved_grids: dict[str, list[list[float]]] = {}
    for row in evaluation_contracts:
        capability_id = str(row.get("capability_id", ""))
        if capability_id not in generated_dose_capabilities:
            continue
        calibration = row.get("dose_calibration")
        if not isinstance(calibration, dict):
            contract = row.get("contract")
            if isinstance(contract, dict):
                calibration = contract.get("dose_calibration")
        if not isinstance(calibration, dict):
            continue
        grid = [
            float(value)
            for value in calibration.get("applied_alpha_grid", [])
        ]
        if len(grid) == len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID):
            resolved_grids.setdefault(capability_id, []).append(grid)
    expected_alpha_ranges = {
        capability_id: [
            {
                "minimum": min(grid[index] for grid in grids),
                "maximum": max(grid[index] for grid in grids),
            }
            for index in range(len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID))
        ]
        for capability_id, grids in sorted(resolved_grids.items())
        if grids
    }
    if real_config.get("applied_alpha_range_by_capability") != (
        expected_alpha_ranges
    ):
        raise ValueError("generation contract-specific alpha ranges mismatch")
    validate_evaluation_qualification_policy(
        evaluation_contracts,
        qualification_policy,
    )
    _validated_structural_donor_commitments(
        manifest,
        bundle=bundle,
        expected_bundle_hash=expected_bundle_hash,
        dataset_id=dataset_id,
    )
    replay_evidence = _current_v4_row_replay_evidence(
        bundle_files=bundle_files,
        config=config,
        dataset_id=dataset_id,
        seed_indexes=expected_seed_indexes,
    )
    expected_main_count = len(
        replay_evidence["univariate_expected_rows"]
    ) + len(replay_evidence["structural_expected_rows"]) + len(
        replay_evidence["structural_sensitivity_expected_rows"]
    ) + len(
        replay_evidence["nonlinear_replay_expected_rows"]
    )
    structural_ablation_count = real_config.get(
        "structural_input_ablation_count"
    )
    sensitivity_ablation_count = real_config.get(
        "structural_sensitivity_input_ablation_count",
        0,
    )
    if (
        not isinstance(structural_ablation_count, int)
        or not isinstance(sensitivity_ablation_count, int)
        or expected_main_count
        + structural_ablation_count
        + sensitivity_ablation_count
        != real_row_count
    ):
        raise ValueError("v4 real-anchored row count is not replayable")
    if replay_evidence_out is not None:
        replay_evidence_out.update(replay_evidence)
    return config


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


def _upstream_main_row_replay_checks(
    row: dict[str, Any],
    *,
    target: np.ndarray | None,
    expected: dict[str, Any] | None,
) -> dict[str, bool]:
    """Compare a main row with the calibration-bound deterministic replay."""

    if expected is None:
        return {
            "upstream_replay_sample_exists": False,
            "upstream_background_assignment_exact": False,
            "upstream_contract_binding_exact": False,
            "upstream_baseline_target_exact": False,
            "upstream_treatment_target_replay_exact": False,
            "upstream_full_row_replay_exact": False,
        }
    try:
        expected_target = np.asarray(expected.get("target"), dtype=float)
    except (TypeError, ValueError):
        expected_target = np.empty((0, 0), dtype=float)
    targets_exact = bool(
        target is not None
        and target.shape == expected_target.shape
        and np.array_equal(target, expected_target)
    )
    metadata = row.get("generation_metadata")
    expected_metadata = expected.get("generation_metadata")
    contract_fields = (
        "contract_sha256",
        "capability_contract_sha256",
        "source_history_sha256",
        "decomposition_history_sha256",
    )
    expected_contract_fields = (
        tuple(
            field
            for field in contract_fields
            if isinstance(expected_metadata, dict)
            and field in expected_metadata
        )
    )
    contract_exact = bool(
        expected_contract_fields
        and isinstance(metadata, dict)
        and isinstance(expected_metadata, dict)
        and row.get("parameter_sampling")
        == expected.get("parameter_sampling")
        and all(
            metadata.get(field) == expected_metadata.get(field)
            for field in expected_contract_fields
        )
    )
    member = row.get("counterfactual_member")
    return {
        "upstream_replay_sample_exists": True,
        "upstream_background_assignment_exact": bool(
            row.get("dataset_id") == expected.get("dataset_id")
            and row.get("capability_id") == expected.get("capability_id")
            and row.get("seed_index") == expected.get("seed_index")
            and row.get("background_id") == expected.get("background_id")
            and row.get("anchor_id") == expected.get("anchor_id")
        ),
        "upstream_contract_binding_exact": contract_exact,
        "upstream_baseline_target_exact": bool(
            member != 0 or targets_exact
        ),
        "upstream_treatment_target_replay_exact": bool(
            member != 1 or targets_exact
        ),
        "upstream_full_row_replay_exact": row == expected,
    }


def _upstream_replay_coverage_failures(
    rows: list[dict[str, Any]],
    expected_rows: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if expected_rows is None:
        return []
    observed = {str(row.get("sample_id", "")) for row in rows}
    expected = set(expected_rows)
    failures: list[dict[str, Any]] = []
    if expected - observed:
        failures.append(
            {
                "reason": "missing_calibration_replay_rows",
                "sample_ids": sorted(expected - observed),
            }
        )
    if observed - expected:
        failures.append(
            {
                "reason": "unexpected_rows_without_calibration_replay",
                "sample_ids": sorted(observed - expected),
            }
        )
    return failures


def _univariate_real_anchored_counterfactual_checks(
    rows: list[dict[str, Any]],
    *,
    expected_row_count: int | None = None,
    expected_replay_rows: dict[str, dict[str, Any]] | None = None,
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
        expected_replay = (
            None
            if expected_replay_rows is None
            else expected_replay_rows.get(sample_id)
        )

        member_valid = isinstance(member, int) and member in (0, 1)
        current_schema = row.get("schema_version") == REAL_ANCHORED_MASTER_SCHEMA
        v4_dose_checks = (
            _v4_dose_row_checks(row) if current_schema else {}
        )
        dose_index = row.get("dose_index")
        dose_index_valid = bool(
            isinstance(dose_index, int)
            and dose_index >= 1
            and (
                not current_schema
                or dose_index
                <= len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
            )
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
            and (
                not current_schema
                or (
                    _validated_row_dose_calibration(row) is not None
                    and alpha_values
                    == _validated_row_dose_calibration(row).get(
                        "applied_alpha_grid"
                    )
                )
            )
        )
        alpha = (
            row.get("applied_alpha")
            if current_schema
            else row.get("dose_value")
        )
        exposed_alphas = [
            alpha,
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
            in {
                "cafe.real_anchored_counterfactual_master.v1",
                REAL_ANCHORED_MASTER_SCHEMA,
            },
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
                and (
                    v4_dose_checks.get("dose_index_valid", False)
                    if current_schema
                    else row.get("dose_parameter") == "alpha"
                    and _same_finite_float(
                        row.get("baseline_dose_value"), 1.0
                    )
                )
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
        if current_schema:
            checks.update(v4_dose_checks)
        if expected_replay_rows is not None:
            checks.update(
                _upstream_main_row_replay_checks(
                    row,
                    target=target if target_valid else None,
                    expected=expected_replay,
                )
            )
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
        "paired_treatment_strength",
        "paired_treatment_applied_alpha",
        "dose_calibration_policy_sha256",
        "dose_calibration",
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
        "dose_response_law",
        "future_innovation_policy",
        "future_innovation_sha256",
        "history_innovation_policy",
        "history_innovation_sha256",
        "history_residual_replay_policy",
        "history_residual_replay_sensitivity_alpha2_rms",
        "dose_response_qualification",
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
                                (
                                    0.0
                                    if baseline.get("schema_version")
                                    == REAL_ANCHORED_MASTER_SCHEMA
                                    else 1.0
                                ),
                            )
                            and (
                                baseline.get("schema_version")
                                != REAL_ANCHORED_MASTER_SCHEMA
                                or _same_finite_float(
                                    baseline.get("applied_alpha"), 1.0
                                )
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
        "dose_calibration_policy_sha256",
        "dose_calibration",
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
                float(
                    pair["treatment"].get(
                        "applied_alpha",
                        pair["treatment"]["dose_value"],
                    )
                )
                for pair in ordered_pairs
            ]
            treatment_strengths = [
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
            treatment_strengths = []
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
        capability_id = (
            str(group_rows[0].get("capability_id", ""))
            if group_rows
            else ""
        )
        nonlinear_dynamic = capability_id == "nonlinear_persistence"
        current_group_schema = any(
            row.get("schema_version") == REAL_ANCHORED_MASTER_SCHEMA
            for row in group_rows
        )
        nonlinear_metadata = [
            pair["treatment"].get("generation_metadata", {})
            for pair in ordered_pairs
        ]
        nonlinear_expected_rms: list[float] = []
        nonlinear_expected_future_rms: list[float] = []
        nonlinear_contract_fields_valid = bool(nonlinear_metadata)
        if nonlinear_dynamic:
            for alpha, metadata in zip(
                treatment_alphas,
                nonlinear_metadata,
                strict=False,
            ):
                qualification = metadata.get("dose_response_qualification")
                selected = (
                    next(
                        (
                            row
                            for row in qualification
                            if _same_finite_float(row.get("alpha"), alpha)
                        ),
                        None,
                    )
                    if isinstance(qualification, list)
                    else None
                )
                nonlinear_contract_fields_valid = bool(
                    nonlinear_contract_fields_valid
                    and metadata.get("dose_response_law")
                    == "dynamic_recursive_nonproportional"
                    and metadata.get("future_innovation_policy")
                    == "zero_future_innovation_paired_rollout_v1"
                    and metadata.get("history_innovation_policy")
                    == "shared_observed_one_step_innovations"
                    and metadata.get("dynamic_contract_replay_verified") is True
                    and _is_sha256(metadata.get("history_innovation_sha256"))
                    and _is_sha256(metadata.get("future_innovation_sha256"))
                    and metadata.get("future_component_source")
                    == "paired_zero_innovation_dynamic_rollout"
                    and metadata.get("history_residual_replay_policy")
                    == "history_residual_replay_qualification_only_v1"
                    and _finite_float(
                        metadata.get(
                            "history_residual_replay_sensitivity_alpha2_rms"
                        )
                    )
                    and float(
                        metadata[
                            "history_residual_replay_sensitivity_alpha2_rms"
                        ]
                    )
                    >= 0.0
                    and isinstance(selected, dict)
                    and _finite_float(selected.get("intervention_rms"))
                    and _finite_float(selected.get("future_effect_rms"))
                )
                if isinstance(selected, dict):
                    nonlinear_expected_rms.append(
                        float(selected["intervention_rms"])
                    )
                    nonlinear_expected_future_rms.append(
                        float(selected["future_effect_rms"])
                    )
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
            "frozen_treatment_grid_exact": bool(
                not current_group_schema
                or (
                    len(group_rows)
                    == 2 * len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
                    and treatment_strengths
                    == list(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
                    and _validated_row_dose_calibration(group_rows[0])
                    is not None
                    and selected_alphas
                    == _validated_row_dose_calibration(group_rows[0]).get(
                        "applied_alpha_grid"
                    )
                )
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
                nonlinear_dynamic
                or (
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
                )
            ),
            "metadata_rms_is_linear_in_alpha_minus_one": bool(
                nonlinear_dynamic
                or (
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
                )
            ),
            "nonlinear_dynamic_contract_valid": bool(
                not nonlinear_dynamic
                or (
                    nonlinear_contract_fields_valid
                    and len(nonlinear_expected_rms) == len(delta_rms)
                    and all(
                        math.isclose(
                            observed,
                            expected,
                            rel_tol=1e-10,
                            abs_tol=1e-12,
                        )
                        for observed, expected in zip(
                            delta_rms,
                            nonlinear_expected_rms,
                            strict=False,
                        )
                    )
                    and all(
                        math.isclose(
                            float(metadata.get("future_effect_rms")),
                            expected,
                            rel_tol=1e-10,
                            abs_tol=1e-12,
                        )
                        for metadata, expected in zip(
                            nonlinear_metadata,
                            nonlinear_expected_future_rms,
                            strict=False,
                        )
                    )
                    and all(
                        math.isclose(
                            float(metadata.get("future_effect_rms")),
                            float(
                                np.sqrt(
                                    np.mean(
                                        pair["delta"][
                                            protocol.REAL_ANCHORED_CONTEXT_LENGTH :
                                        ]
                                        ** 2
                                    )
                                )
                            ),
                            rel_tol=1e-10,
                            abs_tol=1e-12,
                        )
                        for pair, metadata in zip(
                            ordered_pairs,
                            nonlinear_metadata,
                            strict=False,
                        )
                    )
                )
            ),
        }
        if current_group_schema:
            checks.update(_paired_gate_replay_checks(ordered_pairs))
        if not _paired_group_checks_pass(checks):
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
    upstream_replay_coverage_failures = _upstream_replay_coverage_failures(
        rows,
        expected_replay_rows,
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
        and not upstream_replay_coverage_failures
    )
    return {
        "schema_version": "cafe.real_anchored_validation.v5",
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
        "upstream_replay_coverage_failures": (
            upstream_replay_coverage_failures
        ),
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


def _finite_matrix(value: Any) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        return None
    return matrix


def _nonlinear_replay_sensitivity_checks(
    rows: list[dict[str, Any]],
    *,
    source_main_rows: list[dict[str, Any]],
    expected_replay_rows: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Validate the excluded history-residual-replay auxiliary track."""

    expected_length = (
        protocol.REAL_ANCHORED_CONTEXT_LENGTH + protocol.HORIZON
    )
    context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
    sources = {
        str(row.get("sample_id", "")): row for row in source_main_rows
    }
    row_failures: list[dict[str, Any]] = []
    pair_failures: list[dict[str, Any]] = []
    group_failures: list[dict[str, Any]] = []
    targets: dict[str, np.ndarray] = {}
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        source_id = str(row.get("sensitivity_source_sample_id", ""))
        source = sources.get(source_id)
        pair_id = str(row.get("counterfactual_pair_id", ""))
        group_id = str(row.get("paired_group_id", ""))
        member = row.get("counterfactual_member")
        member_valid = isinstance(member, int) and member in (0, 1)
        target = _finite_matrix(row.get("target"))
        target_valid = bool(
            target is not None and target.shape == (expected_length, 1)
        )
        if target_valid:
            assert target is not None
            targets[sample_id] = target
        source_target = (
            None
            if source is None
            else _finite_matrix(source.get("target"))
        )
        metadata = row.get("generation_metadata")
        source_metadata = (
            source.get("generation_metadata")
            if isinstance(source, dict)
            else None
        )
        expected = (
            None
            if expected_replay_rows is None
            else expected_replay_rows.get(sample_id)
        )
        calibration = row.get("intensity_calibration")
        selected_alphas = (
            calibration.get("selected_alphas")
            if isinstance(calibration, dict)
            else None
        )
        v4_dose_checks = _v4_dose_row_checks(row)
        source_identity_exact = bool(
            source is not None
            and source.get("evaluation_table")
            == "real_anchored_counterfactual"
            and source.get("capability_id") == "nonlinear_persistence"
            and row.get("sensitivity_source_pair_id")
            == source.get("counterfactual_pair_id")
            and row.get("sensitivity_source_paired_group_id")
            == source.get("paired_group_id")
            and pair_id
            == f"{source.get('counterfactual_pair_id')}__nonlinear_replay"
            and group_id
            == f"{source.get('paired_group_id')}__nonlinear_replay"
            and sample_id == f"{source_id}__nonlinear_replay"
            and all(
                row.get(field) == source.get(field)
                for field in (
                    "dataset_id",
                    "capability_id",
                    "seed_index",
                    "dose_index",
                    "counterfactual_member",
                    "background_id",
                    "anchor_id",
                    "parameter_sampling",
                    "intensity_calibration",
                    "dose_calibration",
                    "dose_calibration_policy_sha256",
                    "dose_value",
                    "intensity_lambda",
                    "paired_treatment_strength",
                    "applied_alpha",
                    "paired_treatment_applied_alpha",
                )
            )
        )
        source_contract_exact = bool(
            isinstance(metadata, dict)
            and isinstance(source_metadata, dict)
            and all(
                metadata.get(field) == source_metadata.get(field)
                for field in (
                    "contract_sha256",
                    "capability_contract_sha256",
                    "source_history_sha256",
                    "decomposition_history_sha256",
                    "history_innovation_policy",
                    "history_innovation_sha256",
                )
            )
            and source_metadata.get("future_innovation_policy")
            == NONLINEAR_FUTURE_INNOVATION_MAIN_POLICY
        )
        checks = {
            "schema_valid": row.get("schema_version")
            == REAL_ANCHORED_MASTER_SCHEMA,
            "track_valid": bool(
                row.get("benchmark_track")
                == "real_anchored_counterfactual"
                and row.get("evaluation_table")
                == "real_anchored_nonlinear_replay_sensitivity"
                and row.get("generator_family_role") == "real_anchored"
                and row.get("capability_id") == "nonlinear_persistence"
            ),
            "excluded_from_primary_score": row.get(
                "excluded_from_primary_score"
            )
            is True,
            "source_main_binding_exact": source_identity_exact,
            "source_contract_binding_exact": source_contract_exact,
            "sample_identity_valid": bool(
                source_identity_exact
                and row.get("master_sample_id") == sample_id
                and row.get("baseline_sample_id")
                == f"{row.get('sensitivity_source_pair_id')}__m0__nonlinear_replay"
            ),
            "member_valid": member_valid,
            "fixed_shape_valid": bool(
                target_valid
                and row.get("context_length") == context
                and row.get("horizon") == protocol.HORIZON
                and row.get("target_dim") == 1
                and row.get("covariate_dim") == 0
                and row.get("covariates") is None
            ),
            "target_hash_matches": bool(
                target_valid
                and target is not None
                and row.get("target_sha256")
                == protocol.target_and_covariate_sha256(target, None)
            ),
            "future_hash_matches": bool(
                target_valid
                and target is not None
                and row.get("future_sha256") == array_sha256(target[context:])
            ),
            "history_matches_zero_innovation_source": bool(
                target_valid
                and target is not None
                and source_target is not None
                and source_target.shape == target.shape
                and np.array_equal(target[:context], source_target[:context])
            ),
            "residual_replay_policy_valid": bool(
                isinstance(metadata, dict)
                and metadata.get("future_innovation_policy")
                == NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
                and metadata.get("history_residual_replay_policy")
                == NONLINEAR_FUTURE_INNOVATION_SENSITIVITY_POLICY
                and metadata.get("sensitivity_role")
                == "history_residual_replay_auxiliary"
                and _is_sha256(metadata.get("future_innovation_sha256"))
            ),
            "frozen_treatment_grid_exact": selected_alphas
            == (
                _validated_row_dose_calibration(row) or {}
            ).get("applied_alpha_grid"),
            "dose_and_alpha_valid": bool(
                member_valid and all(v4_dose_checks.values())
            ),
            **_upstream_main_row_replay_checks(
                row,
                target=target if target_valid else None,
                expected=expected,
            ),
        }
        checks.update(v4_dose_checks)
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

    complete_pairs: dict[str, dict[str, Any]] = {}
    for pair_id, pair_rows in sorted(by_pair.items()):
        members = [row.get("counterfactual_member") for row in pair_rows]
        complete = bool(
            len(pair_rows) == 2
            and members.count(0) == 1
            and members.count(1) == 1
        )
        checks = {
            "exactly_two_members": complete,
            "pair_fields_shared": _exact_shared(
                pair_rows,
                (
                    "counterfactual_pair_id",
                    "paired_group_id",
                    "baseline_sample_id",
                    "sensitivity_source_pair_id",
                    "sensitivity_source_paired_group_id",
                    "dataset_id",
                    "capability_id",
                    "seed_index",
                    "dose_index",
                    "background_id",
                    "anchor_id",
                    "intensity_calibration",
                    "dose_calibration",
                    "dose_calibration_policy_sha256",
                    "paired_treatment_strength",
                    "paired_treatment_applied_alpha",
                    "excluded_from_primary_score",
                ),
            ),
        }
        if complete:
            baseline = next(
                row for row in pair_rows if row["counterfactual_member"] == 0
            )
            treatment = next(
                row for row in pair_rows if row["counterfactual_member"] == 1
            )
            baseline_target = targets.get(str(baseline.get("sample_id", "")))
            treatment_target = targets.get(str(treatment.get("sample_id", "")))
            if baseline_target is None or treatment_target is None:
                checks["pair_targets_valid"] = False
            else:
                delta = treatment_target - baseline_target
                delta_rms = float(np.sqrt(np.mean(delta**2)))
                checks.update(
                    {
                        "baseline_member_exact": bool(
                            _same_finite_float(baseline.get("dose_value"), 0.0)
                            and _same_finite_float(
                                baseline.get("applied_alpha"), 1.0
                            )
                            and baseline.get("intervention_delta_sha256")
                            == array_sha256(np.zeros_like(baseline_target))
                        ),
                        "treatment_delta_hash_exact": treatment.get(
                            "intervention_delta_sha256"
                        )
                        == array_sha256(delta),
                        "treatment_delta_nonzero": delta_rms > 0.0,
                        "target_feature_matches_pair_delta": bool(
                            _finite_float(treatment.get("target_feature_value"))
                            and _same_finite_float(
                                treatment.get("target_feature_value"),
                                treatment.get(
                                    "intensity_target_feature_value"
                                ),
                                treatment.get("generation_metadata", {}).get(
                                    "intervention_rms"
                                ),
                            )
                            and math.isclose(
                                float(treatment["target_feature_value"]),
                                delta_rms,
                                rel_tol=1e-10,
                                abs_tol=1e-12,
                            )
                        ),
                    }
                )
                complete_pairs[pair_id] = {
                    "baseline": baseline,
                    "treatment": treatment,
                    "baseline_target": baseline_target,
                    "delta": delta,
                }
        if not all(checks.values()):
            pair_failures.append(
                {"counterfactual_pair_id": pair_id, "checks": checks}
            )

    for group_id, group_rows in sorted(by_group.items()):
        pairs = sorted(
            (
                pair
                for pair in complete_pairs.values()
                if pair["baseline"].get("paired_group_id") == group_id
            ),
            key=lambda pair: int(pair["treatment"].get("dose_index", -1)),
        )
        dose_indexes = [
            int(pair["treatment"].get("dose_index", -1)) for pair in pairs
        ]
        treatment_alphas = [
            float(pair["treatment"].get("applied_alpha", math.nan))
            for pair in pairs
        ]
        treatment_strengths = [
            float(pair["treatment"].get("dose_value", math.nan))
            for pair in pairs
        ]
        baselines = [pair["baseline_target"] for pair in pairs]
        checks = {
            "complete_frozen_grid": bool(
                len(group_rows)
                == 2 * len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
                and len(pairs)
                == len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
                and dose_indexes
                == list(
                    range(
                        1,
                        len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID) + 1,
                    )
                )
                and treatment_strengths
                == list(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
                and _validated_row_dose_calibration(group_rows[0])
                is not None
                and treatment_alphas
                == _validated_row_dose_calibration(group_rows[0]).get(
                    "applied_alpha_grid"
                )
            ),
            "duplicate_baselines_are_exact": bool(
                baselines
                and all(
                    np.array_equal(values, baselines[0])
                    for values in baselines[1:]
                )
            ),
            "one_zero_innovation_source_group": bool(
                len(
                    {
                        row.get("sensitivity_source_paired_group_id")
                        for row in group_rows
                    }
                )
                == 1
            ),
        }
        checks.update(_paired_gate_replay_checks(pairs))
        if not _paired_group_checks_pass(checks):
            group_failures.append(
                {"paired_group_id": group_id, "checks": checks}
            )

    coverage_failures = _upstream_replay_coverage_failures(
        rows,
        expected_replay_rows,
    )
    accepted = not any(
        (row_failures, pair_failures, group_failures, coverage_failures)
    )
    return {
        "accepted": accepted,
        "sample_count": len(rows),
        "pair_count": len(by_pair),
        "paired_group_count": len(by_group),
        "row_failures": row_failures,
        "pair_failures": pair_failures,
        "paired_group_failures": group_failures,
        "upstream_replay_coverage_failures": coverage_failures,
        "effective_background_count_by_capability": {},
    }


def _structural_mase_reference_valid(
    row: dict[str, Any],
    *,
    target_dim: int,
) -> bool:
    scales = row.get("mase_scale_by_target")
    effective_periods = row.get("mase_scale_effective_period_by_target")
    fallback_indices = row.get("mase_scale_fallback_target_indices")
    period = row.get("mase_period")
    if not (
        isinstance(scales, list)
        and len(scales) == target_dim
        and scales
        and all(_finite_float(value) and float(value) > 0.0 for value in scales)
        and _finite_float(row.get("mase_scale"))
        and math.isclose(
            float(row["mase_scale"]),
            float(np.mean(np.asarray(scales, dtype=float))),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and isinstance(period, int)
        and 1 <= period < protocol.REAL_ANCHORED_CONTEXT_LENGTH
        and isinstance(effective_periods, list)
        and len(effective_periods) == target_dim
        and all(
            isinstance(value, int) and value in {1, period}
            for value in effective_periods
        )
        and isinstance(fallback_indices, list)
        and len(fallback_indices) == len(set(fallback_indices))
        and all(
            isinstance(index, int)
            and 0 <= index < target_dim
            and effective_periods[index] == 1
            for index in fallback_indices
        )
        and isinstance(row.get("mase_scale_policy"), str)
        and bool(row["mase_scale_policy"])
        and row.get("mase_scale_source")
        == "shared_unmodified_real_l336_history"
    ):
        return False
    return True


def _structural_standardization_valid(
    row: dict[str, Any],
    *,
    target_dim: int,
) -> bool:
    standardization = row.get("shared_standardization")
    if not isinstance(standardization, dict):
        return False
    centers = standardization.get("center_by_target")
    scales = standardization.get("scale_by_target")
    return bool(
        standardization.get("scope")
        == "shared_unmodified_real_l336_history"
        and isinstance(centers, list)
        and len(centers) == target_dim
        and all(_finite_float(value) for value in centers)
        and isinstance(scales, list)
        and len(scales) == target_dim
        and all(_finite_float(value) and float(value) > 0.0 for value in scales)
        and standardization.get("member_specific") is False
    )


def _structural_real_anchored_checks(
    main_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    *,
    donor_commitment_entries: dict[str, dict[str, Any]] | None = None,
    donor_commitment_root_sha256: str | None = None,
    expected_replay_rows: dict[str, dict[str, Any]] | None = None,
    expected_sensitivity_replay_rows: (
        dict[str, dict[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Validate formal and explicitly excluded structural auxiliary rows."""

    expected_length = (
        protocol.REAL_ANCHORED_CONTEXT_LENGTH + protocol.HORIZON
    )
    row_failures: list[dict[str, Any]] = []
    pair_failures: list[dict[str, Any]] = []
    group_failures: list[dict[str, Any]] = []
    ablation_failures: list[dict[str, Any]] = []
    targets: dict[str, np.ndarray] = {}
    covariates: dict[str, np.ndarray | None] = {}
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    structural_capabilities = {
        "common_factor",
        "cross_series_dependence",
        "covariate_response",
    }
    mandatory_ablation_capabilities = {
        "common_factor",
        "cross_series_dependence",
    }
    committed_successor_by_sample: dict[str, str] = {}
    if donor_commitment_entries is not None:
        commitment_cells: dict[tuple[Any, ...], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for entry in donor_commitment_entries.values():
            key = (
                str(entry.get("dataset_id", "")),
                str(entry.get("capability_id", "")),
                int(entry.get("dose_index", -1)),
                int(entry.get("counterfactual_member", -1)),
                int(entry.get("target_dim", -1)),
                str(entry.get("evaluation_table", "")),
            )
            commitment_cells[key].append(entry)
        for entries in commitment_cells.values():
            ordered = sorted(
                entries,
                key=lambda entry: (
                    int(entry["seed_index"]),
                    str(entry["background_id"]),
                    str(entry["sample_id"]),
                ),
            )
            if len({str(entry["background_id"]) for entry in ordered}) < 2:
                continue
            by_background = {
                str(entry["background_id"]): index
                for index, entry in enumerate(ordered)
            }
            for entry in ordered:
                index = by_background[str(entry["background_id"])]
                committed_successor_by_sample[str(entry["sample_id"])] = str(
                    ordered[(index + 1) % len(ordered)]["sample_id"]
                )

    for row_index, row in enumerate(main_rows):
        sample_id = str(row.get("sample_id", ""))
        pair_id = str(row.get("counterfactual_pair_id", ""))
        group_id = str(row.get("paired_group_id", ""))
        capability_id = str(row.get("capability_id", ""))
        evaluation_table = row.get("evaluation_table")
        sensitivity_row = (
            evaluation_table == "real_anchored_structural_sensitivity"
        )
        member = row.get("counterfactual_member")
        target_dim = row.get("target_dim")
        covariate_dim = row.get("covariate_dim")
        target = _finite_matrix(row.get("target"))
        raw_covariates = row.get("covariates")
        covariate = (
            None
            if raw_covariates is None
            else _finite_matrix(raw_covariates)
        )
        target_valid = bool(
            isinstance(target_dim, int)
            and target_dim >= 1
            and target is not None
            and target.shape == (expected_length, target_dim)
        )
        covariate_valid = bool(
            isinstance(covariate_dim, int)
            and covariate_dim >= 0
            and (
                (covariate_dim == 0 and raw_covariates is None)
                or (
                    covariate_dim > 0
                    and covariate is not None
                    and covariate.shape == (expected_length, covariate_dim)
                )
            )
        )
        if target_valid:
            assert target is not None
            targets[sample_id] = target
        if covariate_valid:
            covariates[sample_id] = covariate
        expected_replay = (
            None
            if (
                expected_sensitivity_replay_rows is None
                if sensitivity_row
                else expected_replay_rows is None
            )
            else (
                expected_sensitivity_replay_rows.get(sample_id)
                if sensitivity_row
                else expected_replay_rows.get(sample_id)
            )
        )

        metadata = row.get("generation_metadata")
        parameter_sampling = row.get("parameter_sampling")
        sampled_parameters = row.get("sampled_generator_parameters")
        standardization_valid = bool(
            isinstance(target_dim, int)
            and _structural_standardization_valid(
                row,
                target_dim=target_dim,
            )
        )
        mase_valid = bool(
            isinstance(target_dim, int)
            and _structural_mase_reference_valid(
                row,
                target_dim=target_dim,
            )
        )
        background_id = str(row.get("background_id", ""))
        contract_hash = (
            metadata.get("contract_sha256")
            if isinstance(metadata, dict)
            else None
        )
        contract_valid = bool(
            isinstance(metadata, dict)
            and metadata.get("capability_id") == capability_id
            and background_id
            and row.get("anchor_id") == background_id
            and isinstance(parameter_sampling, dict)
            and parameter_sampling.get("background_id") == background_id
            and parameter_sampling.get("contract_sha256") == contract_hash
            and _is_sha256(contract_hash)
            and metadata.get("target_future_used_for_delta") is False
        )
        truth_delta = (
            _finite_matrix(metadata.get("truth_delta"))
            if isinstance(metadata, dict)
            else None
        )
        truth_delta_valid = bool(
            target_valid
            and truth_delta is not None
            and target is not None
            and truth_delta.shape == target.shape
            and _is_sha256(metadata.get("truth_delta_sha256"))
        )
        member_valid = isinstance(member, int) and member in (0, 1)
        v4_dose_checks = _v4_dose_row_checks(row)
        alpha = row.get("applied_alpha")
        exposed_alphas = (
            alpha,
            metadata.get("alpha") if isinstance(metadata, dict) else None,
            (
                sampled_parameters.get("alpha")
                if isinstance(sampled_parameters, dict)
                else None
            ),
        )
        expected_alpha = bool(
            member_valid
            and _same_finite_float(*exposed_alphas)
            and (
                _same_finite_float(alpha, 1.0)
                if member == 0
                else _finite_float(alpha) and float(alpha) > 1.0
            )
        )
        calibration = row.get("intensity_calibration")
        selected_alphas = (
            calibration.get("selected_alphas")
            if isinstance(calibration, dict)
            else None
        )
        alpha_grid_valid = bool(
            isinstance(selected_alphas, list)
            and selected_alphas
            and all(
                _finite_float(value) and float(value) > 1.0
                for value in selected_alphas
            )
            and all(
                float(right) > float(left)
                for left, right in zip(
                    selected_alphas,
                    selected_alphas[1:],
                )
            )
            and _validated_row_dose_calibration(row) is not None
            and selected_alphas
            == _validated_row_dose_calibration(row).get(
                "applied_alpha_grid"
            )
        )
        panel_dimension_valid = bool(
            isinstance(target_dim, int)
            and (
                target_dim == 2
                if sensitivity_row
                else (
                    target_dim >= FORMAL_PANEL_MINIMUM_DIMENSION
                    if capability_id in mandatory_ablation_capabilities
                    else target_dim >= 1
                )
            )
        )
        sensitivity_contract_eligible = bool(
            not sensitivity_row
            or (
                capability_id in mandatory_ablation_capabilities
                and expected_replay is not None
                and expected_replay.get("evaluation_table")
                == "real_anchored_structural_sensitivity"
                and expected_replay.get("target_dim") == 2
            )
        )
        covariate_names = row.get("covariate_column_names")
        eligible_target_indices = (
            metadata.get("eligible_target_indices")
            if isinstance(metadata, dict)
            else None
        )
        known_future_covariate_contract_valid = bool(
            capability_id != "covariate_response"
            or (
                isinstance(covariate_dim, int)
                and covariate_dim > 0
                and covariate_valid
                and covariate is not None
                and isinstance(covariate_names, list)
                and len(covariate_names) == covariate_dim
                and all(
                    isinstance(name, str) and bool(name)
                    for name in covariate_names
                )
                and isinstance(metadata, dict)
                and metadata.get(
                    "known_future_covariate_path_used_for_delta"
                )
                is True
                and metadata.get("target_future_used_for_delta") is False
                and metadata.get("controlled_component")
                == "known_future_covariate_predictive_response"
                and isinstance(eligible_target_indices, list)
                and bool(eligible_target_indices)
                and isinstance(target_dim, int)
                and all(
                    isinstance(index, int) and 0 <= index < target_dim
                    for index in eligible_target_indices
                )
            )
        )
        mandatory = row.get("mandatory_input_ablation")
        mandatory_valid = capability_id not in mandatory_ablation_capabilities
        if capability_id in mandatory_ablation_capabilities:
            assessed = (
                mandatory.get("assessed_target_indices")
                if isinstance(mandatory, dict)
                else None
            )
            ablated = (
                mandatory.get("ablated_input_indices")
                if isinstance(mandatory, dict)
                else None
            )
            mandatory_valid = bool(
                isinstance(target_dim, int)
                and isinstance(mandatory, dict)
                and mandatory.get("required") is True
                and mandatory.get("evaluation_table")
                == "real_anchored_input_ablation"
                and mandatory.get("target_future_unchanged") is True
                and mandatory.get("excluded_from_primary_score") is True
                and mandatory.get(
                    "reported_as_separate_attribution_audit"
                )
                is True
                and isinstance(assessed, list)
                and assessed
                and isinstance(ablated, list)
                and ablated
                and not (set(assessed) & set(ablated))
                and all(
                    isinstance(index, int) and 0 <= index < target_dim
                    for index in (*assessed, *ablated)
                )
            )

        target_hash_matches = bool(
            target_valid
            and covariate_valid
            and target is not None
            and row.get("target_sha256")
            == protocol.target_and_covariate_sha256(target, covariate)
        )
        future_hash_matches = bool(
            target_valid
            and target is not None
            and row.get("future_sha256")
            == array_sha256(
                target[protocol.REAL_ANCHORED_CONTEXT_LENGTH :]
            )
        )
        checks = {
            "schema_valid": row.get("schema_version")
            == STRUCTURAL_MASTER_SCHEMA,
            "track_valid": bool(
                row.get("benchmark_track")
                == "real_anchored_counterfactual"
                and evaluation_table
                == (
                    "real_anchored_structural_sensitivity"
                    if sensitivity_row
                    else "real_anchored_counterfactual"
                )
                and row.get("generator_family_role")
                == "real_anchored_structural"
            ),
            "formal_capability_valid": capability_id
            in structural_capabilities,
            "hierarchy_formal_row_prohibited": capability_id
            != "hierarchical_coherence",
            "formal_dimension_valid_for_capability": panel_dimension_valid,
            "formal_panel_dimension_valid": panel_dimension_valid,
            "sensitivity_contract_eligible": (
                sensitivity_contract_eligible
            ),
            "known_future_covariate_contract_valid": (
                known_future_covariate_contract_valid
            ),
            "sample_id_valid": bool(
                sample_id
                and member_valid
                and sample_id == f"{pair_id}__m{member}"
                and row.get("master_sample_id") == sample_id
                and row.get("baseline_sample_id") == f"{pair_id}__m0"
            ),
            "pair_id_valid": bool(pair_id and group_id),
            "member_valid": member_valid,
            "fixed_shape_valid": bool(
                row.get("context_length")
                == protocol.REAL_ANCHORED_CONTEXT_LENGTH
                and row.get("horizon") == protocol.HORIZON
                and target_valid
                and covariate_valid
            ),
            "target_hash_matches": target_hash_matches,
            "future_hash_matches": future_hash_matches,
            "alpha_exposure_consistent": expected_alpha,
            "dose_index_valid": bool(
                isinstance(row.get("dose_index"), int)
                and 1
                <= int(row["dose_index"])
                <= len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
                and row.get("intensity") == row.get("dose_index")
                and v4_dose_checks.get("dose_index_valid", False)
            ),
            "alpha_grid_valid": alpha_grid_valid,
            "normalization_reference_valid": standardization_valid,
            "mase_reference_valid": mase_valid,
            "contract_and_background_valid": contract_valid,
            "truth_delta_contract_valid": truth_delta_valid,
            "mandatory_input_ablation_declared": mandatory_valid,
            "primary_score_policy_valid": bool(
                row.get("excluded_from_primary_score") is True
                if sensitivity_row
                else row.get("excluded_from_primary_score") is not True
            ),
            "anti_copy_explicitly_not_applicable": row.get("anti_copy_gate")
            == {
                "status": "not_applicable",
                "reason_code": "intentional_real_anchor_counterfactual",
            },
        }
        checks.update(v4_dose_checks)
        if (
            sensitivity_row
            or expected_replay_rows is not None
            or expected_sensitivity_replay_rows is not None
        ):
            checks.update(
                _upstream_main_row_replay_checks(
                    row,
                    target=target if target_valid else None,
                    expected=expected_replay,
                )
            )
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
        "paired_treatment_strength",
        "paired_treatment_applied_alpha",
        "background_id",
        "anchor_id",
        "target_dim",
        "covariate_dim",
        "covariates",
        "shared_standardization",
        "mase_period",
        "mase_scale",
        "mase_scale_by_target",
        "mase_scale_effective_period_by_target",
        "mase_scale_fallback_target_indices",
        "mase_scale_policy",
        "mase_scale_source",
        "dose_calibration_policy_sha256",
        "dose_calibration",
        "parameter_sampling",
        "intensity_calibration",
        "mandatory_input_ablation",
        "evaluation_table",
        "excluded_from_primary_score",
    )
    metadata_shared_fields = (
        "capability_id",
        "contract_sha256",
        "controlled_component",
        "target_future_used_for_delta",
        "known_future_covariate_path_used_for_delta",
        "mandatory_input_ablation",
        "protected_target_index",
        "response_loadings",
        "driver_index",
        "responder_indices",
        "cross_lag_steps",
        "eligible_target_indices",
    )
    for pair_id, pair_rows in sorted(by_pair.items()):
        members = [row.get("counterfactual_member") for row in pair_rows]
        complete = bool(
            len(pair_rows) == 2
            and members.count(0) == 1
            and members.count(1) == 1
        )
        checks: dict[str, bool] = {
            "exactly_two_members": complete,
            "pair_fields_shared": _exact_shared(
                pair_rows,
                pair_shared_fields,
            ),
            "metadata_references_shared": _metadata_exact_shared(
                pair_rows,
                metadata_shared_fields,
            ),
        }
        if complete:
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
            baseline_target = targets.get(str(baseline.get("sample_id", "")))
            treatment_target = targets.get(str(treatment.get("sample_id", "")))
            if baseline_target is None or treatment_target is None:
                checks["pair_targets_valid"] = False
            else:
                delta = treatment_target - baseline_target
                delta_rms = float(np.sqrt(np.mean(delta**2)))
                baseline_truth = _finite_matrix(
                    baseline.get("generation_metadata", {}).get("truth_delta")
                )
                treatment_truth = _finite_matrix(
                    treatment.get("generation_metadata", {}).get("truth_delta")
                )
                checks.update(
                    {
                        "baseline_member_exact": bool(
                            _same_finite_float(baseline.get("dose_value"), 0.0)
                            and _same_finite_float(
                                baseline.get("applied_alpha"), 1.0
                            )
                            and _same_finite_float(
                                baseline.get("target_feature_value"),
                                0.0,
                            )
                            and baseline_truth is not None
                            and np.array_equal(
                                baseline_truth,
                                np.zeros_like(baseline_target),
                            )
                        ),
                        "treatment_truth_delta_exact": bool(
                            treatment_truth is not None
                            and treatment_truth.shape == delta.shape
                            and np.allclose(
                                treatment_truth,
                                delta,
                                rtol=1e-12,
                                atol=1e-12,
                            )
                        ),
                        "treatment_delta_nonzero": delta_rms > 0.0,
                        "target_feature_matches_pair_delta": bool(
                            _finite_float(
                                treatment.get("target_feature_value")
                            )
                            and _same_finite_float(
                                treatment.get("target_feature_value"),
                                treatment.get("intensity_target_feature_value"),
                            )
                            and math.isclose(
                                float(treatment["target_feature_value"]),
                                delta_rms,
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                        ),
                    }
                )
                complete_pairs[pair_id] = {
                    "baseline": baseline,
                    "treatment": treatment,
                    "baseline_target": baseline_target,
                    "delta": delta,
                    "delta_rms": delta_rms,
                }
        if not all(checks.values()):
            pair_failures.append(
                {
                    "counterfactual_pair_id": pair_id,
                    "checks": checks,
                }
            )

    group_shared_fields = (
        "paired_group_id",
        "dataset_id",
        "capability_id",
        "seed_index",
        "background_id",
        "anchor_id",
        "target_dim",
        "covariate_dim",
        "covariates",
        "shared_standardization",
        "mase_period",
        "mase_scale",
        "mase_scale_by_target",
        "mase_scale_effective_period_by_target",
        "mase_scale_fallback_target_indices",
        "mase_scale_policy",
        "mase_scale_source",
        "parameter_sampling",
        "intensity_calibration",
        "mandatory_input_ablation",
        "evaluation_table",
        "excluded_from_primary_score",
    )
    for group_id, group_rows in sorted(by_group.items()):
        group_pairs = [
            pair
            for pair in complete_pairs.values()
            if pair["baseline"].get("paired_group_id") == group_id
        ]
        try:
            ordered = sorted(
                group_pairs,
                key=lambda pair: int(pair["treatment"]["dose_index"]),
            )
            dose_indexes = [
                int(pair["treatment"]["dose_index"]) for pair in ordered
            ]
            alphas = [
                float(pair["treatment"]["applied_alpha"])
                for pair in ordered
            ]
            strengths = [
                float(pair["treatment"]["dose_value"])
                for pair in ordered
            ]
            selected_alphas = [
                float(value)
                for value in group_rows[0]["intensity_calibration"][
                    "selected_alphas"
                ]
            ]
            rms_values = [float(pair["delta_rms"]) for pair in ordered]
        except (KeyError, TypeError, ValueError, IndexError):
            ordered = []
            dose_indexes = []
            alphas = []
            strengths = []
            selected_alphas = []
            rms_values = []
        baseline_targets = [
            np.asarray(pair["baseline_target"], dtype=float) for pair in ordered
        ]
        scaled_deltas = [
            np.asarray(pair["delta"], dtype=float) / (alpha - 1.0)
            for pair, alpha in zip(ordered, alphas)
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
                and len(ordered) == len(selected_alphas)
                and len(group_rows) == 2 * len(selected_alphas)
            ),
            "frozen_treatment_grid_exact": (
                strengths == list(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
                and _validated_row_dose_calibration(group_rows[0])
                is not None
                and selected_alphas
                == _validated_row_dose_calibration(group_rows[0]).get(
                    "applied_alpha_grid"
                )
            ),
            "dose_indexes_match_grid": dose_indexes
            == list(range(1, len(selected_alphas) + 1)),
            "treatment_alphas_match_grid": alphas == selected_alphas,
            "duplicate_baselines_are_exact": bool(
                baseline_targets
                and all(
                    np.array_equal(target, baseline_targets[0])
                    for target in baseline_targets[1:]
                )
            ),
            "delta_rms_strictly_increases": bool(
                rms_values
                and all(
                    right > left
                    for left, right in zip(rms_values, rms_values[1:])
                )
            ),
            "delta_is_linear_in_alpha_minus_one": bool(
                scaled_deltas
                and len(scaled_deltas) == len(ordered)
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
        }
        checks.update(_paired_gate_replay_checks(ordered))
        if not _paired_group_checks_pass(checks):
            group_failures.append(
                {
                    "paired_group_id": group_id,
                    "dose_indexes": dose_indexes,
                    "treatment_alphas": alphas,
                    "delta_rms": rms_values,
                    "checks": checks,
                }
            )

    main_by_sample = {
        str(row.get("sample_id", "")): row for row in main_rows
    }
    ablations_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(ablation_rows):
        sample_id = str(row.get("sample_id", ""))
        source_id = str(row.get("input_ablation_source_sample_id", ""))
        source = main_by_sample.get(source_id)
        donor_id = str(row.get("donor_sample_id", ""))
        donor = main_by_sample.get(donor_id)
        target = _finite_matrix(row.get("target"))
        row_covariate = (
            None
            if row.get("covariates") is None
            else _finite_matrix(row.get("covariates"))
        )
        metadata = row.get("input_ablation_metadata")
        contract = row.get("mandatory_input_ablation")
        checks: dict[str, bool] = {
            "schema_valid": bool(
                row.get("schema_version") == STRUCTURAL_MASTER_SCHEMA
                and row.get("input_ablation_schema_version")
                == STRUCTURAL_ABLATION_SCHEMA
            ),
            "track_valid": bool(
                row.get("benchmark_track")
                == "real_anchored_counterfactual"
                and row.get("evaluation_table")
                == (
                    "real_anchored_structural_sensitivity_input_ablation"
                    if source is not None
                    and source.get("evaluation_table")
                    == "real_anchored_structural_sensitivity"
                    else "real_anchored_input_ablation"
                )
            ),
            "source_main_exists": source is not None,
            "frozen_treatment_grid_exact": bool(
                all(_v4_dose_row_checks(row).values())
                and isinstance(row.get("dose_index"), int)
                and 1 <= int(row["dose_index"])
                <= len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID)
            ),
            "excluded_from_primary_score": bool(
                row.get("excluded_from_primary_score") is True
                and isinstance(metadata, dict)
                and metadata.get("excluded_from_primary_score") is True
                and metadata.get(
                    "reported_as_separate_attribution_audit"
                )
                is True
            ),
        }
        if source is not None:
            source_target = targets.get(source_id)
            source_covariate = covariates.get(source_id)
            assessed = (
                [int(value) for value in contract["assessed_target_indices"]]
                if isinstance(contract, dict)
                and isinstance(contract.get("assessed_target_indices"), list)
                else []
            )
            ablated = (
                [int(value) for value in contract["ablated_input_indices"]]
                if isinstance(contract, dict)
                and isinstance(contract.get("ablated_input_indices"), list)
                else []
            )
            unchanged = [
                index
                for index in range(int(source.get("target_dim", 0)))
                if index not in ablated
            ]
            arrays_valid = bool(
                source_target is not None
                and target is not None
                and source_target.shape == target.shape
            )
            context = protocol.REAL_ANCHORED_CONTEXT_LENGTH
            donor_background_id = str(row.get("donor_background_id", ""))
            donor_seed_index = row.get("donor_seed_index")
            expected_donor_sample_id = (
                f"cafe_structural_cf__"
                f"{protocol.safe_id(str(source['dataset_id']))}__"
                f"{source['capability_id']}__"
                f"{protocol.safe_id(donor_background_id)}__"
                f"a{source['dose_index']}__m"
                f"{source['counterfactual_member']}"
            )
            donor_history_payload = (
                metadata.get("donor_visible_history_by_channel")
                if isinstance(metadata, dict)
                else None
            )
            affine_payload = (
                metadata.get("affine_match_by_channel")
                if isinstance(metadata, dict)
                else None
            )
            expected_channel_keys = {str(channel) for channel in ablated}
            donor_channels_valid = bool(
                isinstance(donor_history_payload, dict)
                and set(donor_history_payload) == expected_channel_keys
            )
            donor_channel_arrays: dict[int, np.ndarray] = {}
            if donor_channels_valid:
                for channel in ablated:
                    try:
                        donor_channel = np.asarray(
                            donor_history_payload[str(channel)],
                            dtype=float,
                        )
                    except (TypeError, ValueError):
                        donor_channels_valid = False
                        break
                    if (
                        donor_channel.shape != (context,)
                        or not np.isfinite(donor_channel).all()
                    ):
                        donor_channels_valid = False
                        break
                    donor_channel_arrays[channel] = donor_channel
            donor_matrix = (
                np.column_stack(
                    [donor_channel_arrays[channel] for channel in ablated]
                )
                if donor_channels_valid and ablated
                else None
            )
            committed_entry = (
                None
                if donor_commitment_entries is None
                else donor_commitment_entries.get(donor_id)
            )
            committed_channels = (
                committed_entry.get("visible_history_by_channel_sha256")
                if isinstance(committed_entry, dict)
                else None
            )
            upstream_metadata = (
                metadata.get("donor_upstream_commitment")
                if isinstance(metadata, dict)
                else None
            )
            upstream_commitment_valid = bool(
                isinstance(committed_entry, dict)
                and isinstance(committed_channels, dict)
                and donor_matrix is not None
                and isinstance(upstream_metadata, dict)
                and _is_sha256(donor_commitment_root_sha256)
                and row.get("structural_donor_commitment_root_sha256")
                == donor_commitment_root_sha256
                and row.get("donor_structural_commitment_entry_sha256")
                == committed_entry.get("entry_sha256")
                and upstream_metadata.get("manifest_schema_version")
                == STRUCTURAL_DONOR_COMMITMENT_SCHEMA
                and upstream_metadata.get("commitment_policy")
                == STRUCTURAL_DONOR_COMMITMENT_POLICY
                and upstream_metadata.get("commitment_root_sha256")
                == donor_commitment_root_sha256
                and upstream_metadata.get("entry_sha256")
                == committed_entry.get("entry_sha256")
                and committed_entry.get("sample_id") == donor_id
                and committed_entry.get("dataset_id")
                == source.get("dataset_id")
                and committed_entry.get("background_id")
                == donor_background_id
                and committed_entry.get("capability_id")
                == source.get("capability_id")
                and committed_entry.get("dose_index")
                == source.get("dose_index")
                and committed_entry.get("intensity_lambda")
                == source.get("intensity_lambda")
                and committed_entry.get("paired_treatment_strength")
                == source.get("paired_treatment_strength")
                and (
                    donor is None
                    or committed_entry.get("applied_alpha")
                    == donor.get("applied_alpha")
                )
                and (
                    donor is None
                    or committed_entry.get(
                        "paired_treatment_applied_alpha"
                    )
                    == donor.get("paired_treatment_applied_alpha")
                )
                and committed_entry.get("dose_calibration_policy_sha256")
                == source.get("dose_calibration_policy_sha256")
                and committed_entry.get("counterfactual_member")
                == source.get("counterfactual_member")
                and committed_entry.get("evaluation_table")
                == source.get("evaluation_table")
                and committed_entry.get("seed_index") == donor_seed_index
                and committed_entry.get("target_dim")
                == source.get("target_dim")
                and committed_entry.get("context_length") == context
                and committed_successor_by_sample.get(source_id) == donor_id
                and (
                    donor is None
                    or committed_entry.get("source_contract_sha256")
                    == donor.get("generation_metadata", {}).get(
                        "contract_sha256"
                    )
                )
                and committed_entry.get(
                    "source_structural_background_target_sha256"
                )
                == (
                    donor.get("source_structural_background_target_sha256")
                    if donor is not None
                    else committed_entry.get(
                        "source_structural_background_target_sha256"
                    )
                )
                and all(
                    committed_channels.get(str(channel))
                    == array_sha256(donor_channel_arrays[channel])
                    for channel in ablated
                )
            )
            affine_statistics_valid = bool(
                isinstance(affine_payload, dict)
                and set(affine_payload) == expected_channel_keys
                and arrays_valid
                and donor_channels_valid
            )
            affine_replacement_valid = affine_statistics_valid
            if affine_statistics_valid:
                assert source_target is not None
                assert target is not None
                for channel in ablated:
                    statistics = affine_payload.get(str(channel))
                    if not isinstance(statistics, dict):
                        affine_statistics_valid = False
                        affine_replacement_valid = False
                        break
                    donor_channel = donor_channel_arrays[channel]
                    recipient_channel = source_target[:context, channel]
                    donor_center = float(np.mean(donor_channel))
                    donor_scale = max(float(np.std(donor_channel)), 1e-9)
                    recipient_center = float(np.mean(recipient_channel))
                    recipient_scale = max(
                        float(np.std(recipient_channel)),
                        1e-9,
                    )
                    statistics_valid = all(
                        _finite_float(statistics.get(key))
                        and math.isclose(
                            float(statistics[key]),
                            expected,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        for key, expected in (
                            ("donor_center", donor_center),
                            ("donor_scale", donor_scale),
                            ("recipient_center", recipient_center),
                            ("recipient_scale", recipient_scale),
                        )
                    )
                    affine_statistics_valid = bool(
                        affine_statistics_valid and statistics_valid
                    )
                    expected_replacement = (
                        (donor_channel - donor_center)
                        / donor_scale
                        * recipient_scale
                        + recipient_center
                    )
                    affine_replacement_valid = bool(
                        affine_replacement_valid
                        and np.allclose(
                            target[:context, channel],
                            expected_replacement,
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    )
            donor_provenance_valid = bool(
                isinstance(metadata, dict)
                and metadata.get("donor_selection_policy")
                == (
                    "global_eligible_background_successor_"
                    "shard_invariant_v1"
                )
                and donor_matrix is not None
                and metadata.get("donor_visible_history_sha256")
                == array_sha256(donor_matrix)
                and affine_statistics_valid
                and affine_replacement_valid
                and arrays_valid
                and ablated
                and metadata.get("ablated_visible_history_sha256")
                == array_sha256(target[:context, ablated])
                and donor_background_id
                and donor_background_id != source.get("background_id")
                and isinstance(donor_seed_index, int)
                and donor_seed_index >= 0
                and donor_seed_index != source.get("seed_index")
                and donor_id == expected_donor_sample_id
                and upstream_commitment_valid
            )
            checks.update(
                {
                    "source_binding_exact": bool(
                        row.get("input_ablation_source_pair_id")
                        == source.get("counterfactual_pair_id")
                        and row.get("input_ablation_source_paired_group_id")
                        == source.get("paired_group_id")
                        and row.get("counterfactual_pair_id")
                        == f"{source['counterfactual_pair_id']}__input_ablation"
                        and row.get("paired_group_id")
                        == f"{source['paired_group_id']}__input_ablation"
                        and row.get("counterfactual_member")
                        == source.get("counterfactual_member")
                        and row.get("evaluation_table")
                        == (
                            "real_anchored_structural_sensitivity_"
                            "input_ablation"
                            if source.get("evaluation_table")
                            == "real_anchored_structural_sensitivity"
                            else "real_anchored_input_ablation"
                        )
                    ),
                    "sample_identity_valid": bool(
                        sample_id == f"{source_id}__input_ablation"
                        and row.get("master_sample_id") == sample_id
                        and row.get("baseline_sample_id")
                        == f"{source['baseline_sample_id']}__input_ablation"
                    ),
                    "source_fields_shared": all(
                        row.get(field) == source.get(field)
                        for field in (
                            "dataset_id",
                            "capability_id",
                            "seed_index",
                            "dose_index",
                            "intensity",
                            "dose_value",
                            "background_id",
                            "anchor_id",
                            "target_dim",
                            "covariate_dim",
                            "mase_scale",
                            "mase_scale_by_target",
                            "shared_standardization",
                            "intensity_calibration",
                            "dose_calibration",
                            "dose_calibration_policy_sha256",
                            "dose_value",
                            "intensity_lambda",
                            "paired_treatment_strength",
                            "applied_alpha",
                            "paired_treatment_applied_alpha",
                            "paired_minimum_separation_gate",
                        )
                    ),
                    "mandatory_contract_shared": bool(
                        isinstance(contract, dict)
                        and contract == source.get("mandatory_input_ablation")
                        and contract.get("required") is True
                    ),
                    "metadata_contract_valid": bool(
                        isinstance(metadata, dict)
                        and metadata.get("assessed_target_indices") == assessed
                        and metadata.get("ablated_input_indices") == ablated
                        and metadata.get(
                            "assessed_target_history_unchanged"
                        )
                        is True
                        and metadata.get("scored_future_unchanged") is True
                    ),
                    "donor_payload_valid": bool(
                        donor_channels_valid and donor_matrix is not None
                    ),
                    "affine_statistics_valid": affine_statistics_valid,
                    "affine_replacement_valid": affine_replacement_valid,
                    "donor_provenance_valid": donor_provenance_valid,
                    "donor_upstream_commitment_valid": (
                        upstream_commitment_valid
                    ),
                    "target_arrays_valid": arrays_valid,
                    "assessed_history_unchanged": bool(
                        arrays_valid
                        and assessed
                        and np.array_equal(
                            target[:context, assessed],
                            source_target[:context, assessed],
                        )
                    ),
                    "all_nonablated_history_unchanged": bool(
                        arrays_valid
                        and np.array_equal(
                            target[:context, unchanged],
                            source_target[:context, unchanged],
                        )
                    ),
                    "ablated_history_changed": bool(
                        arrays_valid
                        and ablated
                        and not np.array_equal(
                            target[:context, ablated],
                            source_target[:context, ablated],
                        )
                    ),
                    "future_truth_unchanged": bool(
                        arrays_valid
                        and np.array_equal(
                            target[context:],
                            source_target[context:],
                        )
                    ),
                    "covariates_unchanged": bool(
                        (source_covariate is None and row_covariate is None)
                        or (
                            source_covariate is not None
                            and row_covariate is not None
                            and np.array_equal(
                                row_covariate,
                                source_covariate,
                            )
                        )
                    ),
                    "target_hash_matches": bool(
                        arrays_valid
                        and row.get("target_sha256")
                        == protocol.target_and_covariate_sha256(
                            target,
                            row_covariate,
                        )
                    ),
                    "future_hash_matches_source": bool(
                        arrays_valid
                        and row.get("future_sha256")
                        == source.get("future_sha256")
                        == array_sha256(target[context:])
                    ),
                }
            )
        if source is not None and donor is not None:
            donor_target = targets.get(donor_id)
            checks["donor_in_shard_contract_valid"] = bool(
                donor_id != source_id
                and donor.get("background_id")
                == row.get("donor_background_id")
                and donor.get("seed_index") == row.get("donor_seed_index")
                and donor.get("background_id") != source.get("background_id")
                and all(
                    donor.get(field) == source.get(field)
                    for field in (
                        "dataset_id",
                        "capability_id",
                        "dose_index",
                            "counterfactual_member",
                            "target_dim",
                            "evaluation_table",
                        )
                )
                and donor_target is not None
                and isinstance(metadata, dict)
                and donor_matrix is not None
                and np.array_equal(
                    donor_matrix,
                    donor_target[
                        : protocol.REAL_ANCHORED_CONTEXT_LENGTH,
                        ablated,
                    ],
                )
                and metadata.get("donor_visible_history_sha256")
                == array_sha256(
                    donor_target[
                        : protocol.REAL_ANCHORED_CONTEXT_LENGTH,
                        ablated,
                    ]
                )
            )
        if not all(checks.values()):
            ablation_failures.append(
                {
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "source_sample_id": source_id,
                    "checks": checks,
                }
            )
        if source_id:
            ablations_by_source[source_id].append(row)

    coverage_failures: list[dict[str, Any]] = []
    for sample_id, source in sorted(main_by_sample.items()):
        expected = source.get("capability_id") in mandatory_ablation_capabilities
        matched = ablations_by_source.get(sample_id, [])
        if (expected and len(matched) != 1) or (not expected and matched):
            coverage_failures.append(
                {
                    "source_sample_id": sample_id,
                    "capability_id": source.get("capability_id"),
                    "expected_ablation_count": 1 if expected else 0,
                    "observed_ablation_count": len(matched),
                }
            )
    unknown_sources = sorted(
        source_id
        for source_id in ablations_by_source
        if source_id not in main_by_sample
    )
    for source_id in unknown_sources:
        coverage_failures.append(
            {
                "source_sample_id": source_id,
                "capability_id": None,
                "expected_ablation_count": 0,
                "observed_ablation_count": len(
                    ablations_by_source[source_id]
                ),
            }
        )

    ablation_pair_failures: list[dict[str, Any]] = []
    ablation_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ablation_rows:
        ablation_by_pair[str(row.get("counterfactual_pair_id", ""))].append(row)
    for pair_id, pair_rows in sorted(ablation_by_pair.items()):
        members = [row.get("counterfactual_member") for row in pair_rows]
        source_pairs = {
            str(row.get("input_ablation_source_pair_id", ""))
            for row in pair_rows
        }
        checks = {
            "exactly_two_members": bool(
                len(pair_rows) == 2
                and members.count(0) == 1
                and members.count(1) == 1
            ),
            "one_source_main_pair": bool(
                len(source_pairs) == 1 and "" not in source_pairs
            ),
            "pair_id_bound_to_source": bool(
                len(source_pairs) == 1
                and pair_id
                == f"{next(iter(source_pairs))}__input_ablation"
            ),
        }
        if not all(checks.values()):
            ablation_pair_failures.append(
                {
                    "counterfactual_pair_id": pair_id,
                    "checks": checks,
                }
            )

    background_groups: dict[tuple[str, str, str, str], set[str]] = defaultdict(
        set
    )
    seed_backgrounds: dict[tuple[str, str, int, str], set[str]] = defaultdict(
        set
    )
    for group_id, group_rows in by_group.items():
        if not group_rows:
            continue
        first = group_rows[0]
        try:
            key = (
                str(first["dataset_id"]),
                str(first["capability_id"]),
                str(first["background_id"]),
                str(first["evaluation_table"]),
            )
            seed_key = (
                str(first["dataset_id"]),
                str(first["capability_id"]),
                int(first["seed_index"]),
                str(first["evaluation_table"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        background_groups[key].add(group_id)
        seed_backgrounds[seed_key].add(key[2])
    repeated_background_failures = [
        {
            "dataset_id": key[0],
            "capability_id": key[1],
            "background_id": key[2],
            "evaluation_table": key[3],
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
            "evaluation_table": key[3],
            "background_ids": sorted(background_ids),
        }
        for key, background_ids in sorted(seed_backgrounds.items())
        if len(background_ids) > 1
    ]
    formal_rows = [
        row
        for row in main_rows
        if row.get("evaluation_table")
        != "real_anchored_structural_sensitivity"
    ]
    sensitivity_rows = [
        row
        for row in main_rows
        if row.get("evaluation_table")
        == "real_anchored_structural_sensitivity"
    ]
    upstream_replay_coverage_failures = [
        *_upstream_replay_coverage_failures(
            formal_rows,
            expected_replay_rows,
        ),
        *_upstream_replay_coverage_failures(
            sensitivity_rows,
            expected_sensitivity_replay_rows,
        ),
    ]
    accepted = not any(
        (
            row_failures,
            pair_failures,
            group_failures,
            ablation_failures,
            coverage_failures,
            ablation_pair_failures,
            repeated_background_failures,
            seed_assignment_failures,
            upstream_replay_coverage_failures,
        )
    )
    return {
        "accepted": accepted,
        "main_sample_count": len(main_rows),
        "formal_main_sample_count": len(formal_rows),
        "sensitivity_main_sample_count": len(sensitivity_rows),
        "input_ablation_sample_count": len(ablation_rows),
        "main_pair_count": len(by_pair),
        "input_ablation_pair_count": len(ablation_by_pair),
        "paired_group_count": len(by_group),
        "row_failures": row_failures,
        "pair_failures": pair_failures,
        "paired_group_failures": group_failures,
        "input_ablation_failures": ablation_failures,
        "input_ablation_coverage_failures": coverage_failures,
        "input_ablation_pair_failures": ablation_pair_failures,
        "repeated_background_failures": repeated_background_failures,
        "seed_assignment_failures": seed_assignment_failures,
        "upstream_replay_coverage_failures": (
            upstream_replay_coverage_failures
        ),
        "effective_background_count_by_capability": {
            capability_id: len(
                {
                    str(row.get("background_id", ""))
                    for row in formal_rows
                    if row.get("capability_id") == capability_id
                }
            )
            for capability_id in sorted(
                {
                    str(row.get("capability_id", ""))
                    for row in formal_rows
                    if row.get("capability_id")
                }
            )
        },
        "sensitivity_background_count_by_capability": {
            capability_id: len(
                {
                    str(row.get("background_id", ""))
                    for row in sensitivity_rows
                    if row.get("capability_id") == capability_id
                }
            )
            for capability_id in sorted(
                {
                    str(row.get("capability_id", ""))
                    for row in sensitivity_rows
                    if row.get("capability_id")
                }
            )
        },
    }


def real_anchored_counterfactual_checks(
    rows: list[dict[str, Any]],
    *,
    expected_row_count: int | None = None,
    donor_commitment_entries: dict[str, dict[str, Any]] | None = None,
    donor_commitment_root_sha256: str | None = None,
    upstream_replay_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate v4 real-path rows and isolate legacy empty components."""

    univariate_rows: list[dict[str, Any]] = []
    nonlinear_replay_rows: list[dict[str, Any]] = []
    structural_main_rows: list[dict[str, Any]] = []
    structural_ablation_rows: list[dict[str, Any]] = []
    routing_failures: list[dict[str, Any]] = []
    hierarchy_rows: list[str] = []
    undersized_formal_panel_rows: list[str] = []
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        capability_id = str(row.get("capability_id", ""))
        evaluation_table = row.get("evaluation_table")
        if capability_id == "hierarchical_coherence":
            hierarchy_rows.append(sample_id)
        if (
            evaluation_table
            == "real_anchored_nonlinear_replay_sensitivity"
        ):
            nonlinear_replay_rows.append(row)
            continue
        if evaluation_table == "real_anchored_input_ablation" or row.get(
            "input_ablation_schema_version"
        ) is not None:
            structural_ablation_rows.append(row)
            continue
        structural = bool(
            row.get("schema_version") == STRUCTURAL_MASTER_SCHEMA
            or row.get("generator_family_role") == "real_anchored_structural"
        )
        if structural:
            structural_main_rows.append(row)
            if (
                evaluation_table == "real_anchored_counterfactual"
                and capability_id
                in {"common_factor", "cross_series_dependence"}
                and isinstance(row.get("target_dim"), int)
                and int(row["target_dim"])
                < FORMAL_PANEL_MINIMUM_DIMENSION
            ):
                undersized_formal_panel_rows.append(sample_id)
            continue
        schema = row.get("schema_version")
        legacy = schema == "cafe.real_anchored_counterfactual_master.v1"
        evaluation_valid = bool(
            evaluation_table == "real_anchored_counterfactual"
            or legacy and evaluation_table in {None, "main"}
        )
        role_valid = row.get("generator_family_role") == "real_anchored"
        if not evaluation_valid or not role_valid:
            routing_failures.append(
                {
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "checks": {
                        "main_evaluation_table_valid": evaluation_valid,
                        "generator_family_role_valid": role_valid,
                    },
                }
            )
        univariate_rows.append(row)

    univariate_expected = (
        upstream_replay_evidence.get("univariate_expected_rows")
        if isinstance(upstream_replay_evidence, dict)
        else None
    )
    structural_expected = (
        upstream_replay_evidence.get("structural_expected_rows")
        if isinstance(upstream_replay_evidence, dict)
        else None
    )
    structural_sensitivity_expected = (
        upstream_replay_evidence.get(
            "structural_sensitivity_expected_rows"
        )
        if isinstance(upstream_replay_evidence, dict)
        else None
    )
    nonlinear_replay_expected = (
        upstream_replay_evidence.get("nonlinear_replay_expected_rows")
        if isinstance(upstream_replay_evidence, dict)
        else None
    )
    univariate = _univariate_real_anchored_counterfactual_checks(
        univariate_rows,
        expected_replay_rows=(
            univariate_expected
            if isinstance(univariate_expected, dict)
            else None
        ),
    )
    nonlinear_replay = _nonlinear_replay_sensitivity_checks(
        nonlinear_replay_rows,
        source_main_rows=univariate_rows,
        expected_replay_rows=(
            nonlinear_replay_expected
            if isinstance(nonlinear_replay_expected, dict)
            else None
        ),
    )
    structural = _structural_real_anchored_checks(
        structural_main_rows,
        structural_ablation_rows,
        donor_commitment_entries=donor_commitment_entries,
        donor_commitment_root_sha256=donor_commitment_root_sha256,
        expected_replay_rows=(
            structural_expected
            if isinstance(structural_expected, dict)
            else None
        ),
        expected_sensitivity_replay_rows=(
            structural_sensitivity_expected
            if isinstance(structural_sensitivity_expected, dict)
            else None
        ),
    )
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
    manifest_row_count_matches = bool(
        expected_row_count is None or expected_row_count == len(rows)
    )
    policy_failures: list[dict[str, Any]] = []
    if hierarchy_rows:
        policy_failures.append(
            {
                "policy": "hierarchy_qualification_only_no_formal_rows",
                "sample_ids": sorted(hierarchy_rows),
            }
        )
    if undersized_formal_panel_rows:
        policy_failures.append(
            {
                "policy": "formal_panel_dimension_at_least_three",
                "capabilities": [
                    "common_factor",
                    "cross_series_dependence",
                ],
                "sample_ids": sorted(undersized_formal_panel_rows),
            }
        )
    row_failures = [
        *univariate["row_failures"],
        *nonlinear_replay["row_failures"],
        *structural["row_failures"],
        *routing_failures,
    ]
    pair_failures = [
        *univariate["pair_failures"],
        *nonlinear_replay["pair_failures"],
        *structural["pair_failures"],
    ]
    group_failures = [
        *univariate["paired_group_failures"],
        *nonlinear_replay["paired_group_failures"],
        *structural["paired_group_failures"],
    ]
    repeated_background_failures = [
        *univariate["repeated_background_failures"],
        *structural["repeated_background_failures"],
    ]
    seed_assignment_failures = [
        *univariate["seed_assignment_failures"],
        *structural["seed_assignment_failures"],
    ]
    accepted = bool(
        manifest_row_count_matches
        and not duplicate_sample_ids
        and not duplicate_master_ids
        and not policy_failures
        and univariate["accepted"]
        and nonlinear_replay["accepted"]
        and structural["accepted"]
        and not routing_failures
    )
    effective_backgrounds = dict(
        univariate["effective_background_count_by_capability"]
    )
    effective_backgrounds.update(
        structural["effective_background_count_by_capability"]
    )
    return {
        "schema_version": "cafe.real_anchored_validation.v5",
        "status": "evaluated",
        "accepted": accepted,
        "sample_count": len(rows),
        "pair_count": len(
            {
                str(row.get("counterfactual_pair_id", ""))
                for row in rows
                if row.get("counterfactual_pair_id")
            }
        ),
        "paired_group_count": len(
            {
                str(row.get("paired_group_id", ""))
                for row in rows
                if row.get("paired_group_id")
            }
        ),
        "univariate_sample_count": len(univariate_rows),
        "nonlinear_replay_sensitivity_sample_count": len(
            nonlinear_replay_rows
        ),
        "structural_main_sample_count": sum(
            row.get("evaluation_table")
            == "real_anchored_counterfactual"
            for row in structural_main_rows
        ),
        "structural_sensitivity_sample_count": sum(
            row.get("evaluation_table")
            == "real_anchored_structural_sensitivity"
            for row in structural_main_rows
        ),
        "structural_input_ablation_sample_count": len(
            structural_ablation_rows
        ),
        "expected_manifest_row_count": expected_row_count,
        "manifest_row_count_matches": manifest_row_count_matches,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_master_sample_ids": duplicate_master_ids,
        "row_failures": row_failures,
        "pair_failures": pair_failures,
        "paired_group_failures": group_failures,
        "policy_failures": policy_failures,
        "input_ablation_failures": structural[
            "input_ablation_failures"
        ],
        "input_ablation_coverage_failures": structural[
            "input_ablation_coverage_failures"
        ],
        "input_ablation_pair_failures": structural[
            "input_ablation_pair_failures"
        ],
        "repeated_background_failures": repeated_background_failures,
        "seed_assignment_failures": seed_assignment_failures,
        "upstream_replay_coverage_failures": [
            *univariate["upstream_replay_coverage_failures"],
            *nonlinear_replay["upstream_replay_coverage_failures"],
            *structural["upstream_replay_coverage_failures"],
        ],
        "effective_background_count_by_capability": effective_backgrounds,
        "univariate_validation": univariate,
        "nonlinear_replay_sensitivity_validation": nonlinear_replay,
        "structural_validation": structural,
        "background_sampling_policy": (
            "unique_real_background_per_dataset_capability_main_rows_only"
        ),
        "intentional_duplicate_baseline_policy": (
            "allowed_within_paired_group_across_alpha_doses_if_exact"
        ),
        "anti_copy_policy": (
            "not_applicable_intentional_real_anchor_counterfactual"
        ),
        "hierarchy_policy": "qualification_only_formal_rows_rejected",
        "formal_panel_dimension_policy": (
            "common_and_cross_D>=3;D2_sensitivity_only;"
            "covariate_response_D>=1_with_known_future_inputs"
        ),
        "structural_input_ablation_policy": (
            "mandatory_for_common_and_cross_reported_separately_"
            "excluded_from_primary_score"
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
    upstream_replay_evidence: dict[str, Any] = {}
    validated_config = validate_generation_manifest_contract(
        manifest,
        manifest_path=manifest_path,
        calibration_dir=generation_dir.parent / "01_calibration",
        dataset_id=dataset.dataset_id,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        replay_evidence_out=upstream_replay_evidence,
    )
    real_config = validated_config.get("real_anchored_counterfactual", {})
    donor_declaration = (
        real_config.get("structural_donor_commitment")
        if isinstance(real_config, dict)
        else None
    )
    donor_commitment_entries: dict[str, dict[str, Any]] | None = None
    donor_commitment_root_sha256: str | None = None
    if isinstance(donor_declaration, dict):
        donor_sidecar = protocol.read_json(
            Path(manifest["files"]["structural_donor_commitments"]["path"])
        )
        donor_commitment_entries = {
            str(entry["sample_id"]): entry
            for entry in donor_sidecar["entries"]
        }
        donor_commitment_root_sha256 = str(
            donor_sidecar["commitment_root_sha256"]
        )
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
            **real_anchored_counterfactual_checks(
                real_anchored_rows,
                upstream_replay_evidence=(
                    upstream_replay_evidence or None
                ),
            ),
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
            donor_commitment_entries=donor_commitment_entries,
            donor_commitment_root_sha256=(
                donor_commitment_root_sha256
            ),
            upstream_replay_evidence=(upstream_replay_evidence or None),
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
        "schema_version": "cafe.generation_validation.v5",
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
