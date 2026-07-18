#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.services.synthetic_generation_service import (  # noqa: E402
    _generate_accepted_sample_values,
    _seed_for,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    INTENSITY_POLICY_ID,
    resolve_generator_conditioning,
)


SCHEMA_VERSION = "paper_e1_method_validity.v2"
EXPERIMENT_VERSION = "v4"
EXPERIMENT_ID = "E1_method_validity"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp" / EXPERIMENT_VERSION / EXPERIMENT_ID
NINE_CAPABILITY_SUITE_DIR = (
    REPO_ROOT / "runtime/paper_exp/v4/01_nine_capability_suite"
)
GENERATOR_ARTIFACT_PATH = NINE_CAPABILITY_SUITE_DIR / "generator_conditioning_artifact.json"
FEATURE_GATE_ARTIFACT_PATH = NINE_CAPABILITY_SUITE_DIR / "feature_gate_artifact.json"
NEAR_DISTANCE_ARTIFACT_PATH = NINE_CAPABILITY_SUITE_DIR / "near_distance_artifact.json"
SUPPORT_MATRIX_PATH = (
    NINE_CAPABILITY_SUITE_DIR / "dataset_capability_support_matrix.json"
)
DEFAULT_ROUND_SEEDS = (2026071601, 2026071602)
DEFAULT_SAMPLES_PER_ROUND = 64
INTENSITIES = (1, 2, 3, 4, 5)

# These criteria are protocol constants, not values selected after the run.
DOSE_MAX_NORMALIZED_ERROR = 0.25
DOSE_MIN_SPEARMAN = 0.90
CONTROL_MAX_MEDIAN_ABS_PAIRED_SHIFT = 1.00
CONTROL_MAX_ABS_MEDIAN_SIGNED_SHIFT = 0.50
MIN_FIRST_PASS_ACCEPTANCE_RATE = 0.95
MIN_DISTRIBUTION_CLOSER_FRACTION = 0.90
MAX_DUPLICATE_RATE = 0.0
MIN_ORACLE_WIN_RATE = 0.50
NEGATIVE_CONTROL_SHIFT_IQR = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-v4 E1 dataset-local synthetic-method validity experiment."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-round", type=int, default=DEFAULT_SAMPLES_PER_ROUND)
    parser.add_argument("--round-seeds", nargs=2, type=int, default=DEFAULT_ROUND_SEEDS)
    parser.add_argument(
        "--allow-existing-empty",
        action="store_true",
        help="Permit a pre-created but empty output directory; existing result files are never overwritten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_round < 8:
        raise ValueError("samples-per-round must be at least 8")
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, allow_existing_empty=args.allow_existing_empty)

    generator_artifact = read_json(GENERATOR_ARTIFACT_PATH)
    feature_artifact = read_json(FEATURE_GATE_ARTIFACT_PATH)
    near_artifact = read_json(NEAR_DISTANCE_ARTIFACT_PATH)
    support_matrix = read_json(SUPPORT_MATRIX_PATH)
    validate_artifact_alignment(
        generator_artifact,
        feature_artifact,
        near_artifact,
        support_matrix,
    )
    capability_support = capability_support_summary(
        generator_artifact,
        support_matrix,
    )
    if capability_support["supported_count"] == 0:
        raise ValueError("E1 has no supported dataset/profile/capability cells")
    config = experiment_config(
        generator_artifact=generator_artifact,
        samples_per_round=args.samples_per_round,
        round_seeds=tuple(args.round_seeds),
        capability_support=capability_support,
    )
    write_json(output_dir / "config.json", config)

    internal_samples = generate_samples(
        generator_artifact=generator_artifact,
        feature_artifact=feature_artifact,
        near_artifact=near_artifact,
        supported_cells=capability_support["supported_cells"],
        samples_per_round=args.samples_per_round,
        round_seeds=tuple(args.round_seeds),
    )
    sample_rows = [sample["row"] for sample in internal_samples]
    write_jsonl(output_dir / "samples.jsonl", sample_rows)

    dose_rows, dose_summary = dose_response_analysis(sample_rows, generator_artifact)
    selectivity_rows, response_matrix_rows, selectivity_summary = selectivity_analysis(
        sample_rows,
        feature_artifact,
        generator_artifact,
    )
    construction_rows, construction_summary = construction_analysis(sample_rows)
    support_rows, support_summary = support_analysis(sample_rows)
    novelty_rows, novelty_summary = novelty_analysis(sample_rows)
    repetition_rows, repetition_summary = repetition_analysis(internal_samples)
    baseline_rows, baseline_summary = baseline_analysis(sample_rows)
    distribution_rows, distribution_summary = distribution_analysis(
        internal_samples,
        feature_artifact=feature_artifact,
    )

    csv_outputs = {
        "dose_response.csv": dose_rows,
        "selectivity_controls.csv": selectivity_rows,
        "selectivity_response_matrix.csv": response_matrix_rows,
        "construction_predictability.csv": construction_rows,
        "control_support.csv": support_rows,
        "distribution_mmd_swd.csv": distribution_rows,
        "novelty_dcr_nndr.csv": novelty_rows,
        "cross_round_repetition.csv": repetition_rows,
        "baseline_oracle_response.csv": baseline_rows,
        "dataset_capability_support_matrix.csv": support_matrix_output_rows(
            capability_support
        ),
    }
    for filename, rows in csv_outputs.items():
        write_csv(output_dir / filename, rows)

    criteria = evaluate_criteria(
        dose_summary=dose_summary,
        selectivity_summary=selectivity_summary,
        construction_summary=construction_summary,
        support_summary=support_summary,
        distribution_summary=distribution_summary,
        novelty_summary=novelty_summary,
        repetition_summary=repetition_summary,
        baseline_summary=baseline_summary,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "sample_count": len(sample_rows),
        "capability_support": capability_support,
        "dose_response": dose_summary,
        "selectivity": selectivity_summary,
        "construction_predictability": construction_summary,
        "control_support": support_summary,
        "distribution_mmd_swd": distribution_summary,
        "novelty_dcr_nndr": novelty_summary,
        "cross_round_repetition": repetition_summary,
        "baseline_oracle_response": baseline_summary,
        "criteria": criteria,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    write_manifest(output_dir, config=config)

    print(f"wrote E1 experiment: {output_dir}")
    print(f"samples: {len(sample_rows)}")
    print(f"criteria passed: {criteria['passed_count']}/{criteria['criterion_count']}")
    print(f"overall_passed: {criteria['overall_passed']}")
    return 0


def experiment_config(
    *,
    generator_artifact: dict[str, Any],
    samples_per_round: int,
    round_seeds: tuple[int, int],
    capability_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if capability_support is None:
        raise ValueError("paper-v4 E1 requires the dataset capability support matrix")
    support = capability_support
    intensity_policy = generator_artifact["intensity_policy"]
    profile_ids = sorted(
        {str(cell["profile_id"]) for cell in support["supported_cells"]}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "large_models_used": False,
        "intensity_policy": dict(intensity_policy),
        "conditioning_profile_ids": profile_ids,
        "dataset_ids": sorted(
            {str(cell["dataset_id"]) for cell in support["supported_cells"]}
        ),
        "support_matrix_path": relative_or_absolute(SUPPORT_MATRIX_PATH),
        "supported_dataset_profile_capability_count": support["supported_count"],
        "unsupported_dataset_profile_capability_count": support["unsupported_count"],
        "intensities": list(INTENSITIES),
        "round_seeds": list(round_seeds),
        "samples_per_round_per_cell": int(samples_per_round),
        "paired_seed_policy": (
            "within profile/capability/round/sample_index, the same sample seed is reused across "
            "all five intensities"
        ),
        "real_control_vector_source": (
            "frozen dataset-local gate-reference and gate-calibration vectors "
            "inside the feature-gate artifact"
        ),
        "criteria": {
            "dose_max_normalized_error": DOSE_MAX_NORMALIZED_ERROR,
            "dose_min_spearman": DOSE_MIN_SPEARMAN,
            "control_max_median_abs_paired_shift_iqr": (
                CONTROL_MAX_MEDIAN_ABS_PAIRED_SHIFT
            ),
            "control_max_abs_median_signed_shift_iqr": (
                CONTROL_MAX_ABS_MEDIAN_SIGNED_SHIFT
            ),
            "minimum_first_pass_acceptance_rate_per_cell": (
                MIN_FIRST_PASS_ACCEPTANCE_RATE
            ),
            "minimum_distribution_cells_closer_than_shifted_negative_fraction": (
                MIN_DISTRIBUTION_CLOSER_FRACTION
            ),
            "maximum_exact_or_near_duplicate_rate": MAX_DUPLICATE_RATE,
            "minimum_capability_oracle_win_rate": MIN_ORACLE_WIN_RATE,
        },
        "distribution_negative_control_shift_iqr": NEGATIVE_CONTROL_SHIFT_IQR,
        "output_retention": (
            "immutable experiment directory; the runner refuses to overwrite any existing file"
        ),
    }


def validate_artifact_alignment(
    generator_artifact: dict[str, Any],
    feature_artifact: dict[str, Any],
    near_artifact: dict[str, Any],
    support_matrix: dict[str, Any],
) -> None:
    if generator_artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "E1 requires dataset-local generator conditioning artifact "
            f"{ARTIFACT_SCHEMA_VERSION}"
        )
    policy = generator_artifact.get("intensity_policy", {})
    if policy.get("policy_id") != INTENSITY_POLICY_ID:
        raise ValueError(f"E1 requires intensity policy {INTENSITY_POLICY_ID}")
    if (
        support_matrix.get("schema_version")
        != "paper_v4_dataset_capability_support_matrix.v1"
    ):
        raise ValueError("E1 requires the paper-v4 dataset capability support matrix")
    supported_cells = [
        cell for cell in support_matrix.get("cells", [])
        if cell.get("status") == "supported"
    ]
    if not supported_cells:
        raise ValueError("E1 support matrix has no supported cells")
    profile_ids = tuple(
        sorted({str(cell["generator_profile_id"]) for cell in supported_cells})
    )
    generator_profiles = set(generator_artifact["profiles"])
    feature_profiles = set(feature_artifact["buckets"])
    near_profiles = set(near_artifact["buckets"])
    missing = {
        "generator": sorted(set(profile_ids) - generator_profiles),
        "feature_gate": sorted(set(profile_ids) - feature_profiles),
        "near_distance": sorted(set(profile_ids) - near_profiles),
    }
    if any(missing.values()):
        raise ValueError(f"paper-v4 online artifact alignment failed: {missing}")
    for profile_id in profile_ids:
        profile = generator_artifact["profiles"][profile_id]
        if (
            profile.get("conditioning_role")
            != "paper_v4_dataset_local_train_only_master_task"
        ):
            raise ValueError(f"{profile_id} is not a paper-v4 dataset-local master")
        if not str(profile.get("dataset_id", "")).strip():
            raise ValueError(f"{profile_id} has no dataset_id")
    for cell in supported_cells:
        profile_id = str(cell["generator_profile_id"])
        capability_id = str(cell["capability_id"])
        profile = generator_artifact["profiles"][profile_id]
        if str(cell.get("dataset_id")) != str(profile.get("dataset_id")):
            raise ValueError(f"{profile_id}/{capability_id} dataset identity mismatch")
        capability = profile.get("capabilities", {}).get(capability_id)
        if not is_supported_capability(capability):
            raise ValueError(
                f"{profile_id}/{capability_id} is supported in the matrix "
                "but missing supported generator conditioning"
            )
        feature_capabilities = feature_artifact["buckets"][profile_id].get(
            "capabilities", {}
        )
        if capability_id not in feature_capabilities:
            raise ValueError(
                f"{profile_id}/{capability_id} has no feature-gate calibration"
            )


def capability_support_summary(
    generator_artifact: dict[str, Any],
    support_matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    if support_matrix is None:
        raise ValueError("dataset capability support matrix is required")
    seen: set[tuple[str, str]] = set()
    for cell in support_matrix.get("cells", []):
        profile_id = str(cell.get("generator_profile_id", ""))
        capability_id = str(cell.get("capability_id", ""))
        key = (profile_id, capability_id)
        if not profile_id or not capability_id or key in seen:
            continue
        seen.add(key)
        row = {
            "dataset_id": str(cell.get("dataset_id", "")),
            "task_id": str(cell.get("task_id", "")),
            "profile_id": profile_id,
            "capability_id": capability_id,
        }
        profile = generator_artifact.get("profiles", {}).get(profile_id, {})
        capability = profile.get("capabilities", {}).get(capability_id)
        if cell.get("status") == "supported" and is_supported_capability(capability):
            supported.append({**row, "status": "supported", "reason_codes": []})
        else:
            reason_codes = [
                str(reason)
                for reason in (cell.get("reason_codes") or [])
                if str(reason)
            ]
            if cell.get("status") == "supported":
                reason_codes.append("generator_conditioning_missing_or_unsupported")
            if not reason_codes:
                reason_codes = ["unsupported_by_dataset_suite"]
            unsupported.append(
                {
                    **row,
                    "status": "unsupported",
                    "reason_codes": reason_codes,
                    "reason": reason_codes[0],
                }
            )
    return {
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "supported_cells": supported,
        "unsupported_cells": unsupported,
    }


def support_matrix_output_rows(
    capability_support: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        *capability_support["supported_cells"],
        *capability_support["unsupported_cells"],
    ]
    return [
        {
            "dataset_id": row["dataset_id"],
            "task_id": row["task_id"],
            "profile_id": row["profile_id"],
            "capability_id": row["capability_id"],
            "status": row["status"],
            "reason_codes": ";".join(row.get("reason_codes", [])),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["dataset_id"]),
                str(item["capability_id"]),
            ),
        )
    ]


def is_supported_capability(capability: Any) -> bool:
    calibration = (
        capability.get("calibration")
        if isinstance(capability, dict)
        else None
    )
    return bool(
        isinstance(capability, dict)
        and capability.get("status", "supported") == "supported"
        and isinstance(calibration, dict)
        and calibration.get("status") == "supported"
    )


def generate_samples(
    *,
    generator_artifact: dict[str, Any],
    feature_artifact: dict[str, Any],
    near_artifact: dict[str, Any],
    supported_cells: list[dict[str, Any]],
    samples_per_round: int,
    round_seeds: tuple[int, int],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for cell in supported_cells:
        profile_id = str(cell["profile_id"])
        capability_id = str(cell["capability_id"])
        profile = generator_artifact["profiles"][profile_id]
        capability = profile["capabilities"][capability_id]
        if not is_supported_capability(capability):
            raise RuntimeError(f"unsupported E1 cell {profile_id}/{capability_id}")
        conditioning = resolve_generator_conditioning(
            capability_id=capability_id,
            profile_id=profile_id,
            context_length=int(profile["context_length"]),
            horizon=int(profile["horizon"]),
            target_dim=int(profile["target_dim"]),
            artifact=generator_artifact,
        )
        if conditioning is None:
            raise RuntimeError(f"missing conditioning for {profile_id}/{capability_id}")
        for round_index, round_seed in enumerate(round_seeds, start=1):
            for sample_index in range(samples_per_round):
                sample_seed = _seed_for(
                    round_seed,
                    f"{profile_id}:{capability_id}",
                    sample_index,
                )
                for intensity in INTENSITIES:
                    target, latent, covariates, features = _generate_accepted_sample_values(
                        capability_id,
                        int(profile["context_length"]) + int(profile["horizon"]),
                        int(profile["context_length"]),
                        int(profile["target_dim"]),
                        int(profile["season_length"]),
                        intensity,
                        sample_seed,
                        anchor_profile_id=profile_id,
                        generator_conditioning=conditioning,
                        generator_conditioning_artifact=generator_artifact,
                        feature_gate_artifact=feature_artifact,
                        near_distance_artifact=near_artifact,
                        acceptance_profile_ids=(profile_id,),
                    )
                    forecasts = forecast_metrics(
                        capability_id=capability_id,
                        target=target,
                        covariates=covariates,
                        context_length=int(profile["context_length"]),
                        season_length=int(profile["season_length"]),
                        latent=latent,
                    )
                    acceptance = latent["acceptance"]
                    validation = acceptance["validation"]
                    near_bucket = matching_near_bucket(
                        validation["near_distance_gate"], profile_id
                    )
                    row = {
                        "schema_version": "paper_e1_sample.v2",
                        "dataset_id": conditioning.dataset_id,
                        "profile_id": profile_id,
                        "capability_id": capability_id,
                        "intensity": intensity,
                        "round_index": round_index,
                        "round_seed": round_seed,
                        "sample_index": sample_index,
                        "sample_seed": sample_seed,
                        "context_length": int(profile["context_length"]),
                        "horizon": int(profile["horizon"]),
                        "season_length": int(profile["season_length"]),
                        "target_dim": int(profile["target_dim"]),
                        "covariate_dim": (
                            0 if covariates is None else int(covariates.shape[1])
                        ),
                        "target_feature": conditioning.target_feature,
                        "target_percentile_level": conditioning.target_percentile_levels[
                            intensity - 1
                        ],
                        "target_strength": conditioning.target_values[
                            intensity - 1
                        ],
                        "realized_features": clean_float_mapping(features),
                        "construction_predictability": latent["predictability"],
                        "acceptance_attempts": int(acceptance["attempts"]),
                        "feature_gate": compact_feature_gate(validation["feature_gate"]),
                        "near_distance_gate": compact_near_distance_gate(
                            validation["near_distance_gate"], near_bucket
                        ),
                        "forecast_mae": forecasts,
                    }
                    samples.append(
                        {
                            "row": row,
                            "target": np.asarray(target, dtype=float),
                            "covariates": (
                                None
                                if covariates is None
                                else np.asarray(covariates, dtype=float)
                            ),
                        }
                    )
    return samples


def compact_feature_gate(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": bool(gate["accepted"]),
        "enforced": bool(gate["enforced"]),
        "status": gate["status"],
        "matched_profile_id": gate.get("matched_profile_id"),
        "score": gate.get("score"),
        "threshold": gate.get("threshold"),
        "normalized_score": gate.get("normalized_score"),
        "control_features": gate.get("control_features", {}),
    }


def matching_near_bucket(gate: dict[str, Any], profile_id: str) -> dict[str, Any]:
    matches = [
        row for row in gate.get("bucket_results", []) if row.get("profile_id") == profile_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one near-distance result for {profile_id}, got {len(matches)}")
    return matches[0]


def compact_near_distance_gate(
    gate: dict[str, Any],
    bucket: dict[str, Any],
) -> dict[str, Any]:
    names = (
        "strict_risk",
        "combined_risk",
        "full_strict_risk",
        "context_strict_risk",
        "full_combined_risk",
        "context_combined_risk",
        "raw_mae_d1",
        "raw_l2_d1",
        "context_raw_mae_d1",
        "context_raw_l2_d1",
        "feature_l2_d1",
        "raw_mae_nndr",
        "context_raw_mae_nndr",
        "thresholds",
    )
    return {
        "accepted": bool(gate["accepted"]),
        "enforced": bool(gate["enforced"]),
        "status": gate["status"],
        "profile_id": bucket["profile_id"],
        **{name: bucket[name] for name in names},
    }


def dose_response_analysis(
    sample_rows: list[dict[str, Any]],
    generator_artifact: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_rows(sample_rows, "profile_id", "capability_id", "intensity")
    rows: list[dict[str, Any]] = []
    profile_checks: list[dict[str, Any]] = []
    for (profile_id, capability_id, intensity), group in sorted(grouped.items()):
        profile = generator_artifact["profiles"][profile_id]
        capability = profile["capabilities"][capability_id]
        if not is_supported_capability(capability):
            continue
        feature_name = str(capability["target_feature"])
        values = np.asarray(
            [row["realized_features"][feature_name] for row in group], dtype=float
        )
        target_values = [float(value) for value in capability["target_values"]]
        target = target_values[int(intensity) - 1]
        scale = float(target_values[-1]) - float(target_values[0])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"unsupported non-positive target range for {profile_id}/{capability_id}"
            )
        rows.append(
            {
                "dataset_id": profile["dataset_id"],
                "profile_id": profile_id,
                "capability_id": capability_id,
                "intensity": intensity,
                "feature": feature_name,
                "sample_count": len(values),
                "target_percentile_level": capability["target_percentile_levels"][
                    int(intensity) - 1
                ],
                "dataset_local_target": target,
                "dataset_local_target_range": scale,
                "realized_mean": float(np.mean(values)),
                "realized_std": float(np.std(values)),
                "realized_p05": float(np.quantile(values, 0.05)),
                "realized_p95": float(np.quantile(values, 0.95)),
                "normalized_absolute_error": abs(float(np.mean(values)) - target) / scale,
            }
        )
    by_profile = group_rows(rows, "profile_id", "capability_id")
    for (profile_id, capability_id), group in sorted(by_profile.items()):
        ordered = sorted(group, key=lambda item: item["intensity"])
        realized = [float(item["realized_mean"]) for item in ordered]
        max_error = max(float(item["normalized_absolute_error"]) for item in ordered)
        rho = spearman(realized)
        passed = bool(
            rho + 1e-12 >= DOSE_MIN_SPEARMAN
            and realized[-1] > realized[0]
            and max_error <= DOSE_MAX_NORMALIZED_ERROR
        )
        profile_checks.append(
            {
                "dataset_id": generator_artifact["profiles"][profile_id]["dataset_id"],
                "profile_id": profile_id,
                "capability_id": capability_id,
                "spearman": rho,
                "endpoint_delta": realized[-1] - realized[0],
                "max_normalized_absolute_error": max_error,
                "passed": passed,
            }
        )
    return rows, summarize_boolean_checks(profile_checks, detail_key="profile_checks")


def selectivity_analysis(
    sample_rows: list[dict[str, Any]],
    feature_artifact: dict[str, Any],
    generator_artifact: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    control_rows: list[dict[str, Any]] = []
    paired_groups = group_rows(sample_rows, "profile_id", "capability_id")
    for (profile_id, capability_id), group in sorted(paired_groups.items()):
        support = feature_artifact["buckets"][profile_id]["capabilities"][capability_id][
            "control_support"
        ]
        scales = dict(zip(support["feature_names"], support["feature_scale"], strict=True))
        by_intensity = group_rows(group, "intensity")
        paired = group_rows(group, "round_index", "sample_index")
        for feature_name in support["feature_names"]:
            means = [
                float(
                    np.mean(
                        [row["realized_features"][feature_name] for row in by_intensity[(level,)]]
                    )
                )
                for level in INTENSITIES
            ]
            scale = max(float(scales[feature_name]), 1e-9)
            paired_shifts = [
                (
                    next(row for row in rows if row["intensity"] == 5)["realized_features"][
                        feature_name
                    ]
                    - next(row for row in rows if row["intensity"] == 1)["realized_features"][
                        feature_name
                    ]
                )
                / scale
                for rows in paired.values()
            ]
            median_abs = float(np.median(np.abs(paired_shifts)))
            median_signed = float(np.median(paired_shifts))
            passed = bool(
                median_abs <= CONTROL_MAX_MEDIAN_ABS_PAIRED_SHIFT
                and abs(median_signed) <= CONTROL_MAX_ABS_MEDIAN_SIGNED_SHIFT
            )
            control_rows.append(
                {
                    "profile_id": profile_id,
                    "capability_id": capability_id,
                    "control_feature": feature_name,
                    "mean_i1": means[0],
                    "mean_i2": means[1],
                    "mean_i3": means[2],
                    "mean_i4": means[3],
                    "mean_i5": means[4],
                    "spearman": spearman(means),
                    "median_abs_paired_i5_i1_shift_iqr": median_abs,
                    "median_signed_i5_i1_shift_iqr": median_signed,
                    "passed": passed,
                }
            )

    primary_features: dict[str, str] = {}
    for profile in generator_artifact["profiles"].values():
        for capability_id, definition in profile.get("capabilities", {}).items():
            if not is_supported_capability(definition):
                continue
            feature_name = str(definition["target_feature"])
            previous = primary_features.setdefault(capability_id, feature_name)
            if previous != feature_name:
                raise ValueError(
                    f"inconsistent target feature for {capability_id}: "
                    f"{previous!r} != {feature_name!r}"
                )
    response_rows: list[dict[str, Any]] = []
    for (profile_id, capability_id), group in sorted(paired_groups.items()):
        by_intensity = group_rows(group, "intensity")
        available = set.intersection(
            *(set(row["realized_features"]) for row in group)
        )
        for feature_capability, feature_name in primary_features.items():
            if feature_name not in available:
                continue
            means = [
                float(
                    np.mean(
                        [row["realized_features"][feature_name] for row in by_intensity[(level,)]]
                    )
                )
                for level in INTENSITIES
            ]
            response_rows.append(
                {
                    "profile_id": profile_id,
                    "generator_capability": capability_id,
                    "feature_capability": feature_capability,
                    "feature": feature_name,
                    "is_target": capability_id == feature_capability,
                    "spearman": spearman(means),
                    "endpoint_delta": means[-1] - means[0],
                }
            )
    summary = summarize_boolean_checks(control_rows, detail_key="control_checks")
    target_rhos = [row["spearman"] for row in response_rows if row["is_target"]]
    off_target_rhos = [abs(row["spearman"]) for row in response_rows if not row["is_target"]]
    summary["response_matrix"] = {
        "target_median_spearman": float(np.median(target_rhos)) if target_rhos else None,
        "off_target_median_abs_spearman": (
            float(np.median(off_target_rhos)) if off_target_rhos else None
        ),
        "off_target_p90_abs_spearman": (
            float(np.quantile(off_target_rhos, 0.90)) if off_target_rhos else None
        ),
        "row_count": len(response_rows),
    }
    return control_rows, response_rows, summary


def construction_analysis(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_rows(sample_rows, "profile_id", "capability_id")
    rows: list[dict[str, Any]] = []
    for (profile_id, capability_id), group in sorted(grouped.items()):
        validated = [
            bool(row["construction_predictability"].get("construction_validated"))
            for row in group
        ]
        rows.append(
            {
                "profile_id": profile_id,
                "capability_id": capability_id,
                "sample_count": len(group),
                "contract": group[0]["construction_predictability"].get("contract"),
                "validated_count": sum(validated),
                "validated_rate": float(np.mean(validated)),
                "passed": all(validated),
            }
        )
    return rows, summarize_boolean_checks(rows, detail_key="profile_checks")


def support_analysis(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_rows(sample_rows, "profile_id", "capability_id", "intensity")
    rows: list[dict[str, Any]] = []
    for (profile_id, capability_id, intensity), group in sorted(grouped.items()):
        first_pass = np.asarray([row["acceptance_attempts"] == 1 for row in group])
        final = np.asarray([row["feature_gate"]["accepted"] for row in group])
        scores = np.asarray(
            [row["feature_gate"]["normalized_score"] for row in group], dtype=float
        )
        rate = float(np.mean(first_pass))
        rows.append(
            {
                "profile_id": profile_id,
                "capability_id": capability_id,
                "intensity": intensity,
                "sample_count": len(group),
                "first_pass_count": int(np.sum(first_pass)),
                "first_pass_rate": rate,
                "final_support_acceptance_rate": float(np.mean(final)),
                "mean_attempts": float(np.mean([row["acceptance_attempts"] for row in group])),
                "max_attempts": max(row["acceptance_attempts"] for row in group),
                "normalized_support_score_p50": float(np.quantile(scores, 0.50)),
                "normalized_support_score_p95": float(np.quantile(scores, 0.95)),
                "passed": bool(rate >= MIN_FIRST_PASS_ACCEPTANCE_RATE and np.all(final)),
            }
        )
    summary = summarize_boolean_checks(rows, detail_key="cell_checks")
    summary["overall_first_pass_rate"] = float(
        np.mean([row["acceptance_attempts"] == 1 for row in sample_rows])
    )
    summary["maximum_attempts"] = max(row["acceptance_attempts"] for row in sample_rows)
    return rows, summary


def novelty_analysis(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_rows(sample_rows, "profile_id", "capability_id", "intensity")
    rows: list[dict[str, Any]] = []
    for (profile_id, capability_id, intensity), group in sorted(grouped.items()):
        gates = [row["near_distance_gate"] for row in group]
        raw_ratios = np.asarray(
            [gate["raw_mae_d1"] / max(gate["thresholds"]["raw_mae_p05"], 1e-12) for gate in gates]
        )
        context_ratios = np.asarray(
            [
                gate["context_raw_mae_d1"]
                / max(gate["thresholds"]["context_raw_mae_p05"], 1e-12)
                for gate in gates
            ]
        )
        nndr = np.asarray([gate["raw_mae_nndr"] for gate in gates], dtype=float)
        strict_rate = float(np.mean([gate["strict_risk"] for gate in gates]))
        combined_rate = float(np.mean([gate["combined_risk"] for gate in gates]))
        final_rate = float(np.mean([gate["accepted"] for gate in gates]))
        rows.append(
            {
                "profile_id": profile_id,
                "capability_id": capability_id,
                "intensity": intensity,
                "sample_count": len(group),
                "raw_mae_dcr_over_p05_q05": float(np.quantile(raw_ratios, 0.05)),
                "raw_mae_dcr_over_p05_p50": float(np.quantile(raw_ratios, 0.50)),
                "context_mae_dcr_over_p05_q05": float(np.quantile(context_ratios, 0.05)),
                "raw_mae_nndr_q05": float(np.quantile(nndr, 0.05)),
                "raw_mae_nndr_p50": float(np.quantile(nndr, 0.50)),
                "strict_risk_rate": strict_rate,
                "combined_risk_rate": combined_rate,
                "final_novelty_acceptance_rate": final_rate,
                "passed": bool(strict_rate == 0.0 and combined_rate == 0.0 and final_rate == 1.0),
            }
        )
    return rows, summarize_boolean_checks(rows, detail_key="cell_checks")


def repetition_analysis(
    internal_samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_rows(
        internal_samples,
        lambda item: item["row"]["profile_id"],
        lambda item: item["row"]["capability_id"],
        lambda item: item["row"]["intensity"],
    )
    rows: list[dict[str, Any]] = []
    for (profile_id, capability_id, intensity), group in sorted(grouped.items()):
        round_a = [item for item in group if item["row"]["round_index"] == 1]
        round_b = [item for item in group if item["row"]["round_index"] == 2]
        left = np.vstack([item["target"].reshape(-1) for item in round_a])
        right = np.vstack([item["target"].reshape(-1) for item in round_b])
        distances = nearest_mae_distances(right, left)
        exact_left = {array_hash(item["target"]) for item in round_a}
        rounded_left = {array_hash(np.round(item["target"], 6)) for item in round_a}
        exact_rate = float(
            np.mean([array_hash(item["target"]) in exact_left for item in round_b])
        )
        rounded_rate = float(
            np.mean(
                [array_hash(np.round(item["target"], 6)) in rounded_left for item in round_b]
            )
        )
        near_rate = float(np.mean(distances["d1"] <= 1e-6))
        passed = bool(
            exact_rate <= MAX_DUPLICATE_RATE
            and rounded_rate <= MAX_DUPLICATE_RATE
            and near_rate <= MAX_DUPLICATE_RATE
        )
        rows.append(
            {
                "profile_id": profile_id,
                "capability_id": capability_id,
                "intensity": intensity,
                "round_size": len(round_a),
                "exact_duplicate_rate": exact_rate,
                "rounded_1e6_duplicate_rate": rounded_rate,
                "near_duplicate_mae_le_1e6_rate": near_rate,
                "cross_round_dcr_q01": float(np.quantile(distances["d1"], 0.01)),
                "cross_round_dcr_q05": float(np.quantile(distances["d1"], 0.05)),
                "cross_round_dcr_p50": float(np.quantile(distances["d1"], 0.50)),
                "cross_round_nndr_q05": float(np.quantile(distances["nndr"], 0.05)),
                "cross_round_nndr_p50": float(np.quantile(distances["nndr"], 0.50)),
                "passed": passed,
            }
        )
    return rows, summarize_boolean_checks(rows, detail_key="cell_checks")


def baseline_analysis(
    sample_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = group_rows(sample_rows, "profile_id", "capability_id", "intensity")
    rows: list[dict[str, Any]] = []
    for (profile_id, capability_id, intensity), group in sorted(grouped.items()):
        naive = np.asarray([row["forecast_mae"]["naive"] for row in group], dtype=float)
        seasonal = np.asarray(
            [row["forecast_mae"]["seasonal_naive"] for row in group], dtype=float
        )
        oracle = np.asarray(
            [row["forecast_mae"]["capability_oracle"] for row in group], dtype=float
        )
        best = np.minimum(naive, seasonal)
        rows.append(
            {
                "profile_id": profile_id,
                "capability_id": capability_id,
                "intensity": intensity,
                "sample_count": len(group),
                "naive_mae_mean": float(np.mean(naive)),
                "seasonal_naive_mae_mean": float(np.mean(seasonal)),
                "capability_oracle_mae_mean": float(np.mean(oracle)),
                "oracle_vs_best_baseline_ratio": float(np.mean(oracle) / max(np.mean(best), 1e-12)),
                "oracle_win_rate": float(np.mean(oracle < best)),
            }
        )
    capability_rows: list[dict[str, Any]] = []
    for (capability_id,), group in sorted(group_rows(sample_rows, "capability_id").items()):
        oracle = np.asarray(
            [row["forecast_mae"]["capability_oracle"] for row in group], dtype=float
        )
        best = np.asarray(
            [
                min(row["forecast_mae"]["naive"], row["forecast_mae"]["seasonal_naive"])
                for row in group
            ],
            dtype=float,
        )
        win_rate = float(np.mean(oracle < best))
        capability_rows.append(
            {
                "capability_id": capability_id,
                "sample_count": len(group),
                "oracle_win_rate": win_rate,
                "oracle_vs_best_baseline_ratio": float(
                    np.mean(oracle) / max(np.mean(best), 1e-12)
                ),
                "passed": win_rate >= MIN_ORACLE_WIN_RATE,
            }
        )
    summary = summarize_boolean_checks(capability_rows, detail_key="capability_checks")
    summary["cell_rows"] = len(rows)
    return rows, summary


def distribution_analysis(
    internal_samples: list[dict[str, Any]],
    *,
    feature_artifact: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    synthetic_groups = group_rows(
        internal_samples,
        lambda item: item["row"]["profile_id"],
        lambda item: item["row"]["capability_id"],
    )
    rows: list[dict[str, Any]] = []
    for (profile_id, capability_id), group in sorted(synthetic_groups.items()):
        bucket = feature_artifact["buckets"][profile_id]
        support = bucket["capabilities"][capability_id]["control_support"]
        names = tuple(str(name) for name in support["feature_names"])
        base = {
            "dataset_id": str(bucket["dataset_id"]),
            "profile_id": profile_id,
            "capability_id": capability_id,
            "control_features": ";".join(names),
            "real_reference_count": int(support["reference_count"]),
            "real_calibration_count": int(support["calibration_count"]),
            "synthetic_count": len(group),
        }
        if not names:
            rows.append(
                {
                    **base,
                    "status": "not_applicable_no_control_features",
                    "mmd_real_vs_real": "",
                    "mmd_synthetic_vs_real": "",
                    "mmd_shifted_negative_vs_real": "",
                    "mmd_closer_than_shifted_negative": "",
                    "swd_real_vs_real": "",
                    "swd_synthetic_vs_real": "",
                    "swd_shifted_negative_vs_real": "",
                    "swd_closer_than_shifted_negative": "",
                    "passed": "",
                }
            )
            continue
        real_reference = np.asarray(support["reference_control_z"], dtype=float)
        real_calibration = np.asarray(
            support["calibration_control_z"],
            dtype=float,
        )
        expected_shape = (len(names),)
        if (
            real_reference.ndim != 2
            or real_calibration.ndim != 2
            or real_reference.shape[1:] != expected_shape
            or real_calibration.shape[1:] != expected_shape
            or len(real_reference) != int(support["reference_count"])
            or len(real_calibration) != int(support["calibration_count"])
        ):
            raise ValueError(
                f"invalid frozen E1 control vectors for {profile_id}/{capability_id}"
            )
        center = np.asarray(support["feature_center"], dtype=float)
        scale = np.maximum(np.asarray(support["feature_scale"], dtype=float), 1e-9)
        synthetic = np.vstack(
            [
                feature_vector(item["row"]["realized_features"], names, center, scale)
                for item in group
            ]
        )
        seed = _seed_for(20260716, f"distribution:{profile_id}:{capability_id}", 0)
        bandwidth = reference_bandwidth(real_reference, seed=seed)
        negative = real_calibration + NEGATIVE_CONTROL_SHIFT_IQR
        mmd_real = rbf_mmd(real_reference, real_calibration, bandwidth=bandwidth)
        mmd_synthetic = rbf_mmd(real_reference, synthetic, bandwidth=bandwidth)
        mmd_negative = rbf_mmd(real_reference, negative, bandwidth=bandwidth)
        swd_real = sliced_wasserstein_fixed(
            real_reference, real_calibration, seed=seed
        )
        swd_synthetic = sliced_wasserstein_fixed(real_reference, synthetic, seed=seed)
        swd_negative = sliced_wasserstein_fixed(real_reference, negative, seed=seed)
        mmd_passed = mmd_synthetic < mmd_negative
        swd_passed = swd_synthetic < swd_negative
        rows.append(
            {
                **base,
                "status": "evaluated",
                "mmd_real_vs_real": mmd_real,
                "mmd_synthetic_vs_real": mmd_synthetic,
                "mmd_shifted_negative_vs_real": mmd_negative,
                "mmd_closer_than_shifted_negative": mmd_passed,
                "swd_real_vs_real": swd_real,
                "swd_synthetic_vs_real": swd_synthetic,
                "swd_shifted_negative_vs_real": swd_negative,
                "swd_closer_than_shifted_negative": swd_passed,
                "passed": bool(mmd_passed and swd_passed),
            }
        )
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    summary = summarize_boolean_checks(evaluated, detail_key="profile_checks")
    summary["not_applicable_count"] = len(rows) - len(evaluated)
    summary["mmd_closer_fraction"] = float(
        np.mean([row["mmd_closer_than_shifted_negative"] for row in evaluated])
        if evaluated
        else 0.0
    )
    summary["swd_closer_fraction"] = float(
        np.mean([row["swd_closer_than_shifted_negative"] for row in evaluated])
        if evaluated
        else 0.0
    )
    summary["passed"] = bool(
        evaluated
        and
        summary["mmd_closer_fraction"] >= MIN_DISTRIBUTION_CLOSER_FRACTION
        and summary["swd_closer_fraction"] >= MIN_DISTRIBUTION_CLOSER_FRACTION
    )
    return rows, summary


def evaluate_criteria(**sections: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "dose_response": bool(sections["dose_summary"]["all_passed"]),
        "control_selectivity": bool(sections["selectivity_summary"]["all_passed"]),
        "construction_predictability": bool(
            sections["construction_summary"]["all_passed"]
        ),
        "control_feature_support": bool(sections["support_summary"]["all_passed"]),
        "distribution_mmd_swd": bool(sections["distribution_summary"]["passed"]),
        "dcr_nndr": bool(sections["novelty_summary"]["all_passed"]),
        "cross_round_repetition": bool(sections["repetition_summary"]["all_passed"]),
        "capability_oracle": bool(sections["baseline_summary"]["all_passed"]),
    }
    return {
        "checks": checks,
        "passed_count": sum(checks.values()),
        "criterion_count": len(checks),
        "overall_passed": all(checks.values()),
    }


def forecast_metrics(
    *,
    capability_id: str,
    target: np.ndarray,
    covariates: np.ndarray | None,
    context_length: int,
    season_length: int,
    latent: dict[str, Any],
) -> dict[str, float]:
    history = np.asarray(target[:context_length], dtype=float)
    future = np.asarray(target[context_length:], dtype=float)
    naive = np.repeat(history[-1:], len(future), axis=0)
    pattern = history[-min(max(1, season_length), len(history)) :]
    seasonal_naive = np.vstack([pattern[index % len(pattern)] for index in range(len(future))])
    oracle = capability_oracle_forecast(
        capability_id=capability_id,
        target=target,
        covariates=covariates,
        context_length=context_length,
        season_length=season_length,
        latent=latent,
    )
    return {
        "naive": float(np.mean(np.abs(future - naive))),
        "seasonal_naive": float(np.mean(np.abs(future - seasonal_naive))),
        "capability_oracle": float(np.mean(np.abs(future - oracle))),
    }


def capability_oracle_forecast(
    *,
    capability_id: str,
    target: np.ndarray,
    covariates: np.ndarray | None,
    context_length: int,
    season_length: int,
    latent: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(target, dtype=float)
    length = len(values)
    times = np.arange(length, dtype=float)
    periods: list[float] = [float(season_length), float(7 * season_length)]
    degree = 1
    extra: np.ndarray | None = None
    if capability_id == "trend":
        degree = 2
    elif capability_id == "multi_seasonal":
        periods = [float(value) for value in latent.get("periods", periods)] + [
            float(7 * season_length)
        ]
    elif capability_id == "time_varying_seasonality":
        period = float(season_length)
        periods = [period, 2 * period, 4 * period, 0.8 * period, 4 * period / 3]
    elif capability_id == "regime_switching":
        state = alternating_state(length, latent.get("cut_points", []))
        extra = state[:, None]
    elif capability_id == "predictable_intermittency":
        width = max(float(latent.get("pulse_width", 1.0)), 1e-6)
        pulse = np.zeros(length, dtype=float)
        for center in latent.get("pulse_centers", []):
            pulse += np.exp(-0.5 * ((times - float(center)) / width) ** 2)
        extra = pulse[:, None]
    elif capability_id == "nonlinear_persistence":
        return nonlinear_oracle_forecast(
            values,
            context_length=context_length,
            season_length=season_length,
        )
    elif capability_id == "common_factor":
        columns = []
        for channel in range(values.shape[1]):
            design = harmonic_design(
                times,
                context_length=context_length,
                periods=[*periods, float(season_length + 3 * (channel + 1))],
                degree=1,
            )
            columns.append(linear_design_forecast(values[:, channel], design, context_length))
        return np.column_stack(columns)
    elif capability_id == "hierarchical_coherence":
        children = []
        for child_index in range(1, values.shape[1]):
            design = harmonic_design(
                times,
                context_length=context_length,
                periods=[*periods, float(season_length + 2 * child_index)],
                degree=1,
            )
            children.append(
                linear_design_forecast(values[:, child_index], design, context_length)
            )
        child_forecasts = np.column_stack(children)
        return np.column_stack([np.sum(child_forecasts, axis=1), child_forecasts])
    elif capability_id == "covariate_response":
        if covariates is None:
            raise ValueError("covariate oracle requires known-future covariates")
        extra = np.asarray(covariates, dtype=float)

    design = harmonic_design(
        times,
        context_length=context_length,
        periods=periods,
        degree=degree,
        extra=extra,
    )
    return np.column_stack(
        [
            linear_design_forecast(values[:, channel], design, context_length)
            for channel in range(values.shape[1])
        ]
    )


def nonlinear_oracle_forecast(
    values: np.ndarray,
    *,
    context_length: int,
    season_length: int,
) -> np.ndarray:
    seasonal_lag = max(4, int(season_length))
    nonlinear_lag = max(2, seasonal_lag // 2)
    output = np.asarray(values[:context_length], dtype=float).copy()
    predictions: list[np.ndarray] = []
    start = seasonal_lag
    for _future_index in range(values.shape[0] - context_length):
        design_rows = []
        responses = []
        for index in range(start, len(output)):
            phase = 2 * np.pi * index / seasonal_lag
            design_rows.append(
                np.concatenate(
                    [
                        [1.0],
                        output[index - 1],
                        output[index - seasonal_lag],
                        np.sin(2.0 * output[index - nonlinear_lag]),
                        [math.sin(phase), math.cos(phase)],
                    ]
                )
            )
            responses.append(output[index])
        design = np.asarray(design_rows, dtype=float)
        response = np.asarray(responses, dtype=float)
        phase = 2 * np.pi * len(output) / seasonal_lag
        query = np.concatenate(
            [
                [1.0],
                output[-1],
                output[-seasonal_lag],
                np.sin(2.0 * output[-nonlinear_lag]),
                [math.sin(phase), math.cos(phase)],
            ]
        )
        coefficient = ridge_solve(design, response)
        prediction = query @ coefficient
        predictions.append(np.asarray(prediction, dtype=float))
        output = np.vstack([output, prediction])
    return np.vstack(predictions)


def alternating_state(length: int, cut_points: Iterable[int]) -> np.ndarray:
    state = np.ones(length, dtype=float)
    sign = 1.0
    start = 0
    for end in [*sorted(int(value) for value in cut_points), length]:
        state[start:end] = sign
        sign *= -1.0
        start = end
    return state


def harmonic_design(
    times: np.ndarray,
    *,
    context_length: int,
    periods: Iterable[float],
    degree: int,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    normalized_time = (times - (context_length - 1)) / max(context_length - 1, 1)
    columns = [np.ones(len(times), dtype=float)]
    columns.extend(normalized_time**power for power in range(1, degree + 1))
    for period in dict.fromkeys(round(float(value), 8) for value in periods if float(value) >= 2):
        angle = 2 * np.pi * times / period
        columns.extend([np.sin(angle), np.cos(angle)])
    if extra is not None:
        array = np.asarray(extra, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        columns.extend(array[:, index] for index in range(array.shape[1]))
    return np.column_stack(columns)


def linear_design_forecast(
    values: np.ndarray,
    design: np.ndarray,
    context_length: int,
) -> np.ndarray:
    coefficient = ridge_solve(design[:context_length], values[:context_length])
    return np.asarray(design[context_length:] @ coefficient, dtype=float)


def ridge_solve(design: np.ndarray, response: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ response)


def feature_vector(
    features: dict[str, float],
    names: tuple[str, ...],
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray([float(features[name]) for name in names], dtype=float)
    return (values - center) / scale


def reference_bandwidth(values: np.ndarray, *, seed: int) -> float:
    sampled = deterministic_subsample(values, 256, seed=seed)
    distances = np.sqrt(
        np.sum((sampled[:, None, :] - sampled[None, :, :]) ** 2, axis=2)
    )
    positive = distances[distances > 0]
    return max(float(np.median(positive)) if positive.size else 1.0, 1e-6)


def rbf_mmd(left: np.ndarray, right: np.ndarray, *, bandwidth: float) -> float:
    left = deterministic_subsample(left, 512, seed=101)
    right = deterministic_subsample(right, 512, seed=103)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def kernel_mean(first: np.ndarray, second: np.ndarray) -> float:
        distance_sq = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2)
        return float(np.mean(np.exp(-gamma * distance_sq)))

    return max(0.0, kernel_mean(left, left) + kernel_mean(right, right) - 2 * kernel_mean(left, right))


def sliced_wasserstein_fixed(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    projections: int = 128,
) -> float:
    count = min(len(left), len(right), 512)
    left = deterministic_subsample(left, count, seed=seed + 1)
    right = deterministic_subsample(right, count, seed=seed + 2)
    rng = np.random.default_rng(seed + 3)
    directions = rng.normal(size=(projections, left.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    distances = [
        float(np.mean(np.abs(np.sort(left @ direction) - np.sort(right @ direction))))
        for direction in directions
    ]
    return float(np.mean(distances))


def deterministic_subsample(values: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    if len(values) <= count:
        return np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indexes = np.sort(rng.choice(len(values), size=count, replace=False))
    return np.asarray(values[indexes], dtype=float)


def nearest_mae_distances(query: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    distances = np.mean(np.abs(query[:, None, :] - reference[None, :, :]), axis=2)
    part = np.partition(distances, kth=1, axis=1)
    d1 = part[:, 0]
    d2 = part[:, 1]
    return {"d1": d1, "d2": d2, "nndr": d1 / np.maximum(d2, 1e-12)}


def summarize_boolean_checks(
    rows: list[dict[str, Any]],
    *,
    detail_key: str,
) -> dict[str, Any]:
    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "check_count": len(rows),
        "passed_count": passed,
        "failed_count": len(rows) - passed,
        "pass_rate": float(passed / len(rows)) if rows else 0.0,
        "all_passed": bool(rows and passed == len(rows)),
        detail_key: rows,
    }


def group_rows(rows: Iterable[Any], *keys: Any) -> dict[tuple[Any, ...], list[Any]]:
    grouped: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in rows:
        resolved = tuple(key(row) if callable(key) else row[key] for key in keys)
        grouped[resolved].append(row)
    return grouped


def spearman(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2 or np.allclose(array, array[0]):
        return 0.0
    ranks = average_ranks(array)
    centered_x = np.arange(array.size, dtype=float) - (array.size - 1) / 2
    centered_y = ranks - float(np.mean(ranks))
    denominator = float(np.linalg.norm(centered_x) * np.linalg.norm(centered_y))
    return float(np.dot(centered_x, centered_y) / denominator) if denominator > 0 else 0.0


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2
        index = end
    return ranks


def array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def clean_float_mapping(values: dict[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in values.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    }


def render_report(summary: dict[str, Any]) -> str:
    criteria = summary["criteria"]
    capability_support = summary["capability_support"]
    dose = summary["dose_response"]
    selectivity = summary["selectivity"]
    construction = summary["construction_predictability"]
    support = summary["control_support"]
    distribution = summary["distribution_mmd_swd"]
    novelty = summary["novelty_dcr_nndr"]
    repetition = summary["cross_round_repetition"]
    baselines = summary["baseline_oracle_response"]
    lines = [
        "# E1 — Synthetic method validity",
        "",
        f"- Intensity policy: `{summary['config']['intensity_policy']['policy_id']}`; "
        "strength is relative within each dataset/profile/capability.",
        f"- Capability cells: {capability_support['supported_count']} supported; "
        f"{capability_support['unsupported_count']} unsupported and skipped.",
        f"- Samples: {summary['sample_count']} accepted samples; no large forecasting model was used.",
        f"- Overall preregistered checks: {criteria['passed_count']} / {criteria['criterion_count']} passed.",
        "",
        "## Criterion summary",
        "",
        "| Criterion | Passed |",
        "| --- | --- |",
    ]
    for name, passed in criteria["checks"].items():
        lines.append(f"| `{name}` | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Key diagnostics",
            "",
            f"- Dose-response cells: {dose['passed_count']} / {dose['check_count']} passed.",
            f"- Control selectivity checks: {selectivity['passed_count']} / {selectivity['check_count']} passed.",
            f"- Construction contracts: {construction['passed_count']} / {construction['check_count']} passed.",
            f"- First-pass online acceptance: {support['overall_first_pass_rate']:.4f}; maximum attempts: {support['maximum_attempts']}.",
            f"- MMD/SWD closer-than-shifted fractions: {distribution['mmd_closer_fraction']:.4f} / {distribution['swd_closer_fraction']:.4f}.",
            f"- Distribution cells without valid nuisance controls: {distribution['not_applicable_count']} (recorded as not applicable, not failed or silently dropped).",
            f"- DCR/NNDR cells: {novelty['passed_count']} / {novelty['check_count']} passed.",
            f"- Cross-round repetition checks: {repetition['passed_count']} / {repetition['check_count']} passed.",
            f"- Capability-oracle checks: {baselines['passed_count']} / {baselines['check_count']} passed.",
            "",
            "## Failed dose/selectivity checks",
            "",
        ]
    )
    failed_dose = [row for row in dose["profile_checks"] if not row["passed"]]
    failed_controls = [row for row in selectivity["control_checks"] if not row["passed"]]
    if not failed_dose and not failed_controls:
        lines.append("No failed dose-response or control-selectivity checks.")
    else:
        for row in failed_dose:
            lines.append(
                f"- dose `{row['profile_id']}/{row['capability_id']}`: "
                f"rho={row['spearman']:.4f}, max normalized error="
                f"{row['max_normalized_absolute_error']:.4f}."
            )
        for row in failed_controls:
            lines.append(
                f"- control `{row['profile_id']}/{row['capability_id']}/"
                f"{row['control_feature']}`: paired |i5-i1|="
                f"{row['median_abs_paired_i5_i1_shift_iqr']:.4f} IQR, signed="
                f"{row['median_signed_i5_i1_shift_iqr']:.4f} IQR."
            )
    lines.extend(
        [
            "",
            "## Capability-oracle response",
            "",
            "| Capability | Oracle win rate | Oracle / best baseline MAE | Passed |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in baselines["capability_checks"]:
        lines.append(
            f"| `{row['capability_id']}` | {row['oracle_win_rate']:.4f} | "
            f"{row['oracle_vs_best_baseline_ratio']:.4f} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The CSV files in this directory contain all profile/capability/intensity-level results; `samples.jsonl` retains per-sample features, gate evidence, and baseline errors.",
        ]
    )
    return "\n".join(lines) + "\n"


def prepare_output_dir(path: Path, *, allow_existing_empty: bool) -> None:
    if path.exists():
        existing = list(path.iterdir())
        if existing:
            raise FileExistsError(
                f"E1 output directory is immutable and already contains files: {path}"
            )
        if not allow_existing_empty:
            raise FileExistsError(
                f"E1 output directory already exists; pass --allow-existing-empty only for an empty directory: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty E1 table: {path.name}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(output_dir: Path, *, config: dict[str, Any]) -> None:
    inputs = {
        "generator_conditioning_artifact": GENERATOR_ARTIFACT_PATH,
        "feature_gate_artifact": FEATURE_GATE_ARTIFACT_PATH,
        "near_distance_artifact": NEAR_DISTANCE_ARTIFACT_PATH,
        "dataset_capability_support_matrix": SUPPORT_MATRIX_PATH,
        "runner": Path(__file__).resolve(),
    }
    files = {
        path.name: {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": "paper_experiment_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": git_head(REPO_ROOT),
        "inputs": {
            name: {
                "path": relative_or_absolute(path),
                "sha256": sha256_file(path),
            }
            for name, path in inputs.items()
        },
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "files": files,
    }
    write_json(output_dir / "manifest.json", manifest)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
