#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

import paper_v8_pipeline_common as v8
import paper_v8_realism_gate as realism
from app.services.synthetic_v8_generation import (
    common_factor_identifiability_gate,
    cross_series_identifiability_gate,
)


DEFAULT_OUTPUT_ROOT = v8.REPO_ROOT / "runtime" / "paper_exp" / "v8"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_MAX_GENERATION_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate formal Paper v8 deterministic master samples."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Independent capability generation processes. Use 1 for the "
            "serial reference implementation."
        ),
    )
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=v8.CAPABILITIES,
        default=list(v8.CAPABILITIES),
    )
    parser.add_argument(
        "--secondary-modulus",
        type=int,
        default=4,
        help="Seeds whose stable hash is divisible by this value enter secondary/robustness.",
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=DEFAULT_MAX_GENERATION_ATTEMPTS,
        help=(
            "Maximum deterministic candidates for one capability/seed "
            "bundle, including attempt zero."
        ),
    )
    parser.add_argument(
        "--near-distance-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the lightweight anchor-internal DCR/NNDR anti-copy gate. "
            "Use --no-near-distance-gate for an explicit diagnostic bypass."
        ),
    )
    return parser.parse_args()


def selected_sensitivity_seeds(
    dataset_id: str,
    seed_indexes: list[int],
    modulus: int,
) -> set[int]:
    return {
        seed
        for seed in seed_indexes
        if v8.stable_seed(dataset_id, seed, "sensitivity") % modulus == 0
    }


def members_for(capability_id: str) -> tuple[int | None, ...]:
    return (
        (0, 1)
        if capability_id in v8.MAIN_COUNTERFACTUAL_CAPABILITIES
        else (None,)
    )


def clean_seed_bundle(
    dataset: v8.DatasetSpec,
    anchor: dict[str, Any],
    capability_calibration: dict[str, Any],
    *,
    capability_id: str,
    seed_index: int,
    sensitivity_seed: bool,
    generation_attempt: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intensity in v8.INTENSITIES:
        for member in members_for(capability_id):
            rows.append(
                v8.generate_master_sample(
                    dataset,
                    anchor,
                    capability_calibration,
                    capability_id=capability_id,
                    family_role="primary",
                    intensity=intensity,
                    seed_index=seed_index,
                    counterfactual_member=member,
                    generation_attempt=generation_attempt,
                )
            )
    if (
        capability_id in v8.STRICT_COUNTERFACTUAL_CAPABILITIES
        and sensitivity_seed
    ):
        for member in (0, 1):
            rows.append(
                v8.generate_master_sample(
                    dataset,
                    anchor,
                    capability_calibration,
                    capability_id=capability_id,
                    family_role="primary",
                    intensity=5,
                    seed_index=seed_index,
                    counterfactual_member=member,
                    evaluation_table="strict_counterfactual_audit",
                    generation_attempt=generation_attempt,
                )
            )
    if sensitivity_seed:
        for intensity in (3, 5):
            for member in members_for(capability_id):
                rows.append(
                    v8.generate_master_sample(
                        dataset,
                        anchor,
                        capability_calibration,
                        capability_id=capability_id,
                        family_role="secondary",
                        intensity=intensity,
                        seed_index=seed_index,
                        counterfactual_member=member,
                        generation_attempt=generation_attempt,
                    )
                )
    return rows


def structural_seed_bundle_gate(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    strict_rows = [
        row
        for row in rows
        if row.get("evaluation_table") == "strict_counterfactual_audit"
    ]
    if not strict_rows:
        return None
    members = {
        int(row["counterfactual_member"]): row
        for row in strict_rows
    }
    if set(members) != {0, 1}:
        return {
            "accepted": False,
            "enforced": True,
            "reason": "incomplete_strict_counterfactual_pair",
        }
    first, second = members[0], members[1]
    capability_id = str(first["capability_id"])
    arguments = {
        "first_target": np.asarray(first["target"], dtype=float),
        "second_target": np.asarray(second["target"], dtype=float),
        "context_length": int(first["context_length"]),
        "metadata": first["generation_metadata"],
        "enforced": True,
    }
    if capability_id == "common_factor":
        gate = common_factor_identifiability_gate(**arguments)
    elif capability_id == "cross_series_dependence":
        gate = cross_series_identifiability_gate(**arguments)
    else:
        return None
    for row in strict_rows:
        row["structural_generation_gate"] = gate
    return {
        "pair_id": first["counterfactual_pair_id"],
        "capability_id": capability_id,
        **gate,
    }


def iter_clean_samples(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    capability_ids: tuple[str, ...],
    seed_indexes: list[int],
    sensitivity_seeds: set[int],
    gate_context: realism.RealismGateContext,
    max_generation_attempts: int,
    attempt_audits: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for capability_id in capability_ids:
        capability_calibration = calibration["capabilities"][capability_id]
        for seed_index in seed_indexes:
            anchor = v8.anchor_for_seed(
                anchors,
                dataset_id=dataset.dataset_id,
                capability_id=capability_id,
                seed_index=seed_index,
            )
            audit_attempts: list[dict[str, Any]] = []
            for generation_attempt in range(max_generation_attempts):
                candidates = clean_seed_bundle(
                    dataset,
                    anchor,
                    capability_calibration,
                    capability_id=capability_id,
                    seed_index=seed_index,
                    sensitivity_seed=seed_index in sensitivity_seeds,
                    generation_attempt=generation_attempt,
                )
                failed_samples: list[dict[str, Any]] = []
                for row in candidates:
                    gate = realism.evaluate_sample(row, gate_context)
                    row["realism_gate"] = gate
                    row["generation_attempt"] = generation_attempt
                    if not gate["accepted"]:
                        failed_samples.append(
                            {
                                "sample_id": row["sample_id"],
                                "failure_codes": list(
                                    gate["failure_codes"]
                                ),
                                "gate": gate,
                            }
                        )
                structural_gate = structural_seed_bundle_gate(candidates)
                if (
                    structural_gate is not None
                    and not structural_gate["accepted"]
                ):
                    failed_samples.append(
                        {
                            "sample_id": structural_gate["pair_id"],
                            "failure_codes": [
                                "structural_identifiability_gate"
                            ],
                            "gate": {
                                "structural_identifiability": structural_gate
                            },
                        }
                    )
                audit_attempts.append(
                    {
                        "attempt": generation_attempt,
                        "path_seed": int(
                            candidates[0]["parameter_sampling"]["path_seed"]
                        ),
                        "accepted": not failed_samples,
                        "failed_samples": failed_samples,
                    }
                )
                if failed_samples:
                    continue
                attempt_audits.append(
                    {
                        "capability_id": capability_id,
                        "seed_index": seed_index,
                        "anchor_id": anchor["anchor_id"],
                        "selected_attempt": generation_attempt,
                        "attempts": audit_attempts,
                    }
                )
                yield from candidates
                break
            else:
                raise RuntimeError(
                    "Paper-v8 realism gates exhausted the fixed candidate "
                    f"budget for {dataset.dataset_id}/{capability_id}/"
                    f"seed={seed_index}: "
                    f"{v8.canonical_json(audit_attempts)}"
                )


def iter_input_ablations(
    clean_rows: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in clean_rows:
        if (
            row["capability_id"] not in v8.INPUT_ABLATION_CAPABILITIES
            or row["generator_family_role"] != "primary"
            or row.get("evaluation_table", "main") != "main"
        ):
            continue
        key = (str(row["capability_id"]), int(row["intensity"]))
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["seed_index"]))
        if len(rows) < 2:
            continue
        for index, clean in enumerate(rows):
            donor = rows[(index + 1) % len(rows)]
            yield v8.multivariate_input_ablation_sample(clean, donor)


def summarize_real_alignment(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "sample_count": 0,
            "feature_reference_count": 0,
            "feature_inside_count": 0,
            "target_reference_count": 0,
            "target_within_tolerance_count": 0,
            "normalized_target_errors": [],
        }
    )
    for row in rows:
        key = (
            str(row["capability_id"]),
            str(row["generator_family_role"]),
        )
        group = groups[key]
        group["sample_count"] += 1
        gate = row.get("realism_gate", {})
        feature = gate.get("feature_support", {})
        if feature.get("diagnostic_only"):
            group["feature_reference_count"] += 1
            group["feature_inside_count"] += int(
                bool(feature.get("within_reference_support"))
            )
        target = gate.get("intensity_target_match", {})
        if target.get("diagnostic_only"):
            group["target_reference_count"] += 1
            group["target_within_tolerance_count"] += int(
                bool(target.get("within_reference_tolerance"))
            )
            tolerance = float(target.get("tolerance", 0.0))
            if tolerance > 0.0:
                group["normalized_target_errors"].append(
                    float(target["absolute_error"]) / tolerance
                )
    summaries: list[dict[str, Any]] = []
    for (capability_id, family_role), group in sorted(groups.items()):
        feature_count = int(group["feature_reference_count"])
        target_count = int(group["target_reference_count"])
        errors = np.asarray(
            group.pop("normalized_target_errors"),
            dtype=float,
        )
        summaries.append(
            {
                "capability_id": capability_id,
                "family_role": family_role,
                **group,
                "feature_inside_fraction": (
                    float(group["feature_inside_count"] / feature_count)
                    if feature_count
                    else None
                ),
                "target_within_tolerance_fraction": (
                    float(
                        group["target_within_tolerance_count"]
                        / target_count
                    )
                    if target_count
                    else None
                ),
                "target_error_over_tolerance_p50": (
                    float(np.median(errors)) if errors.size else None
                ),
                "acceptance_effect": "none_diagnostic_only",
            }
        )
    return summaries


def generate_capability_shard(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    capability_id: str,
    seed_indexes: list[int],
    sensitivity_seeds: set[int],
    gate_context: realism.RealismGateContext,
    max_generation_attempts: int,
    output_path: Path,
) -> tuple[str, int, list[dict[str, Any]]]:
    attempt_audits: list[dict[str, Any]] = []
    count = v8.write_jsonl(
        output_path,
        iter_clean_samples(
            dataset,
            anchors,
            calibration,
            capability_ids=(capability_id,),
            seed_indexes=seed_indexes,
            sensitivity_seeds=sensitivity_seeds,
            gate_context=gate_context,
            max_generation_attempts=max_generation_attempts,
            attempt_audits=attempt_audits,
        ),
    )
    return capability_id, count, attempt_audits


def merge_jsonl_shards(
    output_path: Path,
    shard_paths: Iterable[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as output:
        for shard_path in shard_paths:
            with shard_path.open("rb") as source:
                shutil.copyfileobj(source, output)
    os.replace(temporary, output_path)


def generate_clean_samples(
    dataset: v8.DatasetSpec,
    anchors: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    capability_ids: tuple[str, ...],
    seed_indexes: list[int],
    sensitivity_seeds: set[int],
    gate_context: realism.RealismGateContext,
    max_generation_attempts: int,
    output_path: Path,
    workers: int,
) -> tuple[int, list[dict[str, Any]]]:
    if workers == 1 or len(capability_ids) == 1:
        attempt_audits: list[dict[str, Any]] = []
        count = v8.write_jsonl(
            output_path,
            iter_clean_samples(
                dataset,
                anchors,
                calibration,
                capability_ids=capability_ids,
                seed_indexes=seed_indexes,
                sensitivity_seeds=sensitivity_seeds,
                gate_context=gate_context,
                max_generation_attempts=max_generation_attempts,
                attempt_audits=attempt_audits,
            ),
        )
        return count, attempt_audits

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    audits: dict[str, list[dict[str, Any]]] = {}
    maximum_workers = min(workers, len(capability_ids))
    submission_order = v8.preparation_capability_order(capability_ids)
    with tempfile.TemporaryDirectory(
        prefix=".v8_capability_shards_",
        dir=output_path.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        shard_paths = {
            capability_id: temporary_root / f"{capability_id}.jsonl"
            for capability_id in capability_ids
        }
        with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
            future_capabilities = {
                executor.submit(
                    generate_capability_shard,
                    dataset,
                    anchors,
                    calibration,
                    capability_id=capability_id,
                    seed_indexes=seed_indexes,
                    sensitivity_seeds=sensitivity_seeds,
                    gate_context=gate_context,
                    max_generation_attempts=max_generation_attempts,
                    output_path=shard_paths[capability_id],
                ): capability_id
                for capability_id in submission_order
            }
            for future in as_completed(future_capabilities):
                capability_id, count, capability_audits = future.result()
                counts[capability_id] = count
                audits[capability_id] = capability_audits
                print(
                    v8.canonical_json(
                        {
                            "dataset_id": dataset.dataset_id,
                            "generated_capability": capability_id,
                            "sample_count": count,
                        }
                    ),
                    flush=True,
                )
        merge_jsonl_shards(
            output_path,
            (shard_paths[capability_id] for capability_id in capability_ids),
        )
    return (
        sum(counts.values()),
        [
            row
            for capability_id in capability_ids
            for row in audits[capability_id]
        ],
    )


def main() -> int:
    run_started = time.perf_counter()
    args = parse_args()
    if args.seed_start < 0 or args.seed_count < 1:
        raise ValueError("seed_start must be non-negative and seed_count positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.secondary_modulus < 1:
        raise ValueError("secondary modulus must be positive")
    if args.max_generation_attempts < 1:
        raise ValueError("max generation attempts must be positive")
    dataset = v8.resolve_dataset(args.dataset_id)
    dataset_root = args.output_root.resolve() / dataset.dataset_id
    calibration_dir = dataset_root / "01_calibration"
    bundle = v8.read_json(calibration_dir / "calibration_bundle.json")
    anchors = list(v8.iter_jsonl(calibration_dir / "anchors.jsonl"))
    real_anchor_masters = list(
        v8.iter_jsonl(calibration_dir / "real_anchor_masters.jsonl")
    )
    calibration = v8.read_json(
        calibration_dir / "capability_calibration.json"
    )
    if bundle["generator_version"] != v8.GENERATOR_VERSION:
        raise ValueError("calibration bundle generator version mismatch")

    seed_indexes = list(
        range(args.seed_start, args.seed_start + args.seed_count)
    )
    sensitivity_seeds = selected_sensitivity_seeds(
        dataset.dataset_id,
        seed_indexes,
        args.secondary_modulus,
    )
    generation_dir = dataset_root / "02_generation"
    shard_dir = generation_dir / "sample_shards"
    shard_name = (
        f"seed_{args.seed_start:06d}_{args.seed_start + args.seed_count:06d}"
    )
    clean_path = shard_dir / f"{shard_name}.jsonl"
    capability_ids = tuple(args.capabilities)
    gate_context = realism.build_realism_gate_context(
        anchors,
        real_anchor_masters,
        calibration,
        near_distance_enabled=bool(args.near_distance_gate),
    )
    clean_generation_started = time.perf_counter()
    clean_count, attempt_audits = generate_clean_samples(
        dataset,
        anchors,
        calibration,
        capability_ids=capability_ids,
        seed_indexes=seed_indexes,
        sensitivity_seeds=sensitivity_seeds,
        gate_context=gate_context,
        max_generation_attempts=args.max_generation_attempts,
        output_path=clean_path,
        workers=args.workers,
    )
    clean_generation_seconds = (
        time.perf_counter() - clean_generation_started
    )

    failure_counts = Counter(
        failure_code
        for group in attempt_audits
        for attempt in group["attempts"]
        if not attempt["accepted"]
        for failed_sample in attempt["failed_samples"]
        for failure_code in failed_sample["failure_codes"]
    )
    attempt_summary = {
        "group_count": len(attempt_audits),
        "retried_group_count": sum(
            int(group["selected_attempt"] > 0) for group in attempt_audits
        ),
        "maximum_selected_attempt": max(
            (int(group["selected_attempt"]) for group in attempt_audits),
            default=0,
        ),
        "selected_attempt_histogram": {
            str(attempt): sum(
                int(group["selected_attempt"] == attempt)
                for group in attempt_audits
            )
            for attempt in range(args.max_generation_attempts)
        },
        "failed_candidate_sample_counts_by_code": dict(
            sorted(failure_counts.items())
        ),
    }
    real_alignment_audit = summarize_real_alignment(
        v8.iter_jsonl(clean_path)
    )
    attempt_path = (
        generation_dir / f"generation_attempts__{shard_name}.json"
    )
    v8.write_json(
        attempt_path,
        {
            "schema_version": "paper_v8_generation_attempts.v1",
            "created_at": v8.utc_now(),
            "dataset_id": dataset.dataset_id,
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "max_generation_attempts": args.max_generation_attempts,
            "atomic_retry_unit": "capability_seed_complete_paired_bundle",
            "gate_policy": gate_context.policy_summary,
            "summary": attempt_summary,
            "groups": attempt_audits,
        },
    )

    derived_tables_started = time.perf_counter()
    ablation_path = shard_dir / f"{shard_name}__input_ablation.jsonl"
    ablation_count = v8.write_jsonl(
        ablation_path,
        iter_input_ablations(v8.iter_jsonl(clean_path)),
    )

    robustness_path = shard_dir / f"{shard_name}__robustness.jsonl"
    robustness_count = v8.write_jsonl(
        robustness_path,
        (
            v8.robustness_sample(row)
            for row in v8.iter_jsonl(clean_path)
            if row["generator_family_role"] == "primary"
            and row.get("evaluation_table", "main") == "main"
            and int(row["intensity"]) in {3, 5}
            and int(row["seed_index"]) in sensitivity_seeds
        ),
    )
    derived_tables_seconds = time.perf_counter() - derived_tables_started
    config = {
        "schema_version": "paper_v8_generation_config.v4",
        "dataset_id": dataset.dataset_id,
        "calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "generator_version": v8.GENERATOR_VERSION,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "seed_indexes": seed_indexes,
        "capabilities": list(args.capabilities),
        "realism_gate_policy": {
            **gate_context.policy_summary,
            "max_generation_attempts": args.max_generation_attempts,
            "atomic_retry_unit": "capability_seed_complete_paired_bundle",
            "attempt_zero_rng": "formal_generation_path_v1",
            "retry_rng": "stable_generation_path_retry_v1",
            "real_feature_alignment": "diagnostic_only",
            "family_target_alignment": "diagnostic_only",
        },
        "secondary_seed_policy": {
            "stable_hash_modulus": args.secondary_modulus,
            "selected_seed_indexes": sorted(sensitivity_seeds),
            "intensities": [3, 5],
        },
        "robustness_policy": {
            "source": "clean_primary",
            "selected_seed_indexes": sorted(sensitivity_seeds),
            "intensities": [3, 5],
            "history_noise_ratio": v8.ROBUSTNESS_NOISE_RATIO,
            "scoring_future": "clean_latent",
        },
        "input_ablation_policy": {
            "capabilities": sorted(v8.INPUT_ABLATION_CAPABILITIES),
            "source": "clean_primary_main",
            "donor_policy": "next_seed_same_capability_and_intensity",
            "marginal_matching": "affine_mean_and_std",
            "scoring_future": "original_clean_latent",
        },
        "strict_counterfactual_policy": {
            "capabilities": sorted(v8.STRICT_COUNTERFACTUAL_CAPABILITIES),
            "selected_seed_indexes": sorted(sensitivity_seeds),
            "intensities": [5],
            "evaluation_table": "strict_counterfactual_audit",
        },
    }
    manifest = {
        "schema_version": "paper_v8_generation_manifest.v4",
        "created_at": v8.utc_now(),
        "execution": {
            "capability_workers": min(args.workers, len(capability_ids)),
            "blas_threads_per_process": 1,
            "timing_seconds": {
                "clean_generation_and_gates": clean_generation_seconds,
                "derived_table_generation": derived_tables_seconds,
                "elapsed_before_manifest_write": (
                    time.perf_counter() - run_started
                ),
            },
        },
        "config": config,
        "config_sha256": v8.json_sha256(config),
        "files": {
            "clean": {
                **v8.file_record(clean_path),
                "row_count": clean_count,
            },
            "robustness": {
                **v8.file_record(robustness_path),
                "row_count": robustness_count,
            },
            "input_ablations": {
                **v8.file_record(ablation_path),
                "row_count": ablation_count,
            },
            "generation_attempts": v8.file_record(attempt_path),
        },
        "generation_attempt_summary": attempt_summary,
        "real_alignment_audit": {
            "policy": (
                "real_anchor_support_and_family_target_diagnostic_only_v1"
            ),
            "groups": real_alignment_audit,
        },
    }
    manifest_path = generation_dir / f"manifest__{shard_name}.json"
    v8.write_json(manifest_path, manifest)
    total_seconds = time.perf_counter() - run_started
    print(
        v8.canonical_json(
            {
                "dataset_id": dataset.dataset_id,
                "clean_sample_count": clean_count,
                "robustness_sample_count": robustness_count,
                "input_ablation_sample_count": ablation_count,
                "sensitivity_seed_count": len(sensitivity_seeds),
                "retried_group_count": attempt_summary[
                    "retried_group_count"
                ],
                "manifest": str(manifest_path),
                "timing_seconds": {
                    **manifest["execution"]["timing_seconds"],
                    "total": total_seconds,
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
