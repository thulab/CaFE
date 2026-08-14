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

from cafe import protocol
from cafe.validation import realism
from cafe.generation.families import (
    common_factor_identifiability_gate,
    cross_series_identifiability_gate,
)
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_ALPHAS,
    REAL_ANCHORED_GENERATOR_VERSION,
    available_capabilities as available_real_anchored_capabilities,
    iter_nonlinear_replay_sensitivity_samples,
    iter_real_anchored_samples,
    real_anchored_assignments,
    validate_availability_contract,
    validate_contract_integrity,
)
from cafe.generation.reference_bank import (
    validate_evaluation_qualification_policy,
    validate_real_anchored_reference_chain,
)
from cafe.generation.structural_real_counterfactuals import (
    STRUCTURAL_ALPHAS,
    STRUCTURAL_CAPABILITIES,
    available_structural_capabilities,
    available_structural_sensitivity_capabilities,
    iter_mandatory_structural_input_ablation_tasks,
    iter_structural_real_anchored_samples,
    validate_structural_donor_commitment_manifest,
    validate_structural_availability,
    validate_structural_contract,
)


DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "experiments"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_MAX_GENERATION_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate formal CaFE deterministic master samples."
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
        choices=protocol.CAPABILITIES,
        default=list(protocol.CAPABILITIES),
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
        if protocol.stable_seed(dataset_id, seed, "sensitivity") % modulus == 0
    }


def resolve_generation_capabilities(
    calibration: dict[str, Any],
    requested_capability_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    records = calibration.get("capabilities")
    if not isinstance(records, dict):
        raise ValueError("capability calibration is missing capability records")
    available: list[str] = []
    unavailable: list[dict[str, Any]] = []
    for capability_id in requested_capability_ids:
        record = records.get(capability_id)
        if not isinstance(record, dict):
            raise ValueError(
                f"capability calibration is missing {capability_id}"
            )
        is_available = bool(record.get("available_for_generation"))
        uses_real_grid = (
            record.get("intensity_calibration_scope")
            == "dataset_real_generator_overlap_reference"
        )
        no_reasons = not list(record.get("unavailable_reason_codes") or [])
        if is_available and uses_real_grid and no_reasons:
            available.append(capability_id)
            continue
        unavailable.append(
            {
                "capability_id": capability_id,
                "availability_status": str(
                    record.get("availability_status", "unavailable")
                ),
                "reason_codes": list(
                    record.get("unavailable_reason_codes")
                    or ["real_calibrated_intensity_grid_unavailable"]
                ),
                "intensity_calibration_scope": record.get(
                    "intensity_calibration_scope"
                ),
            }
        )
    return tuple(available), unavailable


def generation_path_seed(
    dataset_id: str,
    capability_id: str,
    seed_index: int,
    generation_attempt: int,
) -> int:
    if generation_attempt == 0:
        return protocol.stable_seed(
            dataset_id,
            capability_id,
            seed_index,
            "generation-path",
            base=protocol.GENERATION_PATH_SEED,
        )
    return protocol.stable_seed(
        dataset_id,
        capability_id,
        seed_index,
        "generation-path-retry",
        generation_attempt,
        base=protocol.GENERATION_PATH_SEED,
    )


def members_for(capability_id: str) -> tuple[int | None, ...]:
    return (
        (0, 1)
        if capability_id in protocol.MAIN_COUNTERFACTUAL_CAPABILITIES
        else (None,)
    )


def clean_seed_bundle(
    dataset: protocol.DatasetSpec,
    anchor: dict[str, Any],
    capability_calibration: dict[str, Any],
    *,
    capability_id: str,
    seed_index: int,
    sensitivity_seed: bool,
    generation_attempt: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intensity in protocol.INTENSITIES:
        for member in members_for(capability_id):
            rows.append(
                protocol.generate_master_sample(
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
    if capability_id in protocol.PRIMARY_MECHANISM_COUNTERFACTUAL_CAPABILITIES:
        for member in (0, 1):
            rows.append(
                protocol.generate_master_sample(
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
                    protocol.generate_master_sample(
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
    dataset: protocol.DatasetSpec,
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
            anchor = protocol.anchor_for_seed(
                anchors,
                dataset_id=dataset.dataset_id,
                capability_id=capability_id,
                seed_index=seed_index,
            )
            audit_attempts: list[dict[str, Any]] = []
            for generation_attempt in range(max_generation_attempts):
                try:
                    candidates = clean_seed_bundle(
                        dataset,
                        anchor,
                        capability_calibration,
                        capability_id=capability_id,
                        seed_index=seed_index,
                        sensitivity_seed=seed_index in sensitivity_seeds,
                        generation_attempt=generation_attempt,
                    )
                except ValueError as error:
                    audit_attempts.append(
                        {
                            "attempt": generation_attempt,
                            "path_seed": generation_path_seed(
                                dataset.dataset_id,
                                capability_id,
                                seed_index,
                                generation_attempt,
                            ),
                            "accepted": False,
                            "failed_samples": [
                                {
                                    "sample_id": (
                                        f"{dataset.dataset_id}/"
                                        f"{capability_id}/seed={seed_index}"
                                    ),
                                    "failure_codes": [
                                        "candidate_generation_error"
                                    ],
                                    "gate": {
                                        "candidate_generation": {
                                            "accepted": False,
                                            "error": (
                                                f"{type(error).__name__}: "
                                                f"{error}"
                                            ),
                                        }
                                    },
                                }
                            ],
                        }
                    )
                    continue
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
                    "Paper-cafe realism gates exhausted the fixed candidate "
                    f"budget for {dataset.dataset_id}/{capability_id}/"
                    f"seed={seed_index}: "
                    f"{protocol.canonical_json(audit_attempts)}"
                )


def iter_input_ablations(
    clean_rows: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in clean_rows:
        if (
            row["capability_id"] not in protocol.INPUT_ABLATION_CAPABILITIES
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
            yield protocol.multivariate_input_ablation_sample(clean, donor)


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
    dataset: protocol.DatasetSpec,
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
    count = protocol.write_jsonl(
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
    dataset: protocol.DatasetSpec,
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
        count = protocol.write_jsonl(
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
    submission_order = protocol.preparation_capability_order(capability_ids)
    with tempfile.TemporaryDirectory(
        prefix=".cafe_capability_shards_",
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
                    protocol.canonical_json(
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
    dataset = protocol.resolve_dataset(args.dataset_id)
    dataset_root = args.output_root.resolve() / dataset.dataset_id
    calibration_dir = dataset_root / "01_calibration"
    bundle = protocol.read_json(calibration_dir / "calibration_bundle.json")
    expected_bundle_content_sha256 = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    if bundle.get("bundle_content_sha256") != (
        expected_bundle_content_sha256
    ):
        raise ValueError("calibration bundle content hash mismatch")
    for record in bundle.get("files", {}).values():
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if protocol.file_sha256(path) != record["sha256"]:
            raise ValueError(f"calibration bundle file hash mismatch: {path}")
    anchors = list(protocol.iter_jsonl(calibration_dir / "anchors.jsonl"))
    real_anchor_masters = list(
        protocol.iter_jsonl(calibration_dir / "real_anchor_masters.jsonl")
    )
    calibration = protocol.read_json(
        calibration_dir / "capability_calibration.json"
    )
    upstream_pipeline_schema = bundle.get("pipeline_schema_version")
    if upstream_pipeline_schema not in {
        "cafe.pipeline.v1",
        "cafe.pipeline.v2",
        protocol.SCHEMA_VERSION,
    }:
        raise ValueError(
            "unsupported calibration bundle pipeline schema: "
            f"{upstream_pipeline_schema!r}"
        )
    expected_bundle_schema = {
        "cafe.pipeline.v1": "cafe.calibration_bundle.v1",
        "cafe.pipeline.v2": "cafe.calibration_bundle.v2",
        protocol.SCHEMA_VERSION: "cafe.calibration_bundle.v3",
    }[upstream_pipeline_schema]
    if bundle.get("schema_version") != expected_bundle_schema:
        raise ValueError(
            "calibration bundle schema does not match its pipeline schema: "
            f"expected {expected_bundle_schema!r}"
        )
    real_anchored_file_keys = {
        "real_anchored_backgrounds",
        "real_anchored_contracts",
        "real_anchored_availability",
    }
    present_real_anchored_file_keys = real_anchored_file_keys.intersection(
        bundle.get("files", {})
    )
    if present_real_anchored_file_keys and (
        present_real_anchored_file_keys != real_anchored_file_keys
    ):
        raise ValueError(
            "calibration bundle contains an incomplete real-anchored "
            "component"
        )
    real_anchored_files_present = bool(present_real_anchored_file_keys)
    if (
        upstream_pipeline_schema == "cafe.pipeline.v1"
        and real_anchored_files_present
    ):
        raise ValueError(
            "v1 calibration bundle must not declare the v2 real-anchored "
            "component"
        )
    if upstream_pipeline_schema in {
        "cafe.pipeline.v2",
        protocol.SCHEMA_VERSION,
    } and not real_anchored_files_present:
        raise ValueError(
            "v2/v3 calibration bundle is missing the frozen real-anchored "
            "component"
        )
    v3_real_anchored_file_keys = {
        "structural_real_anchored_backgrounds",
        "structural_real_anchored_contracts",
        "structural_real_anchored_donor_commitments",
        "structural_real_anchored_availability",
        "real_anchored_reference_backgrounds",
        "structural_real_anchored_reference_backgrounds",
        "real_anchored_reference_contracts",
        "real_anchored_bank_split_audit",
        "real_anchored_qualification_policy",
        "structural_hierarchy_qualification",
    }
    if upstream_pipeline_schema == protocol.SCHEMA_VERSION:
        missing_v3 = sorted(
            v3_real_anchored_file_keys - set(bundle.get("files", {}))
        )
        if missing_v3:
            raise ValueError(
                "v3 calibration bundle is missing real-anchored files: "
                + ", ".join(missing_v3)
            )
    if real_anchored_files_present:
        real_anchored_backgrounds = list(
            protocol.iter_jsonl(
                Path(bundle["files"]["real_anchored_backgrounds"]["path"])
            )
        )
        real_anchored_contracts = list(
            protocol.iter_jsonl(
                Path(bundle["files"]["real_anchored_contracts"]["path"])
            )
        )
        real_anchored_availability = protocol.read_json(
            Path(bundle["files"]["real_anchored_availability"]["path"])
        )
        for contract_row in real_anchored_contracts:
            validate_contract_integrity(contract_row)
        validate_availability_contract(
            real_anchored_availability,
            real_anchored_contracts,
        )
        if upstream_pipeline_schema == protocol.SCHEMA_VERSION:
            structural_backgrounds = list(
                protocol.iter_jsonl(
                    Path(
                        bundle["files"][
                            "structural_real_anchored_backgrounds"
                        ]["path"]
                    )
                )
            )
            structural_contracts = list(
                protocol.iter_jsonl(
                    Path(
                        bundle["files"][
                            "structural_real_anchored_contracts"
                        ]["path"]
                    )
                )
            )
            structural_by_id = {
                str(row["background_id"]): row
                for row in structural_backgrounds
            }
            for row in structural_contracts:
                contract = row.get("contract")
                if isinstance(contract, dict):
                    validate_structural_contract(
                        contract,
                        structural_by_id[str(row["background_id"])],
                    )
            structural_donor_commitments = protocol.read_json(
                Path(
                    bundle["files"][
                        "structural_real_anchored_donor_commitments"
                    ]["path"]
                )
            )
            validate_structural_donor_commitment_manifest(
                structural_donor_commitments,
                structural_backgrounds,
                structural_contracts,
                dataset_id=dataset.dataset_id,
            )
            structural_availability = protocol.read_json(
                Path(
                    bundle["files"][
                        "structural_real_anchored_availability"
                    ]["path"]
                )
            )
            validate_structural_availability(
                structural_availability,
                structural_contracts,
            )
            qualification_policy = protocol.read_json(
                Path(
                    bundle["files"]["real_anchored_qualification_policy"][
                        "path"
                    ]
                )
            )
            reference_backgrounds = list(
                protocol.iter_jsonl(
                    Path(
                        bundle["files"][
                            "real_anchored_reference_backgrounds"
                        ]["path"]
                    )
                )
            )
            structural_reference_backgrounds = list(
                protocol.iter_jsonl(
                    Path(
                        bundle["files"][
                            "structural_real_anchored_reference_backgrounds"
                        ]["path"]
                    )
                )
            )
            reference_contracts = list(
                protocol.iter_jsonl(
                    Path(
                        bundle["files"][
                            "real_anchored_reference_contracts"
                        ]["path"]
                    )
                )
            )
            bank_split_audit = protocol.read_json(
                Path(
                    bundle["files"]["real_anchored_bank_split_audit"][
                        "path"
                    ]
                )
            )
            validate_real_anchored_reference_chain(
                [*real_anchored_backgrounds, *structural_backgrounds],
                [
                    *reference_backgrounds,
                    *structural_reference_backgrounds,
                ],
                bank_split_audit,
                qualification_policy,
                reference_contract_rows=reference_contracts,
            )
            validate_evaluation_qualification_policy(
                [*real_anchored_contracts, *structural_contracts],
                qualification_policy,
            )
        else:
            structural_backgrounds = []
            structural_contracts = []
            structural_donor_commitments = None
            structural_availability = {}
            qualification_policy = None
    else:
        real_anchored_backgrounds = []
        real_anchored_contracts = []
        real_anchored_availability = {
            "schema_version": "cafe.real_anchored_availability.v2",
            "benchmark_track": "real_anchored_counterfactual",
            "dataset_id": dataset.dataset_id,
            "cells": [],
        }
        structural_backgrounds = []
        structural_contracts = []
        structural_donor_commitments = None
        structural_availability = {}
        qualification_policy = None
    if bundle["generator_version"] != protocol.GENERATOR_VERSION:
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
    requested_capability_ids = tuple(args.capabilities)
    capability_ids, unavailable_capabilities = (
        resolve_generation_capabilities(
            calibration,
            requested_capability_ids,
        )
    )
    real_anchored_capability_ids = (
        tuple(
            capability_id
            for capability_id in available_real_anchored_capabilities(
                real_anchored_availability
            )
            if capability_id in requested_capability_ids
        )
        if upstream_pipeline_schema == protocol.SCHEMA_VERSION
        else ()
    )
    real_anchored_assignment_map = real_anchored_assignments(
        real_anchored_contracts,
        capability_ids=real_anchored_capability_ids,
        seed_indexes=seed_indexes,
    )
    generated_real_anchored_capability_ids = tuple(
        capability_id
        for capability_id in real_anchored_capability_ids
        if real_anchored_assignment_map[capability_id]
    )
    available_structural_ids = set(
        available_structural_capabilities(structural_availability)
    )
    available_structural_sensitivity_ids = set(
        available_structural_sensitivity_capabilities(
            structural_availability
        )
    )
    structural_capability_ids = tuple(
        capability_id
        for capability_id in requested_capability_ids
        if capability_id in STRUCTURAL_CAPABILITIES
        and capability_id != "hierarchical_coherence"
        and capability_id in available_structural_ids
    )
    structural_main_rows = list(
        iter_structural_real_anchored_samples(
            structural_backgrounds,
            [
                row
                for row in structural_contracts
                if row.get("capability_id") in structural_capability_ids
            ],
            alphas=STRUCTURAL_ALPHAS,
            seed_indexes=seed_indexes,
        )
    )
    structural_sensitivity_capability_ids = tuple(
        capability_id
        for capability_id in requested_capability_ids
        if capability_id in {"common_factor", "cross_series_dependence"}
        and capability_id in available_structural_sensitivity_ids
    )
    structural_sensitivity_rows = list(
        iter_structural_real_anchored_samples(
            structural_backgrounds,
            [
                row
                for row in structural_contracts
                if row.get("capability_id")
                in structural_sensitivity_capability_ids
            ],
            alphas=STRUCTURAL_ALPHAS,
            sensitivity=True,
            seed_indexes=seed_indexes,
        )
    )
    structural_donor_rows = list(
        iter_structural_real_anchored_samples(
            structural_backgrounds,
            [
                row
                for row in structural_contracts
                if row.get("capability_id") in structural_capability_ids
            ],
            alphas=STRUCTURAL_ALPHAS,
            seed_indexes=range(len(structural_backgrounds)),
        )
    )
    structural_sensitivity_donor_rows = list(
        iter_structural_real_anchored_samples(
            structural_backgrounds,
            [
                row
                for row in structural_contracts
                if row.get("capability_id")
                in structural_sensitivity_capability_ids
            ],
            alphas=STRUCTURAL_ALPHAS,
            sensitivity=True,
            seed_indexes=range(len(structural_backgrounds)),
        )
    )
    if (
        structural_donor_rows or structural_sensitivity_donor_rows
    ) and structural_donor_commitments is None:
        raise ValueError(
            "structural ablation donors lack calibration commitments"
        )
    structural_ablation_rows = list(
        iter_mandatory_structural_input_ablation_tasks(
            structural_main_rows,
            donor_samples=structural_donor_rows,
            donor_commitment_manifest=structural_donor_commitments,
        )
    )
    structural_sensitivity_ablation_rows = list(
        iter_mandatory_structural_input_ablation_tasks(
            structural_sensitivity_rows,
            donor_samples=structural_sensitivity_donor_rows,
            donor_commitment_manifest=structural_donor_commitments,
        )
    )
    del structural_donor_rows
    del structural_sensitivity_donor_rows
    for row in (
        *structural_sensitivity_rows,
        *structural_sensitivity_ablation_rows,
    ):
        row["excluded_from_primary_score"] = True
        row["excluded_from_univariate_real_anchored_rank"] = True
    nonlinear_replay_rows = list(
        iter_nonlinear_replay_sensitivity_samples(
            real_anchored_backgrounds,
            real_anchored_contracts,
            seed_indexes=seed_indexes,
            alphas=REAL_ANCHORED_ALPHAS,
        )
    ) if "nonlinear_persistence" in real_anchored_capability_ids else []
    generated_structural_capability_ids = tuple(
        capability_id
        for capability_id in structural_capability_ids
        if any(
            row["capability_id"] == capability_id
            for row in structural_main_rows
        )
    )
    if (
        not capability_ids
        and not generated_real_anchored_capability_ids
        and not generated_structural_capability_ids
        and not real_anchor_masters
    ):
        raise ValueError(
            "none of the requested capabilities has a sample in this seed "
            "range in either the synthetic or real-anchored track; "
            "synthetic_unavailable="
            f"{protocol.canonical_json(unavailable_capabilities)}"
        )
    gate_context = realism.build_realism_gate_context(
        anchors,
        real_anchor_masters,
        calibration,
        near_distance_enabled=bool(args.near_distance_gate),
    )
    clean_generation_started = time.perf_counter()
    if capability_ids:
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
    else:
        clean_count = protocol.write_jsonl(clean_path, ())
        attempt_audits = []
    clean_generation_seconds = (
        time.perf_counter() - clean_generation_started
    )

    real_anchored_started = time.perf_counter()
    real_anchored_path = (
        shard_dir / f"{shard_name}__real_anchored_counterfactual.jsonl"
    )
    real_anchored_count = protocol.write_jsonl(
        real_anchored_path,
        (
            row
            for source in (
                iter_real_anchored_samples(
                    real_anchored_backgrounds,
                    real_anchored_contracts,
                    capability_ids=real_anchored_capability_ids,
                    seed_indexes=seed_indexes,
                    alphas=REAL_ANCHORED_ALPHAS,
                ),
                iter(structural_main_rows),
                iter(structural_ablation_rows),
                iter(structural_sensitivity_rows),
                iter(structural_sensitivity_ablation_rows),
                iter(nonlinear_replay_rows),
            )
            for row in source
        ),
    )
    expected_real_anchored_count = (
        sum(
            len(assignments)
            for assignments in real_anchored_assignment_map.values()
        )
        * len(REAL_ANCHORED_ALPHAS)
        * 2
        + len(structural_main_rows)
        + len(structural_ablation_rows)
        + len(structural_sensitivity_rows)
        + len(structural_sensitivity_ablation_rows)
        + len(nonlinear_replay_rows)
    )
    if real_anchored_count != expected_real_anchored_count:
        raise ValueError(
            "real-anchored generation count disagrees with frozen "
            f"availability: {real_anchored_count} != "
            f"{expected_real_anchored_count}"
        )
    real_anchored_generation_seconds = (
        time.perf_counter() - real_anchored_started
    )
    real_anchored_availability_path = generation_dir / (
        f"real_anchored_availability__{shard_name}.json"
    )
    generation_real_anchored_availability = {
        **real_anchored_availability,
        "source_calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "requested_seed_indexes": seed_indexes,
        "generated_capabilities": list(
            (
                *generated_real_anchored_capability_ids,
                *generated_structural_capability_ids,
            )
        ),
        "assigned_seed_indexes_by_capability": {
            capability_id: [
                seed_index for seed_index, _row in assignments
            ]
            for capability_id, assignments in (
                real_anchored_assignment_map.items()
            )
        },
        "effective_background_count_by_capability": {
            capability_id: len(assignments)
            for capability_id, assignments in (
                real_anchored_assignment_map.items()
            )
        },
        "background_sampling_policy": (
            "frozen_global_seed_ordinal_permutation_without_replacement_"
            "truncate_at_eligible_count_v1"
        ),
        "generated_master_count": real_anchored_count,
        "generated_univariate_master_count": (
            real_anchored_count
            - len(structural_main_rows)
            - len(structural_ablation_rows)
            - len(structural_sensitivity_rows)
            - len(structural_sensitivity_ablation_rows)
            - len(nonlinear_replay_rows)
        ),
        "generated_structural_main_count": len(structural_main_rows),
        "generated_structural_input_ablation_count": len(
            structural_ablation_rows
        ),
        "generated_structural_sensitivity_main_count": len(
            structural_sensitivity_rows
        ),
        "generated_structural_sensitivity_input_ablation_count": len(
            structural_sensitivity_ablation_rows
        ),
        "generated_nonlinear_replay_sensitivity_count": len(
            nonlinear_replay_rows
        ),
        "structural_input_ablation_policy": (
            "mandatory_common_cross_separate_attribution_not_score_weighted"
        ),
        "hierarchical_coherence_generation_count": 0,
        "dose_values": list(REAL_ANCHORED_ALPHAS),
    }
    protocol.write_json(
        real_anchored_availability_path,
        generation_real_anchored_availability,
    )
    structural_real_anchored_availability_path = generation_dir / (
        f"structural_real_anchored_availability__{shard_name}.json"
    )
    structural_assigned_seeds = {
        capability_id: sorted(
            {
                int(row["seed_index"])
                for row in structural_main_rows
                if row["capability_id"] == capability_id
            }
        )
        for capability_id in structural_capability_ids
    }
    generation_structural_availability = {
        **structural_availability,
        "source_calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "requested_seed_indexes": seed_indexes,
        "generated_capabilities": list(
            generated_structural_capability_ids
        ),
        "assigned_seed_indexes_by_capability": structural_assigned_seeds,
        "effective_background_count_by_capability": {
            capability_id: len(assigned)
            for capability_id, assigned in structural_assigned_seeds.items()
        },
        "background_sampling_policy": (
            "frozen_global_seed_ordinal_permutation_without_replacement_"
            "truncate_at_eligible_count_v1"
        ),
        "generated_main_master_count": len(structural_main_rows),
        "generated_input_ablation_master_count": len(
            structural_ablation_rows
        ),
        "generated_sensitivity_capabilities": list(
            structural_sensitivity_capability_ids
        ),
        "generated_sensitivity_main_master_count": len(
            structural_sensitivity_rows
        ),
        "generated_sensitivity_input_ablation_master_count": len(
            structural_sensitivity_ablation_rows
        ),
        "hierarchical_coherence_generation_count": 0,
        "dose_values": list(STRUCTURAL_ALPHAS),
    }
    protocol.write_json(
        structural_real_anchored_availability_path,
        generation_structural_availability,
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
        protocol.iter_jsonl(clean_path)
    )
    attempt_path = (
        generation_dir / f"generation_attempts__{shard_name}.json"
    )
    protocol.write_json(
        attempt_path,
        {
            "schema_version": "cafe.generation_attempts.v1",
            "created_at": protocol.utc_now(),
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
    ablation_count = protocol.write_jsonl(
        ablation_path,
        iter_input_ablations(protocol.iter_jsonl(clean_path)),
    )

    robustness_path = shard_dir / f"{shard_name}__robustness.jsonl"
    robustness_count = protocol.write_jsonl(
        robustness_path,
        (
            protocol.robustness_sample(row)
            for row in protocol.iter_jsonl(clean_path)
            if row["generator_family_role"] == "primary"
            and row.get("evaluation_table", "main") == "main"
            and int(row["intensity"]) in {3, 5}
            and int(row["seed_index"]) in sensitivity_seeds
        ),
    )
    derived_tables_seconds = time.perf_counter() - derived_tables_started
    config = {
        "schema_version": "cafe.generation_config.v3",
        "dataset_id": dataset.dataset_id,
        "calibration_bundle_sha256": bundle["bundle_content_sha256"],
        "generator_version": protocol.GENERATOR_VERSION,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "seed_indexes": seed_indexes,
        "requested_capabilities": list(requested_capability_ids),
        "capabilities": list(capability_ids),
        "unavailable_capabilities": unavailable_capabilities,
        "capability_selection_policy": (
            "generate_only_explicit_real_calibrated_intensity_grids_v1"
        ),
        "benchmark_tracks": [
            "deterministic_synthetic",
            "real_anchored_counterfactual",
        ],
        "real_anchored_counterfactual": {
            "generator_version": REAL_ANCHORED_GENERATOR_VERSION,
            "calibrated_available_capabilities": list(
                real_anchored_capability_ids
            ),
            "generated_capabilities": list(
                (
                    *generated_real_anchored_capability_ids,
                    *generated_structural_capability_ids,
                )
            ),
            "effective_background_count_by_capability": {
                capability_id: len(assignments)
                for capability_id, assignments in (
                    real_anchored_assignment_map.items()
                )
            },
            "background_sampling": (
                "frozen_global_seed_ordinal_permutation_without_replacement_"
                "truncate_at_eligible_count_v1"
            ),
            "alpha_grid": list(REAL_ANCHORED_ALPHAS),
            "pairing": "baseline_alpha1_vs_treatment_each_alpha",
            "normalization": "shared_unmodified_real_l336_history",
            "anti_copy": (
                "not_applicable_intentional_real_anchor_counterfactual"
            ),
            "qualification_policy_sha256": (
                None
                if qualification_policy is None
                else qualification_policy["qualification_policy_sha256"]
            ),
            "qualification_threshold_source": (
                (
                    "independent_source_time_disjoint_reference_bank_never_"
                    "evaluation_origins_v1"
                )
                if upstream_pipeline_schema == protocol.SCHEMA_VERSION
                else None
            ),
            "upstream_real_anchored_protocol": upstream_pipeline_schema,
            "legacy_upstream_component_policy": (
                None
                if upstream_pipeline_schema == protocol.SCHEMA_VERSION
                else "validated_but_not_regenerated_or_ranked_as_v3"
            ),
            "structural_main_count": len(structural_main_rows),
            "structural_input_ablation_count": len(
                structural_ablation_rows
            ),
            "structural_sensitivity_capabilities": list(
                structural_sensitivity_capability_ids
            ),
            "structural_sensitivity_main_count": len(
                structural_sensitivity_rows
            ),
            "structural_sensitivity_input_ablation_count": len(
                structural_sensitivity_ablation_rows
            ),
            "nonlinear_replay_sensitivity_count": len(
                nonlinear_replay_rows
            ),
            "structural_input_ablation": (
                "mandatory_common_cross_separate_attribution_never_score_"
                "weighted_v1"
            ),
            "structural_donor_commitment": (
                None
                if structural_donor_commitments is None
                else {
                    "schema_version": structural_donor_commitments[
                        "schema_version"
                    ],
                    "commitment_policy": structural_donor_commitments[
                        "commitment_policy"
                    ],
                    "commitment_root_sha256": structural_donor_commitments[
                        "commitment_root_sha256"
                    ],
                    "source_calibration_bundle_sha256": bundle[
                        "bundle_content_sha256"
                    ],
                    "source_file_sha256": bundle["files"][
                        "structural_real_anchored_donor_commitments"
                    ]["sha256"],
                }
            ),
            "formal_panel_minimum_dimension": 3,
            "panel_d2_policy": (
                "generated_auxiliary_sensitivity_never_formal_rank_v1"
            ),
            "hierarchy_policy": "qualification_only_zero_generation_rows",
            "included_in_synthetic_ranking": False,
        },
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
            "noise_to_clean_mase_scale_ratio": (
                protocol.ROBUSTNESS_NOISE_RATIO
            ),
            "noise_scale_source": (
                "clean_l336_mase_denominator_by_target"
            ),
            "scoring_future": "clean_latent",
        },
        "input_ablation_policy": {
            "capabilities": sorted(protocol.INPUT_ABLATION_CAPABILITIES),
            "source": "clean_primary_main",
            "donor_policy": "next_seed_same_capability_and_intensity",
            "marginal_matching": {
                "common_factor": "replaced_segment_mean_and_std",
                "cross_series_dependence": (
                    "pair_invariant_driver_prefix_mean_and_std"
                ),
            },
            "scoring_future": "original_clean_latent",
        },
        "strict_counterfactual_policy": {
            "capabilities": sorted(protocol.STRICT_COUNTERFACTUAL_CAPABILITIES),
            "selected_seed_indexes": seed_indexes,
            "intensities": [5],
            "evaluation_table": "strict_counterfactual_audit",
            "ranking_role": "primary_mechanism_score",
            "seed_policy": "all_formal_seeds",
        },
    }
    manifest = {
        "schema_version": "cafe.generation_manifest.v3",
        "created_at": protocol.utc_now(),
        "execution": {
            "capability_workers": min(args.workers, len(capability_ids)),
            "blas_threads_per_process": 1,
            "timing_seconds": {
                "clean_generation_and_gates": clean_generation_seconds,
                "real_anchored_generation": (
                    real_anchored_generation_seconds
                ),
                "derived_table_generation": derived_tables_seconds,
                "elapsed_before_manifest_write": (
                    time.perf_counter() - run_started
                ),
            },
        },
        "config": config,
        "config_sha256": protocol.json_sha256(config),
        "files": {
            "clean": {
                **protocol.file_record(clean_path),
                "row_count": clean_count,
            },
            "robustness": {
                **protocol.file_record(robustness_path),
                "row_count": robustness_count,
            },
            "input_ablations": {
                **protocol.file_record(ablation_path),
                "row_count": ablation_count,
            },
            "real_anchored_counterfactuals": {
                **protocol.file_record(real_anchored_path),
                "row_count": real_anchored_count,
            },
            "real_anchored_availability": protocol.file_record(
                real_anchored_availability_path
            ),
            "structural_real_anchored_availability": protocol.file_record(
                structural_real_anchored_availability_path
            ),
            **(
                {}
                if structural_donor_commitments is None
                else {
                    "structural_donor_commitments": {
                        **bundle["files"][
                            "structural_real_anchored_donor_commitments"
                        ],
                        "commitment_root_sha256": (
                            structural_donor_commitments[
                                "commitment_root_sha256"
                            ]
                        ),
                        "source_calibration_bundle_sha256": bundle[
                            "bundle_content_sha256"
                        ],
                    }
                }
            ),
            "generation_attempts": protocol.file_record(attempt_path),
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
    protocol.write_json(manifest_path, manifest)
    total_seconds = time.perf_counter() - run_started
    print(
        protocol.canonical_json(
            {
                "dataset_id": dataset.dataset_id,
                "clean_sample_count": clean_count,
                "robustness_sample_count": robustness_count,
                "input_ablation_sample_count": ablation_count,
                "real_anchored_sample_count": real_anchored_count,
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
