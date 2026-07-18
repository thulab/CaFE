#!/usr/bin/env python3
"""Generate the formal Paper v5 E2 H=48 master sample collection.

Each accepted paired group contains all five intensities generated from the
same attempt seed.  A group is accepted only when every intensity passes
construction, feature-support, and near-distance checks at all four context
suffixes.  The persisted statistical sample is one L=504,H=48 master curve;
model inference later expands it into L={96,168,336,504} views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.services.synthetic_feature_gate import (  # noqa: E402
    evaluate_feature_support_gate,
)
from app.services.synthetic_generation_service import (  # noqa: E402
    PAPER_GENERATOR_VERSION,
    _attempt_seed,
    _generate_sample_values,
    _normalize_covariates,
    _seed_for,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    resolve_generator_conditioning,
)
from app.services.synthetic_near_distance_gate import (  # noqa: E402
    evaluate_near_distance_gate,
)
from build_paper_v4_nine_capability_suite import (  # noqa: E402
    CONTEXT_LENGTHS,
    HORIZON,
    MAX_CONTEXT_LENGTH,
    PRIMARY_TARGET_FEATURE,
    array_sha256,
    gate_profile_id,
    synthetic_paired_view,
    synthetic_view_features,
)


SCHEMA_VERSION = "paper_v5_e2_master_sample_collection.v1"
SAMPLE_SCHEMA_VERSION = "paper_v5_e2_master_sample.v1"
DEFAULT_SUITE_DIR = (
    REPO_ROOT / "runtime/paper_exp/v5/01_nine_capability_suite"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp/v5/E2_dynamic_stability"
)
DEFAULT_ROUND_SEEDS = (
    2026071621,
    2026071622,
    2026071623,
    2026071624,
    2026071625,
)
DEFAULT_SAMPLES_PER_ROUND = 32
DEFAULT_MAX_ATTEMPTS = 512
INTENSITIES = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate formal Paper v5 E2 H=48 master samples with paired "
            "five-intensity and four-context acceptance."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--round-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_ROUND_SEEDS),
    )
    parser.add_argument(
        "--samples-per-round",
        type=int,
        default=DEFAULT_SAMPLES_PER_ROUND,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def supported_cells(
    support_artifact: dict[str, Any],
    generator_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    profiles = generator_artifact.get("profiles", {})
    cells: list[dict[str, Any]] = []
    for cell in support_artifact.get("cells", []):
        if cell.get("status") != "supported":
            continue
        profile_id = str(cell["generator_profile_id"])
        capability_id = str(cell["capability_id"])
        profile = profiles.get(profile_id)
        if not isinstance(profile, dict):
            raise ValueError(
                f"supported cell references missing profile: {profile_id}"
            )
        conditioning = resolve_generator_conditioning(
            capability_id=capability_id,
            profile_id=profile_id,
            context_length=int(profile["context_length"]),
            horizon=int(profile["horizon"]),
            target_dim=int(profile["target_dim"]),
            artifact=generator_artifact,
        )
        if conditioning is None:
            raise ValueError(
                f"supported cell has no conditioning: "
                f"{profile_id}/{capability_id}"
            )
        cells.append(
            {
                "dataset_id": str(cell["dataset_id"]),
                "task_id": str(cell["task_id"]),
                "profile_id": profile_id,
                "capability_id": capability_id,
            }
        )
    return sorted(
        cells,
        key=lambda row: (
            row["dataset_id"],
            row["task_id"],
            row["capability_id"],
        ),
    )


def compact_gate_audit(
    *,
    context_length: int,
    profile_id: str,
    primary_feature: str,
    features: dict[str, float],
    feature_gate: dict[str, Any],
    near_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "context_length": int(context_length),
        "profile_id": profile_id,
        "passed": bool(
            feature_gate["enforced"]
            and feature_gate["accepted"]
            and near_gate["enforced"]
            and near_gate["accepted"]
        ),
        "primary_feature": primary_feature,
        "primary_feature_value": finite_or_none(
            features.get(primary_feature)
        ),
        "feature_gate_status": feature_gate["status"],
        "feature_gate_normalized_score": finite_or_none(
            feature_gate.get("normalized_score")
        ),
        "near_distance_status": near_gate["status"],
        "strict_risk": near_gate.get("strict_risk"),
        "combined_risk": near_gate.get("combined_risk"),
    }


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def generate_intensity_candidate(
    *,
    capability_id: str,
    intensity: int,
    attempt_seed: int,
    profile: dict[str, Any],
    conditioning: Any,
    feature_artifact: dict[str, Any],
    near_artifact: dict[str, Any],
) -> dict[str, Any]:
    target_dim = int(profile["target_dim"])
    season_length = int(profile["season_length"])
    hierarchy = profile.get("hierarchy")
    rng = np.random.default_rng(attempt_seed)
    raw_target, metadata, raw_covariates = _generate_sample_values(
        capability_id,
        MAX_CONTEXT_LENGTH + HORIZON,
        MAX_CONTEXT_LENGTH,
        target_dim,
        season_length,
        intensity,
        rng,
        generator_conditioning=conditioning,
    )
    target = (
        _standardize_hierarchy_by_context(
            raw_target,
            MAX_CONTEXT_LENGTH,
        )
        if capability_id == "hierarchical_coherence"
        else _standardize_by_context(raw_target, MAX_CONTEXT_LENGTH)
    )
    covariates = (
        _normalize_covariates(raw_covariates, MAX_CONTEXT_LENGTH)
        if raw_covariates is not None and raw_covariates.size
        else None
    )
    construction_validated = bool(
        metadata.get("predictability", {}).get(
            "construction_validated",
            False,
        )
    )
    primary_feature = PRIMARY_TARGET_FEATURE[capability_id]
    view_audits: list[dict[str, Any]] = []
    realized_features: dict[str, dict[str, float]] = {}
    all_views_passed = construction_validated
    for context_length in CONTEXT_LENGTHS:
        view_target, view_covariates = synthetic_paired_view(
            target,
            covariates,
            context_length=context_length,
            hierarchy=hierarchy,
        )
        features = synthetic_view_features(
            capability_id=capability_id,
            target=view_target,
            covariates=view_covariates,
            season_length=season_length,
            context_length=context_length,
            latent=metadata,
        )
        view_profile_id = gate_profile_id(
            str(profile["dataset_id"]),
            str(profile["task_id"]),
            context_length,
        )
        feature_gate = evaluate_feature_support_gate(
            capability_id=capability_id,
            features=features,
            profile_ids=(view_profile_id,),
            context_length=context_length,
            horizon=HORIZON,
            target_dim=target_dim,
            artifact=feature_artifact,
        )
        near_gate = evaluate_near_distance_gate(
            target=view_target,
            features=features,
            profile_ids=(view_profile_id,),
            context_length=context_length,
            horizon=HORIZON,
            artifact=near_artifact,
        )
        audit = compact_gate_audit(
            context_length=context_length,
            profile_id=view_profile_id,
            primary_feature=primary_feature,
            features=features,
            feature_gate=feature_gate,
            near_gate=near_gate,
        )
        view_audits.append(audit)
        realized_features[str(context_length)] = {
            str(name): float(value)
            for name, value in features.items()
            if finite_or_none(value) is not None
        }
        all_views_passed = all_views_passed and bool(audit["passed"])
    return {
        "accepted": bool(all_views_passed),
        "construction_validated": construction_validated,
        "target": target,
        "covariates": covariates,
        "metadata": metadata,
        "view_audits": view_audits,
        "realized_features": realized_features,
    }


def generate_paired_group(
    *,
    capability_id: str,
    sample_seed: int,
    profile: dict[str, Any],
    conditioning: Any,
    feature_artifact: dict[str, Any],
    near_artifact: dict[str, Any],
    max_attempts: int,
) -> tuple[int, int, dict[int, dict[str, Any]]]:
    last_summary: dict[str, Any] = {}
    for attempt in range(max_attempts):
        attempt_seed = _attempt_seed(sample_seed, attempt)
        candidates = {
            intensity: generate_intensity_candidate(
                capability_id=capability_id,
                intensity=intensity,
                attempt_seed=attempt_seed,
                profile=profile,
                conditioning=conditioning,
                feature_artifact=feature_artifact,
                near_artifact=near_artifact,
            )
            for intensity in INTENSITIES
        }
        if all(candidate["accepted"] for candidate in candidates.values()):
            return attempt, attempt_seed, candidates
        last_summary = {
            str(intensity): {
                "construction_validated": candidate[
                    "construction_validated"
                ],
                "failed_contexts": [
                    audit["context_length"]
                    for audit in candidate["view_audits"]
                    if not audit["passed"]
                ],
            }
            for intensity, candidate in candidates.items()
            if not candidate["accepted"]
        }
    raise RuntimeError(
        "paired five-intensity group failed joint four-context acceptance "
        f"after {max_attempts} attempts: "
        f"capability={capability_id}, sample_seed={sample_seed}, "
        f"last_failure={json.dumps(last_summary, sort_keys=True)}"
    )


def master_sample_row(
    *,
    cell: dict[str, Any],
    profile: dict[str, Any],
    intensity: int,
    round_index: int,
    round_seed: int,
    sample_index: int,
    sample_seed: int,
    attempt: int,
    attempt_seed: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    paired_group_id = (
        f"{cell['profile_id']}__{cell['capability_id']}__"
        f"r{round_index}__s{sample_index:03d}"
    )
    sample_id = f"{paired_group_id}__i{intensity}"
    metadata = candidate["metadata"]
    conditioning = metadata["generator_conditioning"]
    target = np.asarray(candidate["target"], dtype=float)
    covariates = candidate["covariates"]
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "master_sample_id": sample_id,
        "paired_group_id": paired_group_id,
        "profile_id": cell["profile_id"],
        "dataset_id": cell["dataset_id"],
        "task_id": cell["task_id"],
        "capability_id": cell["capability_id"],
        "intensity": int(intensity),
        "round_index": int(round_index),
        "round_seed": int(round_seed),
        "sample_index": int(sample_index),
        "sample_seed": int(sample_seed),
        "paired_attempt": int(attempt),
        "paired_attempt_seed": int(attempt_seed),
        "context_length": MAX_CONTEXT_LENGTH,
        "context_lengths": list(CONTEXT_LENGTHS),
        "horizon": HORIZON,
        "season_length": int(profile["season_length"]),
        "frequency": str(profile["frequency"]),
        "target_dim": int(profile["target_dim"]),
        "covariate_dim": (
            0 if covariates is None else int(covariates.shape[1])
        ),
        "hierarchy": profile.get("hierarchy"),
        "target": target.tolist(),
        "covariates": (
            None
            if covariates is None
            else np.asarray(covariates, dtype=float).tolist()
        ),
        "target_sha256": array_sha256(target),
        "future_sha256": array_sha256(target[MAX_CONTEXT_LENGTH:]),
        "covariates_sha256": (
            None if covariates is None else array_sha256(covariates)
        ),
        "construction_validated": bool(
            candidate["construction_validated"]
        ),
        "view_qualification": candidate["view_audits"],
        "realized_features_by_context": candidate[
            "realized_features"
        ],
        "generation_metadata": metadata,
        "target_feature": conditioning["target_feature"],
        "target_strength": conditioning["target_strength"],
        "target_relative_level": conditioning.get(
            "target_relative_level",
            conditioning.get("target_percentile_level"),
        ),
        "intensity_comparability": "within_dataset_only",
        "future_view_policy": (
            "all contexts share the exact same 48-step future"
        ),
        "view_standardization_policy": (
            "each context suffix is re-standardized from its own history "
            "before model inference and scoring"
        ),
    }


def shard_path(output_dir: Path, cell: dict[str, Any]) -> Path:
    name = safe_filename(
        f"{cell['dataset_id']}__{cell['task_id']}__{cell['capability_id']}"
    )
    return output_dir / "sample_shards" / f"{name}.jsonl"


def validate_complete_shard(
    path: Path,
    *,
    cell: dict[str, Any],
    expected: int,
) -> dict[str, Any]:
    count = 0
    paired: dict[str, list[dict[str, Any]]] = {}
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row["dataset_id"] != cell["dataset_id"]
                or row["task_id"] != cell["task_id"]
                or row["capability_id"] != cell["capability_id"]
            ):
                raise ValueError(f"cell mismatch in {path}")
            if row["context_lengths"] != list(CONTEXT_LENGTHS):
                raise ValueError(f"context contract mismatch in {path}")
            if int(row["horizon"]) != HORIZON:
                raise ValueError(f"horizon contract mismatch in {path}")
            if not bool(row.get("construction_validated")):
                raise ValueError(f"unvalidated construction in {path}")
            view_qualification = row.get("view_qualification", [])
            observed_contexts = [
                int(audit["context_length"])
                for audit in view_qualification
            ]
            if observed_contexts != list(CONTEXT_LENGTHS):
                raise ValueError(f"qualified view set mismatch in {path}")
            if not all(audit["passed"] for audit in view_qualification):
                raise ValueError(f"unqualified row in {path}")
            if len(row.get("target", [])) != MAX_CONTEXT_LENGTH + HORIZON:
                raise ValueError(f"master target length mismatch in {path}")
            covariates = row.get("covariates")
            if (
                covariates is not None
                and len(covariates) != MAX_CONTEXT_LENGTH + HORIZON
            ):
                raise ValueError(f"master covariate length mismatch in {path}")
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
            sample_ids.add(sample_id)
            paired.setdefault(row["paired_group_id"], []).append(row)
            count += 1
    if count != expected:
        raise ValueError(
            f"incomplete shard {path}: observed={count}, expected={expected}"
        )
    for group_id, rows in paired.items():
        if len(rows) != len(INTENSITIES):
            raise ValueError(f"paired group row count mismatch: {group_id}")
        if {int(row["intensity"]) for row in rows} != set(INTENSITIES):
            raise ValueError(f"paired intensities incomplete: {group_id}")
        if len({int(row["paired_attempt_seed"]) for row in rows}) != 1:
            raise ValueError(f"paired attempt seed mismatch: {group_id}")
    return {
        "path": str(path),
        "row_count": count,
        "paired_group_count": len(paired),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def generate_cell_shard(
    *,
    output_dir: Path,
    cell: dict[str, Any],
    generator_artifact: dict[str, Any],
    feature_artifact: dict[str, Any],
    near_artifact: dict[str, Any],
    round_seeds: tuple[int, ...],
    samples_per_round: int,
    max_attempts: int,
) -> dict[str, Any]:
    path = shard_path(output_dir, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = (
        len(round_seeds) * samples_per_round * len(INTENSITIES)
    )
    if path.exists():
        return validate_complete_shard(
            path,
            cell=cell,
            expected=expected,
        )
    temporary = path.with_suffix(path.suffix + ".in_progress")
    if temporary.exists():
        temporary.unlink()
    profile = generator_artifact["profiles"][cell["profile_id"]]
    conditioning = resolve_generator_conditioning(
        capability_id=cell["capability_id"],
        profile_id=cell["profile_id"],
        context_length=MAX_CONTEXT_LENGTH,
        horizon=HORIZON,
        target_dim=int(profile["target_dim"]),
        artifact=generator_artifact,
    )
    if conditioning is None:
        raise RuntimeError(f"missing conditioning for {cell}")
    attempts: list[int] = []
    with temporary.open("w", encoding="utf-8") as handle:
        for round_index, round_seed in enumerate(round_seeds, start=1):
            for sample_index in range(samples_per_round):
                sample_seed = _seed_for(
                    round_seed,
                    f"{cell['profile_id']}:{cell['capability_id']}",
                    sample_index,
                )
                attempt, attempt_seed, candidates = generate_paired_group(
                    capability_id=cell["capability_id"],
                    sample_seed=sample_seed,
                    profile=profile,
                    conditioning=conditioning,
                    feature_artifact=feature_artifact,
                    near_artifact=near_artifact,
                    max_attempts=max_attempts,
                )
                attempts.append(attempt + 1)
                for intensity in INTENSITIES:
                    row = master_sample_row(
                        cell=cell,
                        profile=profile,
                        intensity=intensity,
                        round_index=round_index,
                        round_seed=round_seed,
                        sample_index=sample_index,
                        sample_seed=sample_seed,
                        attempt=attempt,
                        attempt_seed=attempt_seed,
                        candidate=candidates[intensity],
                    )
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=json_default,
                        )
                        + "\n"
                    )
            print(
                f"[generate] {cell['dataset_id']}/{cell['capability_id']} "
                f"round {round_index}/{len(round_seeds)}",
                flush=True,
            )
    os.replace(temporary, path)
    result = validate_complete_shard(
        path,
        cell=cell,
        expected=expected,
    )
    result["attempt_summary"] = {
        "minimum": min(attempts),
        "median": float(np.median(attempts)),
        "p95": float(np.quantile(attempts, 0.95)),
        "maximum": max(attempts),
    }
    return result


def concatenate_shards(
    output_dir: Path,
    shard_records: list[dict[str, Any]],
) -> Path:
    path = output_dir / "samples.jsonl"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        for record in shard_records:
            with Path(record["path"]).open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    os.replace(temporary, path)
    return path


def generate_collection(
    *,
    suite_dir: Path,
    output_dir: Path,
    round_seeds: tuple[int, ...],
    samples_per_round: int,
    max_attempts: int,
) -> dict[str, Any]:
    suite_dir = suite_dir.resolve()
    output_dir = output_dir.resolve()
    required = {
        "generator": suite_dir / "generator_conditioning_artifact.json",
        "feature": suite_dir / "feature_gate_artifact.json",
        "near": suite_dir / "near_distance_artifact.json",
        "support": suite_dir / "dataset_capability_support_matrix.json",
        "qualification": suite_dir / "qualification.json",
        "manifest": suite_dir / "manifest.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing formal suite {name}: {path}")
    generator_artifact = read_json(required["generator"])
    feature_artifact = read_json(required["feature"])
    near_artifact = read_json(required["near"])
    support_artifact = read_json(required["support"])
    qualification = read_json(required["qualification"])
    if generator_artifact.get("generator_version") != PAPER_GENERATOR_VERSION:
        raise ValueError("formal suite generator version is stale")
    if not qualification.get("all_supported_cells_qualified"):
        raise ValueError("formal calibration suite qualification is not complete")
    cells = supported_cells(support_artifact, generator_artifact)
    if not cells:
        raise ValueError("formal suite has no supported cells")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": PAPER_GENERATOR_VERSION,
        "suite_dir": str(suite_dir.relative_to(REPO_ROOT)),
        "suite_files": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
            }
            for name, path in required.items()
        },
        "context_lengths": list(CONTEXT_LENGTHS),
        "horizon": HORIZON,
        "master_shape": {
            "context_length": MAX_CONTEXT_LENGTH,
            "horizon": HORIZON,
        },
        "intensities": list(INTENSITIES),
        "round_seeds": list(round_seeds),
        "samples_per_round_per_cell": samples_per_round,
        "generation_max_attempts": max_attempts,
        "supported_cell_count": len(cells),
        "expected_paired_group_count": (
            len(cells) * len(round_seeds) * samples_per_round
        ),
        "expected_master_sample_count": (
            len(cells)
            * len(round_seeds)
            * samples_per_round
            * len(INTENSITIES)
        ),
        "expected_model_views_per_master_sample": len(CONTEXT_LENGTHS),
        "view_materialization_policy": (
            "take each history suffix from the L=504 master, retain the "
            "identical H=48 future, and re-standardize using only that "
            "suffix history"
        ),
        "paired_acceptance_rule": (
            "all five intensities share one attempt seed and every intensity "
            "passes construction plus dataset-local feature/near gates at "
            "L=96,168,336,504 with the same H=48 future"
        ),
        "context_analysis_policy": (
            "retain all per-view predictions and metrics; preregistered main "
            "summary may take the per-master-sample best context"
        ),
        "cells": cells,
    }
    config_path = output_dir / "generation_config.json"
    if config_path.exists():
        existing = read_json(config_path)
        comparable = {
            key: value
            for key, value in config.items()
            if key != "created_at"
        }
        observed = {
            key: value
            for key, value in existing.items()
            if key != "created_at"
        }
        if comparable != observed:
            raise ValueError(
                "existing generation_config.json does not match this run"
            )
    else:
        write_json(config_path, config)

    shard_records: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells, start=1):
        print(
            f"[cell {cell_index}/{len(cells)}] "
            f"{cell['dataset_id']}/{cell['task_id']}/"
            f"{cell['capability_id']}",
            flush=True,
        )
        shard_records.append(
            generate_cell_shard(
                output_dir=output_dir,
                cell=cell,
                generator_artifact=generator_artifact,
                feature_artifact=feature_artifact,
                near_artifact=near_artifact,
                round_seeds=round_seeds,
                samples_per_round=samples_per_round,
                max_attempts=max_attempts,
            )
        )
    samples_path = concatenate_shards(output_dir, shard_records)
    expected = int(config["expected_master_sample_count"])
    observed = count_jsonl(samples_path)
    if observed != expected:
        raise AssertionError(
            f"master sample count mismatch: {observed}/{expected}"
        )
    attempt_histogram: Counter[int] = Counter()
    for path in sorted((output_dir / "sample_shards").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line_number % len(INTENSITIES) == 0:
                    row = json.loads(line)
                    attempt_histogram[int(row["paired_attempt"]) + 1] += 1
    result = {
        "schema_version": "paper_v5_e2_master_sample_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": PAPER_GENERATOR_VERSION,
        "supported_cell_count": len(cells),
        "paired_group_count": int(
            config["expected_paired_group_count"]
        ),
        "master_sample_count": observed,
        "all_rows_have_four_qualified_views": True,
        "all_groups_have_paired_intensity_attempt_seed": True,
        "attempt_count_histogram": {
            str(key): value
            for key, value in sorted(attempt_histogram.items())
        },
        "files": {
            "generation_config.json": {
                "size_bytes": config_path.stat().st_size,
                "sha256": file_sha256(config_path),
            },
            "samples.jsonl": {
                "size_bytes": samples_path.stat().st_size,
                "sha256": file_sha256(samples_path),
            },
        },
        "sample_shards": shard_records,
        "generator_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "generator_sha256": file_sha256(Path(__file__)),
    }
    write_json(output_dir / "sample_manifest.json", result)
    return result


def main() -> int:
    args = parse_args()
    round_seeds = tuple(int(seed) for seed in args.round_seeds)
    if len(round_seeds) != 5 or len(set(round_seeds)) != 5:
        raise ValueError("formal E2 generation requires five unique round seeds")
    if args.samples_per_round != 32:
        raise ValueError("formal E2 generation requires 32 samples per round")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    result = generate_collection(
        suite_dir=args.suite_dir,
        output_dir=args.output_dir,
        round_seeds=round_seeds,
        samples_per_round=int(args.samples_per_round),
        max_attempts=int(args.max_attempts),
    )
    print(
        f"formal E2 master samples: {result['master_sample_count']} rows, "
        f"{result['paired_group_count']} paired groups",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
