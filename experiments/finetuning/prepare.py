#!/usr/bin/env python3
"""Prepare disjoint CaFE seed batches for Chronos-2 fine-tuning.

The exported records use the fixed-context main-table task:

    context = master_target[168:336]
    labels = master_target[336:384]

Only primary-family main-table samples are included. Secondary-family records
and strict counterfactual audit records are intentionally excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "cafe.chronos_finetune_split.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-index", type=int, default=32)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            payload = (
                json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
            ).encode()
            handle.write(payload)
            digest.update(payload)
            count += 1
    return count, digest.hexdigest()


def source_shards(
    source_experiment: Path,
    dataset_ids: list[str],
) -> list[Path]:
    paths: list[Path] = []
    for dataset_id in dataset_ids:
        shard_dir = source_experiment / dataset_id / "02_generation" / "sample_shards"
        candidates = sorted(
            path
            for path in shard_dir.glob("seed_*.jsonl")
            if "__" not in path.stem
        )
        if not candidates:
            raise FileNotFoundError(f"no clean generation shard in {shard_dir}")
        paths.extend(candidates)
    return paths


def selected_rows(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        for row in iter_jsonl(path):
            if row.get("evaluation_table") != "main":
                continue
            if row.get("generator_family_role") != "primary":
                continue
            yield row


def compact_record(
    row: dict[str, Any],
    *,
    split: str,
    source_experiment_id: str,
    source_protocol_sha256: str,
    master_context_length: int,
    fixed_context_length: int,
    horizon: int,
) -> dict[str, Any]:
    target = row["target"]
    expected_length = master_context_length + horizon
    if len(target) != expected_length:
        raise ValueError(
            f"{row['sample_id']}: target length {len(target)} != {expected_length}"
        )
    start = master_context_length - fixed_context_length
    compact_target = target[start:]
    covariates = row.get("covariates")
    compact_covariates = None if covariates is None else covariates[start:]
    if compact_covariates is not None and len(compact_covariates) != len(
        compact_target
    ):
        raise ValueError(f"{row['sample_id']}: target/covariate length mismatch")

    return {
        "schema_version": SCHEMA_VERSION,
        "source_experiment_id": source_experiment_id,
        "source_protocol_sha256": source_protocol_sha256,
        "split": split,
        "sample_id": row["sample_id"],
        "master_sample_id": row["master_sample_id"],
        "paired_group_id": row["paired_group_id"],
        "dataset_id": row["dataset_id"],
        "capability_id": row["capability_id"],
        "intensity": int(row["intensity"]),
        "seed_index": int(row["seed_index"]),
        "counterfactual_member": row.get("counterfactual_member"),
        "target_dim": int(row["target_dim"]),
        "covariate_dim": int(row["covariate_dim"]),
        "covariate_column_names": list(row.get("covariate_column_names") or []),
        "frequency": row["frequency"],
        "season_length": int(row["season_length"]),
        "mase_scale_by_target": [
            float(value) for value in row["mase_scale_by_target"]
        ],
        "context_length": fixed_context_length,
        "horizon": horizon,
        "target": compact_target,
        "covariates": compact_covariates,
    }


def validate_coverage(
    rows: list[dict[str, Any]],
    *,
    seed_start: int,
    seed_count: int,
) -> dict[str, Any]:
    expected_seeds = set(range(seed_start, seed_start + seed_count))
    grouped: dict[tuple[str, str, int], Counter[int]] = defaultdict(Counter)
    for row in rows:
        key = (
            str(row["dataset_id"]),
            str(row["capability_id"]),
            int(row["intensity"]),
        )
        grouped[key][int(row["seed_index"])] += 1

    multiplicities: Counter[int] = Counter()
    for key, seed_counts in grouped.items():
        if set(seed_counts) != expected_seeds:
            missing = sorted(expected_seeds - set(seed_counts))
            extra = sorted(set(seed_counts) - expected_seeds)
            raise ValueError(f"{key}: seed coverage mismatch; missing={missing}, extra={extra}")
        observed = set(seed_counts.values())
        if len(observed) != 1:
            raise ValueError(f"{key}: unequal per-seed multiplicity {seed_counts}")
        multiplicities[next(iter(observed))] += 1

    return {
        "dataset_capability_intensity_groups": len(grouped),
        "dataset_capability_cells": len({key[:2] for key in grouped}),
        "per_seed_multiplicity_group_counts": {
            str(key): value for key, value in sorted(multiplicities.items())
        },
    }


def main() -> None:
    args = parse_args()
    source_experiment = args.source_experiment.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")

    experiment_manifest_path = source_experiment / "experiment_manifest.json"
    experiment_manifest = json.loads(experiment_manifest_path.read_text())
    protocol = experiment_manifest["protocol"]
    seed_start = int(protocol["seed_start"])
    seed_count = int(protocol["seed_count"])
    if seed_count != 64:
        raise ValueError(f"expected 64 source seeds, found {seed_count}")
    if not seed_start < args.split_index < seed_start + seed_count:
        raise ValueError("--split-index must fall strictly inside the seed range")

    master_context_length = int(protocol["synthetic_master_context_length"])
    fixed_context_length = int(protocol["fixed_context_length"])
    horizon = int(protocol["horizon"])
    if (master_context_length, fixed_context_length, horizon) != (336, 168, 48):
        raise ValueError(
            "this exporter requires the CaFE 336/168/48 protocol, found "
            f"{master_context_length}/{fixed_context_length}/{horizon}"
        )

    paths = source_shards(source_experiment, list(protocol["dataset_ids"]))
    rows = list(selected_rows(paths))
    coverage = validate_coverage(
        rows,
        seed_start=seed_start,
        seed_count=seed_count,
    )
    split_seeds = {
        "A": list(range(seed_start, args.split_index)),
        "B": list(range(args.split_index, seed_start + seed_count)),
    }
    if any(len(values) != 32 for values in split_seeds.values()):
        raise ValueError(f"expected a 32/32 split, found {split_seeds}")

    output_dir.mkdir(parents=True)
    files: dict[str, Any] = {}
    for split, seeds in split_seeds.items():
        seed_set = set(seeds)
        split_rows = [
            compact_record(
                row,
                split=split,
                source_experiment_id=str(experiment_manifest["experiment_id"]),
                source_protocol_sha256=str(experiment_manifest["protocol_sha256"]),
                master_context_length=master_context_length,
                fixed_context_length=fixed_context_length,
                horizon=horizon,
            )
            for row in rows
            if int(row["seed_index"]) in seed_set
        ]
        path = output_dir / f"{split}.jsonl"
        row_count, sha256 = write_jsonl(path, split_rows)
        files[split] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "row_count": row_count,
            "target_variate_count": sum(
                int(row["target_dim"]) for row in split_rows
            ),
            "seed_indices": seeds,
            "sha256": sha256,
        }

    if files["A"]["row_count"] != files["B"]["row_count"]:
        raise ValueError("A/B row counts differ")
    if files["A"]["target_variate_count"] != files["B"]["target_variate_count"]:
        raise ValueError("A/B target variate counts differ")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "experiment_id": experiment_manifest["experiment_id"],
            "experiment_root": str(source_experiment),
            "protocol_sha256": experiment_manifest["protocol_sha256"],
            "experiment_manifest": str(experiment_manifest_path),
            "source_shards": [str(path) for path in paths],
        },
        "selection": {
            "evaluation_table": "main",
            "generator_family_role": "primary",
            "excluded": [
                "secondary generator-family records",
                "strict counterfactual audit records",
                "robustness records",
                "input-ablation records",
            ],
            "seed_partition": "A=seed_index[0:32], B=seed_index[32:64]",
            "paired_seed_policy": (
                "all intensities, counterfactual members, targets, and nuisance "
                "paths for one seed stay in the same split"
            ),
        },
        "task": {
            "master_context_length": master_context_length,
            "fixed_context_length": fixed_context_length,
            "horizon": horizon,
            "master_slice_start": master_context_length - fixed_context_length,
            "target_slice": "master_target[168:384]",
            "context_slice": "exported_target[:168]",
            "label_slice": "exported_target[168:216]",
            "standardization": (
                "slice exact L336-standardized master without re-standardization"
            ),
        },
        "coverage": coverage,
        "files": files,
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
