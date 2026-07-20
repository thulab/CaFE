#!/usr/bin/env python3
"""Merge task-view-disjoint Paper v5 calibration build shards.

The upstream builder calibrates every dataset independently.  This utility
only unions disjoint dataset/task-view artifacts; it never combines real
windows, profile statistics, intensity targets, or gate thresholds across
task views. Qualification must run once on the merged output after this
command.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_paper_v4_nine_capability_suite import (  # noqa: E402
    ALL_CAPABILITY_IDS,
    record_task_view_id,
    task_view_id,
    write_json,
    write_support_matrix_csv,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runtime/paper_exp/v5/01_nine_capability_suite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge dataset-disjoint Paper v5 calibration build shards."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_unique_mapping(
    shards: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for shard in shards:
        for item_id, item in shard[key].items():
            if item_id in merged:
                raise ValueError(f"duplicate {key} id across shards: {item_id}")
            merged[item_id] = item
    return merged


def inventory_task_view_id(row: dict[str, Any]) -> str:
    stored = row.get("task_view_id")
    if stored:
        return str(stored)
    dataset = row["dataset"]
    stored = dataset.get("task_view_id")
    if stored:
        return str(stored)
    return task_view_id(
        str(dataset["dataset_id"]),
        str(dataset["task_id"]),
    )


def merge_shards(output_dir: Path, shard_dirs: list[Path]) -> None:
    if len(shard_dirs) < 2:
        raise ValueError("at least two calibration shards are required")
    required = (
        "config.json",
        "profile_suite.json",
        "generator_conditioning_artifact.json",
        "feature_gate_artifact.json",
        "near_distance_artifact.json",
        "dataset_capability_support_matrix.json",
    )
    resolved = [path.resolve() for path in shard_dirs]
    for shard_dir in resolved:
        missing = [
            name for name in required if not (shard_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{shard_dir} is missing build outputs: {', '.join(missing)}"
            )

    configs = [read_json(path / "config.json") for path in resolved]
    comparable_fields = (
        "schema_version",
        "generator_version",
        "context_lengths",
        "max_context_length",
        "horizon",
        "validation_embargo",
        "pairing_policy",
        "profile_role",
        "dataset_is_independent_unit",
        "task_view_is_calibration_unit",
        "max_windows_per_dataset",
        "calibration_samples_per_grid_cell",
        "seed",
    )
    reference = configs[0]
    for index, config in enumerate(configs[1:], start=2):
        mismatched = [
            field
            for field in comparable_fields
            if config.get(field) != reference.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"shard {index} config mismatch: {', '.join(mismatched)}"
            )

    profiles = [
        read_json(path / "profile_suite.json") for path in resolved
    ]
    generators = [
        read_json(path / "generator_conditioning_artifact.json")
        for path in resolved
    ]
    features = [
        read_json(path / "feature_gate_artifact.json") for path in resolved
    ]
    distances = [
        read_json(path / "near_distance_artifact.json") for path in resolved
    ]
    matrices = [
        read_json(path / "dataset_capability_support_matrix.json")
        for path in resolved
    ]

    generator_version = generators[0].get("generator_version")
    intensity_policy = generators[0].get("intensity_policy")
    for artifact in (*generators[1:], *features, *distances):
        if artifact.get("generator_version") != generator_version:
            raise ValueError("generator_version differs across shards")
    for artifact in generators[1:]:
        if artifact.get("intensity_policy") != intensity_policy:
            raise ValueError("intensity policy differs across shards")

    created_at = datetime.now(timezone.utc).isoformat()
    inventory_rows = [
        row
        for profile in profiles
        for row in profile["dataset_inventory"]
    ]
    task_view_ids = [inventory_task_view_id(row) for row in inventory_rows]
    if len(set(task_view_ids)) != len(task_view_ids):
        raise ValueError("dataset/task view appears in more than one shard")
    dataset_ids = sorted(
        {
            str(row["dataset"]["dataset_id"])
            for row in inventory_rows
        }
    )
    task_view_ids = sorted(task_view_ids)
    config = {
        **reference,
        "created_at": created_at,
        "dataset_ids": dataset_ids,
        "task_view_ids": task_view_ids,
        "build_shards": [
            str(path.relative_to(REPO_ROOT)) for path in resolved
        ],
        "shard_merge_policy": (
            "dataset/task-view-disjoint union; no statistic pooling"
        ),
    }

    generator_profiles = merge_unique_mapping(generators, "profiles")
    feature_buckets = merge_unique_mapping(features, "buckets")
    near_buckets = merge_unique_mapping(distances, "buckets")
    split_summaries = merge_unique_mapping(profiles, "split_summaries")
    support_matrix = [
        row for matrix in matrices for row in matrix["cells"]
    ]
    support_matrix.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            record_task_view_id(row),
            ALL_CAPABILITY_IDS.index(str(row["capability_id"])),
        )
    )

    profile_suite = {
        "schema_version": profiles[0]["schema_version"],
        "created_at": created_at,
        "config": config,
        "dataset_inventory": [
            row for profile in profiles for row in profile["dataset_inventory"]
        ],
        "structured_dataset_profiles": [
            row
            for profile in profiles
            for row in profile["structured_dataset_profiles"]
        ],
        "calibration_dataset_profiles": [
            row
            for profile in profiles
            for row in profile["calibration_dataset_profiles"]
        ],
        "split_summaries": split_summaries,
        "support_matrix": support_matrix,
    }
    generator_artifact = {
        "schema_version": generators[0]["schema_version"],
        "generator_version": generator_version,
        "created_at": created_at,
        "config": config,
        "intensity_policy": intensity_policy,
        "profiles": generator_profiles,
    }
    feature_artifact = {
        "schema_version": features[0]["schema_version"],
        "generator_version": generator_version,
        "created_at": created_at,
        "config": {
            **config,
            "coverage": features[0]["config"]["coverage"],
            "support_method": features[0]["config"]["support_method"],
        },
        "buckets": feature_buckets,
    }
    near_artifact = {
        "schema_version": distances[0]["schema_version"],
        "generator_version": generator_version,
        "created_at": created_at,
        "dataset_summary_schema_version": distances[0][
            "dataset_summary_schema_version"
        ],
        "config": {
            **config,
            "artifact_reference_count": distances[0]["config"][
                "artifact_reference_count"
            ],
            "strict_rule": distances[0]["config"]["strict_rule"],
            "combined_rule": distances[0]["config"]["combined_rule"],
        },
        "buckets": near_buckets,
    }
    support_artifact = {
        "schema_version": matrices[0]["schema_version"],
        "created_at": created_at,
        "intensity_policy": intensity_policy,
        "cells": support_matrix,
    }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"merged output directory is not empty: {output_dir}")
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "profile_suite.json", profile_suite)
    write_json(
        output_dir / "generator_conditioning_artifact.json",
        generator_artifact,
    )
    write_json(output_dir / "feature_gate_artifact.json", feature_artifact)
    write_json(output_dir / "near_distance_artifact.json", near_artifact)
    write_json(
        output_dir / "dataset_capability_support_matrix.json",
        support_artifact,
    )
    write_support_matrix_csv(
        output_dir / "dataset_capability_support_matrix.csv",
        support_matrix,
    )
    print(
        f"merged {len(resolved)} shards, {len(dataset_ids)} datasets, "
        f"{len(task_view_ids)} task views, "
        f"{len(generator_profiles)} generator profiles, "
        f"{sum(bool(row['supported']) for row in support_matrix)} supported cells "
        f"into {output_dir}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    merge_shards(args.output_dir, args.shard_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
