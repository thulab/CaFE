from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).parents[3] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "merge_paper_v5_calibration_shards.py"


@pytest.fixture()
def module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "merge_paper_v5_calibration_shards",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_shard(path: Path, *, dataset_id: str, task_id: str) -> None:
    path.mkdir()
    task_view_id = f"{dataset_id}::{task_id}"
    profile_id = f"{dataset_id}__{task_id}__L504_H48"
    config = {
        "schema_version": "paper_v4_nine_capability_suite.v3",
        "generator_version": "generator",
        "context_lengths": [96, 168, 336, 504],
        "max_context_length": 504,
        "horizon": 48,
        "validation_embargo": 48,
        "pairing_policy": "paired",
        "profile_role": "dataset-local",
        "dataset_is_independent_unit": True,
        "task_view_is_calibration_unit": True,
        "max_windows_per_dataset": 120,
        "calibration_samples_per_grid_cell": 4,
        "seed": 17,
    }
    inventory = {
        "task_view_id": task_view_id,
        "dataset": {
            "dataset_id": dataset_id,
            "task_id": task_id,
            "task_view_id": task_view_id,
        },
    }
    profile_suite = {
        "schema_version": config["schema_version"],
        "dataset_inventory": [inventory],
        "structured_dataset_profiles": [],
        "calibration_dataset_profiles": [],
        "split_summaries": {},
    }
    generator = {
        "schema_version": "generator.v1",
        "generator_version": "generator",
        "intensity_policy": {"policy": "local"},
        "profiles": {
            profile_id: {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "task_view_id": task_view_id,
            }
        },
    }
    feature = {
        "schema_version": "feature.v1",
        "generator_version": "generator",
        "config": {
            "coverage": 0.95,
            "support_method": "test",
        },
        "buckets": {},
    }
    distance = {
        "schema_version": "distance.v1",
        "generator_version": "generator",
        "dataset_summary_schema_version": config["schema_version"],
        "config": {
            "artifact_reference_count": 1,
            "strict_rule": "test",
            "combined_rule": "test",
        },
        "buckets": {},
    }
    matrix = {
        "schema_version": "matrix.v2",
        "cells": [],
    }
    for name, payload in {
        "config.json": config,
        "profile_suite.json": profile_suite,
        "generator_conditioning_artifact.json": generator,
        "feature_gate_artifact.json": feature,
        "near_distance_artifact.json": distance,
        "dataset_capability_support_matrix.json": matrix,
    }.items():
        write_json(path / name, payload)


def test_merge_accepts_distinct_task_views_of_same_dataset(
    module,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    common = tmp_path / "common"
    hierarchy = tmp_path / "hierarchy"
    output = tmp_path / "merged"
    make_shard(common, dataset_id="shared", task_id="common_factor")
    make_shard(hierarchy, dataset_id="shared", task_id="hierarchy")

    module.merge_shards(output, [common, hierarchy])

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["dataset_ids"] == ["shared"]
    assert config["task_view_ids"] == [
        "shared::common_factor",
        "shared::hierarchy",
    ]


def test_merge_rejects_duplicate_task_view(
    module,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    make_shard(first, dataset_id="shared", task_id="hierarchy")
    make_shard(second, dataset_id="shared", task_id="hierarchy")

    with pytest.raises(
        ValueError,
        match="dataset/task view appears in more than one shard",
    ):
        module.merge_shards(tmp_path / "merged", [first, second])
