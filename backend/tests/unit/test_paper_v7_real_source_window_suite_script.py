import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = (
    REPO_ROOT / "scripts/build_paper_v5_real_source_window_suite.py"
)
INFERENCE_PATH = REPO_ROOT / "scripts/run_paper_v5_e2_inference.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(BUILDER_PATH, "paper_v7_real_source_builder")
inference = load_module(INFERENCE_PATH, "paper_v7_real_source_inference")


def support_cell(
    task_id: str,
    capability_id: str,
    *,
    target_dim: int,
    covariate_dim: int = 0,
    hierarchy: str | None = None,
) -> dict:
    dataset_id = "shared_dataset"
    return {
        "dataset_id": dataset_id,
        "task_id": task_id,
        "task_view_id": f"{dataset_id}::{task_id}",
        "capability_id": capability_id,
        "generator_profile_id": (
            f"{dataset_id}__{task_id}__L504_H48"
        ),
        "status": "supported",
        "structure_audit": {
            "target_dim": target_dim,
            "covariate_dim": covariate_dim,
            "hierarchy": hierarchy,
        },
    }


def fixture_artifacts() -> tuple[dict, dict]:
    steps = builder.MAX_CONTEXT_LENGTH + builder.HORIZON
    reference_count = 2
    univariate = np.stack(
        [
            np.arange(steps, dtype=float),
            np.arange(steps, dtype=float) + 10.0,
        ]
    )
    child_a = np.stack(
        [
            np.arange(steps, dtype=float) + 1.0,
            np.arange(steps, dtype=float) + 2.0,
        ]
    )
    child_b = np.stack(
        [
            np.arange(steps, dtype=float) * 0.5 + 3.0,
            np.arange(steps, dtype=float) * 0.5 + 4.0,
        ]
    )
    hierarchy = np.stack(
        [child_a + child_b, child_a, child_b],
        axis=2,
    )
    cov_target = (univariate / 10.0)[:, :, None]
    covariates = np.stack(
        [
            np.column_stack(
                [
                    np.arange(steps) % 24,
                    (np.arange(steps) % 7 == 0).astype(float),
                ]
            )
            for _ in range(reference_count)
        ]
    )
    support = {
        "cells": [
            support_cell(
                "univariate",
                "trend",
                target_dim=1,
            ),
            support_cell(
                "univariate",
                "regime_switching",
                target_dim=1,
            ),
            support_cell(
                "hierarchy",
                "hierarchical_coherence",
                target_dim=3,
                hierarchy="additive_first",
            ),
            support_cell(
                "covariate",
                "covariate_response",
                target_dim=1,
                covariate_dim=2,
            ),
        ]
    }
    common = {
        "context_length": 504,
        "horizon": 48,
        "reference_count": reference_count,
        "season_length": 24,
        "split": {"policy": "fixture"},
        "reference_group_ids": ["group-a", "group-b"],
        "reference_window_starts": [100, 200],
    }
    near = {
        "config": {},
        "buckets": {
            "shared_dataset__univariate__L504_H48": {
                **common,
                "target_dim": 1,
                "covariate_dim": 0,
                "reference_raw": univariate.tolist(),
            },
            "shared_dataset__hierarchy__L504_H48": {
                **common,
                "target_dim": 3,
                "covariate_dim": 0,
                # The calibration artifact flattens each [552, D] row.
                "reference_raw": hierarchy.reshape(
                    reference_count,
                    -1,
                ).tolist(),
                "target_column_names": ["total", "a", "b"],
                "frequency": "d",
                "provenance": {
                    "hierarchy": "publisher-native"
                },
            },
            "shared_dataset__covariate__L504_H48": {
                **common,
                "target_dim": 1,
                "covariate_dim": 2,
                "reference_raw": cov_target.reshape(
                    reference_count,
                    -1,
                ).tolist(),
                "reference_covariates": covariates.tolist(),
                "target_column_names": ["load"],
                "covariate_column_names": ["hour", "holiday"],
                "known_future_covariates": ["hour", "holiday"],
                "covariate_provenance": "deterministic calendar",
                "frequency": "h",
                "provenance": {
                    "source": {"license": "fixture"}
                },
            },
        },
    }
    return support, near


def write_calibration_fixture(path: Path) -> tuple[dict, dict]:
    path.mkdir()
    support, near = fixture_artifacts()
    inputs = {
        "near_distance_artifact.json": near,
        "dataset_capability_support_matrix.json": support,
    }
    files = []
    for name, payload in inputs.items():
        artifact_path = path / name
        artifact_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files.append(
            {"path": name, "sha256": builder.file_sha256(artifact_path)}
        )
    (path / "manifest.json").write_text(
        json.dumps({"files": files}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return support, near


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_v7_suite_keeps_same_dataset_task_views_and_structures(tmp_path):
    calibration_dir = tmp_path / "calibration"
    output_dir = tmp_path / "real-source"
    support, _near = write_calibration_fixture(calibration_dir)

    result = builder.build_suite(calibration_dir, output_dir)
    rows = read_jsonl(output_dir / "real_source_samples.jsonl")
    config = json.loads((output_dir / "config.json").read_text())
    support_output = json.loads(
        (output_dir / "dataset_support.json").read_text()
    )

    selected = builder.selected_task_views(support)
    assert len(selected) == 3
    assert {row["task_view_id"] for row in selected} == {
        "shared_dataset::univariate",
        "shared_dataset::hierarchy",
        "shared_dataset::covariate",
    }
    assert result["task_view_count"] == 3
    assert result["master_sample_count"] == 6
    assert config["selected_supported_cell_count"] == 4
    assert support_output["supported_dataset_count"] == 1
    assert support_output["supported_task_view_count"] == 3

    hierarchy = next(row for row in rows if row["task_id"] == "hierarchy")
    hierarchy_values = np.asarray(hierarchy["target"], dtype=float)
    assert hierarchy_values.shape == (552, 3)
    assert hierarchy["hierarchy"] == "additive_first"
    assert hierarchy["target_column_names"] == ["total", "a", "b"]
    assert np.allclose(
        hierarchy_values[:, 0],
        hierarchy_values[:, 1:].sum(axis=1),
    )
    assert hierarchy["source_group_id"] == "group-a"
    assert hierarchy["source_window_start"] == 100
    assert hierarchy["source_window_id"] == "group-a::start-100"

    covariate = next(row for row in rows if row["task_id"] == "covariate")
    assert np.asarray(covariate["target"]).shape == (552, 1)
    assert np.asarray(covariate["covariates"]).shape == (552, 2)
    assert covariate["target_column_names"] == ["load"]
    assert covariate["covariate_column_names"] == ["hour", "holiday"]
    assert covariate["frequency"] == "h"
    assert covariate["source_provenance"] == {
        "source": {"license": "fixture"}
    }


def test_structured_real_master_is_directly_readable_by_v7_inference(
    tmp_path,
):
    calibration_dir = tmp_path / "calibration"
    output_dir = tmp_path / "real-source"
    write_calibration_fixture(calibration_dir)
    builder.build_suite(calibration_dir, output_dir)
    rows = read_jsonl(output_dir / "real_source_samples.jsonl")
    master = next(row for row in rows if row["task_id"] == "hierarchy")

    views = [
        inference.master_view(master, context_length)
        for context_length in inference.CONTEXT_LENGTHS
    ]

    assert [len(view["target"]) for view in views] == [144, 216, 384, 552]
    assert all(view["target_dim"] == 3 for view in views)
    assert all(view["hierarchy"] == "additive_first" for view in views)
    assert all(
        view["master_future_sha256"] == master["future_sha256"]
        for view in views
    )
    assert all(view["prediction_kind"] == "real_source" for view in views)


def test_old_covariate_bucket_without_references_fails_closed():
    support, near = fixture_artifacts()
    del near["buckets"][
        "shared_dataset__covariate__L504_H48"
    ]["reference_covariates"]

    with pytest.raises(ValueError, match="lacks reference_covariates"):
        builder.source_rows(
            near_artifact=near,
            task_views=builder.selected_task_views(support),
        )


def test_old_non_covariate_bucket_without_structured_metadata_is_compatible():
    steps = builder.MAX_CONTEXT_LENGTH + builder.HORIZON
    rows, support_rows = builder.source_rows(
        near_artifact={
            "buckets": {
                "old__univariate__L504_H48": {
                    "context_length": 504,
                    "horizon": 48,
                    "target_dim": 1,
                    "covariate_dim": 0,
                    "reference_count": 2,
                    "reference_raw": np.zeros((2, steps)).tolist(),
                    "season_length": 24,
                }
            }
        },
        eligible={"old": ["trend", "regime_switching"]},
    )

    assert len(rows) == 2
    assert rows[0]["target_column_names"] == ["target_0"]
    assert rows[0]["covariates"] is None
    assert support_rows[0]["status"] == "supported"
