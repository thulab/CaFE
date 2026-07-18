from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "scripts/run_paper_v5_e2_inference.py"
SOURCE_BUILDER_PATH = (
    REPO_ROOT / "scripts/build_paper_v5_real_source_window_suite.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_master(module) -> dict:
    time = np.arange(module.MAX_CONTEXT_LENGTH + module.HORIZON)
    target = np.sin(2 * np.pi * time / 24)[:, None]
    return {
        "sample_id": "master-i1",
        "master_sample_id": "master-i1",
        "paired_group_id": "group",
        "profile_id": "dataset__univariate__L504_H48",
        "dataset_id": "dataset",
        "task_id": "univariate",
        "capability_id": "multi_seasonal",
        "intensity": 1,
        "round_index": 1,
        "round_seed": 101,
        "sample_index": 0,
        "context_length": 504,
        "context_lengths": list(module.CONTEXT_LENGTHS),
        "horizon": module.HORIZON,
        "season_length": 24,
        "frequency": "h",
        "target_dim": 1,
        "covariate_dim": 0,
        "hierarchy": None,
        "target": target.tolist(),
        "covariates": None,
        "future_sha256": module.array_sha256(
            target[module.MAX_CONTEXT_LENGTH :]
        ),
    }


def test_master_expands_to_four_independently_standardized_views():
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_views")
    master = synthetic_master(module)

    views = [
        module.master_view(master, context_length)
        for context_length in module.CONTEXT_LENGTHS
    ]

    assert [view["context_length"] for view in views] == list(
        module.CONTEXT_LENGTHS
    )
    assert len({view["view_id"] for view in views}) == 4
    assert {view["master_sample_id"] for view in views} == {"master-i1"}
    assert {
        view["master_future_sha256"] for view in views
    } == {master["future_sha256"]}
    for view in views:
        context = view["context_length"]
        target = np.asarray(view["target"], dtype=float)
        assert target.shape == (context + module.HORIZON, 1)
        assert abs(float(target[:context].mean())) < 1e-10
        assert abs(float(target[:context].std()) - 1.0) < 1e-10


def test_prediction_row_retains_view_and_master_identity():
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_prediction")
    view = module.master_view(synthetic_master(module), 168)
    target = np.asarray(view["target"], dtype=float)
    forecast = target[168:]

    row = module.prediction_row(
        "model",
        "timer_service",
        view,
        forecast,
    )

    assert row["sample_id"] == "master-i1__L168"
    assert row["view_id"] == "master-i1__L168"
    assert row["master_sample_id"] == "master-i1"
    assert row["context_length"] == 168
    assert row["metrics"]["mase"] == 0.0
    assert row["capability_id"] == "multi_seasonal"
    assert row["round_index"] == 1
    assert len(row["target_future"]) == module.HORIZON


def test_tabpfn_execution_configuration_is_frozen():
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_config")

    assert module.DEFAULT_MODELS[-1] == "tabpfn-ts3"
    assert module.MODEL_EXECUTION_CONFIG["tabpfn-ts3"] == {
        "replicas_per_device": 8,
        "http_concurrency": 24,
    }


def test_preflight_selects_one_master_per_cell(tmp_path):
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_preflight")
    source = tmp_path / "source.jsonl"
    rows = []
    for capability in ("trend", "regime_switching"):
        for index in range(2):
            row = synthetic_master(module)
            row["sample_id"] = f"{capability}-{index}"
            row["master_sample_id"] = row["sample_id"]
            row["capability_id"] = capability
            rows.append(row)
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    destination = module.build_preflight_master_file(
        source,
        tmp_path / "preflight.jsonl",
    )
    selected = list(module.iter_jsonl(destination))

    assert len(selected) == 2
    assert {row["capability_id"] for row in selected} == {
        "trend",
        "regime_switching",
    }


def test_real_source_rows_use_multi_capability_univariate_references():
    module = load_module(
        SOURCE_BUILDER_PATH,
        "paper_v5_real_source_builder",
    )
    references = np.vstack(
        [
            np.arange(module.MAX_CONTEXT_LENGTH + module.HORIZON),
            np.arange(module.MAX_CONTEXT_LENGTH + module.HORIZON) + 1,
        ]
    ).astype(float)
    profile_id = "dataset__univariate__L504_H48"
    rows, support = module.source_rows(
        near_artifact={
            "buckets": {
                profile_id: {
                    "context_length": 504,
                    "horizon": 48,
                    "target_dim": 1,
                    "covariate_dim": 0,
                    "reference_count": 2,
                    "reference_raw": references.tolist(),
                    "season_length": 24,
                    "split": {"policy": "dataset_local"},
                }
            }
        },
        eligible={"dataset": ["trend", "regime_switching"]},
    )

    assert len(rows) == 2
    assert support[0]["status"] == "supported"
    assert support[0]["supported_capability_count"] == 2
    assert rows[0]["context_lengths"] == [96, 168, 336, 504]
    assert np.asarray(rows[0]["target"]).shape == (552, 1)
    assert rows[0]["source_role"].startswith(
        "dataset-local near-distance reference_raw"
    )
