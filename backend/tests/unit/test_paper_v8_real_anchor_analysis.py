from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script():
    path = SCRIPT_DIR / "analyze_paper_v8_real_anchors.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_paper_v8_real_anchors_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_anchor_scoring_uses_recorded_history_scale():
    analysis = load_script()
    views = {
        "first": {
            "sample_id": "first",
            "anchor_id": "anchor-first",
            "target": [[0.0], [1.0], [2.0], [4.0]],
            "context_length": 2,
            "horizon": 2,
            "target_dim": 1,
            "mase_scale": 2.0,
        },
        "second": {
            "sample_id": "second",
            "anchor_id": "anchor-second",
            "target": [[0.0], [1.0], [3.0], [5.0]],
            "context_length": 2,
            "horizon": 2,
            "target_dim": 1,
            "mase_scale": 1.0,
        },
    }
    predictions = {
        "accurate": {
            "first": {"forecast": [[2.0], [4.0]]},
            "second": {"forecast": [[3.0], [5.0]]},
        },
        "offset": {
            "first": {"forecast": [[4.0], [6.0]]},
            "second": {"forecast": [[4.0], [6.0]]},
        },
    }

    metrics, scores = analysis.score_real_anchor_dataset(
        dataset_id="dataset",
        views=views,
        predictions_by_model=predictions,
    )

    accurate = next(
        row for row in scores if row["model_id"] == "accurate"
    )
    offset = next(row for row in scores if row["model_id"] == "offset")
    assert len(metrics) == 4
    assert accurate["mean_mase"] == 0.0
    assert accurate["mase_rank"] == 1
    assert offset["mean_mase"] == pytest.approx(1.0)
    assert offset["mase_rank"] == 2


def test_real_anchor_macro_ranking_is_compared_with_capability_ranking():
    analysis = load_script()
    models = ["real-winner", "synthetic-winner"]
    datasets = ["first", "second"]
    capabilities = ["trend"]
    real_dataset_scores = [
        {
            "dataset_id": "first",
            "model_id": "real-winner",
            "mean_mase": 1.0,
            "mase_rank": 1,
        },
        {
            "dataset_id": "first",
            "model_id": "synthetic-winner",
            "mean_mase": 2.0,
            "mase_rank": 2,
        },
        {
            "dataset_id": "second",
            "model_id": "real-winner",
            "mean_mase": 1.0,
            "mase_rank": 1,
        },
        {
            "dataset_id": "second",
            "model_id": "synthetic-winner",
            "mean_mase": 3.0,
            "mase_rank": 2,
        },
    ]
    real_metrics = [
        {
            "dataset_id": row["dataset_id"],
            "model_id": row["model_id"],
            "mase": row["mean_mase"],
        }
        for row in real_dataset_scores
    ]
    synthetic_dataset_scores = [
        {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "mean_capability_accuracy_rank": rank,
            "capability_accuracy_rank": rank,
            "mean_capability_mechanism_rank": rank,
            "capability_mechanism_rank": rank,
        }
        for dataset_id in datasets
        for model_id, rank in (
            ("real-winner", 2),
            ("synthetic-winner", 1),
        )
    ]
    synthetic_scores = [
        {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "capability_id": "trend",
            "context_policy": analysis.FIXED_CONTEXT_POLICY,
            "evaluation_table": "main",
            "generator_family_role": "primary",
            "accuracy_rank": rank,
            "mechanism_rank": rank,
        }
        for dataset_id in datasets
        for model_id, rank in (
            ("real-winner", 2),
            ("synthetic-winner", 1),
        )
    ]

    comparison = analysis.ranking_comparison(
        real_metrics=real_metrics,
        real_dataset_scores=real_dataset_scores,
        synthetic_scores=synthetic_scores,
        synthetic_dataset_scores=synthetic_dataset_scores,
        models=models,
        datasets=datasets,
        capabilities=capabilities,
    )

    by_model = {
        row["model_id"]: row
        for row in comparison["overall_model_rows"]
    }
    assert by_model["real-winner"]["real_anchor_macro_rank"] == 1
    assert (
        by_model["synthetic-winner"][
            "synthetic_capability_accuracy_rank"
        ]
        == 1
    )
    assert comparison["overall_rank_correlations"][
        "real_anchor_vs_synthetic_accuracy_kendall_tau_b"
    ] == pytest.approx(-1.0)
