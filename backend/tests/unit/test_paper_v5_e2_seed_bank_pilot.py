from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts/analyze_paper_v5_e2_seed_bank_pilot.py"
SPEC = importlib.util.spec_from_file_location("seed_bank_pilot", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def round_scores(*, reverse_last_model: bool = False) -> pd.DataFrame:
    rows = []
    models = [f"m{index}" for index in range(6)]
    for round_index in range(1, 6):
        scores = list(range(1, 7))
        if reverse_last_model:
            scores[-1], scores[-2] = scores[-2], scores[-1]
        for model_id, score in zip(models, scores, strict=True):
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_id": "dataset",
                    "task_id": "task",
                    "capability_id": "capability",
                    "intensity": 1,
                    "round_index": round_index,
                    "mase_mean": float(score),
                    "master_sample_count": 32,
                    "model_rank": float(score),
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_bank_scores_ranks_after_pooling_160_samples() -> None:
    result = pilot.aggregate_bank_scores(round_scores(), bank_id="A")

    assert set(result["round_count"]) == {5}
    assert set(result["master_sample_count"]) == {160}
    assert result.sort_values("model_id")["model_rank"].tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]


def test_compare_banks_allows_three_of_fifteen_pair_reversals() -> None:
    left = pilot.aggregate_bank_scores(round_scores(), bank_id="A")
    right = left.copy()
    right.loc[right["model_id"] == "m0", "mase_mean"] = 4.5
    right["model_rank"] = right.groupby(
        pilot.CELL_KEYS
    )["mase_mean"].rank(method="average", ascending=True)

    comparison = pilot.compare_banks(left, right)

    assert comparison.iloc[0]["pairwise_ordering_agreement"] == pytest.approx(
        12 / 15
    )
    assert bool(comparison.iloc[0]["passed"]) is True


def test_compare_banks_rejects_four_of_fifteen_pair_reversals() -> None:
    left = pilot.aggregate_bank_scores(round_scores(), bank_id="A")
    right = left.copy()
    right.loc[right["model_id"] == "m0", "mase_mean"] = 5.5
    right["model_rank"] = right.groupby(
        pilot.CELL_KEYS
    )["mase_mean"].rank(method="average", ascending=True)

    comparison = pilot.compare_banks(left, right)

    assert comparison.iloc[0]["pairwise_ordering_agreement"] == pytest.approx(
        11 / 15
    )
    assert bool(comparison.iloc[0]["passed"]) is False
