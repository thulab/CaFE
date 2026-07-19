from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/analyze_paper_v5_e2_seed_bank_reliability.py"
)
SPEC = importlib.util.spec_from_file_location(
    "seed_bank_reliability",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
reliability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reliability)


def cell_scores(bank_id: str, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for capability in ("cap1", "cap2"):
        for intensity in (1, 2):
            for model_index, model_id in enumerate(("m1", "m2", "m3")):
                value = 1.0 + model_index + intensity / 10 + offset
                rows.append(
                    {
                        "bank_id": bank_id,
                        "model_id": model_id,
                        "dataset_id": "dataset",
                        "task_id": "task",
                        "capability_id": capability,
                        "intensity": intensity,
                        "mase_mean": value,
                        "mase_std": 0.2,
                        "mase_se": 0.2 / np.sqrt(160),
                        "mase_ci_low": value - 0.04,
                        "mase_ci_high": value + 0.04,
                        "relative_log_mase": np.log(value),
                        "model_rank": float(model_index + 1),
                    }
                )
    return pd.DataFrame(rows)


def test_lin_concordance_is_one_only_for_identical_vectors() -> None:
    values = np.asarray([1.0, 2.0, 4.0])

    assert reliability.lin_concordance(values, values) == pytest.approx(1.0)
    assert reliability.lin_concordance(values, values + 1.0) < 1.0


def test_cell_model_comparison_retains_absolute_bank_shift() -> None:
    compared = reliability.compare_cell_model_scores(
        cell_scores("A"),
        cell_scores("B", offset=0.1),
    )

    assert len(compared) == 12
    assert set(compared["mase_difference_b_minus_a"].round(8)) == {0.1}
    assert not compared["mean_ci_overlap"].any()


def test_pair_state_comparison_distinguishes_ties_and_conflicts() -> None:
    keys = {
        "dataset_id": "dataset",
        "task_id": "task",
        "capability_id": "cap",
        "intensity": 1,
        "left_model": "m1",
        "right_model": "m2",
    }
    bank_a = pd.DataFrame(
        [{**keys, "bank_id": "A", "state": "left_better"}]
    )
    bank_b = pd.DataFrame(
        [{**keys, "bank_id": "B", "state": "right_better"}]
    )

    compared, summary = reliability.compare_pair_states(bank_a, bank_b)

    assert bool(compared.iloc[0]["direction_conflict"]) is True
    assert summary["direction_conflict_count"] == 1
    assert summary["both_decisive_directional_agreement"] == 0.0


def test_formal_rank_reliability_is_reported_at_bank_level() -> None:
    compared, summary = reliability.formal_rank_reliability(
        cell_scores("A"),
        cell_scores("B", offset=0.1),
    )

    assert len(compared) == 4
    assert summary["cell_count"] == 4
    assert summary["passed_cell_count"] == 4
    assert summary["pairwise_ordering_agreement"]["mean"] == pytest.approx(
        1.0
    )
