from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_paper_v2_e4_synthetic_real_transfer as e4  # noqa: E402


def test_balanced_stratum_allocations_redistribute_small_stratum() -> None:
    allocations = e4.balanced_stratum_allocations(
        {0: 100, 1: 100, 2: 10},
        max_total=100,
    )

    assert allocations == {0: 45, 1: 45, 2: 10}
    assert sum(allocations.values()) == 100


def test_select_evenly_is_deterministic_and_keeps_endpoints() -> None:
    values = [{"index": index} for index in range(10)]

    selected = e4.select_evenly(values, 4)

    assert [row["index"] for row in selected] == [0, 3, 6, 9]


def test_masked_real_metrics_never_impute_future_labels() -> None:
    context = np.arange(e4.CONTEXT_LENGTH, dtype=float)
    target = np.full(e4.HORIZON, 2.0)
    target[1] = np.nan
    forecast = np.ones(e4.HORIZON)

    metrics = e4.masked_real_metrics(
        target,
        forecast,
        context=context,
        period=e4.SEASON_LENGTH,
    )

    assert metrics["observed_future_count"] == e4.HORIZON - 1
    assert metrics["abs_error_sum"] == pytest.approx(e4.HORIZON - 1)
    assert metrics["mase_scale"] == pytest.approx(e4.SEASON_LENGTH)
    assert metrics["mase"] == pytest.approx(1.0 / e4.SEASON_LENGTH)


def test_family_macro_weights_profiles_not_raw_cells() -> None:
    rows = []
    # family_a has one profile with two capability cells; family_b has one cell.
    for value in (0.0, 1.0):
        rows.append(
            {
                "predictor_id": "p",
                "family_id": "family_a",
                "profile_id": "profile_a",
                **{metric: value for metric in e4.CONCORDANCE_METRICS},
            }
        )
    rows.append(
        {
            "predictor_id": "p",
            "family_id": "family_b",
            "profile_id": "profile_b",
            **{metric: 1.0 for metric in e4.CONCORDANCE_METRICS},
        }
    )

    family = e4.family_macro_frame(
        pd.DataFrame.from_records(rows),
        group_columns=["predictor_id"],
    ).set_index("family_id")

    assert family.loc["family_a", "kendall_tau_b"] == pytest.approx(0.5)
    assert family.loc["family_b", "kendall_tau_b"] == pytest.approx(1.0)
    assert family["kendall_tau_b"].mean() == pytest.approx(0.75)


def test_real_history_timestamps_accept_legacy_hour_frequency() -> None:
    values = e4.real_history_timestamps(
        datetime(2020, 1, 1),
        frequency="H",
        start_index=2,
        periods=3,
    )

    assert values == [
        "2020-01-01T02:00:00",
        "2020-01-01T03:00:00",
        "2020-01-01T04:00:00",
    ]


def test_shared_sample_alias_points_to_frozen_tasks(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text('{"sample_id":"one"}\n', encoding="utf-8")

    e4.ensure_shared_sample_alias(tmp_path)

    alias = tmp_path / "samples.jsonl"
    assert alias.is_symlink()
    assert alias.resolve() == task_path.resolve()
    assert e4.sha256_file(alias) == e4.sha256_file(task_path)
