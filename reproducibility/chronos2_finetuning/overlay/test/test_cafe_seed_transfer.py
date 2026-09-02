from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "cafe_seed_transfer.py"
SPEC = importlib.util.spec_from_file_location("cafe_seed_transfer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
cafe_seed_transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cafe_seed_transfer)


def _dense_row() -> dict:
    return {
        "target": np.arange(16, dtype=np.float32).reshape(8, 2),
        "covariates": np.arange(24, dtype=np.float32).reshape(8, 3),
        "future_covariate_visible": [True, False, True],
        "future_observed_mask": np.ones((2, 2), dtype=bool),
        "mase_scale_by_target": [1.0, 2.0],
        "horizon": 2,
        "dataset_id": "gift_fixture",
        "official_instance_id": "instance-1",
        "sample_id": "sample-1",
        "capability_id": "trend",
        "capability_level": 3,
        "augmentation_seed": 7,
    }


def test_training_adapter_reorders_known_future_covariates_last() -> None:
    row = cafe_seed_transfer._prepared_training_row(_dense_row())

    context = np.asarray(row["context"])
    future = np.asarray(row["future_covariates"])
    original_covariates = _dense_row()["covariates"]

    assert context.shape == (5, 8)
    assert row["n_targets"] == 2
    assert row["n_covariates"] == 3
    assert row["n_future_covariates"] == 2
    np.testing.assert_array_equal(context[2], original_covariates[:, 1])
    np.testing.assert_array_equal(context[3], original_covariates[:, 0])
    np.testing.assert_array_equal(context[4], original_covariates[:, 2])
    assert np.isnan(future[:3]).all()
    np.testing.assert_array_equal(future[3], original_covariates[-2:, 0])
    np.testing.assert_array_equal(future[4], original_covariates[-2:, 2])


def test_evaluation_adapter_hides_future_target() -> None:
    prepared, truth, observed = cafe_seed_transfer._prepared_evaluation_row(_dense_row())

    assert prepared["context"].shape == (5, 6)
    assert prepared["future_covariates"].shape == (5, 2)
    np.testing.assert_array_equal(truth, _dense_row()["target"][-2:])
    assert observed.all()


def test_materialized_evaluation_adapter_hides_future_target() -> None:
    materialized = cafe_seed_transfer._prepared_training_row(_dense_row())
    prepared, truth, observed = cafe_seed_transfer._materialized_evaluation_row(
        materialized
    )

    assert prepared["context"].shape == (5, 6)
    assert prepared["future_covariates"].shape == (5, 2)
    np.testing.assert_array_equal(truth, _dense_row()["target"][-2:])
    assert observed.all()


def test_balanced_selection_excludes_nonlinear_and_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "treatments.parquet"
    rows = []
    for capability in ("trend", "nonlinear_persistence"):
        for level in (1, 2):
            for index in range(10):
                rows.append(
                    {
                        "sample_id": f"{capability}-{level}-{index}",
                        "official_instance_id": f"instance-{index}",
                        "capability_id": capability,
                        "capability_level": level,
                    }
                )
    pq.write_table(pa.Table.from_pylist(rows), path)

    kwargs = {
        "capabilities": frozenset(cafe_seed_transfer.CAPABILITIES),
        "capability_levels": frozenset((1, 2)),
        "max_per_stratum": 3,
        "selection_seed": 42,
        "fold_count": 1,
        "heldout_fold": 0,
        "role": "all",
    }
    selected_1, counts_1 = cafe_seed_transfer._selected_sample_ids(path, **kwargs)
    selected_2, counts_2 = cafe_seed_transfer._selected_sample_ids(path, **kwargs)

    assert selected_1 == selected_2
    assert counts_1 == counts_2 == {("trend", 1): 10, ("trend", 2): 10}
    assert selected_1 is not None and len(selected_1) == 6
    assert all("nonlinear_persistence" not in sample_id for sample_id in selected_1)


def test_balanced_selection_can_keep_only_level_five(tmp_path: Path) -> None:
    path = tmp_path / "treatments.parquet"
    rows = [
        {
            "sample_id": f"trend-{level}-{index}-aug1",
            "official_instance_id": f"instance-{index}",
            "capability_id": "trend",
            "capability_level": level,
        }
        for level in (4, 5)
        for index in range(5)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)

    selected, counts = cafe_seed_transfer._selected_sample_ids(
        path,
        capabilities=frozenset(("trend",)),
        capability_levels=frozenset((5,)),
        max_per_stratum=3,
        selection_seed=42,
        fold_count=1,
        heldout_fold=0,
        role="all",
    )

    assert counts == {("trend", 5): 5}
    assert selected is not None and len(selected) == 3
    assert all("trend-5-" in sample_id for sample_id in selected)


def test_official_instance_holdout_roles_are_disjoint() -> None:
    identifiers = [f"instance-{index}" for index in range(100)]
    train = {
        value
        for value in identifiers
        if cafe_seed_transfer._fold_matches(
            value, fold_count=5, heldout_fold=2, role="train"
        )
    }
    evaluation = {
        value
        for value in identifiers
        if cafe_seed_transfer._fold_matches(
            value, fold_count=5, heldout_fold=2, role="eval"
        )
    }

    assert train.isdisjoint(evaluation)
    assert train | evaluation == set(identifiers)


def test_official_instance_fold_salts_change_window_selection() -> None:
    identifiers = [f"instance-{index}" for index in range(1000)]
    first = {
        value
        for value in identifiers
        if cafe_seed_transfer._fold_matches(
            value,
            fold_count=10,
            heldout_fold=0,
            role="eval",
            fold_salt="seed01-window-v1",
        )
    }
    second = {
        value
        for value in identifiers
        if cafe_seed_transfer._fold_matches(
            value,
            fold_count=10,
            heldout_fold=0,
            role="eval",
            fold_salt="seed02-window-v1",
        )
    }

    assert 70 <= len(first) <= 130
    assert 70 <= len(second) <= 130
    assert len(first & second) < 25


def test_epoch_and_checkpoint_step_calculation() -> None:
    total = cafe_seed_transfer._packed_epoch_steps(
        {1: 64, 40: 3}, batch_size=32, seed=7
    )

    assert total == 5
    assert cafe_seed_transfer._checkpoint_steps(1000) == [
        10,
        20,
        50,
        100,
        200,
        400,
        600,
        800,
        1000,
    ]
    assert cafe_seed_transfer._checkpoint_steps(10_000, 1_000) == list(
        range(1_000, 10_001, 1_000)
    )
    assert cafe_seed_transfer._checkpoint_steps(10_500, 1_000)[-2:] == [
        10_000,
        10_500,
    ]


def test_curve_aggregation_merges_complete_ranks(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    parts.mkdir()
    for rank, mase_sum in enumerate((2.0, 4.0)):
        payload = {
            "schema_version": "chronos2.cafe_seed_transfer_prepared_evaluation_part.v1",
            "corpus": "v14",
            "horizon": 30,
            "step": 10,
            "model": "checkpoint-10",
            "rank": rank,
            "world_size": 2,
            "strata": [
                {
                    "dataset_id": "gift_fixture",
                    "capability_id": "trend",
                    "capability_level": 1,
                    "mase_sum": mase_sum,
                    "row_count": 2,
                    "observed_cells": 4,
                    "cell_weighted_mase_sum": mase_sum * 2,
                }
            ],
        }
        (parts / f"rank-{rank}.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "curve.json"
    cafe_seed_transfer.command_aggregate_curve(
        SimpleNamespace(parts=parts, output=output)
    )
    row = json.loads(output.read_text(encoding="utf-8"))["rows"][0]

    assert row["row_count"] == 4
    assert row["macro_stratum_mase"] == 1.5
    assert row["sample_weighted_mase"] == 1.5
    assert row["cell_weighted_mase"] == 1.5
