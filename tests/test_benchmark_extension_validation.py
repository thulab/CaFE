from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe.benchmark_extension.generation import generate_dataset
from cafe.benchmark_extension.mechanisms import _nonlinear_state_response
from cafe.benchmark_extension.storage import (
    iter_compact_parquet,
    parquet_file_record,
    write_compact_parquet,
)
from cafe.benchmark_extension.validation import validate_generation


def _fixture(tmp_path: Path, monkeypatch) -> Path:
    asset = tmp_path / "gift" / "fixture" / "H"
    asset.mkdir(parents=True)
    t = np.arange(800.0)
    target = 0.02 * t + np.sin(t / 10.0)
    table = pa.table(
        {
            "item_id": ["native-item"],
            "start": ["2020-01-01"],
            "freq": ["H"],
            "target": [target.tolist()],
        }
    )
    with pa.OSFile(str(asset / "data-00000-of-00001.arrow"), "wb") as sink:
        with pa_ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    spec = protocol.DatasetSpec(
        "gift_fixture", "Fixture", "fixture/H", "fixture/H", "Test"
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)
    return tmp_path / "gift"


def _tvs_fixture(tmp_path: Path, monkeypatch) -> Path:
    asset = tmp_path / "gift-tvs" / "fixture" / "H"
    asset.mkdir(parents=True)
    t = np.arange(1200.0)
    carrier = np.sin(2.0 * np.pi * t / 12.0 + 0.2)
    envelope = 1.0 + 0.7 * np.sin(2.0 * np.pi * t / 120.0 + 0.8)
    table = pa.table(
        {
            "item_id": ["native-item"],
            "start": ["2020-01-01"],
            "freq": ["H"],
            "target": [(carrier * envelope).tolist()],
        }
    )
    with pa.OSFile(str(asset / "data-00000-of-00001.arrow"), "wb") as sink:
        with pa_ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    spec = protocol.DatasetSpec(
        "gift_tvs_fixture", "TVS Fixture", "fixture/H", "fixture/H", "Test"
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)
    return tmp_path / "gift-tvs"


def _nonlinear_fixture(tmp_path: Path, monkeypatch) -> Path:
    asset = tmp_path / "gift-nonlinear" / "fixture" / "H"
    asset.mkdir(parents=True)
    rng = np.random.default_rng(123)
    # Short-term GIFT evaluation keeps rolling windows; w00 therefore starts
    # 576 points before the source end.  This leaves the audited history
    # ending at the selected extreme state at index 4969.
    target = np.zeros(4970 + 576, dtype=float)
    target[0] = 1.0
    for index in range(1, target.size):
        target[index] = (
            0.05 * target[index - 1]
            + float(_nonlinear_state_response(target[index - 1]))
            + rng.normal(0.0, 0.6)
        )
        target[index] = float(np.clip(target[index], -6.0, 6.0))
    table = pa.table(
        {
            "item_id": ["native-item"],
            "start": ["2020-01-01"],
            "freq": ["H"],
            "target": [target.tolist()],
        }
    )
    with pa.OSFile(str(asset / "data-00000-of-00001.arrow"), "wb") as sink:
        with pa_ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    spec = protocol.DatasetSpec(
        "gift_nonlinear_fixture",
        "Nonlinear Fixture",
        "fixture/H",
        "fixture/H",
        "Test",
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)
    return tmp_path / "gift-nonlinear"


def test_validation_accepts_exact_generated_pairs(tmp_path: Path, monkeypatch) -> None:
    gift_root = _fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment" / "gift_fixture"
    generate_dataset(
        "gift_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=14,
        capability_ids=("trend", "regime_switching", "predictable_intermittency"),
        max_instances=1,
    )
    report = validate_generation(dataset_root)
    assert report["accepted"]
    assert report["validation_mode"] == "research"
    assert report["source_distance_gate_checked_count"] == 15
    assert report["mechanism_scoring_gate_checked_count"] == 15
    assert report["official_baseline_count"] == 1
    assert report["treatment_count"] == 15

    publication = validate_generation(
        dataset_root,
        mode="publication",
        workers=2,
    )
    assert publication["accepted"]
    assert publication["validation_mode"] == "publication"


def test_validation_rejects_self_consistent_treatment_source_rewrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment" / "gift_fixture"
    generate_dataset(
        "gift_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=14,
        capability_ids=("trend",),
        max_instances=1,
    )
    treatment_path = dataset_root / "01_generation" / "treatment_contracts.parquet"
    rows = list(iter_compact_parquet(treatment_path))
    rows[0]["source_history_sha256"] = "0" * 64
    write_compact_parquet(treatment_path, rows)
    manifest_path = dataset_root / "01_generation" / "manifest.json"
    manifest = protocol.read_json(manifest_path)
    manifest["files"]["capability_treatments"] = {
        **parquet_file_record(treatment_path, row_count=len(rows)),
    }
    protocol.write_json(manifest_path, manifest)
    research_report = validate_generation(dataset_root)
    assert research_report["accepted"]

    report = validate_generation(dataset_root, mode="publication", workers=1)
    assert not report["accepted"]
    assert report["validation_mode"] == "publication"
    assert any(
        row["reason"] == "deterministic_replay_mismatch"
        for row in report["failures"]
    )


def test_research_validation_rejects_stored_distance_below_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment" / "gift_fixture"
    generate_dataset(
        "gift_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=14,
        capability_ids=("trend",),
        max_instances=1,
    )
    treatment_path = dataset_root / "01_generation" / "treatment_contracts.parquet"
    rows = list(iter_compact_parquet(treatment_path))
    gate = rows[0]["source_distance_gate"]
    gate["minimum_observed_macro_distance"] = 0.01
    gate["accepted"] = False
    gate["reason"] = "below_minimum_source_distance"
    write_compact_parquet(treatment_path, rows)

    report = validate_generation(dataset_root, workers=2)
    assert not report["accepted"]
    assert report["failure_count"] == 1
    assert report["failures"][0]["reason"] in {
        "source_distance_gate_observed_mismatch",
        "source_distance_gate_minimum_mismatch",
        "source_distance_below_minimum",
        "source_distance_rejected",
    }


def test_research_validation_rejects_invalid_mechanism_scoring_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment" / "gift_fixture"
    generate_dataset(
        "gift_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=14,
        capability_ids=("predictable_intermittency",),
        max_instances=1,
    )
    treatment_path = dataset_root / "01_generation" / "treatment_contracts.parquet"
    rows = list(iter_compact_parquet(treatment_path))
    rows[0]["mechanism_scoring_gate"]["truth_effect_mase_rms"] = 0.0
    write_compact_parquet(treatment_path, rows)

    report = validate_generation(dataset_root, workers=2)
    assert not report["accepted"]
    assert report["failure_count"] == 1
    assert report["failures"][0]["reason"] in {
        "mechanism_scoring_gate_status",
        "mechanism_scoring_gate_reason",
        "mechanism_scoring_gate_ranking_flag",
        "intermittency_future_effect_not_scoreable",
    }


def test_research_validation_checks_tvs_horizon_support_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _tvs_fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment" / "gift_tvs_fixture"
    generate_dataset(
        "gift_tvs_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=31,
        capability_ids=("time_varying_seasonality",),
        max_instances=1,
    )
    accepted = validate_generation(dataset_root)
    assert accepted["accepted"]
    assert accepted["horizon_support_gate_checked_count"] == 5
    publication = validate_generation(
        dataset_root,
        gift_eval_dir=gift_root,
        mode="publication",
        workers=1,
    )
    assert publication["accepted"]

    treatment_path = dataset_root / "01_generation" / "treatment_contracts.parquet"
    rows = list(iter_compact_parquet(treatment_path))
    rows[0]["horizon_support_gate"][
        "minimum_observed_active_fraction"
    ] = 0.0
    write_compact_parquet(treatment_path, rows)
    rejected = validate_generation(dataset_root)
    assert not rejected["accepted"]
    assert rejected["failures"][0]["reason"] == (
        "horizon_support_gate_minimum_mismatch"
    )


def test_research_validation_checks_nonlinear_identifiability_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = _nonlinear_fixture(tmp_path, monkeypatch)
    dataset_root = tmp_path / "experiment" / "gift_nonlinear_fixture"
    generate_dataset(
        "gift_nonlinear_fixture",
        gift_eval_dir=gift_root,
        dataset_root=dataset_root,
        term="short",
        augmentation_seed=7,
        capability_ids=("nonlinear_persistence",),
        max_instances=1,
    )
    accepted = validate_generation(dataset_root)
    assert accepted["accepted"]
    assert accepted["nonlinear_identifiability_gate_checked_count"] == 5

    treatment_path = dataset_root / "01_generation" / "treatment_contracts.parquet"
    rows = list(iter_compact_parquet(treatment_path))
    audit = rows[0]["group_metadata"]["nonlinear_identifiability_gate"][
        "diagnostics_by_target"
    ]["0"]
    audit["holdout_incremental_r2"] = 0.0
    write_compact_parquet(treatment_path, rows)
    rejected = validate_generation(dataset_root)
    assert not rejected["accepted"]
    assert rejected["failures"][0]["reason"] == (
        "nonlinear_identifiability_target_invalid"
    )
