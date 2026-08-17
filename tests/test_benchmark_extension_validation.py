from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe.benchmark_extension.generation import generate_dataset
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
    assert report["official_baseline_count"] == 1
    assert report["treatment_count"] == 15


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
    report = validate_generation(dataset_root)
    assert not report["accepted"]
    assert any(
        row["reason"] == "deterministic_replay_mismatch"
        for row in report["failures"]
    )
