from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe import pipeline


def test_v6_pipeline_has_no_calibration_stage() -> None:
    assert pipeline.STAGES == ("generation", "validation", "inference", "analysis")


def test_preparation_pipeline_uses_official_instances_without_services(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gift_root = tmp_path / "gift"
    asset = gift_root / "fixture" / "H"
    asset.mkdir(parents=True)
    t = np.arange(800.0)
    table = pa.table(
        {
            "item_id": ["item"],
            "start": ["2020-01-01"],
            "freq": ["H"],
            "target": [(0.02 * t + np.sin(t / 10.0)).tolist()],
        }
    )
    with pa.OSFile(str(asset / "data-00000-of-00001.arrow"), "wb") as sink:
        with pa_ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    spec = protocol.DatasetSpec(
        "gift_fixture", "Fixture", "fixture/H", "fixture/H", "Test"
    )
    monkeypatch.setitem(protocol.DATASET_REGISTRY, spec.dataset_id, spec)
    args = argparse.Namespace(
        dataset_id=["gift_fixture"],
        dataset_ids=None,
        output_root=tmp_path / "runtime",
        experiment_id="v6-smoke",
        gift_eval_dir=gift_root,
        term="short",
        augmentation_seed=3,
        capabilities=["trend", "regime_switching"],
        models=[],
        endpoints=[],
        api_prefix="/ai/api/v1",
        devices="0",
        max_instances=1,
        start_at="generation",
        stop_after="validation",
        resume_inference=False,
        prepare_only=False,
    )
    experiment_root = pipeline.run_pipeline(args)
    assert not (experiment_root / "stage_contracts" / "calibration.json").exists()
    assert (experiment_root / "stage_contracts" / "generation.json").is_file()
    assert (experiment_root / "stage_contracts" / "validation.json").is_file()
    assert protocol.read_json(
        experiment_root / "gift_fixture" / "02_validation" / "report.json"
    )["accepted"]
