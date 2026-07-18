from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).parents[3] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "build_paper_v4_real_evaluation_suite.py"


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_paper_v4_real_evaluation_suite",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluation_origins_never_expose_development_targets() -> None:
    module = load_module()

    gift_origins = module.evaluation_origins(
        kind="gift_univariate",
        series_length=1_000,
        official_test_tail_steps=96,
    )
    assert gift_origins == [
        (0, 904, 904, 856, "gift_official_short_term_test_tail"),
        (1, 952, 904, 856, "gift_official_short_term_test_tail"),
    ]
    for _index, origin, split_start, development_end, _role in gift_origins:
        assert origin >= split_start
        assert origin >= development_end
        assert origin + module.HORIZON <= 1_000

    tsf_origins = module.evaluation_origins(
        kind="tsf_univariate",
        series_length=1_000,
        official_test_tail_steps=0,
    )
    assert tsf_origins == [
        (0, 952, 952, 952, "tsf_final_internal_validation_horizon")
    ]


def test_four_lookbacks_share_the_exact_raw_future() -> None:
    module = load_module()
    dataset = module.DatasetSpec(
        "gift_demo_h",
        "Gift Demo/H",
        "Demo",
        "gift_univariate",
        "demo/H",
    )
    values = np.arange(700, dtype=float)

    rows, support = module.build_rows_from_records(
        dataset,
        records=[("series-1", values)],
        frequency="1h",
        official_test_tail_steps=48,
        max_samples=1,
    )

    assert support["status"] == "supported"
    assert support["selected_master_sample_count"] == 1
    assert len(rows) == 4
    assert {row["lookback"] for row in rows} == {96, 168, 336, 504}
    assert {row["master_sample_id"] for row in rows} == {
        rows[0]["master_sample_id"]
    }
    assert len({row["sample_id"] for row in rows}) == 4
    assert len({row["future_sha256"] for row in rows}) == 1
    assert all(
        row["target_future"] == list(np.arange(652, 700, dtype=float))
        for row in rows
    )
    assert {
        row["lookback"]: len(row["target_history"])
        for row in rows
    } == {96: 96, 168: 168, 336: 336, 504: 504}
    assert {row["frequency"] for row in rows} == {"h"}
    assert {row["source_origin"] for row in rows} == {652}
    assert {row["evaluation_split_start"] for row in rows} == {652}
    assert {row["development_read_end_exclusive"] for row in rows} == {
        604
    }


def test_unsupported_dataset_does_not_block_other_datasets(
    monkeypatch,
) -> None:
    module = load_module()
    unsupported = module.DatasetSpec(
        "unsupported_h",
        "Unsupported/H",
        "Demo",
        "gift_univariate",
        "missing/H",
    )
    supported = module.DatasetSpec(
        "supported_h",
        "Supported/H",
        "Demo",
        "gift_univariate",
        "present/H",
    )

    def fake_build_dataset(dataset, **_kwargs):
        if dataset.dataset_id == "unsupported_h":
            raise ValueError("insufficient evaluation history")
        return [
            {
                "dataset_id": dataset.dataset_id,
                "origin_index": 0,
                "series_id": "series",
                "lookback": 96,
            }
        ], {
            "dataset_id": dataset.dataset_id,
            "status": "supported",
            "reason_codes": [],
            "selected_master_sample_count": 1,
            "output_row_count": 1,
        }

    monkeypatch.setattr(module, "build_dataset", fake_build_dataset)
    rows, support, _config = module.build_suite(
        (unsupported, supported),
        gift_eval_dir=Path("/unused"),
        data_dir=Path("/unused"),
        max_samples=1,
    )

    assert [row["dataset_id"] for row in rows] == ["supported_h"]
    assert support["supported_dataset_count"] == 1
    assert support["unsupported_dataset_count"] == 1
    assert support["datasets"][0]["status"] == "unsupported"
    assert support["datasets"][0]["reason_codes"] == [
        "dataset_build_failed"
    ]
    assert support["datasets"][1]["status"] == "supported"


def test_write_outputs_creates_sealed_suite(tmp_path) -> None:
    module = load_module()
    output_dir = tmp_path / "real-eval"
    rows = [{"sample_id": "sample-1", "target_future": [1.0]}]
    support = {
        "schema_version": module.SUPPORT_SCHEMA_VERSION,
        "datasets": [],
    }
    config = {"schema_version": module.SCHEMA_VERSION}

    module.write_outputs(
        output_dir,
        rows=rows,
        support=support,
        config=config,
    )

    assert {
        path.name for path in output_dir.iterdir()
    } == {
        "config.json",
        "dataset_support.json",
        "real_samples.jsonl",
        "manifest.json",
    }
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == module.MANIFEST_SCHEMA_VERSION
    assert set(manifest["files"]) == {
        "config.json",
        "dataset_support.json",
        "real_samples.jsonl",
    }
