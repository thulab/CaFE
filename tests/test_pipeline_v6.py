from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe import pipeline
from cafe.benchmark_extension import generation
from cafe.benchmark_extension.mechanisms import (
    CAPABILITY_IDS,
    DEFAULT_CAPABILITY_IDS,
)


def test_v7_pipeline_has_no_calibration_stage() -> None:
    assert pipeline.STAGES == ("generation", "validation", "inference", "analysis")


def test_default_pipeline_runs_eight_capabilities_but_keeps_persistence_optional(
    monkeypatch,
) -> None:
    assert len(DEFAULT_CAPABILITY_IDS) == 8
    assert "nonlinear_persistence" in CAPABILITY_IDS
    assert "nonlinear_persistence" not in DEFAULT_CAPABILITY_IDS
    assert "hierarchical_coherence" not in DEFAULT_CAPABILITY_IDS

    monkeypatch.setattr("sys.argv", ["cafe.pipeline"])
    assert tuple(pipeline.parse_args().capabilities) == DEFAULT_CAPABILITY_IDS
    monkeypatch.setattr("sys.argv", ["cafe.generation", "--dataset-id", "fixture"])
    assert tuple(generation.parse_args().capabilities) == DEFAULT_CAPABILITY_IDS


def test_medium_long_defaults_use_official_config_intersection_and_six_models() -> None:
    expected_datasets = [
        "gift_electricity_h",
        "gift_solar_h",
        "gift_ett1_h",
        "gift_ett2_h",
        "gift_jena_weather_h",
        "gift_kdd_cup_h",
        "gift_loop_seattle_h",
        "gift_m_dense_h",
        "gift_bizitobs_l2c_h",
        "gift_bizitobs_application",
        "gift_bizitobs_service",
    ]
    expected_models = [
        "Timer-4.0",
        "Chronos-2",
        "timesfm2.5",
        "moirai2",
        "Timer-3.5",
        "toto2.0",
    ]
    for term in ("medium", "long"):
        args = argparse.Namespace(
            term=term,
            dataset_id=None,
            dataset_ids=None,
            models=None,
        )
        assert pipeline.selected_dataset_ids(args) == expected_datasets
        assert pipeline.selected_model_ids(args) == expected_models


def test_medium_long_reject_tirex_and_nonofficial_registered_config() -> None:
    with np.testing.assert_raises_regex(ValueError, "fixed medium"):
        pipeline.selected_model_ids(
            argparse.Namespace(term="medium", models=["tirex2"])
        )
    with np.testing.assert_raises_regex(ValueError, "not in the source-available"):
        pipeline.selected_dataset_ids(
            argparse.Namespace(
                term="long",
                dataset_id=["gift_bitbrains_fast_h"],
                dataset_ids=None,
            )
        )


def test_storage_preflight_is_stable_across_resume(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    computed = {
        "schema_version": "cafe.storage_preflight.v1",
        "policy": "test",
        "dataset_count": 1,
        "model_count": 7,
        "maximum_views_per_instance": 56,
        "estimated_steady_state_bytes": 100,
        "estimated_peak_bytes": 120,
        "datasets": [{"dataset_id": "fixture"}],
    }
    first = pipeline._freeze_storage_preflight(
        experiment_root, computed, disk_budget_gb=1.0
    )
    path = experiment_root / "storage_preflight.json"
    first_hash = protocol.file_sha256(path)
    second = pipeline._freeze_storage_preflight(
        experiment_root, computed, disk_budget_gb=1.0
    )
    assert second == first
    assert protocol.file_sha256(path) == first_hash


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
        experiment_id="v7-smoke",
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
        generation_workers=2,
        generation_shard_size=2,
        validation_dataset_workers=1,
        preprocess_workers=2,
        analysis_workers=2,
        max_open_shape_groups=8,
        max_inflight_batches=2,
        max_inflight_mib=64,
        disk_budget_gb=1.0,
    )
    experiment_root = pipeline.run_pipeline(args)
    assert not (experiment_root / "stage_contracts" / "calibration.json").exists()
    assert (experiment_root / "stage_contracts" / "generation.json").is_file()
    assert (experiment_root / "stage_contracts" / "validation.json").is_file()
    assert protocol.read_json(
        experiment_root / "gift_fixture" / "02_validation" / "report.json"
    )["accepted"]
def test_generation_shard_sizes_balance_small_datasets_without_fragmenting_large() -> None:
    preflight = {
        "datasets": [
            {
                "dataset_id": "small",
                "official_instance_upper_bound": 15,
            },
            {
                "dataset_id": "panel",
                "official_instance_upper_bound": 315,
            },
            {
                "dataset_id": "large",
                "official_instance_upper_bound": 95_912,
            },
        ]
    }
    assert pipeline._generation_shard_sizes(
        preflight, requested_shard_size=256
    ) == {"small": 5, "panel": 105, "large": 256}
    assert pipeline._generation_shard_sizes(
        preflight, requested_shard_size=32
    ) == {"small": 5, "panel": 32, "large": 32}
