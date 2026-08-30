from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cafe import fev_pipeline
from cafe.benchmark_extension.fev_bench import (
    FEV_MINI_SUITE_ID,
    FevBenchAdapter,
    audit_fev_suite,
)
from cafe.benchmark_extension.generation import (
    DEFAULT_NATIVE_GENERATION_BATCH_BYTES,
    _native_instance_batches,
    generate_benchmark_task,
    iter_replayed_samples,
)
from cafe.benchmark_extension.mechanisms import build_capability_group
from cafe.benchmark_extension.validation import validate_generation


def test_fev_model_contract_uses_cafe_operational_context_cap(
    monkeypatch,
) -> None:
    service_model = {
        "forecast_limits": {
            "max_input_length": 16384,
            "max_output_length": 2048,
        }
    }
    monkeypatch.setattr(
        fev_pipeline,
        "health_catalog",
        lambda endpoint, api_prefix: (endpoint, {"moirai2": service_model}),
    )
    contracts = fev_pipeline._service_model_contracts(
        argparse.Namespace(
            endpoints=["http://service:10810"],
            api_prefix="/ai/api/v1",
            models=["moirai2"],
        )
    )
    assert contracts["moirai2"]["service_maximum_context"] == 16384
    assert contracts["moirai2"]["maximum_context"] == 16384
    assert contracts["moirai2"]["context_policy"] == (
        "min_service_and_cafe_operational_limit_v1"
    )


def _fixture(tmp_path: Path, *, length: int = 12) -> tuple[Path, Path]:
    timestamps = [
        datetime(2024, 1, 1) + timedelta(hours=index) for index in range(length)
    ]
    table = pa.table(
        {
            "id": pa.array(["b", "a"], type=pa.string()),
            "timestamp": pa.array(
                [timestamps, timestamps],
                type=pa.list_(pa.timestamp("us")),
            ),
            "target_a": pa.array(
                [
                    np.linspace(10.0, 10.0 + length - 1, length).tolist(),
                    np.linspace(0.0, length - 1, length).tolist(),
                ]
            ),
            "target_b": pa.array(
                [
                    np.linspace(30.0, 30.0 + length - 1, length).tolist(),
                    np.linspace(20.0, 20.0 + length - 1, length).tolist(),
                ]
            ),
            "past_driver": pa.array(
                [
                    np.linspace(100.0, 100.0 + length - 1, length).tolist(),
                    np.linspace(200.0, 200.0 + length - 1, length).tolist(),
                ]
            ),
            "temperature": pa.array(
                [
                    np.linspace(5.0, 8.0, length).tolist(),
                    np.linspace(9.0, 12.0, length).tolist(),
                ]
            ),
            "holiday": pa.array(
                [
                    ([0, 0, 0, 0, 1, 1] * ((length + 5) // 6))[:length],
                    ([0, 0, 0, 0, 1, 1] * ((length + 5) // 6))[:length],
                ]
            ),
            "region": pa.array(["west", "east"], type=pa.string()),
        }
    )
    dataset_path = tmp_path / "fixture.parquet"
    pq.write_table(table, dataset_path)
    suite_path = tmp_path / "tasks_mini.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "tasks:",
                f"- dataset_path: {dataset_path}",
                "  horizon: 2",
                "  num_windows: 2",
                "  seasonality: 2",
                "  eval_metric: MASE",
                "  target: [target_a, target_b]",
                "  known_dynamic_columns: [temperature, holiday]",
                "  past_dynamic_columns: [past_driver]",
                "  static_columns: [region]",
                "  task_name: fixture_task",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return suite_path, dataset_path


def test_fev_adapter_preserves_official_windows_and_covariate_roles(
    tmp_path: Path,
) -> None:
    suite_path, _ = _fixture(tmp_path)
    adapter = FevBenchAdapter(suite_path, num_proc=1)
    tasks = adapter.list_tasks(FEV_MINI_SUITE_ID)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.horizon == 2
    assert task.seasonality == 2
    assert task.target_column_names == ("target_a", "target_b")

    instances = list(
        adapter.iter_instances(
            task,
            selected_model_max_contexts={"short": 4, "long": 10},
        )
    )
    assert len(instances) == 4
    # FEV sorts series IDs and iterates windows in source task order.
    assert [(row.window_index, row.item_id) for row in instances] == [
        (0, "a"),
        (0, "b"),
        (1, "a"),
        (1, "b"),
    ]
    first = instances[0]
    assert first.benchmark_id == "fev_bench"
    assert first.task_id == "fev__fixture_task"
    assert first.context_length == 8
    assert first.history.shape == (8, 2)
    assert first.future.shape == (2, 2)
    np.testing.assert_array_equal(first.future[:, 0], [8.0, 9.0])
    assert first.covariate_column_names == (
        "past_driver",
        "holiday",
        "temperature",
    )
    assert first.covariate_availability == (
        "past_only",
        "known_future",
        "known_future",
    )
    assert first.future_covariate_visible == (False, True, True)
    assert first.covariate_types == (
        "continuous_numeric",
        "binary",
        "continuous_numeric",
    )
    np.testing.assert_array_equal(
        first.future_covariates[:, 0], first.history_covariates[-1, 0]
    )
    assert first.static_covariates == {"region": "east"}
    assert first.selected_model_max_contexts == {"short": 4, "long": 10}


def test_fev_adapter_audit_and_continuous_covariate_eligibility(
    tmp_path: Path,
) -> None:
    suite_path, _ = _fixture(tmp_path)
    adapter = FevBenchAdapter(suite_path, num_proc=1)
    audit = audit_fev_suite(adapter)
    assert audit["task_count"] == 1
    assert audit["instance_count"] == 4
    assert audit["forecast_target_cell_count"] == 16

    task = adapter.list_tasks()[0]
    instance = next(
        adapter.iter_instances(
            task,
            selected_model_max_contexts={"fixture": 8},
        )
    )
    group = build_capability_group(
        instance,
        "covariate_impulse_response",
        augmentation_seed=7,
    )
    # The short fixture may fail a mechanism history requirement, but it must
    # never select the binary holiday channel as a continuous impulse carrier.
    if group.available:
        selected = {
            int(row.metadata["covariate_index"]) for row in group.treatments
        }
        assert selected <= {0, 2}


def test_fev_generation_replay_and_research_validation(tmp_path: Path) -> None:
    suite_path, _ = _fixture(tmp_path)
    adapter = FevBenchAdapter(suite_path, num_proc=1)
    task = adapter.list_tasks()[0]
    dataset_root = tmp_path / "experiment" / task.task_id
    manifest = generate_benchmark_task(
        adapter,
        task,
        dataset_root=dataset_root,
        augmentation_seed=17,
        capability_ids=("trend",),
        model_max_contexts={"fixture": 8},
        max_instances=2,
        workers=1,
        shard_size=1,
    )

    assert manifest["benchmark_id"] == "fev_bench"
    assert manifest["official_instance_count"] == 2
    replayed = list(iter_replayed_samples(manifest))
    baselines = [
        row
        for row in replayed
        if row["evaluation_table"] == "benchmark_official_baseline"
    ]
    assert len(baselines) == 2
    assert {row["task_id"] for row in baselines} == {task.task_id}
    assert all(np.asarray(row["target"])[-2:].shape == (2, 2) for row in baselines)

    report = validate_generation(dataset_root, mode="research")
    assert report["accepted"]
    assert report["benchmark_id"] == "fev_bench"


def test_native_generation_compute_batches_do_not_change_source_shards(
    tmp_path: Path,
) -> None:
    suite_path, _ = _fixture(tmp_path)
    adapter = FevBenchAdapter(suite_path, num_proc=1)
    task = adapter.list_tasks()[0]

    batches = list(
        _native_instance_batches(
            adapter,
            task,
            augmentation_seed=17,
            capability_ids=("trend",),
            max_instances=None,
            shard_size=256,
            model_max_contexts={"fixture": 8},
            maximum_batch_bytes=1,
        )
    )

    assert DEFAULT_NATIVE_GENERATION_BATCH_BYTES == 16 * 1024 * 1024
    assert len(batches) == 4
    assert all(batch["source_shard_size"] == 256 for batch in batches)
    assert [index for batch in batches for index, _ in batch["instances"]] == [
        0,
        1,
        2,
        3,
    ]


def test_fev_research_validation_accepts_native_term_treatments(
    tmp_path: Path,
) -> None:
    suite_path, _ = _fixture(tmp_path, length=80)
    adapter = FevBenchAdapter(suite_path, num_proc=1)
    task = adapter.list_tasks()[0]
    dataset_root = tmp_path / "long-experiment" / task.task_id
    manifest = generate_benchmark_task(
        adapter,
        task,
        dataset_root=dataset_root,
        augmentation_seed=19,
        capability_ids=("trend", "covariate_impulse_response"),
        model_max_contexts={"fixture": 64},
        max_instances=1,
        workers=1,
        shard_size=1,
    )
    assert manifest["treatment_count"] == 10
    assert validate_generation(dataset_root, mode="research", workers=1)[
        "accepted"
    ]


def test_fev_adapter_preserves_missing_masks_and_categorical_semantics(
    tmp_path: Path,
) -> None:
    length = 40
    timestamps = [
        datetime(2024, 1, 1) + timedelta(days=index) for index in range(length)
    ]
    target = np.arange(length, dtype=float)
    target[4] = np.nan
    target[36] = np.nan
    dataset_path = tmp_path / "missing-categorical.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["series"],
                "timestamp": pa.array(
                    [timestamps], type=pa.list_(pa.timestamp("us"))
                ),
                "target": [target.tolist()],
                "state": [["closed", "open"] * (length // 2)],
                "promo_week": [np.nan],
            }
        ),
        dataset_path,
    )
    suite_path = tmp_path / "categorical.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "tasks:",
                f"- dataset_path: {dataset_path}",
                "  horizon: 2",
                "  num_windows: 2",
                "  seasonality: 7",
                "  eval_metric: MASE",
                "  target: target",
                "  known_dynamic_columns: [state]",
                "  static_columns: [promo_week]",
                "  task_name: categorical_task",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = FevBenchAdapter(suite_path, num_proc=1)
    instance = next(adapter.iter_instances(adapter.list_tasks()[0]))
    assert not bool(instance.history_observed_mask[4, 0])
    assert np.all(np.isfinite(instance.history))
    assert not bool(instance.future_observed_mask[0, 0])
    assert np.all(np.isfinite(instance.future))
    assert instance.covariate_types == ("categorical",)
    assert instance.static_covariates == {"promo_week": None}
    assert instance.native_protocol["covariate_encodings"]["state"] == {
        "encoding": "sorted_string_category_codes_v1",
        "categories": ["closed", "open"],
    }
