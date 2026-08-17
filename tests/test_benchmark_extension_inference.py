from __future__ import annotations

import asyncio

import numpy as np

from cafe.benchmark_extension import inference as inference_module
from cafe.benchmark_extension.inference import (
    _run_streaming_bulk_model,
    model_task_row,
)
from cafe.benchmark_extension.storage import iter_prediction_parquet


def _sample() -> dict:
    target = np.arange(250 * 3.0).reshape(250, 3)
    return {
        "schema_version": "cafe.benchmark_extension_sample.v1",
        "sample_id": "sample",
        "context_length": 200,
        "horizon": 50,
        "target_dim": 3,
        "covariate_dim": 0,
        "covariates": None,
        "frequency": "H",
        "target": target.tolist(),
    }


def test_model_context_truncation_happens_after_full_history_treatment() -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {"max_input_length": 96, "min_input_length": 1},
    }
    row = model_task_row(_sample(), model)
    assert row["source_context_length"] == 200
    assert row["context_length"] == 96
    assert np.asarray(row["target"]).shape == (146, 3)
    np.testing.assert_array_equal(
        np.asarray(row["target"]),
        np.asarray(_sample()["target"])[104:],
    )


def test_native_panel_is_preserved_in_generation_task() -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {"max_input_length": -1, "min_input_length": 1},
    }
    row = model_task_row(_sample(), model)
    assert row["target_dim"] == 3
    assert np.asarray(row["target"]).shape == (250, 3)


def test_streaming_bulk_writes_atomic_source_shards_without_task_file(
    tmp_path,
    monkeypatch,
) -> None:
    model = {
        "model_id": "fixture",
        "forecast_limits": {
            "max_input_length": 16,
            "min_input_length": 1,
            "max_output_length": 8,
            "input_mode": {
                "max_target_count": -1,
                "max_history_covariate_count": 0,
                "supports_future_covariates": False,
            },
        },
    }

    def samples():
        for index in range(2):
            yield {
                "schema_version": "fixture",
                "sample_id": f"sample-{index}",
                "source_shard_index": index,
                "context_length": 10,
                "horizon": 2,
                "target_dim": 1,
                "target_column_names": ["target_0"],
                "covariate_dim": 0,
                "covariate_column_names": [],
                "covariates": None,
                "frequency": "H",
                "target": np.arange(12.0)[:, None],
            }

    async def fake_forecast(
        _client,
        *,
        children,
        **_kwargs,
    ):
        return {
            "forecasts": np.zeros((len(children), 1, 2), dtype=np.float32),
            "attempts": 1,
            "elapsed_seconds": 0.01,
            "error": None,
        }

    monkeypatch.setattr(inference_module, "_forecast_bulk_with_retry", fake_forecast)
    summary, stats, parts = asyncio.run(
        _run_streaming_bulk_model(
            model_id="fixture",
            model=model,
            execution={"task_batch_size": 2, "http_concurrency": 2},
            endpoints=["http://endpoint-a", "http://endpoint-b"],
            api_prefix="/ai/api/v1",
            sample_factory=samples,
            prediction_dir=tmp_path / "predictions",
            failure_path=tmp_path / "failures.jsonl",
            forecast_timeout_seconds=30,
            max_attempts=1,
            maximum_open_groups=2,
            maximum_inflight_batches=1,
            maximum_inflight_bytes=1024 * 1024,
        )
    )
    assert summary["compatible_sample_count"] == 2
    assert stats["prediction_count"] == 2
    assert stats["endpoint_count"] == 2
    assert [row["source_shard_index"] for row in parts] == [0, 1]
    assert not (tmp_path / "tasks").exists()
    assert [
        row["sample_id"]
        for part in parts
        for row in iter_prediction_parquet(part["path"])
    ] == ["sample-0", "sample-1"]
