from app.services.forecast_store import ForecastStore


def test_forecast_store_writes_success_rows_with_schema(tmp_path):
    artifact = ForecastStore(tmp_path).write_forecasts(
        run_id="run-1",
        task_id="task-1",
        model_id="model-1",
        shard_id="shard-1",
        rows=[
            {
                "sample_id": "sample-1",
                "forecast": [[1.0], [2.0]],
                "future_timestamps": ["t1", "t2"],
                "metrics": {"mse": 0.1, "mae": 0.2},
            }
        ],
    )

    rows = ForecastStore(tmp_path).read_forecasts(artifact.storage_uri)

    assert artifact.schema_version == "forecast.v1"
    assert artifact.sample_count == 1
    assert rows[0]["schema_version"] == "forecast.v1"
    assert rows[0]["benchmarking_run_id"] == "run-1"
    assert rows[0]["task_id"] == "task-1"
    assert rows[0]["model_id"] == "model-1"
    assert rows[0]["shard_id"] == "shard-1"
    assert rows[0]["status"] == "succeeded"
    assert rows[0]["forecast"] == [[1.0], [2.0]]
    assert rows[0]["future_timestamps"] == ["t1", "t2"]
