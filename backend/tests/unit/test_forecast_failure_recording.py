from app.services.forecast_store import ForecastStore


def test_forecast_store_persists_failure_rows_with_error_fields(tmp_path):
    artifact = ForecastStore(tmp_path).write_forecasts(
        run_id="run-1",
        task_id="task-1",
        model_id="model-1",
        shard_id="shard-1",
        rows=[{"sample_id": "sample-1", "status": "failed", "error_code": "adapter_error", "error_message": "boom"}],
    )

    row = ForecastStore(tmp_path).read_forecasts(artifact.storage_uri)[0]

    assert row["status"] == "failed"
    assert row["forecast"] is None
    assert row["error_code"] == "adapter_error"
    assert row["error_message"] == "boom"
