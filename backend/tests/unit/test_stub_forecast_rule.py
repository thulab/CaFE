from app.services.stub_timer_adapter import StubTimerAdapter


def test_stub_forecast_uses_last_value_naive_shape_with_deterministic_adjustment():
    forecast = StubTimerAdapter(seed=0).forecast(
        {
            "sample_id": "sample-1",
            "target_history": [[10.0], [12.0]],
            "target_future": [[0.0], [0.0], [0.0]],
        },
        {"model_id": "model-a"},
        timeout_seconds=300,
    )

    assert len(forecast) == 3
    assert all(len(row) == 1 for row in forecast)
    assert all(abs(row[0] - 12.0) < 1.0 for row in forecast)
