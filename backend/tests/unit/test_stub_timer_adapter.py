from app.services.stub_timer_adapter import StubTimerAdapter


def sample(sample_id: str = "sample-1"):
    return {
        "sample_id": sample_id,
        "target_history": [[1.0], [2.0], [3.0]],
        "target_future": [[4.0], [5.0]],
        "future_timestamps": ["2026-01-01T03:00:00", "2026-01-01T04:00:00"],
    }


def test_stub_forecast_is_deterministic_for_model_sample_and_seed():
    adapter = StubTimerAdapter(seed=7)

    first = adapter.forecast(sample(), {"model_id": "model-a"}, timeout_seconds=300)
    second = adapter.forecast(sample(), {"model_id": "model-a"}, timeout_seconds=300)

    assert first == second


def test_different_model_bias_can_change_output():
    adapter = StubTimerAdapter(seed=7)

    first = adapter.forecast(sample(), {"model_id": "model-a"}, timeout_seconds=300)
    second = adapter.forecast(sample(), {"model_id": "model-b"}, timeout_seconds=300)

    assert first != second
