"""TDD tests for input/answer separation.

These tests verify that:
1. build_model_input() strips target_future from the model input view.
2. StubTimerAdapter.forecast() works with model_input (no target_future, only horizon).
3. TimerRestAdapter.forecast() with model_input sends output_length=[horizon] and parses
   the response without needing target_future.
"""
import json

import httpx
import pytest

from app.services.model_input import build_model_input
from app.services.stub_timer_adapter import StubTimerAdapter
from app.services.timer_rest_adapter import TimerRestAdapter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _full_sample(horizon: int = 3, target_dim: int = 2) -> dict:
    """A complete materialized sample dict (sample.v1 shape)."""
    return {
        "sample_id": "sample-42",
        "shard_id": "shard-0",
        "sample_index": 0,
        "target_column_names": ["col_a", "col_b"],
        "history_timestamps": ["2024-01-01T00:00:00", "2024-01-01T01:00:00", "2024-01-01T02:00:00"],
        "future_timestamps": [f"2024-01-01T0{3 + i}:00:00" for i in range(horizon)],
        "target_history": [[float(i), float(i + 0.5)] for i in range(3)],
        "target_future": [[float(10 + i), float(10.5 + i)] for i in range(horizon)],
        "history_cov": None,
        "future_cov": None,
        "source_row_start": 0,
        "source_row_end": 5,
    }


# ---------------------------------------------------------------------------
# Tests for build_model_input
# ---------------------------------------------------------------------------

class TestBuildModelInput:
    def test_target_future_not_in_model_input(self):
        sample = _full_sample(horizon=3)
        model_input = build_model_input(sample)
        assert "target_future" not in model_input

    def test_horizon_equals_len_target_future(self):
        sample = _full_sample(horizon=5)
        model_input = build_model_input(sample)
        assert model_input["horizon"] == 5
        assert isinstance(model_input["horizon"], int)

    def test_preserves_required_keys(self):
        sample = _full_sample(horizon=3)
        model_input = build_model_input(sample)
        for key in ("target_history", "history_timestamps", "future_timestamps",
                    "target_column_names", "sample_id"):
            assert key in model_input, f"expected key '{key}' to be present"

    def test_preserves_target_history_values(self):
        sample = _full_sample(horizon=3)
        model_input = build_model_input(sample)
        assert model_input["target_history"] == sample["target_history"]

    def test_preserves_history_cov_when_present(self):
        sample = _full_sample(horizon=3)
        sample["history_cov"] = [[0.1], [0.2], [0.3]]
        model_input = build_model_input(sample)
        assert model_input.get("history_cov") == sample["history_cov"]

    def test_preserves_future_cov_when_present(self):
        sample = _full_sample(horizon=3)
        sample["future_cov"] = [[0.4], [0.5], [0.6]]
        model_input = build_model_input(sample)
        assert model_input.get("future_cov") == sample["future_cov"]

    def test_does_not_contain_ground_truth_values(self):
        """No answer values should leak into model_input."""
        sample = _full_sample(horizon=3)
        model_input = build_model_input(sample)
        # target_future values must not appear as a value in the dict
        assert "target_future" not in model_input
        # And the horizon key contains an int, not the actual future list
        assert not isinstance(model_input["horizon"], list)


# ---------------------------------------------------------------------------
# Tests for StubTimerAdapter with model_input (no target_future)
# ---------------------------------------------------------------------------

class TestStubTimerAdapterWithModelInput:
    def _model_input(self, horizon: int = 3, target_dim: int = 2) -> dict:
        sample = _full_sample(horizon=horizon, target_dim=target_dim)
        return build_model_input(sample)

    def test_forecast_returns_correct_shape(self):
        adapter = StubTimerAdapter(seed=0)
        model_input = self._model_input(horizon=4, target_dim=2)
        model = {"model_id": "model-x"}
        result = adapter.forecast(model_input, model, timeout_seconds=1)
        assert len(result) == 4
        assert all(len(row) == 2 for row in result)

    def test_forecast_is_deterministic(self):
        adapter = StubTimerAdapter(seed=7)
        model_input = self._model_input(horizon=3)
        model = {"model_id": "model-y", "stub_seed": 42}
        first = adapter.forecast(model_input, model, timeout_seconds=1)
        second = adapter.forecast(model_input, model, timeout_seconds=1)
        assert first == second

    def test_determinism_across_adapter_instances(self):
        model_input = self._model_input(horizon=3)
        model = {"model_id": "model-z"}
        result_a = StubTimerAdapter(seed=5).forecast(model_input, model, timeout_seconds=1)
        result_b = StubTimerAdapter(seed=5).forecast(model_input, model, timeout_seconds=1)
        assert result_a == result_b


# ---------------------------------------------------------------------------
# Tests for TimerRestAdapter with model_input (no target_future)
# ---------------------------------------------------------------------------

def _fake_forecast_response(horizon: int, n_cols: int) -> dict:
    """Build a minimal /forecast response payload."""
    columns = ["time"] + [f"col_{i}" for i in range(n_cols)]
    data = [[f"2024-01-01T0{3 + i}:00:00"] + [float(i) * 0.1] * n_cols for i in range(horizon)]
    return {
        "code": 200,
        "data": {
            "results": [
                {"columns": columns, "data": data}
            ]
        },
    }


class _FakeTransport(httpx.BaseTransport):
    """Fake httpx transport that captures the last request and returns a preset response."""

    def __init__(self, response_payload: dict, status_code: int = 200):
        self._payload = response_payload
        self._status_code = status_code
        self.last_request_body: dict | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body_bytes = request.read()
        self.last_request_body = json.loads(body_bytes)
        return httpx.Response(
            status_code=self._status_code,
            json=self._payload,
            request=request,
        )


class TestTimerRestAdapterWithModelInput:
    def _model_input(self, horizon: int = 3, n_cols: int = 1) -> dict:
        sample = _full_sample(horizon=horizon, target_dim=n_cols)
        return build_model_input(sample)

    def _adapter_with_fake(self, response_payload: dict) -> tuple[TimerRestAdapter, _FakeTransport]:
        transport = _FakeTransport(response_payload)
        client = httpx.Client(transport=transport)
        adapter = TimerRestAdapter(
            base_url="http://fake-timer-service",
            api_prefix="/ai/api/v1",
            client=client,
        )
        return adapter, transport

    def test_request_body_output_length_equals_horizon(self):
        horizon = 5
        model_input = self._model_input(horizon=horizon, n_cols=1)
        resp = _fake_forecast_response(horizon=horizon, n_cols=1)
        adapter, transport = self._adapter_with_fake(resp)

        adapter.forecast(model_input, {"model_id": "m1", "remote_model_id": "Timer-3.5"}, timeout_seconds=5)

        assert transport.last_request_body is not None
        assert transport.last_request_body["output_length"] == [horizon]

    def test_response_parsed_without_target_future(self):
        horizon = 3
        n_cols = 2
        model_input = self._model_input(horizon=horizon, n_cols=n_cols)
        assert "target_future" not in model_input  # sanity-check the fixture

        resp = _fake_forecast_response(horizon=horizon, n_cols=n_cols)
        adapter, _ = self._adapter_with_fake(resp)

        result = adapter.forecast(model_input, {"model_id": "m1", "remote_model_id": "Timer-3.5"}, timeout_seconds=5)

        assert len(result) == horizon
        assert all(len(row) == n_cols for row in result)

    def test_forecast_truncates_to_horizon(self):
        """Even if the service returns extra rows, we truncate to horizon."""
        horizon = 2
        model_input = self._model_input(horizon=horizon, n_cols=1)
        # Response has more rows than requested
        resp = _fake_forecast_response(horizon=5, n_cols=1)
        adapter, _ = self._adapter_with_fake(resp)

        result = adapter.forecast(model_input, {"model_id": "m1", "remote_model_id": "Timer-3.5"}, timeout_seconds=5)

        assert len(result) == horizon
