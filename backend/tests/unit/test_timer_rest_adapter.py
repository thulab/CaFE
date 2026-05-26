from fastapi.testclient import TestClient
import httpx

from app.services.timer_rest_adapter import TimerRestAdapter, TimerServiceError
from stub_service.main import create_app


def _adapter_against_stub() -> TimerRestAdapter:
    # TestClient 是包了 ASGI app 的同步 httpx 客户端，可直接注入适配器。
    client = TestClient(create_app())
    return TimerRestAdapter(base_url=str(client.base_url), api_prefix="/ai/api/v1", client=client)


def _sample(horizon=2):
    return {
        "sample_id": "sample-1",
        "target_column_names": ["target"],
        "history_timestamps": ["2024-01-01T00:00:00", "2024-01-01T01:00:00", "2024-01-01T02:00:00"],
        "future_timestamps": ["2024-01-01T03:00:00"][:horizon] or ["t"] * horizon,
        "target_history": [[1.0], [2.0], [3.0]],
        "target_future": [[0.0]] * horizon,
    }


def test_adapter_returns_horizon_by_dim_forecast():
    adapter = _adapter_against_stub()
    forecast = adapter.forecast(_sample(horizon=2), {"model_id": "x", "remote_model_id": "Timer-3.5"}, timeout_seconds=30)

    assert len(forecast) == 2
    assert all(len(row) == 1 for row in forecast)
    assert all(isinstance(row[0], float) for row in forecast)


def test_adapter_is_deterministic():
    adapter = _adapter_against_stub()
    model = {"model_id": "x", "remote_model_id": "Timer-3.5"}
    first = adapter.forecast(_sample(), model, timeout_seconds=30)
    second = adapter.forecast(_sample(), model, timeout_seconds=30)
    assert first == second


def test_adapter_raises_on_service_error():
    # 指向一个不存在的端点（无 client → 真实 httpx，连接失败）。
    adapter = TimerRestAdapter(base_url="http://127.0.0.1:1", api_prefix="/ai/api/v1")
    try:
        adapter.forecast(_sample(), {"model_id": "x"}, timeout_seconds=1)
    except TimerServiceError:
        return
    raise AssertionError("expected TimerServiceError")


def test_adapter_raises_on_business_error_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 400,
                "message": "Model is not available",
                "service_info": {},
                "data": {"results": []},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    adapter = TimerRestAdapter(base_url=str(client.base_url), api_prefix="/ai/api/v1", client=client)

    try:
        adapter.forecast(_sample(), {"model_id": "x", "remote_model_id": "Timer-3.0"}, timeout_seconds=30)
    except TimerServiceError as exc:
        assert "service code 400" in str(exc)
        return
    raise AssertionError("expected TimerServiceError")


def test_adapter_list_models_returns_loaded_builtins():
    adapter = _adapter_against_stub()
    models = adapter.list_models()
    ids = [m["model_id"] for m in models]
    assert "Timer-3.5" in ids
    assert all(m["loaded"] for m in models if m["model_id"].startswith("Timer"))


def test_adapter_loads_unloaded_model_before_forecast():
    calls: list[tuple[str, str]] = []
    list_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal list_calls
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/models/list"):
            list_calls += 1
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "message": "Success",
                    "data": {
                        "models": [
                            {
                                "model_id": "Timer-3.0",
                                "model_type": "sundial",
                                "category": "builtin",
                                "state": "active",
                                "loaded": list_calls > 1,
                                "base_model_id": None,
                            }
                        ]
                    },
                },
            )
        if request.url.path.endswith("/models/load"):
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "message": "loaded",
                    "data": {"model_id": "Timer-3.0", "devices": ["cpu"]},
                },
            )
        return httpx.Response(404, json={"code": 404, "message": "not found", "data": None})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    adapter = TimerRestAdapter(base_url=str(client.base_url), api_prefix="/ai/api/v1", client=client)

    adapter.ensure_model_loaded({"model_id": "local", "remote_model_id": "Timer-3.0"}, timeout_seconds=30)

    assert calls == [
        ("GET", "/ai/api/v1/models/list"),
        ("POST", "/ai/api/v1/models/load"),
        ("GET", "/ai/api/v1/models/list"),
    ]


def test_adapter_surfaces_model_load_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models/list"):
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "message": "Success",
                    "data": {
                        "models": [
                            {
                                "model_id": "Timer-3.0",
                                "model_type": "sundial",
                                "category": "builtin",
                                "state": "active",
                                "loaded": False,
                                "base_model_id": None,
                            }
                        ]
                    },
                },
            )
        return httpx.Response(
            503,
            json={
                "code": 503,
                "message": "Failed to load model: Failed to spawn ModelWorker",
                "data": None,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")
    adapter = TimerRestAdapter(base_url=str(client.base_url), api_prefix="/ai/api/v1", client=client)

    try:
        adapter.ensure_model_loaded({"model_id": "local", "remote_model_id": "Timer-3.0"}, timeout_seconds=30)
    except TimerServiceError as exc:
        assert "Failed to spawn ModelWorker" in str(exc)
        return
    raise AssertionError("expected TimerServiceError")
