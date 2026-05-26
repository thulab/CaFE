from fastapi.testclient import TestClient

import app.services.model_catalog as model_catalog
from app.core.config import get_settings
from app.main import create_app


class _FakeTimerRestAdapter:
    loaded_by_id = {"toto2.0": True, "Timer-3.0": False}

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        pass

    def list_models(self, timeout_seconds: int = 10) -> list[dict]:  # noqa: ARG002
        return [
            {
                "model_id": "toto2.0",
                "model_type": "toto2p0",
                "category": "builtin",
                "state": "active",
                "loaded": self.loaded_by_id["toto2.0"],
                "base_model_id": None,
            },
            {
                "model_id": "Timer-3.0",
                "model_type": "sundial",
                "category": "builtin",
                "state": "active",
                "loaded": self.loaded_by_id["Timer-3.0"],
                "base_model_id": None,
            },
        ]

    def ensure_model_loaded(self, model: dict, timeout_seconds: int = 600) -> None:  # noqa: ARG002
        self.loaded_by_id[str(model["remote_model_id"])] = True


def test_models_endpoint_uses_timer_service_catalog_in_rest_mode(monkeypatch):
    monkeypatch.setenv("TSBENCHMARK_MODEL_ADAPTER", "rest")
    get_settings.cache_clear()
    _FakeTimerRestAdapter.loaded_by_id = {"toto2.0": True, "Timer-3.0": False}
    monkeypatch.setattr(model_catalog, "TimerRestAdapter", _FakeTimerRestAdapter)

    app = create_app()
    client = TestClient(app)
    token = client.post("/auth/login", json={"username": "admin", "password": "test-admin-pw"}).json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"

    response = client.get("/models")

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["model_id"] for item in items] == ["toto2.0", "Timer-3.0"]
    assert [item["loaded"] for item in items] == [True, False]
    assert [item["remote_model_id"] for item in items] == ["toto2.0", "Timer-3.0"]


def test_load_model_endpoint_loads_timer_service_model(monkeypatch):
    monkeypatch.setenv("TSBENCHMARK_MODEL_ADAPTER", "rest")
    get_settings.cache_clear()
    _FakeTimerRestAdapter.loaded_by_id = {"toto2.0": True, "Timer-3.0": False}
    monkeypatch.setattr(model_catalog, "TimerRestAdapter", _FakeTimerRestAdapter)

    app = create_app()
    client = TestClient(app)
    token = client.post("/auth/login", json={"username": "admin", "password": "test-admin-pw"}).json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"

    response = client.post("/models/Timer-3.0/load")

    assert response.status_code == 200
    assert response.json()["model_id"] == "Timer-3.0"
    assert response.json()["loaded"] is True
