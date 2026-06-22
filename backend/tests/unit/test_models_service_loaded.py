from fastapi.testclient import TestClient

from app.core.config import Settings
from app.services.model_adapter import service_loaded_model_ids
from app.services.timer_rest_adapter import TimerRestAdapter
from stub_service.main import create_app


def test_service_loaded_ids_none_in_stub_mode():
    assert service_loaded_model_ids(Settings(model_adapter="stub")) is None


def test_service_loaded_ids_from_stub_service():
    client = TestClient(create_app())
    adapter = TimerRestAdapter(base_url=str(client.base_url), api_prefix="/ai/api/v1", client=client)

    loaded = service_loaded_model_ids(Settings(model_adapter="rest"), adapter=adapter)

    assert loaded is not None
    assert {"Timer-3.5", "Timer-3.0", "Chronos-2"} <= loaded


def test_service_loaded_ids_none_when_unreachable():
    settings = Settings(model_adapter="rest", timer_service_base_url="http://127.0.0.1:1")
    assert service_loaded_model_ids(settings) is None
