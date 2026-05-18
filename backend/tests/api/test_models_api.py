from fastapi.testclient import TestClient

from app.main import create_app


def test_models_api_lists_five_mvp_stub_models():
    client = TestClient(create_app())

    response = client.get("/models")

    assert response.status_code == 200
    names = [model["name"] for model in response.json()["items"]]
    assert names == ["Timer 3.5", "Timer 3.0", "Chronos 2", "toto", "TimesFM 2.5"]
    assert all(model["adapter_type"] == "timer_service" for model in response.json()["items"])
