from fastapi.testclient import TestClient

from app.main import create_app


def test_create_model_rejects_duplicate_name():
    client = TestClient(create_app())
    payload = {"name": "Dup Model", "model_family": "Timer", "model_version": "1"}

    first = client.post("/models", json=payload)
    assert first.status_code == 200

    second = client.post("/models", json=payload)
    assert second.status_code == 400
    assert second.json()["error_code"] == "model_name_taken"
