from fastapi.testclient import TestClient

from app.main import create_app
from tests.api.test_benchmarking_run_create import create_track


def test_ranking_list_api_returns_policy_view():
    client = TestClient(create_app())
    track_id, model_id = create_track(client)
    run = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_id]}).json()
    for _ in range(20):
        progress = client.get(f"/benchmarking-runs/{run['benchmarking_run_id']}/progress").json()
        if progress["status"] == "succeeded":
            break

    response = client.get(f"/tracks/{track_id}/ranking", params={"metric": "mse", "policy": "latest_valid_result"})

    assert response.status_code == 200
    assert response.json()["items"][0]["model_id"] == model_id
    assert response.json()["items"][0]["rank"] == 1
