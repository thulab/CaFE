from fastapi.testclient import TestClient

from app.main import create_app
from tests.api.test_benchmarking_run_create import create_track


def test_run_progress_dto_contains_run_units_tasks_counts_and_events():
    client = TestClient(create_app())
    track_id, model_id = create_track(client)
    run = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_id]}).json()

    response = client.get(f"/benchmarking-runs/{run['benchmarking_run_id']}/progress")

    assert response.status_code == 200
    body = response.json()
    assert body["benchmarking_run_id"] == run["benchmarking_run_id"]
    assert body["status"] in {"queued", "running", "succeeded"}
    assert body["progress"]["total_models"] == 1
    assert body["progress"]["total_tasks"] == 1
    assert body["units"][0]["model_id"] == model_id
    assert body["tasks"][0]["model_id"] == model_id
    assert body["recent_events"][0]["created_at"]
