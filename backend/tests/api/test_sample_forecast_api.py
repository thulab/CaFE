from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models.model_registry import Model
from app.main import create_app
from tests.api.test_benchmarking_run_create import create_track


def test_sample_forecast_api_returns_history_truth_forecasts_and_metrics(client):
    track_id, model_id = create_track(client)
    run = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_id]}).json()
    for _ in range(20):
        progress = client.get(f"/benchmarking-runs/{run['benchmarking_run_id']}/progress").json()
        if progress["status"] == "succeeded":
            break
    report = client.get(f"/benchmarking-runs/{run['benchmarking_run_id']}/progress").json()
    sample_id = client.get(f"/reports/{report['report_id']}").json()["sample_forecast_links"][0]["sample_id"]

    response = client.get(f"/samples/{sample_id}/forecast", params={"run_id": run["benchmarking_run_id"]})

    assert response.status_code == 200
    body = response.json()
    assert body["sample_id"] == sample_id
    assert body["target_history"]
    assert body["target_future"]
    assert body["sample_index"] == 0
    assert body["horizon_start"] <= body["horizon_end"]
    assert body["forecast_start_at"]
    assert body["forecast_end_at"]
    assert body["models"][0]["model_id"] == model_id
    assert "mse" in body["models"][0]["metrics"]
    # 回看链接：前端要能从样本预测跳回报告与榜单，report/ranking 必须是真实 id 而非 None。
    links = body["links"]
    assert links["run"] == run["benchmarking_run_id"]
    assert links["report"] == report["report_id"]
    assert links["ranking"]


def test_track_sample_forecast_aggregates_latest_successful_model_runs(client):
    track_id, model_a_id = create_track(client)
    with Session(client.app.state.engine) as session:
        model_b = Model(name="Model B", model_family="Timer", model_version="b", endpoint_uri="stub://b")
        session.add(model_b)
        session.commit()
        session.refresh(model_b)
        model_b_id = model_b.model_id

    run_a = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_a_id]}).json()
    run_b = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_b_id]}).json()
    for run in [run_a, run_b]:
        for _ in range(20):
            progress = client.get(f"/benchmarking-runs/{run['benchmarking_run_id']}/progress").json()
            if progress["status"] == "succeeded":
                break

    track_results = client.get(f"/tracks/{track_id}/results", params={"sample_link_limit": 1}).json()
    sample_id = track_results["sample_forecast_links"][0]["sample_id"]

    response = client.get(f"/samples/{sample_id}/forecast", params={"track_id": track_id})

    assert response.status_code == 200
    body = response.json()
    assert body["sample_id"] == sample_id
    assert body["links"]["track"] == track_id
    assert body["links"]["run"] is None
    assert body["links"]["report"] is None
    assert {model["model_id"] for model in body["models"]} == {model_a_id, model_b_id}
