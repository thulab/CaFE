from fastapi.testclient import TestClient

from app.main import create_app
from tests.api.test_benchmarking_run_create import create_track


def test_sample_forecast_api_returns_history_truth_forecasts_and_metrics():
    client = TestClient(create_app())
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
    assert body["models"][0]["model_id"] == model_id
    assert "mse" in body["models"][0]["metrics"]
    # 回看链接：前端要能从样本预测跳回报告与榜单，report/ranking 必须是真实 id 而非 None。
    links = body["links"]
    assert links["run"] == run["benchmarking_run_id"]
    assert links["report"] == report["report_id"]
    assert links["ranking"]
