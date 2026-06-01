from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_mvp_benchmarking_flow_from_csv_upload_to_sample_forecast(client):
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"

    with source.open("rb") as file:
        upload = client.post("/dataset-manifests/upload", files={"file": ("valid_hourly_20.csv", file, "text/csv")})
    assert upload.status_code == 200
    source_uri = upload.json()["source_uri"]

    manifest = client.post(
        "/dataset-manifests",
        json={"name": "valid hourly", "domain": "energy", "source_uri": source_uri, "file_format": "csv", "time_column": "time"},
    )
    assert manifest.status_code == 200

    job = client.post(
        "/dataset-load-jobs",
        json={"dataset_manifest_id": manifest.json()["dataset_manifest_id"], "split_config": {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]}},
    )
    assert job.status_code == 200
    shard_id = job.json()["output_shard_id"]

    shard = client.get(f"/shards/{shard_id}")
    assert shard.json()["sample_count"] == 4

    track = client.post("/wizard/real-dataset-track", json={"name": "real track", "shard_ids": [shard_id], "primary_metric_id": "mse"})
    assert track.status_code == 200
    track_id = track.json()["track_id"]

    models = client.get("/models")
    assert models.status_code == 200
    selected_model_ids = [item["model_id"] for item in models.json()["items"][:2]]

    run = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": selected_model_ids})
    assert run.status_code == 200
    run_id = run.json()["benchmarking_run_id"]

    progress = {}
    for _ in range(30):
        progress = client.get(f"/benchmarking-runs/{run_id}/progress").json()
        if progress["status"] == "succeeded":
            break
    assert progress["status"] == "succeeded"
    assert progress["report_id"]

    ranking = client.get(f"/tracks/{track_id}/ranking", params={"metric": "mse", "policy": "latest_valid_result"})
    assert ranking.status_code == 200
    assert len(ranking.json()["items"]) == 2

    report = client.get(f"/reports/{progress['report_id']}")
    assert report.status_code == 200
    sample_id = report.json()["sample_forecast_links"][0]["sample_id"]

    sample_forecast = client.get(f"/samples/{sample_id}/forecast", params={"run_id": run_id})
    assert sample_forecast.status_code == 200
    assert len(sample_forecast.json()["models"]) == 2
