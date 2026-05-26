from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def create_track(client: TestClient) -> tuple[str, str]:
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    with source.open("rb") as file:
        upload = client.post("/dataset-manifests/upload", files={"file": ("valid_hourly_20.csv", file, "text/csv")}).json()
    manifest = client.post(
        "/dataset-manifests",
        json={"name": "valid hourly", "domain": "energy", "source_uri": upload["source_uri"], "file_format": "csv", "time_column": "time", "value_columns": ["target"]},
    ).json()
    job = client.post(
        "/dataset-load-jobs",
        json={"dataset_manifest_id": manifest["dataset_manifest_id"], "split_config": {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]}},
    ).json()
    wizard = client.post("/wizard/real-dataset-track", json={"name": "real track", "shard_ids": [job["output_shard_id"]], "primary_metric_id": "mse"}).json()
    model = client.get("/models").json()["items"][0]
    return wizard["track_id"], model["model_id"]


def test_create_run_returns_immediately_with_queued_or_running_status(client):
    track_id, model_id = create_track(client)

    response = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_id]})

    assert response.status_code == 200
    assert response.json()["benchmarking_run_id"]
    assert response.json()["status"] in {"queued", "running"}
