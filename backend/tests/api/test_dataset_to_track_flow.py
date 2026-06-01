from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def loaded_shard_id(client: TestClient) -> str:
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    with source.open("rb") as file:
        upload = client.post("/dataset-manifests/upload", files={"file": ("valid_hourly_20.csv", file, "text/csv")}).json()
    manifest = client.post(
        "/dataset-manifests",
        json={
            "name": "valid hourly",
            "domain": "energy",
            "source_uri": upload["source_uri"],
            "file_format": "csv",
            "time_column": "time",
        },
    ).json()
    job = client.post(
        "/dataset-load-jobs",
        json={"dataset_manifest_id": manifest["dataset_manifest_id"], "split_config": {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]}},
    ).json()
    return job["output_shard_id"]


def test_wizard_real_dataset_track_returns_track_block_and_ranking_ids(client):
    shard_id = loaded_shard_id(client)

    response = client.post(
        "/wizard/real-dataset-track",
        json={"name": "real track", "shard_ids": [shard_id], "primary_metric_id": "mse"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["track_id"]
    assert body["capability_block_id"]
    assert body["ranking_list_id"]
