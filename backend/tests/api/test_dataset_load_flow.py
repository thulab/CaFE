from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_dataset_load_flow_uploads_manifest_loads_shard_and_previews_samples():
    client = TestClient(create_app())
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"

    with source.open("rb") as file:
        upload = client.post("/dataset-manifests/upload", files={"file": ("valid_hourly_20.csv", file, "text/csv")})
    assert upload.status_code == 200
    upload_body = upload.json()
    assert upload_body["detected_delimiter"] == ","
    assert upload_body["columns"][1]["name"] == "target"

    manifest = client.post(
        "/dataset-manifests",
        json={
            "name": "valid hourly",
            "domain": "energy",
            "source_uri": upload_body["source_uri"],
            "file_format": "csv",
            "time_column": "time",
            "value_columns": ["target"],
        },
    )
    assert manifest.status_code == 200
    manifest_id = manifest.json()["dataset_manifest_id"]

    job = client.post(
        "/dataset-load-jobs",
        json={"dataset_manifest_id": manifest_id, "split_config": {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]}, "seed": 7},
    )
    assert job.status_code == 200
    job_body = job.json()
    assert job_body["status"] == "succeeded"
    shard_id = job_body["output_shard_id"]

    loaded_job = client.get(f"/dataset-load-jobs/{job_body['load_job_id']}")
    assert loaded_job.json()["status"] == "succeeded"

    shard = client.get(f"/shards/{shard_id}")
    assert shard.status_code == 200
    assert shard.json()["sample_count"] == 4

    samples = client.get(f"/shards/{shard_id}/samples", params={"limit": 20, "offset": 0})
    assert samples.status_code == 200
    sample_items = samples.json()["items"]
    assert len(sample_items) == 4

    preview = client.get(f"/samples/{sample_items[0]['sample_id']}/preview")
    assert preview.status_code == 200
    assert preview.json()["target_history"][0] == [10.0]
    assert preview.json()["target_future"][0] == [16.0]
