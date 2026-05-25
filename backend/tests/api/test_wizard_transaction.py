from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.routes import wizard
from app.core.errors import ApiError
from app.main import create_app
from app.models.benchmark import CapabilityBlock
from app.models.dataset import Shard


def loaded_shard_id(client: TestClient) -> str:
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    with source.open("rb") as file:
        upload = client.post(
            "/dataset-manifests/upload",
            files={"file": ("valid_hourly_20.csv", file, "text/csv")},
        ).json()
    manifest = client.post(
        "/dataset-manifests",
        json={
            "name": "valid hourly",
            "domain": "energy",
            "source_uri": upload["source_uri"],
            "file_format": "csv",
            "time_column": "time",
            "value_columns": ["target"],
        },
    ).json()
    job = client.post(
        "/dataset-load-jobs",
        json={
            "dataset_manifest_id": manifest["dataset_manifest_id"],
            "split_config": {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]},
        },
    ).json()
    return job["output_shard_id"]


def test_wizard_step2_failure_leaves_no_orphan_block_or_assigned_shard(monkeypatch: pytest.MonkeyPatch):
    app = create_app()
    client = TestClient(app)
    shard_id = loaded_shard_id(client)

    def boom(*args, **kwargs):
        raise ApiError("track_create_failed", "boom")

    monkeypatch.setattr(wizard, "create_track_with_blocks", boom)

    response = client.post(
        "/wizard/real-dataset-track",
        json={"name": "real track", "shard_ids": [shard_id], "primary_metric_id": "mse"},
    )

    assert response.status_code >= 400

    with Session(app.state.engine) as session:
        blocks = session.exec(select(CapabilityBlock)).all()
        assert len(blocks) == 0, "step 2 failure must not leave an orphan capability block"

        shard = session.get(Shard, shard_id)
        assert shard is not None
        assert shard.capability_block_id is None, "shard must be freed so it can be reused"


def test_wizard_step1_failure_creates_nothing(monkeypatch: pytest.MonkeyPatch):
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/wizard/real-dataset-track",
        json={"name": "real track", "shard_ids": ["does-not-exist"], "primary_metric_id": "mse"},
    )

    assert response.status_code >= 400
    with Session(app.state.engine) as session:
        blocks = session.exec(select(CapabilityBlock)).all()
        assert len(blocks) == 0
