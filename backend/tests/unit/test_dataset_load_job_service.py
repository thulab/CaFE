from pathlib import Path

import pytest
from sqlmodel import Session, create_engine, select

from app.core.errors import ApiError
from app.db.init_db import init_db
from app.models.dataset import DatasetManifest, Shard
from app.models.sample import SampleIndex
from app.services.dataset_load_service import DatasetLoadService


def make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return Session(engine)


def create_manifest(session: Session, source_uri: str) -> DatasetManifest:
    manifest = DatasetManifest(
        name="demo",
        domain="energy",
        source_uri=source_uri,
        time_column="time",
        value_columns=["target"],
    )
    session.add(manifest)
    session.commit()
    session.refresh(manifest)
    return manifest


def test_successful_manifest_cannot_reload(tmp_path):
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    with make_session(tmp_path) as session:
        manifest = create_manifest(session, str(source))
        service = DatasetLoadService(tmp_path / "runtime")

        job = service.create_load_job(session, manifest.dataset_manifest_id, {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]})

        assert job.status == "succeeded"
        assert session.exec(select(SampleIndex).where(SampleIndex.shard_id == job.output_shard_id)).all()
        with pytest.raises(ApiError) as exc:
            service.create_load_job(session, manifest.dataset_manifest_id, {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]})
        assert exc.value.error_code == "dataset_manifest_already_loaded"


def test_failed_load_cleans_intermediate_artifacts_and_can_retry(tmp_path):
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    with make_session(tmp_path) as session:
        manifest = create_manifest(session, str(source))
        service = DatasetLoadService(tmp_path / "runtime")

        failed = service.create_load_job(session, manifest.dataset_manifest_id, {"context_length": 30, "horizon": 3, "stride": 3, "target_columns": ["target"]})

        assert failed.status == "failed"
        assert not list((tmp_path / "runtime" / "tsfiles").glob("*.tsfile"))

        retried = service.create_load_job(session, manifest.dataset_manifest_id, {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]})

        assert retried.status == "succeeded"
        assert retried.output_shard_id is not None


def test_load_job_persists_optional_shard_name(tmp_path):
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    with make_session(tmp_path) as session:
        manifest = create_manifest(session, str(source))
        service = DatasetLoadService(tmp_path / "runtime")

        job = service.create_load_job(
            session,
            manifest.dataset_manifest_id,
            {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"], "shard_name": "Energy validation slice"},
        )

        shard = session.get(Shard, job.output_shard_id)
        assert shard.name == "Energy validation slice"
