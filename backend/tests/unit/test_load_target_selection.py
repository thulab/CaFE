"""选择期目标列校验：单变量 MVP 只从 load job 选择 target。"""
from pathlib import Path

from sqlmodel import Session, create_engine

from app.db.init_db import init_db
from app.models.dataset import DatasetManifest, Shard
from app.services.dataset_load_service import DatasetLoadService


def _make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return Session(engine)


def _manifest(session):
    source = Path(__file__).parents[1] / "fixtures" / "valid_hourly_20.csv"
    manifest = DatasetManifest(name="demo", domain="energy", source_uri=str(source), time_column="time")
    session.add(manifest)
    session.commit()
    session.refresh(manifest)
    return manifest


def test_valid_target_selection_records_only_target_columns(tmp_path):
    with _make_session(tmp_path) as session:
        manifest = _manifest(session)
        service = DatasetLoadService(tmp_path / "runtime")

        job = service.create_load_job(
            session, manifest.dataset_manifest_id, {"context_length": 6, "horizon": 3, "target_columns": ["target"]}
        )

        assert job.status == "succeeded"
        shard = session.get(Shard, job.output_shard_id)
        assert shard.target_columns == ["target"]
        assert shard.target_dim == 1
        assert "value_columns" not in shard.model_dump()


def test_missing_target_columns_fails(tmp_path):
    with _make_session(tmp_path) as session:
        manifest = _manifest(session)
        service = DatasetLoadService(tmp_path / "runtime")

        job = service.create_load_job(session, manifest.dataset_manifest_id, {"context_length": 6, "horizon": 3})

        assert job.status == "failed"
        assert job.error_code == "load_target_columns_invalid"


def test_unknown_target_column_fails(tmp_path):
    with _make_session(tmp_path) as session:
        manifest = _manifest(session)
        service = DatasetLoadService(tmp_path / "runtime")

        job = service.create_load_job(
            session, manifest.dataset_manifest_id, {"context_length": 6, "horizon": 3, "target_columns": ["missing"]}
        )

        assert job.status == "failed"
        assert job.error_code == "csv_target_column_missing"
