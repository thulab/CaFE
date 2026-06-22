import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.core.errors import ApiError
from app.db.init_db import assert_manifest_can_create_successful_real_shard, assert_manifest_can_succeed_load
from app.models.dataset import DatasetLoadJob, DatasetManifest, Shard


def make_session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_manifest_allows_many_failed_load_jobs_but_only_one_succeeded_job():
    with make_session() as session:
        manifest = DatasetManifest(name="demo", domain="energy", source_uri="file.csv", time_column="time")
        session.add(manifest)
        session.commit()

        session.add(DatasetLoadJob(dataset_manifest_id=manifest.dataset_manifest_id, status="failed", split_config={}))
        session.add(DatasetLoadJob(dataset_manifest_id=manifest.dataset_manifest_id, status="failed", split_config={}))
        session.commit()

        assert_manifest_can_succeed_load(session, manifest.dataset_manifest_id)
        session.add(DatasetLoadJob(dataset_manifest_id=manifest.dataset_manifest_id, status="succeeded", split_config={}))
        session.commit()

        with pytest.raises(ApiError, match="already has a successful load job"):
            assert_manifest_can_succeed_load(session, manifest.dataset_manifest_id)


def test_manifest_allows_only_one_successful_real_shard():
    with make_session() as session:
        manifest = DatasetManifest(name="demo", domain="energy", source_uri="file.csv", time_column="time")
        session.add(manifest)
        session.commit()

        assert_manifest_can_create_successful_real_shard(session, manifest.dataset_manifest_id)
        session.add(Shard(dataset_manifest_id=manifest.dataset_manifest_id, shard_type="real", status="ready", source_uri="file.csv"))
        session.commit()

        with pytest.raises(ApiError, match="already has a successful real shard"):
            assert_manifest_can_create_successful_real_shard(session, manifest.dataset_manifest_id)
