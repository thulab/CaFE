"""选择期目标列校验：load job 从上传列中选择一个或多个 target。"""
from pathlib import Path

from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.dataset import DatasetManifest, Shard
from app.models.sample import SampleIndex
from app.services.dataset_load_service import DatasetLoadService
from app.services.sample_store import SampleStore


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


def test_multi_target_selection_materializes_two_dimensional_samples(tmp_path):
    source = tmp_path / "multi.csv"
    source.write_text(
        "\n".join(
            [
                "time,load,temperature",
                "2026-01-01 00:00:00,10,20",
                "2026-01-01 01:00:00,11,21",
                "2026-01-01 02:00:00,12,22",
                "2026-01-01 03:00:00,13,23",
                "2026-01-01 04:00:00,14,24",
                "2026-01-01 05:00:00,15,25",
            ]
        ),
        encoding="utf-8",
    )
    with _make_session(tmp_path) as session:
        manifest = DatasetManifest(name="multi", domain="energy", source_uri=str(source), time_column="time")
        session.add(manifest)
        session.commit()
        session.refresh(manifest)
        service = DatasetLoadService(tmp_path / "runtime")

        job = service.create_load_job(
            session,
            manifest.dataset_manifest_id,
            {"context_length": 3, "horizon": 2, "stride": 2, "target_columns": ["load", "temperature"]},
        )

        assert job.status == "succeeded"
        shard = session.get(Shard, job.output_shard_id)
        assert shard.target_columns == ["load", "temperature"]
        assert shard.target_dim == 2
        sample_index = session.exec(select(SampleIndex).where(SampleIndex.shard_id == shard.shard_id)).first()
        sample = SampleStore().read_by_ref(session, sample_index.storage_ref)
        assert sample["target_column_names"] == ["load", "temperature"]
        assert sample["target_history"] == [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]
        assert sample["target_future"] == [[13.0, 23.0], [14.0, 24.0]]


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
