from pathlib import Path

import pytest
from sqlmodel import Session, create_engine, select

from app.core.errors import ApiError
from app.db.init_db import init_db
from app.models.dataset import DatasetManifest, Shard
from app.models.model_registry import Model
from app.models.sample import SampleIndex
from app.services.dataset_load_service import DatasetLoadService
from app.services.run_executor import create_benchmarking_run
from app.services.sample_store import SampleStore
from app.services.track_service import create_real_capability_block, create_track_with_blocks


def _make_covariate_track(session: Session, tmp_path: Path):
    source = tmp_path / "with_covariates.csv"
    source.write_text(
        "\n".join(
            [
                "time,target,promo,temp",
                "2026-01-01 00:00:00,10,0,21",
                "2026-01-01 01:00:00,11,0,22",
                "2026-01-01 02:00:00,12,1,23",
                "2026-01-01 03:00:00,13,1,24",
                "2026-01-01 04:00:00,14,0,25",
                "2026-01-01 05:00:00,15,0,26",
            ]
        ),
        encoding="utf-8",
    )
    manifest = DatasetManifest(name="covariates", domain="energy", source_uri=str(source), time_column="time")
    session.add(manifest)
    session.commit()
    session.refresh(manifest)
    job = DatasetLoadService(tmp_path / "runtime").create_load_job(
        session,
        manifest.dataset_manifest_id,
        {
            "context_length": 3,
            "horizon": 2,
            "stride": 2,
            "target_columns": ["target"],
            "covariate_columns": ["promo", "temp"],
        },
    )
    block = create_real_capability_block(session, "covariate block", [job.output_shard_id])
    track, _ranking = create_track_with_blocks(session, "covariate track", [block.capability_block_id], "mase")
    return track, job


def test_load_job_materializes_known_future_covariates(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        _track, job = _make_covariate_track(session, tmp_path)
        shard = session.get(Shard, job.output_shard_id)
        sample_index = session.exec(select(SampleIndex).where(SampleIndex.shard_id == shard.shard_id)).first()

        sample = SampleStore().read_by_ref(session, sample_index.storage_ref)

        assert shard.covariate_columns == ["promo", "temp"]
        assert shard.covariate_dim == 2
        assert sample["covariate_column_names"] == ["promo", "temp"]
        assert sample["history_cov"] == [[0.0, 21.0], [0.0, 22.0], [1.0, 23.0]]
        assert sample["future_cov"] == [[1.0, 24.0], [0.0, 25.0]]
        assert sample["target_history"] == [[10.0], [11.0], [12.0]]
        assert job.validation_summary["covariate_columns"] == ["promo", "temp"]


def test_covariate_run_rejects_models_without_covariate_capacity(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _job = _make_covariate_track(session, tmp_path)
        model = Model(
            name="Timer 3.0",
            model_family="Timer",
            model_version="3.0",
            endpoint_uri="stub://timer",
            forecast_limits={"max_target_count": 1, "max_covariate_count": 0},
        )
        session.add(model)
        session.commit()
        session.refresh(model)

        with pytest.raises(ApiError) as exc:
            create_benchmarking_run(session, track.track_id, [model.model_id])

        assert exc.value.error_code == "model_covariate_dim_unsupported"
        assert exc.value.details["covariate_dim"] == 2
        assert exc.value.details["model_ids"] == [model.model_id]


def test_covariate_run_allows_models_with_covariate_capacity(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _job = _make_covariate_track(session, tmp_path)
        model = Model(
            name="Chronos 2",
            model_family="Chronos",
            model_version="2",
            endpoint_uri="stub://chronos",
            forecast_limits={"max_target_count": 1, "max_covariate_count": 50},
        )
        session.add(model)
        session.commit()
        session.refresh(model)

        run = create_benchmarking_run(session, track.track_id, [model.model_id])

        assert run.status == "queued"
