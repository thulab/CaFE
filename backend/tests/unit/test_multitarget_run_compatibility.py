from pathlib import Path

import pytest
from sqlmodel import Session, create_engine

from app.core.errors import ApiError
from app.db.init_db import init_db
from app.models.dataset import DatasetManifest
from app.models.model_registry import Model
from app.services.dataset_load_service import DatasetLoadService
from app.services.run_executor import create_benchmarking_run
from app.services.track_service import create_real_capability_block, create_track_with_blocks


def _make_multi_track(session: Session, tmp_path: Path):
    source = tmp_path / "multi.csv"
    source.write_text(
        "\n".join(
            [
                "time,target_a,target_b",
                "2026-01-01 00:00:00,1,10",
                "2026-01-01 01:00:00,2,11",
                "2026-01-01 02:00:00,3,12",
                "2026-01-01 03:00:00,4,13",
                "2026-01-01 04:00:00,5,14",
                "2026-01-01 05:00:00,6,15",
            ]
        ),
        encoding="utf-8",
    )
    manifest = DatasetManifest(name="multi", domain="energy", source_uri=str(source), time_column="time")
    session.add(manifest)
    session.commit()
    session.refresh(manifest)
    job = DatasetLoadService(tmp_path / "runtime").create_load_job(
        session,
        manifest.dataset_manifest_id,
        {"context_length": 3, "horizon": 2, "stride": 2, "target_columns": ["target_a", "target_b"]},
    )
    block = create_real_capability_block(session, "multi block", [job.output_shard_id])
    track, _ranking = create_track_with_blocks(session, "multi track", [block.capability_block_id], "mase")
    return track


def test_multitarget_run_rejects_models_with_single_target_limit(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track = _make_multi_track(session, tmp_path)
        model = Model(
            name="Timer 3.0",
            model_family="Timer",
            model_version="3.0",
            endpoint_uri="stub://timer",
            forecast_limits={"max_target_count": 1},
        )
        session.add(model)
        session.commit()
        session.refresh(model)

        with pytest.raises(ApiError) as exc:
            create_benchmarking_run(session, track.track_id, [model.model_id])

        assert exc.value.error_code == "model_target_dim_unsupported"
        assert exc.value.details["target_dim"] == 2
        assert exc.value.details["model_ids"] == [model.model_id]


def test_multitarget_run_allows_models_with_unbounded_target_limit(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track = _make_multi_track(session, tmp_path)
        model = Model(
            name="toto2.0",
            model_family="toto",
            model_version="2.0",
            endpoint_uri="stub://toto",
            forecast_limits={"max_target_count": None},
        )
        session.add(model)
        session.commit()
        session.refresh(model)

        run = create_benchmarking_run(session, track.track_id, [model.model_id])

        assert run.status == "queued"
        assert run.model_ids == [model.model_id]
