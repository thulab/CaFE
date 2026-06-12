from sqlmodel import Session, create_engine

from app.db.init_db import init_db
from app.models.benchmark import FailedSampleRerunJob
from app.services.run_executor import create_benchmarking_run, recover_interrupted_runs
from tests.run_helpers import create_loaded_track_with_models


def test_startup_recovery_marks_unfinished_runs_interrupted(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        run.status = "running"
        session.add(run)
        session.commit()

        recover_interrupted_runs(session)

        session.refresh(run)
        assert run.status == "failed"


def test_startup_recovery_marks_active_failed_sample_reruns_interrupted(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        run.status = "partial_succeeded"
        job = FailedSampleRerunJob(
            benchmarking_run_id=run.benchmarking_run_id,
            status="running",
            activity_status="forecasting",
            total_samples=4,
            processed_samples=2,
        )
        session.add(run)
        session.add(job)
        session.commit()

        recover_interrupted_runs(session)

        session.refresh(run)
        session.refresh(job)
        assert run.status == "partial_succeeded"
        assert job.status == "failed"
        assert job.activity_status == "failed"
        assert job.error_code == "interrupted_by_server_restart"
        assert job.finished_at is not None
