from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import Task, Unit
from app.services.run_executor import create_benchmarking_run
from tests.run_helpers import create_loaded_track_with_models


def test_run_creation_expands_units_and_tasks(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=2)

        run = create_benchmarking_run(session, track.track_id, [model.model_id for model in models])

        assert run.status == "queued"
        assert run.model_count == 2
        assert run.task_count == 2
        assert run.sample_count == 8
        assert len(session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).all()) == 2
        assert len(session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).all()) == 2
