from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import Unit
from app.services.run_executor import create_benchmarking_run, execute_run
from tests.run_helpers import create_loaded_track_with_models


def test_run_with_one_failed_model_is_partial_success(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=2)
        models[1].endpoint_uri = "stub://fail"
        session.add(models[1])
        session.commit()
        run = create_benchmarking_run(session, track.track_id, [model.model_id for model in models])

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        statuses = sorted(unit.status for unit in session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).all())
        assert run.status == "partial_succeeded"
        assert statuses == ["failed", "succeeded"]
