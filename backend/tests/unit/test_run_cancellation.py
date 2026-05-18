from sqlmodel import Session, create_engine

from app.db.init_db import init_db
from app.services.run_executor import cancel_run, create_benchmarking_run, execute_run
from tests.run_helpers import create_loaded_track_with_models


def test_cancelled_run_stops_cooperatively(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        cancel_run(session, run.benchmarking_run_id)

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert run.status == "cancelled"
        assert run.cancel_requested is True
