import pytest
from sqlmodel import Session, create_engine, select

from app.core.errors import ApiError
from app.db.init_db import init_db
from app.models.benchmark import RunEvent
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


def test_cancel_rejects_run_in_terminal_state(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")
        session.refresh(run)
        assert run.status in {"succeeded", "partial_succeeded", "failed"}
        run_status_before = run.status

        with pytest.raises(ApiError) as exc:
            cancel_run(session, run.benchmarking_run_id)
        assert exc.value.status_code == 409
        assert exc.value.error_code == "run_in_terminal_state"

        session.refresh(run)
        assert run.status == run_status_before
        assert run.cancel_requested is False


def test_cancel_is_idempotent_when_already_requested(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        cancel_run(session, run.benchmarking_run_id)
        events_after_first = session.exec(
            select(RunEvent).where(RunEvent.benchmarking_run_id == run.benchmarking_run_id, RunEvent.event_type == "cancel_requested")
        ).all()

        cancel_run(session, run.benchmarking_run_id)
        cancel_run(session, run.benchmarking_run_id)

        events_after_repeat = session.exec(
            select(RunEvent).where(RunEvent.benchmarking_run_id == run.benchmarking_run_id, RunEvent.event_type == "cancel_requested")
        ).all()
        assert len(events_after_repeat) == len(events_after_first) == 1
