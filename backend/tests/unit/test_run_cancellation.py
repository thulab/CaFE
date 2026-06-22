import pytest
from sqlmodel import Session, create_engine, select

from app.core.config import get_settings
from app.core.errors import ApiError
from app.db.init_db import init_db
from app.models.benchmark import RunEvent
from app.models.report import Report
from app.models.ranking import RankingEntry
from app.services.run_executor import cancel_run, create_benchmarking_run, execute_run
from app.services.stub_timer_adapter import StubTimerAdapter
from tests.run_helpers import create_loaded_track_with_models


class _CancelsAfterFirstForecastAdapter:
    def __init__(self, engine, run_id: str):
        self.engine = engine
        self.run_id = run_id
        self.calls = 0
        self.stub = StubTimerAdapter()

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        self.calls += 1
        forecast = self.stub.forecast(sample, model, timeout_seconds)
        if self.calls == 1:
            with Session(self.engine) as session:
                cancel_run(session, self.run_id)
        return forecast


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


def test_running_cancel_stops_before_remaining_samples_and_skips_report_and_ranking(tmp_path, monkeypatch):
    monkeypatch.setenv("TSBENCHMARK_RUN_SAMPLE_PARALLELISM", "1")
    get_settings.cache_clear()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        adapter = _CancelsAfterFirstForecastAdapter(engine, run.benchmarking_run_id)
        monkeypatch.setattr("app.services.run_executor.get_model_adapter", lambda settings: adapter)

        try:
            execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

            session.refresh(run)
            assert run.status == "cancelled"
            assert run.report_id is None
            assert adapter.calls < run.sample_count
            assert not session.exec(select(Report).where(Report.benchmarking_run_id == run.benchmarking_run_id)).all()
            assert not session.exec(select(RankingEntry).where(RankingEntry.benchmarking_run_id == run.benchmarking_run_id)).all()
            assert not (tmp_path / "runtime" / "reports" / f"{run.benchmarking_run_id}.json").exists()
        finally:
            get_settings.cache_clear()


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
