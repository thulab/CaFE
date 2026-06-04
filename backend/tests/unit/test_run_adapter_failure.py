"""#16 — adapter errors must not crash the run nor wedge the queue.

When ``adapter.forecast`` raises for a sample, the executor must:
- record a *failed* forecast row for that sample (``status == "failed"``
  with an ``error_code``) instead of letting the exception escape;
- still drive the run to a terminal status (never stuck in ``running``);
- and the background queue must still be drained so a subsequently
  submitted run can run to a terminal status too.
"""
from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import ForecastArtifact, Task, Unit
from app.services.forecast_store import ForecastStore
from app.services.run_executor import create_benchmarking_run, execute_run
from app.services.stub_timer_adapter import StubTimerAdapter
from app.workers.run_queue import RunQueue
from tests.run_helpers import create_loaded_track_with_models

TERMINAL = {"succeeded", "partial_succeeded", "failed", "cancelled"}


class _AlwaysRaisingAdapter:
    """Adapter whose ``forecast`` raises for every sample."""

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        raise RuntimeError("adapter boom")


class _FirstBadShapeAdapter:
    def __init__(self):
        self._calls = 0
        self._stub = StubTimerAdapter()

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        self._calls += 1
        if self._calls == 1:
            return []
        return self._stub.forecast(sample, model, timeout_seconds)


class _LoadFailingAdapter:
    def ensure_model_loaded(self, model, timeout_seconds):  # noqa: ANN001, ANN201
        raise RuntimeError("load boom")

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        raise AssertionError("forecast should not be called when model load fails")


def _all_forecast_rows(forecasts_dir, artifacts: list[ForecastArtifact]) -> list[dict]:
    store = ForecastStore(forecasts_dir)
    rows: list[dict] = []
    for artifact in artifacts:
        rows.extend(store.read_forecasts(artifact.storage_uri))
    return rows


def test_adapter_failure_does_not_crash_run_and_marks_sample_failed(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _AlwaysRaisingAdapter(),
        )

        # Must not raise even though every adapter.forecast call raises.
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert run.status in TERMINAL
        assert run.status != "running"

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)
        ).all()
        rows = _all_forecast_rows(tmp_path / "runtime" / "forecasts", artifacts)
        assert rows, "expected forecast rows to be written even on adapter failure"
        assert all(row["status"] == "failed" for row in rows)
        assert all(row["error_code"] for row in rows)


def test_sample_metric_error_marks_only_that_sample_failed_and_continues(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _FirstBadShapeAdapter(),
        )

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)
        ).all()
        rows = _all_forecast_rows(tmp_path / "runtime" / "forecasts", artifacts)
        assert [row["status"] for row in rows].count("failed") == 1
        assert [row["status"] for row in rows].count("succeeded") > 0
        failed = next(row for row in rows if row["status"] == "failed")
        assert failed["error_code"] == "metric_error"
        assert "same flattened length" in failed["error_message"]


def test_model_load_failure_marks_unit_failed_without_forecast(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _LoadFailingAdapter(),
        )

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert run.status == "failed"
        unit = session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).one()
        task = session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).one()
        assert unit.status == "failed"
        assert task.status == "failed"
        assert task.error_code == "model_load_error"
        assert "load boom" in task.error_message
        assert not session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)).all()


def test_queue_drains_after_adapter_failure_so_next_run_completes(tmp_path, monkeypatch):
    """A failing run must not wedge the queue: a later run still reaches terminal."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        run_id = run.benchmarking_run_id

    monkeypatch.setattr(
        "app.services.run_executor.get_model_adapter",
        lambda settings: _AlwaysRaisingAdapter(),
    )

    queue = RunQueue()

    # Exercise the real background entrypoint under test.
    from app.api.routes.benchmarking_runs import _execute_in_background

    assert queue.submit(run_id) == "running"
    _execute_in_background(engine, run_id, tmp_path / "runtime", queue)

    # complete() must have been called via try/finally despite the failing run.
    assert queue.running_run_id is None

    # And the next submission can now run, proving the queue isn't wedged.
    assert queue.submit("next-run") == "running"
