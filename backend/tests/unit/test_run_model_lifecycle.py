from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import RunEvent, Task, Unit
from app.services.run_executor import create_benchmarking_run, execute_run
from app.services.stub_timer_adapter import StubTimerAdapter
from tests.run_helpers import create_loaded_track_with_models


class _LifecycleAdapter:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []
        self._stub = StubTimerAdapter()

    def unload_all_models(self, timeout_seconds):  # noqa: ANN001, ANN201
        del timeout_seconds
        self.calls.append(("unload_all", None))

    def ensure_model_loaded(self, model, timeout_seconds):  # noqa: ANN001, ANN201
        del timeout_seconds
        self.calls.append(("load", model["remote_model_id"]))

    def unload_model(self, model, timeout_seconds):  # noqa: ANN001, ANN201
        del timeout_seconds
        self.calls.append(("unload", model["remote_model_id"]))

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        self.calls.append(("forecast", model["remote_model_id"]))
        return self._stub.forecast(sample, model, timeout_seconds)


class _LoadFailingLifecycleAdapter(_LifecycleAdapter):
    def ensure_model_loaded(self, model, timeout_seconds):  # noqa: ANN001, ANN201
        del timeout_seconds
        self.calls.append(("load", model["remote_model_id"]))
        raise RuntimeError("load boom")

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        del sample, model, timeout_seconds
        raise AssertionError("forecast should not run when load fails")


def test_run_unloads_all_then_loads_and_unloads_each_model(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=2)
        run = create_benchmarking_run(session, track.track_id, [model.model_id for model in models])
        adapter = _LifecycleAdapter()

        monkeypatch.setattr("app.services.run_executor.get_model_adapter", lambda settings: adapter)

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert run.status == "succeeded"
        lifecycle_calls = [call for call in adapter.calls if call[0] != "forecast"]
        assert lifecycle_calls == [
            ("unload_all", None),
            ("load", "Timer-0"),
            ("unload", "Timer-0"),
            ("load", "Timer-1"),
            ("unload", "Timer-1"),
        ]
        for remote_model_id in ("Timer-0", "Timer-1"):
            load_index = adapter.calls.index(("load", remote_model_id))
            unload_index = adapter.calls.index(("unload", remote_model_id))
            forecast_indices = [
                index for index, call in enumerate(adapter.calls) if call == ("forecast", remote_model_id)
            ]
            assert forecast_indices
            assert all(load_index < index < unload_index for index in forecast_indices)


def test_run_unloads_model_after_load_failure(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        adapter = _LoadFailingLifecycleAdapter()

        monkeypatch.setattr("app.services.run_executor.get_model_adapter", lambda settings: adapter)

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        unit = session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).one()
        task = session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).one()
        event_types = [
            event.event_type
            for event in session.exec(select(RunEvent).where(RunEvent.benchmarking_run_id == run.benchmarking_run_id)).all()
        ]
        assert run.status == "failed"
        assert unit.status == "failed"
        assert task.error_code == "model_load_error"
        assert adapter.calls == [
            ("unload_all", None),
            ("load", "Timer-0"),
            ("unload", "Timer-0"),
        ]
        assert "model_load_failed" in event_types
        assert "model_unloaded" in event_types
