"""#18 — run terminal status must account for partial_succeeded units.

``execute_run`` previously counted only ``succeeded`` / ``failed`` units
and ignored ``partial_succeeded``:
- ``[1 succeeded + 1 partial_succeeded]`` was mislabeled ``succeeded``;
- ``[all partial_succeeded]`` was mislabeled ``failed``.

Correct rule:
- any success AND any non-success  -> ``partial_succeeded``
- all non-success                  -> ``failed``
- all success                      -> ``succeeded``
"""
from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import Unit
from app.services.run_executor import create_benchmarking_run, execute_run
from app.services.stub_timer_adapter import StubTimerAdapter
from tests.run_helpers import create_loaded_track_with_models


class _PerModelAdapter:
    """Succeeds via the real stub, except for model_ids that must fail.

    Failing every sample of a model makes its only shard return no metrics,
    so its task -> unit collapses to ``partial_succeeded``.
    """

    def __init__(self, failing_model_ids: set[str]):
        self._failing = failing_model_ids
        self._stub = StubTimerAdapter()

    def forecast(self, sample, model, timeout_seconds):  # noqa: ANN001, ANN201
        if model["model_id"] in self._failing:
            raise RuntimeError("adapter boom for this model")
        return self._stub.forecast(sample, model, timeout_seconds)


def _unit_statuses(session: Session, run_id: str) -> list[str]:
    return sorted(
        unit.status
        for unit in session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    )


def test_one_succeeded_one_partial_is_partial_succeeded(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=2)
        run = create_benchmarking_run(session, track.track_id, [m.model_id for m in models])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _PerModelAdapter({models[1].model_id}),
        )

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert _unit_statuses(session, run.benchmarking_run_id) == ["partial_succeeded", "succeeded"]
        assert run.status == "partial_succeeded"


def test_all_partial_units_is_partial_succeeded(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=2)
        run = create_benchmarking_run(session, track.track_id, [m.model_id for m in models])

        monkeypatch.setattr(
            "app.services.run_executor.get_model_adapter",
            lambda settings: _PerModelAdapter({m.model_id for m in models}),
        )

        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        session.refresh(run)
        assert _unit_statuses(session, run.benchmarking_run_id) == ["partial_succeeded", "partial_succeeded"]
        assert run.status == "partial_succeeded"
