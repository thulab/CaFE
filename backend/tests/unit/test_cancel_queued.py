"""#19 — cancelling a QUEUED run must not wedge the queue.

``cancel_run`` only flips the run's status to ``cancel_requested``. If the
cancelled run is still *queued* (not the one running), the queue pointer
would later advance to it via ``complete`` but nobody starts a thread for
it, so the queue stalls forever and no later run ever executes.

The fix removes a still-queued cancelled run from the queue (via
``RunQueue.remove``) so the drain logic skips it. Cancelling the
currently-running run keeps its existing cooperative-stop behavior.
"""
import time

from sqlmodel import Session, create_engine

from app.api.routes.benchmarking_runs import _execute_in_background
from app.db.init_db import init_db
from app.models.benchmark import BenchmarkingRun
from app.services.run_executor import cancel_run, create_benchmarking_run
from app.workers.run_queue import RunQueue
from tests.run_helpers import create_loaded_track_with_models

TERMINAL = {"succeeded", "partial_succeeded", "failed", "cancelled"}


def _wait_terminal(engine, run_id: str, timeout: float = 60.0) -> str:
    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        with Session(engine) as session:
            status = session.get(BenchmarkingRun, run_id).status
        if status in TERMINAL:
            return status
        time.sleep(0.1)
    return status


def test_remove_drops_queued_run_from_the_queue():
    queue = RunQueue()
    assert queue.submit("run-a") == "running"
    assert queue.submit("run-b") == "queued"

    queue.remove("run-b")

    # Completing the running run must NOT advance to the removed run.
    assert queue.complete("run-a") is None


def test_cancelling_queued_run_does_not_wedge_queue(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        track_id = track.track_id
        model_id = models[0].model_id
        run_a = create_benchmarking_run(session, track_id, [model_id])
        run_b = create_benchmarking_run(session, track_id, [model_id])
        run_a_id = run_a.benchmarking_run_id
        run_b_id = run_b.benchmarking_run_id

    queue = RunQueue()
    assert queue.submit(run_a_id) == "running"
    assert queue.submit(run_b_id) == "queued"

    # Cancel the still-queued run B: drop it from the queue so drain skips it.
    with Session(engine) as session:
        cancel_run(session, run_b_id)
    queue.remove(run_b_id)

    # Drain run A. B must NOT execute (stays cancel_requested).
    _execute_in_background(engine, run_a_id, tmp_path / "runtime", queue)
    assert _wait_terminal(engine, run_a_id) in TERMINAL

    with Session(engine) as session:
        assert session.get(BenchmarkingRun, run_b_id).status == "cancel_requested"

    # Queue must be drained, not stuck on the cancelled run.
    assert queue.running_run_id is None

    # A subsequently submitted run C still runs to a terminal state.
    with Session(engine) as session:
        run_c = create_benchmarking_run(session, track_id, [model_id])
        run_c_id = run_c.benchmarking_run_id

    assert queue.submit(run_c_id) == "running"
    _execute_in_background(engine, run_c_id, tmp_path / "runtime", queue)
    assert _wait_terminal(engine, run_c_id) in TERMINAL, "queue was wedged; later run never executed"
    assert queue.running_run_id is None
