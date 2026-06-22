"""#17 — a queued run must be driven to execution automatically.

The execution thread is only started by ``create_run`` when the queue
admits the run as ``running``. A second concurrent run is admitted as
``queued`` and only ever advanced by ``queue.complete`` moving the
pointer — nothing ever *starts* it. So the queue's background entrypoint
must, after completing a run, drive the next queued run to execution.
"""
import time

from sqlmodel import Session, create_engine

from app.api.routes.benchmarking_runs import _execute_in_background
from app.db.init_db import init_db
from app.models.benchmark import BenchmarkingRun
from app.services.run_executor import create_benchmarking_run
from app.workers.run_queue import RunQueue
from tests.run_helpers import create_loaded_track_with_models

TERMINAL = {"succeeded", "partial_succeeded", "failed", "cancelled"}


def _wait_terminal(engine, run_id: str, timeout: float = 60.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with Session(engine) as session:
            status = session.get(BenchmarkingRun, run_id).status
        if status in TERMINAL:
            return status
        time.sleep(0.1)
    return status


def test_second_queued_run_executes_to_terminal(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run_one = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        run_two = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        run_one_id = run_one.benchmarking_run_id
        run_two_id = run_two.benchmarking_run_id

    queue = RunQueue()
    assert queue.submit(run_one_id) == "running"
    assert queue.submit(run_two_id) == "queued"

    # Run the first run's background entrypoint synchronously; completing it
    # must drive the second (queued) run to execution on its own thread.
    _execute_in_background(engine, run_one_id, tmp_path / "runtime", queue)

    assert _wait_terminal(engine, run_one_id) in TERMINAL
    assert _wait_terminal(engine, run_two_id) in TERMINAL, "second queued run never executed"

    # Queue must be fully drained once both runs finish.
    assert _wait_for_drained(queue) is None


def _wait_for_drained(queue: RunQueue, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if queue.running_run_id is None:
            return None
        time.sleep(0.1)
    return queue.running_run_id
