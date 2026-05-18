import threading
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import get_db_session
from app.core.config import get_settings
from app.models.benchmark import BenchmarkingRun
from app.services.run_executor import build_run_progress, cancel_run, create_benchmarking_run, execute_run
from app.workers.run_queue import RunQueue

router = APIRouter(prefix="/benchmarking-runs", tags=["benchmarking-runs"])


class BenchmarkingRunCreate(BaseModel):
    track_id: str
    model_ids: list[str]


@router.post("")
def create_run(payload: BenchmarkingRunCreate, request: Request, session: Session = Depends(get_db_session)) -> BenchmarkingRun:
    run = create_benchmarking_run(session, payload.track_id, payload.model_ids)
    queue: RunQueue = request.app.state.run_queue
    run.status = queue.submit(run.benchmarking_run_id)
    session.add(run)
    session.commit()
    session.refresh(run)
    if run.status == "running":
        thread = threading.Thread(
            target=_execute_in_background,
            args=(request.app.state.engine, run.benchmarking_run_id, get_settings().runtime_dir, queue),
            daemon=True,
        )
        thread.start()
    return run


def _execute_in_background(engine, run_id: str, runtime_dir: Path, queue: RunQueue) -> None:
    with Session(engine) as session:
        execute_run(session, run_id, runtime_dir)
    queue.complete(run_id)


@router.get("/{benchmarking_run_id}/progress")
def get_run_progress(benchmarking_run_id: str, session: Session = Depends(get_db_session)) -> dict:
    return build_run_progress(session, benchmarking_run_id)


@router.post("/{benchmarking_run_id}/cancel")
def cancel_benchmarking_run(benchmarking_run_id: str, session: Session = Depends(get_db_session)) -> BenchmarkingRun:
    return cancel_run(session, benchmarking_run_id)
