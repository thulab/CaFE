import threading
from pathlib import Path

from fastapi import Depends, Request
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.config import get_settings
from app.models.benchmark import BenchmarkingRun
from app.services.run_executor import build_run_progress, cancel_run, create_benchmarking_run, execute_run
from app.workers.run_queue import RunQueue

router = make_router(prefix="/benchmarking-runs", tags=["benchmarking-runs"])


class BenchmarkingRunCreate(BaseModel):
    track_id: str
    model_ids: list[str]


@router.post("", tier="perm", perm="run.execute")
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
    try:
        with Session(engine) as session:
            execute_run(session, run_id, runtime_dir)
    finally:
        # complete() 必须执行，否则队列永久卡死；它返回下一个待执行 run，
        # 即便本次 run 异常退出也要把下一个排队 run 起起来。
        next_run_id = queue.complete(run_id)
        if next_run_id is not None:
            threading.Thread(
                target=_execute_in_background,
                args=(engine, next_run_id, runtime_dir, queue),
                daemon=True,
            ).start()


@router.get("/{benchmarking_run_id}/progress", tier="authed")
def get_run_progress(benchmarking_run_id: str, session: Session = Depends(get_db_session)) -> dict:
    return build_run_progress(session, benchmarking_run_id)


@router.post("/{benchmarking_run_id}/cancel", tier="perm", perm="run.cancel")
def cancel_benchmarking_run(benchmarking_run_id: str, request: Request, session: Session = Depends(get_db_session)) -> BenchmarkingRun:
    queue: RunQueue = request.app.state.run_queue
    # 取消一个仍在排队（非当前运行）的 run 时，必须把它从队列里摘掉，
    # 否则 complete 会把指针推进到它，但没人起线程执行 → 队列永久卡死。
    if queue.running_run_id != benchmarking_run_id:
        queue.remove(benchmarking_run_id)
    return cancel_run(session, benchmarking_run_id)
