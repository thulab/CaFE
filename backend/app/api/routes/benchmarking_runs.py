import threading
from pathlib import Path

from fastapi import Depends, Request
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.config import get_settings
from app.core.errors import ApiError
from app.models.benchmark import BenchmarkingRun
from app.services.resource_lifecycle import (
    RESOURCE_RUN,
    archive_resource,
    deletion_impact,
    purge_run,
    restore_resource,
    row_with_archive,
    visible_rows,
)
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


@router.get("", tier="authed")
def list_runs(
    track_id: str | None = None,
    include_archived: bool = False,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_db_session),
) -> dict:
    base = select(BenchmarkingRun)
    if track_id:
        base = base.where(BenchmarkingRun.track_id == track_id)
    rows = session.exec(base.order_by(BenchmarkingRun.created_at.desc())).all()
    rows = visible_rows(session, RESOURCE_RUN, rows, "benchmarking_run_id", include_archived)
    total = len(rows)
    items = rows[offset : offset + limit]
    return {
        "items": [row_with_archive(session, RESOURCE_RUN, item, "benchmarking_run_id") for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{benchmarking_run_id}/deletion-impact", tier="perm", perm="run.delete")
def get_run_deletion_impact(benchmarking_run_id: str, session: Session = Depends(get_db_session)) -> dict:
    return deletion_impact(session, RESOURCE_RUN, benchmarking_run_id)


@router.post("/{benchmarking_run_id}/archive", tier="perm", perm="run.delete")
def archive_run(benchmarking_run_id: str, session: Session = Depends(get_db_session)) -> dict:
    return archive_resource(session, RESOURCE_RUN, benchmarking_run_id)


@router.post("/{benchmarking_run_id}/restore", tier="perm", perm="run.delete")
def restore_run(benchmarking_run_id: str, session: Session = Depends(get_db_session)) -> dict:
    return restore_resource(session, RESOURCE_RUN, benchmarking_run_id)


@router.delete("/{benchmarking_run_id}", tier="perm", perm="admin.purge")
def delete_run(benchmarking_run_id: str, session: Session = Depends(get_db_session)) -> dict:
    return purge_run(session, benchmarking_run_id)


@router.get("/{benchmarking_run_id}/progress", tier="authed")
def get_run_progress(benchmarking_run_id: str, session: Session = Depends(get_db_session)) -> dict:
    if session.get(BenchmarkingRun, benchmarking_run_id) is None:
        raise ApiError("resource_not_found", "resource not found", {"resource_type": RESOURCE_RUN, "resource_id": benchmarking_run_id}, 404)
    return build_run_progress(session, benchmarking_run_id)


@router.post("/{benchmarking_run_id}/cancel", tier="perm", perm="run.cancel")
def cancel_benchmarking_run(benchmarking_run_id: str, request: Request, session: Session = Depends(get_db_session)) -> BenchmarkingRun:
    queue: RunQueue = request.app.state.run_queue
    # 取消一个仍在排队（非当前运行）的 run 时，必须把它从队列里摘掉，
    # 否则 complete 会把指针推进到它，但没人起线程执行 → 队列永久卡死。
    if queue.running_run_id != benchmarking_run_id:
        queue.remove(benchmarking_run_id)
    return cancel_run(session, benchmarking_run_id)
