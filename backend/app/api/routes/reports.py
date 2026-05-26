from fastapi import Depends
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.models.report import Report
from app.services.report_service import read_report

router = make_router(prefix="/reports", tags=["reports"])


@router.get("", tier="authed")
def list_reports(
    benchmarking_run_id: str | None = None,
    track_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_db_session),
) -> dict:
    base = select(Report)
    if benchmarking_run_id:
        base = base.where(Report.benchmarking_run_id == benchmarking_run_id)
    if track_id:
        base = base.where(Report.track_id == track_id)
    total = len(session.exec(base).all())
    items = session.exec(
        base.order_by(Report.created_at.desc()).offset(offset).limit(limit)
    ).all()
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/{report_id}", tier="authed")
def get_report(report_id: str, session: Session = Depends(get_db_session)) -> dict:
    report = session.get(Report, report_id)
    payload = read_report(report)
    payload["report_id"] = report.report_id
    return payload
