from fastapi import Depends
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.errors import ApiError
from app.models.report import Report
from app.services.report_service import SampleLinkSort, read_report

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
def get_report(
    report_id: str,
    sample_link_limit: int | None = None,
    sample_link_offset: int = 0,
    sample_link_capability_block_id: str | None = None,
    sample_link_metric: str | None = None,
    sample_link_model_id: str | None = None,
    sample_link_sort: SampleLinkSort = "sample_index",
    session: Session = Depends(get_db_session),
) -> dict:
    report = session.get(Report, report_id)
    if report is None:
        raise ApiError("resource_not_found", "resource not found", {"resource_type": "report", "resource_id": report_id}, 404)
    payload = read_report(
        report,
        session=session,
        sample_link_limit=sample_link_limit,
        sample_link_offset=sample_link_offset,
        sample_link_capability_block_id=sample_link_capability_block_id,
        sample_link_metric=sample_link_metric,
        sample_link_model_id=sample_link_model_id,
        sample_link_sort=sample_link_sort,
    )
    payload["report_id"] = report.report_id
    return payload
