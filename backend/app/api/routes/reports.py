from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import get_db_session
from app.models.report import Report
from app.services.report_service import read_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{report_id}")
def get_report(report_id: str, session: Session = Depends(get_db_session)) -> dict:
    report = session.get(Report, report_id)
    payload = read_report(report)
    payload["report_id"] = report.report_id
    return payload
