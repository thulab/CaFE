from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import get_db_session
from app.models.sample import SampleIndex
from app.services.sample_forecast_service import build_sample_forecast
from app.services.sample_store import SampleStore

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("/{sample_id}/preview")
def get_sample_preview(sample_id: str, session: Session = Depends(get_db_session)) -> dict:
    sample = session.get(SampleIndex, sample_id)
    record = SampleStore().read_by_ref(sample.materialized_sample_uri, sample.storage_ref)
    return record


@router.get("/{sample_id}/forecast")
def get_sample_forecast(sample_id: str, run_id: str, session: Session = Depends(get_db_session)) -> dict:
    return build_sample_forecast(session, sample_id, run_id)
