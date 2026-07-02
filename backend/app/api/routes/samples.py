from fastapi import Depends
from sqlmodel import Session

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.errors import ApiError
from app.models.sample import SampleIndex
from app.services.sample_forecast_service import build_sample_forecast, build_track_sample_forecast
from app.services.sample_store import SampleStore

router = make_router(prefix="/samples", tags=["samples"])


@router.get("/{sample_id}/preview", tier="authed")
def get_sample_preview(sample_id: str, session: Session = Depends(get_db_session)) -> dict:
    sample = session.get(SampleIndex, sample_id)
    record = SampleStore().read_by_ref(session, sample.storage_ref)
    return record


@router.get("/{sample_id}/forecast", tier="authed")
def get_sample_forecast(
    sample_id: str,
    run_id: str | None = None,
    track_id: str | None = None,
    session: Session = Depends(get_db_session),
) -> dict:
    if track_id:
        return build_track_sample_forecast(session, sample_id, track_id)
    if not run_id:
        raise ApiError("forecast_context_required", "run_id or track_id is required", {"sample_id": sample_id}, 422)
    return build_sample_forecast(session, sample_id, run_id)
