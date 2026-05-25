from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import get_db_session
from app.models.benchmark import Track
from app.services.ranking_service import query_ranking

router = APIRouter(prefix="/tracks", tags=["ranking-lists"])


@router.get("/{track_id}/ranking")
def get_track_ranking(track_id: str, metric: str | None = None, policy: str | None = None, session: Session = Depends(get_db_session)) -> dict:
    track = session.get(Track, track_id)
    resolved_metric = metric or (track.primary_metric_id if track else "mase")
    resolved_policy = policy or (track.default_ranking_policy if track else "latest_valid_result")
    entries = query_ranking(session, track_id, resolved_metric, resolved_policy)
    return {"track_id": track_id, "metric": resolved_metric, "policy": resolved_policy, "items": entries}
