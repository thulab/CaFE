from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import get_db_session
from app.services.ranking_service import query_ranking

router = APIRouter(prefix="/tracks", tags=["ranking-lists"])


@router.get("/{track_id}/ranking")
def get_track_ranking(track_id: str, metric: str = "mse", policy: str = "latest_valid_result", session: Session = Depends(get_db_session)) -> dict:
    entries = query_ranking(session, track_id, metric, policy)
    return {"track_id": track_id, "metric": metric, "policy": policy, "items": entries}
