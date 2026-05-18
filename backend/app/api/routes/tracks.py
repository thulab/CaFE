from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import get_db_session
from app.services.track_service import create_track_with_blocks

router = APIRouter(prefix="/tracks", tags=["tracks"])


class TrackCreate(BaseModel):
    name: str
    capability_block_ids: list[str]
    primary_metric_id: str = "mse"


@router.post("")
def create_track(payload: TrackCreate, session: Session = Depends(get_db_session)) -> dict:
    track, ranking = create_track_with_blocks(session, payload.name, payload.capability_block_ids, payload.primary_metric_id)
    return {"track_id": track.track_id, "ranking_list_id": ranking.ranking_list_id}
