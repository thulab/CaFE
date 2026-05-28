from fastapi import Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.errors import ApiError
from app.models.benchmark import Track
from app.services.track_service import create_track_with_blocks, track_summary

router = make_router(prefix="/tracks", tags=["tracks"])


class TrackCreate(BaseModel):
    name: str
    capability_block_ids: list[str]
    primary_metric_id: str = "mase"


@router.get("", tier="public")
def list_tracks(session: Session = Depends(get_db_session)) -> dict:
    tracks = session.exec(select(Track).order_by(Track.created_at)).all()
    return {"items": [track_summary(session, t) for t in tracks]}


@router.get("/{track_id}", tier="public")
def get_track(track_id: str, session: Session = Depends(get_db_session)) -> dict:
    track = session.get(Track, track_id)
    if track is None:
        raise ApiError("track_not_found", "track not found", {"track_id": track_id}, 404)
    return track_summary(session, track)


@router.post("", tier="perm", perm="track.manage")
def create_track(payload: TrackCreate, session: Session = Depends(get_db_session)) -> dict:
    track, ranking = create_track_with_blocks(session, payload.name, payload.capability_block_ids, payload.primary_metric_id)
    return {"track_id": track.track_id, "ranking_list_id": ranking.ranking_list_id}
