from fastapi import Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.errors import ApiError
from app.models.benchmark import Track
from app.services.resource_lifecycle import (
    RESOURCE_TRACK,
    archive_resource,
    archived_at,
    deletion_impact,
    purge_track,
    restore_resource,
    visible_rows,
)
from app.services.report_service import SampleLinkSort, read_track_results
from app.services.track_service import create_track_with_blocks, track_summary

router = make_router(prefix="/tracks", tags=["tracks"])


class TrackCreate(BaseModel):
    name: str
    capability_block_ids: list[str]
    primary_metric_id: str = "mase"


@router.get("", tier="public")
def list_tracks(include_archived: bool = False, session: Session = Depends(get_db_session)) -> dict:
    tracks = session.exec(select(Track).order_by(Track.created_at)).all()
    tracks = visible_rows(session, RESOURCE_TRACK, tracks, "track_id", include_archived)
    return {"items": [_track_summary_with_archive(session, t) for t in tracks], "total": len(tracks)}


@router.get("/{track_id}", tier="public")
def get_track(track_id: str, session: Session = Depends(get_db_session)) -> dict:
    track = session.get(Track, track_id)
    if track is None:
        raise ApiError("track_not_found", "track not found", {"track_id": track_id}, 404)
    return _track_summary_with_archive(session, track)


@router.get("/{track_id}/results", tier="authed")
def get_track_results(
    track_id: str,
    sample_link_limit: int | None = None,
    sample_link_offset: int = 0,
    sample_link_capability_block_id: str | None = None,
    sample_link_metric: str | None = None,
    sample_link_model_id: str | None = None,
    sample_link_sort: SampleLinkSort = "sample_index",
    session: Session = Depends(get_db_session),
) -> dict:
    return read_track_results(
        session,
        track_id,
        sample_link_limit=sample_link_limit,
        sample_link_offset=sample_link_offset,
        sample_link_capability_block_id=sample_link_capability_block_id,
        sample_link_metric=sample_link_metric,
        sample_link_model_id=sample_link_model_id,
        sample_link_sort=sample_link_sort,
    )


@router.post("", tier="perm", perm="track.manage")
def create_track(payload: TrackCreate, session: Session = Depends(get_db_session)) -> dict:
    track, ranking = create_track_with_blocks(session, payload.name, payload.capability_block_ids, payload.primary_metric_id)
    return {"track_id": track.track_id, "ranking_list_id": ranking.ranking_list_id}


@router.get("/{track_id}/deletion-impact", tier="perm", perm="track.delete")
def get_track_deletion_impact(track_id: str, session: Session = Depends(get_db_session)) -> dict:
    return deletion_impact(session, RESOURCE_TRACK, track_id)


@router.post("/{track_id}/archive", tier="perm", perm="track.delete")
def archive_track(track_id: str, session: Session = Depends(get_db_session)) -> dict:
    return archive_resource(session, RESOURCE_TRACK, track_id)


@router.post("/{track_id}/restore", tier="perm", perm="track.delete")
def restore_track(track_id: str, session: Session = Depends(get_db_session)) -> dict:
    return restore_resource(session, RESOURCE_TRACK, track_id)


@router.delete("/{track_id}", tier="perm", perm="admin.purge")
def delete_track(track_id: str, cascade: bool = False, session: Session = Depends(get_db_session)) -> dict:
    return purge_track(session, track_id, cascade)


def _track_summary_with_archive(session: Session, track: Track) -> dict:
    data = track_summary(session, track)
    data["archived_at"] = archived_at(session, RESOURCE_TRACK, track.track_id)
    return data
