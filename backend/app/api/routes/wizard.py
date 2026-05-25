from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import get_db_session
from app.services.track_service import create_real_capability_block, create_track_with_blocks

router = APIRouter(prefix="/wizard", tags=["wizard"])


class RealDatasetTrackCreate(BaseModel):
    name: str
    shard_ids: list[str]
    primary_metric_id: str = "mase"


@router.post("/real-dataset-track")
def create_real_dataset_track(payload: RealDatasetTrackCreate, session: Session = Depends(get_db_session)) -> dict:
    block = create_real_capability_block(session, f"{payload.name} real data", payload.shard_ids)
    track, ranking = create_track_with_blocks(session, payload.name, [block.capability_block_id], payload.primary_metric_id)
    return {
        "track_id": track.track_id,
        "capability_block_id": block.capability_block_id,
        "ranking_list_id": ranking.ranking_list_id,
    }
