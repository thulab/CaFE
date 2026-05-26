from fastapi import Depends
from sqlmodel import Session, select
from pydantic import BaseModel

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.models.benchmark import CapabilityBlock
from app.models.dataset import Shard
from app.services.track_service import create_real_capability_block, create_track_with_blocks

router = make_router(prefix="/wizard", tags=["wizard"])


class RealDatasetTrackCreate(BaseModel):
    name: str
    shard_ids: list[str]
    primary_metric_id: str = "mase"


def _delete_capability_block(session: Session, capability_block_id: str) -> None:
    """Compensate a committed step-1 block: free its shards then delete it."""
    for shard in session.exec(select(Shard).where(Shard.capability_block_id == capability_block_id)).all():
        shard.capability_block_id = None
        session.add(shard)
    block = session.get(CapabilityBlock, capability_block_id)
    if block is not None:
        session.delete(block)
    session.commit()


@router.post("/real-dataset-track", tier="perm", perm="run.execute")
def create_real_dataset_track(payload: RealDatasetTrackCreate, session: Session = Depends(get_db_session)) -> dict:
    block = create_real_capability_block(session, f"{payload.name} real data", payload.shard_ids)
    # Step 1 commits on its own (other callers rely on that), so if step 2 fails we
    # must compensate: delete the just-created block and free its shards for reuse.
    try:
        track, ranking = create_track_with_blocks(
            session, payload.name, [block.capability_block_id], payload.primary_metric_id
        )
    except Exception:
        session.rollback()
        _delete_capability_block(session, block.capability_block_id)
        raise
    return {
        "track_id": track.track_id,
        "capability_block_id": block.capability_block_id,
        "ranking_list_id": ranking.ranking_list_id,
    }
