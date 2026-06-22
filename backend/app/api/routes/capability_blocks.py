from fastapi import Depends
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.models.benchmark import CapabilityBlock
from app.services.track_service import create_real_capability_block

router = make_router(prefix="/capability-blocks", tags=["capability-blocks"])


class CapabilityBlockCreate(BaseModel):
    name: str
    shard_ids: list[str]


@router.post("", tier="perm", perm="track.manage")
def create_capability_block(payload: CapabilityBlockCreate, session: Session = Depends(get_db_session)) -> CapabilityBlock:
    return create_real_capability_block(session, payload.name, payload.shard_ids)
