from fastapi import Depends
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.models.dataset import Shard
from app.models.sample import SampleIndex

router = make_router(prefix="/shards", tags=["shards"])


@router.get("/{shard_id}", tier="authed")
def get_shard(shard_id: str, session: Session = Depends(get_db_session)) -> Shard:
    return session.get(Shard, shard_id)


@router.get("/{shard_id}/samples", tier="authed")
def list_shard_samples(shard_id: str, limit: int = 20, offset: int = 0, session: Session = Depends(get_db_session)) -> dict:
    items = session.exec(
        select(SampleIndex)
        .where(SampleIndex.shard_id == shard_id)
        .order_by(SampleIndex.sample_index)
        .offset(offset)
        .limit(limit)
    ).all()
    total = len(session.exec(select(SampleIndex).where(SampleIndex.shard_id == shard_id)).all())
    return {"items": items, "total": total, "limit": limit, "offset": offset}
