from sqlmodel import Session, select

from app.core.errors import ApiError
from app.models.benchmark import CapabilityBlock, Track
from app.models.dataset import Shard
from app.models.model_registry import Model
from app.models.ranking import RankingList


MVP_STUB_MODELS = [
    ("Timer 3.5", "Timer", "3.5"),
    ("Timer 3.0", "Timer", "3.0"),
    ("Chronos 2", "Chronos", "2"),
    ("toto", "toto", "mvp"),
    ("TimesFM 2.5", "TimesFM", "2.5"),
]


def create_real_capability_block(session: Session, name: str, shard_ids: list[str]) -> CapabilityBlock:
    if not shard_ids:
        raise ApiError("capability_block_requires_shard", "real capability block requires at least one shard")
    shards = [session.get(Shard, shard_id) for shard_id in shard_ids]
    missing = [shard_id for shard_id, shard in zip(shard_ids, shards, strict=True) if shard is None]
    if missing:
        raise ApiError("shard_not_found", "shard not found", {"shard_ids": missing}, 404)
    assigned = [shard.shard_id for shard in shards if shard.capability_block_id]
    if assigned:
        raise ApiError("shard_already_assigned", "shard already belongs to a capability block", {"shard_ids": assigned})

    block = CapabilityBlock(
        name=name,
        block_type="real",
        capability_type="real_data",
        shard_count=len(shards),
        sample_count=sum(shard.sample_count for shard in shards),
        target_dim=max((shard.target_dim for shard in shards), default=1),
    )
    session.add(block)
    session.commit()
    session.refresh(block)

    for shard in shards:
        shard.capability_block_id = block.capability_block_id
        session.add(shard)
    session.commit()
    session.refresh(block)
    return block


def create_track_with_blocks(
    session: Session,
    name: str,
    capability_block_ids: list[str],
    primary_metric_id: str = "mse",
) -> tuple[Track, RankingList]:
    if not capability_block_ids:
        raise ApiError("track_requires_block", "track requires at least one capability block")
    blocks = [session.get(CapabilityBlock, block_id) for block_id in capability_block_ids]
    missing = [block_id for block_id, block in zip(capability_block_ids, blocks, strict=True) if block is None]
    if missing:
        raise ApiError("capability_block_not_found", "capability block not found", {"capability_block_ids": missing}, 404)

    track = Track(name=name, primary_metric_id=primary_metric_id)
    session.add(track)
    session.commit()
    session.refresh(track)

    for block in blocks:
        block.track_id = track.track_id
        session.add(block)
    ranking = RankingList(track_id=track.track_id, default_metric_id=primary_metric_id)
    session.add(ranking)
    session.commit()
    session.refresh(track)
    session.refresh(ranking)
    return track, ranking


def seed_mvp_models(session: Session) -> None:
    existing = {model.name for model in session.exec(select(Model)).all()}
    for name, family, version in MVP_STUB_MODELS:
        if name in existing:
            continue
        session.add(
            Model(
                name=name,
                model_family=family,
                model_version=version,
                endpoint_uri=f"stub://timer-service/{name.lower().replace(' ', '-')}",
            )
        )
        existing.add(name)
    session.commit()
