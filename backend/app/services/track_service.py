from sqlmodel import Session, select

from app.core.errors import ApiError
from app.models.benchmark import CapabilityBlock, CapabilityBlockShard, Track
from app.models.dataset import Shard
from app.models.model_registry import Model
from app.models.ranking import RankingList


MVP_STUB_MODELS = [
    ("Timer 3.5", "Timer", "3.5", {"max_target_count": 1}),
    ("Timer 3.0", "Timer", "3.0", {"max_target_count": 1}),
    ("Chronos 2", "Chronos", "2", {"max_target_count": 1}),
    ("toto", "toto", "mvp", {"max_target_count": None}),
    ("TimesFM 2.5", "TimesFM", "2.5", {"max_target_count": 1}),
]


def create_real_capability_block(session: Session, name: str, shard_ids: list[str]) -> CapabilityBlock:
    if not shard_ids:
        raise ApiError("capability_block_requires_shard", "real capability block requires at least one shard")
    unique_shard_ids = list(dict.fromkeys(shard_ids))
    shards = [session.get(Shard, shard_id) for shard_id in unique_shard_ids]
    missing = [shard_id for shard_id, shard in zip(unique_shard_ids, shards, strict=True) if shard is None]
    if missing:
        raise ApiError("shard_not_found", "shard not found", {"shard_ids": missing}, 404)

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
        session.add(CapabilityBlockShard(capability_block_id=block.capability_block_id, shard_id=shard.shard_id))
    session.commit()
    session.refresh(block)
    return block


def shards_for_capability_block(session: Session, capability_block_id: str) -> list[Shard]:
    links = session.exec(
        select(CapabilityBlockShard).where(CapabilityBlockShard.capability_block_id == capability_block_id)
    ).all()
    if links:
        shards = [session.get(Shard, link.shard_id) for link in links]
        return [shard for shard in shards if shard is not None]
    return list(session.exec(select(Shard).where(Shard.capability_block_id == capability_block_id)).all())


def track_summary(session: Session, track: Track) -> dict:
    blocks = session.exec(select(CapabilityBlock).where(CapabilityBlock.track_id == track.track_id)).all()
    shards_by_id: dict[str, Shard] = {}
    for block in blocks:
        shards = shards_for_capability_block(session, block.capability_block_id)
        for shard in shards:
            shards_by_id.setdefault(shard.shard_id, shard)
    shard_ids = list(shards_by_id)
    data = track.model_dump()
    data.update(
        {
            "capability_block_ids": [block.capability_block_id for block in blocks],
            "shard_ids": shard_ids,
            "shard_count": len(shard_ids),
            "sample_count": sum(shard.sample_count for shard in shards_by_id.values()),
        }
    )
    return data


def create_track_with_blocks(
    session: Session,
    name: str,
    capability_block_ids: list[str],
    primary_metric_id: str = "mase",
) -> tuple[Track, RankingList]:
    if not capability_block_ids:
        raise ApiError("track_requires_block", "track requires at least one capability block")
    blocks = [session.get(CapabilityBlock, block_id) for block_id in capability_block_ids]
    missing = [block_id for block_id, block in zip(capability_block_ids, blocks, strict=True) if block is None]
    if missing:
        raise ApiError("capability_block_not_found", "capability block not found", {"capability_block_ids": missing}, 404)
    assigned = [block.capability_block_id for block in blocks if block.track_id]
    if assigned:
        raise ApiError("capability_block_already_assigned", "capability block already belongs to a track", {"capability_block_ids": assigned})

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
    existing = {model.name: model for model in session.exec(select(Model)).all()}
    for name, family, version, forecast_limits in MVP_STUB_MODELS:
        if name in existing:
            model = existing[name]
            if not isinstance(model.forecast_limits, dict) or not model.forecast_limits:
                model.forecast_limits = forecast_limits
                session.add(model)
            continue
        session.add(
            Model(
                name=name,
                model_family=family,
                model_version=version,
                endpoint_uri=f"stub://timer-service/{name.lower().replace(' ', '-')}",
                forecast_limits=forecast_limits,
            )
        )
    session.commit()
