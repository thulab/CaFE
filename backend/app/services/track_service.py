from collections import defaultdict
from typing import Any

from sqlmodel import Session, select

from app.core.errors import ApiError
from app.models.benchmark import CapabilityBlock, CapabilityBlockShard, Track
from app.models.dataset import Shard
from app.models.model_registry import Model
from app.models.ranking import RankingList
from app.services.synthetic_generation_service import CAPABILITIES_BY_ID


MVP_STUB_MODELS = [
    ("Timer 3.5", "Timer", "3.5", {"max_target_count": 1, "max_covariate_count": 0}),
    ("Timer 3.0", "Timer", "3.0", {"max_target_count": 1, "max_covariate_count": 0}),
    ("Chronos 2", "Chronos", "2", {"max_target_count": 1, "max_covariate_count": 50}),
    ("toto", "toto", "mvp", {"max_target_count": None, "max_covariate_count": 0}),
    ("TimesFM 2.5", "TimesFM", "2.5", {"max_target_count": 1, "max_covariate_count": 0}),
]


def create_capability_block_from_shards(
    session: Session,
    name: str,
    shard_ids: list[str],
    block_type: str,
    capability_type: str,
    generation_config: dict[str, Any] | None = None,
) -> CapabilityBlock:
    if not shard_ids:
        raise ApiError("capability_block_requires_shard", "capability block requires at least one shard")
    unique_shard_ids = list(dict.fromkeys(shard_ids))
    shards = [session.get(Shard, shard_id) for shard_id in unique_shard_ids]
    missing = [shard_id for shard_id, shard in zip(unique_shard_ids, shards, strict=True) if shard is None]
    if missing:
        raise ApiError("shard_not_found", "shard not found", {"shard_ids": missing}, 404)

    block = CapabilityBlock(
        name=name,
        block_type=block_type,
        capability_type=capability_type,
        task_type=_task_type_for_shards(shards),
        shard_count=len(shards),
        sample_count=sum(shard.sample_count for shard in shards),
        target_dim=max((shard.target_dim for shard in shards), default=1),
        covariate_dim=max((shard.covariate_dim for shard in shards), default=0),
        generation_config=generation_config or _generation_config_for_block(block_type, capability_type, shards),
    )
    session.add(block)
    session.commit()
    session.refresh(block)

    for shard in shards:
        session.add(CapabilityBlockShard(capability_block_id=block.capability_block_id, shard_id=shard.shard_id))
    session.commit()
    session.refresh(block)
    return block


def create_real_capability_block(session: Session, name: str, shard_ids: list[str]) -> CapabilityBlock:
    return create_capability_block_from_shards(session, name, shard_ids, "real", "real_data")


def create_track_from_shards(
    session: Session,
    name: str,
    shard_ids: list[str],
    primary_metric_id: str = "mase",
) -> tuple[Track, RankingList, list[CapabilityBlock]]:
    if not shard_ids:
        raise ApiError("track_requires_shard", "track requires at least one shard")
    unique_shard_ids = list(dict.fromkeys(shard_ids))
    shards = [session.get(Shard, shard_id) for shard_id in unique_shard_ids]
    missing = [shard_id for shard_id, shard in zip(unique_shard_ids, shards, strict=True) if shard is None]
    if missing:
        raise ApiError("shard_not_found", "shard not found", {"shard_ids": missing}, 404)

    grouped: dict[tuple[str, str], list[Shard]] = defaultdict(list)
    for shard in shards:
        block_type = "synthetic" if shard.shard_type == "synthetic" else "real"
        capability_type = shard.capability_type or (
            str((shard.generation_config or {}).get("capability_id")) if shard.shard_type == "synthetic" else "real_data"
        )
        if not capability_type or capability_type == "None":
            capability_type = "synthetic_data" if block_type == "synthetic" else "real_data"
        grouped[(block_type, capability_type)].append(shard)

    blocks: list[CapabilityBlock] = []
    try:
        for (block_type, capability_type), block_shards in grouped.items():
            block = create_capability_block_from_shards(
                session,
                _block_name(name, block_type, capability_type),
                [shard.shard_id for shard in block_shards],
                block_type,
                capability_type,
            )
            blocks.append(block)
        track_type = _track_type_for_blocks(blocks)
        track, ranking = create_track_with_blocks(
            session,
            name,
            [block.capability_block_id for block in blocks],
            primary_metric_id,
            track_type=track_type,
        )
        return track, ranking, blocks
    except Exception:
        session.rollback()
        delete_capability_blocks(session, [block.capability_block_id for block in blocks])
        raise


def delete_capability_blocks(session: Session, capability_block_ids: list[str]) -> None:
    for capability_block_id in capability_block_ids:
        for link in session.exec(select(CapabilityBlockShard).where(CapabilityBlockShard.capability_block_id == capability_block_id)).all():
            session.delete(link)
        block = session.get(CapabilityBlock, capability_block_id)
        if block is not None:
            session.delete(block)
    session.commit()


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
    track_type: str | None = None,
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

    track = Track(name=name, primary_metric_id=primary_metric_id, track_type=track_type or _track_type_for_blocks(blocks))
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


def _task_type_for_shards(shards: list[Shard]) -> str:
    if any(shard.covariate_dim > 0 for shard in shards):
        return "covariate_forecast"
    if any(shard.target_dim > 1 for shard in shards):
        return "multivariate_forecast"
    return "univariate_forecast"


def _generation_config_for_block(block_type: str, capability_type: str, shards: list[Shard]) -> dict[str, Any]:
    if block_type != "synthetic":
        return {}
    return {
        "schema_version": "capability_block_generation.v1",
        "capability_id": capability_type,
        "shard_count": len(shards),
        "shard_generation_configs": [shard.generation_config for shard in shards if shard.generation_config],
    }


def _block_name(track_name: str, block_type: str, capability_type: str) -> str:
    if block_type == "real":
        return f"{track_name} real data"
    capability = CAPABILITIES_BY_ID.get(capability_type)
    label = capability.label if capability else capability_type.replace("_", " ")
    return f"{track_name} {label}"


def _track_type_for_blocks(blocks: list[CapabilityBlock]) -> str:
    block_types = {block.block_type for block in blocks}
    if block_types == {"synthetic"}:
        return "synthetic_dataset"
    if "synthetic" in block_types:
        return "mixed_dataset"
    return "real_dataset"


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
