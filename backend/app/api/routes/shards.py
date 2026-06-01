from fastapi import Depends
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.errors import ApiError
from app.models.dataset import DatasetManifest, Shard
from app.models.sample import SampleIndex
from app.services.resource_lifecycle import (
    RESOURCE_SHARD,
    archive_resource,
    deletion_impact,
    purge_shard,
    restore_resource,
    row_with_archive,
    visible_rows,
)

router = make_router(prefix="/shards", tags=["shards"])


@router.get("", tier="authed")
def list_shards(
    dataset_manifest_id: str | None = None,
    q: str | None = None,
    include_archived: bool = False,
    limit: int = 50,
    offset: int = 0,
    session: Session = Depends(get_db_session),
) -> dict:
    base = select(Shard)
    if dataset_manifest_id:
        base = base.where(Shard.dataset_manifest_id == dataset_manifest_id)
    rows = session.exec(base.order_by(Shard.created_at.desc())).all()
    rows = visible_rows(session, RESOURCE_SHARD, rows, "shard_id", include_archived)
    manifest_names = _manifest_names(session, rows)
    rows = _filter_shards(rows, manifest_names, q)
    total = len(rows)
    items = rows[offset : offset + limit]
    return {
        "items": [_row_with_dataset_name(session, item, manifest_names) for item in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{shard_id}/deletion-impact", tier="perm", perm="dataset.delete")
def get_shard_deletion_impact(shard_id: str, session: Session = Depends(get_db_session)) -> dict:
    return deletion_impact(session, RESOURCE_SHARD, shard_id)


@router.post("/{shard_id}/archive", tier="perm", perm="dataset.delete")
def archive_shard(shard_id: str, session: Session = Depends(get_db_session)) -> dict:
    return archive_resource(session, RESOURCE_SHARD, shard_id)


@router.post("/{shard_id}/restore", tier="perm", perm="dataset.delete")
def restore_shard(shard_id: str, session: Session = Depends(get_db_session)) -> dict:
    return restore_resource(session, RESOURCE_SHARD, shard_id)


@router.delete("/{shard_id}", tier="perm", perm="admin.purge")
def delete_shard(shard_id: str, cascade: bool = False, session: Session = Depends(get_db_session)) -> dict:
    return purge_shard(session, shard_id, cascade)


@router.get("/{shard_id}", tier="authed")
def get_shard(shard_id: str, session: Session = Depends(get_db_session)) -> dict:
    shard = session.get(Shard, shard_id)
    if shard is None:
        raise ApiError("resource_not_found", "resource not found", {"resource_type": RESOURCE_SHARD, "resource_id": shard_id}, 404)
    return row_with_archive(session, RESOURCE_SHARD, shard, "shard_id")


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


def _manifest_names(session: Session, rows: list[Shard]) -> dict[str, str]:
    manifest_ids = list({row.dataset_manifest_id for row in rows})
    if not manifest_ids:
        return {}
    manifests = session.exec(select(DatasetManifest).where(DatasetManifest.dataset_manifest_id.in_(manifest_ids))).all()
    return {manifest.dataset_manifest_id: manifest.name for manifest in manifests}


def _filter_shards(rows: list[Shard], manifest_names: dict[str, str], query: str | None) -> list[Shard]:
    needle = (query or "").strip().lower()
    if not needle:
        return rows

    def matches(row: Shard) -> bool:
        values = [
            row.name or "",
            row.shard_id,
            row.source_uri,
            manifest_names.get(row.dataset_manifest_id, ""),
            " ".join(row.target_columns or []),
        ]
        return any(needle in value.lower() for value in values)

    return [row for row in rows if matches(row)]


def _row_with_dataset_name(session: Session, row: Shard, manifest_names: dict[str, str]) -> dict:
    data = row_with_archive(session, RESOURCE_SHARD, row, "shard_id")
    data["dataset_name"] = manifest_names.get(row.dataset_manifest_id)
    return data
