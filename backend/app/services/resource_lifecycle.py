from pathlib import Path
from typing import Any, Iterable

from sqlmodel import Session, select

from app.core.config import get_settings
from app.core.errors import ApiError
from app.models.benchmark import BenchmarkingRun, CapabilityBlock, CapabilityBlockShard, FailedSampleRerunJob, ForecastArtifact, RunEvent, Task, Track, Unit
from app.models.dataset import DatasetLoadJob, DatasetManifest, Shard
from app.models.lifecycle import ArchivedResource
from app.models.metric import MetricResult
from app.models.ranking import RankingEntry, RankingList
from app.models.report import Report
from app.models.sample import SampleIndex
from app.models.series_point import SeriesPoint

RESOURCE_DATASET = "dataset_manifest"
RESOURCE_SHARD = "shard"
RESOURCE_TRACK = "track"
RESOURCE_RUN = "benchmarking_run"
RESOURCE_TYPES = {RESOURCE_DATASET, RESOURCE_SHARD, RESOURCE_TRACK, RESOURCE_RUN}
TERMINAL_RUN_STATUSES = {"succeeded", "partial_succeeded", "failed", "cancelled"}

AFFECTED_KEYS = [
    "dataset_manifests",
    "load_jobs",
    "shards",
    "series_points",
    "sample_indices",
    "capability_blocks",
    "tracks",
    "benchmarking_runs",
    "units",
    "tasks",
    "reports",
    "ranking_lists",
    "ranking_entries",
    "forecast_artifacts",
    "metric_results",
    "failed_sample_rerun_jobs",
    "run_events",
]


def archive_resource(
    session: Session,
    resource_type: str,
    resource_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    _ensure_resource(session, resource_type, resource_id)
    if resource_type == RESOURCE_RUN:
        _assert_run_terminal(session, resource_id)
    existing = session.get(ArchivedResource, (resource_type, resource_id))
    if existing is None:
        existing = ArchivedResource(resource_type=resource_type, resource_id=resource_id, archived_reason=reason)
    else:
        existing.archived_reason = reason or existing.archived_reason
    session.add(existing)
    session.commit()
    session.refresh(existing)
    return _archive_payload(existing)


def restore_resource(session: Session, resource_type: str, resource_id: str) -> dict[str, Any]:
    _ensure_resource(session, resource_type, resource_id)
    existing = session.get(ArchivedResource, (resource_type, resource_id))
    if existing is not None:
        session.delete(existing)
        session.commit()
    return {"resource_type": resource_type, "resource_id": resource_id, "archived_at": None}


def archived_at(session: Session, resource_type: str, resource_id: str):
    row = session.get(ArchivedResource, (resource_type, resource_id))
    return row.archived_at if row else None


def is_archived(session: Session, resource_type: str, resource_id: str) -> bool:
    return archived_at(session, resource_type, resource_id) is not None


def archive_map(session: Session, resource_type: str, resource_ids: Iterable[str]) -> dict[str, Any]:
    ids = list(dict.fromkeys(resource_ids))
    if not ids:
        return {}
    rows = session.exec(
        select(ArchivedResource).where(
            ArchivedResource.resource_type == resource_type,
            ArchivedResource.resource_id.in_(ids),
        )
    ).all()
    return {row.resource_id: row.archived_at for row in rows}


def visible_rows(
    session: Session,
    resource_type: str,
    rows: list[Any],
    id_attr: str,
    include_archived: bool = False,
) -> list[Any]:
    archived = archive_map(session, resource_type, [getattr(row, id_attr) for row in rows])
    if include_archived:
        return rows
    return [row for row in rows if getattr(row, id_attr) not in archived]


def row_with_archive(session: Session, resource_type: str, row: Any, id_attr: str) -> dict[str, Any]:
    data = row.model_dump()
    data["archived_at"] = archived_at(session, resource_type, getattr(row, id_attr))
    return data


def deletion_impact(session: Session, resource_type: str, resource_id: str) -> dict[str, Any]:
    _ensure_resource(session, resource_type, resource_id)
    if resource_type == RESOURCE_DATASET:
        return _dataset_impact(session, resource_id)
    if resource_type == RESOURCE_SHARD:
        return _shard_impact(session, resource_id)
    if resource_type == RESOURCE_TRACK:
        return _track_impact(session, resource_id)
    if resource_type == RESOURCE_RUN:
        return _run_impact(session, resource_id)
    raise ApiError("resource_type_invalid", "invalid resource type", {"resource_type": resource_type}, 400)


def purge_dataset_manifest(session: Session, dataset_manifest_id: str, cascade: bool = False) -> dict[str, Any]:
    impact = _dataset_impact(session, dataset_manifest_id)
    if impact["cascade_required"] and not cascade:
        _raise_cascade_required(impact)
    manifest = session.get(DatasetManifest, dataset_manifest_id)
    source_uri = manifest.source_uri if manifest is not None else None
    for track_id in _track_ids_for_dataset(session, dataset_manifest_id):
        purge_track(session, track_id, cascade=True, commit=False)
    for shard_id in _shard_ids_for_dataset(session, dataset_manifest_id):
        if session.get(Shard, shard_id) is not None:
            purge_shard(session, shard_id, cascade=True, commit=False)
    _delete_where(session, DatasetLoadJob, DatasetLoadJob.dataset_manifest_id == dataset_manifest_id)
    _delete_archive_marks(session, RESOURCE_DATASET, [dataset_manifest_id])
    if manifest is not None:
        _unlink_managed_upload(source_uri)
        session.delete(manifest)
    session.commit()
    return {"ok": True, "purged": impact}


def purge_shard(session: Session, shard_id: str, cascade: bool = False, commit: bool = True) -> dict[str, Any]:
    impact = _shard_impact(session, shard_id)
    if impact["cascade_required"] and not cascade:
        _raise_cascade_required(impact)
    shard = session.get(Shard, shard_id)
    storage_uri = shard.storage_uri if shard is not None else None
    for track_id in _track_ids_for_shard(session, shard_id):
        purge_track(session, track_id, cascade=True, commit=False)
    sample_ids = _sample_ids_for_shards(session, [shard_id])
    _delete_metric_results(session, shard_ids=[shard_id], sample_ids=sample_ids)
    _delete_forecast_artifacts(session, shard_ids=[shard_id])
    _delete_where(session, SeriesPoint, SeriesPoint.shard_id == shard_id)
    _delete_where(session, SampleIndex, SampleIndex.shard_id == shard_id)
    _delete_where(session, CapabilityBlockShard, CapabilityBlockShard.shard_id == shard_id)
    _delete_orphan_blocks(session)
    _delete_archive_marks(session, RESOURCE_SHARD, [shard_id])
    if shard is not None:
        if storage_uri:
            _unlink(storage_uri)
        session.delete(shard)
    if commit:
        session.commit()
    return {"ok": True, "purged": impact}


def purge_track(session: Session, track_id: str, cascade: bool = False, commit: bool = True) -> dict[str, Any]:
    impact = _track_impact(session, track_id)
    run_ids = _run_ids_for_track(session, track_id)
    if run_ids and not cascade:
        _raise_cascade_required(impact)
    for run_id in run_ids:
        purge_run(session, run_id, commit=False)
    block_ids = _block_ids_for_track(session, track_id)
    _delete_where(session, RankingEntry, RankingEntry.track_id == track_id)
    _delete_where(session, RankingList, RankingList.track_id == track_id)
    if block_ids:
        _delete_where(session, CapabilityBlockShard, CapabilityBlockShard.capability_block_id.in_(block_ids))
        _delete_where(session, MetricResult, MetricResult.capability_block_id.in_(block_ids))
        _delete_where(session, CapabilityBlock, CapabilityBlock.capability_block_id.in_(block_ids))
    _delete_archive_marks(session, RESOURCE_TRACK, [track_id])
    track = session.get(Track, track_id)
    if track is not None:
        session.delete(track)
    if commit:
        session.commit()
    return {"ok": True, "purged": impact}


def purge_run(session: Session, run_id: str, commit: bool = True) -> dict[str, Any]:
    impact = _run_impact(session, run_id)
    _assert_run_terminal(session, run_id)
    _delete_forecast_artifacts(session, run_ids=[run_id])
    _delete_reports(session, run_ids=[run_id])
    _delete_where(session, RankingEntry, RankingEntry.benchmarking_run_id == run_id)
    _delete_where(session, MetricResult, MetricResult.benchmarking_run_id == run_id)
    _delete_where(session, FailedSampleRerunJob, FailedSampleRerunJob.benchmarking_run_id == run_id)
    _delete_where(session, RunEvent, RunEvent.benchmarking_run_id == run_id)
    _delete_where(session, Task, Task.benchmarking_run_id == run_id)
    _delete_where(session, Unit, Unit.benchmarking_run_id == run_id)
    _delete_archive_marks(session, RESOURCE_RUN, [run_id])
    run = session.get(BenchmarkingRun, run_id)
    if run is not None:
        session.delete(run)
    if commit:
        session.commit()
    return {"ok": True, "purged": impact}


def _dataset_impact(session: Session, dataset_manifest_id: str) -> dict[str, Any]:
    shard_ids = _shard_ids_for_dataset(session, dataset_manifest_id)
    track_ids = _track_ids_for_dataset(session, dataset_manifest_id)
    return _impact(
        RESOURCE_DATASET,
        dataset_manifest_id,
        _counts_for_scope(session, dataset_manifest_ids=[dataset_manifest_id], shard_ids=shard_ids, track_ids=track_ids),
        cascade_required=bool(track_ids),
    )


def _shard_impact(session: Session, shard_id: str) -> dict[str, Any]:
    track_ids = _track_ids_for_shard(session, shard_id)
    return _impact(
        RESOURCE_SHARD,
        shard_id,
        _counts_for_scope(session, shard_ids=[shard_id], track_ids=track_ids),
        cascade_required=bool(track_ids),
    )


def _track_impact(session: Session, track_id: str) -> dict[str, Any]:
    return _impact(
        RESOURCE_TRACK,
        track_id,
        _counts_for_scope(session, track_ids=[track_id]),
        cascade_required=bool(_run_ids_for_track(session, track_id)),
    )


def _run_impact(session: Session, run_id: str) -> dict[str, Any]:
    return _impact(
        RESOURCE_RUN,
        run_id,
        _counts_for_scope(session, run_ids=[run_id]),
        cascade_required=False,
        warnings=[] if session.get(BenchmarkingRun, run_id).status in TERMINAL_RUN_STATUSES else ["run_not_terminal"],
    )


def _impact(resource_type: str, resource_id: str, affected: dict[str, int], cascade_required: bool, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "archive_available": True,
        "purge_available": True,
        "cascade_required": cascade_required,
        "affected": {key: affected.get(key, 0) for key in AFFECTED_KEYS},
        "warnings": warnings or [],
    }


def _counts_for_scope(
    session: Session,
    dataset_manifest_ids: list[str] | None = None,
    shard_ids: list[str] | None = None,
    track_ids: list[str] | None = None,
    run_ids: list[str] | None = None,
) -> dict[str, int]:
    dataset_manifest_ids = _unique(dataset_manifest_ids or [])
    shard_ids = _unique(shard_ids or [])
    track_ids = _unique(track_ids or [])
    run_ids = _unique(run_ids or [])
    block_ids = _unique([*_block_ids_for_tracks(session, track_ids), *_block_ids_for_shards(session, shard_ids)])
    track_ids = _unique([*track_ids, *_track_ids_for_blocks(session, block_ids)])
    run_ids = _unique([*run_ids, *_run_ids_for_tracks(session, track_ids)])
    sample_ids = _sample_ids_for_shards(session, shard_ids)
    ranking_list_ids = _ranking_list_ids_for_tracks(session, track_ids)

    counts = {key: 0 for key in AFFECTED_KEYS}
    counts["dataset_manifests"] = len(dataset_manifest_ids)
    counts["load_jobs"] = _count_where(session, DatasetLoadJob, DatasetLoadJob.dataset_manifest_id.in_(dataset_manifest_ids)) if dataset_manifest_ids else 0
    counts["shards"] = len(shard_ids)
    counts["series_points"] = _count_where(session, SeriesPoint, SeriesPoint.shard_id.in_(shard_ids)) if shard_ids else 0
    counts["sample_indices"] = len(sample_ids)
    counts["capability_blocks"] = len(block_ids)
    counts["tracks"] = len(track_ids)
    counts["benchmarking_runs"] = len(run_ids)
    counts["units"] = _count_where(session, Unit, Unit.benchmarking_run_id.in_(run_ids)) if run_ids else 0
    counts["tasks"] = _count_where(session, Task, Task.benchmarking_run_id.in_(run_ids)) if run_ids else 0
    counts["reports"] = _count_where(session, Report, Report.benchmarking_run_id.in_(run_ids)) if run_ids else 0
    counts["ranking_lists"] = len(ranking_list_ids)
    counts["ranking_entries"] = _count_ranking_entries(session, track_ids, run_ids, ranking_list_ids)
    counts["forecast_artifacts"] = _count_forecast_artifacts(session, run_ids, shard_ids)
    counts["metric_results"] = _count_metric_results(session, run_ids, shard_ids, sample_ids, block_ids)
    counts["failed_sample_rerun_jobs"] = _count_where(session, FailedSampleRerunJob, FailedSampleRerunJob.benchmarking_run_id.in_(run_ids)) if run_ids else 0
    counts["run_events"] = _count_where(session, RunEvent, RunEvent.benchmarking_run_id.in_(run_ids)) if run_ids else 0
    return counts


def _ensure_resource(session: Session, resource_type: str, resource_id: str) -> None:
    if resource_type == RESOURCE_DATASET and session.get(DatasetManifest, resource_id) is not None:
        return
    if resource_type == RESOURCE_SHARD and session.get(Shard, resource_id) is not None:
        return
    if resource_type == RESOURCE_TRACK and session.get(Track, resource_id) is not None:
        return
    if resource_type == RESOURCE_RUN and session.get(BenchmarkingRun, resource_id) is not None:
        return
    raise ApiError("resource_not_found", "resource not found", {"resource_type": resource_type, "resource_id": resource_id}, 404)


def _assert_run_terminal(session: Session, run_id: str) -> None:
    run = session.get(BenchmarkingRun, run_id)
    if run is None:
        raise ApiError("resource_not_found", "resource not found", {"resource_type": RESOURCE_RUN, "resource_id": run_id}, 404)
    if run.status not in TERMINAL_RUN_STATUSES:
        raise ApiError("run_not_terminal", "run must be terminal before this operation", {"run_id": run_id, "status": run.status}, 409)


def _raise_cascade_required(impact: dict[str, Any]) -> None:
    raise ApiError("purge_requires_cascade", "purge requires cascade", {"impact": impact}, 409)


def _archive_payload(row: ArchivedResource) -> dict[str, Any]:
    return {
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "archived_at": row.archived_at,
        "archived_reason": row.archived_reason,
    }


def _shard_ids_for_dataset(session: Session, dataset_manifest_id: str) -> list[str]:
    return _ids(session.exec(select(Shard).where(Shard.dataset_manifest_id == dataset_manifest_id)).all(), "shard_id")


def _block_ids_for_track(session: Session, track_id: str) -> list[str]:
    return _ids(session.exec(select(CapabilityBlock).where(CapabilityBlock.track_id == track_id)).all(), "capability_block_id")


def _block_ids_for_tracks(session: Session, track_ids: list[str]) -> list[str]:
    if not track_ids:
        return []
    return _ids(session.exec(select(CapabilityBlock).where(CapabilityBlock.track_id.in_(track_ids))).all(), "capability_block_id")


def _block_ids_for_shards(session: Session, shard_ids: list[str]) -> list[str]:
    if not shard_ids:
        return []
    ids = _ids(session.exec(select(CapabilityBlockShard).where(CapabilityBlockShard.shard_id.in_(shard_ids))).all(), "capability_block_id")
    legacy = [
        shard.capability_block_id
        for shard in session.exec(select(Shard).where(Shard.shard_id.in_(shard_ids))).all()
        if shard.capability_block_id
    ]
    return _unique([*ids, *legacy])


def _track_ids_for_dataset(session: Session, dataset_manifest_id: str) -> list[str]:
    return _track_ids_for_shards(session, _shard_ids_for_dataset(session, dataset_manifest_id))


def _track_ids_for_shard(session: Session, shard_id: str) -> list[str]:
    return _track_ids_for_shards(session, [shard_id])


def _track_ids_for_shards(session: Session, shard_ids: list[str]) -> list[str]:
    return _track_ids_for_blocks(session, _block_ids_for_shards(session, shard_ids))


def _track_ids_for_blocks(session: Session, block_ids: list[str]) -> list[str]:
    if not block_ids:
        return []
    blocks = session.exec(select(CapabilityBlock).where(CapabilityBlock.capability_block_id.in_(block_ids))).all()
    return _unique([block.track_id for block in blocks if block.track_id])


def _run_ids_for_track(session: Session, track_id: str) -> list[str]:
    return _run_ids_for_tracks(session, [track_id])


def _run_ids_for_tracks(session: Session, track_ids: list[str]) -> list[str]:
    if not track_ids:
        return []
    return _ids(session.exec(select(BenchmarkingRun).where(BenchmarkingRun.track_id.in_(track_ids))).all(), "benchmarking_run_id")


def _sample_ids_for_shards(session: Session, shard_ids: list[str]) -> list[str]:
    if not shard_ids:
        return []
    return _ids(session.exec(select(SampleIndex).where(SampleIndex.shard_id.in_(shard_ids))).all(), "sample_id")


def _ranking_list_ids_for_tracks(session: Session, track_ids: list[str]) -> list[str]:
    if not track_ids:
        return []
    return _ids(session.exec(select(RankingList).where(RankingList.track_id.in_(track_ids))).all(), "ranking_list_id")


def _count_ranking_entries(session: Session, track_ids: list[str], run_ids: list[str], ranking_list_ids: list[str]) -> int:
    ids: set[str] = set()
    if track_ids:
        ids.update(_ids(session.exec(select(RankingEntry).where(RankingEntry.track_id.in_(track_ids))).all(), "ranking_entry_id"))
    if run_ids:
        ids.update(_ids(session.exec(select(RankingEntry).where(RankingEntry.benchmarking_run_id.in_(run_ids))).all(), "ranking_entry_id"))
    if ranking_list_ids:
        ids.update(_ids(session.exec(select(RankingEntry).where(RankingEntry.ranking_list_id.in_(ranking_list_ids))).all(), "ranking_entry_id"))
    return len(ids)


def _count_forecast_artifacts(session: Session, run_ids: list[str], shard_ids: list[str]) -> int:
    ids: set[str] = set()
    if run_ids:
        ids.update(_ids(session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id.in_(run_ids))).all(), "forecast_artifact_id"))
    if shard_ids:
        ids.update(_ids(session.exec(select(ForecastArtifact).where(ForecastArtifact.shard_id.in_(shard_ids))).all(), "forecast_artifact_id"))
    return len(ids)


def _count_metric_results(session: Session, run_ids: list[str], shard_ids: list[str], sample_ids: list[str], block_ids: list[str]) -> int:
    ids: set[str] = set()
    if run_ids:
        ids.update(_ids(session.exec(select(MetricResult).where(MetricResult.benchmarking_run_id.in_(run_ids))).all(), "metric_result_id"))
    if shard_ids:
        ids.update(_ids(session.exec(select(MetricResult).where(MetricResult.shard_id.in_(shard_ids))).all(), "metric_result_id"))
    if sample_ids:
        ids.update(_ids(session.exec(select(MetricResult).where(MetricResult.sample_id.in_(sample_ids))).all(), "metric_result_id"))
    if block_ids:
        ids.update(_ids(session.exec(select(MetricResult).where(MetricResult.capability_block_id.in_(block_ids))).all(), "metric_result_id"))
    return len(ids)


def _delete_metric_results(
    session: Session,
    run_ids: list[str] | None = None,
    shard_ids: list[str] | None = None,
    sample_ids: list[str] | None = None,
    block_ids: list[str] | None = None,
) -> None:
    seen: set[str] = set()
    rows: list[MetricResult] = []
    for condition in [
        MetricResult.benchmarking_run_id.in_(run_ids or []) if run_ids else None,
        MetricResult.shard_id.in_(shard_ids or []) if shard_ids else None,
        MetricResult.sample_id.in_(sample_ids or []) if sample_ids else None,
        MetricResult.capability_block_id.in_(block_ids or []) if block_ids else None,
    ]:
        if condition is None:
            continue
        for row in session.exec(select(MetricResult).where(condition)).all():
            if row.metric_result_id not in seen:
                rows.append(row)
                seen.add(row.metric_result_id)
    for row in rows:
        session.delete(row)


def _delete_forecast_artifacts(
    session: Session,
    run_ids: list[str] | None = None,
    shard_ids: list[str] | None = None,
) -> None:
    seen: set[str] = set()
    rows: list[ForecastArtifact] = []
    for condition in [
        ForecastArtifact.benchmarking_run_id.in_(run_ids or []) if run_ids else None,
        ForecastArtifact.shard_id.in_(shard_ids or []) if shard_ids else None,
    ]:
        if condition is None:
            continue
        for row in session.exec(select(ForecastArtifact).where(condition)).all():
            if row.forecast_artifact_id not in seen:
                rows.append(row)
                seen.add(row.forecast_artifact_id)
    for row in rows:
        _unlink(row.storage_uri)
        session.delete(row)


def _delete_reports(session: Session, run_ids: list[str]) -> None:
    for report in session.exec(select(Report).where(Report.benchmarking_run_id.in_(run_ids))).all():
        if report.storage_uri:
            _unlink(report.storage_uri)
        session.delete(report)


def _delete_archive_marks(session: Session, resource_type: str, resource_ids: list[str]) -> None:
    if not resource_ids:
        return
    rows = session.exec(
        select(ArchivedResource).where(
            ArchivedResource.resource_type == resource_type,
            ArchivedResource.resource_id.in_(resource_ids),
        )
    ).all()
    for row in rows:
        session.delete(row)


def _delete_orphan_blocks(session: Session) -> None:
    links = session.exec(select(CapabilityBlockShard)).all()
    linked_block_ids = {link.capability_block_id for link in links}
    blocks = session.exec(select(CapabilityBlock)).all()
    for block in blocks:
        if block.capability_block_id not in linked_block_ids and block.track_id is None:
            session.delete(block)


def _delete_where(session: Session, model: Any, *conditions: Any) -> None:
    for row in session.exec(select(model).where(*conditions)).all():
        session.delete(row)


def _count_where(session: Session, model: Any, *conditions: Any) -> int:
    return len(session.exec(select(model).where(*conditions)).all())


def _ids(rows: Iterable[Any], attr: str) -> list[str]:
    return _unique([getattr(row, attr) for row in rows])


def _unique(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys([value for value in values if value]))


def _unlink(uri: str) -> None:
    try:
        path = Path(uri)
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _unlink_managed_upload(uri: str | None) -> None:
    if not uri:
        return
    try:
        uploads_dir = get_settings().uploads_dir.resolve()
        path = Path(uri).resolve()
        if path.is_file() and uploads_dir in path.parents:
            path.unlink()
    except OSError:
        pass
