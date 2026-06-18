import json
from pathlib import Path
from typing import Literal

from sqlalchemy import func
from sqlmodel import Session, select

from app.core.errors import ApiError
from app.models.benchmark import BenchmarkingRun, CapabilityBlock, CapabilityBlockShard, ForecastArtifact, Task, Track, Unit
from app.models.metric import MetricResult
from app.models.model_registry import Model
from app.models.report import Report
from app.models.sample import SampleIndex
from app.models.series_point import SeriesPoint
from app.services.metric_service import _mase_scale
from app.services.sample_store import SampleStore

SampleLinkSort = Literal["sample_index", "metric_desc", "metric_asc"]
KNOWN_SAMPLE_METRICS = ["mase", "mse", "rmse", "mae", "smape", "mape"]


def generate_run_report(session: Session, run_id: str, runtime_dir: Path) -> Report:
    run = session.get(BenchmarkingRun, run_id)
    if run.status == "cancelled":
        raise ApiError("cancelled_run_has_no_report", "cancelled runs do not generate reports", {"benchmarking_run_id": run_id}, 409)
    report_dir = Path(runtime_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"{run_id}.json"
    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    tasks = session.exec(select(Task).where(Task.benchmarking_run_id == run_id)).all()
    metrics = session.exec(select(MetricResult).where(MetricResult.benchmarking_run_id == run_id)).all()
    artifacts = session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)).all()
    task_sample_counts = _task_sample_counts_from_artifacts(artifacts)

    payload = {
        "benchmarking_run_id": run_id,
        "track_id": run.track_id,
        "status": run.status,
        "model_metrics": [_unit_metrics(session, unit, metrics) for unit in units],
        "task_summaries": [_task_summary(task, metrics, task_sample_counts) for task in tasks],
        "capability_blocks": _capability_blocks(session, tasks),
        "capability_metrics": [_capability_metric(session, task, metrics, task_sample_counts) for task in tasks],
        "sample_forecast_links": _sample_links(session, run_id, artifacts),
        "cancellation_reason": "cancel_requested" if run.status == "cancelled" else None,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    existing = session.exec(select(Report).where(Report.benchmarking_run_id == run_id)).first()
    report = existing or Report(benchmarking_run_id=run_id, track_id=run.track_id)
    report.status = "ready"
    report.storage_uri = str(output)
    report.summary = {"status": run.status, "model_count": len(units), "task_count": len(tasks)}
    run.report_id = report.report_id
    session.add(report)
    session.add(run)
    session.commit()
    session.refresh(report)
    return report


def read_report(
    report: Report,
    session: Session | None = None,
    sample_link_limit: int | None = None,
    sample_link_offset: int = 0,
    sample_link_capability_block_id: str | None = None,
    sample_link_metric: str | None = None,
    sample_link_sort: SampleLinkSort = "sample_index",
) -> dict:
    payload = json.loads(Path(report.storage_uri).read_text(encoding="utf-8"))
    offset = max(0, int(sample_link_offset or 0))
    limit = int(sample_link_limit) if sample_link_limit is not None else None
    if limit is not None:
        limit = max(0, limit)

    metric_id = _resolve_sample_link_metric(session, report, payload, sample_link_metric) if session is not None else sample_link_metric
    if session is not None and metric_id is not None:
        links, total = _sample_links_page_from_metrics(
            session,
            report,
            metric_id,
            offset,
            limit,
            sample_link_capability_block_id,
            sample_link_sort,
        )
        payload["sample_forecast_links"] = links
    else:
        links = _sample_links_page_from_payload(
            payload,
            offset,
            limit,
            sample_link_capability_block_id,
            metric_id,
            sample_link_sort,
        )
        total = links["total"]
        payload["sample_forecast_links"] = links["items"]

    payload["sample_forecast_links_total"] = total
    payload["sample_forecast_links_limit"] = limit if limit is not None else total
    payload["sample_forecast_links_offset"] = offset
    payload["sample_forecast_links_capability_block_id"] = sample_link_capability_block_id
    payload["sample_forecast_links_metric"] = metric_id
    payload["sample_forecast_links_sort"] = sample_link_sort
    return payload


def _unit_metrics(session: Session, unit: Unit, metrics: list[MetricResult]) -> dict:
    model = session.get(Model, unit.model_id)
    unit_metrics = {
        metric.metric_id: metric.value
        for metric in metrics
        if metric.result_level == "unit" and metric.unit_id == unit.unit_id
    }
    entry = {
        "unit_id": unit.unit_id,
        "model_id": unit.model_id,
        "model_name": model.name if model else unit.model_id,
        "status": unit.status,
        "metrics": unit_metrics,
    }
    # Item #14: a succeeded unit whose samples are all flat (stationary) produces
    # no MASE at all. Rather than letting the primary metric silently vanish from
    # the report, surface that MASE is unavailable and WHY.
    if unit.status == "succeeded" and "mase" not in unit_metrics:
        reason = _mase_unavailable_reason_for_unit(session, unit, metrics)
        if reason is not None:
            entry["metrics"]["mase"] = None
            entry["mase_unavailable_reason"] = reason
    return entry


def _mase_unavailable_reason_for_unit(session: Session, unit: Unit, metrics: list[MetricResult]) -> str | None:
    """Why does this succeeded unit have no MASE? Returns a reason code or None.

    A unit can legitimately lack a MASE unit-metric for reasons unrelated to flat
    history (e.g. it never produced any sample at all). We only report a reason
    when the unit has succeeded samples (mse rows exist) yet not a single MASE
    sample row — then we recompute the cause from one sample's history.
    """
    has_sample_metric = any(m.result_level == "sample" and m.unit_id == unit.unit_id for m in metrics)
    has_sample_mase = any(
        m.result_level == "sample" and m.unit_id == unit.unit_id and m.metric_id == "mase" for m in metrics
    )
    if not has_sample_metric or has_sample_mase:
        return None

    store = SampleStore()
    for sample_id in _unit_sample_ids(session, unit):
        sample_index = session.get(SampleIndex, sample_id)
        if sample_index is None:
            continue
        sample = store.read_by_ref(session, sample_index.storage_ref)
        _scale, reason = _mase_scale(sample["target_history"])
        if reason is not None:
            return reason
    return None


def _unit_sample_ids(session: Session, unit: Unit) -> list[str]:
    shard_ids = {
        metric.shard_id
        for metric in session.exec(
            select(MetricResult).where(
                MetricResult.result_level == "sample",
                MetricResult.unit_id == unit.unit_id,
            )
        ).all()
        if metric.shard_id is not None
    }
    if not shard_ids:
        return []
    samples = session.exec(select(SampleIndex).where(SampleIndex.shard_id.in_(shard_ids))).all()
    return [sample.sample_id for sample in samples]


def _task_summary(task: Task, metrics: list[MetricResult], task_sample_counts: dict[str, dict[str, int]] | None = None) -> dict:
    counts = _task_sample_counts(task, task_sample_counts)
    return {
        "task_id": task.task_id,
        "unit_id": task.unit_id,
        "model_id": task.model_id,
        "capability_block_id": task.capability_block_id,
        "status": task.status,
        "sample_count": task.sample_count,
        "processed_sample_count": counts["processed"],
        "failed_sample_count": counts["failed"],
        "error_code": task.error_code,
        "error_message": task.error_message,
        "metrics": {
            metric.metric_id: metric.value
            for metric in metrics
            if metric.result_level == "task" and metric.task_id == task.task_id
        },
    }


def _capability_blocks(session: Session, tasks: list[Task]) -> list[dict]:
    block_ids = list(dict.fromkeys(task.capability_block_id for task in tasks))
    blocks = [session.get(CapabilityBlock, block_id) for block_id in block_ids]
    return [_capability_block_summary(session, block) for block in blocks if block is not None]


def _capability_block_summary(session: Session, block: CapabilityBlock) -> dict:
    links = session.exec(
        select(CapabilityBlockShard).where(CapabilityBlockShard.capability_block_id == block.capability_block_id)
    ).all()
    return {
        "capability_block_id": block.capability_block_id,
        "name": block.name,
        "block_type": block.block_type,
        "capability_type": block.capability_type,
        "capability_label": _capability_label(block),
        "task_type": block.task_type,
        "target_dim": block.target_dim,
        "covariate_dim": block.covariate_dim,
        "shard_count": block.shard_count,
        "sample_count": block.sample_count,
        "aggregation_policy": block.aggregation_policy,
        "generation_config": block.generation_config or {},
        "shard_ids": [link.shard_id for link in links],
    }


def _capability_label(block: CapabilityBlock) -> str:
    if block.block_type == "real":
        return "Real data"
    config = block.generation_config or {}
    direct = config.get("capability_label")
    if isinstance(direct, str) and direct:
        return direct
    for shard_config in config.get("shard_generation_configs") or []:
        if isinstance(shard_config, dict):
            label = shard_config.get("capability_label")
            if isinstance(label, str) and label:
                return label
    return block.capability_type.replace("_", " ").title()


def _capability_metric(session: Session, task: Task, metrics: list[MetricResult], task_sample_counts: dict[str, dict[str, int]] | None = None) -> dict:
    model = session.get(Model, task.model_id)
    counts = _task_sample_counts(task, task_sample_counts)
    return {
        "task_id": task.task_id,
        "unit_id": task.unit_id,
        "model_id": task.model_id,
        "model_name": model.name if model else task.model_id,
        "capability_block_id": task.capability_block_id,
        "status": task.status,
        "sample_count": task.sample_count,
        "processed_sample_count": counts["processed"],
        "failed_sample_count": counts["failed"],
        "error_code": task.error_code,
        "error_message": task.error_message,
        "metrics": {
            metric.metric_id: metric.value
            for metric in metrics
            if metric.result_level == "task" and metric.task_id == task.task_id
        },
    }


def _task_sample_counts_from_artifacts(artifacts: list[ForecastArtifact]) -> dict[str, dict[str, int]]:
    counts_by_task: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        counts = counts_by_task.setdefault(artifact.task_id, {"processed": 0, "failed": 0})
        try:
            with Path(artifact.storage_uri).open(encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line)
                    if row.get("status") in {"succeeded", "failed"}:
                        counts["processed"] += 1
                    if row.get("status") == "failed":
                        counts["failed"] += 1
        except FileNotFoundError:
            continue
    return counts_by_task


def _task_sample_counts(task: Task, task_sample_counts: dict[str, dict[str, int]] | None = None) -> dict[str, int]:
    counts = (task_sample_counts or {}).get(task.task_id)
    if counts and counts["processed"] > 0:
        return counts
    return {
        "processed": int(task.processed_sample_count or 0),
        "failed": int(task.failed_sample_count or 0),
    }


def _resolve_sample_link_metric(session: Session | None, report: Report, payload: dict, requested: str | None) -> str | None:
    if session is None:
        return requested
    available = session.exec(
        select(MetricResult.metric_id)
        .where(
            MetricResult.benchmarking_run_id == report.benchmarking_run_id,
            MetricResult.result_level == "sample",
        )
        .distinct()
    ).all()
    available_set = {str(metric_id) for metric_id in available if metric_id}
    if requested:
        return requested

    track_metric = None
    if report.track_id:
        track = session.get(Track, report.track_id)
        track_metric = track.primary_metric_id if track else None
    if track_metric and track_metric in available_set:
        return track_metric

    for metric_id in _payload_metric_keys(payload):
        if metric_id in available_set:
            return metric_id
    for metric_id in KNOWN_SAMPLE_METRICS:
        if metric_id in available_set:
            return metric_id
    return next(iter(sorted(available_set)), None)


def _payload_metric_keys(payload: dict) -> list[str]:
    keys: set[str] = set()
    for item in payload.get("model_metrics") or []:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics")
        if isinstance(metrics, dict):
            keys.update(str(key) for key, value in metrics.items() if isinstance(value, int | float))
    return sorted(
        keys,
        key=lambda key: (
            KNOWN_SAMPLE_METRICS.index(key) if key in KNOWN_SAMPLE_METRICS else len(KNOWN_SAMPLE_METRICS),
            key,
        ),
    )


def _sample_links_page_from_metrics(
    session: Session,
    report: Report,
    metric_id: str,
    offset: int,
    limit: int | None,
    capability_block_id: str | None,
    sort_mode: SampleLinkSort,
) -> tuple[list[dict], int]:
    metric_value = func.avg(MetricResult.value).label("metric_value")
    model_count = func.count(func.distinct(MetricResult.model_id)).label("model_count")
    base = (
        select(
            MetricResult.sample_id,
            MetricResult.capability_block_id,
            metric_value,
            model_count,
            SampleIndex.sample_index,
        )
        .join(SampleIndex, SampleIndex.sample_id == MetricResult.sample_id)
        .where(
            MetricResult.benchmarking_run_id == report.benchmarking_run_id,
            MetricResult.result_level == "sample",
            MetricResult.metric_id == metric_id,
            MetricResult.sample_id.is_not(None),
        )
        .group_by(MetricResult.sample_id, MetricResult.capability_block_id, SampleIndex.sample_index)
    )
    if capability_block_id:
        base = base.where(MetricResult.capability_block_id == capability_block_id)

    total = int(session.exec(select(func.count()).select_from(base.subquery())).one() or 0)
    if sort_mode == "metric_desc":
        ordered = base.order_by(metric_value.desc(), SampleIndex.sample_index.asc(), MetricResult.sample_id.asc())
    elif sort_mode == "metric_asc":
        ordered = base.order_by(metric_value.asc(), SampleIndex.sample_index.asc(), MetricResult.sample_id.asc())
    else:
        ordered = base.order_by(SampleIndex.sample_index.asc(), MetricResult.sample_id.asc())
    if limit is not None:
        ordered = ordered.offset(offset).limit(limit)
    else:
        ordered = ordered.offset(offset)

    rows = session.exec(ordered).all()
    return _sample_links_from_metric_rows(session, report.benchmarking_run_id, metric_id, rows), total


def _sample_links_from_metric_rows(session: Session, run_id: str, metric_id: str, rows: list) -> list[dict]:
    sample_ids = [str(row[0]) for row in rows if row[0] is not None]
    if not sample_ids:
        return []

    samples = session.exec(select(SampleIndex).where(SampleIndex.sample_id.in_(sample_ids))).all()
    samples_by_id = {sample.sample_id: sample for sample in samples}

    block_ids = [str(row[1]) for row in rows if row[1]]
    blocks_by_id: dict[str, CapabilityBlock] = {}
    if block_ids:
        blocks = session.exec(select(CapabilityBlock).where(CapabilityBlock.capability_block_id.in_(list(set(block_ids))))).all()
        blocks_by_id = {block.capability_block_id: block for block in blocks}

    shard_ids = list({sample.shard_id for sample in samples})
    artifacts_by_shard: dict[str, list[ForecastArtifact]] = {}
    if shard_ids:
        artifacts = session.exec(
            select(ForecastArtifact).where(
                ForecastArtifact.benchmarking_run_id == run_id,
                ForecastArtifact.shard_id.in_(shard_ids),
            )
        ).all()
        for artifact in artifacts:
            artifacts_by_shard.setdefault(artifact.shard_id, []).append(artifact)

    links: list[dict] = []
    for row in rows:
        sample_id = str(row[0])
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        artifacts = artifacts_by_shard.get(sample.shard_id, [])
        artifact_ids = [artifact.forecast_artifact_id for artifact in artifacts]
        link = _sample_link(session, run_id, sample_id, artifact_ids[0] if artifact_ids else None)
        link["forecast_artifact_ids"] = artifact_ids
        link["model_count"] = int(row[3] or 0)
        link["metric_id"] = metric_id
        link["metric_value"] = float(row[2]) if row[2] is not None else None

        block_id = str(row[1]) if row[1] else None
        block = blocks_by_id.get(block_id) if block_id else None
        if block_id:
            link["capability_block_id"] = block_id
        if block is not None:
            link["capability_block_name"] = block.name
            link["capability_type"] = block.capability_type
            link["capability_label"] = _capability_label(block)
        links.append(link)
    return links


def _sample_links_page_from_payload(
    payload: dict,
    offset: int,
    limit: int | None,
    capability_block_id: str | None,
    metric_id: str | None,
    sort_mode: SampleLinkSort,
) -> dict:
    links = list(payload.get("sample_forecast_links") or [])
    if capability_block_id:
        links = [link for link in links if link.get("capability_block_id") == capability_block_id]
    if sort_mode in {"metric_desc", "metric_asc"}:
        def metric_sort_key(link: dict) -> tuple:
            value = link.get("metric_value")
            numeric = float(value) if isinstance(value, int | float) else 0.0
            if sort_mode == "metric_desc":
                numeric = -numeric
            return (
                value is None,
                numeric,
                int(link.get("sample_index") if link.get("sample_index") is not None else 10**12),
                str(link.get("sample_id") or ""),
            )

        links.sort(key=metric_sort_key)
    else:
        links.sort(
            key=lambda link: (
                int(link.get("sample_index") if link.get("sample_index") is not None else 10**12),
                str(link.get("sample_id") or ""),
            )
        )
    if metric_id:
        for link in links:
            link.setdefault("metric_id", metric_id)
    total = len(links)
    items = links[offset : offset + limit] if limit is not None else links[offset:]
    return {"items": items, "total": total}


def _sample_links(session: Session, run_id: str, artifacts: list[ForecastArtifact]) -> list[dict]:
    links_by_sample: dict[str, dict] = {}
    models_by_sample: dict[str, set[str]] = {}
    for artifact in artifacts:
        with Path(artifact.storage_uri).open(encoding="utf-8") as file:
            for line in file:
                sample_id = json.loads(line)["sample_id"]
                if sample_id not in links_by_sample:
                    links_by_sample[sample_id] = _sample_link(session, run_id, sample_id, artifact.forecast_artifact_id)
                    models_by_sample[sample_id] = set()
                models_by_sample[sample_id].add(artifact.model_id)
                links_by_sample[sample_id]["forecast_artifact_ids"].append(artifact.forecast_artifact_id)
    links = list(links_by_sample.values())
    for link in links:
        link["model_count"] = len(models_by_sample.get(link["sample_id"], set()))
    return links


def _sample_link(session: Session, run_id: str, sample_id: str, forecast_artifact_id: str | None) -> dict:
    sample_index = session.get(SampleIndex, sample_id)
    link = {
        "sample_id": sample_id,
        "run_id": run_id,
        "forecast_artifact_id": forecast_artifact_id,
        "forecast_artifact_ids": [],
        "model_count": 0,
    }
    if sample_index is None:
        return link

    link.update(
        {
            "sample_index": sample_index.sample_index,
            "context_start": sample_index.context_start,
            "context_end": sample_index.context_end,
            "horizon_start": sample_index.horizon_start,
            "horizon_end": sample_index.horizon_end,
        }
    )
    history_start_at, history_end_at, forecast_start_at, forecast_end_at = _sample_time_bounds(session, sample_index)
    link.update(
        {
            "history_start_at": history_start_at,
            "history_end_at": history_end_at,
            "forecast_start_at": forecast_start_at,
            "forecast_end_at": forecast_end_at,
        }
    )
    return link


def _sample_time_bounds(session: Session, sample_index: SampleIndex) -> tuple[str | None, str | None, str | None, str | None]:
    wanted = {
        sample_index.context_start,
        sample_index.context_end,
        sample_index.horizon_start,
        sample_index.horizon_end,
    }
    rows = session.exec(
        select(SeriesPoint).where(
            SeriesPoint.shard_id == sample_index.shard_id,
            SeriesPoint.row_index.in_(wanted),
        )
    ).all()
    by_index = {row.row_index: row.ts for row in rows}
    return (
        by_index.get(sample_index.context_start),
        by_index.get(sample_index.context_end),
        by_index.get(sample_index.horizon_start),
        by_index.get(sample_index.horizon_end),
    )
