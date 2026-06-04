import json
from pathlib import Path

from sqlmodel import Session, select

from app.models.benchmark import BenchmarkingRun, ForecastArtifact, Task, Unit
from app.models.metric import MetricResult
from app.models.model_registry import Model
from app.models.report import Report
from app.models.sample import SampleIndex
from app.services.metric_service import _mase_scale
from app.services.sample_store import SampleStore


def generate_run_report(session: Session, run_id: str, runtime_dir: Path) -> Report:
    run = session.get(BenchmarkingRun, run_id)
    report_dir = Path(runtime_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"{run_id}.json"
    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    tasks = session.exec(select(Task).where(Task.benchmarking_run_id == run_id)).all()
    metrics = session.exec(select(MetricResult).where(MetricResult.benchmarking_run_id == run_id)).all()
    artifacts = session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)).all()

    payload = {
        "benchmarking_run_id": run_id,
        "track_id": run.track_id,
        "status": run.status,
        "model_metrics": [_unit_metrics(session, unit, metrics) for unit in units],
        "task_summaries": [_task_summary(task, metrics) for task in tasks],
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


def read_report(report: Report, sample_link_limit: int | None = None, sample_link_offset: int = 0) -> dict:
    payload = json.loads(Path(report.storage_uri).read_text(encoding="utf-8"))
    links = list(payload.get("sample_forecast_links") or [])
    total = len(links)
    offset = max(0, int(sample_link_offset or 0))
    limit = int(sample_link_limit) if sample_link_limit is not None else None
    if limit is not None:
        limit = max(0, limit)
        payload["sample_forecast_links"] = links[offset : offset + limit]
    payload["sample_forecast_links_total"] = total
    payload["sample_forecast_links_limit"] = limit if limit is not None else total
    payload["sample_forecast_links_offset"] = offset
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


def _task_summary(task: Task, metrics: list[MetricResult]) -> dict:
    return {
        "task_id": task.task_id,
        "unit_id": task.unit_id,
        "model_id": task.model_id,
        "capability_block_id": task.capability_block_id,
        "status": task.status,
        "error_code": task.error_code,
        "error_message": task.error_message,
        "metrics": {
            metric.metric_id: metric.value
            for metric in metrics
            if metric.result_level == "task" and metric.task_id == task.task_id
        },
    }


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


def _sample_link(session: Session, run_id: str, sample_id: str, forecast_artifact_id: str) -> dict:
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
    try:
        sample = SampleStore().read_by_ref(session, sample_index.storage_ref)
    except Exception:  # noqa: BLE001 — report generation must tolerate missing sample detail metadata
        return link
    history_timestamps = sample.get("history_timestamps") or []
    future_timestamps = sample.get("future_timestamps") or []
    link.update(
        {
            "history_start_at": history_timestamps[0] if history_timestamps else None,
            "history_end_at": history_timestamps[-1] if history_timestamps else None,
            "forecast_start_at": future_timestamps[0] if future_timestamps else None,
            "forecast_end_at": future_timestamps[-1] if future_timestamps else None,
        }
    )
    return link
