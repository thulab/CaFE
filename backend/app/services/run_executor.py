from pathlib import Path

from sqlmodel import Session, select

from app.core.errors import ApiError
from app.core.time import utc_now
from app.models.benchmark import BenchmarkingRun, CapabilityBlock, RunEvent, Task, Unit
from app.models.dataset import Shard
from app.models.metric import MetricResult
from app.models.model_registry import Model
from app.models.sample import SampleIndex
from app.services.forecast_store import ForecastStore
from app.services.metric_service import aggregate_metric, compute_sample_metrics
from app.services.sample_store import SampleStore
from app.services.stub_timer_adapter import StubTimerAdapter


def create_benchmarking_run(session: Session, track_id: str, model_ids: list[str]) -> BenchmarkingRun:
    if not model_ids:
        raise ApiError("run_requires_model", "benchmarking run requires at least one model")
    blocks = session.exec(select(CapabilityBlock).where(CapabilityBlock.track_id == track_id)).all()
    if not blocks:
        raise ApiError("track_has_no_blocks", "track has no capability blocks", {"track_id": track_id})

    block_sample_count = sum(block.sample_count for block in blocks)
    run = BenchmarkingRun(
        track_id=track_id,
        model_ids=model_ids,
        status="queued",
        model_count=len(model_ids),
        task_count=len(model_ids) * len(blocks),
        sample_count=len(model_ids) * block_sample_count,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    for model_id in model_ids:
        unit = Unit(
            benchmarking_run_id=run.benchmarking_run_id,
            model_id=model_id,
            task_count=len(blocks),
            sample_count=block_sample_count,
        )
        session.add(unit)
        session.commit()
        session.refresh(unit)
        for block in blocks:
            session.add(
                Task(
                    benchmarking_run_id=run.benchmarking_run_id,
                    unit_id=unit.unit_id,
                    model_id=model_id,
                    capability_block_id=block.capability_block_id,
                    shard_count=block.shard_count,
                    sample_count=block.sample_count,
                )
            )
    session.add(RunEvent(benchmarking_run_id=run.benchmarking_run_id, message="run queued"))
    session.commit()
    session.refresh(run)
    return run


def cancel_run(session: Session, run_id: str) -> BenchmarkingRun:
    run = session.get(BenchmarkingRun, run_id)
    run.cancel_requested = True
    run.cancel_requested_at = utc_now()
    run.status = "cancel_requested"
    session.add(RunEvent(benchmarking_run_id=run_id, level="warning", event_type="cancel_requested", message="cancel requested"))
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def recover_interrupted_runs(session: Session) -> None:
    unfinished = session.exec(
        select(BenchmarkingRun).where(BenchmarkingRun.status.in_(["queued", "running", "cancel_requested"]))
    ).all()
    for run in unfinished:
        run.status = "failed"
        run.finished_at = utc_now()
        session.add(RunEvent(benchmarking_run_id=run.benchmarking_run_id, level="error", event_type="interrupted_by_server_restart", message="run interrupted by server restart"))
        session.add(run)
    session.commit()


def execute_run(session: Session, run_id: str, runtime_dir: Path) -> BenchmarkingRun:
    run = session.get(BenchmarkingRun, run_id)
    if run.cancel_requested:
        run.status = "cancelled"
        run.finished_at = utc_now()
        session.add(RunEvent(benchmarking_run_id=run_id, level="warning", event_type="cancelled", message="run cancelled"))
        session.add(run)
        session.commit()
        session.refresh(run)
        return run

    run.status = "running"
    run.started_at = run.started_at or utc_now()
    session.add(RunEvent(benchmarking_run_id=run_id, message="run started"))
    session.add(run)
    session.commit()

    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    for unit in units:
        _execute_unit(session, run, unit, Path(runtime_dir))

    succeeded = len([unit for unit in units if unit.status == "succeeded"])
    failed = len([unit for unit in units if unit.status == "failed"])
    if succeeded and failed:
        run.status = "partial_succeeded"
    elif succeeded:
        run.status = "succeeded"
    else:
        run.status = "failed"
    run.finished_at = utc_now()
    run.updated_at = utc_now()
    session.add(RunEvent(benchmarking_run_id=run_id, message=f"run {run.status}"))
    session.add(run)
    from app.services.ranking_service import refresh_ranking
    from app.services.report_service import generate_run_report

    report = generate_run_report(session, run_id, runtime_dir)
    run.report_id = report.report_id
    if run.status != "cancelled":
        for metric_id in ["mse", "mae"]:
            refresh_ranking(session, run.track_id, metric_id)
    session.refresh(run)
    return run


def _execute_unit(session: Session, run: BenchmarkingRun, unit: Unit, runtime_dir: Path) -> None:
    model = session.get(Model, unit.model_id)
    unit.status = "running"
    unit.started_at = utc_now()
    session.add(unit)
    session.commit()
    if model and model.endpoint_uri == "stub://fail":
        _fail_unit(session, unit, "adapter_error", "stub failure")
        return

    tasks = session.exec(select(Task).where(Task.unit_id == unit.unit_id)).all()
    task_metrics: list[dict[str, float] | None] = []
    for task in tasks:
        task_metrics.append(_execute_task(session, run, unit, task, model, runtime_dir))
    mse = aggregate_metric(task_metrics, "mse")
    mae = aggregate_metric(task_metrics, "mae")
    if mse:
        session.add(_metric("mse", "unit", run, unit, None, model.model_id, mse["value"]))
    if mae:
        session.add(_metric("mae", "unit", run, unit, None, model.model_id, mae["value"]))
    unit.status = "succeeded" if all(metric is not None for metric in task_metrics) else "partial_succeeded"
    unit.finished_at = utc_now()
    session.add(unit)
    session.commit()


def _fail_unit(session: Session, unit: Unit, code: str, message: str) -> None:
    tasks = session.exec(select(Task).where(Task.unit_id == unit.unit_id)).all()
    for task in tasks:
        task.status = "failed"
        task.error_code = code
        task.error_message = message
        task.finished_at = utc_now()
        session.add(task)
    unit.status = "failed"
    unit.finished_at = utc_now()
    session.add(unit)
    session.commit()


def _execute_task(session: Session, run: BenchmarkingRun, unit: Unit, task: Task, model: Model, runtime_dir: Path) -> dict[str, float] | None:
    task.status = "running"
    task.started_at = utc_now()
    session.add(task)
    session.commit()
    block = session.get(CapabilityBlock, task.capability_block_id)
    shards = session.exec(select(Shard).where(Shard.capability_block_id == block.capability_block_id)).all()
    shard_metrics: list[dict[str, float] | None] = []
    for shard in shards:
        shard_metrics.append(_execute_shard(session, run, unit, task, model, shard, runtime_dir))
    mse = aggregate_metric(shard_metrics, "mse")
    mae = aggregate_metric(shard_metrics, "mae")
    if mse:
        session.add(_metric("mse", "task", run, unit, task, model.model_id, mse["value"], capability_block_id=block.capability_block_id))
    if mae:
        session.add(_metric("mae", "task", run, unit, task, model.model_id, mae["value"], capability_block_id=block.capability_block_id))
    task.status = "succeeded" if all(metric is not None for metric in shard_metrics) else "partial_succeeded"
    task.finished_at = utc_now()
    session.add(task)
    session.commit()
    return {"mse": mse["value"], "mae": mae["value"]} if mse and mae else None


def _execute_shard(session: Session, run: BenchmarkingRun, unit: Unit, task: Task, model: Model, shard: Shard, runtime_dir: Path) -> dict[str, float] | None:
    samples = session.exec(select(SampleIndex).where(SampleIndex.shard_id == shard.shard_id).order_by(SampleIndex.sample_index)).all()
    store = SampleStore(runtime_dir / "samples")
    adapter = StubTimerAdapter(seed=model.stub_seed if model else 0)
    rows = []
    sample_metrics = []
    for sample_index in samples:
        sample = store.read_by_ref(sample_index.materialized_sample_uri, sample_index.storage_ref)
        forecast = adapter.forecast(sample, {"model_id": model.model_id}, timeout_seconds=300)
        metrics = compute_sample_metrics(sample["target_future"], forecast)
        for metric_name, value in metrics.items():
            session.add(_metric(metric_name, "sample", run, unit, task, model.model_id, value, shard.shard_id, sample_index.sample_id, task.capability_block_id))
        rows.append(
            {
                "sample_id": sample_index.sample_id,
                "unit_id": unit.unit_id,
                "forecast": forecast,
                "future_timestamps": sample["future_timestamps"],
                "metrics": metrics,
            }
        )
        sample_metrics.append(metrics)
    artifact = ForecastStore(runtime_dir / "forecasts").write_forecasts(
        run.benchmarking_run_id,
        task.task_id,
        model.model_id,
        shard.shard_id,
        rows,
    )
    artifact.unit_id = unit.unit_id
    session.add(artifact)
    mse = aggregate_metric(sample_metrics, "mse")
    mae = aggregate_metric(sample_metrics, "mae")
    if mse:
        session.add(_metric("mse", "shard", run, unit, task, model.model_id, mse["value"], shard.shard_id, capability_block_id=task.capability_block_id))
    if mae:
        session.add(_metric("mae", "shard", run, unit, task, model.model_id, mae["value"], shard.shard_id, capability_block_id=task.capability_block_id))
    session.commit()
    return {"mse": mse["value"], "mae": mae["value"]} if mse and mae else None


def _metric(
    metric_name: str,
    level: str,
    run: BenchmarkingRun,
    unit: Unit,
    task: Task | None,
    model_id: str,
    value: float,
    shard_id: str | None = None,
    sample_id: str | None = None,
    capability_block_id: str | None = None,
) -> MetricResult:
    return MetricResult(
        metric_id=metric_name,
        result_level=level,
        benchmarking_run_id=run.benchmarking_run_id,
        unit_id=unit.unit_id,
        task_id=task.task_id if task else None,
        sample_id=sample_id,
        shard_id=shard_id,
        model_id=model_id,
        capability_block_id=capability_block_id,
        value=value,
        aggregation="raw" if level == "sample" else f"mean_over_{level}s",
    )


def build_run_progress(session: Session, run_id: str) -> dict:
    run = session.get(BenchmarkingRun, run_id)
    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    tasks = session.exec(select(Task).where(Task.benchmarking_run_id == run_id)).all()
    events = session.exec(select(RunEvent).where(RunEvent.benchmarking_run_id == run_id).order_by(RunEvent.created_at.desc()).limit(20)).all()
    return {
        "benchmarking_run_id": run.benchmarking_run_id,
        "status": run.status,
        "progress": {
            "total_models": run.model_count,
            "completed_models": len([unit for unit in units if unit.status in {"succeeded", "failed", "partial_succeeded", "cancelled"}]),
            "total_tasks": run.task_count,
            "completed_tasks": len([task for task in tasks if task.status in {"succeeded", "failed", "partial_succeeded", "cancelled"}]),
            "total_samples": run.sample_count,
            "completed_samples": 0,
            "failed_samples": 0,
        },
        "units": [
            {
                "unit_id": unit.unit_id,
                "model_id": unit.model_id,
                "model_name": (session.get(Model, unit.model_id).name if session.get(Model, unit.model_id) else unit.model_id),
                "status": unit.status,
                "task_count": unit.task_count,
                "completed_task_count": len([task for task in tasks if task.unit_id == unit.unit_id and task.status in {"succeeded", "failed", "partial_succeeded", "cancelled"}]),
                "metrics": {},
            }
            for unit in units
        ],
        "tasks": [
            {
                "task_id": task.task_id,
                "unit_id": task.unit_id,
                "model_id": task.model_id,
                "capability_block_id": task.capability_block_id,
                "capability_block_name": (session.get(CapabilityBlock, task.capability_block_id).name if session.get(CapabilityBlock, task.capability_block_id) else task.capability_block_id),
                "status": task.status,
                "shard_count": task.shard_count,
                "sample_count": task.sample_count,
                "completed_sample_count": 0,
                "metrics": {},
                "error_code": task.error_code,
                "error_message": task.error_message,
            }
            for task in tasks
        ],
        "recent_events": [
            {
                "level": event.level,
                "event_type": event.event_type,
                "message": event.message,
                "created_at": event.created_at.isoformat(),
            }
            for event in events
        ],
        "report_id": run.report_id,
        "ranking_list_id": run.ranking_list_id,
    }
