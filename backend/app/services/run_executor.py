from pathlib import Path

from sqlmodel import Session, select

from app.core.errors import ApiError
from app.core.time import utc_now
from app.models.benchmark import BenchmarkingRun, CapabilityBlock, ForecastArtifact, RunEvent, Task, Unit
from app.models.dataset import Shard
from app.models.metric import MetricResult
from app.models.model_registry import Model
from app.models.sample import SampleIndex
from app.core.config import get_settings
from app.services.forecast_store import ForecastStore
from app.services.metric_service import aggregate_metric, compute_sample_metrics
from app.services.model_adapter import ModelAdapter, get_model_adapter, remote_model_id
from app.services.model_catalog import ensure_catalog_models_exist
from app.services.model_input import build_model_input
from app.services.resource_lifecycle import RESOURCE_TRACK, is_archived
from app.services.sample_store import SampleStore
from app.services.track_service import shards_for_capability_block

# 计算并入榜的指标集合：mase 为主排名，mse/mae 为诊断。
METRIC_NAMES = ["mase", "mse", "mae"]


def create_benchmarking_run(session: Session, track_id: str, model_ids: list[str]) -> BenchmarkingRun:
    if not model_ids:
        raise ApiError("run_requires_model", "benchmarking run requires at least one model")
    if is_archived(session, RESOURCE_TRACK, track_id):
        raise ApiError("resource_archived", "track is archived", {"track_id": track_id}, 409)
    ensure_catalog_models_exist(session, get_settings(), model_ids)
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


_TERMINAL_RUN_STATUSES = {"succeeded", "partial_succeeded", "failed", "cancelled"}


def cancel_run(session: Session, run_id: str) -> BenchmarkingRun:
    run = session.get(BenchmarkingRun, run_id)
    if run.status in _TERMINAL_RUN_STATUSES:
        raise ApiError(
            "run_in_terminal_state",
            f"run already finished with status '{run.status}'",
            {"run_id": run_id, "status": run.status},
            status_code=409,
        )
    if run.status == "cancel_requested":
        return run
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

    adapter = get_model_adapter(get_settings())
    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    for unit in units:
        _execute_unit(session, run, unit, Path(runtime_dir), adapter)

    statuses = [unit.status for unit in units]
    succeeded = len([status for status in statuses if status == "succeeded"])
    partial = len([status for status in statuses if status == "partial_succeeded"])
    if succeeded == len(statuses):
        terminal_status = "succeeded"
    elif succeeded or partial:
        # 既有成功（或部分成功），又有未完全成功的 unit → 部分成功。
        terminal_status = "partial_succeeded"
    else:
        terminal_status = "failed"

    from app.services.ranking_service import refresh_ranking
    from app.services.report_service import generate_run_report

    # 竞态防护：终态 status / report_id / 三张榜单必须在同一次提交里一起对外可见，
    # 否则轮询可能看到 run 已 succeeded、却查不到榜单。先在内存里置终态（refresh_ranking
    # 经 session 标识映射即可读到，用于筛选 valid rows），各榜单 commit=False 只挂起不提交，
    # 最后由 generate_run_report 的提交一次性落盘 status + report_id + 全部榜单。
    run.status = terminal_status
    run.finished_at = utc_now()
    run.updated_at = utc_now()
    session.add(RunEvent(benchmarking_run_id=run_id, message=f"run {terminal_status}"))
    session.add(run)
    for metric_id in METRIC_NAMES:
        refresh_ranking(session, run.track_id, metric_id, commit=False)
    generate_run_report(session, run_id, runtime_dir)  # 记录终态、置 report_id，并一次性提交（含挂起的榜单）
    session.refresh(run)
    return run


def _execute_unit(session: Session, run: BenchmarkingRun, unit: Unit, runtime_dir: Path, adapter: ModelAdapter) -> None:
    model = session.get(Model, unit.model_id)
    unit.status = "running"
    unit.started_at = utc_now()
    session.add(unit)
    session.commit()
    if model is None:
        _fail_unit(session, unit, "model_not_found", f"model not found: {unit.model_id}")
        return
    if model and model.endpoint_uri == "stub://fail":
        _fail_unit(session, unit, "adapter_error", "stub failure")
        return
    model_payload = {
        "model_id": model.model_id,
        "remote_model_id": remote_model_id(model),
        "stub_seed": model.stub_seed,
    }
    ensure_model_loaded = getattr(adapter, "ensure_model_loaded", None)
    if ensure_model_loaded is not None:
        try:
            ensure_model_loaded(
                model_payload,
                timeout_seconds=get_settings().timer_service_model_load_timeout_seconds,
            )
        except Exception as error:  # noqa: BLE001 — load failure is recorded on the unit, not raised out of the run
            _fail_unit(session, unit, "model_load_error", str(error))
            session.add(
                RunEvent(
                    benchmarking_run_id=run.benchmarking_run_id,
                    level="error",
                    event_type="model_load_failed",
                    message=f"model load failed for {model_payload['remote_model_id']}: {error}",
                )
            )
            session.commit()
            return

    tasks = session.exec(select(Task).where(Task.unit_id == unit.unit_id)).all()
    task_metrics: list[dict[str, float] | None] = []
    for task in tasks:
        task_metrics.append(_execute_task(session, run, unit, task, model, runtime_dir, adapter))
    for metric_name in METRIC_NAMES:
        aggregated = aggregate_metric(task_metrics, metric_name)
        if aggregated:
            session.add(_metric(metric_name, "unit", run, unit, None, model.model_id, aggregated["value"]))
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


def _execute_task(session: Session, run: BenchmarkingRun, unit: Unit, task: Task, model: Model, runtime_dir: Path, adapter: ModelAdapter) -> dict[str, float] | None:
    task.status = "running"
    task.started_at = utc_now()
    session.add(task)
    session.commit()
    block = session.get(CapabilityBlock, task.capability_block_id)
    shards = shards_for_capability_block(session, block.capability_block_id)
    shard_metrics: list[dict[str, float] | None] = []
    for shard in shards:
        shard_metrics.append(_execute_shard(session, run, unit, task, model, shard, runtime_dir, adapter))
    task_result: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        aggregated = aggregate_metric(shard_metrics, metric_name)
        if aggregated:
            session.add(_metric(metric_name, "task", run, unit, task, model.model_id, aggregated["value"], capability_block_id=block.capability_block_id))
            task_result[metric_name] = aggregated["value"]
    task.status = "succeeded" if all(metric is not None for metric in shard_metrics) else "partial_succeeded"
    task.finished_at = utc_now()
    session.add(task)
    session.commit()
    return task_result or None


def _execute_shard(session: Session, run: BenchmarkingRun, unit: Unit, task: Task, model: Model, shard: Shard, runtime_dir: Path, adapter: ModelAdapter) -> dict[str, float] | None:
    samples = session.exec(select(SampleIndex).where(SampleIndex.shard_id == shard.shard_id).order_by(SampleIndex.sample_index)).all()
    store = SampleStore()
    model_payload = {
        "model_id": model.model_id,
        "remote_model_id": remote_model_id(model),
        "stub_seed": model.stub_seed if model else 0,
    }
    timeout_seconds = get_settings().sample_forecast_timeout_seconds
    rows = []
    sample_metrics = []
    for sample_index in samples:
        sample = store.read_by_ref(session, sample_index.storage_ref)
        model_input = build_model_input(sample)
        try:
            forecast = adapter.forecast(model_input, model_payload, timeout_seconds=timeout_seconds)
        except Exception as error:  # noqa: BLE001 — adapter failure must not crash the run
            # 单样本预测失败：写一条 forecast 失败行、该样本计为失败，但不崩整 run。
            rows.append(
                {
                    "sample_id": sample_index.sample_id,
                    "unit_id": unit.unit_id,
                    "status": "failed",
                    "forecast": None,
                    "future_timestamps": sample["future_timestamps"],
                    "metrics": {},
                    "error_code": "adapter_error",
                    "error_message": str(error),
                }
            )
            sample_metrics.append(None)
            continue
        metrics = compute_sample_metrics(sample["target_future"], forecast, sample["target_history"])
        for metric_name, value in metrics.items():
            session.add(_metric(metric_name, "sample", run, unit, task, model.model_id, value, shard.shard_id, sample_index.sample_id, task.capability_block_id))
        rows.append(
            {
                "sample_id": sample_index.sample_id,
                "unit_id": unit.unit_id,
                "status": "succeeded",
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
    shard_result: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        aggregated = aggregate_metric(sample_metrics, metric_name)
        if aggregated:
            session.add(_metric(metric_name, "shard", run, unit, task, model.model_id, aggregated["value"], shard.shard_id, capability_block_id=task.capability_block_id))
            shard_result[metric_name] = aggregated["value"]
    session.commit()
    return shard_result or None


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


def _sample_counts(session: Session, run_id: str) -> tuple[int, int, dict[str, int]]:
    """Count per-sample forecast rows actually written for a run.

    Returns ``(completed, failed, completed_by_task)`` where completed = rows
    with status ``succeeded`` and failed = rows with status ``failed``, read
    back from the JSONL forecast artifacts produced during ``_execute_shard``.
    """
    from app.services.forecast_store import ForecastStore

    artifacts = session.exec(
        select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)
    ).all()
    completed = 0
    failed = 0
    completed_by_task: dict[str, int] = {}
    for artifact in artifacts:
        try:
            rows = ForecastStore(Path(artifact.storage_uri).parent).read_forecasts(artifact.storage_uri)
        except FileNotFoundError:
            continue
        artifact_completed = sum(1 for row in rows if row.get("status") == "succeeded")
        completed += artifact_completed
        failed += sum(1 for row in rows if row.get("status") == "failed")
        completed_by_task[artifact.task_id] = completed_by_task.get(artifact.task_id, 0) + artifact_completed
    return completed, failed, completed_by_task


def build_run_progress(session: Session, run_id: str) -> dict:
    run = session.get(BenchmarkingRun, run_id)
    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    tasks = session.exec(select(Task).where(Task.benchmarking_run_id == run_id)).all()
    events = session.exec(select(RunEvent).where(RunEvent.benchmarking_run_id == run_id).order_by(RunEvent.created_at.desc()).limit(20)).all()
    completed_samples, failed_samples, completed_by_task = _sample_counts(session, run_id)
    return {
        "benchmarking_run_id": run.benchmarking_run_id,
        "status": run.status,
        "progress": {
            "total_models": run.model_count,
            "completed_models": len([unit for unit in units if unit.status in {"succeeded", "failed", "partial_succeeded", "cancelled"}]),
            "total_tasks": run.task_count,
            "completed_tasks": len([task for task in tasks if task.status in {"succeeded", "failed", "partial_succeeded", "cancelled"}]),
            "total_samples": run.sample_count,
            "completed_samples": completed_samples,
            "failed_samples": failed_samples,
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
                "completed_sample_count": completed_by_task.get(task.task_id, 0),
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
