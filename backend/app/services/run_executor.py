from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _PreparedSample:
    sample_id: str
    sample_index: int
    sample: dict
    model_input: dict


@dataclass(frozen=True)
class _ForecastOutcome:
    forecast: list[list[float]] | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _ShardExecutionResult:
    metrics: dict[str, float] | None
    processed_count: int
    failed_count: int


@dataclass(frozen=True)
class _TaskExecutionResult:
    metrics: dict[str, float] | None
    failed_count: int


def create_benchmarking_run(session: Session, track_id: str, model_ids: list[str]) -> BenchmarkingRun:
    if not model_ids:
        raise ApiError("run_requires_model", "benchmarking run requires at least one model")
    if is_archived(session, RESOURCE_TRACK, track_id):
        raise ApiError("resource_archived", "track is archived", {"track_id": track_id}, 409)
    ensure_catalog_models_exist(session, get_settings(), model_ids)
    blocks = session.exec(select(CapabilityBlock).where(CapabilityBlock.track_id == track_id)).all()
    if not blocks:
        raise ApiError("track_has_no_blocks", "track has no capability blocks", {"track_id": track_id})
    target_dim = max((block.target_dim for block in blocks), default=1)
    covariate_dim = max((block.covariate_dim for block in blocks), default=0)
    _validate_models_support_target_dim(session, model_ids, target_dim)
    _validate_models_support_covariate_dim(session, model_ids, covariate_dim)

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


def _validate_models_support_target_dim(session: Session, model_ids: list[str], target_dim: int) -> None:
    if target_dim <= 1:
        return
    unsupported: list[str] = []
    limits_by_model: dict[str, dict] = {}
    for model_id in model_ids:
        model = session.get(Model, model_id)
        if model is None:
            continue
        limits = model.forecast_limits if isinstance(model.forecast_limits, dict) else {}
        limits_by_model[model_id] = limits
        if not _forecast_limits_support_target_dim(limits, target_dim):
            unsupported.append(model_id)
    if unsupported:
        raise ApiError(
            "model_target_dim_unsupported",
            "selected model does not support the track target dimension",
            {"model_ids": unsupported, "target_dim": target_dim, "forecast_limits": limits_by_model},
            status_code=400,
        )


def _forecast_limits_support_target_dim(limits: dict, target_dim: int) -> bool:
    if "max_target_count" not in limits:
        return False
    max_target_count = limits.get("max_target_count")
    if max_target_count is None:
        return True
    try:
        return int(max_target_count) >= target_dim
    except (TypeError, ValueError):
        return False


def _validate_models_support_covariate_dim(session: Session, model_ids: list[str], covariate_dim: int) -> None:
    if covariate_dim <= 0:
        return
    unsupported: list[str] = []
    limits_by_model: dict[str, dict] = {}
    for model_id in model_ids:
        model = session.get(Model, model_id)
        if model is None:
            continue
        limits = model.forecast_limits if isinstance(model.forecast_limits, dict) else {}
        limits_by_model[model_id] = limits
        if not _forecast_limits_support_covariate_dim(limits, covariate_dim):
            unsupported.append(model_id)
    if unsupported:
        raise ApiError(
            "model_covariate_dim_unsupported",
            "selected model does not support the track covariate dimension",
            {"model_ids": unsupported, "covariate_dim": covariate_dim, "forecast_limits": limits_by_model},
            status_code=400,
        )


def _forecast_limits_support_covariate_dim(limits: dict, covariate_dim: int) -> bool:
    if "max_covariate_count" not in limits:
        return False
    try:
        return int(limits.get("max_covariate_count")) >= covariate_dim
    except (TypeError, ValueError):
        return False


_TERMINAL_RUN_STATUSES = {"succeeded", "partial_succeeded", "failed", "cancelled"}


class _RunCancelled(Exception):
    pass


def cancel_run(session: Session, run_id: str) -> BenchmarkingRun:
    run = session.get(BenchmarkingRun, run_id)
    if run.status == "cancelled":
        return run
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
    session.add(RunEvent(benchmarking_run_id=run_id, level="warning", event_type="cancel_requested", message="cancel requested"))
    if run.status in {"created", "queued"}:
        run.status = "cancelled"
        run.finished_at = utc_now()
        session.add(RunEvent(benchmarking_run_id=run_id, level="warning", event_type="cancelled", message="run cancelled"))
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    run.status = "cancel_requested"
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
    if run.status in _TERMINAL_RUN_STATUSES:
        return run
    if run.cancel_requested:
        return _finish_cancelled_run(session, run_id)

    try:
        run.status = "running"
        run.started_at = run.started_at or utc_now()
        session.add(RunEvent(benchmarking_run_id=run_id, message="run started"))
        session.add(run)
        session.commit()

        settings = get_settings()
        adapter = get_model_adapter(settings)
        _raise_if_cancel_requested(session, run)
        if _uses_sequential_model_lifecycle(settings):
            _unload_all_models_before_run(session, run, adapter, settings.timer_service_model_load_timeout_seconds)
            _raise_if_cancel_requested(session, run)
        units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
        for unit in units:
            _raise_if_cancel_requested(session, run)
            _execute_unit(session, run, unit, Path(runtime_dir), adapter, settings)
            _raise_if_cancel_requested(session, run)

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

        _raise_if_cancel_requested(session, run)
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
    except _RunCancelled:
        return _finish_cancelled_run(session, run_id)


def _uses_sequential_model_lifecycle(settings) -> bool:
    return settings.model_lifecycle_mode != "keep_loaded"


def _raise_if_cancel_requested(session: Session, run: BenchmarkingRun) -> None:
    session.refresh(run)
    if run.cancel_requested or run.status == "cancel_requested":
        raise _RunCancelled()


def _finish_cancelled_run(session: Session, run_id: str) -> BenchmarkingRun:
    run = session.get(BenchmarkingRun, run_id)
    if run.status == "cancelled":
        return run
    for task in session.exec(select(Task).where(Task.benchmarking_run_id == run_id)).all():
        if task.status not in _TERMINAL_RUN_STATUSES:
            task.status = "cancelled"
            task.finished_at = utc_now()
            session.add(task)
    for unit in session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all():
        if unit.status not in _TERMINAL_RUN_STATUSES:
            unit.status = "cancelled"
            unit.finished_at = utc_now()
            session.add(unit)
    run.cancel_requested = True
    run.cancel_requested_at = run.cancel_requested_at or utc_now()
    run.status = "cancelled"
    run.finished_at = utc_now()
    run.updated_at = utc_now()
    session.add(RunEvent(benchmarking_run_id=run_id, level="warning", event_type="cancelled", message="run cancelled"))
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def _unload_all_models_before_run(session: Session, run: BenchmarkingRun, adapter: ModelAdapter, timeout_seconds: int) -> None:
    unload_all_models = getattr(adapter, "unload_all_models", None)
    if unload_all_models is None:
        return
    _add_run_event(
        session,
        run.benchmarking_run_id,
        event_type="model_unload_all_started",
        message="unloading loaded models before run",
    )
    try:
        unload_all_models(timeout_seconds=timeout_seconds)
    except Exception as error:  # noqa: BLE001 — best-effort cleanup must not prevent the run from loading its first model
        _add_run_event(
            session,
            run.benchmarking_run_id,
            level="warning",
            event_type="model_unload_all_failed",
            message=f"failed to unload loaded models before run: {error}",
        )
        return
    _add_run_event(
        session,
        run.benchmarking_run_id,
        event_type="model_unload_all_finished",
        message="loaded models unloaded before run",
    )


def _execute_unit(session: Session, run: BenchmarkingRun, unit: Unit, runtime_dir: Path, adapter: ModelAdapter, settings) -> None:
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
    try:
        _raise_if_cancel_requested(session, run)
        ensure_model_loaded = getattr(adapter, "ensure_model_loaded", None)
        if ensure_model_loaded is not None:
            _add_run_event(
                session,
                run.benchmarking_run_id,
                unit_id=unit.unit_id,
                event_type="model_load_started",
                message=f"loading model {model_payload['remote_model_id']}",
                payload={"model_id": model.model_id, "remote_model_id": model_payload["remote_model_id"]},
            )
            try:
                ensure_model_loaded(
                    model_payload,
                    timeout_seconds=settings.timer_service_model_load_timeout_seconds,
                )
            except Exception as error:  # noqa: BLE001 — load failure is recorded on the unit, not raised out of the run
                _fail_unit(session, unit, "model_load_error", str(error))
                _add_run_event(
                    session,
                    run.benchmarking_run_id,
                    unit_id=unit.unit_id,
                    level="error",
                    event_type="model_load_failed",
                    message=f"model load failed for {model_payload['remote_model_id']}: {error}",
                    payload={"model_id": model.model_id, "remote_model_id": model_payload["remote_model_id"]},
                )
                return
            _add_run_event(
                session,
                run.benchmarking_run_id,
                unit_id=unit.unit_id,
                event_type="model_loaded",
                message=f"model {model_payload['remote_model_id']} loaded",
                payload={"model_id": model.model_id, "remote_model_id": model_payload["remote_model_id"]},
            )
            _raise_if_cancel_requested(session, run)

        tasks = session.exec(select(Task).where(Task.unit_id == unit.unit_id)).all()
        task_results: list[_TaskExecutionResult] = []
        for task in tasks:
            _raise_if_cancel_requested(session, run)
            task_results.append(_execute_task(session, run, unit, task, model, runtime_dir, adapter))
            _raise_if_cancel_requested(session, run)
        for metric_name in METRIC_NAMES:
            aggregated = aggregate_metric([result.metrics for result in task_results], metric_name)
            if aggregated:
                session.add(_metric(metric_name, "unit", run, unit, None, model.model_id, aggregated["value"]))
        unit.status = "succeeded" if all(result.metrics is not None and result.failed_count == 0 for result in task_results) else "partial_succeeded"
        unit.finished_at = utc_now()
        session.add(unit)
        session.commit()
    finally:
        if _uses_sequential_model_lifecycle(settings):
            _unload_model_after_unit(session, run, unit, model_payload, adapter, settings.timer_service_model_load_timeout_seconds)


def _unload_model_after_unit(
    session: Session,
    run: BenchmarkingRun,
    unit: Unit,
    model_payload: dict,
    adapter: ModelAdapter,
    timeout_seconds: int,
) -> None:
    unload_model = getattr(adapter, "unload_model", None)
    if unload_model is None:
        return
    remote_id = model_payload["remote_model_id"]
    _add_run_event(
        session,
        run.benchmarking_run_id,
        unit_id=unit.unit_id,
        event_type="model_unload_started",
        message=f"unloading model {remote_id}",
        payload={"model_id": model_payload["model_id"], "remote_model_id": remote_id},
    )
    try:
        unload_model(model_payload, timeout_seconds=timeout_seconds)
    except Exception as error:  # noqa: BLE001 — evaluation results are already persisted; record cleanup failure only
        _add_run_event(
            session,
            run.benchmarking_run_id,
            unit_id=unit.unit_id,
            level="warning",
            event_type="model_unload_failed",
            message=f"failed to unload model {remote_id}: {error}",
            payload={"model_id": model_payload["model_id"], "remote_model_id": remote_id},
        )
        return
    _add_run_event(
        session,
        run.benchmarking_run_id,
        unit_id=unit.unit_id,
        event_type="model_unloaded",
        message=f"model {remote_id} unloaded",
        payload={"model_id": model_payload["model_id"], "remote_model_id": remote_id},
    )


def _add_run_event(
    session: Session,
    run_id: str,
    message: str,
    event_type: str,
    level: str = "info",
    unit_id: str | None = None,
    task_id: str | None = None,
    payload: dict | None = None,
) -> None:
    session.add(
        RunEvent(
            benchmarking_run_id=run_id,
            unit_id=unit_id,
            task_id=task_id,
            level=level,
            event_type=event_type,
            message=message,
            payload=payload or {},
        )
    )
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


def _execute_task(session: Session, run: BenchmarkingRun, unit: Unit, task: Task, model: Model, runtime_dir: Path, adapter: ModelAdapter) -> _TaskExecutionResult:
    task.status = "running"
    task.started_at = utc_now()
    task.processed_sample_count = 0
    task.failed_sample_count = 0
    session.add(task)
    session.commit()
    _raise_if_cancel_requested(session, run)
    block = session.get(CapabilityBlock, task.capability_block_id)
    shards = shards_for_capability_block(session, block.capability_block_id)
    shard_results: list[_ShardExecutionResult] = []
    for shard in shards:
        _raise_if_cancel_requested(session, run)
        shard_results.append(_execute_shard(session, run, unit, task, model, shard, runtime_dir, adapter))
        _raise_if_cancel_requested(session, run)
    task_result: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        aggregated = aggregate_metric([result.metrics for result in shard_results], metric_name)
        if aggregated:
            session.add(_metric(metric_name, "task", run, unit, task, model.model_id, aggregated["value"], capability_block_id=block.capability_block_id))
            task_result[metric_name] = aggregated["value"]
    failed_count = sum(result.failed_count for result in shard_results)
    task.status = "succeeded" if all(result.metrics is not None and result.failed_count == 0 for result in shard_results) else "partial_succeeded"
    task.finished_at = utc_now()
    session.add(task)
    session.commit()
    return _TaskExecutionResult(task_result or None, failed_count)


def _execute_shard(session: Session, run: BenchmarkingRun, unit: Unit, task: Task, model: Model, shard: Shard, runtime_dir: Path, adapter: ModelAdapter) -> _ShardExecutionResult:
    _raise_if_cancel_requested(session, run)
    samples = session.exec(select(SampleIndex).where(SampleIndex.shard_id == shard.shard_id).order_by(SampleIndex.sample_index)).all()
    store = SampleStore()
    model_payload = {
        "model_id": model.model_id,
        "remote_model_id": remote_model_id(model),
        "stub_seed": model.stub_seed if model else 0,
    }
    settings = get_settings()
    timeout_seconds = settings.sample_forecast_timeout_seconds
    parallelism = max(1, int(settings.run_sample_parallelism or 1))
    progress_interval = max(1, int(settings.run_progress_update_interval_samples or 1))
    prepared_samples: list[_PreparedSample] = []
    for sample_index in samples:
        with session.no_autoflush:
            sample = store.read_by_ref(session, sample_index.storage_ref)
        prepared_samples.append(
            _PreparedSample(
                sample_id=sample_index.sample_id,
                sample_index=sample_index.sample_index,
                sample=sample,
                model_input=build_model_input(sample),
            )
        )

    rows_with_order: list[tuple[int, dict]] = []
    sample_metrics: list[dict[str, float] | None] = []
    metric_rows: list[MetricResult] = []
    base_processed_count = int(task.processed_sample_count or 0)
    base_failed_count = int(task.failed_sample_count or 0)
    processed_count = 0
    failed_count = 0

    def handle_outcome(prepared: _PreparedSample, outcome: _ForecastOutcome) -> None:
        nonlocal processed_count, failed_count
        sample = prepared.sample
        processed_count += 1
        if outcome.error_code:
            failed_count += 1
            rows_with_order.append((prepared.sample_index, _failed_forecast_row(unit, prepared, sample, outcome.error_code, outcome.error_message or "")))
            sample_metrics.append(None)
            return

        try:
            metrics = compute_sample_metrics(sample["target_future"], outcome.forecast or [], sample["target_history"])
        except Exception as error:  # noqa: BLE001 — bad output for one sample must not stop later samples
            failed_count += 1
            rows_with_order.append((prepared.sample_index, _failed_forecast_row(unit, prepared, sample, "metric_error", str(error))))
            sample_metrics.append(None)
            return

        for metric_name, value in metrics.items():
            metric_rows.append(_metric(metric_name, "sample", run, unit, task, model.model_id, value, shard.shard_id, prepared.sample_id, task.capability_block_id))
        rows_with_order.append(
            (
                prepared.sample_index,
                {
                    "sample_id": prepared.sample_id,
                    "unit_id": unit.unit_id,
                    "status": "succeeded",
                    "forecast": outcome.forecast,
                    "future_timestamps": sample["future_timestamps"],
                    "metrics": metrics,
                },
            )
        )
        sample_metrics.append(metrics)

    if parallelism == 1 or len(prepared_samples) <= 1:
        for prepared in prepared_samples:
            _raise_if_cancel_requested(session, run)
            handle_outcome(prepared, _forecast_prepared_sample(adapter, prepared, model_payload, timeout_seconds))
            if processed_count % progress_interval == 0:
                _record_task_sample_progress(session, task, base_processed_count + processed_count, base_failed_count + failed_count)
            _raise_if_cancel_requested(session, run)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            sample_iter = iter(prepared_samples)
            futures = {}

            def submit_next() -> bool:
                _raise_if_cancel_requested(session, run)
                try:
                    prepared = next(sample_iter)
                except StopIteration:
                    return False
                futures[executor.submit(_forecast_prepared_sample, adapter, prepared, model_payload, timeout_seconds)] = prepared
                return True

            try:
                for _ in range(min(parallelism, len(prepared_samples))):
                    submit_next()
                while futures:
                    future = next(as_completed(list(futures)))
                    prepared = futures.pop(future)
                    handle_outcome(prepared, future.result())
                    if processed_count % progress_interval == 0:
                        _record_task_sample_progress(session, task, base_processed_count + processed_count, base_failed_count + failed_count)
                    _raise_if_cancel_requested(session, run)
                    submit_next()
            except _RunCancelled:
                for future in futures:
                    future.cancel()
                raise
    _record_task_sample_progress(session, task, base_processed_count + processed_count, base_failed_count + failed_count)

    session.add_all(metric_rows)
    rows = [row for _sample_order, row in sorted(rows_with_order, key=lambda item: item[0])]
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
    return _ShardExecutionResult(shard_result or None, processed_count, failed_count)


def _forecast_prepared_sample(adapter: ModelAdapter, prepared: _PreparedSample, model_payload: dict, timeout_seconds: int) -> _ForecastOutcome:
    try:
        return _ForecastOutcome(forecast=adapter.forecast(prepared.model_input, model_payload, timeout_seconds=timeout_seconds))
    except Exception as error:  # noqa: BLE001 — adapter failure must not crash the run
        return _ForecastOutcome(error_code="adapter_error", error_message=str(error))


def _failed_forecast_row(unit: Unit, prepared: _PreparedSample, sample: dict, error_code: str, error_message: str) -> dict:
    return {
        "sample_id": prepared.sample_id,
        "unit_id": unit.unit_id,
        "status": "failed",
        "forecast": None,
        "future_timestamps": sample["future_timestamps"],
        "metrics": {},
        "error_code": error_code,
        "error_message": error_message,
    }


def _record_task_sample_progress(session: Session, task: Task, processed_sample_count: int, failed_sample_count: int) -> None:
    task.processed_sample_count = processed_sample_count
    task.failed_sample_count = failed_sample_count
    task.updated_at = utc_now()
    session.add(task)
    session.commit()


def list_failed_samples(session: Session, run_id: str) -> dict:
    run = session.get(BenchmarkingRun, run_id)
    if run is None:
        raise ApiError("resource_not_found", "resource not found", {"resource_type": "benchmarking_run", "resource_id": run_id}, 404)
    items: list[dict] = []
    for artifact in session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)).all():
        store = ForecastStore(Path(artifact.storage_uri).parent)
        try:
            rows = store.read_forecasts(artifact.storage_uri)
        except FileNotFoundError:
            continue
        unit = session.get(Unit, artifact.unit_id)
        task = session.get(Task, artifact.task_id)
        model = session.get(Model, artifact.model_id)
        block = session.get(CapabilityBlock, task.capability_block_id) if task else None
        for row in rows:
            if row.get("status") != "failed":
                continue
            sample = session.get(SampleIndex, row.get("sample_id"))
            items.append(
                {
                    "forecast_artifact_id": artifact.forecast_artifact_id,
                    "sample_id": row.get("sample_id"),
                    "sample_index": sample.sample_index if sample else None,
                    "model_id": artifact.model_id,
                    "model_name": model.name if model else artifact.model_id,
                    "unit_id": artifact.unit_id,
                    "task_id": artifact.task_id,
                    "capability_block_id": task.capability_block_id if task else None,
                    "capability_block_name": block.name if block else None,
                    "shard_id": artifact.shard_id,
                    "error_code": row.get("error_code"),
                    "error_message": row.get("error_message"),
                    "unit_status": unit.status if unit else None,
                    "task_status": task.status if task else None,
                }
            )
    items.sort(key=lambda item: (item.get("model_name") or "", item.get("sample_index") is None, item.get("sample_index") or 0, item.get("sample_id") or ""))
    return {"items": items, "total": len(items)}


def rerun_failed_samples(session: Session, run_id: str, runtime_dir: Path) -> dict:
    run = session.get(BenchmarkingRun, run_id)
    if run is None:
        raise ApiError("resource_not_found", "resource not found", {"resource_type": "benchmarking_run", "resource_id": run_id}, 404)
    if run.status not in _TERMINAL_RUN_STATUSES or run.status == "cancelled":
        raise ApiError("run_not_terminal", "run must be terminal before rerunning failed samples", {"run_id": run_id, "status": run.status}, 409)

    artifacts = session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)).all()
    artifacts_with_failures: list[tuple[ForecastArtifact, list[dict]]] = []
    for artifact in artifacts:
        store = ForecastStore(Path(artifact.storage_uri).parent)
        try:
            rows = store.read_forecasts(artifact.storage_uri)
        except FileNotFoundError:
            continue
        if any(row.get("status") == "failed" for row in rows):
            artifacts_with_failures.append((artifact, rows))
    if not artifacts_with_failures:
        return {"rerun_samples": 0, "remaining_failed_samples": 0}

    settings = get_settings()
    adapter = get_model_adapter(settings)
    touched_unit_ids: set[str] = set()
    rerun_count = 0
    for artifact, rows in artifacts_with_failures:
        unit = session.get(Unit, artifact.unit_id)
        model = session.get(Model, artifact.model_id)
        model_payload = {
            "model_id": artifact.model_id,
            "remote_model_id": remote_model_id(model) if model else artifact.model_id,
            "stub_seed": model.stub_seed if model else 0,
        }
        load_error: Exception | None = None
        ensure_model_loaded = getattr(adapter, "ensure_model_loaded", None)
        if model is None:
            load_error = RuntimeError(f"model not found: {artifact.model_id}")
        elif ensure_model_loaded is not None:
            try:
                ensure_model_loaded(model_payload, timeout_seconds=settings.timer_service_model_load_timeout_seconds)
            except Exception as error:  # noqa: BLE001 - rerun records failures per sample instead of aborting the whole request
                load_error = error
        updated_rows: list[dict] = []
        for row in rows:
            if row.get("status") != "failed":
                updated_rows.append(row)
                continue
            rerun_count += 1
            if load_error is not None:
                updated_rows.append(_failed_record_from_existing(row, "model_load_error", str(load_error)))
                continue
            try:
                prepared = _prepare_sample_by_id(session, str(row.get("sample_id")))
                outcome = _forecast_prepared_sample(adapter, prepared, model_payload, settings.sample_forecast_timeout_seconds)
                updated_rows.append(_record_from_rerun_outcome(run, unit, artifact, prepared, outcome))
            except Exception as error:  # noqa: BLE001 - one corrupt sample should not block other failed samples
                updated_rows.append(_failed_record_from_existing(row, "rerun_error", str(error)))
        store = ForecastStore(Path(artifact.storage_uri).parent)
        artifact.sample_count, artifact.checksum = store.overwrite_forecasts(artifact.storage_uri, updated_rows)
        session.add(artifact)
        touched_unit_ids.add(artifact.unit_id)
        if unit is not None and _uses_sequential_model_lifecycle(settings):
            _unload_model_after_unit(session, run, unit, model_payload, adapter, settings.timer_service_model_load_timeout_seconds)
    if touched_unit_ids:
        _rebuild_metrics_for_units(session, run, touched_unit_ids)
        _finalize_run_after_rerun(session, run, runtime_dir)
    remaining_failed = list_failed_samples(session, run_id)["total"]
    return {"rerun_samples": rerun_count, "remaining_failed_samples": remaining_failed}


def _prepare_sample_by_id(session: Session, sample_id: str) -> _PreparedSample:
    sample_index = session.get(SampleIndex, sample_id)
    if sample_index is None:
        raise ApiError("sample_not_found", "sample not found", {"sample_id": sample_id}, 404)
    sample = SampleStore().read_by_ref(session, sample_index.storage_ref)
    return _PreparedSample(
        sample_id=sample_index.sample_id,
        sample_index=sample_index.sample_index,
        sample=sample,
        model_input=build_model_input(sample),
    )


def _record_from_rerun_outcome(
    run: BenchmarkingRun,
    unit: Unit | None,
    artifact: ForecastArtifact,
    prepared: _PreparedSample,
    outcome: _ForecastOutcome,
) -> dict:
    if outcome.error_code:
        return _failed_forecast_record(run, unit, artifact, prepared, outcome.error_code, outcome.error_message or "")
    try:
        metrics = compute_sample_metrics(prepared.sample["target_future"], outcome.forecast or [], prepared.sample["target_history"])
    except Exception as error:  # noqa: BLE001
        return _failed_forecast_record(run, unit, artifact, prepared, "metric_error", str(error))
    return {
        "schema_version": "forecast.v1",
        "benchmarking_run_id": run.benchmarking_run_id,
        "unit_id": unit.unit_id if unit else artifact.unit_id,
        "task_id": artifact.task_id,
        "model_id": artifact.model_id,
        "shard_id": artifact.shard_id,
        "sample_id": prepared.sample_id,
        "status": "succeeded",
        "forecast": outcome.forecast,
        "future_timestamps": prepared.sample["future_timestamps"],
        "metrics": metrics,
        "error_code": None,
        "error_message": None,
    }


def _failed_forecast_record(
    run: BenchmarkingRun,
    unit: Unit | None,
    artifact: ForecastArtifact,
    prepared: _PreparedSample,
    error_code: str,
    error_message: str,
) -> dict:
    return {
        "schema_version": "forecast.v1",
        "benchmarking_run_id": run.benchmarking_run_id,
        "unit_id": unit.unit_id if unit else artifact.unit_id,
        "task_id": artifact.task_id,
        "model_id": artifact.model_id,
        "shard_id": artifact.shard_id,
        "sample_id": prepared.sample_id,
        "status": "failed",
        "forecast": None,
        "future_timestamps": prepared.sample["future_timestamps"],
        "metrics": {},
        "error_code": error_code,
        "error_message": error_message,
    }


def _failed_record_from_existing(row: dict, error_code: str, error_message: str) -> dict:
    return {
        **row,
        "status": "failed",
        "forecast": None,
        "metrics": {},
        "error_code": error_code,
        "error_message": error_message,
    }


def _rebuild_metrics_for_units(session: Session, run: BenchmarkingRun, unit_ids: set[str]) -> None:
    existing = session.exec(
        select(MetricResult).where(
            MetricResult.benchmarking_run_id == run.benchmarking_run_id,
            MetricResult.unit_id.in_(unit_ids),
        )
    ).all()
    for metric in existing:
        session.delete(metric)
    session.flush()
    for unit_id in unit_ids:
        unit = session.get(Unit, unit_id)
        if unit is None:
            continue
        task_results: list[_TaskExecutionResult] = []
        tasks = session.exec(select(Task).where(Task.unit_id == unit_id)).all()
        for task in tasks:
            task_results.append(_rebuild_task_metrics_from_artifacts(session, run, unit, task))
        for metric_name in METRIC_NAMES:
            aggregated = aggregate_metric([result.metrics for result in task_results], metric_name)
            if aggregated:
                session.add(_metric(metric_name, "unit", run, unit, None, unit.model_id, aggregated["value"]))
        if all(result.metrics is not None and result.failed_count == 0 for result in task_results):
            unit.status = "succeeded"
        elif any(result.metrics is not None or result.failed_count > 0 for result in task_results):
            unit.status = "partial_succeeded"
        else:
            unit.status = "failed"
        unit.finished_at = utc_now()
        unit.updated_at = utc_now()
        session.add(unit)
    session.commit()


def _rebuild_task_metrics_from_artifacts(session: Session, run: BenchmarkingRun, unit: Unit, task: Task) -> _TaskExecutionResult:
    artifacts = session.exec(select(ForecastArtifact).where(ForecastArtifact.task_id == task.task_id)).all()
    shard_results: list[_ShardExecutionResult] = []
    processed_total = 0
    failed_total = 0
    for artifact in artifacts:
        rows = ForecastStore(Path(artifact.storage_uri).parent).read_forecasts(artifact.storage_uri)
        sample_metrics: list[dict[str, float] | None] = []
        processed = 0
        failed = 0
        for row in rows:
            if row.get("status") == "succeeded":
                processed += 1
                metrics = {key: float(value) for key, value in (row.get("metrics") or {}).items()}
                sample_metrics.append(metrics)
                for metric_name, value in metrics.items():
                    session.add(_metric(metric_name, "sample", run, unit, task, artifact.model_id, value, artifact.shard_id, row.get("sample_id"), task.capability_block_id))
            elif row.get("status") == "failed":
                processed += 1
                failed += 1
                sample_metrics.append(None)
        shard_result: dict[str, float] = {}
        for metric_name in METRIC_NAMES:
            aggregated = aggregate_metric(sample_metrics, metric_name)
            if aggregated:
                session.add(_metric(metric_name, "shard", run, unit, task, artifact.model_id, aggregated["value"], artifact.shard_id, capability_block_id=task.capability_block_id))
                shard_result[metric_name] = aggregated["value"]
        shard_results.append(_ShardExecutionResult(shard_result or None, processed, failed))
        processed_total += processed
        failed_total += failed
    task_result: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        aggregated = aggregate_metric([result.metrics for result in shard_results], metric_name)
        if aggregated:
            session.add(_metric(metric_name, "task", run, unit, task, task.model_id, aggregated["value"], capability_block_id=task.capability_block_id))
            task_result[metric_name] = aggregated["value"]
    task.processed_sample_count = processed_total
    task.failed_sample_count = failed_total
    if all(result.metrics is not None and result.failed_count == 0 for result in shard_results):
        task.status = "succeeded"
        task.error_code = None
        task.error_message = None
    elif processed_total > 0:
        task.status = "partial_succeeded"
    else:
        task.status = "failed"
    task.finished_at = utc_now()
    task.updated_at = utc_now()
    session.add(task)
    return _TaskExecutionResult(task_result or None, failed_total)


def _finalize_run_after_rerun(session: Session, run: BenchmarkingRun, runtime_dir: Path) -> None:
    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run.benchmarking_run_id)).all()
    statuses = [unit.status for unit in units]
    succeeded = len([status for status in statuses if status == "succeeded"])
    partial = len([status for status in statuses if status == "partial_succeeded"])
    if succeeded == len(statuses):
        terminal_status = "succeeded"
    elif succeeded or partial:
        terminal_status = "partial_succeeded"
    else:
        terminal_status = "failed"
    run.status = terminal_status
    run.finished_at = utc_now()
    run.updated_at = utc_now()
    session.add(RunEvent(benchmarking_run_id=run.benchmarking_run_id, message=f"failed samples rerun; run {terminal_status}"))
    session.add(run)
    from app.services.ranking_service import refresh_ranking
    from app.services.report_service import generate_run_report

    for metric_id in METRIC_NAMES:
        refresh_ranking(session, run.track_id, metric_id, commit=False)
    generate_run_report(session, run.benchmarking_run_id, runtime_dir)


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


def _sample_counts(session: Session, run_id: str) -> tuple[int, int, int, dict[str, int], dict[str, int], dict[str, int]]:
    """Count per-sample forecast rows actually written for a run.

    Returns ``(completed, failed, processed, completed_by_task, failed_by_task, processed_by_task)``.
    During a running task, task counters are visible before the forecast JSONL
    artifact is written; after artifact write, JSONL remains the source of
    truth and counters are only used when they are ahead.
    """
    from app.services.forecast_store import ForecastStore

    tasks = session.exec(select(Task).where(Task.benchmarking_run_id == run_id)).all()
    artifacts = session.exec(
        select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)
    ).all()
    succeeded_by_task: dict[str, int] = {task.task_id: 0 for task in tasks}
    failed_by_task: dict[str, int] = {task.task_id: 0 for task in tasks}
    processed_by_task: dict[str, int] = {task.task_id: 0 for task in tasks}
    for artifact in artifacts:
        try:
            rows = ForecastStore(Path(artifact.storage_uri).parent).read_forecasts(artifact.storage_uri)
        except FileNotFoundError:
            continue
        artifact_completed = sum(1 for row in rows if row.get("status") == "succeeded")
        artifact_failed = sum(1 for row in rows if row.get("status") == "failed")
        succeeded_by_task[artifact.task_id] = succeeded_by_task.get(artifact.task_id, 0) + artifact_completed
        failed_by_task[artifact.task_id] = failed_by_task.get(artifact.task_id, 0) + artifact_failed
        processed_by_task[artifact.task_id] = processed_by_task.get(artifact.task_id, 0) + artifact_completed + artifact_failed
    for task in tasks:
        task_processed = int(task.processed_sample_count or 0)
        task_failed = int(task.failed_sample_count or 0)
        task_succeeded = max(0, task_processed - task_failed)
        succeeded_by_task[task.task_id] = max(succeeded_by_task.get(task.task_id, 0), task_succeeded)
        failed_by_task[task.task_id] = max(failed_by_task.get(task.task_id, 0), task_failed)
        processed_by_task[task.task_id] = max(processed_by_task.get(task.task_id, 0), task_processed)
    completed = sum(succeeded_by_task.values())
    failed = sum(failed_by_task.values())
    processed = sum(processed_by_task.values())
    return completed, failed, processed, succeeded_by_task, failed_by_task, processed_by_task


def build_run_progress(session: Session, run_id: str) -> dict:
    run = session.get(BenchmarkingRun, run_id)
    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    tasks = session.exec(select(Task).where(Task.benchmarking_run_id == run_id)).all()
    events = session.exec(select(RunEvent).where(RunEvent.benchmarking_run_id == run_id).order_by(RunEvent.created_at.desc()).limit(20)).all()
    completed_samples, failed_samples, processed_samples, completed_by_task, failed_by_task, processed_by_task = _sample_counts(session, run_id)
    activity_status = _activity_status(run.status, events, processed_samples, run.sample_count)
    return {
        "benchmarking_run_id": run.benchmarking_run_id,
        "status": run.status,
        "activity_status": activity_status,
        "progress": {
            "total_models": run.model_count,
            "completed_models": len([unit for unit in units if unit.status in {"succeeded", "failed", "partial_succeeded", "cancelled"}]),
            "total_tasks": run.task_count,
            "completed_tasks": len([task for task in tasks if task.status in {"succeeded", "failed", "partial_succeeded", "cancelled"}]),
            "total_samples": run.sample_count,
            "completed_samples": completed_samples,
            "failed_samples": failed_samples,
            "processed_samples": processed_samples,
        },
        "units": [
            {
                "unit_id": unit.unit_id,
                "model_id": unit.model_id,
                "model_name": (session.get(Model, unit.model_id).name if session.get(Model, unit.model_id) else unit.model_id),
                "status": unit.status,
                "activity_status": _unit_activity_status(session, unit, tasks, processed_by_task),
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
                "failed_sample_count": failed_by_task.get(task.task_id, 0),
                "processed_sample_count": processed_by_task.get(task.task_id, 0),
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


def build_run_activity_status(session: Session, run: BenchmarkingRun) -> str:
    events = session.exec(
        select(RunEvent)
        .where(RunEvent.benchmarking_run_id == run.benchmarking_run_id)
        .order_by(RunEvent.created_at.desc())
        .limit(1)
    ).all()
    tasks = session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).all()
    processed_samples = sum(int(task.processed_sample_count or 0) for task in tasks)
    return _activity_status(run.status, events, processed_samples, run.sample_count)


def _unit_activity_status(session: Session, unit: Unit, tasks: list[Task], processed_by_task: dict[str, int]) -> str:
    events = session.exec(
        select(RunEvent)
        .where(RunEvent.benchmarking_run_id == unit.benchmarking_run_id)
        .where(RunEvent.unit_id == unit.unit_id)
        .order_by(RunEvent.created_at.desc())
        .limit(1)
    ).all()
    processed_samples = sum(int(processed_by_task.get(task.task_id, 0)) for task in tasks if task.unit_id == unit.unit_id)
    return _activity_status(unit.status, events, processed_samples, unit.sample_count, prefer_event=True)


def _activity_status(run_status: str, events: list[RunEvent], processed_samples: int, total_samples: int, prefer_event: bool = False) -> str:
    latest_event_type = events[0].event_type if events else ""
    activity_by_event = {
        "model_load_started": "model_loading",
        "model_loaded": "forecasting",
        "model_unload_started": "model_unloading",
        "model_load_failed": "model_loading_failed",
        "model_unload_failed": "model_unloading_failed",
    }
    if prefer_event and latest_event_type in activity_by_event:
        return activity_by_event[latest_event_type]
    if run_status in _TERMINAL_RUN_STATUSES:
        return run_status
    if latest_event_type in activity_by_event:
        return activity_by_event[latest_event_type]
    if run_status == "running":
        if total_samples and processed_samples >= total_samples:
            return "finalizing"
        return "forecasting" if processed_samples else "running"
    return run_status
