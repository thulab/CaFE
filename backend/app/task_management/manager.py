from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from pathlib import Path
from uuid import uuid4

from ..config import AppSettings, get_settings
from ..data_management.domain import DatasetBatch, TrackKind
from ..data_management.manager import DataManager
from ..domain.common import utc_now
from ..errors import BenchmarkError, InternalBenchmarkError, NotFoundError
from ..model_management import HuggingFaceForecast, ModelManager
from ..model_management.domain import HuggingFaceConfig, ModelAdapter, ModelRecord, ModelRuntimeParameterDefinition
from ..storage import FileRepository
from .domain import (
    DEFAULT_EVALUATION_METRICS,
    DEFAULT_EXECUTION_REPEAT_COUNT,
    AggregatedMetrics,
    BenchmarkReport,
    EvaluationTask,
    SampleOutcome,
    TaskDatasetSpec,
    TaskRunRecord,
    TaskRunRequest,
    TaskRunStatus,
    TaskSpec,
    TaskStatus,
)


class TaskManager:
    def __init__(
        self,
        runtime_root: Path,
        data_manager: DataManager,
        model_manager: ModelManager,
        settings: AppSettings | None = None,
        repository: FileRepository | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = repository or FileRepository(runtime_root)
        self.data_manager = data_manager
        self.model_manager = model_manager

    def list_tasks(self) -> list[EvaluationTask]:
        tasks = [EvaluationTask.parse_obj(item) for item in self.repo.list("tasks")]
        return sorted(tasks, key=lambda item: item.created_at, reverse=True)

    def get_task(self, task_id: str) -> EvaluationTask:
        if not self.repo.exists("tasks", task_id):
            raise NotFoundError(f"task {task_id} not found")
        return EvaluationTask.parse_obj(self.repo.load("tasks", task_id))

    def get_report(self, report_id: str) -> BenchmarkReport:
        if not self.repo.exists("reports", report_id):
            raise NotFoundError(f"report {report_id} not found")
        return BenchmarkReport.parse_obj(self.repo.load("reports", report_id))

    def run_task(self, request: TaskRunRequest) -> EvaluationTask:
        batch = self.data_manager.get_batch(request.batch_id)
        model = self.model_manager.get_model(request.model_id)
        task_id = f"task-{uuid4().hex[:8]}"
        execution_repeat_count = self._resolve_execution_repeat_count(request.execution_repeat_count)
        task_spec = self._build_task_spec(model=model, batch=batch, request=request, execution_repeat_count=execution_repeat_count)
        running = EvaluationTask(
            task_id=task_id,
            model_id=model.model_id,
            batch_id=batch.batch_id,
            track=batch.track,
            track_variant_id=batch.track_variant_id,
            status=TaskStatus.RUNNING,
            spec=task_spec,
        )
        self.repo.save("tasks", task_id, running)

        try:
            effective_model = self._resolve_model_for_task(model=model, runtime_parameters=request.model_runtime_parameters)
            task_runs = self._execute_task_runs(
                task_id=task_id,
                model=effective_model,
                batch=batch,
                execution_repeat_count=execution_repeat_count,
            )
            aggregated_outcomes = self._aggregate_sample_outcomes(task_runs)
            metrics = self._aggregate_metrics(aggregated_outcomes, run_metrics=[run.metrics for run in task_runs if run.metrics is not None])
            report = self._build_report(
                task_id=task_id,
                task_runs=task_runs,
                model=model,
                batch=batch,
                outcomes=aggregated_outcomes,
                metrics=metrics,
                evaluation_metrics=task_spec.evaluation_metrics,
            )
            self.repo.save("reports", report.report_id, report)
            finished = running.copy(
                update={
                    "status": TaskStatus.SUCCEEDED,
                    "metrics": metrics,
                    "report_id": report.report_id,
                    "sample_outcomes": aggregated_outcomes,
                    "task_runs": task_runs,
                    "error_message": None,
                }
            )
            self.repo.save("tasks", task_id, finished)
            return finished
        except BenchmarkError as exc:
            self._save_failed_task(running, str(exc))
            raise
        except Exception as exc:
            self._save_failed_task(running, str(exc))
            raise InternalBenchmarkError(f"task {task_id} failed: {exc}") from exc

    def _execute_task_runs(
        self,
        *,
        task_id: str,
        model: ModelRecord,
        batch: DatasetBatch,
        execution_repeat_count: int,
    ) -> list[TaskRunRecord]:
        task_runs: list[TaskRunRecord] = []
        for run_no in range(1, execution_repeat_count + 1):
            run = TaskRunRecord(
                run_id=f"run-{uuid4().hex[:8]}",
                task_id=task_id,
                run_no=run_no,
                status=TaskRunStatus.RUNNING,
                started_at=utc_now(),
            )
            self.repo.save("task_runs", run.run_id, run)
            try:
                sample_outcomes = self._score_samples(model=model, samples=batch.samples, track=batch.track)
                metrics = self._aggregate_metrics(sample_outcomes)
                finished_run = run.copy(
                    update={
                        "status": TaskRunStatus.SUCCEEDED,
                        "finished_at": utc_now(),
                        "metrics": metrics,
                        "sample_outcomes": sample_outcomes,
                        "error_message": None,
                    }
                )
                self.repo.save("task_runs", run.run_id, finished_run)
                task_runs.append(finished_run)
            except Exception as exc:
                failed_run = run.copy(update={"status": TaskRunStatus.FAILED, "finished_at": utc_now(), "error_message": str(exc)})
                self.repo.save("task_runs", run.run_id, failed_run)
                raise
        return task_runs

    def _score_samples(self, model: ModelRecord, samples, track: TrackKind) -> list[SampleOutcome]:
        if model.adapter in {ModelAdapter.HUGGINGFACE_CHRONOS2, ModelAdapter.HUGGINGFACE_SUNDIAL}:
            forecasts = self.model_manager.execute_huggingface_model_batch(model=model, samples=samples, track=track)
            return [
                self._sample_outcome_from_forecast(sample=sample, forecast=forecast)
                for sample, forecast in zip(samples, forecasts, strict=True)
            ]
        return [self._score_sample(model=model, sample=sample, track=track) for sample in samples]

    def _score_sample(self, model: ModelRecord, sample, track: TrackKind) -> SampleOutcome:
        execution = self.model_manager.execute_model(model, sample, track)
        mse = self._mse(execution.prediction, sample.target)
        mae = self._mae(execution.prediction, sample.target)
        smape = self._smape(execution.prediction, sample.target)
        return SampleOutcome(
            sample_id=sample.sample_id,
            mse=round(mse, 6),
            mae=round(mae, 6),
            smape=round(smape, 6),
            latency_ms=round(execution.latency_ms, 3),
            token_count=execution.token_count,
            prediction=execution.prediction,
            notes=execution.notes,
        )

    def _sample_outcome_from_forecast(self, sample, forecast: HuggingFaceForecast) -> SampleOutcome:
        mse = self._mse(forecast.prediction, sample.target)
        mae = self._mae(forecast.prediction, sample.target)
        smape = self._smape(forecast.prediction, sample.target)
        return SampleOutcome(
            sample_id=sample.sample_id,
            mse=round(mse, 6),
            mae=round(mae, 6),
            smape=round(smape, 6),
            latency_ms=round(forecast.latency_ms, 3),
            token_count=forecast.token_count,
            prediction=forecast.prediction,
            notes=forecast.notes,
        )

    def _aggregate_metrics(
        self,
        outcomes: list[SampleOutcome],
        *,
        run_metrics: list[AggregatedMetrics] | None = None,
    ) -> AggregatedMetrics:
        scoring = self.settings.benchmark.scoring
        mse = sum(item.mse for item in outcomes) / len(outcomes)
        mae = sum(item.mae for item in outcomes) / len(outcomes)
        smape = sum(item.smape for item in outcomes) / len(outcomes)
        latency = sum(item.latency_ms for item in outcomes) / len(outcomes)
        tokens = sum(item.token_count for item in outcomes) / len(outcomes)
        composite = (
            scoring.composite_base / (scoring.composite_mse_offset + mse)
            - latency * scoring.composite_latency_penalty
            - tokens * scoring.composite_token_penalty
        )
        stability_stats: dict[str, float] = {}
        if run_metrics:
            stability_stats = {
                "run_count": float(len(run_metrics)),
                "mse_std": round(self._metric_std([item.mse for item in run_metrics]), 6),
                "mae_std": round(self._metric_std([item.mae for item in run_metrics]), 6),
                "smape_std": round(self._metric_std([item.smape for item in run_metrics]), 6),
                "latency_ms_std": round(self._metric_std([item.mean_latency_ms for item in run_metrics]), 6),
                "token_count_std": round(self._metric_std([item.mean_token_count for item in run_metrics]), 6),
                "composite_score_std": round(self._metric_std([item.composite_score for item in run_metrics]), 6),
            }
        return AggregatedMetrics(
            mse=round(mse, 6),
            mae=round(mae, 6),
            smape=round(smape, 6),
            mean_latency_ms=round(latency, 3),
            mean_token_count=round(tokens, 3),
            composite_score=round(composite, 3),
            stability_stats=stability_stats,
        )

    def _aggregate_sample_outcomes(self, task_runs: list[TaskRunRecord]) -> list[SampleOutcome]:
        grouped: dict[str, list[SampleOutcome]] = defaultdict(list)
        for run in task_runs:
            for outcome in run.sample_outcomes:
                grouped[outcome.sample_id].append(outcome)

        aggregated: list[SampleOutcome] = []
        for sample_id, outcomes in sorted(grouped.items()):
            reference = outcomes[0]
            aggregated.append(
                SampleOutcome(
                    sample_id=sample_id,
                    mse=round(mean(item.mse for item in outcomes), 6),
                    mae=round(mean(item.mae for item in outcomes), 6),
                    smape=round(mean(item.smape for item in outcomes), 6),
                    latency_ms=round(mean(item.latency_ms for item in outcomes), 3),
                    token_count=int(round(mean(item.token_count for item in outcomes))),
                    prediction=self._mean_prediction([item.prediction for item in outcomes]),
                    run_count=len(outcomes),
                    mse_std=round(self._metric_std([item.mse for item in outcomes]), 6),
                    mae_std=round(self._metric_std([item.mae for item in outcomes]), 6),
                    smape_std=round(self._metric_std([item.smape for item in outcomes]), 6),
                    latency_ms_std=round(self._metric_std([item.latency_ms for item in outcomes]), 6),
                    token_count_std=round(self._metric_std([float(item.token_count) for item in outcomes]), 6),
                    notes=dict(reference.notes, aggregated_from_runs=len(outcomes)),
                )
            )
        return aggregated

    def _build_report(
        self,
        task_id: str,
        task_runs: list[TaskRunRecord],
        model: ModelRecord,
        batch: DatasetBatch,
        outcomes: list[SampleOutcome],
        metrics: AggregatedMetrics,
        evaluation_metrics: list[str],
    ) -> BenchmarkReport:
        reporting = self.settings.benchmark.reporting
        sorted_outcomes = sorted(outcomes, key=lambda item: item.mse, reverse=True)
        worst = sorted_outcomes[: reporting.bad_case_count]
        strengths: list[str] = []
        risks: list[str] = []
        if metrics.mse < reporting.strength_mse_threshold:
            strengths.append("整体误差保持在可接受区间，具备作为基线的稳定性。")
        if metrics.mean_latency_ms < reporting.strength_latency_ms_threshold:
            strengths.append("推理延迟较低，适合高频批量评测。")
        if metrics.stability_stats.get("mse_std", 0.0) < 1e-6:
            strengths.append("重复执行波动较小，结果稳定性较好。")
        if batch.track == TrackKind.COVARIATE_ROBUSTNESS and model.adapter != ModelAdapter.COVARIATE_TRAP:
            strengths.append("在协变量干扰赛道中未表现出明显的顺序依赖。")
        if model.adapter in {
            ModelAdapter.HUGGINGFACE_TEXT_GENERATION,
            ModelAdapter.HUGGINGFACE_CHRONOS2,
            ModelAdapter.HUGGINGFACE_SUNDIAL,
        }:
            strengths.append("模型通过 Hugging Face 接入，可用于统一管理第三方提交模型。")

        if batch.track == TrackKind.COVARIATE_ROBUSTNESS and model.adapter == ModelAdapter.COVARIATE_TRAP:
            risks.append("模型存在协变量顺序敏感性，容易被无关变量误导。")
        if batch.track == TrackKind.NOISE_ROBUSTNESS:
            risks.append("噪声放大后误差波动可能显著，需要配合更稳健的去噪或上下文选择策略。")
        if metrics.mean_token_count > reporting.risk_token_threshold:
            risks.append("Token 成本偏高，适合纳入成本赛道做进一步约束。")
        if metrics.stability_stats.get("mse_std", 0.0) > 0.0:
            risks.append("多次执行存在一定波动，建议结合稳定性统计一起解读结果。")

        summary_metrics = [f"重复执行={len(task_runs)}次"]
        if "mse" in evaluation_metrics:
            summary_metrics.append(f"MSE={metrics.mse:.4f}")
        if "mae" in evaluation_metrics:
            summary_metrics.append(f"MAE={metrics.mae:.4f}")
        if "smape" in evaluation_metrics:
            summary_metrics.append(f"sMAPE={metrics.smape:.4f}")
        if "latency_ms" in evaluation_metrics:
            summary_metrics.append(f"平均延迟={metrics.mean_latency_ms:.2f}ms")
        if "token_count" in evaluation_metrics:
            summary_metrics.append(f"平均Token={metrics.mean_token_count:.2f}")
        if "composite_score" in evaluation_metrics:
            summary_metrics.append(f"综合分={metrics.composite_score:.3f}")
        if metrics.stability_stats:
            summary_metrics.append(f"MSE标准差={metrics.stability_stats.get('mse_std', 0.0):.6f}")
        summary = (
            f"模型 {model.name} 在批次 {batch.batch_id} 的 {batch.track_variant_id} 赛道上完成 {batch.sample_count} 个样本评测，"
            + "，".join(summary_metrics)
            + "。"
        )
        bad_cases = [
            f"{item.sample_id}: mse={item.mse:.4f}, mse_std={item.mse_std or 0.0:.6f}, 备注={item.notes}"
            for item in worst
        ]
        distribution = {
            "mse_p50": round(sorted_outcomes[len(sorted_outcomes) // 2].mse, 6),
            "mse_max": round(sorted_outcomes[0].mse, 6),
            "mse_min": round(sorted_outcomes[-1].mse, 6),
            "mse_std": round(metrics.stability_stats.get("mse_std", 0.0), 6),
            "latency_ms_std": round(metrics.stability_stats.get("latency_ms_std", 0.0), 6),
        }
        return BenchmarkReport(
            report_id=f"report-{task_id}",
            task_id=task_id,
            summary=summary,
            strengths=strengths,
            risks=risks,
            bad_cases=bad_cases,
            distribution=distribution,
            run_ids=[run.run_id for run in task_runs],
        )

    def _mse(self, prediction: list[float], target: list[float]) -> float:
        return sum((pred - real) ** 2 for pred, real in zip(prediction, target)) / len(target)

    def _mae(self, prediction: list[float], target: list[float]) -> float:
        return sum(abs(pred - real) for pred, real in zip(prediction, target)) / len(target)

    def _smape(self, prediction: list[float], target: list[float]) -> float:
        values = []
        for pred, real in zip(prediction, target):
            denom = abs(pred) + abs(real) + 1e-6
            values.append(2.0 * abs(pred - real) / denom)
        return sum(values) / len(values)

    def _save_failed_task(self, running: EvaluationTask, error_message: str) -> None:
        failed = running.copy(update={"status": TaskStatus.FAILED, "error_message": error_message})
        self.repo.save("tasks", running.task_id, failed)

    def _build_task_spec(
        self,
        model: ModelRecord,
        batch: DatasetBatch,
        request: TaskRunRequest,
        *,
        execution_repeat_count: int,
    ) -> TaskSpec:
        return TaskSpec(
            model_id=model.model_id,
            model_runtime_parameters=request.model_runtime_parameters,
            dataset=TaskDatasetSpec(
                batch_id=batch.batch_id,
                track=batch.track,
                track_variant_id=batch.track_variant_id,
                sample_count=batch.sample_count,
                input_length=batch.input_length,
                prediction_length=batch.prediction_length,
                context_length=batch.context_length,
                horizon=batch.horizon,
            ),
            evaluation_metrics=self._normalize_evaluation_metrics(request.evaluation_metrics),
            execution_repeat_count=execution_repeat_count,
        )

    def _normalize_evaluation_metrics(self, metrics: list[str]) -> list[str]:
        if not metrics:
            return list(DEFAULT_EVALUATION_METRICS)
        normalized = [metric.strip() for metric in metrics if metric.strip()]
        unsupported = [metric for metric in normalized if metric not in DEFAULT_EVALUATION_METRICS]
        if unsupported:
            raise BenchmarkError(f"unsupported evaluation metrics: {', '.join(unsupported)}")
        return normalized

    def _resolve_execution_repeat_count(self, value: int | None) -> int:
        if value is None:
            return DEFAULT_EXECUTION_REPEAT_COUNT
        if value <= 0:
            raise BenchmarkError("execution_repeat_count must be positive")
        return value

    def _resolve_model_for_task(self, model: ModelRecord, runtime_parameters: dict[str, Any]) -> ModelRecord:
        if not runtime_parameters:
            return model
        if model.huggingface is None:
            raise BenchmarkError(f"model {model.model_id} does not accept runtime parameter overrides")
        allowed = {definition.name: definition for definition in model.spec.runtime_parameter_definitions}
        unknown = [name for name in runtime_parameters if name not in allowed]
        if unknown:
            raise BenchmarkError(f"unsupported runtime parameters for {model.model_id}: {', '.join(sorted(unknown))}")
        effective_config = self._merge_runtime_parameters(model.huggingface, runtime_parameters, allowed)
        return model.copy(update={"huggingface": effective_config})

    def _merge_runtime_parameters(
        self,
        base_config: HuggingFaceConfig,
        runtime_parameters: dict[str, Any],
        allowed: dict[str, ModelRuntimeParameterDefinition],
    ) -> HuggingFaceConfig:
        payload = base_config.dict()
        payload.update(runtime_parameters)
        try:
            return HuggingFaceConfig.parse_obj(payload)
        except Exception as exc:
            expected = ", ".join(f"{name}:{definition.value_type.value}" for name, definition in sorted(allowed.items()))
            raise BenchmarkError(f"invalid runtime parameters, expected types [{expected}]: {exc}") from exc

    def _metric_std(self, values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        return pstdev(values)

    def _mean_prediction(self, predictions: list[list[float]]) -> list[float]:
        if not predictions:
            return []
        horizon = len(predictions[0])
        return [round(mean(prediction[index] for prediction in predictions), 6) for index in range(horizon)]
