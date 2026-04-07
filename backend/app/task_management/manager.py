from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ..config import AppSettings, get_settings
from ..data_management.manager import DataManager
from ..data_management.domain import DatasetBatch, TrackKind
from ..errors import BenchmarkError, InternalBenchmarkError, NotFoundError
from ..model_management.domain import ModelAdapter, ModelRecord
from ..model_management import HuggingFaceForecast, ModelManager
from ..storage import FileRepository
from .domain import AggregatedMetrics, BenchmarkReport, EvaluationTask, SampleOutcome, TaskRunRequest, TaskStatus


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
        tasks = [EvaluationTask.model_validate(item) for item in self.repo.list("tasks")]
        return sorted(tasks, key=lambda item: item.created_at, reverse=True)

    def get_task(self, task_id: str) -> EvaluationTask:
        if not self.repo.exists("tasks", task_id):
            raise NotFoundError(f"task {task_id} not found")
        return EvaluationTask.model_validate(self.repo.load("tasks", task_id))

    def get_report(self, report_id: str) -> BenchmarkReport:
        if not self.repo.exists("reports", report_id):
            raise NotFoundError(f"report {report_id} not found")
        return BenchmarkReport.model_validate(self.repo.load("reports", report_id))

    def run_task(self, request: TaskRunRequest) -> EvaluationTask:
        batch = self.data_manager.get_batch(request.batch_id)
        model = self.model_manager.get_model(request.model_id)
        task_id = f"task-{uuid4().hex[:8]}"
        running = EvaluationTask(
            task_id=task_id,
            model_id=model.model_id,
            batch_id=batch.batch_id,
            track=batch.track,
            status=TaskStatus.RUNNING,
        )
        self.repo.save("tasks", task_id, running)

        try:
            sample_outcomes = self._score_samples(model=model, samples=batch.samples, track=batch.track)
            metrics = self._aggregate_metrics(sample_outcomes)
            report = self._build_report(task_id=task_id, model=model, batch=batch, outcomes=sample_outcomes, metrics=metrics)
            self.repo.save("reports", report.report_id, report)
            finished = running.model_copy(
                update={
                    "status": TaskStatus.SUCCEEDED,
                    "metrics": metrics,
                    "report_id": report.report_id,
                    "sample_outcomes": sample_outcomes,
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

    def _score_samples(self, model: ModelRecord, samples, track: TrackKind) -> list[SampleOutcome]:
        if model.adapter == ModelAdapter.HUGGINGFACE_CHRONOS2:
            forecasts = self.model_manager.execute_huggingface_model_batch(model=model, samples=samples, track=track)
            return [self._sample_outcome_from_forecast(sample=sample, forecast=forecast) for sample, forecast in zip(samples, forecasts, strict=True)]
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

    def _aggregate_metrics(self, outcomes: list[SampleOutcome]) -> AggregatedMetrics:
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
        return AggregatedMetrics(
            mse=round(mse, 6),
            mae=round(mae, 6),
            smape=round(smape, 6),
            mean_latency_ms=round(latency, 3),
            mean_token_count=round(tokens, 3),
            composite_score=round(composite, 3),
        )

    def _build_report(
        self,
        task_id: str,
        model: ModelRecord,
        batch: DatasetBatch,
        outcomes: list[SampleOutcome],
        metrics: AggregatedMetrics,
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
        if batch.track == TrackKind.COVARIATE_ROBUSTNESS and model.adapter != ModelAdapter.COVARIATE_TRAP:
            strengths.append("在协变量干扰赛道中未表现出明显的顺序依赖。")
        if model.adapter in {ModelAdapter.HUGGINGFACE_TEXT_GENERATION, ModelAdapter.HUGGINGFACE_CHRONOS2}:
            strengths.append("模型通过 Hugging Face 接入，可用于统一管理第三方提交模型。")

        if batch.track == TrackKind.COVARIATE_ROBUSTNESS and model.adapter == ModelAdapter.COVARIATE_TRAP:
            risks.append("模型存在协变量顺序敏感性，容易被无关变量误导。")
        if batch.track == TrackKind.NOISE_ROBUSTNESS:
            risks.append("噪声放大后误差波动可能显著，需要配合更稳健的去噪或上下文选择策略。")
        if metrics.mean_token_count > reporting.risk_token_threshold:
            risks.append("Token 成本偏高，适合纳入成本赛道做进一步约束。")

        summary = (
            f"模型 {model.name} 在批次 {batch.batch_id} 的 {batch.track.value} 赛道上完成 {batch.sample_count} 个样本评测，"
            f"MSE={metrics.mse:.4f}，sMAPE={metrics.smape:.4f}，平均延迟={metrics.mean_latency_ms:.2f}ms。"
        )
        bad_cases = [f"{item.sample_id}: mse={item.mse:.4f}, 备注={item.notes}" for item in worst]
        distribution = {
            "mse_p50": round(sorted_outcomes[len(sorted_outcomes) // 2].mse, 6),
            "mse_max": round(sorted_outcomes[0].mse, 6),
            "mse_min": round(sorted_outcomes[-1].mse, 6),
        }
        return BenchmarkReport(
            report_id=f"report-{task_id}",
            task_id=task_id,
            summary=summary,
            strengths=strengths,
            risks=risks,
            bad_cases=bad_cases,
            distribution=distribution,
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
        failed = running.model_copy(update={"status": TaskStatus.FAILED, "error_message": error_message})
        self.repo.save("tasks", running.task_id, failed)
