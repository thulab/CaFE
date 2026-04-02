from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .data_processors import DataProcessorError, build_default_dataset_processor_pipeline
from .data_validators import DataValidationContext, build_default_dataset_validation_pipeline
from .domain import (
    AdminDashboardOverview,
    AggregatedMetrics,
    BatchGenerationRequest,
    BatchSummary,
    BenchmarkReport,
    DatasetBatch,
    DatasetLoadRequest,
    EvaluationTask,
    HuggingFaceConfig,
    HuggingFaceModelRegistrationRequest,
    HuggingFaceTask,
    LeaderboardEntry,
    ModelAdapter,
    ModelRecord,
    ModelRegistrationRequest,
    ModelRuntimeStatus,
    SampleOutcome,
    SeriesSample,
    SeriesTruth,
    TaskRunRequest,
    TaskStatus,
    TaskSummary,
    TrackKind,
    TrackLeaderboard,
    TrackSpec,
    UserDashboardOverview,
    ValidationReport,
    utc_now,
)
from .dataloaders import DataLoaderError, build_default_dataset_loader_registry
from .huggingface import HuggingFaceForecast, HuggingFaceModelRunner, HuggingFaceRunnerError
from .storage import FileRepository


class BenchmarkError(RuntimeError):
    pass


class NotFoundError(BenchmarkError):
    pass


@dataclass
class ExecutionResult:
    prediction: list[float]
    latency_ms: float
    token_count: int
    notes: dict[str, str]


class BenchmarkEngine:
    MAX_GENERATION_ATTEMPTS = 5

    def __init__(self, runtime_root: Path) -> None:
        self.repo = FileRepository(runtime_root)
        self.track_specs = self._build_track_specs()
        self.dataset_loader_registry = build_default_dataset_loader_registry()
        self.dataset_processor_pipeline = build_default_dataset_processor_pipeline()
        self.dataset_validation_pipeline = build_default_dataset_validation_pipeline()
        self.huggingface_runner_factory = HuggingFaceModelRunner
        self._huggingface_runners: dict[str, HuggingFaceModelRunner] = {}
        self._bootstrap_builtin_models()

    def _build_track_specs(self) -> dict[TrackKind, TrackSpec]:
        return {
            TrackKind.FORECAST_ACCURACY: TrackSpec(
                track=TrackKind.FORECAST_ACCURACY,
                name="Forecast Accuracy",
                description="标准零样本预测赛道，主要考察趋势、周期和多尺度外推能力。",
                fairness_policy="monthly_replay",
                default_context_length=96,
                default_horizon=24,
                suggested_sample_count=12,
                knobs=["period_count", "trend_strength", "noise_level"],
            ),
            TrackKind.COVARIATE_ROBUSTNESS: TrackSpec(
                track=TrackKind.COVARIATE_ROBUSTNESS,
                name="Covariate Robustness",
                description="打乱协变量顺序并注入无关变量，评估模型抗协变量干扰能力。",
                fairness_policy="monthly_replay_with_scoreboard_points",
                default_context_length=96,
                default_horizon=24,
                suggested_sample_count=12,
                knobs=["covariate_count", "distractor_ratio", "order_shuffle"],
            ),
            TrackKind.NOISE_ROBUSTNESS: TrackSpec(
                track=TrackKind.NOISE_ROBUSTNESS,
                name="Noise Robustness",
                description="在输入序列上增加噪声和相位漂移，评估模型抗噪能力。",
                fairness_policy="monthly_replay_with_std",
                default_context_length=96,
                default_horizon=24,
                suggested_sample_count=12,
                knobs=["noise_level", "phase_shift", "amplitude_drift"],
            ),
            TrackKind.COST_INTENSIVE: TrackSpec(
                track=TrackKind.COST_INTENSIVE,
                name="Cost Intensive",
                description="复杂长上下文任务，联合评估效果与延迟/Token 成本。",
                fairness_policy="seasonal_points_ranking",
                default_context_length=192,
                default_horizon=48,
                suggested_sample_count=8,
                knobs=["context_length", "horizon", "token_budget", "latency_budget"],
            ),
        }

    def _bootstrap_builtin_models(self) -> None:
        builtins = [
            ModelRecord(
                model_id="seasonal-naive-stub",
                name="Seasonal Naive Stub",
                adapter=ModelAdapter.SEASONAL_NAIVE,
                source_type="builtin_stub",
                manual="重复历史中的主周期片段，适合做周期理解赛道基线。",
                capabilities=["forecast", "multi_period"],
                runtime_status=ModelRuntimeStatus.READY,
            ),
            ModelRecord(
                model_id="recent-mean-stub",
                name="Recent Mean Stub",
                adapter=ModelAdapter.RECENT_MEAN,
                source_type="builtin_stub",
                manual="用近期窗口均值外推，作为简单平滑型基线。",
                capabilities=["forecast", "low_cost"],
                runtime_status=ModelRuntimeStatus.READY,
            ),
            ModelRecord(
                model_id="covariate-trap-stub",
                name="Covariate Trap Stub",
                adapter=ModelAdapter.COVARIATE_TRAP,
                source_type="builtin_stub",
                manual="故意依赖协变量顺序的桩模型，用于验证协变量赛道是否能识别脆弱行为。",
                capabilities=["covariate", "bad_case_generation"],
                runtime_status=ModelRuntimeStatus.READY,
            ),
        ]
        for record in builtins:
            if not self.repo.exists("models", record.model_id):
                self.repo.save("models", record.model_id, record)

    def list_tracks(self) -> list[TrackSpec]:
        return list(self.track_specs.values())

    def list_models(self) -> list[ModelRecord]:
        models = [ModelRecord.model_validate(item) for item in self.repo.list("models")]
        return sorted(models, key=lambda item: item.created_at, reverse=True)

    def register_model(self, request: ModelRegistrationRequest) -> ModelRecord:
        if request.adapter in {
            ModelAdapter.HUGGINGFACE_TEXT_GENERATION,
            ModelAdapter.HUGGINGFACE_CHRONOS2,
        }:
            raise BenchmarkError("use the dedicated Hugging Face registration endpoint for huggingface models")
        self._ensure_model_not_exists(request.model_id)
        record = ModelRecord(
            **request.model_dump(),
            runtime_status=ModelRuntimeStatus.READY,
        )
        self.repo.save("models", record.model_id, record)
        return record

    def register_huggingface_model(self, request: HuggingFaceModelRegistrationRequest) -> ModelRecord:
        model_id = self._normalize_model_id(request.model_id or request.repo_id)
        self._ensure_model_not_exists(model_id)
        adapter = (
            ModelAdapter.HUGGINGFACE_CHRONOS2
            if request.task == HuggingFaceTask.CHRONOS2
            else ModelAdapter.HUGGINGFACE_TEXT_GENERATION
        )
        record = ModelRecord(
            model_id=model_id,
            name=request.name or request.repo_id,
            adapter=adapter,
            source_type="huggingface_hub",
            manual=request.manual,
            capabilities=request.capabilities,
            metadata=request.metadata,
            runtime_status=ModelRuntimeStatus.REGISTERED,
            huggingface=HuggingFaceConfig(
                repo_id=request.repo_id,
                task=request.task,
                revision=request.revision,
                trust_remote_code=request.trust_remote_code,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                device_map=request.device_map,
                torch_dtype=request.torch_dtype,
                attn_implementation=request.attn_implementation,
                batch_size=request.batch_size,
                context_length=request.context_length,
                use_covariates=request.use_covariates,
                cross_learning=request.cross_learning,
                max_output_patches=request.max_output_patches,
                load_retries=request.load_retries,
                load_retry_backoff_seconds=request.load_retry_backoff_seconds,
            ),
        )
        self.repo.save("models", record.model_id, record)
        return record

    def load_model(self, model_id: str) -> ModelRecord:
        model = self._get_model(model_id)
        if not self._is_huggingface_adapter(model.adapter):
            if model.runtime_status != ModelRuntimeStatus.READY:
                model = model.model_copy(update={"runtime_status": ModelRuntimeStatus.READY, "last_error": None})
                self.repo.save("models", model.model_id, model)
            return model

        try:
            self._get_or_load_huggingface_runner(model)
        except BenchmarkError:
            raise
        except Exception as exc:
            updated = model.model_copy(
                update={"runtime_status": ModelRuntimeStatus.LOAD_FAILED, "last_error": str(exc)}
            )
            self.repo.save("models", model.model_id, updated)
            raise BenchmarkError(str(exc)) from exc

        updated = model.model_copy(
            update={
                "runtime_status": ModelRuntimeStatus.READY,
                "last_loaded_at": utc_now(),
                "last_error": None,
            }
        )
        self.repo.save("models", model.model_id, updated)
        return updated

    def list_batches(self) -> list[DatasetBatch]:
        batches = [DatasetBatch.model_validate(item) for item in self.repo.list("batches")]
        return sorted(batches, key=lambda item: item.created_at, reverse=True)

    def get_batch(self, batch_id: str) -> DatasetBatch:
        if not self.repo.exists("batches", batch_id):
            raise NotFoundError(f"batch {batch_id} not found")
        return DatasetBatch.model_validate(self.repo.load("batches", batch_id))

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

    def generate_batch(self, request: BatchGenerationRequest) -> DatasetBatch:
        if request.sample_count <= 0:
            raise BenchmarkError("sample_count must be positive")
        spec = self.track_specs[request.track]
        context_length = request.context_length or spec.default_context_length
        horizon = request.horizon or spec.default_horizon
        batch_id = ""
        selected_seed = request.seed
        samples: list[SeriesSample] = []
        validation = ValidationReport(passed=False, issues=["generation not attempted"])

        for attempt in range(self.MAX_GENERATION_ATTEMPTS):
            attempt_seed = request.seed + attempt
            selected_seed = attempt_seed
            batch_id = f"{request.track.value}-{attempt_seed}-{uuid4().hex[:8]}"
            samples = [
                self._generate_sample(
                    rng=random.Random(attempt_seed * 1000 + index),
                    sample_id=f"{batch_id}-sample-{index + 1:03d}",
                    track=request.track,
                    context_length=context_length,
                    horizon=horizon,
                )
                for index in range(request.sample_count)
            ]
            validation = self._validate_dataset(samples, context_length=context_length, horizon=horizon)
            if validation.passed:
                break

        if not validation.passed:
            raise BenchmarkError(
                f"generated dataset failed validation after {self.MAX_GENERATION_ATTEMPTS} attempts: {validation.issues}"
            )

        batch = DatasetBatch(
            batch_id=batch_id,
            track=request.track,
            policy=spec.fairness_policy,
            seed=selected_seed,
            sample_count=request.sample_count,
            context_length=context_length,
            horizon=horizon,
            samples=samples,
            validation=validation,
        )
        self.repo.save("batches", batch.batch_id, batch)
        return batch

    def load_batch(self, request: DatasetLoadRequest) -> DatasetBatch:
        if request.context_length <= 0:
            raise BenchmarkError("context_length must be positive")
        if request.horizon <= 0:
            raise BenchmarkError("horizon must be positive")
        if request.max_samples is not None and request.max_samples <= 0:
            raise BenchmarkError("max_samples must be positive when provided")

        spec = self.track_specs[request.track]
        try:
            loader = self.dataset_loader_registry.get(request.source_type)
            samples = loader.load_samples(request)
        except DataLoaderError as exc:
            raise BenchmarkError(str(exc)) from exc
        except ValueError as exc:
            raise BenchmarkError(str(exc)) from exc

        try:
            samples = self.dataset_processor_pipeline.process(samples, request)
        except DataProcessorError as exc:
            raise BenchmarkError(str(exc)) from exc

        batch_id = f"{request.batch_id_prefix}-{request.track.value}-{uuid4().hex[:8]}"
        validation = self._validate_dataset(samples, context_length=request.context_length, horizon=request.horizon)
        if not validation.passed:
            raise BenchmarkError(f"loaded dataset failed validation and must be regenerated: {validation.issues}")
        batch = DatasetBatch(
            batch_id=batch_id,
            track=request.track,
            policy=spec.fairness_policy,
            seed=0,
            sample_count=len(samples),
            context_length=request.context_length,
            horizon=request.horizon,
            samples=samples,
            validation=validation,
        )
        self.repo.save("batches", batch.batch_id, batch)
        return batch

    def run_task(self, request: TaskRunRequest) -> EvaluationTask:
        batch = self.get_batch(request.batch_id)
        model = self._get_model(request.model_id)
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
            report = self._build_report(
                task_id=task_id,
                model=model,
                batch=batch,
                outcomes=sample_outcomes,
                metrics=metrics,
            )
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
        except Exception as exc:
            failed = running.model_copy(update={"status": TaskStatus.FAILED, "error_message": str(exc)})
            self.repo.save("tasks", task_id, failed)
            raise BenchmarkError(f"task {task_id} failed: {exc}") from exc

    def leaderboard(self, track: TrackKind | None = None) -> list[LeaderboardEntry]:
        models = {model.model_id: model for model in self.list_models()}
        entries: list[LeaderboardEntry] = []
        for task in self.list_tasks():
            if task.status != TaskStatus.SUCCEEDED or task.metrics is None:
                continue
            if track is not None and task.track != track:
                continue
            model = models.get(task.model_id)
            entries.append(
                LeaderboardEntry(
                    task_id=task.task_id,
                    model_id=task.model_id,
                    model_name=model.name if model else task.model_id,
                    batch_id=task.batch_id,
                    track=task.track,
                    composite_score=task.metrics.composite_score,
                    mse=task.metrics.mse,
                    smape=task.metrics.smape,
                    mean_latency_ms=task.metrics.mean_latency_ms,
                )
            )
        return sorted(entries, key=lambda item: item.composite_score, reverse=True)

    def user_overview(self) -> UserDashboardOverview:
        return UserDashboardOverview(
            tracks=self.list_tracks(),
            models=self.list_models(),
            overall_leaderboard=self.leaderboard()[:10],
            track_leaderboards=[
                TrackLeaderboard(track=track.track, entries=self.leaderboard(track=track.track)[:10])
                for track in self.list_tracks()
            ],
        )

    def admin_overview(self) -> AdminDashboardOverview:
        models = self.list_models()
        model_map = {model.model_id: model for model in models}
        return AdminDashboardOverview(
            tracks=self.list_tracks(),
            models=models,
            batches=[self._batch_summary(batch) for batch in self.list_batches()[:8]],
            recent_tasks=[self._task_summary(task, model_map) for task in self.list_tasks()[:8]],
            leaderboard=self.leaderboard()[:8],
        )

    def overview(self) -> AdminDashboardOverview:
        return self.admin_overview()

    def _batch_summary(self, batch: DatasetBatch) -> BatchSummary:
        return BatchSummary(
            batch_id=batch.batch_id,
            track=batch.track,
            policy=batch.policy,
            created_at=batch.created_at,
            sample_count=batch.sample_count,
            context_length=batch.context_length,
            horizon=batch.horizon,
            validation_passed=batch.validation.passed,
        )

    def _task_summary(self, task: EvaluationTask, model_map: dict[str, ModelRecord]) -> TaskSummary:
        model = model_map.get(task.model_id)
        return TaskSummary(
            task_id=task.task_id,
            model_id=task.model_id,
            model_name=model.name if model else task.model_id,
            batch_id=task.batch_id,
            track=task.track,
            status=task.status,
            created_at=task.created_at,
            composite_score=task.metrics.composite_score if task.metrics else None,
            report_id=task.report_id,
            error_message=task.error_message,
        )

    def _ensure_model_not_exists(self, model_id: str) -> None:
        if self.repo.exists("models", model_id):
            raise BenchmarkError(f"model {model_id} already exists")

    def _normalize_model_id(self, raw_value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", raw_value.lower()).strip("-")
        return normalized or f"model-{uuid4().hex[:6]}"

    def _get_model(self, model_id: str) -> ModelRecord:
        if not self.repo.exists("models", model_id):
            raise NotFoundError(f"model {model_id} not found")
        return ModelRecord.model_validate(self.repo.load("models", model_id))

    def _get_or_load_huggingface_runner(self, model: ModelRecord) -> HuggingFaceModelRunner:
        if model.huggingface is None:
            raise BenchmarkError(f"model {model.model_id} has no huggingface config")
        runner = self._huggingface_runners.get(model.model_id)
        if runner is None:
            runner = self.huggingface_runner_factory(model.huggingface)
            self._huggingface_runners[model.model_id] = runner
        try:
            runner.load()
        except HuggingFaceRunnerError as exc:
            raise BenchmarkError(str(exc)) from exc
        return runner

    def _is_huggingface_adapter(self, adapter: ModelAdapter) -> bool:
        return adapter in {
            ModelAdapter.HUGGINGFACE_TEXT_GENERATION,
            ModelAdapter.HUGGINGFACE_CHRONOS2,
        }

    def _generate_sample(
        self,
        rng: random.Random,
        sample_id: str,
        track: TrackKind,
        context_length: int,
        horizon: int,
    ) -> SeriesSample:
        total_length = context_length + horizon
        periods = self._choose_periods(track, context_length)
        dominant_period = periods[-1]
        phase_shift = rng.random() > 0.55
        amplitude_mode = rng.choice(["stable", "slow_drift", "mid_spike"])
        trend_type = rng.choice(["linear", "piecewise_linear", "smooth_curve"])
        difficulty = rng.choice(["easy", "medium", "hard"])
        noise_level = self._noise_level_for(track, difficulty)

        series = []
        for step in range(total_length):
            value = self._trend_value(step, total_length, trend_type)
            for order, period in enumerate(periods, start=1):
                amplitude = 0.8 * order
                if amplitude_mode == "slow_drift":
                    amplitude *= 1.0 + 0.3 * math.sin(step / max(period, 2))
                elif amplitude_mode == "mid_spike" and step > total_length // 2:
                    amplitude *= 1.3
                phase = math.pi / 5 if phase_shift and step > context_length else 0.0
                value += amplitude * math.sin(2 * math.pi * step / period + phase)
            value += rng.gauss(0.0, noise_level)
            series.append(round(value, 4))

        history = series[:context_length]
        target = series[context_length:]
        covariates: dict[str, list[float]] = {}
        track_tags = [track.value, difficulty]
        notes: dict[str, object] = {"total_length": total_length}

        if track == TrackKind.COVARIATE_ROBUSTNESS:
            helpful = [round(v * 0.7 + rng.gauss(0.0, noise_level / 2), 4) for v in series]
            distractors = {
                f"distractor_{index + 1}": [
                    round(
                        0.5 * math.sin(2 * math.pi * step / rng.choice([5, 9, 11])) + rng.gauss(0.0, 0.7),
                        4,
                    )
                    for step in range(total_length)
                ]
                for index in range(3)
            }
            ordered_items = [("helpful_covariate", helpful), *distractors.items()]
            rng.shuffle(ordered_items)
            covariates = {key: values for key, values in ordered_items}
            notes["covariate_order"] = list(covariates.keys())
            notes["future_known_covariates"] = ["helpful_covariate"]
            track_tags.append("order_shuffle")
        elif track == TrackKind.NOISE_ROBUSTNESS:
            history = [round(value + rng.gauss(0.0, noise_level * 1.8), 4) for value in history]
            covariates["noise_probe"] = [round(rng.gauss(0.0, 1.0), 4) for _ in range(total_length)]
            track_tags.append("noise_augmented")
        elif track == TrackKind.COST_INTENSIVE:
            covariates["calendar_signal"] = [round(math.sin(2 * math.pi * step / 24), 4) for step in range(total_length)]
            covariates["load_signal"] = [round(0.2 * step / total_length + rng.random(), 4) for step in range(total_length)]
            notes["future_known_covariates"] = ["calendar_signal", "load_signal"]
            track_tags.extend(["long_context", "cost_sensitive"])

        truth = SeriesTruth(
            trend_type=trend_type,
            periods=periods,
            dominant_period=dominant_period,
            amplitude_mode=amplitude_mode,
            phase_shift=phase_shift,
            noise_level=noise_level,
            difficulty=difficulty,
        )
        return SeriesSample(
            sample_id=sample_id,
            history=history,
            target=target,
            covariates=covariates,
            track_tags=track_tags,
            truth=truth,
            notes=notes,
        )

    def _choose_periods(self, track: TrackKind, context_length: int) -> list[int]:
        if track == TrackKind.COST_INTENSIVE:
            return [12, 24, 48]
        if track == TrackKind.NOISE_ROBUSTNESS:
            return [8, 24]
        if track == TrackKind.COVARIATE_ROBUSTNESS:
            return [6, 18]
        if context_length >= 96:
            return [12, 24]
        return [6, 12]

    def _noise_level_for(self, track: TrackKind, difficulty: str) -> float:
        base = {
            TrackKind.FORECAST_ACCURACY: 0.18,
            TrackKind.COVARIATE_ROBUSTNESS: 0.2,
            TrackKind.NOISE_ROBUSTNESS: 0.35,
            TrackKind.COST_INTENSIVE: 0.22,
        }[track]
        factor = {"easy": 0.8, "medium": 1.0, "hard": 1.3}[difficulty]
        return round(base * factor, 4)

    def _trend_value(self, step: int, total_length: int, trend_type: str) -> float:
        ratio = step / max(total_length - 1, 1)
        if trend_type == "linear":
            return 4.0 * ratio
        if trend_type == "piecewise_linear":
            if ratio < 0.4:
                return 2.5 * ratio
            if ratio < 0.75:
                return 1.0 + 4.0 * (ratio - 0.4)
            return 2.4 + 1.5 * (ratio - 0.75)
        return 0.6 + 2.0 * ratio + math.sin(ratio * math.pi) * 0.5

    def _validate_dataset(self, samples: list[SeriesSample], context_length: int, horizon: int) -> ValidationReport:
        context = DataValidationContext(context_length=context_length, horizon=horizon)
        return self.dataset_validation_pipeline.validate(samples, context)

    def _score_sample(self, model: ModelRecord, sample: SeriesSample, track: TrackKind) -> SampleOutcome:
        execution = self._execute_model(model, sample, track)
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

    def _score_samples(self, model: ModelRecord, samples: list[SeriesSample], track: TrackKind) -> list[SampleOutcome]:
        if model.adapter == ModelAdapter.HUGGINGFACE_CHRONOS2:
            forecasts = self._execute_huggingface_model_batch(model=model, samples=samples, track=track)
            return [
                self._sample_outcome_from_forecast(sample=sample, forecast=forecast)
                for sample, forecast in zip(samples, forecasts, strict=True)
            ]
        return [self._score_sample(model=model, sample=sample, track=track) for sample in samples]

    def _sample_outcome_from_forecast(self, sample: SeriesSample, forecast: HuggingFaceForecast) -> SampleOutcome:
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

    def _execute_model(self, model: ModelRecord, sample: SeriesSample, track: TrackKind) -> ExecutionResult:
        history = sample.history
        horizon = len(sample.target)
        covariates = sample.covariates
        dominant_period = sample.truth.dominant_period
        base_tokens = 90 + len(history) // 2 + len(covariates) * 24

        if model.adapter == ModelAdapter.SEASONAL_NAIVE:
            if len(history) >= dominant_period:
                season = history[-dominant_period:]
                prediction = [round(season[index % len(season)], 4) for index in range(horizon)]
            else:
                prediction = [round(history[-1], 4)] * horizon
            latency = 4.5 + horizon * 0.12 + len(covariates) * 0.2
            notes = {"decision": "repeat dominant period"}
        elif model.adapter == ModelAdapter.RECENT_MEAN:
            window = history[-min(8, len(history)) :]
            baseline = sum(window) / len(window)
            prediction = [round(baseline, 4)] * horizon
            latency = 2.0 + horizon * 0.08
            notes = {"decision": "smooth recent window"}
        elif model.adapter == ModelAdapter.COVARIATE_TRAP:
            if covariates:
                _, first_signal = next(iter(covariates.items()))
                prediction = [round(first_signal[len(history) + index], 4) for index in range(horizon)]
            else:
                prediction = [round(history[-1], 4)] * horizon
            latency = 5.5 + horizon * 0.16 + len(covariates) * 0.6
            notes = {"decision": "trust first covariate", "first_covariate": next(iter(covariates.keys()), "none")}
        elif model.adapter in {
            ModelAdapter.HUGGINGFACE_TEXT_GENERATION,
            ModelAdapter.HUGGINGFACE_CHRONOS2,
        }:
            huggingface_result = self._execute_huggingface_model(model, sample, track)
            prediction = huggingface_result.prediction
            latency = huggingface_result.latency_ms
            base_tokens = huggingface_result.token_count
            notes = huggingface_result.notes
        else:
            raise BenchmarkError(f"unsupported model adapter {model.adapter}")

        if track == TrackKind.COST_INTENSIVE:
            latency *= 1.8
            base_tokens += 150
        elif track == TrackKind.NOISE_ROBUSTNESS:
            latency *= 1.2

        token_count = int(base_tokens + horizon * 3)
        return ExecutionResult(prediction=prediction, latency_ms=latency, token_count=token_count, notes=notes)

    def _execute_huggingface_model(
        self,
        model: ModelRecord,
        sample: SeriesSample,
        track: TrackKind,
    ) -> HuggingFaceForecast:
        if model.huggingface is None:
            raise BenchmarkError(f"model {model.model_id} has no huggingface config")
        runner = self._get_or_load_huggingface_runner(model)
        return runner.forecast(sample=sample, track=track)

    def _execute_huggingface_model_batch(
        self,
        model: ModelRecord,
        samples: list[SeriesSample],
        track: TrackKind,
    ) -> list[HuggingFaceForecast]:
        if model.huggingface is None:
            raise BenchmarkError(f"model {model.model_id} has no huggingface config")
        runner = self._get_or_load_huggingface_runner(model)
        return runner.forecast_batch(samples=samples, track=track)

    def _aggregate_metrics(self, outcomes: list[SampleOutcome]) -> AggregatedMetrics:
        mse = sum(item.mse for item in outcomes) / len(outcomes)
        mae = sum(item.mae for item in outcomes) / len(outcomes)
        smape = sum(item.smape for item in outcomes) / len(outcomes)
        latency = sum(item.latency_ms for item in outcomes) / len(outcomes)
        tokens = sum(item.token_count for item in outcomes) / len(outcomes)
        composite = 100.0 / (1.0 + mse) - latency * 0.35 - tokens * 0.01
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
        sorted_outcomes = sorted(outcomes, key=lambda item: item.mse, reverse=True)
        worst = sorted_outcomes[:3]
        strengths: list[str] = []
        risks: list[str] = []
        if metrics.mse < 2.2:
            strengths.append("整体误差保持在可接受区间，具备作为基线的稳定性。")
        if metrics.mean_latency_ms < 10:
            strengths.append("推理延迟较低，适合高频批量评测。")
        if batch.track == TrackKind.COVARIATE_ROBUSTNESS and model.adapter != ModelAdapter.COVARIATE_TRAP:
            strengths.append("在协变量干扰赛道中未表现出明显的顺序依赖。")
        if model.adapter in {
            ModelAdapter.HUGGINGFACE_TEXT_GENERATION,
            ModelAdapter.HUGGINGFACE_CHRONOS2,
        }:
            strengths.append("模型通过 Hugging Face 接入，可用于统一管理第三方提交模型。")

        if batch.track == TrackKind.COVARIATE_ROBUSTNESS and model.adapter == ModelAdapter.COVARIATE_TRAP:
            risks.append("模型存在协变量顺序敏感性，容易被无关变量误导。")
        if batch.track == TrackKind.NOISE_ROBUSTNESS:
            risks.append("噪声放大后误差波动可能显著，需要配合更稳健的去噪或上下文选择策略。")
        if metrics.mean_token_count > 220:
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
