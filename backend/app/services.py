from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from .domain import (
    AggregatedMetrics,
    BatchGenerationRequest,
    BenchmarkReport,
    DashboardOverview,
    DatasetBatch,
    EvaluationTask,
    LeaderboardEntry,
    ModelAdapter,
    ModelRecord,
    ModelRegistrationRequest,
    SampleOutcome,
    SeriesSample,
    SeriesTruth,
    TaskRunRequest,
    TaskStatus,
    TrackKind,
    TrackSpec,
    ValidationReport,
)
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
    def __init__(self, runtime_root: Path) -> None:
        self.repo = FileRepository(runtime_root)
        self.track_specs = self._build_track_specs()
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
            ),
            ModelRecord(
                model_id="recent-mean-stub",
                name="Recent Mean Stub",
                adapter=ModelAdapter.RECENT_MEAN,
                source_type="builtin_stub",
                manual="用近期窗口均值外推，作为简单平滑型基线。",
                capabilities=["forecast", "low_cost"],
            ),
            ModelRecord(
                model_id="covariate-trap-stub",
                name="Covariate Trap Stub",
                adapter=ModelAdapter.COVARIATE_TRAP,
                source_type="builtin_stub",
                manual="故意依赖协变量顺序的桩模型，用于验证协变量赛道是否能识别脆弱行为。",
                capabilities=["covariate", "bad_case_generation"],
            ),
        ]
        for record in builtins:
            if not self.repo.exists("models", record.model_id):
                self.repo.save("models", record.model_id, record)

    def list_tracks(self) -> list[TrackSpec]:
        return list(self.track_specs.values())

    def list_models(self) -> list[ModelRecord]:
        return [ModelRecord.model_validate(item) for item in self.repo.list("models")]

    def register_model(self, request: ModelRegistrationRequest) -> ModelRecord:
        record = ModelRecord(**request.model_dump())
        self.repo.save("models", record.model_id, record)
        return record

    def list_batches(self) -> list[DatasetBatch]:
        return [DatasetBatch.model_validate(item) for item in self.repo.list("batches")]

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
        spec = self.track_specs[request.track]
        context_length = request.context_length or spec.default_context_length
        horizon = request.horizon or spec.default_horizon
        batch_id = f"{request.track.value}-{request.seed}-{len(self.list_batches()) + 1:03d}"
        rng = random.Random(request.seed)
        samples = [
            self._generate_sample(
                rng=random.Random(request.seed * 1000 + index),
                sample_id=f"{batch_id}-sample-{index + 1:03d}",
                track=request.track,
                context_length=context_length,
                horizon=horizon,
            )
            for index in range(request.sample_count)
        ]
        validation = self._validate_samples(samples, context_length=context_length, horizon=horizon)
        batch = DatasetBatch(
            batch_id=batch_id,
            track=request.track,
            policy=spec.fairness_policy,
            seed=request.seed,
            sample_count=request.sample_count,
            context_length=context_length,
            horizon=horizon,
            samples=samples,
            validation=validation,
        )
        self.repo.save("batches", batch.batch_id, batch)
        return batch

    def run_task(self, request: TaskRunRequest) -> EvaluationTask:
        batch = self.get_batch(request.batch_id)
        model = self._get_model(request.model_id)
        task_id = f"task-{len(self.list_tasks()) + 1:03d}"
        running = EvaluationTask(
            task_id=task_id,
            model_id=model.model_id,
            batch_id=batch.batch_id,
            track=batch.track,
            status=TaskStatus.RUNNING,
        )
        self.repo.save("tasks", task_id, running)

        sample_outcomes = [self._score_sample(model=model, sample=sample, track=batch.track) for sample in batch.samples]
        metrics = self._aggregate_metrics(sample_outcomes)
        report = self._build_report(task_id=task_id, model=model, batch=batch, outcomes=sample_outcomes, metrics=metrics)
        self.repo.save("reports", report.report_id, report)

        finished = running.model_copy(
            update={
                "status": TaskStatus.SUCCEEDED,
                "metrics": metrics,
                "report_id": report.report_id,
                "sample_outcomes": sample_outcomes,
            }
        )
        self.repo.save("tasks", task_id, finished)
        return finished

    def leaderboard(self, track: TrackKind | None = None) -> list[LeaderboardEntry]:
        models = {model.model_id: model for model in self.list_models()}
        entries: list[LeaderboardEntry] = []
        for task in self.list_tasks():
            if task.status != TaskStatus.SUCCEEDED or task.metrics is None:
                continue
            if track is not None and task.track != track:
                continue
            model = models[task.model_id]
            entries.append(
                LeaderboardEntry(
                    task_id=task.task_id,
                    model_id=task.model_id,
                    model_name=model.name,
                    batch_id=task.batch_id,
                    track=task.track,
                    composite_score=task.metrics.composite_score,
                    mse=task.metrics.mse,
                    smape=task.metrics.smape,
                    mean_latency_ms=task.metrics.mean_latency_ms,
                )
            )
        return sorted(entries, key=lambda item: item.composite_score, reverse=True)

    def overview(self) -> DashboardOverview:
        batches = sorted(self.list_batches(), key=lambda item: item.created_at, reverse=True)
        tasks = self.list_tasks()
        return DashboardOverview(
            tracks=self.list_tracks(),
            models=self.list_models(),
            batches=batches[:6],
            recent_tasks=tasks[:6],
            leaderboard=self.leaderboard()[:6],
        )

    def _get_model(self, model_id: str) -> ModelRecord:
        if not self.repo.exists("models", model_id):
            raise NotFoundError(f"model {model_id} not found")
        return ModelRecord.model_validate(self.repo.load("models", model_id))

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
            track_tags.append("order_shuffle")
        elif track == TrackKind.NOISE_ROBUSTNESS:
            history = [round(value + rng.gauss(0.0, noise_level * 1.8), 4) for value in history]
            covariates["noise_probe"] = [round(rng.gauss(0.0, 1.0), 4) for _ in range(total_length)]
            track_tags.append("noise_augmented")
        elif track == TrackKind.COST_INTENSIVE:
            covariates["calendar_signal"] = [round(math.sin(2 * math.pi * step / 24), 4) for step in range(total_length)]
            covariates["load_signal"] = [round(0.2 * step / total_length + rng.random(), 4) for step in range(total_length)]
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

    def _validate_samples(self, samples: list[SeriesSample], context_length: int, horizon: int) -> ValidationReport:
        issues: list[str] = []
        for sample in samples:
            if len(sample.history) != context_length:
                issues.append(f"{sample.sample_id}: context length mismatch")
            if len(sample.target) != horizon:
                issues.append(f"{sample.sample_id}: horizon length mismatch")
            if not all(math.isfinite(value) for value in sample.history + sample.target):
                issues.append(f"{sample.sample_id}: non-finite value found")
            if max(sample.history) - min(sample.history) < 0.3:
                issues.append(f"{sample.sample_id}: low variance sequence")
        return ValidationReport(passed=not issues, issues=issues)

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
        else:
            if covariates:
                first_name, first_signal = next(iter(covariates.items()))
                prediction = [round(first_signal[len(history) + index], 4) for index in range(horizon)]
            else:
                prediction = [round(history[-1], 4)] * horizon
            latency = 5.5 + horizon * 0.16 + len(covariates) * 0.6
            notes = {"decision": "trust first covariate", "first_covariate": next(iter(covariates.keys()), "none")}

        if track == TrackKind.COST_INTENSIVE:
            latency *= 1.8
            base_tokens += 150
        elif track == TrackKind.NOISE_ROBUSTNESS:
            latency *= 1.2

        token_count = int(base_tokens + horizon * 3)
        return ExecutionResult(prediction=prediction, latency_ms=latency, token_count=token_count, notes=notes)

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
        bad_cases = [
            f"{item.sample_id}: mse={item.mse:.4f}, 备注={item.notes}"
            for item in worst
        ]
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
