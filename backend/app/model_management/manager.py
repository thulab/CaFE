from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ..config import AppSettings, get_settings
from ..domain import (
    HuggingFaceConfig,
    HuggingFaceModelRegistrationRequest,
    HuggingFaceTask,
    ModelAdapter,
    ModelRecord,
    ModelRegistrationRequest,
    ModelRuntimeStatus,
    SeriesSample,
    TrackKind,
    utc_now,
)
from ..errors import BenchmarkError, InternalBenchmarkError, NotFoundError
from ..storage import FileRepository
from .huggingface import HuggingFaceForecast, HuggingFaceModelRunner, HuggingFaceRunnerError


@dataclass
class ExecutionResult:
    prediction: list[float]
    latency_ms: float
    token_count: int
    notes: dict[str, str]


class ModelManager:
    def __init__(self, runtime_root: Path, settings: AppSettings | None = None, repository: FileRepository | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo = repository or FileRepository(runtime_root)
        self.huggingface_runner_factory = HuggingFaceModelRunner
        self._huggingface_runners: dict[str, HuggingFaceModelRunner] = {}
        self._bootstrap_builtin_models()

    def list_models(self) -> list[ModelRecord]:
        models = [ModelRecord.model_validate(item) for item in self.repo.list("models")]
        return sorted(models, key=lambda item: item.created_at, reverse=True)

    def get_model(self, model_id: str) -> ModelRecord:
        if not self.repo.exists("models", model_id):
            raise NotFoundError(f"model {model_id} not found")
        return ModelRecord.model_validate(self.repo.load("models", model_id))

    def register_model(self, request: ModelRegistrationRequest) -> ModelRecord:
        if request.adapter in {ModelAdapter.HUGGINGFACE_TEXT_GENERATION, ModelAdapter.HUGGINGFACE_CHRONOS2}:
            raise BenchmarkError("use the dedicated Hugging Face registration endpoint for huggingface models")
        self._ensure_model_not_exists(request.model_id)
        record = ModelRecord(**request.model_dump(), runtime_status=ModelRuntimeStatus.READY)
        self.repo.save("models", record.model_id, record)
        return record

    def register_huggingface_model(self, request: HuggingFaceModelRegistrationRequest) -> ModelRecord:
        model_id = self._normalize_model_id(request.model_id or request.repo_id)
        self._ensure_model_not_exists(model_id)
        adapter = ModelAdapter.HUGGINGFACE_CHRONOS2 if request.task == HuggingFaceTask.CHRONOS2 else ModelAdapter.HUGGINGFACE_TEXT_GENERATION
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
        model = self.get_model(model_id)
        if not self._is_huggingface_adapter(model.adapter):
            if model.runtime_status != ModelRuntimeStatus.READY:
                model = model.model_copy(update={"runtime_status": ModelRuntimeStatus.READY, "last_error": None})
                self.repo.save("models", model.model_id, model)
            return model

        try:
            self.get_or_load_huggingface_runner(model)
        except InternalBenchmarkError as exc:
            updated = model.model_copy(update={"runtime_status": ModelRuntimeStatus.LOAD_FAILED, "last_error": str(exc)})
            self.repo.save("models", model.model_id, updated)
            raise
        except Exception as exc:
            updated = model.model_copy(update={"runtime_status": ModelRuntimeStatus.LOAD_FAILED, "last_error": str(exc)})
            self.repo.save("models", model.model_id, updated)
            raise InternalBenchmarkError(str(exc)) from exc

        updated = model.model_copy(
            update={"runtime_status": ModelRuntimeStatus.READY, "last_loaded_at": utc_now(), "last_error": None}
        )
        self.repo.save("models", model.model_id, updated)
        return updated

    def execute_model(self, model: ModelRecord, sample: SeriesSample, track: TrackKind) -> ExecutionResult:
        scoring = self.settings.benchmark.scoring
        stubs = self.settings.benchmark.stub_models
        history = sample.history
        horizon = len(sample.target)
        covariates = sample.covariates
        dominant_period = sample.truth.dominant_period
        base_tokens = scoring.base_tokens + len(history) // scoring.history_token_divisor + len(covariates) * scoring.covariate_token_weight

        if model.adapter == ModelAdapter.SEASONAL_NAIVE:
            if len(history) >= dominant_period:
                season = history[-dominant_period:]
                prediction = [round(season[index % len(season)], 4) for index in range(horizon)]
            else:
                prediction = [round(history[-1], 4)] * horizon
            latency = stubs.seasonal_naive.latency_base + horizon * stubs.seasonal_naive.latency_per_horizon + len(covariates) * stubs.seasonal_naive.latency_per_covariate
            notes = {"decision": "repeat dominant period"}
        elif model.adapter == ModelAdapter.RECENT_MEAN:
            window = history[-min(stubs.recent_mean.window, len(history)) :]
            baseline = sum(window) / len(window)
            prediction = [round(baseline, 4)] * horizon
            latency = stubs.recent_mean.latency_base + horizon * stubs.recent_mean.latency_per_horizon
            notes = {"decision": "smooth recent window"}
        elif model.adapter == ModelAdapter.COVARIATE_TRAP:
            if covariates:
                _, first_signal = next(iter(covariates.items()))
                prediction = [round(first_signal[len(history) + index], 4) for index in range(horizon)]
            else:
                prediction = [round(history[-1], 4)] * horizon
            latency = stubs.covariate_trap.latency_base + horizon * stubs.covariate_trap.latency_per_horizon + len(covariates) * stubs.covariate_trap.latency_per_covariate
            notes = {"decision": "trust first covariate", "first_covariate": next(iter(covariates.keys()), "none")}
        elif model.adapter in {ModelAdapter.HUGGINGFACE_TEXT_GENERATION, ModelAdapter.HUGGINGFACE_CHRONOS2}:
            huggingface_result = self.execute_huggingface_model(model, sample, track)
            prediction = huggingface_result.prediction
            latency = huggingface_result.latency_ms
            base_tokens = huggingface_result.token_count
            notes = huggingface_result.notes
        else:
            raise BenchmarkError(f"unsupported model adapter {model.adapter}")

        if track == TrackKind.COST_INTENSIVE:
            latency *= scoring.cost_track_latency_multiplier
            base_tokens += scoring.cost_track_token_bonus
        elif track == TrackKind.NOISE_ROBUSTNESS:
            latency *= scoring.noise_track_latency_multiplier

        return ExecutionResult(
            prediction=prediction,
            latency_ms=latency,
            token_count=int(base_tokens + horizon * scoring.token_per_horizon),
            notes=notes,
        )

    def execute_huggingface_model(self, model: ModelRecord, sample: SeriesSample, track: TrackKind) -> HuggingFaceForecast:
        if model.huggingface is None:
            raise InternalBenchmarkError(f"model {model.model_id} has no huggingface config")
        runner = self.get_or_load_huggingface_runner(model)
        return runner.forecast(sample=sample, track=track)

    def execute_huggingface_model_batch(
        self,
        model: ModelRecord,
        samples: list[SeriesSample],
        track: TrackKind,
    ) -> list[HuggingFaceForecast]:
        if model.huggingface is None:
            raise InternalBenchmarkError(f"model {model.model_id} has no huggingface config")
        runner = self.get_or_load_huggingface_runner(model)
        return runner.forecast_batch(samples=samples, track=track)

    def get_or_load_huggingface_runner(self, model: ModelRecord) -> HuggingFaceModelRunner:
        if model.huggingface is None:
            raise InternalBenchmarkError(f"model {model.model_id} has no huggingface config")
        runner = self._huggingface_runners.get(model.model_id)
        if runner is None:
            runner = self.huggingface_runner_factory(model.huggingface)
            self._huggingface_runners[model.model_id] = runner
        try:
            runner.load()
        except HuggingFaceRunnerError as exc:
            raise InternalBenchmarkError(str(exc)) from exc
        return runner

    def _bootstrap_builtin_models(self) -> None:
        for builtin in self.settings.benchmark.builtin_models:
            record = ModelRecord(
                model_id=builtin.model_id,
                name=builtin.name,
                adapter=ModelAdapter(builtin.adapter),
                source_type=builtin.source_type,
                manual=builtin.manual,
                capabilities=builtin.capabilities,
                runtime_status=ModelRuntimeStatus.READY,
            )
            if not self.repo.exists("models", record.model_id):
                self.repo.save("models", record.model_id, record)

    def _ensure_model_not_exists(self, model_id: str) -> None:
        if self.repo.exists("models", model_id):
            raise BenchmarkError(f"model {model_id} already exists")

    def _normalize_model_id(self, raw_value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", raw_value.lower()).strip("-")
        return normalized or f"model-{uuid4().hex[:6]}"

    def _is_huggingface_adapter(self, adapter: ModelAdapter) -> bool:
        return adapter in {ModelAdapter.HUGGINGFACE_TEXT_GENERATION, ModelAdapter.HUGGINGFACE_CHRONOS2}
