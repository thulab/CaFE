from __future__ import annotations

import random
from pathlib import Path
from uuid import uuid4

from ..config import AppSettings, get_settings
from ..domain import BatchGenerationRequest, DatasetBatch, DatasetLoadRequest, TrackKind, TrackSpec, ValidationReport
from ..errors import BenchmarkError, NotFoundError
from ..storage import FileRepository
from .data_loader import DataLoaderError, build_default_dataset_loader_registry
from .processors import DataProcessorError, build_default_dataset_processor_pipeline
from .synthetic import SyntheticDatasetGenerator
from .validators import DataValidationContext, build_default_dataset_validation_pipeline


class DataManager:
    def __init__(self, runtime_root: Path, settings: AppSettings | None = None, repository: FileRepository | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo = repository or FileRepository(runtime_root)
        self.track_specs = self._build_track_specs()
        self.dataset_loader_registry = build_default_dataset_loader_registry()
        self.dataset_processor_pipeline = build_default_dataset_processor_pipeline()
        self.dataset_validation_pipeline = build_default_dataset_validation_pipeline()
        self.synthetic_generator = SyntheticDatasetGenerator(self.settings)

    def list_tracks(self) -> list[TrackSpec]:
        return list(self.track_specs.values())

    def get_track_spec(self, track: TrackKind) -> TrackSpec:
        return self.track_specs[track]

    def list_batches(self) -> list[DatasetBatch]:
        batches = [DatasetBatch.model_validate(item) for item in self.repo.list("batches")]
        return sorted(batches, key=lambda item: item.created_at, reverse=True)

    def get_batch(self, batch_id: str) -> DatasetBatch:
        if not self.repo.exists("batches", batch_id):
            raise NotFoundError(f"batch {batch_id} not found")
        return DatasetBatch.model_validate(self.repo.load("batches", batch_id))

    def generate_batch(self, request: BatchGenerationRequest) -> DatasetBatch:
        if request.sample_count <= 0:
            raise BenchmarkError("sample_count must be positive")
        spec = self.track_specs[request.track]
        context_length = request.context_length or spec.default_context_length
        horizon = request.horizon or spec.default_horizon
        batch_id = ""
        selected_seed = request.seed
        samples = []
        validation = ValidationReport(passed=False, issues=["generation not attempted"])

        for attempt in range(self.settings.benchmark.synthetic_generation.max_generation_attempts):
            attempt_seed = request.seed + attempt
            selected_seed = attempt_seed
            batch_id = f"{request.track.value}-{attempt_seed}-{uuid4().hex[:8]}"
            samples = [
                self.synthetic_generator.generate_sample(
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
                "generated dataset failed validation after "
                f"{self.settings.benchmark.synthetic_generation.max_generation_attempts} attempts: {validation.issues}"
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

    def _build_track_specs(self) -> dict[TrackKind, TrackSpec]:
        specs: dict[TrackKind, TrackSpec] = {}
        for track in TrackKind:
            config = self.settings.benchmark.tracks[track.value]
            specs[track] = TrackSpec(track=track, **config.model_dump())
        return specs

    def _validate_dataset(self, samples, context_length: int, horizon: int) -> ValidationReport:
        context = DataValidationContext(context_length=context_length, horizon=horizon)
        return self.dataset_validation_pipeline.validate(samples, context)
