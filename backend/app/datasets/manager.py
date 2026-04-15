from __future__ import annotations

import random
from pathlib import Path
from uuid import uuid4

from ..config import AppSettings, get_settings
from ..errors import BenchmarkError, NotFoundError
from ..storage import FileRepository
from .data_loader import DataLoaderError, build_default_dataset_loader_registry
from .domain import (
    BatchGenerationRequest,
    DatasetBatch,
    DatasetFeatureProfile,
    DatasetLoadRequest,
    DatasetRecord,
    DatasetSourceRecord,
    DatasetSourceType,
    ExecutionConstraint,
    NoiseMode,
    TrackKind,
    TrackSpec,
    TrackTemplateKind,
    ValidationReport,
)
from .processors import DataProcessorError, build_default_dataset_processor_pipeline
from .synthetic import SyntheticDatasetGenerator
from .validators import DataValidationContext, build_default_dataset_validation_pipeline


class DataManager:
    def __init__(self, runtime_root: Path, settings: AppSettings | None = None, repository: FileRepository | None = None) -> None:
        self.settings = settings or get_settings()
        self.repo = repository or FileRepository(runtime_root)
        self.track_specs = self._build_track_specs()
        self.track_aliases = self._build_track_aliases()
        self.dataset_loader_registry = build_default_dataset_loader_registry()
        self.dataset_processor_pipeline = build_default_dataset_processor_pipeline()
        self.dataset_validation_pipeline = build_default_dataset_validation_pipeline()
        self.synthetic_generator = SyntheticDatasetGenerator(self.settings)

    def list_tracks(self) -> list[TrackSpec]:
        return list(self.track_specs.values())

    def get_track_spec(self, track: TrackKind | str) -> TrackSpec:
        key = self._normalize_track_key(track)
        return self.track_specs[key]

    def list_batches(self) -> list[DatasetBatch]:
        batches = [DatasetBatch.parse_obj(item) for item in self.repo.list("batches")]
        return sorted(batches, key=lambda item: item.created_at, reverse=True)

    def get_batch(self, batch_id: str) -> DatasetBatch:
        if not self.repo.exists("batches", batch_id):
            raise NotFoundError(f"batch {batch_id} not found")
        return DatasetBatch.parse_obj(self.repo.load("batches", batch_id))

    def generate_batch(self, request: BatchGenerationRequest) -> DatasetBatch:
        if request.sample_count <= 0:
            raise BenchmarkError("sample_count must be positive")
        track_spec = self._resolve_track_spec(request.track_variant_id, request.track)
        input_length = request.input_length or track_spec.default_context_length
        prediction_length = request.prediction_length or track_spec.default_horizon
        batch_id = ""
        selected_seed = request.seed
        samples = []
        validation = ValidationReport(passed=False, issues=["generation not attempted"])

        for attempt in range(self.settings.benchmark.synthetic_generation.max_generation_attempts):
            attempt_seed = request.seed + attempt
            selected_seed = attempt_seed
            batch_id = f"{track_spec.track_variant_id}-{attempt_seed}-{uuid4().hex[:8]}"
            samples = [
                self.synthetic_generator.generate_sample(
                    rng=random.Random(attempt_seed * 1000 + index),
                    sample_id=f"{batch_id}-sample-{index + 1:03d}",
                    track_spec=track_spec,
                    input_length=input_length,
                    prediction_length=prediction_length,
                )
                for index in range(request.sample_count)
            ]
            validation = self._validate_dataset(samples, input_length=input_length, prediction_length=prediction_length)
            if validation.passed:
                break

        if not validation.passed:
            raise BenchmarkError(
                "generated dataset failed validation after "
                f"{self.settings.benchmark.synthetic_generation.max_generation_attempts} attempts: {validation.issues}"
            )

        source = self._save_source_record(
            source_type=DatasetSourceType.SYNTHETIC,
            source_path=None,
            source_schema=self._source_schema_for_track(track_spec),
            metadata={"seed": selected_seed, "track_variant_id": track_spec.track_variant_id},
        )
        feature_profile = self._build_feature_profile(samples)
        batch = self._build_batch(
            batch_id=batch_id,
            track_spec=track_spec,
            source=source,
            seed=selected_seed,
            sample_count=request.sample_count,
            input_length=input_length,
            prediction_length=prediction_length,
            samples=samples,
            validation=validation,
            feature_profile=feature_profile,
        )
        self._save_dataset_record(batch=batch)
        self.repo.save("batches", batch.batch_id, batch)
        return batch

    def load_batch(self, request: DatasetLoadRequest) -> DatasetBatch:
        if request.input_length <= 0:
            raise BenchmarkError("context_length must be positive")
        if request.prediction_length <= 0:
            raise BenchmarkError("horizon must be positive")
        if request.max_samples is not None and request.max_samples <= 0:
            raise BenchmarkError("max_samples must be positive when provided")

        track_spec = self._resolve_track_spec(request.track_variant_id, request.track)
        try:
            loader = self.dataset_loader_registry.get(request.source_type)
            samples = loader.load_samples(request, track_spec)
        except DataLoaderError as exc:
            raise BenchmarkError(str(exc)) from exc
        except ValueError as exc:
            raise BenchmarkError(str(exc)) from exc

        try:
            samples = self.dataset_processor_pipeline.process(samples, request, track_spec)
        except DataProcessorError as exc:
            raise BenchmarkError(str(exc)) from exc

        batch_id = f"{request.batch_id_prefix}-{track_spec.track_variant_id}-{uuid4().hex[:8]}"
        validation = self._validate_dataset(samples, input_length=request.input_length, prediction_length=request.prediction_length)
        if not validation.passed:
            raise BenchmarkError(f"loaded dataset failed validation and must be regenerated: {validation.issues}")
        source = self._save_source_record(
            source_type=request.source_type,
            source_path=self._load_source_path(request),
            source_schema=self._source_schema_from_request(request, track_spec),
            metadata={"track_variant_id": track_spec.track_variant_id},
        )
        feature_profile = self._build_feature_profile(samples)
        batch = self._build_batch(
            batch_id=batch_id,
            track_spec=track_spec,
            source=source,
            seed=0,
            sample_count=len(samples),
            input_length=request.input_length,
            prediction_length=request.prediction_length,
            samples=samples,
            validation=validation,
            feature_profile=feature_profile,
        )
        self._save_dataset_record(batch=batch)
        self.repo.save("batches", batch.batch_id, batch)
        return batch

    def _build_track_specs(self) -> dict[str, TrackSpec]:
        template_specs = [
            self._make_track_spec(
                track_variant_id="univariate_forecast.clean",
                template_kind=TrackTemplateKind.UNIVARIATE_FORECAST,
                noise_mode=NoiseMode.CLEAN,
                runtime_track=TrackKind.FORECAST_ACCURACY,
                config_key="forecast_accuracy",
                input_channels=["target"],
                target_channels=["target"],
                future_known_channels=[],
                execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="univariate_forecast.noisy",
                template_kind=TrackTemplateKind.UNIVARIATE_FORECAST,
                noise_mode=NoiseMode.NOISY,
                runtime_track=TrackKind.NOISE_ROBUSTNESS,
                config_key="noise_robustness",
                input_channels=["target"],
                target_channels=["target"],
                future_known_channels=[],
                execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="multivariate_forecast_all_to_all.clean",
                template_kind=TrackTemplateKind.MULTIVARIATE_FORECAST_ALL_TO_ALL,
                noise_mode=NoiseMode.CLEAN,
                runtime_track=TrackKind.FORECAST_ACCURACY,
                config_key="forecast_accuracy",
                input_channels=["series_1", "series_2", "series_3"],
                target_channels=["series_1", "series_2", "series_3"],
                future_known_channels=[],
                execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="multivariate_forecast_all_to_all.noisy",
                template_kind=TrackTemplateKind.MULTIVARIATE_FORECAST_ALL_TO_ALL,
                noise_mode=NoiseMode.NOISY,
                runtime_track=TrackKind.NOISE_ROBUSTNESS,
                config_key="noise_robustness",
                input_channels=["series_1", "series_2", "series_3"],
                target_channels=["series_1", "series_2", "series_3"],
                future_known_channels=[],
                execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="multivariate_forecast_all_to_subset.clean",
                template_kind=TrackTemplateKind.MULTIVARIATE_FORECAST_ALL_TO_SUBSET,
                noise_mode=NoiseMode.CLEAN,
                runtime_track=TrackKind.COVARIATE_ROBUSTNESS,
                config_key="covariate_robustness",
                input_channels=["series_1", "series_2", "series_3", "series_4", "series_5"],
                target_channels=["series_4", "series_5"],
                future_known_channels=[],
                execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="multivariate_forecast_all_to_subset.noisy",
                template_kind=TrackTemplateKind.MULTIVARIATE_FORECAST_ALL_TO_SUBSET,
                noise_mode=NoiseMode.NOISY,
                runtime_track=TrackKind.COVARIATE_ROBUSTNESS,
                config_key="covariate_robustness",
                input_channels=["series_1", "series_2", "series_3", "series_4", "series_5"],
                target_channels=["series_4", "series_5"],
                future_known_channels=[],
                execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="multivariate_forecast_with_future_covariates.clean",
                template_kind=TrackTemplateKind.MULTIVARIATE_FORECAST_WITH_FUTURE_COVARIATES,
                noise_mode=NoiseMode.CLEAN,
                runtime_track=TrackKind.COVARIATE_ROBUSTNESS,
                config_key="covariate_robustness",
                input_channels=["series_1", "series_2", "series_3", "series_4", "series_5"],
                target_channels=["series_4", "series_5"],
                future_known_channels=["series_1", "series_2", "series_3"],
                execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="multivariate_forecast_with_future_covariates.noisy",
                template_kind=TrackTemplateKind.MULTIVARIATE_FORECAST_WITH_FUTURE_COVARIATES,
                noise_mode=NoiseMode.NOISY,
                runtime_track=TrackKind.COVARIATE_ROBUSTNESS,
                config_key="covariate_robustness",
                input_channels=["series_1", "series_2", "series_3", "series_4", "series_5"],
                target_channels=["series_4", "series_5"],
                future_known_channels=["series_1", "series_2", "series_3"],
                execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="multivariate_forecast_via_univariate.clean",
                template_kind=TrackTemplateKind.MULTIVARIATE_FORECAST_VIA_UNIVARIATE,
                noise_mode=NoiseMode.CLEAN,
                runtime_track=TrackKind.FORECAST_ACCURACY,
                config_key="forecast_accuracy",
                input_channels=["series_1", "series_2", "series_3"],
                target_channels=["series_1", "series_2", "series_3"],
                future_known_channels=[],
                execution_constraint=ExecutionConstraint.PER_CHANNEL_UNIVARIATE,
            ),
            self._make_track_spec(
                track_variant_id="multivariate_forecast_via_univariate.noisy",
                template_kind=TrackTemplateKind.MULTIVARIATE_FORECAST_VIA_UNIVARIATE,
                noise_mode=NoiseMode.NOISY,
                runtime_track=TrackKind.NOISE_ROBUSTNESS,
                config_key="noise_robustness",
                input_channels=["series_1", "series_2", "series_3"],
                target_channels=["series_1", "series_2", "series_3"],
                future_known_channels=[],
                execution_constraint=ExecutionConstraint.PER_CHANNEL_UNIVARIATE,
            ),
        ]
        return {item.track_variant_id: item for item in template_specs}

    def _make_track_spec(
        self,
        *,
        track_variant_id: str,
        template_kind: TrackTemplateKind,
        noise_mode: NoiseMode,
        runtime_track: TrackKind,
        config_key: str,
        input_channels: list[str],
        target_channels: list[str],
        future_known_channels: list[str],
        execution_constraint: ExecutionConstraint,
    ) -> TrackSpec:
        config = self.settings.benchmark.tracks[config_key]
        return TrackSpec(
            track=runtime_track,
            track_variant_id=track_variant_id,
            track_template_kind=template_kind,
            noise_mode=noise_mode,
            execution_constraint=execution_constraint,
            name=f"{template_kind.value}.{noise_mode.value}",
            description=self._describe_track(template_kind, noise_mode),
            fairness_policy=config.fairness_policy,
            default_context_length=config.default_context_length,
            default_horizon=config.default_horizon,
            suggested_sample_count=config.suggested_sample_count,
            input_channels=input_channels,
            target_channels=target_channels,
            future_known_channels=future_known_channels,
            knobs=list(config.knobs),
            aliases=[runtime_track.value] if noise_mode == NoiseMode.CLEAN else [],
        )

    def _describe_track(self, template_kind: TrackTemplateKind, noise_mode: NoiseMode) -> str:
        suffix = "，带噪声版本" if noise_mode == NoiseMode.NOISY else "，无噪声版本"
        descriptions = {
            TrackTemplateKind.UNIVARIATE_FORECAST: "单变量预测，单通道输入预测同一通道",
            TrackTemplateKind.MULTIVARIATE_FORECAST_ALL_TO_ALL: "多变量预测1，1/2/3 输入预测 1/2/3",
            TrackTemplateKind.MULTIVARIATE_FORECAST_ALL_TO_SUBSET: "多变量预测2，1/2/3/4/5 输入预测 4/5",
            TrackTemplateKind.MULTIVARIATE_FORECAST_WITH_FUTURE_COVARIATES: "多变量预测3，带未来已知 1/2/3 协变量预测 4/5",
            TrackTemplateKind.MULTIVARIATE_FORECAST_VIA_UNIVARIATE: "多变量预测4，使用单变量方式分别完成 1/2/3 的预测",
        }
        return descriptions[template_kind] + suffix

    def _build_track_aliases(self) -> dict[str, str]:
        aliases = {spec.track_variant_id: spec.track_variant_id for spec in self.track_specs.values()}
        aliases.update(
            {
                TrackKind.FORECAST_ACCURACY.value: "univariate_forecast.clean",
                TrackKind.NOISE_ROBUSTNESS.value: "univariate_forecast.noisy",
                TrackKind.COVARIATE_ROBUSTNESS.value: "multivariate_forecast_all_to_subset.clean",
                TrackKind.COST_INTENSIVE.value: "multivariate_forecast_with_future_covariates.clean",
            }
        )
        for spec in self.track_specs.values():
            for alias in spec.aliases:
                aliases.setdefault(alias, spec.track_variant_id)
        return aliases

    def _validate_dataset(self, samples, input_length: int, prediction_length: int) -> ValidationReport:
        context = DataValidationContext(context_length=input_length, horizon=prediction_length)
        return self.dataset_validation_pipeline.validate(samples, context)

    def _resolve_track_spec(self, track_variant_id: str | None, track: TrackKind | str) -> TrackSpec:
        key = track_variant_id or self._normalize_track_key(track)
        if key not in self.track_specs:
            raise BenchmarkError(f"unsupported track variant {key}")
        return self.track_specs[key]

    def _normalize_track_key(self, track: TrackKind | str) -> str:
        key = getattr(track, "value", str(track))
        return self.track_aliases.get(key, key)

    def _save_source_record(
        self,
        *,
        source_type: DatasetSourceType,
        source_path: str | None,
        source_schema: dict[str, object],
        metadata: dict[str, object],
    ) -> DatasetSourceRecord:
        source = DatasetSourceRecord(
            source_id=f"source-{uuid4().hex[:8]}",
            source_type=source_type,
            source_path=source_path,
            source_schema=source_schema,
            metadata=metadata,
        )
        self.repo.save("dataset_sources", source.source_id, source)
        return source

    def _build_batch(
        self,
        *,
        batch_id: str,
        track_spec: TrackSpec,
        source: DatasetSourceRecord,
        seed: int,
        sample_count: int,
        input_length: int,
        prediction_length: int,
        samples,
        validation: ValidationReport,
        feature_profile: DatasetFeatureProfile,
    ) -> DatasetBatch:
        return DatasetBatch(
            batch_id=batch_id,
            track=track_spec.track,
            track_variant_id=track_spec.track_variant_id,
            track_template_kind=track_spec.track_template_kind,
            noise_mode=track_spec.noise_mode,
            execution_constraint=track_spec.execution_constraint,
            input_channels=list(track_spec.input_channels),
            target_channels=list(track_spec.target_channels),
            future_known_channels=list(track_spec.future_known_channels),
            policy=track_spec.fairness_policy,
            seed=seed,
            source_type=source.source_type,
            source_id=source.source_id,
            dataset_id=f"dataset-{uuid4().hex[:8]}",
            sample_count=sample_count,
            input_length=input_length,
            prediction_length=prediction_length,
            context_length=input_length,
            horizon=prediction_length,
            samples=samples,
            validation=validation,
            feature_profile=feature_profile,
        )

    def _save_dataset_record(self, batch: DatasetBatch) -> None:
        record = DatasetRecord(
            dataset_id=batch.dataset_id,
            source_id=batch.source_id,
            batch_id=batch.batch_id,
            track_variant_id=batch.track_variant_id,
            sample_count=batch.sample_count,
            input_length=batch.input_length,
            prediction_length=batch.prediction_length,
            feature_profile=batch.feature_profile,
            metadata={
                "track_template_kind": batch.track_template_kind.value,
                "noise_mode": batch.noise_mode.value,
                "execution_constraint": batch.execution_constraint.value,
            },
        )
        self.repo.save("datasets", record.dataset_id, record)

    def _build_feature_profile(self, samples) -> DatasetFeatureProfile:
        if not samples:
            return DatasetFeatureProfile()
        dominant_periods = sorted({sample.truth.dominant_period for sample in samples})
        trend_tags = sorted({sample.truth.trend_type for sample in samples})
        mean_noise = round(sum(sample.truth.noise_level for sample in samples) / len(samples), 4)
        return DatasetFeatureProfile(
            trend_tags=trend_tags,
            seasonality_tags=["synthetic_periodic" if dominant_periods else "unknown"],
            dominant_periods=dominant_periods,
            noise_level=mean_noise,
            missing_rate=0.0,
            outlier_rate=0.0,
            feature_summary={
                "sample_count": len(samples),
                "difficulty_levels": sorted({sample.truth.difficulty for sample in samples}),
                "channels": sorted(
                    {
                        *samples[0].channel_layout.input_channels,
                        *samples[0].channel_layout.target_channels,
                        *samples[0].channel_layout.future_known_channels,
                    }
                ),
            },
        )

    def _source_schema_for_track(self, track_spec: TrackSpec) -> dict[str, object]:
        return {
            "input_channels": list(track_spec.input_channels),
            "target_channels": list(track_spec.target_channels),
            "future_known_channels": list(track_spec.future_known_channels),
            "noise_mode": track_spec.noise_mode.value,
        }

    def _source_schema_from_request(self, request: DatasetLoadRequest, track_spec: TrackSpec) -> dict[str, object]:
        base = self._source_schema_for_track(track_spec)
        if hasattr(request, "target_columns"):
            base["target_columns"] = list(getattr(request, "target_columns"))
        if hasattr(request, "input_columns"):
            base["input_columns"] = list(getattr(request, "input_columns"))
        if hasattr(request, "future_known_columns"):
            base["future_known_columns"] = list(getattr(request, "future_known_columns"))
        return base

    def _load_source_path(self, request: DatasetLoadRequest) -> str | None:
        if hasattr(request, "csv_path"):
            return str(getattr(request, "csv_path"))
        return None
