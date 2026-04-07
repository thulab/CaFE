from __future__ import annotations

from ..domain import DataProcessorConfig, DataProcessorType, DatasetLoadRequest, SeriesSample
from .base import DataProcessor, DataProcessorError
from .helpers import rebuild_sample


class IdentityProcessor(DataProcessor):
    processor_type = DataProcessorType.IDENTITY

    def process(
        self,
        samples: list[SeriesSample],
        request: DatasetLoadRequest,
        config: DataProcessorConfig,
    ) -> list[SeriesSample]:
        return [
            rebuild_sample(
                sample,
                request.track,
                list(sample.history),
                list(sample.target),
                {name: list(values) for name, values in sample.covariates.items()},
                extra_tags=["processor_identity"],
                processor_note={"name": self.processor_type.value},
            )
            for sample in samples
        ]


class ScaleProcessor(DataProcessor):
    processor_type = DataProcessorType.SCALE

    def process(
        self,
        samples: list[SeriesSample],
        request: DatasetLoadRequest,
        config: DataProcessorConfig,
    ) -> list[SeriesSample]:
        factor = float(config.params.get("factor", 1.0))
        include_covariates = bool(config.params.get("include_covariates", True))
        return [
            rebuild_sample(
                sample,
                request.track,
                [round(value * factor, 6) for value in sample.history],
                [round(value * factor, 6) for value in sample.target],
                {
                    name: [round(value * factor, 6) for value in values] if include_covariates else list(values)
                    for name, values in sample.covariates.items()
                },
                extra_tags=["processor_scale"],
                processor_note={
                    "name": self.processor_type.value,
                    "factor": factor,
                    "include_covariates": include_covariates,
                },
            )
            for sample in samples
        ]


class ClipProcessor(DataProcessor):
    processor_type = DataProcessorType.CLIP

    def process(
        self,
        samples: list[SeriesSample],
        request: DatasetLoadRequest,
        config: DataProcessorConfig,
    ) -> list[SeriesSample]:
        min_value = config.params.get("min_value")
        max_value = config.params.get("max_value")
        include_covariates = bool(config.params.get("include_covariates", True))
        if min_value is None or max_value is None:
            raise DataProcessorError("clip processor requires both min_value and max_value")
        min_value = float(min_value)
        max_value = float(max_value)
        if min_value > max_value:
            raise DataProcessorError("clip processor requires min_value <= max_value")

        def clamp(value: float) -> float:
            return round(min(max(value, min_value), max_value), 6)

        return [
            rebuild_sample(
                sample,
                request.track,
                [clamp(value) for value in sample.history],
                [clamp(value) for value in sample.target],
                {
                    name: [clamp(value) for value in values] if include_covariates else list(values)
                    for name, values in sample.covariates.items()
                },
                extra_tags=["processor_clip"],
                processor_note={
                    "name": self.processor_type.value,
                    "min_value": min_value,
                    "max_value": max_value,
                    "include_covariates": include_covariates,
                },
            )
            for sample in samples
        ]


class CovariateFilterProcessor(DataProcessor):
    processor_type = DataProcessorType.COVARIATE_FILTER

    def process(
        self,
        samples: list[SeriesSample],
        request: DatasetLoadRequest,
        config: DataProcessorConfig,
    ) -> list[SeriesSample]:
        keep = config.params.get("keep")
        if not isinstance(keep, list) or not all(isinstance(item, str) for item in keep):
            raise DataProcessorError("covariate_filter processor requires params.keep as a list of strings")
        keep_set = set(keep)
        return [
            rebuild_sample(
                sample,
                request.track,
                list(sample.history),
                list(sample.target),
                {name: list(values) for name, values in sample.covariates.items() if name in keep_set},
                extra_tags=["processor_covariate_filter"],
                processor_note={"name": self.processor_type.value, "keep": keep},
            )
            for sample in samples
        ]
