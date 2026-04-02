from __future__ import annotations

from ...domain import DatasetLoadRequest, SeriesSample
from .base import DataProcessorError
from .builtin import ClipProcessor, CovariateFilterProcessor, IdentityProcessor, ScaleProcessor
from .registry import DataProcessorRegistry


class DatasetProcessorPipeline:
    def __init__(self, registry: DataProcessorRegistry) -> None:
        self.registry = registry

    def process(self, samples: list[SeriesSample], request: DatasetLoadRequest) -> list[SeriesSample]:
        current = samples
        for config in request.processors:
            if not config.enabled:
                continue
            try:
                processor = self.registry.get(config.processor_type)
                current = processor.process(current, request, config)
            except ValueError as exc:
                raise DataProcessorError(str(exc)) from exc
        return current


def build_default_dataset_processor_pipeline() -> DatasetProcessorPipeline:
    registry = DataProcessorRegistry()
    registry.register(IdentityProcessor())
    registry.register(ScaleProcessor())
    registry.register(ClipProcessor())
    registry.register(CovariateFilterProcessor())
    return DatasetProcessorPipeline(registry)
