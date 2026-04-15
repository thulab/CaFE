from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain import DataProcessorConfig, DataProcessorType, DatasetLoadRequest, SeriesSample, TrackSpec


class DataProcessorError(RuntimeError):
    pass


class DataProcessor(ABC):
    processor_type: DataProcessorType

    @abstractmethod
    def process(
        self,
        samples: list[SeriesSample],
        request: DatasetLoadRequest,
        track_spec: TrackSpec,
        config: DataProcessorConfig,
    ) -> list[SeriesSample]:
        raise NotImplementedError
