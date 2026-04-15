from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain import DatasetLoadRequest, DatasetSourceType, SeriesSample, TrackSpec


class DataLoaderError(RuntimeError):
    pass


class DatasetLoader(ABC):
    source_type: DatasetSourceType

    @abstractmethod
    def load_samples(self, request: DatasetLoadRequest, track_spec: TrackSpec | None = None) -> list[SeriesSample]:
        raise NotImplementedError
