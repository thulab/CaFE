from __future__ import annotations

from .base import DatasetLoader
from .csv_loader import CsvDatasetLoader


class DatasetLoaderRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, DatasetLoader] = {}

    def register(self, loader: DatasetLoader) -> None:
        key = loader.source_type.value
        if key in self._loaders:
            raise ValueError(f"dataset loader already registered for source type {key}")
        self._loaders[key] = loader

    def get(self, source_type: object) -> DatasetLoader:
        key = getattr(source_type, "value", str(source_type))
        if key not in self._loaders:
            available = ", ".join(sorted(self._loaders))
            raise ValueError(f"unsupported dataset source type {key}; available: {available}")
        return self._loaders[key]


def build_default_dataset_loader_registry() -> DatasetLoaderRegistry:
    registry = DatasetLoaderRegistry()
    registry.register(CsvDatasetLoader())
    return registry
