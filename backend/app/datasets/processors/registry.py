from __future__ import annotations

from .base import DataProcessor


class DataProcessorRegistry:
    def __init__(self) -> None:
        self._processors: dict[str, DataProcessor] = {}

    def register(self, processor: DataProcessor) -> None:
        key = processor.processor_type.value
        if key in self._processors:
            raise ValueError(f"data processor already registered for type {key}")
        self._processors[key] = processor

    def get(self, processor_type: object) -> DataProcessor:
        key = getattr(processor_type, "value", str(processor_type))
        if key not in self._processors:
            available = ", ".join(sorted(self._processors))
            raise ValueError(f"unsupported data processor type {key}; available: {available}")
        return self._processors[key]
