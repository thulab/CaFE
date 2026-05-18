from typing import Protocol


class ModelAdapter(Protocol):
    def forecast(self, sample: dict, model: dict, timeout_seconds: int) -> list[list[float]]:
        ...
