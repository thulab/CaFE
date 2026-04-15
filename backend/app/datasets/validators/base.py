from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..domain import SeriesSample


@dataclass(frozen=True)
class DataValidationContext:
    context_length: int
    horizon: int


class DataValidator(ABC):
    name: str

    @abstractmethod
    def validate(self, samples: list[SeriesSample], context: DataValidationContext) -> list[str]:
        raise NotImplementedError
