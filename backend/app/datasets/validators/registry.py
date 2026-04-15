from __future__ import annotations

from .base import DataValidator


class DataValidatorRegistry:
    def __init__(self) -> None:
        self._validators: list[DataValidator] = []

    def register(self, validator: DataValidator) -> None:
        if any(item.name == validator.name for item in self._validators):
            raise ValueError(f"data validator already registered with name {validator.name}")
        self._validators.append(validator)

    def list(self) -> list[DataValidator]:
        return list(self._validators)
