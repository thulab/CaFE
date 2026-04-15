from __future__ import annotations

from ..domain import SeriesSample, ValidationReport
from .base import DataValidationContext
from .builtin import ContextLengthValidator, FiniteValueValidator, HorizonLengthValidator, LowVarianceValidator
from .registry import DataValidatorRegistry


class DatasetValidationPipeline:
    def __init__(self, registry: DataValidatorRegistry) -> None:
        self.registry = registry

    def validate(self, samples: list[SeriesSample], context: DataValidationContext) -> ValidationReport:
        issues: list[str] = []
        for validator in self.registry.list():
            issues.extend(validator.validate(samples, context))
        return ValidationReport(passed=not issues, issues=issues)


def build_default_dataset_validation_pipeline() -> DatasetValidationPipeline:
    registry = DataValidatorRegistry()
    registry.register(ContextLengthValidator())
    registry.register(HorizonLengthValidator())
    registry.register(FiniteValueValidator())
    registry.register(LowVarianceValidator())
    return DatasetValidationPipeline(registry)
