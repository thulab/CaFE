from __future__ import annotations

import math

from .base import DataValidationContext, DataValidator


class ContextLengthValidator(DataValidator):
    name = "context_length"

    def validate(self, samples, context: DataValidationContext) -> list[str]:
        issues: list[str] = []
        for sample in samples:
            if len(sample.history) != context.context_length:
                issues.append(f"{sample.sample_id}: context length mismatch")
        return issues


class HorizonLengthValidator(DataValidator):
    name = "horizon_length"

    def validate(self, samples, context: DataValidationContext) -> list[str]:
        issues: list[str] = []
        for sample in samples:
            if len(sample.target) != context.horizon:
                issues.append(f"{sample.sample_id}: horizon length mismatch")
        return issues


class FiniteValueValidator(DataValidator):
    name = "finite_value"

    def validate(self, samples, context: DataValidationContext) -> list[str]:
        issues: list[str] = []
        for sample in samples:
            if not all(math.isfinite(value) for value in sample.history + sample.target):
                issues.append(f"{sample.sample_id}: non-finite value found")
        return issues


class LowVarianceValidator(DataValidator):
    name = "low_variance"

    def validate(self, samples, context: DataValidationContext) -> list[str]:
        issues: list[str] = []
        for sample in samples:
            if not sample.history:
                issues.append(f"{sample.sample_id}: empty history")
                continue
            if max(sample.history) - min(sample.history) < 0.3:
                issues.append(f"{sample.sample_id}: low variance sequence")
        return issues
