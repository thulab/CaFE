from __future__ import annotations

import numpy as np


def standardize_by_context(
    values: np.ndarray,
    context_length: int,
) -> np.ndarray:
    context = values[:context_length]
    mean = context.mean(axis=0, keepdims=True)
    std = context.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (values - mean) / std


def standardize_hierarchy_by_context(
    values: np.ndarray,
    context_length: int,
) -> np.ndarray:
    context = values[:context_length]
    mean = context.mean(axis=0, keepdims=True)
    centered = values - mean
    scale = float(np.std(context[:, 0]))
    if scale <= 1e-6:
        scale = float(np.mean(np.std(context, axis=0)))
    if scale <= 1e-6:
        scale = 1.0
    return centered / scale


def normalize_covariates(
    covariates: np.ndarray,
    context_length: int,
) -> np.ndarray:
    normalized = covariates.copy()
    for index in range(normalized.shape[1]):
        column = normalized[:context_length, index]
        if set(np.unique(normalized[:, index])).issubset({0.0, 1.0}):
            continue
        mean = float(column.mean())
        std = float(column.std()) or 1.0
        normalized[:, index] = (normalized[:, index] - mean) / std
    return normalized
