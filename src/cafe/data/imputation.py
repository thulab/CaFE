from __future__ import annotations

import numpy as np


def impute_observed_window(
    values: np.ndarray,
    *,
    minimum_observed_fraction: float = 0.5,
) -> tuple[np.ndarray | None, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(array)
    observed_fraction = float(np.mean(finite)) if array.size else 0.0
    if (
        array.size == 0
        or observed_fraction < minimum_observed_fraction
        or int(np.sum(finite)) < 2
    ):
        return None, observed_fraction
    indexes = np.arange(array.size, dtype=float)
    imputed = np.interp(indexes, indexes[finite], array[finite])
    return np.asarray(imputed, dtype=float), observed_fraction
