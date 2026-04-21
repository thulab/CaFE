from __future__ import annotations

from pathlib import Path

FEATURE_COLUMNS = [
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "acf_half_life",
    "changepoint_density",
    "variance_shift",
    "intermittency",
    "outlier_rate",
]

DIAGNOSTIC_FAMILIES = [
    "trend",
    "multi_seasonal",
    "regime_switching",
    "long_memory_nonlinear",
    "intermittent_heteroskedastic",
]

HORIZON_RATIOS = [0.25, 0.5, 1.0]
DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]
DEFAULT_ARTIFACT_SUBDIR = Path("benchmark_v1")

