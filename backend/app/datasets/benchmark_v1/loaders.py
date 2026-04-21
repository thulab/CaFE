from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DIAGNOSTIC_FAMILIES
from .utils import seeded_rng


@dataclass(slots=True)
class LoadedSeries:
    source: str
    path: str
    values: np.ndarray


def _first_numeric_series(frame: pd.DataFrame) -> np.ndarray | None:
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        return None
    for column in numeric.columns:
        values = numeric[column].dropna().to_numpy(dtype=float)
        if len(values) >= 48:
            return values
    return None


def _scan_root(root: Path | None, source: str) -> list[LoadedSeries]:
    if root is None or not root.exists():
        return []
    results: list[LoadedSeries] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                frame = pd.read_csv(path)
            elif suffix in {".parquet", ".pq"}:
                frame = pd.read_parquet(path)
            elif suffix == ".npy":
                values = np.load(path)
                if values.ndim == 1 and len(values) >= 48:
                    results.append(LoadedSeries(source=source, path=str(path), values=values.astype(float)))
                continue
            else:
                continue
        except Exception:
            continue
        values = _first_numeric_series(frame)
        if values is not None:
            results.append(LoadedSeries(source=source, path=str(path), values=values))
    return results


def scan_real_corpora(gift_root: Path | None, tfb_root: Path | None) -> list[LoadedSeries]:
    series = []
    series.extend(_scan_root(gift_root, "gift_eval"))
    series.extend(_scan_root(tfb_root, "tfb"))
    return series


def bootstrap_anchor_corpus(size: int, seed: int) -> list[LoadedSeries]:
    rng = seeded_rng(seed)
    corpus: list[LoadedSeries] = []
    families = list(DIAGNOSTIC_FAMILIES)
    for idx in range(size):
        family = families[idx % len(families)]
        length = int(rng.integers(160, 360))
        t = np.arange(length, dtype=float)
        noise = rng.normal(0.0, 0.2, size=length)
        if family == "trend":
            values = 2 + rng.uniform(-0.05, 0.08) * t + 0.3 * np.sin(2 * np.pi * t / rng.integers(24, 48)) + noise
        elif family == "multi_seasonal":
            s1 = rng.integers(12, 24)
            s2 = rng.integers(24, 72)
            values = 1.5 * np.sin(2 * np.pi * t / s1 + rng.uniform(0, np.pi)) + 0.7 * np.cos(2 * np.pi * t / s2) + noise
        elif family == "regime_switching":
            state = 0
            values = np.zeros(length)
            for i in range(1, length):
                if rng.random() < 0.05:
                    state = 1 - state
                phi = 0.2 if state == 0 else 0.85
                mu = -0.5 if state == 0 else 1.2
                values[i] = mu + phi * values[i - 1] + noise[i]
        elif family == "long_memory_nonlinear":
            values = np.zeros(length)
            delay = 18
            values[: delay + 1] = 1.2
            for i in range(delay, length - 1):
                delayed = values[i - delay]
                values[i + 1] = values[i] + 0.18 * delayed / (1 + delayed**8) - 0.09 * values[i] + 0.05 * noise[i]
        else:
            demand = rng.poisson(1.3, size=length).astype(float)
            demand[rng.random(length) < rng.uniform(0.55, 0.8)] = 0.0
            vol = np.ones(length)
            for i in range(1, length):
                vol[i] = 0.05 + 0.25 * abs(noise[i - 1]) + 0.7 * vol[i - 1]
            values = demand + noise * vol * 3
        corpus.append(LoadedSeries(source="bootstrap", path=f"bootstrap:{idx}", values=values.astype(float)))
    return corpus
