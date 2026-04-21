from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    frame.to_parquet(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def adjacent_meta_path(path: Path) -> Path:
    return Path(path).with_suffix(".meta.json")


def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

