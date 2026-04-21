from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import build_model_adapter
from .domain import EvalResult
from .metrics import mase, smape
from .utils import adjacent_meta_path, read_json, write_parquet


def _as_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float)


def run_model_eval(model_name: str, benchmark_path: Path, output_dir: Path, seeds: list[int]) -> Path:
    frame = pd.read_parquet(benchmark_path)
    meta_path = adjacent_meta_path(benchmark_path)
    benchmark_version = str(frame["benchmark_version"].iloc[0]) if "benchmark_version" in frame.columns else (
        str(read_json(meta_path).get("benchmark_version", benchmark_path.stem)) if meta_path.exists() else benchmark_path.stem
    )
    adapter = build_model_adapter(model_name)
    results = []
    for seed in seeds:
        batch_rows = [
            {
                "id": row.id,
                "context": _as_array(row.context),
                "target": _as_array(row.target),
                "horizon": int(row.horizon),
                "season_length": int(row.season_length),
            }
            for row in frame.itertuples(index=False)
        ]
        started = time.perf_counter()
        forecasts = adapter.predict_many(batch_rows, seed=seed)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_item_ms = elapsed_ms / max(1, len(batch_rows))
        for row, forecast in zip(frame.itertuples(index=False), forecasts, strict=False):
            context = _as_array(row.context)
            target = _as_array(row.target)
            season_length = int(row.season_length)
            results.append(
                EvalResult(
                    model=model_name,
                    series_id=row.id,
                    seed=seed,
                    benchmark_version=benchmark_version,
                    runtime_ms=per_item_ms,
                    mase=mase(context, target, forecast, season_length),
                    smape=smape(target, forecast),
                ).to_dict()
            )
    result_frame = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}.parquet"
    write_parquet(result_frame, output_path)
    return output_path

