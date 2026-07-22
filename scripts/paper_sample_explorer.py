#!/usr/bin/env python3
"""Serve an offline browser for Paper v7 synthetic samples and forecasts.

The experiment artifacts are intentionally left in JSONL form.  A compact
SQLite index stores byte offsets only, so the browser can seek directly to the
five intensity rows and their model forecasts without copying the large data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(__file__).with_suffix("").resolve()
DEFAULT_DATA_DIR = REPO_ROOT / "runtime/paper_exp/v7/E2_dynamic_stability"
DEFAULT_INDEX_NAME = ".sample-explorer-index.sqlite3"
INDEX_SCHEMA_VERSION = "paper-sample-explorer.v1"
DEFAULT_CONTEXTS = (96, 168, 336, 504)
MODEL_ORDER = (
    "Timer-3.5",
    "Timer-3.0",
    "Chronos-2",
    "moirai2",
    "toto2.0",
    "timesfm2.5",
    "tirex2",
    "tabpfn-ts3",
    "naive",
    "seasonal_naive",
)


def load_oracle_context_rankings(data_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate intensity-level oracle-context means over every cell sample."""

    score_path = data_dir / "cell_full_pool_scores.csv"
    if not score_path.exists():
        return {}
    accumulators: dict[tuple[str, str, str], dict[str, float | int]] = {}
    with score_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("score_policy") != "oracle_context":
                continue
            key = (
                str(row["dataset_id"]),
                str(row["capability_id"]),
                str(row["model_id"]),
            )
            count = int(row["master_sample_count"])
            mase_mean = float(row["mase_mean"])
            accumulator = accumulators.setdefault(
                key, {"weighted_sum": 0.0, "sample_count": 0, "intensity_count": 0}
            )
            accumulator["weighted_sum"] = (
                float(accumulator["weighted_sum"]) + mase_mean * count
            )
            accumulator["sample_count"] = int(accumulator["sample_count"]) + count
            accumulator["intensity_count"] = int(accumulator["intensity_count"]) + 1

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (dataset_id, capability_id, model_id), accumulator in accumulators.items():
        sample_count = int(accumulator["sample_count"])
        if sample_count <= 0:
            continue
        grouped.setdefault((dataset_id, capability_id), []).append(
            {
                "modelId": model_id,
                "maseMean": float(accumulator["weighted_sum"]) / sample_count,
                "sampleCount": sample_count,
                "intensityCount": int(accumulator["intensity_count"]),
            }
        )

    rankings: dict[tuple[str, str], dict[str, Any]] = {}
    for cell, models in grouped.items():
        models.sort(key=lambda model: (model["maseMean"], model["modelId"]))
        ranked_models = [
            {**model, "rank": rank}
            for rank, model in enumerate(models, start=1)
        ]
        best = ranked_models[0]
        runner_up = ranked_models[1] if len(ranked_models) > 1 else None
        rankings[cell] = {
            "scorePolicy": "oracle_context",
            "best": best,
            "runnerUp": runner_up,
            "gapToRunnerUp": (
                runner_up["maseMean"] - best["maseMean"]
                if runner_up is not None
                else None
            ),
            "models": ranked_models,
        }
    return rankings

STRING_FIELDS = {
    name: re.compile(rb'"' + name.encode() + rb'"\s*:\s*"([^"]*)"')
    for name in (
        "analysis_block_id",
        "capability_id",
        "dataset_id",
        "frequency",
        "master_sample_id",
        "paired_group_id",
    )
}
NUMBER_FIELDS = {
    name: re.compile(
        rb'"'
        + name.encode()
        + rb'"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    )
    for name in (
        "analysis_block_index",
        "context_length",
        "intensity",
        "pool_index",
        "round_index",
        "sample_index",
        "season_length",
        "target_dim",
        "target_strength",
    )
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse Paper v7 synthetic samples and model forecasts."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="E2_dynamic_stability directory (default: runtime/paper_exp/v7).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Override the byte-offset SQLite index location.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Discard and rebuild the derived byte-offset index.",
    )
    parser.add_argument(
        "--build-index-only",
        action="store_true",
        help="Build/validate the index and exit without starting HTTP.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the explorer URL in the default browser.",
    )
    return parser.parse_args(argv)


def _extract_string(line: bytes, field: str) -> str:
    match = STRING_FIELDS[field].search(line)
    if match is None:
        raise ValueError(f"JSONL row is missing string field {field!r}")
    return match.group(1).decode("utf-8")


def _extract_number(line: bytes, field: str, *, optional: bool = False) -> float | None:
    match = NUMBER_FIELDS[field].search(line)
    if match is None:
        if optional:
            return None
        raise ValueError(f"JSONL row is missing numeric field {field!r}")
    return float(match.group(1))


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _resolve_artifact_path(data_dir: Path, configured: str) -> Path:
    raw = Path(configured)
    if raw.is_absolute():
        return raw.resolve()
    candidates = (data_dir / raw, REPO_ROOT / raw, data_dir / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / raw).resolve()


def discover_prediction_sources(data_dir: Path) -> list[dict[str, Any]]:
    manifest_path = data_dir / "inference_manifest.json"
    discovered: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configured = manifest.get("prediction_files", {}).get("synthetic", {})
        for model_id, descriptor in configured.items():
            path = _resolve_artifact_path(data_dir, str(descriptor["path"]))
            if path.exists():
                discovered.append({"model_id": str(model_id), "path": path})
    else:
        for path in sorted((data_dir / "predictions").glob("*.jsonl")):
            discovered.append({"model_id": path.stem, "path": path.resolve()})
    if not discovered:
        raise FileNotFoundError(f"no synthetic prediction JSONL files under {data_dir}")
    order = {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    return sorted(
        discovered,
        key=lambda row: (order.get(row["model_id"], len(order)), row["model_id"]),
    )


def current_input_signature(
    sample_path: Path, sources: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        _file_signature(sample_path),
        *[_file_signature(Path(source["path"])) for source in sources],
    ]


def index_is_valid(
    index_path: Path,
    sample_path: Path,
    sources: list[dict[str, Any]],
) -> bool:
    if not index_path.exists():
        return False
    try:
        with sqlite3.connect(index_path) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        return (
            metadata.get("schema_version") == INDEX_SCHEMA_VERSION
            and json.loads(metadata.get("input_signature", "null"))
            == current_input_signature(sample_path, sources)
        )
    except (json.JSONDecodeError, OSError, sqlite3.DatabaseError):
        return False


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE sources (
            source_id INTEGER PRIMARY KEY,
            model_id TEXT NOT NULL UNIQUE,
            path TEXT NOT NULL
        );
        CREATE TABLE sample_groups (
            group_id TEXT PRIMARY KEY,
            group_order INTEGER NOT NULL UNIQUE,
            dataset_id TEXT NOT NULL,
            capability_id TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            round_index INTEGER NOT NULL,
            analysis_block_id TEXT NOT NULL,
            analysis_block_index INTEGER NOT NULL,
            pool_index INTEGER NOT NULL,
            target_dim INTEGER NOT NULL,
            frequency TEXT NOT NULL,
            season_length INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX sample_groups_cell
            ON sample_groups(dataset_id, capability_id, group_order);
        CREATE TABLE sample_rows (
            sample_ord INTEGER PRIMARY KEY,
            group_id TEXT NOT NULL,
            intensity INTEGER NOT NULL,
            master_sample_id TEXT NOT NULL UNIQUE,
            byte_offset INTEGER NOT NULL,
            byte_length INTEGER NOT NULL,
            target_strength REAL,
            UNIQUE(group_id, intensity)
        );
        CREATE INDEX sample_rows_group ON sample_rows(group_id, intensity);
        CREATE TABLE prediction_rows (
            source_id INTEGER NOT NULL,
            sample_ord INTEGER NOT NULL,
            context_length INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL,
            byte_length INTEGER NOT NULL,
            PRIMARY KEY(source_id, sample_ord, context_length)
        ) WITHOUT ROWID;
        """
    )


def build_index(
    data_dir: Path,
    index_path: Path,
    sources: list[dict[str, Any]],
    *,
    progress: Callable[[str], None] = print,
) -> None:
    sample_path = data_dir / "samples.jsonl"
    if not sample_path.exists():
        raise FileNotFoundError(f"missing master sample file: {sample_path}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    started = time.monotonic()
    connection = sqlite3.connect(temporary)
    sample_ids: dict[str, int] = {}
    try:
        _create_schema(connection)
        progress(f"indexing 1/{len(sources) + 1}: {sample_path.name}")
        group_orders: dict[str, int] = {}
        sample_batch: list[tuple[Any, ...]] = []
        group_batch: list[tuple[Any, ...]] = []
        with sample_path.open("rb") as handle:
            sample_ord = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                group_id = _extract_string(line, "paired_group_id")
                master_sample_id = _extract_string(line, "master_sample_id")
                if master_sample_id in sample_ids:
                    raise ValueError(f"duplicate master_sample_id: {master_sample_id}")
                sample_ids[master_sample_id] = sample_ord
                if group_id not in group_orders:
                    group_order = len(group_orders)
                    group_orders[group_id] = group_order
                    group_batch.append(
                        (
                            group_id,
                            group_order,
                            _extract_string(line, "dataset_id"),
                            _extract_string(line, "capability_id"),
                            int(_extract_number(line, "sample_index")),
                            int(_extract_number(line, "round_index")),
                            _extract_string(line, "analysis_block_id"),
                            int(_extract_number(line, "analysis_block_index")),
                            int(_extract_number(line, "pool_index")),
                            int(_extract_number(line, "target_dim")),
                            _extract_string(line, "frequency"),
                            int(_extract_number(line, "season_length")),
                        )
                    )
                sample_batch.append(
                    (
                        sample_ord,
                        group_id,
                        int(_extract_number(line, "intensity")),
                        master_sample_id,
                        offset,
                        len(line),
                        _extract_number(line, "target_strength", optional=True),
                    )
                )
                sample_ord += 1
                if len(sample_batch) >= 2_000:
                    connection.executemany(
                        "INSERT INTO sample_rows VALUES (?, ?, ?, ?, ?, ?, ?)",
                        sample_batch,
                    )
                    connection.executemany(
                        "INSERT INTO sample_groups VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        group_batch,
                    )
                    sample_batch.clear()
                    group_batch.clear()
        if sample_batch:
            connection.executemany(
                "INSERT INTO sample_rows VALUES (?, ?, ?, ?, ?, ?, ?)", sample_batch
            )
        if group_batch:
            connection.executemany(
                "INSERT INTO sample_groups VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                group_batch,
            )
        connection.commit()

        prediction_total = 0
        for source_id, source in enumerate(sources):
            model_id = str(source["model_id"])
            path = Path(source["path"])
            connection.execute(
                "INSERT INTO sources(source_id, model_id, path) VALUES (?, ?, ?)",
                (source_id, model_id, str(path.resolve())),
            )
            progress(
                f"indexing {source_id + 2}/{len(sources) + 1}: "
                f"{model_id} ({path.name})"
            )
            prediction_batch: list[tuple[int, int, int, int, int]] = []
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    master_sample_id = _extract_string(line, "master_sample_id")
                    try:
                        row_sample_ord = sample_ids[master_sample_id]
                    except KeyError as error:
                        raise ValueError(
                            f"{path.name} references unknown sample {master_sample_id}"
                        ) from error
                    prediction_batch.append(
                        (
                            source_id,
                            row_sample_ord,
                            int(_extract_number(line, "context_length")),
                            offset,
                            len(line),
                        )
                    )
                    prediction_total += 1
                    if len(prediction_batch) >= 10_000:
                        connection.executemany(
                            "INSERT INTO prediction_rows VALUES (?, ?, ?, ?, ?)",
                            prediction_batch,
                        )
                        prediction_batch.clear()
            if prediction_batch:
                connection.executemany(
                    "INSERT INTO prediction_rows VALUES (?, ?, ?, ?, ?)",
                    prediction_batch,
                )
            connection.commit()

        built_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "input_signature": json.dumps(
                current_input_signature(sample_path, sources), separators=(",", ":")
            ),
            "built_at": built_at,
            "sample_count": str(len(sample_ids)),
            "group_count": str(len(group_orders)),
            "prediction_count": str(prediction_total),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
        os.replace(temporary, index_path)
    elapsed = time.monotonic() - started
    progress(f"index ready: {index_path} ({elapsed:.1f}s)")


def ensure_index(
    data_dir: Path,
    index_path: Path,
    *,
    rebuild: bool = False,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    data_dir = data_dir.resolve()
    sample_path = data_dir / "samples.jsonl"
    sources = discover_prediction_sources(data_dir)
    if rebuild or not index_is_valid(index_path, sample_path, sources):
        build_index(data_dir, index_path, sources, progress=progress)
    else:
        progress(f"using existing index: {index_path}")
    return sources


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def standardized_view(
    target: list[list[float]],
    context_length: int,
    horizon: int,
    hierarchy: str | None,
) -> tuple[list[list[float]], list[list[float]]]:
    master_context = len(target) - horizon
    if context_length <= 0 or context_length > master_context:
        raise ValueError(f"invalid context length: {context_length}")
    view = target[master_context - context_length :]
    dimension = len(view[0]) if view else 0
    context = view[:context_length]
    means = [sum(row[index] for row in context) / context_length for index in range(dimension)]
    if hierarchy == "additive_first" and dimension > 1:
        scale = _population_std([row[0] for row in context])
        if scale <= 1e-6:
            scale = sum(
                _population_std([row[index] for row in context])
                for index in range(dimension)
            ) / dimension
        if scale <= 1e-6:
            scale = 1.0
        scales = [scale] * dimension
    else:
        scales = []
        for index in range(dimension):
            scale = _population_std([row[index] for row in context])
            scales.append(scale if scale > 1e-6 else 1.0)
    standardized = [
        [(row[index] - means[index]) / scales[index] for index in range(dimension)]
        for row in view
    ]
    return standardized[:context_length], standardized[context_length:]


class SampleExplorer:
    """Read indexed JSONL rows with thread-safe positional file reads."""

    def __init__(self, data_dir: Path, index_path: Path):
        self.data_dir = data_dir.resolve()
        self.index_path = index_path.resolve()
        self.oracle_context_rankings = load_oracle_context_rankings(self.data_dir)
        self._sample_fd = os.open(self.data_dir / "samples.jsonl", os.O_RDONLY)
        with self._connect() as connection:
            source_rows = connection.execute(
                "SELECT source_id, model_id, path FROM sources ORDER BY source_id"
            ).fetchall()
        self.sources = {
            int(row[0]): {
                "model_id": str(row[1]),
                "path": Path(row[2]),
                "fd": os.open(row[2], os.O_RDONLY),
            }
            for row in source_rows
        }

    def close(self) -> None:
        os.close(self._sample_fd)
        for source in self.sources.values():
            os.close(int(source["fd"]))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.index_path}?mode=ro", uri=True, timeout=10
        )
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _read_json(fd: int, offset: int, length: int) -> dict[str, Any]:
        payload = os.pread(fd, length, offset)
        if len(payload) != length:
            raise OSError(f"short positional read: {len(payload)} != {length}")
        return json.loads(payload)

    def meta(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT dataset_id, capability_id, COUNT(*) AS group_count,
                       MIN(group_order) AS first_order
                FROM sample_groups
                GROUP BY dataset_id, capability_id
                ORDER BY first_order
                """
            ).fetchall()
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        datasets: list[dict[str, Any]] = []
        by_dataset: dict[str, dict[str, Any]] = {}
        for row in rows:
            dataset_id = str(row["dataset_id"])
            dataset = by_dataset.get(dataset_id)
            if dataset is None:
                dataset = {"id": dataset_id, "capabilities": []}
                by_dataset[dataset_id] = dataset
                datasets.append(dataset)
            dataset["capabilities"].append(
                {"id": str(row["capability_id"]), "sampleCount": int(row["group_count"])}
            )
        contexts = list(DEFAULT_CONTEXTS)
        config_path = self.data_dir / "generation_config.json"
        if config_path.exists():
            configured = json.loads(config_path.read_text(encoding="utf-8")).get(
                "context_lengths"
            )
            if configured:
                contexts = [int(value) for value in configured]
        return {
            "schemaVersion": "paper-sample-explorer.api.v1",
            "datasets": datasets,
            "models": [
                {
                    "id": str(source["model_id"]),
                    "kind": (
                        "baseline"
                        if source["model_id"] in {"naive", "seasonal_naive"}
                        else "model"
                    ),
                }
                for source in self.sources.values()
            ],
            "contexts": contexts,
            "intensities": [1, 2, 3, 4, 5],
            "index": {
                "builtAt": metadata.get("built_at"),
                "sampleCount": int(metadata.get("sample_count", 0)),
                "groupCount": int(metadata.get("group_count", 0)),
                "predictionCount": int(metadata.get("prediction_count", 0)),
            },
        }

    def groups(self, dataset_id: str, capability_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT group_id, group_order, sample_index, round_index,
                       analysis_block_id, analysis_block_index, pool_index,
                       target_dim, frequency, season_length
                FROM sample_groups
                WHERE dataset_id = ? AND capability_id = ?
                ORDER BY group_order
                """,
                (dataset_id, capability_id),
            ).fetchall()
        return [
            {
                "id": str(row["group_id"]),
                "order": int(row["group_order"]),
                "sampleIndex": int(row["sample_index"]),
                "roundIndex": int(row["round_index"]),
                "analysisBlock": str(row["analysis_block_id"]),
                "analysisBlockIndex": int(row["analysis_block_index"]),
                "poolIndex": int(row["pool_index"]),
                "targetDim": int(row["target_dim"]),
                "frequency": str(row["frequency"]),
                "seasonLength": int(row["season_length"]),
            }
            for row in rows
        ]

    def sample(
        self,
        group_id: str,
        context_length: int,
        requested_models: list[str] | None = None,
    ) -> dict[str, Any]:
        model_by_id = {
            str(source["model_id"]): source_id
            for source_id, source in self.sources.items()
        }
        model_ids = requested_models or list(model_by_id)
        unknown = sorted(set(model_ids) - set(model_by_id))
        if unknown:
            raise ValueError(f"unknown models: {', '.join(unknown)}")
        source_ids = [model_by_id[model_id] for model_id in model_ids]
        with self._connect() as connection:
            group = connection.execute(
                "SELECT * FROM sample_groups WHERE group_id = ?", (group_id,)
            ).fetchone()
            if group is None:
                raise KeyError(f"unknown sample group: {group_id}")
            sample_rows = connection.execute(
                "SELECT * FROM sample_rows WHERE group_id = ? ORDER BY intensity",
                (group_id,),
            ).fetchall()
            if len(sample_rows) != 5:
                raise ValueError(f"sample group has {len(sample_rows)} intensities, expected 5")
            placeholders = ",".join("?" for _ in source_ids)
            sample_ordinals = [int(row["sample_ord"]) for row in sample_rows]
            sample_placeholders = ",".join("?" for _ in sample_ordinals)
            prediction_rows = connection.execute(
                f"""
                SELECT source_id, sample_ord, byte_offset, byte_length
                FROM prediction_rows
                WHERE context_length = ?
                  AND source_id IN ({placeholders})
                  AND sample_ord IN ({sample_placeholders})
                """,
                [context_length, *source_ids, *sample_ordinals],
            ).fetchall()
        predictions_by_sample: dict[int, list[sqlite3.Row]] = {}
        for row in prediction_rows:
            predictions_by_sample.setdefault(int(row["sample_ord"]), []).append(row)

        intensity_payloads: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for indexed_row in sample_rows:
            sample_ord = int(indexed_row["sample_ord"])
            sample = self._read_json(
                self._sample_fd,
                int(indexed_row["byte_offset"]),
                int(indexed_row["byte_length"]),
            )
            horizon = int(sample["horizon"])
            history, actual = standardized_view(
                sample["target"], context_length, horizon, sample.get("hierarchy")
            )
            model_payloads: dict[str, Any] = {}
            for prediction_index in predictions_by_sample.get(sample_ord, []):
                source_id = int(prediction_index["source_id"])
                source = self.sources[source_id]
                prediction = self._read_json(
                    int(source["fd"]),
                    int(prediction_index["byte_offset"]),
                    int(prediction_index["byte_length"]),
                )
                model_id = str(source["model_id"])
                model_payloads[model_id] = {
                    "forecast": prediction["forecast"],
                    "metrics": prediction.get("metrics", {}),
                    "modelGroup": prediction.get("model_group"),
                    "inputAdaptation": prediction.get("input_adaptation"),
                }
            for model_id in model_ids:
                if model_id not in model_payloads:
                    missing.append(
                        {"intensity": int(sample["intensity"]), "modelId": model_id}
                    )
            realized_by_context = sample.get("realized_features_by_context", {}).get(
                str(context_length), {}
            )
            target_feature = sample.get("target_feature")
            intensity_payloads.append(
                {
                    "intensity": int(sample["intensity"]),
                    "masterSampleId": str(sample["master_sample_id"]),
                    "targetStrength": sample.get("target_strength"),
                    "targetRelativeLevel": sample.get("target_relative_level"),
                    "targetFeature": target_feature,
                    "realizedFeature": (
                        realized_by_context.get(target_feature) if target_feature else None
                    ),
                    "history": history,
                    "actual": actual,
                    "models": model_payloads,
                }
            )
        return {
            "group": {
                "id": str(group["group_id"]),
                "datasetId": str(group["dataset_id"]),
                "capabilityId": str(group["capability_id"]),
                "sampleIndex": int(group["sample_index"]),
                "roundIndex": int(group["round_index"]),
                "analysisBlock": str(group["analysis_block_id"]),
                "analysisBlockIndex": int(group["analysis_block_index"]),
                "poolIndex": int(group["pool_index"]),
                "targetDim": int(group["target_dim"]),
                "frequency": str(group["frequency"]),
                "seasonLength": int(group["season_length"]),
            },
            "contextLength": context_length,
            "horizon": len(intensity_payloads[0]["actual"]),
            "targetColumns": [
                f"target_{index}" for index in range(int(group["target_dim"]))
            ],
            "intensities": intensity_payloads,
            "missingPredictions": missing,
            "oracleContextRanking": self.oracle_context_rankings.get(
                (str(group["dataset_id"]), str(group["capability_id"]))
            ),
        }


class ExplorerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], explorer: SampleExplorer):
        self.explorer = explorer
        super().__init__(address, ExplorerRequestHandler)


class ExplorerRequestHandler(BaseHTTPRequestHandler):
    server: ExplorerHTTPServer

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write(
            f"[{self.log_date_time_string()}] {format_string % args}\n"
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._send_json({"status": "ok"})
            elif parsed.path == "/api/meta":
                self._send_json(self.server.explorer.meta())
            elif parsed.path == "/api/groups":
                query = parse_qs(parsed.query)
                dataset_id = self._required_query(query, "dataset")
                capability_id = self._required_query(query, "capability")
                self._send_json(
                    {"groups": self.server.explorer.groups(dataset_id, capability_id)}
                )
            elif parsed.path == "/api/sample":
                query = parse_qs(parsed.query)
                group_id = self._required_query(query, "group")
                context = int(self._required_query(query, "context"))
                configured_models = query.get("models", [""])[0]
                models = (
                    [value for value in configured_models.split(",") if value]
                    if configured_models
                    else None
                )
                self._send_json(
                    self.server.explorer.sample(group_id, context, models)
                )
            elif parsed.path in {"/", "/index.html"}:
                self._send_asset("index.html", "text/html; charset=utf-8")
            elif parsed.path == "/app.js":
                self._send_asset("app.js", "text/javascript; charset=utf-8")
            elif parsed.path == "/styles.css":
                self._send_asset("styles.css", "text/css; charset=utf-8")
            elif parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except KeyError as error:
            self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except (TypeError, ValueError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # noqa: BLE001 - keep the local browser responsive.
            self.log_error("request failed: %s: %s", type(error).__name__, error)
            self._send_json(
                {"error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    @staticmethod
    def _required_query(query: dict[str, list[str]], name: str) -> str:
        values = query.get(name)
        if not values or not values[0]:
            raise ValueError(f"missing query parameter: {name}")
        return values[0]

    def _base_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )

    def _send_json(
        self, payload: Any, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.send_response(status)
        self._base_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_asset(self, name: str, content_type: str) -> None:
        payload = (ASSET_DIR / name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self._base_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    index_path = (
        args.index_path.resolve()
        if args.index_path is not None
        else data_dir / DEFAULT_INDEX_NAME
    )
    ensure_index(
        data_dir,
        index_path,
        rebuild=args.rebuild_index,
        progress=lambda message: print(message, flush=True),
    )
    if args.build_index_only:
        return 0
    explorer = SampleExplorer(data_dir, index_path)
    server = ExplorerHTTPServer((args.host, args.port), explorer)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"sample explorer ready: {url}", flush=True)
    print("press Ctrl+C to stop", flush=True)
    if args.open:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nstopping sample explorer", flush=True)
    finally:
        server.server_close()
        explorer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
