#!/usr/bin/env python3
"""Serve an offline browser for CaFE synthetic samples and forecasts.

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


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATA_DIR = REPO_ROOT / "runtime" / "experiments"
DEFAULT_INDEX_NAME = ".sample-explorer-index.sqlite3"
INDEX_SCHEMA_VERSION = "paper-sample-explorer.v1"
CAFE_INDEX_SCHEMA_VERSION = "cafe-sample-explorer.v1"
DEFAULT_CONTEXTS = (96, 168, 336, 504)
MODEL_ORDER = (
    "Chronos-2",
    "timesfm2.5",
    "tirex2",
    "moirai2",
    "Timer-3.5",
    "toto2.0",
    "TimePFN",
    "Timer-3.0",
    "tabpfn-ts3",
    "naive",
    "last_value",
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
        "evaluation_table",
        "generator_family_role",
        "master_sample_id",
        "paired_group_id",
        "sample_id",
        "target_feature",
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
        "counterfactual_member",
        "context_length",
        "intensity",
        "pool_index",
        "round_index",
        "sample_index",
        "seed_index",
        "season_length",
        "target_dim",
        "target_feature_value",
        "target_strength",
    )
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse Paper v7/cafe synthetic samples and model forecasts."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "A Paper v7 E2 directory, a CaFE experiment directory, or the "
            "cafe parent directory (default: runtime/experiments, newest "
            "completed cafe experiment)."
        ),
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


def is_v8_experiment_dir(data_dir: Path) -> bool:
    return (
        (data_dir / "distributed_analysis_manifest.json").is_file()
        or (data_dir / "model_major_inference_status.json").is_file()
        or any(data_dir.glob("*/03_inference/seed_*/inference_manifest.json"))
    )


def resolve_data_dir(data_dir: Path) -> Path:
    """Resolve a v7 directory or the newest completed cafe experiment."""

    resolved = data_dir.resolve()
    if (resolved / "samples.jsonl").is_file() or is_v8_experiment_dir(resolved):
        return resolved
    if resolved.name == "cafe" and resolved.is_dir():
        candidates = [
            path
            for path in resolved.iterdir()
            if path.is_dir() and is_v8_experiment_dir(path)
        ]
        if candidates:
            completed = [
                path
                for path in candidates
                if (path / "distributed_analysis_manifest.json").is_file()
            ]
            pool = completed or candidates
            return max(
                pool,
                key=lambda path: max(
                    candidate.stat().st_mtime_ns
                    for candidate in (
                        path,
                        path / "distributed_analysis_manifest.json",
                        path / "model_major_inference_status.json",
                    )
                    if candidate.exists()
                ),
            ).resolve()
    raise FileNotFoundError(
        f"{resolved} is neither a Paper v7 data directory nor a CaFE "
        "experiment directory"
    )


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


def _v8_artifact_path(configured: str | None, fallback: Path) -> Path:
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return candidate.resolve()
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"missing cafe artifact: {configured or fallback}")


def discover_v8_artifacts(data_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    suite_manifest_path = data_dir / "distributed_analysis_manifest.json"
    if suite_manifest_path.exists():
        suite_manifest = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
        dataset_ids = [
            str(row["dataset_id"]) for row in suite_manifest.get("datasets", [])
        ]
        seed_start = int(suite_manifest["seed_start"])
        seed_count = int(suite_manifest["seed_count"])
        shard_name = f"seed_{seed_start:06d}_{seed_start + seed_count:06d}"
    else:
        preparation_path = data_dir / "distributed_preparation_manifest.json"
        if preparation_path.exists():
            preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
            dataset_ids = [
                str(value) for value in preparation["protocol"]["dataset_ids"]
            ]
        else:
            dataset_ids = sorted(
                path.name
                for path in data_dir.iterdir()
                if path.is_dir() and (path / "03_inference").is_dir()
            )
        shard_sets = [
            {
                path.name
                for path in (data_dir / dataset_id / "03_inference").glob("seed_*")
                if (path / "inference_manifest.json").is_file()
            }
            for dataset_id in dataset_ids
        ]
        common_shards = set.intersection(*shard_sets) if shard_sets else set()
        if len(common_shards) != 1:
            raise ValueError(
                "cafe explorer requires exactly one common inference seed shard; "
                f"found {sorted(common_shards)}"
            )
        shard_name = next(iter(common_shards))

    if not dataset_ids:
        raise FileNotFoundError(f"no cafe datasets found under {data_dir}")

    datasets: list[dict[str, Any]] = []
    common_models: list[str] | None = None
    order = {model_id: index for index, model_id in enumerate(MODEL_ORDER)}
    for dataset_id in dataset_ids:
        inference_dir = data_dir / dataset_id / "03_inference" / shard_name
        task_manifest = json.loads(
            (inference_dir / "task_manifest.json").read_text(encoding="utf-8")
        )
        task_path = _v8_artifact_path(
            task_manifest.get("task_file", {}).get("path"),
            inference_dir / "forecast_views.jsonl",
        )
        inference_manifest = json.loads(
            (inference_dir / "inference_manifest.json").read_text(encoding="utf-8")
        )
        if not inference_manifest.get("complete"):
            raise ValueError(f"incomplete cafe inference manifest: {dataset_id}")
        predictions: list[dict[str, Any]] = []
        for descriptor in inference_manifest.get("predictions", {}).get("files", []):
            model_id = str(descriptor["model_id"])
            configured = str(descriptor.get("path", ""))
            fallback_matches = sorted(
                (inference_dir / "model_shards").glob(
                    f"*/predictions/{Path(configured).name}"
                )
            )
            fallback = (
                fallback_matches[0]
                if fallback_matches
                else inference_dir
                / "model_shards"
                / model_id
                / "predictions"
                / f"{model_id}.jsonl"
            )
            predictions.append(
                {
                    "model_id": model_id,
                    "path": _v8_artifact_path(configured, fallback),
                }
            )
        predictions.sort(
            key=lambda row: (
                order.get(str(row["model_id"]), len(order)),
                str(row["model_id"]),
            )
        )
        models = [str(row["model_id"]) for row in predictions]
        if common_models is None:
            common_models = models
        elif models != common_models:
            raise ValueError(
                f"cafe model set/order differs for {dataset_id}: {models} != "
                f"{common_models}"
            )
        score_path = (
            data_dir / dataset_id / "04_analysis" / shard_name / "scores.json"
        )
        datasets.append(
            {
                "dataset_id": dataset_id,
                "task_path": task_path,
                "predictions": predictions,
                "score_path": score_path if score_path.exists() else None,
            }
        )
    return {
        "schema_version": "cafe-sample-explorer-input.v1",
        "experiment_id": data_dir.name,
        "shard_name": shard_name,
        "datasets": datasets,
        "models": common_models or [],
    }


def cafe_input_signature(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for dataset in artifacts["datasets"]:
        signature.append(_file_signature(Path(dataset["task_path"])))
        signature.extend(
            _file_signature(Path(source["path"]))
            for source in dataset["predictions"]
        )
    return signature


def cafe_index_is_valid(
    index_path: Path,
    artifacts: dict[str, Any],
) -> bool:
    if not index_path.exists():
        return False
    try:
        with sqlite3.connect(index_path) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        return (
            metadata.get("schema_version") == CAFE_INDEX_SCHEMA_VERSION
            and json.loads(metadata.get("input_signature", "null"))
            == cafe_input_signature(artifacts)
        )
    except (json.JSONDecodeError, OSError, sqlite3.DatabaseError):
        return False


def _create_v8_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE sample_sources (
            source_id INTEGER PRIMARY KEY,
            dataset_id TEXT NOT NULL UNIQUE,
            path TEXT NOT NULL
        );
        CREATE TABLE prediction_sources (
            source_id INTEGER PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            path TEXT NOT NULL,
            UNIQUE(dataset_id, model_id)
        );
        CREATE TABLE sample_groups (
            group_id TEXT PRIMARY KEY,
            group_order INTEGER NOT NULL UNIQUE,
            dataset_id TEXT NOT NULL,
            capability_id TEXT NOT NULL,
            seed_index INTEGER NOT NULL,
            family_role TEXT NOT NULL,
            evaluation_table TEXT NOT NULL,
            counterfactual_member INTEGER,
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
            UNIQUE(group_id, intensity)
        );
        CREATE INDEX sample_rows_group ON sample_rows(group_id, intensity);
        CREATE TABLE sample_views (
            sample_ord INTEGER NOT NULL,
            context_length INTEGER NOT NULL,
            sample_id TEXT NOT NULL UNIQUE,
            source_id INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL,
            byte_length INTEGER NOT NULL,
            PRIMARY KEY(sample_ord, context_length)
        ) WITHOUT ROWID;
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


def build_v8_index(
    data_dir: Path,
    index_path: Path,
    artifacts: dict[str, Any],
    *,
    progress: Callable[[str], None] = print,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    started = time.monotonic()
    connection = sqlite3.connect(temporary)
    sample_ids: dict[str, tuple[int, int]] = {}
    master_ids: dict[str, int] = {}
    group_orders: dict[str, int] = {}
    sample_count_by_dataset: dict[str, int] = {}
    view_count_by_dataset: dict[str, int] = {}
    prediction_total = 0
    try:
        _create_v8_schema(connection)
        for sample_source_id, dataset in enumerate(artifacts["datasets"]):
            dataset_id = str(dataset["dataset_id"])
            task_path = Path(dataset["task_path"])
            connection.execute(
                "INSERT INTO sample_sources(source_id, dataset_id, path) "
                "VALUES (?, ?, ?)",
                (sample_source_id, dataset_id, str(task_path.resolve())),
            )
            progress(
                f"indexing cafe tasks {sample_source_id + 1}/"
                f"{len(artifacts['datasets'])}: {dataset_id}"
            )
            group_batch: list[tuple[Any, ...]] = []
            sample_batch: list[tuple[Any, ...]] = []
            view_batch: list[tuple[Any, ...]] = []
            with task_path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    if (
                        _extract_string(line, "evaluation_table") != "main"
                        or _extract_string(line, "generator_family_role")
                        != "primary"
                    ):
                        continue
                    master_sample_id = _extract_string(line, "master_sample_id")
                    sample_id = _extract_string(line, "sample_id")
                    context_length = int(_extract_number(line, "context_length"))
                    sample_ord = master_ids.get(master_sample_id)
                    if sample_ord is None:
                        sample_ord = len(master_ids)
                        master_ids[master_sample_id] = sample_ord
                        paired_group_id = _extract_string(line, "paired_group_id")
                        member = _extract_number(
                            line, "counterfactual_member", optional=True
                        )
                        member_index = int(member) if member is not None else None
                        group_id = (
                            paired_group_id
                            if member_index is None
                            else f"{paired_group_id}__member{member_index}"
                        )
                        if group_id not in group_orders:
                            group_order = len(group_orders)
                            group_orders[group_id] = group_order
                            group_batch.append(
                                (
                                    group_id,
                                    group_order,
                                    dataset_id,
                                    _extract_string(line, "capability_id"),
                                    int(_extract_number(line, "seed_index")),
                                    "primary",
                                    "main",
                                    member_index,
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
                            )
                        )
                    if sample_id in sample_ids:
                        raise ValueError(f"duplicate cafe task sample_id: {sample_id}")
                    sample_ids[sample_id] = (sample_ord, context_length)
                    view_batch.append(
                        (
                            sample_ord,
                            context_length,
                            sample_id,
                            sample_source_id,
                            offset,
                            len(line),
                        )
                    )
                    if len(view_batch) >= 4_000:
                        connection.executemany(
                            "INSERT INTO sample_views VALUES (?, ?, ?, ?, ?, ?)",
                            view_batch,
                        )
                        connection.executemany(
                            "INSERT INTO sample_rows VALUES (?, ?, ?, ?)",
                            sample_batch,
                        )
                        connection.executemany(
                            "INSERT INTO sample_groups VALUES "
                            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            group_batch,
                        )
                        view_batch.clear()
                        sample_batch.clear()
                        group_batch.clear()
            if view_batch:
                connection.executemany(
                    "INSERT INTO sample_views VALUES (?, ?, ?, ?, ?, ?)",
                    view_batch,
                )
            if sample_batch:
                connection.executemany(
                    "INSERT INTO sample_rows VALUES (?, ?, ?, ?)",
                    sample_batch,
                )
            if group_batch:
                connection.executemany(
                    "INSERT INTO sample_groups VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    group_batch,
                )
            sample_count_by_dataset[dataset_id] = connection.execute(
                "SELECT COUNT(*) FROM sample_rows r JOIN sample_groups g "
                "ON g.group_id = r.group_id WHERE g.dataset_id = ?",
                (dataset_id,),
            ).fetchone()[0]
            view_count_by_dataset[dataset_id] = connection.execute(
                "SELECT COUNT(*) FROM sample_views v JOIN sample_rows r "
                "ON r.sample_ord = v.sample_ord JOIN sample_groups g "
                "ON g.group_id = r.group_id WHERE g.dataset_id = ?",
                (dataset_id,),
            ).fetchone()[0]
            connection.commit()

        prediction_source_id = 0
        for dataset in artifacts["datasets"]:
            dataset_id = str(dataset["dataset_id"])
            expected = view_count_by_dataset[dataset_id]
            for source in dataset["predictions"]:
                model_id = str(source["model_id"])
                path = Path(source["path"])
                connection.execute(
                    "INSERT INTO prediction_sources"
                    "(source_id, dataset_id, model_id, path) VALUES (?, ?, ?, ?)",
                    (
                        prediction_source_id,
                        dataset_id,
                        model_id,
                        str(path.resolve()),
                    ),
                )
                progress(f"indexing cafe predictions: {dataset_id} / {model_id}")
                batch: list[tuple[int, int, int, int, int]] = []
                indexed_count = 0
                with path.open("rb") as handle:
                    while True:
                        offset = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        if not line.strip():
                            continue
                        identity = sample_ids.get(_extract_string(line, "sample_id"))
                        if identity is None:
                            continue
                        sample_ord, context_length = identity
                        batch.append(
                            (
                                prediction_source_id,
                                sample_ord,
                                context_length,
                                offset,
                                len(line),
                            )
                        )
                        indexed_count += 1
                        prediction_total += 1
                        if len(batch) >= 10_000:
                            connection.executemany(
                                "INSERT INTO prediction_rows VALUES "
                                "(?, ?, ?, ?, ?)",
                                batch,
                            )
                            batch.clear()
                if batch:
                    connection.executemany(
                        "INSERT INTO prediction_rows VALUES (?, ?, ?, ?, ?)",
                        batch,
                    )
                if indexed_count != expected:
                    raise ValueError(
                        f"{dataset_id}/{model_id} has {indexed_count} main-primary "
                        f"predictions, expected {expected}"
                    )
                prediction_source_id += 1
                connection.commit()

        contexts = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT context_length FROM sample_views "
                "ORDER BY context_length"
            )
        ]
        metadata = {
            "schema_version": CAFE_INDEX_SCHEMA_VERSION,
            "input_signature": json.dumps(
                cafe_input_signature(artifacts), separators=(",", ":")
            ),
            "experiment_id": str(artifacts["experiment_id"]),
            "shard_name": str(artifacts["shard_name"]),
            "sample_scope": "main/primary/clean",
            "contexts": json.dumps(contexts, separators=(",", ":")),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": str(len(master_ids)),
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
    progress(f"cafe index ready: {index_path} ({time.monotonic() - started:.1f}s)")


def ensure_v8_index(
    data_dir: Path,
    index_path: Path,
    *,
    rebuild: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    artifacts = discover_v8_artifacts(data_dir)
    if rebuild or not cafe_index_is_valid(index_path, artifacts):
        build_v8_index(
            data_dir,
            index_path,
            artifacts,
            progress=progress,
        )
    else:
        progress(f"using existing cafe index: {index_path}")
    return artifacts


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


def load_v8_oracle_context_rankings(
    artifacts: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    rankings: dict[tuple[str, str], dict[str, Any]] = {}
    allowed_models = set(str(value) for value in artifacts["models"])
    for dataset in artifacts["datasets"]:
        score_path = dataset.get("score_path")
        if score_path is None:
            continue
        rows = json.loads(Path(score_path).read_text(encoding="utf-8")).get(
            "scores", []
        )
        by_capability: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if (
                row.get("context_policy") != "oracle_context"
                or row.get("evaluation_table") != "main"
                or row.get("generator_family_role") != "primary"
                or row.get("model_id") not in allowed_models
                or row.get("accuracy_score") is None
            ):
                continue
            by_capability.setdefault(
                str(row["capability_id"]), []
            ).append(
                {
                    "modelId": str(row["model_id"]),
                    "maseMean": float(row["accuracy_score"]),
                    "sampleCount": int(row["seed_count"]),
                    "intensityCount": len(row["intensities"]),
                }
            )
        for capability_id, capability_models in by_capability.items():
            capability_models.sort(
                key=lambda row: (row["maseMean"], row["modelId"])
            )
            ranked = [
                {**model, "rank": index}
                for index, model in enumerate(capability_models, start=1)
            ]
            best = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else None
            rankings[(str(dataset["dataset_id"]), capability_id)] = {
                "scorePolicy": "oracle_context",
                "best": best,
                "runnerUp": runner_up,
                "gapToRunnerUp": (
                    runner_up["maseMean"] - best["maseMean"]
                    if runner_up is not None
                    else None
                ),
                "models": ranked,
            }
    return rankings


def _prediction_metrics(
    actual: list[list[float]],
    forecast: list[list[float]],
    mase_scale: float,
) -> dict[str, float]:
    if len(actual) != len(forecast):
        raise ValueError(
            f"forecast horizon mismatch: {len(forecast)} != {len(actual)}"
        )
    for index, (truth_row, forecast_row) in enumerate(zip(actual, forecast)):
        if len(truth_row) != len(forecast_row):
            raise ValueError(
                "forecast target dimension mismatch at horizon step "
                f"{index}: {len(forecast_row)} != {len(truth_row)}"
            )
    absolute_errors = [
        abs(float(truth) - float(predicted))
        for truth_row, forecast_row in zip(actual, forecast)
        for truth, predicted in zip(truth_row, forecast_row)
    ]
    if not absolute_errors:
        raise ValueError("empty forecast")
    mae = sum(absolute_errors) / len(absolute_errors)
    return {
        "mae": mae,
        "mase": mae / max(float(mase_scale), 1e-12),
    }


class V8SampleExplorer:
    """Read CaFE task views and per-dataset prediction shards."""

    def __init__(
        self,
        data_dir: Path,
        index_path: Path,
        artifacts: dict[str, Any],
    ):
        self.data_dir = data_dir.resolve()
        self.index_path = index_path.resolve()
        self.artifacts = artifacts
        self.oracle_context_rankings = load_v8_oracle_context_rankings(artifacts)
        with self._connect() as connection:
            sample_sources = connection.execute(
                "SELECT source_id, dataset_id, path FROM sample_sources "
                "ORDER BY source_id"
            ).fetchall()
            prediction_sources = connection.execute(
                "SELECT source_id, dataset_id, model_id, path "
                "FROM prediction_sources ORDER BY source_id"
            ).fetchall()
        self.sample_sources = {
            int(row["source_id"]): {
                "dataset_id": str(row["dataset_id"]),
                "path": Path(row["path"]),
                "fd": os.open(row["path"], os.O_RDONLY),
            }
            for row in sample_sources
        }
        self.prediction_sources = {
            int(row["source_id"]): {
                "dataset_id": str(row["dataset_id"]),
                "model_id": str(row["model_id"]),
                "path": Path(row["path"]),
                "fd": os.open(row["path"], os.O_RDONLY),
            }
            for row in prediction_sources
        }
        self.models = [str(value) for value in artifacts["models"]]

    def close(self) -> None:
        for source in self.sample_sources.values():
            os.close(int(source["fd"]))
        for source in self.prediction_sources.values():
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
                {
                    "id": str(row["capability_id"]),
                    "sampleCount": int(row["group_count"]),
                }
            )
        return {
            "schemaVersion": "paper-sample-explorer.api.v2",
            "experiment": {
                "version": "cafe",
                "id": metadata.get("experiment_id", self.data_dir.name),
                "shard": metadata.get("shard_name"),
                "sampleScope": metadata.get("sample_scope"),
            },
            "datasets": datasets,
            "models": [{"id": model_id, "kind": "model"} for model_id in self.models],
            "contexts": json.loads(
                metadata.get("contexts", json.dumps(DEFAULT_CONTEXTS))
            ),
            "intensities": [1, 2, 3, 4, 5],
            "index": {
                "builtAt": metadata.get("built_at"),
                "sampleCount": int(metadata.get("sample_count", 0)),
                "groupCount": int(metadata.get("group_count", 0)),
                "predictionCount": int(metadata.get("prediction_count", 0)),
                "groupUnit": "seed groups",
            },
        }

    def groups(self, dataset_id: str, capability_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT group_id, group_order, seed_index, family_role,
                       evaluation_table, counterfactual_member, target_dim,
                       frequency, season_length
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
                "seedIndex": int(row["seed_index"]),
                "familyRole": str(row["family_role"]),
                "evaluationTable": str(row["evaluation_table"]),
                "counterfactualMember": (
                    int(row["counterfactual_member"])
                    if row["counterfactual_member"] is not None
                    else None
                ),
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
        model_ids = requested_models or list(self.models)
        unknown = sorted(set(model_ids) - set(self.models))
        if unknown:
            raise ValueError(f"unknown models: {', '.join(unknown)}")
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
                raise ValueError(
                    f"cafe sample group has {len(sample_rows)} intensities, expected 5"
                )
            sample_ordinals = [int(row["sample_ord"]) for row in sample_rows]
            sample_placeholders = ",".join("?" for _ in sample_ordinals)
            sample_views = connection.execute(
                f"""
                SELECT sample_ord, source_id, byte_offset, byte_length
                FROM sample_views
                WHERE context_length = ?
                  AND sample_ord IN ({sample_placeholders})
                """,
                [context_length, *sample_ordinals],
            ).fetchall()
            source_rows = connection.execute(
                "SELECT source_id, model_id FROM prediction_sources "
                "WHERE dataset_id = ?",
                (str(group["dataset_id"]),),
            ).fetchall()
            source_by_model = {
                str(row["model_id"]): int(row["source_id"]) for row in source_rows
            }
            source_ids = [source_by_model[model_id] for model_id in model_ids]
            source_placeholders = ",".join("?" for _ in source_ids)
            prediction_rows = connection.execute(
                f"""
                SELECT source_id, sample_ord, byte_offset, byte_length
                FROM prediction_rows
                WHERE context_length = ?
                  AND source_id IN ({source_placeholders})
                  AND sample_ord IN ({sample_placeholders})
                """,
                [
                    context_length,
                    *source_ids,
                    *sample_ordinals,
                ],
            ).fetchall()

        view_by_sample = {int(row["sample_ord"]): row for row in sample_views}
        predictions_by_sample: dict[int, list[sqlite3.Row]] = {}
        for row in prediction_rows:
            predictions_by_sample.setdefault(int(row["sample_ord"]), []).append(row)
        intensity_payloads: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for indexed_row in sample_rows:
            sample_ord = int(indexed_row["sample_ord"])
            view_row = view_by_sample.get(sample_ord)
            if view_row is None:
                raise ValueError(
                    f"sample {indexed_row['master_sample_id']} has no L{context_length} view"
                )
            sample_source = self.sample_sources[int(view_row["source_id"])]
            sample = self._read_json(
                int(sample_source["fd"]),
                int(view_row["byte_offset"]),
                int(view_row["byte_length"]),
            )
            target = sample["target"]
            history = target[:context_length]
            actual = target[context_length:]
            model_payloads: dict[str, Any] = {}
            for prediction_index in predictions_by_sample.get(sample_ord, []):
                source_id = int(prediction_index["source_id"])
                source = self.prediction_sources[source_id]
                prediction = self._read_json(
                    int(source["fd"]),
                    int(prediction_index["byte_offset"]),
                    int(prediction_index["byte_length"]),
                )
                forecast = prediction["forecast"]
                model_payloads[str(source["model_id"])] = {
                    "forecast": forecast,
                    "metrics": _prediction_metrics(
                        actual,
                        forecast,
                        float(sample["mase_scale"]),
                    ),
                    "modelGroup": None,
                    "inputAdaptation": prediction.get("input_adaptation"),
                }
            for model_id in model_ids:
                if model_id not in model_payloads:
                    missing.append(
                        {"intensity": int(sample["intensity"]), "modelId": model_id}
                    )
            target_feature = sample.get("target_feature")
            intensity_payloads.append(
                {
                    "intensity": int(sample["intensity"]),
                    "masterSampleId": str(sample["master_sample_id"]),
                    "targetStrength": sample.get("target_feature_value"),
                    "targetRelativeLevel": (int(sample["intensity"]) - 1) / 4,
                    "targetFeature": target_feature,
                    "realizedFeature": (
                        sample.get("realized_features", {}).get(target_feature)
                        if target_feature
                        else None
                    ),
                    "history": history,
                    "actual": actual,
                    "models": model_payloads,
                }
            )
        member = (
            int(group["counterfactual_member"])
            if group["counterfactual_member"] is not None
            else None
        )
        return {
            "group": {
                "id": str(group["group_id"]),
                "datasetId": str(group["dataset_id"]),
                "capabilityId": str(group["capability_id"]),
                "seedIndex": int(group["seed_index"]),
                "familyRole": str(group["family_role"]),
                "evaluationTable": str(group["evaluation_table"]),
                "counterfactualMember": member,
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

    def __init__(
        self,
        address: tuple[str, int],
        explorer: SampleExplorer | V8SampleExplorer,
    ):
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
    data_dir = resolve_data_dir(args.data_dir)
    print(f"sample explorer data: {data_dir}", flush=True)
    index_path = (
        args.index_path.resolve()
        if args.index_path is not None
        else data_dir / DEFAULT_INDEX_NAME
    )
    cafe_layout = is_v8_experiment_dir(data_dir)
    if cafe_layout:
        artifacts = ensure_v8_index(
            data_dir,
            index_path,
            rebuild=args.rebuild_index,
            progress=lambda message: print(message, flush=True),
        )
    else:
        ensure_index(
            data_dir,
            index_path,
            rebuild=args.rebuild_index,
            progress=lambda message: print(message, flush=True),
        )
        artifacts = None
    if args.build_index_only:
        return 0
    explorer = (
        V8SampleExplorer(data_dir, index_path, artifacts)
        if artifacts is not None
        else SampleExplorer(data_dir, index_path)
    )
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
