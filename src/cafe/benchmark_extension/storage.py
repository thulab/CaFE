from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from cafe import core as protocol


PARQUET_ARTIFACT_SCHEMA = "cafe.compact_parquet_rows.v1"
DEFAULT_COMPRESSION = "zstd"
DEFAULT_COMPRESSION_LEVEL = 3
DEFAULT_ROW_GROUP_SIZE = 16_384


def _envelope(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep filter columns typed and place heterogeneous metadata in one payload.

    Mechanism contracts have capability-specific nested metadata. A canonical
    JSON payload keeps that schema evolvable while Parquet still supplies
    compression, typed partition keys, batched scans, and atomic shards. Dense
    time-series arrays are excluded by the caller before reaching this layer.
    """

    level = row.get("capability_level")
    return {
        "artifact_schema": PARQUET_ARTIFACT_SCHEMA,
        "record_kind": str(row.get("record_kind") or row.get("evaluation_table") or "row"),
        "dataset_id": None if row.get("dataset_id") is None else str(row["dataset_id"]),
        "official_instance_id": (
            None
            if row.get("official_instance_id") is None
            else str(row["official_instance_id"])
        ),
        "sample_id": None if row.get("sample_id") is None else str(row["sample_id"]),
        "capability_id": (
            None if row.get("capability_id") is None else str(row["capability_id"])
        ),
        "capability_level": None if level is None else int(level),
        "available": (
            None if row.get("available") is None else bool(row.get("available"))
        ),
        "payload_json": protocol.canonical_json(dict(row)),
    }


class CompactParquetWriter:
    """Buffered, atomic writer for compact heterogeneous contract rows."""

    def __init__(
        self,
        path: Path,
        *,
        batch_size: int = DEFAULT_ROW_GROUP_SIZE,
    ) -> None:
        self.path = path.resolve()
        self.temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self.batch_size = max(1, int(batch_size))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self._buffer.append(_envelope(row))
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.temporary,
                table.schema,
                compression=DEFAULT_COMPRESSION,
                compression_level=DEFAULT_COMPRESSION_LEVEL,
                use_dictionary=True,
                write_statistics=True,
            )
        self._writer.write_table(table, row_group_size=self.batch_size)
        self.row_count += len(self._buffer)
        self._buffer.clear()

    def close(self) -> int:
        self._flush()
        if self._writer is None:
            empty = pa.table(
                {
                    "artifact_schema": pa.array([], type=pa.string()),
                    "record_kind": pa.array([], type=pa.string()),
                    "dataset_id": pa.array([], type=pa.string()),
                    "official_instance_id": pa.array([], type=pa.string()),
                    "sample_id": pa.array([], type=pa.string()),
                    "capability_id": pa.array([], type=pa.string()),
                    "capability_level": pa.array([], type=pa.int64()),
                    "available": pa.array([], type=pa.bool_()),
                    "payload_json": pa.array([], type=pa.string()),
                }
            )
            pq.write_table(
                empty,
                self.temporary,
                compression=DEFAULT_COMPRESSION,
                compression_level=DEFAULT_COMPRESSION_LEVEL,
            )
        else:
            self._writer.close()
            self._writer = None
        with self.temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self.temporary, self.path)
        return self.row_count

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.temporary.unlink(missing_ok=True)

    def __enter__(self) -> CompactParquetWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_compact_parquet(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    writer = CompactParquetWriter(path)
    try:
        for row in rows:
            writer.write(row)
        return writer.close()
    except Exception:
        writer.abort()
        raise


def iter_compact_parquet(
    paths: Path | Iterable[Path],
    *,
    batch_size: int = 8_192,
) -> Iterator[dict[str, Any]]:
    selected = [paths] if isinstance(paths, Path) else list(paths)
    for path in selected:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=max(1, int(batch_size)),
            columns=("artifact_schema", "payload_json"),
        ):
            schemas = batch.column(0).to_pylist()
            payloads = batch.column(1).to_pylist()
            for schema, payload in zip(schemas, payloads, strict=True):
                if schema != PARQUET_ARTIFACT_SCHEMA:
                    raise ValueError(f"unsupported compact parquet schema in {path}")
                row = json.loads(str(payload))
                if not isinstance(row, dict):
                    raise ValueError(f"compact parquet payload is not an object: {path}")
                yield row


def parquet_file_record(path: Path, *, row_count: int) -> dict[str, Any]:
    return {
        **protocol.file_record(path),
        "format": "parquet",
        "compression": DEFAULT_COMPRESSION,
        "row_count": int(row_count),
    }


def validate_parquet_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if record.get("format") != "parquet":
        raise ValueError(f"artifact is not declared as parquet: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"parquet artifact size mismatch: {path}")
    if protocol.file_sha256(path) != record["sha256"]:
        raise ValueError(f"parquet artifact hash mismatch: {path}")
    metadata = pq.read_metadata(path)
    if metadata.num_rows != int(record["row_count"]):
        raise ValueError(f"parquet artifact row count mismatch: {path}")
    return path


class PredictionParquetWriter:
    """Atomic float32 prediction writer with bounded row buffering."""

    def __init__(self, path: Path, *, batch_size: int = 2_048) -> None:
        self.path = path.resolve()
        self.temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self.batch_size = max(1, int(batch_size))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def write(
        self,
        *,
        model_id: str,
        sample_id: str,
        forecast: Any,
        input_adaptation: Mapping[str, Any] | None = None,
    ) -> None:
        values = np.asarray(forecast, dtype=np.float32)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"prediction must be finite HxD for {sample_id}")
        self._buffer.append(
            {
                "schema_version": "cafe.benchmark_extension_prediction.v1",
                "model_id": str(model_id),
                "sample_id": str(sample_id),
                "horizon": int(values.shape[0]),
                "target_dim": int(values.shape[1]),
                "forecast": values.reshape(-1).tolist(),
                "input_adaptation_json": (
                    None
                    if input_adaptation is None
                    else protocol.canonical_json(dict(input_adaptation))
                ),
            }
        )
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(
            self._buffer,
            schema=pa.schema(
                [
                    ("schema_version", pa.string()),
                    ("model_id", pa.string()),
                    ("sample_id", pa.string()),
                    ("horizon", pa.int32()),
                    ("target_dim", pa.int32()),
                    ("forecast", pa.list_(pa.float32())),
                    ("input_adaptation_json", pa.string()),
                ]
            ),
        )
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.temporary,
                table.schema,
                compression=DEFAULT_COMPRESSION,
                compression_level=DEFAULT_COMPRESSION_LEVEL,
                use_dictionary=("schema_version", "model_id"),
                write_statistics=True,
            )
        self._writer.write_table(table, row_group_size=self.batch_size)
        self.row_count += len(self._buffer)
        self._buffer.clear()

    def close(self) -> int:
        self._flush()
        if self._writer is None:
            empty = pa.Table.from_pylist(
                [],
                schema=pa.schema(
                    [
                        ("schema_version", pa.string()),
                        ("model_id", pa.string()),
                        ("sample_id", pa.string()),
                        ("horizon", pa.int32()),
                        ("target_dim", pa.int32()),
                        ("forecast", pa.list_(pa.float32())),
                        ("input_adaptation_json", pa.string()),
                    ]
                ),
            )
            pq.write_table(empty, self.temporary, compression=DEFAULT_COMPRESSION)
        else:
            self._writer.close()
            self._writer = None
        os.replace(self.temporary, self.path)
        return self.row_count

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.temporary.unlink(missing_ok=True)


def iter_prediction_parquet(
    path: Path,
    *,
    batch_size: int = 4_096,
) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=max(1, int(batch_size))):
        for row in batch.to_pylist():
            if row.get("schema_version") != "cafe.benchmark_extension_prediction.v1":
                raise ValueError(f"unsupported prediction schema: {path}")
            horizon = int(row["horizon"])
            target_dim = int(row["target_dim"])
            forecast = np.asarray(row["forecast"], dtype=np.float32)
            if forecast.size != horizon * target_dim:
                raise ValueError(f"invalid forecast size for {row['sample_id']}")
            yield {
                "schema_version": row["schema_version"],
                "model_id": row["model_id"],
                "sample_id": row["sample_id"],
                "forecast": forecast.reshape(horizon, target_dim),
                "input_adaptation": (
                    None
                    if row.get("input_adaptation_json") is None
                    else json.loads(row["input_adaptation_json"])
                ),
            }


class TypedParquetWriter:
    """Atomic writer for homogeneous scalar/list metric tables."""

    def __init__(
        self,
        path: Path,
        *,
        schema: pa.Schema,
        batch_size: int = DEFAULT_ROW_GROUP_SIZE,
    ) -> None:
        self.path = path.resolve()
        self.temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self.schema = schema
        self.batch_size = max(1, int(batch_size))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self._buffer.append(dict(row))
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.temporary,
                self.schema,
                compression=DEFAULT_COMPRESSION,
                compression_level=DEFAULT_COMPRESSION_LEVEL,
                use_dictionary=True,
                write_statistics=True,
            )
        self._writer.write_table(table, row_group_size=self.batch_size)
        self.row_count += len(self._buffer)
        self._buffer.clear()

    def close(self) -> int:
        self._flush()
        if self._writer is None:
            pq.write_table(
                pa.Table.from_pylist([], schema=self.schema),
                self.temporary,
                compression=DEFAULT_COMPRESSION,
                compression_level=DEFAULT_COMPRESSION_LEVEL,
            )
        else:
            self._writer.close()
            self._writer = None
        os.replace(self.temporary, self.path)
        return self.row_count

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.temporary.unlink(missing_ok=True)
