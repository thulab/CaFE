import hashlib
import json
from pathlib import Path
from typing import Any

from app.models.sample import SampleIndex
from app.services.dataset_load_service import SampleWindow
from app.services.dataset_reader import DatasetReadResult


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_checksum(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


class SampleStore:
    def __init__(self, samples_dir: Path):
        self.samples_dir = Path(samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def sample_uri(self, shard_id: str) -> Path:
        return self.samples_dir / f"{shard_id}.jsonl"

    def write_samples(
        self,
        shard_id: str,
        read_result: DatasetReadResult,
        windows: list[SampleWindow],
        target_columns: list[str],
    ) -> list[SampleIndex]:
        output = self.sample_uri(shard_id)
        sample_indexes: list[SampleIndex] = []
        with output.open("w", encoding="utf-8") as file:
            for line_number, window in enumerate(windows):
                sample = SampleIndex(
                    shard_id=shard_id,
                    sample_index=line_number,
                    context_start=window.context_start,
                    context_end=window.context_end,
                    horizon_start=window.horizon_start,
                    horizon_end=window.horizon_end,
                    target_columns=target_columns,
                    context_length=window.context_length,
                    horizon=window.horizon,
                    storage_ref={"line": line_number},
                    materialized=True,
                    materialized_sample_uri=str(output),
                )
                record = self._record_for_sample(sample, read_result, window, target_columns)
                sample.checksum = canonical_json_checksum(record)
                file.write(canonical_json(record) + "\n")
                sample_indexes.append(sample)
        return sample_indexes

    def read_by_ref(self, materialized_sample_uri: str | Path, storage_ref: dict[str, Any]) -> dict[str, Any]:
        line_number = int(storage_ref["line"])
        with Path(materialized_sample_uri).open(encoding="utf-8") as file:
            for current, line in enumerate(file):
                if current == line_number:
                    return json.loads(line)
        raise IndexError(f"sample line {line_number} not found")

    def _record_for_sample(
        self,
        sample: SampleIndex,
        read_result: DatasetReadResult,
        window: SampleWindow,
        target_columns: list[str],
    ) -> dict[str, Any]:
        history_range = range(window.context_start, window.context_end + 1)
        future_range = range(window.horizon_start, window.horizon_end + 1)
        return {
            "schema_version": "sample.v1",
            "sample_id": sample.sample_id,
            "shard_id": sample.shard_id,
            "sample_index": sample.sample_index,
            "target_column_names": target_columns,
            "history_timestamps": [read_result.timestamps[index].isoformat() for index in history_range],
            "future_timestamps": [read_result.timestamps[index].isoformat() for index in future_range],
            "target_history": [read_result.target_values[index] for index in history_range],
            "target_future": [read_result.target_values[index] for index in future_range],
            "history_cov": [],
            "future_cov": [],
            "source_row_start": window.source_row_start,
            "source_row_end": window.source_row_end,
        }
