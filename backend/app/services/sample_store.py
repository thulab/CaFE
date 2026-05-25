import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.sample import SampleIndex
from app.services.dataset_load_service import SampleWindow
from app.services.tsfile_store import TsFileSlicer


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_checksum(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _ms_to_iso(ms: int) -> str:
    """ms epoch → ISO 8601。与 TsFileStore.write 的 int(dt.timestamp()*1000) 互逆（同机本地时区往返为恒等）。"""
    return datetime.fromtimestamp(ms / 1000).isoformat()


class SampleStore:
    """指针化样本存储：样本不再逐窗物化值，按 SampleIndex 的行号指针从 TsFile 现切窗口。"""

    def write_samples(
        self,
        shard_id: str,
        dataset_id: str,
        tsfile_path: str | Path,
        windows: list[SampleWindow],
        target_columns: list[str],
    ) -> list[SampleIndex]:
        sample_indexes: list[SampleIndex] = []
        with TsFileSlicer(tsfile_path) as slicer:
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
                    materialized=True,
                    materialized_sample_uri=str(tsfile_path),
                )
                sample.storage_ref = {
                    "dataset_id": dataset_id,
                    "shard_id": shard_id,
                    "sample_id": sample.sample_id,
                    "sample_index": line_number,
                    "target_columns": target_columns,
                    "context": [window.context_start, window.context_end],
                    "horizon": [window.horizon_start, window.horizon_end],
                }
                record = self._assemble(slicer, sample.storage_ref)
                sample.checksum = canonical_json_checksum(record)
                sample_indexes.append(sample)
        return sample_indexes

    def read_by_ref(self, materialized_sample_uri: str | Path, storage_ref: dict[str, Any]) -> dict[str, Any]:
        with TsFileSlicer(materialized_sample_uri) as slicer:
            return self._assemble(slicer, storage_ref)

    def _assemble(self, slicer: TsFileSlicer, storage_ref: dict[str, Any]) -> dict[str, Any]:
        dataset_id = storage_ref["dataset_id"]
        target_columns = storage_ref["target_columns"]
        context_start, context_end = storage_ref["context"]
        horizon_start, horizon_end = storage_ref["horizon"]
        target_history = slicer.slice(dataset_id, target_columns, context_start, context_end + 1)
        target_future = slicer.slice(dataset_id, target_columns, horizon_start, horizon_end + 1)
        history_timestamps = [_ms_to_iso(ms) for ms in slicer.slice_timestamps(dataset_id, context_start, context_end + 1)]
        future_timestamps = [_ms_to_iso(ms) for ms in slicer.slice_timestamps(dataset_id, horizon_start, horizon_end + 1)]
        return {
            "schema_version": "sample.v1",
            "sample_id": storage_ref["sample_id"],
            "shard_id": storage_ref["shard_id"],
            "sample_index": storage_ref["sample_index"],
            "target_column_names": target_columns,
            "history_timestamps": history_timestamps,
            "future_timestamps": future_timestamps,
            "target_history": target_history,
            "target_future": target_future,
            "history_cov": [],
            "future_cov": [],
            "source_row_start": context_start,
            "source_row_end": horizon_end,
        }
