import hashlib
import json
from typing import Any

from sqlmodel import Session

from app.models.sample import SampleIndex
from app.services.dataset_load_service import SampleWindow
from app.services.dataset_reader import DatasetReadResult
from app.services.series_store import SeriesStore


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_checksum(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


class SampleStore:
    """指针化样本存储：样本只存行号指针，值从 SQLite 的 SeriesPoint 现切。

    `read_by_ref(session, storage_ref)` 用调用方的 session 读 SeriesPoint
    （同引擎/事务，能看到已提交的序列），重拼回 sample.v1（对上层契约不变）。
    """

    # ----- 写：只建 SampleIndex 指针 + 内容 checksum（不含随机 ID，跨加载可比，#7）-----
    def write_samples(
        self,
        shard_id: str,
        windows: list[SampleWindow],
        target_columns: list[str],
        covariate_columns: list[str],
        read_result: DatasetReadResult,
    ) -> list[SampleIndex]:
        target_matrix = read_result.column_matrix(target_columns)
        covariate_matrix = read_result.column_matrix(covariate_columns) if covariate_columns else []
        sample_indexes: list[SampleIndex] = []
        for line_number, window in enumerate(windows):
            sample = SampleIndex(
                shard_id=shard_id,
                sample_index=line_number,
                context_start=window.context_start,
                context_end=window.context_end,
                horizon_start=window.horizon_start,
                horizon_end=window.horizon_end,
                target_columns=target_columns,
                covariate_columns=covariate_columns,
                context_length=window.context_length,
                horizon=window.horizon,
                materialized=False,
            )
            sample.storage_ref = {
                "shard_id": shard_id,
                "sample_id": sample.sample_id,
                "sample_index": line_number,
                "target_columns": target_columns,
                "covariate_columns": covariate_columns,
                "context": [window.context_start, window.context_end],
                "horizon": [window.horizon_start, window.horizon_end],
            }
            sample.checksum = canonical_json_checksum(
                self._content(read_result, target_matrix, covariate_matrix, window, target_columns, covariate_columns)
            )
            sample_indexes.append(sample)
        return sample_indexes

    def _content(
        self,
        read_result: DatasetReadResult,
        target_matrix: list[list[float]],
        covariate_matrix: list[list[float]],
        window: SampleWindow,
        target_columns: list[str],
        covariate_columns: list[str],
    ) -> dict[str, Any]:
        """checksum 输入：仅样本内容（值/时间戳/列名/行范围），不含随机 sample_id/shard_id。"""
        history = range(window.context_start, window.context_end + 1)
        future = range(window.horizon_start, window.horizon_end + 1)
        return {
            "target_column_names": target_columns,
            "covariate_column_names": covariate_columns,
            "history_timestamps": [read_result.timestamps[i].isoformat() for i in history],
            "future_timestamps": [read_result.timestamps[i].isoformat() for i in future],
            "target_history": [target_matrix[i] for i in history],
            "target_future": [target_matrix[i] for i in future],
            "history_cov": [covariate_matrix[i] for i in history] if covariate_columns else [],
            "future_cov": [covariate_matrix[i] for i in future] if covariate_columns else [],
            "source_row_start": window.source_row_start,
            "source_row_end": window.source_row_end,
        }

    # ----- 读：用调用方 session 做 SeriesPoint 范围查询 → 重拼 sample.v1（契约不变）-----
    def read_by_ref(self, session: Session, storage_ref: dict[str, Any]) -> dict[str, Any]:
        return self._assemble(session, storage_ref)

    def _assemble(self, session: Session, storage_ref: dict[str, Any]) -> dict[str, Any]:
        shard_id = storage_ref["shard_id"]
        target_columns = storage_ref["target_columns"]
        covariate_columns = storage_ref.get("covariate_columns") or []
        context_start, context_end = storage_ref["context"]
        horizon_start, horizon_end = storage_ref["horizon"]
        store = SeriesStore()
        return {
            "schema_version": "sample.v1",
            "sample_id": storage_ref["sample_id"],
            "shard_id": shard_id,
            "sample_index": storage_ref["sample_index"],
            "target_column_names": target_columns,
            "covariate_column_names": covariate_columns,
            "history_timestamps": store.slice_timestamps(session, shard_id, context_start, context_end),
            "future_timestamps": store.slice_timestamps(session, shard_id, horizon_start, horizon_end),
            "target_history": store.slice(session, shard_id, target_columns, context_start, context_end),
            "target_future": store.slice(session, shard_id, target_columns, horizon_start, horizon_end),
            "history_cov": store.slice(session, shard_id, covariate_columns, context_start, context_end) if covariate_columns else [],
            "future_cov": store.slice(session, shard_id, covariate_columns, horizon_start, horizon_end) if covariate_columns else [],
            "source_row_start": context_start,
            "source_row_end": horizon_end,
        }
