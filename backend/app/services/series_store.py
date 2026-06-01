"""SQLite 逐点行序列存储：取代 per-shard TsFile，作为样本真值的单一来源。

- write     —— 把全列矩阵逐行写入 SeriesPoint（每点一行 + {列:值} 字典）。
- slice     —— 按 (shard_id, row_index) 闭区间范围查询，返回 row-major 值矩阵。
- slice_timestamps —— 同范围取 ISO 时间戳。

行号区间一律**闭区间** [row_start, row_end]，对齐 SampleWindow 与 SQL BETWEEN，
调用方无需 +1。
"""
from __future__ import annotations

from datetime import datetime

from sqlmodel import Session, select

from app.models.series_point import SeriesPoint


class SeriesStore:
    def write(
        self,
        session: Session,
        shard_id: str,
        timestamps: list[datetime],
        columns: list[str],
        values: list[list[float]],
    ) -> None:
        """写入 N 行；values 为 [N, C] 行主序，列序对应 columns。"""
        for row_index, (ts, row) in enumerate(zip(timestamps, values, strict=True)):
            session.add(
                SeriesPoint(
                    shard_id=shard_id,
                    row_index=row_index,
                    ts=ts.isoformat(),
                    values_json={col: float(row[i]) for i, col in enumerate(columns)},
                )
            )
        session.flush()  # 让同事务内的后续切片（算 checksum）可见

    def _rows(self, session: Session, shard_id: str, row_start: int, row_end: int) -> list[SeriesPoint]:
        return list(
            session.exec(
                select(SeriesPoint)
                .where(
                    SeriesPoint.shard_id == shard_id,
                    SeriesPoint.row_index >= row_start,
                    SeriesPoint.row_index <= row_end,
                )
                .order_by(SeriesPoint.row_index)
            ).all()
        )

    def slice(
        self,
        session: Session,
        shard_id: str,
        columns: list[str],
        row_start: int,
        row_end: int,
    ) -> list[list[float]]:
        """返回闭区间 [row_start, row_end] 各行在 columns 上的值，shape [n, len(columns)]。"""
        rows = self._rows(session, shard_id, row_start, row_end)
        return [[float(row.values_json[col]) for col in columns] for row in rows]

    def slice_timestamps(self, session: Session, shard_id: str, row_start: int, row_end: int) -> list[str]:
        rows = self._rows(session, shard_id, row_start, row_end)
        return [row.ts for row in rows]
