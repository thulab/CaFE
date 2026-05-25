"""TsFile 输入读取器：把单设备（表模型）TsFile 读成 DatasetReadResult（与 CSV 同契约）。

设计（见 spec input-readers-and-sqlite-storage）：CSV 或 TsFile 输入各自经 reader →
统一的 DatasetReadResult（内存矩阵）→ 再存进 SQLite。约束：单设备 + 表模型；
不等间隔 / 混合时区由共享的 validate_time_axis 拒绝（与 CSV 同一套时间轴校验）。
TsFile 时间是内建轴，`time_column` 参数被忽略。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.core.errors import ApiError
from app.services.dataset_reader import DatasetReadResult
from app.services.time_axis import validate_time_axis


class TsFileDatasetReader:
    def read(
        self,
        path: Path,
        time_column: str,
        value_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> DatasetReadResult:
        del time_column  # TsFile 时间为内建轴，无具名时间列
        import tsfile

        tf = tsfile.TsFileDataFrame(str(path))
        try:
            series = list(tf.list_timeseries())
            if not series:
                raise ApiError("tsfile_empty", "tsfile contains no timeseries", {"path": str(path)})
            # series 形如 "<table>.<device>.<measurement>"；device 维 = 去掉最后一段。
            devices = {".".join(name.split(".")[:-1]) for name in series}
            if len(devices) != 1:
                raise ApiError(
                    "tsfile_multiple_devices",
                    "MVP supports exactly one device per tsfile",
                    {"devices": sorted(devices)},
                )
            prefix = next(iter(devices))
            discovered = [name.split(".")[-1] for name in series if name.startswith(prefix + ".")]

            if value_columns:
                missing = [column for column in value_columns if column not in discovered]
                if missing:
                    raise ApiError(
                        "tsfile_value_column_missing",
                        "requested value column not found in tsfile",
                        {"missing": missing, "available": discovered},
                    )
                columns = list(value_columns)
            else:
                columns = discovered

            ts_ms = [int(value) for value in tf[f"{prefix}.{columns[0]}"].timestamps[:]]
            row_count = len(ts_ms)
            timestamps = [datetime.fromtimestamp(ms / 1000) for ms in ts_ms]
            column_arrays = {col: [float(v) for v in tf[f"{prefix}.{col}"][0:row_count]] for col in columns}
        finally:
            tf.__exit__(None, None, None)

        values = [[column_arrays[col][row] for col in columns] for row in range(row_count)]
        inferred_frequency = validate_time_axis(timestamps, frequency)
        return DatasetReadResult(
            columns=columns,
            rows=[{} for _ in range(row_count)],
            timestamps=timestamps,
            value_columns=columns,
            values=values,
            frequency=inferred_frequency,
            encoding="tsfile",
            delimiter="",
        )
