"""TsFile 输入读取器：把单设备（表模型）TsFile 读成 DatasetReadResult（与 CSV 同契约）。

设计（见 spec input-readers-and-sqlite-storage）：CSV 或 TsFile 输入各自经 reader →
统一的 DatasetReadResult（内存矩阵）→ 再存进 SQLite。约束：单设备 + 表模型；
不等间隔 / 混合时区由共享的 validate_time_axis 拒绝（与 CSV 同一套时间轴校验）。
TsFile 时间是内建轴，`time_column` 参数被忽略。
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.core.errors import ApiError
from app.services.dataset_reader import DatasetReadResult
from app.services.time_axis import validate_time_axis


class TsFileDatasetReader:
    def read(
        self,
        path: Path,
        time_column: str,
        target_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> DatasetReadResult:
        del time_column  # TsFile 时间为内建轴，无具名时间列
        import tsfile

        tf = tsfile.TsFileDataFrame(str(path))
        try:
            series = list(tf.list_timeseries())
            if not series:
                raise ApiError("tsfile_empty", "tsfile contains no timeseries", {"path": str(path)})
            selected_series, columns = _resolve_series_selection(series, target_columns)

            ts_ms = [int(value) for value in tf[selected_series[0]].timestamps[:]]
            row_count = len(ts_ms)
            timestamps = [datetime.fromtimestamp(ms / 1000, UTC).replace(tzinfo=None) for ms in ts_ms]
            column_arrays = {}
            for col, series_name in zip(columns, selected_series, strict=True):
                current_ts = [int(value) for value in tf[series_name].timestamps[:]]
                if current_ts != ts_ms:
                    raise ApiError(
                        "tsfile_target_axis_mismatch",
                        "selected target series must share the same timestamp axis",
                        {"target_columns": columns},
                    )
                column_arrays[col] = [float(v) for v in tf[series_name][0:row_count]]
        finally:
            tf.__exit__(None, None, None)

        values = [[column_arrays[col][row] for col in columns] for row in range(row_count)]
        inferred_frequency = validate_time_axis(timestamps, frequency)
        return DatasetReadResult(
            columns=columns,
            rows=[{} for _ in range(row_count)],
            timestamps=timestamps,
            target_columns=columns,
            values=values,
            frequency=inferred_frequency,
            encoding="tsfile",
            delimiter="",
        )


def tsfile_device_path(series_name: str) -> str:
    return ".".join(series_name.split(".")[:-1])


def tsfile_field_name(series_name: str) -> str:
    return series_name.split(".")[-1]


def _resolve_series_selection(series: list[str], target_columns: list[str] | None) -> tuple[list[str], list[str]]:
    """Resolve target columns to concrete TsFile series paths.

    For ordinary single-device files, target columns are measurement names. For
    TimeBench table-model shards that contain many timeseries_id tag values,
    callers may pass full series paths (e.g. ``table.device.value``) to select
    one device from a multi-device file.
    """
    selected_targets = list(target_columns or [])
    if not selected_targets or len(set(selected_targets)) != len(selected_targets):
        raise ApiError(
            "tsfile_target_columns_invalid",
            "at least one distinct target column must be selected",
            {"target_columns": selected_targets},
        )

    if all(column in series for column in selected_targets):
        selected_series = selected_targets
        devices = {tsfile_device_path(name) for name in selected_series}
        if len(devices) != 1:
            raise ApiError(
                "tsfile_multiple_devices",
                "MVP supports exactly one selected device per tsfile",
                {"devices": sorted(devices)},
            )
        return selected_series, selected_targets

    devices = {tsfile_device_path(name) for name in series}
    if len(devices) != 1:
        raise ApiError(
            "tsfile_multiple_devices",
            "MVP supports exactly one device per tsfile unless target_columns selects one full series path",
            {"devices": sorted(devices)[:50], "device_count": len(devices)},
        )
    prefix = next(iter(devices))
    discovered = [tsfile_field_name(name) for name in series if name.startswith(prefix + ".")]

    missing = [column for column in selected_targets if column not in discovered]
    if missing:
        raise ApiError(
            "tsfile_target_column_missing",
            "requested target column not found in tsfile",
            {"missing": missing, "available": discovered},
        )
    columns = selected_targets
    return [f"{prefix}.{column}" for column in columns], columns
