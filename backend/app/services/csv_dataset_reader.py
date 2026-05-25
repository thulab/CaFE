import csv
import math
from datetime import datetime
from pathlib import Path

from app.core.errors import ApiError
from app.services.dataset_reader import DatasetReadResult


SUPPORTED_DELIMITERS = [",", "\t", ";"]


class CsvDatasetReader:
    def read(
        self,
        path: Path,
        time_column: str,
        value_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> DatasetReadResult:
        text, encoding = self._read_text(path)
        delimiter = self._detect_delimiter(text)
        rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
        if not rows:
            raise ApiError("csv_missing_header", "CSV must include a header row")

        columns = [name.strip().removeprefix("﻿") for name in rows[0]]
        if self._looks_like_data_row(columns):
            raise ApiError("csv_missing_header", "CSV must include a header row")
        if len(set(columns)) != len(columns):
            raise ApiError("csv_duplicate_columns", "CSV column names must be unique")
        if time_column not in columns:
            raise ApiError("csv_time_column_missing", "time_column was not found", {"time_column": time_column})

        # 全列摄入：未显式声明 value_columns 时，除时间列外的所有列都作为数值列。
        if value_columns:
            for value_column in value_columns:
                if value_column not in columns:
                    raise ApiError("csv_value_column_missing", "value column was not found", {"value_column": value_column})
            selected = list(value_columns)
        else:
            selected = [column for column in columns if column != time_column]
        if not selected:
            raise ApiError("csv_no_value_columns", "CSV must contain at least one value column besides the time column")

        dict_rows: list[dict[str, str]] = []
        timestamps: list[datetime] = []
        value_matrix: list[list[float]] = []
        seen_times: set[datetime] = set()

        for row_index, row in enumerate(rows[1:], start=2):
            if not row or all(not value.strip() for value in row):
                continue
            values = {column: row[position].strip() if position < len(row) else "" for position, column in enumerate(columns)}
            parsed_time = self._parse_time(values.get(time_column, ""), row_index)
            if parsed_time in seen_times:
                raise ApiError("csv_duplicate_timestamp", "time_column must not contain duplicate timestamps", {"row_index": row_index})
            if timestamps and parsed_time < timestamps[-1]:
                raise ApiError("csv_time_not_monotonic", "time_column must be strictly increasing", {"row_index": row_index})
            seen_times.add(parsed_time)
            timestamps.append(parsed_time)

            row_values: list[float] = []
            for value_column in selected:
                raw_value = values.get(value_column, "")
                if raw_value == "":
                    raise ApiError("csv_value_missing", "value is missing", {"row_index": row_index, "column": value_column})
                try:
                    parsed_value = float(raw_value)
                except ValueError as exc:
                    raise ApiError("csv_value_not_float", "value must be numeric", {"row_index": row_index, "column": value_column}) from exc
                if not math.isfinite(parsed_value):
                    raise ApiError("csv_value_not_finite", "value must be finite", {"row_index": row_index, "column": value_column})
                row_values.append(parsed_value)

            dict_rows.append(values)
            value_matrix.append(row_values)

        inferred_frequency = self._infer_frequency(timestamps)
        if frequency is not None and frequency != inferred_frequency:
            raise ApiError(
                "csv_frequency_mismatch",
                "provided frequency does not match inferred frequency",
                {"provided": frequency, "inferred": inferred_frequency},
            )

        return DatasetReadResult(
            columns=columns,
            rows=dict_rows,
            timestamps=timestamps,
            value_columns=selected,
            values=value_matrix,
            frequency=inferred_frequency,
            encoding=encoding,
            delimiter=delimiter,
        )

    def _read_text(self, path: Path) -> tuple[str, str]:
        raw = path.read_bytes()
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            raise ApiError("csv_encoding_unsupported", "CSV encoding must be UTF-8") from exc

    def _detect_delimiter(self, text: str) -> str:
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        if not first_line:
            raise ApiError("csv_missing_header", "CSV must include a header row")
        return max(SUPPORTED_DELIMITERS, key=first_line.count)

    def _parse_time(self, value: str, row_index: int) -> datetime:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ApiError("csv_time_parse_failed", "time_column contains an unsupported datetime", {"row_index": row_index}) from exc

    def _infer_frequency(self, timestamps: list[datetime]) -> str:
        if len(timestamps) < 2:
            raise ApiError("csv_frequency_not_inferable", "at least two timestamps are required")
        intervals = [timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))]
        first = intervals[0]
        if any(interval != first for interval in intervals):
            raise ApiError("csv_time_not_equidistant", "time_column must be equally spaced")
        total_seconds = int(first.total_seconds())
        if total_seconds <= 0:
            raise ApiError("csv_time_not_monotonic", "time_column must be strictly increasing")
        if total_seconds % 86400 == 0:
            return f"{total_seconds // 86400}d"
        if total_seconds % 3600 == 0:
            return f"{total_seconds // 3600}h"
        if total_seconds % 60 == 0:
            return f"{total_seconds // 60}m"
        return f"{total_seconds}s"

    def _looks_like_data_row(self, columns: list[str]) -> bool:
        if len(columns) < 2:
            return False
        try:
            datetime.fromisoformat(columns[0])
            float(columns[1])
            return True
        except ValueError:
            return False
