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
        target_columns: list[str],
        frequency: str | None = None,
    ) -> DatasetReadResult:
        if len(target_columns) != 1:
            raise ApiError("csv_single_target_only", "MVP supports exactly one target column")

        text, encoding = self._read_text(path)
        delimiter = self._detect_delimiter(text)
        rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
        if not rows:
            raise ApiError("csv_missing_header", "CSV must include a header row")

        columns = [name.strip().removeprefix("\ufeff") for name in rows[0]]
        if self._looks_like_data_row(columns):
            raise ApiError("csv_missing_header", "CSV must include a header row")
        if len(set(columns)) != len(columns):
            raise ApiError("csv_duplicate_columns", "CSV column names must be unique")
        if time_column not in columns:
            raise ApiError("csv_time_column_missing", "time_column was not found", {"time_column": time_column})
        for target_column in target_columns:
            if target_column not in columns:
                raise ApiError("csv_target_column_missing", "target column was not found", {"target_column": target_column})

        dict_rows: list[dict[str, str]] = []
        timestamps: list[datetime] = []
        target_values: list[list[float]] = []
        seen_times: set[datetime] = set()
        target_column = target_columns[0]

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

            raw_target = values.get(target_column, "")
            if raw_target == "":
                raise ApiError("csv_target_missing", "target value is missing", {"row_index": row_index, "target_column": target_column})
            try:
                target = float(raw_target)
            except ValueError as exc:
                raise ApiError("csv_target_not_float", "target value must be numeric", {"row_index": row_index}) from exc
            if not math.isfinite(target):
                raise ApiError("csv_target_not_finite", "target value must be finite", {"row_index": row_index})

            dict_rows.append(values)
            target_values.append([target])

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
            target_values=target_values,
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
