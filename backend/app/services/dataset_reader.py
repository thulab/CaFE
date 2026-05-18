from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DatasetReadResult:
    columns: list[str]
    rows: list[dict[str, str]]
    timestamps: list[datetime]
    target_values: list[list[float]]
    frequency: str
    encoding: str
    delimiter: str

    @property
    def row_count(self) -> int:
        return len(self.rows)


class DatasetReader(Protocol):
    def read(
        self,
        path: Path,
        time_column: str,
        target_columns: list[str],
        frequency: str | None = None,
    ) -> DatasetReadResult:
        ...
