from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from ...config import future_known_covariates, infer_difficulty, infer_periods_for_track, infer_trend_type
from ..domain import (
    ChannelLayout,
    CsvBatchLoadRequest,
    DatasetLoadRequest,
    DatasetSourceType,
    SeriesSample,
    SeriesTruth,
    TrackKind,
    TrackSpec,
)
from .base import DataLoaderError, DatasetLoader


class CsvLoaderError(DataLoaderError):
    pass


class CsvDatasetLoader(DatasetLoader):
    source_type = DatasetSourceType.CSV

    def load_samples(self, request: DatasetLoadRequest, track_spec: TrackSpec) -> list[SeriesSample]:
        if not isinstance(request, CsvBatchLoadRequest):
            raise CsvLoaderError(f"csv loader received unsupported request type: {type(request).__name__}")

        csv_path = Path(request.csv_path).expanduser().resolve()
        if not csv_path.exists():
            raise CsvLoaderError(f"csv file not found: {csv_path}")

        grouped_rows = self._read_rows(csv_path=csv_path, request=request)
        samples: list[SeriesSample] = []
        for index, (sample_key, rows) in enumerate(sorted(grouped_rows.items())):
            if request.max_samples is not None and index >= request.max_samples:
                break
            rows.sort(key=lambda item: item["step"])
            expected = request.input_length + request.prediction_length
            if len(rows) < expected:
                raise CsvLoaderError(f"sample {sample_key} has only {len(rows)} rows, requires at least {expected}")

            trimmed_rows = rows[:expected]
            primary = request.primary_target_column or request.target_columns[0]
            history = [row["targets"][primary] for row in trimmed_rows[: request.input_length]]
            target = [row["targets"][primary] for row in trimmed_rows[request.input_length : expected]]

            input_channel_values = {
                channel: [row["inputs"][channel] for row in trimmed_rows[: request.input_length]]
                for channel in track_spec.input_channels
                if channel in trimmed_rows[0]["inputs"]
            }
            input_channel_values.setdefault(primary, list(history))

            target_channel_values = {
                channel: [row["targets"][channel] for row in trimmed_rows[request.input_length : expected]]
                for channel in request.target_columns
            }
            target_channel_values.setdefault(primary, list(target))

            future_known_channel_values = {
                channel: [row["future_known"][channel] for row in trimmed_rows]
                for channel in request.future_known_columns
                if channel in trimmed_rows[0]["future_known"]
            }
            auxiliary_covariates = {
                channel: [row["inputs"][channel] for row in trimmed_rows]
                for channel in request.covariate_columns
                if channel in trimmed_rows[0]["inputs"] and channel != primary
            }
            covariates = {
                name: list(values)
                for name, values in {
                    **{
                        channel: [row["inputs"][channel] for row in trimmed_rows]
                        for channel in request.input_columns
                        if channel != primary and channel in trimmed_rows[0]["inputs"]
                    },
                    **auxiliary_covariates,
                    **future_known_channel_values,
                }.items()
            }
            truth = self._build_truth(series=history + target, track=track_spec.track)
            notes: dict[str, object] = {
                "source_type": request.source_type.value,
                "source_path": str(csv_path),
                "row_count": len(trimmed_rows),
                "future_known_covariates": self._future_known_covariates(request.future_known_columns or request.covariate_columns),
            }
            samples.append(
                SeriesSample(
                    sample_id=str(sample_key),
                    history=history,
                    target=target,
                    covariates=covariates,
                    input_channel_values=input_channel_values,
                    target_channel_values=target_channel_values,
                    future_known_channel_values=future_known_channel_values,
                    channel_layout=ChannelLayout(
                        primary_target_channel=primary,
                        input_channels=list(track_spec.input_channels),
                        target_channels=list(track_spec.target_channels),
                        future_known_channels=list(track_spec.future_known_channels),
                    ),
                    track_tags=[track_spec.track_variant_id, request.source_type.value, "csv_loaded"],
                    truth=truth,
                    notes=notes,
                )
            )

        if not samples:
            raise CsvLoaderError(f"csv file {csv_path} did not produce any samples")
        return samples

    def _read_rows(self, csv_path: Path, request: CsvBatchLoadRequest) -> dict[str, list[dict[str, object]]]:
        grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        target_columns = list(dict.fromkeys(request.target_columns))
        input_columns = [column for column in dict.fromkeys(request.input_columns + request.covariate_columns) if column not in target_columns]
        future_known_columns = [column for column in dict.fromkeys(request.future_known_columns) if column not in target_columns]
        required_columns = [request.sample_id_column, request.step_column, *target_columns, *input_columns, *future_known_columns]

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=request.delimiter)
            missing_columns = {column for column in required_columns if reader.fieldnames is None or column not in reader.fieldnames}
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise CsvLoaderError(f"csv file missing required columns: {missing}")

            for row_number, row in enumerate(reader, start=2):
                sample_id = row[request.sample_id_column]
                if sample_id is None or sample_id == "":
                    raise CsvLoaderError(f"row {row_number} missing sample id")
                try:
                    step = int(row[request.step_column])
                    targets = {column: float(row[column]) for column in target_columns}
                    inputs = {column: float(row[column]) if row[column] not in (None, "") else 0.0 for column in input_columns}
                    future_known = {
                        column: float(row[column]) if row[column] not in (None, "") else 0.0 for column in future_known_columns
                    }
                except ValueError as exc:
                    raise CsvLoaderError(f"row {row_number} contains invalid numeric value: {exc}") from exc
                grouped_rows[sample_id].append(
                    {
                        "step": step,
                        "targets": targets,
                        "inputs": inputs,
                        "future_known": future_known,
                    }
                )
        return grouped_rows

    def _build_truth(self, series: list[float], track: TrackKind) -> SeriesTruth:
        trend_type = infer_trend_type(series)
        difficulty = infer_difficulty(series)
        noise_level = self._estimate_noise(series)
        periods = self._infer_periods(track=track, length=len(series))
        return SeriesTruth(
            trend_type=trend_type,
            periods=periods,
            dominant_period=periods[-1],
            amplitude_mode="stable",
            phase_shift=False,
            noise_level=noise_level,
            difficulty=difficulty,
        )

    def _infer_periods(self, track: TrackKind, length: int) -> list[int]:
        return infer_periods_for_track(track, length)

    def _estimate_noise(self, series: list[float]) -> float:
        if len(series) < 3:
            return 0.0
        second_diffs = [series[index] - 2 * series[index - 1] + series[index - 2] for index in range(2, len(series))]
        variance = sum(value * value for value in second_diffs) / max(len(second_diffs), 1)
        return round(math.sqrt(max(variance, 0.0)), 4)

    def _future_known_covariates(self, covariate_columns: list[str]) -> list[str]:
        return future_known_covariates(covariate_columns)
