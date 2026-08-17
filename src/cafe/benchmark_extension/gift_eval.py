from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol


GIFT_EVAL_ADAPTER_SCHEMA = "cafe.gift_eval_native_adapter.v1"
GIFT_EVAL_SOURCE_REVISION = (
    "SalesforceAIResearch/gift-eval@26df7582a5a2a2ef7602b5ded3a9a12fd4da74b2:"
    "src/gift_eval/data.py"
)
SHORT_PREDICTION_LENGTHS = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}
M4_PREDICTION_LENGTHS = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}
TERM_MULTIPLIERS = {"short": 1, "medium": 10, "long": 15}


def iter_gift_arrow_target_records(path: Path) -> Iterator[tuple[str, str, np.ndarray]]:
    """Stream native records from Arrow without materializing the whole table."""

    arrow_files = sorted(path.glob("data-*.arrow"))
    if len(arrow_files) != 1:
        raise ValueError(f"expected one canonical data-*.arrow in {path}")
    with pa.memory_map(str(arrow_files[0]), "r") as source:
        reader = pa_ipc.open_stream(source)
        required = {"item_id", "target", "freq"}
        if not required.issubset(reader.schema.names):
            raise ValueError(
                "GIFT-Eval Arrow schema missing "
                f"{sorted(required - set(reader.schema.names))}"
            )
        indexes = {name: reader.schema.get_field_index(name) for name in required}
        for batch in reader:
            item_ids = batch.column(indexes["item_id"])
            targets = batch.column(indexes["target"])
            frequencies = batch.column(indexes["freq"])
            for index in range(batch.num_rows):
                yield (
                    str(item_ids[index].as_py()),
                    str(frequencies[index].as_py()),
                    np.asarray(targets[index].as_py(), dtype=float),
                )


def gift_arrow_target_summary(path: Path) -> tuple[str, int, int]:
    frequencies: set[str] = set()
    minimum_length: int | None = None
    count = 0
    for _item_id, frequency, target in iter_gift_arrow_target_records(path):
        frequencies.add(frequency)
        values = np.asarray(target)
        length = int(values.shape[-1])
        minimum_length = length if minimum_length is None else min(minimum_length, length)
        count += 1
    if len(frequencies) != 1 or minimum_length is None:
        raise ValueError("GIFT-Eval config must contain records with one frequency")
    return next(iter(frequencies)), int(minimum_length), count


def read_gift_arrow_targets(path: Path) -> tuple[str, list[tuple[str, np.ndarray]]]:
    """Compatibility reader; execution paths should prefer the streaming API."""

    records: list[tuple[str, np.ndarray]] = []
    frequencies: set[str] = set()
    for item_id, frequency, target in iter_gift_arrow_target_records(path):
        frequencies.add(frequency)
        records.append((item_id, target))
    if len(frequencies) != 1:
        raise ValueError("GIFT-Eval config must have one frequency")
    return next(iter(frequencies)), records


@dataclass(frozen=True)
class GiftEvalInstance:
    dataset_id: str
    config_id: str
    item_id: str
    official_instance_id: str
    frequency: str
    term: str
    window_index: int
    window_count: int
    forecast_origin: int
    prediction_length: int
    history: np.ndarray
    future: np.ndarray
    future_observed_mask: np.ndarray
    history_covariates: np.ndarray
    future_covariates: np.ndarray
    covariate_column_names: tuple[str, ...]
    target_column_names: tuple[str, ...]
    source_target_length: int
    history_imputation: dict[str, object]

    @property
    def target_dim(self) -> int:
        return int(self.history.shape[1])

    @property
    def context_length(self) -> int:
        return int(self.history.shape[0])


def _normalized_frequency(frequency: str) -> str:
    raw = str(frequency).strip()
    aliases = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
    }
    normalized = aliases.get(raw, raw)
    if normalized.startswith("W-"):
        return "W"
    for suffix in ("A", "Q", "M", "W", "D", "H", "T", "S"):
        prefix = normalized[: -len(suffix)]
        if normalized.endswith(suffix) and (not prefix or prefix.isdigit()):
            return suffix
    raise ValueError(f"unsupported GIFT-Eval frequency {frequency!r}")


def prediction_length(
    dataset_id: str,
    frequency: str,
    *,
    term: str,
) -> int:
    try:
        multiplier = TERM_MULTIPLIERS[str(term)]
    except KeyError as error:
        raise ValueError(f"unsupported GIFT-Eval term {term!r}") from error
    normalized = _normalized_frequency(frequency)
    mapping = (
        M4_PREDICTION_LENGTHS
        if "m4" in dataset_id.lower()
        else SHORT_PREDICTION_LENGTHS
    )
    try:
        base = mapping[normalized]
    except KeyError as error:
        raise ValueError(
            f"unsupported GIFT-Eval prediction frequency {frequency!r}"
        ) from error
    return int(multiplier * base)


def official_window_count(
    dataset_id: str,
    records: list[tuple[str, np.ndarray]],
    prediction_length_value: int,
) -> int:
    if "m4" in dataset_id.lower():
        return 1
    minimum_length = min(int(np.asarray(values).shape[-1]) for _, values in records)
    count = math.ceil(0.1 * minimum_length / int(prediction_length_value))
    return int(min(max(1, count), 20))


def official_window_count_from_minimum_length(
    dataset_id: str,
    minimum_length: int,
    prediction_length_value: int,
) -> int:
    if "m4" in dataset_id.lower():
        return 1
    count = math.ceil(0.1 * int(minimum_length) / int(prediction_length_value))
    return int(min(max(1, count), 20))


def official_forecast_origins(
    target_length: int,
    *,
    prediction_length_value: int,
    window_count: int,
) -> tuple[int, ...]:
    first = int(target_length) - int(prediction_length_value) * int(window_count)
    return tuple(
        first + index * int(prediction_length_value)
        for index in range(int(window_count))
    )


def _time_major(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array.T
    raise ValueError(f"unsupported GIFT-Eval target shape {array.shape}")


def _impute_history(history: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(history, dtype=float).copy()
    fractions: list[float] = []
    empty_channels: list[int] = []
    x = np.arange(values.shape[0], dtype=float)
    for channel in range(values.shape[1]):
        column = values[:, channel]
        finite = np.isfinite(column)
        fractions.append(float(np.mean(finite)))
        if not np.any(finite):
            values[:, channel] = 0.0
            empty_channels.append(channel)
        elif int(np.sum(finite)) == 1:
            values[:, channel] = float(column[finite][0])
        else:
            values[:, channel] = np.interp(x, x[finite], column[finite])
    return values, {
        "policy": "history_only_linear_interpolation_edge_hold_v1",
        "observed_fraction_by_target": fractions,
        "all_missing_target_indices": empty_channels,
    }


def _fill_future_for_storage(
    future: np.ndarray,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(future, dtype=float).copy()
    observed = np.isfinite(values)
    for channel in range(values.shape[1]):
        fallback = float(history[-1, channel])
        values[~observed[:, channel], channel] = fallback
    return values, observed


def _calendar_period(frequency: str) -> int:
    normalized = _normalized_frequency(frequency)
    return {
        "A": 1,
        "Q": 4,
        "M": 12,
        "W": 52,
        "D": 7,
        "H": 24,
        "T": 60,
        "S": 60,
    }[normalized]


def _known_calendar_covariates(
    length: int,
    frequency: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    period = _calendar_period(frequency)
    if period <= 1:
        return np.zeros((length, 0), dtype=float), ()
    index = np.arange(length, dtype=float)
    angle = 2.0 * math.pi * index / float(period)
    values = np.column_stack((np.sin(angle), np.cos(angle)))
    return values, (f"calendar_sin_p{period}", f"calendar_cos_p{period}")


def gift_eval_asset_path(dataset_id: str, gift_eval_dir: Path) -> Path:
    dataset = protocol.resolve_dataset(dataset_id)
    if dataset.real_data_adapter not in {"gift_arrow", "gift_hierarchical_sales"}:
        raise ValueError(
            f"v7 currently supports GIFT-Eval datasets only: {dataset_id}"
        )
    path = gift_eval_dir.resolve() / dataset.asset_name
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def iter_gift_eval_instances(
    dataset_id: str,
    gift_eval_dir: Path,
    *,
    term: str = "short",
    max_instances: int | None = None,
) -> Iterator[GiftEvalInstance]:
    """Yield the exact native GIFT-Eval test instances in source order.

    The origin formula mirrors ``gift_eval.data.Dataset.test_data``:
    split at ``-prediction_length * windows`` and generate ``windows``
    instances with distance equal to the prediction length. Native
    multivariate records remain multivariate.
    """

    dataset = protocol.resolve_dataset(dataset_id)
    asset_path = gift_eval_asset_path(dataset_id, gift_eval_dir)
    frequency, minimum_length, _record_count = gift_arrow_target_summary(asset_path)
    horizon = prediction_length(dataset_id, frequency, term=term)
    windows = official_window_count_from_minimum_length(
        dataset_id, minimum_length, horizon
    )
    emitted = 0
    for item_id, record_frequency, raw_target in iter_gift_arrow_target_records(asset_path):
        if record_frequency != frequency:
            raise ValueError("GIFT-Eval config must have one frequency")
        for instance in gift_eval_instances_for_record(
            dataset_id=dataset_id,
            config_id=dataset.config_id,
            item_id=item_id,
            frequency=frequency,
            term=term,
            raw_target=raw_target,
            prediction_length_value=horizon,
            window_count=windows,
        ):
            yield instance
            emitted += 1
            if max_instances is not None and emitted >= int(max_instances):
                return


def gift_eval_instances_for_record(
    *,
    dataset_id: str,
    config_id: str,
    item_id: str,
    frequency: str,
    term: str,
    raw_target: np.ndarray,
    prediction_length_value: int,
    window_count: int,
    maximum_windows: int | None = None,
) -> Iterator[GiftEvalInstance]:
    """Build official windows for one native record without cross-record state."""

    target = _time_major(raw_target)
    calendar, calendar_names = _known_calendar_covariates(target.shape[0], frequency)
    origins = official_forecast_origins(
        target.shape[0],
        prediction_length_value=prediction_length_value,
        window_count=window_count,
    )
    emitted = 0
    for window_index, origin in enumerate(origins):
        if origin <= 0 or origin + prediction_length_value > target.shape[0]:
            continue
        raw_history = target[:origin]
        raw_future = target[origin : origin + prediction_length_value]
        history, imputation = _impute_history(raw_history)
        future, observed = _fill_future_for_storage(raw_future, history)
        token = protocol.safe_id(str(item_id)) or "item"
        instance_id = (
            f"gift__{protocol.safe_id(dataset_id)}__{token}__"
            f"w{window_index:02d}__o{origin}"
        )
        yield GiftEvalInstance(
            dataset_id=dataset_id,
            config_id=config_id,
            item_id=str(item_id),
            official_instance_id=instance_id,
            frequency=frequency,
            term=str(term),
            window_index=int(window_index),
            window_count=int(window_count),
            forecast_origin=int(origin),
            prediction_length=int(prediction_length_value),
            history=history,
            future=future,
            future_observed_mask=observed,
            history_covariates=calendar[:origin],
            future_covariates=calendar[origin : origin + prediction_length_value],
            covariate_column_names=calendar_names,
            target_column_names=tuple(
                f"target_{index}" for index in range(history.shape[1])
            ),
            source_target_length=int(target.shape[0]),
            history_imputation=imputation,
        )
        emitted += 1
        if maximum_windows is not None and emitted >= int(maximum_windows):
            return
