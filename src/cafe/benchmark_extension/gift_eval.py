from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from cafe import core as protocol
from cafe.benchmark_extension.adapters import (
    BenchmarkSourceSpec,
    BenchmarkTaskSpec,
)
from cafe.benchmark_extension.native import (
    GiftEvalInstance,
    NativeForecastInstance,
    default_seasonality,
    fill_unobserved_future,
    impute_dynamic_covariates,
    impute_history_only,
)


GIFT_EVAL_ADAPTER_SCHEMA = "cafe.gift_eval_native_adapter.v3"
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

# GIFT-Eval evaluates medium and long horizons on this fixed configuration
# subset.  CaFE currently carries twenty source configurations; the active
# suite is therefore the intersection between this list and DATASET_REGISTRY.
OFFICIAL_MEDIUM_LONG_CONFIG_IDS = frozenset(
    {
        "electricity/15T",
        "electricity/H",
        "solar/10T",
        "solar/H",
        "kdd_cup_2018_with_missing/H",
        "LOOP_SEATTLE/5T",
        "LOOP_SEATTLE/H",
        "SZ_TAXI/15T",
        "M_DENSE/H",
        "ett1/15T",
        "ett1/H",
        "ett2/15T",
        "ett2/H",
        "jena_weather/10T",
        "jena_weather/H",
        "bitbrains_fast_storage/5T",
        "bitbrains_rnd/5T",
        "bizitobs_application",
        "bizitobs_service",
        "bizitobs_l2c/5T",
        "bizitobs_l2c/H",
    }
)


def configured_dataset_ids_for_term(term: str) -> tuple[str, ...]:
    """Return the source-available official configuration suite for ``term``."""

    normalized = str(term)
    if normalized not in TERM_MULTIPLIERS:
        raise ValueError(f"unsupported GIFT-Eval term {term!r}")
    registered = tuple(protocol.DATASET_REGISTRY)
    if normalized == "short":
        return registered
    return tuple(
        dataset_id
        for dataset_id in registered
        if protocol.DATASET_REGISTRY[dataset_id].config_id
        in OFFICIAL_MEDIUM_LONG_CONFIG_IDS
    )


@dataclass(frozen=True)
class GiftArrowRecord:
    item_id: str
    frequency: str
    target: np.ndarray
    past_covariates: np.ndarray | None
    known_future_covariates: np.ndarray | None


def iter_gift_arrow_records(path: Path) -> Iterator[GiftArrowRecord]:
    """Stream native targets and benchmark-declared covariates from Arrow."""

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
        past_index = reader.schema.get_field_index("past_feat_dynamic_real")
        known_index = reader.schema.get_field_index("feat_dynamic_real")
        for batch in reader:
            item_ids = batch.column(indexes["item_id"])
            targets = batch.column(indexes["target"])
            frequencies = batch.column(indexes["freq"])
            past_covariates = (
                None if past_index < 0 else batch.column(past_index)
            )
            known_covariates = (
                None if known_index < 0 else batch.column(known_index)
            )
            for index in range(batch.num_rows):
                past_value = (
                    None
                    if past_covariates is None
                    else past_covariates[index].as_py()
                )
                known_value = (
                    None
                    if known_covariates is None
                    else known_covariates[index].as_py()
                )
                yield GiftArrowRecord(
                    item_id=str(item_ids[index].as_py()),
                    frequency=str(frequencies[index].as_py()),
                    target=np.asarray(targets[index].as_py(), dtype=float),
                    past_covariates=(
                        None
                        if past_value is None
                        else np.asarray(past_value, dtype=float)
                    ),
                    known_future_covariates=(
                        None
                        if known_value is None
                        else np.asarray(known_value, dtype=float)
                    ),
                )


def iter_gift_arrow_target_records(path: Path) -> Iterator[tuple[str, str, np.ndarray]]:
    """Compatibility target-only view over :func:`iter_gift_arrow_records`."""

    for record in iter_gift_arrow_records(path):
        yield record.item_id, record.frequency, record.target


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


def future_label_window_audit(
    raw_target: np.ndarray,
    *,
    prediction_length_value: int,
    window_count: int,
) -> dict[str, int]:
    """Count complete and incomplete official forecast windows.

    CaFE's formal extension uses complete-case forecast labels: every target
    cell in the complete H x D forecast horizon must be finite.  History
    missingness remains a separate input-imputation concern.
    """

    target = _time_major(raw_target)
    complete = partially_missing = fully_missing = 0
    for origin in official_forecast_origins(
        target.shape[0],
        prediction_length_value=prediction_length_value,
        window_count=window_count,
    ):
        if origin <= 0 or origin + prediction_length_value > target.shape[0]:
            continue
        observed = np.isfinite(target[origin : origin + prediction_length_value])
        if bool(np.all(observed)):
            complete += 1
        elif bool(np.any(observed)):
            partially_missing += 1
        else:
            fully_missing += 1
    return {
        "official_window_count": complete + partially_missing + fully_missing,
        "complete_future_label_count": complete,
        "partially_missing_future_label_count": partially_missing,
        "fully_missing_future_label_count": fully_missing,
    }


def _time_major(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array.T
    raise ValueError(f"unsupported GIFT-Eval target shape {array.shape}")


def _impute_history(history: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    values, _observed, audit = impute_history_only(history)
    return values, audit


def _fill_future_for_storage(
    future: np.ndarray,
    history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return fill_unobserved_future(future, history)


def _covariate_time_major(values: np.ndarray | None, target_length: int) -> np.ndarray:
    if values is None:
        return np.empty((target_length, 0), dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.size != target_length:
            raise ValueError("GIFT-Eval covariate length does not match target")
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"unsupported GIFT-Eval covariate shape {array.shape}")
    if array.shape[-1] == target_length:
        return array.T
    if array.shape[0] == target_length:
        return array
    raise ValueError("GIFT-Eval covariate length does not match target")


def _impute_covariates(
    history: np.ndarray,
    future: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    return impute_dynamic_covariates(history, future)


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
    selected_model_max_contexts: Mapping[str, int] | None = None,
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
    for record in iter_gift_arrow_records(asset_path):
        if record.frequency != frequency:
            raise ValueError("GIFT-Eval config must have one frequency")
        for instance in gift_eval_instances_for_record(
            dataset_id=dataset_id,
            config_id=dataset.config_id,
            item_id=record.item_id,
            frequency=frequency,
            term=term,
            raw_target=record.target,
            prediction_length_value=horizon,
            window_count=windows,
            raw_past_covariates=record.past_covariates,
            raw_known_future_covariates=record.known_future_covariates,
            selected_model_max_contexts=selected_model_max_contexts,
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
    raw_past_covariates: np.ndarray | None = None,
    raw_known_future_covariates: np.ndarray | None = None,
    selected_model_max_contexts: Mapping[str, int] | None = None,
) -> Iterator[GiftEvalInstance]:
    """Build complete-label official windows without cross-record state."""

    if maximum_windows is not None and int(maximum_windows) <= 0:
        return

    target = _time_major(raw_target)
    past_covariates = _covariate_time_major(
        raw_past_covariates, target.shape[0]
    )
    known_covariates = _covariate_time_major(
        raw_known_future_covariates, target.shape[0]
    )
    covariates = np.column_stack((past_covariates, known_covariates))
    covariate_names = tuple(
        [
            f"past_feat_dynamic_real_{index}"
            for index in range(past_covariates.shape[1])
        ]
        + [
            f"feat_dynamic_real_{index}"
            for index in range(known_covariates.shape[1])
        ]
    )
    availability = tuple(
        ["past_only"] * past_covariates.shape[1]
        + ["known_future"] * known_covariates.shape[1]
    )
    future_visible = tuple(value == "known_future" for value in availability)
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
        if not bool(np.all(observed)):
            continue
        history_covariates, future_covariates, covariate_imputation = (
            _impute_covariates(
                covariates[:origin],
                covariates[origin : origin + prediction_length_value],
            )
        )
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
            history_covariates=history_covariates,
            future_covariates=future_covariates,
            covariate_column_names=covariate_names,
            covariate_availability=availability,
            future_covariate_visible=future_visible,
            target_column_names=tuple(
                f"target_{index}" for index in range(history.shape[1])
            ),
            source_target_length=int(target.shape[0]),
            history_imputation={
                **imputation,
                "covariates": covariate_imputation,
            },
            benchmark_id="gift_eval",
            suite_id=str(term),
            task_id=f"{dataset_id}__{term}",
            history_observed_mask=np.ones_like(history, dtype=bool),
            covariate_types=tuple("continuous_numeric" for _ in covariate_names),
            source_locator={
                "item_id": str(item_id),
                "window_index": int(window_index),
                "forecast_origin": int(origin),
            },
            native_protocol={
                "split": "gift_eval_official_rolling_origin",
                "term": str(term),
                "window_count": int(window_count),
            },
            selected_model_max_contexts=dict(
                selected_model_max_contexts or {}
            ),
        )
        emitted += 1
        if maximum_windows is not None and emitted >= int(maximum_windows):
            return


class GiftEvalAdapter:
    """Adapter exposing GIFT-Eval tasks through the native window contract."""

    benchmark_id = "gift_eval"
    adapter_schema_version = GIFT_EVAL_ADAPTER_SCHEMA

    def __init__(self, source_root: Path, *, term: str = "short") -> None:
        if str(term) not in TERM_MULTIPLIERS:
            raise ValueError(f"unsupported GIFT-Eval term {term!r}")
        self.source_root = source_root.resolve()
        self.term = str(term)

    def list_tasks(self, suite_id: str) -> tuple[BenchmarkTaskSpec, ...]:
        term = self.term if not suite_id else str(suite_id)
        tasks: list[BenchmarkTaskSpec] = []
        for dataset_id in configured_dataset_ids_for_term(term):
            dataset = protocol.resolve_dataset(dataset_id)
            asset = gift_eval_asset_path(dataset_id, self.source_root)
            frequency, minimum_length, _ = gift_arrow_target_summary(asset)
            horizon = prediction_length(dataset_id, frequency, term=term)
            windows = official_window_count_from_minimum_length(
                dataset_id, minimum_length, horizon
            )
            native_config = {
                "dataset_id": dataset_id,
                "config_id": dataset.config_id,
                "term": term,
            }
            tasks.append(
                BenchmarkTaskSpec(
                    benchmark_id=self.benchmark_id,
                    suite_id=term,
                    task_id=f"{dataset_id}__{term}",
                    source_dataset_id=dataset_id,
                    display_name=dataset.logical_name,
                    frequency=frequency,
                    horizon=horizon,
                    seasonality=default_seasonality(frequency),
                    window_count=windows,
                    native_config=native_config,
                    config_sha256=protocol.json_sha256(native_config),
                )
            )
        return tuple(tasks)

    def iter_instances(
        self,
        task: BenchmarkTaskSpec,
        *,
        max_instances: int | None = None,
        selected_model_max_contexts: Mapping[str, int] | None = None,
    ) -> Iterator[NativeForecastInstance]:
        if task.benchmark_id != self.benchmark_id:
            raise ValueError("task does not belong to the GIFT-Eval adapter")
        yield from iter_gift_eval_instances(
            task.source_dataset_id,
            self.source_root,
            term=task.suite_id,
            max_instances=max_instances,
            selected_model_max_contexts=selected_model_max_contexts,
        )

    def source_spec(self) -> BenchmarkSourceSpec:
        return BenchmarkSourceSpec(
            benchmark_id=self.benchmark_id,
            adapter_schema_version=self.adapter_schema_version,
            source_revision=GIFT_EVAL_SOURCE_REVISION,
            source_root=self.source_root,
        )

    def source_artifacts(self, task: BenchmarkTaskSpec) -> tuple[Path, ...]:
        asset = gift_eval_asset_path(task.source_dataset_id, self.source_root)
        return tuple(sorted(asset.glob("data-*.arrow")))
