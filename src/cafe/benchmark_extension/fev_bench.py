from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from cafe import core as protocol
from cafe.benchmark_extension.adapters import (
    BenchmarkSourceSpec,
    BenchmarkTaskSpec,
)
from cafe.benchmark_extension.native import (
    NativeForecastInstance,
    fill_unobserved_future,
    impute_dynamic_covariates,
    impute_history_only,
)


FEV_ADAPTER_SCHEMA = "cafe.fev_native_adapter.v3"
FEV_BENCHMARK_ID = "fev_bench"
FEV_MINI_SUITE_ID = "mini20"
FEV_PACKAGE_VERSION = "0.8.0"


def _import_fev() -> Any:
    try:
        import fev
    except ImportError as error:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "FEV support requires `uv sync --extra fev`"
        ) from error
    version = str(getattr(fev, "__version__", ""))
    if version != FEV_PACKAGE_VERSION:
        raise RuntimeError(
            f"FEV adapter requires fev=={FEV_PACKAGE_VERSION}, found {version}"
        )
    return fev


def _public_task_config(task: Any) -> dict[str, Any]:
    names = (
        "dataset_path",
        "dataset_config",
        "horizon",
        "num_windows",
        "initial_cutoff",
        "window_step_size",
        "min_context_length",
        "max_context_length",
        "seasonality",
        "eval_metric",
        "extra_metrics",
        "quantile_levels",
        "id_column",
        "timestamp_column",
        "target",
        "generate_univariate_targets_from",
        "known_dynamic_columns",
        "past_dynamic_columns",
        "static_columns",
        "task_name",
    )
    return {name: getattr(task, name) for name in names}


def _as_json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _numeric_sequence(row: Mapping[str, Any], column: str) -> np.ndarray:
    raw = np.asarray(row[column])
    if not (
        np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.bool_)
    ):
        raise ValueError(
            f"FEV dynamic column {column!r} is non-numeric ({raw.dtype})"
        )
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"FEV sequence column {column!r} is not one-dimensional")
    return values


def _missing_scalar(value: Any) -> bool:
    return value is None or (
        isinstance(value, (float, np.floating)) and not np.isfinite(value)
    )


def _encoded_dynamic_pair(
    past_row: Mapping[str, Any],
    future_row: Mapping[str, Any] | None,
    column: str,
) -> tuple[np.ndarray, np.ndarray | None, str, dict[str, Any]]:
    raw_history = np.asarray(past_row[column])
    raw_future = (
        None if future_row is None else np.asarray(future_row[column])
    )
    if raw_history.ndim != 1 or (
        raw_future is not None and raw_future.ndim != 1
    ):
        raise ValueError(f"FEV sequence column {column!r} is not one-dimensional")
    if np.issubdtype(raw_history.dtype, np.number) or np.issubdtype(
        raw_history.dtype, np.bool_
    ):
        history = np.asarray(raw_history, dtype=float)
        future = (
            None if raw_future is None else np.asarray(raw_future, dtype=float)
        )
        return history, future, _covariate_type(history, future), {
            "encoding": "identity_numeric"
        }

    combined = [*raw_history.tolist()]
    if raw_future is not None:
        combined.extend(raw_future.tolist())
    categories = sorted(
        {str(value) for value in combined if not _missing_scalar(value)}
    )
    mapping = {value: index for index, value in enumerate(categories)}

    def encode(raw: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                float("nan")
                if _missing_scalar(value)
                else float(mapping[str(value)])
                for value in raw.tolist()
            ],
            dtype=float,
        )

    return (
        encode(raw_history),
        None if raw_future is None else encode(raw_future),
        "categorical",
        {"encoding": "sorted_string_category_codes_v1", "categories": categories},
    )


def _time_major(
    row: Mapping[str, Any],
    columns: tuple[str, ...],
    *,
    length: int | None = None,
) -> np.ndarray:
    if not columns:
        if length is None:
            raise ValueError("empty FEV column selection requires an explicit length")
        return np.empty((int(length), 0), dtype=float)
    values = [_numeric_sequence(row, column) for column in columns]
    lengths = {int(value.size) for value in values}
    if len(lengths) != 1:
        raise ValueError("FEV sequence columns have inconsistent lengths")
    return np.column_stack(values)


def _covariate_type(history: np.ndarray, future: np.ndarray | None) -> str:
    values = history if future is None else np.concatenate((history, future))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "continuous_numeric"
    unique = np.unique(finite)
    if unique.size <= 2 and np.all(np.isin(unique, (0.0, 1.0))):
        return "binary"
    return "continuous_numeric"


class FevBenchAdapter:
    """Official FEV task/window adapter with no CaFE-side re-splitting."""

    benchmark_id = FEV_BENCHMARK_ID
    adapter_schema_version = FEV_ADAPTER_SCHEMA

    def __init__(
        self,
        suite_path: Path,
        *,
        source_root: Path | None = None,
        source_revision: str = "unpinned",
        num_proc: int = 1,
    ) -> None:
        self.suite_path = suite_path.resolve()
        if not self.suite_path.is_file():
            raise FileNotFoundError(self.suite_path)
        self.source_root = (
            self.suite_path.parent
            if source_root is None
            else source_root.resolve()
        )
        self.source_revision = str(source_revision)
        self.num_proc = max(1, int(num_proc))
        self._benchmark: Any | None = None
        self._tasks_by_id: dict[str, Any] = {}
        self._specs_by_id: dict[str, BenchmarkTaskSpec] = {}
        manifest_path = self.source_root / "source_manifest.json"
        self._source_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )

    def _load_benchmark(self) -> Any:
        if self._benchmark is None:
            fev = _import_fev()
            self._benchmark = fev.Benchmark.from_yaml(self.suite_path)
        return self._benchmark

    def _prepare_task_source(self, task: Any) -> Any:
        dataset_config = task.dataset_config
        files = self._source_manifest.get("dataset_files") or {}
        if dataset_config is None or dataset_config not in files:
            return task
        record = files[str(dataset_config)]
        path = (self.source_root / str(record["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha256 = str(record.get("sha256") or "")
        if expected_sha256 and protocol.file_sha256(path) != expected_sha256:
            raise ValueError(f"FEV source snapshot hash mismatch: {path}")
        task.dataset_path = str(path)
        task.dataset_config = None
        return task

    def _task_entries(self) -> list[tuple[Any, str, dict[str, Any], str]]:
        tasks = list(self._load_benchmark().tasks)
        configs = [_public_task_config(task) for task in tasks]
        hashes = [protocol.json_sha256(config) for config in configs]
        base_ids = [protocol.safe_id(str(task.task_name)) for task in tasks]
        duplicate_ids = {
            value for value, count in Counter(base_ids).items() if count > 1
        }
        return [
            (
                task,
                (
                    f"fev__{base_id}__{config_sha256[:8]}"
                    if base_id in duplicate_ids
                    else f"fev__{base_id}"
                ),
                config,
                config_sha256,
            )
            for task, base_id, config, config_sha256 in zip(
                tasks, base_ids, configs, hashes, strict=True
            )
        ]

    def available_task_ids(
        self, suite_id: str = FEV_MINI_SUITE_ID
    ) -> tuple[str, ...]:
        if suite_id != FEV_MINI_SUITE_ID:
            raise ValueError(f"unsupported FEV suite {suite_id!r}")
        return tuple(entry[1] for entry in self._task_entries())

    def list_tasks(
        self,
        suite_id: str = FEV_MINI_SUITE_ID,
        *,
        selected_task_ids: tuple[str, ...] | None = None,
    ) -> tuple[BenchmarkTaskSpec, ...]:
        if suite_id != FEV_MINI_SUITE_ID:
            raise ValueError(f"unsupported FEV suite {suite_id!r}")
        entries = self._task_entries()
        requested = (
            None
            if selected_task_ids is None
            else set(str(value) for value in selected_task_ids)
        )
        available = {entry[1] for entry in entries}
        if requested is not None and not requested <= available:
            raise ValueError(
                "unknown FEV task ids: " + ", ".join(sorted(requested - available))
            )
        specs: list[BenchmarkTaskSpec] = []
        for task, task_id, config, config_sha256 in entries:
            if requested is not None and task_id not in requested:
                continue
            if task_id in self._specs_by_id:
                specs.append(self._specs_by_id[task_id])
                continue
            self._prepare_task_source(task)
            task.load_full_dataset(num_proc=self.num_proc)
            target_columns = tuple(str(value) for value in task.target_columns)
            spec = BenchmarkTaskSpec(
                benchmark_id=self.benchmark_id,
                suite_id=suite_id,
                task_id=task_id,
                source_dataset_id=str(task.dataset_config or task.task_name),
                display_name=str(task.task_name),
                frequency=str(task.freq),
                horizon=int(task.horizon),
                seasonality=int(task.seasonality),
                window_count=int(task.num_windows),
                target_column_names=target_columns,
                past_dynamic_columns=tuple(
                    str(value) for value in task.past_dynamic_columns
                ),
                known_dynamic_columns=tuple(
                    str(value) for value in task.known_dynamic_columns
                ),
                static_columns=tuple(str(value) for value in task.static_columns),
                native_config={
                    **config,
                    "fev_version": FEV_PACKAGE_VERSION,
                    "dataset_fingerprint": str(task._dataset_fingerprint),
                },
                config_sha256=config_sha256,
            )
            self._tasks_by_id[task_id] = task
            self._specs_by_id[task_id] = spec
            specs.append(spec)
        return tuple(specs)

    def _task(self, task_spec: BenchmarkTaskSpec) -> Any:
        if task_spec.benchmark_id != self.benchmark_id:
            raise ValueError("task does not belong to the FEV adapter")
        if task_spec.task_id not in self._tasks_by_id:
            config = {
                key: value
                for key, value in task_spec.native_config.items()
                if key not in {"fev_version", "dataset_fingerprint"}
            }
            fev = _import_fev()
            self._tasks_by_id[task_spec.task_id] = self._prepare_task_source(
                fev.Task(**config)
            )
        try:
            return self._tasks_by_id[task_spec.task_id]
        except KeyError as error:
            raise ValueError(f"unknown FEV task {task_spec.task_id!r}") from error

    def iter_instances(
        self,
        task: BenchmarkTaskSpec,
        *,
        max_instances: int | None = None,
        selected_model_max_contexts: Mapping[str, int] | None = None,
    ) -> Iterator[NativeForecastInstance]:
        fev_task = self._task(task)
        emitted = 0
        full_dataset = fev_task.load_full_dataset(num_proc=self.num_proc)
        full_lengths = {
            str(row[fev_task.id_column]): len(row[fev_task.timestamp_column])
            for row in full_dataset
        }
        past_columns = tuple(str(value) for value in fev_task.past_dynamic_columns)
        known_columns = tuple(str(value) for value in fev_task.known_dynamic_columns)
        covariate_columns = past_columns + known_columns
        covariate_availability = tuple(
            ["past_only"] * len(past_columns)
            + ["known_future"] * len(known_columns)
        )
        future_visible = tuple(
            [False] * len(past_columns) + [True] * len(known_columns)
        )
        target_columns = tuple(str(value) for value in fev_task.target_columns)
        for window_index, window in enumerate(
            fev_task.iter_windows(num_proc=self.num_proc)
        ):
            past_data, future_known = window.get_input_data()
            ground_truth = window.get_ground_truth()
            if not (len(past_data) == len(future_known) == len(ground_truth)):
                raise ValueError("FEV past/future/ground-truth row counts differ")
            for past_row, known_row, truth_row in zip(
                past_data, future_known, ground_truth, strict=True
            ):
                series_id = str(past_row[fev_task.id_column])
                if str(known_row[fev_task.id_column]) != series_id:
                    raise ValueError("FEV future-known series order mismatch")
                if str(truth_row[fev_task.id_column]) != series_id:
                    raise ValueError("FEV ground-truth series order mismatch")
                raw_history = _time_major(past_row, target_columns)
                raw_future = _time_major(truth_row, target_columns)
                if raw_future.shape != (int(fev_task.horizon), len(target_columns)):
                    raise ValueError("FEV future target shape differs from task horizon")
                history, history_observed, target_imputation = impute_history_only(
                    raw_history
                )
                future, future_observed = fill_unobserved_future(
                    raw_future, history
                )

                history_columns: list[np.ndarray] = []
                future_columns: list[np.ndarray] = []
                covariate_types_list: list[str] = []
                covariate_encodings: dict[str, Any] = {}
                for column in past_columns:
                    encoded_history, _unused, semantic_type, encoding = (
                        _encoded_dynamic_pair(past_row, None, column)
                    )
                    history_columns.append(encoded_history)
                    future_columns.append(
                        np.repeat(encoded_history[-1], int(fev_task.horizon))
                    )
                    covariate_types_list.append(semantic_type)
                    covariate_encodings[column] = encoding
                for column in known_columns:
                    encoded_history, encoded_future, semantic_type, encoding = (
                        _encoded_dynamic_pair(past_row, known_row, column)
                    )
                    assert encoded_future is not None
                    history_columns.append(encoded_history)
                    future_columns.append(encoded_future)
                    covariate_types_list.append(semantic_type)
                    covariate_encodings[column] = encoding
                history_covariates = (
                    np.column_stack(history_columns)
                    if history_columns
                    else np.empty((history.shape[0], 0), dtype=float)
                )
                future_covariates = (
                    np.column_stack(future_columns)
                    if future_columns
                    else np.empty((future.shape[0], 0), dtype=float)
                )
                covariate_types = tuple(covariate_types_list)
                (
                    history_covariates,
                    future_covariates,
                    covariate_imputation,
                ) = impute_dynamic_covariates(
                    history_covariates, future_covariates
                )
                static_covariates = {
                    str(column): _as_json_scalar(past_row[column])
                    for column in fev_task.static_columns
                }
                source_length = int(full_lengths[series_id])
                cutoff = window.cutoff
                origin: int | str
                if isinstance(cutoff, int):
                    origin = cutoff if cutoff >= 0 else source_length + cutoff
                else:
                    origin = str(cutoff)
                token = protocol.safe_id(series_id) or "series"
                origin_token = protocol.safe_id(str(origin)) or "origin"
                instance_id = (
                    f"fev__{protocol.safe_id(task.task_id)}__{token}__"
                    f"w{window_index:02d}__o{origin_token}"
                )
                yield NativeForecastInstance(
                    dataset_id=task.source_dataset_id,
                    config_id=str(fev_task.dataset_config or fev_task.task_name),
                    item_id=series_id,
                    official_instance_id=instance_id,
                    frequency=str(fev_task.freq),
                    term="native",
                    window_index=int(window_index),
                    window_count=int(fev_task.num_windows),
                    forecast_origin=origin,
                    prediction_length=int(fev_task.horizon),
                    history=history,
                    future=future,
                    future_observed_mask=future_observed,
                    history_covariates=history_covariates,
                    future_covariates=future_covariates,
                    covariate_column_names=covariate_columns,
                    covariate_availability=covariate_availability,
                    future_covariate_visible=future_visible,
                    target_column_names=target_columns,
                    source_target_length=source_length,
                    history_imputation={
                        **target_imputation,
                        "benchmark_policy": "fev_native_missing_value_mask_v1",
                        "covariates": covariate_imputation,
                    },
                    benchmark_id=self.benchmark_id,
                    suite_id=task.suite_id,
                    task_id=task.task_id,
                    seasonality=int(fev_task.seasonality),
                    history_observed_mask=history_observed,
                    covariate_types=covariate_types,
                    static_covariates=static_covariates,
                    source_locator={
                        "task_id": task.task_id,
                        "series_id": series_id,
                        "window_index": int(window_index),
                        "cutoff": _as_json_scalar(cutoff),
                    },
                    native_protocol={
                        "split": "fev_task_iter_windows",
                        "task_config_sha256": task.config_sha256,
                        "min_context_length": int(fev_task.min_context_length),
                        "max_context_length": fev_task.max_context_length,
                        "window_step_size": fev_task.window_step_size,
                        "eval_metric": fev_task.eval_metric,
                        "extra_metrics": list(fev_task.extra_metrics),
                        "quantile_levels": list(fev_task.quantile_levels),
                        "covariate_encodings": covariate_encodings,
                    },
                    selected_model_max_contexts=dict(
                        selected_model_max_contexts or {}
                    ),
                )
                emitted += 1
                if max_instances is not None and emitted >= int(max_instances):
                    return

    def source_spec(self) -> BenchmarkSourceSpec:
        manifest = self.source_root / "source_manifest.json"
        return BenchmarkSourceSpec(
            benchmark_id=self.benchmark_id,
            adapter_schema_version=self.adapter_schema_version,
            source_revision=self.source_revision,
            source_root=self.source_root,
            suite_artifact=self.suite_path,
            source_manifest=manifest if manifest.is_file() else None,
        )

    def source_artifacts(self, task: BenchmarkTaskSpec) -> tuple[Path, ...]:
        fev_task = self._task(task)
        dataset_path = Path(str(fev_task.dataset_path))
        artifacts = [self.suite_path]
        if fev_task.dataset_config is None and dataset_path.is_file():
            artifacts.append(dataset_path.resolve())
        manifest = self.source_root / "source_manifest.json"
        if manifest.is_file():
            artifacts.append(manifest.resolve())
        return tuple(dict.fromkeys(artifacts))


def audit_fev_suite(
    adapter: FevBenchAdapter,
    *,
    suite_id: str = FEV_MINI_SUITE_ID,
) -> dict[str, Any]:
    """Enumerate adapter output and return a frozen pre-generation audit."""

    rows: list[dict[str, Any]] = []
    total_instances = 0
    total_target_cells = 0
    for task in adapter.list_tasks(suite_id):
        instance_count = 0
        target_cells = 0
        target_dims: set[int] = set()
        context_lengths: list[int] = []
        observed_covariate_types: set[str] = set()
        for instance in adapter.iter_instances(task):
            instance_count += 1
            target_cells += instance.horizon * instance.target_dim
            target_dims.add(instance.target_dim)
            context_lengths.append(instance.context_length)
            observed_covariate_types.update(instance.covariate_types)
        rows.append(
            {
                "task_id": task.task_id,
                "source_dataset_id": task.source_dataset_id,
                "instance_count": instance_count,
                "forecast_target_cell_count": target_cells,
                "target_dimensions": sorted(target_dims),
                "minimum_context_length": min(context_lengths),
                "maximum_context_length": max(context_lengths),
                "horizon": task.horizon,
                "window_count": task.window_count,
                "covariate_types": sorted(observed_covariate_types),
            }
        )
        total_instances += instance_count
        total_target_cells += target_cells
    return {
        "schema_version": "cafe.fev_suite_audit.v1",
        "benchmark_id": adapter.benchmark_id,
        "suite_id": suite_id,
        "task_count": len(rows),
        "instance_count": total_instances,
        "forecast_target_cell_count": total_target_cells,
        "tasks": rows,
    }
