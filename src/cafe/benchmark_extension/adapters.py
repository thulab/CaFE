from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

from cafe.benchmark_extension.native import NativeForecastInstance


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    """Serializable identity and native protocol for one benchmark task."""

    benchmark_id: str
    suite_id: str
    task_id: str
    source_dataset_id: str
    display_name: str
    frequency: str
    horizon: int
    seasonality: int
    window_count: int
    target_column_names: tuple[str, ...] = ()
    past_dynamic_columns: tuple[str, ...] = ()
    known_dynamic_columns: tuple[str, ...] = ()
    static_columns: tuple[str, ...] = ()
    native_config: Mapping[str, Any] = field(default_factory=dict)
    config_sha256: str = ""


@dataclass(frozen=True)
class BenchmarkSourceSpec:
    """Content-addressed source information frozen into generation manifests."""

    benchmark_id: str
    adapter_schema_version: str
    source_revision: str
    source_root: Path
    suite_artifact: Path | None = None
    source_manifest: Path | None = None


@runtime_checkable
class BenchmarkAdapter(Protocol):
    benchmark_id: str
    adapter_schema_version: str

    def list_tasks(self, suite_id: str) -> tuple[BenchmarkTaskSpec, ...]: ...

    def iter_instances(
        self,
        task: BenchmarkTaskSpec,
        *,
        max_instances: int | None = None,
        selected_model_max_contexts: Mapping[str, int] | None = None,
    ) -> Iterator[NativeForecastInstance]: ...

    def source_spec(self) -> BenchmarkSourceSpec: ...

    def source_artifacts(self, task: BenchmarkTaskSpec) -> tuple[Path, ...]: ...


def task_spec_to_dict(task: BenchmarkTaskSpec) -> dict[str, Any]:
    return {
        "benchmark_id": task.benchmark_id,
        "suite_id": task.suite_id,
        "task_id": task.task_id,
        "source_dataset_id": task.source_dataset_id,
        "display_name": task.display_name,
        "frequency": task.frequency,
        "horizon": task.horizon,
        "seasonality": task.seasonality,
        "window_count": task.window_count,
        "target_column_names": list(task.target_column_names),
        "past_dynamic_columns": list(task.past_dynamic_columns),
        "known_dynamic_columns": list(task.known_dynamic_columns),
        "static_columns": list(task.static_columns),
        "native_config": dict(task.native_config),
        "config_sha256": task.config_sha256,
    }


def task_spec_from_dict(value: Mapping[str, Any]) -> BenchmarkTaskSpec:
    return BenchmarkTaskSpec(
        benchmark_id=str(value["benchmark_id"]),
        suite_id=str(value["suite_id"]),
        task_id=str(value["task_id"]),
        source_dataset_id=str(value["source_dataset_id"]),
        display_name=str(value["display_name"]),
        frequency=str(value["frequency"]),
        horizon=int(value["horizon"]),
        seasonality=int(value["seasonality"]),
        window_count=int(value["window_count"]),
        target_column_names=tuple(
            str(item) for item in value.get("target_column_names") or []
        ),
        past_dynamic_columns=tuple(
            str(item) for item in value.get("past_dynamic_columns") or []
        ),
        known_dynamic_columns=tuple(
            str(item) for item in value.get("known_dynamic_columns") or []
        ),
        static_columns=tuple(
            str(item) for item in value.get("static_columns") or []
        ),
        native_config=dict(value.get("native_config") or {}),
        config_sha256=str(value.get("config_sha256") or ""),
    )
