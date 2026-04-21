from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ...config import AppSettings, get_settings
from ...errors import BenchmarkError
from .aggregate import make_report_artifacts
from .anchor import build_anchor_stats_artifacts
from .domain import (
    BenchmarkV1ArtifactSummary,
    BuildAnchorStatsRequest,
    BuildBenchmarkV1Request,
    MakeBenchmarkV1ReportRequest,
    RunBenchmarkV1EvalRequest,
)
from .generate import build_benchmark_artifacts
from .runner import run_model_eval
from .utils import adjacent_meta_path, read_json, write_json


class BenchmarkV1Manager:
    def __init__(self, runtime_root: Path, settings: AppSettings | None = None) -> None:
        self.runtime_root = runtime_root
        self.settings = settings or get_settings()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    @property
    def artifact_root(self) -> Path:
        configured = getattr(self.settings.benchmark.v1, "artifact_root", "benchmark_v1")
        path = Path(configured)
        if path.is_absolute():
            return path
        return self.runtime_root / "generated" / path

    def default_anchor_stats_path(self) -> Path:
        return self.artifact_root / "anchor_stats.parquet"

    def default_benchmark_path(self) -> Path:
        return self.artifact_root / "benchmark_v1.parquet"

    def default_eval_dir(self) -> Path:
        return self.artifact_root / "eval"

    def default_report_dir(self) -> Path:
        return self.artifact_root / "reports"

    def build_anchor_stats(self, request: BuildAnchorStatsRequest) -> BenchmarkV1ArtifactSummary:
        output = self._safe_parquet_path(request.output_name, default=self.default_anchor_stats_path())
        path = build_anchor_stats_artifacts(
            output_path=output,
            gift_root=Path(request.gift_root) if request.gift_root else None,
            tfb_root=Path(request.tfb_root) if request.tfb_root else None,
            n_clusters=request.n_clusters,
            bootstrap_size=request.bootstrap_size,
            seed=request.seed,
        )
        return self._summarize_path("anchor_stats", path)

    def build_benchmark(self, request: BuildBenchmarkV1Request) -> BenchmarkV1ArtifactSummary:
        anchor_stats_path = Path(request.anchor_stats_path) if request.anchor_stats_path else self.default_anchor_stats_path()
        if not anchor_stats_path.exists():
            raise BenchmarkError(f"anchor stats not found: {anchor_stats_path}")
        output = self._safe_parquet_path(request.output_name, default=self.default_benchmark_path())
        path = build_benchmark_artifacts(
            anchor_stats_path=anchor_stats_path,
            output_path=output,
            anchor_track_size=request.anchor_track_size,
            diagnostic_per_cell=request.diagnostic_per_cell,
            seed=request.seed,
            version=request.version,
        )
        return self._summarize_path("benchmark", path)

    def run_eval(self, request: RunBenchmarkV1EvalRequest) -> BenchmarkV1ArtifactSummary:
        benchmark_path = Path(request.benchmark_path) if request.benchmark_path else self.default_benchmark_path()
        if not benchmark_path.exists():
            raise BenchmarkError(f"benchmark not found: {benchmark_path}")
        output_dir = Path(request.output_dir) if request.output_dir else self.default_eval_dir()
        path = run_model_eval(
            model_name=request.model,
            benchmark_path=benchmark_path,
            output_dir=output_dir,
            seeds=request.seeds,
        )
        return self._summarize_path("eval", path)

    def make_report(self, request: MakeBenchmarkV1ReportRequest) -> BenchmarkV1ArtifactSummary:
        benchmark_path = Path(request.benchmark_path) if request.benchmark_path else self.default_benchmark_path()
        eval_dir = Path(request.eval_dir) if request.eval_dir else self.default_eval_dir()
        output_dir = Path(request.output_dir) if request.output_dir else self.default_report_dir()
        real_eval_path = Path(request.real_eval_path) if request.real_eval_path else None
        if not benchmark_path.exists():
            raise BenchmarkError(f"benchmark not found: {benchmark_path}")
        if not eval_dir.exists():
            raise BenchmarkError(f"eval dir not found: {eval_dir}")
        path = make_report_artifacts(
            benchmark_path=benchmark_path,
            eval_dir=eval_dir,
            output_dir=output_dir,
            real_eval_path=real_eval_path,
        )
        return self._summarize_path("report", path)

    def list_artifacts(self) -> list[BenchmarkV1ArtifactSummary]:
        summaries: list[BenchmarkV1ArtifactSummary] = []
        for path in sorted(self.artifact_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in {".parquet", ".json", ".md"}:
                continue
            if path.name.endswith(".meta.json"):
                continue
            kind = self._infer_kind(path)
            if kind:
                summaries.append(self._summarize_path(kind, path))
        return summaries

    def _safe_parquet_path(self, name: str, *, default: Path) -> Path:
        value = (name or "").strip()
        if not value:
            return default
        path = Path(value)
        if path.suffix != ".parquet":
            path = path.with_suffix(".parquet")
        if path.is_absolute():
            return path
        return self.artifact_root / path

    def _infer_kind(self, path: Path) -> str | None:
        if path.name == "anchor_stats.parquet" or "anchor_stats" in path.name:
            return "anchor_stats"
        if path.name == "benchmark_v1.parquet" or "benchmark" in path.name:
            return "benchmark"
        if path.parent.name == "eval":
            return "eval"
        if path.parent.name == "reports" or path.name.startswith("summary"):
            return "report"
        return None

    def _summarize_path(self, kind: str, path: Path) -> BenchmarkV1ArtifactSummary:
        path = Path(path)
        meta_path = adjacent_meta_path(path) if path.is_file() else path / "summary.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        validation = meta.get("validation_summary") or meta.get("validation") or {}
        n_series = None
        if path.suffix == ".parquet":
            try:
                n_series = int(len(pd.read_parquet(path)))
            except Exception:
                n_series = None
        return BenchmarkV1ArtifactSummary(
            kind=kind,
            path=str(path),
            created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None,
            benchmark_version=meta.get("benchmark_version"),
            anchor_mode=meta.get("anchor_mode") or validation.get("anchor_mode"),
            n_series=n_series or validation.get("n_series"),
            validation_summary=validation,
        )

    def write_artifact_note(self, name: str, payload: dict[str, object]) -> Path:
        path = self.artifact_root / f"{name}.json"
        write_json(payload, path)
        return path
