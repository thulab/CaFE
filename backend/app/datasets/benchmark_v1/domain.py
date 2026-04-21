from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .constants import DIAGNOSTIC_FAMILIES, DIFFICULTY_LEVELS, HORIZON_RATIOS


class BuildAnchorStatsRequest(BaseModel):
    output_name: str = "anchor_stats"
    gift_root: str | None = None
    tfb_root: str | None = None
    n_clusters: int = Field(default=12, ge=2)
    bootstrap_size: int = Field(default=256, ge=32)
    seed: int = 20260407


class BuildBenchmarkV1Request(BaseModel):
    anchor_stats_path: str | None = None
    output_name: str = "benchmark_v1"
    anchor_track_size: int = Field(default=2000, ge=1)
    diagnostic_per_cell: int = Field(default=100, ge=1)
    seed: int = 20260407
    version: str | None = None


class RunBenchmarkV1EvalRequest(BaseModel):
    model: str
    benchmark_path: str | None = None
    output_dir: str | None = None
    seeds: list[int] = Field(default_factory=lambda: [0])


class MakeBenchmarkV1ReportRequest(BaseModel):
    benchmark_path: str | None = None
    eval_dir: str | None = None
    output_dir: str | None = None
    real_eval_path: str | None = None


class BenchmarkV1ArtifactSummary(BaseModel):
    kind: str
    path: str
    created_at: str | None = None
    benchmark_version: str | None = None
    anchor_mode: str | None = None
    n_series: int | None = None
    validation_summary: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkV1Config:
    artifact_root: Path
    anchor_track_size: int = 2000
    diagnostic_per_cell: int = 100
    n_anchor_clusters: int = 12
    anchor_tolerance_sigma: float = 1.5
    diagnostic_tolerance_sigma: float = 2.5
    max_context: int = 512
    min_horizon: int = 12
    max_horizon: int = 96
    min_burn_in: int = 200
    bootstrap_corpus_size: int = 256
    calibration_candidates_per_family: int = 350
    random_seed: int = 20260407
    diagnostic_families: list[str] = None  # type: ignore[assignment]
    difficulty_levels: list[int] = None  # type: ignore[assignment]
    horizon_ratios: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.diagnostic_families is None:
            self.diagnostic_families = list(DIAGNOSTIC_FAMILIES)
        if self.difficulty_levels is None:
            self.difficulty_levels = list(DIFFICULTY_LEVELS)
        if self.horizon_ratios is None:
            self.horizon_ratios = list(HORIZON_RATIOS)


@dataclass(slots=True)
class SeriesSpec:
    track: str
    family: str
    difficulty: int
    horizon_ratio: float
    seed: int
    anchor_cluster_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SeriesMeta:
    latent_params: dict[str, Any]
    realized_features: dict[str, float]
    season_length: int
    dominant_scale: int
    baseline_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SeriesSample:
    id: str
    context: list[float]
    target: list[float]
    horizon: int
    spec: SeriesSpec
    meta: SeriesMeta

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "context": list(self.context),
            "target": list(self.target),
            "horizon": self.horizon,
            "track": self.spec.track,
            "family": self.spec.family,
            "difficulty": self.spec.difficulty,
            "horizon_ratio": self.spec.horizon_ratio,
            "seed": self.spec.seed,
            "anchor_cluster_id": self.spec.anchor_cluster_id,
            "latent_params": self.meta.latent_params,
            "realized_features": self.meta.realized_features,
            "season_length": self.meta.season_length,
            "dominant_scale": self.meta.dominant_scale,
            "baseline_type": self.meta.baseline_type,
        }


@dataclass(slots=True)
class EvalResult:
    model: str
    series_id: str
    seed: int
    benchmark_version: str
    runtime_ms: float
    mase: float
    smape: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

