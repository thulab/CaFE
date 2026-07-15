#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlmodel import Session, select


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.ids import new_id  # noqa: E402
from app.core.time import utc_now  # noqa: E402
from app.db.init_db import init_db  # noqa: E402
from app.db.session import create_db_engine  # noqa: E402
from app.models.dataset import DatasetManifest, Shard  # noqa: E402
from app.models.sample import SampleIndex  # noqa: E402
from app.services.dataset_load_service import SampleWindow  # noqa: E402
from app.services.dataset_reader import DatasetReadResult  # noqa: E402
from app.services.sample_store import SampleStore  # noqa: E402
from app.services.series_store import SeriesStore  # noqa: E402
from app.services.synthetic_generation_service import (  # noqa: E402
    CAPABILITIES_BY_ID,
    MOCK_ANCHOR,
    _frequency_delta,
    _generate_accepted_sample_values,
    _write_generation_manifest,
    _seed_for,
)


DEFAULT_DATABASE_URL = f"sqlite:///{REPO_ROOT / 'backend/runtime/tsbenchmark.db'}"
DEFAULT_RUNTIME_DIR = REPO_ROOT / "backend/runtime"
DEFAULT_CAPABILITIES = (
    "trend",
    "multi_seasonal",
    "regime_switching",
    "time_varying_seasonality",
    "nonlinear_persistence",
    "predictable_intermittency",
    "common_factor",
    "hierarchical_coherence",
    "covariate_response",
)
DEFAULT_INTENSITIES = (1, 2, 3, 4, 5)
DEFAULT_CONTEXT_LENGTH = 168
DEFAULT_HORIZON = 24
DEFAULT_SEASON_LENGTH = 24
DEFAULT_SAMPLE_COUNT = 12
DEFAULT_TARGET_DIM = 3
DEFAULT_SEED = 20260701


@dataclass(frozen=True)
class ImportConfig:
    name: str
    capabilities: tuple[str, ...]
    intensities: tuple[int, ...]
    sample_count: int
    context_length: int
    horizon: int
    season_length: int
    target_dim: int
    seed: int
    frequency: str
    database_url: str
    runtime_dir: Path
    source_summary: Path | None = None
    allow_duplicates: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import synthetic v2 experiment samples as platform test case sets."
    )
    parser.add_argument("--summary", type=Path, help="Read capabilities and window parameters from a synthetic v2 summary.json.")
    parser.add_argument("--name", help="Dataset/test-case-set name prefix.")
    parser.add_argument("--capabilities", nargs="+", help="Synthetic capability ids to import.")
    parser.add_argument("--intensities", nargs="+", type=int, help="Structure intensity levels to import.")
    parser.add_argument("--difficulties", nargs="+", type=int, help="Deprecated alias for --intensities.")
    parser.add_argument("--sample-count", type=int, help="Samples per capability and intensity.")
    parser.add_argument("--context-length", type=int, help="History length.")
    parser.add_argument("--horizon", type=int, help="Forecast length.")
    parser.add_argument("--season-length", type=int, help="Primary seasonal period.")
    parser.add_argument("--target-dim", type=int, default=DEFAULT_TARGET_DIM, help="Target count for multi-target capabilities.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frequency", default="h")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("TSBENCHMARK_DATABASE_URL") or DEFAULT_DATABASE_URL,
        help="Platform database URL. Defaults to backend/runtime/tsbenchmark.db.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path(os.environ.get("TSBENCHMARK_RUNTIME_DIR") or DEFAULT_RUNTIME_DIR),
        help="Platform runtime directory. Defaults to backend/runtime.",
    )
    parser.add_argument("--allow-duplicates", action="store_true", help="Import even if matching shards already exist.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ImportConfig:
    summary = read_summary(args.summary) if args.summary else {}
    capabilities = tuple(args.capabilities or summary.get("requested_capabilities") or DEFAULT_CAPABILITIES)
    sample_count = int(
        args.sample_count
        or summary.get("sample_count_per_capability_intensity")
        or summary.get("sample_count_per_capability_difficulty")
        or DEFAULT_SAMPLE_COUNT
    )
    context_length = int(args.context_length or summary.get("context_length") or DEFAULT_CONTEXT_LENGTH)
    horizon = int(args.horizon or summary.get("horizon") or DEFAULT_HORIZON)
    season_length = int(args.season_length or summary.get("season_length") or DEFAULT_SEASON_LENGTH)
    name = args.name or default_name(args.summary, capabilities)
    return ImportConfig(
        name=name,
        capabilities=capabilities,
        intensities=tuple(args.intensities or args.difficulties or DEFAULT_INTENSITIES),
        sample_count=sample_count,
        context_length=context_length,
        horizon=horizon,
        season_length=season_length,
        target_dim=int(args.target_dim),
        seed=int(args.seed),
        frequency=str(args.frequency),
        database_url=str(args.database_url),
        runtime_dir=Path(args.runtime_dir),
        source_summary=args.summary,
        allow_duplicates=bool(args.allow_duplicates),
    )


def read_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def default_name(summary_path: Path | None, capabilities: tuple[str, ...]) -> str:
    if summary_path is not None:
        return f"Imported {summary_path.parent.name}"
    if len(capabilities) == len(DEFAULT_CAPABILITIES):
        return "Imported synthetic v2 all capabilities"
    return "Imported synthetic v2 " + "-".join(capabilities)


def validate_config(config: ImportConfig) -> None:
    missing = [capability_id for capability_id in config.capabilities if capability_id not in CAPABILITIES_BY_ID]
    if missing:
        raise SystemExit(f"unknown synthetic capabilities: {', '.join(missing)}")
    if not config.intensities:
        raise SystemExit("at least one intensity is required")
    invalid_intensities = [intensity for intensity in config.intensities if intensity < 1 or intensity > 5]
    if invalid_intensities:
        raise SystemExit(f"intensities must be in [1, 5]: {invalid_intensities}")
    positive = {
        "sample_count": config.sample_count,
        "context_length": config.context_length,
        "horizon": config.horizon,
        "season_length": config.season_length,
        "target_dim": config.target_dim,
    }
    invalid = {key: value for key, value in positive.items() if int(value) <= 0}
    if invalid:
        raise SystemExit(f"positive parameters required: {invalid}")


def import_experiment_shards(config: ImportConfig) -> dict[str, Any]:
    validate_config(config)
    config.runtime_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = config.runtime_dir / "synthetic" / "imports"
    storage_dir.mkdir(parents=True, exist_ok=True)

    engine = create_db_engine(config.database_url)
    init_db(engine)
    with Session(engine) as session:
        existing = existing_import_keys(session)
        planned = [
            (capability_id, intensity, import_key(config, capability_id, intensity))
            for capability_id in config.capabilities
            for intensity in config.intensities
        ]
        to_create = [
            (capability_id, intensity, key)
            for capability_id, intensity, key in planned
            if config.allow_duplicates or key not in existing
        ]
        skipped = [
            {"capability_id": capability_id, "intensity": intensity, "difficulty": intensity, "import_key": key}
            for capability_id, intensity, key in planned
            if not config.allow_duplicates and key in existing
        ]
        if not to_create:
            return {
                "created_count": 0,
                "skipped_count": len(skipped),
                "created_shards": [],
                "skipped": skipped,
                "database_url": config.database_url,
                "runtime_dir": str(config.runtime_dir),
            }

        generation_id = new_id()
        source_uri = f"synthetic-v2-import://{generation_id}"
        manifest = DatasetManifest(
            name=config.name,
            domain="synthetic",
            source_type="synthetic",
            source_uri=source_uri,
            file_format="synthetic",
            time_column="time",
            frequency=config.frequency,
            status="loaded",
        )
        session.add(manifest)
        session.flush()

        created: list[dict[str, Any]] = []
        try:
            for capability_id, intensity, key in to_create:
                shard = create_imported_shard(
                    session,
                    config,
                    manifest,
                    source_uri,
                    storage_dir,
                    generation_id,
                    capability_id,
                    intensity,
                    key,
                )
                session.add(shard)
                session.flush()
                first_sample = session.exec(
                    select(SampleIndex).where(SampleIndex.shard_id == shard.shard_id).order_by(SampleIndex.sample_index)
                ).first()
                created.append(
                    {
                        "shard_id": shard.shard_id,
                        "name": shard.name,
                        "capability_id": capability_id,
                        "intensity": intensity,
                        "difficulty": intensity,
                        "sample_count": shard.sample_count,
                        "first_sample_id": first_sample.sample_id if first_sample else None,
                    }
                )
            manifest.updated_at = utc_now()
            session.add(manifest)
            session.commit()
        except Exception:
            session.rollback()
            raise
    return {
        "created_count": len(created),
        "skipped_count": len(skipped),
        "created_shards": created,
        "skipped": skipped,
        "database_url": config.database_url,
        "runtime_dir": str(config.runtime_dir),
    }

def existing_import_keys(session: Session) -> set[str]:
    keys: set[str] = set()
    for shard in session.exec(select(Shard).where(Shard.shard_type == "synthetic")).all():
        value = (shard.generation_config or {}).get("import_key")
        if isinstance(value, str):
            keys.add(value)
    return keys


def import_key(config: ImportConfig, capability_id: str, intensity: int) -> str:
    payload = {
        "schema_version": "synthetic_v2_platform_import.v1",
        "capability_id": capability_id,
        "intensity": intensity,
        "difficulty": intensity,
        "sample_count": config.sample_count,
        "context_length": config.context_length,
        "horizon": config.horizon,
        "season_length": config.season_length,
        "target_dim": experiment_target_dim(capability_id, config.target_dim),
        "seed": config.seed,
        "frequency": config.frequency,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_target_dim(capability_id: str, requested_target_dim: int) -> int:
    capability = CAPABILITIES_BY_ID[capability_id]
    if capability.target_dim_mode == "multi":
        return max(2, int(requested_target_dim))
    return 1


def create_imported_shard(
    session: Session,
    config: ImportConfig,
    manifest: DatasetManifest,
    source_uri: str,
    storage_dir: Path,
    generation_id: str,
    capability_id: str,
    intensity: int,
    key: str,
) -> Shard:
    capability = CAPABILITIES_BY_ID[capability_id]
    context = int(config.context_length)
    horizon = int(config.horizon)
    sample_length = context + horizon
    target_dim = experiment_target_dim(capability_id, config.target_dim)
    target_columns = [f"target_{index}" for index in range(target_dim)]
    covariate_columns = list(capability.covariate_columns)
    columns = [*target_columns, *covariate_columns]
    base_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    delta = _frequency_delta(config.frequency)

    all_timestamps: list[datetime] = []
    all_values: list[list[float]] = []
    windows: list[SampleWindow] = []
    sample_metadata: list[dict[str, Any]] = []
    for sample_index in range(config.sample_count):
        experiment_index = intensity * 10_000 + sample_index
        sample_seed = _seed_for(config.seed, capability_id, sample_index)
        target, latent_params, covariates, realized_features = _generate_accepted_sample_values(
            capability_id,
            sample_length,
            context,
            target_dim,
            config.season_length,
            intensity,
            sample_seed,
        )
        values = np.concatenate([target, covariates], axis=1) if covariates is not None and covariates.size else target
        row_start = sample_index * sample_length
        row_end = row_start + sample_length - 1
        windows.append(
            SampleWindow(
                source_row_start=row_start,
                source_row_end=row_end,
                context_start=row_start,
                context_end=row_start + context - 1,
                horizon_start=row_start + context,
                horizon_end=row_end,
                context_length=context,
                horizon=horizon,
            )
        )
        all_timestamps.extend(base_start + delta * row for row in range(row_start, row_end + 1))
        all_values.extend(values.astype(float).tolist())
        sample_metadata.append(
            {
                "schema_version": "synthetic_v2_import_sample_metadata.v1",
                "experiment_sample_id": f"{capability_id}-i{intensity}-{sample_index:03d}",
                "experiment_sample_index": experiment_index,
                "capability_id": capability_id,
                "capability_label": capability.label,
                "intensity": intensity,
                "difficulty": intensity,
                "seed": config.seed,
                "sample_seed": sample_seed,
                "paired_seed_across_intensity": True,
                "latent_params": latent_params,
                "realized_features": realized_features,
                **MOCK_ANCHOR,
            }
        )

    generation_config = {
        "schema_version": "synthetic_v2_platform_import.v1",
        "generation_id": generation_id,
        "import_key": key,
        "source_summary": str(config.source_summary) if config.source_summary else None,
        "capability_id": capability_id,
        "capability_label": capability.label,
        "task_type": capability.task_type,
        "intensity": intensity,
        "difficulty": intensity,
        "intensity_definition": (
            "capability-global canonical realized strength; not a required monotonic model-error difficulty"
        ),
        "seed": config.seed,
        "context_length": context,
        "horizon": horizon,
        "sample_count": config.sample_count,
        "season_length": config.season_length,
        "target_dim": target_dim,
        "requested_target_dim": config.target_dim,
        "covariate_columns": covariate_columns,
        "frequency": config.frequency,
        **MOCK_ANCHOR,
    }
    shard = Shard(
        name=f"{config.name} - {capability.label} i{intensity}",
        shard_type="synthetic",
        capability_type=capability_id,
        dataset_manifest_id=manifest.dataset_manifest_id,
        source_uri=source_uri,
        storage_uri=str(storage_dir / f"{generation_id}-{capability_id}-i{intensity}.json"),
        time_range_start=all_timestamps[0].isoformat(),
        time_range_end=all_timestamps[-1].isoformat(),
        row_count=len(all_values),
        target_columns=target_columns,
        target_dim=target_dim,
        covariate_columns=covariate_columns,
        covariate_dim=len(covariate_columns),
        frequency=config.frequency,
        context_length=context,
        horizon=horizon,
        stride=horizon,
        sample_count=config.sample_count,
        generation_config=generation_config,
        status="ready",
    )
    session.add(shard)
    session.flush()

    read_result = DatasetReadResult(
        columns=["time", *columns],
        rows=[{"time": timestamp.isoformat()} for timestamp in all_timestamps],
        timestamps=all_timestamps,
        target_columns=target_columns,
        covariate_columns=covariate_columns,
        values=all_values,
        frequency=config.frequency,
        encoding="synthetic_v2_import",
        delimiter=",",
    )
    SeriesStore().write(session, shard.shard_id, all_timestamps, columns, all_values)
    sample_indexes = SampleStore().write_samples(shard.shard_id, windows, target_columns, covariate_columns, read_result)
    for sample_index, metadata in zip(sample_indexes, sample_metadata, strict=True):
        sample_index.sample_metadata = metadata
        session.add(sample_index)
    _write_generation_manifest(Path(shard.storage_uri), generation_config, sample_metadata)
    return shard


def print_summary(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"created shards: {summary['created_count']}")
    print(f"skipped shards: {summary['skipped_count']}")
    for shard in summary["created_shards"]:
        first = shard.get("first_sample_id") or "-"
        print(
            f"- {shard['name']} | shard={shard['shard_id']} | first_sample={first} | samples={shard['sample_count']}"
        )


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    summary = import_experiment_shards(config)
    print_summary(summary, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
