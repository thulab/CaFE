#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SCRIPT_DIR = REPO_ROOT / "scripts"
for import_path in (BACKEND_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.services.synthetic_generation_service import _realized_features  # noqa: E402
from paper_v2_transfer_common import impute_observed_window  # noqa: E402
from run_synthetic_v2_near_distance_calibration import (  # noqa: E402
    is_informative_target,
    standardize_target,
)
from synthetic_feature_profile import (  # noqa: E402
    WindowSpec,
    gift_eval_short_term_test_holdout_steps,
    limit_candidates,
    read_gift_arrow_targets,
    read_tsf_series_records,
    window_starts,
)


SCHEMA_VERSION = "paper_v4_multi_lookback_profile_suite.v1"
CONTEXT_LENGTHS = (96, 168, 336, 504)
MAX_CONTEXT_LENGTH = max(CONTEXT_LENGTHS)
HORIZON = 48
SEASON_LENGTH = 24
DEFAULT_MAX_WINDOWS_PER_SOURCE = 240
DEFAULT_GIFT_EVAL_DIR = Path.home() / "xmy/gift-eval"
DEFAULT_DATA_DIR = REPO_ROOT / "runtime/research"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp/v4/00_profile_suite"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-18-paper-v4-multi-lookback-profile-protocol.md"
)
QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    dataset_name: str
    family_id: str
    domain: str
    kind: str
    asset_name: str
    frequency: str = "h"


SOURCE_SPECS = (
    SourceSpec(
        "m4_hourly",
        "M4 Hourly",
        "m4_hourly",
        "Econ/Fin",
        "tsf_univariate",
        "m4_hourly_dataset.zip",
    ),
    SourceSpec(
        "gift_electricity_h",
        "Electricity/H",
        "electricity",
        "Energy",
        "gift_univariate",
        "electricity/H",
    ),
    SourceSpec(
        "gift_solar_h",
        "Solar/H",
        "solar",
        "Energy",
        "gift_univariate",
        "solar/H",
    ),
    SourceSpec(
        "gift_ett1_h",
        "ETT1/H",
        "ETT",
        "Energy",
        "gift_univariate",
        "ett1/H",
    ),
    SourceSpec(
        "gift_ett2_h",
        "ETT2/H",
        "ETT",
        "Energy",
        "gift_univariate",
        "ett2/H",
    ),
    SourceSpec(
        "gift_jena_weather_h",
        "Jena Weather/H",
        "jena_weather",
        "Nature",
        "gift_univariate",
        "jena_weather/H",
    ),
    SourceSpec(
        "gift_kdd_cup_h",
        "KDD Cup 2018/H",
        "kdd_cup_2018",
        "Nature",
        "gift_univariate",
        "kdd_cup_2018_with_missing/H",
    ),
    SourceSpec(
        "gift_loop_seattle_h",
        "Loop Seattle/H",
        "LOOP_SEATTLE",
        "Transport",
        "gift_univariate",
        "LOOP_SEATTLE/H",
    ),
    SourceSpec(
        "gift_sz_taxi_h",
        "SZ-Taxi/H",
        "SZ_TAXI",
        "Transport",
        "gift_univariate",
        "SZ_TAXI/H",
    ),
    SourceSpec(
        "gift_m_dense_h",
        "M_DENSE/H",
        "M_DENSE",
        "Transport",
        "gift_univariate",
        "M_DENSE/H",
    ),
    SourceSpec(
        "gift_bitbrains_fast_h",
        "Bitbrains Fast Storage/H",
        "bitbrains",
        "Web/CloudOps",
        "gift_univariate",
        "bitbrains_fast_storage/H",
    ),
    SourceSpec(
        "gift_bitbrains_rnd_h",
        "Bitbrains RND/H",
        "bitbrains",
        "Web/CloudOps",
        "gift_univariate",
        "bitbrains_rnd/H",
    ),
    SourceSpec(
        "gift_bizitobs_l2c_h",
        "BizITObs L2C/H",
        "bizitobs_l2c",
        "Web/CloudOps",
        "gift_univariate",
        "bizitobs_l2c/H",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe, paired real-data profiles for H=48 and "
            "L in {96, 168, 336, 504}."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gift-eval-dir", type=Path, default=DEFAULT_GIFT_EVAL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--max-windows-per-source",
        type=int,
        default=DEFAULT_MAX_WINDOWS_PER_SOURCE,
    )
    parser.add_argument("--sources", nargs="*", default=None)
    return parser.parse_args()


def selected_source_specs(source_ids: Iterable[str] | None) -> tuple[SourceSpec, ...]:
    if source_ids is None:
        return SOURCE_SPECS
    requested = tuple(str(source_id) for source_id in source_ids)
    by_id = {source.source_id: source for source in SOURCE_SPECS}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError("unknown profile sources: " + ", ".join(unknown))
    return tuple(by_id[source_id] for source_id in requested)


def source_asset_path(
    source: SourceSpec,
    *,
    gift_eval_dir: Path,
    data_dir: Path,
) -> Path:
    if source.kind == "gift_univariate":
        return gift_eval_dir / source.asset_name
    if source.kind == "tsf_univariate":
        return data_dir / source.asset_name
    raise ValueError(f"unsupported source kind: {source.kind}")


def nested_view(master_window: np.ndarray, context_length: int) -> np.ndarray:
    master = np.asarray(master_window, dtype=float).reshape(-1)
    expected = MAX_CONTEXT_LENGTH + HORIZON
    if master.size != expected:
        raise ValueError(f"master window must contain {expected} points")
    if context_length not in CONTEXT_LENGTHS:
        raise ValueError(f"unsupported context length: {context_length}")
    start = MAX_CONTEXT_LENGTH - int(context_length)
    return np.asarray(master[start:], dtype=float)


def weighted_quantile(
    values: Iterable[float],
    weights: Iterable[float],
    levels: Iterable[float] = QUANTILE_LEVELS,
) -> list[float]:
    value_array = np.asarray(tuple(values), dtype=float)
    weight_array = np.asarray(tuple(weights), dtype=float)
    level_array = np.asarray(tuple(levels), dtype=float)
    finite = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    value_array = value_array[finite]
    weight_array = weight_array[finite]
    if not value_array.size:
        return []
    order = np.argsort(value_array, kind="stable")
    value_array = value_array[order]
    weight_array = weight_array[order]
    centers = (np.cumsum(weight_array) - 0.5 * weight_array) / np.sum(weight_array)
    return [
        round_float(value)
        for value in np.interp(
            level_array,
            centers,
            value_array,
            left=value_array[0],
            right=value_array[-1],
        )
    ]


def source_balanced_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    """Give every family equal mass, then every config inside a family equal mass."""

    families = sorted({str(row["family_id"]) for row in rows})
    sources_by_family = {
        family_id: sorted(
            {
                str(row["source_id"])
                for row in rows
                if str(row["family_id"]) == family_id
            }
        )
        for family_id in families
    }
    counts_by_source = {
        source_id: sum(str(row["source_id"]) == source_id for row in rows)
        for source_ids in sources_by_family.values()
        for source_id in source_ids
    }
    weights = []
    for row in rows:
        family_id = str(row["family_id"])
        source_id = str(row["source_id"])
        weights.append(
            1.0
            / len(families)
            / len(sources_by_family[family_id])
            / counts_by_source[source_id]
        )
    return np.asarray(weights, dtype=float)


def select_series_balanced_candidates(
    candidates: list[tuple[str, str, int, np.ndarray, int, int]],
    max_items: int,
) -> list[tuple[str, str, int, np.ndarray, int, int]]:
    """Spread a capped sample across series/channels before adding time origins."""

    if max_items <= 0 or not candidates:
        return []
    by_series: dict[str, list[tuple[str, str, int, np.ndarray, int, int]]] = {}
    for candidate in candidates:
        by_series.setdefault(str(candidate[0]), []).append(candidate)
    series_ids = sorted(by_series)
    if len(series_ids) > max_items:
        series_ids = limit_candidates(series_ids, max_items)
    rounds_per_series = max(1, int(np.ceil(max_items / len(series_ids))))
    while True:
        capacity = sum(
            min(len(by_series[series_id]), rounds_per_series)
            for series_id in series_ids
        )
        if capacity >= max_items or all(
            len(by_series[series_id]) <= rounds_per_series
            for series_id in series_ids
        ):
            break
        rounds_per_series += max(
            1,
            int(np.ceil((max_items - capacity) / len(series_ids))),
        )
    spread_by_series = {
        series_id: limit_candidates(
            by_series[series_id],
            min(len(by_series[series_id]), rounds_per_series),
        )
        for series_id in series_ids
    }
    selected: list[tuple[str, str, int, np.ndarray, int, int]] = []
    round_index = 0
    while len(selected) < max_items:
        active = [
            series_id
            for series_id in series_ids
            if round_index < len(spread_by_series[series_id])
        ]
        if not active:
            break
        slots = max_items - len(selected)
        if len(active) > slots:
            active = limit_candidates(active, slots)
        selected.extend(
            spread_by_series[series_id][round_index]
            for series_id in active
        )
        round_index += 1
    return selected


def summarize_feature_rows(
    rows: list[dict[str, Any]],
    *,
    weights: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    feature_names = sorted(
        {
            name
            for row in rows
            for name in row
            if name.startswith("full__") or name.startswith("measure__")
        }
    )
    if weights is None:
        weights = np.ones(len(rows), dtype=float)
    summary: dict[str, dict[str, float]] = {}
    for feature_name in feature_names:
        values = np.asarray(
            [float(row.get(feature_name, float("nan"))) for row in rows],
            dtype=float,
        )
        quantiles = weighted_quantile(values, weights)
        if not quantiles:
            continue
        summary[feature_name] = {
            "p05": quantiles[0],
            "p25": quantiles[1],
            "p50": quantiles[2],
            "p75": quantiles[3],
            "p95": quantiles[4],
            "finite_count": int(np.isfinite(values).sum()),
        }
    return summary


def build_profile_rows(
    source: SourceSpec,
    *,
    gift_eval_dir: Path,
    data_dir: Path,
    max_windows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = source_asset_path(
        source,
        gift_eval_dir=gift_eval_dir,
        data_dir=data_dir,
    )
    if source.kind == "gift_univariate":
        frequency, native_records = read_gift_arrow_targets(path)
        test_tail_steps = gift_eval_short_term_test_holdout_steps(
            frequency,
            native_records,
        )
        validation_steps = HORIZON
        records = [
            (item_id, np.asarray(values, dtype=float))
            for item_id, values in native_records
        ]
        split_policy = (
            "exclude the official GIFT-Eval short-term test tail and one "
            "immediately preceding validation horizon"
        )
    else:
        metadata, tsf_records = read_tsf_series_records(path)
        frequency = str(metadata.get("frequency", "hourly"))
        declared_horizon = int(metadata.get("horizon", HORIZON))
        if declared_horizon != HORIZON:
            raise ValueError(
                f"{source.source_id} declares horizon={declared_horizon}, expected {HORIZON}"
            )
        test_tail_steps = 0
        validation_steps = HORIZON
        records = [
            (record.series_id, np.asarray(record.values, dtype=float))
            for record in tsf_records
        ]
        split_policy = (
            "the Monash TSF contains training histories only; exclude the final "
            "48 points as an internal validation horizon"
        )

    candidates: list[tuple[str, str, int, np.ndarray, int, int]] = []
    expanded_series_count = 0
    master_spec = WindowSpec(MAX_CONTEXT_LENGTH, HORIZON, HORIZON)
    for item_id, native_values in records:
        channels = (
            native_values
            if native_values.ndim == 2
            else native_values.reshape(1, -1)
        )
        for channel_index, values in enumerate(channels):
            expanded_series_count += 1
            cutoff = int(len(values) - test_tail_steps - validation_steps)
            if cutoff < master_spec.length:
                continue
            series_id = (
                str(item_id)
                if native_values.ndim == 1
                else f"{item_id}:dim:{channel_index}"
            )
            candidates.extend(
                (
                    series_id,
                    str(item_id),
                    int(channel_index),
                    np.asarray(values, dtype=float),
                    int(start),
                    cutoff,
                )
                for start in window_starts(cutoff, master_spec)
            )

    primary_candidates = select_series_balanced_candidates(
        candidates,
        max_windows,
    )
    primary_keys = {
        (candidate[0], candidate[2], candidate[4])
        for candidate in primary_candidates
    }
    backup_candidates = select_series_balanced_candidates(
        [
            candidate
            for candidate in candidates
            if (candidate[0], candidate[2], candidate[4]) not in primary_keys
        ],
        max_windows * 5,
    )
    selected_candidates = [*primary_candidates, *backup_candidates]
    feature_rows: list[dict[str, Any]] = []
    accepted_master_count = 0
    rejected_missing_count = 0
    rejected_uninformative_count = 0
    used_series: set[str] = set()
    for series_id, item_id, channel_index, values, start, cutoff in selected_candidates:
        raw_master = np.asarray(
            values[start : start + master_spec.length],
            dtype=float,
        )
        view_payloads: list[tuple[int, np.ndarray, float]] = []
        rejected = False
        for context_length in CONTEXT_LENGTHS:
            raw_view = nested_view(raw_master, context_length)
            imputed_view, observed_fraction = impute_observed_window(raw_view)
            if imputed_view is None:
                rejected_missing_count += 1
                rejected = True
                break
            target = imputed_view[:, None]
            if not is_informative_target(target, context_length):
                rejected_uninformative_count += 1
                rejected = True
                break
            view_payloads.append((context_length, target, observed_fraction))
        if rejected:
            continue

        master_window_id = (
            f"{source.source_id}:{series_id}:{start}:{MAX_CONTEXT_LENGTH}:{HORIZON}"
        )
        raw_future = raw_master[MAX_CONTEXT_LENGTH:]
        future_sha256 = hashlib.sha256(
            np.asarray(raw_future, dtype="<f8").tobytes()
        ).hexdigest()
        for context_length, target, observed_fraction in view_payloads:
            standardized = standardize_target(target, context_length)
            full_features = _realized_features(
                standardized,
                None,
                SEASON_LENGTH,
                context_length,
            )
            measurement = standardized[: context_length + SEASON_LENGTH]
            measurement_features = _realized_features(
                measurement,
                None,
                SEASON_LENGTH,
                context_length,
            )
            row: dict[str, Any] = {
                "source_id": source.source_id,
                "dataset_name": source.dataset_name,
                "family_id": source.family_id,
                "domain": source.domain,
                "master_window_id": master_window_id,
                "future_sha256": future_sha256,
                "series_id": series_id,
                "item_id": item_id,
                "channel_index": channel_index,
                "window_start": start,
                "source_cutoff": cutoff,
                "context_length": context_length,
                "horizon": HORIZON,
                "season_length": SEASON_LENGTH,
                "observed_fraction": round_float(observed_fraction),
            }
            row.update(
                {
                    f"full__{name}": round_float(value)
                    for name, value in full_features.items()
                    if np.isfinite(value)
                }
            )
            row.update(
                {
                    f"measure__{name}": round_float(value)
                    for name, value in measurement_features.items()
                    if np.isfinite(value)
                }
            )
            feature_rows.append(row)
        accepted_master_count += 1
        used_series.add(series_id)
        if accepted_master_count >= max_windows:
            break

    if accepted_master_count < 30:
        raise ValueError(
            f"{source.source_id} produced only {accepted_master_count} paired windows"
        )
    return feature_rows, {
        **asdict(source),
        "asset_path": str(path),
        "asset_sha256": sha256_path(path),
        "source_frequency": frequency,
        "split_policy": split_policy,
        "official_test_tail_steps": int(test_tail_steps),
        "validation_excluded_steps": int(validation_steps),
        "expanded_series_count": int(expanded_series_count),
        "candidate_master_window_count": len(candidates),
        "selected_candidate_count": len(selected_candidates),
        "paired_master_window_count": int(accepted_master_count),
        "view_row_count": int(len(feature_rows)),
        "used_series_count": len(used_series),
        "rejected_missing_count": int(rejected_missing_count),
        "rejected_uninformative_count": int(rejected_uninformative_count),
    }


def build_suite(
    *,
    sources: tuple[SourceSpec, ...],
    gift_eval_dir: Path,
    data_dir: Path,
    max_windows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        print(
            f"[{index + 1}/{len(sources)}] profiling {source.source_id}",
            flush=True,
        )
        rows, inventory = build_profile_rows(
            source,
            gift_eval_dir=gift_eval_dir,
            data_dir=data_dir,
            max_windows=max_windows,
        )
        all_rows.extend(rows)
        source_inventory.append(inventory)

    profiles: dict[str, Any] = {}
    global_profiles: dict[str, Any] = {}
    for context_length in CONTEXT_LENGTHS:
        length_rows = [
            row for row in all_rows if int(row["context_length"]) == context_length
        ]
        for source in sources:
            source_rows = [
                row
                for row in length_rows
                if str(row["source_id"]) == source.source_id
            ]
            profile_id = f"{source.source_id}__L{context_length}_H{HORIZON}"
            profiles[profile_id] = {
                "profile_id": profile_id,
                "source_id": source.source_id,
                "dataset_name": source.dataset_name,
                "family_id": source.family_id,
                "domain": source.domain,
                "context_length": context_length,
                "horizon": HORIZON,
                "season_length": SEASON_LENGTH,
                "window_count": len(source_rows),
                "features": summarize_feature_rows(source_rows),
            }
        weights = source_balanced_weights(length_rows)
        global_id = f"family_macro__L{context_length}_H{HORIZON}"
        global_profiles[global_id] = {
            "profile_id": global_id,
            "role": "family-balanced reference distribution",
            "context_length": context_length,
            "horizon": HORIZON,
            "season_length": SEASON_LENGTH,
            "source_count": len({row["source_id"] for row in length_rows}),
            "family_count": len({row["family_id"] for row in length_rows}),
            "domain_count": len({row["domain"] for row in length_rows}),
            "window_count": len(length_rows),
            "weighting": (
                "equal family mass; equal source-config mass within family; "
                "equal paired-window mass within source config"
            ),
            "features": summarize_feature_rows(length_rows, weights=weights),
        }

    suite = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "context_lengths": list(CONTEXT_LENGTHS),
            "horizon": HORIZON,
            "season_length": SEASON_LENGTH,
            "max_windows_per_source": max_windows,
            "pairing_policy": (
                "select one raw 504+48 master window, then expose suffix contexts "
                "of length 96, 168, 336, and 504 with the identical raw future"
            ),
            "sampling_policy": (
                "sample family uniformly, source config uniformly within family, "
                "then paired master window uniformly within source"
            ),
            "profile_role": (
                "real-data nuisance/support calibration; dataset identity is "
                "provenance and a robustness stratum, not a benchmark axis"
            ),
        },
        "selection": {
            "source_count": len(sources),
            "family_count": len({source.family_id for source in sources}),
            "domain_count": len({source.domain for source in sources}),
            "domains": sorted({source.domain for source in sources}),
            "inclusion_rule": (
                "public GIFT-Eval/Monash hourly source; supports a leakage-safe "
                "504+48 window; univariate or channel-wise univariate evaluation"
            ),
            "performance_blind": True,
        },
        "sources": source_inventory,
        "profiles": profiles,
        "global_profiles": global_profiles,
    }
    return suite, all_rows, source_inventory


def write_outputs(
    output_dir: Path,
    *,
    suite: dict[str, Any],
    rows: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"profile suite is already sealed: {output_dir}")
    write_json(output_dir / "profile_suite.json", suite)
    write_csv(output_dir / "profile_rows.csv", rows)
    write_csv(output_dir / "source_inventory.csv", inventory)
    (output_dir / "report.md").write_text(render_report(suite), encoding="utf-8")
    manifest_files = [
        "profile_suite.json",
        "profile_rows.csv",
        "source_inventory.csv",
        "report.md",
    ]
    manifest = {
        "schema_version": "paper_v4_profile_suite_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "builder_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "builder_sha256": sha256_path(Path(__file__).resolve()),
        "files": {
            name: {
                "sha256": sha256_path(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in manifest_files
        },
    }
    write_json(output_dir / "manifest.json", manifest)


def render_report(suite: dict[str, Any]) -> str:
    config = suite["config"]
    lines = [
        "# Paper v4 multi-lookback profile suite",
        "",
        f"- Shape: `H={config['horizon']}`, `L={config['context_lengths']}`, "
        f"`period={config['season_length']}`.",
        f"- Coverage: {suite['selection']['source_count']} configs, "
        f"{suite['selection']['family_count']} families, "
        f"{suite['selection']['domain_count']} domains.",
        "- Pairing: every accepted master window contributes all four L views "
        "with the same raw 48-step future.",
        "- Aggregation: family -> source config -> paired window, equal mass at "
        "each level.",
        "",
        "| source | domain | family | paired windows | used series | missing rejects |",
        "|---|---|---|---:|---:|---:|",
    ]
    for source in suite["sources"]:
        lines.append(
            f"| {source['dataset_name']} | {source['domain']} | "
            f"{source['family_id']} | {source['paired_master_window_count']} | "
            f"{source['used_series_count']} | {source['rejected_missing_count']} |"
        )
    lines.extend(
        [
            "",
            "The dataset label is retained only for provenance and source-domain "
            "robustness analysis. Synthetic sampling uses the frozen hierarchical "
            "sampling policy and does not expose a dataset identity to models.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(file for file in path.rglob("*") if file.is_file())
        for file in files:
            digest.update(str(file.relative_to(path)).encode("utf-8"))
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_float(value: float) -> float:
    return float(round(float(value), 8))


def main() -> int:
    args = parse_args()
    if args.max_windows_per_source < 30:
        raise ValueError("--max-windows-per-source must be at least 30")
    sources = selected_source_specs(args.sources)
    suite, rows, inventory = build_suite(
        sources=sources,
        gift_eval_dir=args.gift_eval_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        max_windows=int(args.max_windows_per_source),
    )
    write_outputs(args.output_dir.resolve(), suite=suite, rows=rows, inventory=inventory)
    print(f"profile suite: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
