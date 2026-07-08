#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for path in (BACKEND_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.synthetic_generation_service import (  # noqa: E402
    ACCEPTANCE_PROFILE_BY_CAPABILITY,
    ACCEPTANCE_PROFILE_GROUPS,
    ANCHOR_FEATURE_QUANTILES,
    BOUNDED_ACCEPTANCE_FEATURES,
    CAPABILITIES_BY_ID,
    PILOT_ACCEPTANCE_CAPS,
    PILOT_ACCEPTANCE_MINS,
    TARGET_FEATURES_BY_CAPABILITY,
    _attempt_seed,
    _generate_sample_values,
    _normalize_covariates,
    _realized_features,
    _resolve_seasonality,
    _seed_for,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/research/synthetic-v2-acceptance-sweep"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs/superpowers/baselines/2026-07-08-synthetic-v2-acceptance-sweep.md"
DEFAULT_CAPABILITIES = tuple(CAPABILITIES_BY_ID)
CONTEXT_LENGTH = 168
HORIZON = 24
MAX_ATTEMPTS = 12
DEFAULT_TARGET_DIM = 3
PROFILE_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0, 2.5)
EVENT_MULTIPLIERS = (2.0, 3.0, 5.0, 7.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep synthetic v2 hard-acceptance cap multipliers.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--sample-count", type=int, default=32, help="Base samples per capability and intensity.")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--target-dim", type=int, default=DEFAULT_TARGET_DIM)
    parser.add_argument("--frequency", default="h")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_sweep(
        sample_count=args.sample_count,
        seed=args.seed,
        context_length=args.context_length,
        horizon=args.horizon,
        target_dim=args.target_dim,
        frequency=args.frequency,
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(summary, output_dir=args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote summary: {summary_path}")
    print(f"wrote report: {args.report}")
    return 0


def run_sweep(
    *,
    sample_count: int,
    seed: int,
    context_length: int = CONTEXT_LENGTH,
    horizon: int = HORIZON,
    target_dim: int = DEFAULT_TARGET_DIM,
    frequency: str = "h",
    capabilities: tuple[str, ...] = DEFAULT_CAPABILITIES,
) -> dict[str, Any]:
    cap_sets = build_cap_sets()
    attempts = generate_attempt_rows(
        capabilities=capabilities,
        sample_count=sample_count,
        seed=seed,
        context_length=context_length,
        horizon=horizon,
        target_dim=target_dim,
        frequency=frequency,
    )
    cells: list[dict[str, Any]] = []
    for strategy_id, cap_set in cap_sets.items():
        for capability_id in capabilities:
            for intensity in range(1, 6):
                key = (capability_id, intensity)
                cells.append(
                    evaluate_cell(
                        strategy_id,
                        capability_id,
                        intensity,
                        attempts[key],
                        caps=cap_set["caps"],
                        mins=cap_set["mins"],
                    )
                )
    strategy_summary = summarize_strategies(cells)
    recommendation = recommend_strategy(strategy_summary)
    return {
        "schema_version": "synthetic_v2_acceptance_sweep.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "sample_count_per_capability_intensity": sample_count,
            "seed": seed,
            "context_length": context_length,
            "horizon": horizon,
            "target_dim": target_dim,
            "frequency": frequency,
            "max_attempts": MAX_ATTEMPTS,
            "profile_multipliers": list(PROFILE_MULTIPLIERS),
            "event_multipliers": list(EVENT_MULTIPLIERS),
        },
        "cap_sets": cap_sets,
        "seasonality": seasonality_summary(capabilities, seed=seed, frequency=frequency),
        "cells": cells,
        "strategy_summary": strategy_summary,
        "recommendation": recommendation,
    }


def build_cap_sets() -> dict[str, dict[str, Any]]:
    cap_sets: dict[str, dict[str, Any]] = {
        "current": {
            "label": "Current backend caps",
            "profile_multiplier": None,
            "event_multiplier": None,
            "caps": copy_caps(PILOT_ACCEPTANCE_CAPS),
            "mins": copy_caps(PILOT_ACCEPTANCE_MINS),
        }
    }
    for multiplier in PROFILE_MULTIPLIERS:
        strategy_id = f"profile_m{fmt_id(multiplier)}_event5"
        cap_sets[strategy_id] = {
            "label": f"profile p95 * {multiplier:g}; event_lift p95 * 5",
            "profile_multiplier": multiplier,
            "event_multiplier": 5.0,
            "caps": profile_caps(multiplier=multiplier, event_multiplier=5.0),
            "mins": copy_caps(PILOT_ACCEPTANCE_MINS),
        }
    for event_multiplier in EVENT_MULTIPLIERS:
        strategy_id = f"profile_m1_5_event{fmt_id(event_multiplier)}"
        cap_sets[strategy_id] = {
            "label": f"profile p95 * 1.5; event_lift p95 * {event_multiplier:g}",
            "profile_multiplier": 1.5,
            "event_multiplier": event_multiplier,
            "caps": profile_caps(multiplier=1.5, event_multiplier=event_multiplier),
            "mins": copy_caps(PILOT_ACCEPTANCE_MINS),
        }
    return cap_sets


def profile_caps(*, multiplier: float, event_multiplier: float) -> dict[str, dict[str, float]]:
    caps: dict[str, dict[str, float]] = {}
    for capability_id, current_caps in PILOT_ACCEPTANCE_CAPS.items():
        profile_ids = profile_ids_for_capability(capability_id)
        caps[capability_id] = {}
        for feature, current_cap in current_caps.items():
            feature_multiplier = event_multiplier if feature == "event_lift_abs" else multiplier
            cap = cap_from_profiles(feature, profile_ids, multiplier=feature_multiplier, default=current_cap)
            if feature == "hierarchy_residual_mean_abs":
                cap = max(cap, 1e-6)
            caps[capability_id][feature] = float(cap)
    return caps


def cap_from_profiles(feature: str, profile_ids: tuple[str, ...], *, multiplier: float, default: float) -> float:
    values = [
        ANCHOR_FEATURE_QUANTILES[profile_id][feature]["p95"]
        for profile_id in profile_ids
        if feature in ANCHOR_FEATURE_QUANTILES.get(profile_id, {})
    ]
    if not values:
        return float(default)
    cap = max(float(value) for value in values) * multiplier
    if feature in BOUNDED_ACCEPTANCE_FEATURES:
        cap = min(cap, 1.0)
    return float(cap)


def profile_ids_for_capability(capability_id: str) -> tuple[str, ...]:
    profile_id = ACCEPTANCE_PROFILE_BY_CAPABILITY.get(capability_id)
    if profile_id is None:
        return ()
    return ACCEPTANCE_PROFILE_GROUPS.get(profile_id, (profile_id,))


def copy_caps(caps: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {capability_id: {feature: float(value) for feature, value in rows.items()} for capability_id, rows in caps.items()}


def generate_attempt_rows(
    *,
    capabilities: tuple[str, ...],
    sample_count: int,
    seed: int,
    context_length: int,
    horizon: int,
    target_dim: int,
    frequency: str,
) -> dict[tuple[str, int], list[list[dict[str, Any]]]]:
    length = context_length + horizon
    rows: dict[tuple[str, int], list[list[dict[str, Any]]]] = {}
    for capability_id in capabilities:
        resolved_target_dim = target_dim_for_capability(capability_id, target_dim)
        seasonality = _resolve_seasonality(capability_id, requested_frequency=frequency, seed=_seed_for(seed, capability_id, -1))
        for intensity in range(1, 6):
            sample_attempts: list[list[dict[str, Any]]] = []
            for sample_index in range(sample_count):
                sample_seed = _seed_for(seed, capability_id, intensity * 100_000 + sample_index)
                attempts: list[dict[str, Any]] = []
                for attempt in range(MAX_ATTEMPTS):
                    rng = np.random.default_rng(_attempt_seed(sample_seed, attempt))
                    target, _latent_params, covariates = _generate_sample_values(
                        capability_id,
                        length,
                        context_length,
                        resolved_target_dim,
                        seasonality.season_length,
                        intensity,
                        rng,
                    )
                    target = (
                        _standardize_hierarchy_by_context(target, context_length)
                        if capability_id == "hierarchical_coherence"
                        else _standardize_by_context(target, context_length)
                    )
                    if covariates is not None and covariates.size:
                        covariates = _normalize_covariates(covariates, context_length)
                    features = _realized_features(target, covariates, seasonality.season_length, context_length)
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "features": features,
                            "season_length": seasonality.season_length,
                            "target_dim": resolved_target_dim,
                        }
                    )
                sample_attempts.append(attempts)
            rows[(capability_id, intensity)] = sample_attempts
    return rows


def target_dim_for_capability(capability_id: str, requested: int) -> int:
    mode = CAPABILITIES_BY_ID[capability_id].target_dim_mode
    if mode == "fixed_1":
        return 1
    if mode == "multi":
        return max(2, int(requested))
    return 1


def evaluate_cell(
    strategy_id: str,
    capability_id: str,
    intensity: int,
    sample_attempts: list[list[dict[str, Any]]],
    *,
    caps: dict[str, dict[str, float]],
    mins: dict[str, dict[str, float]],
) -> dict[str, Any]:
    accepted_features: list[dict[str, float]] = []
    accepted_attempts: list[int] = []
    first_failed = Counter()
    terminal_failed = Counter()
    rejected = 0
    for attempts in sample_attempts:
        first_accept = None
        first_failure = failed_features(capability_id, attempts[0]["features"], caps=caps, mins=mins)
        if first_failure:
            first_failed.update(first_failure)
        for attempt in attempts:
            failed = failed_features(capability_id, attempt["features"], caps=caps, mins=mins)
            if not failed:
                first_accept = attempt
                break
        if first_accept is None:
            rejected += 1
            terminal_failed.update(failed_features(capability_id, attempts[-1]["features"], caps=caps, mins=mins))
        else:
            accepted_features.append(first_accept["features"])
            accepted_attempts.append(int(first_accept["attempt"]))

    total = len(sample_attempts)
    target_features = TARGET_FEATURES_BY_CAPABILITY.get(capability_id, ())
    cap_features = tuple(caps.get(capability_id, {}))
    return {
        "strategy_id": strategy_id,
        "capability_id": capability_id,
        "intensity": intensity,
        "sample_count": total,
        "accepted_count": len(accepted_features),
        "rejected_count": rejected,
        "acceptance_rate": float(len(accepted_features) / total) if total else 0.0,
        "first_attempt_acceptance_rate": float(sum(attempt == 1 for attempt in accepted_attempts) / total) if total else 0.0,
        "mean_attempts_accepted": float(np.mean(accepted_attempts)) if accepted_attempts else None,
        "p95_attempts_accepted": float(np.quantile(accepted_attempts, 0.95)) if accepted_attempts else None,
        "first_attempt_failed_features": dict(first_failed.most_common()),
        "terminal_failed_features": dict(terminal_failed.most_common()),
        "target_features": summarize_features(accepted_features, target_features),
        "cap_features": summarize_features(accepted_features, cap_features),
    }


def failed_features(
    capability_id: str,
    features: dict[str, float],
    *,
    caps: dict[str, dict[str, float]],
    mins: dict[str, dict[str, float]],
) -> list[str]:
    failed: list[str] = []
    for feature, cap in caps.get(capability_id, {}).items():
        value = features.get(feature)
        if value is not None and np.isfinite(value) and value > cap:
            failed.append(feature)
    for feature, floor in mins.get(capability_id, {}).items():
        value = features.get(feature)
        if value is not None and np.isfinite(value) and value < floor and feature not in failed:
            failed.append(feature)
    return failed


def summarize_features(rows: list[dict[str, float]], features: tuple[str, ...]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for feature in features:
        values = np.asarray([row[feature] for row in rows if feature in row and np.isfinite(row[feature])], dtype=float)
        if values.size:
            summary[feature] = {
                "mean": float(np.mean(values)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }
    return summary


def summarize_strategies(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[cell["strategy_id"]].append(cell)
    out: dict[str, dict[str, Any]] = {}
    for strategy_id, rows in sorted(grouped.items()):
        acceptance = np.asarray([row["acceptance_rate"] for row in rows], dtype=float)
        mean_attempts = np.asarray(
            [row["mean_attempts_accepted"] for row in rows if row["mean_attempts_accepted"] is not None],
            dtype=float,
        )
        failed = Counter()
        terminal = Counter()
        for row in rows:
            failed.update(row["first_attempt_failed_features"])
            terminal.update(row["terminal_failed_features"])
        out[strategy_id] = {
            "cell_count": len(rows),
            "min_acceptance_rate": float(np.min(acceptance)) if acceptance.size else 0.0,
            "p10_acceptance_rate": float(np.quantile(acceptance, 0.10)) if acceptance.size else 0.0,
            "median_acceptance_rate": float(np.median(acceptance)) if acceptance.size else 0.0,
            "mean_acceptance_rate": float(np.mean(acceptance)) if acceptance.size else 0.0,
            "cells_below_0_90": int(np.sum(acceptance < 0.90)),
            "cells_below_0_95": int(np.sum(acceptance < 0.95)),
            "max_mean_attempts_accepted": float(np.max(mean_attempts)) if mean_attempts.size else None,
            "mean_attempts_accepted": float(np.mean(mean_attempts)) if mean_attempts.size else None,
            "top_first_attempt_failed_features": dict(failed.most_common(8)),
            "top_terminal_failed_features": dict(terminal.most_common(8)),
        }
    return out


def recommend_strategy(strategy_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        (strategy_id, stats)
        for strategy_id, stats in strategy_summary.items()
        if strategy_id.startswith("profile_m")
        and stats["min_acceptance_rate"] >= 0.90
        and stats["cells_below_0_95"] <= 2
        and (stats["max_mean_attempts_accepted"] is None or stats["max_mean_attempts_accepted"] <= 3.0)
    ]
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(strategy_sort_key(item[0])),
                item[1]["cells_below_0_95"],
                item[1]["max_mean_attempts_accepted"] or 0.0,
            ),
        )
        strategy_id, stats = ranked[0]
        return {
            "strategy_id": strategy_id,
            "reason": "smallest profile-derived strategy satisfying min acceptance >= 0.90, at most two cells below 0.95, and max mean attempts <= 3.",
            "stats": stats,
        }
    current = strategy_summary.get("current", {})
    return {
        "strategy_id": "current",
        "reason": "no profile-derived sweep strategy satisfied the operational criteria; keep current caps pending generator or cap adjustment.",
        "stats": current,
    }


def strategy_sort_key(strategy_id: str) -> float:
    if "_m1_25_" in strategy_id:
        return 1.25
    if "_m1_5_" in strategy_id:
        return 1.5
    if "_m2_5_" in strategy_id:
        return 2.5
    if "_m2_" in strategy_id:
        return 2.0
    if "_m1_" in strategy_id:
        return 1.0
    return 99.0


def seasonality_summary(capabilities: tuple[str, ...], *, seed: int, frequency: str) -> dict[str, Any]:
    return {
        capability_id: {
            "season_length": resolution.season_length,
            "source": resolution.source,
            "candidate_periods": list(resolution.candidate_periods),
            "profile_ids": list(resolution.profile_ids),
        }
        for capability_id in capabilities
        for resolution in [_resolve_seasonality(capability_id, requested_frequency=frequency, seed=_seed_for(seed, capability_id, -1))]
    }


def render_report(summary: dict[str, Any], *, output_dir: Path) -> str:
    summary_path = output_dir / "summary.json"
    try:
        summary_display_path = summary_path.relative_to(REPO_ROOT)
    except ValueError:
        summary_display_path = summary_path
    lines = [
        "# Synthetic v2 Acceptance Threshold Sweep",
        "",
        f"日期：{datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Purpose",
        "",
        "比较 hard acceptance 阈值策略对生成通过率、重采样成本和失败特征的影响，为论文阶段固定真实分布验收阈值提供依据。",
        "",
        "## Design",
        "",
        f"- Samples: {summary['config']['sample_count_per_capability_intensity']} base samples per capability/intensity, {summary['config']['max_attempts']} deterministic attempts per sample.",
        f"- Window: context={summary['config']['context_length']}, horizon={summary['config']['horizon']}, requested multi target_dim={summary['config']['target_dim']}, frequency={summary['config']['frequency']}.",
        "- Each strategy is evaluated on the same raw attempt pool, so strategy differences come only from thresholds.",
        "- `current` is the backend cap set. `profile_m*_event*` rebuilds caps from real-profile p95 values; bounded features are clipped at 1.0.",
        "- Operational screen used for the automatic recommendation: min acceptance >= 0.90, at most two capability/intensity cells below 0.95, and max mean attempts <= 3.",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | min acc | p10 acc | median acc | cells <0.95 | max mean attempts | top terminal failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for strategy_id, stats in sorted(summary["strategy_summary"].items(), key=lambda item: (item[0] != "current", strategy_sort_key(item[0]), item[0])):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{strategy_id}`",
                    fmt(stats["min_acceptance_rate"]),
                    fmt(stats["p10_acceptance_rate"]),
                    fmt(stats["median_acceptance_rate"]),
                    str(stats["cells_below_0_95"]),
                    fmt(stats["max_mean_attempts_accepted"]),
                    top_features(stats["top_terminal_failed_features"]),
                ]
            )
            + " |"
        )
    recommendation = summary["recommendation"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Recommended strategy: `{recommendation['strategy_id']}`.",
            f"- Reason: {recommendation['reason']}",
            "",
            "## Capability Detail",
            "",
            "下表只展示 `current`、推荐策略，以及相邻的 profile-derived 策略，便于判断是否过紧或过松。",
            "",
        ]
    )
    detail_strategies = detail_strategy_ids(summary)
    for strategy_id in detail_strategies:
        lines.extend(render_strategy_detail(summary, strategy_id))
    lines.extend(
        [
            "",
            "## Seasonality Resolution",
            "",
            "| Capability | season_length | source | candidates |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for capability_id, row in summary["seasonality"].items():
        lines.append(
            f"| `{capability_id}` | {row['season_length']} | `{row['source']}` | "
            f"{', '.join(str(value) for value in row['candidate_periods'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `event_lift_abs` is swept separately because the M5 event profile is sparse and would otherwise dominate covariate acceptance.",
            "- `hierarchy_residual_mean_abs` keeps a fixed floating-point tolerance floor of `1e-6` even though the real M5 p95 is 0.",
            "- This is a first-pass operational sweep at the recorded sample size. Before freezing paper thresholds, rerun with a larger cached attempt pool and keep the same report schema.",
            "- This sweep evaluates generation-side acceptance only. Near-distance DCR/NNDR thresholds still need the separate real-holdout calibration experiment.",
            "",
            f"Full JSON summary: `{summary_display_path}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def detail_strategy_ids(summary: dict[str, Any]) -> list[str]:
    ids = ["current", summary["recommendation"]["strategy_id"], "profile_m1_25_event5", "profile_m1_5_event5", "profile_m2_event5"]
    out: list[str] = []
    for strategy_id in ids:
        if strategy_id in summary["strategy_summary"] and strategy_id not in out:
            out.append(strategy_id)
    return out


def render_strategy_detail(summary: dict[str, Any], strategy_id: str) -> list[str]:
    cells = [cell for cell in summary["cells"] if cell["strategy_id"] == strategy_id]
    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_capability[cell["capability_id"]].append(cell)
    lines = [
        f"### `{strategy_id}`",
        "",
        "| Capability | min acc | i1 acc | i3 acc | i5 acc | max mean attempts | main terminal failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for capability_id, rows in sorted(by_capability.items()):
        rows_by_intensity = {row["intensity"]: row for row in rows}
        acceptance = [row["acceptance_rate"] for row in rows]
        mean_attempts = [row["mean_attempts_accepted"] for row in rows if row["mean_attempts_accepted"] is not None]
        terminal = Counter()
        for row in rows:
            terminal.update(row["terminal_failed_features"])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{capability_id}`",
                    fmt(min(acceptance)),
                    fmt(rows_by_intensity[1]["acceptance_rate"]),
                    fmt(rows_by_intensity[3]["acceptance_rate"]),
                    fmt(rows_by_intensity[5]["acceptance_rate"]),
                    fmt(max(mean_attempts) if mean_attempts else None),
                    top_features(dict(terminal.most_common(4))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    number = float(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    return f"{number:.4g}"


def fmt_id(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def top_features(values: dict[str, int]) -> str:
    if not values:
        return "-"
    return ", ".join(f"{feature}:{count}" for feature, count in list(values.items())[:4])


if __name__ == "__main__":
    raise SystemExit(main())
