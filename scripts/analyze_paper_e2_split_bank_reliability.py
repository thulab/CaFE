#!/usr/bin/env python3
"""Estimate E2 measurement reliability by splitting one paired-group pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_paper_v5_e2 as e2  # noqa: E402
import analyze_paper_v5_e2_seed_bank_reliability as reliability  # noqa: E402
import run_paper_e2_dynamic_stability as stats  # noqa: E402


SCHEMA_VERSION = "paper_e2_split_bank_reliability.v2"
DEFAULT_E2_DIR = REPO_ROOT / "runtime/paper_exp/v7/E2_dynamic_stability"
DEFAULT_BANK_SIZES = (160,)
DEFAULT_SPLIT_SEED = 20260720
DEFAULT_MINIMUM_AGREEMENT = 0.80
DEFAULT_EQUIVALENCE_MARGINS = (0.01, 0.02, 0.05)
DEFAULT_PRIMARY_EQUIVALENCE_MARGIN = 0.02
DEFAULT_PAIR_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_PAIR_CI_LEVEL = 0.95
FORMAL_V7_ROUND_COUNT = 5
FORMAL_V7_SAMPLES_PER_ROUND = 64
FORMAL_V7_POOL_SIZE = 320
FORMAL_V7_BANK_SIZE = 160
NORMAL_95 = 1.959963984540054
PROFILE_KEYS = ["dataset_id", "task_id", "capability_id"]
CELL_KEYS = [*PROFILE_KEYS, "intensity"]
MODEL_CELL_KEYS = ["model_id", *CELL_KEYS]
SCORE_POLICIES = {
    "oracle_context": "oracle_mase",
    "fixed_l504": "fixed_l504_mase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split one E2 paired-group pool into two disjoint banks and "
            "compare continuous scores, capability profiles, tie-aware "
            "model contrasts, and rankings."
        )
    )
    parser.add_argument("--e2-dir", type=Path, default=DEFAULT_E2_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <e2-dir>/split_bank_reliability.",
    )
    parser.add_argument(
        "--bank-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BANK_SIZES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Defaults to requested_models in inference_config.json.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Analyze only these dataset IDs; defaults to all datasets.",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=0,
        help=(
            "Additionally evaluate this many deterministic random disjoint "
            "splits per bank size."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
    )
    parser.add_argument(
        "--minimum-agreement",
        type=float,
        default=DEFAULT_MINIMUM_AGREEMENT,
    )
    parser.add_argument(
        "--equivalence-margins",
        type=float,
        nargs="+",
        default=list(DEFAULT_EQUIVALENCE_MARGINS),
        help=(
            "Relative MASE margins used for practical-equivalence "
            "classification."
        ),
    )
    parser.add_argument(
        "--primary-equivalence-margin",
        type=float,
        default=DEFAULT_PRIMARY_EQUIVALENCE_MARGIN,
    )
    parser.add_argument(
        "--pair-bootstrap-replicates",
        type=int,
        default=DEFAULT_PAIR_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--pair-ci-level",
        type=float,
        default=DEFAULT_PAIR_CI_LEVEL,
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def requested_models(e2_dir: Path, explicit: list[str] | None) -> list[str]:
    if explicit is not None:
        models = [str(model_id) for model_id in explicit]
    else:
        config = json.loads(
            (e2_dir / "inference_config.json").read_text(encoding="utf-8")
        )
        models = [str(model_id) for model_id in config["requested_models"]]
    if len(models) < 2 or len(models) != len(set(models)):
        raise ValueError("at least two unique models are required")
    return models


def ensure_oracle_paths(e2_dir: Path, models: list[str]) -> list[Path]:
    paths: list[Path] = []
    for model_id in models:
        destination = e2.oracle_path(
            e2_dir,
            model_id,
            prediction_kind="synthetic",
        )
        if not destination.is_file():
            source = e2.prediction_path(
                e2_dir,
                model_id,
                prediction_kind="synthetic",
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            e2.compact_prediction_file(
                source,
                destination,
                model_id=model_id,
                prediction_kind="synthetic",
            )
        paths.append(destination)
    return paths


def load_oracle_pool(
    paths: Iterable[Path],
    *,
    datasets: set[str] | None,
) -> pd.DataFrame:
    columns = [
        "model_id",
        "master_sample_id",
        "dataset_id",
        "task_id",
        "capability_id",
        "intensity",
        "paired_group_id",
        "oracle_mase",
        "fixed_l504_mase",
    ]
    optional_identity_columns = ("round_index", "sample_index")
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in e2.iter_jsonl(path):
            dataset_id = str(row["dataset_id"])
            if datasets is not None and dataset_id not in datasets:
                continue
            record = {column: row[column] for column in columns}
            record.update(
                {
                    column: row.get(column)
                    for column in optional_identity_columns
                }
            )
            rows.append(record)
    if not rows:
        raise ValueError("no synthetic oracle scores matched the selection")
    frame = pd.DataFrame(rows)
    frame["intensity"] = frame["intensity"].astype(int)
    for column in ("oracle_mase", "fixed_l504_mase"):
        frame[column] = frame[column].astype(float)
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"non-finite score in {column}")

    has_round_identity = frame[
        list(optional_identity_columns)
    ].notna().all(axis=1)
    if bool(has_round_identity.any()) and not bool(has_round_identity.all()):
        raise ValueError("round/sample identity is only partially populated")
    if bool(has_round_identity.all()):
        frame["round_index"] = frame["round_index"].astype(int)
        frame["sample_index"] = frame["sample_index"].astype(int)
        group_order = frame[
            [
                *PROFILE_KEYS,
                "paired_group_id",
                "round_index",
                "sample_index",
            ]
        ].drop_duplicates()
        identity_counts = group_order.groupby(
            [*PROFILE_KEYS, "paired_group_id"],
            sort=False,
        ).size()
        if (identity_counts != 1).any():
            raise ValueError(
                "paired group maps to multiple round/sample identities"
            )
        if group_order.duplicated(
            [*PROFILE_KEYS, "round_index", "sample_index"]
        ).any():
            raise ValueError(
                "multiple paired groups share one round/sample identity"
            )
        group_order = group_order.sort_values(
            [*PROFILE_KEYS, "round_index", "sample_index"],
            kind="stable",
        )
    else:
        group_order = (
            frame[[*PROFILE_KEYS, "paired_group_id"]]
            .drop_duplicates()
            .sort_values(
                [*PROFILE_KEYS, "paired_group_id"],
                kind="stable",
            )
        )
    group_order["pool_index"] = group_order.groupby(
        PROFILE_KEYS,
        sort=False,
    ).cumcount()
    frame = frame.merge(
        group_order[
            [*PROFILE_KEYS, "paired_group_id", "pool_index"]
        ],
        on=[*PROFILE_KEYS, "paired_group_id"],
        how="left",
        validate="many_to_one",
    )
    validate_oracle_pool(frame)
    return frame


def validate_oracle_pool(frame: pd.DataFrame) -> None:
    duplicate_keys = [*MODEL_CELL_KEYS, "paired_group_id"]
    if frame.duplicated(duplicate_keys).any():
        raise ValueError("duplicate model/cell/paired-group oracle score")

    profile_sizes = (
        frame.groupby(PROFILE_KEYS, sort=False)["paired_group_id"]
        .nunique()
        .rename("profile_size")
    )
    if profile_sizes.empty or int(profile_sizes.min()) < 2:
        raise ValueError("each profile requires at least two paired groups")

    observed = (
        frame.groupby(MODEL_CELL_KEYS, sort=False)["paired_group_id"]
        .nunique()
        .rename("observed")
        .reset_index()
        .merge(
            profile_sizes.reset_index(),
            on=PROFILE_KEYS,
            how="left",
            validate="many_to_one",
        )
    )
    incomplete = observed[observed["observed"] != observed["profile_size"]]
    if not incomplete.empty:
        raise ValueError(
            "model/cell does not cover the complete paired-group pool: "
            f"{incomplete.head().to_dict(orient='records')}"
        )

    models_per_cell = frame.groupby(CELL_KEYS, sort=False)["model_id"].nunique()
    if (models_per_cell < 2).any():
        raise ValueError("every analyzed cell requires at least two models")


def pool_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        column
        for column in ("round_index", "sample_index")
        if column in frame.columns and frame[column].notna().all()
    ]
    return (
        frame[
            [
                *PROFILE_KEYS,
                "paired_group_id",
                "pool_index",
                *identity_columns,
            ]
        ]
        .drop_duplicates()
        .sort_values([*PROFILE_KEYS, "pool_index"], kind="stable")
        .reset_index(drop=True)
    )


def validate_formal_v7_pool(catalog: pd.DataFrame) -> dict[str, Any]:
    required = {"round_index", "sample_index", "pool_index"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(
            "formal v7 320-pool requires round/sample identity columns: "
            + ", ".join(missing)
        )
    expected_identity = {
        (round_index, sample_index)
        for round_index in range(1, FORMAL_V7_ROUND_COUNT + 1)
        for sample_index in range(FORMAL_V7_SAMPLES_PER_ROUND)
    }
    profile_count = 0
    for profile, group in catalog.groupby(PROFILE_KEYS, sort=True):
        profile_count += 1
        if len(group) != FORMAL_V7_POOL_SIZE:
            raise ValueError(
                f"formal v7 profile {profile} must contain exactly "
                f"{FORMAL_V7_POOL_SIZE} groups, got {len(group)}"
            )
        observed_identity = set(
            zip(
                group["round_index"].astype(int),
                group["sample_index"].astype(int),
                strict=True,
            )
        )
        if observed_identity != expected_identity:
            raise ValueError(
                f"formal v7 profile {profile} has incomplete "
                "round/sample identities"
            )
        if set(group["pool_index"].astype(int)) != set(
            range(FORMAL_V7_POOL_SIZE)
        ):
            raise ValueError(
                f"formal v7 profile {profile} has invalid pool indexes"
            )
    assignments = split_assignments(
        catalog,
        bank_size=FORMAL_V7_BANK_SIZE,
        split_kind="ordered",
        repeat_index=0,
        split_seed=0,
    )
    block_counts = (
        assignments.groupby([*PROFILE_KEYS, "bank_id"], sort=True)[
            "paired_group_id"
        ]
        .nunique()
    )
    if not bool((block_counts == FORMAL_V7_BANK_SIZE).all()):
        raise AssertionError("formal v7 analysis blocks are incomplete")
    if assignments.duplicated(
        [*PROFILE_KEYS, "paired_group_id"]
    ).any():
        raise AssertionError("formal v7 analysis blocks overlap")
    return {
        "ordering": ["round_index", "sample_index"],
        "total_per_intensity": FORMAL_V7_POOL_SIZE,
        "block_size": FORMAL_V7_BANK_SIZE,
        "block_ids": ["A", "B"],
        "profile_count": profile_count,
        "mutually_exclusive": True,
        "complete_partition": True,
    }


def stable_profile_seed(
    split_seed: int,
    repeat_index: int,
    profile: tuple[Any, ...],
) -> int:
    payload = "|".join(
        [str(split_seed), str(repeat_index), *(str(value) for value in profile)]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def split_assignments(
    catalog: pd.DataFrame,
    *,
    bank_size: int,
    split_kind: str,
    repeat_index: int,
    split_seed: int,
) -> pd.DataFrame:
    if bank_size < 2:
        raise ValueError("bank size must be at least two")
    if split_kind not in {"ordered", "random"}:
        raise ValueError(f"unsupported split kind: {split_kind}")

    rows: list[dict[str, Any]] = []
    for profile, group in catalog.groupby(PROFILE_KEYS, sort=True):
        ordered = group.sort_values("pool_index", kind="stable")
        if 2 * bank_size > len(ordered):
            raise ValueError(
                f"bank size {bank_size} requires {2 * bank_size} groups, "
                f"but profile {profile} contains {len(ordered)}"
            )
        if split_kind == "random":
            rng = np.random.default_rng(
                stable_profile_seed(split_seed, repeat_index, profile)
            )
            ordered = ordered.iloc[
                rng.permutation(len(ordered))
            ].reset_index(drop=True)
        left = ordered.iloc[:bank_size]
        right = ordered.iloc[-bank_size:]
        for bank_id, selected in (("A", left), ("B", right)):
            rows.extend(
                {
                    **dict(zip(PROFILE_KEYS, profile, strict=True)),
                    "paired_group_id": str(record["paired_group_id"]),
                    "bank_id": bank_id,
                    "analysis_block_index": block_index,
                    "pool_index": int(record["pool_index"]),
                    **{
                        column: int(record[column])
                        for column in ("round_index", "sample_index")
                        if column in record
                        and pd.notna(record[column])
                    },
                }
                for block_index, record in enumerate(
                    selected.to_dict(orient="records")
                )
            )
    assignments = pd.DataFrame(rows)
    if assignments.duplicated(
        [*PROFILE_KEYS, "paired_group_id"]
    ).any():
        raise AssertionError("split banks overlap")
    return assignments


def cell_model_scores(
    split_frame: pd.DataFrame,
    *,
    score_column: str,
    bank_size: int,
) -> pd.DataFrame:
    result = (
        split_frame.groupby(["bank_id", *MODEL_CELL_KEYS], sort=True)[
            score_column
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mase_mean",
                "std": "mase_std",
                "count": "sample_count",
            }
        )
    )
    if not (result["sample_count"] == bank_size).all():
        raise ValueError("split cell score has an unexpected sample count")
    result["mase_se"] = result["mase_std"] / np.sqrt(
        result["sample_count"]
    )
    result["mase_ci_low"] = (
        result["mase_mean"] - NORMAL_95 * result["mase_se"]
    )
    result["mase_ci_high"] = (
        result["mase_mean"] + NORMAL_95 * result["mase_se"]
    )
    if (result["mase_mean"] <= 0).any():
        raise ValueError("MASE means must be positive")
    result["log_mase"] = np.log(result["mase_mean"])
    result["relative_log_mase"] = result["log_mase"] - result.groupby(
        ["bank_id", *CELL_KEYS],
        sort=False,
    )["log_mase"].transform("mean")
    result["model_rank"] = result.groupby(
        ["bank_id", *CELL_KEYS],
        sort=False,
    )["mase_mean"].rank(method="average", ascending=True)
    return result


def stable_bootstrap_seed(
    base_seed: int,
    values: tuple[Any, ...],
) -> int:
    payload = "|".join([str(base_seed), *(str(value) for value in values)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def practical_equivalence_state(
    *,
    ci_low: float,
    ci_high: float,
    margin: float,
) -> str:
    if ci_high < -margin:
        return "left_better"
    if ci_low > margin:
        return "right_better"
    if ci_low >= -margin and ci_high <= margin:
        return "equivalent"
    return "unresolved"


def tie_aware_pair_states(
    split_frame: pd.DataFrame,
    *,
    score_column: str,
    bank_size: int,
    equivalence_margins: tuple[float, ...],
    bootstrap_replicates: int,
    ci_level: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alpha = (1.0 - ci_level) / 2.0
    for key, group in split_frame.groupby(
        ["bank_id", *CELL_KEYS],
        sort=True,
    ):
        bank_id, *cell_values = key
        matrix = group.pivot(
            index="paired_group_id",
            columns="model_id",
            values=score_column,
        )
        if len(matrix) != bank_size or matrix.isna().any().any():
            raise ValueError(f"incomplete paired model matrix for {key}")
        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        values = matrix.to_numpy(dtype=float)
        rng = np.random.default_rng(
            stable_bootstrap_seed(bootstrap_seed, tuple(key))
        )
        draws = rng.integers(
            0,
            bank_size,
            size=(bootstrap_replicates, bank_size),
        )
        bootstrap_means = values[draws].mean(axis=1)
        point_means = values.mean(axis=0)
        for left_index, right_index in combinations(
            range(len(matrix.columns)),
            2,
        ):
            left_model = str(matrix.columns[left_index])
            right_model = str(matrix.columns[right_index])
            point_denominator = max(
                abs(point_means[left_index])
                + abs(point_means[right_index]),
                1e-12,
            )
            relative_gap = float(
                2.0
                * (
                    point_means[left_index]
                    - point_means[right_index]
                )
                / point_denominator
            )
            bootstrap_denominator = np.maximum(
                np.abs(bootstrap_means[:, left_index])
                + np.abs(bootstrap_means[:, right_index]),
                1e-12,
            )
            bootstrap_gap = (
                2.0
                * (
                    bootstrap_means[:, left_index]
                    - bootstrap_means[:, right_index]
                )
                / bootstrap_denominator
            )
            low, high = np.quantile(
                bootstrap_gap,
                [alpha, 1.0 - alpha],
            )
            for margin in equivalence_margins:
                rows.append(
                    {
                        **dict(
                            zip(CELL_KEYS, cell_values, strict=True)
                        ),
                        "left_model": left_model,
                        "right_model": right_model,
                        "bank_id": bank_id,
                        "left_mase_mean": float(
                            point_means[left_index]
                        ),
                        "right_mase_mean": float(
                            point_means[right_index]
                        ),
                        "relative_mase_gap": relative_gap,
                        "bootstrap_ci_low": float(low),
                        "bootstrap_ci_high": float(high),
                        "bootstrap_replicates": bootstrap_replicates,
                        "ci_level": ci_level,
                        "equivalence_margin": margin,
                        "state": practical_equivalence_state(
                            ci_low=float(low),
                            ci_high=float(high),
                            margin=margin,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compare_pair_states(
    pair_a: pd.DataFrame,
    pair_b: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = [
        *CELL_KEYS,
        "left_model",
        "right_model",
        "equivalence_margin",
    ]
    compared = pair_a.merge(
        pair_b,
        on=keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    decisive = {"left_better", "right_better"}
    compared["state_match"] = compared["state_a"] == compared["state_b"]
    compared["both_decisive"] = compared["state_a"].isin(
        decisive
    ) & compared["state_b"].isin(decisive)
    compared["direction_conflict"] = (
        compared["both_decisive"]
        & (compared["state_a"] != compared["state_b"])
    )
    compared["both_equivalent"] = (
        (compared["state_a"] == "equivalent")
        & (compared["state_b"] == "equivalent")
    )
    compared["both_unresolved"] = (
        (compared["state_a"] == "unresolved")
        & (compared["state_b"] == "unresolved")
    )
    compared["one_decisive_one_equivalent"] = (
        compared["state_a"].isin(decisive)
        ^ compared["state_b"].isin(decisive)
    ) & (
        (compared["state_a"] == "equivalent")
        | (compared["state_b"] == "equivalent")
    )
    compared["one_decisive_one_unresolved"] = (
        compared["state_a"].isin(decisive)
        ^ compared["state_b"].isin(decisive)
    ) & (
        (compared["state_a"] == "unresolved")
        | (compared["state_b"] == "unresolved")
    )
    compared["equivalent_vs_unresolved"] = [
        frozenset((left, right))
        == frozenset(("equivalent", "unresolved"))
        for left, right in zip(
            compared["state_a"],
            compared["state_b"],
            strict=True,
        )
    ]

    summaries: dict[str, Any] = {}
    for margin, group in compared.groupby("equivalence_margin", sort=True):
        both_decisive = group[group["both_decisive"]]
        directional_agreement = (
            float(
                (
                    both_decisive["state_a"]
                    == both_decisive["state_b"]
                ).mean()
            )
            if len(both_decisive)
            else 1.0
        )
        summaries[f"{float(margin):g}"] = {
            "equivalence_margin": float(margin),
            "model_pair_cell_count": len(group),
            "state_match_rate": float(group["state_match"].mean()),
            "both_decisive_count": int(group["both_decisive"].sum()),
            "both_decisive_directional_agreement": (
                directional_agreement
            ),
            "direction_conflict_count": int(
                group["direction_conflict"].sum()
            ),
            "conclusion_compatibility_rate": float(
                1.0 - group["direction_conflict"].mean()
            ),
            "both_equivalent_count": int(
                group["both_equivalent"].sum()
            ),
            "both_unresolved_count": int(
                group["both_unresolved"].sum()
            ),
            "one_decisive_one_equivalent_count": int(
                group["one_decisive_one_equivalent"].sum()
            ),
            "one_decisive_one_unresolved_count": int(
                group["one_decisive_one_unresolved"].sum()
            ),
            "equivalent_vs_unresolved_count": int(
                group["equivalent_vs_unresolved"].sum()
            ),
        }
    return compared, summaries


def partial_order_ranks(
    pair_states: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    primary_margin: float,
) -> pd.DataFrame:
    selected = pair_states[
        np.isclose(
            pair_states["equivalence_margin"],
            primary_margin,
        )
    ]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(
        ["bank_id", *CELL_KEYS],
        sort=True,
    ):
        bank_id, *cell_values = key
        models = sorted(
            set(group["left_model"]) | set(group["right_model"])
        )
        outgoing = {model: set() for model in models}
        incoming = {model: set() for model in models}
        equivalent_count = {model: 0 for model in models}
        unresolved_count = {model: 0 for model in models}
        for record in group.to_dict(orient="records"):
            left = str(record["left_model"])
            right = str(record["right_model"])
            state = str(record["state"])
            if state == "left_better":
                outgoing[left].add(right)
                incoming[right].add(left)
            elif state == "right_better":
                outgoing[right].add(left)
                incoming[left].add(right)
            elif state == "equivalent":
                equivalent_count[left] += 1
                equivalent_count[right] += 1
            else:
                unresolved_count[left] += 1
                unresolved_count[right] += 1

        remaining = set(models)
        tiers: dict[str, int] = {}
        tier_index = 1
        while remaining:
            current = sorted(
                model
                for model in remaining
                if not (incoming[model] & remaining)
            )
            if not current:
                raise ValueError(f"decisive comparison cycle for {key}")
            for model in current:
                tiers[model] = tier_index
            remaining.difference_update(current)
            tier_index += 1

        cell_scores = scores
        for column, value in zip(CELL_KEYS, cell_values, strict=True):
            cell_scores = cell_scores[cell_scores[column] == value]
        cell_scores = cell_scores[cell_scores["bank_id"] == bank_id]
        score_by_model = cell_scores.set_index("model_id")
        for model in models:
            best_rank = 1 + len(incoming[model])
            worst_rank = len(models) - len(outgoing[model])
            rows.append(
                {
                    **dict(zip(CELL_KEYS, cell_values, strict=True)),
                    "bank_id": bank_id,
                    "model_id": model,
                    "model_count": len(models),
                    "mase_mean": float(
                        score_by_model.loc[model, "mase_mean"]
                    ),
                    "point_estimate_rank": float(
                        score_by_model.loc[model, "model_rank"]
                    ),
                    "partial_order_tier": tiers[model],
                    "best_rank": best_rank,
                    "worst_rank": worst_rank,
                    "rank_interval": (
                        str(best_rank)
                        if best_rank == worst_rank
                        else f"{best_rank}-{worst_rank}"
                    ),
                    "decisively_better_count": len(outgoing[model]),
                    "decisively_worse_count": len(incoming[model]),
                    "equivalent_pair_count": equivalent_count[model],
                    "unresolved_pair_count": unresolved_count[model],
                    "primary_equivalence_margin": primary_margin,
                }
            )
    return pd.DataFrame(rows)


def compare_partial_order_ranks(
    ranks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = [*CELL_KEYS, "model_id"]
    columns = [
        *keys,
        "partial_order_tier",
        "best_rank",
        "worst_rank",
        "rank_interval",
        "point_estimate_rank",
    ]
    left = ranks[ranks["bank_id"] == "A"][columns]
    right = ranks[ranks["bank_id"] == "B"][columns]
    compared = left.merge(
        right,
        on=keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    compared["tier_match"] = (
        compared["partial_order_tier_a"]
        == compared["partial_order_tier_b"]
    )
    compared["rank_interval_overlap"] = (
        np.maximum(compared["best_rank_a"], compared["best_rank_b"])
        <= np.minimum(
            compared["worst_rank_a"],
            compared["worst_rank_b"],
        )
    )

    cell_rows: list[dict[str, Any]] = []
    for cell, group in ranks.groupby(CELL_KEYS, sort=True):
        left_top = set(
            group[
                (group["bank_id"] == "A")
                & (group["partial_order_tier"] == 1)
            ]["model_id"]
        )
        right_top = set(
            group[
                (group["bank_id"] == "B")
                & (group["partial_order_tier"] == 1)
            ]["model_id"]
        )
        union = left_top | right_top
        cell_rows.append(
            {
                **dict(zip(CELL_KEYS, cell, strict=True)),
                "top_tier_a": ";".join(sorted(left_top)),
                "top_tier_b": ";".join(sorted(right_top)),
                "top_tier_exact_match": left_top == right_top,
                "top_tier_jaccard": (
                    len(left_top & right_top) / len(union)
                    if union
                    else 1.0
                ),
            }
        )
    cells = pd.DataFrame(cell_rows)
    return compared, cells, {
        "cell_model_count": len(compared),
        "tier_match_rate": float(compared["tier_match"].mean()),
        "rank_interval_overlap_rate": float(
            compared["rank_interval_overlap"].mean()
        ),
        "cell_count": len(cells),
        "top_tier_exact_match_rate": float(
            cells["top_tier_exact_match"].mean()
        ),
        "top_tier_jaccard_mean": float(cells["top_tier_jaccard"].mean()),
    }


def practical_tie_ranks(
    scores: pd.DataFrame,
    *,
    equivalence_margins: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in scores.groupby(["bank_id", *CELL_KEYS], sort=True):
        bank_id, *cell_values = key
        ordered = group.sort_values(
            ["mase_mean", "model_id"],
            kind="stable",
        )
        for margin in equivalence_margins:
            tier = 0
            tier_anchor = 0.0
            competition_rank = 0
            tier_members: list[dict[str, Any]] = []
            assigned: list[dict[str, Any]] = []
            for position, record in enumerate(
                ordered.to_dict(orient="records"),
                start=1,
            ):
                score = float(record["mase_mean"])
                relative_to_anchor = (
                    0.0
                    if not tier_members
                    else 2.0
                    * (score - tier_anchor)
                    / max(abs(score) + abs(tier_anchor), 1e-12)
                )
                if not tier_members or relative_to_anchor > margin:
                    if tier_members:
                        for member in tier_members:
                            member["tie_tier_size"] = len(tier_members)
                        assigned.extend(tier_members)
                    tier += 1
                    competition_rank = position
                    tier_anchor = score
                    tier_members = []
                    relative_to_anchor = 0.0
                tier_members.append(
                    {
                        **dict(
                            zip(CELL_KEYS, cell_values, strict=True)
                        ),
                        "bank_id": bank_id,
                        "model_id": str(record["model_id"]),
                        "mase_mean": score,
                        "point_estimate_rank": float(
                            record["model_rank"]
                        ),
                        "equivalence_margin": margin,
                        "practical_tie_tier": tier,
                        "practical_tie_rank": competition_rank,
                        "tier_anchor_mase": tier_anchor,
                        "relative_gap_to_tier_anchor": (
                            relative_to_anchor
                        ),
                    }
                )
            for member in tier_members:
                member["tie_tier_size"] = len(tier_members)
            assigned.extend(tier_members)
            rows.extend(assigned)
    return pd.DataFrame(rows)


def compare_practical_tie_ranks(
    ranks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = [*CELL_KEYS, "model_id", "equivalence_margin"]
    columns = [
        *keys,
        "practical_tie_tier",
        "practical_tie_rank",
        "tie_tier_size",
    ]
    compared = ranks[ranks["bank_id"] == "A"][columns].merge(
        ranks[ranks["bank_id"] == "B"][columns],
        on=keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    compared["tie_tier_match"] = (
        compared["practical_tie_tier_a"]
        == compared["practical_tie_tier_b"]
    )
    compared["tie_rank_match"] = (
        compared["practical_tie_rank_a"]
        == compared["practical_tie_rank_b"]
    )

    cell_rows: list[dict[str, Any]] = []
    for key, group in ranks.groupby(
        ["equivalence_margin", *CELL_KEYS],
        sort=True,
    ):
        margin, *cell_values = key
        left = (
            group[group["bank_id"] == "A"]
            .set_index("model_id")
            .sort_index()
        )
        right = (
            group[group["bank_id"] == "B"]
            .set_index("model_id")
            .sort_index()
        )
        if list(left.index) != list(right.index):
            raise ValueError(f"practical tie model mismatch for {key}")
        pair_matches: list[bool] = []
        direction_conflicts = 0
        for left_model, right_model in combinations(left.index, 2):
            states = []
            for frame in (left, right):
                left_tier = int(
                    frame.loc[left_model, "practical_tie_tier"]
                )
                right_tier = int(
                    frame.loc[right_model, "practical_tie_tier"]
                )
                if left_tier == right_tier:
                    state = "tie"
                elif left_tier < right_tier:
                    state = "left_better"
                else:
                    state = "right_better"
                states.append(state)
            pair_matches.append(states[0] == states[1])
            if {
                states[0],
                states[1],
            } == {"left_better", "right_better"}:
                direction_conflicts += 1
        left_top = set(
            left[left["practical_tie_tier"] == 1].index
        )
        right_top = set(
            right[right["practical_tie_tier"] == 1].index
        )
        union = left_top | right_top
        cell_rows.append(
            {
                **dict(zip(CELL_KEYS, cell_values, strict=True)),
                "equivalence_margin": float(margin),
                "model_count": len(left),
                "pair_count": len(pair_matches),
                "tie_pair_state_agreement": float(
                    np.mean(pair_matches)
                ),
                "direction_conflict_count": direction_conflicts,
                "conclusion_compatibility_rate": (
                    1.0
                    - direction_conflicts / max(len(pair_matches), 1)
                ),
                "exact_tie_rank_vector": bool(
                    np.array_equal(
                        left["practical_tie_rank"].to_numpy(dtype=int),
                        right["practical_tie_rank"].to_numpy(dtype=int),
                    )
                ),
                "top_tier_a": ";".join(sorted(left_top)),
                "top_tier_b": ";".join(sorted(right_top)),
                "top_tier_exact_match": left_top == right_top,
                "top_tier_jaccard": (
                    len(left_top & right_top) / len(union)
                    if union
                    else 1.0
                ),
                "top_tier_size_a": len(left_top),
                "top_tier_size_b": len(right_top),
                "tier_count_a": int(
                    left["practical_tie_tier"].max()
                ),
                "tier_count_b": int(
                    right["practical_tie_tier"].max()
                ),
            }
        )
    cells = pd.DataFrame(cell_rows)
    summaries: dict[str, Any] = {}
    for margin, group in cells.groupby("equivalence_margin", sort=True):
        model_group = compared[
            np.isclose(compared["equivalence_margin"], margin)
        ]
        summaries[f"{float(margin):g}"] = {
            "equivalence_margin": float(margin),
            "cell_count": len(group),
            "cell_model_count": len(model_group),
            "tie_pair_state_agreement": {
                "mean": float(group["tie_pair_state_agreement"].mean()),
                "median": float(
                    group["tie_pair_state_agreement"].median()
                ),
                "minimum": float(group["tie_pair_state_agreement"].min()),
            },
            "direction_conflict_count": int(
                group["direction_conflict_count"].sum()
            ),
            "conclusion_compatibility_rate": float(
                group["conclusion_compatibility_rate"].mean()
            ),
            "exact_tie_rank_vector_rate": float(
                group["exact_tie_rank_vector"].mean()
            ),
            "tie_tier_match_rate": float(
                model_group["tie_tier_match"].mean()
            ),
            "tie_rank_match_rate": float(
                model_group["tie_rank_match"].mean()
            ),
            "top_tier_exact_match_rate": float(
                group["top_tier_exact_match"].mean()
            ),
            "top_tier_jaccard_mean": float(
                group["top_tier_jaccard"].mean()
            ),
            "top_tier_size_mean": float(
                pd.concat(
                    [
                        group["top_tier_size_a"],
                        group["top_tier_size_b"],
                    ],
                    ignore_index=True,
                ).mean()
            ),
            "top_tier_single_model_rate": float(
                pd.concat(
                    [
                        group["top_tier_size_a"],
                        group["top_tier_size_b"],
                    ],
                    ignore_index=True,
                ).eq(1).mean()
            ),
            "tier_count_mean": float(
                pd.concat(
                    [
                        group["tier_count_a"],
                        group["tier_count_b"],
                    ],
                    ignore_index=True,
                ).mean()
            ),
        }
    return compared, cells, summaries


def rank_reliability(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    *,
    minimum_agreement: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, left_group in scores_a.groupby(CELL_KEYS, sort=True):
        right_group = scores_b
        for column, value in zip(CELL_KEYS, cell, strict=True):
            right_group = right_group[right_group[column] == value]
        left = left_group.set_index("model_id").sort_index()
        right = right_group.set_index("model_id").sort_index()
        if list(left.index) != list(right.index):
            raise ValueError(f"split-bank model mismatch for {cell}")
        left_score = left["mase_mean"].to_numpy(dtype=float)
        right_score = right["mase_mean"].to_numpy(dtype=float)
        left_rank = left["model_rank"].to_numpy(dtype=float)
        right_rank = right["model_rank"].to_numpy(dtype=float)
        ordering = stats.pairwise_ordering_agreement(
            left_score,
            right_score,
        )["agreement"]
        agreement = 1.0 if ordering is None else float(ordering)
        top_k = min(3, len(left))
        left_top = set(left["mase_mean"].nsmallest(top_k).index)
        right_top = set(right["mase_mean"].nsmallest(top_k).index)
        rows.append(
            {
                **dict(zip(CELL_KEYS, cell, strict=True)),
                "model_count": len(left),
                "pair_count": len(left) * (len(left) - 1) // 2,
                "pairwise_ordering_agreement": agreement,
                "passed": agreement >= minimum_agreement,
                "spearman_rho": float(
                    stats.spearman_rank_correlation(
                        left_rank,
                        right_rank,
                    )
                ),
                "kendall_tau_b": float(
                    stats.kendall_tau_b(left_rank, right_rank)
                ),
                "exact_rank_vector": bool(
                    np.array_equal(left_rank, right_rank)
                ),
                "top1_agreement": bool(
                    left["mase_mean"].idxmin()
                    == right["mase_mean"].idxmin()
                ),
                "top3_overlap_rate": float(
                    len(left_top & right_top) / top_k
                ),
            }
        )
    frame = pd.DataFrame(rows)
    agreement = frame["pairwise_ordering_agreement"].to_numpy(dtype=float)
    summary = {
        "cell_count": len(frame),
        "passed_cell_count": int(frame["passed"].sum()),
        "passed_cell_rate": float(frame["passed"].mean()),
        "pairwise_ordering_agreement": {
            "mean": float(np.mean(agreement)),
            "median": float(np.median(agreement)),
            "minimum": float(np.min(agreement)),
            "maximum": float(np.max(agreement)),
        },
        "spearman_rho": {
            "mean": float(frame["spearman_rho"].mean()),
            "median": float(frame["spearman_rho"].median()),
        },
        "kendall_tau_b": {
            "mean": float(frame["kendall_tau_b"].mean()),
            "median": float(frame["kendall_tau_b"].median()),
        },
        "exact_rank_vector_rate": float(frame["exact_rank_vector"].mean()),
        "top1_agreement_rate": float(frame["top1_agreement"].mean()),
        "top3_overlap_mean": float(frame["top3_overlap_rate"].mean()),
    }
    return frame, summary


def analyze_split(
    oracle: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    score_column: str,
    bank_size: int,
    minimum_agreement: float,
    equivalence_margins: tuple[float, ...],
    primary_equivalence_margin: float,
    pair_bootstrap_replicates: int,
    pair_ci_level: float,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    split_frame = oracle.merge(
        assignments,
        on=[*PROFILE_KEYS, "paired_group_id"],
        how="inner",
        validate="many_to_one",
    )
    scores = cell_model_scores(
        split_frame,
        score_column=score_column,
        bank_size=bank_size,
    )
    scores_a = scores[scores["bank_id"] == "A"].copy()
    scores_b = scores[scores["bank_id"] == "B"].copy()
    compared = reliability.compare_cell_model_scores(scores_a, scores_b)

    profiles = reliability.capability_profiles(scores)
    profile_a = profiles[profiles["bank_id"] == "A"].copy()
    profile_b = profiles[profiles["bank_id"] == "B"].copy()
    profile_comparison, profile_summary = (
        reliability.compare_capability_profiles(profile_a, profile_b)
    )

    pair_states = tie_aware_pair_states(
        split_frame,
        score_column=score_column,
        bank_size=bank_size,
        equivalence_margins=equivalence_margins,
        bootstrap_replicates=pair_bootstrap_replicates,
        ci_level=pair_ci_level,
        bootstrap_seed=bootstrap_seed,
    )
    pair_a = pair_states[pair_states["bank_id"] == "A"].copy()
    pair_b = pair_states[pair_states["bank_id"] == "B"].copy()
    pair_comparison, pair_summaries = compare_pair_states(
        pair_a,
        pair_b,
    )
    primary_margin_key = f"{primary_equivalence_margin:g}"
    primary_pair_summary = pair_summaries[primary_margin_key]
    partial_ranks = partial_order_ranks(
        pair_states,
        scores,
        primary_margin=primary_equivalence_margin,
    )
    (
        partial_rank_comparison,
        partial_rank_cells,
        partial_rank_summary,
    ) = compare_partial_order_ranks(partial_ranks)
    practical_ranks = practical_tie_ranks(
        scores,
        equivalence_margins=equivalence_margins,
    )
    (
        practical_rank_comparison,
        practical_rank_cells,
        practical_rank_summaries,
    ) = compare_practical_tie_ranks(practical_ranks)
    rank_comparison, rank_summary = rank_reliability(
        scores_a,
        scores_b,
        minimum_agreement=minimum_agreement,
    )
    summary = reliability.summarize_continuous(
        compared,
        profile_summary,
        primary_pair_summary,
    )
    summary["tie_aware_model_contrasts"] = {
        "primary_equivalence_margin": primary_equivalence_margin,
        "primary": primary_pair_summary,
        "by_margin": pair_summaries,
    }
    summary["partial_order_rank_reliability"] = partial_rank_summary
    summary["practical_tie_rank_reliability"] = {
        "primary_equivalence_margin": primary_equivalence_margin,
        "primary": practical_rank_summaries[primary_margin_key],
        "by_margin": practical_rank_summaries,
    }
    summary["formal_rank_reliability"] = rank_summary
    return summary, {
        "cell_model_scores": scores,
        "cell_model_reliability": compared,
        "capability_profiles": profiles,
        "capability_profile_reliability": profile_comparison,
        "tie_aware_pair_states": pair_states,
        "tie_aware_model_contrasts": pair_comparison,
        "partial_order_ranks": partial_ranks,
        "partial_order_rank_reliability": partial_rank_comparison,
        "partial_order_top_tier_reliability": partial_rank_cells,
        "practical_tie_ranks": practical_ranks,
        "practical_tie_rank_reliability": practical_rank_comparison,
        "practical_tie_top_tier_reliability": practical_rank_cells,
        "rank_reliability": rank_comparison,
    }


def flat_summary_row(
    summary: dict[str, Any],
    *,
    bank_size: int,
    split_kind: str,
    repeat_index: int,
    score_policy: str,
) -> dict[str, Any]:
    raw = summary["raw_mase_reliability"]
    normalized = summary["cell_normalized_score_reliability"]
    profile = summary["capability_profile_reliability"]["overall"]
    relative = summary["symmetric_relative_difference"]
    pairs = summary["tie_aware_model_contrasts"]["primary"]
    partial = summary["partial_order_rank_reliability"]
    practical = summary["practical_tie_rank_reliability"]["primary"]
    rank = summary["formal_rank_reliability"]
    return {
        "bank_size": bank_size,
        "split_kind": split_kind,
        "repeat_index": repeat_index,
        "score_policy": score_policy,
        "cell_count": rank["cell_count"],
        "rank_pairwise_agreement_mean": (
            rank["pairwise_ordering_agreement"]["mean"]
        ),
        "rank_pairwise_agreement_median": (
            rank["pairwise_ordering_agreement"]["median"]
        ),
        "rank_pairwise_agreement_minimum": (
            rank["pairwise_ordering_agreement"]["minimum"]
        ),
        "rank_passed_cell_rate": rank["passed_cell_rate"],
        "rank_spearman_mean": rank["spearman_rho"]["mean"],
        "rank_kendall_mean": rank["kendall_tau_b"]["mean"],
        "exact_rank_vector_rate": rank["exact_rank_vector_rate"],
        "top1_agreement_rate": rank["top1_agreement_rate"],
        "top3_overlap_mean": rank["top3_overlap_mean"],
        "raw_mase_lin_ccc": raw["lin_ccc"],
        "raw_mase_spearman": raw["spearman_rho"],
        "normalized_score_lin_ccc": normalized["lin_ccc"],
        "normalized_score_spearman": normalized["spearman_rho"],
        "capability_profile_lin_ccc": profile["lin_ccc"],
        "capability_profile_spearman": profile["spearman_rho"],
        "symmetric_relative_difference_median": relative["median"],
        "symmetric_relative_difference_p90": relative["p90"],
        "tie_state_match_rate": pairs["state_match_rate"],
        "tie_decisive_direction_agreement": (
            pairs["both_decisive_directional_agreement"]
        ),
        "tie_direction_conflict_count": pairs[
            "direction_conflict_count"
        ],
        "tie_conclusion_compatibility_rate": pairs[
            "conclusion_compatibility_rate"
        ],
        "tie_both_equivalent_count": pairs["both_equivalent_count"],
        "tie_both_unresolved_count": pairs["both_unresolved_count"],
        "partial_tier_match_rate": partial["tier_match_rate"],
        "rank_interval_overlap_rate": partial[
            "rank_interval_overlap_rate"
        ],
        "top_tier_exact_match_rate": partial[
            "top_tier_exact_match_rate"
        ],
        "top_tier_jaccard_mean": partial["top_tier_jaccard_mean"],
        "practical_tie_pair_agreement_mean": practical[
            "tie_pair_state_agreement"
        ]["mean"],
        "practical_exact_tie_rank_vector_rate": practical[
            "exact_tie_rank_vector_rate"
        ],
        "practical_tie_tier_match_rate": practical[
            "tie_tier_match_rate"
        ],
        "practical_top_tier_exact_match_rate": practical[
            "top_tier_exact_match_rate"
        ],
        "practical_top_tier_jaccard_mean": practical[
            "top_tier_jaccard_mean"
        ],
        "practical_conclusion_compatibility_rate": practical[
            "conclusion_compatibility_rate"
        ],
        "practical_top_tier_size_mean": practical[
            "top_tier_size_mean"
        ],
        "practical_tier_count_mean": practical["tier_count_mean"],
    }


def distribution(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def random_repeat_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    metrics = [
        "rank_pairwise_agreement_mean",
        "rank_passed_cell_rate",
        "rank_spearman_mean",
        "top1_agreement_rate",
        "top3_overlap_mean",
        "normalized_score_lin_ccc",
        "capability_profile_lin_ccc",
        "symmetric_relative_difference_median",
        "tie_state_match_rate",
        "partial_tier_match_rate",
        "rank_interval_overlap_rate",
        "top_tier_jaccard_mean",
        "practical_tie_pair_agreement_mean",
        "practical_exact_tie_rank_vector_rate",
        "practical_top_tier_jaccard_mean",
        "practical_conclusion_compatibility_rate",
        "practical_top_tier_size_mean",
        "practical_tier_count_mean",
    ]
    result: dict[str, Any] = {}
    random_frame = frame[frame["split_kind"] == "random"]
    for (policy, bank_size), group in random_frame.groupby(
        ["score_policy", "bank_size"],
        sort=True,
    ):
        result.setdefault(str(policy), {})[str(int(bank_size))] = {
            "repeat_count": len(group),
            "metrics": {
                metric: distribution(group[metric])
                for metric in metrics
            },
        }
    return result


def add_split_metadata(
    frame: pd.DataFrame,
    *,
    bank_size: int,
    split_kind: str,
    repeat_index: int,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "repeat_index", repeat_index)
    result.insert(0, "split_kind", split_kind)
    result.insert(0, "bank_size", bank_size)
    return result


def rank_breakdown(
    frame: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    return (
        frame.groupby(
            ["bank_size", "split_kind", "repeat_index", group_column],
            sort=True,
        )
        .agg(
            cell_count=("passed", "size"),
            passed_cell_count=("passed", "sum"),
            passed_cell_rate=("passed", "mean"),
            pairwise_agreement_mean=(
                "pairwise_ordering_agreement",
                "mean",
            ),
            pairwise_agreement_median=(
                "pairwise_ordering_agreement",
                "median",
            ),
            pairwise_agreement_minimum=(
                "pairwise_ordering_agreement",
                "min",
            ),
            spearman_mean=("spearman_rho", "mean"),
            kendall_mean=("kendall_tau_b", "mean"),
            exact_rank_vector_rate=("exact_rank_vector", "mean"),
            top1_agreement_rate=("top1_agreement", "mean"),
            top3_overlap_mean=("top3_overlap_rate", "mean"),
        )
        .reset_index()
    )


def render_report(summary: dict[str, Any], flat: pd.DataFrame) -> str:
    lines = [
        "# E2 paired-group split-bank reliability",
        "",
        "每个 dataset/task/capability 的 paired-group pool 被直接切成两个"
        "不相交 bank；round 不作为统计层级。Ordered split 按 paired_group_id "
        "确定性排序后取前 N 与后 N。",
        "",
        f"排名 cell 通过阈值：`{summary['minimum_pairwise_agreement']:.2f}`。",
        f"Practical-equivalence 主界值：相对 MASE "
        f"`±{summary['primary_equivalence_margin']:.1%}`；"
        f"paired bootstrap {summary['pair_bootstrap_replicates']} 次，"
        f"CI={summary['pair_ci_level']:.0%}。",
        "",
    ]
    ordered = flat[flat["split_kind"] == "ordered"]
    for policy, title in (
        ("oracle_context", "Oracle context"),
        ("fixed_l504", "固定 L=504"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| N | Point rank | Tie-state exact | No contradiction | "
                "Top-tier Jaccard | Mean top-tier size |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected = ordered[ordered["score_policy"] == policy].sort_values(
            "bank_size"
        )
        for row in selected.to_dict(orient="records"):
            lines.append(
                f"| {int(row['bank_size'])} | "
                f"{row['rank_pairwise_agreement_mean']:.4f} | "
                f"{row['practical_tie_pair_agreement_mean']:.4f} | "
                f"{row['practical_conclusion_compatibility_rate']:.4f} | "
                f"{row['practical_top_tier_jaccard_mean']:.4f} | "
                f"{row['practical_top_tier_size_mean']:.2f} |"
            )
        lines.append("")
        if policy == "oracle_context":
            result = summary["ordered_split"][policy][
                str(int(selected["bank_size"].max()))
            ]["practical_tie_rank_reliability"]["by_margin"]
            lines.extend(
                [
                    "| Margin | Exact pair state | No contradiction | "
                    "Exact tie vector | Mean top-tier size |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for margin, margin_result in sorted(
                result.items(),
                key=lambda item: float(item[0]),
            ):
                lines.append(
                    f"| {float(margin):.1%} | "
                    f"{margin_result['tie_pair_state_agreement']['mean']:.4f} | "
                    f"{margin_result['conclusion_compatibility_rate']:.4f} | "
                    f"{margin_result['exact_tie_rank_vector_rate']:.4f} | "
                    f"{margin_result['top_tier_size_mean']:.2f} |"
                )
            lines.append("")
    if summary["random_repeats"] > 0:
        lines.extend(
            [
                "## Repeated random split",
                "",
                f"每个 N 额外执行 {summary['random_repeats']} 次固定种子的"
                "不相交随机二分；完整分布见 `summary.json` 和 "
                "`split_comparison_summary.csv`。",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Split-half reliability 可用于估计当前生成分布下达到稳定测量所需的"
            "样本数，但两个 bank 来自同一个冻结 pool，因此不能替代独立重新生成"
            "的 external seed-bank replication。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    e2_dir: Path,
    output_dir: Path,
    *,
    bank_sizes: list[int],
    models: list[str],
    datasets: set[str] | None,
    random_repeats: int,
    split_seed: int,
    minimum_agreement: float,
    equivalence_margins: list[float],
    primary_equivalence_margin: float,
    pair_bootstrap_replicates: int,
    pair_ci_level: float,
) -> dict[str, Any]:
    e2_dir = e2_dir.resolve()
    output_dir = output_dir.resolve()
    if random_repeats < 0:
        raise ValueError("random repeats cannot be negative")
    if not 0.0 <= minimum_agreement <= 1.0:
        raise ValueError("minimum agreement must be between zero and one")
    margins = tuple(sorted(set(float(value) for value in equivalence_margins)))
    if not margins or any(value <= 0.0 or value >= 1.0 for value in margins):
        raise ValueError("equivalence margins must be between zero and one")
    if not any(
        np.isclose(primary_equivalence_margin, value)
        for value in margins
    ):
        raise ValueError("primary equivalence margin must be in margins")
    if pair_bootstrap_replicates < 100:
        raise ValueError("pair bootstrap requires at least 100 replicates")
    if not 0.50 < pair_ci_level < 1.0:
        raise ValueError("pair CI level must be between 0.50 and 1.0")
    sizes = sorted(set(int(value) for value in bank_sizes))
    if not sizes:
        raise ValueError("at least one bank size is required")

    oracle_paths = ensure_oracle_paths(e2_dir, models)
    oracle = load_oracle_pool(oracle_paths, datasets=datasets)
    observed_models = set(oracle["model_id"])
    missing_models = sorted(set(models) - observed_models)
    if missing_models:
        raise ValueError(
            "selected oracle pool is missing models: "
            + ", ".join(missing_models)
        )
    catalog = pool_catalog(oracle)
    profile_sizes = catalog.groupby(PROFILE_KEYS, sort=False).size()
    minimum_pool_size = int(profile_sizes.min())
    formal_v7_partition = (
        validate_formal_v7_pool(catalog)
        if FORMAL_V7_BANK_SIZE in sizes
        else None
    )
    if 2 * max(sizes) > minimum_pool_size:
        raise ValueError(
            f"largest bank size {max(sizes)} exceeds half the minimum "
            f"profile pool size {minimum_pool_size}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict[str, Any]] = []
    ordered_summaries: dict[str, dict[str, Any]] = {
        policy: {} for policy in SCORE_POLICIES
    }
    ordered_frames: dict[str, dict[str, list[pd.DataFrame]]] = {
        policy: {} for policy in SCORE_POLICIES
    }
    split_specs = [("ordered", 0), *(
        ("random", repeat_index)
        for repeat_index in range(1, random_repeats + 1)
    )]
    for split_kind, repeat_index in split_specs:
        for bank_size in sizes:
            assignments = split_assignments(
                catalog,
                bank_size=bank_size,
                split_kind=split_kind,
                repeat_index=repeat_index,
                split_seed=split_seed,
            )
            for score_policy, score_column in SCORE_POLICIES.items():
                policy_summary, frames = analyze_split(
                    oracle,
                    assignments,
                    score_column=score_column,
                    bank_size=bank_size,
                    minimum_agreement=minimum_agreement,
                    equivalence_margins=margins,
                    primary_equivalence_margin=(
                        primary_equivalence_margin
                    ),
                    pair_bootstrap_replicates=(
                        pair_bootstrap_replicates
                    ),
                    pair_ci_level=pair_ci_level,
                    bootstrap_seed=stable_bootstrap_seed(
                        split_seed,
                        (
                            split_kind,
                            repeat_index,
                            bank_size,
                            score_policy,
                        ),
                    ),
                )
                flat_rows.append(
                    flat_summary_row(
                        policy_summary,
                        bank_size=bank_size,
                        split_kind=split_kind,
                        repeat_index=repeat_index,
                        score_policy=score_policy,
                    )
                )
                if split_kind == "ordered":
                    ordered_summaries[score_policy][str(bank_size)] = (
                        policy_summary
                    )
                    for name, frame in frames.items():
                        ordered_frames[score_policy].setdefault(
                            name,
                            [],
                        ).append(
                            add_split_metadata(
                                frame,
                                bank_size=bank_size,
                                split_kind=split_kind,
                                repeat_index=repeat_index,
                            )
                        )

    flat = pd.DataFrame(flat_rows).sort_values(
        ["split_kind", "repeat_index", "bank_size", "score_policy"],
        kind="stable",
    )
    flat.to_csv(output_dir / "split_comparison_summary.csv", index=False)
    table_rows = {"split_comparison_summary.csv": len(flat)}
    for policy, frame_groups in ordered_frames.items():
        for name, frames in frame_groups.items():
            filename = f"ordered_{name}_{policy}.csv"
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(output_dir / filename, index=False)
            table_rows[filename] = len(combined)
            if name == "rank_reliability":
                for group_column in (
                    "capability_id",
                    "dataset_id",
                    "intensity",
                ):
                    breakdown_name = (
                        "ordered_rank_reliability_by_"
                        f"{group_column.removesuffix('_id')}_{policy}.csv"
                    )
                    breakdown = rank_breakdown(
                        combined,
                        group_column=group_column,
                    )
                    breakdown.to_csv(
                        output_dir / breakdown_name,
                        index=False,
                    )
                    table_rows[breakdown_name] = len(breakdown)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "e2_dir": e2.display_path(e2_dir),
        "models": models,
        "datasets": sorted(oracle["dataset_id"].unique()),
        "sample_unit": "paired_group",
        "pool_order": "lexicographic paired_group_id within profile",
        "round_interpretation": (
            "round fields are ignored; the pool is one flat collection of "
            "independent paired groups"
        ),
        "profile_count": len(profile_sizes),
        "cell_count": int(
            oracle[CELL_KEYS].drop_duplicates().shape[0]
        ),
        "profile_pool_size": {
            "minimum": minimum_pool_size,
            "maximum": int(profile_sizes.max()),
            "unique": sorted(int(value) for value in profile_sizes.unique()),
        },
        "bank_sizes": sizes,
        "formal_v7_partition": formal_v7_partition,
        "random_repeats": random_repeats,
        "split_seed": split_seed,
        "minimum_pairwise_agreement": minimum_agreement,
        "equivalence_margins": list(margins),
        "primary_equivalence_margin": primary_equivalence_margin,
        "pair_bootstrap_replicates": pair_bootstrap_replicates,
        "pair_ci_level": pair_ci_level,
        "ordered_split": ordered_summaries,
        "random_split": random_repeat_summary(flat),
        "interpretation": (
            "within-pool split-half measurement reliability; useful for "
            "sample-size selection but weaker than an independently "
            "generated seed-bank replication"
        ),
        "inputs": {
            "inference_config": {
                "path": e2.display_path(e2_dir / "inference_config.json"),
                "sha256": file_sha256(
                    e2_dir / "inference_config.json"
                ),
            },
            "oracle_scores": [
                {
                    "path": e2.display_path(path),
                    "sha256": file_sha256(path),
                }
                for path in oracle_paths
            ],
        },
        "table_rows": table_rows,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        render_report(summary, flat),
        encoding="utf-8",
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "paper_e2_split_bank_manifest.v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis_path": str(Path(__file__).relative_to(REPO_ROOT)),
            "analysis_sha256": file_sha256(Path(__file__)),
            "files": {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    return summary


def main() -> int:
    args = parse_args()
    e2_dir = args.e2_dir.resolve()
    output_dir = (
        (e2_dir / "split_bank_reliability")
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    models = requested_models(
        e2_dir,
        None if args.models is None else [str(value) for value in args.models],
    )
    summary = analyze(
        e2_dir,
        output_dir,
        bank_sizes=[int(value) for value in args.bank_sizes],
        models=models,
        datasets=(
            None
            if args.datasets is None
            else {str(value) for value in args.datasets}
        ),
        random_repeats=int(args.random_repeats),
        split_seed=int(args.split_seed),
        minimum_agreement=float(args.minimum_agreement),
        equivalence_margins=[
            float(value) for value in args.equivalence_margins
        ],
        primary_equivalence_margin=float(
            args.primary_equivalence_margin
        ),
        pair_bootstrap_replicates=int(
            args.pair_bootstrap_replicates
        ),
        pair_ci_level=float(args.pair_ci_level),
    )
    oracle_80 = summary["ordered_split"]["oracle_context"].get("80")
    if oracle_80 is None:
        largest = str(max(summary["bank_sizes"]))
        oracle_80 = summary["ordered_split"]["oracle_context"][largest]
    rank = oracle_80["formal_rank_reliability"]
    print(
        "split-bank reliability complete: rank agreement="
        f"{rank['pairwise_ordering_agreement']['mean']:.4f}, "
        f"passed={rank['passed_cell_count']}/{rank['cell_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
