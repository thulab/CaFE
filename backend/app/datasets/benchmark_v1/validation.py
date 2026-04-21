from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .baselines import baseline_by_name
from .constants import FEATURE_COLUMNS
from .metrics import mase
from .utils import adjacent_meta_path, read_json


def _vectorize_feature_map(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(series.tolist(), index=series.index)


def _series_to_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float)


def validate_benchmark(benchmark_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(benchmark_path)
    meta = read_json(adjacent_meta_path(benchmark_path))
    anchor_meta = meta["anchor_meta"]
    means = np.asarray(anchor_meta["feature_means"], dtype=float)
    stds = np.asarray(anchor_meta["feature_stds"], dtype=float)
    prototypes = {int(item["anchor_cluster_id"]): item for item in anchor_meta["prototypes"]}

    feature_frame = _vectorize_feature_map(frame["realized_features"])
    drifts = []
    for idx, row in frame.iterrows():
        cluster = int(row["anchor_cluster_id"])
        proto = prototypes[cluster]
        realized = np.array([feature_frame.loc[idx, name] for name in FEATURE_COLUMNS], dtype=float)
        target = np.array([proto[name] for name in FEATURE_COLUMNS], dtype=float)
        z = (realized - means) / stds
        z_target = (target - means) / stds
        drifts.append(float(np.median(np.abs(z - z_target))))
    frame = frame.copy()
    frame["non_target_drift"] = drifts

    difficulty_summary = []
    for family, family_frame in frame[frame["track"] == "diagnostic"].groupby("family"):
        objective = {
            "trend": "trend_strength",
            "multi_seasonal": "seasonal_strength",
            "regime_switching": "changepoint_density",
            "long_memory_nonlinear": "acf_half_life",
            "intermittent_heteroskedastic": "intermittency",
        }[family]
        grouped = family_frame.assign(
            objective_value=[features[objective] for features in family_frame["realized_features"]]
        ).groupby("difficulty")["objective_value"].mean()
        values = grouped.to_numpy(dtype=float)
        corr = 1.0 if len(np.unique(np.round(values, 8))) <= 1 else spearmanr(grouped.index.to_numpy(dtype=float), values).statistic
        difficulty_summary.append({"family": family, "objective": objective, "spearman": float(corr)})

    behavior_rows = []
    selectors = {
        "seasonal_easy": lambda row: row["family"] == "multi_seasonal" and row["difficulty"] == 1,
        "random_walk_like": lambda row: row["family"] == "trend" and row["difficulty"] == 1,
        "high_entropy": lambda row: row["realized_features"]["spectral_entropy"] >= 0.95,
    }
    for behavior_name, selector in selectors.items():
        subset = frame[[selector(row) for _, row in frame.iterrows()]]
        if subset.empty:
            continue
        scores = {}
        for predictor_name in ["last_value", "seasonal_naive", "auto_theta"]:
            predictor = baseline_by_name(predictor_name)
            local = []
            for row in subset.itertuples(index=False):
                history = _series_to_array(row.context)
                target = _series_to_array(row.target)
                forecast = predictor.predict(history, len(target), int(row.season_length))
                local.append(mase(history, target, forecast, int(row.season_length)))
            scores[predictor_name] = float(np.mean(local))
        behavior_rows.append({"scenario": behavior_name, **scores})

    return {
        "n_series": int(len(frame)),
        "anchor_mode": str(meta.get("anchor_mode", anchor_meta.get("anchor_mode", "unknown"))),
        "median_non_target_drift": float(frame["non_target_drift"].median()),
        "difficulty_objective_spearman": difficulty_summary,
        "behavior_baselines": behavior_rows,
    }


def validate_external_alignment(
    eval_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    real_eval_path: Path | None,
) -> dict[str, object] | None:
    if real_eval_path is None or not real_eval_path.exists():
        return None
    real_frame = pd.read_parquet(real_eval_path)
    synth = eval_frame.merge(benchmark_frame[["id", "track"]], left_on="series_id", right_on="id")
    synth = synth[synth["track"] == "anchor"]
    synth_rank = synth.groupby("model")["mase"].mean().sort_values()
    real_rank = real_frame.groupby("model")["mase"].mean().sort_values()
    common = [name for name in synth_rank.index if name in real_rank.index]
    pair_total = 0
    pair_match = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            pair_total += 1
            pair_match += int((synth_rank[a] <= synth_rank[b]) == (real_rank[a] <= real_rank[b]))
    return {"pair_match": pair_match, "pair_total": pair_total}

