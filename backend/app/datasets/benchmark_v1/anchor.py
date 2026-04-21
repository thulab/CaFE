from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS
from .domain import BenchmarkV1Config
from .features import extract_features, infer_dominant_scale, infer_season_length
from .loaders import bootstrap_anchor_corpus, scan_real_corpora
from .utils import adjacent_meta_path, read_json, write_json, write_parquet


@dataclass(slots=True)
class AnchorArtifacts:
    feature_columns: list[str]
    feature_means: list[float]
    feature_stds: list[float]
    covariance: list[list[float]]
    prototypes: list[dict[str, float]]
    cluster_weights: list[float]
    anchor_mode: str


def _assign_medoids(data: np.ndarray, medoid_indices: list[int]) -> np.ndarray:
    medoids = data[medoid_indices]
    distances = ((data[:, None, :] - medoids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def _fit_k_medoids(data: np.ndarray, n_clusters: int, seed: int, n_iter: int = 12) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    n_clusters = min(n_clusters, len(data))
    medoid_indices = rng.choice(len(data), size=n_clusters, replace=False).tolist()
    for _ in range(n_iter):
        labels = _assign_medoids(data, medoid_indices)
        changed = False
        for cluster in range(n_clusters):
            members = np.where(labels == cluster)[0]
            if len(members) == 0:
                continue
            pairwise = ((data[members, None, :] - data[None, members, :]) ** 2).sum(axis=2)
            best_member = members[int(np.argmin(pairwise.sum(axis=1)))]
            if medoid_indices[cluster] != int(best_member):
                medoid_indices[cluster] = int(best_member)
                changed = True
        if not changed:
            break
    return _assign_medoids(data, medoid_indices), medoid_indices


def build_anchor_stats_artifacts(
    output_path: Path,
    gift_root: Path | None,
    tfb_root: Path | None,
    n_clusters: int,
    bootstrap_size: int,
    seed: int,
) -> Path:
    config = BenchmarkV1Config(
        artifact_root=output_path.parent,
        n_anchor_clusters=n_clusters,
        bootstrap_corpus_size=bootstrap_size,
        random_seed=seed,
    )
    loaded = scan_real_corpora(gift_root=gift_root, tfb_root=tfb_root)
    anchor_mode = "real" if loaded else "bootstrap"
    if not loaded:
        loaded = bootstrap_anchor_corpus(size=config.bootstrap_corpus_size, seed=config.random_seed)
    rows: list[dict[str, float | str | int]] = []
    for item in loaded:
        values = np.asarray(item.values, dtype=float)
        features = extract_features(values)
        rows.append(
            {
                "source": item.source,
                "path": item.path,
                "length": int(len(values)),
                "season_length": int(infer_season_length(values)),
                "dominant_scale": int(infer_dominant_scale(values)),
                **features,
            }
        )
    frame = pd.DataFrame(rows)
    feature_frame = frame[FEATURE_COLUMNS].astype(float)
    means = feature_frame.mean(axis=0)
    stds = feature_frame.std(axis=0).replace(0.0, 1.0)
    standardized = (feature_frame - means) / stds
    labels, medoids = _fit_k_medoids(standardized.to_numpy(), n_clusters=config.n_anchor_clusters, seed=config.random_seed)
    frame["anchor_cluster_id"] = labels.astype(int)
    frame["anchor_mode"] = anchor_mode
    cov = np.cov(standardized.to_numpy(), rowvar=False).tolist()
    prototypes: list[dict[str, float]] = []
    cluster_weights: list[float] = []
    for cluster_id, medoid_index in enumerate(medoids):
        row = frame.iloc[medoid_index]
        prototypes.append(
            {
                "anchor_cluster_id": int(cluster_id),
                "path": str(row["path"]),
                "source": str(row["source"]),
                "season_length": int(row["season_length"]),
                "dominant_scale": int(row["dominant_scale"]),
                **{name: float(row[name]) for name in FEATURE_COLUMNS},
            }
        )
        cluster_weights.append(float((labels == cluster_id).mean()))
    write_parquet(frame, output_path)
    meta = AnchorArtifacts(
        feature_columns=list(FEATURE_COLUMNS),
        feature_means=[float(means[name]) for name in FEATURE_COLUMNS],
        feature_stds=[float(stds[name]) for name in FEATURE_COLUMNS],
        covariance=cov,
        prototypes=prototypes,
        cluster_weights=cluster_weights,
        anchor_mode=anchor_mode,
    )
    write_json(asdict(meta), adjacent_meta_path(output_path))
    return output_path


def load_anchor_artifacts(anchor_stats_path: Path) -> tuple[pd.DataFrame, AnchorArtifacts]:
    frame = pd.read_parquet(anchor_stats_path)
    meta = AnchorArtifacts(**read_json(adjacent_meta_path(anchor_stats_path)))
    return frame, meta

