#!/usr/bin/env python3
"""Plot oracle-context capability MASE and mechanism-score heatmaps.

The script consumes the immutable experiment-level stage-4 aggregate. It does
not rescan predictions or recompute scores. The score file is verified against
the stage-4 analysis manifest before any figure is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_EXPERIMENT_ROOT = (
    REPOSITORY_ROOT
    / "runtime"
    / "experiments"
    / "cafe-gifteval20-inputmode-v2-20260804T112326Z"
)
ANALYSIS_RELATIVE = (
    Path("04_analysis")
    / "seed_000000_000064"
    / "capability_scores_oracle_context.json"
)
ANALYSIS_MANIFEST_RELATIVE = (
    Path("04_analysis") / "seed_000000_000064" / "analysis_manifest.json"
)

CAPABILITY_LABELS = {
    "trend": "Trend",
    "multi_seasonal": "Multi-seasonal",
    "time_varying_seasonality": "TV seasonality",
    "regime_switching": "Regime switching",
    "nonlinear_persistence": "Nonlinear persistence",
    "predictable_intermittency": "Intermittency",
    "common_factor": "Common factor",
    "hierarchical_coherence": "Hierarchy",
    "cross_series_dependence": "Cross-series",
    "covariate_response": "Covariate response",
}
MODEL_LABELS = {
    "Timer-4.0": "Timer\n4.0",
    "Chronos-2": "Chronos\n2",
    "timesfm2.5": "TimesFM\n2.5",
    "tirex2": "TiRex\n2",
    "moirai2": "Moirai\n2",
    "Timer-3.5": "Timer\n3.5",
    "toto2.0": "Toto\n2.0",
}
SCORE_FIELDS = {
    "mase": "macro_mean_accuracy_score",
    "mechanism": "macro_mean_mechanism_score",
}
OUTPUT_STEMS = {
    "mase": "oracle_context_mase_heatmap",
    "mechanism": "oracle_context_mechanism_score_heatmap",
}
BEST_TEXT_COLOR = "#C7441C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Completed CaFE experiment root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Figure output directory. The default is "
            "figures/<experiment-id>/oracle_context."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing figure outputs in the selected directory.",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.5,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_scores(
    experiment_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
]:
    experiment_path = experiment_root / "experiment.json"
    analysis_manifest_path = experiment_root / ANALYSIS_MANIFEST_RELATIVE
    score_path = experiment_root / ANALYSIS_RELATIVE
    for path in (experiment_path, analysis_manifest_path, score_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    experiment = read_json(experiment_path)
    analysis_manifest = read_json(analysis_manifest_path)
    score_payload = read_json(score_path)
    experiment_id = str(experiment.get("experiment_id", ""))
    if not experiment_id:
        raise ValueError(f"Missing experiment_id in {experiment_path}")
    if analysis_manifest.get("experiment_id") != experiment_id:
        raise ValueError("Experiment and analysis manifest IDs do not match")
    if "oracle_context" not in analysis_manifest.get("context_policies", []):
        raise ValueError("Analysis manifest does not declare oracle_context")

    declared_score = analysis_manifest.get("files", {}).get("oracle_scores", {})
    if (
        int(declared_score.get("bytes", -1)) != score_path.stat().st_size
        or declared_score.get("sha256") != sha256_file(score_path)
    ):
        raise ValueError("Oracle score file does not match the analysis manifest")

    capabilities = analysis_manifest.get("capabilities")
    models = analysis_manifest.get("models")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or len(capabilities) != len(set(capabilities))
    ):
        raise ValueError("Invalid capability roster in analysis manifest")
    if not isinstance(models, list) or not models or len(models) != len(set(models)):
        raise ValueError("Invalid model roster in analysis manifest")

    rows = score_payload.get("scores")
    if not isinstance(rows, list):
        raise ValueError(f"Missing score list in {score_path}")
    expected_pairs = {
        (str(capability_id), str(model_id))
        for capability_id in capabilities
        for model_id in models
    }
    observed_pairs: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every score row must be an object")
        pair = (str(row.get("capability_id")), str(row.get("model_id")))
        if pair not in expected_pairs:
            raise ValueError(f"Unexpected capability/model pair: {pair}")
        if pair in observed_pairs:
            raise ValueError(f"Duplicate capability/model pair: {pair}")
        observed_pairs.add(pair)
        if row.get("context_policy") != "oracle_context":
            raise ValueError(f"Non-oracle score row: {pair}")
        dataset_ids = row.get("dataset_ids")
        if not isinstance(dataset_ids, list) or len(dataset_ids) != int(
            row.get("dataset_count", -1)
        ):
            raise ValueError(f"Dataset support mismatch: {pair}")
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError(f"Duplicate dataset support: {pair}")
        for field in SCORE_FIELDS.values():
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {field} for {pair}: {value}")
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        raise RuntimeError(f"Incomplete oracle-context score matrix: {missing}")

    return (
        rows,
        analysis_manifest,
        experiment,
        score_path,
        analysis_manifest_path,
        experiment_path,
    )


def score_matrix(
    rows: list[dict[str, Any]],
    capabilities: list[str],
    models: list[str],
    score_key: str,
) -> np.ndarray:
    field = SCORE_FIELDS[score_key]
    capability_indices = {item: index for index, item in enumerate(capabilities)}
    model_indices = {item: index for index, item in enumerate(models)}
    matrix = np.full((len(capabilities), len(models)), np.nan)
    for row in rows:
        i = capability_indices[str(row["capability_id"])]
        j = model_indices[str(row["model_id"])]
        matrix[i, j] = float(row[field])
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"Incomplete matrix for {field}")
    return matrix


def soft_heatmap_cmap() -> mpl.colors.ListedColormap:
    colors = mpl.colormaps["YlGnBu"](np.linspace(0.50, 0.10, 256))
    return mpl.colors.ListedColormap(colors)


def score_norm(matrix: np.ndarray) -> mpl.colors.Normalize:
    return mpl.colors.Normalize(
        vmin=float(np.quantile(matrix, 0.02)),
        vmax=float(np.quantile(matrix, 0.98)),
        clip=True,
    )


def display_label(identifier: str, labels: dict[str, str]) -> str:
    return labels.get(identifier, identifier.replace("_", " ").title())


def plot_heatmap(
    matrix: np.ndarray,
    *,
    capabilities: list[str],
    models: list[str],
    title: str,
    colorbar_label: str,
    footer: str,
) -> plt.Figure:
    norm = score_norm(matrix)
    fig, ax = plt.subplots(figsize=(6.0, 3.75))
    image = ax.imshow(
        matrix,
        cmap=soft_heatmap_cmap(),
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title(title, pad=9, fontweight="bold")
    ax.set_xticks(
        np.arange(len(models)),
        [display_label(item, MODEL_LABELS) for item in models],
        ha="center",
    )
    ax.set_yticks(
        np.arange(len(capabilities)),
        [display_label(item, CAPABILITY_LABELS) for item in capabilities],
    )
    ax.set_xlabel("Model", labelpad=4, fontweight="bold")
    ax.tick_params(axis="x", pad=5)

    best_columns = np.argmin(matrix, axis=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            is_best = j == best_columns[i]
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.4f}",
                ha="center",
                va="center",
                fontsize=6.3,
                color=BEST_TEXT_COLOR if is_best else "#202124",
                fontweight="bold" if is_best else "normal",
                zorder=4,
            )
            if is_best:
                ax.add_patch(
                    Rectangle(
                        (j - 0.46, i - 0.46),
                        0.92,
                        0.92,
                        facecolor="none",
                        edgecolor=BEST_TEXT_COLOR,
                        linewidth=0.65,
                        zorder=3,
                    )
                )

    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(capabilities), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.033, pad=0.022, extend="both")
    colorbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 3))
    colorbar.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=6.2, length=2.0, pad=1.5)
    colorbar.set_label(colorbar_label, fontsize=7.2, labelpad=3.0)

    fig.subplots_adjust(left=0.235, right=0.94, top=0.88, bottom=0.19)
    position = ax.get_position()
    fig.text(
        position.x0 - 0.018,
        position.y1 + 0.030,
        "Capability",
        ha="right",
        va="center",
        fontsize=8.4,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        footer,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    return fig


def ensure_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite to replace:\n"
            f"{formatted}"
        )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(
        paths[0],
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
        metadata={"Software": "CaFE oracle capability heatmap script"},
    )
    fig.savefig(
        paths[1],
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return paths


def write_score_csv(
    path: Path,
    rows: list[dict[str, Any]],
    capabilities: list[str],
    models: list[str],
) -> None:
    capability_indices = {item: index for index, item in enumerate(capabilities)}
    model_indices = {item: index for index, item in enumerate(models)}
    fields = [
        "context_policy",
        "capability_id",
        "model_id",
        "macro_mean_accuracy_score",
        "macro_mean_mechanism_score",
        "dataset_count",
        "dataset_ids",
    ]
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            capability_indices[str(row["capability_id"])],
            model_indices[str(row["model_id"])],
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    **{field: row[field] for field in fields[:-1]},
                    "dataset_ids": ";".join(str(item) for item in row["dataset_ids"]),
                }
            )


def row_winners(
    matrix: np.ndarray, capabilities: list[str], models: list[str]
) -> dict[str, str]:
    return {
        capability: models[int(np.argmin(matrix[index]))]
        for index, capability in enumerate(capabilities)
    }


def write_manifest(
    path: Path,
    *,
    experiment_id: str,
    analysis_manifest: dict[str, Any],
    capabilities: list[str],
    models: list[str],
    score_path: Path,
    analysis_manifest_path: Path,
    experiment_path: Path,
    csv_path: Path,
    figure_paths: list[Path],
    mase_matrix: np.ndarray,
    mechanism_matrix: np.ndarray,
) -> None:
    payload = {
        "schema_version": "cafe.oracle_capability_heatmap_manifest.v1",
        "experiment_id": experiment_id,
        "context_policy": "oracle_context",
        "aggregation_policy": analysis_manifest.get("aggregation_policy"),
        "oracle_selection_policy": analysis_manifest.get("oracle_selection_policy"),
        "capabilities": capabilities,
        "models": models,
        "score_fields": SCORE_FIELDS,
        "best_value_rule": (
            "Minimum full-precision score within each capability row; lower is better."
        ),
        "color_scale": (
            "Shared across all cells within each figure and clipped at the 2nd and "
            "98th percentiles; annotations show unclipped scores."
        ),
        "mase_row_winners": row_winners(mase_matrix, capabilities, models),
        "mechanism_row_winners": row_winners(
            mechanism_matrix, capabilities, models
        ),
        "source_files": [
            artifact_record(score_path),
            artifact_record(analysis_manifest_path),
            artifact_record(experiment_path),
        ],
        "producer_script": artifact_record(SCRIPT_PATH),
        "data_outputs": [artifact_record(csv_path)],
        "figure_outputs": [artifact_record(item) for item in figure_paths],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    (
        rows,
        analysis_manifest,
        experiment,
        score_path,
        analysis_manifest_path,
        experiment_path,
    ) = load_scores(experiment_root)
    experiment_id = str(experiment["experiment_id"])
    capabilities = [str(item) for item in analysis_manifest["capabilities"]]
    models = [str(item) for item in analysis_manifest["models"]]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPOSITORY_ROOT / "figures" / experiment_id / "oracle_context"
    )

    stems = list(OUTPUT_STEMS.values())
    figure_targets = [
        output_dir / f"{stem}.{extension}"
        for stem in stems
        for extension in ("png", "pdf")
    ]
    csv_path = output_dir / "oracle_context_capability_scores.csv"
    manifest_path = output_dir / "figure_manifest.json"
    ensure_outputs_available(
        [*figure_targets, csv_path, manifest_path], args.overwrite
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mase_matrix = score_matrix(rows, capabilities, models, "mase")
    mechanism_matrix = score_matrix(rows, capabilities, models, "mechanism")
    configure_style()
    figure_paths: list[Path] = []
    figure_paths.extend(
        save_figure(
            plot_heatmap(
                mase_matrix,
                capabilities=capabilities,
                models=models,
                title="Forecast accuracy under Oracle Context",
                colorbar_label="MASE (lower is better)",
                footer=(
                    "Orange: row minimum; annotations are exact; colors are clipped "
                    "at the 2nd–98th percentiles."
                ),
            ),
            output_dir,
            OUTPUT_STEMS["mase"],
        )
    )
    figure_paths.extend(
        save_figure(
            plot_heatmap(
                mechanism_matrix,
                capabilities=capabilities,
                models=models,
                title="Mechanism score under Oracle Context",
                colorbar_label="Mechanism score (lower is better)",
                footer=(
                    "Context chosen by MASE; orange: row minimum; annotations are exact; "
                    "colors are clipped at the 2nd–98th percentiles."
                ),
            ),
            output_dir,
            OUTPUT_STEMS["mechanism"],
        )
    )
    write_score_csv(csv_path, rows, capabilities, models)
    write_manifest(
        manifest_path,
        experiment_id=experiment_id,
        analysis_manifest=analysis_manifest,
        capabilities=capabilities,
        models=models,
        score_path=score_path,
        analysis_manifest_path=analysis_manifest_path,
        experiment_path=experiment_path,
        csv_path=csv_path,
        figure_paths=figure_paths,
        mase_matrix=mase_matrix,
        mechanism_matrix=mechanism_matrix,
    )

    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "context_policy": "oracle_context",
                "score_rows": len(rows),
                "mase_row_winners": row_winners(
                    mase_matrix, capabilities, models
                ),
                "mechanism_row_winners": row_winners(
                    mechanism_matrix, capabilities, models
                ),
                "outputs": [
                    str(path)
                    for path in [*figure_paths, csv_path, manifest_path]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
