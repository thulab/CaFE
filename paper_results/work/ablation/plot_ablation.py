from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm


PANEL_ORDER = (
    ("gift", "short", "GIFT-Short"),
    ("gift", "medium", "GIFT-Medium"),
    ("gift", "long", "GIFT-Long"),
    ("fev", "native", "FEV-Mini20"),
)
CAPABILITIES = (
    "common_factor",
    "covariate_impulse_response",
    "cross_series_dependence",
)
CAPABILITY_LABELS = {
    "common_factor": "Common\nfactor",
    "covariate_impulse_response": "Covariate\nimpulse",
    "cross_series_dependence": "Cross-\nseries",
}
MODEL_ORDER = (
    "Chronos-2",
    "timesfm2.5",
    "Timer-3.5",
    "tirex2",
    "moirai2",
    "toto2.0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--output-data", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in csv.DictReader(args.input.open(encoding="utf-8"))
        if row["model_id"] in MODEL_ORDER
    ]
    if args.output_data is not None:
        args.output_data.parent.mkdir(parents=True, exist_ok=True)
        with args.output_data.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    lookup = {
        (
            row["suite"],
            row["term"],
            row["model_id"],
            row["capability_id"],
        ): row
        for row in rows
    }
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), constrained_layout=True)
    norm = SymLogNorm(linthresh=0.01, linscale=1.0, vmin=-3.0, vmax=3.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    image = None
    for axis, (suite, term, title) in zip(axes.flat, PANEL_ORDER, strict=True):
        models = [
            model
            for model in MODEL_ORDER
            if any((suite, term, model, capability) in lookup for capability in CAPABILITIES)
        ]
        matrix = np.full((len(models), len(CAPABILITIES)), np.nan, dtype=float)
        for y_index, model in enumerate(models):
            for x_index, capability in enumerate(CAPABILITIES):
                row = lookup.get((suite, term, model, capability))
                if row is not None:
                    matrix[y_index, x_index] = float(
                        row["task_level_equal_mase_degradation"]
                    )
        image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axis.set_title(title)
        axis.set_xticks(range(len(CAPABILITIES)))
        axis.set_xticklabels([CAPABILITY_LABELS[value] for value in CAPABILITIES])
        axis.set_yticks(range(len(models)))
        axis.set_yticklabels(models)
        axis.tick_params(length=0)
        for y_index, model in enumerate(models):
            for x_index, capability in enumerate(CAPABILITIES):
                row = lookup.get((suite, term, model, capability))
                if row is None:
                    axis.text(x_index, y_index, "—", ha="center", va="center")
                    continue
                value = float(row["task_level_equal_mase_degradation"])
                lower = float(row["task_bootstrap_95_ci_lower"])
                upper = float(row["task_bootstrap_95_ci_upper"])
                marker = "*" if lower > 0.0 or upper < 0.0 else ""
                task_count = int(row["task_count"])
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "white" if luminance < 0.42 else "black"
                axis.text(
                    x_index,
                    y_index,
                    f"{value:+.3f}{marker}\n(n={task_count})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
    if image is None:
        raise RuntimeError("no ablation observations plotted")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.86, pad=0.025)
    colorbar.set_label(
        "Task- and level-equal ΔMASE after input removal\n"
        "(positive: auxiliary input helped; symmetric-log color scale)"
    )
    figure.suptitle(
        "Target-only/relevant-covariate-removal attribution",
        fontsize=15,
    )
    figure.text(
        0.5,
        -0.01,
        "* descriptive 95% task-bootstrap interval excludes zero; no multiple-comparison correction.",
        ha="center",
        fontsize=9,
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
