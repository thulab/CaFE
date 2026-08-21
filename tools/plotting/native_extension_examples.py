from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cafe import core
from cafe.benchmark_extension.generation import iter_replayed_samples
from cafe.benchmark_extension.mechanisms import CAPABILITY_IDS


PLOTTER_SCHEMA = "cafe.native_extension_example_plotter.v4"
REAL_COLOR = "#1565c0"
TREATMENT_COLOR = "#9aa0a6"
DELTA_COLOR = "#d84315"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot five-level examples by replaying validated v9 contracts."
    )
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_roots(roots: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    baselines: dict[str, dict[str, Any]] = {}
    treatments: list[dict[str, Any]] = []
    selected: set[str] = set()
    required = set(CAPABILITY_IDS) - {"hierarchical_coherence"}
    for root in roots:
        validation = core.read_json(root / "02_validation" / "report.json")
        if not validation.get("accepted"):
            raise ValueError(f"generation is not validated: {root}")
        manifest = core.read_json(root / "01_generation" / "manifest.json")
        gift_root = Path(manifest["config"]["gift_eval_source_root"])
        current_baseline: dict[str, Any] | None = None
        current_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in iter_replayed_samples(manifest, gift_eval_dir=gift_root):
            if row["evaluation_table"] == "gift_eval_official_baseline":
                current_baseline = row
                current_groups.clear()
                continue
            if (
                row["evaluation_table"] != "gift_eval_capability_treatment"
                or str(row["capability_id"]) in selected
            ):
                continue
            capability = str(row["capability_id"])
            current_groups[capability].append(row)
            if len(current_groups[capability]) == 5:
                if current_baseline is None:
                    raise ValueError("treatment appeared before its baseline")
                selected.add(capability)
                treatments.extend(current_groups[capability])
                baselines[str(current_baseline["sample_id"])] = current_baseline
            if selected == required:
                break
        if selected == required:
            break
    return baselines, treatments


def _groups(treatments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in treatments:
        grouped[(str(row["capability_id"]), str(row["official_instance_id"]))].append(row)
    selected: dict[str, list[dict[str, Any]]] = {}
    for (capability, _instance), rows in sorted(grouped.items()):
        if capability not in selected and len(rows) == 5:
            selected[capability] = sorted(rows, key=lambda row: row["capability_level"])
    return selected


def _plot_group(
    capability: str,
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    baseline_target = np.asarray(baseline["target"], dtype=float)
    context = int(baseline["context_length"])
    horizon = int(baseline["horizon"])
    affected = [int(value) for value in rows[0]["affected_target_indices"]]
    channel = affected[0]
    start = 0
    x = np.arange(start - context, horizon)
    source = baseline_target[start:, channel]
    treatment_values = [np.asarray(row["target"], dtype=float)[start:, channel] for row in rows]
    all_values = np.concatenate((source, *treatment_values))
    margin = max(1e-6, 0.05 * float(np.ptp(all_values)))
    y_limits = (float(np.min(all_values) - margin), float(np.max(all_values) + margin))
    delta_values = [values - source for values in treatment_values]
    displayed_deltas = delta_values
    delta_limit = max(
        max(float(np.max(np.abs(delta))) for delta in displayed_deltas),
        1e-6,
    )
    figure, axes = plt.subplots(2, 5, figsize=(16, 5.4), sharex=True)
    for index, (row, values, delta, displayed_delta) in enumerate(
        zip(rows, treatment_values, delta_values, displayed_deltas, strict=True)
    ):
        axis = axes[0, index]
        axis.plot(
            x,
            source,
            color=REAL_COLOR,
            linewidth=1.2,
            label="real / authentic source",
            zorder=3,
        )
        axis.plot(
            x,
            values,
            color=TREATMENT_COLOR,
            linewidth=1.0,
            label="modified treatment",
            zorder=2,
        )
        axis.axvline(0, color="#263238", linewidth=0.8)
        axis.axvspan(0, horizon - 1, color="#ffecb3", alpha=0.35)
        axis.set_ylim(*y_limits)
        axis.set_title(
            f"level {row['capability_level']}\n"
            f"{row['controlled_coordinate']}={row['sampled_coordinate']:.3f}",
            fontsize=8,
        )
        bottom = axes[1, index]
        bottom.plot(x, displayed_delta, color=DELTA_COLOR, linewidth=1.2)
        bottom.axhline(0, color="#9e9e9e", linewidth=0.7)
        bottom.axvline(0, color="#263238", linewidth=0.8)
        bottom.axvspan(0, horizon - 1, color="#ffecb3", alpha=0.35)
        bottom.set_ylim(-1.05 * delta_limit, 1.05 * delta_limit)
        bottom.set_xlabel("time relative to forecast origin")
        if capability == "trend":
            slope = float(np.mean(np.diff(delta[: context - start])))
            bottom.text(
                0.04,
                0.90,
                f"slope={slope:.4g}/step",
                transform=bottom.transAxes,
                fontsize=7,
                va="top",
            )
    axes[0, 0].set_ylabel("authentic units")
    axes[1, 0].set_ylabel(
        "relative trend delta" if capability == "trend" else "treatment − source"
    )
    axes[0, 0].legend(loc="upper left", fontsize=7)
    figure.suptitle(
        f"{capability}: five treatments on one official {baseline['dataset_id']} instance",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    file_record = core.file_record(output_path)
    file_record["path"] = output_path.name
    return {
        "capability_id": capability,
        "status": "generated_example",
        "dataset_id": baseline["dataset_id"],
        "official_instance_id": baseline["official_instance_id"],
        "baseline_sample_id": baseline["sample_id"],
        "affected_target_index_shown": channel,
        "display_history": context,
        "delta_display": "treatment_minus_source",
        "full_treatment_history": context,
        "horizon": horizon,
        "augmentation_seed": rows[0]["augmentation_seed"],
        "levels": [
            {
                "level": row["capability_level"],
                "coordinate_interval": row["coordinate_interval"],
                "sampled_coordinate": row["sampled_coordinate"],
                "source_distance": row["source_distance_gate"][
                    "minimum_observed_macro_distance"
                ],
                "future_effect_mase_rms": row["mechanism_scoring_gate"][
                    "truth_effect_mase_rms"
                ],
                "mechanism_scoreable": row["mechanism_scoring_gate"][
                    "accepted"
                ],
                "horizon_support_gate": row.get("horizon_support_gate"),
            }
            for row in rows
        ],
        "file": file_record,
    }


def _plot_hierarchy(output_path: Path) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(12, 3.4))
    axis.axis("off")
    axis.text(
        0.5,
        0.62,
        "Hierarchical coherence",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
    )
    axis.text(
        0.5,
        0.36,
        "Qualification-only in the v9 GIFT-Eval adapter\n"
        "No explicit summing matrix is inferred from separate univariate records,\n"
        "so no five-level treatment or formal rank is emitted.",
        ha="center",
        va="center",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    file_record = core.file_record(output_path)
    file_record["path"] = output_path.name
    return {
        "capability_id": "hierarchical_coherence",
        "status": "qualification_only_no_generation",
        "file": file_record,
    }


def main() -> int:
    args = parse_args()
    baselines, treatments = _load_roots([path.resolve() for path in args.dataset_root])
    selected = _groups(treatments)
    names = {
        capability: f"{index:02d}_{capability}__five_levels.png"
        for index, capability in enumerate(CAPABILITY_IDS, start=1)
    }
    records: list[dict[str, Any]] = []
    for capability in CAPABILITY_IDS:
        output_path = args.output_dir.resolve() / names[capability]
        if capability == "hierarchical_coherence":
            records.append(_plot_hierarchy(output_path))
            continue
        rows = selected.get(capability)
        if rows is None:
            raise ValueError(f"no validated five-level example found for {capability}")
        records.append(
            _plot_group(
                capability,
                rows,
                baselines[str(rows[0]["baseline_sample_id"])],
                output_path,
            )
        )
    manifest = {
        "schema_version": PLOTTER_SCHEMA,
        "created_at": core.utc_now(),
        "selection_policy": (
            "lexicographically_first_validated_official_instance_per_capability"
        ),
        "figure_semantics": (
            "benchmark_truth_paths_not_model_predictions_complete_official_"
            "instance_history_and_future"
        ),
        "color_semantics": {
            "real_authentic_source": REAL_COLOR,
            "modified_treatment": TREATMENT_COLOR,
            "treatment_minus_source": DELTA_COLOR,
        },
        "records": records,
    }
    core.write_json(args.output_dir.resolve() / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
