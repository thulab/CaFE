from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cafe import core
from cafe.benchmark_extension.generation import iter_replayed_samples
from cafe.benchmark_extension.mechanisms import (
    CAPABILITY_IDS,
    _nonlinear_innovation_bootstrap_paths,
    _nonlinear_state_response,
    _scale_by_target,
)


PLOTTER_SCHEMA = "cafe.native_extension_example_plotter.v7"
REAL_COLOR = "#1565c0"
TREATMENT_COLOR = "#9aa0a6"
DELTA_COLOR = "#d84315"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot five-level examples by replaying validated contracts."
    )
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=CAPABILITY_IDS,
        default=list(CAPABILITY_IDS),
        help="Capabilities to redraw; omitted capabilities remain in the manifest.",
    )
    return parser.parse_args()


def _load_roots(
    roots: list[Path],
    required: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    baselines: dict[str, dict[str, Any]] = {}
    treatments: list[dict[str, Any]] = []
    selected: set[str] = set()
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


def _nonlinear_future_path_band(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    channel: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    baseline_target = np.asarray(baseline["target"], dtype=float)
    context = int(baseline["context_length"])
    horizon = int(baseline["horizon"])
    history = baseline_target[:context, channel]
    scale = float(_scale_by_target(history[:, None])[0])
    source = (history - float(np.mean(history))) / scale
    metadata = rows[0]["mechanism_metadata"]
    audit = metadata["identifiability_by_target"][str(channel)]
    intercept = float(audit["linear_intercept"])
    persistence = float(audit["linear_persistence_coefficient"])
    innovations = source[1:] - (
        intercept + persistence * source[:-1]
    )
    bootstrap = metadata["future_innovation_bootstrap_by_target"][str(channel)]
    paths, _ = _nonlinear_innovation_bootstrap_paths(
        innovations,
        horizon=horizon,
        path_count=int(bootstrap["path_count"]),
        seed=int(bootstrap["seed"]),
    )
    coefficients = np.asarray(
        [
            row["mechanism_metadata"]["level_diagnostics_by_target"][
                str(channel)
            ]["nonlinear_persistence_coefficient"]
            for row in rows
        ],
        dtype=float,
    )
    treated_last = np.asarray(
        [
            (
                np.asarray(row["target"], dtype=float)[context - 1, channel]
                - float(np.mean(history))
            )
            / scale
            for row in rows
        ],
        dtype=float,
    )
    linear_state = np.full(paths.shape[0], source[-1], dtype=float)
    nonlinear_state = np.broadcast_to(
        treated_last[:, None], (len(rows), paths.shape[0])
    ).copy()
    path_deltas = np.empty((len(rows), paths.shape[0], horizon), dtype=float)
    for step in range(horizon):
        innovation = paths[:, step]
        linear_state = intercept + persistence * linear_state + innovation
        nonlinear_state = (
            intercept
            + persistence * nonlinear_state
            + coefficients[:, None]
            * np.asarray(_nonlinear_state_response(nonlinear_state))
            + innovation[None, :]
        )
        path_deltas[:, :, step] = scale * (
            nonlinear_state - linear_state[None, :]
        )
    gains = np.asarray(
        [float(row["applied_component_gain"]) for row in rows], dtype=float
    )
    path_deltas *= gains[:, None, None]
    expected = np.stack(
        [
            np.asarray(row["target"], dtype=float)[context:, channel]
            - baseline_target[context:, channel]
            for row in rows
        ]
    )
    np.testing.assert_allclose(
        np.mean(path_deltas, axis=1), expected, rtol=1e-10, atol=1e-8
    )
    lower, upper = np.quantile(path_deltas, (0.10, 0.90), axis=1)
    return lower, upper, {
        "display_only": True,
        "quantiles": [0.10, 0.90],
        "path_count": int(paths.shape[0]),
        "formal_target": "paired_path_mean",
    }


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
    if capability == "nonlinear_persistence":
        display_history = min(context, max(512, 8 * horizon))
        start = context - display_history
    else:
        display_history = context
        start = 0
    x = np.arange(start - context, horizon)
    source = baseline_target[start:, channel]
    treatment_values = [np.asarray(row["target"], dtype=float)[start:, channel] for row in rows]
    future_band: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
    if capability == "nonlinear_persistence":
        future_band = _nonlinear_future_path_band(rows, baseline, channel)
    band_values: list[np.ndarray] = []
    if future_band is not None:
        lower, upper, _band_metadata = future_band
        future_source = baseline_target[context:, channel]
        band_values.extend(
            [
                (future_source[None, :] + lower).ravel(),
                (future_source[None, :] + upper).ravel(),
            ]
        )
    all_values = np.concatenate((source, *treatment_values, *band_values))
    margin = max(1e-6, 0.05 * float(np.ptp(all_values)))
    y_limits = (float(np.min(all_values) - margin), float(np.max(all_values) + margin))
    delta_values = [values - source for values in treatment_values]
    displayed_deltas = delta_values
    delta_limit = max(
        max(float(np.max(np.abs(delta))) for delta in displayed_deltas),
        (
            float(np.max(np.abs(np.concatenate(future_band[:2]))))
            if future_band is not None
            else 0.0
        ),
        1e-6,
    )
    figure, axes = plt.subplots(2, 5, figsize=(16, 5.4), sharex=True)
    for index, (row, values, delta, displayed_delta) in enumerate(
        zip(rows, treatment_values, delta_values, displayed_deltas, strict=True)
    ):
        axis = axes[0, index]
        if future_band is not None:
            lower, upper, _band_metadata = future_band
            future_source = baseline_target[context:, channel]
            axis.fill_between(
                np.arange(horizon),
                future_source + lower[index],
                future_source + upper[index],
                color=TREATMENT_COLOR,
                alpha=0.28,
                linewidth=0,
                label=(
                    "10–90% paired response paths"
                    if index == 0
                    else "_nolegend_"
                ),
                zorder=1,
            )
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
        if capability == "nonlinear_persistence":
            title = (
                f"level {row['capability_level']} · "
                f"headroom ρ={row['sampled_coordinate']:.3f}"
            )
        else:
            title = (
                f"level {row['capability_level']}\n"
                f"{row['controlled_coordinate']}={row['sampled_coordinate']:.3f}"
            )
        axis.set_title(title, fontsize=8)
        bottom = axes[1, index]
        if future_band is not None:
            lower, upper, _band_metadata = future_band
            bottom.fill_between(
                np.arange(horizon),
                lower[index],
                upper[index],
                color=DELTA_COLOR,
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
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
        (
            "state-dependent persistence"
            if capability == "nonlinear_persistence"
            else capability
        )
        + f": five treatments on one official {baseline['dataset_id']} instance",
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
        "display_history": display_history,
        "display_policy": (
            "recent_history_and_complete_future"
            if capability == "nonlinear_persistence"
            else "complete_official_history_and_future"
        ),
        "delta_display": "treatment_minus_source",
        "full_treatment_history": context,
        "horizon": horizon,
        "augmentation_seed": rows[0]["augmentation_seed"],
        "future_estimand": rows[0].get("group_metadata", {}).get(
            "future_estimand"
        ),
        "future_innovation_policy": rows[0].get("mechanism_metadata", {}).get(
            "future_innovation_policy"
        ),
        "future_path_band": (
            future_band[2] if future_band is not None else None
        ),
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


def _plot_covariate_group(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    capability = "covariate_impulse_response"
    baseline_target = np.asarray(baseline["target"], dtype=float)
    baseline_covariates = np.asarray(baseline["covariates"], dtype=float)
    context = int(baseline["context_length"])
    horizon = int(baseline["horizon"])
    target_channel = int(rows[0]["affected_target_indices"][0])
    metadata = dict(rows[0]["mechanism_metadata"])
    covariate_channel = int(metadata["covariate_index"])
    display_history = min(context, 6 * horizon)
    start = context - display_history
    x = np.arange(-display_history, horizon)
    source_target = baseline_target[start:, target_channel]
    source_covariate = baseline_covariates[start:, covariate_channel]
    treatment_targets = [
        np.asarray(row["target"], dtype=float)[start:, target_channel]
        for row in rows
    ]
    treatment_covariates = [
        np.asarray(row["covariates"], dtype=float)[start:, covariate_channel]
        for row in rows
    ]
    target_deltas = [values - source_target for values in treatment_targets]
    covariate_deltas = [
        values - source_covariate for values in treatment_covariates
    ]
    all_target_values = np.concatenate((source_target, *treatment_targets))
    target_margin = max(1e-6, 0.05 * float(np.ptp(all_target_values)))
    target_limits = (
        float(np.min(all_target_values) - target_margin),
        float(np.max(all_target_values) + target_margin),
    )
    target_delta_limit = max(
        max(float(np.max(np.abs(delta))) for delta in target_deltas),
        1e-6,
    )
    covariate_delta_limit = max(
        max(float(np.max(np.abs(delta))) for delta in covariate_deltas),
        1e-6,
    )
    figure, axes = plt.subplots(3, 5, figsize=(16, 7.2), sharex=True)
    for index, (row, target, target_delta, covariate_delta) in enumerate(
        zip(
            rows,
            treatment_targets,
            target_deltas,
            covariate_deltas,
            strict=True,
        )
    ):
        top = axes[0, index]
        top.plot(
            x,
            source_target,
            color=REAL_COLOR,
            linewidth=1.2,
            label="authentic target",
            zorder=3,
        )
        top.plot(
            x,
            target,
            color=TREATMENT_COLOR,
            linewidth=1.0,
            label="treated target",
            zorder=2,
        )
        top.set_ylim(*target_limits)
        top.set_title(
            f"level {row['capability_level']}\n"
            f"target distance={row['sampled_coordinate']:.3f}",
            fontsize=8,
        )

        middle = axes[1, index]
        middle.plot(x, target_delta, color=DELTA_COLOR, linewidth=1.2)
        middle.axhline(0, color="#9e9e9e", linewidth=0.7)
        middle.set_ylim(-1.05 * target_delta_limit, 1.05 * target_delta_limit)

        bottom = axes[2, index]
        bottom.plot(x, covariate_delta, color="#2e7d32", linewidth=1.2)
        bottom.axhline(0, color="#9e9e9e", linewidth=0.7)
        bottom.set_ylim(
            -0.08 * covariate_delta_limit,
            1.08 * covariate_delta_limit,
        )
        bottom.set_xlabel("time relative to forecast origin")

        for axis in (top, middle, bottom):
            axis.axvline(0, color="#263238", linewidth=0.8)
            axis.axvspan(0, horizon - 1, color="#ffecb3", alpha=0.35)

    axes[0, 0].set_ylabel("target / authentic units")
    axes[1, 0].set_ylabel("target treatment − source")
    axes[2, 0].set_ylabel("injected covariate impulse")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    visibility = str(metadata["covariate_availability"])
    figure.suptitle(
        "covariate impulse response: target response and injected "
        f"{visibility} covariate impulse",
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
        "affected_target_index_shown": target_channel,
        "covariate_index_shown": covariate_channel,
        "covariate_name": metadata["covariate_name"],
        "covariate_availability": visibility,
        "display_history": display_history,
        "display_policy": "recent_history_and_complete_future",
        "full_treatment_history": context,
        "horizon": horizon,
        "augmentation_seed": rows[0]["augmentation_seed"],
        "delta_display": (
            "target_treatment_minus_source_and_injected_covariate_delta"
        ),
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
        "Qualification-only in the v11 GIFT-Eval adapter\n"
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
    requested = tuple(dict.fromkeys(args.capabilities))
    required = set(requested) - {"hierarchical_coherence"}
    baselines, treatments = _load_roots(
        [path.resolve() for path in args.dataset_root], required
    )
    selected = _groups(treatments)
    names = {
        capability: f"{index:02d}_{capability}__five_levels.png"
        for index, capability in enumerate(CAPABILITY_IDS, start=1)
    }
    generated_records: list[dict[str, Any]] = []
    for capability in requested:
        output_path = args.output_dir.resolve() / names[capability]
        if capability == "hierarchical_coherence":
            generated_records.append(_plot_hierarchy(output_path))
            continue
        rows = selected.get(capability)
        if rows is None:
            raise ValueError(f"no validated five-level example found for {capability}")
        baseline = baselines[str(rows[0]["baseline_sample_id"])]
        if capability == "covariate_impulse_response":
            generated_records.append(
                _plot_covariate_group(rows, baseline, output_path)
            )
            continue
        generated_records.append(
            _plot_group(
                capability,
                rows,
                baseline,
                output_path,
            )
        )
    manifest_path = args.output_dir.resolve() / "manifest.json"
    records_by_capability: dict[str, dict[str, Any]] = {}
    if manifest_path.exists() and set(requested) != set(CAPABILITY_IDS):
        previous = core.read_json(manifest_path)
        records_by_capability.update(
            (str(record["capability_id"]), record)
            for record in previous.get("records", [])
        )
    records_by_capability.update(
        (str(record["capability_id"]), record) for record in generated_records
    )
    records = [
        records_by_capability[capability]
        for capability in CAPABILITY_IDS
        if capability in records_by_capability
    ]
    manifest = {
        "schema_version": PLOTTER_SCHEMA,
        "created_at": core.utc_now(),
        "selection_policy": (
            "lexicographically_first_validated_official_instance_per_capability"
        ),
        "figure_semantics": (
            "benchmark_truth_paths_not_model_predictions; complete official future; "
            "complete history except capability-specific recent-history focus"
        ),
        "updated_capabilities": list(requested),
        "color_semantics": {
            "real_authentic_source": REAL_COLOR,
            "modified_treatment": TREATMENT_COLOR,
            "treatment_minus_source": DELTA_COLOR,
        },
        "records": records,
    }
    core.write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
