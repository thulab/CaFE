#!/usr/bin/env python3
"""Render the ten documented real-anchored counterfactual examples.

The plotter consumes completed calibration artifacts, verifies their file
records, selects the production seed-0 background, and replays the production
counterfactual operators.  It never fits a display-only component and never
uses model predictions or held-out target values to choose an example.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from cafe import protocol
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_ALPHAS,
    iter_real_anchored_samples,
)
from cafe.generation.structural_real_counterfactuals import (
    STRUCTURAL_ALPHAS,
    iter_structural_real_anchored_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs" / "figures" / "real-anchored-capability-examples"
)
DISPLAY_HISTORY = 168
DISPLAY_HORIZON = protocol.HORIZON
VISIBLE_START = (
    protocol.REAL_ANCHORED_CONTEXT_LENGTH - DISPLAY_HISTORY
)
ALPHAS = tuple(float(value) for value in REAL_ANCHORED_ALPHAS)
PLOTTER_VERSION = "cafe.real_anchored_example_plotter.v1"
SELECTION_POLICY = (
    "production_seed0_stable_background_permutation_no_future_or_model_selection_v1"
)

BASELINE_COLOR = "#6B7280"
TREATMENT_COLOR = "#0072B2"
DELTA_COLOR = "#D55E00"
INPUT_COLOR = "#009E73"
FUTURE_COLOR = "#E8F1FA"
GRID_COLOR = "#D1D5DB"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "svg.hashsalt": PLOTTER_VERSION,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class CalibrationArtifact:
    directory: Path
    bundle: Mapping[str, Any]
    backgrounds: tuple[Mapping[str, Any], ...]
    contracts: tuple[Mapping[str, Any], ...]
    structural_backgrounds: tuple[Mapping[str, Any], ...]
    structural_contracts: tuple[Mapping[str, Any], ...]
    hierarchy_contracts: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Example:
    capability_id: str
    title: str
    status: str
    dataset_id: str
    background_id: str
    bank_role: str
    contract_sha256: str
    qualification_policy_sha256: str
    baseline: np.ndarray
    treatments: tuple[np.ndarray, ...]
    channel_names: tuple[str, ...]
    source_bundle_sha256: str
    source_file_records: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]
    covariate: np.ndarray | None = None
    covariate_name: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(
    directory: Path,
    bundle: Mapping[str, Any],
    key: str,
) -> Path:
    record = bundle.get("files", {}).get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"calibration bundle is missing {key!r}")
    recorded_path = Path(str(record["path"]))
    path = directory / recorded_path.name
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{key} byte count disagrees with calibration bundle")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"{key} hash disagrees with calibration bundle")
    return path


def _rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    return tuple(protocol.iter_jsonl(path))


def _load_calibration(directory: Path) -> CalibrationArtifact:
    directory = directory.resolve()
    bundle_path = directory / "calibration_bundle.json"
    bundle = protocol.read_json(bundle_path)
    records = bundle.get("files")
    if not isinstance(records, Mapping):
        raise ValueError("calibration bundle has no file records")
    expected_bundle_hash = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    if bundle.get("bundle_content_sha256") != expected_bundle_hash:
        raise ValueError("calibration bundle content hash mismatch")
    return CalibrationArtifact(
        directory=directory,
        bundle=bundle,
        backgrounds=_rows(_verified_file(directory, bundle, "real_anchored_backgrounds")),
        contracts=_rows(_verified_file(directory, bundle, "real_anchored_contracts")),
        structural_backgrounds=_rows(
            _verified_file(directory, bundle, "structural_real_anchored_backgrounds")
        ),
        structural_contracts=_rows(
            _verified_file(directory, bundle, "structural_real_anchored_contracts")
        ),
        hierarchy_contracts=_rows(
            _verified_file(directory, bundle, "structural_hierarchy_qualification")
        ),
    )


def _source_records(
    artifact: CalibrationArtifact,
    keys: Iterable[str],
) -> dict[str, Mapping[str, Any]]:
    return {
        key: {
            "bytes": int(artifact.bundle["files"][key]["bytes"]),
            "sha256": str(artifact.bundle["files"][key]["sha256"]),
        }
        for key in keys
    }


def _unstandardize_univariate(
    values: np.ndarray,
    standardization: Mapping[str, Any],
) -> np.ndarray:
    return (
        np.asarray(values, dtype=float) * float(standardization["scale"])
        + float(standardization["location"])
    )


def _unstandardize_panel(
    values: np.ndarray,
    standardization: Mapping[str, Any],
) -> np.ndarray:
    scale = np.asarray(standardization["scale_by_target"], dtype=float)
    center = np.asarray(standardization["center_by_target"], dtype=float)
    return np.asarray(values, dtype=float) * scale[None, :] + center[None, :]


def _unstandardize_hierarchy(
    values: np.ndarray,
    standardization: Mapping[str, Any],
) -> np.ndarray:
    center = np.asarray(standardization["center_by_node"], dtype=float)
    return (
        np.asarray(values, dtype=float) * float(standardization["shared_scale"])
        + center[None, :]
    )


def _rows_by_dose(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    treatments = [
        row for row in rows if int(row["counterfactual_member"]) == 1
    ]
    treatments.sort(key=lambda row: int(row["dose_index"]))
    if [float(row["dose_value"]) for row in treatments] != list(ALPHAS):
        raise ValueError("example rows do not contain the frozen alpha grid")
    return treatments


def _univariate_example(
    artifact: CalibrationArtifact,
    capability_id: str,
    title: str,
) -> Example:
    rows = list(
        iter_real_anchored_samples(
            artifact.backgrounds,
            artifact.contracts,
            capability_ids=(capability_id,),
            seed_indexes=(0,),
            alphas=ALPHAS,
        )
    )
    treatments = _rows_by_dose(rows)
    baselines = [
        row for row in rows if int(row["counterfactual_member"]) == 0
    ]
    if len(treatments) != len(ALPHAS) or len(baselines) != len(ALPHAS):
        raise ValueError(f"{capability_id} did not produce five complete pairs")
    background_id = str(treatments[0]["background_id"])
    background = next(
        row for row in artifact.backgrounds
        if str(row["background_id"]) == background_id
    )
    contract_row = next(
        row for row in artifact.contracts
        if str(row["background_id"]) == background_id
        and str(row["capability_id"]) == capability_id
    )
    standardization = background["standardization"]
    baseline = _unstandardize_univariate(
        np.asarray(baselines[0]["target"], dtype=float)[:, 0],
        standardization,
    )
    treatment_values = tuple(
        _unstandardize_univariate(
            np.asarray(row["target"], dtype=float)[:, 0],
            standardization,
        )
        for row in treatments
    )
    metadata = treatments[0]["generation_metadata"]
    return Example(
        capability_id=capability_id,
        title=title,
        status="formal",
        dataset_id=str(treatments[0]["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(metadata["contract_sha256"]),
        qualification_policy_sha256=str(
            contract_row["qualification_policy_sha256"]
        ),
        baseline=baseline,
        treatments=treatment_values,
        channel_names=(str(background["channel_id"]),),
        source_bundle_sha256=str(artifact.bundle["bundle_content_sha256"]),
        source_file_records=_source_records(
            artifact,
            ("real_anchored_backgrounds", "real_anchored_contracts"),
        ),
        metadata={
            "item_id": str(background["item_id"]),
            "channel_id": str(background["channel_id"]),
            "forecast_origin": int(background["forecast_origin"]),
            "seed_index": 0,
            "controlled_component": str(metadata["controlled_component"]),
            "target_future_used_for_contract_or_selection": False,
        },
    )


def _structural_example(
    artifact: CalibrationArtifact,
    capability_id: str,
    title: str,
    *,
    sensitivity: bool,
) -> Example:
    capability_rows = tuple(
        row for row in artifact.structural_contracts
        if str(row["capability_id"]) == capability_id
    )
    rows = list(
        iter_structural_real_anchored_samples(
            artifact.structural_backgrounds,
            capability_rows,
            alphas=STRUCTURAL_ALPHAS,
            sensitivity=sensitivity,
            seed_indexes=(0,),
        )
    )
    treatments = _rows_by_dose(rows)
    baselines = [
        row for row in rows if int(row["counterfactual_member"]) == 0
    ]
    if len(treatments) != len(ALPHAS) or len(baselines) != len(ALPHAS):
        raise ValueError(f"{capability_id} did not produce five complete pairs")
    background_id = str(treatments[0]["background_id"])
    background = next(
        row for row in artifact.structural_backgrounds
        if str(row["background_id"]) == background_id
    )
    contract_row = next(
        row for row in capability_rows
        if str(row["background_id"]) == background_id
    )
    contract = contract_row["contract"]
    baseline = _unstandardize_panel(
        np.asarray(baselines[0]["target"], dtype=float),
        background["target_standardization"],
    )
    treatment_values = tuple(
        _unstandardize_panel(
            np.asarray(row["target"], dtype=float),
            background["target_standardization"],
        )
        for row in treatments
    )
    diagnostics = contract["fit_diagnostics"]
    metadata: dict[str, Any] = {
        "item_id": str(background["item_id"]),
        "forecast_origin": int(background["forecast_origin"]),
        "seed_index": 0,
        "controlled_component": str(
            treatments[0]["generation_metadata"]["controlled_component"]
        ),
        "target_future_used_for_contract_or_selection": False,
    }
    covariate: np.ndarray | None = None
    covariate_name: str | None = None
    if capability_id == "common_factor":
        protected = int(diagnostics["protected_target_index"])
        loadings = np.abs(np.asarray(diagnostics["loadings"], dtype=float))
        other = [index for index in np.argsort(-loadings) if index != protected]
        selected = (protected, int(other[0]), int(other[1]))
        metadata.update(
            {
                "display_channel_indices": list(selected),
                "protected_target_index": protected,
                "mandatory_input_ablation_not_shown": True,
            }
        )
    elif capability_id == "cross_series_dependence":
        driver = int(diagnostics["source"])
        responder = int(diagnostics["responders"][0])
        selected = (driver, responder)
        metadata.update(
            {
                "display_channel_indices": list(selected),
                "driver_index": driver,
                "responder_indices": [responder],
                "lag": int(diagnostics["lag"]),
                "mandatory_input_ablation_not_shown": True,
            }
        )
    elif capability_id == "covariate_response":
        target_index = int(diagnostics["eligible_target_indices"][0])
        selected = (target_index,)
        payload = background["known_future_covariates"]
        target_diagnostic = next(
            row for row in diagnostics["target_diagnostics"]
            if int(row["target_index"]) == target_index
        )
        beta = np.asarray(
            target_diagnostic["beta_by_covariate_and_lag"], dtype=float
        )
        covariate_index = int(np.argmax(np.linalg.norm(beta, axis=1)))
        normalized_covariate = np.asarray(payload["target"], dtype=float)[
            :, covariate_index
        ]
        covariate_normalization = payload["normalization"]
        covariate = (
            normalized_covariate
            * float(covariate_normalization["scale_by_covariate"][covariate_index])
            + float(
                covariate_normalization["center_by_covariate"][covariate_index]
            )
        )
        covariate_name = str(payload["column_names"][covariate_index])
        metadata.update(
            {
                "display_channel_indices": list(selected),
                "eligible_target_indices": list(
                    diagnostics["eligible_target_indices"]
                ),
                "display_covariate_index": covariate_index,
                "display_covariate_name": covariate_name,
                "known_future_covariate_path_unchanged_across_doses": True,
            }
        )
    else:
        raise ValueError(f"unsupported structural capability {capability_id}")
    channel_names = tuple(
        str(background["channel_ids"][index]) for index in selected
    )
    if capability_id == "common_factor":
        channel_names = (
            f"protected {channel_names[0]}",
            f"auxiliary {channel_names[1]}",
            f"auxiliary {channel_names[2]}",
        )
    elif capability_id == "cross_series_dependence":
        channel_names = (
            f"driver {channel_names[0]}",
            f"responder {channel_names[1]}",
        )
    elif capability_id == "covariate_response":
        channel_names = (f"eligible target {channel_names[0]}",)
    return Example(
        capability_id=capability_id,
        title=title,
        status=("sensitivity_D2" if sensitivity else "formal"),
        dataset_id=str(background["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(contract["contract_sha256"]),
        qualification_policy_sha256=str(
            contract_row["frozen_qualification_policy_sha256"]
        ),
        baseline=baseline[:, selected],
        treatments=tuple(values[:, selected] for values in treatment_values),
        channel_names=channel_names,
        source_bundle_sha256=str(artifact.bundle["bundle_content_sha256"]),
        source_file_records=_source_records(
            artifact,
            (
                "structural_real_anchored_backgrounds",
                "structural_real_anchored_contracts",
            ),
        ),
        metadata=metadata,
        covariate=covariate,
        covariate_name=covariate_name,
    )


def _hierarchy_example(
    artifact: CalibrationArtifact,
) -> Example:
    eligible = [
        row for row in artifact.hierarchy_contracts
        if isinstance(row.get("contract"), Mapping)
        and row["contract"].get("qualification_only") is True
        and row["contract"]["fit_diagnostics"].get("qualification_passed") is True
    ]
    eligible.sort(
        key=lambda row: (
            protocol.stable_seed(
                str(row["dataset_id"]),
                "hierarchical_coherence",
                str(row["background_id"]),
                "structural-real-anchored-background-permutation",
                base=protocol.REAL_ANCHORED_SAMPLE_SEED,
            ),
            str(row["background_id"]),
        )
    )
    if not eligible:
        raise ValueError("no qualified hierarchy example is available")
    row = eligible[0]
    contract = row["contract"]
    background_id = str(row["background_id"])
    background = next(
        value for value in artifact.structural_backgrounds
        if str(value["background_id"]) == background_id
    )
    hierarchy = background["hierarchy"]
    baseline_standardized = np.asarray(hierarchy["target"], dtype=float)
    component = np.asarray(contract["component"], dtype=float)[
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
    ]
    treatments_standardized = tuple(
        baseline_standardized + (alpha - 1.0) * component
        for alpha in ALPHAS
    )
    baseline = _unstandardize_hierarchy(
        baseline_standardized,
        hierarchy["standardization"],
    )
    treatments = tuple(
        _unstandardize_hierarchy(values, hierarchy["standardization"])
        for values in treatments_standardized
    )
    parent = int(hierarchy["parent_index"])
    children = tuple(int(value) for value in hierarchy["child_indices"])
    selected = (parent, *children)
    negativity = contract["fit_diagnostics"]["raw_negativity_audit_by_alpha"]
    return Example(
        capability_id="hierarchical_coherence",
        title="Forecastable zero-sum hierarchical contrast",
        status="qualification_only_no_forecast_task",
        dataset_id=str(background["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(contract["contract_sha256"]),
        qualification_policy_sha256=str(
            row["frozen_qualification_policy_sha256"]
        ),
        baseline=baseline[:, selected],
        treatments=tuple(values[:, selected] for values in treatments),
        channel_names=tuple(
            str(hierarchy["node_ids"][index]) for index in selected
        ),
        source_bundle_sha256=str(artifact.bundle["bundle_content_sha256"]),
        source_file_records=_source_records(
            artifact,
            (
                "structural_real_anchored_backgrounds",
                "structural_hierarchy_qualification",
            ),
        ),
        metadata={
            "item_id": str(background["item_id"]),
            "forecast_origin": int(background["forecast_origin"]),
            "seed_index": 0,
            "controlled_component": "forecastable_zero_sum_hierarchical_contrast",
            "display_channel_indices": list(selected),
            "aggregation_law": str(hierarchy["aggregation_law"]),
            "zero_sum_component_max_abs": float(
                contract["fit_diagnostics"]["zero_sum_component_max_abs"]
            ),
            "raw_negativity_audit_by_alpha": negativity,
            "target_future_used_for_contract_or_selection": False,
            "real_future_used_only_for_post_fit_raw_negativity_audit": True,
        },
    )


def _display(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape[0] != protocol.REAL_ANCHORED_MASTER_LENGTH:
        raise ValueError("example path does not have L336+H48 geometry")
    return array[VISIBLE_START:]


def _axis_limits(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    low = min(float(np.min(array)) for array in arrays)
    high = max(float(np.max(array)) for array in arrays)
    span = high - low
    padding = 0.06 * span if span > 0.0 else max(1.0, 0.06 * abs(low))
    return low - padding, high + padding


def _decorate_axis(axis: Axes, *, bottom: bool) -> None:
    axis.axvspan(0, DISPLAY_HORIZON - 1, color=FUTURE_COLOR, zorder=0)
    axis.axvline(0, color="#111827", linewidth=0.8, linestyle=":", zorder=4)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.55)
    axis.set_xlim(-DISPLAY_HISTORY, DISPLAY_HORIZON - 1)
    axis.set_xticks((-168, -96, -48, 0, 47))
    if not bottom:
        axis.tick_params(axis="x", labelbottom=False)
    else:
        axis.set_xlabel("time relative to forecast origin")


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
) -> dict[str, Mapping[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Mapping[str, Any]] = {}
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        metadata = {"Date": None} if suffix == "svg" else None
        figure.savefig(
            path,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        records[suffix] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    plt.close(figure)
    return records


def _plot_univariate(
    example: Example,
    output_dir: Path,
    stem: str,
) -> dict[str, Mapping[str, Any]]:
    x = np.arange(-DISPLAY_HISTORY, DISPLAY_HORIZON)
    baseline = _display(example.baseline)
    treatments = tuple(_display(values) for values in example.treatments)
    deltas = tuple(values - baseline for values in treatments)
    full_limits = _axis_limits((baseline, *treatments))
    delta_bound = max(float(np.max(np.abs(values))) for values in deltas)
    delta_bound = 1.08 * delta_bound if delta_bound > 0.0 else 1.0
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(17.2, 5.7),
        sharex=True,
        constrained_layout=True,
    )
    for column, (alpha, treatment, delta) in enumerate(
        zip(ALPHAS, treatments, deltas)
    ):
        upper = axes[0, column]
        lower = axes[1, column]
        upper.plot(
            x,
            baseline,
            color=BASELINE_COLOR,
            linewidth=1.05,
            linestyle="--",
            label="real baseline (alpha=1)",
        )
        upper.plot(
            x,
            treatment,
            color=TREATMENT_COLOR,
            linewidth=1.25,
            label="counterfactual truth",
        )
        upper.set_ylim(full_limits)
        upper.set_title(rf"$\alpha={alpha:.1f}$", fontsize=11)
        lower.plot(x, delta, color=DELTA_COLOR, linewidth=1.15)
        lower.axhline(0.0, color=BASELINE_COLOR, linewidth=0.6)
        lower.set_ylim(-delta_bound, delta_bound)
        _decorate_axis(upper, bottom=False)
        _decorate_axis(lower, bottom=True)
    axes[0, 0].set_ylabel("raw target")
    axes[1, 0].set_ylabel(r"truth delta $X^{(\alpha)}-X^{(1)}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper right",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        f"{example.title} | {example.dataset_id} | {example.status}",
        fontsize=13,
        fontweight="bold",
    )
    return _save_figure(figure, output_dir, stem)


def _plot_structural(
    example: Example,
    output_dir: Path,
    stem: str,
) -> dict[str, Mapping[str, Any]]:
    x = np.arange(-DISPLAY_HISTORY, DISPLAY_HORIZON)
    baseline = _display(example.baseline)
    treatments = tuple(_display(values) for values in example.treatments)
    target_rows = baseline.shape[1]
    include_covariate = example.covariate is not None
    row_count = target_rows + int(include_covariate)
    figure, axes = plt.subplots(
        row_count,
        5,
        figsize=(17.2, 2.25 * row_count + 1.3),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row_index in range(target_rows):
        limits = _axis_limits(
            (
                baseline[:, row_index],
                *(values[:, row_index] for values in treatments),
            )
        )
        for column, (alpha, treatment) in enumerate(zip(ALPHAS, treatments)):
            axis = axes[row_index, column]
            axis.plot(
                x,
                baseline[:, row_index],
                color=BASELINE_COLOR,
                linewidth=1.0,
                linestyle="--",
                label="real baseline (alpha=1)",
            )
            axis.plot(
                x,
                treatment[:, row_index],
                color=TREATMENT_COLOR,
                linewidth=1.2,
                label="counterfactual truth",
            )
            axis.set_ylim(limits)
            _decorate_axis(
                axis,
                bottom=(row_index == row_count - 1 and not include_covariate),
            )
            if row_index == 0:
                axis.set_title(rf"$\alpha={alpha:.1f}$", fontsize=11)
        axes[row_index, 0].set_ylabel(example.channel_names[row_index])
    if include_covariate:
        covariate = _display(np.asarray(example.covariate, dtype=float))
        limits = _axis_limits((covariate,))
        for column in range(5):
            axis = axes[-1, column]
            axis.step(
                x,
                covariate,
                where="mid",
                color=INPUT_COLOR,
                linewidth=1.15,
                label="unchanged known-future input",
            )
            axis.set_ylim(limits)
            _decorate_axis(axis, bottom=True)
        axes[-1, 0].set_ylabel(str(example.covariate_name))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if include_covariate:
        cov_handles, cov_labels = axes[-1, 0].get_legend_handles_labels()
        handles += cov_handles
        labels += cov_labels
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    status_label = {
        "formal": "formal",
        "sensitivity_D2": "D=2 sensitivity (rank weight 0)",
        "qualification_only_no_forecast_task": (
            "qualification only (no forecast task/rank)"
        ),
    }.get(example.status, example.status)
    figure.suptitle(
        f"{example.title} | {example.dataset_id} | {status_label}",
        fontsize=13,
        fontweight="bold",
    )
    return _save_figure(figure, output_dir, stem)


def _manifest_entry(
    example: Example,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "capability_id": example.capability_id,
        "status": example.status,
        "dataset_id": example.dataset_id,
        "background_id": example.background_id,
        "background_bank_role": example.bank_role,
        "contract_sha256": example.contract_sha256,
        "qualification_policy_sha256": example.qualification_policy_sha256,
        "source_calibration_bundle_content_sha256": example.source_bundle_sha256,
        "source_file_records": dict(example.source_file_records),
        "selection_policy": SELECTION_POLICY,
        "figure_seed": int(protocol.REAL_ANCHORED_SAMPLE_SEED),
        "alpha_grid": list(ALPHAS),
        "display_history": DISPLAY_HISTORY,
        "display_horizon": DISPLAY_HORIZON,
        "fit_history": protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH,
        "raw_units": True,
        "is_model_prediction": False,
        "files": dict(files),
        **dict(example.metadata),
    }


def _write_manifest(
    output_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> None:
    manifest = {
        "schema_version": "cafe.real_anchored_example_figure_manifest.v1",
        "plotter_version": PLOTTER_VERSION,
        "plotter_source_sha256": _sha256(Path(__file__).resolve()),
        "selection_policy": SELECTION_POLICY,
        "figure_seed": int(protocol.REAL_ANCHORED_SAMPLE_SEED),
        "alpha_grid": list(ALPHAS),
        "alpha_one_role": "repeated_unmodified_real_baseline_not_a_dose_column",
        "display_geometry": {
            "history": DISPLAY_HISTORY,
            "horizon": DISPLAY_HORIZON,
            "fit_history": protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH,
        },
        "qualification_threshold_semantics": (
            "protocol_constants_predeclared_then_verified_frozen_and_hash_bound_"
            "by_source_time_disjoint_reference_bank_not_learned_from_evaluation"
        ),
        "target_future_selection_policy": "prohibited",
        "model_output_selection_policy": "prohibited",
        "examples": list(entries),
    }
    protocol.write_json(output_dir / "manifest.json", manifest)


def build_examples(args: argparse.Namespace) -> list[tuple[Example, str]]:
    ett = _load_calibration(args.ett_calibration)
    electricity = _load_calibration(args.electricity_calibration)
    bitbrains = _load_calibration(args.bitbrains_calibration)
    hierarchy = _load_calibration(args.hierarchy_calibration)
    return [
        (
            _univariate_example(ett, "trend", "Local nonlinear trend continuation"),
            "01_trend__five_doses",
        ),
        (
            _univariate_example(
                ett, "multi_seasonal", "Independent multi-seasonality"
            ),
            "02_multi_seasonal__five_doses",
        ),
        (
            _univariate_example(
                ett,
                "time_varying_seasonality",
                "Carrier amplitude modulation",
            ),
            "03_time_varying_seasonality__five_doses",
        ),
        (
            _univariate_example(
                ett, "regime_switching", "Observed persistent level shift"
            ),
            "04_regime_switching__five_doses",
        ),
        (
            _univariate_example(
                ett,
                "nonlinear_persistence",
                "Nonlinear autoregressive persistence",
            ),
            "05_nonlinear_persistence__five_doses",
        ),
        (
            _univariate_example(
                electricity,
                "predictable_intermittency",
                "Predictable recurrent intermittency",
            ),
            "06_predictable_intermittency__five_doses",
        ),
        (
            _structural_example(
                ett, "common_factor", "Forecastable common factor", sensitivity=False
            ),
            "07_common_factor__five_doses",
        ),
        (
            _hierarchy_example(hierarchy),
            "08_hierarchical_coherence__five_doses",
        ),
        (
            _structural_example(
                bitbrains,
                "cross_series_dependence",
                "Directed cross-series predictive transfer",
                sensitivity=True,
            ),
            "09_cross_series_dependence__five_doses",
        ),
        (
            _structural_example(
                hierarchy,
                "covariate_response",
                "Known-future conditional predictive response",
                sensitivity=False,
            ),
            "10_covariate_response__five_doses",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ett-calibration", type=Path, required=True)
    parser.add_argument("--electricity-calibration", type=Path, required=True)
    parser.add_argument("--bitbrains-calibration", type=Path, required=True)
    parser.add_argument("--hierarchy-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    entries: list[Mapping[str, Any]] = []
    for example, stem in build_examples(args):
        if example.capability_id in {
            "trend",
            "multi_seasonal",
            "time_varying_seasonality",
            "regime_switching",
            "nonlinear_persistence",
            "predictable_intermittency",
        }:
            files = _plot_univariate(example, output_dir, stem)
        else:
            files = _plot_structural(example, output_dir, stem)
        entries.append(_manifest_entry(example, files))
    _write_manifest(output_dir, entries)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "example_count": len(entries),
                "manifest": str(output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
