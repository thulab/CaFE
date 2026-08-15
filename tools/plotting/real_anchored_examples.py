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
from cafe.generation.real_anchored_policy import (
    REAL_ANCHORED_CANONICAL_STRENGTH_GRID,
    REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION,
    REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION,
    REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION,
    REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION,
    REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM,
)
from cafe.generation.real_counterfactuals import iter_real_anchored_samples
from cafe.generation.structural_real_counterfactuals import (
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
STRENGTHS = tuple(
    float(value) for value in REAL_ANCHORED_CANONICAL_STRENGTH_GRID
)
PLOTTER_VERSION = "cafe.real_anchored_example_plotter.v5"
SELECTION_POLICY = (
    "production_seed0_or_stable_qualified_audit_background_"
    "no_future_or_model_selection_v2"
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
    availability: Mapping[str, Any]
    structural_availability: Mapping[str, Any]
    qualification_policy: Mapping[str, Any]


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
    dose_calibration_policy_sha256: str
    canonical_strengths: tuple[float, ...]
    applied_alphas: tuple[float, ...]
    paired_minimum_separation_status: str
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


def _validate_current_v5_policy(
    bundle: Mapping[str, Any],
    qualification_policy: Mapping[str, Any],
) -> None:
    if bundle.get("schema_version") != "cafe.calibration_bundle.v5":
        raise ValueError("figure source is not a v5 calibration bundle")
    decisions = qualification_policy.get("decisions")
    dose_policy = qualification_policy.get("dose_policy")
    if not isinstance(decisions, Mapping) or not isinstance(
        dose_policy, Mapping
    ):
        raise ValueError("figure source has no frozen v5 dose policy")
    if decisions.get("schema_version") != "cafe.real_anchored_protocol.v5":
        raise ValueError("figure source does not use real-anchored protocol v5")
    if dose_policy.get("evaluation_origins_used_for_mapping") is not False:
        raise ValueError("figure dose mapping used evaluation origins")
    if tuple(float(value) for value in dose_policy.get("strength_grid", ())) != (
        STRENGTHS
    ):
        raise ValueError("figure source changed the canonical lambda grid")
    expected_upper = {
        "maximum_history_macro_separation": (
            REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION
        ),
        "maximum_future_macro_separation": (
            REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION
        ),
        "maximum_affected_channel_separation": (
            REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION
        ),
        "minimum_acceptance_fraction": (
            REAL_ANCHORED_MINIMUM_SEPARATION_ACCEPTANCE_FRACTION
        ),
        "treatment_source_distance_minimum": (
            REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
        ),
    }
    for key, expected in expected_upper.items():
        observed = dose_policy.get(key)
        if observed is None or not np.isclose(float(observed), float(expected)):
            raise ValueError(
                f"figure source does not freeze current v5 {key}"
            )


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
    qualification_policy = protocol.read_json(
        _verified_file(
            directory,
            bundle,
            "real_anchored_qualification_policy",
        )
    )
    _validate_current_v5_policy(bundle, qualification_policy)
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
        availability=protocol.read_json(
            _verified_file(directory, bundle, "real_anchored_availability")
        ),
        structural_availability=protocol.read_json(
            _verified_file(
                directory,
                bundle,
                "structural_real_anchored_availability",
            )
        ),
        qualification_policy=qualification_policy,
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


def _availability_status(
    artifact: CalibrationArtifact,
    capability_id: str,
    *,
    structural: bool,
) -> str:
    availability = (
        artifact.structural_availability
        if structural
        else artifact.availability
    )
    cell = next(
        (
            row
            for row in availability.get("cells", [])
            if str(row.get("capability_id")) == capability_id
        ),
        None,
    )
    if not isinstance(cell, Mapping):
        return "availability_cell_missing"
    if structural and capability_id == "hierarchical_coherence":
        return "qualification_only_no_forecast_task"
    if str(cell.get("status")) == "available":
        return "formal"
    if str(cell.get("sensitivity_status")) == "available":
        return "sensitivity_D2"
    count = cell.get(
        "formal_background_count" if structural else "eligible_background_count"
    )
    return f"evaluation_example_below_formal_N_{int(count or 0)}"


def _rows_by_dose(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    treatments = [
        row for row in rows if int(row["counterfactual_member"]) == 1
    ]
    treatments.sort(key=lambda row: int(row["dose_index"]))
    if [float(row["dose_value"]) for row in treatments] != list(STRENGTHS):
        raise ValueError("example rows do not contain the canonical strength grid")
    alphas = [float(row["applied_alpha"]) for row in treatments]
    if len(alphas) != len(STRENGTHS) or any(
        right <= left for left, right in zip(alphas, alphas[1:])
    ):
        raise ValueError("example rows do not contain a frozen increasing alpha grid")
    return treatments


def _frozen_dose_calibration(
    artifact: CalibrationArtifact,
    capability_id: str,
) -> Mapping[str, Any]:
    capability = artifact.qualification_policy.get("capabilities", {}).get(
        capability_id
    )
    if not isinstance(capability, Mapping):
        raise ValueError(f"{capability_id} has no frozen capability policy")
    dose_calibration = capability.get("dose_calibration")
    if not isinstance(dose_calibration, Mapping):
        raise ValueError(f"{capability_id} has no frozen dose calibration")
    return dose_calibration


def _dose_metadata(
    treatments: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], tuple[float, ...], str, str]:
    strengths = tuple(float(row["dose_value"]) for row in treatments)
    alphas = tuple(float(row["applied_alpha"]) for row in treatments)
    calibration_hashes = {
        str(row["dose_calibration_policy_sha256"]) for row in treatments
    }
    gate_statuses = {
        str(row["paired_minimum_separation_gate"]["status"])
        for row in treatments
    }
    if len(calibration_hashes) != 1 or gate_statuses != {"passed"}:
        raise ValueError("example rows are not bound to one passed dose policy")
    return strengths, alphas, next(iter(calibration_hashes)), "passed"


def _univariate_example(
    artifact: CalibrationArtifact,
    capability_id: str,
    title: str,
) -> Example:
    dose_calibration = _frozen_dose_calibration(artifact, capability_id)
    if dose_calibration.get("status") != "available":
        return _univariate_unavailable_audit(
            artifact,
            capability_id,
            title,
            dose_calibration,
        )
    rows = list(
        iter_real_anchored_samples(
            artifact.backgrounds,
            artifact.contracts,
            capability_ids=(capability_id,),
            seed_indexes=(0,),
        )
    )
    treatments = _rows_by_dose(rows)
    baselines = [
        row for row in rows if int(row["counterfactual_member"]) == 0
    ]
    if len(treatments) != len(STRENGTHS) or len(baselines) != len(STRENGTHS):
        raise ValueError(f"{capability_id} did not produce five complete pairs")
    strengths, alphas, dose_hash, gate_status = _dose_metadata(treatments)
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
        status=_availability_status(
            artifact,
            capability_id,
            structural=False,
        ),
        dataset_id=str(treatments[0]["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(metadata["contract_sha256"]),
        qualification_policy_sha256=str(
            contract_row["qualification_policy_sha256"]
        ),
        dose_calibration_policy_sha256=dose_hash,
        canonical_strengths=strengths,
        applied_alphas=alphas,
        paired_minimum_separation_status=gate_status,
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


def _univariate_unavailable_audit(
    artifact: CalibrationArtifact,
    capability_id: str,
    title: str,
    dose_calibration: Mapping[str, Any],
) -> Example:
    candidates = [
        row
        for row in artifact.contracts
        if str(row.get("capability_id")) == capability_id
        and isinstance(row.get("contract"), Mapping)
        and isinstance(row.get("dose_design_reference"), Mapping)
    ]
    candidates.sort(
        key=lambda row: (
            protocol.stable_seed(
                str(row["dataset_id"]),
                capability_id,
                str(row["background_id"]),
                "real-anchored-unavailable-figure-audit",
                base=protocol.REAL_ANCHORED_SAMPLE_SEED,
            ),
            str(row["background_id"]),
        )
    )
    if not candidates:
        raise ValueError(
            f"no mechanism-qualified {capability_id} audit is available"
        )
    contract_row = candidates[0]
    background_id = str(contract_row["background_id"])
    background = next(
        row
        for row in artifact.backgrounds
        if str(row["background_id"]) == background_id
    )
    baseline = _unstandardize_univariate(
        np.asarray(background["target"], dtype=float)[:, 0],
        background["standardization"],
    )
    strengths = tuple(
        float(value) for value in dose_calibration["strength_grid"]
    )
    if strengths != STRENGTHS:
        raise ValueError("univariate unavailable audit changed lambda grid")
    contract = contract_row["contract"]
    return Example(
        capability_id=capability_id,
        title=title,
        status="unavailable_dose_mapping",
        dataset_id=str(contract_row["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(contract["capability_contract_sha256"]),
        qualification_policy_sha256=str(
            contract_row["qualification_policy_sha256"]
        ),
        dose_calibration_policy_sha256=str(
            dose_calibration["policy_sha256"]
        ),
        canonical_strengths=strengths,
        applied_alphas=(),
        paired_minimum_separation_status=(
            "not_evaluated_dose_mapping_unavailable"
        ),
        baseline=baseline[:, None],
        treatments=(),
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
            "controlled_component": capability_id,
            "dose_mapping_status": "unavailable",
            "dose_mapping_unavailable_reason": str(
                dose_calibration["unavailable_reason"]
            ),
            "reference_evidence_count": int(
                dose_calibration["reference_evidence_count"]
            ),
            "reference_balanced_effect_count": int(
                dose_calibration.get("reference_balanced_effect_count", 0)
            ),
            "minimum_reference_balanced_effect_count": int(
                dose_calibration.get(
                    "minimum_reference_balanced_effect_count", 0
                )
            ),
            "display_alpha_grid_role": (
                "none_dose_mapping_unavailable_no_treatment_curves"
            ),
            "target_future_used_for_contract_or_selection": False,
        },
    )


def _common_factor_unavailable_audit(
    artifact: CalibrationArtifact,
    title: str,
) -> Example:
    """Build an honest no-treatment audit when common dose mapping failed.

    The real baseline and a mechanism-qualified history fit may still be shown,
    but there is no applied-alpha grid and therefore no counterfactual curve.
    """

    capability_id = "common_factor"
    dose_calibration = _frozen_dose_calibration(artifact, capability_id)
    if dose_calibration.get("status") == "available":
        raise ValueError("common-factor audit requested for an available mapping")
    candidates = [
        row
        for row in artifact.structural_contracts
        if str(row.get("capability_id")) == capability_id
        and isinstance(row.get("contract"), Mapping)
        and row["contract"].get("component_gate", {}).get("passed") is True
        and row["contract"].get("fit_diagnostics", {}).get(
            "observable_passed"
        )
        is True
    ]
    candidates.sort(
        key=lambda row: (
            protocol.stable_seed(
                str(row["dataset_id"]),
                capability_id,
                str(row["background_id"]),
                "real-anchored-unavailable-figure-audit",
                base=protocol.REAL_ANCHORED_SAMPLE_SEED,
            ),
            str(row["background_id"]),
        )
    )
    if not candidates:
        raise ValueError("no mechanism-qualified common-factor audit is available")
    contract_row = candidates[0]
    contract = contract_row["contract"]
    diagnostics = contract["fit_diagnostics"]
    background_id = str(contract_row["background_id"])
    background = next(
        row
        for row in artifact.structural_backgrounds
        if str(row["background_id"]) == background_id
    )
    baseline_panel = _unstandardize_panel(
        np.asarray(background["target"], dtype=float),
        background["target_standardization"],
    )
    protected = int(diagnostics["protected_target_index"])
    loadings = np.abs(np.asarray(diagnostics["loadings"], dtype=float))
    other = [index for index in np.argsort(-loadings) if index != protected]
    if len(other) < 2:
        raise ValueError("common-factor audit has fewer than three channels")
    selected = (protected, int(other[0]), int(other[1]))
    channel_ids = tuple(
        str(background["channel_ids"][index]) for index in selected
    )
    strengths = tuple(
        float(value) for value in dose_calibration["strength_grid"]
    )
    if strengths != STRENGTHS:
        raise ValueError("common-factor unavailable audit changed lambda grid")
    return Example(
        capability_id=capability_id,
        title=title,
        status="unavailable_dose_mapping",
        dataset_id=str(background["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(contract["contract_sha256"]),
        qualification_policy_sha256=str(
            contract_row["frozen_qualification_policy_sha256"]
        ),
        dose_calibration_policy_sha256=str(
            dose_calibration["policy_sha256"]
        ),
        canonical_strengths=strengths,
        applied_alphas=(),
        paired_minimum_separation_status=(
            "not_evaluated_dose_mapping_unavailable"
        ),
        baseline=baseline_panel[:, selected],
        treatments=(),
        channel_names=(
            f"protected {channel_ids[0]}",
            f"auxiliary {channel_ids[1]}",
            f"auxiliary {channel_ids[2]}",
        ),
        source_bundle_sha256=str(artifact.bundle["bundle_content_sha256"]),
        source_file_records=_source_records(
            artifact,
            (
                "structural_real_anchored_backgrounds",
                "structural_real_anchored_contracts",
            ),
        ),
        metadata={
            "item_id": str(background["item_id"]),
            "forecast_origin": int(background["forecast_origin"]),
            "controlled_component": "history_fitted_common_factor",
            "display_channel_indices": list(selected),
            "protected_target_index": protected,
            "dose_mapping_status": "unavailable",
            "dose_mapping_unavailable_reason": str(
                dose_calibration["unavailable_reason"]
            ),
            "reference_evidence_count": int(
                dose_calibration["reference_evidence_count"]
            ),
            "reference_balanced_effect_count": int(
                dose_calibration.get("reference_balanced_effect_count", 0)
            ),
            "minimum_reference_balanced_effect_count": int(
                dose_calibration.get(
                    "minimum_reference_balanced_effect_count", 0
                )
            ),
            "top_factor_share": float(diagnostics["top_factor_share"]),
            "loading_split_cosine": float(
                diagnostics["loading_split_cosine"]
            ),
            "factor_one_step_holdout_r2": float(
                diagnostics["factor_one_step_holdout_r2"]
            ),
            "display_alpha_grid_role": (
                "none_dose_mapping_unavailable_no_treatment_curves"
            ),
            "target_future_used_for_contract_or_selection": False,
            "mandatory_input_ablation_not_applicable_without_treatment": True,
        },
    )


def _cross_series_unavailable_audit(
    artifact: CalibrationArtifact,
    title: str,
) -> Example:
    capability_id = "cross_series_dependence"
    dose_calibration = _frozen_dose_calibration(artifact, capability_id)
    if dose_calibration.get("status") == "available":
        raise ValueError("cross-series audit requested for an available mapping")
    candidates = [
        row
        for row in artifact.structural_contracts
        if str(row.get("capability_id")) == capability_id
        and isinstance(row.get("contract"), Mapping)
        and row["contract"].get("component_gate", {}).get("passed") is True
        and row["contract"].get("fit_diagnostics", {}).get("edge_passed")
        is True
    ]
    candidates.sort(
        key=lambda row: (
            protocol.stable_seed(
                str(row["dataset_id"]),
                capability_id,
                str(row["background_id"]),
                "real-anchored-unavailable-figure-audit",
                base=protocol.REAL_ANCHORED_SAMPLE_SEED,
            ),
            str(row["background_id"]),
        )
    )
    if not candidates:
        raise ValueError("no mechanism-qualified cross-series audit is available")
    contract_row = candidates[0]
    contract = contract_row["contract"]
    diagnostics = contract["fit_diagnostics"]
    background_id = str(contract_row["background_id"])
    background = next(
        row
        for row in artifact.structural_backgrounds
        if str(row["background_id"]) == background_id
    )
    baseline_panel = _unstandardize_panel(
        np.asarray(background["target"], dtype=float),
        background["target_standardization"],
    )
    driver = int(diagnostics["source"])
    responder = int(diagnostics["responders"][0])
    selected = (driver, responder)
    channel_ids = tuple(
        str(background["channel_ids"][index]) for index in selected
    )
    strengths = tuple(
        float(value) for value in dose_calibration["strength_grid"]
    )
    if strengths != STRENGTHS:
        raise ValueError("cross-series unavailable audit changed lambda grid")
    corrected = np.asarray(
        diagnostics["corrected_incremental_r2_by_responder"], dtype=float
    )
    return Example(
        capability_id=capability_id,
        title=title,
        status="sensitivity_unavailable_dose_mapping",
        dataset_id=str(background["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(contract["contract_sha256"]),
        qualification_policy_sha256=str(
            contract_row["frozen_qualification_policy_sha256"]
        ),
        dose_calibration_policy_sha256=str(
            dose_calibration["policy_sha256"]
        ),
        canonical_strengths=strengths,
        applied_alphas=(),
        paired_minimum_separation_status=(
            "not_evaluated_dose_mapping_unavailable"
        ),
        baseline=baseline_panel[:, selected],
        treatments=(),
        channel_names=(
            f"driver {channel_ids[0]}",
            f"responder {channel_ids[1]}",
        ),
        source_bundle_sha256=str(artifact.bundle["bundle_content_sha256"]),
        source_file_records=_source_records(
            artifact,
            (
                "structural_real_anchored_backgrounds",
                "structural_real_anchored_contracts",
            ),
        ),
        metadata={
            "item_id": str(background["item_id"]),
            "forecast_origin": int(background["forecast_origin"]),
            "controlled_component": "directed_predictive_transfer",
            "display_channel_indices": list(selected),
            "driver_index": driver,
            "responder_indices": [responder],
            "lag": int(diagnostics["lag"]),
            "corrected_incremental_r2_by_responder": corrected.tolist(),
            "dose_mapping_status": "unavailable",
            "dose_mapping_unavailable_reason": str(
                dose_calibration["unavailable_reason"]
            ),
            "reference_evidence_count": int(
                dose_calibration["reference_evidence_count"]
            ),
            "reference_balanced_effect_count": int(
                dose_calibration.get("reference_balanced_effect_count", 0)
            ),
            "minimum_reference_balanced_effect_count": int(
                dose_calibration.get(
                    "minimum_reference_balanced_effect_count", 0
                )
            ),
            "display_alpha_grid_role": (
                "none_dose_mapping_unavailable_no_treatment_curves"
            ),
            "target_future_used_for_contract_or_selection": False,
            "mandatory_input_ablation_not_applicable_without_treatment": True,
        },
    )
def _structural_example(
    artifact: CalibrationArtifact,
    capability_id: str,
    title: str,
    *,
    sensitivity: bool,
) -> Example:
    dose_calibration = _frozen_dose_calibration(artifact, capability_id)
    if dose_calibration.get("status") != "available":
        if capability_id == "common_factor" and not sensitivity:
            return _common_factor_unavailable_audit(artifact, title)
        if capability_id == "cross_series_dependence" and sensitivity:
            return _cross_series_unavailable_audit(artifact, title)
        raise ValueError(
            f"{capability_id} has no available applied-alpha mapping"
        )
    capability_rows = tuple(
        row for row in artifact.structural_contracts
        if str(row["capability_id"]) == capability_id
    )
    rows = list(
        iter_structural_real_anchored_samples(
            artifact.structural_backgrounds,
            capability_rows,
            sensitivity=sensitivity,
            seed_indexes=(0,),
        )
    )
    treatments = _rows_by_dose(rows)
    baselines = [
        row for row in rows if int(row["counterfactual_member"]) == 0
    ]
    if len(treatments) != len(STRENGTHS) or len(baselines) != len(STRENGTHS):
        raise ValueError(f"{capability_id} did not produce five complete pairs")
    strengths, alphas, dose_hash, gate_status = _dose_metadata(treatments)
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
        status=(
            "sensitivity_D2"
            if sensitivity
            else _availability_status(
                artifact,
                capability_id,
                structural=True,
            )
        ),
        dataset_id=str(background["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(contract["contract_sha256"]),
        qualification_policy_sha256=str(
            contract_row["frozen_qualification_policy_sha256"]
        ),
        dose_calibration_policy_sha256=dose_hash,
        canonical_strengths=strengths,
        applied_alphas=alphas,
        paired_minimum_separation_status=gate_status,
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
    dose_calibration = contract.get("dose_calibration")
    if not isinstance(dose_calibration, Mapping):
        raise ValueError("hierarchy example has no dose calibration contract")
    strengths = tuple(
        float(value) for value in dose_calibration["strength_grid"]
    )
    mapping_available = dose_calibration.get("status") == "available"
    alphas = (
        tuple(float(value) for value in dose_calibration["applied_alpha_grid"])
        if mapping_available
        else ()
    )
    if strengths != STRENGTHS or (
        mapping_available and len(alphas) != len(STRENGTHS)
    ):
        raise ValueError("hierarchy example dose grid is malformed")
    hierarchy = background["hierarchy"]
    baseline_standardized = np.asarray(hierarchy["target"], dtype=float)
    component = np.asarray(contract["component"], dtype=float)[
        protocol.REAL_ANCHORED_DECOMPOSITION_CONTEXT_LENGTH
        - protocol.REAL_ANCHORED_CONTEXT_LENGTH :
    ]
    treatments_standardized = tuple(
        baseline_standardized + (alpha - 1.0) * component
        for alpha in alphas
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
    child_offsets = tuple(range(1, 1 + len(children)))
    negativity = {
        str(alpha): {
            "negative_value_count_by_child": [
                int(np.sum(values[:, offset] < 0.0))
                for offset in child_offsets
            ],
            "total_negative_value_count": int(
                sum(
                    np.sum(values[:, offset] < 0.0)
                    for offset in child_offsets
                )
            ),
            "minimum_augmented_child_value": float(
                np.min(values[:, child_offsets])
            ),
        }
        for alpha, values in zip(alphas, treatments, strict=True)
    }
    return Example(
        capability_id="hierarchical_coherence",
        title="Forecastable zero-sum hierarchical contrast",
        status=(
            "qualification_only_no_forecast_task"
            if mapping_available
            else "qualification_only_dose_mapping_unavailable"
        ),
        dataset_id=str(background["dataset_id"]),
        background_id=background_id,
        bank_role=str(background["background_bank_role"]),
        contract_sha256=str(contract["contract_sha256"]),
        qualification_policy_sha256=str(
            row["frozen_qualification_policy_sha256"]
        ),
        dose_calibration_policy_sha256=str(
            dose_calibration["policy_sha256"]
        ),
        canonical_strengths=strengths,
        applied_alphas=alphas,
        paired_minimum_separation_status=(
            str(
                contract.get(
                    "paired_minimum_separation_qualification",
                    {},
                ).get("status", "qualification_only_not_evaluated")
            )
            if mapping_available
            else "not_evaluated_dose_mapping_unavailable"
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
            "display_alpha_grid_role": (
                "reference_frozen_applied_grid"
                if mapping_available
                else "none_dose_mapping_unavailable_no_treatment_curves"
            ),
            "dose_mapping_status": str(dose_calibration["status"]),
            "dose_mapping_unavailable_reason": (
                None
                if mapping_available
                else str(dose_calibration["unavailable_reason"])
            ),
            "reference_evidence_count": int(
                dose_calibration["reference_evidence_count"]
            ),
            "reference_balanced_effect_count": int(
                dose_calibration.get("reference_balanced_effect_count", 0)
            ),
            "minimum_reference_balanced_effect_count": int(
                dose_calibration.get(
                    "minimum_reference_balanced_effect_count", 0
                )
            ),
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


def _status_label(status: str) -> str:
    labels = {
        "formal": "formal",
        "sensitivity_D2": "D=2 sensitivity (rank weight 0)",
        "qualification_only_no_forecast_task": (
            "qualification only (no forecast task/rank)"
        ),
        "qualification_only_dose_mapping_unavailable": (
            "qualification only (dose mapping unavailable; no treatment)"
        ),
        "unavailable_dose_mapping": (
            "unavailable (reference-frozen dose mapping failed)"
        ),
        "sensitivity_unavailable_dose_mapping": (
            "D=2 sensitivity unavailable (dose mapping failed)"
        ),
    }
    if status.startswith("evaluation_example_below_formal_N_"):
        count = status.rsplit("_", 1)[-1]
        return f"evaluation illustration (formal N={count}<4; not ranked)"
    return labels.get(status, status)


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
    for column, (strength, alpha, treatment, delta) in enumerate(
        zip(
            example.canonical_strengths,
            example.applied_alphas,
            treatments,
            deltas,
            strict=True,
        )
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
        upper.set_title(
            rf"$\lambda={strength:.1f}$" + "\n" + rf"$\alpha={alpha:.3g}$",
            fontsize=10,
        )
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
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        f"{example.title} | {example.dataset_id} | "
        f"{_status_label(example.status)}",
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
        for column, (strength, alpha, treatment) in enumerate(
            zip(
                example.canonical_strengths,
                example.applied_alphas,
                treatments,
                strict=True,
            )
        ):
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
                axis.set_title(
                    rf"$\lambda={strength:.1f}$"
                    + "\n"
                    + rf"$\alpha={alpha:.3g}$",
                    fontsize=10,
                )
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
    status_label = _status_label(example.status)
    figure.suptitle(
        f"{example.title} | {example.dataset_id} | {status_label}",
        fontsize=13,
        fontweight="bold",
    )
    return _save_figure(figure, output_dir, stem)


def _plot_unavailable_audit(
    example: Example,
    output_dir: Path,
    stem: str,
) -> dict[str, Mapping[str, Any]]:
    """Show the authentic baseline without inventing an applied-alpha grid."""

    if example.treatments or example.applied_alphas:
        raise ValueError("unavailable audit unexpectedly contains treatments")
    if example.canonical_strengths != STRENGTHS:
        raise ValueError("unavailable audit changed the canonical lambda grid")
    x = np.arange(-DISPLAY_HISTORY, DISPLAY_HORIZON)
    baseline = _display(example.baseline)
    row_count = baseline.shape[1]
    figure, axes = plt.subplots(
        row_count,
        5,
        figsize=(17.2, 2.25 * row_count + 1.5),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row_index in range(row_count):
        limits = _axis_limits((baseline[:, row_index],))
        for column, strength in enumerate(example.canonical_strengths):
            axis = axes[row_index, column]
            axis.plot(
                x,
                baseline[:, row_index],
                color=BASELINE_COLOR,
                linewidth=1.05,
                label="authentic real baseline",
            )
            axis.set_ylim(limits)
            _decorate_axis(axis, bottom=(row_index == row_count - 1))
            axis.text(
                0.5,
                0.08,
                "no treatment curve",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                color=DELTA_COLOR,
                fontsize=8.5,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.5,
                },
            )
            if row_index == 0:
                axis.set_title(
                    rf"$\lambda={strength:.1f}$" + "\n" + r"$\alpha=\mathrm{N/A}$",
                    fontsize=10,
                )
        axes[row_index, 0].set_ylabel(example.channel_names[row_index])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        frameon=False,
    )
    reason = str(
        example.metadata.get(
            "dose_mapping_unavailable_reason",
            "dose_mapping_unavailable",
        )
    )
    figure.suptitle(
        f"{example.title} | {example.dataset_id} | "
        f"{_status_label(example.status)}\n"
        f"reason: {reason}; authentic baseline only, no applied alpha exists",
        fontsize=12.5,
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
        "dose_calibration_policy_sha256": (
            example.dose_calibration_policy_sha256
        ),
        "canonical_strength_grid": list(example.canonical_strengths),
        "applied_alpha_grid": list(example.applied_alphas),
        "applied_alpha_grid_available": bool(example.applied_alphas),
        "treatment_curve_count": len(example.treatments),
        "paired_minimum_separation_status": (
            example.paired_minimum_separation_status
        ),
        "source_calibration_bundle_content_sha256": example.source_bundle_sha256,
        "source_file_records": dict(example.source_file_records),
        "selection_policy": SELECTION_POLICY,
        "figure_seed": int(protocol.REAL_ANCHORED_SAMPLE_SEED),
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
        "schema_version": "cafe.real_anchored_example_figure_manifest.v3",
        "plotter_version": PLOTTER_VERSION,
        "plotter_source_sha256": _sha256(Path(__file__).resolve()),
        "selection_policy": SELECTION_POLICY,
        "figure_seed": int(protocol.REAL_ANCHORED_SAMPLE_SEED),
        "canonical_strength_grid": list(STRENGTHS),
        "physical_alpha_grid_semantics": (
            "contract_specific_history_only_resolved_recorded_per_example"
        ),
        "unavailable_figure_semantics": (
            "lambda_columns_with_alpha_na_and_authentic_baseline_only_"
            "never_a_candidate_or_uniform_alpha_grid"
        ),
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
        "paired_minimum_separation_semantics": (
            "treatment_only_distance_from_its_authentic_source_baseline_"
            "plus_local_augmentation_upper_budget_not_synthetic_loo_dcr_nndr"
        ),
        "paired_separation_policy": {
            "treatment_source_distance_minimum": float(
                REAL_ANCHORED_SOURCE_DISTANCE_MINIMUM
            ),
            "adjacent_treatment_distance_role": "diagnostic_only",
            "maximum_history_macro_separation": float(
                REAL_ANCHORED_MAXIMUM_HISTORY_MACRO_SEPARATION
            ),
            "maximum_future_macro_separation": float(
                REAL_ANCHORED_MAXIMUM_FUTURE_MACRO_SEPARATION
            ),
            "maximum_affected_channel_separation": float(
                REAL_ANCHORED_MAXIMUM_AFFECTED_CHANNEL_SEPARATION
            ),
        },
        "target_future_selection_policy": "prohibited",
        "model_output_selection_policy": "prohibited",
        "examples": list(entries),
    }
    protocol.write_json(output_dir / "manifest.json", manifest)


def build_examples(args: argparse.Namespace) -> list[tuple[Example, str]]:
    ett = _load_calibration(args.ett_calibration)
    trend = _load_calibration(args.trend_calibration or args.ett_calibration)
    nonlinear = _load_calibration(
        args.nonlinear_calibration or args.ett_calibration
    )
    common = _load_calibration(args.common_calibration or args.ett_calibration)
    electricity = _load_calibration(args.electricity_calibration)
    bitbrains = _load_calibration(args.bitbrains_calibration)
    hierarchy = _load_calibration(args.hierarchy_calibration)
    return [
        (
            _univariate_example(
                trend,
                "trend",
                "Local nonlinear trend continuation",
            ),
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
                nonlinear,
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
                common,
                "common_factor",
                "Forecastable common factor",
                sensitivity=False,
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
                sensitivity=False,
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
    parser.add_argument("--trend-calibration", type=Path)
    parser.add_argument("--nonlinear-calibration", type=Path)
    parser.add_argument("--common-calibration", type=Path)
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
        if not example.treatments:
            files = _plot_unavailable_audit(example, output_dir, stem)
        elif example.capability_id in {
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
