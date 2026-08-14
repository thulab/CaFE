#!/usr/bin/env python3
"""Offline qualification for CaFE's real-anchored counterfactual track.

This utility reads local GIFT-Eval assets, fits history-only contracts, and
prints one JSON summary to stdout. It does not start model services or write
pipeline artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cafe import protocol
from cafe.generation.real_counterfactuals import (
    REAL_ANCHORED_ALPHAS,
    REAL_ANCHORED_SUPPORTED_CAPABILITIES,
    available_capabilities,
    fit_background_capability_contracts,
    iter_real_anchored_samples,
    public_background,
    reconstruct_source_baseline,
    validate_contract_integrity,
)
from cafe.generation.real_anchored_policy import (
    QUALIFICATION_THRESHOLD_SOURCE_POLICY,
)
from cafe.generation.reference_bank import (
    freeze_real_anchored_qualification_policy,
    split_real_anchored_background_banks,
    validate_evaluation_qualification_policy,
)


DEFAULT_SOURCE_ROOT = protocol.REPO_ROOT / "data" / "gift-eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify real-anchored history-only contracts against local "
            "GIFT-Eval data and emit a JSON summary on stdout."
        )
    )
    parser.add_argument(
        "--dataset-id",
        dest="dataset_ids",
        action="append",
        required=True,
        help="CaFE GIFT-Eval dataset ID; repeat to qualify multiple datasets.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root containing the downloaded GIFT-Eval config directories.",
    )
    parser.add_argument(
        "--maximum-backgrounds",
        type=int,
        default=8,
        help="Maximum authentic L504+H48 windows per dataset.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=2,
        help="Number of deterministic paired seeds to audit when available.",
    )
    return parser.parse_args()


def _counts(values: Iterable[object]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(value) for value in values).items())
    )


def _finite_summary(values: Sequence[float]) -> dict[str, float] | None:
    finite = np.asarray(
        [value for value in values if np.isfinite(float(value))],
        dtype=float,
    )
    if finite.size == 0:
        return None
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
    }


def _resolution_status_counts(
    rows: Sequence[Mapping[str, Any]],
    resolution_key: str,
) -> dict[str, int]:
    statuses: list[str] = []
    for row in rows:
        resolution = row.get(resolution_key)
        if not isinstance(resolution, Mapping):
            statuses.append("not_evaluated")
        elif resolution.get("available") is True:
            statuses.append("available")
        else:
            statuses.append(
                str(resolution.get("unavailable_reason", "unavailable"))
            )
    return _counts(statuses)


def _available_resolution_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    resolution_key: str,
    value_key: str,
) -> list[object]:
    values: list[object] = []
    for row in rows:
        resolution = row.get(resolution_key)
        if (
            isinstance(resolution, Mapping)
            and resolution.get("available") is True
            and resolution.get(value_key) is not None
        ):
            values.append(resolution[value_key])
    return values


def _audit_pairs(
    samples: Sequence[Mapping[str, Any]],
    backgrounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not samples:
        return {
            "status": "not_applicable",
            "reason_code": "no_available_real_anchored_cell",
            "sample_count": 0,
            "pair_count": 0,
            "checks": {},
            "passed": None,
        }
    by_background = {
        str(background["background_id"]): background
        for background in backgrounds
    }
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_pair[str(sample["counterfactual_pair_id"])].append(sample)
    exact_baselines = True
    shared_mase = True
    exact_pair_cardinality = True
    doses: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = defaultdict(
        list
    )
    for pair in by_pair.values():
        if len(pair) != 2:
            exact_pair_cardinality = False
            continue
        baseline_rows = [
            row for row in pair if int(row["counterfactual_member"]) == 0
        ]
        treatment_rows = [
            row for row in pair if int(row["counterfactual_member"]) == 1
        ]
        if len(baseline_rows) != 1 or len(treatment_rows) != 1:
            exact_pair_cardinality = False
            continue
        baseline = baseline_rows[0]
        treatment = treatment_rows[0]
        background = by_background[str(baseline["background_id"])]
        baseline_target = np.asarray(baseline["target"], dtype=float)
        expected_target = np.asarray(background["target"], dtype=float)
        treatment_target = np.asarray(treatment["target"], dtype=float)
        exact_baselines &= bool(np.array_equal(baseline_target, expected_target))
        shared_mase &= (
            baseline["mase_scale"] == treatment["mase_scale"]
            and baseline["mase_scale_by_target"]
            == treatment["mase_scale_by_target"]
            and baseline["shared_standardization"]
            == treatment["shared_standardization"]
        )
        doses[
            (str(treatment["capability_id"]), int(treatment["seed_index"]))
        ].append(
            (
                float(treatment["dose_value"]),
                treatment_target - baseline_target,
            )
        )
    monotone_dose_response = True
    proportional_dose_response = True
    nonlinear_dynamic_contract = True
    for (capability_id, _seed_index), rows in doses.items():
        rows.sort(key=lambda row: row[0])
        norms = [float(np.linalg.norm(delta)) for _alpha, delta in rows]
        monotone_dose_response &= all(
            left < right for left, right in zip(norms, norms[1:])
        )
        if rows and capability_id != "nonlinear_persistence":
            alpha, delta = rows[0]
            unit_delta = delta / (alpha - 1.0)
            proportional_dose_response &= all(
                np.allclose(
                    candidate / (candidate_alpha - 1.0),
                    unit_delta,
                    rtol=0.0,
                    atol=2e-14,
                )
                for candidate_alpha, candidate in rows[1:]
            )
        elif rows:
            nonlinear_rows = [
                row
                for pair in by_pair.values()
                for row in pair
                if row.get("capability_id") == "nonlinear_persistence"
                and int(row.get("seed_index", -1)) == _seed_index
                and int(row.get("counterfactual_member", -1)) == 1
            ]
            nonlinear_dynamic_contract &= bool(
                nonlinear_rows
                and all(
                    row.get("generation_metadata", {}).get(
                        "dose_response_law"
                    )
                    == "dynamic_recursive_nonproportional"
                    and row.get("generation_metadata", {}).get(
                        "future_innovation_policy"
                    )
                    == "zero_future_innovation_paired_rollout_v1"
                    for row in nonlinear_rows
                )
            )
    checks = {
        "exact_pair_cardinality": exact_pair_cardinality,
        "exact_public_background_baselines": exact_baselines,
        "shared_pair_mase_and_normalization": shared_mase,
        "strictly_monotone_alpha_response": monotone_dose_response,
        "linear_alpha_delta_scaling": proportional_dose_response,
        "nonlinear_dynamic_contract": nonlinear_dynamic_contract,
    }
    return {
        "status": "evaluated",
        "sample_count": len(samples),
        "pair_count": len(by_pair),
        "checks": checks,
        "passed": all(checks.values()),
    }


def qualify_dataset(
    dataset_id: str,
    *,
    source_root: Path,
    maximum_backgrounds: int,
    seed_count: int,
) -> dict[str, Any]:
    dataset = protocol.resolve_dataset(dataset_id)
    if not dataset.real_data_adapter.startswith("gift_"):
        raise ValueError(
            f"{dataset_id!r} is not backed by a GIFT-Eval real-data adapter"
        )
    candidate_backgrounds, background_metadata = (
        protocol.build_real_anchored_backgrounds(
            dataset,
            source_root=source_root,
            # The public count is an evaluation-bank cap.  Qualification gets
            # a separate, source-time-disjoint bank of the same maximum size.
            maximum_backgrounds=2 * maximum_backgrounds,
        )
    )
    backgrounds, reference_backgrounds, bank_split_audit = (
        split_real_anchored_background_banks(
            candidate_backgrounds,
            maximum_evaluation_backgrounds=maximum_backgrounds,
            maximum_reference_backgrounds=maximum_backgrounds,
            source_window_length=protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH,
        )
    )
    if not backgrounds or not reference_backgrounds:
        raise ValueError(
            f"{dataset_id!r} did not yield non-empty source-time-disjoint "
            "reference and evaluation banks"
        )
    reference_contract_rows, _reference_availability = (
        fit_background_capability_contracts(
            reference_backgrounds,
            capability_ids=REAL_ANCHORED_SUPPORTED_CAPABILITIES,
        )
    )
    qualification_policy = freeze_real_anchored_qualification_policy(
        reference_contract_rows,
        reference_background_ids=[
            str(row["background_id"]) for row in reference_backgrounds
        ],
        bank_split_audit=bank_split_audit,
    )
    contract_rows, availability = fit_background_capability_contracts(
        backgrounds,
        capability_ids=REAL_ANCHORED_SUPPORTED_CAPABILITIES,
        qualification_policy=qualification_policy,
    )
    validate_evaluation_qualification_policy(
        contract_rows,
        qualification_policy,
    )
    for row in contract_rows:
        validate_contract_integrity(row)
    public_backgrounds = [
        public_background(background) for background in backgrounds
    ]
    reconstruct_exact = all(
        np.array_equal(
            reconstruct_source_baseline(public),
            np.concatenate(
                (
                    np.asarray(private["_decomposition_history"], dtype=float),
                    np.asarray(private["target"], dtype=float)[
                        -protocol.HORIZON :, 0
                    ],
                )
            ),
        )
        for private, public in zip(
            backgrounds,
            public_backgrounds,
            strict=True,
        )
    )
    enabled = available_capabilities(availability)
    generation_arguments = {
        "capability_ids": enabled,
        "seed_indexes": range(seed_count),
        "alphas": REAL_ANCHORED_ALPHAS,
    }
    samples = list(
        iter_real_anchored_samples(
            public_backgrounds,
            contract_rows,
            **generation_arguments,
        )
    )
    deterministic_generation = samples == list(
        iter_real_anchored_samples(
            public_backgrounds,
            contract_rows,
            **generation_arguments,
        )
    )
    pair_audit = _audit_pairs(samples, public_backgrounds)
    contract_summary: dict[str, Any] = {}
    for capability_id in REAL_ANCHORED_SUPPORTED_CAPABILITIES:
        selected = [
            row
            for row in contract_rows
            if row["capability_id"] == capability_id
        ]
        eligible = [row for row in selected if row["available"] is True]
        modulation_periods = _available_resolution_values(
            selected,
            resolution_key="modulation_resolution",
            value_key="modulation_period",
        )
        regime_join_indexes = _available_resolution_values(
            selected,
            resolution_key="regime_resolution",
            value_key="regime_join_index",
        )
        contract_summary[capability_id] = {
            "eligible_background_count": len(eligible),
            "unavailable_reason_counts": _counts(
                row["unavailable_reason"]
                for row in selected
                if row["available"] is not True
            ),
            "controlled_component_rms": _finite_summary(
                [
                    float(row["controlled_component_rms"])
                    for row in eligible
                    if row.get("controlled_component_rms") is not None
                ]
            ),
            "controlled_component_future_rms": _finite_summary(
                [
                    float(row["controlled_component_future_rms"])
                    for row in eligible
                    if row.get("controlled_component_future_rms") is not None
                ]
            ),
            "secondary_period_counts": _counts(
                round(float(period), 6)
                for row in eligible
                if isinstance(row.get("period_resolution"), Mapping)
                for period in row["period_resolution"].get(
                    "secondary_periods",
                    [],
                )
            ),
            "modulation_resolution_status_counts": (
                _resolution_status_counts(
                    selected,
                    "modulation_resolution",
                )
            ),
            "modulation_period_counts": _counts(
                round(float(period), 6) for period in modulation_periods
            ),
            "regime_resolution_status_counts": _resolution_status_counts(
                selected,
                "regime_resolution",
            ),
            "regime_join_index_counts": _counts(
                int(index) for index in regime_join_indexes
            ),
        }
    integrity_checks = {
        "public_background_reconstruction": reconstruct_exact,
        "contract_integrity": True,
        "deterministic_generation": deterministic_generation,
        "paired_generation": pair_audit["passed"] is not False,
    }
    return {
        "dataset_id": dataset.dataset_id,
        "config_id": dataset.config_id,
        "real_data_adapter": dataset.real_data_adapter,
        "source_candidate_background_count": len(candidate_backgrounds),
        "accepted_background_count": len(backgrounds),
        "reference_background_count": len(reference_backgrounds),
        "requested_background_limit": maximum_backgrounds,
        "qualification_threshold_source": (
            QUALIFICATION_THRESHOLD_SOURCE_POLICY
        ),
        "qualification_policy_sha256": qualification_policy[
            "qualification_policy_sha256"
        ],
        "reference_bank_split_audit": bank_split_audit,
        "official_holdout": background_metadata["official_holdout"],
        "rejection_counts": {
            key: background_metadata[key]
            for key in (
                "rejected_fit_missing_count",
                "rejected_future_missing_count",
                "rejected_uninformative_count",
            )
        },
        "feature_period_counts": _counts(
            background["feature_period"] for background in backgrounds
        ),
        "availability": availability["cells"],
        "contracts": contract_summary,
        "paired_generation_audit": pair_audit,
        "integrity_checks": integrity_checks,
        "integrity_passed": all(integrity_checks.values()),
    }


def main() -> int:
    arguments = parse_args()
    if arguments.maximum_backgrounds < 1:
        raise ValueError("--maximum-backgrounds must be positive")
    if arguments.seed_count < 1:
        raise ValueError("--seed-count must be positive")
    source_root = arguments.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    dataset_ids = tuple(dict.fromkeys(arguments.dataset_ids))
    datasets = [
        qualify_dataset(
            dataset_id,
            source_root=source_root,
            maximum_backgrounds=arguments.maximum_backgrounds,
            seed_count=arguments.seed_count,
        )
        for dataset_id in dataset_ids
    ]
    result = {
        "schema_version": "cafe.real_anchored_offline_qualification.v2",
        "source_root": str(source_root),
        "requested_dataset_ids": list(dataset_ids),
        "maximum_backgrounds": arguments.maximum_backgrounds,
        "seed_count": arguments.seed_count,
        "alphas": list(REAL_ANCHORED_ALPHAS),
        "datasets": datasets,
        "integrity_passed": all(
            dataset["integrity_passed"] for dataset in datasets
        ),
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
