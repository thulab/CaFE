from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pytest

from cafe import protocol
from cafe.generation import runner
from cafe.generation.real_counterfactuals import (
    build_availability,
    real_anchored_assignments,
)
from cafe.generation.real_anchored_dose import (
    additive_dose_reference,
    resolve_contract_dose_calibration,
)
from cafe.generation.real_anchored_policy import (
    REAL_ANCHORED_CANONICAL_STRENGTH_GRID,
)
from cafe.generation.reference_bank import (
    build_combined_real_anchored_bank_split_audit,
    freeze_real_anchored_qualification_policy,
    split_real_anchored_background_banks,
)
from cafe.generation.structural_real_counterfactuals import (
    build_structural_donor_commitment_manifest,
)


DATASET_ID = "gift_electricity_h"
CAPABILITY_ID = "trend"


def _write_calibration_bundle(
    output_root: Path,
    *,
    pipeline_schema_version: str,
    include_real_anchored: bool,
    synthetic_available: bool,
) -> Path:
    calibration_dir = output_root / DATASET_ID / "01_calibration"
    anchors_path = calibration_dir / "anchors.jsonl"
    real_anchors_path = calibration_dir / "real_anchor_masters.jsonl"
    calibration_path = calibration_dir / "capability_calibration.json"
    protocol.write_jsonl(anchors_path, ())
    protocol.write_jsonl(real_anchors_path, ())
    protocol.write_json(
        calibration_path,
        {
            "capabilities": {
                CAPABILITY_ID: {
                    "available_for_generation": synthetic_available,
                    "availability_status": (
                        "available" if synthetic_available else "unavailable"
                    ),
                    "unavailable_reason_codes": (
                        [] if synthetic_available else ["fixture_synthetic_off"]
                    ),
                    "intensity_calibration_scope": (
                        "dataset_real_generator_overlap_reference"
                        if synthetic_available
                        else None
                    ),
                }
            }
        },
    )
    files = {
        "anchors": protocol.file_record(anchors_path),
        "real_anchor_masters": protocol.file_record(real_anchors_path),
        "capability_calibration": protocol.file_record(calibration_path),
    }
    reference_backgrounds: list[dict[str, Any]] = []
    reference_contracts: list[dict[str, Any]] = []
    combined_bank_split_audit: dict[str, Any] | None = None
    if include_real_anchored:
        backgrounds_path = calibration_dir / "real_anchored_backgrounds.jsonl"
        contracts_path = calibration_dir / "real_anchored_contracts.jsonl"
        availability_path = calibration_dir / "real_anchored_availability.json"
        candidate_backgrounds = [
            {
                "dataset_id": DATASET_ID,
                "background_id": f"background_{index}",
                "item_id": f"item_{index}",
                "decomposition_start": 0,
            }
            for index in range(8)
        ]
        if pipeline_schema_version == protocol.SCHEMA_VERSION:
            backgrounds, reference_backgrounds, base_split_audit = (
                split_real_anchored_background_banks(
                    candidate_backgrounds,
                    maximum_evaluation_backgrounds=4,
                    maximum_reference_backgrounds=4,
                    source_window_length=(
                        protocol.REAL_ANCHORED_SOURCE_WINDOW_LENGTH
                    ),
                )
            )
            combined_bank_split_audit = (
                build_combined_real_anchored_bank_split_audit(
                    backgrounds,
                    reference_backgrounds,
                    base_split_audit=base_split_audit,
                )
            )
        else:
            backgrounds = candidate_backgrounds[:4]
        background_ids = [str(row["background_id"]) for row in backgrounds]
        contracts = [
            {
                "schema_version": (
                    "cafe.real_anchored_background_capability.v2"
                ),
                "dataset_id": DATASET_ID,
                "background_id": background_id,
                "capability_id": CAPABILITY_ID,
                "benchmark_track": "real_anchored_counterfactual",
                "qualification_policy_id": "fixture.trend.v1",
                "qualification_threshold_source": (
                    "independent_source_time_disjoint_reference_bank_never_"
                    "evaluation_origins_v1"
                ),
                "qualification_thresholds": {},
                "available": True,
                "unavailable_reason": None,
                "contract": {},
            }
            for background_id in background_ids
        ]
        if reference_backgrounds:
            reference_contracts = [
                {
                    **contracts[0],
                    "background_id": str(background["background_id"]),
                    "dose_design_reference": additive_dose_reference(
                        capability_id=CAPABILITY_ID,
                        background_id=str(background["background_id"]),
                        unit_gain_history_separation=0.12,
                        unit_gain_future_separation=0.24,
                        affected_channel_indices=(0,),
                    ),
                }
                for background in reference_backgrounds
            ]
        availability = build_availability(
            contracts,
            requested_capability_ids=(CAPABILITY_ID,),
            minimum_eligible_backgrounds=4,
        )
        if pipeline_schema_version == "cafe.pipeline.v2":
            availability["schema_version"] = (
                "cafe.real_anchored_availability.v1"
            )
            for cell in availability["cells"]:
                gate = cell["controlled_component_rms_gate"]
                for field in (
                    "visible_history_source",
                    "visible_history_minimum_rms_ratios",
                    "visible_context_lengths",
                    "visible_history_rms_range",
                    "visible_history_threshold_range",
                ):
                    gate.pop(field, None)
        protocol.write_jsonl(backgrounds_path, backgrounds)
        protocol.write_jsonl(contracts_path, contracts)
        protocol.write_json(availability_path, availability)
        files.update(
            {
                "real_anchored_backgrounds": protocol.file_record(
                    backgrounds_path
                ),
                "real_anchored_contracts": protocol.file_record(
                    contracts_path
                ),
                "real_anchored_availability": protocol.file_record(
                    availability_path
                ),
            }
        )
    if pipeline_schema_version == protocol.SCHEMA_VERSION:
        for name in (
            "structural_real_anchored_backgrounds",
            "structural_real_anchored_contracts",
            "structural_real_anchored_reference_backgrounds",
            "structural_hierarchy_qualification",
        ):
            path = calibration_dir / f"{name}.jsonl"
            protocol.write_jsonl(path, ())
            files[name] = protocol.file_record(path)
        donor_commitment_path = (
            calibration_dir
            / "structural_real_anchored_donor_commitments.json"
        )
        protocol.write_json(
            donor_commitment_path,
            build_structural_donor_commitment_manifest(
                (),
                (),
                dataset_id=DATASET_ID,
            ),
        )
        files["structural_real_anchored_donor_commitments"] = (
            protocol.file_record(donor_commitment_path)
        )
        assert combined_bank_split_audit is not None
        qualification_policy = freeze_real_anchored_qualification_policy(
            reference_contracts,
            reference_background_ids=[
                str(row["background_id"])
                for row in reference_backgrounds
            ],
            bank_split_audit=combined_bank_split_audit,
        )
        for name, payload in (
            ("real_anchored_reference_backgrounds", reference_backgrounds),
            ("real_anchored_reference_contracts", reference_contracts),
        ):
            path = calibration_dir / f"{name}.jsonl"
            protocol.write_jsonl(path, payload)
            files[name] = protocol.file_record(path)
        if include_real_anchored:
            for row in contracts:
                row["qualification_policy_sha256"] = (
                    qualification_policy["qualification_policy_sha256"]
                )
                evidence = additive_dose_reference(
                    capability_id=CAPABILITY_ID,
                    background_id=str(row["background_id"]),
                    unit_gain_history_separation=0.12,
                    unit_gain_future_separation=0.24,
                    affected_channel_indices=(0,),
                )
                row["dose_design_reference"] = evidence
                row["dose_calibration"] = resolve_contract_dose_calibration(
                    qualification_policy[
                    "capabilities"
                    ][CAPABILITY_ID]["dose_calibration"],
                    evidence,
                )
            protocol.write_jsonl(contracts_path, contracts)
            files["real_anchored_contracts"] = protocol.file_record(
                contracts_path
            )
            availability = build_availability(
                contracts,
                requested_capability_ids=(CAPABILITY_ID,),
                minimum_eligible_backgrounds=4,
            )
            protocol.write_json(availability_path, availability)
            files["real_anchored_availability"] = protocol.file_record(
                availability_path
            )
        for name, payload in (
            (
                "structural_real_anchored_availability",
                {"schema_version": "fixture", "cells": []},
            ),
            (
                "real_anchored_bank_split_audit",
                combined_bank_split_audit,
            ),
            (
                "real_anchored_qualification_policy",
                qualification_policy,
            ),
        ):
            path = calibration_dir / f"{name}.json"
            protocol.write_json(path, payload)
            files[name] = protocol.file_record(path)
    bundle = {
        "schema_version": (
            "cafe.calibration_bundle.v1"
            if pipeline_schema_version == "cafe.pipeline.v1"
            else (
                "cafe.calibration_bundle.v5"
                if pipeline_schema_version == protocol.SCHEMA_VERSION
                else "cafe.calibration_bundle.v2"
            )
        ),
        "pipeline_schema_version": pipeline_schema_version,
        "dataset": {"dataset_id": DATASET_ID},
        "source": {"fixture": True},
        "files": files,
        "generator_version": protocol.GENERATOR_VERSION,
    }
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(calibration_dir / "calibration_bundle.json", bundle)
    return calibration_dir


def _fake_real_anchored_samples(
    _backgrounds: Sequence[Mapping[str, Any]],
    contract_rows: Sequence[Mapping[str, Any]],
    *,
    capability_ids: Iterable[str],
    seed_indexes: Iterable[int],
) -> Iterable[dict[str, Any]]:
    assignments = real_anchored_assignments(
        contract_rows,
        capability_ids=capability_ids,
        seed_indexes=seed_indexes,
    )
    for capability_id, rows in assignments.items():
        for seed_index, contract_row in rows:
            dose_calibration = contract_row["dose_calibration"]
            for dose_index, (strength, alpha) in enumerate(
                zip(
                    dose_calibration["strength_grid"],
                    dose_calibration["applied_alpha_grid"],
                    strict=True,
                ),
                start=1,
            ):
                for member in (0, 1):
                    yield {
                        "capability_id": capability_id,
                        "seed_index": seed_index,
                        "background_id": contract_row["background_id"],
                        "dose_index": dose_index,
                        "dose_parameter": "canonical_strength_lambda",
                        "dose_value": 0.0 if member == 0 else strength,
                        "paired_treatment_strength": strength,
                        "applied_alpha": 1.0 if member == 0 else alpha,
                        "dose_calibration_policy_sha256": dose_calibration[
                            "dose_policy_sha256"
                        ],
                        "counterfactual_member": member,
                    }


def _install_lightweight_runner_fakes(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "validate_contract_integrity",
        lambda _row: None,
    )
    monkeypatch.setattr(
        runner,
        "iter_real_anchored_samples",
        _fake_real_anchored_samples,
    )
    monkeypatch.setattr(
        runner.realism,
        "build_realism_gate_context",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy_summary={"fixture": True}
        ),
    )

    def fake_generate_clean_samples(
        *_args,
        output_path: Path,
        **_kwargs,
    ) -> tuple[int, list[dict[str, Any]]]:
        return protocol.write_jsonl(output_path, ()), []

    monkeypatch.setattr(
        runner,
        "generate_clean_samples",
        fake_generate_clean_samples,
    )


def _run_generation(
    monkeypatch,
    output_root: Path,
    *,
    seed_start: int,
    seed_count: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: argparse.Namespace(
            dataset_id=DATASET_ID,
            output_root=output_root,
            seed_start=seed_start,
            seed_count=seed_count,
            workers=1,
            capabilities=[CAPABILITY_ID],
            secondary_modulus=4,
            max_generation_attempts=1,
            near_distance_gate=False,
        ),
    )
    assert runner.main() == 0
    shard_name = f"seed_{seed_start:06d}_{seed_start + seed_count:06d}"
    generation_dir = output_root / DATASET_ID / "02_generation"
    manifest = protocol.read_json(
        generation_dir / f"manifest__{shard_name}.json"
    )
    availability = protocol.read_json(
        generation_dir / f"real_anchored_availability__{shard_name}.json"
    )
    rows = list(
        protocol.iter_jsonl(
            generation_dir
            / "sample_shards"
            / f"{shard_name}__real_anchored_counterfactual.jsonl"
        )
    )
    return manifest, availability, rows


def test_runner_without_replacement_counts_and_mapping_are_shard_invariant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_calibration_bundle(
        tmp_path,
        pipeline_schema_version=protocol.SCHEMA_VERSION,
        include_real_anchored=True,
        synthetic_available=False,
    )
    _install_lightweight_runner_fakes(monkeypatch)

    full_manifest, full_availability, full_rows = _run_generation(
        monkeypatch,
        tmp_path,
        seed_start=0,
        seed_count=10,
    )
    _first_manifest, first_availability, first_rows = _run_generation(
        monkeypatch,
        tmp_path,
        seed_start=0,
        seed_count=2,
    )
    _second_manifest, second_availability, second_rows = _run_generation(
        monkeypatch,
        tmp_path,
        seed_start=2,
        seed_count=8,
    )

    expected_full_count = (
        4 * len(REAL_ANCHORED_CANONICAL_STRENGTH_GRID) * 2
    )
    assert len(full_rows) == expected_full_count
    assert full_rows == first_rows + second_rows
    assert full_availability["requested_seed_indexes"] == list(range(10))
    assert full_availability["assigned_seed_indexes_by_capability"] == {
        CAPABILITY_ID: [0, 1, 2, 3]
    }
    assert full_availability["effective_background_count_by_capability"] == {
        CAPABILITY_ID: 4
    }
    assert full_availability["cells"][0]["status"] == "available"
    assert full_availability["cells"][0]["eligible_background_count"] == 4
    assert full_availability["background_sampling_policy"] == (
        "frozen_global_seed_ordinal_permutation_without_replacement_"
        "truncate_at_eligible_count_v1"
    )
    assert full_availability["generated_master_count"] == expected_full_count
    assert first_availability["assigned_seed_indexes_by_capability"] == {
        CAPABILITY_ID: [0, 1]
    }
    assert second_availability["assigned_seed_indexes_by_capability"] == {
        CAPABILITY_ID: [2, 3]
    }
    assert first_availability["generated_master_count"] == 20
    assert second_availability["generated_master_count"] == 20
    assert full_manifest["files"]["real_anchored_counterfactuals"][
        "row_count"
    ] == expected_full_count
    assert full_manifest["config"]["real_anchored_counterfactual"][
        "effective_background_count_by_capability"
    ] == {CAPABILITY_ID: 4}
    assert full_manifest["config"]["real_anchored_counterfactual"][
        "generated_capabilities"
    ] == [CAPABILITY_ID]
    real_config = full_manifest["config"][
        "real_anchored_counterfactual"
    ]
    assert real_config["canonical_strength_grid"] == list(
        REAL_ANCHORED_CANONICAL_STRENGTH_GRID
    )
    assert real_config["applied_alpha_grid_by_capability"][CAPABILITY_ID] == []
    assert real_config["applied_alpha_scope"] == (
        "contract_specific_history_only"
    )
    assert len(
        real_config["applied_alpha_range_by_capability"][CAPABILITY_ID]
    ) == 5
    assert len(
        real_config["dose_calibration_sha256_by_capability"][CAPABILITY_ID]
    ) == 64
    treatment_rows = [
        row for row in full_rows if row["counterfactual_member"] == 1
    ]
    assert {int(row["seed_index"]) for row in treatment_rows} == {0, 1, 2, 3}
    assert len({str(row["background_id"]) for row in treatment_rows}) == 4


def test_generation_rejects_bundle_rehashed_wrong_reference_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calibration_dir = _write_calibration_bundle(
        tmp_path,
        pipeline_schema_version=protocol.SCHEMA_VERSION,
        include_real_anchored=True,
        synthetic_available=False,
    )
    bundle_path = calibration_dir / "calibration_bundle.json"
    bundle = protocol.read_json(bundle_path)
    reference_path = Path(
        bundle["files"]["real_anchored_reference_backgrounds"]["path"]
    )
    assert list(protocol.iter_jsonl(reference_path))
    protocol.write_jsonl(reference_path, ())
    bundle["files"]["real_anchored_reference_backgrounds"] = (
        protocol.file_record(reference_path)
    )
    bundle["bundle_content_sha256"] = protocol.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    protocol.write_json(bundle_path, bundle)
    _install_lightweight_runner_fakes(monkeypatch)

    with pytest.raises(ValueError, match="split audit disagrees"):
        _run_generation(
            monkeypatch,
            tmp_path,
            seed_start=0,
            seed_count=1,
        )


def test_generation_accepts_legacy_v1_calibration_without_anchored_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_calibration_bundle(
        tmp_path,
        pipeline_schema_version="cafe.pipeline.v1",
        include_real_anchored=False,
        synthetic_available=True,
    )
    _install_lightweight_runner_fakes(monkeypatch)

    manifest, availability, rows = _run_generation(
        monkeypatch,
        tmp_path,
        seed_start=0,
        seed_count=1,
    )

    assert rows == []
    assert availability["cells"] == []
    assert availability["generated_capabilities"] == []
    assert availability["assigned_seed_indexes_by_capability"] == {}
    assert availability["effective_background_count_by_capability"] == {}
    assert availability["generated_master_count"] == 0
    assert manifest["files"]["real_anchored_counterfactuals"][
        "row_count"
    ] == 0
    assert manifest["config"]["real_anchored_counterfactual"][
        "calibrated_available_capabilities"
    ] == []
    assert manifest["config"]["capabilities"] == [CAPABILITY_ID]


def test_generation_accepts_immutable_v2_anchored_availability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_calibration_bundle(
        tmp_path,
        pipeline_schema_version="cafe.pipeline.v2",
        include_real_anchored=True,
        synthetic_available=True,
    )
    _install_lightweight_runner_fakes(monkeypatch)

    manifest, availability, rows = _run_generation(
        monkeypatch,
        tmp_path,
        seed_start=0,
        seed_count=1,
    )

    assert rows == []
    assert availability["schema_version"] == (
        "cafe.real_anchored_availability.v1"
    )
    assert manifest["files"]["real_anchored_counterfactuals"][
        "row_count"
    ] == 0
    assert manifest["config"]["real_anchored_counterfactual"][
        "calibrated_available_capabilities"
    ] == []
    assert manifest["config"]["real_anchored_counterfactual"][
        "legacy_upstream_component_policy"
    ] == "validated_but_not_regenerated_or_ranked_as_v5"
    assert manifest["config"]["capabilities"] == [CAPABILITY_ID]


def test_generation_rejects_v2_anchored_files_hidden_in_v1_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_calibration_bundle(
        tmp_path,
        pipeline_schema_version="cafe.pipeline.v1",
        include_real_anchored=True,
        synthetic_available=False,
    )
    _install_lightweight_runner_fakes(monkeypatch)

    with pytest.raises(
        ValueError,
        match="v1 calibration bundle must not declare",
    ):
        _run_generation(
            monkeypatch,
            tmp_path,
            seed_start=0,
            seed_count=1,
        )
