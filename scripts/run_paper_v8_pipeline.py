#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paper_v8_pipeline_common as v8
import run_paper_v8_inference as v8_inference

DEFAULT_OUTPUT_ROOT = v8.REPO_ROOT / "runtime" / "paper_exp" / "v8"
DEFAULT_MODELS = v8_inference.DEFAULT_MODELS
STEPS = ("calibration", "generation", "validation", "inference", "analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete formal Paper v8 pipeline."
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help=(
            "One registered dataset id. Repeat the flag to run several "
            "datasets. Defaults to gift_electricity_h."
        ),
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=None,
        help="Convenience form for passing several registered dataset ids.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "Immutable experiment directory name. When omitted, derive one "
            "from the generator version, protocol hash, and UTC start time."
        ),
    )
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=Path("/root/xmy/gift-eval"),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument(
        "--preparation-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Capability-level processes used by calibration and generation.",
    )
    parser.add_argument(
        "--calibration-seeds",
        type=int,
        default=v8.DEFAULT_CALIBRATION_PATH_COUNT,
    )
    parser.add_argument(
        "--max-calibration-seeds",
        type=int,
        default=v8.MAX_CALIBRATION_PATH_COUNT,
    )
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=v8.CAPABILITIES,
        default=list(v8.CAPABILITIES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=list(v8_inference.DEFAULT_ENDPOINTS),
    )
    v8_inference.add_endpoint_topology_arguments(parser)
    parser.add_argument("--start-at", choices=STEPS, default="calibration")
    parser.add_argument("--stop-after", choices=STEPS, default="analysis")
    parser.add_argument("--resume-inference", action="store_true")
    parser.add_argument(
        "--upgrade-inference-execution-policy",
        action="store_true",
        help=(
            "Explicitly migrate an existing pre-inference experiment from an "
            "older model execution/scheduling policy. An active preparation-"
            "only run may continue; refuses if that run can enter inference "
            "or after inference artifacts exist."
        ),
    )
    return parser.parse_args()


def run(script: str, arguments: list[str]) -> None:
    command = [sys.executable, str(v8.REPO_ROOT / "scripts" / script), *arguments]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=v8.REPO_ROOT / "backend", check=True)


def requested_dataset_ids(args: argparse.Namespace) -> list[str]:
    if args.dataset_id and args.dataset_ids:
        raise ValueError("use either --dataset-id or --dataset-ids, not both")
    values = list(args.dataset_ids or args.dataset_id or ["gift_electricity_h"])
    if len(values) != len(set(values)):
        raise ValueError("v8 dataset ids must be unique")
    for dataset_id in values:
        v8.resolve_dataset(dataset_id)
    return values


def commands_for_dataset(
    args: argparse.Namespace,
    dataset_id: str,
    *,
    experiment_root: Path,
) -> dict[str, tuple[str, list[str]]]:
    common = [
        "--dataset-id",
        dataset_id,
        "--output-root",
        str(experiment_root),
    ]
    seed = [
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
    ]
    return {
        "calibration": (
            "calibrate_paper_v8.py",
            [
                *common,
                "--gift-eval-dir",
                str(args.gift_eval_dir.resolve()),
                "--max-anchors",
                str(args.max_anchors),
                "--calibration-seeds",
                str(args.calibration_seeds),
                "--max-calibration-seeds",
                str(args.max_calibration_seeds),
                "--workers",
                str(args.preparation_workers),
                "--capabilities",
                *args.capabilities,
            ],
        ),
        "generation": (
            "generate_paper_v8_samples.py",
            [
                *common,
                *seed,
                "--workers",
                str(args.preparation_workers),
                "--capabilities",
                *args.capabilities,
            ],
        ),
        "validation": (
            "validate_paper_v8_samples.py",
            [*common, *seed],
        ),
        "inference": (
            "run_paper_v8_inference.py",
            [
                *common,
                *seed,
                "--models",
                *args.models,
                "--endpoints",
                *args.endpoints,
                *v8_inference.endpoint_topology_cli_arguments(args),
                *(["--resume"] if args.resume_inference else []),
            ],
        ),
        "analysis": (
            "analyze_paper_v8.py",
            [*common, *seed, "--models", *args.models],
        ),
    }


def protocol_config(
    args: argparse.Namespace,
    dataset_ids: list[str],
) -> dict[str, Any]:
    missing_configs = sorted(
        set(args.models) - set(v8_inference.MODEL_EXECUTION_CONFIG)
    )
    if missing_configs:
        raise ValueError(
            "missing model execution configs: " + ", ".join(missing_configs)
        )
    return {
        "schema_version": "paper_v8_experiment_protocol.v2",
        "pipeline_schema_version": v8.SCHEMA_VERSION,
        "generator_version": v8.GENERATOR_VERSION,
        "dataset_ids": list(dataset_ids),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "max_anchors": int(args.max_anchors),
        "calibration_seeds": int(args.calibration_seeds),
        "max_calibration_seeds": int(args.max_calibration_seeds),
        "calibration_path_policy": (
            "formal_generation_seed_bank_" "fixed_base_hard_failure_only_expansion_v2"
        ),
        "capabilities": list(args.capabilities),
        "models": list(args.models),
        "model_execution_config": {
            model_id: dict(v8_inference.MODEL_EXECUTION_CONFIG[model_id])
            for model_id in args.models
        },
        "dataset_execution_policy": (
            "sequential_in_declared_order_complete_each_before_next"
        ),
        "model_scheduling_policy": {
            "policy_id": v8_inference.SCHEDULING_POLICY_ID,
            "phase_order": "models_in_declared_order",
            "service_collaboration": (
                "all_compatible_services_run_deterministic_parts_of_each_model"
            ),
            "resume_part_identity": "preserved_when_service_count_changes",
        },
        "real_calibration_context_length": (
            v8.REAL_CALIBRATION_CONTEXT_LENGTH
        ),
        "synthetic_master_context_length": v8.CONTEXT_LENGTH,
        "fixed_context_length": v8.FIXED_CONTEXT_LENGTH,
        "horizon": v8.HORIZON,
        "view_context_lengths": list(v8.VIEW_CONTEXT_LENGTHS),
        "intensities": list(v8.INTENSITIES),
        "aggregation_policy": (
            "dataset-isolated outputs and reports; no implicit "
            "cross-dataset averaging"
        ),
    }


def default_experiment_id(
    protocol_sha256: str,
    *,
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    generator_tag = v8.safe_id(v8.GENERATOR_VERSION)
    return f"v8_{generator_tag}_{protocol_sha256[:12]}_{timestamp}"


def code_provenance() -> dict[str, Any]:
    def git_value(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *arguments],
                cwd=v8.REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
    }


def initialize_experiment(
    *,
    storage_root: Path,
    experiment_id: str,
    protocol: dict[str, Any],
    endpoints: list[str],
    endpoint_profiles: dict[str, dict[str, Any]] | None = None,
    allow_inference_execution_upgrade: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if v8.safe_id(experiment_id) != experiment_id:
        raise ValueError("experiment-id may contain only letters, digits, '_' and '-'")
    experiment_root = storage_root.resolve() / experiment_id
    manifest_path = experiment_root / "experiment_manifest.json"
    protocol_sha256 = v8.json_sha256(protocol)
    manifest = {
        "schema_version": "paper_v8_experiment_manifest.v1",
        "experiment_id": experiment_id,
        "created_at": v8.utc_now(),
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
        "execution_environment": {
            "requested_endpoints": list(endpoints),
            "requested_endpoint_profiles": endpoint_profiles,
            **code_provenance(),
        },
        "storage": {
            "experiment_root": str(experiment_root),
            "dataset_layout": (
                "<dataset_id>/{01_calibration,02_generation,"
                "03_inference,04_analysis}"
            ),
            "seed_shards": "append-only files named by [seed_start, seed_end)",
            "cross_dataset_aggregation": "not_performed",
        },
    }
    if manifest_path.exists():
        existing = v8.read_json(manifest_path)
        exact_match = (
            existing.get("experiment_id") == experiment_id
            and existing.get("protocol_sha256") == protocol_sha256
            and existing.get("protocol") == protocol
        )
        if not exact_match and allow_inference_execution_upgrade:
            existing = upgrade_inference_execution_policy(
                experiment_root,
                existing,
                experiment_id=experiment_id,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
            )
            exact_match = True
        if not exact_match:
            raise ValueError(
                "existing experiment manifest does not match requested protocol"
            )
        return experiment_root, existing
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise ValueError(
            "refusing to use a non-empty experiment directory without "
            "an experiment manifest"
        )
    v8.write_json(manifest_path, manifest)
    return experiment_root, manifest


def upgrade_inference_execution_policy(
    experiment_root: Path,
    existing: dict[str, Any],
    *,
    experiment_id: str,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    """Explicitly migrate only the pre-inference execution policy.

    Generation and analysis identities remain immutable.  The migration is
    rejected once any inference file exists and records the complete prior
    protocol/hash for auditability.
    """

    if existing.get("experiment_id") != experiment_id:
        raise ValueError("cannot migrate a different experiment identity")
    old_protocol = dict(existing.get("protocol") or {})
    execution_keys = {
        "model_execution_config",
        "model_scheduling_policy",
    }
    old_identity = {
        key: value for key, value in old_protocol.items() if key not in execution_keys
    }
    new_identity = {
        key: value for key, value in protocol.items() if key not in execution_keys
    }
    if old_identity != new_identity:
        raise ValueError(
            "inference execution upgrade may not change generation or "
            "analysis protocol fields"
        )
    status_path = experiment_root / "pipeline_status.json"
    status: dict[str, Any] | None = None
    if status_path.exists():
        status = v8.read_json(status_path)
        if status.get("state") == "running":
            preparation_steps = {"calibration", "generation", "validation"}
            active_step = status.get("active_step")
            stop_after = status.get("stop_after")
            if (
                active_step not in preparation_steps
                or stop_after not in preparation_steps
                or status.get("protocol_sha256") != existing.get("protocol_sha256")
            ):
                raise ValueError(
                    "active pipeline may enter inference or does not match "
                    "the recorded protocol; wait before upgrading the "
                    "inference execution policy"
                )
    inference_files = [
        path for path in experiment_root.glob("*/03_inference/**/*") if path.is_file()
    ]
    if inference_files:
        raise ValueError(
            "cannot upgrade inference execution policy after inference "
            "artifacts exist"
        )
    upgraded = dict(existing)
    history = list(upgraded.get("protocol_history") or [])
    history.append(
        {
            "changed_at": v8.utc_now(),
            "reason": "explicit_pre_inference_execution_policy_upgrade",
            "protocol_sha256": existing.get("protocol_sha256"),
            "protocol": old_protocol,
            "concurrent_preparation_status": (
                status
                if status is not None and status.get("state") == "running"
                else None
            ),
        }
    )
    upgraded["protocol_history"] = history
    upgraded["protocol"] = protocol
    upgraded["protocol_sha256"] = protocol_sha256
    environment = dict(upgraded.get("execution_environment") or {})
    environment["inference_execution_upgrade"] = code_provenance()
    upgraded["execution_environment"] = environment
    v8.write_json(
        experiment_root / "experiment_manifest.json",
        upgraded,
    )
    return upgraded


def write_pipeline_status(
    experiment_root: Path,
    *,
    experiment_id: str,
    protocol_sha256: str,
    state: str,
    start_at: str,
    stop_after: str,
    completed: list[dict[str, Any]],
    active_dataset_id: str | None = None,
    active_step: str | None = None,
    error: str | None = None,
) -> None:
    v8.write_json(
        experiment_root / "pipeline_status.json",
        {
            "schema_version": "paper_v8_pipeline_status.v1",
            "updated_at": v8.utc_now(),
            "experiment_id": experiment_id,
            "protocol_sha256": protocol_sha256,
            "state": state,
            "start_at": start_at,
            "stop_after": stop_after,
            "active_dataset_id": active_dataset_id,
            "active_step": active_step,
            "completed": completed,
            "error": error,
        },
    )


def main() -> int:
    args = parse_args()
    dataset_ids = requested_dataset_ids(args)
    if len(args.models) != len(set(args.models)):
        raise ValueError("model ids must be unique")
    if len(args.endpoints) != len(set(args.endpoints)):
        raise ValueError("inference endpoints must be unique")
    endpoint_presets = v8_inference.endpoint_presets_with_defaults(
        list(args.endpoints),
        list(args.endpoint_preset),
    )
    endpoint_profiles = v8_inference.build_endpoint_profiles(
        list(args.endpoints),
        default_devices=args.devices,
        endpoint_presets=endpoint_presets,
        endpoint_devices=list(args.endpoint_devices),
        endpoint_capacities=list(args.endpoint_capacity),
        endpoint_concurrency_scales=list(args.endpoint_concurrency_scale),
        endpoint_model_capacities=list(args.endpoint_model_capacity),
        endpoint_model_concurrencies=list(args.endpoint_model_concurrency),
    )
    if args.seed_start < 0 or args.seed_count < 1:
        raise ValueError("seed_start must be non-negative and seed_count positive")
    if args.preparation_workers < 1:
        raise ValueError("preparation_workers must be positive")
    if (
        args.max_anchors < 1
        or args.calibration_seeds < 1
        or args.max_calibration_seeds < args.calibration_seeds
    ):
        raise ValueError(
            "anchor and calibration path budgets must be positive and "
            "maximums must not be smaller than base counts"
        )
    start = STEPS.index(args.start_at)
    stop = STEPS.index(args.stop_after)
    if stop < start:
        raise ValueError("stop-after must not precede start-at")
    protocol = protocol_config(args, dataset_ids)
    protocol_sha256 = v8.json_sha256(protocol)
    experiment_id = args.experiment_id or default_experiment_id(protocol_sha256)
    experiment_root, manifest = initialize_experiment(
        storage_root=args.output_root,
        experiment_id=experiment_id,
        protocol=protocol,
        endpoints=list(args.endpoints),
        endpoint_profiles={
            endpoint: profile.as_dict()
            for endpoint, profile in endpoint_profiles.items()
        },
        allow_inference_execution_upgrade=(args.upgrade_inference_execution_policy),
    )
    completed: list[dict[str, Any]] = []
    write_pipeline_status(
        experiment_root,
        experiment_id=experiment_id,
        protocol_sha256=manifest["protocol_sha256"],
        state="running",
        start_at=args.start_at,
        stop_after=args.stop_after,
        completed=completed,
    )
    try:
        for dataset_id in dataset_ids:
            commands = commands_for_dataset(
                args,
                dataset_id,
                experiment_root=experiment_root,
            )
            completed_steps: list[str] = []
            for step in STEPS[start : stop + 1]:
                write_pipeline_status(
                    experiment_root,
                    experiment_id=experiment_id,
                    protocol_sha256=manifest["protocol_sha256"],
                    state="running",
                    start_at=args.start_at,
                    stop_after=args.stop_after,
                    completed=completed,
                    active_dataset_id=dataset_id,
                    active_step=step,
                )
                script, arguments = commands[step]
                run(script, arguments)
                completed_steps.append(step)
            completed.append(
                {
                    "dataset_id": dataset_id,
                    "steps": completed_steps,
                    "output_dir": str(experiment_root / dataset_id),
                }
            )
    except Exception as error:
        write_pipeline_status(
            experiment_root,
            experiment_id=experiment_id,
            protocol_sha256=manifest["protocol_sha256"],
            state="failed",
            start_at=args.start_at,
            stop_after=args.stop_after,
            completed=completed,
            active_dataset_id=dataset_id,
            active_step=step,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    write_pipeline_status(
        experiment_root,
        experiment_id=experiment_id,
        protocol_sha256=manifest["protocol_sha256"],
        state="complete",
        start_at=args.start_at,
        stop_after=args.stop_after,
        completed=completed,
    )
    print(
        v8.canonical_json(
            {
                "experiment_id": experiment_id,
                "protocol_sha256": manifest["protocol_sha256"],
                "dataset_count": len(dataset_ids),
                "output": str(experiment_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
