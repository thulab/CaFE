#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = REPO_ROOT / "runtime/paper_exp/v2/00_transfer_protocol_freeze"
GENERATOR_ARTIFACT_PATH = FREEZE_DIR / "generator_conditioning_artifact.json"
FEATURE_GATE_ARTIFACT_PATH = FREEZE_DIR / "feature_gate_artifact.json"
NEAR_DISTANCE_ARTIFACT_PATH = FREEZE_DIR / "near_distance_artifact.json"
FREEZE_MANIFEST_PATH = FREEZE_DIR / "manifest.json"
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-17-paper-v2-synthetic-real-transfer-protocol.md"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runtime/paper_exp/v2/E2_dynamic_stability"

SCHEMA_VERSION = "paper_e2_dynamic_stability.v2"
EXPERIMENT_VERSION = "v2"
EXPERIMENT_ID = "E2_dynamic_stability"
DEFAULT_ROUND_SEEDS = (
    2026071721,
    2026071722,
    2026071723,
    2026071724,
    2026071725,
)
DEFAULT_SAMPLES_PER_ROUND = 16
EXPECTED_PROFILE_COUNT = 9
EXPECTED_CAPABILITIES_PER_PROFILE = 6
EXPECTED_CONTEXT_LENGTH = 504
EXPECTED_HORIZON = 48
EXPECTED_SEASON_LENGTH = 24
EXPECTED_TARGET_DIM = 1


import run_paper_e2_dynamic_stability as base  # noqa: E402


def configure_base_module() -> None:
    """Reuse the audited v1 inference/statistics engine with v2 experiment inputs."""

    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.EXPERIMENT_VERSION = EXPERIMENT_VERSION
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    base.GENERATOR_ARTIFACT_PATH = GENERATOR_ARTIFACT_PATH
    base.FEATURE_GATE_ARTIFACT_PATH = FEATURE_GATE_ARTIFACT_PATH
    base.NEAR_DISTANCE_ARTIFACT_PATH = NEAR_DISTANCE_ARTIFACT_PATH
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.RUNNER_PATH = Path(__file__).resolve()


configure_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper-v2 E2 dynamic stability on six univariate capabilities "
            "conditioned by the frozen held-out GIFT profiles."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:10810")
    parser.add_argument("--api-prefix", default="/ai/api/v1")
    parser.add_argument("--models", nargs="+", default=list(base.DEFAULT_MODELS))
    parser.add_argument(
        "--round-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_ROUND_SEEDS),
    )
    parser.add_argument("--samples-per-round", type=int, default=DEFAULT_SAMPLES_PER_ROUND)
    parser.add_argument("--devices", default=base.DEFAULT_DEVICES)
    parser.add_argument(
        "--request-max-attempts",
        type=int,
        default=base.DEFAULT_REQUEST_MAX_ATTEMPTS,
    )
    parser.add_argument("--forecast-timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-load-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=base.DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "infer", "analyze"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-loaded", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base.validate_cli_args(args)
    artifacts = load_frozen_artifacts()
    config = experiment_config(args, artifacts["generator"])
    output_dir = args.output_dir.resolve()
    base.prepare_or_resume_output(output_dir, config=config, resume=args.resume)

    if args.stage in {"all", "generate"}:
        generate_samples_if_needed(output_dir, config=config, artifacts=artifacts)
    if args.stage in {"all", "infer"}:
        base.require_file(output_dir / "samples.jsonl")
        base.run_inference(output_dir, config=config, args=args)
    if args.stage in {"all", "analyze"}:
        base.require_file(output_dir / "samples.jsonl")
        summary = base.analyze_experiment(
            output_dir,
            config=config,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        base.write_json(output_dir / "summary.json", summary)
        (output_dir / "report.md").write_text(
            base.render_report(summary),
            encoding="utf-8",
        )
        base.write_manifest(output_dir, config=config)
        print(
            f"E2-v2 criteria: {summary['criteria']['passed_count']}/"
            f"{summary['criteria']['criterion_count']}, "
            f"overall={summary['criteria']['overall_passed']}",
            flush=True,
        )
    print(f"E2-v2 output: {output_dir}", flush=True)
    return 0


def load_frozen_artifacts() -> dict[str, dict[str, Any]]:
    required = {
        "generator": GENERATOR_ARTIFACT_PATH,
        "feature_gate": FEATURE_GATE_ARTIFACT_PATH,
        "near_distance": NEAR_DISTANCE_ARTIFACT_PATH,
        "manifest": FREEZE_MANIFEST_PATH,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "paper-v2 transfer freeze is incomplete: " + ", ".join(missing)
        )
    artifacts = {name: base.read_json(path) for name, path in required.items()}
    manifest = artifacts["manifest"]
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("transfer freeze manifest is not paper-v2")
    return artifacts


def experiment_config(
    args: argparse.Namespace,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    validate_frozen_design(artifact)
    config = base.experiment_config(args, artifact)
    config.update(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_version": EXPERIMENT_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "transfer_freeze": {
                "path": base.relative_path(FREEZE_DIR),
                "manifest_sha256": base.sha256_file(FREEZE_MANIFEST_PATH),
                "generator_conditioning_sha256": base.sha256_file(
                    GENERATOR_ARTIFACT_PATH
                ),
                "feature_gate_sha256": base.sha256_file(FEATURE_GATE_ARTIFACT_PATH),
                "near_distance_sha256": base.sha256_file(NEAR_DISTANCE_ARTIFACT_PATH),
            },
            "fixed_request_shape": {
                "context_length": EXPECTED_CONTEXT_LENGTH,
                "horizon": EXPECTED_HORIZON,
                "season_length": EXPECTED_SEASON_LENGTH,
                "target_dim": EXPECTED_TARGET_DIM,
            },
            "generation_policy": (
                "use the held-out dataset profile's frozen generator, feature-support, "
                "and near-distance artifacts; pair base seeds across intensities and models"
            ),
        }
    )
    return config


def validate_frozen_design(artifact: dict[str, Any]) -> None:
    profile_ids = list(artifact["config"]["online_conditioning_profile_ids"])
    if len(profile_ids) != EXPECTED_PROFILE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_PROFILE_COUNT} frozen profiles, got {len(profile_ids)}"
        )
    for profile_id in profile_ids:
        profile = artifact["profiles"][profile_id]
        observed_shape = (
            int(profile["context_length"]),
            int(profile["horizon"]),
            int(profile["season_length"]),
            int(profile["target_dim"]),
        )
        expected_shape = (
            EXPECTED_CONTEXT_LENGTH,
            EXPECTED_HORIZON,
            EXPECTED_SEASON_LENGTH,
            EXPECTED_TARGET_DIM,
        )
        if observed_shape != expected_shape:
            raise ValueError(
                f"{profile_id} has shape {observed_shape}, expected {expected_shape}"
            )
        capabilities = tuple(sorted(profile["capabilities"]))
        if len(capabilities) != EXPECTED_CAPABILITIES_PER_PROFILE:
            raise ValueError(
                f"{profile_id} has {len(capabilities)} capabilities, "
                f"expected {EXPECTED_CAPABILITIES_PER_PROFILE}"
            )


def generate_samples_if_needed(
    output_dir: Path,
    *,
    config: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    sample_path = output_dir / "samples.jsonl"
    if sample_path.exists():
        observed = base.count_jsonl(sample_path)
        expected = int(config["expected_generated_sample_count"])
        if observed != expected:
            raise ValueError(
                f"existing samples.jsonl is incomplete: observed={observed}, expected={expected}"
            )
        print(f"samples already complete: {observed}", flush=True)
        return

    temporary = output_dir / "samples.jsonl.in_progress"
    if temporary.exists():
        raise FileExistsError(
            f"partial sample file exists and is retained for diagnosis: {temporary}"
        )
    generator_artifact = artifacts["generator"]
    feature_gate_artifact = artifacts["feature_gate"]
    near_distance_artifact = artifacts["near_distance"]
    created = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for profile_id in config["online_conditioning_profile_ids"]:
            profile = generator_artifact["profiles"][profile_id]
            for capability_id in sorted(profile["capabilities"]):
                conditioning = base.resolve_generator_conditioning(
                    capability_id=capability_id,
                    profile_id=profile_id,
                    context_length=int(profile["context_length"]),
                    horizon=int(profile["horizon"]),
                    target_dim=int(profile["target_dim"]),
                    artifact=generator_artifact,
                )
                if conditioning is None:
                    raise RuntimeError(
                        f"missing conditioning for {profile_id}/{capability_id}"
                    )
                for round_index, round_seed in enumerate(
                    config["round_seeds"],
                    start=1,
                ):
                    for sample_index in range(
                        config["samples_per_round_per_cell"]
                    ):
                        sample_seed = base._seed_for(
                            int(round_seed),
                            f"{profile_id}:{capability_id}",
                            sample_index,
                        )
                        for intensity in base.INTENSITIES:
                            target, latent, covariates, features = (
                                base._generate_accepted_sample_values(
                                    capability_id,
                                    int(profile["context_length"])
                                    + int(profile["horizon"]),
                                    int(profile["context_length"]),
                                    int(profile["target_dim"]),
                                    int(profile["season_length"]),
                                    intensity,
                                    sample_seed,
                                    anchor_profile_id=profile_id,
                                    generator_conditioning=conditioning,
                                    generator_conditioning_artifact=generator_artifact,
                                    feature_gate_artifact=feature_gate_artifact,
                                    near_distance_artifact=near_distance_artifact,
                                    acceptance_profile_ids=(profile_id,),
                                )
                            )
                            row = base.sample_row(
                                profile=profile,
                                profile_id=profile_id,
                                capability_id=capability_id,
                                intensity=intensity,
                                round_index=round_index,
                                round_seed=int(round_seed),
                                sample_index=sample_index,
                                sample_seed=sample_seed,
                                target=target,
                                covariates=covariates,
                                features=features,
                                latent=latent,
                            )
                            handle.write(
                                json.dumps(
                                    row,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            created += 1
                            if created % 500 == 0:
                                print(
                                    f"generated {created}/"
                                    f"{config['expected_generated_sample_count']}",
                                    flush=True,
                                )
    os.replace(temporary, sample_path)
    if created != int(config["expected_generated_sample_count"]):
        raise AssertionError(f"unexpected generated sample count: {created}")
    print(f"generated samples complete: {created}", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
