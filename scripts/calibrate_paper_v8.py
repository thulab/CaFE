#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import paper_v8_pipeline_common as v8


DEFAULT_OUTPUT_ROOT = (
    v8.REPO_ROOT / "runtime" / "paper_exp" / "v8_test" / "full_pipeline"
)
DEFAULT_GIFT_EVAL_DIR = Path("/root/xmy/gift-eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the formal Paper v8 real-data calibration bundle."
    )
    parser.add_argument("--dataset-id", default="gift_electricity_h")
    parser.add_argument(
        "--gift-eval-dir",
        type=Path,
        default=DEFAULT_GIFT_EVAL_DIR,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument("--calibration-seeds", type=int, default=12)
    parser.add_argument("--minimum-observed-fraction", type=float, default=0.5)
    parser.add_argument(
        "--capabilities",
        nargs="+",
        choices=v8.CAPABILITIES,
        default=list(v8.CAPABILITIES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_anchors < 1 or args.calibration_seeds < 1:
        raise ValueError("anchor and calibration seed counts must be positive")
    if not 0.0 < args.minimum_observed_fraction <= 1.0:
        raise ValueError("minimum observed fraction must be in (0, 1]")
    dataset = v8.resolve_dataset(args.dataset_id)
    output_dir = args.output_root.resolve() / dataset.dataset_id / "01_calibration"
    anchors, source_metadata = v8.build_calibration_anchors(
        dataset,
        gift_eval_dir=args.gift_eval_dir.resolve(),
        maximum_anchors=args.max_anchors,
        minimum_observed_fraction=args.minimum_observed_fraction,
    )
    anchor_path = output_dir / "anchors.jsonl"
    v8.write_jsonl(anchor_path, anchors)
    capability_calibration = v8.calibrate_capabilities(
        dataset,
        anchors,
        calibration_seed_count=args.calibration_seeds,
        capability_ids=args.capabilities,
    )
    capability_path = output_dir / "capability_calibration.json"
    v8.write_json(capability_path, capability_calibration)
    bundle = {
        "schema_version": "paper_v8_calibration_bundle.v2",
        "created_at": v8.utc_now(),
        "pipeline_schema_version": v8.SCHEMA_VERSION,
        "generator_version": v8.GENERATOR_VERSION,
        "dataset": source_metadata["dataset"],
        "source": source_metadata,
        "anchor_count": len(anchors),
        "capabilities": list(args.capabilities),
        "feature_contract": {
            "background_features": (
                "direct finite feature row from one real L504 anchor"
            ),
            "univariate_primary_strength": (
                "dataset q10-q90 intersected with generator response support"
            ),
            "structural_primary_strength": (
                "fixed cross-dataset evenly spaced realized-strength grid"
            ),
            "structural_identifiability": "measured on generated samples only",
            "removed_features": ["future_abs_covariate_target_corr"],
            "time_scale_semantics": {
                "calendar_season_length": (
                    "frequency-derived provenance"
                ),
                "feature_period": (
                    "calendar season when two cycles fit L504, otherwise "
                    "observable profile dominant period"
                ),
                "generator_period": (
                    "capability-specific observable clipping of the direct "
                    "anchor profile dominant period"
                ),
                "mase_period": (
                    "calendar season when defined inside L504, otherwise "
                    "non-seasonal lag 1"
                ),
            },
        },
        "files": {
            "anchors": v8.file_record(anchor_path),
            "capability_calibration": v8.file_record(capability_path),
        },
    }
    bundle["bundle_content_sha256"] = v8.json_sha256(
        {
            "dataset": bundle["dataset"],
            "source": bundle["source"],
            "files": bundle["files"],
            "generator_version": bundle["generator_version"],
        }
    )
    v8.write_json(output_dir / "calibration_bundle.json", bundle)
    print(
        v8.canonical_json(
            {
                "dataset_id": dataset.dataset_id,
                "anchor_count": len(anchors),
                "output": str(output_dir),
                "bundle_content_sha256": bundle["bundle_content_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
