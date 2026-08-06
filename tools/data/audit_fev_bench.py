#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx

from cafe import provenance
from cafe.data.fev_audit import build_fev_metadata_audit
from cafe.data.fev_audit import config_inventory_csv
from cafe.data.fev_audit import parse_fev_readme
from cafe.data.fev_audit import parse_fev_tasks
from cafe.data.fev_audit import task_matrix_csv
from cafe.data.fev_audit import write_jsonl
from cafe.data.fev_bench import FEV_DATASET_REPOSITORY
from cafe.data.fev_bench import FEV_DATASET_REVISION
from cafe.data.fev_bench import FEV_TASK_REPOSITORY
from cafe.data.fev_bench import FEV_TASK_REVISION
from cafe.data.fev_bench import FEV_TASKS_SHA256
from cafe.protocol import DEFAULT_CALIBRATION_PATH_COUNT
from cafe.protocol import REPO_ROOT
from cafe.protocol import canonical_json
from cafe.protocol import utc_now


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runtime" / "fev_bench_audits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all official FEV-Bench task metadata without downloading "
            "the Parquet datasets or running CaFE calibration."
        )
    )
    parser.add_argument("--audit-id", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--maximum-anchors", type=int, default=256)
    parser.add_argument(
        "--calibration-path-count",
        type=int,
        default=DEFAULT_CALIBRATION_PATH_COUNT,
    )
    return parser.parse_args()


def _client() -> httpx.Client:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return httpx.Client(
        follow_redirects=True,
        timeout=120.0,
        proxy=proxy,
        trust_env=False,
    )


def _get_text(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    return response.text


def _get_json(client: httpx.Client, url: str) -> Any:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _human_bytes(value: int) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or suffix == "TiB":
            return f"{amount:.2f} {suffix}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _report_markdown(audit: dict[str, Any], provenance_record: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# FEV-Bench full metadata audit",
        "",
        "This is a metadata-only Phase 1 report. No full Parquet dataset was "
        "downloaded and no CaFE calibration or generation stage was started.",
        "",
        "## Frozen sources",
        "",
        f"- Dataset revision: `{provenance_record['dataset_revision']}`",
        f"- Task revision: `{provenance_record['task_revision']}`",
        f"- Task YAML SHA256: `{provenance_record['tasks_sha256']}`",
        f"- Git revision: `{provenance_record['code']['git_revision']}`",
        "",
        "## Inventory",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Unique benchmark configs: {summary['config_count']}",
        f"- Parquet files: {summary['parquet_file_count']}",
        f"- Estimated download: {_human_bytes(summary['download_bytes'])}",
        f"- Declared in-memory dataset bytes: "
        f"{_human_bytes(summary['dataset_bytes'])}",
        f"- Declared observations: {summary['observation_count']:,}",
        f"- Exact-frequency overlaps with current non-FEV CaFE sources: "
        f"{summary['source_overlap_config_count']}",
        f"- Configs sourced from GIFT-Eval: "
        f"{summary['gift_eval_source_config_count']}",
        f"- Pilot adapters already registered: "
        f"{summary['pilot_registered_config_count']}",
        f"- Metadata schema errors: {summary['schema_error_task_count']}",
        f"- Multivariate task views: {summary['multivariate_task_count']}",
        f"- Task views with known dynamic covariates: "
        f"{summary['known_dynamic_task_count']}",
        f"- Task views with any covariate type: "
        f"{summary['any_covariate_task_count']}",
        f"- Task views requiring categorical known-covariate scans: "
        f"{summary['categorical_known_task_count']}",
        f"- Configs with median length below 216: "
        f"{summary['median_below_window_config_count']}",
        "",
        "## Workload estimate",
        "",
        f"- Metadata-candidate task/capability cells: "
        f"{summary['metadata_candidate_capability_cells']}",
        f"- Cells requiring a full length scan: "
        f"{summary['conditional_length_scan_cells']}",
        f"- Default qualification-path units before data-level rejection: "
        f"{summary['default_qualification_path_units']:,}",
        "",
        "Candidate means only that task metadata is compatible. Phase 2 must "
        "still scan actual sequence lengths, missingness, finite-window support, "
        "categorical levels, and realized structural feature support.",
        "",
        "## Capability status counts",
        "",
        "| Capability | Status counts |",
        "|---|---|",
    ]
    for capability, counts in summary["capability_status_counts"].items():
        rendered = ", ".join(
            f"{status}={count}" for status, count in counts.items()
        )
        lines.append(f"| {capability} | {rendered} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `task_capability_matrix.csv`: one row per official task view.",
            "- `config_inventory.csv`: 96-config aggregate with duplicate flags.",
            "- `task_inventory.jsonl`: complete normalized task metadata.",
            "- `download_manifest.jsonl`: pinned Parquet paths, sizes, and hashes.",
            "- `metadata/`: exact task YAML, dataset README, and repository tree.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.maximum_anchors < 1 or args.calibration_path_count < 1:
        raise ValueError("audit workload budgets must be positive")
    output_dir = args.output_root.resolve() / args.audit_id
    if output_dir.exists():
        raise FileExistsError(
            f"FEV audit output already exists and is immutable: {output_dir}"
        )

    tasks_url = (
        "https://raw.githubusercontent.com/"
        f"{FEV_TASK_REPOSITORY}/{FEV_TASK_REVISION}/"
        "benchmarks/fev_bench/tasks.yaml"
    )
    readme_url = (
        "https://huggingface.co/datasets/"
        f"{FEV_DATASET_REPOSITORY}/raw/{FEV_DATASET_REVISION}/README.md"
    )
    tree_url = (
        "https://huggingface.co/api/datasets/"
        f"{FEV_DATASET_REPOSITORY}/tree/{FEV_DATASET_REVISION}"
        "?recursive=true&expand=false&limit=1000"
    )
    with _client() as client:
        tasks_text = _get_text(client, tasks_url)
        readme_text = _get_text(client, readme_url)
        tree_entries = _get_json(client, tree_url)
    tasks_digest = _sha256(tasks_text.encode("utf-8"))
    if tasks_digest != FEV_TASKS_SHA256:
        raise ValueError(
            f"FEV task YAML checksum mismatch: expected {FEV_TASKS_SHA256}, "
            f"got {tasks_digest}"
        )
    if not isinstance(tree_entries, list):
        raise ValueError("FEV repository tree response is not a list")

    tasks = parse_fev_tasks(tasks_text)
    readme_metadata, statistics = parse_fev_readme(readme_text)
    audit = build_fev_metadata_audit(
        tasks=tasks,
        readme_metadata=readme_metadata,
        statistics=statistics,
        tree_entries=tree_entries,
        maximum_anchors=args.maximum_anchors,
        calibration_path_count=args.calibration_path_count,
    )
    code = provenance.code_provenance(REPO_ROOT)
    provenance_record = {
        "created_at": utc_now(),
        "dataset_repository": FEV_DATASET_REPOSITORY,
        "dataset_revision": FEV_DATASET_REVISION,
        "dataset_readme_sha256": _sha256(readme_text.encode("utf-8")),
        "task_repository": FEV_TASK_REPOSITORY,
        "task_revision": FEV_TASK_REVISION,
        "tasks_sha256": tasks_digest,
        "tree_response_sha256": _sha256(
            canonical_json(tree_entries).encode("utf-8")
        ),
        "code": code,
        "policy": {
            "metadata_only": True,
            "parquet_downloaded": False,
            "calibration_started": False,
            "generation_started": False,
            "minimum_window_length": 216,
            "minimum_finite_window_count": 12,
            "maximum_anchors": args.maximum_anchors,
            "calibration_path_count": args.calibration_path_count,
        },
    }
    output_dir.mkdir(parents=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "tasks.yaml").write_text(tasks_text, encoding="utf-8")
    (metadata_dir / "dataset_README.md").write_text(
        readme_text,
        encoding="utf-8",
    )
    (metadata_dir / "tree.json").write_text(
        json.dumps(tree_entries, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "task_inventory.jsonl", audit["task_rows"])
    write_jsonl(output_dir / "download_manifest.jsonl", audit["download_files"])
    (output_dir / "task_capability_matrix.csv").write_text(
        task_matrix_csv(audit["task_rows"]),
        encoding="utf-8",
    )
    (output_dir / "config_inventory.csv").write_text(
        config_inventory_csv(audit["config_rows"]),
        encoding="utf-8",
    )
    (output_dir / "audit.json").write_text(
        json.dumps(
            {
                "schema_version": "cafe.fev_bench_metadata_audit.v1",
                "provenance": provenance_record,
                "summary": audit["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        _report_markdown(audit, provenance_record),
        encoding="utf-8",
    )
    print(
        canonical_json(
            {
                "audit_id": args.audit_id,
                "output_dir": str(output_dir),
                **audit["summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
