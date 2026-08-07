#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import httpx

from cafe import provenance
from cafe import protocol
from cafe.data.fev_bench import FEV_DATASET_REPOSITORY
from cafe.data.fev_bench import FEV_DATASET_REVISION
from cafe.data.fev_bench import FEV_TASK_REVISION
from cafe.data.fev_bench import FEV_TASKS_SHA256
from cafe.data.fev_qualification import qualification_matrix_csv
from cafe.data.fev_qualification import qualify_task_view
from cafe.data.fev_qualification import select_qualification_configs
from cafe.data.fev_qualification import summarize_qualification
from cafe.data.fev_qualification import write_jsonl


DEFAULT_PHASE1_DIR = (
    protocol.REPO_ROOT
    / "runtime"
    / "fev_bench_audits"
    / "fev-full-phase1-20260806t135540z"
)
DEFAULT_DATA_ROOT = protocol.REPO_ROOT / "data" / "fev-bench"
DEFAULT_OUTPUT_ROOT = protocol.REPO_ROOT / "runtime" / "fev_bench_qualifications"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Phase 1-pinned FEV corpus and run exact CaFE "
            "data/feature qualification without calibration or generation."
        )
    )
    parser.add_argument("--qualification-id", required=True)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--maximum-anchors", type=int, default=256)
    parser.add_argument("--minimum-observed-fraction", type=float, default=0.5)
    parser.add_argument(
        "--config-id",
        dest="config_ids",
        action="append",
        help=(
            "Qualify only this config from the frozen Phase 1 inventory; "
            "repeat for a deterministic subset. The default is all configs."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require all assets to exist locally and only verify/scan them.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _verify_phase1(
    phase1_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    audit_path = phase1_dir / "audit.json"
    task_path = phase1_dir / "task_inventory.jsonl"
    manifest_path = phase1_dir / "download_manifest.jsonl"
    for path in (audit_path, task_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase 1 artifact: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source = audit["provenance"]
    expected = {
        "dataset_revision": FEV_DATASET_REVISION,
        "task_revision": FEV_TASK_REVISION,
        "tasks_sha256": FEV_TASKS_SHA256,
    }
    actual = {key: source[key] for key in expected}
    if actual != expected:
        raise ValueError(
            f"Phase 1 source contract mismatch: expected={expected}, actual={actual}"
        )
    tasks = _read_jsonl(task_path)
    files = _read_jsonl(manifest_path)
    if len(tasks) != int(audit["summary"]["task_count"]):
        raise ValueError("Phase 1 task inventory count mismatch")
    if len(files) != int(audit["summary"]["parquet_file_count"]):
        raise ValueError("Phase 1 download manifest count mismatch")
    return audit, tasks, files


def _target_path(data_root: Path, file_row: dict[str, Any]) -> Path:
    configs = list(file_row["configs"])
    if len(configs) != 1:
        raise ValueError(
            f"Phase 2 requires one config per Parquet asset: {file_row['path']}"
        )
    return data_root / str(configs[0]) / Path(str(file_row["path"])).name


def _download_url(file_row: dict[str, Any]) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{FEV_DATASET_REPOSITORY}/resolve/{FEV_DATASET_REVISION}/"
        f"{file_row['path']}?download=true"
    )


def _verify_asset(path: Path, file_row: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != int(file_row["size_bytes"]):
        raise ValueError(
            f"size mismatch for {path}: expected {file_row['size_bytes']}, "
            f"got {actual_size}"
        )
    digest = _sha256(path)
    if digest != str(file_row["sha256"]):
        raise ValueError(
            f"checksum mismatch for {path}: expected {file_row['sha256']}, "
            f"got {digest}"
        )
    return {
        "config_id": str(file_row["configs"][0]),
        "source_path": str(file_row["path"]),
        "local_path": str(path),
        "size_bytes": actual_size,
        "sha256": digest,
        "source_url": _download_url(file_row),
    }


def _download_one(
    data_root: Path,
    file_row: dict[str, Any],
    *,
    skip_download: bool,
) -> dict[str, Any]:
    target = _target_path(data_root, file_row)
    if target.exists() or skip_download:
        record = _verify_asset(target, file_row)
        record["download_status"] = "reused_verified"
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=180.0,
                proxy=proxy,
                trust_env=False,
            ) as client:
                with client.stream("GET", _download_url(file_row)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_bytes():
                            output.write(chunk)
            _verify_asset(temporary, file_row)
            os.replace(temporary, target)
            record = _verify_asset(target, file_row)
            record["download_status"] = "downloaded_verified"
            return record
        except (httpx.HTTPError, OSError, ValueError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(float(attempt))
        finally:
            temporary.unlink(missing_ok=True)
    assert last_error is not None
    raise last_error


def _download_assets(
    data_root: Path,
    files: list[dict[str, Any]],
    *,
    workers: int,
    skip_download: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                data_root,
                row,
                skip_download=skip_download,
            ): row
            for row in files
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                protocol.canonical_json(
                    {
                        "phase": "download",
                        "completed": len(records),
                        "total": len(files),
                        "config_id": record["config_id"],
                        "status": record["download_status"],
                    }
                ),
                flush=True,
            )
    return sorted(records, key=lambda row: row["config_id"])


def _blocked_task_row(task: dict[str, Any], error: Exception) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    for capability_id, metadata_status in task["capability_status"].items():
        statuses[capability_id] = (
            str(metadata_status)
            if str(metadata_status).startswith("not_applicable")
            else "blocked_data_error"
        )
    return {
        "task_index": int(task["task_index"]),
        "task_view_id": str(task["task_view_id"]),
        "config_id": str(task["config_id"]),
        "frequency": str(task["frequency"]),
        "frequency_class": str(task["frequency_class"]),
        "target_columns": list(task["target_columns"]),
        "target_count": int(task["target_count"]),
        "known_dynamic_columns": list(task["known_dynamic_columns"]),
        "categorical_known_columns": list(task["categorical_known_columns"]),
        "categorical_scan": {},
        "source_is_gift_eval": bool(task["source_is_gift_eval"]),
        "existing_cafe_source_overlaps": list(
            task["existing_cafe_source_overlaps"]
        ),
        "asset_path": None,
        "asset_sha256": None,
        "asset_size_bytes": int(task["download_bytes"]),
        "native_record_count": 0,
        "minimum_length": 0,
        "median_length": 0.0,
        "maximum_length": 0,
        "stratum_count": 0,
        "accepted_anchor_count": 0,
        "anchor_error": f"{type(error).__name__}: {error}",
        "target_value_count": 0,
        "target_nonfinite_count": 0,
        "target_nonfinite_fraction": 0.0,
        "known_covariate_value_count": 0,
        "known_covariate_nonfinite_count": 0,
        "known_covariate_nonfinite_fraction": 0.0,
        "rejected_missing_count": 0,
        "rejected_uninformative_count": 0,
        "feature_qualification": {},
        "capability_status": statuses,
    }


def _human_bytes(value: int) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or suffix == "TiB":
            return f"{amount:.2f} {suffix}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _report(
    summary: dict[str, Any],
    provenance_record: dict[str, Any],
) -> str:
    lines = [
        "# FEV-Bench full data qualification",
        "",
        "Phase 2 downloaded and checksum-verified the Phase 1-pinned Parquet "
        "corpus, then reused CaFE's exact anchor and feature extraction logic. "
        "No response-curve calibration or synthetic generation was started.",
        "",
        "## Frozen inputs",
        "",
        f"- Phase 1 audit SHA256: `{provenance_record['phase1_audit_sha256']}`",
        f"- Dataset revision: `{provenance_record['dataset_revision']}`",
        f"- Task revision: `{provenance_record['task_revision']}`",
        f"- Git revision: `{provenance_record['code']['git_revision']}`",
        "",
        "## Data verification",
        "",
        f"- Verified assets: {summary['verified_asset_count']}",
        f"- Verified bytes: {_human_bytes(summary['verified_asset_bytes'])}",
        f"- Newly downloaded: {summary['download_status_counts'].get('downloaded_verified', 0)}",
        f"- Reused locally: {summary['download_status_counts'].get('reused_verified', 0)}",
        "",
        "## Qualification",
        "",
        f"- Task views: {summary['task_count']}",
        f"- Unique configs: {summary['config_count']}",
        f"- Tasks with at least one usable anchor: {summary['task_with_usable_anchor_count']}",
        f"- Tasks with at least 12 anchors: {summary['task_with_minimum_anchor_count']}",
        f"- Eligible task/capability cells: {summary['eligible_capability_cells']}",
        f"- Phase 3 default qualification-path units: {summary['default_phase3_qualification_path_units']:,}",
        f"- Tasks with target missingness: {summary['task_with_target_missingness_count']}",
        f"- Tasks with known-covariate missingness: {summary['task_with_known_covariate_missingness_count']}",
        "",
        "## Capability status counts",
        "",
        "| Capability | Status counts |",
        "|---|---|",
    ]
    for capability_id, counts in summary["capability_status_counts"].items():
        rendered = ", ".join(
            f"{status}={count}" for status, count in counts.items()
        )
        lines.append(f"| {capability_id} | {rendered} |")
    lines.extend(
        [
            "",
            "Eligibility here means the real-data feature coordinate has at "
            "least 12 finite anchors and a non-collapsed p10-p90 range. Phase "
            "3 response-curve calibration can still reject a cell.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.download_workers < 1 or args.maximum_anchors < 1:
        raise ValueError("worker and anchor budgets must be positive")
    if not 0.0 < args.minimum_observed_fraction <= 1.0:
        raise ValueError("minimum observed fraction must be in (0, 1]")
    phase1_dir = args.phase1_dir.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_dir = output_root / args.qualification_id
    incomplete_dir = output_root / f".{args.qualification_id}.incomplete"
    if output_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(
            "FEV qualification output already exists and is immutable or "
            f"incomplete: {output_dir}"
        )
    phase1, tasks, files = _verify_phase1(phase1_dir)
    tasks, files = select_qualification_configs(
        tasks,
        files,
        args.config_ids,
    )
    files_by_config = {
        str(row["configs"][0]): row for row in files
    }
    if set(files_by_config) != {str(row["config_id"]) for row in tasks}:
        raise ValueError("Phase 1 task/config and file inventories disagree")

    output_root.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    try:
        assets = _download_assets(
            data_root,
            files,
            workers=args.download_workers,
            skip_download=args.skip_download,
        )
        write_jsonl(incomplete_dir / "asset_inventory.jsonl", assets)

        qualification_rows: list[dict[str, Any]] = []
        partial_path = incomplete_dir / "task_qualification.partial.jsonl"
        for index, task in enumerate(tasks, start=1):
            try:
                row = qualify_task_view(
                    task_row=task,
                    file_row=files_by_config[str(task["config_id"])],
                    data_root=data_root,
                    maximum_anchors=args.maximum_anchors,
                    minimum_observed_fraction=args.minimum_observed_fraction,
                )
            except Exception as error:
                row = _blocked_task_row(task, error)
            qualification_rows.append(row)
            with partial_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            print(
                protocol.canonical_json(
                    {
                        "phase": "qualification",
                        "completed": index,
                        "total": len(tasks),
                        "config_id": row["config_id"],
                        "accepted_anchor_count": row["accepted_anchor_count"],
                        "eligible_capability_count": sum(
                            status.startswith("eligible")
                            for status in row["capability_status"].values()
                        ),
                        "error": row["anchor_error"],
                    }
                ),
                flush=True,
            )

        qualification_rows.sort(key=lambda row: int(row["task_index"]))
        summary = summarize_qualification(qualification_rows)
        summary.update(
            {
                "verified_asset_count": len(assets),
                "verified_asset_bytes": sum(
                    int(row["size_bytes"]) for row in assets
                ),
                "download_status_counts": dict(
                    sorted(Counter(row["download_status"] for row in assets).items())
                ),
            }
        )
        code = provenance.code_provenance(protocol.REPO_ROOT)
        provenance_record = {
            "created_at": protocol.utc_now(),
            "phase1_dir": str(phase1_dir),
            "phase1_audit_sha256": _sha256(phase1_dir / "audit.json"),
            "phase1_task_inventory_sha256": _sha256(
                phase1_dir / "task_inventory.jsonl"
            ),
            "phase1_download_manifest_sha256": _sha256(
                phase1_dir / "download_manifest.jsonl"
            ),
            "phase1_code_revision": phase1["provenance"]["code"][
                "git_revision"
            ],
            "dataset_revision": FEV_DATASET_REVISION,
            "task_revision": FEV_TASK_REVISION,
            "tasks_sha256": FEV_TASKS_SHA256,
            "code": code,
            "policy": {
                "parquet_downloaded_and_verified": True,
                "data_qualification_completed": True,
                "calibration_started": False,
                "generation_started": False,
                "minimum_window_length": protocol.REAL_FORECAST_MASTER_LENGTH,
                "minimum_finite_feature_count": protocol.MIN_REAL_FEATURE_COUNT,
                "maximum_anchors": args.maximum_anchors,
                "minimum_observed_fraction": args.minimum_observed_fraction,
                "selected_config_ids": sorted(
                    {str(row["config_id"]) for row in tasks}
                ),
            },
        }
        partial_path.unlink()
        write_jsonl(
            incomplete_dir / "task_qualification.jsonl",
            qualification_rows,
        )
        (incomplete_dir / "capability_matrix.csv").write_text(
            qualification_matrix_csv(qualification_rows),
            encoding="utf-8",
        )
        (incomplete_dir / "qualification.json").write_text(
            json.dumps(
                {
                    "schema_version": "cafe.fev_bench_data_qualification.v1",
                    "provenance": provenance_record,
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (incomplete_dir / "REPORT.md").write_text(
            _report(summary, provenance_record),
            encoding="utf-8",
        )
        os.replace(incomplete_dir, output_dir)
    except BaseException:
        raise
    print(
        protocol.canonical_json(
            {
                "qualification_id": args.qualification_id,
                "output_dir": str(output_dir),
                **summary,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
