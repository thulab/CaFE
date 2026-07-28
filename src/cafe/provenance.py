from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_SCHEMA = "cafe.experiment.v1"
STAGE_CONTRACT_SCHEMA = "cafe.stage_contract.v1"
STAGES = ("calibration", "generation", "validation", "inference", "analysis")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Atomically create an immutable JSON record.

    Repeating the same write is idempotent. A different payload at the same
    path is rejected so an existing stage can never be silently redefined.
    """

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"immutable record already exists with other content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_provenance(repository_root: Path) -> dict[str, Any]:
    revision = _git_output(repository_root, "rev-parse", "HEAD")
    branch = _git_output(repository_root, "branch", "--show-current")
    status = _git_output(repository_root, "status", "--porcelain=v1") or ""
    diff = _git_output(repository_root, "diff", "--binary", "HEAD") or ""
    return {
        "git_revision": revision,
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def initialize_experiment(
    experiment_root: Path,
    *,
    experiment_id: str,
    created_at: str,
) -> dict[str, Any]:
    """Create only the stable experiment identity.

    Stage code and configuration deliberately do not live here. They are
    frozen when the corresponding stage starts, so later code may consume
    existing upstream artifacts without rewriting their provenance.
    """

    path = experiment_root / "experiment.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") != EXPERIMENT_SCHEMA
            or existing.get("experiment_id") != experiment_id
        ):
            raise ValueError(f"experiment identity does not match: {path}")
        return existing
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA,
        "experiment_id": experiment_id,
        "created_at": created_at,
        "stage_contract_layout": "stage_contracts/<stage>.json",
        "artifact_layout": (
            "<dataset_id>/{01_calibration,02_generation,"
            "03_inference,04_analysis}"
        ),
    }
    write_json_once(path, manifest)
    return manifest


def artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = relative_to.resolve()
    try:
        display_path = str(resolved.relative_to(root))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def upstream_records(
    paths: Iterable[Path],
    *,
    relative_to: Path,
) -> list[dict[str, Any]]:
    return [
        artifact_record(path, relative_to=relative_to)
        for path in sorted((item.resolve() for item in paths), key=str)
    ]


def stage_contract(
    *,
    stage: str,
    created_at: str,
    repository_root: Path,
    config: dict[str, Any],
    upstream: list[dict[str, Any]],
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown CaFE stage: {stage!r}")
    identity = {
        "stage": stage,
        "config": config,
        "upstream": upstream,
        "code": code_provenance(repository_root),
    }
    return {
        "schema_version": STAGE_CONTRACT_SCHEMA,
        "created_at": created_at,
        "contract_sha256": value_sha256(identity),
        **identity,
    }


def ensure_stage_contract(
    experiment_root: Path,
    *,
    stage: str,
    created_at: str,
    repository_root: Path,
    config: dict[str, Any],
    upstream: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = stage_contract(
        stage=stage,
        created_at=created_at,
        repository_root=repository_root,
        config=config,
        upstream=upstream,
    )
    path = experiment_root / "stage_contracts" / f"{stage}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_keys = ("stage", "config", "upstream", "code")
        if any(existing.get(key) != contract.get(key) for key in comparable_keys):
            raise ValueError(
                f"{stage} is already frozen with a different stage contract"
            )
        return existing
    write_json_once(path, contract)
    return contract
