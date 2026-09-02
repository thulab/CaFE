#!/usr/bin/env python3
"""Materialize the pinned Chronos-2 tree used by the CaFE fine-tuning runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PATCH = HERE / "patches" / "chronos2-cafe.patch"
OVERLAY = HERE / "overlay"
MARKER = ".cafe-finetuning-overlay.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def overlay_files() -> list[Path]:
    return sorted(path for path in OVERLAY.rglob("*") if path.is_file())


def expected_manifest() -> dict[str, Any]:
    return {
        "schema_version": "cafe.chronos2_overlay.v1",
        "upstream_commit": CONFIG["upstream"]["commit"],
        "patch_sha256": sha256(PATCH),
        "overlay_files": {
            str(path.relative_to(OVERLAY)): sha256(path) for path in overlay_files()
        },
    }


def verify(destination: Path) -> None:
    marker_path = destination / MARKER
    if not marker_path.is_file():
        raise FileNotFoundError(f"missing overlay marker: {marker_path}")
    recorded = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = expected_manifest()
    if recorded != expected:
        raise RuntimeError("overlay marker does not match this CaFE checkout")
    head = run(["git", "rev-parse", "HEAD"], cwd=destination)
    if head != CONFIG["upstream"]["commit"]:
        raise RuntimeError(f"unexpected Chronos HEAD: {head}")
    for relative, expected_hash in expected["overlay_files"].items():
        target = destination / relative
        if not target.is_file() or sha256(target) != expected_hash:
            raise RuntimeError(f"overlay file mismatch: {relative}")
    print(f"verified CaFE Chronos overlay at {destination}")


def materialize(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}; use --check for an existing overlay"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    upstream = CONFIG["upstream"]
    run(["git", "clone", str(upstream["repository"]), str(destination)])
    run(["git", "checkout", "--detach", str(upstream["commit"])], cwd=destination)
    run(["git", "apply", "--check", str(PATCH)], cwd=destination)
    run(["git", "apply", str(PATCH)], cwd=destination)
    for source in overlay_files():
        relative = source.relative_to(OVERLAY)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = expected_manifest()
    (destination / MARKER).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing materialized tree instead of creating it",
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    if args.check:
        verify(destination)
    else:
        materialize(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
