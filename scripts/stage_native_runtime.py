#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


MODEL_KEYS = (
    "timer_4p0",
    "timer_3p5",
    "chronos_2",
    "moirai2",
    "toto_2p0",
    "timesfm2p5",
    "tirex2",
)
CODE_PATHS = (
    "core",
    "inference/__init__.py",
    "inference/_locale.py",
    "inference/i18n.py",
    "inference/pipeline",
)
MANIFEST_SCHEMA = "cafe.staged_native_model_runtime.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage model weights and the model-only runtime used by CaFE."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Re-hash an existing staged runtime without copying large weights.",
    )
    return parser.parse_args()


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)


def _files(root: Path) -> Iterable[Path]:
    yield from sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(source: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    code_destination = destination / "model_runtime"
    model_destination = destination / "models" / "builtin"
    targets = [code_destination, model_destination]
    if args.manifest_only:
        missing = [path for path in targets if not path.exists()]
        missing.extend(
            model_destination / key
            for key in args.models
            if not (model_destination / key).is_dir()
        )
        if missing:
            raise FileNotFoundError(
                "cannot refresh an incomplete staged runtime: "
                + ", ".join(map(str, missing))
            )
    else:
        existing = [path for path in targets if path.exists()]
        if existing and not args.replace:
            raise FileExistsError(
                "staged runtime already exists; pass --replace to replace exactly: "
                + ", ".join(map(str, existing))
            )
        for path in existing:
            shutil.rmtree(path)

        for relative in CODE_PATHS:
            _copy(source / relative, code_destination / relative)
        for key in args.models:
            _copy(
                source / "data" / "models" / "builtin" / key,
                model_destination / key,
            )

    records = []
    for root in targets:
        for path in _files(root):
            records.append(
                {
                    "path": str(path.relative_to(destination)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source": str(source),
        "source_git_revision": _git_value(source, "rev-parse", "HEAD"),
        "source_git_status": _git_value(
            source, "status", "--short", "--untracked-files=no"
        ),
        "models": list(args.models),
        "files": records,
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "model_runtime_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
