#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from cafe.data.fev_bench import (
    FEV_BENCH_CONFIGS,
    FEV_DATASET_REPOSITORY,
    FEV_DATASET_REVISION,
    FEV_TASK_REPOSITORY,
    FEV_TASK_REVISION,
    FEV_TASKS_SHA256,
)
from cafe.protocol import REPO_ROOT, canonical_json, utc_now


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "fev-bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned CaFE pilot subset of FEV-Bench."
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=sorted(FEV_BENCH_CONFIGS),
        help="One selected FEV config. Repeat to limit the download.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(client: httpx.Client, url: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return sha256(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(target)


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    config_ids = tuple(args.config or sorted(FEV_BENCH_CONFIGS))
    downloaded: list[dict[str, Any]] = []
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    with httpx.Client(
        follow_redirects=True,
        timeout=120.0,
        proxy=proxy,
        trust_env=False,
    ) as client:
        for config_id in config_ids:
            config = FEV_BENCH_CONFIGS[config_id]
            target = output_root / config_id / config.parquet_name
            url = (
                "https://huggingface.co/datasets/"
                f"{FEV_DATASET_REPOSITORY}/resolve/{FEV_DATASET_REVISION}/"
                f"{config.source_path}?download=true"
            )
            digest = download_file(client, url, target)
            if digest != config.sha256:
                raise ValueError(
                    f"checksum mismatch for {config_id}: "
                    f"expected {config.sha256}, got {digest}"
                )
            actual_size = target.stat().st_size
            if actual_size != config.size_bytes:
                raise ValueError(
                    f"size mismatch for {config_id}: expected "
                    f"{config.size_bytes}, got {actual_size}"
                )
            downloaded.append(
                {
                    "config": asdict(config),
                    "local_path": str(target),
                    "sha256": digest,
                    "size_bytes": actual_size,
                    "source_url": url,
                }
            )
            print(
                canonical_json(
                    {
                        "config_id": config_id,
                        "path": str(target),
                        "size_bytes": actual_size,
                        "status": "verified",
                    }
                ),
                flush=True,
            )
    write_manifest(
        output_root / "snapshot.json",
        {
            "schema_version": "cafe.fev_bench_snapshot.v1",
            "created_at": utc_now(),
            "dataset_repository": FEV_DATASET_REPOSITORY,
            "dataset_revision": FEV_DATASET_REVISION,
            "task_repository": FEV_TASK_REPOSITORY,
            "task_revision": FEV_TASK_REVISION,
            "tasks_sha256": FEV_TASKS_SHA256,
            "configs": downloaded,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
