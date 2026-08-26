#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


FEV_VERSION = "0.8.0"
FEV_TAG_COMMIT = "f1afffbf97bc51a4a233080d331633c6f7ab32f6"
SUITE_URL = (
    "https://raw.githubusercontent.com/autogluon/fev/v0.8.0/"
    "benchmarks/fev_bench/tasks_mini.yaml"
)
SUITE_SHA256 = "992ac3b7fa76f0a400d2783535b989549decab381006da5a866e28b61b963edb"
DATASET_REPOSITORY = "autogluon/fev_datasets"
DATASET_REVISION = "f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8"
MINI_DATASET_PATHS = {
    "boomlet_1282": "boomlet/1282/train-00000-of-00001.parquet",
    "boomlet_1676": "boomlet/1676/train-00000-of-00001.parquet",
    "boomlet_619": "boomlet/619/train-00000-of-00001.parquet",
    "bizitobs_l2c_5T": "bizitobs_l2c/5T/train-00000-of-00001.parquet",
    "epf_np": "epf_np/train-00000-of-00001.parquet",
    "ETT_15T": "ETT/15T/train-00000-of-00001.parquet",
    "ETT_1H": "ETT/1H/train-00000-of-00001.parquet",
    "proenfo_gfc14": "proenfo_gfc14/train-00000-of-00001.parquet",
    "hospital_admissions_1W": (
        "hospital_admissions/1W/train-00000-of-00001.parquet"
    ),
    "hospital_admissions_1D": (
        "hospital_admissions/1D/train-00000-of-00001.parquet"
    ),
    "jena_weather_1H": "jena_weather/1H/train-00000-of-00001.parquet",
    "M_DENSE_1D": "M_DENSE/1D/train-00000-of-00001.parquet",
    "rohlik_orders_1D": "rohlik_orders/1D/train-00000-of-00001.parquet",
    "rossmann_1W": "rossmann/1W/train-00000-of-00001.parquet",
    "rossmann_1D": "rossmann/1D/train-00000-of-00001.parquet",
    "solar_with_weather_1H": (
        "solar_with_weather/1H/train-00000-of-00001.parquet"
    ),
    "uci_air_quality_1H": "uci_air_quality/1H/train-00000-of-00001.parquet",
    "uk_covid_nation_1D": "uk_covid_nation/1D/train-00000-of-00001.parquet",
    "us_consumption_1Y": "us_consumption/1Y/train-00000-of-00001.parquet",
    "world_co2_emissions": "world_co2_emissions/train-00000-of-00001.parquet",
}


def _download(client: httpx.Client, url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as error:
            last_error = error
            if attempt < 4:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot the exact FEV v0.8.0 Mini-20 suite definition."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fev-mini-v0.8.0"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"FEV source snapshot is immutable; target already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="fev-mini-snapshot-", dir=output_dir.parent)
    )
    try:
        suite_path = temporary / "tasks_mini.yaml"
        client = httpx.Client(follow_redirects=True, timeout=300)
        payload = _download(client, SUITE_URL)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != SUITE_SHA256:
            raise ValueError(
                f"FEV suite hash mismatch: expected {SUITE_SHA256}, received {digest}"
            )
        suite_path.write_bytes(payload)

        dataset_dir = temporary / "datasets"
        dataset_dir.mkdir()
        dataset_files: dict[str, dict[str, object]] = {}
        for config_name, upstream_path in MINI_DATASET_PATHS.items():
            url = (
                "https://huggingface.co/datasets/"
                f"{DATASET_REPOSITORY}/resolve/{DATASET_REVISION}/{upstream_path}"
            )
            destination = dataset_dir / f"{config_name}.parquet"
            data = _download(client, url)
            destination.write_bytes(data)
            dataset_files[config_name] = {
                "path": str(Path("datasets") / destination.name),
                "upstream_path": upstream_path,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        manifest = {
            "schema_version": "cafe.fev_source_snapshot.v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_id": "fev_bench",
            "suite_id": "mini20",
            "fev_package_version": FEV_VERSION,
            "upstream_git_tag": f"v{FEV_VERSION}",
            "upstream_git_commit": FEV_TAG_COMMIT,
            "suite_url": SUITE_URL,
            "suite_path": "tasks_mini.yaml",
            "suite_sha256": digest,
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "dataset_snapshot_policy": "local_commit_pinned_single_copy_v1",
            "dataset_files": dataset_files,
        }
        (temporary / "source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        client.close()
        os.replace(temporary, output_dir)
    except Exception:
        if "client" in locals():
            client.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
