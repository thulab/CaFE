from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    logical_name: str
    config_id: str
    asset_name: str
    domain: str
    real_data_adapter: str = "gift_arrow"


_GIFT_DATASETS = (
    ("gift_electricity_h", "Electricity", "electricity/H", "Energy"),
    ("gift_solar_h", "Solar", "solar/H", "Energy"),
    ("gift_ett1_h", "ETT1", "ett1/H", "Energy"),
    ("gift_ett2_h", "ETT2", "ett2/H", "Energy"),
    ("gift_jena_weather_h", "Jena Weather", "jena_weather/H", "Nature"),
    ("gift_kdd_cup_h", "KDD Cup 2018", "kdd_cup_2018_with_missing/H", "Nature"),
    ("gift_loop_seattle_h", "Loop Seattle", "LOOP_SEATTLE/H", "Transport"),
    ("gift_sz_taxi_h", "SZ-Taxi", "SZ_TAXI/H", "Transport"),
    ("gift_m_dense_h", "M_DENSE", "M_DENSE/H", "Transport"),
    (
        "gift_bitbrains_fast_h",
        "Bitbrains Fast Storage",
        "bitbrains_fast_storage/H",
        "Web/CloudOps",
    ),
    ("gift_bitbrains_rnd_h", "Bitbrains RND", "bitbrains_rnd/H", "Web/CloudOps"),
    ("gift_bizitobs_l2c_h", "BizITObs L2C", "bizitobs_l2c/H", "Web/CloudOps"),
    (
        "gift_bizitobs_application",
        "BizITObs Application",
        "bizitobs_application",
        "Web/CloudOps",
    ),
    ("gift_bizitobs_service", "BizITObs Service", "bizitobs_service", "Web/CloudOps"),
    ("gift_restaurant_d", "Restaurant", "restaurant", "Business"),
    (
        "gift_hierarchical_sales_d",
        "Hierarchical Sales",
        "hierarchical_sales/D",
        "Business",
    ),
    ("gift_m4_hourly", "M4 Hourly", "m4_hourly", "Mixed"),
    ("gift_us_births_d", "US Births", "us_births/D", "Nature"),
    ("gift_saugeenday_d", "Saugeen River Flow", "saugeenday/D", "Nature"),
    (
        "gift_temperature_rain_d",
        "Temperature Rain",
        "temperature_rain_with_missing",
        "Nature",
    ),
)


DATASET_REGISTRY = {
    dataset_id: DatasetSpec(
        dataset_id=dataset_id,
        logical_name=logical_name,
        config_id=asset_name,
        asset_name=asset_name,
        domain=domain,
    )
    for dataset_id, logical_name, asset_name, domain in _GIFT_DATASETS
}


def resolve_dataset(dataset_id: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[dataset_id]
    except KeyError as error:
        raise ValueError(
            f"unknown GIFT-Eval dataset {dataset_id!r}; registered={sorted(DATASET_REGISTRY)}"
        ) from error


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(*parts: Any, base: int = 0) -> int:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int((base + int.from_bytes(digest[:8], "big")) % (2**32 - 1))


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


# Compatibility names used by the model-service transport module. The active
# v6 task builder provides variable contexts and does not call legacy views.
FIXED_CONTEXT_LENGTH = 168
REAL_CALIBRATION_CONTEXT_LENGTH = 168
VIEW_CONTEXT_LENGTHS = (96, 168, 336)
