import hashlib
import json
from pathlib import Path
from typing import Any

from app.models.benchmark import ForecastArtifact
from app.services.sample_store import canonical_json


class ForecastStore:
    def __init__(self, forecasts_dir: Path):
        self.forecasts_dir = Path(forecasts_dir)
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)

    def write_forecasts(
        self,
        run_id: str,
        task_id: str,
        model_id: str,
        shard_id: str,
        rows: list[dict[str, Any]],
    ) -> ForecastArtifact:
        output_dir = self.forecasts_dir / run_id / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{model_id}_{shard_id}.jsonl"
        checksum = hashlib.sha256()
        with output.open("w", encoding="utf-8") as file:
            for row in rows:
                record = {
                    "schema_version": "forecast.v1",
                    "benchmarking_run_id": run_id,
                    "task_id": task_id,
                    "model_id": model_id,
                    "shard_id": shard_id,
                    "sample_id": row["sample_id"],
                    "status": row.get("status", "succeeded"),
                    "forecast": row.get("forecast"),
                    "future_timestamps": row.get("future_timestamps", []),
                    "metrics": row.get("metrics", {}),
                    "error_code": row.get("error_code"),
                    "error_message": row.get("error_message"),
                }
                line = canonical_json(record)
                checksum.update(line.encode("utf-8"))
                file.write(line + "\n")
        return ForecastArtifact(
            benchmarking_run_id=run_id,
            unit_id=rows[0].get("unit_id", ""),
            task_id=task_id,
            model_id=model_id,
            shard_id=shard_id,
            storage_uri=str(output),
            schema_version="forecast.v1",
            sample_count=len(rows),
            checksum=checksum.hexdigest(),
        )

    def read_forecasts(self, storage_uri: str | Path) -> list[dict[str, Any]]:
        with Path(storage_uri).open(encoding="utf-8") as file:
            return [json.loads(line) for line in file]
