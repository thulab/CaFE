from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class FileRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        for category in ("models", "dataset_sources", "datasets", "batches", "tasks", "task_runs", "reports"):
            self._dir(category).mkdir(parents=True, exist_ok=True)

    def _dir(self, category: str) -> Path:
        return self.root / "generated" / category

    def save(self, category: str, key: str, payload: BaseModel | dict[str, Any]) -> None:
        path = self._dir(category) / f"{key}.json"
        if isinstance(payload, BaseModel):
            if hasattr(payload, "model_dump"):
                data = payload.model_dump(mode="json")
            else:
                data = payload.dict()
        else:
            data = payload
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, category: str, key: str) -> dict[str, Any]:
        path = self._dir(category) / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, category: str, key: str) -> bool:
        return (self._dir(category) / f"{key}.json").exists()

    def list(self, category: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self._dir(category).glob("*.json")):
            items.append(json.loads(path.read_text(encoding="utf-8")))
        return items

    def delete(self, category: str, key: str) -> None:
        path = self._dir(category) / f"{key}.json"
        if path.exists():
            path.unlink()
