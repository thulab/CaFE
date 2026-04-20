from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import BaseModel

from backend.app.storage import FileRepository
from test.support.helpers import temporary_runtime_dir


class DummyPayload(BaseModel):
    name: str
    value: int


class FileRepositoryTest(unittest.TestCase):
    def test_init_creates_required_directories(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            for category in ("models", "dataset_sources", "datasets", "batches", "tasks", "task_runs", "reports"):
                dir_path = repo._dir(category)
                self.assertTrue(dir_path.exists(), f"Directory {category} should exist")

    def test_save_and_load_pydantic_model(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            payload = DummyPayload(name="test", value=42)
            repo.save("batches", "test-batch", payload)

            loaded = repo.load("batches", "test-batch")
            self.assertEqual(loaded["name"], "test")
            self.assertEqual(loaded["value"], 42)

    def test_save_and_load_dict(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            payload = {"name": "test", "value": 42}
            repo.save("batches", "test-batch-dict", payload)

            loaded = repo.load("batches", "test-batch-dict")
            self.assertEqual(loaded["name"], "test")
            self.assertEqual(loaded["value"], 42)

    def test_exists_returns_true_for_saved_item(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            payload = DummyPayload(name="test", value=42)
            repo.save("batches", "test-batch", payload)

            self.assertTrue(repo.exists("batches", "test-batch"))

    def test_exists_returns_false_for_nonexistent_item(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            self.assertFalse(repo.exists("batches", "nonexistent"))

    def test_list_returns_all_items_in_category(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            repo.save("batches", "batch-1", {"id": 1})
            repo.save("batches", "batch-2", {"id": 2})
            repo.save("batches", "batch-3", {"id": 3})

            items = repo.list("batches")
            self.assertEqual(len(items), 3)
            ids = sorted(item["id"] for item in items)
            self.assertEqual(ids, [1, 2, 3])

    def test_list_returns_empty_list_for_empty_category(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            items = repo.list("models")
            self.assertEqual(items, [])

    def test_delete_removes_item(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            repo.save("batches", "test-batch", {"id": 1})
            self.assertTrue(repo.exists("batches", "test-batch"))

            repo.delete("batches", "test-batch")
            self.assertFalse(repo.exists("batches", "test-batch"))

    def test_delete_nonexistent_item_does_not_raise(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            repo.delete("batches", "nonexistent")

    def test_save_overwrites_existing_item(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            repo.save("batches", "test-batch", {"version": 1})
            repo.save("batches", "test-batch", {"version": 2})

            loaded = repo.load("batches", "test-batch")
            self.assertEqual(loaded["version"], 2)

    def test_list_is_sorted(self) -> None:
        with temporary_runtime_dir(prefix="storage-test-") as runtime_root:
            repo = FileRepository(runtime_root)
            for i in [3, 1, 4, 1, 5, 9, 2, 6]:
                repo.save("batches", f"batch-{i:02d}", {"n": i})

            items = repo.list("batches")
            names = [item["n"] for item in items]
            self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()