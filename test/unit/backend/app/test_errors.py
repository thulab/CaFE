from __future__ import annotations

import unittest

from backend.app.errors import BenchmarkError, InternalBenchmarkError, NotFoundError


class BenchmarkErrorTest(unittest.TestCase):
    def test_benchmark_error_is_runtime_error(self) -> None:
        error = BenchmarkError("test error")
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "test error")

    def test_internal_benchmark_error_inherits_from_benchmark_error(self) -> None:
        error = InternalBenchmarkError("internal error")
        self.assertIsInstance(error, BenchmarkError)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "internal error")

    def test_not_found_error_inherits_from_benchmark_error(self) -> None:
        error = NotFoundError("resource not found")
        self.assertIsInstance(error, BenchmarkError)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "resource not found")

    def test_error_hierarchy(self) -> None:
        benchmark = BenchmarkError("benchmark")
        internal = InternalBenchmarkError("internal")
        not_found = NotFoundError("not found")

        self.assertNotIsInstance(benchmark, InternalBenchmarkError)
        self.assertNotIsInstance(benchmark, NotFoundError)
        self.assertNotIsInstance(internal, NotFoundError)
        self.assertNotIsInstance(not_found, InternalBenchmarkError)


if __name__ == "__main__":
    unittest.main()