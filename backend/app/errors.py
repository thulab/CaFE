from __future__ import annotations


class BenchmarkError(RuntimeError):
    pass


class InternalBenchmarkError(BenchmarkError):
    pass


class NotFoundError(BenchmarkError):
    pass
