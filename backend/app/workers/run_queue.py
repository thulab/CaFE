from collections import deque


class RunQueue:
    def __init__(self):
        self.running_run_id: str | None = None
        self._queued: deque[str] = deque()

    def submit(self, run_id: str) -> str:
        if self.running_run_id is None:
            self.running_run_id = run_id
            return "running"
        self._queued.append(run_id)
        return "queued"

    def complete(self, run_id: str) -> str | None:
        if self.running_run_id == run_id:
            self.running_run_id = self._queued.popleft() if self._queued else None
        return self.running_run_id
