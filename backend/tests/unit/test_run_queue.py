from app.workers.run_queue import RunQueue


def test_run_queue_allows_only_one_running_run():
    queue = RunQueue()

    assert queue.submit("run-1") == "running"
    assert queue.submit("run-2") == "queued"
    assert queue.complete("run-1") == "run-2"
    assert queue.running_run_id == "run-2"
