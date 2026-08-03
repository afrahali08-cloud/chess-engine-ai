"""Background runner tests. No pygame, no window."""

import threading
import time

import pytest

from gui.engine_worker import BackgroundRunner


def drain(runner, *, timeout=5.0, expected=1):
    """Poll until ``expected`` results have arrived or the timeout expires."""
    results = []
    deadline = time.monotonic() + timeout
    while len(results) < expected and time.monotonic() < deadline:
        results.extend(runner.poll())
        if len(results) < expected:
            time.sleep(0.005)
    return results


@pytest.fixture
def runner():
    made = BackgroundRunner()
    yield made
    made.shutdown()


def test_worker_thread_is_a_daemon(runner):
    """A non-daemon worker would hang the app on close during a long search."""
    assert runner._thread.daemon is True


def test_submit_then_poll_returns_the_value(runner):
    job_id = runner.submit("engine", lambda: 21 * 2)
    (result,) = drain(runner)

    assert result.job_id == job_id
    assert result.kind == "engine"
    assert result.value == 42
    assert result.ok is True
    assert result.stale is False


def test_poll_is_empty_when_nothing_has_finished(runner):
    assert runner.poll() == []


def test_payload_round_trips(runner):
    runner.submit("coach", lambda: "analysis", payload=7)
    (result,) = drain(runner)

    assert result.payload == 7
    assert result.value == "analysis"


def test_jobs_never_run_concurrently(runner):
    """The search clears a shared transposition table, so overlap corrupts it."""
    active = threading.Lock()
    overlaps = []

    def job():
        if not active.acquire(blocking=False):
            overlaps.append(True)
            return "overlap"
        try:
            time.sleep(0.02)
            return "ok"
        finally:
            active.release()

    for _ in range(6):
        runner.submit("engine", job)
    results = drain(runner, expected=6)

    assert len(results) == 6
    assert overlaps == []
    assert all(result.value == "ok" for result in results)


def test_jobs_run_in_submission_order(runner):
    order = []
    for index in range(5):
        runner.submit("engine", lambda i=index: order.append(i) or i)
    results = drain(runner, expected=5)

    assert order == [0, 1, 2, 3, 4]
    assert [result.value for result in results] == [0, 1, 2, 3, 4]


def test_a_raising_job_is_reported_and_the_runner_keeps_going(runner):
    def boom():
        raise RuntimeError("search exploded")

    runner.submit("engine", boom)
    (failed,) = drain(runner)

    assert failed.ok is False
    assert isinstance(failed.error, RuntimeError)
    assert "search exploded" in str(failed.error)
    assert failed.value is None

    runner.submit("engine", lambda: "still alive")
    (recovered,) = drain(runner)
    assert recovered.value == "still alive"


def test_bump_generation_marks_in_flight_results_stale(runner):
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(2.0)
        return "late"

    runner.submit("engine", slow)
    assert started.wait(2.0)
    runner.bump_generation()  # the board moved on while the search ran
    release.set()

    (result,) = drain(runner)
    assert result.stale is True
    assert result.value == "late"


def test_results_submitted_after_a_bump_are_fresh(runner):
    runner.bump_generation()
    runner.submit("engine", lambda: "fresh")
    (result,) = drain(runner)

    assert result.stale is False


def test_bump_generation_increments_and_returns(runner):
    first = runner.bump_generation()
    assert runner.generation == first
    assert runner.bump_generation() == first + 1


def test_outstanding_tracks_queued_work(runner):
    release = threading.Event()
    runner.submit("engine", lambda: release.wait(2.0))
    runner.submit("coach", lambda: None)

    assert runner.outstanding("engine") >= 1
    assert runner.outstanding() >= 2
    assert runner.busy() is True

    release.set()
    drain(runner, expected=2)
    assert runner.outstanding() == 0
    assert runner.busy() is False


def test_outstanding_is_zero_for_unknown_kinds(runner):
    assert runner.outstanding("nothing") == 0


def test_shutdown_returns_promptly():
    made = BackgroundRunner()
    started = time.monotonic()
    made.shutdown(timeout=0.25)
    assert time.monotonic() - started < 1.0


def test_shutdown_does_not_block_on_a_running_job():
    """Daemon threading is what lets the window close mid-search."""
    made = BackgroundRunner()
    started = threading.Event()
    made.submit("engine", lambda: (started.set(), time.sleep(1.5)))
    assert started.wait(2.0)

    began = time.monotonic()
    made.shutdown(timeout=0.25)
    assert time.monotonic() - began < 1.0
