"""Runs search jobs off the UI thread.

The search in :mod:`engine` keeps a module-level transposition table that it
clears on entry, so two concurrent searches would corrupt each other. This
runner therefore owns exactly **one** worker thread and executes jobs strictly
one at a time, in submission order.

Results are delivered through a plain :class:`queue.Queue` rather than
``pygame.event.post`` -- the SDL event queue is bounded and silently drops
events when full, which would lose a completed search.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable


SHUTDOWN = object()


@dataclass(frozen=True)
class JobResult:
    job_id: int
    generation: int
    kind: str
    value: Any = None
    error: BaseException | None = None
    elapsed: float = 0.0
    payload: Any = None
    stale: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class _Job:
    job_id: int
    generation: int
    kind: str
    fn: Callable[[], Any]
    payload: Any = None


class BackgroundRunner:
    """A single-threaded FIFO job runner with generation-stamped results."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._generation = 0
        self._next_id = 0
        self._outstanding: dict[str, int] = {}
        self._running: _Job | None = None
        self._stopping = False
        # Daemon so a mid-search exit can never hang the process.
        self._thread = threading.Thread(
            target=self._loop, name="engine-worker", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def bump_generation(self) -> int:
        """Invalidate every queued and in-flight job.

        Call this on any action that changes the board out from under a search
        (new game, undo, side swap). Results submitted under an older
        generation come back with ``stale=True``.
        """
        with self._lock:
            self._generation += 1
            return self._generation

    def submit(
        self,
        kind: str,
        fn: Callable[[], Any],
        *,
        payload: Any = None,
    ) -> int:
        with self._lock:
            self._next_id += 1
            job = _Job(
                job_id=self._next_id,
                generation=self._generation,
                kind=kind,
                fn=fn,
                payload=payload,
            )
            self._outstanding[kind] = self._outstanding.get(kind, 0) + 1
        self._jobs.put(job)
        return job.job_id

    def outstanding(self, kind: str | None = None) -> int:
        with self._lock:
            if kind is None:
                return sum(self._outstanding.values())
            return self._outstanding.get(kind, 0)

    def busy(self) -> bool:
        return self.outstanding() > 0

    def poll(self) -> list[JobResult]:
        """Drain finished jobs. Non-blocking; call once per frame."""
        drained: list[JobResult] = []
        while True:
            try:
                result: JobResult = self._results.get_nowait()
            except queue.Empty:
                break
            current = self.generation
            drained.append(
                result
                if result.generation == current
                else JobResult(
                    job_id=result.job_id,
                    generation=result.generation,
                    kind=result.kind,
                    value=result.value,
                    error=result.error,
                    elapsed=result.elapsed,
                    payload=result.payload,
                    stale=True,
                )
            )
        return drained

    def shutdown(self, timeout: float = 0.25) -> None:
        with self._lock:
            self._stopping = True
        self._jobs.put(SHUTDOWN)
        self._thread.join(timeout)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is SHUTDOWN:
                return
            with self._lock:
                if self._stopping:
                    return
                # Guards against a future refactor adding a second worker,
                # which would silently corrupt the shared transposition table.
                assert self._running is None, "two jobs ran concurrently"
                self._running = job

            started = monotonic()
            value: Any = None
            error: BaseException | None = None
            try:
                value = job.fn()
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                error = exc
            finally:
                with self._lock:
                    self._running = None
                    remaining = self._outstanding.get(job.kind, 1) - 1
                    if remaining > 0:
                        self._outstanding[job.kind] = remaining
                    else:
                        self._outstanding.pop(job.kind, None)

            self._results.put(
                JobResult(
                    job_id=job.job_id,
                    generation=job.generation,
                    kind=job.kind,
                    value=value,
                    error=error,
                    elapsed=monotonic() - started,
                    payload=job.payload,
                )
            )
