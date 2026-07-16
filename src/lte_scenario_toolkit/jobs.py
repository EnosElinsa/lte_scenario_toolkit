"""Thread-safe coordination for one local CPU-intensive job."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from queue import Queue
from threading import Event, Lock
from typing import Any
from uuid import uuid4


class JobBusyError(RuntimeError):
    """Raised when a second job is requested while one is still active."""

    code = "job.busy"

    def __init__(
        self,
        message: str,
        *,
        active_job_id: str,
        active_kind: str,
        requested_kind: str,
    ) -> None:
        super().__init__(message)
        self.details = {
            "active_job_id": active_job_id,
            "active_kind": active_kind,
            "requested_kind": requested_kind,
        }


class JobCoordinatorClosedError(RuntimeError):
    """Raised when work is submitted after application shutdown."""

    code = "job.coordinator_closed"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = {}


@dataclass(frozen=True)
class Job:
    """Immutable handle whose synchronization primitives are worker-safe."""

    job_id: str
    kind: str
    cancel_event: Event
    progress: Queue[Any]
    future: Future[Any] | None = None


@dataclass(frozen=True)
class JobSnapshot:
    """Immutable, render-safe view of the active coordinator state."""

    active: bool
    job_id: str | None = None
    kind: str | None = None
    cancel_requested: bool = False
    done: bool = False


class JobCoordinator:
    """Run at most one background worker and expose cooperative cancellation."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lte-job")
        self._active: Job | None = None
        self._shutdown = False

    @staticmethod
    def _validate_kind(kind: Any) -> str:
        if type(kind) is not str or not kind.strip():
            raise ValueError("job kind must be a non-empty string")
        return kind

    def _reserve_locked(self, kind: str) -> Job:
        if self._shutdown:
            raise JobCoordinatorClosedError("Job coordinator has been shut down")
        if self._active is not None:
            raise JobBusyError(
                f"Cannot start {kind!r}; active job is "
                f"{self._active.kind!r} ({self._active.job_id})",
                active_job_id=self._active.job_id,
                active_kind=self._active.kind,
                requested_kind=kind,
            )
        job = Job(
            job_id=str(uuid4()),
            kind=kind,
            cancel_event=Event(),
            progress=Queue(),
        )
        self._active = job
        return job

    def start(self, kind: str) -> Job:
        """Reserve the one active slot without submitting a worker."""

        validated_kind = self._validate_kind(kind)
        with self._lock:
            return self._reserve_locked(validated_kind)

    def submit(
        self,
        kind: str,
        worker: Callable[[Event, Callable[[Any], None]], Any],
    ) -> Job:
        """Reserve the active slot and run ``worker(cancel, emit)``."""

        if not callable(worker):
            raise ValueError("worker must be callable")
        validated_kind = self._validate_kind(kind)
        with self._lock:
            reserved = self._reserve_locked(validated_kind)
            try:
                future = self._executor.submit(
                    worker,
                    reserved.cancel_event,
                    reserved.progress.put,
                )
            except BaseException:
                self._active = None
                raise
            submitted = replace(reserved, future=future)
            self._active = submitted
            return submitted

    def finish(self, job_id: str) -> bool:
        """Clear the active slot only when ``job_id`` still owns it."""

        with self._lock:
            if self._active is None or self._active.job_id != job_id:
                return False
            if self._active.future is not None and not self._active.future.done():
                return False
            self._active = None
            return True

    def cancel(self, job_id: str) -> bool:
        """Request cancellation only for the matching active job."""

        with self._lock:
            if self._active is None or self._active.job_id != job_id:
                return False
            self._active.cancel_event.set()
            return True

    def snapshot(self) -> JobSnapshot:
        """Return an immutable status suitable for GUI polling."""

        with self._lock:
            job = self._active
            if job is None:
                return JobSnapshot(active=False)
            return JobSnapshot(
                active=True,
                job_id=job.job_id,
                kind=job.kind,
                cancel_requested=job.cancel_event.is_set(),
                done=job.future.done() if job.future is not None else False,
            )

    def shutdown(self) -> None:
        """Cancel the active token and join the worker during application exit."""

        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            active = self._active
            if active is not None:
                active.cancel_event.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._active = None


__all__ = [
    "Job",
    "JobBusyError",
    "JobCoordinatorClosedError",
    "JobCoordinator",
    "JobSnapshot",
]
