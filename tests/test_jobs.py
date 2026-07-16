from __future__ import annotations

from threading import Event

import pytest

from lte_scenario_toolkit.jobs import (
    JobBusyError,
    JobCoordinator,
    JobCoordinatorClosedError,
)


def test_coordinator_allows_only_one_active_job():
    coordinator = JobCoordinator()
    first = coordinator.start("scan")

    with pytest.raises(JobBusyError, match="scan") as captured:
        coordinator.start("figure")

    assert captured.value.code == "job.busy"
    assert captured.value.details == {
        "active_job_id": first.job_id,
        "active_kind": "scan",
        "requested_kind": "figure",
    }

    coordinator.finish(first.job_id)
    second = coordinator.start("figure")
    assert second.kind == "figure"
    coordinator.shutdown()


def test_cancel_sets_only_the_matching_active_job_token():
    coordinator = JobCoordinator()
    job = coordinator.start("scan")

    assert coordinator.cancel("another-job") is False
    assert not job.cancel_event.is_set()
    assert coordinator.cancel(job.job_id) is True

    assert job.cancel_event.is_set()
    coordinator.shutdown()


def test_finish_clears_only_the_matching_active_job():
    coordinator = JobCoordinator()
    job = coordinator.start("scan")

    assert coordinator.finish("another-job") is False
    assert coordinator.snapshot().job_id == job.job_id
    assert coordinator.finish(job.job_id) is True
    assert coordinator.snapshot().active is False
    coordinator.shutdown()


def test_submit_runs_worker_with_progress_queue():
    coordinator = JobCoordinator()

    job = coordinator.submit("scan", lambda cancel, emit: (emit({"checked": 1}), 7)[1])

    assert job.future is not None
    assert job.future.result(timeout=2) == 7
    assert job.progress.get(timeout=1) == {"checked": 1}
    coordinator.finish(job.job_id)
    coordinator.shutdown()


def test_finish_cannot_release_the_slot_while_submitted_worker_is_running():
    coordinator = JobCoordinator()
    worker_started = Event()
    release_worker = Event()

    def worker(cancel, emit):
        del cancel, emit
        worker_started.set()
        assert release_worker.wait(timeout=2)

    job = coordinator.submit("scan", worker)
    assert worker_started.wait(timeout=1)

    assert coordinator.finish(job.job_id) is False
    with pytest.raises(JobBusyError):
        coordinator.start("figure")

    release_worker.set()
    assert job.future is not None
    assert job.future.result(timeout=1) is None
    assert coordinator.finish(job.job_id) is True
    coordinator.shutdown()


def test_snapshot_is_an_immutable_view_of_active_job():
    coordinator = JobCoordinator()
    job = coordinator.start("statistics")

    before_cancel = coordinator.snapshot()
    coordinator.cancel(job.job_id)
    after_cancel = coordinator.snapshot()

    assert before_cancel.active is True
    assert before_cancel.kind == "statistics"
    assert before_cancel.cancel_requested is False
    assert after_cancel.cancel_requested is True
    with pytest.raises(AttributeError):
        after_cancel.kind = "other"
    coordinator.shutdown()


def test_shutdown_cancels_running_worker_and_rejects_new_jobs():
    coordinator = JobCoordinator()
    worker_started = Event()

    def worker(cancel, emit):
        del emit
        worker_started.set()
        assert cancel.wait(timeout=2)

    job = coordinator.submit("scan", worker)
    assert worker_started.wait(timeout=1)

    coordinator.shutdown()

    assert job.cancel_event.is_set()
    assert job.future is not None
    assert job.future.result(timeout=1) is None
    with pytest.raises(JobCoordinatorClosedError, match="shut down") as captured:
        coordinator.start("figure")
    assert captured.value.code == "job.coordinator_closed"
    assert captured.value.details == {}
