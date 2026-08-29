"""Tests for the killable build worker.

The reason this module exists is that a coil build cannot be interrupted:
it spends minutes inside numpy with nothing checking a flag, and Python
has no safe way to stop a thread mid-computation. So the interesting
behaviours to pin down are that a long build really can be stopped, that
the worker survives being stopped, and that a failure to start a process
degrades to something that still works.

These tests start real subprocesses, so they are marked ``slow``.
"""

from __future__ import annotations

import time

import pytest

from nanocarbon_lab.gui.worker import BuildWorker
from nanocarbon_lab.jobs import Job


def collect(worker: BuildWorker, timeout: float = 300.0):
    """Poll until the worker reports, mimicking the GUI's timer."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = worker.poll()
        if result is not None:
            return result
        time.sleep(0.05)
    return None


@pytest.fixture
def worker():
    instance = BuildWorker()
    yield instance
    instance.shutdown()


@pytest.mark.slow
class TestProcessWorker:
    def test_a_build_comes_back_through_the_queue(self, worker):
        worker.submit(Job("fullerene", {"freq": 1, "family": "C60"}))
        result = collect(worker)
        assert result is not None, "no result within the timeout"
        _job_id, kind, atoms = result
        assert kind == "done"
        assert len(atoms) == 60
        assert not worker.busy

    def test_a_failing_build_returns_strings_not_an_exception(self, worker):
        """A builder may raise something that does not survive pickling,
        and failing to unpickle a failure is a confusing way to lose an
        error message. Both fields must be plain text."""
        worker.submit(Job("coil (relaxed)",
                          {"coil_radius": 30.0, "pitch": 8.0, "tube_radius": 6.0}))
        result = collect(worker)
        assert result is not None
        _job_id, kind, (text, tb) = result
        assert kind == "error"
        assert isinstance(text, str) and isinstance(tb, str)
        assert "pitch" in text, text

    def test_a_long_build_can_actually_be_stopped(self, worker):
        """The whole point of using a process instead of a thread."""
        worker.submit(Job("coil (relaxed)", {"coil_radius": 60.0, "pitch": 30.0,
                                             "turns": 3.0, "tube_radius": 6.0}))
        time.sleep(3.0)  # let it get properly into numpy
        assert worker.busy
        started = time.time()
        worker.cancel()
        assert time.time() - started < 10.0, "cancel should be prompt"
        assert not worker.busy

    def test_the_worker_still_builds_after_a_cancel(self, worker):
        worker.submit(Job("coil (relaxed)", {"coil_radius": 60.0, "pitch": 30.0,
                                             "turns": 3.0, "tube_radius": 6.0}))
        time.sleep(2.0)
        worker.cancel()
        worker.submit(Job("fullerene", {"freq": 2, "family": "C60"}))
        result = collect(worker)
        assert result is not None and result[1] == "done"
        assert len(result[2]) == 240

    def test_two_builds_at_once_are_refused(self, worker):
        worker.submit(Job("fullerene", {"freq": 1}))
        with pytest.raises(RuntimeError, match="already in flight"):
            worker.submit(Job("fullerene", {"freq": 1}))
        collect(worker)

    def test_shutdown_is_idempotent(self, worker):
        worker.start()
        worker.shutdown()
        worker.shutdown()  # must not raise
        assert not worker.alive


class TestDegradedFallback:
    """When no process can be started, builds must still happen.

    That case is not exotic: a frozen executable, a sandbox without
    shared memory, or any script that constructs the GUI without an
    ``if __name__ == "__main__"`` guard, which ``spawn`` needs.
    """

    def test_a_failed_spawn_falls_back_to_a_thread(self, monkeypatch):
        worker = BuildWorker()

        def refuse(*_args, **_kwargs):
            raise OSError("no processes here")

        monkeypatch.setattr(worker._ctx, "Process", refuse)
        worker.start()
        assert worker.degraded
        assert worker.alive, "degraded mode must not look dead"

        worker.submit(Job("fullerene", {"freq": 1, "family": "C60"}))
        result = collect(worker, timeout=120)
        assert result is not None and result[1] == "done"
        assert len(result[2]) == 60
        worker.shutdown()

    def test_cancelling_in_degraded_mode_detaches_rather_than_hanging(
        self, monkeypatch
    ):
        """A thread cannot be killed, so cancel only stops waiting for it.
        What must not happen is the UI staying stuck as busy."""
        worker = BuildWorker()
        monkeypatch.setattr(
            worker._ctx, "Process",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no processes")),
        )
        worker.start()
        worker.submit(Job("fullerene", {"freq": 1, "family": "C60"}))
        worker.cancel()
        assert not worker.busy
        worker.shutdown()
