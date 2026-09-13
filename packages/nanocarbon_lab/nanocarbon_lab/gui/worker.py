"""A build process the GUI can actually kill.

Builds used to run on a worker *thread*, which meant they could not be
stopped. That is fine for a 60-atom cage and unacceptable for a coil:
:func:`nanocarbon_lab.builders.swept.build_coil` spends minutes inside
numpy and scipy, none of it checking a cancel flag, so a mistyped
parameter left the window frozen with no way out but killing the app.
Python offers no safe way to interrupt a thread mid-computation, so the
work goes to a **subprocess** and cancelling means terminating it.

The process is long-lived rather than per-build. Spawning costs a second
or two of interpreter start plus numpy and scipy imports, which is
invisible against a six-minute coil but would triple the cost of a cage
that builds in a tenth of a second. So one worker is started up front and
reused; cancelling kills it and a fresh one takes its place.

Results come back over a queue as plain data. Exceptions are **not**
sent as objects: a builder may raise something whose ``__reduce__`` does
not round-trip, and a failure to unpickle the failure is a confusing way
to lose an error message. The repr and the formatted traceback are sent
as strings instead.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from typing import Any

from ..jobs import Job, build

# Sent when a worker dies without reporting -- almost always because the
# user cancelled it, but also covers a segfault in a native library.
WORKER_DIED = "worker-died"


def _worker_loop(requests: Any, results: Any) -> None:  # pragma: no cover
    """Child-process main loop: build whatever arrives, report back.

    Runs in the subprocess, so it is exercised by the integration test
    rather than by direct call, hence the coverage exemption.
    """
    while True:
        item = requests.get()
        if item is None:
            return
        job_id, job = item
        try:
            results.put((job_id, "done", build(job)))
        except BaseException as exc:  # noqa: BLE001 - reported, not handled
            results.put((job_id, "error", (repr(exc), traceback.format_exc())))


class BuildWorker:
    """Runs :class:`nanocarbon_lab.jobs.Job` builds in a killable process.

    Usage is poll-based so a Tk main loop can drive it without threads::

        worker = BuildWorker()
        worker.start()
        job_id = worker.submit(job)
        ...            # later, from a timer callback
        result = worker.poll()

    Parameters
    ----------
    context
        Multiprocessing start method. Defaults to ``"spawn"``, which is
        the only one available on Windows and the only one safe to mix
        with a running Tk main loop on macOS -- ``fork`` would duplicate
        the parent's GUI state into the child.
    """

    def __init__(self, context: str = "spawn") -> None:
        self._ctx = mp.get_context(context)
        self._requests: Any = None
        self._results: Any = None
        self._proc: Any = None
        self._next_id = 0
        self._pending: int | None = None
        #: True once spawning has failed and builds run on a thread
        #: instead. Cancelling is then advisory -- see :meth:`cancel`.
        self.degraded = False

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start the worker process, replacing any dead one.

        Falls back to a thread if a process cannot be started at all.
        That happens in more places than it should: inside a frozen
        executable, in sandboxes without shared memory, and in any script
        that builds the GUI without an ``if __name__ == "__main__"``
        guard, which ``spawn`` needs to re-import the parent safely.
        Losing the ability to cancel is much better than losing the
        ability to build.
        """
        if self.alive or self.degraded:
            return
        try:
            self._requests = self._ctx.Queue()
            self._results = self._ctx.Queue()
            self._proc = self._ctx.Process(
                target=_worker_loop, args=(self._requests, self._results),
                daemon=True,
            )
            self._proc.start()
        except Exception:  # noqa: BLE001 - any spawn failure degrades, never fails
            self.degraded = True
            self._proc = None
            self._requests = None
            self._results = queue.Queue()

    @property
    def alive(self) -> bool:
        if self.degraded:
            return True  # the "worker" is this process; it cannot die alone
        return self._proc is not None and self._proc.is_alive()

    @property
    def busy(self) -> bool:
        """True while a submitted job has neither finished nor been cancelled."""
        return self._pending is not None

    def shutdown(self, timeout: float = 2.0) -> None:
        """Ask the worker to exit, then make sure it has."""
        if self.degraded or self._proc is None:
            self._pending = None
            return
        try:
            if self._proc.is_alive():
                self._requests.put(None)
                self._proc.join(timeout)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout)
        except (OSError, ValueError):  # queue already closed
            pass
        finally:
            self._proc = None
            self._pending = None

    # ----------------------------------------------------------------- work
    def submit(self, job: Job) -> int:
        """Queue a job and return its id. Starts the worker if needed."""
        if self.busy:
            raise RuntimeError("a build is already in flight; cancel it first.")
        self.start()
        self._next_id += 1
        self._pending = self._next_id
        if self.degraded:
            self._run_on_thread(self._next_id, job)
        else:
            self._requests.put((self._next_id, job))
        return self._next_id

    def _run_on_thread(self, job_id: int, job: Job) -> None:
        """Degraded path: build here, on a daemon thread."""
        import threading

        def run() -> None:
            try:
                self._results.put((job_id, "done", build(job)))
            except BaseException as exc:  # noqa: BLE001 - reported, not handled
                self._results.put(
                    (job_id, "error", (repr(exc), traceback.format_exc()))
                )

        threading.Thread(target=run, daemon=True).start()

    def cancel(self) -> None:
        """Stop the running build and stand up a fresh worker.

        Terminating the process is the whole point: there is no
        cooperative exit from a numpy call. The replacement is started
        eagerly so the next build does not pay the startup cost on the
        critical path.

        In the degraded thread mode there is nothing to terminate --
        Python cannot safely stop a thread mid-computation -- so cancel
        only detaches: the result is discarded when it arrives and the UI
        is usable again, while the work runs on to completion in the
        background. Callers that want to tell the user the difference can
        read :attr:`degraded`.
        """
        self._pending = None
        if self.degraded:
            return
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2.0)
        self._proc = None
        self.start()

    def poll(self) -> tuple[int, str, Any] | None:
        """Return the next ``(job_id, kind, payload)``, or ``None``.

        ``kind`` is ``"done"`` with an ``Atoms``, or ``"error"`` with a
        ``(repr, traceback)`` pair. A worker that died without reporting
        -- which a native crash would do -- surfaces as an ``"error"``
        carrying :data:`WORKER_DIED` rather than hanging the caller
        forever waiting for a result that is never coming.
        """
        if self._results is None:
            return None
        try:
            job_id, kind, payload = self._results.get_nowait()
        except queue.Empty:
            if self._pending is not None and not self.alive:
                pending, self._pending = self._pending, None
                return (
                    pending,
                    "error",
                    (WORKER_DIED, "The build process exited unexpectedly."),
                )
            return None
        # Results from a cancelled job still arrive if it finished in the
        # race between terminate() and the queue read; drop them.
        if job_id != self._pending:
            return None
        self._pending = None
        return job_id, kind, payload


__all__ = ["WORKER_DIED", "BuildWorker"]
