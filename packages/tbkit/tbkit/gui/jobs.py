"""The window's calculations run in a child process: cancelling kills it.

A Python thread cannot be stopped from outside, so a calculation started from the window
used to run to the end after «Cancelar» and its result was thrown away. Here every
calculation runs in a separate process (``spawn``: safe next to Qt on every platform),
which can be killed at once, given its own cores and watched for memory.

The process is started ahead of time and imports numpy, scipy, ASE and tbkit while idle,
so a calculation starts without that delay; it serves one job after another and is replaced
only when killed. Messages from the child, through a pipe:

``("ready", pid)``, ``("out", text)`` (what the job prints), ``("progress", dict)``
(:mod:`tbkit.progress` counters inside the job), ``("frame", positions)`` (a relaxation's
current geometry), ``("done", result)`` and ``("fail", message, traceback)``.

Nothing here imports Qt: the window polls :meth:`Worker.messages` from a timer in its own
thread, so there is no worker thread in the GUI process at all.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings


class _Pipe:
    """File-like: writes become ``("out", text)`` messages."""

    def __init__(self, conn):
        self.conn = conn

    def write(self, text: str) -> int:
        if text:
            self.conn.send(("out", text))
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _throttled(conn, kind: str, every: float):
    last = [0.0]

    def send(value):
        now = time.monotonic()
        finished = isinstance(value, dict) and value.get("total") is not None and \
            value.get("done", 0) >= value["total"]
        if now - last[0] >= every or finished:
            last[0] = now
            conn.send((kind, value))
    return send


#: The processors the job process started with: a job without a core limit gets them all
#: back (a previous job may have narrowed them).
_ALLOWED: list[int] | None = None


def set_cores(cores: int | None) -> list[int] | None:
    """Limit this process to ``cores`` processors (the first ones it may use) and its linear
    algebra to as many threads; ``None`` restores every processor it started with. Returns
    the processors chosen (None: no limit)."""
    global _ALLOWED
    if _ALLOWED is None and hasattr(os, "sched_getaffinity"):
        _ALLOWED = sorted(os.sched_getaffinity(0))
    if not cores:
        if _ALLOWED is not None:
            os.sched_setaffinity(0, _ALLOWED)
        try:
            from threadpoolctl import threadpool_limits

            threadpool_limits(limits=len(_ALLOWED) if _ALLOWED else os.cpu_count())
        except ImportError:
            pass
        return None
    chosen = None
    if hasattr(os, "sched_setaffinity"):
        chosen = _ALLOWED[:cores]
        os.sched_setaffinity(0, chosen)
    else:
        try:
            import psutil

            process = psutil.Process()
            chosen = sorted(process.cpu_affinity())[:cores]
            process.cpu_affinity(chosen)
        except (ImportError, AttributeError, OSError):
            pass                              # macOS: no affinity, threads only
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=cores)
    except ImportError:
        pass
    return chosen


def child_main(conn) -> None:
    """The child's loop (top-level so ``spawn`` can import it)."""
    # BLAS threads are set per job (set_cores); without a limit numpy takes every core.
    import numpy  # noqa: F401  (warm: the first job does not pay for the imports)
    import scipy.linalg  # noqa: F401

    from .. import progress
    from . import actions

    set_cores(None)                           # remember the processors it may use

    progress.SINK = _throttled(conn, "progress", 0.5)
    progress.EVERY = 5.0                      # a printed line every 5 s in the terminal
    actions.FRAME_SINK = _throttled(conn, "frame", 0.5)
    sys.stdout = sys.stderr = _Pipe(conn)
    conn.send(("ready", os.getpid()))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message[0] == "quit":
            return
        _, func, args, kwargs, cores = message
        try:
            chosen = set_cores(cores)
            if chosen is not None:
                print(f"(núcleos {chosen}, {cores} hilos de álgebra lineal)")
            with warnings.catch_warnings():
                warnings.showwarning = lambda m, c, *a, **k: print(f"AVISO ({c.__name__}): {m}")
                result = func(*args, **kwargs)
            try:
                conn.send(("done", result))
            except Exception as error:  # noqa: BLE001  (an object that cannot cross processes)
                conn.send(("fail", f"El resultado no se pudo devolver a la ventana: {error}",
                           traceback.format_exc()))
        except Exception as error:  # noqa: BLE001  (shown in the window, never swallowed)
            conn.send(("fail", str(error), traceback.format_exc()))


def probe(seconds: float = 0.0, size: int = 0, steps: int = 0) -> dict:
    """A job for tests and for checking the setup: prints, counts ``steps`` with a progress
    counter, holds ``size`` MB of memory and sleeps; returns where and how it ran."""
    import numpy as np

    from ..progress import Progress

    print("probe: empieza")
    block = np.ones(int(size * 1e6 / 8)) if size else None
    if steps:
        with Progress(steps, "probe") as bar:
            for _ in range(steps):
                time.sleep(seconds / steps)
                bar.step()
    else:
        time.sleep(seconds)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {"pid": os.getpid(), "affinity": affinity,
            "held_mb": 0 if block is None else block.nbytes / 1e6}


class Worker:
    """One child process and its pipe. ``messages()`` never blocks."""

    def __init__(self):
        context = mp.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(target=child_main, args=(child,), daemon=True,
                                       name="tbkit-calculo")
        self.process.start()
        child.close()
        self.pid = self.process.pid
        self.ready = False
        self.busy = False

    @property
    def alive(self) -> bool:
        return self.process.is_alive()

    def submit(self, func, args=(), kwargs=None, cores: int | None = None) -> None:
        self.conn.send(("run", func, tuple(args), dict(kwargs or {}), cores))
        self.busy = True

    def messages(self) -> list[tuple]:
        out = []
        try:
            while self.conn.poll():
                message = self.conn.recv()
                if message[0] == "ready":
                    self.ready = True
                elif message[0] in ("done", "fail"):
                    self.busy = False
                out.append(message)
        except (EOFError, OSError):
            if self.busy:
                out.append(("fail", ("El proceso del cálculo terminó sin respuesta "
                                     f"(código {self.process.exitcode})."), ""))
                self.busy = False
        return out

    def kill(self) -> None:
        """Stop the process now (SIGKILL / TerminateProcess) and wait for it."""
        if self.process.is_alive():
            self.process.kill()
        self.process.join(timeout=5)
        self.busy = False
        self.conn.close()

    def close(self) -> None:
        """Ask an idle process to leave; kill it if it does not."""
        try:
            if self.process.is_alive() and not self.busy:
                self.conn.send(("quit",))
                self.process.join(timeout=2)
        except (OSError, BrokenPipeError):
            pass
        if self.process.is_alive():
            self.kill()
