"""Run a calculation in a child process and hand the result back as a signal.

The calculation runs in :class:`tbkit.gui.jobs.Worker` (a separate process), polled from a
timer of the GUI thread: there is no worker thread in the window's process, so Python's
collector never runs Qt destructors off the GUI thread (the crash the old thread runner had
to guard against), and «Cancelar» kills the calculation at once instead of waiting for it.

Each job gets the cores chosen in the window (affinity + BLAS threads) and a RAM limit:
the process is killed when its memory passes it, before the operating system kills
something else (or the window) for lack of memory.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QTimer, Signal

from . import jobs
from .resources import Monitor, gb


class Runner(QObject):
    """One calculation at a time; ``busy`` tells the window to grey out buttons.

    Callbacks (``on_done``, ``on_error``) run in the GUI thread, from the polling timer.
    """

    busy = Signal(bool)
    message = Signal(str)
    #: The terminal: stamped lines of the runner's own, and the jobs' raw output.
    log_line = Signal(str)
    output = Signal(str)
    #: A progress counter inside the job (``tbkit.progress``): dict with ``line``.
    progress = Signal(dict)
    #: Positions of the structure being relaxed, as it moves.
    frame = Signal(object)
    #: Every second while a job runs: {"pid", "rss_gb", "cpu_percent", "seconds"}.
    usage = Signal(dict)

    POLL_MS = 50

    def __init__(self, parent=None, prestart: bool = True):
        super().__init__(parent)
        self.worker: jobs.Worker | None = None
        self.cores: int | None = None          # None: every core the window may use
        self.memory_limit_gb: float | None = None
        self.monitor = Monitor()
        self._current = None                   # (label, on_done, on_error)
        self.queue: list[tuple] = []           # jobs asked for while one runs, in order
        self._started = 0.0
        self._last_usage = 0.0
        self.last_usage: dict = {}
        self.output_text: list[str] = []       # this job's output (the Procesos tab)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(self.POLL_MS)
        if prestart:
            self._ensure_worker()

    # ------------------------------------------------------------------ state
    @property
    def running(self) -> bool:
        return self._current is not None

    @property
    def pid(self) -> int | None:
        return self.worker.pid if self.worker is not None else None

    @property
    def label(self) -> str | None:
        return self._current[0] if self._current else None

    def _ensure_worker(self):
        if self.worker is None or not self.worker.alive:
            self.worker = jobs.Worker()

    # ------------------------------------------------------------------ jobs
    def start(self, label, func, *args, on_done=None, on_error=None, **kwargs):
        """Run ``func(*args, **kwargs)`` in the job process; with one running, queue it."""
        if self.running:
            self.queue.append((label, func, args, kwargs, on_done, on_error))
            self.log_line.emit(f"en cola: {label} (detrás de {self.label})")
            self.message.emit(f"{label}: en cola.")
            return True
        from .console import command_line

        self._ensure_worker()
        self._current = (label, on_done, on_error)
        self._started = time.perf_counter()
        self.output_text = []
        cores = f" · {self.cores} núcleos" if self.cores else ""
        limit = f" · RAM ≤ {self.memory_limit_gb:.1f} GB" if self.memory_limit_gb else ""
        self.log_line.emit(f"$ {label}: {command_line(func, args, kwargs)}"
                           f"  [proceso {self.worker.pid}{cores}{limit}]")
        try:
            self.worker.submit(func, args, kwargs, self.cores)
        except Exception as error:             # arguments that cannot cross processes
            self._current = None
            self.log_line.emit(f"✗ {label}: no se pudo enviar al proceso de cálculo: {error}")
            if on_error:
                on_error(f"No se pudo enviar el cálculo: {error}", "")
            return False
        self.busy.emit(True)
        self.message.emit(f"{label}…")
        return True

    def _poll(self):
        if self.worker is None:
            return
        for message in self.worker.messages():
            kind = message[0]
            if kind == "out":
                self.output_text.append(message[1])
                self.output.emit(message[1])
            elif kind == "progress":
                self.progress.emit(message[1])
            elif kind == "frame":
                self.frame.emit(message[1])
            elif kind == "done":
                self._finish(result=message[1])
            elif kind == "fail":
                self._finish(error=(message[1], message[2]))
        if self.running:
            self._watch()
        elif self.worker is not None and not self.worker.alive:
            self.worker = None
            self._ensure_worker()

    def _watch(self):
        """Once a second: the job's memory and CPU; kill it past the RAM limit."""
        now = time.perf_counter()
        if now - self._last_usage < 1.0:
            return
        self._last_usage = now
        info = self.monitor.process(self.worker.pid)
        if info is None:
            if not self.worker.alive:
                self._finish(error=((f"El proceso del cálculo murió (código "
                                     f"{self.worker.process.exitcode}); puede ser falta de "
                                     "memoria."), ""))
            return
        self.last_usage = {"pid": self.worker.pid, "seconds": now - self._started, **info}
        self.usage.emit(self.last_usage)
        if self.memory_limit_gb and info["rss_gb"] > self.memory_limit_gb:
            label = self.label
            self.kill(reason=f"pasó el límite de RAM ({gb(info['rss_gb'])} > "
                             f"{self.memory_limit_gb:.1f} GB)")
            self.message.emit(f"{label}: detenido por memoria.")

    def _finish(self, result=None, error=None):
        if self._current is None:
            return
        label, on_done, on_error = self._current
        self._current = None
        self.last_usage = {}
        elapsed = time.perf_counter() - self._started
        self.busy.emit(False)
        if error is None:
            self.log_line.emit(f"✓ {label} ({elapsed:.1f} s)")
            self.message.emit(f"{label}: listo.")
            if on_done:
                on_done(result)
        else:
            text, trace = error
            self.log_line.emit(f"✗ {label} ({elapsed:.1f} s): {text}")
            if trace:
                self.output.emit(trace)
            self.message.emit(f"{label}: error.")
            if on_error:
                on_error(text, trace)
        self._next()

    def _next(self):
        if self.queue and not self.running:
            label, func, args, kwargs, on_done, on_error = self.queue.pop(0)
            self.start(label, func, *args, on_done=on_done, on_error=on_error, **kwargs)

    def kill(self, reason: str = "cancelado"):
        """Kill the running job now; a fresh process is started for the next one."""
        if not self.running or self.worker is None:
            return
        label, _, on_error = self._current
        self._current = None
        self.last_usage = {}
        pid = self.worker.pid
        self.worker.kill()
        self.worker = None
        elapsed = time.perf_counter() - self._started
        self.log_line.emit(f"■ {label} detenido tras {elapsed:.1f} s ({reason}); "
                           f"proceso {pid} terminado")
        self.busy.emit(False)
        self._ensure_worker()
        if reason != "cancelado" and on_error:
            on_error(f"{label} detenido: {reason}.", "")

    def cancel(self):
        """Stop the running job (its process is killed: nothing keeps computing) and empty
        the queue."""
        if self.queue:
            self.log_line.emit(f"cola vaciada: {', '.join(q[0] for q in self.queue)}")
            self.queue.clear()
        if self.running:
            self.message.emit("Cancelado: el cálculo se detuvo.")
            self.kill("cancelado")

    def shutdown(self):
        """When the window closes: stop the job and the idle process."""
        self._timer.stop()
        if self.worker is not None:
            if self.running:
                self.worker.kill()
            else:
                self.worker.close()
            self.worker = None
