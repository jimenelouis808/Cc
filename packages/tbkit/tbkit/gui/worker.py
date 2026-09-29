"""Run a calculation off the GUI thread and hand the result back as a signal."""

from __future__ import annotations

import contextlib
import gc
import time
import traceback
import warnings

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot


class _Job(QObject):
    done = Signal(object)
    failed = Signal(str, str)

    def __init__(self, func, args, kwargs, stream=None):
        super().__init__()
        self.func, self.args, self.kwargs = func, args, kwargs
        self.stream = stream

    def run(self):
        # What the calculation prints (and its warnings) goes to the window's
        # terminal while it runs. sys.stdout is global: prints from the GUI
        # thread in that time land there too, which is where they belong.
        redirect = (contextlib.redirect_stdout(self.stream), contextlib.redirect_stderr(self.stream)) \
            if self.stream is not None else ()
        try:
            with contextlib.ExitStack() as stack, warnings.catch_warnings():
                for context in redirect:
                    stack.enter_context(context)
                if self.stream is not None:
                    warnings.showwarning = self._warning
                self.done.emit(self.func(*self.args, **self.kwargs))
        except Exception as error:            # shown to the user, never swallowed
            self.failed.emit(str(error), traceback.format_exc())

    def _warning(self, message, category, filename, lineno, file=None, line=None):
        self.stream.write(f"AVISO ({category.__name__}): {message}\n")


class Runner(QObject):
    """One calculation at a time; ``busy`` tells the window to grey out buttons.

    Results come back through slots of this object, which lives in the GUI
    thread (queued connections): callbacks never run in the worker thread.
    Python threads cannot be killed safely: "cancel" only drops the result of
    the running job when it arrives.
    """

    busy = Signal(bool)
    message = Signal(str)
    #: The terminal: stamped lines of the runner's own, and the jobs' raw output.
    log_line = Signal(str)
    output = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._job = None
        self._generation = 0
        self._current = None          # (generation, label, on_done, on_error)
        # Finished QThread/_Job pairs, released here in the GUI thread. Dropping
        # them at once let Python's collector destroy Qt objects from whatever
        # thread it ran in -- the next worker -- and corrupt memory (segfaults
        # at random places).
        self._finished = []
        self._gc_enabled = True
        self._started = 0.0
        from .console import Stream

        self.stream = Stream()
        self.stream.text.connect(self.output, Qt.QueuedConnection)

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self, label, func, *args, on_done=None, on_error=None, **kwargs):
        if self.running:
            self.message.emit("Ya hay un cálculo en marcha.")
            return False
        self._finished.clear()        # GUI thread: safe to destroy the old pairs
        # Python's cyclic collector runs in whichever thread allocates when it
        # triggers: in the worker it destroyed Qt objects of the GUI thread
        # (matplotlib/Qt cycles) and crashed the window. Pause it while a job
        # runs (arrays are still freed by reference counting) and collect in
        # the GUI thread when the job ends.
        self._gc_enabled = gc.isenabled()
        gc.disable()
        self._generation += 1
        self._current = (self._generation, label, on_done, on_error)
        from .console import command_line

        self._started = time.perf_counter()
        self.log_line.emit(f"$ {label}: {command_line(func, args, kwargs)}")
        thread = QThread()
        job = _Job(func, args, kwargs, self.stream)
        job.moveToThread(thread)
        thread.started.connect(job.run)
        job.done.connect(self._done, Qt.QueuedConnection)
        job.failed.connect(self._failed, Qt.QueuedConnection)
        self._thread, self._job = thread, job
        self.busy.emit(True)
        self.message.emit(f"{label}…")
        thread.start()
        return True

    def _finish(self):
        self._thread.quit()
        self._thread.wait()
        self._finished.append((self._thread, self._job))
        self._thread = self._job = None
        if self._gc_enabled:
            gc.enable()
        gc.collect()
        self.busy.emit(False)
        current, self._current = self._current, None
        return current

    @Slot(object)
    def _done(self, result):
        generation, label, on_done, _ = self._finish()
        elapsed = time.perf_counter() - self._started
        self.log_line.emit(f"✓ {label} ({elapsed:.1f} s)" if generation == self._generation
                           else f"· {label} terminó tras cancelarse; resultado descartado")
        if generation == self._generation:
            self.message.emit(f"{label}: listo.")
            if on_done:
                on_done(result)

    @Slot(str, str)
    def _failed(self, text, trace):
        generation, label, _, on_error = self._finish()
        elapsed = time.perf_counter() - self._started
        self.log_line.emit(f"✗ {label} ({elapsed:.1f} s): {text}")
        self.output.emit(trace)
        if generation == self._generation:
            self.message.emit(f"{label}: error.")
            if on_error:
                on_error(text, trace)

    def cancel(self):
        """Forget the running job: its result will be ignored."""
        if self.running:
            self._generation += 1
            self.message.emit("Cancelado: el resultado se descartará al llegar.")
            self.log_line.emit("cancelado: el cálculo sigue hasta terminar y su resultado se "
                               "descarta")
