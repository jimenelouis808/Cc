"""Plumbing shared by every section of the suite.

Four instruments, four sections, and the same four problems in each: embed
a matplotlib figure, run a slow calculation without freezing the window,
show a status line, and report an exception without a traceback in the
console. :class:`SectionApp` holds those and nothing else, so a section is
its layout and its logic rather than a copy of this file.

The threading rule is the one that matters. Tk is not thread-safe, so a
worker thread never touches a widget: it returns through a queue and the
main thread drains it. Getting this wrong does not crash — it corrupts the
display intermittently on some platforms and not at all on others, which
is the worst kind of bug to be handed.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Callable, Optional

from .theme import PAD, Palette, matplotlib_style


def _clock(seconds: float) -> str:
    """Tiempo transcurrido como lo lee un cronómetro.

    Segundos por debajo del minuto y m:ss por encima: «143 s» obliga a
    dividir y «2:23» no.
    """
    if seconds < 60.0:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


class SectionApp:
    """Base for one instrument's section: canvases, threading, status."""

    def __init__(self, root, container, palette: Palette, fonts: dict) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.container = container
        self.palette = palette
        self.fonts = fonts
        self.queue: queue.Queue = queue.Queue()
        self.busy = False
        self._canvases: dict[str, Any] = {}
        self._figures: dict[str, Any] = {}
        self._dirty: set[str] = set()
        self.status_var = tk.StringVar(value="")
        #: Tiempo transcurrido del cálculo en curso, junto a la barra.
        self.elapsed_var = tk.StringVar(value="")
        #: Palette for the FIGURES. Starts as the chrome palette and is
        #: replaced by the suite when the user asks for a figure
        #: background that differs from the window's.
        self.figure_palette = palette
        self._started: float | None = None
        self._clock_job: str | None = None
        self.progress = None

    # -- canvases ------------------------------------------------------
    def make_canvas(self, parent, key: str, subplots: Callable, figsize=(7.6, 5.0)):
        """Embed a matplotlib figure, remembering it under ``key``."""
        import matplotlib
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        with matplotlib.rc_context(matplotlib_style(self.figure_palette)):
            figure = Figure(figsize=figsize, dpi=100)
            subplots(figure)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        widget = canvas.get_tk_widget()
        widget.configure(background=self.palette.surface, highlightthickness=0)
        widget.pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.configure(background=self.palette.surface_alt)
        toolbar.update()
        toolbar.pack(fill="x")
        # A real reset, next to matplotlib's own Home. Home rewinds the
        # view STACK, so after a redraw with new data it restores limits
        # that belonged to the previous figure; and if the stack is empty
        # it does nothing at all, which reads as a dead button. This
        # redraws from the data and autoscales, which is what "reset zoom"
        # means to anyone who presses it.
        self.ttk.Button(toolbar, text="Restablecer zoom",
                        command=lambda k=key: self.reset_zoom(k)).pack(
            side="right", padx=PAD["xs"])
        self._canvases[key] = canvas
        self._figures[key] = figure
        return figure, canvas

    def reset_zoom(self, key: str) -> None:
        """Redraw one canvas at the limits its data imply."""
        canvas = self._canvases.get(key)
        if canvas is None:
            return
        toolbar = getattr(canvas, "toolbar", None)
        if toolbar is not None and hasattr(toolbar, "_nav_stack"):
            try:
                toolbar._nav_stack.clear()
            except Exception:  # noqa: BLE001 - private API, best effort
                pass
        drawer = getattr(self, "_drawers", {}).get(key)
        if drawer is not None:
            self.with_style(key, drawer)
            return
        for axes in self._figures[key].axes:
            axes.relim()
            axes.autoscale()
        canvas.draw_idle()

    def with_style(self, key: str, draw: Callable) -> None:
        """Redraw one canvas under the current palette's matplotlib style."""
        import matplotlib

        figure = self._figures.get(key)
        canvas = self._canvases.get(key)
        if figure is None or canvas is None:
            return
        with matplotlib.rc_context(matplotlib_style(self.figure_palette)):
            figure.clear()
            draw(figure)
            figure.tight_layout()
        canvas.draw_idle()

    def mark_dirty(self, *keys: str) -> None:
        self._dirty.update(keys)

    def flush_dirty(self, visible: Optional[tuple[str, ...]] = None) -> None:
        """Redraw the canvases that are both dirty and on screen.

        Redrawing every figure on every change is the slowest thing a
        section does. Anything not currently visible is left dirty and
        drawn when its tab is opened.
        """
        targets = self._dirty if visible is None else (self._dirty & set(visible))
        for key in list(targets):
            drawer = getattr(self, "_drawers", {}).get(key)
            if drawer is None:
                continue
            self.with_style(key, drawer)
            self._dirty.discard(key)

    # -- threading -----------------------------------------------------
    def run_async(
        self, work: Callable[[], Any], done: Callable[[Any], None], message: str
    ) -> None:
        """Run ``work`` on a thread; call ``done`` on the main thread after."""
        if self.busy:
            self.set_status("Ya hay un cálculo en marcha; espera a que termine.")
            return
        self.busy = True
        if self.progress is not None:
            self.progress.start(12)
        self.set_status(message)
        self._started = time.monotonic()
        self._tick_clock()

        def target() -> None:
            try:
                result = work()
                self.queue.put(("ok", done, result))
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self.queue.put(("error", done, (exc, traceback.format_exc())))

        threading.Thread(target=target, daemon=True).start()

    def report_progress(self, text: str) -> None:
        """Show a line from a worker thread, without ending the job.

        A worker thread must never touch a widget — doing it wrong does
        not crash, it corrupts the display occasionally on some platforms
        and never on others — so progress goes through the same queue as
        the result and the main thread puts it on screen. What separates
        it from a result is that it leaves ``busy`` alone: a refinement
        reporting its twentieth iteration has not finished, and stopping
        the progress bar there would say it had.
        """
        self.queue.put(("progress", None, text))

    def drain_queue(self) -> None:
        try:
            while True:
                kind, done, payload = self.queue.get_nowait()
                if kind == "progress":
                    self.status_var.set(str(payload))
                    continue
                self.busy = False
                if self.progress is not None:
                    self.progress.stop()
                self._stop_clock()
                if kind == "ok":
                    done(payload)
                else:
                    exception, tb = payload
                    self.show_error(exception, tb)
        except queue.Empty:
            pass
        self.root.after(150, self.drain_queue)

    # -- chrome --------------------------------------------------------
    def build_status(self, parent) -> None:
        ttk = self.ttk
        bar = ttk.Frame(parent, style="Toolbar.TFrame",
                        padding=(PAD["lg"], PAD["sm"]))
        bar.pack(fill="x", side="bottom")
        ttk.Label(bar, textvariable=self.status_var,
                  style="Status.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
        self.progress.pack(side="right")
        ttk.Label(bar, textvariable=self.elapsed_var,
                  style="Status.TLabel").pack(side="right", padx=(0, PAD["sm"]))

    def _tick_clock(self) -> None:
        """Cuenta mientras el cálculo corre.

        Una barra indeterminada dice «algo está pasando» y nada más, que
        es justo la duda que crea un refinamiento Rietveld o un lote de
        cincuenta espectros. El tiempo transcurrido dice cuánto lleva
        pasando, y es lo que separa un cálculo lento de uno colgado.
        """
        if getattr(self, "_started", None) is None:
            return
        self.elapsed_var.set(_clock(time.monotonic() - self._started))
        self._clock_job = self.root.after(250, self._tick_clock)

    def _stop_clock(self) -> None:
        job = getattr(self, "_clock_job", None)
        if job is not None:
            self.root.after_cancel(job)
            self._clock_job = None
        started = getattr(self, "_started", None)
        self._started = None
        if started is not None:
            self.elapsed_var.set(_clock(time.monotonic() - started))

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def flush_messages(self, messages) -> None:
        """Move the newest queued message into the status bar."""
        if not messages:
            return
        level, text = messages[-1]
        prefix = {"error": "✗ ", "warning": "⚠ ", "info": ""}.get(level, "")
        self.set_status(prefix + text)

    def warn(self, title: str, message: str) -> None:
        from tkinter import messagebox

        messagebox.showwarning(title, message, parent=self.root)

    def canvas_limits(self, key: str) -> Optional[tuple[float, float]]:
        """The x-limits currently shown on one canvas, low first.

        For the sections that let the user pick a range by zooming to it
        rather than by typing two numbers. Returned low-first because
        several of these axes are inverted — binding energy runs right to
        left — and ``get_xlim`` gives them back in display order.
        """
        figure = self._figures.get(key)
        if figure is None or not figure.axes:
            return None
        left, right = figure.axes[0].get_xlim()
        return (float(min(left, right)), float(max(left, right)))

    def ask_yes_no(self, title: str, message: str) -> bool:
        """Confirm before something the user cannot undo."""
        from tkinter import messagebox

        return bool(messagebox.askyesno(title, message, parent=self.root))

    def show_text(self, title: str, text: str) -> None:
        """A scrollable, selectable, copyable window of text.

        Not a message box. A refinement report is a page long and the
        reason to look at it is usually to copy a number out of it, and a
        message box lets you do neither.
        """
        from .widgets import scrolled_text, set_text

        window = self.tk.Toplevel(self.root)
        window.title(title)
        window.geometry("820x620")
        window.configure(background=self.palette.background)
        frame = self.ttk.Frame(window, padding=PAD["md"])
        frame.pack(fill="both", expand=True)
        widget = scrolled_text(frame, self.palette, self.fonts["mono"], height=34)
        set_text(widget, text)
        self.ttk.Button(frame, text="Cerrar", command=window.destroy).pack(
            anchor="e", pady=(PAD["sm"], 0))
        return window

    def show_error(self, exception: Exception, tb: str) -> None:
        from tkinter import messagebox

        self.set_status(f"✗ {exception}")
        messagebox.showerror(
            "Error",
            f"{type(exception).__name__}: {exception}\n\n{tb[-1500:]}",
            parent=self.root,
        )


def placeholder(ax, text: str, palette: Palette) -> None:
    """Empty-state message on an otherwise blank axes."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            color=palette.text_muted, fontsize=10, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


__all__ = ["SectionApp", "placeholder"]
