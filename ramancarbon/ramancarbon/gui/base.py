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
import traceback
from typing import Any, Callable, Optional

from .theme import PAD, Palette, matplotlib_style


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

        with matplotlib.rc_context(matplotlib_style(self.palette)):
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
        self._canvases[key] = canvas
        self._figures[key] = figure
        return figure, canvas

    def with_style(self, key: str, draw: Callable) -> None:
        """Redraw one canvas under the current palette's matplotlib style."""
        import matplotlib

        figure = self._figures.get(key)
        canvas = self._canvases.get(key)
        if figure is None or canvas is None:
            return
        with matplotlib.rc_context(matplotlib_style(self.palette)):
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

        def target() -> None:
            try:
                result = work()
                self.queue.put(("ok", done, result))
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self.queue.put(("error", done, (exc, traceback.format_exc())))

        threading.Thread(target=target, daemon=True).start()

    def drain_queue(self) -> None:
        try:
            while True:
                kind, done, payload = self.queue.get_nowait()
                self.busy = False
                if self.progress is not None:
                    self.progress.stop()
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
