"""Tkinter desktop application for carbonforge.

Layout: parameters on the left, a live 3D preview plus a validation summary
on the right. Structures are built on a worker thread so the window never
freezes on a large nanocoil, and results are marshalled back to the main
thread via ``root.after`` (Tk is not thread-safe).

The tabs live in :mod:`carbonforge.gui.tabs`, one module each; this class
only owns what they share.

Run with ``carbonforge-gui`` or ``python -m carbonforge.gui``.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any, Optional

from ase import Atoms

from .params import ADVANCED_KEY, ParamSpec
from .session import Session
from .tabs import AnalysisTab, BuilderTab, EdlcTab, ImportTab, PrepareTab, PreviewPanel
from .tabs.builder import _clock  # noqa: F401  (re-exported: tests and callers)

_TK_MISSING_MSG = """
No se encontró Tkinter, que es lo que dibuja la ventana.

  • Windows / macOS: reinstala Python desde python.org marcando la opción
    "tcl/tk and IDLE" durante la instalación.
  • Ubuntu / Debian:  sudo apt install python3-tk
  • Fedora:           sudo dnf install python3-tkinter
  • Arch:             sudo pacman -S tk

Mientras tanto puedes seguir usando la línea de comandos, que no necesita
Tkinter:

  carbonforge cnt --n 6 --m 6 --length 10 --out salida --format both
""".strip()


class CarbonForgeApp(BuilderTab, PreviewPanel, ImportTab, PrepareTab, EdlcTab, AnalysisTab):
    """Main application window: the notebook, and what its tabs share."""

    def __init__(self, root, vibspec_workdir: Optional[Path] = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("carbonforge — estructuras, cálculos y espectros de nanocarbono")
        self.root.geometry("1240x800")
        self.root.minsize(940, 620)

        self.atoms: Optional[Atoms] = None
        self._param_vars: dict[str, Any] = {}
        self._modifier_vars: dict[str, Any] = {}
        self._calculation_vars: dict[str, Any] = {}
        self._preset_vars: dict[str, Any] = {}
        self._functionalization_vars: dict[str, Any] = {}
        self._format_vars: dict[str, Any] = {}
        self._edlc_vars: dict[str, Any] = {}
        self.edlc_cell = None
        self._edlc_source: Optional[Atoms] = None
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        #: Cuándo empezó la construcción en curso, o None si no hay.
        self._build_started: float | None = None
        #: `after` pendiente del reloj, para poder cancelarlo.
        self._clock_job: str | None = None

        #: Fix panels (Construir and Preparar show the same list).
        self._fix_frames: list[Any] = []
        #: The structure the tabs hand to each other (gui/session.py).
        self.session = Session()
        self._vibspec = None
        self._vibspec_workdir: Optional[Path] = vibspec_workdir

        self._build_layout()
        self._rebuild_param_fields()
        self._poll_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    #: Sections in the order the work happens, and their pages.
    SECTIONS: dict[str, tuple[str, ...]] = {
        "Estructura": ("Construir", "Importar", "Modelo finito (IR)"),
        "Preparar": ("Cálculo (QE, SIESTA, LAMMPS)", "Celda EDLC (LAMMPS)"),
        "Calcular": ("IR con GPAW",),
        "Resultados": ("Bandas y espectros", "IR frente a FTIR"),
    }
    #: vibspec's pages and where they live in the window.
    VIBSPEC_PAGES: dict[str, str] = {
        "Modelo": "Modelo finito (IR)",
        "Cálculo": "IR con GPAW",
        "Resultados": "IR frente a FTIR",
    }

    def _build_layout(self) -> None:
        """Sections by task, each a notebook of pages.

        Make or bring a structure (Estructura), turn it into inputs
        (Preparar: QE, SIESTA, LAMMPS, EDLC),
        run what runs from here (Calcular), read results back (Resultados).
        The vibspec pages are built the first time one is shown: they probe
        for GPAW and start a job poller, which a builder-only session does
        not need.
        """
        tk, ttk = self.tk, self.ttk

        bar = ttk.Frame(self.root, padding=(8, 2))
        bar.pack(side="bottom", fill="x")
        self.current_var = tk.StringVar(value="Estructura actual: ninguna")
        ttk.Label(bar, textvariable=self.current_var, foreground="#335").pack(side="left")
        self.page_status_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.page_status_var, foreground="#667").pack(side="right")

        sections = ttk.Notebook(self.root)
        sections.pack(fill="both", expand=True, padx=6, pady=6)
        self._sections = sections
        self.pages: dict[str, Any] = {}
        self._page_home: dict[str, tuple[Any, Any]] = {}
        for section, pages in self.SECTIONS.items():
            holder = ttk.Frame(sections)
            sections.add(holder, text=f"  {section}  ")
            inner = ttk.Notebook(holder)
            inner.pack(fill="both", expand=True)
            inner.bind("<<NotebookTabChanged>>", self._on_page_changed)
            for page in pages:
                frame = ttk.Frame(inner, padding=4 if page in self.VIBSPEC_PAGES.values() else 0)
                inner.add(frame, text=f" {page} ")
                self.pages[page] = frame
                self._page_home[page] = (holder, inner)
        sections.bind("<<NotebookTabChanged>>", self._on_page_changed)

        self._build_builder_tab(self.pages["Construir"])
        self._build_import_tab(self.pages["Importar"])
        self._build_prepare_tab(self.pages["Cálculo (QE, SIESTA, LAMMPS)"])
        self._build_edlc_tab(self.pages["Celda EDLC (LAMMPS)"])
        self._build_analysis_tab(self.pages["Bandas y espectros"])
        self.session.subscribe(self._on_session_change)

    def select_page(self, page: str) -> None:
        """Show ``page`` (a name from :data:`SECTIONS`) and its section."""
        holder, inner = self._page_home[page]
        self._sections.select(holder)
        inner.select(self.pages[page])
        self._on_page_changed()

    def _current_page(self) -> Optional[str]:
        try:
            holder = self._sections.nametowidget(self._sections.select())
        except (KeyError, AttributeError, ValueError):
            return None
        for page, (home, inner) in self._page_home.items():
            if home is holder and str(inner.select()) == str(self.pages[page]):
                return page
        return None

    def _on_page_changed(self, _event=None) -> None:
        if self._current_page() in self.VIBSPEC_PAGES.values():
            self._ensure_vibspec()

    def _ensure_vibspec(self):
        """Build the vibspec pages once, into their frames."""
        if self._vibspec is None:
            from ..vibspec.gui.app import VibspecApp

            self._vibspec = VibspecApp(
                self.root, self._vibspec_workdir,
                frames={name: self.pages[page] for name, page in self.VIBSPEC_PAGES.items()},
                select=lambda name: self.select_page(self.VIBSPEC_PAGES[name]),
                status_var=self.page_status_var, session=self.session,
            )
        return self._vibspec

    def _on_session_change(self, current) -> None:
        self.current_var.set("Estructura actual: "
                             + ("ninguna" if current is None else current.describe()))

    def _on_close(self) -> None:
        """Close the window, unless vibspec jobs are running and the user
        prefers to keep them."""
        if self._vibspec is not None and not self._vibspec.confirm_close():
            return
        self.root.destroy()

    def _add_field(self, parent, spec: ParamSpec, store: dict[str, Any]) -> None:
        tk, ttk = self.tk, self.ttk
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)

        if spec.kind == "bool":
            var = tk.BooleanVar(value=bool(spec.default))
            ttk.Checkbutton(row, text=spec.label, variable=var).pack(anchor="w")
        else:
            ttk.Label(row, text=spec.label).pack(anchor="w")
            var = tk.StringVar(value=str(spec.default))
            if spec.kind == "choice":
                ttk.Combobox(
                    row, textvariable=var, values=list(spec.choices or ()),
                    state="readonly",
                ).pack(fill="x")
            else:
                ttk.Entry(row, textvariable=var).pack(fill="x")

        if spec.help:
            ttk.Label(
                row, text=spec.help, wraplength=320, justify="left",
                foreground="#777777", font=("TkDefaultFont", 8),
            ).pack(anchor="w")
        store[spec.key] = var

    def _all_values(self) -> dict[str, Any]:
        """The whole form, both pages: geometry and calculation."""
        return {
            **self._read_raw(self._param_vars),
            **self._read_raw(self._modifier_vars),
            **self._read_raw(self._functionalization_vars),
            **self._read_raw(self._calculation_vars),
            **self._read_raw(self._preset_vars),
            ADVANCED_KEY: getattr(self, "_advanced", {}),
        }

    def _read_raw(self, store: dict[str, Any]) -> dict[str, Any]:
        return {key: var.get() for key, var in store.items()}

    def _poll_queue(self) -> None:
        """Drain worker results on the main thread (Tk is not thread-safe)."""
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "built":
                    self._on_built(payload)
                elif kind == "edlc":
                    self._on_edlc_built(payload)
                elif kind == "edlc_error":
                    exc, tb = payload
                    self.edlc_build_button.configure(state="normal")
                    self.edlc_status_var.set("Error al construir la celda.")
                    self._show_error(exc, tb)
                elif kind == "error":
                    exc, tb = payload
                    self._set_busy(False, "Error.")
                    self._show_error(exc, tb)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    # ------------------------------------------------------------------
    def _show_error(self, exc: Exception, tb: str) -> None:
        from tkinter import messagebox

        # Builders and validators raise ValueError with user-facing text; only
        # unexpected exception types warrant dumping a traceback.
        if isinstance(exc, (ValueError, IndexError, RuntimeError)):
            messagebox.showerror("No se pudo construir", str(exc))
        else:
            messagebox.showerror(
                "Error inesperado", f"{type(exc).__name__}: {exc}\n\n{tb}"
            )
        self._set_info(f"❌ {type(exc).__name__}: {exc}")


def main(page: Optional[str] = None, vibspec_workdir: Optional[Path] = None) -> int:
    """Launch the GUI, optionally at ``page``. Returns a process exit code."""
    try:
        import tkinter as tk
    except ImportError:
        print(_TK_MISSING_MSG)
        return 1

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "Falta matplotlib, necesario para la vista previa 3D.\n"
            "Instálalo con:  pip install matplotlib"
        )
        return 1

    root = tk.Tk()
    app = CarbonForgeApp(root, vibspec_workdir=vibspec_workdir)
    if page is not None:
        app.select_page(page)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
