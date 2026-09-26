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
from typing import Any, Optional

from ase import Atoms

from .params import ParamSpec
from .tabs import AnalysisTab, BuilderTab, EdlcTab, ImportTab, PreviewPanel
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


class CarbonForgeApp(BuilderTab, PreviewPanel, ImportTab, EdlcTab, AnalysisTab):
    """Main application window: the notebook, and what its tabs share."""

    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("carbonforge — generador de nanoestructuras de carbono")
        self.root.geometry("1180x760")
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

        self._build_layout()
        self._rebuild_param_fields()
        self._poll_queue()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        """Four tabs, in the order the work happens.

        Build or import a structure, turn it into an EDLC cell if that is
        what you are after, then read the finished run back.
        """
        ttk = self.ttk

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)

        build_tab = ttk.Frame(notebook)
        import_tab = ttk.Frame(notebook)
        edlc_tab = ttk.Frame(notebook)
        analyse_tab = ttk.Frame(notebook)
        notebook.add(build_tab, text="  Construir estructura  ")
        notebook.add(import_tab, text="  Importar y preparar  ")
        notebook.add(edlc_tab, text="  Celda EDLC (LAMMPS)  ")
        notebook.add(analyse_tab, text="  Analizar resultados  ")

        self._build_builder_tab(build_tab)
        self._build_import_tab(import_tab)
        self._build_edlc_tab(edlc_tab)
        self._build_analysis_tab(analyse_tab)

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


def main() -> int:
    """Launch the GUI. Returns a process exit code."""
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
    CarbonForgeApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
