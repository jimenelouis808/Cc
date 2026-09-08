"""The suite window: four instruments, one application.

Raman of carbon, Raman of dichalcogenides, X-ray diffraction and
electrochemistry, as four top-level sections. They are separate because
the measurements are separate — a diffractogram and a voltammogram have
nothing in common but the sample — and because a program that tried to
decide between "multi-walled nanotube" and "bilayer MoS₂" would be solving
a problem nobody has.

What they *do* share is one thing worth sharing: the two Raman sections
hold the same :class:`~ramancarbon.gui.state.Session`, so a spectrum
loaded in either appears in both. A file is a file.

Sections are built lazily, on first visit. Constructing all four at start
means importing matplotlib, building a dozen figures and reading the
reference library before the window appears, which turns a fast start into
a two-second one for the three sections the user did not open.
"""

from __future__ import annotations

from typing import Any, Optional

from ..plotting.style import PRESETS as PLOT_PRESET_LABELS
from .state import Session
from .theme import PAD, PALETTES, apply_theme

#: The plot presets offered in the header, in the order they are shown.
#: Screen first, then the journals, then the rest — which is the order
#: somebody actually needs them in.
PLOT_PRESETS: tuple[str, ...] = (
    "predeterminado", "acs", "acs-doble", "rsc", "elsevier", "nature",
    "aps", "wiley", "tesis", "presentacion", "poster", "grises", "cascada",
)

WINDOW_TITLE = "ramancarbon — caracterización de nanomateriales"

#: ``(key, label, subtitle)`` for each section, in the order they appear.
SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("carbono", "  Raman · carbono  ",
     "Nanotubos, grafeno, carbones: bandas D y G, cocientes, diámetros."),
    ("tmd", "  Raman · TMD  ",
     "Dicalcogenuros: número de capas, fase 2H/1T′, óxidos."),
    ("drx", "  DRX  ",
     "Identificación de fases contra estructuras cristalinas y Rietveld."),
    ("echem", "  Electroquímica  ",
     "CV, carga-descarga, impedancia, HER y OER."),
)


class Suite:
    """The top-level window holding the four sections."""

    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.session = Session()
        self.palette = PALETTES["claro"]
        self.fonts = apply_theme(root, self.palette)
        self.sections: dict[str, Any] = {}
        self._frames: dict[str, Any] = {}

        root.title(WINDOW_TITLE)
        root.geometry("1480x940")
        root.minsize(1100, 740)

        self._build_header()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=PAD["md"],
                           pady=(0, PAD["md"]))
        for key, label, _ in SECTIONS:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=label)
            self._frames[key] = frame
        self.notebook.bind("<<NotebookTabChanged>>", self._on_section_changed)
        self._ensure("carbono")

    # ==================================================================
    def _build_header(self) -> None:
        ttk = self.ttk
        header = ttk.Frame(
            self.root, padding=(PAD["lg"], PAD["md"], PAD["lg"], PAD["xs"])
        )
        header.pack(fill="x")
        ttk.Label(header, text="ramancarbon", style="Title.TLabel").pack(side="left")
        self.subtitle_var = self.tk.StringVar(value=SECTIONS[0][2])
        ttk.Label(header, textvariable=self.subtitle_var,
                  style="Muted.TLabel").pack(side="left", padx=(PAD["md"], 0))
        ttk.Button(header, text="Tema", command=self._toggle_theme).pack(side="right")

        # The plot preset lives in the header rather than inside one
        # section because it applies to every figure the suite saves, and
        # because the point of the presets is that the figure for the
        # manuscript comes out of the same window the work is done in.
        self.preset_var = self.tk.StringVar(value=self.session.plot_preset)
        chooser = ttk.Combobox(
            header, textvariable=self.preset_var, width=16, state="readonly",
            values=list(PLOT_PRESETS),
        )
        chooser.pack(side="right", padx=(0, PAD["sm"]))
        chooser.bind("<<ComboboxSelected>>", self._on_preset_changed)
        ttk.Label(header, text="Figura:", style="Muted.TLabel").pack(
            side="right", padx=(0, PAD["xs"]))

    def _on_preset_changed(self, _event=None) -> None:
        """Remember the chosen preset, and say what it means."""
        chosen = self.preset_var.get()
        self.session.plot_preset = chosen
        self.session.remember()
        description = PLOT_PRESET_LABELS.get(chosen, "")
        for section in self.sections.values():
            setter = getattr(section, "_set_status", None)
            if callable(setter):
                setter(f"Figuras: {chosen}"
                       + (f" — {description}" if description else ""))

    def _current_key(self) -> Optional[str]:
        try:
            index = self.notebook.index(self.notebook.select())
        except Exception:  # noqa: BLE001 - nothing selected yet
            return None
        if 0 <= index < len(SECTIONS):
            return SECTIONS[index][0]
        return None

    def _on_section_changed(self, _event=None) -> None:
        key = self._current_key()
        if key is None:
            return
        self.subtitle_var.set(dict((k, s) for k, _, s in SECTIONS)[key])
        self._ensure(key)
        section = self.sections.get(key)
        # The dichalcogenide section shares the carbon section's spectrum
        # list, so it has to re-read it every time it comes to the front.
        if key == "tmd" and section is not None:
            section.refresh()

    def _ensure(self, key: str) -> None:
        """Build a section the first time it is shown."""
        if key in self.sections:
            return
        frame = self._frames[key]
        if key == "carbono":
            from .app import RamanCarbonApp

            self.sections[key] = RamanCarbonApp(
                self.root, container=frame, session=self.session
            )
        elif key == "tmd":
            from .tmd_app import TMDApp

            section = TMDApp(self.root, frame, self.palette, self.fonts,
                             self.session)
            section.refresh()
            self.sections[key] = section
        elif key == "drx":
            from .xrd_app import XRDApp

            self.sections[key] = XRDApp(self.root, frame, self.palette, self.fonts)
        elif key == "echem":
            from .echem_app import EchemApp

            self.sections[key] = EchemApp(self.root, frame, self.palette, self.fonts)

    def _toggle_theme(self) -> None:
        from tkinter import messagebox

        self.session.palette_name = (
            "oscuro" if self.session.palette_name == "claro" else "claro"
        )
        messagebox.showinfo(
            "Cambio de tema",
            "El tema se aplicará completamente al reiniciar la aplicación.\n\n"
            "Las figuras sí cambian ahora mismo.",
            parent=self.root,
        )
        self.palette = PALETTES[self.session.palette_name]
        for section in self.sections.values():
            section.palette = self.palette
            redraw = getattr(section, "_redraw", None) or getattr(
                section, "_redraw_all", None
            )
            if redraw is not None:
                redraw()


_TK_MISSING = """
No se ha podido importar Tkinter, que es lo que dibuja la ventana.

Tkinter viene con Python pero algunas distribuciones lo empaquetan aparte:

  Debian / Ubuntu   sudo apt install python3-tk
  Fedora            sudo dnf install python3-tkinter
  Arch              sudo pacman -S tk
  macOS (Homebrew)  brew install python-tk
  Windows           reinstala Python desde python.org marcando «tcl/tk»

La línea de comandos funciona sin Tkinter:

  ramancarbon analizar espectro.txt --laser 532
  ramancarbon drx patron.xy
  ramancarbon echem cv.txt --velocidad 20 --masa 2
"""


def main() -> int:
    """Entry point for ``ramancarbon-gui``. Returns a process exit code."""
    try:
        import tkinter as tk
    except ImportError:
        print(_TK_MISSING)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"No se pudo abrir la ventana: {exc}")
        print("\n¿Estás en una sesión sin entorno gráfico (SSH sin X11)?")
        print("La línea de comandos funciona sin pantalla:\n")
        print("  ramancarbon analizar espectro.txt --laser 532")
        return 1
    Suite(root)
    root.mainloop()
    return 0


__all__ = ["SECTIONS", "WINDOW_TITLE", "Suite", "main"]
