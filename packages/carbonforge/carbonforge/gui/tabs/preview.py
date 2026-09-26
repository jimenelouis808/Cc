"""The 3D preview panel: matplotlib canvas, named viewpoints, zoom, PNG."""

from __future__ import annotations

import traceback
from pathlib import Path

from ase import Atoms


class PreviewPanel:
    """The 3D view shown beside the builder form.

    Uses from the host: ``tk``, ``ttk``, ``root``, ``atoms``, ``status_var``.
    """

    def _build_preview(self, parent) -> None:
        ttk = self.ttk
        from matplotlib.backends.backend_tkagg import (  # noqa: WPS433
            FigureCanvasTkAgg,
        )
        from matplotlib.figure import Figure  # noqa: WPS433

        preview = ttk.LabelFrame(parent, text="Vista previa 3D", padding=4)
        preview.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(6, 4.6), dpi=100)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_title("Pulsa «Construir y previsualizar»")

        self.canvas = FigureCanvasTkAgg(self.figure, master=preview)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # Matplotlib's toolbar is built for a 2D axes. On this one its
        # magnifier and its pan cross do not do what their icons say, and
        # it has no rotate button -- rotation is the drag gesture, which
        # is what makes the toolbar look like it is missing one. So: the
        # canonical viewpoints, a zoom that scales the axis ranges, and
        # the gesture written down. Its save button did work here, so
        # that is kept rather than lost with the rest.
        self._build_view_bar(preview)
        self.canvas.draw()

        info = ttk.LabelFrame(parent, text="Resumen y validación", padding=4)
        info.pack(fill="both", expand=False, pady=(8, 0))

        self.info_text = self.tk.Text(info, height=11, wrap="word")
        info_scroll = ttk.Scrollbar(
            info, orient="vertical", command=self.info_text.yview
        )
        self.info_text.configure(yscrollcommand=info_scroll.set, state="disabled")
        self.info_text.pack(side="left", fill="both", expand=True)
        info_scroll.pack(side="right", fill="y")

    #: Vistas con nombre, como (elevación, azimut).
    VIEWPOINTS: tuple[tuple[str, float, float], ...] = (
        ("+x", 0.0, 0.0), ("+y", 0.0, 90.0), ("+z", 90.0, -90.0),
        ("iso", 22.0, -60.0),
    )

    def _build_view_bar(self, parent) -> None:
        """Puntos de vista, zoom y guardado, bajo el lienzo."""
        ttk = self.ttk
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(4, 0))

        ttk.Label(bar, text="vista:").pack(side="left")
        for label, elev, azim in self.VIEWPOINTS:
            ttk.Button(bar, text=label, width=4,
                       command=lambda e=elev, a=azim: self._set_view(e, a)
                       ).pack(side="left", padx=1)

        ttk.Label(bar, text="  zoom:").pack(side="left")
        ttk.Button(bar, text="+", width=3,
                   command=lambda: self._zoom(1 / 1.25)).pack(side="left", padx=1)
        ttk.Button(bar, text="−", width=3,
                   command=lambda: self._zoom(1.25)).pack(side="left", padx=1)
        ttk.Button(bar, text="ajustar", width=8,
                   command=self._zoom_fit).pack(side="left", padx=1)

        ttk.Button(bar, text="guardar imagen…", width=15,
                   command=self._save_image).pack(side="right")
        ttk.Label(bar, text="arrastra para rotar",
                  font=("TkDefaultFont", 8)).pack(side="right", padx=(0, 8))

    def _set_view(self, elevation: float, azimuth: float) -> None:
        self.axes.view_init(elev=elevation, azim=azimuth)
        self.canvas.draw_idle()

    def _zoom(self, factor: float) -> None:
        """Escala los tres rangos a la vez, sobre su centro común.

        Los tres juntos y por el mismo factor: escalarlos por separado
        deformaría la estructura, y un eje 3D no tiene recuadro elástico
        que arrastrar.
        """
        self._zoom_scale = max(0.05, min(20.0,
                                         getattr(self, "_zoom_scale", 1.0) * factor))
        self._apply_limits()

    def _zoom_fit(self) -> None:
        self._zoom_scale = 1.0
        self._apply_limits()

    def _apply_limits(self) -> None:
        """Límites de aspecto igual, al zoom actual."""
        atoms = getattr(self, "atoms", None)
        if atoms is None or not len(atoms):
            return
        pos = atoms.get_positions()
        scale = getattr(self, "_zoom_scale", 1.0)
        span = (float((pos.max(axis=0) - pos.min(axis=0)).max()) / 2.0 or 1.0) * scale
        mid = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
        self.axes.set_xlim(mid[0] - span, mid[0] + span)
        self.axes.set_ylim(mid[1] - span, mid[1] + span)
        self.axes.set_zlim(mid[2] - span, mid[2] + span)
        self.canvas.draw_idle()

    def _save_image(self) -> None:
        """Guarda la vista previa tal como está, con su punto de vista."""
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            title="Guardar la vista previa", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="estructura.png",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except (OSError, ValueError) as exc:
            self.status_var.set(f"No se pudo guardar: {exc}")
            return
        self.status_var.set(f"Guardado {Path(path).name}")

    def _render(self, atoms: Atoms) -> None:
        from ...viz.plot import draw_structure_on_axes

        self.axes.clear()
        # Bond inference is O(N^2); skip the wireframe on very large models so
        # the preview stays responsive. The atoms themselves still render.
        draw_structure_on_axes(atoms, self.axes, show_bonds=len(atoms) <= 4000)
        self.canvas.draw_idle()

    def _on_save_png(self) -> None:
        from tkinter import filedialog, messagebox

        if self.atoms is None:
            return
        path = filedialog.asksaveasfilename(
            title="Guardar imagen",
            defaultextension=".png",
            filetypes=[("Imagen PNG", "*.png")],
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:
            self._show_error(exc, traceback.format_exc())
            return
        self.status_var.set(f"Imagen guardada en {path}")
        messagebox.showinfo("Imagen guardada", str(path))
