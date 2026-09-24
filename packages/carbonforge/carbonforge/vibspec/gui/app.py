"""The vibspec window: model, calculation, results. Widgets only.

Every decision lives in :mod:`carbonforge.vibspec.gui.logic` (and below it in
:mod:`carbonforge.vibspec.core`); this module lays out Tk widgets, reads
them, calls the logic and draws what comes back. Calculations run as
subprocesses through :class:`~carbonforge.vibspec.gui.logic.JobQueue`, polled
from Tk's event loop with ``after``, so the window never blocks on DFT.

Run with ``carbonforge vibspec gui`` or ``python -m carbonforge.vibspec.gui``.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import logic

_POLL_MS = 1000
_FRAME_MS = 60

_TK_MISSING = (
    "No se encontró Tkinter, que es lo que dibuja la ventana.\n"
    "  • Windows: reinstala Python desde python.org con la opción 'tcl/tk'.\n"
    "  • Ubuntu:  sudo apt install python3-tk\n"
    "La línea de comandos (carbonforge vibspec ...) funciona sin Tkinter."
)


class VibspecApp:
    """Main window. Holds widget state only; the logic layer holds the rest."""

    def __init__(self, root, workdir: Optional[Path] = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.root = tk, ttk, root
        root.title("carbonforge · vibspec — IR de nanocintas funcionalizadas")
        root.geometry("1280x800")

        self.workdir = Path(workdir or Path.cwd() / "calculos")
        self.queue = logic.JobQueue()
        self.model: Optional[logic.ModelResult] = None
        self.results_dir: Optional[Path] = None
        self.results_record = None
        self.experiment = None
        self.animation: Optional[dict[str, Any]] = None
        self.can_run, self.cannot_run_reason = logic.can_run_locally()

        # Packed before the notebook, so the notebook cannot squeeze it out.
        self.status_var = tk.StringVar(value="Construye un modelo para empezar.")
        ttk.Label(root, textvariable=self.status_var, anchor="w",
                  padding=(8, 2)).pack(side="bottom", fill="x")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook
        self.tabs = {name: ttk.Frame(notebook, padding=8)
                     for name in ("Modelo", "Cálculo", "Resultados")}
        for name, frame in self.tabs.items():
            notebook.add(frame, text=name)

        self._build_model_tab(self.tabs["Modelo"])
        self._build_calc_tab(self.tabs["Cálculo"])
        self._build_results_tab(self.tabs["Resultados"])

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(_POLL_MS, self._poll)

    # ------------------------------------------------------------------ helpers

    def _form(self, parent, specs, store: dict[str, Any]) -> None:
        """One labelled widget per ParamSpec, values kept as Tk variables."""
        tk, ttk = self.tk, self.ttk
        for row, spec in enumerate(specs):
            ttk.Label(parent, text=spec.label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(spec.default))
            if spec.kind == "choice":
                widget = ttk.Combobox(parent, textvariable=var, values=list(spec.choices),
                                      state="readonly", width=18)
            else:
                widget = ttk.Entry(parent, textvariable=var, width=20)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            if spec.help:
                ttk.Label(parent, text="?", foreground="#6b6b66").grid(row=row, column=2, padx=4)
                widget.bind("<Enter>", lambda _e, h=spec.help: self._status(h))
            store[spec.key] = var
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _read(store: dict[str, Any]) -> dict[str, Any]:
        return {key: var.get() for key, var in store.items()}

    def _text(self, parent, height: int):
        text = self.tk.Text(parent, height=height, wrap="word", font=("TkFixedFont", 9),
                            relief="flat", background="#f6f6f3")
        text.configure(state="disabled")
        return text

    @staticmethod
    def _set_text(widget, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _canvas(self, parent, figure):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    def _status(self, message: str) -> None:
        self.status_var.set(message)

    def _error(self, exc: Exception) -> None:
        from tkinter import messagebox

        traceback.print_exception(exc)
        messagebox.showerror("vibspec", str(exc) or type(exc).__name__)

    # ------------------------------------------------------------------ model tab

    def _build_model_tab(self, tab) -> None:
        from matplotlib.figure import Figure

        ttk = self.ttk
        left = ttk.Frame(tab, width=360)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True)

        tk = self.tk
        origin = ttk.LabelFrame(left, text="Origen de la geometría", padding=6)
        origin.pack(fill="x")
        self.source_mode = tk.StringVar(value="build")
        ttk.Radiobutton(origin, text="Construir una cinta", value="build",
                        variable=self.source_mode).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(origin, text="Desde archivo (tus átomos y grupos)", value="file",
                        variable=self.source_mode).grid(row=1, column=0, columnspan=3, sticky="w")
        self.source_var = tk.StringVar(value="")
        ttk.Entry(origin, textvariable=self.source_var).grid(row=2, column=0, columnspan=2,
                                                             sticky="ew")
        ttk.Button(origin, text="…", width=3, command=self._on_pick_source).grid(row=2, column=2)
        ttk.Label(origin, text="Biblioteca").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.library_var = tk.StringVar(value=str(self.workdir.parent / "estructuras"))
        self.library_combo = ttk.Combobox(origin, state="readonly", width=24)
        self.library_combo.grid(row=3, column=1, sticky="ew", pady=(4, 0))
        self.library_combo.bind("<<ComboboxSelected>>", self._on_library_pick)
        ttk.Button(origin, text="…", width=3, command=self._on_pick_library).grid(row=3, column=2,
                                                                                pady=(4, 0))
        origin.columnconfigure(1, weight=1)
        self._refresh_library()

        form = ttk.LabelFrame(left, text="Cinta y funcionalización", padding=6)
        form.pack(fill="x", pady=(6, 0))
        self.builder_vars: dict[str, Any] = {}
        self._form(form, logic.BUILDER_PARAMS, self.builder_vars)
        ttk.Label(form, text="Con un archivo, borde/ancho/largo se ignoran; la "
                             "funcionalización se añade encima.",
                  wraplength=330, foreground="#6b6b66").grid(
            row=len(logic.BUILDER_PARAMS), column=0, columnspan=3, sticky="w")

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="Construir", command=self._on_build).pack(side="left")
        ttk.Button(buttons, text="Guardar…", command=self._on_save_structure
                   ).pack(side="left", padx=4)
        ttk.Button(buttons, text="A la biblioteca", command=self._on_save_to_library
                   ).pack(side="left")
        ttk.Button(buttons, text="Ir a Cálculo",
                   command=lambda: self.notebook.select(self.tabs["Cálculo"])
                   ).pack(side="right")

        ttk.Label(left, text="Comprobaciones").pack(anchor="w")
        self.model_report = self._text(left, 18)
        self.model_report.pack(fill="both", expand=True)

        self.model_figure = Figure(figsize=(6, 6), layout="constrained")
        self.model_ax = self.model_figure.add_subplot(projection="3d")
        self.model_canvas = self._canvas(right, self.model_figure)

    # -- geometry source and library

    def _on_pick_source(self) -> None:
        from tkinter import filedialog

        from ..core import STRUCTURE_EXTENSIONS

        pattern = " ".join(f"*{ext}" for ext in STRUCTURE_EXTENSIONS)
        path = filedialog.askopenfilename(filetypes=[("Estructuras", pattern), ("Todos", "*.*")])
        if path:
            self.source_var.set(path)
            self.source_mode.set("file")

    def _refresh_library(self) -> None:
        files = logic.library(Path(self.library_var.get()))
        self.library_files = {path.name: path for path in files}
        self.library_combo.configure(values=list(self.library_files))
        self.library_combo.set("" if files else "(vacía)")

    def _on_pick_library(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(initialdir=self.library_var.get())
        if path:
            self.library_var.set(path)
            self._refresh_library()

    def _on_library_pick(self, _event=None) -> None:
        path = self.library_files.get(self.library_combo.get())
        if path is not None:
            self.source_var.set(str(path))
            self.source_mode.set("file")
            self._on_build()

    def _on_save_to_library(self) -> None:
        from tkinter import simpledialog

        if self.model is None:
            self._status("Primero construye o carga un modelo.")
            return
        name = simpledialog.askstring("A la biblioteca", "Nombre del modelo:",
                                      initialvalue=logic.job_name(self.model.atoms))
        if not name:
            return
        try:
            path = logic.save_to_library(self.model.atoms, Path(self.library_var.get()), name)
        except Exception as exc:        # noqa: BLE001
            self._error(exc)
            return
        self._refresh_library()
        self.library_combo.set(path.name)
        self._status(f"Guardado en la biblioteca: {path}")

    def _on_build(self) -> None:
        source = None
        if self.source_mode.get() == "file":
            if not self.source_var.get().strip():
                self._status("Elige un archivo o una estructura de la biblioteca.")
                return
            source = Path(self.source_var.get())
        try:
            self.model = logic.build_model(self._read(self.builder_vars), source=source)
        except Exception as exc:        # noqa: BLE001 -- surfaced to the user
            self._error(exc)
            return
        self._set_text(self.model_report, self.model.summary())
        self._draw_structure(self.model_ax, self.model.atoms)
        self.model_canvas.draw_idle()
        self.name_var.set(logic.job_name(self.model.atoms))
        self._status(f"Modelo listo: {self.model.atoms.get_chemical_formula()}.")

    def _draw_structure(self, ax, atoms) -> None:
        from ...viz.plot import draw_structure_on_axes

        ax.clear()
        draw_structure_on_axes(atoms, ax, title=atoms.get_chemical_formula())
        ax.view_init(*logic.view_angles(atoms))
        # A nearly flat axis only collects overlapping tick labels.
        spans = np.ptp(atoms.positions, axis=0)
        for axis, span in zip((ax.xaxis, ax.yaxis, ax.zaxis), spans, strict=True):
            if span < 0.15 * spans.max():
                axis.set_ticks([])
                axis.label.set_visible(False)

    def _on_save_structure(self) -> None:
        from tkinter import filedialog

        if self.model is None:
            self._status("Primero construye un modelo.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xyz",
                                            filetypes=[("extxyz", "*.xyz")])
        if path:
            self.model.atoms.write(path)
            self._status(f"Estructura guardada en {path}.")

    # ------------------------------------------------------------------ calc tab

    def _build_calc_tab(self, tab) -> None:
        tk, ttk = self.tk, self.ttk
        left = ttk.Frame(tab, width=380)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True)

        form = ttk.LabelFrame(left, text="Parámetros (GPAW)", padding=6)
        form.pack(fill="x")
        self.calc_vars: dict[str, Any] = {}
        self._form(form, logic.CALC_PARAMS, self.calc_vars)

        where = ttk.LabelFrame(left, text="Dónde y cómo", padding=6)
        where.pack(fill="x", pady=6)
        self.run_vars: dict[str, Any] = {}
        self._form(where, logic.RUN_PARAMS, self.run_vars)
        ttk.Label(where, text="Directorio").grid(row=1, column=0, sticky="w")
        self.workdir_var = tk.StringVar(value=str(self.workdir))
        ttk.Entry(where, textvariable=self.workdir_var).grid(row=1, column=1, sticky="ew")
        ttk.Button(where, text="…", width=3, command=self._on_pick_workdir).grid(row=1, column=2)
        ttk.Label(where, text="Nombre").grid(row=2, column=0, sticky="w")
        self.name_var = tk.StringVar(value="")
        ttk.Entry(where, textvariable=self.name_var).grid(row=2, column=1, sticky="ew")

        buttons = ttk.Frame(left)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Comprobar", command=self._on_check).pack(side="left")
        ttk.Button(buttons, text="Preparar", command=lambda: self._on_prepare(False)
                   ).pack(side="left", padx=4)
        run = ttk.Button(buttons, text="Preparar y correr", command=lambda: self._on_prepare(True))
        run.pack(side="left")
        if not self.can_run:
            run.state(["disabled"])
        ttk.Label(left, text="" if self.can_run else self.cannot_run_reason,
                  wraplength=360, foreground="#9a4f1c").pack(fill="x", pady=4)
        self.calc_report = self._text(left, 12)
        self.calc_report.pack(fill="both", expand=True)

        jobs = ttk.LabelFrame(right, text="Cola de trabajos", padding=6)
        jobs.pack(fill="both", expand=True)
        columns = ("estado", "progreso")
        self.job_tree = ttk.Treeview(jobs, columns=columns, height=8)
        self.job_tree.heading("#0", text="cálculo")
        self.job_tree.heading("estado", text="estado")
        self.job_tree.heading("progreso", text="progreso")
        self.job_tree.column("#0", width=260)
        self.job_tree.pack(fill="x")
        self.job_tree.bind("<<TreeviewSelect>>", lambda _e: self._refresh_log())
        bar = ttk.Frame(jobs)
        bar.pack(fill="x", pady=4)
        add = ttk.Button(bar, text="Añadir directorio preparado…", command=self._on_add_existing)
        add.pack(side="left")
        if not self.can_run:
            add.state(["disabled"])
        ttk.Button(bar, text="Cancelar", command=self._on_cancel).pack(side="left", padx=4)
        ttk.Button(bar, text="Ver resultados", command=self._on_show_results).pack(side="left")
        ttk.Label(jobs, text="Log").pack(anchor="w")
        self.job_log = self._text(jobs, 20)
        self.job_log.pack(fill="both", expand=True)

    def _on_pick_workdir(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(initialdir=self.workdir_var.get())
        if path:
            self.workdir_var.set(path)

    def _spec(self):
        return logic.spec_from_form(self._read(self.calc_vars))

    def _on_check(self) -> None:
        try:
            spec = self._spec()
            report = spec.validate(self.model.atoms if self.model else None)
        except Exception as exc:        # noqa: BLE001
            self._error(exc)
            return
        text = report.summary()
        if self.model is None:
            text += "\n\n(Sin modelo: solo se comprobaron los parámetros.)"
        self._set_text(self.calc_report, text)

    def _on_prepare(self, and_run: bool) -> None:
        if self.model is None:
            self._status("Primero construye un modelo en la pestaña Modelo.")
            return
        try:
            spec = self._spec()
            directory = logic.prepare_job(self.model.atoms, spec, Path(self.workdir_var.get()),
                                          name=self.name_var.get() or None)
        except Exception as exc:        # noqa: BLE001
            self._error(exc)
            return
        self._set_text(self.calc_report, f"Preparado en {directory}\n\n"
                       f"Para correrlo en otra máquina:\n  cd {directory}\n"
                       "  mpiexec -n 4 gpaw python run.py")
        if and_run:
            self._submit(directory)
        self._status(f"{directory.name} preparado.")

    def _submit(self, directory: Path) -> None:
        try:
            run_values = logic.collect_values(logic.RUN_PARAMS, self._read(self.run_vars))
            nprocs = int(run_values["nprocs"])
            job = self.queue.submit(directory, nprocs=nprocs)
        except Exception as exc:        # noqa: BLE001
            self._error(exc)
            return
        self.job_tree.insert("", "end", iid=str(job.directory), text=job.name,
                             values=(job.state, logic.progress(job.directory)))
        self.queue.poll()

    def _on_add_existing(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(initialdir=self.workdir_var.get())
        if path:
            self._submit(Path(path))

    def _selected_job(self) -> Optional[logic.Job]:
        selection = self.job_tree.selection()
        if not selection:
            return None
        return next((j for j in self.queue.jobs if str(j.directory) == selection[0]), None)

    def _on_cancel(self) -> None:
        job = self._selected_job()
        if job is not None:
            self.queue.cancel(job)
            self._refresh_jobs()

    def _on_show_results(self) -> None:
        job = self._selected_job()
        if job is not None:
            self.results_dir_var.set(str(job.directory))
            self.notebook.select(self.tabs["Resultados"])
            self._on_load_results()

    def _refresh_jobs(self) -> None:
        for job in self.queue.jobs:
            iid = str(job.directory)
            if self.job_tree.exists(iid):
                self.job_tree.item(iid, values=(job.state, logic.progress(job.directory)))

    def _refresh_log(self) -> None:
        job = self._selected_job()
        if job is not None:
            self._set_text(self.job_log, logic.job_log(job))
            self.job_log.see("end")

    def _poll(self) -> None:
        try:
            self.queue.poll()
            self._refresh_jobs()
            self._refresh_log()
        finally:
            self.root.after(_POLL_MS, self._poll)

    def _on_close(self) -> None:
        from tkinter import messagebox

        if self.queue.active() and not messagebox.askyesno(
            "vibspec", "Hay cálculos corriendo. ¿Cancelarlos y salir? "
                       "(se pueden reanudar con run.py)"):
            return
        self.queue.shutdown()
        self.root.destroy()

    # ------------------------------------------------------------------ results tab

    def _build_results_tab(self, tab) -> None:
        from matplotlib.figure import Figure

        tk, ttk = self.tk, self.ttk
        left = ttk.Frame(tab, width=340)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True)

        pick = ttk.LabelFrame(left, text="Cálculo", padding=6)
        pick.pack(fill="x")
        self.results_dir_var = tk.StringVar(value="")
        ttk.Entry(pick, textvariable=self.results_dir_var).pack(fill="x")
        row = ttk.Frame(pick)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Abrir…", command=self._on_pick_results).pack(side="left")
        ttk.Button(row, text="Cargar", command=self._on_load_results).pack(side="left", padx=4)

        exp = ttk.LabelFrame(left, text="FTIR experimental", padding=6)
        exp.pack(fill="x", pady=6)
        self.ftir_var = tk.StringVar(value="")
        ttk.Entry(exp, textvariable=self.ftir_var).pack(fill="x")
        row = ttk.Frame(exp)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Abrir…", command=self._on_pick_ftir).pack(side="left")
        ttk.Button(row, text="Quitar", command=lambda: self.ftir_var.set("")).pack(side="left",
                                                                                  padx=4)
        self.quantity_var = tk.StringVar(value=logic.AUTO)
        ttk.Combobox(exp, textvariable=self.quantity_var, state="readonly",
                     values=[logic.AUTO, "absorbance", "transmittance"]).pack(fill="x")
        self.baseline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(exp, text="Restar línea base", variable=self.baseline_var).pack(anchor="w")

        view = ttk.LabelFrame(left, text="Espectro", padding=6)
        view.pack(fill="x")
        self.view_vars: dict[str, Any] = {}
        fields = (("fwhm", "FWHM (cm-1)", "10"), ("scale", "Factor de escala", ""),
                  ("tolerance", "Tolerancia (cm-1)", "30"),
                  ("min_intensity", "Intensidad mínima", "0.05"),
                  ("xmin", "Desde (cm-1)", "400"), ("xmax", "Hasta (cm-1)", "4000"))
        for key, label, default in fields:
            line = ttk.Frame(view)
            line.pack(fill="x", pady=1)
            ttk.Label(line, text=label, width=18).pack(side="left")
            var = tk.StringVar(value=default)
            ttk.Entry(line, textvariable=var, width=10).pack(side="left", fill="x", expand=True)
            self.view_vars[key] = var
        self.profile_var = tk.StringVar(value="lorentzian")
        ttk.Combobox(view, textvariable=self.profile_var, state="readonly",
                     values=["lorentzian", "gaussian"]).pack(fill="x", pady=2)
        row = ttk.Frame(view)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Dibujar", command=lambda: self._on_plot(False)).pack(side="left")
        ttk.Button(row, text="Ajustar escala", command=lambda: self._on_plot(True)
                   ).pack(side="left", padx=4)
        ttk.Button(row, text="Guardar PNG…", command=self._on_save_plot).pack(side="left")

        ttk.Label(left, text="Bandas (clic en el espectro para ver el modo)").pack(anchor="w")
        self.band_table = self._text(left, 14)
        self.band_table.pack(fill="both", expand=True)

        self.spectrum_figure = Figure(figsize=(7, 3.6), layout="constrained")
        self.spectrum_ax = self.spectrum_figure.add_subplot()
        top = ttk.Frame(right)
        top.pack(fill="both", expand=True)
        self.spectrum_canvas = self._canvas(top, self.spectrum_figure)
        self.spectrum_canvas.mpl_connect("button_press_event", self._on_spectrum_click)

        bottom = ttk.Frame(right)
        bottom.pack(fill="both", expand=True)
        self.mode_var = tk.StringVar(value="Ningún modo seleccionado.")
        ttk.Label(bottom, textvariable=self.mode_var, anchor="w").pack(fill="x")
        self.mode_figure = Figure(figsize=(7, 3.4), layout="constrained")
        self.mode_ax = self.mode_figure.add_subplot(projection="3d")
        self.mode_canvas = self._canvas(bottom, self.mode_figure)

    def _on_pick_results(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(initialdir=self.workdir_var.get())
        if path:
            self.results_dir_var.set(path)
            self._on_load_results()

    def _on_pick_ftir(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("Espectros", "*.csv *.txt *.dat"),
                                                     ("Todos", "*.*")])
        if path:
            self.ftir_var.set(path)

    def _on_load_results(self) -> None:
        from ..core import CalcRecord

        try:
            directory = Path(self.results_dir_var.get())
            record = CalcRecord.load(directory)
        except Exception as exc:        # noqa: BLE001
            self._error(exc)
            return
        if record.status != "done":
            self._status(f"{directory.name}: {logic.progress(directory)}. Aún no hay espectro.")
            return
        self.results_dir, self.results_record = directory, record
        self.view_vars["scale"].set(f"{record.spec.get('scale_factor', 1.0):g}")
        self._on_plot(False)

    def _float(self, key: str) -> float:
        raw = self.view_vars[key].get().strip().replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"'{raw}' no es un número válido.") from None

    def _on_plot(self, fit_scale: bool) -> None:
        from ..core import (
            collect,
            draw_ir_comparison,
            find_bands,
            match_bands,
            match_table,
            prepare_experiment,
            read_ftir,
            search_scale_factor,
        )
        from ..core.checks import SCALE_FACTOR_RANGE

        if self.results_dir is None:
            self._status("Carga primero un cálculo terminado.")
            return
        try:
            spectrum = collect(self.results_dir)
            scale, window = self._float("scale"), (self._float("xmin"), self._float("xmax"))
            tolerance, fwhm = self._float("tolerance"), self._float("fwhm")
            min_intensity = self._float("min_intensity")
            experiment, matches, notes = None, None, []
            if self.ftir_var.get().strip():
                quantity = self.quantity_var.get()
                measured = read_ftir(self.ftir_var.get(),
                                     quantity=None if quantity == logic.AUTO else quantity)
                if measured.quantity_source == "values":
                    notes.append(f"AVISO: se interpretó como {measured.quantity} por los "
                                 "valores; elige el tipo arriba si no es así.")
                experiment = prepare_experiment(measured, baseline=self.baseline_var.get(),
                                                window=window)
                bands = find_bands(*experiment)
                matches = None
                if fit_scale:
                    # A refusal to fit is information, not a failure: say why
                    # and keep drawing with the current factor.
                    try:
                        scale, matches = search_scale_factor(
                            spectrum, bands, tolerance_cm1=tolerance,
                            min_relative_intensity=min_intensity, bounds=SCALE_FACTOR_RANGE)
                    except ValueError as exc:
                        notes.append(f"AVISO: {exc} Se mantiene el factor {scale:g}.")
                    else:
                        self.view_vars["scale"].set(f"{scale:.4f}")
                        notes.append(f"Factor de escala ajustado: {scale:.4f}")
                if matches is None:
                    matches = match_bands(spectrum, bands, scale, tolerance_cm1=tolerance,
                                          min_relative_intensity=min_intensity)
            elif fit_scale:
                notes.append("Para ajustar la escala hace falta un FTIR.")
            self.spectrum_ax.clear()
            draw_ir_comparison(self.spectrum_ax, spectrum, experiment=experiment, fwhm_cm1=fwhm,
                               profile=self.profile_var.get(), scale_factor=scale, window=window,
                               matches=matches,
                               experiment_label=Path(self.ftir_var.get()).stem or "FTIR")
            self.spectrum_ax.set_title(self.results_dir.name, fontsize=9, loc="left")
            self.spectrum_canvas.draw_idle()
            table = match_table(matches) if matches is not None else self._mode_list(scale)
            self._set_text(self.band_table, "\n".join(notes + [table]))
            self.experiment = experiment
        except Exception as exc:        # noqa: BLE001
            self._error(exc)

    def _mode_list(self, scale: float) -> str:
        results = self.results_record.results
        rows = [f"{'modo':>5} {'cm-1 (escalado)':>16} {'I rel':>6}"]
        intensities = np.asarray(results["ir_intensity"])
        top = intensities.max() if intensities.size and intensities.max() > 0 else 1.0
        for index, frequency, intensity in zip(results["mode_indices"], results["frequencies_cm1"],
                                               intensities, strict=True):
            if intensity / top >= 0.02:
                rows.append(f"{index:5d} {frequency * scale:16.1f} {intensity / top:6.2f}")
        return "\n".join(rows)

    def _on_spectrum_click(self, event) -> None:
        if event.inaxes is not self.spectrum_ax or self.results_record is None:
            return
        try:
            scale = self._float("scale")
        except ValueError:
            scale = 1.0
        index = logic.mode_at(self.results_record, event.xdata, scale)
        if index is not None:
            self._animate_mode(index, scale)

    def _animate_mode(self, index: int, scale: float) -> None:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        from ...topology.graph import build_bond_graph
        from ...viz.plot import _ELEMENT_COLORS, _ELEMENT_SIZES

        try:
            atoms, modes, _ = logic.load_modes(self.results_dir)
        except Exception as exc:        # noqa: BLE001
            self._error(exc)
            return
        vector = logic.mode_vector(modes, index)
        frames = logic.mode_frames(atoms, vector)
        frequency = float(modes["frequencies_cm1"][index]) * scale
        self.mode_var.set(f"Modo {index}: {frequency:.1f} cm-1 (escalado) — "
                          f"{logic.mode_character(atoms, vector)}")

        ax = self.mode_ax
        ax.clear()
        symbols = atoms.get_chemical_symbols()
        positions = frames[0]
        scatter = ax.scatter(*positions.T, c=[_ELEMENT_COLORS.get(s, "#888888") for s in symbols],
                             s=[_ELEMENT_SIZES.get(s, 30) for s in symbols],
                             edgecolors="black", linewidths=0.3)
        bonds = [(i, j) for i, j in build_bond_graph(atoms).edges]
        lines = Line3DCollection([(positions[i], positions[j]) for i, j in bonds],
                                 colors="#555555", linewidths=0.6)
        ax.add_collection3d(lines)
        # A cube around the molecule, so no axis is squashed whatever the view.
        centre = atoms.get_positions().mean(axis=0)
        half = float(np.ptp(atoms.get_positions(), axis=0).max()) / 2 + 1.0
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*logic.view_angles(atoms))
        ax.set_axis_off()
        if self.animation is not None:
            self.root.after_cancel(self.animation["after"])
        self.animation = {"frames": frames, "k": 0, "scatter": scatter, "lines": lines,
                          "bonds": bonds}
        self._next_frame()

    def _next_frame(self) -> None:
        state = self.animation
        if state is None:
            return
        positions = state["frames"][state["k"] % len(state["frames"])]
        state["scatter"]._offsets3d = tuple(positions.T)
        state["lines"].set_segments([(positions[i], positions[j]) for i, j in state["bonds"]])
        self.mode_canvas.draw_idle()
        state["k"] += 1
        state["after"] = self.root.after(_FRAME_MS, self._next_frame)

    def _on_save_plot(self) -> None:
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                                                       ("SVG", "*.svg")])
        if path:
            self.spectrum_figure.savefig(path, dpi=200)
            self._status(f"Figura guardada en {path}.")


def main(workdir: Optional[str] = None) -> int:
    """Open the window. Returns a process exit code."""
    try:
        import tkinter as tk
    except ImportError:
        print(_TK_MISSING)
        return 1
    root = tk.Tk()
    VibspecApp(root, Path(workdir) if workdir else None)
    root.mainloop()
    return 0
