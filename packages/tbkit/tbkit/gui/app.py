"""tbkit's window: structure and model on the left, 3D view in the middle, tasks on the right."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import actions
from .viewer import StructureView
from .worker import Runner


class PlotPanel(QWidget):
    """A matplotlib figure with its toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def axes(self):
        self.figure.clear()
        return self.figure.add_subplot(111)

    def draw(self):
        self.canvas.draw_idle()


def table(headers) -> QTableWidget:
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectRows)
    widget.horizontalHeader().setStretchLastSection(True)
    widget.verticalHeader().setVisible(False)
    return widget


def fill(widget: QTableWidget, rows):
    widget.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            widget.setItem(r, c, QTableWidgetItem(text))


class MainWindow(QMainWindow):
    def __init__(self, interactive: bool = True):
        super().__init__()
        self.setWindowTitle("tbkit")
        self.resize(1500, 900)
        self.atoms = None
        self.model = None
        self.model_name = None
        self.state: Optional[actions.GroundState] = None
        self.runner = Runner(self)
        self.runner.message.connect(self.statusBar().showMessage)
        self.runner.busy.connect(self._busy)
        self.timer = QTimer(self)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._left_panel())
        if interactive:
            from pyvistaqt import QtInteractor

            self.plotter = QtInteractor(splitter)
            splitter.addWidget(self.plotter.interactor)
        else:                                          # tests, no OpenGL window
            import pyvista as pv

            self.plotter = pv.Plotter(off_screen=True)
            splitter.addWidget(QLabel("(vista 3D fuera de pantalla)"))
        self.view = StructureView(self.plotter)
        self.tabs = QTabWidget()
        self.pages = {}
        for page in self.page_classes():
            widget = page(self)
            self.pages[widget.title] = widget
            self.tabs.addTab(widget, widget.title)
        splitter.addWidget(self.tabs)
        splitter.setSizes([300, 700, 600])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Abre una estructura (xyz, extxyz, cif, POSCAR…).")

    def page_classes(self):
        return [ElectronicPage, OrbitalPage, MagnetismPage]

    # --- left panel ---------------------------------------------------------------

    def _left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        structure = QGroupBox("Estructura")
        form = QFormLayout(structure)
        open_button = QPushButton("Abrir…")
        open_button.clicked.connect(self.open_dialog)
        self.structure_label = QLabel("—")
        self.structure_label.setWordWrap(True)
        form.addRow(open_button)
        form.addRow(self.structure_label)
        layout.addWidget(structure)

        model = QGroupBox("Modelo")
        form = QFormLayout(model)
        self.model_combo = QComboBox()
        for name, (label, _) in actions.MODELS.items():
            self.model_combo.addItem(label, name)
        self.model_combo.addItem("Archivo de parámetros…", "file")
        self.model_combo.activated.connect(self._model_chosen)
        self.charge = QDoubleSpinBox()
        self.charge.setRange(-10, 10)
        self.charge.setSingleStep(1)
        self.charge.valueChanged.connect(self.invalidate)
        self.scc = QComboBox()
        self.scc.addItems(["según el modelo", "sí", "no"])
        self.scc.currentIndexChanged.connect(self.invalidate)
        self.kT = QDoubleSpinBox()
        self.kT.setRange(0.001, 1.0)
        self.kT.setDecimals(3)
        self.kT.setValue(0.01)
        self.kT.setSuffix(" eV")
        self.kT.valueChanged.connect(self.invalidate)
        self.model_label = QLabel("—")
        self.model_label.setWordWrap(True)
        form.addRow("Parámetros", self.model_combo)
        form.addRow("Carga (e)", self.charge)
        form.addRow("SCC", self.scc)
        form.addRow("kT", self.kT)
        form.addRow(self.model_label)
        layout.addWidget(model)

        colour = QGroupBox("Colorear átomos")
        form = QFormLayout(colour)
        self.colour = QComboBox()
        self.colour.addItems(["por elemento", "por carga (Mulliken)"])
        self.colour.currentIndexChanged.connect(self.redraw)
        form.addRow(self.colour)
        layout.addWidget(colour)

        self.cancel_button = QPushButton("Cancelar cálculo")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.runner.cancel)
        layout.addWidget(self.cancel_button)
        layout.addStretch(1)
        return panel

    def _busy(self, busy: bool):
        self.cancel_button.setEnabled(busy)

    # --- structure and model ------------------------------------------------------------

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir estructura", "",
                                              "Estructuras (*.xyz *.extxyz *.cif *.vasp "
                                              "POSCAR* *.pdb *.json *.traj);;Todos (*)")
        if path:
            self.open_structure(path)

    def open_structure(self, path):
        try:
            atoms = actions.read_structure(path)
        except Exception as error:
            self.error(f"No se pudo leer {path}", str(error))
            return
        self.set_atoms(atoms, Path(path).name)

    def set_atoms(self, atoms, name: str = ""):
        self.atoms = atoms
        info = actions.structure_summary(atoms)
        self.structure_label.setText(f"{name}\n{info['formula']} · {info['atoms']} átomos · "
                                     f"periódica: {info['periodic']}")
        suggested = actions.suggest_model(atoms)
        if self.model is None or actions.check_model(atoms, self.model):
            self.set_model(suggested)
        self.invalidate()
        self.view.show(atoms)

    def _model_chosen(self, index):
        name = self.model_combo.itemData(index)
        if name == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Parámetros", "", "JSON (*.json)")
            if path:
                self.set_model(path)
        else:
            self.set_model(name)

    def set_model(self, name_or_path: str):
        try:
            self.model = actions.load_model(name_or_path)
        except Exception as error:
            self.error("No se pudo cargar el modelo", str(error))
            return
        self.model_name = name_or_path
        index = self.model_combo.findData(name_or_path)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        text = f"{self.model.name}\nSCC: {'sí' if self.model.scc else 'no'} · " \
               f"elementos: {', '.join(sorted(self.model.orbitals))}"
        if self.atoms is not None:
            problems = actions.check_model(self.atoms, self.model)
            if problems:
                text += "\n⚠ " + " ".join(problems)
        self.model_label.setText(text)
        self.invalidate()

    def scc_choice(self) -> Optional[bool]:
        return {0: None, 1: True, 2: False}[self.scc.currentIndex()]

    def invalidate(self, *_):
        self.state = None
        for page in self.pages.values():
            page.invalidated()

    def redraw(self, *_):
        if self.atoms is None:
            return
        if self.colour.currentIndex() == 1 and self.state is not None:
            charges = np.array([self.state.charges.get(i, 0.0) for i in range(len(self.atoms))])
            self.view.show(self.atoms, charges, "carga (e)", keep_camera=True)
        else:
            self.view.show(self.atoms, keep_camera=True)

    # --- shared ground state --------------------------------------------------------------

    def require(self) -> bool:
        if self.atoms is None:
            self.error("Sin estructura", "Abre primero una estructura.")
            return False
        if self.model is None:
            self.error("Sin modelo", "Elige un modelo.")
            return False
        problems = actions.check_model(self.atoms, self.model, self.scc_choice())
        if problems:
            self.error("El modelo no sirve para esta estructura", " ".join(problems))
            return False
        return True

    def with_ground_state(self, then):
        """Call ``then(state)``, computing the ground state first if needed."""
        if self.state is not None:
            then(self.state)
            return
        if not self.require():
            return

        def store(state):
            self.state = state
            self.redraw()
            for page in self.pages.values():
                page.ground_state_ready(state)
            then(state)

        self.runner.start("Estado fundamental", actions.ground_state, self.atoms, self.model,
                          charge=self.charge.value(), kT=self.kT.value(), scc=self.scc_choice(),
                          on_done=store, on_error=self.error)

    def error(self, title, detail=""):
        self.statusBar().showMessage(title)
        box = QMessageBox(QMessageBox.Warning, "tbkit", title, parent=self)
        if detail:
            box.setInformativeText(detail.splitlines()[0][:400] if "\n" in detail else detail)
            box.setDetailedText(detail)
        if QApplication.instance().platformName() != "offscreen":
            box.exec()


class Page(QWidget):
    title = ""

    def __init__(self, window: MainWindow):
        super().__init__()
        self.window = window

    def invalidated(self):
        pass

    def ground_state_ready(self, state):
        pass


class ElectronicPage(Page):
    title = "Electrónica"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.compute = QPushButton("Calcular estado fundamental")
        self.compute.clicked.connect(lambda: self.window.with_ground_state(self.show_state))
        row.addWidget(self.compute)
        row.addWidget(QLabel("PDOS por"))
        self.projection = QComboBox()
        self.projection.addItems(["elemento", "átomo", "orbital", "ninguna"])
        self.projection.currentIndexChanged.connect(self.plot_dos)
        row.addWidget(self.projection)
        row.addWidget(QLabel("σ"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.01, 1.0)
        self.sigma.setValue(0.1)
        self.sigma.setSuffix(" eV")
        self.sigma.valueChanged.connect(self.plot_dos)
        row.addWidget(self.sigma)
        layout.addLayout(row)
        self.summary = QLabel("—")
        layout.addWidget(self.summary)
        tabs = QTabWidget()
        self.dos_plot = PlotPanel()
        tabs.addTab(self.dos_plot, "DOS / PDOS")
        self.levels = table(["#", "E (eV)", "ocupación", ""])
        tabs.addTab(self.levels, "Niveles")
        bands = QWidget()
        bl = QVBoxLayout(bands)
        br = QHBoxLayout()
        br.addWidget(QLabel("Camino"))
        self.path = QLineEdit()
        self.path.setPlaceholderText("automático (p. ej. GMKG)")
        br.addWidget(self.path)
        self.bands_button = QPushButton("Calcular bandas")
        self.bands_button.clicked.connect(self.compute_bands)
        br.addWidget(self.bands_button)
        bl.addLayout(br)
        self.bands_plot = PlotPanel()
        bl.addWidget(self.bands_plot)
        tabs.addTab(bands, "Bandas")
        self.charges = table(["átomo", "elemento", "carga (e)"])
        tabs.addTab(self.charges, "Cargas")
        layout.addWidget(tabs)

    def invalidated(self):
        self.summary.setText("—")

    def ground_state_ready(self, state):
        self.show_state(state)

    def show_state(self, state):
        info = state.info
        gap = f"{info['gap']:.3f} eV" if info["gap"] and info["gap"] > 0 else "sin gap"
        scc = f" · SCC en {state.iterations} iteraciones" if state.scc else ""
        self.summary.setText(f"gap {gap} · HOMO {info['homo']:.3f} · LUMO {info['lumo']:.3f} · "
                             f"E_F {info['fermi']:.3f} eV · {info['electrons']:.0f} electrones"
                             f"{scc}")
        fill(self.levels, actions.levels_table(state, around=15))
        symbols = self.window.atoms.get_chemical_symbols()
        fill(self.charges, [(i, symbols[i], float(q)) for i, q in sorted(state.charges.items())])
        self.plot_dos()

    def plot_dos(self, *_):
        state = self.window.state
        if state is None:
            return
        by = {0: "element", 1: "atom", 2: "orbital", 3: None}[self.projection.currentIndex()]
        if by == "atom" and len(self.window.atoms) > 30:
            by = "element"
        curves = actions.dos_curves(state, self.sigma.value(), by)
        ax = self.dos_plot.axes()
        ax.plot(curves["energy"], curves["total"], color="black", lw=1.2, label="total")
        for label, values in curves["projected"].items():
            ax.plot(curves["energy"], values, lw=0.9, label=label)
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("E − E_F (eV)")
        ax.set_ylabel("estados/eV")
        ax.legend(fontsize=8)
        self.dos_plot.draw()

    def compute_bands(self):
        if not self.window.require():
            return
        self.window.runner.start("Bandas", actions.band_curves, self.window.atoms,
                                 self.window.model, self.path.text().strip() or None,
                                 kT=self.window.kT.value(), on_done=self.plot_bands,
                                 on_error=self.window.error)

    def plot_bands(self, curves):
        ax = self.bands_plot.axes()
        ax.plot(curves["x"], curves["energies"], color="black", lw=0.8)
        for tick in curves["ticks"]:
            ax.axvline(tick, color="grey", lw=0.5)
        ax.set_xticks(curves["ticks"], [lab.replace("G", "Γ") for lab in curves["labels"]])
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_ylabel("E − E_F (eV)")
        ax.set_xlim(curves["x"][0], curves["x"][-1])
        self.bands_plot.draw()


class OrbitalPage(Page):
    title = "Orbitales"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        self.hint = QLabel("Calcula el estado fundamental y elige un nivel.")
        layout.addWidget(self.hint)
        self.list = table(["#", "E (eV)", "ocupación", ""])
        self.list.itemSelectionChanged.connect(self.selected)
        layout.addWidget(self.list)
        row = QHBoxLayout()
        self.load = QPushButton("Cargar niveles")
        self.load.clicked.connect(lambda: self.window.with_ground_state(self.ground_state_ready))
        row.addWidget(self.load)
        row.addWidget(QLabel("isovalor"))
        self.level = QSlider(Qt.Horizontal)
        self.level.setRange(2, 90)
        self.level.setValue(25)
        self.level.valueChanged.connect(self.redraw)
        row.addWidget(self.level)
        clear = QPushButton("Quitar")
        clear.clicked.connect(self.window.view.clear_isosurface)
        row.addWidget(clear)
        layout.addLayout(row)
        self.grid = None
        self.rows = []

    def invalidated(self):
        self.list.setRowCount(0)
        self.grid = None

    def ground_state_ready(self, state):
        self.rows = actions.levels_table(state, around=12)
        fill(self.list, self.rows)

    def selected(self):
        rows = self.list.selectionModel().selectedRows()
        if not rows or self.window.state is None:
            return
        band = self.rows[rows[0].row()][0]
        self.window.runner.start(f"Orbital {band}", actions.orbital_grid, self.window.state,
                                 band, on_done=self.show_grid, on_error=self.window.error)

    def show_grid(self, grid):
        self.grid = grid
        self.redraw()

    def redraw(self, *_):
        if self.grid is None:
            return
        values = self.grid["values"]
        level = self.level.value() / 100 * float(np.abs(values).max())
        self.window.view.isosurface(self.grid["origin"], self.grid["spacing"], values, level)
        self.hint.setText(f"E = {self.grid['energy']:.3f} eV · isovalor {level:.3g}")


def _spin(minimum, maximum, value, step=1.0, decimals=2, suffix=""):
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return box


class MagnetismPage(Page):
    """Mean-field Hubbard: moments on the structure, m(E), spin DOS, sweeps."""

    title = "Magnetismo"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.use_model_u = QComboBox()
        self.use_model_u.addItems(["U del modelo", "U fija"])
        self.U = _spin(0.0, 20.0, 3.0, 0.1, 2, " eV")
        row = QHBoxLayout()
        row.addWidget(self.use_model_u)
        row.addWidget(self.U)
        form.addRow("Hubbard U", row)
        self.guess = QComboBox()
        for key, label in actions.GUESSES.items():
            self.guess.addItem(label, key)
        form.addRow("Punto de partida", self.guess)
        self.kT = _spin(0.0005, 0.5, 0.005, 0.001, 4, " eV")
        form.addRow("kT", self.kT)
        self.field = _spin(-2000, 2000, 0.0, 10, 1, " T")
        form.addRow("Campo (Zeeman)", self.field)
        self.kmesh = QDoubleSpinBox()
        self.kmesh.setRange(1, 400)
        self.kmesh.setDecimals(0)
        self.kmesh.setValue(24)
        form.addRow("Malla k (periódicos)", self.kmesh)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.solve = QPushButton("Resolver")
        self.solve.clicked.connect(self.run)
        buttons.addWidget(self.solve)
        self.compare = QPushButton("Comparar puntos de partida")
        self.compare.clicked.connect(self.run_compare)
        buttons.addWidget(self.compare)
        layout.addLayout(buttons)
        self.summary = QLabel("Los momentos son el parámetro de orden del campo medio, no un "
                              "estado correlacionado.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        tabs = QTabWidget()
        self.m_plot = PlotPanel()
        tabs.addTab(self.m_plot, "m(E) y DOS de espín")
        self.moments = table(["átomo", "elemento", "m (μB)"])
        tabs.addTab(self.moments, "Momentos")
        self.guesses = table(["partida", "E − E_min (eV)", "M (μB)", "|m| máx", "nota"])
        tabs.addTab(self.guesses, "Comparación")
        sweep = QWidget()
        sl = QVBoxLayout(sweep)
        sr = QHBoxLayout()
        self.sweep_kind = QComboBox()
        self.sweep_kind.addItems(["campo (T)", "dopaje (e)"])
        self.sweep_from = _spin(-5000, 5000, 0.0, 1, 2)
        self.sweep_to = _spin(-5000, 5000, 200.0, 1, 2)
        self.sweep_n = _spin(2, 200, 11, 1, 0)
        for widget, text in ((self.sweep_kind, None), (self.sweep_from, "de"),
                             (self.sweep_to, "a"), (self.sweep_n, "puntos")):
            if text:
                sr.addWidget(QLabel(text))
            sr.addWidget(widget)
        self.sweep_button = QPushButton("Barrer")
        self.sweep_button.clicked.connect(self.run_sweep)
        sr.addWidget(self.sweep_button)
        sl.addLayout(sr)
        self.sweep_plot = PlotPanel()
        sl.addWidget(self.sweep_plot)
        tabs.addTab(sweep, "Barridos")
        layout.addWidget(tabs)
        self.result = None

    def settings(self) -> dict:
        return {"U": None if self.use_model_u.currentIndex() == 0 else self.U.value(),
                "charge": self.window.charge.value(), "kT": self.kT.value(),
                "guess": self.guess.currentData(), "kmesh": int(self.kmesh.value())}

    def _ready(self) -> bool:
        if self.window.atoms is None or self.window.model is None:
            self.window.error("Sin estructura o modelo", "Abre una estructura y elige un modelo.")
            return False
        if self.use_model_u.currentIndex() == 0 and not self.window.model.hubbard_u:
            self.window.error("El modelo no tiene U", "Elige «U fija».")
            return False
        return True

    def run(self):
        if self._ready():
            self.window.runner.start("Hubbard", actions.hubbard_solution, self.window.atoms,
                                     self.window.model, field_tesla=self.field.value(),
                                     **self.settings(), on_done=self.show,
                                     on_error=self.window.error)

    def show(self, out):
        self.result = out
        state = "convergido" if out["converged"] else "SIN CONVERGER"
        gap = f"{out['gap']:.3f} eV" if out["gap"] and out["gap"] > 0 else "sin gap"
        self.summary.setText(
            f"{state} en {out['iterations']} iteraciones · M = {out['magnetization']:+.4f} μB · "
            f"|m| máx = {np.abs(out['moments']).max():.4f} μB · E = {out['energy']:.5f} eV · "
            f"gap {gap} · U = {out['U']:.2f} eV. Momentos: parámetro de orden del campo medio.")
        symbols = self.window.atoms.get_chemical_symbols()
        fill(self.moments, [(i, symbols[i], float(m)) for i, m in enumerate(out["moments"])])
        self.window.view.show(self.window.atoms, out["moments"], "m (μB)", cmap="PuOr",
                              keep_camera=True)
        fig = self.m_plot.figure
        fig.clear()
        top, bottom = fig.subplots(2, 1, sharex=True)
        top.plot(out["grid"], out["dos_up"], color="tab:red", lw=1, label="↑")
        top.plot(out["grid"], -out["dos_down"], color="tab:blue", lw=1, label="↓")
        top.axhline(0, color="black", lw=0.5)
        top.set_ylabel("DOS (estados/eV)")
        top.legend(fontsize=8)
        bottom.plot(out["grid"], out["m_of_E"], color="black", lw=1.2, label="m(E)")
        bottom.plot(out["grid"], out["dm_dE"], color="tab:green", lw=0.8, label="dm/dE")
        bottom.set_xlabel("E − E_F (eV)")
        bottom.set_ylabel("μB")
        # ρ↑ = ρ↓ exactly (an antiferromagnetic state): do not blow rounding up
        # into structure on the axis.
        limit = max(0.05, 1.1 * float(np.abs(np.r_[out["m_of_E"], out["dm_dE"]]).max()))
        bottom.set_ylim(-limit, limit)
        bottom.legend(fontsize=8)
        for ax in (top, bottom):
            ax.axvline(0, color="grey", ls="--", lw=0.8)
        self.m_plot.draw()

    def run_compare(self):
        if self._ready():
            settings = self.settings()
            settings.pop("guess")
            self.window.runner.start("Comparar puntos de partida", actions.compare_guesses,
                                     self.window.atoms, self.window.model,
                                     field_tesla=self.field.value(), **settings,
                                     on_done=self.show_compare, on_error=self.window.error)

    def show_compare(self, rows):
        fill(self.guesses, [(actions.GUESSES[r["guess"]], r["delta"], r["M"], r["max_moment"],
                             r["note"] or ("" if r["converged"] else "sin converger"))
                            for r in rows])

    def run_sweep(self):
        if not self._ready():
            return
        values = np.linspace(self.sweep_from.value(), self.sweep_to.value(),
                             int(self.sweep_n.value()))
        settings = self.settings()
        if self.sweep_kind.currentIndex() == 0:
            self.window.runner.start("Barrido en campo", actions.field_sweep, self.window.atoms,
                                     self.window.model, values, **settings,
                                     on_done=self.show_sweep, on_error=self.window.error)
        else:
            settings.pop("charge")
            self.window.runner.start("Barrido en dopaje", actions.doping_sweep,
                                     self.window.atoms, self.window.model, values, **settings,
                                     on_done=self.show_sweep, on_error=self.window.error)

    def show_sweep(self, out):
        ax = self.sweep_plot.axes()
        if "tesla" in out:
            ax.plot(out["tesla"], out["M"], "o-", ms=3)
            ax.set_xlabel("B (T)")
            twin = ax.twinx()
            twin.plot(out["tesla"], out["chi_per_tesla"], color="tab:orange", lw=0.8)
            twin.set_ylabel("χ = dM/dB (μB/T)", color="tab:orange")
        else:
            ax.plot(out["charge"], out["M"], "o-", ms=3)
            ax.set_xlabel("carga añadida (e; + quita electrones)")
        ax.set_ylabel("M (μB)")
        self.sweep_plot.draw()


def main(argv=None):
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    args = sys.argv[1:] if argv is None else argv[1:]
    if args:
        window.open_structure(args[0])
    window.show()
    return app.exec()
