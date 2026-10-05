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
    QCheckBox,
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
        #: Γ modes of the current structure, shared by the pages: (frequencies, L, source).
        self.phonons = None
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
        self.info_boxes = []
        for page in self.page_classes():
            widget = page(self)
            self.pages[widget.title] = widget
            self.tabs.addTab(self._with_info(widget), widget.title)
        splitter.addWidget(self.tabs)
        splitter.setSizes([430, 620, 600])
        self.setCentralWidget(splitter)
        self.last_record = None
        self._menu()
        self._apply_help()
        self.statusBar().showMessage("Abre una estructura (xyz, extxyz, cif, POSCAR…).")

    def _menu(self):
        menu = self.menuBar().addMenu("Archivo")
        for text, slot in (("Abrir estructura…", self.open_dialog),
                           ("Guardar estructura…", self.save_structure),
                           (None, None),
                           ("Guardar registro del último cálculo…", self.save_record),
                           ("Abrir registro…", self.load_record),
                           (None, None),
                           ("Salir", self.close)):
            if text is None:
                menu.addSeparator()
            else:
                menu.addAction(text).triggered.connect(slot)
        help_menu = self.menuBar().addMenu("Ayuda")
        descriptions = help_menu.addAction("Mostrar descripciones de cada pestaña")
        descriptions.setCheckable(True)
        descriptions.setChecked(True)
        descriptions.toggled.connect(self.show_descriptions)
        help_menu.addAction("Pasa el ratón sobre un botón o campo para ver qué hace")\
            .setEnabled(False)

    def _with_info(self, page):
        """The page under a short box that says what it is for (Ayuda > Descripciones)."""
        text = actions.PANEL_HELP.get(page.title)
        if not text:
            return page
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        info = QLabel("ℹ " + text)
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { background: palette(alternate-base); border: 1px solid "
                           "palette(mid); border-radius: 4px; padding: 6px; }")
        self.info_boxes.append(info)
        layout.addWidget(info)
        layout.addWidget(page)
        return box

    def _apply_help(self):
        """Tooltips from actions.HELP on every button, check box, group and form row."""
        def tip(widget, text):
            if widget is not None and text and not widget.toolTip():
                widget.setToolTip(text)

        for kind in (QPushButton, QCheckBox):
            for widget in self.findChildren(kind):
                tip(widget, actions.HELP.get(widget.text()))
        for group in self.findChildren(QGroupBox):
            tip(group, actions.HELP.get(group.title()))
        for form in self.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                label = form.itemAt(row, QFormLayout.LabelRole)
                field = form.itemAt(row, QFormLayout.FieldRole)
                if label is None or label.widget() is None or field is None:
                    continue
                text = actions.HELP.get(label.widget().text())
                tip(label.widget(), text)
                if field.widget() is not None:
                    tip(field.widget(), text)
                elif field.layout() is not None:
                    for k in range(field.layout().count()):
                        tip(field.layout().itemAt(k).widget(), text)

    def show_descriptions(self, visible: bool):
        for info in self.info_boxes:
            info.setVisible(visible)

    def remember(self, task: str, settings: dict, results: dict):
        """The last finished calculation, for «Guardar registro»."""
        if self.atoms is not None and self.model is not None:
            self.last_record = (self.atoms.copy(), self.model, task, dict(settings), results)

    def save_structure(self):
        if self.atoms is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar estructura", "estructura.extxyz",
                                              "extxyz (*.extxyz);;xyz (*.xyz);;Todos (*)")
        if path:
            from ase.io import write

            write(path, self.atoms)
            self.statusBar().showMessage(f"Guardada {path}")

    def save_record(self, path: Optional[str] = None):
        if self.last_record is None:
            self.error("Nada que registrar", "Haz antes un cálculo.")
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Guardar registro", "registro.json",
                                                  "JSON (*.json)")
        if not path:
            return None
        import json

        atoms, model, task, settings, results = self.last_record
        data = actions.record(atoms, model, task, settings, results)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        self.statusBar().showMessage(f"Registro guardado en {path}")
        return path

    def load_record(self, path: Optional[str] = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Abrir registro", "", "JSON (*.json)")
        if not path:
            return
        try:
            atoms, model, data = actions.open_record(path)
        except Exception as error:
            self.error("No es un registro de tbkit", str(error))
            return
        self.model, self.model_name = model, str(path)
        self.model_label.setText(f"{model.name}\n(del registro: {data.get('task')}, "
                                 f"{data.get('date')}, tbkit {data.get('tbkit')}, "
                                 f"commit {str(data.get('commit'))[:8]})")
        self.set_atoms(atoms, Path(path).name)

    def page_classes(self):
        return [ElectronicPage, OrbitalPage, MagnetismPage, GeometryPage, PhononPage,
                SpectraPage, GraphenePage]

    # --- left panel ---------------------------------------------------------------

    def _left_panel(self) -> QWidget:
        """Controls on top, the terminal below, with a divider between them."""
        column = QSplitter(Qt.Vertical)
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

        terminal = QGroupBox("Terminal")
        box = QVBoxLayout(terminal)
        box.setContentsMargins(4, 4, 4, 4)
        from .console import ConsolePanel

        self.console = ConsolePanel()
        box.addWidget(self.console)
        self.runner.log_line.connect(self.console.line)
        self.runner.output.connect(self.console.write)
        layout.addStretch(1)
        column.addWidget(panel)
        column.addWidget(terminal)
        column.setStretchFactor(0, 0)
        column.setStretchFactor(1, 1)
        return column

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
        self.console.line(f"estructura: {name or '—'} · {info['formula']} · {info['atoms']} "
                          f"átomos · periódica: {info['periodic']}")
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
        self.console.line(f"modelo: {self.model.name} (SCC {'sí' if self.model.scc else 'no'})")
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
        self.phonons = None
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
        if hasattr(self, "console") and not detail.startswith("Traceback") and \
                "Traceback" not in detail:
            self.console.line(f"✗ {title}" + (f": {detail.splitlines()[0]}" if detail else ""))
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
        self.scc_button = QPushButton("Comparar SCC sí/no")
        self.scc_button.clicked.connect(self.compare_scc)
        row.addWidget(self.scc_button)
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
        scc = QWidget()
        sl = QVBoxLayout(scc)
        self.scc_summary = table(["magnitud", "sin SCC", "con SCC", "diferencia"])
        sl.addWidget(self.scc_summary)
        self.scc_charges = table(["átomo", "elemento", "q sin SCC", "q con SCC", "Δq"])
        sl.addWidget(self.scc_charges)
        tabs.addTab(scc, "SCC sí/no")
        self.tabs = tabs
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
        self.window.remember("estado fundamental",
                             {"charge": self.window.charge.value(), "kT": self.window.kT.value(),
                              "scc": state.scc},
                             {**state.info, "charges": state.charges,
                              "levels": state.solution.energies[0, 0],
                              "scc_iterations": state.iterations})
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

    def compare_scc(self):
        if not self.window.require():
            return
        self.window.runner.start("SCC sí/no", actions.compare_scc, self.window.atoms,
                                 self.window.model, charge=self.window.charge.value(),
                                 kT=self.window.kT.value(), on_done=self.show_scc,
                                 on_error=self.window.error)

    def show_scc(self, out):
        self.window.remember("SCC sí/no", {"charge": self.window.charge.value(),
                                           "kT": self.window.kT.value()},
                             {"rows": out["rows"], "charges": out["charges"],
                              "iterations": out["iterations"]})
        fill(self.scc_summary, out["rows"])
        fill(self.scc_charges, out["charges"])
        self.tabs.setCurrentIndex(self.tabs.count() - 1)
        self.window.view.show(self.window.atoms, [c[4] for c in out["charges"]],
                              "Δq (SCC − sin SCC)", cmap="coolwarm", keep_camera=True)

    def compute_bands(self):
        if not self.window.require():
            return
        self.window.runner.start("Bandas", actions.band_curves, self.window.atoms,
                                 self.window.model, self.path.text().strip() or None,
                                 kT=self.window.kT.value(), on_done=self.plot_bands,
                                 on_error=self.window.error)

    def plot_bands(self, curves):
        self.window.remember("bandas", {"path": self.path.text().strip() or "auto"}, curves)
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
        self.window.remember("hubbard", {**self.settings(), "field_tesla": self.field.value()},
                             out)
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
        self.window.remember("barrido de magnetización", self.settings(), out)
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


class GeometryPage(Page):
    """Relaxation with the model's forces, Γ modes, vibrational DOS, animated modes."""

    title = "Geometría y modos"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fmax = _spin(0.001, 1.0, 0.02, 0.005, 3, " eV/Å")
        form.addRow("fmax", self.fmax)
        self.steps = _spin(10, 5000, 500, 50, 0)
        form.addRow("pasos máx.", self.steps)
        self.kmesh = _spin(1, 64, 8, 1, 0)
        form.addRow("malla k (periódicos)", self.kmesh)
        self.site = QComboBox()
        self.site.addItem("(frecuencia)")
        self.site.currentIndexChanged.connect(self.sort_by_site)
        form.addRow("Ordenar modos por sitio", self.site)
        other = QHBoxLayout()
        self.other_model = QComboBox()
        for key, (label, _) in actions.MODELS.items():
            if key != "pi":
                self.other_model.addItem(label, key)
        other.addWidget(self.other_model)
        self.other_button = QPushButton("Frecuencia con el otro modelo")
        self.other_button.clicked.connect(self.run_other)
        other.addWidget(self.other_button)
        form.addRow("Comparar el modo con", other)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.relax = QPushButton("Relajar")
        self.relax.clicked.connect(self.run_relax)
        row.addWidget(self.relax)
        self.undo = QPushButton("Volver a la original")
        self.undo.setEnabled(False)
        self.undo.clicked.connect(self.restore)
        row.addWidget(self.undo)
        self.modes_button = QPushButton("Modos en Γ")
        self.modes_button.clicked.connect(self.run_modes)
        row.addWidget(self.modes_button)
        self.save = QPushButton("Guardar modos…")
        self.save.setEnabled(False)
        self.save.clicked.connect(self.save_modes)
        row.addWidget(self.save)
        layout.addLayout(row)
        self.summary = QLabel("Relaja antes de calcular modos: fuera del mínimo las "
                              "frecuencias no son las armónicas.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        tabs = QTabWidget()
        self.table = table(["#", "ω (cm⁻¹)", "en el sitio", "enriquecimiento", "participación"])
        self.table.itemSelectionChanged.connect(self.selected)
        self.row_modes = []
        tabs.addTab(self.table, "Modos")
        self.vdos_plot = PlotPanel()
        tabs.addTab(self.vdos_plot, "DOS vibracional")
        self.relax_plot = PlotPanel()
        tabs.addTab(self.relax_plot, "Relajación")
        layout.addWidget(tabs)
        anim = QHBoxLayout()
        anim.addWidget(QLabel("amplitud"))
        self.amplitude = _spin(0.05, 2.0, 0.4, 0.05, 2, " Å")
        anim.addWidget(self.amplitude)
        self.show_arrows = QPushButton("Flechas")
        self.show_arrows.clicked.connect(self.arrows)
        anim.addWidget(self.show_arrows)
        stop = QPushButton("Parar")
        stop.clicked.connect(self.stop)
        anim.addWidget(stop)
        layout.addLayout(anim)
        self.original = None
        self.result = None

    def invalidated(self):
        self.result = None
        self.table.setRowCount(0)
        self.row_modes = []
        self.save.setEnabled(False)
        self.site.blockSignals(True)
        self.site.clear()
        self.site.addItem("(frecuencia)")
        self.site.blockSignals(False)

    def sort_by_site(self, *_):
        if self.result is None:
            return
        name = self.site.currentText() if self.site.currentIndex() > 0 else None
        rows = actions.modes_by_site(self.result, name)
        self.row_modes = [r[0] for r in rows]
        fill(self.table, [(r[0], r[1], "" if np.isnan(r[2]) else f"{r[2]:.0%}",
                           "" if np.isnan(r[3]) else f"×{r[3]:.1f}", r[4]) for r in rows])

    def run_other(self):
        if self.result is None:
            self.window.error("Sin modos", "Calcula primero los modos en Γ.")
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.window.error("Elige un modo", "Selecciona un modo en la tabla.")
            return
        index = self.row_modes[rows[0].row()]
        other = actions.load_model(self.other_model.currentData())
        self.window.runner.start("Modo con otro modelo", actions.mode_with_other_model,
                                 self.window.atoms, self.result["vibrations"], index, other,
                                 kmesh=int(self.kmesh.value()), on_done=self.show_other,
                                 on_error=self.window.error)

    def show_other(self, out):
        self.window.remember("modo con otro modelo", {"mode": out["index"]}, out)
        self.summary.setText(
            f"Modo {out['index']}: {out['frequency_cm1']:.1f} cm⁻¹ con el modelo actual; "
            f"{out['other_cm1']:.1f} cm⁻¹ con «{out['other']}» (mismo patrón de desplazamiento: "
            "cociente de Rayleigh, cota superior si el modo no es propio de ese modelo).")

    def run_relax(self):
        if not self.window.require():
            return
        self.window.runner.start("Relajación", actions.relax_structure, self.window.atoms,
                                 self.window.model, fmax=self.fmax.value(),
                                 steps=int(self.steps.value()), kmesh=int(self.kmesh.value()),
                                 scc=self.window.scc_choice(), on_done=self.relaxed,
                                 on_error=self.window.error)

    def relaxed(self, out):
        self.window.remember("relajación", {"fmax": self.fmax.value(),
                                            "steps": int(self.steps.value())}, out)
        if self.original is None:
            self.original = self.window.atoms.copy()
        state = "convergida" if out["converged"] else "SIN CONVERGER"
        self.summary.setText(
            f"Relajación {state} en {out['steps']} pasos · ΔE = "
            f"{out['energy'] - out['energy_start']:+.4f} eV · fuerza máx. "
            f"{out['max_force']:.4f} eV/Å · desplazamiento máx. {out['max_displacement']:.3f} Å")
        ax = self.relax_plot.axes()
        ax.plot(out["trajectory"] - out["trajectory"][-1], "o-", ms=3)
        ax.set_xlabel("paso")
        ax.set_ylabel("E − E_final (eV)")
        ax.set_yscale("symlog", linthresh=1e-3)
        self.relax_plot.draw()
        self.window.set_atoms(out["atoms"], "relajada")
        self.undo.setEnabled(True)

    def restore(self):
        if self.original is not None:
            self.window.set_atoms(self.original, "original")
            self.original = None
            self.undo.setEnabled(False)

    def run_modes(self):
        if not self.window.require():
            return
        self.window.runner.start("Modos en Γ", actions.vibration_modes, self.window.atoms,
                                 self.window.model, kmesh=int(self.kmesh.value()),
                                 scc=self.window.scc_choice(), on_done=self.show_modes,
                                 on_error=self.window.error)

    def show_modes(self, out):
        self.result = out
        vib = out["vibrations"]
        self.window.phonons = (vib.frequencies, vib.modes, f"modelo ({self.window.model.name})")
        self.window.remember("modos en Γ", {}, {"frequencies_cm1": vib.frequencies,
                                                "modes": vib.modes, "rows": out["rows"],
                                                "warnings": out["warnings"]})
        self.site.blockSignals(True)
        self.site.clear()
        self.site.addItem("(frecuencia)")
        for name in out["site_sizes"]:
            self.site.addItem(name)
        self.site.blockSignals(False)
        self.sort_by_site()
        warnings = " ".join(out["warnings"])
        self.summary.setText(f"{len(out['rows'])} modos. Clic en uno para animarlo. {warnings}")
        vdos = out["vdos"]
        ax = self.vdos_plot.axes()
        ax.plot(vdos["grid"], vdos["total"], color="black", lw=1.2, label="total")
        for name, values in vdos.items():
            if name not in ("grid", "total"):
                ax.plot(vdos["grid"], values, lw=0.9, label=name)
        ax.set_xlabel("ω (cm⁻¹)")
        ax.set_ylabel("estados/cm⁻¹")
        ax.legend(fontsize=8)
        self.vdos_plot.draw()
        self.save.setEnabled(True)

    def _mode(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.result is None:
            return None
        return self.result["vibrations"].modes[self.row_modes[rows[0].row()]]

    def selected(self):
        mode = self._mode()
        if mode is not None:
            self.window.view.show(self.window.atoms, keep_camera=True)
            self.window.view.animate_mode(mode, self.amplitude.value(), self.window.timer)

    def arrows(self):
        mode = self._mode()
        if mode is not None:
            self.window.view.stop_animation()
            self.window.view.show(self.window.atoms, keep_camera=True)
            largest = float(np.linalg.norm(mode, axis=1).max()) or 1.0
            self.window.view.arrows(mode, 3.0 * self.amplitude.value() / largest)

    def stop(self):
        self.window.view.stop_animation()
        self.window.view.show(self.window.atoms, keep_camera=True)

    def save_modes(self):
        if self.result is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Carpeta (se crea modes.npz)")
        if directory:
            try:
                path = self.result["vibrations"].save(directory)
                self.window.statusBar().showMessage(f"Guardado {path}")
            except FileExistsError as error:
                self.window.error("No se sobrescribe", str(error))


class PhononPage(Page):
    """Phonons in the whole Brillouin zone through phonopy: dispersion, DOS, Γ symmetry."""

    title = "Fonones (ZB)"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        cells = QHBoxLayout()
        self.supercell = [_spin(1, 12, 4, 1, 0) for _ in range(3)]
        for spin in self.supercell:
            cells.addWidget(spin)
        form.addRow("Supercelda", cells)
        self.delta = _spin(0.001, 0.05, 0.01, 0.002, 3, " Å")
        form.addRow("Desplazamiento", self.delta)
        self.kmesh = _spin(1, 64, 12, 1, 0)
        form.addRow("Malla k de la celda", self.kmesh)
        self.temperature = _spin(1, 3000, 300, 50, 0, " K")
        form.addRow("Temperatura", self.temperature)
        self.cache = QLineEdit()
        self.cache.setPlaceholderText("vacío: sin caché")
        form.addRow("Carpeta (reanudable)", self.cache)
        layout.addLayout(form)
        self.run_button = QPushButton("Calcular fonones en la ZB")
        self.run_button.clicked.connect(self.run)
        layout.addWidget(self.run_button)
        self.summary = QLabel("Relaja la estructura antes. Necesita phonopy "
                              "(pip install 'tbkit[phonons]').")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        tabs = QTabWidget()
        self.bands = PlotPanel()
        tabs.addTab(self.bands, "Dispersión")
        self.dos = PlotPanel()
        tabs.addTab(self.dos, "DOS de fonones")
        self.irreps = table(["ω (cm⁻¹)", "degeneración", "representación"])
        tabs.addTab(self.irreps, "Simetrías en Γ")
        layout.addWidget(tabs)

    def run(self):
        if not self.window.require():
            return
        if not actions.phonopy_available():
            self.window.error("Falta phonopy", "Instálalo con pip install 'tbkit[phonons]'.")
            return
        self.window.runner.start("Fonones (phonopy)", actions.bz_phonons, self.window.atoms,
                                 self.window.model,
                                 supercell=tuple(int(s.value()) for s in self.supercell),
                                 delta=self.delta.value(), kmesh=int(self.kmesh.value()),
                                 scc=self.window.scc_choice(),
                                 temperature=self.temperature.value(),
                                 cache_dir=self.cache.text().strip() or None,
                                 on_done=self.show, on_error=self.window.error)

    def show(self, out):
        self.window.remember("fonones en la ZB",
                             {"supercell": out["supercell"], "delta": self.delta.value()}, out)
        t = out["thermal"]
        warnings = " ".join(out["warnings"])
        self.summary.setText(
            f"Supercelda {out['supercell']} · {out['displacements']} desplazamientos · grupo "
            f"puntual {out['point_group']} · a {t['T_K']:.0f} K por celda: F = {t['F_eV']:.4f} eV, "
            f"S = {t['S_meV_per_K']:.4f} meV/K, Cv = {t['Cv_meV_per_K']:.4f} meV/K. {warnings}")
        d = out["dispersion"]
        ax = self.bands.axes()
        ax.plot(d["x"], d["frequencies"], color="black", lw=0.8)
        for tick in d["ticks"]:
            ax.axvline(tick, color="grey", lw=0.5)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(d["ticks"], d["labels"])
        ax.set_xlim(d["x"][0], d["x"][-1])
        ax.set_ylabel("ω (cm⁻¹)")
        self.bands.draw()
        dos = out["dos"]
        ax = self.dos.axes()
        ax.plot(dos["grid"], dos["total"], color="black", lw=1.2, label="total")
        for name, values in dos.items():
            if name not in ("grid", "total"):
                ax.plot(dos["grid"], values, lw=0.9, label=name)
        ax.set_xlabel("ω (cm⁻¹)")
        ax.set_ylabel("estados/cm⁻¹ por celda")
        ax.legend(fontsize=8)
        self.dos.draw()
        fill(self.irreps, [(r["frequency_cm1"], r["degeneracy"], r["irrep"])
                           for r in out["irreps"]])


class SpectraPage(Page):
    """Raman (non-resonant and resonant) and IR on the shared or imported modes."""

    title = "Espectros"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        source = QHBoxLayout()
        source.addWidget(QLabel("Fonones:"))
        self.source = QComboBox()
        self.source.addItems(["los de «Geometría y modos», o calcularlos con el modelo",
                              "archivo de Quantum ESPRESSO (dynmat/matdyn)…"])
        self.source.activated.connect(self._source_chosen)
        source.addWidget(self.source)
        self.source_label = QLabel("")
        source.addWidget(self.source_label)
        layout.addLayout(source)
        self.qe = None
        form = QFormLayout()
        self.laser = _spin(200, 2000, 532, 1, 0, " nm")
        self.temperature = _spin(1, 2000, 300, 10, 0, " K")
        self.fwhm = _spin(0.5, 100, 8, 0.5, 1, " cm⁻¹")
        row = QHBoxLayout()
        for label, widget in (("láser", self.laser), ("T", self.temperature),
                              ("FWHM", self.fwhm)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        form.addRow("Raman / IR", row)
        self.scale = QCheckBox("Escalar frecuencias")
        form.addRow("", self.scale)
        self.lasers = QLineEdit("1.96 2.33 2.54 3.5")
        self.eta = _spin(0.01, 1.0, 0.1, 0.01, 2, " eV")
        row = QHBoxLayout()
        row.addWidget(self.lasers)
        row.addWidget(QLabel("η"))
        row.addWidget(self.eta)
        form.addRow("Resonante (eV)", row)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.raman_button = QPushButton("Raman")
        self.raman_button.clicked.connect(self.run_raman)
        self.resonant_button = QPushButton("Raman resonante")
        self.resonant_button.clicked.connect(self.run_resonant)
        self.ir_button = QPushButton("IR")
        self.ir_button.clicked.connect(self.run_ir)
        self.export = QPushButton("Exportar CSV…")
        self.export.clicked.connect(self.export_csv)
        self.open_saved = QPushButton("Abrir espectro guardado…")
        self.open_saved.clicked.connect(self.open_spectrum)
        for b in (self.raman_button, self.resonant_button, self.ir_button, self.export,
                  self.open_saved):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self.summary = QLabel("Raman no resonante: sistemas con gap y capa cerrada. Las "
                              "energías de resonancia son las del modelo (gaps TB/KS pequeños), "
                              "no energías ópticas.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        tabs = QTabWidget()
        self.plot = PlotPanel()
        tabs.addTab(self.plot, "Espectro")
        self.table = table(["ω (cm⁻¹)", "deg.", "actividad / intensidad", "ρ"])
        self.table.itemSelectionChanged.connect(self.selected)
        tabs.addTab(self.table, "Modos activos")
        self.profile = PlotPanel()
        tabs.addTab(self.profile, "Perfil de excitación")
        self.tabs = tabs
        layout.addWidget(tabs)
        self.last = None
        self.resonant = None
        self.saved = None

    def invalidated(self):
        self.qe = None
        self.source.setCurrentIndex(0)
        self.source_label.setText("")

    def _source_chosen(self, index):
        if index != 1:
            self.qe = None
            self.source_label.setText("")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Modos de QE", "",
                                              "Modos (*.modes *.out *.eig *.vec *);;Todos (*)")
        if not path or self.window.atoms is None:
            self.source.setCurrentIndex(0)
            return
        try:
            self.qe = actions.qe_phonons(self.window.atoms, path)
            self.source_label.setText(f"{Path(path).name}: {len(self.qe[0])} modos")
        except Exception as error:
            self.window.error("No se pudieron leer los modos de QE", str(error))
            self.source.setCurrentIndex(0)

    def phonons(self):
        """QE modes, the window's shared modes, or None (the model computes them)."""
        if self.qe is not None:
            return self.qe
        if self.window.phonons is not None:
            return self.window.phonons[:2]
        return None

    def _scale(self) -> bool:
        """Scale only the model's own frequencies, never imported QE modes."""
        return self.scale.isChecked() and self.qe is None

    @staticmethod
    def _scale_note(out) -> str:
        factor = out.get("frequency_scale")
        return f"Frecuencias × {factor:.4f} (factor del conjunto frente a GPAW). " \
            if factor else ""

    def _ready(self):
        if self.window.atoms is None or self.window.model is None:
            self.window.error("Sin estructura o modelo", "Abre una estructura y elige un modelo.")
            return False
        problems = actions.check_model(self.window.atoms, self.window.model,
                                       self.window.scc_choice())
        if problems:
            self.window.error("El modelo no sirve para esta estructura", " ".join(problems))
            return False
        return True

    def run_raman(self):
        if self._ready():
            self.window.runner.start("Raman", actions.raman_spectrum, self.window.atoms,
                                     self.window.model, phonons=self.phonons(),
                                     laser_nm=self.laser.value(),
                                     temperature_k=self.temperature.value(),
                                     fwhm=self.fwhm.value(), scale=self._scale(),
                                     on_done=self.show_raman,
                                     on_error=self.window.error)

    def open_spectrum(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Espectro guardado", "",
                                                  "Espectros (*.csv *.npz)")
        if not path:
            return
        try:
            self.show_saved(actions.load_spectrum(path))
        except (OSError, ValueError) as error:
            self.window.error("No se pudo abrir", str(error))

    def show_saved(self, saved):
        """Saved curves, each normalised to its maximum, over the current spectrum if any."""
        self.saved = saved
        ax = self.plot.axes()
        for label, values in saved["curves"].items():
            top = float(np.max(values)) or 1.0
            ax.plot(saved["grid"], values / top, lw=1.1, label=label)
        if self.last is not None and self.last[0] == "raman":
            out = self.last[1]
            top = float(np.max(out["intensity"])) or 1.0
            ax.plot(out["grid"], out["intensity"] / top, color="black", lw=1.4, ls="--",
                    label="calculado aquí")
        ax.set_xlabel("desplazamiento Raman (cm⁻¹)")
        ax.set_ylabel("intensidad (normalizada)")
        ax.legend(fontsize=8)
        self.plot.draw()
        rows = []
        if saved["sticks"]:
            label, (freq, act) = next(iter(saved["sticks"].items()))
            order = np.argsort(-np.asarray(act))[:40]
            rows = [(float(freq[k]), 1, float(act[k]), "") for k in order]
        fill(self.table, rows)
        self.summary.setText(f"Abierto {saved['source']}: {len(saved['curves'])} curvas"
                             + (f"; tabla: modos más activos de «{next(iter(saved['sticks']))}»"
                                if saved["sticks"] else "") + ".")
        self.tabs.setCurrentIndex(0)

    def show_raman(self, out):
        self.last = ("raman", out)
        self.window.remember("raman", self._settings(), out)
        self._table(out["rows"], "actividad (Å⁴/amu)")
        ax = self.plot.axes()
        ax.plot(out["grid"], out["intensity"], color="black", lw=1)
        ax.set_xlabel("desplazamiento Raman (cm⁻¹)")
        ax.set_ylabel(f"intensidad (láser {self.laser.value():.0f} nm, "
                      f"{self.temperature.value():.0f} K)")
        self.plot.draw()
        alpha = float(np.trace(out["alpha"]) / 3)
        self.summary.setText(self._scale_note(out) + f"{out['method']} · α medio {alpha:.2f} Å³ · "
                             f"{len(out['rows'])} conjuntos activos. " + " ".join(out["warnings"]))

    def run_resonant(self):
        if not self._ready():
            return
        try:
            lasers = [float(v) for v in self.lasers.text().replace(",", " ").split()]
        except ValueError:
            self.window.error("Energías de láser", "Escribe números en eV separados por espacios.")
            return
        self.window.runner.start("Raman resonante", actions.resonant_spectrum, self.window.atoms,
                                 self.window.model, lasers, eta=self.eta.value(),
                                 phonons=self.phonons(), scale=self._scale(),
                                 on_done=self.show_resonant,
                                 on_error=self.window.error)

    def show_resonant(self, out):
        self.last = ("resonant", out)
        self.window.remember("raman resonante", {**self._settings(), "eta": self.eta.value(),
                                                 "lasers_ev": list(out["lasers"])}, out)
        self.resonant = out
        lasers = out["lasers"]
        headers = ["ω (cm⁻¹)"] + [f"{e:.2f} eV" for e in lasers]
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        rows = [(float(out["frequencies"][k]),) + tuple(float(a) for a in out["activities"][:, k])
                for k in out["active"]]
        fill(self.table, rows)
        ax = self.plot.axes()
        for i, energy in enumerate(lasers):
            ax.vlines(out["frequencies"], 0, out["activities"][i], lw=1.2,
                      color=f"C{i}", label=f"{energy:.2f} eV")
        ax.set_yscale("log")
        ax.set_xlabel("ω (cm⁻¹)")
        ax.set_ylabel("actividad (Å⁴/amu)")
        ax.legend(fontsize=8)
        self.plot.draw()
        self.summary.setText(f"{out['method']} · elige un modo en «Modos activos» para ver su "
                             "perfil. " + " ".join(out["warnings"]))

    def selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.last is None or self.last[0] != "resonant":
            return
        out = self.resonant
        k = out["active"][rows[0].row()]
        profile = out["result"].profile(float(out["frequencies"][k]))
        ax = self.profile.axes()
        ax.plot(out["lasers"], profile, "o-")
        ax.set_xlabel("ħω_L (eV)")
        ax.set_ylabel("actividad (Å⁴/amu)")
        ax.set_title(f"{out['frequencies'][k]:.1f} cm⁻¹", fontsize=9)
        self.profile.draw()
        self.tabs.setCurrentWidget(self.profile)

    def run_ir(self):
        if self._ready():
            self.window.runner.start("IR", actions.ir_spectrum_of, self.window.atoms,
                                     self.window.model, phonons=self.phonons(),
                                     fwhm=self.fwhm.value(), scale=self._scale(),
                                     on_done=self.show_ir,
                                     on_error=self.window.error)

    def show_ir(self, out):
        self.last = ("ir", out)
        self.window.remember("ir", self._settings(), out)
        self._table(out["rows"], "intensidad (km/mol)")
        ax = self.plot.axes()
        ax.plot(out["grid"], out["absorption"], color="tab:red", lw=1)
        ax.set_xlabel("ω (cm⁻¹)")
        ax.set_ylabel("absorción (km/mol por cm⁻¹)")
        ax.invert_xaxis()
        self.plot.draw()
        self.summary.setText(self._scale_note(out) + f"IR con el dipolo del modelo (cargas + dipolos intraatómicos) · "
                             f"μ = {out['dipole_debye']:.2f} D · semicuantitativo (factor ~2 por "
                             "modo frente a GPAW). " + " ".join(out["warnings"]))

    def _settings(self):
        source = "QE" if self.qe is not None else \
            (self.window.phonons[2] if self.window.phonons is not None else "modelo")
        return {"phonons": source, "laser_nm": self.laser.value(),
                "temperature_k": self.temperature.value(), "fwhm_cm1": self.fwhm.value()}

    def _table(self, rows, label):
        self.table.clear()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ω (cm⁻¹)", "deg.", label, "ρ"])
        fill(self.table, rows)

    def export_csv(self):
        if self.last is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar", "espectro.csv", "CSV (*.csv)")
        if not path:
            return
        kind, out = self.last
        if kind == "raman":
            actions.write_csv(path, {"shift_cm1": out["grid"], "intensity": out["intensity"]})
        elif kind == "ir":
            actions.write_csv(path, {"wavenumber_cm1": out["grid"],
                                     "absorption_km_mol_per_cm1": out["absorption"]})
        else:
            columns = {"frequency_cm1": out["frequencies"]}
            for i, energy in enumerate(out["lasers"]):
                columns[f"activity_{energy:.3f}eV"] = out["activities"][i]
            actions.write_csv(path, columns)
        self.window.statusBar().showMessage(f"Exportado {path}")


class GraphenePage(Page):
    """Pristine graphene: G (third order) and 2D, 2D′ (double resonance) per laser."""

    title = "Grafeno"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.lasers = QLineEdit("1.96 2.41 2.71")
        form.addRow("Láseres (eV)", self.lasers)
        self.phonons = QComboBox()
        for key, label in actions.GRAPHENE_PHONONS.items():
            self.phonons.addItem(label, key)
        self.phonons.addItem("archivo JSON de constantes de fuerza…", "file")
        form.addRow("Fonones", self.phonons)
        self.gamma = _spin(0.01, 1.0, 0.1, 0.01, 2, " eV")
        form.addRow("γ electrónico", self.gamma)
        self.dk = _spin(0.002, 0.1, 0.01, 0.002, 3, " 1/Å")
        self.dq = _spin(0.005, 0.2, 0.03, 0.005, 3, " 1/Å")
        row = QHBoxLayout()
        row.addWidget(self.dk)
        row.addWidget(QLabel("dq"))
        row.addWidget(self.dq)
        form.addRow("malla dk", row)
        self.workers = _spin(1, 64, 1, 1, 0)
        form.addRow("procesos", self.workers)
        layout.addLayout(form)
        self.run_button = QPushButton("Calcular G, 2D y 2D′")
        self.run_button.clicked.connect(self.run)
        layout.addWidget(self.run_button)
        self.summary = QLabel("Grafeno prístino, electrones π con t(d). Cuesta minutos con la "
                              "malla fina; dk = 0,05, dq = 0,1 da una vista rápida.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        tabs = QTabWidget()
        self.spectra = PlotPanel()
        tabs.addTab(self.spectra, "Segundo orden")
        self.table = table(["láser (eV)", "G (cm⁻¹)", "2D (cm⁻¹)", "2D′ (cm⁻¹)", "I(2D)/I(G)"])
        tabs.addTab(self.table, "Posiciones")
        self.dispersion = PlotPanel()
        tabs.addTab(self.dispersion, "Dispersión de la 2D")
        layout.addWidget(tabs)

    def run(self):
        try:
            lasers = [float(v) for v in self.lasers.text().replace(",", " ").split()]
        except ValueError:
            self.window.error("Energías de láser", "Escribe números en eV separados por espacios.")
            return
        source = self.phonons.currentData()
        if source == "file":
            source, _ = QFileDialog.getOpenFileName(self, "Constantes de fuerza", "",
                                                    "JSON (*.json)")
            if not source:
                return
        self.settings = {"lasers_ev": lasers, "phonons": source, "gamma": self.gamma.value(),
                         "dk": self.dk.value(), "dq": self.dq.value()}
        self.window.runner.start("Grafeno", actions.graphene_spectra, lasers, source,
                                 gamma=self.gamma.value(), dk=self.dk.value(),
                                 dq=self.dq.value(), workers=int(self.workers.value()),
                                 on_done=self.show, on_error=self.window.error)

    def show(self, out):
        if self.window.atoms is not None and self.window.model is not None:
            self.window.remember("grafeno G/2D/2D′", self.settings, out)
        fill(self.table, out["rows"])
        ax = self.spectra.axes()
        for i, r in enumerate(out["results"]):
            ax.plot(r["grid"], r["spectrum"] / r["g_intensity"], lw=1, color=f"C{i}",
                    label=f"{r['laser_ev']:.2f} eV")
        ax.set_xlabel("desplazamiento Raman (cm⁻¹)")
        ax.set_ylabel("I / I(G)")
        ax.legend(fontsize=8)
        self.spectra.draw()
        ax = self.dispersion.axes()
        ax.plot(out["lasers"], [r[2] for r in out["rows"]], "o-", label="2D")
        ax.set_xlabel("ħω_L (eV)")
        ax.set_ylabel("posición de la 2D (cm⁻¹)")
        self.dispersion.draw()
        slope = out["dispersion_2d"]
        text = f"dispersión de la 2D: {slope:.0f} cm⁻¹/eV (exp. ~100)" if np.isfinite(slope) \
            else "un solo láser: sin dispersión"
        self.summary.setText(f"{text}. Posiciones altas por los fonones PBE; energías de "
                             "resonancia del modelo π.")


def main(argv=None):
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    args = sys.argv[1:] if argv is None else argv[1:]
    if args:
        window.open_structure(args[0])
    window.show()
    return app.exec()
