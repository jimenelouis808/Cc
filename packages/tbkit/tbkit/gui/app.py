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
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
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
        self.resize(1600, 950)
        self.atoms = None
        #: Every geometry of this session (opened, relaxed, with a cell, a supercell…), as
        #: (name, Atoms); the active one is self.atoms and every tab computes on it.
        self.geometries: list[tuple[str, object]] = []
        #: Recipes launched from this window: dicts with popen, name, cores, log, started.
        self.launched: list[dict] = []
        #: What was done on each geometry, by id (the Guía tab): «relajada», «raman», «dft»…
        self.done_tasks: dict[int, set] = {}
        self._picked = None
        self.model = None
        self.model_name = None
        self.state: Optional[actions.GroundState] = None
        #: Γ modes of the current structure, shared by the pages: (frequencies, L, source).
        self.phonons = None
        self.runner = Runner(self)
        self.runner.message.connect(self.statusBar().showMessage)
        self.runner.busy.connect(self._busy)
        self.runner.progress.connect(self._job_progress)
        self.timer = QTimer(self)
        self._status_widgets()

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
        self.runner.frame.connect(self.view.update_positions)
        if interactive:
            self.view.enable_picking(self.atom_clicked)
        self.tabs = QTabWidget()
        self.pages = {}
        self.info_boxes = []
        for page in self.page_classes():
            widget = page(self)
            self.pages[widget.title] = widget
            self.tabs.addTab(self._with_info(widget), widget.title)
        splitter.addWidget(self.tabs)
        self.tabs.currentChanged.connect(
            lambda _: self.pages["Guía"].update_view()
            if self.tabs.currentIndex() == list(self.pages).index("Guía") else None)
        # left column narrow, the 3D view the widest: it is where the structure is read
        splitter.setSizes([320, 680, 600])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        self.splitter = splitter
        self.setCentralWidget(splitter)
        self.last_record = None
        self._menu()
        self._apply_help()
        self.statusBar().showMessage("Abre una estructura (xyz, extxyz, cif, POSCAR…).")

    def _menu(self):
        menu = self.menuBar().addMenu("Archivo")
        for text, slot in (("Abrir proyecto…", self.open_project),
                           ("Guardar proyecto", self.save_project),
                           ("Guardar proyecto como…", self.save_project_as),
                           (None, None),
                           ("Abrir estructura…", self.open_dialog),
                           ("Guardar estructura…", self.save_structure),
                           ("Guardar imagen de la vista…", self.save_image),
                           (None, None),
                           ("Guardar registro del último cálculo…", self.save_record),
                           ("Abrir registro…", self.load_record),
                           (None, None),
                           ("Salir", self.close)):
            if text is None:
                menu.addSeparator()
            else:
                action = menu.addAction(text)
                action.triggered.connect(lambda _=False, slot=slot: slot())
                if text == "Guardar proyecto":
                    action.setShortcut("Ctrl+S")
                elif text == "Abrir proyecto…":
                    action.setShortcut("Ctrl+O")
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
        # In a scroll area: a page's buttons no longer set the tab column's minimum width,
        # so the 3D view keeps its share of the window (a scroll bar appears instead).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(box)
        return scroll

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
        """The last finished calculation, for «Guardar registro»; the Guía tab keeps what was
        done on each geometry."""
        if self.atoms is not None and self.model is not None:
            self.last_record = (self.atoms.copy(), self.model, task, dict(settings), results)
        self.mark_done(task)

    def mark_done(self, task: str, atoms=None):
        """Remember that ``task`` was done on a geometry (the active one by default)."""
        atoms = self.atoms if atoms is None else atoms
        if atoms is not None:
            self.done_tasks.setdefault(id(atoms), set()).add(task)

    def done(self, task: str) -> bool:
        return self.atoms is not None and task in self.done_tasks.get(id(self.atoms), set())

    def save_structure(self):
        """The active geometry to a file (extxyz keeps cell and periodic axes)."""
        if self.atoms is None:
            return
        import re

        name = self.geometries[self.geometry_combo.currentIndex()][0] if self.geometries \
            else "estructura"
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("_") or "estructura"
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar geometría", f"{stem}.extxyz",
            "extxyz (*.extxyz);;xyz (con celda) (*.xyz);;Materials Studio (*.xtl);;"
            "CIF (*.cif);;VASP (POSCAR*);;Todos (*)")
        if path:
            from ..structure_io import write_structure

            try:
                write_structure(self.atoms, path)
            except ValueError as error:
                self.error("No se pudo guardar", str(error))
                return
            self.statusBar().showMessage(f"Guardada {path}")
            self.console.line(f"geometría «{name}» guardada en {path}")

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
        # The guide first, then in the order of the work: geometry (relax), then what is
        # computed on it.
        return [GuidePage, GeometryPage, ElectronicPage, OrbitalPage, MagnetismPage, PhononPage,
                SpectraPage, GraphenePage, RecipesPage, ProcessesPage]

    # --- left panel ---------------------------------------------------------------

    def _left_panel(self) -> QWidget:
        """Controls on top, the terminal below, with a divider between them."""
        column = QSplitter(Qt.Vertical)
        panel = QWidget()
        layout = QVBoxLayout(panel)

        structure = QGroupBox("Estructura")
        form = QFormLayout(structure)
        buttons = QHBoxLayout()
        for text, slot in (("Abrir…", self.open_dialog), ("Guardar…", self.save_structure),
                           ("Celda…", self.cell_dialog)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        form.addRow(buttons)
        self.geometry_combo = QComboBox()
        self.geometry_combo.activated.connect(self.choose_geometry)
        form.addRow("Geometría", self.geometry_combo)
        repeat = QHBoxLayout()
        self.repeat = []
        for _ in range(3):
            box = QSpinBox()
            box.setRange(1, 20)
            self.repeat.append(box)
            repeat.addWidget(box)
        see = QPushButton("Ver")
        see.clicked.connect(lambda: self.view.set_repeat([b.value() for b in self.repeat]))
        repeat.addWidget(see)
        make = QPushButton("Crear")
        make.clicked.connect(self.make_supercell)
        repeat.addWidget(make)
        form.addRow("Repetir", repeat)
        self.structure_label = QLabel("—")
        self.structure_label.setWordWrap(True)
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

        resources = QGroupBox("Recursos del cálculo")
        form = QFormLayout(resources)
        from .resources import cores as available_cores

        self.cores_box = QSpinBox()
        self.cores_box.setRange(1, available_cores())
        self.cores_box.setValue(available_cores())
        self.cores_box.setSuffix(f" de {available_cores()}")
        self.cores_box.valueChanged.connect(self._resources_changed)
        form.addRow("Núcleos", self.cores_box)
        machine = self.monitor.machine()
        total = machine["ram_total_gb"] or 8.0
        self.ram_box = QDoubleSpinBox()
        self.ram_box.setRange(0.2, total)
        self.ram_box.setDecimals(1)
        self.ram_box.setSingleStep(0.5)
        self.ram_box.setValue(round(0.8 * total, 1))
        self.ram_box.setSuffix(f" GB de {total:.1f}")
        self.ram_box.valueChanged.connect(self._resources_changed)
        form.addRow("RAM máx.", self.ram_box)
        self.memory_label = QLabel("—")
        self.memory_label.setWordWrap(True)
        form.addRow(self.memory_label)
        layout.addWidget(resources)
        self._resources_changed()

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
        panel.setMaximumWidth(380)
        terminal.setMaximumWidth(380)
        for combo in (self.model_combo, self.geometry_combo, self.scc, self.colour):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        column.addWidget(panel)
        column.addWidget(terminal)
        column.setStretchFactor(0, 0)
        column.setStretchFactor(1, 1)
        return column

    def _busy(self, busy: bool):
        self.cancel_button.setEnabled(busy)
        self.job_bar.setVisible(busy)
        if not busy:
            self.job_label.setText("")

    # --- status bar: the running job's progress and the machine's resources -----------

    def _status_widgets(self):
        from .resources import Monitor

        self.monitor = Monitor()
        bar = self.statusBar()
        self.job_label = QLabel("")
        self.job_bar = QProgressBar()
        self.job_bar.setMaximumWidth(160)
        self.job_bar.setMaximumHeight(14)
        self.job_bar.setVisible(False)
        self.usage_label = QLabel("")
        bar.addPermanentWidget(self.job_label)
        bar.addPermanentWidget(self.job_bar)
        bar.addPermanentWidget(self.usage_label)
        self.usage_timer = QTimer(self)
        self.usage_timer.timeout.connect(self.update_usage)
        self.usage_timer.start(1500)

    def _job_progress(self, data: dict):
        self.job_label.setText(data.get("line", "")[:90])
        if data.get("total"):
            self.job_bar.setRange(0, int(data["total"]))
            self.job_bar.setValue(int(data["done"]))
        else:
            self.job_bar.setRange(0, 0)                # open-ended: a moving bar

    def update_usage(self):
        """CPU, RAM (and GPU) of the machine, and what the running job uses."""
        from .resources import gb

        m = self.monitor.machine()
        parts = []
        if m["cpu_percent"] is not None:
            parts.append(f"CPU {m['cpu_percent']:.0f} %")
        if m["ram_total_gb"]:
            parts.append(f"RAM {m['ram_used_gb']:.1f}/{m['ram_total_gb']:.1f} GB")
        for g in m["gpus"]:
            parts.append(f"GPU {g['percent']:.0f} % {g['memory_used_gb']:.1f}/"
                         f"{g['memory_total_gb']:.1f} GB")
        job = self.runner.last_usage
        if job:
            cpu = f", {job['cpu_percent']:.0f} % CPU" if job.get("cpu_percent") is not None \
                else ""
            parts.append(f"cálculo {gb(job['rss_gb'])}{cpu}")
        self.usage_label.setText(" · ".join(parts) or "(sin datos de recursos)")
        tight = m["ram_total_gb"] and m["ram_used_gb"] / m["ram_total_gb"] > 0.85
        self.usage_label.setStyleSheet("QLabel { color: #b00020; font-weight: bold; }"
                                       if tight else "")
        self.machine_usage = m

    def _resources_changed(self, *_):
        from .resources import cores as available_cores

        cores = int(self.cores_box.value())
        self.runner.cores = None if cores >= available_cores() else cores
        self.runner.memory_limit_gb = float(self.ram_box.value())
        self._memory_note()

    def _memory_note(self):
        if not hasattr(self, "memory_label"):
            return
        if self.atoms is None or self.model is None:
            self.memory_label.setText("—")
            return
        need = actions.memory_estimate_gb(self.atoms, self.model)
        n = actions.orbital_count(self.atoms, self.model)
        from .resources import gb

        text = f"{n} orbitales: ≈ {gb(need)} por diagonalización"
        if need > self.ram_box.value():
            text = "⚠ " + text + " (más que el límite)"
        self.memory_label.setText(text)

    def allocated_cores(self) -> int:
        """Cores given to what runs now from this window: the window's job and the recipes
        launched from the Recetas tab that are still alive."""
        from .resources import cores as available_cores

        used = (self.runner.cores or available_cores()) if self.runner.running else 0
        for entry in self.launched:
            if entry["popen"].poll() is None:
                used += entry["cores"]
        return used

    def closeEvent(self, event):
        alive = [e for e in self.launched if e["popen"].poll() is None]
        if self.runner.running:
            answer = QMessageBox.question(self, "tbkit", "Hay un cálculo en marcha. ¿Detenerlo y "
                                          "cerrar? (Las recetas lanzadas siguen corriendo.)"
                                          if alive else "Hay un cálculo en marcha. ¿Detenerlo "
                                          "y cerrar?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self.runner.shutdown()
        if self.atoms is not None:                 # never lose a session: autosave on close
            try:
                path = self.save_project(self.autosave_path(), quiet=True)
                print(f"tbkit: sesión guardada en {path}")
            except Exception as error:  # noqa: BLE001  (closing must not fail)
                print(f"tbkit: no se pudo autoguardar: {error}")
        event.accept()

    # --- projects: a snapshot of the session -------------------------------------------

    @staticmethod
    def autosave_path() -> Path:
        return Path.home() / ".tbkit" / "ultima_sesion.tbkproj"

    def project_state(self):
        """The session as a :class:`project.Project`."""
        from ..params import model_to_dict
        from .project import Project

        window = {"model_name": self.model_name, "charge": self.charge.value(),
                  "scc": self.scc.currentIndex(), "kT": self.kT.value(),
                  "colour": self.colour.currentIndex(), "cores": self.cores_box.value(),
                  "ram_gb": self.ram_box.value(), "repeat": [b.value() for b in self.repeat],
                  "view_repeat": list(self.view.repeat), "tab": self.tabs.currentIndex()}
        pages = {}
        for title, page in self.pages.items():
            values = widget_values(page)
            if hasattr(page, "extra_state"):
                values["__extra__"] = page.extra_state()
            pages[title] = values
        modes = None
        geometry = self.pages["Geometría y modos"]
        if geometry.result is not None:
            vib = geometry.result["vibrations"]
            modes = {"hessian": vib.hessian, "masses": vib.masses, "atoms": vib.atoms,
                     "source": vib.source}
        record = None
        if self.last_record is not None:
            atoms, model, task, settings, results = self.last_record
            record = actions.record(atoms, model, task, settings, results)
        elif getattr(self, "loaded_record", None):
            record = self.loaded_record
        return Project(list(self.geometries), max(self.geometry_combo.currentIndex(), 0),
                       window, pages, model_to_dict(self.model) if self.model else None,
                       modes, record, self.console.text())

    def save_project(self, path=None, quiet: bool = False):
        """Save the session; the first time (or «como…»), asks where."""
        from . import project

        path = path or getattr(self, "project_path", None)
        if not path:
            return self.save_project_as()
        if self.runner.running and not quiet:
            self.statusBar().showMessage("Se guarda el estado; el cálculo en marcha no se "
                                         "guarda (al terminar, guarda otra vez).")
        written = project.save(path, self.project_state())
        if not quiet:
            self.project_path = written
            self.setWindowTitle(f"tbkit — {written.name}")
            self.console.line(f"proyecto guardado: {written}")
            self.statusBar().showMessage(f"Proyecto guardado en {written}")
        return written

    def save_project_as(self):
        from .project import SUFFIX

        path, _ = QFileDialog.getSaveFileName(self, "Guardar proyecto", f"proyecto{SUFFIX}",
                                              f"Proyecto de tbkit (*{SUFFIX})")
        if path:
            self.project_path = None
            return self.save_project(path)
        return None

    def open_project(self, path=None):
        from . import project

        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Abrir proyecto", "",
                                                  f"Proyecto de tbkit (*{project.SUFFIX})")
        if not path:
            return
        try:
            state = project.load(path)
        except Exception as error:  # noqa: BLE001  (a bad file is reported, not raised)
            self.error("No se pudo abrir el proyecto", str(error))
            return
        self.restore_project(state)
        self.project_path = Path(path)
        self.setWindowTitle(f"tbkit — {Path(path).name}")
        self.console.line(f"proyecto abierto: {path} (guardado {state.saved}, "
                          f"tbkit {state.tbkit})")

    def restore_project(self, state):
        from ..params import model_from_dict

        if state.console:
            self.console.write(state.console.rstrip("\n") + "\n── proyecto reabierto ──\n")
        self.geometries = []
        self.geometry_combo.clear()
        for name, atoms in state.geometries:
            self.set_atoms(atoms, name)
        if state.geometries:
            self.choose_geometry(min(state.active, len(state.geometries) - 1))
        w = state.window
        name = w.get("model_name")
        if state.model is not None:
            builtin = name in actions.MODELS if isinstance(name, str) else False
            self.set_model(name, None if builtin else model_from_dict(state.model))
        for box, key in ((self.charge, "charge"), (self.kT, "kT"), (self.cores_box, "cores"),
                         (self.ram_box, "ram_gb")):
            if key in w:
                box.setValue(w[key])
        for combo, key in ((self.scc, "scc"), (self.colour, "colour")):
            if key in w:
                combo.setCurrentIndex(int(w[key]))
        for box, n in zip(self.repeat, w.get("repeat", []), strict=False):
            box.setValue(int(n))
        self.view.set_repeat(w.get("view_repeat", (1, 1, 1)))
        for title, values in state.pages.items():
            page = self.pages.get(title)
            if page is None:
                continue
            extra = values.pop("__extra__", None)
            set_widget_values(page, values)
            if extra is not None and hasattr(page, "restore_extra"):
                page.restore_extra(extra)
        if "tab" in w:
            self.tabs.setCurrentIndex(int(w["tab"]))
        self.loaded_record = state.record
        if state.modes is not None:
            m = state.modes
            page = self.pages["Geometría y modos"]
            self.runner.start("Modos del proyecto", actions.modes_from_hessian, m["atoms"],
                              m["hessian"], m["masses"], m["source"],
                              on_done=page.show_modes, on_error=self.error)

    # --- geometries of the session ---------------------------------------------------------

    def launch_dft(self, folder: str, mpi: int = 1):
        """Run a DFT request with carbonforge (``carbonforge dft correr``) in the background;
        it shows in Procesos with its log."""
        import shutil
        import subprocess
        import sys
        import time

        command = [shutil.which("carbonforge")] if shutil.which("carbonforge") else None
        if command is None:
            try:
                import importlib.util

                if importlib.util.find_spec("carbonforge") is not None:
                    command = [sys.executable, "-m", "carbonforge.cli.main"]
            except (ImportError, ValueError):
                command = None
        if command is None:
            self.error("carbonforge no está instalado aquí",
                       f"La solicitud está en {folder}: córrela donde esté carbonforge "
                       "(carbonforge dft correr CARPETA) o en un clúster (trabajo.slurm).")
            return None
        log = Path(folder) / "registro.log"
        with open(log, "a", encoding="utf-8") as handle:
            process = subprocess.Popen(command + ["dft", "correr", str(folder), "--mpi", str(mpi)],
                                       cwd=folder, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.launched.append({"popen": process, "name": f"DFT {Path(folder).name}",
                              "cores": mpi, "log": log, "started": time.time(),
                              "command": "carbonforge dft correr"})
        self.console.line(f"DFT lanzado con carbonforge (PID {process.pid}, {mpi} MPI); "
                          "avance en la pestaña Procesos")
        return process

    def choose_geometry(self, index: int):
        if 0 <= index < len(self.geometries):
            name, atoms = self.geometries[index]
            self.set_atoms(atoms, name)

    def add_geometry(self, name: str, atoms):
        """A new geometry of the session, made the active one."""
        self.set_atoms(atoms, name)

    def make_supercell(self):
        if self.atoms is None:
            return
        repeat = [b.value() for b in self.repeat]
        try:
            bigger = actions.supercell(self.atoms, repeat)
        except ValueError as error:
            self.error("No se puede repetir", str(error))
            return
        name = self.geometries[self.geometry_combo.currentIndex()][0] \
            if self.geometries else "estructura"
        self.view.set_repeat((1, 1, 1))
        self.add_geometry(f"{name} {'x'.join(map(str, repeat))}", bigger)

    def cell_dialog(self):
        if self.atoms is None:
            self.error("Sin estructura", "Abre primero una estructura.")
            return
        dialog = CellDialog(self.atoms, self)
        if dialog.exec() == QDialog.Accepted:
            try:
                atoms = dialog.result_atoms()
            except ValueError as error:
                self.error("Celda no válida", str(error))
                return
            name = self.geometries[self.geometry_combo.currentIndex()][0] \
                if self.geometries else "estructura"
            self.add_geometry(f"{name} (celda editada)", atoms)

    def atom_clicked(self, index: int):
        """The clicked atom in the status bar; two clicks in a row give their distance."""
        atom = self.atoms[index]
        x, y, z = atom.position
        text = f"átomo {index} {atom.symbol} ({x:.3f}, {y:.3f}, {z:.3f}) Å"
        if self._picked is not None and self._picked != index:
            d = self.atoms.get_distance(self._picked, index, mic=bool(self.atoms.pbc.any()))
            text += f" · d({self._picked}–{index}) = {d:.3f} Å"
        self._picked = index
        self.statusBar().showMessage(text)
        self.console.line(text)

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar imagen", "estructura.png",
                                              "PNG (*.png)")
        if path:
            self.view.screenshot(path)
            self.statusBar().showMessage(f"Imagen guardada en {path}")

    # --- structure and model ------------------------------------------------------------

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir estructura", "",
                                              "Estructuras (*.xyz *.extxyz *.xtl *.cif *.vasp "
                                              "POSCAR* CONTCAR* *.pdb *.json *.traj *.in "
                                              "*.gen);;Todos (*)")
        if path:
            self.open_structure(path)

    def open_structure(self, path):
        try:
            atoms = actions.read_structure(path)
        except Exception as error:
            self.error(f"No se pudo leer {path}", str(error))
            return
        self.geometries = []
        self.geometry_combo.clear()
        self.view.set_repeat((1, 1, 1))
        self.set_atoms(atoms, Path(path).name)
        if atoms.info.get("periodicity"):
            self.console.line(atoms.info["periodicity"])
        elif not atoms.pbc.any():
            self.console.line("sin ejes periódicos: si la estructura lo es, «Celda…» fija la "
                              "celda y los ejes")

    def set_atoms(self, atoms, name: str = ""):
        """Make ``atoms`` the active geometry (adding it to the session's list if new)."""
        for k, (_, known) in enumerate(self.geometries):
            if known is atoms:
                break
        else:
            name = name or f"geometría {len(self.geometries) + 1}"
            self.geometries.append((name, atoms))
            self.geometry_combo.addItem(name)
            k = len(self.geometries) - 1
        self.geometry_combo.setCurrentIndex(k)
        self._picked = None
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
        self._memory_note()
        self.view.show(atoms)

    def _model_chosen(self, index):
        name = self.model_combo.itemData(index)
        if name == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Parámetros", "", "JSON (*.json)")
            if path:
                self.set_model(path)
        else:
            self.set_model(name)

    def set_model(self, name_or_path: str, model=None):
        """Use a built-in set (its name) or a parameter file; ``model``: an already built
        TBModel (a project's own parameters), shown under ``name_or_path``."""
        try:
            self.model = model if model is not None else actions.load_model(name_or_path)
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
        self._memory_note()

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
        need = actions.memory_estimate_gb(self.atoms, self.model)
        free = getattr(self, "machine_usage", {}).get("ram_available_gb")
        if need > self.ram_box.value() or (free and need > 0.9 * free):
            limit = f"límite {self.ram_box.value():.1f} GB" + (f", libres {free:.1f} GB"
                                                                 if free else "")
            self.error("No cabe en memoria",
                       f"Una diagonalización de esta estructura necesita ≈ {need:.1f} GB "
                       f"({limit}). Sube «RAM máx.» si la máquina la tiene, o usa una "
                       "estructura más pequeña o el modelo π (un orbital por átomo).")
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


class GuidePage(Page):
    """Where the work on the active geometry stands, what to do next and what applies to this
    system with this model (from what tbkit measured)."""

    title = "Guía"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        self.header = QLabel("—")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)
        self.steps = table(["", "paso", "para qué"])
        self.steps.cellDoubleClicked.connect(self.go)
        layout.addWidget(self.steps, stretch=2)
        row = QHBoxLayout()
        self.next_button = QPushButton("Ir al paso siguiente")
        self.next_button.clicked.connect(self.go_next)
        row.addWidget(self.next_button)
        many = QPushButton("Muchos sistemas: nuevo estudio")
        many.clicked.connect(self.study)
        row.addWidget(many)
        layout.addLayout(row)
        self.caveats = QLabel("—")
        self.caveats.setWordWrap(True)
        self.caveats.setTextFormat(Qt.RichText)
        layout.addWidget(self.caveats, stretch=1)
        self.validity = QLabel("")
        self.validity.setWordWrap(True)
        self.validity.setStyleSheet("QLabel { color: palette(mid); }")
        layout.addWidget(self.validity)
        self.current = []
        self.poll = QTimer(self)
        self.poll.timeout.connect(self.refresh)
        self.poll.start(2000)

    def info(self) -> dict:
        w = self.window
        geometry = w.pages["Geometría y modos"]
        modes = geometry.result is not None and w.atoms is not None and \
            len(geometry.result["vibrations"].atoms) == len(w.atoms)
        return {"atoms": w.atoms, "model": w.model,
                "model_problems": actions.check_model(w.atoms, w.model)
                if w.atoms is not None and w.model is not None else ["sin modelo"],
                "relaxed": w.done("relajada"), "state": w.state,
                "modes": modes or w.done("modos en Γ"),
                "spectra": any(w.done(t) for t in ("raman", "raman resonante", "ir")),
                "dft_checked": w.done("dft"),
                "project_saved": getattr(w, "project_path", None) is not None}

    def refresh(self, *_):
        """From the timer: only while the tab is seen (the first time, always)."""
        if self.isVisible() or not self.current:
            self.update_view()

    def update_view(self):
        out = actions.guide(self.info())
        self.current = out["steps"]
        marks = {"hecho": "✓", "siguiente": "➜", "pendiente": "·"}
        fill(self.steps, [(marks[s["status"]], s["title"], s["why"]) for s in out["steps"]])
        for r, step in enumerate(out["steps"]):
            if step["status"] == "siguiente":
                for c in range(3):
                    font = self.steps.item(r, c).font()
                    font.setBold(True)
                    self.steps.item(r, c).setFont(font)
        self.steps.setColumnWidth(0, 28)
        self.steps.setColumnWidth(1, 240)
        self.steps.resizeRowsToContents()
        nxt = next((s for s in out["steps"] if s["status"] == "siguiente"), None)
        atoms = self.window.atoms
        what = (f"{atoms.get_chemical_formula()} · {len(atoms)} átomos · periódica en "
                f"{int(atoms.pbc.sum())} eje(s)") if atoms is not None else "sin estructura"
        self.header.setText(f"<b>{what}</b>" + (f" — siguiente: {nxt['title']}" if nxt
                                                 else " — todos los pasos hechos"))
        self.next_button.setEnabled(nxt is not None)
        items = "".join(f"<li>{c}</li>" for c in out["caveats"])
        self.caveats.setText(f"<b>Para este sistema</b><ul>{items}</ul>" if items
                             else "<b>Para este sistema</b>: nada que advertir todavía.")
        validity = out["validity"]
        self.validity.setText(f"Validez del modelo (de su archivo): {validity[:600]}"
                              + ("…" if len(validity) > 600 else "") if validity else "")

    def go(self, row: int, *_):
        if not 0 <= row < len(self.current):
            return
        step = self.current[row]
        if step["key"] == "estructura":
            self.window.open_dialog()
        elif step["key"] == "celda":
            self.window.cell_dialog()
        elif step["key"] == "proyecto":
            self.window.save_project()
        elif step["tab"]:
            self.window.tabs.setCurrentIndex(list(self.window.pages).index(step["tab"]))

    def go_next(self):
        for r, step in enumerate(self.current):
            if step["status"] == "siguiente":
                self.go(r)
                return

    def study(self):
        self.window.tabs.setCurrentIndex(list(self.window.pages).index("Recetas"))
        self.window.pages["Recetas"].new_study()


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
        self._dos_running = False
        self._dos_again = False

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
        """DOS/PDOS in the job process (a PDOS by orbital of a large structure takes a while
        and froze the window when computed here); asked again while one runs, it runs once
        more when that one ends, with the latest choice."""
        state = self.window.state
        if state is None:
            return
        if self._dos_running:
            self._dos_again = True
            return
        by = {0: "element", 1: "atom", 2: "orbital", 3: None}[self.projection.currentIndex()]
        if by == "atom" and len(self.window.atoms) > 30:
            by = "element"
        self._dos_running = True
        self.window.runner.start("DOS/PDOS", actions.dos_curves, state, self.sigma.value(), by,
                                 on_done=self.draw_dos, on_error=self._dos_failed)

    def _dos_failed(self, text, trace=""):
        self._dos_running = False
        self.window.error("DOS/PDOS", trace or text)

    def draw_dos(self, curves):
        self._dos_running = False
        if self._dos_again:
            self._dos_again = False
            self.plot_dos()
            return
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
        export = QPushButton("Exportar geometría…")
        export.clicked.connect(self.window.save_structure)
        row.addWidget(export)
        self.modes_button = QPushButton("Modos en Γ")
        self.modes_button.clicked.connect(self.run_modes)
        row.addWidget(self.modes_button)
        self.save = QPushButton("Guardar modos…")
        self.save.setEnabled(False)
        self.save.clicked.connect(self.save_modes)
        row.addWidget(self.save)
        load = QPushButton("Abrir modos…")
        load.clicked.connect(self.open_modes)
        row.addWidget(load)
        layout.addLayout(row)
        row = QHBoxLayout()
        dft = QPushButton("Validar con DFT…")
        dft.clicked.connect(self.validate_dft)
        row.addWidget(dft)
        dft_result = QPushButton("Ver resultado DFT…")
        dft_result.clicked.connect(self.show_dft)
        row.addWidget(dft_result)
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
        name = self.window.geometries[self.window.geometry_combo.currentIndex()][0] \
            if self.window.geometries else "estructura"
        self.original = self.window.atoms
        self.window.add_geometry(f"{name} relajada (fmax {self.fmax.value():g})", out["atoms"])
        self.window.mark_done("relajada")
        self.undo.setEnabled(True)

    def restore(self):
        """Back to the geometry the relaxation started from (it stays in the list)."""
        if self.original is not None:
            self.window.set_atoms(self.original)
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
        self.window.phonons = (vib.frequencies, vib.modes,
                               vib.source or f"modelo ({self.window.model.name})")
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

    def selected_modes(self) -> list[int]:
        if self.result is None:
            return []
        return sorted({self.row_modes[r.row()]
                       for r in self.table.selectionModel().selectedRows()})

    def validate_dft(self):
        """A DFT request for the active geometry (and the selected modes), run by carbonforge."""
        if not self.window.require():
            return
        dialog = DftDialog(self.window, self.selected_modes(), self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            folder = dialog.write()
        except (ValueError, FileExistsError) as error:
            self.window.error("No se pudo escribir la solicitud", str(error))
            return
        self.window.console.line(f"solicitud DFT escrita en {folder}")
        if dialog.run_now.isChecked():
            self.window.launch_dft(folder, int(dialog.mpi.value()))
        else:
            self.window.console.line(f"córrela con: carbonforge dft correr {folder} "
                                     f"(o copia la carpeta a un clúster: trabajo.slurm)")

    def show_dft(self, folder: str | None = None):
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Carpeta de la solicitud DFT")
        if not folder:
            return
        if self.window.model is None:
            self.window.error("Sin modelo", "Elige el modelo con el que comparar.")
            return

        def shown(out):
            lines = actions.dft_summary_lines(out["summary"])
            for line in lines:
                self.window.console.line(line)
            self.summary.setText("DFT: " + " · ".join(lines[:3]))
            self.window.mark_done("dft")
            if out["relaxed"] is not None:
                self.window.add_geometry(f"{Path(folder).name} relajada (DFT)", out["relaxed"])

        self.window.runner.start("Resultado DFT", actions.dft_summary, folder,
                                 self.window.model, on_done=shown, on_error=self.window.error)

    def open_modes(self, path: str | None = None):
        """Modes computed before (modes.npz, or hessian.npy + its structure): their structure
        becomes the active geometry and the modes are shown, without recomputing."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Abrir modos", "", "Modos (modes.npz hessian.npy);;Todos (*)")
        if not path:
            return

        def loaded(out):
            name = f"{Path(path).parent.name} (modos guardados)"
            self.window.add_geometry(name, out["vibrations"].atoms)
            self.show_modes(out)

        self.window.runner.start("Abrir modos", actions.modes_from_file, path,
                                 on_done=loaded, on_error=self.window.error)

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
        self.report_button = QPushButton("Exportar reporte…")
        self.report_button.clicked.connect(self.export_report)
        self.open_result_button = QPushButton("Abrir resultado…")
        self.open_result_button.clicked.connect(self.open_result)
        for b in (self.raman_button, self.resonant_button, self.ir_button, self.export,
                  self.open_saved, self.report_button, self.open_result_button):
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

    def export_report(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Resultados para el reporte", "",
                "Resultados (report.json *.npz *.csv)")
        if not path:
            return
        try:
            written = actions.export_report(path)
        except (OSError, ValueError, KeyError) as error:
            self.window.error("No se pudo escribir el reporte", str(error))
            return
        figures = written.get("figuras")
        extra = f"; {len(figures)} figuras" if isinstance(figures, list) else \
            (f"; {figures}" if figures else "")
        self.summary.setText(f"Reporte: {written['html']} (datos en CSV{extra}).")

    def open_result(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Resultado o validación", "",
                "Resultados (*.json *.npz *.csv)")
        if not path:
            return
        try:
            page = actions.open_result(path)
        except (OSError, ValueError, KeyError) as error:
            self.window.error("No se pudo abrir", str(error))
            return
        self.summary.setText(f"Abierto en el navegador: {page}")

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


class RecipesPage(Page):
    """The recipes (coil, doped Raman, GPAW references…): settings, cores, launch, progress.

    Nothing runs in the window: a launch starts ``tbkit lanzar`` in its own session, so
    closing the window leaves it running, and the progress comes from the recipes'
    ``progreso*.json`` (as ``tbkit estado``)."""

    title = "Recetas"

    def __init__(self, window):
        super().__init__(window)
        self.recipes = actions.recipe_list()
        self.usage = {}
        self.process = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.recipe = QComboBox()
        for r in self.recipes:
            self.recipe.addItem(f"{r['name']}  ({r['count']} ajustes, {r['options']} opciones)",
                                r["name"])
        self.recipe.currentIndexChanged.connect(self.choose)
        form.addRow("Receta", self.recipe)
        self.step = QComboBox()
        self.step.currentIndexChanged.connect(self.choose_step)
        form.addRow("Paso", self.step)
        self.arguments = QLineEdit()
        self.arguments.setPlaceholderText("vacío casi siempre: las opciones van en su tabla")
        form.addRow("Otros argumentos", self.arguments)
        self.workdir = QLineEdit(str(Path.cwd()))
        form.addRow("Carpeta de trabajo", self.workdir)
        layout.addLayout(form)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        editable = (QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                    | QAbstractItemView.AnyKeyPressed)
        self.settings = table(["ajuste", "valor", "unidad", "grupo", "qué es"])
        self.settings.setEditTriggers(editable)
        self.options = table(["opción", "valor", "valores posibles", "qué es"])
        self.options.setEditTriggers(editable)
        self.option_rows = []
        self.tables = QTabWidget()
        self.tables.addTab(self.settings, "Ajustes de la receta")
        self.tables.addTab(self.options, "Opciones del paso")
        layout.addWidget(self.tables, stretch=2)
        row = QHBoxLayout()
        reset = QPushButton("Restaurar valores")
        reset.clicked.connect(self.choose)
        row.addWidget(reset)
        usage = QPushButton("Ayuda de la receta")
        usage.clicked.connect(self.show_usage)
        row.addWidget(usage)
        layout.addLayout(row)
        row = QHBoxLayout()
        geometry = QPushButton("Usar geometría activa")
        geometry.clicked.connect(self.use_geometry)
        row.addWidget(geometry)
        new = QPushButton("Nueva receta…")
        new.clicked.connect(self.new_recipe)
        row.addWidget(new)
        study = QPushButton("Nuevo estudio…")
        study.clicked.connect(self.new_study)
        row.addWidget(study)
        layout.addLayout(row)

        resources = QGroupBox("Recursos")
        rform = QFormLayout(resources)
        self.parts = _spin(1, 256, 1, 1, 0)
        rform.addRow("Partes en paralelo", self.parts)
        self.threads = _spin(1, 256, 1, 1, 0)
        rform.addRow("Hilos por parte", self.threads)
        self.mpi = _spin(1, 256, 1, 1, 0)
        rform.addRow("Procesos MPI por parte", self.mpi)
        self.retries = _spin(0, 10, 1, 1, 0)
        rform.addRow("Reintentos", self.retries)
        self.cores_label = QLabel(f"{actions.cores()} núcleos disponibles")
        rform.addRow(self.cores_label)
        for box in (self.parts, self.threads, self.mpi):
            box.valueChanged.connect(self.check_resources)
        layout.addWidget(resources)
        self.launch_button = QPushButton("Lanzar")
        self.launch_button.clicked.connect(self.launch)
        layout.addWidget(self.launch_button)

        progress = QGroupBox("Avance")
        playout = QVBoxLayout(progress)
        self.show_finished = QCheckBox("Mostrar terminados")
        self.show_finished.toggled.connect(self.refresh)
        refresh = QPushButton("Actualizar avance")
        refresh.clicked.connect(self.refresh)
        top = QHBoxLayout()
        top.addWidget(self.show_finished)
        top.addWidget(refresh)
        playout.addLayout(top)
        self.status = QLabel("—")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status.setStyleSheet("QLabel { font-family: monospace; }")
        playout.addWidget(self.status)
        layout.addWidget(progress, stretch=1)
        self.poll = QTimer(self)
        self.poll.timeout.connect(self.refresh)
        self.poll.start(10000)
        self.choose()

    def name(self) -> str:
        return self.recipe.currentData()

    def choose(self, *_):
        name = self.name()
        if name is None:
            return
        if name not in self.usage:
            self.usage[name] = actions.recipe_usage(name)
        info = self.usage[name]
        self.step.blockSignals(True)
        self.step.clear()
        self.step.addItems(info["steps"])
        self.step.blockSignals(False)
        summary = next(r["summary"] for r in self.recipes if r["name"] == name)
        self.summary.setText(summary + ("" if info["parts"] else
                                        " · Esta receta corre en una sola parte."))
        rows = actions.recipe_settings(name)
        self.settings.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(("name", "value", "unit", "group", "description")):
                item = QTableWidgetItem(row[key])
                if key != "value":
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if row["why"]:
                    item.setToolTip("por qué: " + row["why"])
                self.settings.setItem(r, c, item)
        self.settings.resizeColumnsToContents()
        self.tables.setTabText(0, f"Ajustes de la receta ({len(rows)})")
        self.choose_step()
        self.check_resources()

    def choose_step(self, *_):
        """The options of the recipe, plus those of the chosen step if it has its own."""
        info = self.usage.get(self.name())
        if info is None:
            return
        step = self.step.currentText()
        self.option_rows = info["options"].get("", []) + \
            (info["options"].get(step, []) if step else [])
        self.options.setRowCount(len(self.option_rows))
        for r, row in enumerate(self.option_rows):
            label = row["flag"] or row["dest"] + (" (obligatorio)" if row["required"] else "")
            possible = " | ".join(row["choices"]) or ("sí | no" if row["boolean"] else "")
            for c, text in enumerate((label, row["default"], possible, row["help"])):
                item = QTableWidgetItem(text)
                if c != 1:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.options.setItem(r, c, item)
        self.options.resizeColumnsToContents()
        self.tables.setTabText(1, f"Opciones del paso ({len(self.option_rows)})")
        self.check_resources()

    def edited(self) -> dict:
        return {self.settings.item(r, 0).text(): self.settings.item(r, 1).text()
                for r in range(self.settings.rowCount())}

    def extra_state(self) -> dict:
        """The edited tables (a project keeps them)."""
        return {"settings": self.edited(),
                "options": {row["dest"]: self.options.item(r, 1).text()
                            for r, row in enumerate(self.option_rows)}}

    def restore_extra(self, extra: dict) -> None:
        for r in range(self.settings.rowCount()):
            key = self.settings.item(r, 0).text()
            if key in extra.get("settings", {}):
                self.settings.item(r, 1).setText(extra["settings"][key])
        for r, row in enumerate(self.option_rows):
            if row["dest"] in extra.get("options", {}):
                self.options.item(r, 1).setText(extra["options"][row["dest"]])

    def check_resources(self, *_):
        info = self.usage.get(self.name(), {"parts": True, "parallel_steps": None})
        splits = actions.step_splits(info, self.step.currentText())
        if not splits and self.parts.value() != 1:
            self.parts.setValue(1)
        self.parts.setEnabled(splits)
        parts, threads, mpi = (int(b.value()) for b in (self.parts, self.threads, self.mpi))
        warnings = actions.launch_warnings(parts, threads, mpi, splits)
        total = actions.cores()
        busy = self.window.allocated_cores()
        mine = parts * threads * mpi
        left = total - busy - mine
        serial = "" if splits else ("Este paso corre en serie (un proceso): usa «Hilos por "
                                    "parte» para darle núcleos.\n")
        text = (f"{serial}Esta receta: {parts} × {threads} × {mpi} = {mine} núcleos · ya en "
                f"uso desde la ventana: {busy} · quedan {max(left, 0)} de {total}")
        if left < 0:
            text += f"\n⚠ faltan {-left} núcleos: lo que corre a la vez se estorbará"
        self.cores_label.setText(text + "".join(f"\n⚠ {w}" for w in warnings))
        return warnings

    def show_usage(self):
        QMessageBox.information(self, f"tbkit.recipes.{self.name()}",
                                self.usage.get(self.name(), {}).get("help", ""))

    def command(self) -> list[str]:
        import shlex

        overrides = actions.recipe_overrides(self.name(), self.edited())
        values = {row["dest"]: self.options.item(r, 1).text()
                  for r, row in enumerate(self.option_rows)}
        words = [self.step.currentText()] if self.step.currentText() else []
        words += actions.recipe_arguments(self.option_rows, values)
        arguments = shlex.join(words) + (" " + self.arguments.text() if self.arguments.text()
                                         else "")
        return actions.launch_command(self.name(), arguments, overrides,
                                      parts=int(self.parts.value()),
                                      threads=int(self.threads.value()),
                                      mpi=int(self.mpi.value()),
                                      retries=int(self.retries.value()),
                                      workdir=self.workdir.text().strip() or ".")

    def launch(self):
        try:
            command = self.command()
        except ValueError as error:
            self.window.error("Ajuste no válido", str(error))
            return
        info = self.usage.get(self.name(), {"parts": True})
        if int(self.parts.value()) > 1 and not info["parts"]:
            self.window.error("No se divide en partes", "Esta receta no tiene --part: usa 1 parte.")
            return
        workdir = self.workdir.text().strip() or "."
        import time

        log = Path(workdir) / "out" / "registros" / \
            f"{self.name()}_{time.strftime('%Y%m%d_%H%M%S')}_lanzador.log"
        self.process = actions.launch_recipe(command, workdir, log)
        cores = int(self.parts.value() * self.threads.value() * self.mpi.value())
        self.window.launched.append({"popen": self.process, "name": f"{self.name()} "
                                     f"{self.step.currentText()}".strip(), "cores": cores,
                                     "log": log, "started": time.time(),
                                     "command": " ".join(command[2:])})
        self.window.statusBar().showMessage(
            f"Lanzado (PID {self.process.pid}): {' '.join(command[2:])} · registro en {log}")
        self.check_resources()
        self.refresh()

    def use_geometry(self):
        """Save the active geometry as a file and put its path in the selected row (an
        option such as «structure», or a setting such as SOURCE)."""
        if self.window.atoms is None:
            self.window.error("Sin estructura", "Abre o crea primero una geometría.")
            return
        import re

        name = self.window.geometries[self.window.geometry_combo.currentIndex()][0]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "geometria"
        folder = Path(self.workdir.text().strip() or ".") / "geometrias"
        folder.mkdir(parents=True, exist_ok=True)
        path = (folder / f"{safe}.extxyz").resolve()
        self.window.atoms.write(path, format="extxyz")
        table_ = self.options if self.tables.currentIndex() == 1 else self.settings
        row = table_.currentRow()
        if row < 0:                       # nothing selected: the first structure-like option
            names = [table_.item(r, 0).text().lower() for r in range(table_.rowCount())]
            row = next((r for r, n in enumerate(names)
                        if any(k in n for k in ("structure", "source", "estructura"))), -1)
        if row < 0:
            self.window.statusBar().showMessage(f"Geometría guardada en {path} (elige la fila "
                                                "donde va y vuelve a pulsar)")
            return
        value = str(path) if table_ is self.options else f'"{path}"'
        table_.item(row, 1).setText(value)
        self.window.statusBar().showMessage(f"{table_.item(row, 0).text()} = {path}")

    def new_study(self):
        """A study (recipes/study.py) from geometries of the session: one base and its
        changes (dopants, groups), each embedded in the base."""
        if len(self.window.geometries) < 1:
            self.window.error("Sin geometrías", "Abre la base y las estructuras cambiadas.")
            return
        dialog = NewStudyDialog(self.window, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            path = dialog.write(Path(self.workdir.text().strip() or "."))
        except ValueError as error:
            self.window.error("Estudio no válido", str(error))
            return
        self.recipe.setCurrentIndex(self.recipe.findData("study"))
        self.choose()
        self.step.setCurrentText("todo")
        for r, row in enumerate(self.option_rows):
            if row["dest"] == "estudio":
                self.options.item(r, 1).setText(str(path))
        self.window.statusBar().showMessage(f"Estudio escrito en {path}: «Lanzar» lo corre "
                                            "(paso «todo», o paso a paso)")

    def new_recipe(self):
        if self.window.atoms is None:
            self.window.error("Sin estructura", "La receta parte de una geometría: abre una.")
            return
        dialog = NewRecipeDialog(self.window, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            path = dialog.write(Path(self.workdir.text().strip() or "."))
        except ValueError as error:
            self.window.error("Receta no válida", str(error))
            return
        index = self.recipe.findData("pipeline")
        self.recipe.setCurrentIndex(index)
        self.choose()
        for r, row in enumerate(self.option_rows):
            if row["dest"] == "receta":
                self.options.item(r, 1).setText(str(path))
        self.window.statusBar().showMessage(f"Receta escrita en {path}: «Lanzar» la corre")

    def refresh(self, *_):
        folder = Path(self.workdir.text().strip() or ".") / "out"
        lines = actions.recipe_status(folder, self.show_finished.isChecked()) \
            if folder.exists() else []
        self.status.setText("\n".join(lines) or "Ningún cálculo con contador en marcha en "
                            f"{folder}.")
        self.check_resources()


class CellDialog(QDialog):
    """Cell lengths and angles and the periodic axes of the active geometry."""

    def __init__(self, atoms, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Celda y ejes periódicos")
        self.atoms = atoms
        form = QFormLayout(self)
        lengths, angles = QHBoxLayout(), QHBoxLayout()
        par = atoms.cell.cellpar() if atoms.cell.rank == 3 else [0, 0, 0, 90, 90, 90]
        if atoms.cell.rank < 3:              # no cell: suggest the box around the atoms + vacuum
            span = atoms.positions.max(axis=0) - atoms.positions.min(axis=0)
            par = [*(span + 10.0), 90, 90, 90]
        self.lengths = [_spin(0.0, 1000.0, float(v), 0.1, 4, " Å") for v in par[:3]]
        self.angles = [_spin(1.0, 179.0, float(v), 1.0, 3, "°") for v in par[3:]]
        for box in self.lengths:
            lengths.addWidget(box)
        for box in self.angles:
            angles.addWidget(box)
        form.addRow("Longitudes a, b, c", lengths)
        form.addRow("Ángulos α, β, γ", angles)
        axes = QHBoxLayout()
        self.pbc = []
        for k, text in enumerate(("eje a", "eje b", "eje c")):
            box = QCheckBox(text)
            box.setChecked(bool(atoms.pbc[k]))
            self.pbc.append(box)
            axes.addWidget(box)
        detect = QPushButton("Detectar por enlaces")
        detect.clicked.connect(self.detect)
        axes.addWidget(detect)
        form.addRow("Periódica en", axes)
        self.keep = QCheckBox("Escalar las posiciones con la celda")
        form.addRow(self.keep)
        self.note = QLabel("La celda no se adivina: escríbela si el archivo no la trae. "
                           "«Detectar» marca los ejes por los que hay enlaces que cruzan.")
        self.note.setWordWrap(True)
        form.addRow(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def cellpar(self):
        return [b.value() for b in self.lengths] + [b.value() for b in self.angles]

    def detect(self):
        from ..structure_io import periodic_axes, with_cell

        try:
            probe = with_cell(self.atoms, self.cellpar(), [False] * 3,
                              scale_atoms=self.keep.isChecked())
        except ValueError as error:
            self.note.setText(f"⚠ {error}")
            return
        axes = periodic_axes(probe)
        for box, on in zip(self.pbc, axes, strict=True):
            box.setChecked(on)
        self.note.setText("Ejes con enlaces que cruzan la celda: "
                          + (", ".join("abc"[k] for k, on in enumerate(axes) if on) or "ninguno"))

    def result_atoms(self):
        from ..structure_io import with_cell

        return with_cell(self.atoms, self.cellpar(), [b.isChecked() for b in self.pbc],
                         scale_atoms=self.keep.isChecked())


#: Tasks a new recipe can chain (``tbkit.tasks.TASKS``), with a short description.
RECIPE_TASKS = {"relax": "relajar", "levels": "niveles y gap", "dos": "DOS",
                "bands": "bandas", "phonons": "modos en Γ", "raman": "Raman",
                "hubbard": "Hubbard (magnetismo)"}


class NewRecipeDialog(QDialog):
    """A recipe of one's own: a geometry of the session, a model and tasks in order."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("Nueva receta")
        form = QFormLayout(self)
        self.name = QLineEdit("mi_receta")
        form.addRow("Nombre", self.name)
        self.geometry = QComboBox()
        for name, _ in window.geometries:
            self.geometry.addItem(name)
        self.geometry.setCurrentIndex(window.geometry_combo.currentIndex())
        form.addRow("Geometría de partida", self.geometry)
        self.model = QComboBox()
        for key, (label, _) in actions.MODELS.items():
            self.model.addItem(label, key)
        index = self.model.findData(window.model_name)
        if index >= 0:
            self.model.setCurrentIndex(index)
        form.addRow("Modelo de la receta", self.model)
        self.checks = {}
        tasks_row = QHBoxLayout()
        for key, label in RECIPE_TASKS.items():
            box = QCheckBox(label)
            box.setChecked(key in ("relax", "levels", "dos"))
            self.checks[key] = box
            tasks_row.addWidget(box)
        form.addRow("Pasos, en este orden", tasks_row)
        self.kmesh = _spin(1, 64, 4, 1, 0)
        form.addRow("Malla k de la receta", self.kmesh)
        self.kT = _spin(0.001, 1.0, window.kT.value(), 0.005, 3, " eV")
        form.addRow("kT de la receta", self.kT)
        self.fmax = _spin(0.001, 1.0, 0.02, 0.005, 3, " eV/Å")
        form.addRow("fmax de la relajación", self.fmax)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def steps(self) -> list[dict]:
        common = {"kmesh": int(self.kmesh.value()), "kT": float(self.kT.value())}
        out = []
        for key, box in self.checks.items():
            if not box.isChecked():
                continue
            # bands takes a path, not a mesh or a temperature (tbkit.tasks.band_structure)
            settings = {} if key == "bands" else dict(common)
            if key == "relax":
                settings["fmax"] = float(self.fmax.value())
            out.append({"task": key, "settings": settings})
        return out

    def write(self, workdir: Path) -> Path:
        import re

        from ..recipes import pipeline

        steps = self.steps()
        if not steps:
            raise ValueError("Elige al menos un paso.")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name.text().strip()) or "mi_receta"
        _, atoms = self.window.geometries[self.geometry.currentIndex()]
        return pipeline.write(Path(workdir) / "recetas" / f"{name}.json", name, atoms,
                              self.model.currentData(), steps)


class DftDialog(QDialog):
    """What to ask the DFT (through carbonforge) about the active geometry."""

    def __init__(self, window, modes, parent=None):
        super().__init__(parent)
        self.window, self.modes = window, modes
        self.setWindowTitle("Validar con DFT")
        form = QFormLayout(self)
        default = Path(window.pages["Recetas"].workdir.text().strip() or ".") / "dft"
        name = window.geometries[window.geometry_combo.currentIndex()][0] \
            if window.geometries else "estructura"
        import re

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("_") or "estructura"
        self.folder = QLineEdit(str(default / safe))
        form.addRow("Carpeta de la solicitud", self.folder)
        self.code = QComboBox()
        for key, label in actions.DFT_CODES.items():
            self.code.addItem(label, key)
        form.addRow("Código DFT", self.code)
        self.relax = QCheckBox("Relajar con DFT primero")
        self.single = QCheckBox("Fuerzas en la geometría")
        self.single.setChecked(True)
        self.dos = QCheckBox("DOS y gap")
        self.use_modes = QCheckBox(f"Frecuencia de los {len(modes)} modos seleccionados")
        self.use_modes.setChecked(bool(modes))
        self.use_modes.setEnabled(bool(modes))
        for box in (self.relax, self.single, self.dos, self.use_modes):
            form.addRow(box)
        self.mpi = _spin(1, 256, min(4, actions.cores()), 1, 0)
        form.addRow("Procesos MPI", self.mpi)
        self.extra = QLineEdit()
        self.extra.setPlaceholderText('QE: {"pseudos": {"C": "C.pbe-n-kjpaw_psl.1.0.0.UPF"}, '
                                      '"pseudo_dir": "…"} · DFTB+: {"skf_dir": "…"}')
        form.addRow("Ajustes del código (JSON)", self.extra)
        self.run_now = QCheckBox("Correrla ahora con carbonforge")
        self.run_now.setChecked(True)
        form.addRow(self.run_now)
        note = QLabel("GPAW usa los ajustes de todas las referencias de tbkit (PBE, LCAO dzp, "
                      "h 0.2 Å, Fermi-Dirac 0.05 eV) y una malla k de ≥ 30 Å por eje periódico. "
                      "La carpeta se puede copiar a otra máquina o a un clúster (trabajo.slurm).")
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def write(self) -> str:
        import json

        settings = json.loads(self.extra.text()) if self.extra.text().strip() else None
        modes = self.modes if self.use_modes.isChecked() else []
        geometry = self.window.pages["Geometría y modos"]
        vibrations = geometry.result["vibrations"] if modes else None
        if vibrations is not None and len(vibrations.atoms) != len(self.window.atoms):
            raise ValueError("Los modos son de otra geometría: calcúlalos para la activa.")
        return actions.dft_request(self.folder.text().strip(), self.window.atoms,
                                   self.code.currentData(), relax=self.relax.isChecked(),
                                   single_point=self.single.isChecked(), dos=self.dos.isChecked(),
                                   vibrations=vibrations, modes=modes,
                                   mpi=int(self.mpi.value()), settings=settings)


class NewStudyDialog(QDialog):
    """Base and changes for a study, among the session's geometries."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("Nuevo estudio")
        form = QFormLayout(self)
        self.name = QLineEdit("mi_estudio")
        form.addRow("Nombre del estudio", self.name)
        self.base = QComboBox()
        for name, _ in window.geometries:
            self.base.addItem(name)
        form.addRow("Sistema base", self.base)
        self.changes = []
        box = QVBoxLayout()
        for name, _ in window.geometries:
            check = QCheckBox(name)
            self.changes.append(check)
            box.addWidget(check)
        form.addRow("Cambios sobre la base", box)
        self.model = QComboBox()
        for key, (label, _) in actions.MODELS.items():
            self.model.addItem(label, key)
        index = self.model.findData(window.model_name)
        self.model.setCurrentIndex(index if index >= 0 else self.model.findData("chn"))
        form.addRow("Modelo del estudio", self.model)
        self.radius = _spin(2.0, 20.0, 6.0, 0.5, 1, " Å")
        form.addRow("Radio del cambio", self.radius)
        note = QLabel("Cada cambio debe tener los átomos de la base primero y en su orden "
                      "(sustituir conserva el índice; los añadidos van al final): así sus "
                      "modos se calculan solo cerca del cambio. Los átomos cambiados se "
                      "detectan solos.")
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def write(self, workdir: Path) -> Path:
        import re

        base = self.window.geometries[self.base.currentIndex()]
        changes = [self.window.geometries[k] for k, c in enumerate(self.changes)
                   if c.isChecked() and k != self.base.currentIndex()]
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name.text().strip()) or "estudio"
        return actions.write_study(Path(workdir) / "estudios" / name, name, base, changes,
                                   model=self.model.currentData(),
                                   radius=float(self.radius.value()))


class ProcessesPage(Page):
    """Everything running: the window's calculation, the recipes launched from here and
    any recipe with a counter under the work folder; CPU and RAM of each, a stop button and
    the end of its log."""

    title = "Procesos"

    def __init__(self, window):
        super().__init__(window)
        layout = QVBoxLayout(self)
        self.summary = QLabel("—")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = table(["origen", "nombre", "PID", "estado", "tiempo", "CPU", "RAM",
                            "núcleos", "avance"])
        self.table.itemSelectionChanged.connect(self.show_log)
        layout.addWidget(self.table, stretch=2)
        row = QHBoxLayout()
        stop = QPushButton("Detener seleccionado")
        stop.clicked.connect(self.stop_selected)
        row.addWidget(stop)
        refresh = QPushButton("Actualizar lista")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        layout.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setStyleSheet("QPlainTextEdit { background: #1e1f22; color: #d7dae0; "
                               "font-family: monospace; }")
        layout.addWidget(self.log, stretch=1)
        self.rows: list[dict] = []
        self.poll = QTimer(self)
        self.poll.timeout.connect(self.refresh)
        self.poll.start(2000)

    def entries(self) -> list[dict]:
        """One dict per process: origin, name, pid, state, seconds, cores, progress, log."""
        import time

        from .resources import alive

        out = []
        runner = self.window.runner
        if runner.running:
            out.append({"origin": "ventana", "name": runner.label, "pid": runner.pid,
                        "state": "calculando", "seconds": runner.last_usage.get("seconds"),
                        "cores": runner.cores or actions.cores(),
                        "progress": self.window.job_label.text(), "kind": "runner"})
        for label, *_ in runner.queue:
            out.append({"origin": "ventana", "name": label, "pid": None, "state": "en cola",
                        "kind": "queued"})
        pids = set()
        for entry in self.window.launched:
            code = entry["popen"].poll()
            pids.add(entry["popen"].pid)
            out.append({"origin": "receta", "name": entry["name"], "pid": entry["popen"].pid,
                        "state": "corriendo" if code is None else f"terminó ({code})",
                        "seconds": time.time() - entry["started"] if code is None else None,
                        "cores": entry["cores"], "log": entry["log"], "kind": "launched"})
        folder = Path(self.window.pages["Recetas"].workdir.text().strip() or ".") / "out"
        if folder.exists():
            from ..progress import status_files

            for row in status_files(folder):
                pid = row.get("pid")
                if row["finished"] or not pid or not alive(pid) or pid in pids:
                    continue
                out.append({"origin": "contador", "name": row["label"], "pid": pid,
                            "state": "¿detenido?" if row["stale"] else "corriendo",
                            "seconds": time.time() - row["started"], "progress": row["line"],
                            "kind": "status"})
        return out

    def refresh(self, *_):
        from .resources import cores as available_cores
        from .resources import gb

        self.rows = self.entries()
        monitor = self.window.monitor
        rows = []
        for e in self.rows:
            usage = monitor.process(e["pid"]) if e.get("pid") else None
            cpu = f"{usage['cpu_percent']:.0f} %" if usage and usage["cpu_percent"] is not None \
                else "—"
            ram = gb(usage["rss_gb"]) if usage else "—"
            seconds = e.get("seconds")
            from ..progress import duration

            rows.append((e["origin"], e["name"], e.get("pid") or "—", e["state"],
                         duration(seconds) if seconds else "—", cpu, ram, e.get("cores") or "—",
                         e.get("progress") or ""))
        selected = self.table.currentRow()
        fill(self.table, rows)
        if 0 <= selected < len(rows):
            self.table.selectRow(selected)
        busy = self.window.allocated_cores()
        total = available_cores()
        m = getattr(self.window, "machine_usage", {}) or {}
        ram = (f" · RAM {m['ram_used_gb']:.1f} de {m['ram_total_gb']:.1f} GB "
               f"({m['ram_available_gb']:.1f} libres)") if m.get("ram_total_gb") else ""
        self.summary.setText(f"Núcleos: {total} · asignados desde la ventana: {busy} · libres: "
                             f"{max(total - busy, 0)}{ram}")

    def show_log(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.rows):
            return
        entry = self.rows[row]
        if entry["kind"] == "runner":
            text = "".join(self.window.runner.output_text)[-20000:]
        elif entry.get("log") and Path(entry["log"]).exists():
            data = Path(entry["log"]).read_bytes()[-20000:]
            text = data.decode("utf-8", errors="replace")
        else:
            text = entry.get("progress") or "(sin registro: solo su contador)"
        self.log.setPlainText(text)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def stop_selected(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.rows):
            return
        entry = self.rows[row]
        if entry["kind"] == "runner":
            self.window.runner.cancel()
        elif entry["kind"] == "queued":
            self.window.runner.queue = [q for q in self.window.runner.queue
                                        if q[0] != entry["name"]]
        elif entry.get("pid"):
            answer = QMessageBox.question(self, "tbkit", f"¿Detener {entry['name']} "
                                          f"(PID {entry['pid']}) y sus procesos hijos?")
            if answer != QMessageBox.Yes:
                return
            from .resources import terminate

            pids = terminate(entry["pid"])
            self.window.console.line(f"detenido {entry['name']}: procesos {pids}")
        self.refresh()


# --------------------------------------------------------------------------- snapshots

#: Controls a project does not restore by value: the geometry and model lists are restored
#: from the project itself, the mode-site list depends on the modes.
_NOT_SAVED = {"geometry_combo", "model_combo", "site", "other_model"}


def _value(widget):
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
        return widget.value()
    if isinstance(widget, QComboBox):
        data = widget.currentData()
        return {"text": widget.currentText(),
                "data": data if isinstance(data, (str, int, float)) else None}
    if isinstance(widget, QLineEdit):
        return widget.text()
    return None


def widget_values(owner) -> dict:
    """Every check box, spin box, combo box and text field that is an attribute of
    ``owner`` (lists of them too), by attribute name."""
    out = {}
    for name, widget in vars(owner).items():
        if name.startswith("_") or name in _NOT_SAVED:
            continue
        value = _value(widget)
        if value is not None:
            out[name] = value
        elif isinstance(widget, list) and widget and all(_value(w) is not None
                                                         for w in widget):
            out[name] = [_value(w) for w in widget]
    return out


def _set(widget, value):
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(value))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setValue(float(value))
    elif isinstance(widget, QComboBox):
        index = widget.findData(value.get("data")) if value.get("data") is not None else -1
        if index < 0:
            index = widget.findText(value.get("text", ""))
        if index >= 0:
            widget.setCurrentIndex(index)
        elif widget.isEditable():
            widget.setEditText(value.get("text", ""))
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value))


def set_widget_values(owner, values: dict) -> None:
    """The inverse of :func:`widget_values`; a control that no longer exists is skipped."""
    for name, value in values.items():
        widget = getattr(owner, name, None)
        if widget is None or name in _NOT_SAVED:
            continue
        try:
            if isinstance(widget, list):
                for w, v in zip(widget, value, strict=False):
                    _set(w, v)
            else:
                _set(widget, value)
        except (TypeError, ValueError, AttributeError):
            continue


def main(argv=None):
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    window = MainWindow()
    args = sys.argv[1:] if argv is None else argv[1:]
    if args:
        window.open_structure(args[0])
    window.show()
    return app.exec()
