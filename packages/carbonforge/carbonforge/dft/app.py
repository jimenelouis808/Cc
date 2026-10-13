"""Las tres páginas del frente de GPAW, en una ventana o embebidas.

Estructura, Receta y Resultados. Mismo contrato que vibspec, para que
quepan en la ventana única: ``frames`` para embeberlas, ``session`` para
recibir y publicar estructuras, ``queue`` para lanzar. **La ventana nunca
corre un cálculo en su proceso**: escribe el directorio y lo manda a la
cola, que es la única que sabe cancelarlo.

Lo que une las tres páginas es la selección de átomos. Elegir sitios en el
visor no es un adorno: es la entrada del LDOS y del hueco de core del XPS,
así que la selección viaja sola al parámetro ``sites`` del cálculo que la
necesite. Sin eso habría que copiar índices a mano de una pestaña a otra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ase import Atoms

from .kinds import KINDS
from .recipe import write as write_recipe
from .recipe_panel import RecipePanel
from .results import available_results, read_bands, read_dos, read_ldos, read_phonons
from .structure_panel import StructurePanel

__all__ = ["DftApp", "main"]

#: De dónde sale el nombre de esta página en la sesión de la ventana.
ORIGIN = "DFT (GPAW)"


class DftApp:
    """Compone, prepara y lee cálculos de GPAW."""

    def __init__(self, root, workdir: Optional[Path] = None, *,
                 frames: Optional[dict[str, Any]] = None,
                 session=None, queue=None, status_var=None,
                 on_submit: Optional[Callable[[Any], None]] = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.root = tk, ttk, root
        self.session = session
        self.queue = queue
        self.on_submit = on_submit
        self.workdir = Path(workdir or Path.cwd() / "calculos")
        self.atoms: Optional[Atoms] = None
        self.last_directory: Optional[Path] = None
        self.status_var = status_var or tk.StringVar(value="Listo.")
        self.var_workdir = tk.StringVar(value=str(self.workdir))
        self.var_results = tk.StringVar(value="")

        if frames is None:
            notebook = ttk.Notebook(root)
            frames = {}
            for name in ("Estructura", "Receta", "Resultados"):
                frame = ttk.Frame(notebook)
                notebook.add(frame, text=name)
                frames[name] = frame
            # La barra de estado antes del cuerpo que se expande: al reves
            # se queda en cero pixeles, que es como desaparecen.
            ttk.Label(root, textvariable=self.status_var, anchor="w",
                      foreground="#0a6").pack(side="bottom", fill="x")
            notebook.pack(fill="both", expand=True)

        self._build_structure(frames["Estructura"])
        self._build_recipe(frames["Receta"])
        self._build_results(frames["Resultados"])

    # ------------------------------------------------------------ estructura

    def _build_structure(self, parent) -> None:
        ttk = self.ttk
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=6, pady=6)
        ttk.Button(bar, text="Abrir estructura…",
                   command=self.open_structure).pack(side="left")
        self.take_button = ttk.Button(bar, text="Traer la estructura actual",
                                      command=self.take_current)
        self.take_button.pack(side="left", padx=4)
        ttk.Button(bar, text="Publicarla en la ventana",
                   command=self.publish).pack(side="left")

        self.structure_panel = StructurePanel(
            parent, on_selection=self._on_selection)
        self.structure_panel.pack(fill="both", expand=True)

    def set_structure(self, atoms: Atoms, origin: str = "") -> None:
        """Adopta ``atoms`` y la pasa a las dos páginas que la usan."""
        self.atoms = atoms
        self.structure_panel.show(atoms)
        self.recipe_panel.set_structure(atoms)
        self._status(f"Estructura de {origin or 'archivo'}: "
                     f"{len(atoms)} átomos.")

    def open_structure(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            filetypes=[("Estructuras", "*.xyz *.cif *.traj *.vasp *.pdb"),
                       ("Todos", "*.*")])
        if not path:
            return
        from ase.io import read

        try:
            self.set_structure(read(path), origin=Path(path).name)
        except Exception as problem:                     # noqa: BLE001
            self._status(f"No se pudo leer: {problem}")

    def take_current(self) -> None:
        """La estructura que la ventana tenga, si la hay."""
        current = self.session.current if self.session is not None else None
        if current is None:
            self._status("No hay estructura actual: constrúyela, impórtala "
                         "o ábrela aquí.")
            return
        self.set_structure(self.session.take(), origin=current.origin)

    def publish(self) -> None:
        """Deja esta estructura como la actual de la ventana."""
        if self.atoms is None or self.session is None:
            self._status("No hay estructura que publicar.")
            return
        self.session.publish(self.atoms, origin=ORIGIN)
        self._status("Publicada como la estructura actual de la ventana.")

    def _on_selection(self, indices: tuple[int, ...]) -> None:
        """Lleva los átomos elegidos al cálculo que los pida.

        El LDOS y el XPS se definen POR SITIO. Que la selección del visor
        no llegara sola al parámetro obligaría a copiar índices a mano
        entre pestañas, que es donde se cuelan los errores.
        """
        panel = getattr(self, "recipe_panel", None)
        if panel is None or "sites" not in panel._vars:
            return
        panel._vars["sites"].set(", ".join(str(i) for i in indices))

    # ---------------------------------------------------------------- receta

    def _build_recipe(self, parent) -> None:
        ttk = self.ttk
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=6, pady=6)
        ttk.Label(bar, text="Carpeta:").pack(side="left")
        ttk.Entry(bar, textvariable=self.var_workdir, width=38).pack(
            side="left", padx=4)
        ttk.Button(bar, text="…", width=3,
                   command=self._pick_workdir).pack(side="left")
        ttk.Button(bar, text="Preparar",
                   command=lambda: self.prepare(False)).pack(side="left",
                                                             padx=(8, 0))
        self.run_button = ttk.Button(bar, text="Preparar y correr",
                                     command=lambda: self.prepare(True))
        self.run_button.pack(side="left", padx=4)
        if self.queue is None:
            self.run_button.configure(state="disabled")

        self.recipe_panel = RecipePanel(parent)
        self.recipe_panel.pack(fill="both", expand=True)

    def _pick_workdir(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory()
        if path:
            self.var_workdir.set(path)

    def prepare(self, and_run: bool) -> Optional[Path]:
        """Escribe el directorio de la receta, y opcionalmente lo lanza."""
        if self.atoms is None:
            self._status("Primero trae o abre una estructura.")
            return None
        recipe = self.recipe_panel.recipe()
        if not recipe.steps:
            self._status("La receta está vacía: añade al menos un cálculo.")
            return None
        target = Path(self.var_workdir.get()) / recipe.name
        try:
            directory = write_recipe(recipe, self.atoms, target)
        except (ValueError, NotImplementedError) as problem:
            self._status(str(problem))
            return None
        self.last_directory = directory
        self.var_results.set(str(directory))
        self._status(f"Receta escrita en {directory}.")

        if and_run and self.queue is not None:
            job = self.queue.submit(directory)
            if self.on_submit is not None:
                self.on_submit(job)
            self._status(f"Lanzado: {directory.name}.")
        return directory

    # ------------------------------------------------------------ resultados

    def _build_results(self, parent) -> None:
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=6, pady=6)
        ttk.Entry(bar, textvariable=self.var_results, width=46).pack(
            side="left")
        ttk.Button(bar, text="…", width=3,
                   command=self._pick_results).pack(side="left", padx=2)
        ttk.Button(bar, text="Buscar",
                   command=self.scan_results).pack(side="left", padx=4)
        self.results_combo = ttk.Combobox(bar, state="readonly", width=18,
                                          values=[])
        self.results_combo.pack(side="left", padx=4)
        ttk.Button(bar, text="Dibujar",
                   command=self.draw_result).pack(side="left")

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(6.0, 4.0), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Elige un directorio y pulsa «Buscar»")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.txt_results = tk.Text(parent, height=5, wrap="word",
                                   relief="flat", font=("TkFixedFont", 8))
        self.txt_results.pack(fill="x")
        self.txt_results.configure(state="disabled")

    def _pick_results(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory()
        if path:
            self.var_results.set(path)

    def scan_results(self) -> dict[str, Path]:
        """Qué resultados hay ya en el directorio, y sólo esos."""
        directory = Path(self.var_results.get() or ".")
        found = available_results(directory)
        self.results_combo.configure(values=[KINDS[k].label for k in found])
        self._found = found
        if found:
            self.results_combo.set(KINDS[next(iter(found))].label)
            self._status(f"{len(found)} resultado(s) en {directory.name}.")
        else:
            self._status(f"Sin resultados en {directory}: el paso que los "
                         "escribe no ha corrido, o fallo.")
        return found

    def draw_result(self) -> None:
        """Dibuja el resultado elegido con el dibujo del propio paquete."""
        found = getattr(self, "_found", {})
        labels = {KINDS[k].label: k for k in found}
        chosen = labels.get(self.results_combo.get())
        if chosen is None:
            self._status("Nada que dibujar: pulsa «Buscar» primero.")
            return
        path = found[chosen]
        self.axes.clear()
        try:
            note = self._draw(chosen, path)
        except Exception as problem:                     # noqa: BLE001
            self._status(f"No se pudo dibujar: {problem}")
            return
        self.canvas.draw_idle()
        self._set_results_text(note)

    def _draw(self, chosen: str, path: Path) -> str:
        from ..results.bands import draw_bands_on_axes
        from ..results.dos import draw_dos_on_axes

        if chosen == "bands":
            bands = read_bands(path)
            draw_bands_on_axes(bands, self.axes,
                               reference=bands.fermi_energy)
            return (f"{bands.n_bands} bandas en {bands.n_kpoints} puntos k."
                    + (f" Nivel de Fermi {bands.fermi_energy:.3f} eV."
                       if bands.fermi_energy is not None else ""))
        if chosen == "dos":
            total, projections = read_dos(path)
            draw_dos_on_axes(total, self.axes,
                             reference=total.fermi_energy)
            for name, values in projections.items():
                self.axes.plot(total.energies, values, lw=0.9, label=name)
            if projections:
                self.axes.legend(fontsize=7)
            return self._dos_note(total, projections)
        if chosen == "ldos":
            sites, fermi = read_ldos(path)
            for site in sites:
                self.axes.plot(site.energies, site.total,
                               label=f"{site.index} {site.symbol}")
            if fermi is not None:
                self.axes.axvline(fermi, color="#888", lw=0.8, ls="--")
            self.axes.set_xlabel("E (eV)")
            self.axes.set_ylabel("LDOS (estados/eV)")
            self.axes.legend(fontsize=7)
            self.axes.set_title("DOS local por sitio")
            return (f"{len(sites)} sitio(s). Compararlos sólo vale si se "
                    "calcularon con el mismo ensanchamiento.")
        phonons = read_phonons(path)
        real = phonons.frequencies[phonons.frequencies > 1.0]
        self.axes.hist(real, bins=40, color="#36c")
        self.axes.set_xlabel("cm⁻¹")
        self.axes.set_ylabel("modos")
        self.axes.set_title("Fonones de Γ")
        if phonons.is_a_minimum:
            return f"{len(real)} modos reales; ninguno imaginario."
        return (f"{len(phonons.imaginary)} modo(s) IMAGINARIO(s): la "
                "geometría no es un mínimo de este modelo, y entonces "
                "ninguna frecuencia de la lista describe nada.")

    @staticmethod
    def _dos_note(total, projections: dict) -> str:
        note = ""
        if total.fermi_energy is not None:
            note = (f"DOS en el nivel de Fermi: "
                    f"{total.at_fermi():.3f} estados/eV. ")
        if projections:
            note += ("Las proyecciones NO suman el total: la proyección "
                     "sobre orbitales atómicos es incompleta.")
        return note

    def _set_results_text(self, text: str) -> None:
        self.txt_results.configure(state="normal")
        self.txt_results.delete("1.0", "end")
        self.txt_results.insert("1.0", text)
        self.txt_results.configure(state="disabled")

    # ---------------------------------------------------------------- varios

    def _status(self, text: str) -> None:
        self.status_var.set(text)


def main() -> None:                                      # pragma: no cover
    """Abre la ventana por su cuenta."""
    import tkinter as tk

    root = tk.Tk()
    root.title("carbonforge — DFT (GPAW)")
    root.geometry("1100x760")
    DftApp(root)
    root.mainloop()
