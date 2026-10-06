"""Recetas: una lista de cálculos encadenados, escrita como un directorio.

Una receta es lo que el usuario edita y guarda: "relaja, saca el estado
base, y de ahí bandas y DOS". :func:`write` la convierte en un directorio
de trabajo con la estructura, la receta en JSON, **un script de Python por
paso** y el manifiesto que ya sabe ejecutar
:mod:`carbonforge.jobs.run`.

**Por qué scripts y no llamadas.** El script generado es texto: se lee, se
corrige a mano y se vuelve a lanzar sin tocar carbonforge. Y sobre todo,
corre en OTRO intérprete. Eso no es un rodeo: GPAW 24.6 exige ``numpy<2``
y el lector de DOS de carbonforge necesita numpy 2, así que los dos no
caben en el mismo entorno. El paso declara el programa ``gpaw-python`` y
:func:`carbonforge.jobs.run.find_program` lo resuelve con la variable
``CARBONFORGE_GPAW_PYTHON``, que apunta al intérprete donde viva GPAW.

Lo que una receta **no** hace es adivinar: si un paso necesita el estado
base y no está antes, se dice al escribirla y no al correrla.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ase import Atoms
from ase.io import write as ase_write

from ..jobs.manifest import JobManifest, Step
from .kinds import KINDS, kind, validate_chain

__all__ = [
    "GPAW_PROGRAM",
    "Recipe",
    "RecipeStep",
    "script_for",
    "write",
]

#: Nombre con que el manifiesto pide el intérprete de GPAW. Se resuelve con
#: ``CARBONFORGE_GPAW_PYTHON``; sin ella, el adaptador dice que falta y
#: dónde ponerlo, en vez de correr con el intérprete equivocado.
GPAW_PROGRAM = "gpaw-python"

#: Fichero con la estructura de partida.
STRUCTURE = "structure.xyz"

#: La receta tal cual se editó.
RECIPE = "recipe.json"


@dataclass
class RecipeStep:
    """Un cálculo de la receta, con los valores que el usuario fijó."""

    kind: str
    values: dict[str, Any] = field(default_factory=dict)

    def filled(self) -> dict[str, Any]:
        """Los valores, con los que falten puestos por defecto."""
        return kind(self.kind).defaults() | dict(self.values)


@dataclass
class Recipe:
    """Cálculos encadenados sobre una misma estructura."""

    name: str
    steps: list[RecipeStep] = field(default_factory=list)

    def check(self, atoms: Optional[Atoms] = None) -> list[str]:
        """Todo lo que impide correrla, de la cadena y de cada paso."""
        problems = list(validate_chain([s.kind for s in self.steps]))
        for position, step in enumerate(self.steps, start=1):
            if step.kind not in KINDS:
                continue
            for message in kind(step.kind).check(step.filled(), atoms):
                problems.append(f"Paso {position} ({step.kind}): {message}")
        return problems

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Recipe:
        data = json.loads(text)
        return cls(name=data["name"],
                   steps=[RecipeStep(**s) for s in data["steps"]])


# ------------------------------------------------------------------ scripts

_HEADER = '''"""{title}

Generado por carbonforge. Es texto: corrígelo y vuelve a lanzarlo.
Corre en el entorno donde esté GPAW, que no es el de carbonforge.
"""
from pathlib import Path

import numpy as np
from ase.io import read, write
from gpaw import GPAW

HERE = Path(__file__).resolve().parent
'''


def _calculator(values: dict[str, Any], extra: str = "") -> str:
    """El trozo que construye GPAW a partir de los ajustes compartidos."""
    mode = values["mode"]
    if mode == "pw":
        from_mode = f"PW({float(values['ecut']):g})"
        extra_import = "from gpaw import PW\n"
    elif mode == "lcao":
        from_mode = "'lcao'"
        extra_import = ""
    else:
        from_mode = "'fd'"
        extra_import = ""
    kpts = tuple(int(x) for x in str(values["kpts"]).replace(" ", "").split(","))
    lines = [extra_import, "calc = GPAW("]
    lines.append(f"    mode={from_mode},")
    if mode == "lcao":
        lines.append(f"    basis={values['basis']!r},")
    if mode in ("lcao", "fd"):
        lines.append(f"    h={float(values['h']):g},")
    lines.extend([
        f"    xc={values['xc']!r},",
        f"    kpts={kpts},",
        f"    spinpol={bool(values['spinpol'])},",
        ("    occupations={'name': 'fermi-dirac', "
         f"'width': {float(values['occupations_width']):g}}},"),
        f"    convergence={{'density': {float(values['convergence_density']):g}}},",
        "    txt=str(HERE / 'gpaw.txt'),",
    ])
    if extra:
        lines.append(extra)
    lines.append(")")
    return "\n".join(lines)


def _script_relax(values: dict[str, Any], source: str) -> str:
    return _HEADER.format(title="Relajación de geometría.") + f'''
atoms = read(str(HERE / {source!r}))
{_calculator(values)}
atoms.calc = calc

from ase.optimize import {values["optimizer"]}

optimiser = {values["optimizer"]}(atoms, logfile=str(HERE / "relax.log"),
                 trajectory=str(HERE / "relaxed.traj"))
optimiser.run(fmax={float(values["fmax"]):g}, steps={int(values["steps"])})

forces = np.abs(atoms.get_forces()).max()
write(str(HERE / "relaxed.xyz"), atoms, format="extxyz")
print(f"|F|max final = {{forces:.4f}} eV/A")
if forces > {float(values["fmax"]):g}:
    raise SystemExit(
        f"No convergio: |F|max {{forces:.4f}} sigue por encima de "
        f"{float(values["fmax"]):g} eV/A. Lo que hay NO es un minimo.")
'''


def _script_scf(values: dict[str, Any], source: str) -> str:
    nbands = values.get("nbands", "auto")
    bands = "" if str(nbands).strip() in ("", "auto") else \
        f"    nbands={int(nbands)},"
    return _HEADER.format(title="Estado base: densidad autoconsistente.") + f'''
atoms = read(str(HERE / {source!r}))
{_calculator(values, bands)}
atoms.calc = calc

energy = atoms.get_potential_energy()
calc.write(str(HERE / "gs.gpw"), mode="all")
print(f"E = {{energy:.6f}} eV")
'''


def _script_bands(values: dict[str, Any], source: str) -> str:
    path = values["path"]
    chosen = ("atoms.cell.bandpath(npoints=%d)" % int(values["npoints"])
              if str(path).strip().lower() == "auto"
              else "atoms.cell.bandpath(%r, npoints=%d)"
                   % (path, int(values["npoints"])))
    return _HEADER.format(title="Estructura de bandas a densidad fija.") + f'''
import json

base = GPAW(str(HERE / "gs.gpw"))
atoms = base.get_atoms()
path = {chosen}

calc = base.fixed_density(
    bandpath=path,
    nbands=base.get_number_of_bands() + {int(values["nbands_extra"])},
    symmetry="off",
    txt=str(HERE / "bands.txt"))
structure = calc.band_structure()
structure.write(str(HERE / "bands.json"))
print("camino:", path.path, "| puntos:", len(path.kpts))
'''


def _script_dos(values: dict[str, Any], source: str) -> str:
    projection = values["projection"]
    return _HEADER.format(title="Densidad de estados.") + f'''
import json

calc = GPAW(str(HERE / "gs.gpw"))
atoms = calc.get_atoms()
fermi = calc.get_fermi_level()

from ase.dft.dos import DOS

dos = DOS(calc, width={float(values["width"]):g},
          npts={int(values["npts"])})
energies = dos.get_energies()
out = {{"fermi": float(fermi), "energies": energies.tolist(),
       "total": dos.get_dos().tolist(), "projection": {projection!r}}}

if {projection!r} == "elemento":
    out["por_elemento"] = {{}}
    for symbol in sorted(set(atoms.get_chemical_symbols())):
        indices = [i for i, s in enumerate(atoms.get_chemical_symbols())
                   if s == symbol]
        total = np.zeros_like(energies)
        for index in indices:
            for momentum in "spd":
                e, weights = calc.get_orbital_ldos(
                    a=index, angular=momentum,
                    width={float(values["width"]):g}, npts={int(values["npts"])})
                total += weights
        out["por_elemento"][symbol] = total.tolist()
elif {projection!r} == "orbital":
    out["por_orbital"] = {{}}
    for momentum in "spd":
        total = np.zeros_like(energies)
        for index in range(len(atoms)):
            e, weights = calc.get_orbital_ldos(
                a=index, angular=momentum,
                width={float(values["width"]):g}, npts={int(values["npts"])})
            total += weights
        out["por_orbital"][momentum] = total.tolist()

(HERE / "dos.json").write_text(json.dumps(out), encoding="utf-8")
print("DOS escrito; nivel de Fermi", round(float(fermi), 4), "eV")
'''


def _script_ldos(values: dict[str, Any], source: str) -> str:
    sites = [int(i) for i in values.get("sites", ())]
    orbitals = values["orbitals"]
    which = "spd" if orbitals == "todos" else orbitals
    return _HEADER.format(title="DOS local sobre los átomos elegidos.") + f'''
import json

calc = GPAW(str(HERE / "gs.gpw"))
atoms = calc.get_atoms()
symbols = atoms.get_chemical_symbols()
sites = {sites!r}

out = {{"fermi": float(calc.get_fermi_level()), "sitios": {{}}}}
for index in sites:
    por_orbital = {{}}
    for momentum in {which!r}:
        energies, weights = calc.get_orbital_ldos(
            a=index, angular=momentum,
            width={float(values["width"]):g}, npts=801)
        por_orbital[momentum] = weights.tolist()
        out["energies"] = energies.tolist()
    out["sitios"][str(index)] = {{"simbolo": symbols[index],
                                 "orbitales": por_orbital}}

(HERE / "ldos.json").write_text(json.dumps(out), encoding="utf-8")
print("LDOS escrito para", len(sites), "sitio(s)")
'''


def _script_phonons(values: dict[str, Any], source: str) -> str:
    supercell = tuple(int(x) for x in
                      str(values["supercell"]).replace(" ", "").split(","))
    return _HEADER.format(title="Fonones por desplazamientos finitos.") + f'''
import json

from ase.phonons import Phonons

atoms = read(str(HERE / {source!r}))
{_calculator(values)}

phonons = Phonons(atoms, calc, supercell={supercell!r},
                  delta={float(values["delta"]):g},
                  name=str(HERE / "phonon"))
phonons.run()
phonons.read(acoustic=True)
frequencies = phonons.band_structure([[0, 0, 0]])[0] * 8065.54  # eV -> cm-1

(HERE / "phonons.json").write_text(
    json.dumps({{"gamma_cm1": [float(f) for f in frequencies]}}),
    encoding="utf-8")
negative = [f for f in frequencies if f < -1.0]
print(len(frequencies), "modos en Gamma;", len(negative), "imaginarios")
if negative:
    print("AVISO: hay modos imaginarios. La geometria no es un minimo "
          "de ESTE modelo, o la malla es demasiado gruesa.")
'''


#: Qué script escribe cada tipo. Los que faltan se dicen por su nombre al
#: pedirlos, en vez de emitir código que no se ha podido comprobar.
_WRITERS = {
    "relax": _script_relax,
    "scf": _script_scf,
    "bands": _script_bands,
    "dos": _script_dos,
    "ldos": _script_ldos,
    "phonons": _script_phonons,
}


def script_for(step: RecipeStep, source: str = STRUCTURE) -> str:
    """El script de Python que ejecuta ``step``.

    ``source`` es de dónde lee la estructura: el fichero de partida, o el
    ``relaxed.xyz`` que dejó un paso de relajación anterior.
    """
    try:
        writer = _WRITERS[step.kind]
    except KeyError:
        raise NotImplementedError(
            f"Todavía no se genera el script de '{step.kind}'. "
            "XPS necesita confirmar contra tu instalación cómo se llama el "
            "setup con hueco de core, y el Raman resonante arrastra LrTDDFT; "
            "los dos se escriben a mano de momento."
        ) from None
    return writer(step.filled(), source)


# ----------------------------------------------------------------- escribir


def write(recipe: Recipe, atoms: Atoms, directory: str | Path,
          nprocs: int = 1) -> Path:
    """Deja la receta lista para correr en ``directory``.

    Escribe la estructura, la receta, un script por paso y el manifiesto.
    Se niega si la receta no cierra: lanzar una cadena que no puede
    terminar sólo consume cola.
    """
    problems = recipe.check(atoms)
    if problems:
        raise ValueError("La receta no se puede correr:\n- "
                         + "\n- ".join(problems))

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    clean = atoms.copy()
    clean.info = {}
    ase_write(str(directory / STRUCTURE), clean, format="extxyz")
    (directory / RECIPE).write_text(recipe.to_json(), encoding="utf-8")

    steps, source = [], STRUCTURE
    for position, step in enumerate(recipe.steps, start=1):
        name = f"{position:02d}_{step.kind}"
        (directory / f"{name}.py").write_text(
            script_for(step, source), encoding="utf-8")
        steps.append(Step(
            name=step.kind, program=GPAW_PROGRAM, input=f"{name}.py",
            output=f"{name}.out", parallel=True))
        if step.kind == "relax":
            # Lo que sigue parte de la geometría relajada, no de la de entrada.
            source = "relaxed.xyz"

    manifest = JobManifest(engine="gpaw", title=recipe.name, steps=steps)
    manifest.save(directory)
    return directory
