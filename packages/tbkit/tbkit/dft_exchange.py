"""DFT requests and their results, ``nanocarbon-dft/1`` (docs/INTERCAMBIO_DFT.md of the workspace).

tbkit does not run DFT itself here: it writes a request folder (structure, tasks, settings) that
carbonforge prepares and runs (``carbonforge dft correr CARPETA``, its job queue, or a cluster
through ``trabajo.slurm``), and reads the result back to compare with the model:

* forces on the same geometry (RMS and largest difference, per element);
* the frequency of each tbkit mode pattern with the DFT forces (Rayleigh quotient from the ±
  displacements: exact if the pattern is a DFT eigenmode too, an upper bound otherwise) and
  its IR intensity from the DFT dipole (non-periodic axes only: along a periodic axis the
  dipole needs the Berry phase);
* the DOS and gap.

No import of carbonforge: the request is files. The same comparison the coil's GPAW check made
(``recipes/doped_gpaw``) for any structure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms

SCHEMA = "nanocarbon-dft/1"

#: The settings every GPAW reference of tbkit used (molecules, crystals, the coil): PBE, LCAO dzp,
#: h 0.2 Å, Fermi-Dirac 0.05 eV. k points: :func:`kpts_for`.
GPAW_DEFAULTS = {"xc": "PBE", "modo": "lcao", "base": "dzp", "h": 0.2,
                 "ensanchamiento_ev": 0.05, "espin": False, "carga": 0}


def kpts_for(atoms: Atoms, length: float = 30.0) -> list[int]:
    """k points per axis so that k × |a| ≥ ``length`` Å on periodic axes, 1 elsewhere. 30 Å was
    the coil's converged mesh (kz 2 with Lz 15 Å: forces within 6e-4 eV/Å of kz 4)."""
    lengths = atoms.cell.lengths()
    return [max(1, int(np.ceil(length / lengths[k]))) if atoms.pbc[k] and lengths[k] > 0 else 1
            for k in range(3)]


def write_request(folder: str | Path, atoms: Atoms, tasks: list[dict], code: str = "gpaw",
                  settings: dict | None = None, resources: dict | None = None,
                  vectors: np.ndarray | None = None, name: str | None = None,
                  why: str = "") -> Path:
    """Write ``solicitud.json`` + ``estructura.extxyz`` (+ ``vectores.npy``) in a new folder.

    ``tasks``: list of ``{"id", "tipo", ...}`` (punto_simple, relajar, desplazamientos,
    hessiana, dos). A folder that already holds a request is refused: results are never
    mixed between requests."""
    from . import __version__

    folder = Path(folder)
    if (folder / "solicitud.json").exists():
        raise FileExistsError(f"{folder} ya tiene una solicitud: usa otra carpeta.")
    folder.mkdir(parents=True, exist_ok=True)
    atoms.write(folder / "estructura.extxyz", format="extxyz")
    settings = {**(GPAW_DEFAULTS if code == "gpaw" else {}), **(settings or {})}
    settings.setdefault("kpts", kpts_for(atoms))
    if vectors is not None:
        np.save(folder / "vectores.npy", np.asarray(vectors, dtype=float))
        for task in tasks:
            if task.get("tipo") == "desplazamientos":
                task.setdefault("vectores", "vectores.npy")
    request = {"esquema": SCHEMA, "nombre": name or folder.name,
               "origen": {"programa": "tbkit", "version": __version__,
                          "fecha": time.strftime("%Y-%m-%d %H:%M"), "motivo": why},
               "estructura": "estructura.extxyz", "codigo": code, "ajustes": settings,
               "tareas": tasks, "recursos": resources or {"mpi": 1, "hilos": 1}}
    (folder / "solicitud.json").write_text(json.dumps(request, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    return folder


def mode_check_request(folder: str | Path, vibrations, modes: list[int], code: str = "gpaw",
                       relax_first: bool = False, amplitude: float = 0.01, **kwargs) -> Path:
    """A request that checks ``modes`` (indices of ``vibrations``) with DFT: a single point,
    then ± along each mode pattern L = e/√m. With ``relax_first``, the DFT relaxes the
    structure first and the patterns are applied there (same atom order)."""
    tasks = [{"id": "sp", "tipo": "punto_simple"}]
    source = "entrada"
    if relax_first:
        tasks.insert(0, {"id": "relax", "tipo": "relajar", "fmax": 0.05, "pasos": 300})
        source = "relax"
        tasks[1]["sobre"] = "relax"
    tasks.append({"id": "modos", "tipo": "desplazamientos", "amplitud": amplitude,
                  "sobre": source})
    folder = write_request(folder, vibrations.atoms, tasks, code, vectors=vibrations.modes[modes],
                           why="comprobar modos de tbkit con DFT", **kwargs)
    meta = {"modes": [int(m) for m in modes],
            "tb_frequencies_cm1": [float(vibrations.frequencies[m]) for m in modes],
            "masses_amu": vibrations.masses.tolist(), "source": vibrations.source}
    (folder / "tbkit_modos.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder


def read_result(folder: str | Path) -> dict:
    """``resultado.json`` checked (schema and units); unfinished results are returned too,
    with ``terminado`` False."""
    path = Path(folder) / "resultado.json"
    if not path.exists():
        raise FileNotFoundError(f"{folder}: todavía no hay resultado.json (¿se corrió?)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("esquema") != SCHEMA:
        raise ValueError(f"esquema {data.get('esquema')!r}; tbkit lee {SCHEMA!r}")
    units = data.get("unidades", {})
    if units.get("energia") != "eV" or units.get("fuerza") != "eV/Å":
        raise ValueError(f"unidades inesperadas {units}: el formato pide eV y eV/Å")
    return data


def compare_forces(atoms: Atoms, dft_forces, model, kmesh=4, kT: float = 0.05,
                   scc: bool | None = None) -> dict:
    """tbkit forces on the same geometry against the DFT ones."""
    from .calculator import TBCalculator

    probe = atoms.copy()
    probe.calc = TBCalculator(model, kpts=kmesh, kT=kT, scc=scc)
    tb = probe.get_forces()
    dft = np.asarray(dft_forces)
    diff = np.linalg.norm(tb - dft, axis=1)
    per_element = {}
    for symbol in sorted(set(atoms.get_chemical_symbols())):
        mask = np.array(atoms.get_chemical_symbols()) == symbol
        per_element[symbol] = float(np.sqrt(np.mean(diff[mask] ** 2)))
    return {"rms_eV_A": float(np.sqrt(np.mean(diff ** 2))), "max_eV_A": float(diff.max()),
            "max_atom": int(diff.argmax()), "rms_dft_eV_A":
            float(np.sqrt(np.mean(np.linalg.norm(dft, axis=1) ** 2))),
            "per_element_rms_eV_A": per_element}


def mode_frequencies(result: dict, folder: str | Path, pbc=(False, False, False)) -> list[dict]:
    """DFT frequency and IR intensity of each tbkit mode pattern of a ``mode_check_request``."""
    from ase.units import _amu, _e, _hbar, invcm

    from .infrared import DEBYE_PER_EA, KM_PER_MOL

    folder = Path(folder)
    meta = json.loads((folder / "tbkit_modos.json").read_text(encoding="utf-8"))
    vectors = np.load(folder / "vectores.npy")
    masses = np.asarray(meta["masses_amu"])
    task = result["tareas"].get("modos")
    if task is None:
        return []
    scale = _hbar * 1e10 / np.sqrt(_e * _amu) / invcm
    by_key = {(d["vector"], d["signo"]): d for d in task["desplazamientos"]}
    axes = [k for k in range(3) if not pbc[k]]
    rows = []
    for i, mode in enumerate(meta["modes"]):
        if (i, 1) not in by_key or (i, -1) not in by_key:
            continue
        plus, minus, s = by_key[(i, 1)], by_key[(i, -1)], task["escalas"][i]
        u = vectors[i]
        k_u = -(np.array(plus["fuerzas"]) - np.array(minus["fuerzas"])) / (2 * s)
        curvature, inertia = float(np.sum(u * k_u)), float(np.sum(masses[:, None] * u ** 2))
        dft = float(np.sign(curvature) * scale * np.sqrt(abs(curvature) / inertia))
        ir = None
        if plus.get("dipolo") is not None and axes:
            dmu = (np.array(plus["dipolo"]) - np.array(minus["dipolo"]))[axes] / (2 * s)
            ir = float(np.sum(dmu ** 2) * DEBYE_PER_EA ** 2 * KM_PER_MOL)
        tb = meta["tb_frequencies_cm1"][i]
        rows.append({"mode": mode, "tb_cm1": tb, "dft_cm1": dft,
                     "dft_over_tb": dft / tb if tb else None, "dft_ir_km_mol": ir})
    return rows


def summary(folder: str | Path, model=None, kmesh=4, kT: float = 0.05) -> dict:
    """Everything a request folder can say against tbkit: forces of every single point (with
    ``model``), mode frequencies, DOS/gap. Used by the window («Importar resultado DFT…»)."""
    from ase.io import read

    folder = Path(folder)
    result = read_result(folder)
    request = json.loads((folder / "solicitud.json").read_text(encoding="utf-8"))
    atoms = read(folder / request["estructura"])
    out = {"codigo": result["codigo"], "terminado": result["terminado"],
           "version_codigo": result.get("version_codigo"), "tareas": {}}
    # each task starts from the input or from an earlier relaxation («sobre»)
    geometries = {"entrada": atoms, None: atoms}
    for task in request["tareas"]:
        data = result["tareas"].get(task["id"])
        if data is None:
            out["tareas"][task["id"]] = {"estado": "pendiente"}
            continue
        entry = {"tipo": task["tipo"]}
        start = geometries.get(task.get("sobre", "entrada"), atoms)
        if task["tipo"] in ("punto_simple", "relajar"):
            entry["energia_eV"] = data["energia"]
            geometry = read(folder / data["estructura"]) if "estructura" in data else start
            if task["tipo"] == "relajar":
                geometries[task["id"]] = geometry
            if task["tipo"] == "relajar":
                entry.update(convergida=data.get("convergida"), pasos=data.get("pasos"),
                             desplazamiento_max_A=float(np.linalg.norm(
                                 geometry.positions - start.positions, axis=1).max()))
            if model is not None:
                entry["fuerzas_tbkit_vs_dft"] = compare_forces(geometry, data["fuerzas"],
                                                               model, kmesh, kT)
        elif task["tipo"] == "desplazamientos" and (folder / "tbkit_modos.json").exists():
            entry["modos"] = mode_frequencies(result, folder, tuple(atoms.pbc))
        elif task["tipo"] == "dos":
            entry["gap_eV"] = data.get("gap")
        out["tareas"][task["id"]] = entry
    return out
