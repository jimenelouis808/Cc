"""GPAW reference data for transition metals with d orbitals: molybdenum first.

Run (needs GPAW; resumable, one file per molecule)::

    python -m tbkit.recipes.tm_references Mo gpaw_mo.json --workers 4

Same protocol as :mod:`tbkit.recipes.bsp_references` (PBE, LCAO dzp, spin paired; each
training molecule relaxed, then four random displacements and two uniform scalings; test
molecules relaxed only). The molecules are closed shell, so a spin-paired reference is the
right one: d⁰ Mo(VI) (oxo, hydroxo, methyl, amido, thiolato, sulfido, hydrido) and the d⁶
hexacarbonyl, with the Mo₂ dimer for Mo-Mo. They cover every pair the parameter set must
define on top of xu_chnos (Mo-H, Mo-C, Mo-N, Mo-O, Mo-S, Mo-Mo).

Starting geometries are built here around the metal (RDKit does not place transition-metal
ligands reliably): ideal polyhedra, ligands at typical bond lengths, a small fixed rattle
(seed per molecule) so that GPAW does not sit on a symmetric saddle. GPAW relaxes them.

Iron is magnetic in most of what matters for carbon (Fe-N4 sites, clusters, adatoms): its
references must be spin polarised and its set fitted with the spin SCC, a separate stage.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms

#: Typical bond lengths, Å (starting values only: GPAW relaxes).
BOND = {"O": 1.71, "OH": 1.90, "CH3": 2.12, "NH2": 1.95, "SH": 2.32, "S": 2.14, "H": 1.70,
        "CO": 2.06}

_TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
_OCTA = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def _perpendicular(u):
    v = np.cross(u, [0.0, 0.0, 1.0])
    if np.linalg.norm(v) < 1e-6:
        v = np.cross(u, [0.0, 1.0, 0.0])
    return v / np.linalg.norm(v)


def _ligand(kind: str, direction) -> tuple[str, list]:
    """Symbols and positions of a ligand bonded along ``direction`` from a metal at 0."""
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    x = BOND[kind] * u
    w = _perpendicular(u)
    w2 = np.cross(u, w)
    if kind in ("O", "S", "H"):
        return kind, [x]
    if kind == "OH":
        return "OH", [x, x + 0.97 * (0.33 * u + 0.94 * w)]
    if kind == "SH":
        return "SH", [x, x + 1.34 * (0.25 * u + 0.97 * w)]
    if kind == "NH2":
        return "NHH", [x, x + 1.01 * (0.5 * u + 0.87 * w), x + 1.01 * (0.5 * u - 0.87 * w)]
    if kind == "CO":
        return "CO", [x, x + 1.15 * u]
    if kind == "CH3":
        hs = [x + 1.09 * (0.33 * u + 0.94 * (np.cos(a) * w + np.sin(a) * w2))
              for a in (0.0, 2.094, 4.189)]
        return "CHHH", [x] + hs
    raise KeyError(kind)


def complex_of(metal: str, ligands: list[str], polyhedron: np.ndarray, seed: int) -> Atoms:
    symbols, positions = metal, [np.zeros(3)]
    for kind, direction in zip(ligands, polyhedron, strict=False):
        s, p = _ligand(kind, direction)
        symbols += s
        positions += p
    atoms = Atoms(symbols, positions=positions)
    atoms.rattle(0.02, seed=seed)
    return atoms


#: name -> (ligands, polyhedron); per element: (training, test)
MOLECULES = {
    "Mo": ({"MoO3": (["O", "O", "O"], _TETRA), "MoO2(OH)2": (["O", "O", "OH", "OH"], _TETRA),
            "MoO2Me2": (["O", "O", "CH3", "CH3"], _TETRA),
            "MoO2(NH2)2": (["O", "O", "NH2", "NH2"], _TETRA),
            "MoO2(SH)2": (["O", "O", "SH", "SH"], _TETRA), "MoS3": (["S", "S", "S"], _TETRA),
            "MoOS2": (["O", "S", "S"], _TETRA), "MoH6": (["H"] * 6, _OCTA),
            "MoMe6": (["CH3"] * 6, _OCTA), "Mo(CO)6": (["CO"] * 6, _OCTA), "Mo2": None},
           {"MoO2(SH)Me": (["O", "O", "SH", "CH3"], _TETRA),
            "MoS2(OH)2": (["S", "S", "OH", "OH"], _TETRA)}),
}


def structure(name: str) -> Atoms:
    for element, (train, test) in MOLECULES.items():
        table = {**train, **test}
        if name not in table:
            continue
        if name == f"{element}2":
            return Atoms(f"{element}2", positions=[[0, 0, 0], [0, 0, 1.95]])
        ligands, polyhedron = table[name]
        seed = sum(map(ord, name))
        return complex_of(element, ligands, polyhedron, seed)
    raise KeyError(name)


def _one(args):
    name, role, index, settings = args
    from tbkit.references import generate_gpaw

    kwargs = {} if role == "train" else {"n_random": 0, "scales": ()}
    items = generate_gpaw({name: structure(name)}, settings, role=role, seed=3000 + 100 * index,
                          **kwargs)
    return name, [item.to_dict() for item in items]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("element", choices=sorted(MOLECULES))
    parser.add_argument("out", type=Path,
                        help="archivo JSON de referencias que se escribe (los cálculos parciales "
                             "van a ARCHIVO.parts/, reanudable)")
    parser.add_argument("--workers", type=int, default=4,
                        help="procesos GPAW en paralelo (cada uno, un cálculo; vigila la memoria)")
    args = parser.parse_args(argv)
    from tbkit.references import GPAW_DEFAULTS, gpaw_settings_record

    settings = dict(GPAW_DEFAULTS)
    train, test = MOLECULES[args.element]
    jobs = [(n, "train", k, settings) for k, n in enumerate(train)]
    jobs += [(n, "test", 50 + k, settings) for k, n in enumerate(test)]
    jobs.sort(key=lambda job: -len(structure(job[0])))
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (parts / f"{job[0]}.json").exists()]
    from ..progress import Progress

    bar = Progress(len(jobs), f"referencias GPAW {args.element}", status=parts / "progreso.json",
                   done=len(jobs) - len(todo))
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_one, job) for job in todo]
        for future in as_completed(futures):
            name, result = future.result()
            (parts / f"{name}.json").write_text(json.dumps(result))
            print(f"{name}: {len(result)} estructuras", flush=True)
            bar.step()
    structures = []
    for job in jobs:
        structures += json.loads((parts / f"{job[0]}.json").read_text())
    data = {"settings": gpaw_settings_record(settings), "structures": structures}
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(structures)} estructuras en {args.out}")


if __name__ == "__main__":
    main()
