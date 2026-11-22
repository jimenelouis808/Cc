"""CO2/N2 selectivity of the structures already studied: a dry flue gas
(15 % CO2 in N2) at 0.1 MPa and 298 K, by mixture GCMC
(:mod:`carbonforge.adsorption.mixture`).

Structures: the 6.6 A slit and the two cages of the pure-CO2 study
(neutral Steele walls, as there), and the five functionalised slits on
the adsorption force field (published group charges). One json per
structure in ``selectividad/``; a finished one is skipped.

    python selectividad.py corre <name>
    python selectividad.py tabla
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from carbonforge.adsorption.co2 import wall_parameters
from carbonforge.adsorption.mixture import run_mixture_gcmc

import funcional_co2
from isoterma_co2 import framework as pure_framework

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "selectividad"
TEMPERATURE = 298.0
PRESSURE = 0.1
FRACTIONS = (0.15, 0.85)
#: Longer than the pure isotherms: N2 adsorbs an order of magnitude less,
#: and the selectivity's error is set by the minority count.
EQUILIBRATION, PRODUCTION = 200_000, 600_000

PURE = ("rendija-6.6", "super-cube", "super-tetrahedron")
FUNCTIONAL = tuple(f"funcional-{n}" for n in funcional_co2.VARIANTS)
#: Crystalline (knee-route) superstructures from nanocarbon_lab, in
#: ``cristalinas/`` as extxyz. Schwarz D's channel (6.72 A) is the 6.6 A
#: slit's width; the others bracket it.
CRYSTALLINE = tuple(f"cristalina-{p.stem}" for p in
                    sorted((HERE / "cristalinas").glob("*.extxyz")))


def structure(name: str):
    """(atoms, wall) for a structure name."""
    if name in PURE:
        return pure_framework(name), None
    if name.startswith("cristalina-"):
        from ase.io import read
        from llenado_super import embed
        path = HERE / "cristalinas" / f"{name.split('-', 1)[1]}.extxyz"
        return embed(read(str(path)))[0], None
    atoms = funcional_co2.slit(name.split("-", 1)[1])
    return atoms, wall_parameters(atoms, forcefield="adsorption")


def run(name: str) -> None:
    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / f"{name}.json"
    if path.exists():
        return
    atoms, wall = structure(name)
    started = time.time()
    result = run_mixture_gcmc(atoms, TEMPERATURE, PRESSURE, FRACTIONS,
                              equilibration=EQUILIBRATION,
                              production=PRODUCTION, seed=0, wall=wall)
    record = asdict(result) | {"segundos": time.time() - started,
                               "formula": atoms.get_chemical_formula()}
    path.write_text(json.dumps(record, indent=1))
    print(f"{name}: S = {result.selectivity:.1f}", flush=True)


def table() -> None:
    print(f"{'estructura':24s} {'CO2 mol/kg':>10s} {'N2 mol/kg':>10s} "
          f"{'x_CO2':>6s} {'S CO2/N2':>14s}")
    for name in PURE + FUNCTIONAL + CRYSTALLINE:
        path = FOLDER / f"{name}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        print(f"{name:24s} {r['mol_per_kg'][0]:10.3f} {r['mol_per_kg'][1]:10.3f} "
              f"{r['adsorbed_fractions'][0]:6.3f} {r['selectivity']:7.1f} ± "
              f"{r['selectivity_error']:4.1f}  avisos {len(r['warnings'])}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "corre":
        run(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "tabla":
        table()
    else:
        print(__doc__)
        sys.exit(1)
