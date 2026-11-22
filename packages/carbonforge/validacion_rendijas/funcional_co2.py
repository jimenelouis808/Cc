"""CO2 on a functionalised 6.6 A slit: does chemistry on the wall pay?

    uv run python validacion_rendijas/funcional_co2.py henry
    uv run python validacion_rendijas/funcional_co2.py corre OH
    uv run python validacion_rendijas/funcional_co2.py tabla

The 6.6 A slit is the CO2 winner of `henry_co2.py`. Here its sheet
carries, in turn: nothing; 5 % graphitic N (substituted, in plane); four
hydroxyls; three carboxyls; four amines -- per 240-carbon sheet, on the
side that faces the pore, so every pore has one decorated wall. The
groups are basal (sp3 anchors), which is what oxidised or aminated
carbon looks like.

Every atom is typed by `carbonforge.forcefields` and carries that
table's LJ and charge; the wall is neutralised over its sp2 carbons
(`co2.wall_parameters`). **Those charges are representative, not
established** -- the parameter table marks each one so -- and this
study inherits that: it says which way and roughly how much a group
moves CO2 uptake, not the number a DFT-charged model would give.

A grafted group also NARROWS the pore it sticks into (OH 1.7 A, COOH
2.7 A, NH2 1.8 A above the sheet), so geometry and chemistry change
together. `henry` reports K_H and q_st with and without the charges
on the same structure, which separates the two.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from carbonforge.adsorption.co2 import henry_co2, run_co2_gcmc, wall_parameters
from carbonforge.adsorption.porosity import analyse_porosity
from carbonforge.builders import build_graphene_supercell
from carbonforge.dopants.substitutional import dope_random
from carbonforge.functionalization.attach import functionalize_random

from llenado import MIN_EDGE

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "isotermas_co2"
WIDTH = 6.6
TEMPERATURE = 298.0
#: Five, not the seven of the pristine study: with every wall atom
#: charged (the neutralising shift touches all of them) a move costs
#: about 1 ms, and five variants at seven points is most of a day.
PRESSURES = (1e-3, 1e-2, 3e-2, 0.1, 1.0)
#: Wall force field: "adsorption" (published, neutral group charges; the
#: default) or "electrode" (the EDLC table, whose carboxyl is a
#: carboxylate). Set with CAMPO=electrode. The first study ran on the
#: electrode table and its files carry no suffix; adsorption runs are
#: suffixed "-ads". Pristine and N5 are the same in both (no groups).
FORCEFIELD = os.environ.get("CAMPO", "adsorption")
SUFFIX = "" if FORCEFIELD == "electrode" else "-ads"

#: Variant -> how its sheet is made. Seeds fixed; N's seed is the first
#: that puts no two N atoms side by side, an environment the typing
#: refuses (it has no parameters for it).
VARIANTS = {
    "pristina": lambda a: a,
    "N5": lambda a: dope_random(a, "N", 0.05, seed=1),
    "OH": lambda a: functionalize_random(a, "OH", n_groups=4,
                                         site_kind="basal", seed=0),
    "COOH": lambda a: functionalize_random(a, "COOH", n_groups=3,
                                           site_kind="basal", seed=0),
    "NH2": lambda a: functionalize_random(a, "NH2", n_groups=4,
                                          site_kind="basal", seed=0),
}


def slit(name: str):
    """The decorated sheet, stacked at WIDTH into a 3-periodic slit."""
    sheet = VARIANTS[name](build_graphene_supercell(nx=10, ny=6,
                                                    vacuum=10.0))
    sheet.set_cell(np.diag([sheet.cell[0, 0], sheet.cell[1, 1], WIDTH]))
    sheet.center(axis=2)
    layers = max(1, int(np.ceil(MIN_EDGE / WIDTH)))
    stack = sheet * (1, 1, layers)
    stack.set_pbc((True, True, True))
    stack.wrap()
    return stack


def henry() -> None:
    out = HERE / f"henry_co2_funcional{SUFFIX}.txt"
    done = out.read_text() if out.exists() else ""
    for name in VARIANTS:
        if SUFFIX and name in ("pristina", "N5"):
            continue                    # no groups: same in both fields
        if any(line.split()[:1] == [name] for line in done.splitlines()):
            continue
        atoms = slit(name)
        wall = wall_parameters(atoms, forcefield=FORCEFIELD)
        neutral = dict(wall, charge=np.zeros(len(atoms)))
        started = time.time()
        rows = []
        for label, params in (("con cargas", wall), ("sin cargas", neutral)):
            k, q = zip(*[henry_co2(atoms, TEMPERATURE, samples=120_000,
                                   seed=s, wall=params) for s in range(2)])
            rows.append(f"{label}: K_H {np.mean(k):8.1f} ({min(k):.0f}-"
                        f"{max(k):.0f})  q_st {np.mean(q):5.2f}")
        line = (f"{name:9s} {atoms.get_chemical_formula():>16s}  carga neta "
                f"{wall['net_charge']:+.2f} e/celda -> 0  |  "
                + "  |  ".join(rows) + f"  ({time.time() - started:.0f} s)")
        print(line, flush=True)
        with out.open("a") as handle:
            handle.write(line + "\n")


def run(name: str) -> None:
    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / f"funcional-{name}{SUFFIX}.json"
    stored = json.loads(path.read_text()) if path.exists() else {}
    atoms = slit(name)
    wall = wall_parameters(atoms, forcefield=FORCEFIELD)
    if "vacio_helio" not in stored:
        stored["vacio_helio"] = analyse_porosity(atoms).helium_void_fraction
        stored["formula"] = atoms.get_chemical_formula()
        stored["carga_neta_corregida"] = wall["net_charge"]
        stored["puntos"] = []
        path.write_text(json.dumps(stored, indent=1))
    done = {p["pressure_mpa"] for p in stored["puntos"]}
    for k, pressure in enumerate(PRESSURES):
        if pressure in done:
            continue
        started = time.time()
        result = run_co2_gcmc(atoms, TEMPERATURE, pressure,
                              equilibration=100_000, production=200_000,
                              seed=k, void_fraction=stored["vacio_helio"],
                              wall=wall)
        stored["puntos"].append(asdict(result)
                                | {"segundos": time.time() - started})
        stored["puntos"].sort(key=lambda r: r["pressure_mpa"])
        path.write_text(json.dumps(stored, indent=1))
        print(f"{name} {pressure} MPa: {result.mol_per_kg:.3f} mol/kg",
              flush=True)


def table() -> None:
    for name in VARIANTS:
        path = FOLDER / f"funcional-{name}{SUFFIX}.json"
        if not path.exists() and SUFFIX:
            path = FOLDER / f"funcional-{name}.json"     # no groups: same
        if not path.exists():
            continue
        stored = json.loads(path.read_text())
        print(f"== {name} {stored['formula']} (vacío de helio "
              f"{stored['vacio_helio']:.3f})")
        for p in stored["puntos"]:
            print(f"  {p['pressure_mpa']:7.3g} MPa  {p['mol_per_kg']:7.3f} "
                  f"mol/kg abs  {p['excess_mol_per_kg']:7.3f} exc  ± "
                  f"{p['count_error']:.2f} moléc.  avisos {len(p['warnings'])}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "henry":
        henry()
    elif len(sys.argv) > 2 and sys.argv[1] == "corre":
        run(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "tabla":
        table()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
