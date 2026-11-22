"""Equilibrium snapshots of the adsorption study, for the report's figures.

Short runs (illustrative, not the production averages): one final
configuration per case, written as extxyz (framework tag 0, adsorbate
tag 1) and drawn as a PNG. H2 in the cages reuses the snapshots already
written by the isotherm campaign (isotermas/<cage>/instantaneas).

    python instantaneas.py corre <case>
    python instantaneas.py dibuja
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "instantaneas"

#: Element drawn for each site of each species (None: not drawn, e.g.
#: N2's massless centre charge).
SITES = {"CO2": ("O", "C", "O"), "N2": ("N", None, "N"),
         "H2O": ("H", "O", "H")}
#: Colours (RGB 0-1). The wall is a quiet grey so the molecules read.
COLOURS = {"wall": (0.62, 0.65, 0.68), "C": (0.25, 0.25, 0.25),
           "O": (0.85, 0.18, 0.15), "N": (0.20, 0.35, 0.85),
           "H": (0.97, 0.97, 0.97), "Hw": (0.97, 0.97, 0.97),
           "H2": (0.95, 0.55, 0.75), "Nw": (0.2, 0.55, 0.85),
           "Ow": (0.85, 0.18, 0.15)}


def combine(framework: Atoms, configuration: list[dict]) -> Atoms:
    out = framework.copy()
    out.set_tags(0)
    for mol in configuration:
        sites = np.asarray(mol["sites"]) % np.diag(framework.cell)
        for symbol, pos in zip(SITES[mol["species"]], sites):
            if symbol is None:
                continue
            a = Atoms(symbol, positions=[pos], tags=[1])
            out += a
    return out


def case(name: str):
    """(atoms with adsorbate, title) for a case, running the GCMC."""
    from carbonforge.adsorption.co2 import (CO2, N2, WATER, run_co2_gcmc,
                                            wall_parameters)
    from carbonforge.adsorption.mixture import run_mixture_gcmc
    import funcional_co2
    import humedad
    import selectividad
    if name == "h2-rendija":
        from carbonforge.adsorption.gcmc import run_gcmc
        from llenado import slit
        atoms = slit(10.0)
        r = run_gcmc(atoms, 77.0, 10.0, equilibration=60_000,
                     production=20_000, snapshots=1, seed=0)
        out = atoms.copy()
        out.set_tags(0)
        out += Atoms("H" * len(r.snapshots[-1]), positions=r.snapshots[-1],
                     tags=[2] * len(r.snapshots[-1]))
        return out
    if name == "co2-rendija":
        atoms = funcional_co2.slit("pristina")
        r = run_co2_gcmc(atoms, 298.0, 1.0, equilibration=80_000,
                         production=5_000, keep_configuration=True)
        return combine(atoms, r.configuration)
    if name == "co2-schwarzita-d":
        atoms, _ = selectividad.structure("cristalina-knee-schwarzite-diamond")
        r = run_co2_gcmc(atoms, 298.0, 1.0, equilibration=80_000,
                         production=5_000, keep_configuration=True)
        return combine(atoms, r.configuration)
    if name == "mezcla-n5":
        atoms = funcional_co2.slit("N5")
        r = run_mixture_gcmc(atoms, 298.0, 0.1, (0.15, 0.85),
                             equilibration=100_000, production=5_000,
                             wall=wall_parameters(atoms),
                             keep_configuration=True)
        return combine(atoms, r.configuration)
    if name == "mezcla-super-cube":
        atoms, _ = selectividad.structure("super-cube")
        r = run_mixture_gcmc(atoms, 298.0, 0.1, (0.15, 0.85),
                             equilibration=100_000, production=5_000,
                             keep_configuration=True)
        return combine(atoms, r.configuration)
    if name == "humeda-n5":
        atoms = funcional_co2.slit("N5")
        r = run_mixture_gcmc(atoms, 298.0, 0.1, humedad.fractions(80),
                             species=(CO2, N2, WATER),
                             equilibration=200_000, production=5_000,
                             wall=wall_parameters(atoms),
                             keep_configuration=True)
        return combine(atoms, r.configuration)
    raise SystemExit(f"unknown case {name}")


CASES = ("h2-rendija", "co2-rendija", "co2-schwarzita-d", "mezcla-n5",
         "mezcla-super-cube", "humeda-n5")


def colours(atoms: Atoms) -> list:
    out = []
    for s, t in zip(atoms.get_chemical_symbols(), atoms.get_tags()):
        if t == 0:
            out.append(COLOURS["Nw"] if s == "N" else
                       COLOURS["Ow"] if s == "O" else
                       COLOURS["H"] if s == "H" else COLOURS["wall"])
        elif t == 2:
            out.append(COLOURS["H2"])
        else:
            out.append(COLOURS[s])
    return out


VIEWS = {"lado": "-80x,10y", "arriba": "0x", "perspectiva": "-30x,20y"}


def draw(path: Path, view: str) -> None:
    atoms = read(str(path))
    tags = atoms.get_tags()
    radii = [0.55 if t == 0 else (1.0 if t == 2 else
             {"H": 0.35, "C": 0.6, "O": 0.62, "N": 0.65}[s])
             for s, t in zip(atoms.get_chemical_symbols(), tags)]
    write(str(path.with_name(f"{path.stem}_{view}.png")), atoms,
          rotation=VIEWS[view],
          colors=colours(atoms), radii=radii, scale=22, show_unit_cell=0)


if __name__ == "__main__":
    FOLDER.mkdir(exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == "corre":
        name = sys.argv[2]
        target = FOLDER / f"{name}.extxyz"
        if not target.exists():
            write(str(target), case(name))
    elif len(sys.argv) > 1 and sys.argv[1] == "dibuja":
        for name in CASES:
            path = FOLDER / f"{name}.extxyz"
            if path.exists():
                for view in ("lado", "arriba"):
                    draw(path, view)
        for cage in ("super-cube", "super-tetrahedron"):
            src = HERE / "isotermas" / cage / "instantaneas" / "P10MPa_01.extxyz"
            dst = FOLDER / f"h2-{cage}.extxyz"
            if src.exists():
                atoms = read(str(src))
                atoms.set_tags([2 if t else 0 for t in atoms.get_tags()])
                write(str(dst), atoms)
                draw(dst, "perspectiva")
        print(json.dumps(sorted(p.name for p in FOLDER.glob("*.png"))))
    else:
        print(__doc__)
        sys.exit(1)
