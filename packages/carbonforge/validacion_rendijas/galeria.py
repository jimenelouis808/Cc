"""Every superstructure (and the SWCNT bundle) with every gas adsorbed:
equilibrium snapshots for the report's supplementary gallery.

Short illustrative runs, not production averages. Gases:
  h2      H2, 77 K, 10 MPa (Feynman-Hibbs)
  co2     CO2, 298 K, 0.1 MPa
  mezcla  15 % CO2 in N2, 298 K, 0.1 MPa
  humeda  the same flue gas at 80 % relative humidity (SPC/E water)
Walls are neutral Steele carbon. Each (structure, gas) is one extxyz in
galeria/ and one PNG; a finished one is skipped.

    python galeria.py corre <structure> <gas>
    python galeria.py lista            # all pairs, one per line
    python galeria.py dibuja
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from instantaneas import combine, colours

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "galeria"

#: name -> (file relative to HERE or "bundle", drawing style)
STRUCTURES = {
    "swcnt-bundle": ("bundle", "axis"),
    "schwarzite-P": ("cristalinas/knee-schwarzite-primitive.extxyz", "slab"),
    "schwarzite-D": ("cristalinas/knee-schwarzite-diamond.extxyz", "slab"),
    "gyroid": ("cristalinas/knee-schwarzite-gyroid.extxyz", "slab"),
    "knee-super-graphene": ("cristalinas/knee-super-graphene_k22.extxyz", "slab"),
    "knee-super-square": ("cristalinas/knee-super-square_k12_n7.extxyz", "slab"),
    "super-cubic": ("superestructuras/super-cubic.extxyz", "cage"),
    "super-diamond": ("superestructuras/super-diamond.extxyz", "net"),
    "super-fcc": ("superestructuras/super-fcc.extxyz", "net"),
    "super-graphene": ("superestructuras/super-graphene.extxyz", "net"),
    "super-square": ("superestructuras/super-square.extxyz", "net"),
    "super-cube": ("jaulas/super-cube.extxyz", "cage"),
    "super-tetrahedron": ("jaulas/super-tetrahedron.extxyz", "cage"),
    "super-hypercube": ("superestructuras/super-hypercube.extxyz", "cage"),
}
GASES = ("h2", "co2", "mezcla", "humeda")
SLAB = 8.0          # A, thickness of the cut drawn for 3D-periodic cells


def framework(name: str) -> Atoms:
    source = STRUCTURES[name][0]
    if source == "bundle":
        from carbonforge.adsorption.bundle import levesque_bundle
        return levesque_bundle()
    from llenado_super import embed
    return embed(read(str(HERE / source)))[0]


def simulate(name: str, gas: str) -> Atoms:
    from carbonforge.adsorption.co2 import CO2, N2, WATER, run_co2_gcmc
    from carbonforge.adsorption.mixture import run_mixture_gcmc
    import humedad
    atoms = framework(name)
    if gas == "h2":
        from carbonforge.adsorption.gcmc import run_gcmc
        r = run_gcmc(atoms, 77.0, 10.0, equilibration=50_000,
                     production=2_000, snapshots=1, seed=0)
        out = atoms.copy()
        out.set_tags(0)
        out += Atoms("H" * len(r.snapshots[-1]), positions=r.snapshots[-1],
                     tags=[2] * len(r.snapshots[-1]))
        return out
    if gas == "co2":
        r = run_co2_gcmc(atoms, 298.0, 0.1, equilibration=60_000,
                         production=2_000, keep_configuration=True)
    elif gas == "mezcla":
        r = run_mixture_gcmc(atoms, 298.0, 0.1, (0.15, 0.85),
                             equilibration=80_000, production=2_000,
                             keep_configuration=True)
    elif gas == "humeda":
        r = run_mixture_gcmc(atoms, 298.0, 0.1, humedad.fractions(80),
                             species=(CO2, N2, WATER),
                             equilibration=120_000, production=2_000,
                             keep_configuration=True)
    else:
        raise SystemExit(f"unknown gas {gas}")
    return combine(atoms, r.configuration)


def draw(path: Path) -> Path:
    atoms = read(str(path))
    style = STRUCTURES[path.stem.split("__")[0]][1]
    if style in ("slab", "net"):
        # A meshed net is mostly vacuum: an 8 A cut can miss every strut,
        # so it gets 40 % of the cell instead.
        cell = np.diag(atoms.cell)
        width = SLAB if style == "slab" else 0.4 * cell[1]
        y = atoms.positions[:, 1] % cell[1]
        atoms = atoms[np.abs(y - cell[1] / 2) < width / 2]
        rotation = "-90x"
    elif style == "axis":
        rotation = "-90x"           # the bundle's tubes run along y
    else:
        rotation = "-30x,20y"
    tags = atoms.get_tags()
    radii = [0.55 if t == 0 else (1.0 if t == 2 else
             {"H": 0.35, "C": 0.6, "O": 0.62, "N": 0.65}[s])
             for s, t in zip(atoms.get_chemical_symbols(), tags)]
    png = path.with_suffix(".png")
    write(str(png), atoms, rotation=rotation, colors=colours(atoms),
          radii=radii, scale=16 if len(atoms) > 6000 else 20,
          show_unit_cell=0)
    return png


if __name__ == "__main__":
    FOLDER.mkdir(exist_ok=True)
    if len(sys.argv) > 3 and sys.argv[1] == "corre":
        target = FOLDER / f"{sys.argv[2]}__{sys.argv[3]}.extxyz"
        if not target.exists():
            write(str(target), simulate(sys.argv[2], sys.argv[3]))
            draw(target)
    elif len(sys.argv) > 1 and sys.argv[1] == "lista":
        for gas in GASES:
            for name in STRUCTURES:
                print(name, gas)
    elif len(sys.argv) > 1 and sys.argv[1] == "dibuja":
        for path in sorted(FOLDER.glob("*.extxyz")):
            draw(path)
    else:
        print(__doc__)
        sys.exit(1)
