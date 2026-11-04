"""GPAW references of transition-metal monolayers: MoS2 (2H) and the MoCS Janus layer.

Run from packages/tbkit (resumable, one file per point)::

    python -m tbkit.recipes.tm_crystals gpaw [NAME]
    python -m tbkit.recipes.tm_crystals list

For each monolayer: GPAW PBE plane waves relaxes the atoms and the in-plane cell (the
vacuum direction fixed), then single points of the relaxed layer strained by ±2 % and of
three random distortions (energies and forces, the validation of a set fitted on
molecules), and PBE bands along Γ-M-K-Γ in the format of ``band_validation``.

* ``MoS2``: 2H (trigonal prismatic), the reference sulfide.
* ``MoCS``: the Janus layer of the user's work: C on one side of the Mo plane, S on the
  other, trigonal prismatic like 2H; a starting geometry from MoS2 with the top S → C at
  a shorter height. Its topology (spin-orbit, Z2) is a later stage (ESTADO.md).

Spin-paired, no spin-orbit. The 6-electron Mo setup, as in the molecular references.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

from ..settings import Param, add_arguments, apply

WORK = Path("out/tm_crystals")
PW_CUTOFF = 500.0
KMESH = 12
KT = 0.05
VACUUM_C = 20.0
NK_PATH = 90
STRAINS = (0.98, 1.02)
N_RANDOM = 3
SIGMA = 0.05
FMAX = 0.02
SYSTEMS = ("MoS2", "MoCS")


def start(name: str) -> Atoms:
    a = 3.18
    cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, VACUUM_C]]
    frac = np.array([[1 / 3, 2 / 3], [2 / 3, 1 / 3]])
    xy = frac @ np.array(cell)[:2, :2]
    z0 = VACUUM_C / 2
    if name == "MoS2":
        positions = [[*xy[0], z0], [*xy[1], z0 + 1.56], [*xy[1], z0 - 1.56]]
        symbols = "MoS2"
    elif name == "MoCS":
        positions = [[*xy[0], z0], [*xy[1], z0 + 1.20], [*xy[1], z0 - 1.56]]
        symbols = "MoCS"
    else:
        raise KeyError(name)
    # plane waves need a periodic cell: the slab repeats along z with VACUUM_C of vacuum
    return Atoms(symbols, positions=positions, cell=cell, pbc=True)


def _calc(txt: Path):
    from gpaw import GPAW, PW

    return GPAW(mode=PW(PW_CUTOFF), xc="PBE", kpts=(KMESH, KMESH, 1),
                occupations={"name": "fermi-dirac", "width": KT}, setups={"Mo": "6"},
                txt=str(txt))


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _point(atoms: Atoms, label: str, extra=None) -> dict:
    return {"label": label, "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.positions.tolist(), "cell": np.asarray(atoms.cell).tolist(),
            "pbc": [bool(p) for p in atoms.pbc], "energy": float(atoms.get_potential_energy()),
            "forces": atoms.get_forces().tolist(), "extra": extra or {}}


def relax(name: str) -> Atoms:
    from ase.filters import FrechetCellFilter
    from ase.io import read
    from ase.optimize import BFGS

    folder = WORK / name
    out = folder / "relaxed.json"
    if out.exists():
        data = json.loads(out.read_text())
        return Atoms(data["symbols"], positions=data["positions"], cell=data["cell"],
                     pbc=data["pbc"])
    folder.mkdir(parents=True, exist_ok=True)
    traj = folder / "relax.traj"
    atoms = read(traj, index=-1) if traj.exists() else start(name)
    atoms.pbc = True
    atoms.calc = _calc(folder / "relax.txt")
    mask = [True, True, False, False, False, True]          # in-plane cell only
    BFGS(FrechetCellFilter(atoms, mask=mask), trajectory=str(traj), logfile=None).run(
        fmax=FMAX, steps=200)
    _write(out, _point(atoms, f"{name}/relaxed",
                       {"a_A": float(atoms.cell.lengths()[0])}))
    return atoms


def run(name: str) -> None:
    relaxed = relax(name)
    folder = WORK / name
    points = [(f"x{s}", s, None) for s in STRAINS] + \
        [(f"rnd{k}", 1.0, k) for k in range(N_RANDOM)]
    for label, scale, seed in points:
        out = folder / f"{label}.json"
        if out.exists():
            continue
        atoms = relaxed.copy()
        if scale != 1.0:
            atoms.set_cell(np.asarray(atoms.cell) * [[scale], [scale], [1.0]],
                           scale_atoms=True)
            atoms.positions[:, 2] = relaxed.positions[:, 2]
        if seed is not None:
            atoms.rattle(SIGMA, seed=100 + seed)
        atoms.calc = _calc(out.with_suffix(".txt"))
        _write(out, _point(atoms, f"{name}/{label}"))
    bands = folder / "bands.json"
    if not bands.exists():
        atoms = relaxed.copy()
        atoms.calc = _calc(folder / "scf.txt")
        atoms.get_potential_energy()
        fermi = float(atoms.calc.get_fermi_level())
        bp = atoms.cell.bandpath("GMKG", npoints=NK_PATH, pbc=(True, True, False))
        n_occ = int(round(sum({"Mo": 6, "S": 6, "C": 4}[s]
                              for s in atoms.get_chemical_symbols()) / 2))
        calc = atoms.calc.fixed_density(kpts=bp.kpts, symmetry="off", txt=None,
                                        nbands=n_occ + 8, convergence={"bands": n_occ + 6})
        eps = np.array([calc.get_eigenvalues(kpt=i) for i in range(len(bp.kpts))])
        _write(bands, {"system": name, "code": "gpaw_pw", "fermi_eV": fermi, "n_occ": n_occ,
                       "kpts": bp.kpts.tolist(), "bands": eps[:, :n_occ + 6].tolist(),
                       "settings": f"GPAW PBE PW {PW_CUTOFF} eV, Mo.6, FD {KT} eV, "
                                   f"k {KMESH}x{KMESH}, sin espín-órbita"})


PARAMS = [
    Param("PW_CUTOFF", "energía de corte de las ondas planas", "eV", "", "convergencia"),
    Param("KMESH", "malla k en el plano (n×n)", "", "", "convergencia"),
    Param("KT", "ensanchamiento de Fermi-Dirac", "eV", "", "cálculo"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("gpaw", "list"))
    parser.add_argument("system", nargs="?", help="una monocapa (vacío: todas)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    for name in ([args.system] if args.system else SYSTEMS):
        if args.step == "gpaw":
            run(name)
            print(name, "hecho", flush=True)
        else:
            print(name, sorted(p.name for p in (WORK / name).glob("*.json")))


if __name__ == "__main__":
    main()
