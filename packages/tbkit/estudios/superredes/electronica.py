"""π electronic structure of the carbon superstructures (tbkit π Hückel model).

Structures arrive as extxyz files (estructuras/), written by nanocarbon_lab
and used for the adsorption study; nothing is imported from another package.

For each structure: Fermi level and DOS on a k mesh, band structure along
the lattice's special path (periodic ones), HOMO-LUMO gap (finite ones).
The π model (t = -2.7 eV, nearest neighbours) is a textbook model: good
for band topology and trends (Dirac points, flat bands, gaps from rings),
not for quantitative band energies.

    python electronica.py            -> resultados.json, figuras/*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from tbkit.analysis import bands, dos
from tbkit.hamiltonian import System
from tbkit.kpoints import band_path, gamma, mesh
from tbkit.params import pi_model
from tbkit.solver import solve

HERE = Path(__file__).resolve().parent
FIG = HERE / "figuras"

#: name -> (file, band path or None, k mesh)
STRUCTURES = {
    "knee-super-graphene": ("knee-super-graphene_k22.extxyz", "GMKG", 24),
    "knee-super-square": ("knee-super-square_k12_n7.extxyz", "GXMG", 24),
    "schwarzite-P": ("knee-schwarzite-primitive.extxyz", "GXMGRX", 6),
    "schwarzite-D": ("knee-schwarzite-diamond.extxyz", "GXMGRX", 5),
    "gyroid": ("knee-schwarzite-gyroid.extxyz", "GXMGRX", 4),
    "meshed-super-graphene": ("super-graphene.extxyz", "GXSYG", 12),
    "super-cube": ("super-cube.extxyz", None, 1),
    "super-hypercube": ("super-hypercube.extxyz", None, 1),
}


def hexagonal_primitive(atoms: Atoms) -> Atoms:
    """A rectangular cell with b = sqrt(3) a holds two hexagonal primitive
    cells; return one (a1 = A, a2 = (A + B)/2), with duplicates removed."""
    a, b, c = np.diag(atoms.cell)
    if abs(b / a - np.sqrt(3)) > 1e-3:
        raise ValueError(f"b/a = {b / a:.4f}, not sqrt(3)")
    cell = np.array([[a, 0, 0], [a / 2, b / 2, 0], [0, 0, c]])
    frac = np.linalg.solve(cell.T, atoms.positions.T).T
    frac[:, :2] %= 1.0
    pos = frac @ cell
    keep = []
    for i, p in enumerate(pos):
        if all(np.linalg.norm(_mic(p - pos[j], cell)) > 0.3 for j in keep):
            keep.append(i)
    out = Atoms(np.array(atoms.get_chemical_symbols())[keep], positions=pos[keep],
                cell=cell, pbc=atoms.pbc)
    if 2 * len(out) != len(atoms):
        raise ValueError(f"{len(atoms)} atoms -> {len(out)}, expected half")
    return out


def _mic(d, cell):
    f = np.linalg.solve(cell.T, d)
    f[:2] -= np.round(f[:2])
    return f @ cell


def load(name: str) -> Atoms:
    atoms = read(str(HERE / "estructuras" / STRUCTURES[name][0]))
    if name == "knee-super-graphene":
        atoms = hexagonal_primitive(atoms)
    if not atoms.pbc.any():
        atoms.pbc = False
    return atoms


def study(name: str, model) -> dict:
    path, kmesh = STRUCTURES[name][1], STRUCTURES[name][2]
    atoms = load(name)
    started = time.time()
    system = System.build(atoms, model)
    kpts = mesh(system, kmesh) if system.periodic else gamma()
    solution = solve(system, *kpts, kT=0.01)
    ef = solution.fermi
    grid = np.linspace(ef - 3, ef + 3, 1201)
    _, total = dos(solution, grid, sigma=0.03)
    out = {"atoms": len(atoms), "periodic": bool(system.periodic), "fermi": ef,
           "gap": solution.gap(), "kmesh": kmesh,
           "dos_at_fermi_per_atom": float(np.interp(ef, grid, total)) / len(atoms),
           "dos": {"e_minus_ef": (grid - ef).round(4).tolist(),
                   "states_per_ev_per_atom": (total / len(atoms)).round(5).tolist()}}
    homo, lumo = solution.homo_lumo()
    out["homo"], out["lumo"] = float(homo), float(lumo)
    if path and system.periodic:
        bp = band_path(atoms, path, 160)
        x, sx, labels = bp.get_linear_kpoint_axis()
        e = bands(system, bp)[0] - ef
        near = np.any(np.abs(e) < 2.0, axis=0)
        out["bands"] = {"k": x.round(5).tolist(), "special_k": sx.round(5).tolist(),
                        "labels": list(labels), "e_minus_ef": e[:, near].round(5).tolist()}
        # Smallest direct gap at E_F along the path, and where.
        above = np.where(e > 0, e, np.inf).min(axis=1)
        below = np.where(e <= 0, e, -np.inf).max(axis=1)
        i = int(np.argmin(above - below))
        out["min_direct_gap_on_path"] = float(above[i] - below[i])
        out["min_gap_at_k"] = float(x[i])
    out["seconds"] = time.time() - started
    return out


def main(names) -> None:
    model = pi_model()
    target = HERE / "resultados.json"
    results = json.loads(target.read_text()) if target.exists() else {}
    for name in names:
        if name in results:
            continue
        r = study(name, model)
        results[name] = r
        target.write_text(json.dumps(results, indent=0))
        extra = (f"  min. direct gap on path {r['min_direct_gap_on_path']:.4f} eV"
                 if "bands" in r else "")
        print(f"{name:24s} {r['atoms']:5d} atoms  gap {r['gap']:.4f} eV  "
              f"DOS(E_F) {r['dos_at_fermi_per_atom']:.4f} /eV/atom{extra}  "
              f"({r['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(STRUCTURES))
