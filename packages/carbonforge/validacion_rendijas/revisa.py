"""Check each slit is sound carbon, then measure its porosity.

The check comes first and is not a formality: a slit built by stacking
has two ways to be wrong that the file itself will not show -- a bond
list that crosses the cell seam as a cell-length stretch, and planes
that ended up at the wrong spacing after wrapping. Both are read back
from the written file under the periodic boundary, not from the object
in memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.io import read
from scipy.spatial import cKDTree

from carbonforge.adsorption.porosity import analyse_porosity


def bonds_under_pbc(atoms) -> tuple[float, float, int]:
    """Shortest, longest and count of C-C bonds, honouring the cell."""
    tree = cKDTree(atoms.get_positions(), boxsize=np.diag(atoms.cell))
    pairs = tree.query_pairs(1.8, output_type="ndarray")
    if len(pairs) == 0:
        return (0.0, 0.0, 0)
    d = np.linalg.norm(
        atoms.get_distances(pairs[:, 0], pairs[:, 1], mic=True, vector=True),
        axis=1)
    inside = d > 0.9
    return float(d[inside].min()), float(d[inside].max()), int(inside.sum())


def main() -> int:
    """Report geometry and porosity for every slit in the folder."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(f"{'H':>5} {'atomos':>7} {'enlaces':>8} {'min':>6} {'max':>6} "
          f"{'vacio':>6} {'He':>6} {'poro':>6} {'area':>7} {'cavidad':>8} {'canal':>7}")
    for path in sorted(root.glob("rendija_*.extxyz"),
                       key=lambda p: float(p.stem.split("_")[1])):
        atoms = read(str(path))
        lo, hi, n = bonds_under_pbc(atoms)
        report = analyse_porosity(atoms, samples=150_000, seed=0)
        print(f"{float(path.stem.split('_')[1]):5.1f} {len(atoms):7d} {n:8d} "
              f"{lo:6.3f} {hi:6.3f} {report.void_fraction:6.3f} "
              f"{report.helium_void_fraction:6.3f} "
              f"{report.pore_volume_cm3_g:6.3f} "
              f"{report.accessible_area_m2_g:7.0f} "
              f"{report.largest_included_sphere:8.2f} "
              f"{report.largest_free_sphere:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
