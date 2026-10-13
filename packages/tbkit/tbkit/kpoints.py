"""k-point sets: Γ for finite systems, Monkhorst-Pack meshes and band paths."""

from __future__ import annotations

import numpy as np
from ase import Atoms


def mesh(atoms, n: int | tuple[int, int, int] = 12) -> tuple[np.ndarray, np.ndarray]:
    """Γ-centred uniform mesh along the periodic axes; ``(kpts, weights)``.

    ``n`` points per periodic axis (an int applies to all). Non-periodic axes
    get one point. Γ-centred on purpose: the K point of graphene, where the
    physics is, lies on a Γ-centred mesh when ``n`` is a multiple of 3.

    Given a :class:`~tbkit.hamiltonian.System` instead of an ``Atoms``, an axis is
    sampled only if some bond crosses the cell along it (``System.k_axes``). A tube in
    a box declared periodic in x, y and z has H(k) independent of k_x and k_y, so one
    point there is exact, not an approximation; sampling them anyway turned a
    6-point calculation into 6³ = 216 (and 24 into 13 824, 62 GB of eigenvectors).
    """
    from .hamiltonian import System

    pbc = atoms.k_axes if isinstance(atoms, System) else atoms.get_pbc()
    counts = (n, n, n) if isinstance(n, int) else tuple(n)
    counts = tuple(c if p else 1 for c, p in zip(counts, pbc, strict=True))
    axes = [np.arange(c) / c for c in counts]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    return grid, np.full(len(grid), 1.0 / len(grid))


def gamma() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((1, 3)), np.ones(1)


def band_path(atoms: Atoms, path: str | None = None, npoints: int = 200):
    """A band path from ASE (special points of the lattice); returns the ASE BandPath.

    For a 1D system (one periodic axis) the path is Γ-X along that axis.
    """
    pbc = atoms.get_pbc()
    if int(pbc.sum()) == 1:
        axis = int(np.flatnonzero(pbc)[0])
        end = np.zeros(3)
        end[axis] = 0.5
        from ase.dft.kpoints import BandPath

        kpts = np.linspace(np.zeros(3), end, npoints)
        return BandPath(atoms.cell, kpts=kpts, special_points={"G": np.zeros(3), "X": end},
                        path="GX")
    return atoms.cell.bandpath(path, npoints=npoints, pbc=pbc)
