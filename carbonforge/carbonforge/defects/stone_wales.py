"""Stone-Wales (55-77) defect generator.

A Stone-Wales defect rotates a single C-C bond by 90° in-plane, transforming
four adjacent hexagons into the classic 5-7-7-5 pattern. This module
implements a geometric approximation that is valid for nearly flat sp2
carbons (graphene, large-diameter CNT walls).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms

from ..utils.geometry import minimum_image_distances
from ..utils.rng import make_rng


def _pick_random_bond(
    atoms: Atoms,
    rng: np.random.Generator,
    bond_cutoff: float = 1.80,
) -> tuple[int, int]:
    """Pick a random bonded pair of carbons."""
    symbols = atoms.get_chemical_symbols()
    dmat = minimum_image_distances(atoms)
    np.fill_diagonal(dmat, np.inf)
    n = len(atoms)
    bonded = []
    for i in range(n):
        if symbols[i] != "C":
            continue
        for j in range(i + 1, n):
            if symbols[j] != "C":
                continue
            if 0.9 < dmat[i, j] < bond_cutoff:
                bonded.append((i, j))
    if not bonded:
        raise RuntimeError("No C-C bond found to rotate.")
    pick = bonded[int(rng.integers(0, len(bonded)))]
    return pick


def stone_wales_defect(
    atoms: Atoms,
    bond: Optional[tuple[int, int]] = None,
    seed: Optional[int] = None,
) -> Atoms:
    """Apply a single Stone-Wales rotation to a C-C bond.

    The two atoms defining the bond are rotated by 90° around the bond
    midpoint, around an axis locally normal to the sp2 plane. For planar
    graphene this gives the textbook 5-7-7-5 pattern. For CNTs / curved
    sheets the rotation is still applied in the local tangent plane, so
    downstream relaxation is advisable.

    Parameters
    ----------
    atoms
        Host structure (not mutated).
    bond
        Explicit ``(i, j)`` pair. If ``None``, a random bonded pair is picked.
    seed
        RNG seed used only when ``bond`` is ``None``.

    Returns
    -------
    ase.Atoms
        Defective structure with metadata under ``atoms.info["defects"]``.
    """
    rng = make_rng(seed)
    if bond is None:
        i, j = _pick_random_bond(atoms, rng)
    else:
        i, j = bond
        if i == j or not (0 <= i < len(atoms)) or not (0 <= j < len(atoms)):
            raise ValueError(f"Invalid bond indices {bond}.")

    out = atoms.copy()
    out.info = {**atoms.info}
    positions = out.get_positions()
    p_i, p_j = positions[i], positions[j]
    midpoint = 0.5 * (p_i + p_j)
    bond_vec = p_j - p_i

    # Local normal: approximate as the average normal of the two atoms'
    # neighbour planes. For flat graphene this collapses to ±z.
    dmat = minimum_image_distances(atoms)
    normals = []
    for k in (i, j):
        nbrs = np.argsort(dmat[k])[1:4]
        vecs = positions[nbrs] - positions[k]
        # Normal ~ cross of first two neighbour vectors.
        if len(vecs) >= 2:
            n = np.cross(vecs[0], vecs[1])
            norm = np.linalg.norm(n)
            if norm > 1e-6:
                normals.append(n / norm)
    if not normals:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = np.mean(normals, axis=0)
        axis = axis / (np.linalg.norm(axis) + 1e-12)

    # 90° rotation matrix around axis (Rodrigues).
    theta = np.pi / 2.0
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    R = np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    half = bond_vec / 2.0
    new_half = R @ half
    positions[i] = midpoint - new_half
    positions[j] = midpoint + new_half
    out.set_positions(positions)

    out.info.setdefault("defects", []).append(
        {"type": "stone_wales", "bond": [int(i), int(j)], "seed": seed}
    )
    return out
