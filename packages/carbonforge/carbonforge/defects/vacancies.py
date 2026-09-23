"""Mono- and divacancy generation."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
from ase import Atoms

from ..utils.geometry import minimum_image_distances
from ..utils.rng import make_rng


VacancyKind = Literal["mono", "di"]


def introduce_vacancies(
    atoms: Atoms,
    n_defects: int = 1,
    kind: VacancyKind = "mono",
    seed: Optional[int] = None,
    min_separation: float = 4.0,
    sites: Optional[Sequence[int]] = None,
) -> Atoms:
    """Remove atoms to introduce ``n_defects`` vacancies.

    Parameters
    ----------
    atoms
        Input structure (not mutated).
    n_defects
        Number of vacancies to introduce.
    kind
        ``"mono"`` removes a single atom per defect; ``"di"`` removes two
        nearest-neighbour atoms per defect (divacancy).
    seed
        RNG seed for the random choice of sites.
    min_separation
        Minimum distance between defect centres (Å) to avoid overlapping
        defects in the same neighbourhood.
    sites
        Explicit atom indices, one per defect, to remove instead of drawing
        them at random. Overrides ``n_defects``. For ``"di"`` the partner is
        still the nearest available neighbour. Use this when the position of
        the defect is part of the question -- a vacancy at the centre of a
        flake and one beside its edge are different systems.

    Returns
    -------
    ase.Atoms
        Defective structure. The list of removed indices (in the original
        numbering) is stored in ``atoms.info["vacancies"]``.
    """
    if sites is not None:
        sites = [int(i) for i in sites]
        n_defects = len(sites)
        for i in sites:
            if not 0 <= i < len(atoms):
                raise IndexError(f"Atom index {i} out of range (n={len(atoms)}).")
    if n_defects <= 0:
        raise ValueError("n_defects must be >= 1.")
    rng = make_rng(seed)

    dmat = minimum_image_distances(atoms)
    n = len(atoms)
    available = set(range(n))
    removed: list[int] = []
    defect_centres: list[np.ndarray] = []
    positions = atoms.get_positions()

    for k in range(n_defects):
        if sites is not None:
            candidates = [sites[k]] if sites[k] in available else []
            if not candidates:
                raise RuntimeError(f"Site {sites[k]} was already removed.")
            i = candidates[0]
            available.discard(i)
            removed.append(i)
            centre = positions[i].copy()
            if kind == "di":
                centre = _remove_partner(i, dmat, available, removed, positions)
            defect_centres.append(centre)
            continue
        # Filter candidates that respect min_separation wrt existing defects.
        candidates = []
        for idx in available:
            pos = positions[idx]
            if all(
                np.linalg.norm(pos - c) >= min_separation for c in defect_centres
            ):
                candidates.append(idx)
        if not candidates:
            raise RuntimeError(
                f"Ran out of candidate sites after {len(defect_centres)} vacancies. "
                "Lower min_separation or n_defects."
            )
        i = int(rng.choice(candidates))
        available.discard(i)
        removed.append(i)
        centre = positions[i].copy()

        if kind == "di":
            centre = _remove_partner(i, dmat, available, removed, positions)

        defect_centres.append(centre)

    keep = [i for i in range(n) if i not in set(removed)]
    out = atoms[keep]
    out.info = {**atoms.info}
    out.info.setdefault("defects", []).append(
        {
            "type": f"{kind}vacancy",
            "n": n_defects,
            "removed_indices": sorted(removed),
            "seed": seed,
        }
    )
    return out


def _remove_partner(
    i: int,
    dmat: np.ndarray,
    available: set[int],
    removed: list[int],
    positions: np.ndarray,
) -> np.ndarray:
    """Remove the nearest available neighbour of ``i``; return the defect centre."""
    nbrs = np.argsort(dmat[i])
    j = next(
        (int(k) for k in nbrs if k in available and k != i),
        None,
    )
    if j is None:
        raise RuntimeError("No neighbour available for divacancy.")
    available.discard(j)
    removed.append(j)
    return 0.5 * (positions[i] + positions[j])
