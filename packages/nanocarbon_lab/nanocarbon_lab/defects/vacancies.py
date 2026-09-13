"""Mono- and divacancy generation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms

from ..utils.geometry import minimum_image_distances
from ..utils.metadata import keep_indices, remap_after_removal
from ..utils.rng import make_rng

VacancyKind = Literal["mono", "di"]


def introduce_vacancies(
    atoms: Atoms,
    n_defects: int = 1,
    kind: VacancyKind = "mono",
    seed: int | None = None,
    min_separation: float = 4.0,
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

    Returns
    -------
    ase.Atoms
        Defective structure. The list of removed indices (in the original
        numbering) is stored in ``atoms.info["vacancies"]``.
    """
    if n_defects <= 0:
        raise ValueError("n_defects must be >= 1.")
    rng = make_rng(seed)

    dmat = minimum_image_distances(atoms)
    n = len(atoms)
    available = set(range(n))
    removed: list[int] = []
    defect_centres: list[np.ndarray] = []
    positions = atoms.get_positions()

    for _ in range(n_defects):
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
            # Find the nearest available neighbour and remove it too.
            nbrs = np.argsort(dmat[i])
            j = next(
                (int(k) for k in nbrs if k in available and k != i),
                None,
            )
            if j is None:
                raise RuntimeError("No neighbour available for divacancy.")
            available.discard(j)
            removed.append(j)
            centre = 0.5 * (positions[i] + positions[j])

        defect_centres.append(centre)

    keep = keep_indices(n, removed)
    out = atoms[keep]
    # Renumbered, not copied: the recorded bond graph and ring list are
    # indices into the *old* numbering, and carrying them across a
    # deletion leaves metadata that is present, plausible and wrong.
    out.info = remap_after_removal(atoms.info, keep)
    out.info.setdefault("defects", []).append(
        {
            "type": f"{kind}vacancy",
            "n": n_defects,
            "removed_indices": sorted(removed),
            "seed": seed,
        }
    )
    return out
