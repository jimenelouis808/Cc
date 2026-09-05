"""Dopant placement chosen by ring size rather than at random.

Where a heteroatom sits matters more than how many there are. On a
curved nanocarbon the pentagons are not decoration: they are the sites
that carry the positive curvature, they are pyramidalised, and that
pyramidalisation makes them the most reactive carbons in the structure --
which is why a fullerene's chemistry happens at its pentagons and a
capped tube's happens at its cap. A nitrogen placed on a pentagon is a
different object from a nitrogen placed 40 Å away on the cylinder, and
until now the package could only ask for the second.

The ring data is already there. Every builder that goes through the
triangulated dual -- capped tubes, fullerenes, nano-onions, junctions,
schwarzites, assemblies -- records ``atoms.info["rings"]`` as the actual
list of atom indices per ring, so this module needs no geometry at all:
no distance cutoffs, no cycle basis, no ring perception. That matters,
because re-deriving rings from coordinates on a curved shell is exactly
the thing `builders/fullerene_mesh.py` exists to avoid, and it produced
silently wrong ring counts when it was tried.

A structure with no ``rings`` metadata (a plain ``build_cnt`` sheet, an
imported file) therefore cannot be doped this way, and says so rather
than guessing.
"""

from __future__ import annotations

import warnings

import numpy as np
from ase import Atoms

from ..utils.rng import make_rng
from .chemistry import get_chemistry
from .substitutional import substitute_atoms


def ring_sites(atoms: Atoms, ring_size: int = 5) -> np.ndarray:
    """Indices of the carbons belonging to at least one ring of that size.

    Parameters
    ----------
    atoms
        A structure carrying ``info["rings"]``.
    ring_size
        Ring size to select: 5 for pentagons, 7 for heptagons, 8 for
        octagons. 6 is accepted but selects nearly the whole structure.

    Returns
    -------
    numpy.ndarray
        Sorted carbon indices, possibly empty (a straight all-hexagon
        tube has no pentagons, and that is a fact about the structure
        rather than an error).

    Raises
    ------
    ValueError
        If the structure records no rings at all. Ring membership cannot
        be re-derived reliably from coordinates on a curved shell, so
        this refuses rather than approximating.
    """
    rings = atoms.info.get("rings")
    if not rings:
        raise ValueError(
            "This structure carries no ring metadata, so dopants cannot be "
            "placed by ring size. The mesh-based builders record it "
            "(capped tube, fullerene, nano-onion, junction, schwarzite, "
            "multi-wall, bundle); a plain build_cnt or build_graphene sheet "
            "does not. Use random, edge or bulk placement instead."
        )
    if ring_size < 3:
        raise ValueError(f"ring_size must be at least 3; got {ring_size}.")

    symbols = atoms.get_chemical_symbols()
    chosen: set[int] = set()
    for ring in rings:
        if len(ring) == ring_size:
            chosen.update(int(a) for a in ring if symbols[int(a)] == "C")
    return np.array(sorted(chosen), dtype=int)


def ring_size_census(atoms: Atoms) -> dict[int, int]:
    """How many carbons belong to at least one ring of each size.

    Not the same as the ring census in ``info["ring_counts"]``, which
    counts rings: an atom sits on three rings at once, so these numbers
    overlap. This is the one that says how many doping sites a given
    ring size actually offers.
    """
    rings = atoms.info.get("rings")
    if not rings:
        return {}
    sizes = {len(ring) for ring in rings}
    return {size: len(ring_sites(atoms, size)) for size in sorted(sizes)}


def dope_rings(
    atoms: Atoms,
    element: str,
    ring_size: int = 5,
    concentration: float | None = None,
    count: int | None = None,
    seed: int | None = None,
) -> Atoms:
    """Substitute carbons that sit on rings of a given size.

    Give exactly one of ``concentration`` or ``count``. A concentration
    is a fraction of the **available sites of that ring size**, not of
    the whole structure -- asking for 10% pentagon doping on a C60 means
    six of its sixty pentagon carbons, not six of some larger pool. The
    achieved fraction against the whole structure is recorded, since
    those two numbers are far apart and reporting only the first would
    be misleading.

    Parameters
    ----------
    atoms
        Host structure carrying ``info["rings"]`` (not mutated).
    element
        Dopant symbol, from :data:`~nanocarbon_lab.dopants.chemistry.DOPANT_ELEMENTS`.
    ring_size
        5 for pentagons (the default and the interesting case), 7 or 8
        for the negative-curvature rings of a junction or schwarzite.
    concentration
        Fraction in ``[0, 1]`` of the eligible sites to substitute.
    count
        Absolute number of sites instead of a fraction.
    seed
        RNG seed; placement is reproducible.

    Returns
    -------
    ase.Atoms
        Doped structure, with ``info["doping_mode"]`` set to
        ``f"ring{ring_size}"`` and the site census recorded alongside.

    Raises
    ------
    ValueError
        If neither or both of ``concentration`` and ``count`` are given,
        if the structure has no ring metadata, or if it has no rings of
        the requested size.
    """
    get_chemistry(element)  # rejects an unknown dopant before any work
    if (concentration is None) == (count is None):
        raise ValueError(
            "Give exactly one of concentration or count."
        )

    sites = ring_sites(atoms, ring_size)
    if len(sites) == 0:
        census = ring_size_census(atoms)
        have = ", ".join(f"{k}-rings: {v} sites" for k, v in census.items())
        raise ValueError(
            f"This structure has no {ring_size}-membered rings to dope. "
            f"It has {have or 'no rings at all'}."
        )

    if concentration is not None:
        if not 0.0 <= concentration <= 1.0:
            raise ValueError("concentration must be in [0, 1].")
        n_sub = int(round(concentration * len(sites)))
    else:
        if count <= 0:
            raise ValueError("count must be > 0.")
        n_sub = int(count)

    if n_sub == 0:
        return atoms.copy()
    if n_sub > len(sites):
        warnings.warn(
            f"Asked for {n_sub} dopants on {ring_size}-rings but only "
            f"{len(sites)} such carbons exist; substituting all of them.",
            RuntimeWarning,
            stacklevel=2,
        )
        n_sub = len(sites)

    rng = make_rng(seed)
    chosen = rng.choice(sites, size=n_sub, replace=False)
    result = substitute_atoms(atoms, sorted(int(i) for i in chosen), element)
    result.info["doping_mode"] = f"ring{ring_size}"
    result.info["doping_seed"] = seed
    result.info["doping_ring_size"] = ring_size
    result.info["doping_sites_available"] = int(len(sites))
    # Two fractions, because they differ by a large factor and quoting
    # either one alone reads as the other.
    result.info["doping_concentration"] = n_sub / len(sites)
    result.info["doping_concentration_overall"] = n_sub / max(1, len(atoms))
    return result


__all__ = ["dope_rings", "ring_sites", "ring_size_census"]
