"""Substitutional doping of nanocarbon structures.

All public functions return a **new** :class:`ase.Atoms` (the input is not
mutated) and record the list of substituted indices in ``atoms.info["dopants"]``.
Random placement is fully reproducible via the ``seed`` argument.

Chemistry guardrails (warnings only, not errors):
  * N, B → sp2 compatible, single substitution is well behaved.
  * S, P → larger covalent radii, prefer defect/edge sites; the module flags
    clustered S/P placements since they almost always need local relaxation.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
from ase import Atoms

from ..utils.constants import DOPANT_ELEMENTS
from ..utils.geometry import minimum_image_distances
from ..utils.rng import make_rng


Placement = Literal["random", "edges", "bulk", "cluster"]


def _validate_element(element: str) -> None:
    if element not in DOPANT_ELEMENTS:
        raise ValueError(
            f"Unsupported dopant '{element}'. Allowed: {DOPANT_ELEMENTS}."
        )


def substitute_atoms(atoms: Atoms, indices: Iterable[int], element: str) -> Atoms:
    """Replace the chemical symbol of ``indices`` with ``element``.

    Parameters
    ----------
    atoms
        Structure to dope (not mutated).
    indices
        Iterable of atom indices to substitute.
    element
        Target chemical symbol. Must be in :data:`DOPANT_ELEMENTS`.

    Returns
    -------
    ase.Atoms
        New doped structure.
    """
    _validate_element(element)
    out = atoms.copy()
    out.info = {**atoms.info}
    symbols = out.get_chemical_symbols()
    idx_list = list(indices)
    for i in idx_list:
        if not (0 <= i < len(out)):
            raise IndexError(f"Atom index {i} out of range (n={len(out)}).")
        if symbols[i] != "C":
            warnings.warn(
                f"Atom {i} is {symbols[i]}, not C — overwriting anyway.",
                RuntimeWarning,
                stacklevel=2,
            )
        symbols[i] = element
    out.set_chemical_symbols(symbols)
    existing = out.info.get("dopants", [])
    out.info["dopants"] = existing + [{"element": element, "indices": idx_list}]
    return out


def _carbon_indices(atoms: Atoms) -> np.ndarray:
    return np.array(
        [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "C"], dtype=int
    )


def _edge_indices(atoms: Atoms, cutoff: float = 1.80) -> np.ndarray:
    """Return C atoms with coordination < 3 (edge / under-coordinated)."""
    dmat = minimum_image_distances(atoms)
    np.fill_diagonal(dmat, np.inf)
    coord = ((dmat < cutoff) & (dmat > 0.5)).sum(axis=1)
    carbons = _carbon_indices(atoms)
    return carbons[coord[carbons] < 3]


def _bulk_indices(atoms: Atoms, cutoff: float = 1.80) -> np.ndarray:
    """Return C atoms with coordination == 3 (fully sp2 bonded)."""
    dmat = minimum_image_distances(atoms)
    np.fill_diagonal(dmat, np.inf)
    coord = ((dmat < cutoff) & (dmat > 0.5)).sum(axis=1)
    carbons = _carbon_indices(atoms)
    return carbons[coord[carbons] == 3]


def dope_random(
    atoms: Atoms,
    element: str,
    concentration: float,
    seed: Optional[int] = None,
) -> Atoms:
    """Randomly substitute a fraction of carbons with ``element``.

    Parameters
    ----------
    atoms
        Host nanocarbon.
    element
        Dopant symbol (N, B, S, P).
    concentration
        Fraction in ``[0, 1]`` of carbon atoms to replace.
    seed
        RNG seed.

    Returns
    -------
    ase.Atoms
        Doped structure.
    """
    _validate_element(element)
    if not 0.0 <= concentration <= 1.0:
        raise ValueError("concentration must be in [0, 1].")
    rng = make_rng(seed)
    carbons = _carbon_indices(atoms)
    n_sub = int(round(concentration * len(carbons)))
    if n_sub == 0:
        return atoms.copy()
    chosen = rng.choice(carbons, size=n_sub, replace=False)
    result = substitute_atoms(atoms, chosen.tolist(), element)
    result.info["doping_mode"] = "random"
    result.info["doping_seed"] = seed
    result.info["doping_concentration"] = concentration
    return result


def dope_directed(
    atoms: Atoms,
    element: str,
    where: Placement = "edges",
    count: int = 1,
    seed: Optional[int] = None,
    cluster_radius: float = 3.0,
) -> Atoms:
    """Place ``count`` dopants according to a structural criterion.

    Modes
    -----
    ``"edges"``
        Pick from atoms with coordination < 3 (ribbon edges, flake rims).
    ``"bulk"``
        Pick from fully sp2 atoms (coordination == 3).
    ``"cluster"``
        Pick one seed at random, then fill ``count - 1`` neighbours within
        ``cluster_radius``.

    Parameters
    ----------
    atoms
        Host structure.
    element
        Dopant symbol.
    where
        Placement strategy (see above).
    count
        Number of atoms to substitute.
    seed
        RNG seed.
    cluster_radius
        Only used when ``where="cluster"``.

    Returns
    -------
    ase.Atoms
        Doped structure.
    """
    _validate_element(element)
    if count <= 0:
        raise ValueError("count must be > 0.")
    rng = make_rng(seed)

    if where == "edges":
        pool = _edge_indices(atoms)
        if len(pool) == 0:
            raise ValueError("No edge atoms found (structure looks fully bonded).")
        chosen = rng.choice(pool, size=min(count, len(pool)), replace=False)
    elif where == "bulk":
        pool = _bulk_indices(atoms)
        if len(pool) == 0:
            raise ValueError("No bulk atoms found (all atoms under-coordinated).")
        chosen = rng.choice(pool, size=min(count, len(pool)), replace=False)
    elif where == "cluster":
        carbons = _carbon_indices(atoms)
        seed_atom = int(rng.choice(carbons))
        dmat = minimum_image_distances(atoms)
        neighbours = np.argsort(dmat[seed_atom])
        chosen = [int(i) for i in neighbours if i in carbons][:count]
        # ensure they are within cluster_radius
        chosen = [i for i in chosen if dmat[seed_atom, i] <= cluster_radius]
        if len(chosen) < count:
            warnings.warn(
                f"Cluster placement: only {len(chosen)} atoms within "
                f"{cluster_radius} Å of seed {seed_atom}.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        raise ValueError(f"Unknown placement mode: {where!r}")

    result = substitute_atoms(atoms, list(map(int, chosen)), element)
    result.info["doping_mode"] = where
    result.info["doping_seed"] = seed
    return result


def codope(
    atoms: Atoms,
    spec: Sequence[tuple[str, float]],
    seed: Optional[int] = None,
) -> Atoms:
    """Multi-element doping with independent random placement per species.

    Example
    -------
    >>> codope(sheet, [("N", 0.02), ("B", 0.02)], seed=42)

    Parameters
    ----------
    atoms
        Host structure.
    spec
        Sequence of ``(element, concentration)`` tuples, applied sequentially.
        Each concentration is relative to the **remaining carbon atoms**.
    seed
        Base RNG seed. Each species is seeded as ``seed + k``.

    Returns
    -------
    ase.Atoms
        Co-doped structure.
    """
    out = atoms
    base = 0 if seed is None else int(seed)
    for k, (element, conc) in enumerate(spec):
        sub_seed = None if seed is None else base + k
        out = dope_random(out, element, conc, seed=sub_seed)
    out.info["doping_mode"] = "codope"
    out.info["codoping_spec"] = list(spec)
    return out


def dope_at_sites(
    atoms: Atoms,
    element: str,
    count: int = 1,
    region: str = "basal",
    indices: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    min_separation: float = 2.5,
    avoid_radius: float = 2.6,
    avoid_occupied: bool = True,
    remove_edge_hydrogen: Optional[bool] = None,
) -> Atoms:
    """Substitute carbons with ``element`` at sites of a chosen kind.

    Where :func:`dope_random` picks any carbon, this picks from a **region**
    (see :data:`carbonforge.placement.REGIONS`: basal, edge, edge_zigzag,
    edge_armchair, pentagon, heptagon, defect_57, near_defect,
    vacancy_rim...) or from explicit ``indices``, keeping new dopants apart
    from each other and away from heteroatoms and groups already present.

    Parameters
    ----------
    atoms
        Host structure (not mutated).
    element
        N, B, S or P.
    count
        How many dopants. Ignored when ``indices`` is given.
    region
        Kind of site to draw from.
    indices
        Exact carbons to substitute; overrides ``region``, ``count`` and the
        distance constraints (an explicit choice is respected, and recorded).
    seed
        RNG seed.
    min_separation
        Minimum dopant-dopant distance, Å.
    avoid_radius
        Minimum distance from existing heteroatoms and attached groups, Å.
    avoid_occupied
        Apply ``avoid_radius``. Turn off to allow dopants next to groups.
    remove_edge_hydrogen
        For an H-terminated edge carbon: remove its H. Defaults to True for
        N (a pyridinic edge N has a lone pair, not an N-H) and False
        otherwise (B-H, P-H edges keep their termination).

    Returns
    -------
    ase.Atoms
        With ``info["dopants"]`` extended and ``info["doping_region"]`` set.
    """
    from ..placement import candidate_sites, choose_sites, classify_sites, occupied_atoms

    _validate_element(element)
    info = classify_sites(atoms)
    if indices is not None:
        chosen = [int(i) for i in indices]
        wrong = [i for i in chosen if i not in info or info[i].element != "C"]
        if wrong:
            raise ValueError(f"Los átomos {wrong} no son carbonos de la red.")
        region = "índices"
    else:
        candidates = candidate_sites(atoms, region, "C", info=info)
        avoid = occupied_atoms(atoms, info) if avoid_occupied else []
        chosen = choose_sites(atoms, candidates, count, seed=seed,
                              min_separation=min_separation, avoid=avoid,
                              avoid_radius=avoid_radius)

    if remove_edge_hydrogen is None:
        remove_edge_hydrogen = element == "N"
    out = substitute_atoms(atoms, chosen, element)
    removed_h: list[int] = []
    if remove_edge_hydrogen:
        removed_h = sorted({h for i in chosen for h in info[i].hydrogens}, reverse=True)
        for h in removed_h:
            del out[h]
        # Re-map the recorded dopant indices past the removed hydrogens.
        shift = lambda i: i - sum(1 for h in removed_h if h < i)  # noqa: E731
        out.info["dopants"][-1]["indices"] = [shift(i) for i in chosen]
    out.info["doping_region"] = region
    out.info["doping_seed"] = seed
    if removed_h:
        out.info["doping_removed_hydrogens"] = len(removed_h)
    return out
