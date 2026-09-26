"""Where a dopant or a functional group goes: site classes and controlled choice.

Every modification used to pick its sites on its own, blind to the others:
random doping chose any carbon, groups were kept apart from each other but
not from the dopants, and a vacancy rim counted as an "edge". The result was
a nitrogen right beside a hydroxyl more often than not. This module is the
one place that decides:

* :func:`classify_sites` tags each framework atom with what it is: a bare or
  H-terminated edge, of armchair or zigzag type (or a corner of both), a
  basal carbon, a member of a 5-, 7- or 8-membered ring (the Stone-Wales
  5-7 defect, a divacancy's 5-8-5), a vacancy rim, or next to a defect.
* :func:`candidate_sites` turns a **region** name into atom indices.
* :func:`choose_sites` picks from the candidates reproducibly (seeded),
  keeping a minimum distance between the new sites **and** from what is
  already there -- heteroatoms and attached groups.

Regions (the names the CLI and GUI use)
---------------------------------------
``any``, ``edge``, ``edge_armchair``, ``edge_zigzag``, ``basal``,
``pentagon``, ``heptagon``, ``defect_57``, ``near_defect``,
``vacancy_rim``.

The framework is the sp2 network: C plus substitutional N, B, S, P. Only H
counts as an edge termination; any other attached atom (O, N of an amine...)
marks the anchor as already functionalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np
from ase import Atoms

from .topology.graph import build_bond_graph
from .utils.rng import make_rng

FRAMEWORK: tuple[str, ...] = ("C", "N", "B", "S", "P")

REGIONS: dict[str, str] = {
    "any": "Cualquier carbono de la red",
    "edge": "Borde (suelto o terminado en H)",
    "edge_armchair": "Borde armchair",
    "edge_zigzag": "Borde zigzag",
    "basal": "Plano basal (3 vecinos de red, nada anclado)",
    "pentagon": "En un pentágono",
    "heptagon": "En un heptágono",
    "defect_57": "En un defecto 5-7 (pentágono o heptágono)",
    "near_defect": "Vecino de un defecto (anillo 5/7/8 o vacante)",
    "vacancy_rim": "Borde de una vacante",
}

#: Two outward directions closer than this to parallel are one zigzag edge.
_PARALLEL_COS = float(np.cos(np.radians(20.0)))


@dataclass
class SiteInfo:
    """What one framework atom is."""

    index: int
    element: str
    framework_neighbours: list[int]
    hydrogens: list[int]
    substituents: list[int]
    rings: list[int] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    outward: Optional[np.ndarray] = None


def _unit(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def _vector(atoms: Atoms, i: int, j: int) -> np.ndarray:
    """Minimum-image vector from atom ``i`` to atom ``j``."""
    return atoms.get_distance(i, j, mic=True, vector=True)


def classify_sites(atoms: Atoms) -> dict[int, SiteInfo]:
    """Tag every framework atom with its site classes.

    Returns
    -------
    dict
        Atom index -> :class:`SiteInfo`, for framework atoms only.
    """
    graph = build_bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()
    framework = [i for i, s in enumerate(symbols) if s in FRAMEWORK]
    info: dict[int, SiteInfo] = {}
    for i in framework:
        neighbours = list(graph.neighbors(i))
        info[i] = SiteInfo(
            index=i,
            element=symbols[i],
            framework_neighbours=[n for n in neighbours if symbols[n] in FRAMEWORK],
            hydrogens=[n for n in neighbours if symbols[n] == "H"],
            substituents=[n for n in neighbours if symbols[n] not in FRAMEWORK
                          and symbols[n] != "H"],
        )

    # Rings: chordless cycles of up to 8 atoms are the faces of an sp2 net
    # (a chord would mean two smaller fused rings, e.g. pentalene's 8-cycle).
    # A cycle whose bond vectors do not sum to zero wraps a periodic axis --
    # it is not a ring, however short the cell makes it.
    sub = graph.subgraph(framework)
    for cycle in nx.chordless_cycles(sub, length_bound=8):
        if len(cycle) < 3:
            continue
        closure = sum(_vector(atoms, cycle[k], cycle[(k + 1) % len(cycle)])
                      for k in range(len(cycle)))
        if np.linalg.norm(closure) > 1.0:
            continue
        for i in cycle:
            info[i].rings.append(len(cycle))

    positions = atoms.get_positions()
    for site in info.values():
        tags = site.tags
        if site.element != "C":
            tags.add("heteroatom")
        if site.substituents:
            tags.add("substituted")
        n_frame = len(site.framework_neighbours)
        if n_frame == 3 and not site.hydrogens and not site.substituents:
            tags.add("basal")
        if n_frame == 2 and not site.substituents and len(site.hydrogens) <= 1:
            if site.hydrogens:
                tags.add("terminated_edge")
                site.outward = _unit(_vector(atoms, site.index, site.hydrogens[0]))
            else:
                tags.add("bare_edge")
                total = sum(_vector(atoms, site.index, n) for n in site.framework_neighbours)
                site.outward = _unit(-total)
        for size, name in ((5, "pentagon"), (7, "heptagon"), (8, "octagon")):
            if size in site.rings:
                tags.add(name)
        if "pentagon" in tags or "heptagon" in tags:
            tags.add("defect_57")

    # A two-coordinated atom whose free valence points at lattice is on the
    # rim of a hole, not on the outer edge: step one bond outward and look.
    for site in info.values():
        if site.outward is None or "bare_edge" not in site.tags:
            continue
        probe = positions[site.index] + 1.42 * site.outward
        others = [j for j in info if j != site.index and j not in site.framework_neighbours]
        if others:
            from ase.geometry import get_distances

            _, distances = get_distances([probe], positions[others],
                                         cell=atoms.cell, pbc=atoms.pbc)
            if float(distances.min()) < 1.6:
                site.tags.discard("bare_edge")
                site.tags.add("vacancy_rim")
    for site in info.values():
        if site.tags & {"bare_edge", "terminated_edge"}:
            site.tags.add("edge")

    _tag_edge_types(atoms, info)

    defect = {i for i, s in info.items() if s.tags & {"defect_57", "octagon", "vacancy_rim"}}
    for i, site in info.items():
        if i not in defect and any(n in defect for n in site.framework_neighbours):
            site.tags.add("near_defect")
    return info


def _tag_edge_types(atoms: Atoms, info: dict[int, SiteInfo]) -> None:
    """Armchair or zigzag for every edge atom; a corner can be both.

    Zigzag: consecutive edge sites are second neighbours with parallel
    outward directions. Armchair: edge sites bonded to each other in pairs.
    An isolated edge atom with no edge neighbour is zigzag.
    """
    edge = {i for i, s in info.items() if "edge" in s.tags}
    for i in edge:
        site = info[i]
        paired = any(n in edge for n in site.framework_neighbours)
        zigzag = not paired
        for middle in site.framework_neighbours:
            if middle in edge:
                continue
            for other in info[middle].framework_neighbours:
                if other != i and other in edge and info[other].outward is not None \
                        and site.outward is not None \
                        and float(site.outward @ info[other].outward) > _PARALLEL_COS:
                    zigzag = True
        if paired:
            site.tags.add("edge_armchair")
        if zigzag:
            site.tags.add("edge_zigzag")
        if paired and zigzag:
            site.tags.add("corner")


def candidate_sites(
    atoms: Atoms,
    region: str = "any",
    element: str = "C",
    info: Optional[dict[int, SiteInfo]] = None,
    include_corners: bool = False,
) -> list[int]:
    """Atoms of ``element`` in ``region``, ordered by index.

    Parameters
    ----------
    atoms
        Structure.
    region
        One of :data:`REGIONS`.
    element
        Only atoms of this element (usually C: what gets substituted or
        decorated).
    info
        A precomputed :func:`classify_sites` result.
    include_corners
        For ``edge_armchair`` / ``edge_zigzag``, also return corner atoms,
        which belong to both edges. Off by default: a corner is rarely the
        site anyone means by "the zigzag edge".
    """
    if region not in REGIONS:
        raise ValueError(f"Región desconocida: '{region}'. Opciones: {', '.join(REGIONS)}.")
    info = info if info is not None else classify_sites(atoms)
    out = []
    for i, site in sorted(info.items()):
        if site.element != element:
            continue
        if region == "any":
            wanted = not site.substituents
        else:
            wanted = region in site.tags
        if wanted and region in ("edge_armchair", "edge_zigzag") and not include_corners:
            wanted = "corner" not in site.tags
        if wanted:
            out.append(i)
    return out


def occupied_atoms(atoms: Atoms, info: Optional[dict[int, SiteInfo]] = None) -> list[int]:
    """Atoms new modifications should keep away from: heteroatoms and groups.

    Substitutional heteroatoms in the framework, every attached non-H atom,
    and the carbons those groups hang from.
    """
    info = info if info is not None else classify_sites(atoms)
    symbols = atoms.get_chemical_symbols()
    occupied = {i for i, s in info.items() if s.element != "C" or s.substituents}
    occupied |= {i for i, s in enumerate(symbols) if s not in FRAMEWORK and s != "H"}
    return sorted(occupied)


def choose_sites(
    atoms: Atoms,
    candidates: Sequence[int],
    count: int,
    seed: Optional[int] = None,
    min_separation: float = 2.5,
    avoid: Iterable[int] = (),
    avoid_radius: float = 2.6,
) -> list[int]:
    """Pick ``count`` candidates, reproducibly, spaced out and clear of ``avoid``.

    Parameters
    ----------
    candidates
        Allowed atom indices (from :func:`candidate_sites`, or explicit).
    count
        How many to pick.
    seed
        RNG seed; the same inputs always give the same sites.
    min_separation
        Minimum distance between picked sites, Å. The default 2.5 Å keeps
        them off each other's first neighbours.
    avoid
        Atoms to keep away from, typically :func:`occupied_atoms`.
    avoid_radius
        Minimum distance from any ``avoid`` atom, Å. 2.6 Å excludes the
        first neighbours of a heteroatom or of a group's anchor.

    Raises
    ------
    ValueError
        If fewer than ``count`` sites satisfy the constraints; the message
        says how many do, so the constraint to relax is obvious.
    """
    from ase.geometry import get_distances

    if count <= 0:
        raise ValueError("count debe ser >= 1.")
    positions = atoms.get_positions()
    avoid = sorted(set(avoid))
    pool = [int(i) for i in candidates if int(i) not in avoid]
    if avoid and pool:
        _, d = get_distances(positions[pool], positions[avoid], cell=atoms.cell, pbc=atoms.pbc)
        pool = [i for i, row in zip(pool, d, strict=True) if row.min() >= avoid_radius]
    rng = make_rng(seed)
    chosen: list[int] = []
    for k in rng.permutation(len(pool)):
        candidate = pool[int(k)]
        if chosen and min_separation > 0:
            _, d = get_distances(positions[[candidate]], positions[chosen],
                                 cell=atoms.cell, pbc=atoms.pbc)
            if d.min() < min_separation:
                continue
        chosen.append(candidate)
        if len(chosen) == count:
            return sorted(chosen)
    raise ValueError(
        f"Solo {len(chosen)} sitio(s) cumplen las condiciones (de {len(candidates)} "
        f"candidatos; separación {min_separation} Å, a {avoid_radius} Å de "
        f"{len(avoid)} átomo(s) ya ocupados) y se pidieron {count}. Reduce la cantidad, "
        "la separación o la distancia de exclusión, o cambia de región."
    )


def parse_indices(text: str) -> list[int]:
    """``"3, 7 12-15"`` -> ``[3, 7, 12, 13, 14, 15]``; empty text -> ``[]``."""
    out: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token.strip("-"):
            low, high = token.split("-", 1)
            out.extend(range(int(low), int(high) + 1))
        else:
            out.append(int(token))
    return out


def plane_normal(atoms: Atoms, face: str = "+") -> Optional[np.ndarray]:
    """Unit normal of a flat structure, pointing to ``face``; ``None`` if curved.

    ``"+"`` is the side where the normal's largest Cartesian component is
    positive (for a ribbon in the x-z plane: +y). A structure whose
    thinnest principal extent exceeds 1 Å is not flat, and has no single
    face.
    """
    framework = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in FRAMEWORK]
    if len(framework) < 3:
        return None
    positions = atoms.get_positions()[framework]
    centred = positions - positions.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    if singular[-1] / np.sqrt(len(framework)) > 1.0:
        return None
    normal = vt[-1]
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    return normal if face == "+" else -normal
