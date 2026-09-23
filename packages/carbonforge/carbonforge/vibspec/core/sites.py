"""Choosing *where* on a flake a functionalisation goes, deterministically.

For band assignment the position of a group is part of the question: an amine
on an armchair edge and one on a zigzag edge are different molecules with
different spectra. So every choice here is reproducible -- named positions
("middle", "center", "near_edge") or explicit atom indices -- and never a
random draw.

The edge type of a C-H is read from the structure, not from the builder's
label, so it survives a round trip through a file. Connectivity alone is not
enough: at a corner the last carbon of a zigzag edge is also bonded to the
first C-H of the armchair end, exactly like an armchair pair. What does tell
them apart is the direction of the C-H bonds. Along a **zigzag** edge,
second-neighbour C-H bonds are parallel; across an **armchair** valley they
are 60 degrees apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
from ase import Atoms

from ...topology.graph import build_bond_graph

#: Two C-H bonds closer than this to parallel count as the same zigzag edge.
_PARALLEL_COS = float(np.cos(np.radians(20.0)))

EdgeKind = Literal["armchair", "zigzag"]
EdgePosition = Union[Literal["middle"], int]
InteriorPosition = Union[Literal["center", "near_edge"], int]

#: Atoms that make up the sp2 framework, as opposed to terminations.
_FRAMEWORK = ("C", "N", "B")


@dataclass(frozen=True)
class EdgeSite:
    """A terminated edge carbon: the carbon, its hydrogen and the edge type."""

    carbon: int
    hydrogen: int
    edge: EdgeKind


def _framework_neighbours(graph, symbols, index: int) -> list[int]:
    return [n for n in graph.neighbors(index) if symbols[n] in _FRAMEWORK]


def _edge_analysis(atoms: Atoms) -> tuple[dict[int, int], list[tuple[int, int]], set[int]]:
    """Terminated edge carbons, the zigzag links between them, and lone C-H.

    Returns ``(hydrogen_of, links, lone)``: the H of every singly terminated
    edge carbon; pairs of such carbons that are second neighbours with
    parallel C-H bonds (consecutive sites of one zigzag edge); and those
    with no terminated carbon neighbour at all.
    """
    graph = build_bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()
    hydrogen_of: dict[int, int] = {}
    for index, symbol in enumerate(symbols):
        if symbol != "C":
            continue
        hydrogens = [n for n in graph.neighbors(index) if symbols[n] == "H"]
        if len(hydrogens) == 1 and len(_framework_neighbours(graph, symbols, index)) == 2:
            hydrogen_of[index] = hydrogens[0]

    positions = atoms.get_positions()
    bond_dir = {}
    for carbon, hydrogen in hydrogen_of.items():
        vector = positions[hydrogen] - positions[carbon]
        bond_dir[carbon] = vector / np.linalg.norm(vector)

    links: set[tuple[int, int]] = set()
    lone: set[int] = set()
    for carbon in hydrogen_of:
        neighbours = _framework_neighbours(graph, symbols, carbon)
        if not any(n in hydrogen_of for n in neighbours):
            lone.add(carbon)
        for middle in neighbours:
            if middle in hydrogen_of:
                continue
            for other in _framework_neighbours(graph, symbols, middle):
                if (
                    other != carbon
                    and other in hydrogen_of
                    and float(bond_dir[carbon] @ bond_dir[other]) > _PARALLEL_COS
                ):
                    links.add((min(carbon, other), max(carbon, other)))
    return hydrogen_of, sorted(links), lone


def edge_sites(atoms: Atoms) -> list[EdgeSite]:
    """Every carbon that carries exactly one hydrogen, with its edge type.

    A carbon at a corner belongs to both edges; it is labelled zigzag when
    it continues a zigzag run. The ``"middle"`` site of a long edge, which
    is what the presets use by default, is never a corner.
    """
    hydrogen_of, links, lone = _edge_analysis(atoms)
    zigzag = lone | {i for pair in links for i in pair}
    return [
        EdgeSite(carbon, hydrogen, "zigzag" if carbon in zigzag else "armchair")
        for carbon, hydrogen in sorted(hydrogen_of.items())
    ]


def zigzag_runs(atoms: Atoms) -> list[list[int]]:
    """Terminated carbons grouped into continuous zigzag edges, longest first.

    The length of a run is what decides whether an edge can carry the
    localised, spin-polarised edge state: a two-site zigzag step at the
    corner of an armchair flake cannot, a zigzag end four or more sites long
    can.
    """
    hydrogen_of, links, lone = _edge_analysis(atoms)
    parent = {c: c for c in hydrogen_of}

    def root(c: int) -> int:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for a, b in links:
        parent[root(a)] = root(b)
    members = lone | {i for pair in links for i in pair}
    runs: dict[int, list[int]] = {}
    for carbon in sorted(members):
        runs.setdefault(root(carbon), []).append(carbon)
    return sorted(runs.values(), key=lambda run: (-len(run), run[0]))


def interior_carbons(atoms: Atoms) -> list[int]:
    """Carbons bonded to three framework atoms and nothing else."""
    graph = build_bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()
    return [
        index
        for index, symbol in enumerate(symbols)
        if symbol == "C"
        and graph.degree[index] == 3
        and len(_framework_neighbours(graph, symbols, index)) == 3
    ]


def _framework_centre(atoms: Atoms) -> np.ndarray:
    symbols = atoms.get_chemical_symbols()
    mask = [s in _FRAMEWORK for s in symbols]
    return atoms.get_positions()[mask].mean(axis=0)


def pick_edge_site(
    atoms: Atoms,
    position: EdgePosition = "middle",
    edge: Optional[EdgeKind] = None,
) -> EdgeSite:
    """Choose one terminated edge carbon.

    Parameters
    ----------
    atoms
        Terminated flake.
    position
        ``"middle"`` picks the site of the requested edge type closest to the
        centre of the flake, i.e. the middle of a long edge, as far as
        possible from the corners. An integer picks that carbon.
    edge
        Restrict ``"middle"`` to this edge type. Defaults to the ribbon's
        own edge (``atoms.info["edge"]``) when known.
    """
    sites = edge_sites(atoms)
    if not sites:
        raise ValueError("La estructura no tiene carbonos de borde terminados en H.")

    if not isinstance(position, str):
        for site in sites:
            if site.carbon == int(position):
                return site
        raise ValueError(
            f"El átomo {position} no es un carbono de borde con un H "
            f"(sitios válidos: {[s.carbon for s in sites]})."
        )
    if position != "middle":
        raise ValueError(f"Posición de borde desconocida: '{position}'.")

    wanted = edge or atoms.info.get("edge")
    candidates = [s for s in sites if wanted is None or s.edge == wanted]
    if not candidates:
        raise ValueError(f"No hay bordes '{wanted}' terminados en H en la estructura.")
    centre = _framework_centre(atoms)
    positions = atoms.get_positions()
    return min(
        candidates,
        key=lambda s: (round(float(np.linalg.norm(positions[s.carbon] - centre)), 6), s.carbon),
    )


def pick_interior_carbon(
    atoms: Atoms,
    position: InteriorPosition = "center",
    edge: Optional[EdgeKind] = None,
) -> int:
    """Choose one interior carbon.

    ``"center"`` is the one closest to the centre of the framework, as far
    from every edge as the flake allows. ``"near_edge"`` is the interior
    neighbour of the ``"middle"`` edge site: the same substitution, one bond
    in from the edge. An integer picks that carbon.
    """
    interior = interior_carbons(atoms)
    if not interior:
        raise ValueError("La estructura no tiene carbonos interiores (coordinación 3 sin H).")

    if not isinstance(position, str):
        if int(position) not in interior:
            raise ValueError(f"El átomo {position} no es un carbono interior.")
        return int(position)

    positions = atoms.get_positions()
    if position == "center":
        centre = _framework_centre(atoms)
        return min(
            interior,
            key=lambda i: (round(float(np.linalg.norm(positions[i] - centre)), 6), i),
        )
    if position == "near_edge":
        site = pick_edge_site(atoms, "middle", edge=edge)
        graph = build_bond_graph(atoms)
        neighbours = sorted(n for n in graph.neighbors(site.carbon) if n in interior)
        if not neighbours:
            raise ValueError("El sitio de borde elegido no tiene vecinos interiores.")
        return neighbours[0]
    raise ValueError(f"Posición interior desconocida: '{position}'.")


def strip_hydrogen(atoms: Atoms, site: EdgeSite) -> tuple[Atoms, int]:
    """Remove a site's hydrogen; return the new structure and the carbon's new index."""
    out = atoms.copy()
    out.info = {**atoms.info}
    del out[site.hydrogen]
    carbon = site.carbon - (1 if site.hydrogen < site.carbon else 0)
    return out, carbon
