"""Ring perception for structures that carry no ring metadata.

Every builder in this package **records** its rings, and
:mod:`nanocarbon_lab.builders.fullerene_mesh` exists because re-deriving
them from coordinates on a curved shell produced silently wrong counts.
Nothing here contradicts that. A file someone else wrote simply has no
recorded rings, so perception is the only option available -- and the
right response is to do it by a method whose failure modes are known, and
to say which they are.

There are two methods, and which one runs is reported.

**Face tracing** is used whenever the structure is a *surface*: a net in
which every atom has three neighbours, as graphene, every nanotube, every
fullerene, every schwarzite and every junction do. A ring is then not a
short cycle, it is a **face of the embedded graph**, and the faces can be
read off exactly. Ordering each atom's neighbours by angle about its
surface normal gives a rotation system, and the faces are the orbits of
"reverse the dart, then take the next neighbour round". Measured against
the builders' own recorded rings, this reproduces them exactly: a Y
junction's 5:28 / 6:451 / 7:16 and a Schwarz P cell's 5:41 / 6:457 /
7:65, with the face count matching ``E - V + 2 - 2g`` on the nose.

**Shortest-path rings** (King; Franzblau) are the fallback for everything
else -- a bulk crystal, an MX2 sandwich, a molecule -- where there is no
surface to trace faces on. For each bond, the smallest cycle containing
it, over **all** shortest paths and not one, since each bond of graphene
lies in two hexagons and following a single path would undercount the
sheet by half.

The distinction matters and is not pedantry. SP rings **systematically
miss large rings on a tiled surface**: a heptagon every one of whose
bonds also borders a hexagon is never the smallest ring through any bond,
so it is never emitted. On the Y junction above, SP rings find 4 of the
16 heptagons; on Schwarz P, 10 of the 65. Both are honest answers to the
question SP rings ask, and both are the wrong answer to "what tiles this
surface".

Neither is ``networkx.cycle_basis``, which returns *a* set of independent
cycles whose sizes depend on traversal order -- on a C60 it will happily
report rings of nine and ten members. A cycle basis answers "how many
independent loops", a question with a unique answer; ring statistics ask
what shapes tile the surface, which it cannot answer at all.

Two limits are reported rather than hidden:

* A cell small enough that an atom bonds to the **same neighbour through
  two different images** cannot be represented as a simple graph at all,
  and its ring sizes come out wrong however they are derived. That is
  detected and said.
* Under SP rings, a ring larger than ``max_size`` is not found, so a bond
  in one is reported as belonging to no ring rather than to a small one.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from ase import Atoms

#: Largest ring searched for. Ten covers every ring these materials
#: build -- a schwarzite's octagons, a junction's decagons -- and keeps
#: the bounded search local, which is what makes it fast: the walk stops
#: as soon as the far end of the bond is reached, so a graphitic sheet
#: costs a depth-5 traversal per bond and never approaches this bound.
MAX_RING_SIZE: int = 10


def _adjacency(atoms: Atoms, pairs: np.ndarray) -> list[list[int]]:
    neighbours: list[list[int]] = [[] for _ in range(len(atoms))]
    for first, second in pairs:
        neighbours[int(first)].append(int(second))
        neighbours[int(second)].append(int(first))
    return neighbours


def _shortest_rings_through(neighbours: list[list[int]],
                            start: int, goal: int,
                            max_size: int) -> list[tuple[int, ...]]:
    """Every shortest cycle containing the bond ``start``-``goal``.

    A breadth-first walk from ``start`` that may not use the bond
    itself, kept to the depth a ring of ``max_size`` allows. The whole
    level on which ``goal`` first appears is finished before stopping, so
    every shortest path is found and not merely the first.
    """
    limit = max_size - 1
    distance = {start: 0}
    parents: dict[int, list[int]] = {start: []}
    frontier = deque([start])
    found_at: int | None = None

    while frontier:
        current = frontier.popleft()
        step = distance[current] + 1
        if found_at is not None and step > found_at:
            break
        if step > limit:
            continue
        for other in neighbours[current]:
            if current == start and other == goal:
                continue  # the bond itself is the cycle's closing edge
            if other not in distance:
                distance[other] = step
                parents[other] = [current]
                frontier.append(other)
                if other == goal:
                    found_at = step
            elif distance[other] == step:
                parents[other].append(current)

    if found_at is None:
        return []

    rings: list[tuple[int, ...]] = []
    stack: list[tuple[int, list[int]]] = [(goal, [goal])]
    while stack:
        node, path = stack.pop()
        if node == start:
            rings.append(tuple(path))
            continue
        for parent in parents[node]:
            stack.append((parent, [*path, parent]))
    return rings


def perceive_rings(atoms: Atoms,
                   pairs: np.ndarray,
                   max_size: int = MAX_RING_SIZE) -> list[list[int]]:
    """Shortest-path rings of a structure, as lists of atom indices.

    Parameters
    ----------
    atoms
        Structure to analyse.
    pairs
        Its bond list, ``(n_bonds, 2)``.
    max_size
        Largest ring to look for. Bonds whose smallest ring is larger
        than this are reported in :func:`ring_report` as belonging to no
        ring, which is honest; calling them unringed would not be.

    Returns
    -------
    list of list of int
        Each ring once, ordered around the cycle.
    """
    neighbours = _adjacency(atoms, pairs)
    seen: set[frozenset[int]] = set()
    rings: list[list[int]] = []
    for first, second in pairs:
        for ring in _shortest_rings_through(neighbours, int(first),
                                            int(second), max_size):
            key = frozenset(ring)
            if len(key) == len(ring) and key not in seen:
                seen.add(key)
                rings.append(list(ring))
    return rings


def is_surface_net(atoms: Atoms, pairs: np.ndarray,
                   min_trivalent: float = 0.9) -> bool:
    """Whether the structure is a trivalent net, so faces can be traced.

    Graphene, every nanotube, every fullerene, every schwarzite and every
    junction are: each atom has exactly three neighbours and the net
    tiles a surface. A bulk crystal, an MX2 sandwich (six-coordinate
    metal) and a molecule are not, and face tracing means nothing there.

    A little slack, because a real file has edges: a graphene flake's rim
    atoms have two neighbours, and refusing to trace faces on it over a
    handful of boundary atoms would be pedantry.
    """
    if not len(pairs):
        return False
    degree = np.bincount(pairs.ravel(), minlength=len(atoms))
    bonded = degree[degree > 0]
    if not bonded.size:
        return False
    return float(np.mean(bonded == 3)) >= min_trivalent


def trace_faces(atoms: Atoms, pairs: np.ndarray,
                max_size: int = MAX_RING_SIZE) -> tuple[list[list[int]], int]:
    """Faces of the graph embedded in the surface it tiles.

    Each atom's neighbours are ordered by angle about its surface normal,
    giving a rotation system; the faces are the orbits of "reverse the
    dart, then take the next neighbour round". For a closed orientable
    surface this is exact, and the number of faces satisfies Euler by
    construction rather than by luck.

    Returns
    -------
    (faces, n_boundary)
        Faces no larger than ``max_size``, and the number of traced
        orbits discarded for being larger. On a closed structure that
        count is zero; on a flake or a ribbon it is the outer boundary,
        which is a legitimate orbit of the walk and not a ring.

    Notes
    -----
    Requires a **consistently oriented** normal field, which is what
    ``functionalize.attach.surface_normals`` propagates across the bond
    graph. With an inconsistent one the walk crosses itself and returns
    orbits of length one and two -- which is exactly what it did before
    the pyramidal threshold there was corrected, and is the reason that
    constant is measured rather than guessed.
    """
    from ..functionalize.attach import _mic_vectors, surface_normals

    normals = surface_normals(atoms)
    neighbours: list[list[int]] = [[] for _ in range(len(atoms))]
    offsets: list[list[np.ndarray]] = [[] for _ in range(len(atoms))]
    vectors = _mic_vectors(atoms, pairs)
    for (first, second), vector in zip(pairs, vectors, strict=True):
        first, second = int(first), int(second)
        neighbours[first].append(second)
        offsets[first].append(vector)
        neighbours[second].append(first)
        offsets[second].append(-vector)

    successor: dict[int, dict[int, int]] = {}
    for index in range(len(atoms)):
        if not neighbours[index]:
            continue
        normal = normals[index]
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, normal)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        first_axis = np.cross(normal, helper)
        first_axis /= np.linalg.norm(first_axis)
        second_axis = np.cross(normal, first_axis)
        angles = [float(np.arctan2(np.dot(vector, second_axis),
                                   np.dot(vector, first_axis)))
                  for vector in offsets[index]]
        ordered = [node for _, node in sorted(zip(angles, neighbours[index],
                                                  strict=True))]
        successor[index] = {node: ordered[(position + 1) % len(ordered)]
                            for position, node in enumerate(ordered)}

    faces: list[list[int]] = []
    boundary = 0
    used: set[tuple[int, int]] = set()
    limit = max(max_size, 4) * 8  # a runaway walk is a torn surface
    for start in range(len(atoms)):
        for step in neighbours[start]:
            if (start, step) in used:
                continue
            face: list[int] = []
            here, ahead = start, step
            while (here, ahead) not in used and len(face) <= limit:
                used.add((here, ahead))
                face.append(here)
                here, ahead = ahead, successor[ahead][here]
            if 3 <= len(face) <= max_size:
                faces.append(face)
            else:
                boundary += 1
    return faces, boundary


def collapsed_images(atoms: Atoms, tolerance: float = 0.30) -> int:
    """Bonds lost because a pair is bonded through more than one image.

    In a cell only one or two repeats wide, an atom reaches the same
    neighbour twice -- once directly and once around the cell. A simple
    graph holds one edge for that pair, so the second is gone and the
    ring sizes derived from the graph are wrong.

    This is not a hypothetical: a one-cell graphene primitive cell has
    two atoms and nine bonds between them, and any graph-based ring
    census on it is meaningless. Returning the count lets
    :func:`ring_report` say so instead of quoting a number that is not
    about the structure.
    """
    from ase.neighborlist import neighbor_list

    from ..utils.geometry import guess_bonds

    if not len(atoms) or not guess_bonds(atoms, tolerance=tolerance):
        return 0
    cutoffs = _cutoffs(atoms, tolerance)
    first, second, offsets = neighbor_list("ijS", atoms, cutoff=cutoffs)
    unique_pairs = set()
    directed = 0
    for i, j, offset in zip(first, second, offsets, strict=True):
        if i > j or (i == j and tuple(offset) < tuple(-offset)):
            continue
        directed += 1
        unique_pairs.add((int(i), int(j)))
    return max(0, directed - len(unique_pairs))


def _cutoffs(atoms: Atoms, tolerance: float) -> dict:
    """Per-pair bond cutoffs, matching ``utils.geometry.guess_bonds``."""
    from ..utils.constants import (
        BOND_CUTOFF_OVERRIDE,
        COVALENT_RADII,
        MAX_CC_DISTANCE,
    )

    symbols = sorted(set(atoms.get_chemical_symbols()))
    cutoffs = {}
    for first in symbols:
        for second in symbols:
            override = BOND_CUTOFF_OVERRIDE.get((first, second))
            if override is not None:
                cutoffs[(first, second)] = override
                continue
            if first in COVALENT_RADII and second in COVALENT_RADII:
                cutoffs[(first, second)] = (COVALENT_RADII[first]
                                            + COVALENT_RADII[second]
                                            + tolerance)
            else:
                cutoffs[(first, second)] = MAX_CC_DISTANCE
    return cutoffs


def ring_report(atoms: Atoms,
                pairs: np.ndarray,
                max_size: int = MAX_RING_SIZE) -> dict:
    """Ring census, plus what the census can and cannot be trusted for.

    Traces faces when the structure is a trivalent surface net and falls
    back to shortest-path rings otherwise. ``method`` says which ran,
    because the two answer different questions and only one of them is
    right for a tiled surface.

    Returns
    -------
    dict
        ``rings`` the perceived rings; ``counts`` the size census;
        ``euler_deficit`` ``sum(6 - n)``, the quantity that must equal
        ``6 * chi``; ``method``; ``reliable`` and
        ``collapsed_image_bonds``, which say whether the cell is wide
        enough for a graph-based census to mean anything at all; and,
        for the shortest-path method, ``n_unringed_bonds``.
    """
    collapsed = collapsed_images(atoms)
    if collapsed:
        return {
            "rings": [], "counts": {}, "n_rings": 0,
            "n_unringed_bonds": len(pairs), "euler_deficit": 0,
            "collapsed_image_bonds": collapsed, "reliable": False,
            "method": "none", "max_size_searched": max_size,
            "n_boundary_walks": 0,
            "caveat": (
                f"{collapsed} bond(s) join a pair of atoms through more than "
                "one periodic image, so the structure cannot be written as a "
                "simple graph and no ring census on it would describe it. "
                "Repeat the cell before asking for rings."
            ),
        }

    boundary = 0
    if is_surface_net(atoms, pairs):
        method = "faces"
        rings, boundary = trace_faces(atoms, pairs, max_size)
        caveat = ""
    else:
        method = "shortest-path"
        rings = perceive_rings(atoms, pairs, max_size)
        caveat = (
            "Not a trivalent surface net, so the rings are shortest-path "
            "rings rather than faces. Those systematically miss a large "
            "ring every one of whose bonds also borders a smaller one, so "
            "read the census as a lower bound on the big rings."
        )

    counts: dict[int, int] = {}
    for ring in rings:
        counts[len(ring)] = counts.get(len(ring), 0) + 1

    in_a_ring = {frozenset(pair) for ring in rings
                 for pair in zip(ring, [*ring[1:], ring[0]], strict=True)}
    unringed = sum(1 for first, second in pairs
                   if frozenset((int(first), int(second))) not in in_a_ring)

    if method == "faces" and boundary:
        caveat = (
            f"{boundary} traced walk(s) came out longer than {max_size} atoms "
            "and were not counted as rings. On an open structure that is the "
            "outer boundary, which is a legitimate orbit of the walk and not "
            "a ring; on a closed one it means the surface is torn."
        )

    return {
        "rings": rings,
        "counts": dict(sorted(counts.items())),
        "n_rings": len(rings),
        "n_unringed_bonds": int(unringed),
        "euler_deficit": int(sum(6 - len(ring) for ring in rings)),
        "collapsed_image_bonds": 0,
        "reliable": True,
        "method": method,
        "n_boundary_walks": boundary,
        "max_size_searched": max_size,
        "caveat": caveat,
    }


__all__ = [
    "MAX_RING_SIZE",
    "collapsed_images",
    "is_surface_net",
    "perceive_rings",
    "ring_report",
    "trace_faces",
]
