"""Wythoff's kaleidoscope: uniform cages and tilings as superstructure graphs.

A uniform polyhedron is the orbit of one point under a reflection group.
Three mirrors meeting at angles ``pi/p``, ``pi/2`` and ``pi/q`` generate
the finite group ``[p, q]`` -- ``[3, 3]`` tetrahedral, ``[4, 3]``
octahedral, ``[5, 3]`` icosahedral -- and a choice of **ringed** mirrors
(the Coxeter-Dynkin diagram) fixes the generator: it lies *on* every
unringed mirror and at the same distance from every ringed one, which is
what makes every edge the same length. Reflecting it through the group
gives the vertices; the edges are the shortest pairs.

So the whole Platonic and Archimedean catalogue (the snubs aside, which
need the rotation subgroup) is seven ring patterns over three groups, and
no vertex table is written anywhere. The same construction over the
Euclidean triangle groups ``[6, 3]`` and ``[4, 4]`` gives the uniform
plane tilings, built here in an orthorhombic cell so that
``minimum_image`` can measure them.

Each graph is a :class:`~.supernetwork.SuperGraph`, so the geodesic
carrier (:mod:`.geodesic`) builds any of them as a superstructure of
tubes.
"""

from __future__ import annotations

import numpy as np

from .supernetwork import SuperGraph, edges_from_positions

#: Ring patterns in the order of the diagram's three nodes, with the
#: Coxeter name each produces from ``{p, q}``.
RINGS: dict[str, tuple[int, int, int]] = {
    "{p,q}": (1, 0, 0),        # regular
    "r{p,q}": (0, 1, 0),       # rectified
    "{q,p}": (0, 0, 1),        # dual
    "t{p,q}": (1, 1, 0),       # truncated
    "rr{p,q}": (1, 0, 1),      # cantellated (rhombi-)
    "t{q,p}": (0, 1, 1),       # truncated dual
    "tr{p,q}": (1, 1, 1),      # omnitruncated
}


def mirrors(p: int, q: int) -> np.ndarray:
    """Unit normals of the three mirrors of the spherical group ``[p, q]``.

    Mirrors 1-2 meet at ``pi/p``, 2-3 at ``pi/q`` and 1-3 at ``pi/2``, so
    the normals' pairwise dot products are ``-cos(pi/p)``, ``-cos(pi/q)``
    and ``0``.
    """
    n1 = np.array([1.0, 0.0, 0.0])
    n2 = np.array([-np.cos(np.pi / p), np.sin(np.pi / p), 0.0])
    a = -np.cos(np.pi / q) / np.sin(np.pi / p)
    if abs(a) >= 1.0:
        raise ValueError(f"[{p}, {q}] is not a finite (spherical) group.")
    n3 = np.array([0.0, a, np.sqrt(1.0 - a * a)])
    return np.array([n1, n2, n3])


def _reflect(x: np.ndarray, n: np.ndarray) -> np.ndarray:
    return x - 2.0 * np.dot(x, n) * n


def wythoff_points(p: int, q: int, rings: tuple[int, int, int],
                   with_maps: bool = False):
    """Vertices of the uniform polyhedron ``[p, q]`` with these rings.

    The generator solves ``n_i . x = rings_i`` (on the unringed mirrors,
    equidistant from the ringed ones); its orbit under the three
    reflections is the vertex set, normalised to the unit sphere. With
    ``with_maps`` it also returns, per vertex, the orthogonal matrix of
    one group element taking the generator there.
    """
    if not any(rings):
        raise ValueError("ring at least one mirror.")
    normals = mirrors(p, q)
    seed = np.linalg.solve(normals, np.asarray(rings, dtype=float))
    seed /= np.linalg.norm(seed)
    found = {tuple(np.round(seed, 9)): (seed, np.eye(3))}
    frontier = [(seed, np.eye(3))]
    reflections = [np.eye(3) - 2.0 * np.outer(n, n) for n in normals]
    while frontier:
        nxt = []
        for x, g in frontier:
            for r in reflections:
                y = r @ x
                key = tuple(np.round(y, 9))
                if key not in found:
                    found[key] = (y, r @ g)
                    nxt.append((y, r @ g))
        frontier = nxt
        if len(found) > 1000:
            raise ValueError("orbit did not close; not a finite group.")
    points = np.array([v[0] for v in found.values()])
    if with_maps:
        return points, np.array([v[1] for v in found.values()])
    return points


#: The named cages, as (p, q, rings). Platonic and Archimedean, by the
#: names supernetwork.CAGES uses where it has them.
WYTHOFF_SOLIDS: dict[str, tuple[int, int, tuple[int, int, int]]] = {
    "tetrahedron": (3, 3, (1, 0, 0)),
    "octahedron": (4, 3, (0, 0, 1)),
    "cube": (4, 3, (1, 0, 0)),
    "icosahedron": (5, 3, (0, 0, 1)),
    "dodecahedron": (5, 3, (1, 0, 0)),
    "cuboctahedron": (4, 3, (0, 1, 0)),
    "icosidodecahedron": (5, 3, (0, 1, 0)),
    "truncated-tetrahedron": (3, 3, (1, 1, 0)),
    "truncated-octahedron": (4, 3, (0, 1, 1)),
    "truncated-cube": (4, 3, (1, 1, 0)),
    "truncated-icosahedron": (5, 3, (0, 1, 1)),
    "truncated-dodecahedron": (5, 3, (1, 1, 0)),
    "rhombicuboctahedron": (4, 3, (1, 0, 1)),
    "rhombicosidodecahedron": (5, 3, (1, 0, 1)),
    "truncated-cuboctahedron": (4, 3, (1, 1, 1)),
    "truncated-icosidodecahedron": (5, 3, (1, 1, 1)),
}


def wythoff_cage(name: str, strut: float = 1.0) -> SuperGraph:
    """A uniform polyhedron as a cage graph with struts of length ``strut``.

    ``name`` is a key of :data:`WYTHOFF_SOLIDS`, with or without the
    ``"wythoff-"`` prefix the catalogues use.
    """
    key = name.removeprefix("wythoff-")
    if key not in WYTHOFF_SOLIDS:
        raise ValueError(f"unknown solid {name!r}; built here: "
                         f"{sorted(WYTHOFF_SOLIDS)}.")
    p, q, rings = WYTHOFF_SOLIDS[key]
    points = wythoff_points(p, q, rings)
    edges = edges_from_positions(points, (1.0, 1.0, 1.0),
                                 (False, False, False))
    length = float(np.linalg.norm(points[edges[0][1]] - points[edges[0][0]]))
    points = points * (strut / length)
    graph = SuperGraph(
        name=f"wythoff-{key}", nodes=points, edges=edges,
        pbc=(False, False, False),
        note=f"Wythoff [{p},{q}] with rings {rings}: {len(points)} "
             f"vertices, {len(edges)} edges.")
    _ = graph.coordination  # uniform by construction; raises if not
    return graph


def wythoff_frames(name: str, freq: int, starts: int = 32, seed: int = 0,
                   window: int = 6) -> tuple[np.ndarray, float]:
    """One node-sphere frame per vertex, carried by the group itself.

    The frame is fitted **once**, at the generator, to that vertex's own
    arms; every other vertex gets it transported by the group element
    that carries the generator there. So the carrier has the cage's whole
    symmetry -- mirrors included -- and no frame is left to a random fit.
    That is the cure for the mirror noise the srs builds measured (two
    rebuilds of mirror images 4 % apart): here a mirror image is exactly
    a mirror image, because a reflection maps an icosahedral frame onto
    an icosahedral frame.

    Returns ``(frames, worst_miss_deg)``, frames shaped ``(V, 12, 3)`` in
    the order of the cage's nodes.
    """
    from .geodesic import _fit_arms

    key = name.removeprefix("wythoff-")
    p, q, rings = WYTHOFF_SOLIDS[key]
    points, maps = wythoff_points(p, q, rings, with_maps=True)
    graph = WYTHOFF_CAGES[f"wythoff-{key}"]
    order = [int(np.argmin(np.linalg.norm(points - v / np.linalg.norm(v),
                                          axis=1)))
             for v in graph.nodes]
    start = order.index(0) if 0 in order else 0
    centre = graph.nodes[start]
    arms = []
    for i, j, _ in graph.edges:
        if i == start:
            arms.append(graph.nodes[j] - centre)
        elif j == start:
            arms.append(graph.nodes[i] - centre)
    arms = np.array([a / np.linalg.norm(a) for a in arms])
    frame, miss = _fit_arms(arms, freq, starts, seed, window)
    base = maps[order[start]]
    frames = np.array([(maps[k] @ base.T) @ frame.T for k in order])
    return np.transpose(frames, (0, 2, 1)), miss


#: Every Wythoff cage under the catalogue name the carriers look up.
WYTHOFF_CAGES: dict[str, SuperGraph] = {
    f"wythoff-{key}": wythoff_cage(key) for key in WYTHOFF_SOLIDS}
