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


# --------------------------------------------------------------------------
# Euclidean kaleidoscopes: periodic nets
# --------------------------------------------------------------------------
#
# The cubic-family cages stack face to face and fill space, and the edge
# skeletons of those stackings are the uniform honeycombs: Wythoff's
# construction again, over the affine group [4,3,4] (four mirrors
# bounding a tetrahedron in the cube) instead of a spherical one. The
# plane's [4,4] and [6,3] give the uniform tilings. The icosahedral cages
# have no periodic version: five-fold symmetry does not tile space.
#
# Each group is a list of mirrors (inward unit normal, offset) bounding
# its fundamental simplex, in a cell that the group's translations tile:
# the unit cube for [4,3,4] and [4,4], the orthohexagonal (1, sqrt 3)
# rectangle for [6,3].

_S2 = np.sqrt(2.0)
_S3 = np.sqrt(3.0)

#: group -> (mirrors as (normal, offset) with n.x + d >= 0 inside, cell,
#: periodic axes). Mirrors are listed along the Coxeter diagram.
EUCLIDEAN_GROUPS: dict[str, tuple[list[tuple[np.ndarray, float]],
                                  tuple[float, float, float],
                                  tuple[bool, bool, bool]]] = {
    # [4,3,4]: tetrahedron (0,0,0), (1/2,0,0), (1/2,1/2,0), (1/2,1/2,1/2).
    "[4,3,4]": ([(np.array([-1.0, 0.0, 0.0]), 0.5),
                 (np.array([1.0, -1.0, 0.0]) / _S2, 0.0),
                 (np.array([0.0, 1.0, -1.0]) / _S2, 0.0),
                 (np.array([0.0, 0.0, 1.0]), 0.0)],
                (1.0, 1.0, 1.0), (True, True, True)),
    # [4,4]: triangle (0,0), (1/2,0), (1/2,1/2).
    "[4,4]": ([(np.array([-1.0, 0.0, 0.0]), 0.5),
               (np.array([1.0, -1.0, 0.0]) / _S2, 0.0),
               (np.array([0.0, 1.0, 0.0]), 0.0)],
              (1.0, 1.0, 1.0), (True, True, False)),
    # [6,3]: triangle (0,0) [six-fold], (1/2,0) [two-fold],
    # (1/2, sqrt3/6) [three-fold], on the unit triangular lattice.
    "[6,3]": ([(np.array([0.0, 1.0, 0.0]), 0.0),
               (np.array([1.0, -_S3, 0.0]) / 2.0, 0.0),
               (np.array([-1.0, 0.0, 0.0]), 0.5)],
              (1.0, _S3, 1.0), (True, True, False)),
}


def euclidean_wythoff(group: str, rings: tuple[int, ...]) -> tuple:
    """Vertices (fractional) and cell of a uniform tiling or honeycomb.

    The generator lies on every unringed mirror and at one distance from
    every ringed one; its orbit under the reflections, folded into the
    group's cell, is the vertex set. Returns ``(fractional_nodes, shape,
    pbc)`` for a :class:`~.supernetwork.SuperGraph`.
    """
    mirrors, shape, pbc = EUCLIDEAN_GROUPS[group]
    if len(rings) != len(mirrors) or not any(rings):
        raise ValueError(f"{group} takes {len(mirrors)} rings, one at least "
                         "set.")
    live = [k for k in range(3) if pbc[k]]
    dim = len(live)
    # Solve n_i . x + d_i = rings_i * t in the live coordinates.
    a = np.zeros((len(mirrors), dim + 1))
    b = np.zeros(len(mirrors))
    for i, (normal, offset) in enumerate(mirrors):
        a[i, :dim] = normal[live]
        a[i, dim] = -float(rings[i])
        b[i] = -offset
    solution = np.linalg.solve(a, b)
    seed = np.zeros(3)
    seed[live] = solution[:dim]
    box = np.array(shape)

    # The orbit is grown unfolded inside a window a few cells wide and
    # only then cut to the cell. Folding after every reflection is wrong
    # for [6,3]: its rotations do not preserve the orthohexagonal cell's
    # translations ((1, 0) turns into (1/2, sqrt3/2)), and the folded
    # orbit came out with half its points.
    low = -2.0 * box
    high = 3.0 * box

    def key(x):
        return tuple(np.round(x[live], 7))

    found = {key(seed): seed}
    frontier = [seed]
    while frontier:
        nxt = []
        for x in frontier:
            for normal, offset in mirrors:
                y = x - 2.0 * (np.dot(normal, x) + offset) * normal
                if np.any(y[live] < low[live]) or np.any(y[live] > high[live]):
                    continue
                k_ = key(y)
                if k_ not in found:
                    found[k_] = y
                    nxt.append(y)
        frontier = nxt
        if len(found) > 200000:
            raise ValueError("orbit did not close in the window.")
    inside = {}
    for x in found.values():
        y = x.copy()
        if np.all(y[live] >= -1e-9) and np.all(y[live] < box[live] - 1e-9):
            y[live] = np.clip(y[live], 0.0, None)
            inside[key(y)] = y
    found = inside
    nodes = np.array(list(found.values())) / box
    nodes[:, [k for k in range(3) if not pbc[k]]] = 0.5
    return nodes, tuple(shape), pbc


def _euclidean_names() -> dict[str, tuple[str, tuple[int, ...]]]:
    """Every distinct ring pattern of the three groups, by its own name."""
    from itertools import product

    names = {}
    for group, (mirrors, _, _) in EUCLIDEAN_GROUPS.items():
        for rings in product((0, 1), repeat=len(mirrors)):
            if any(rings):
                code = "".join(str(r) for r in rings)
                names[f"wythoff-{group}-{code}"] = (group, rings)
    return names


def euclidean_net(name: str) -> SuperGraph:
    """A uniform tiling or honeycomb as a periodic superstructure graph."""
    from .supernetwork import _periodic_graph

    group, rings = _euclidean_names()[name]
    nodes, shape, pbc = euclidean_wythoff(group, rings)
    graph = _periodic_graph(name, nodes, shape, pbc,
                            f"Wythoff {group} rings {rings}.")
    _ = graph.coordination
    return graph


#: The distinct Euclidean Wythoff nets the other catalogues do not
#: already hold, by name -> (group, rings). Cubic (1000), square (100),
#: honeycomb ([6,3] 100) and kagome ([6,3] 010) are super-cubic,
#: super-square, super-graphene and super-kagome already; the rest of
#: the patterns repeat these (0001 is 1000 seen from the other end).
#: Each 3D honeycomb is the periodic version of the cages it is made of.
WYTHOFF_NET_PATTERNS: dict[str, tuple[str, tuple[int, ...], str]] = {
    "wythoff-rectified-cubic": (
        "[4,3,4]", (0, 1, 0, 0), "octahedra and cuboctahedra, 8 arms"),
    "wythoff-truncated-cubic": (
        "[4,3,4]", (1, 1, 0, 0), "truncated cubes and octahedra"),
    "wythoff-bitruncated-cubic": (
        "[4,3,4]", (0, 1, 1, 0),
        "truncated octahedra face to face: the sodalite net"),
    "wythoff-cantellated-cubic": (
        "[4,3,4]", (1, 0, 1, 0),
        "rhombicuboctahedra, cuboctahedra and cubes"),
    "wythoff-cantitruncated-cubic": (
        "[4,3,4]", (1, 1, 1, 0),
        "truncated cuboctahedra, truncated octahedra and cubes"),
    "wythoff-runcitruncated-cubic": (
        "[4,3,4]", (1, 1, 0, 1),
        "truncated cubes, rhombicuboctahedra, octagonal prisms, cubes"),
    "wythoff-omnitruncated-cubic": (
        "[4,3,4]", (1, 1, 1, 1),
        "truncated cuboctahedra and octagonal prisms"),
    "wythoff-truncated-square": ("[4,4]", (0, 1, 1), "4.8.8 tiling"),
    "wythoff-triangular": ("[6,3]", (0, 0, 1), "3^6 tiling, 6 arms"),
    "wythoff-truncated-hexagonal": ("[6,3]", (0, 1, 1), "3.12.12 tiling"),
    "wythoff-rhombitrihexagonal": ("[6,3]", (1, 0, 1), "3.4.6.4 tiling"),
    "wythoff-truncated-trihexagonal": ("[6,3]", (1, 1, 1), "4.6.12 tiling"),
}


def wythoff_net(name: str) -> SuperGraph:
    """A named uniform honeycomb or tiling as a periodic graph."""
    from .supernetwork import _periodic_graph

    group, rings, note = WYTHOFF_NET_PATTERNS[name]
    nodes, shape, pbc = euclidean_wythoff(group, rings)
    graph = _periodic_graph(name, nodes, shape, pbc,
                            f"Wythoff {group} {rings}: {note}.")
    _ = graph.coordination
    return graph


#: Every named Euclidean Wythoff net, built once.
WYTHOFF_NETS: dict[str, SuperGraph] = {
    name: wythoff_net(name) for name in WYTHOFF_NET_PATTERNS}
