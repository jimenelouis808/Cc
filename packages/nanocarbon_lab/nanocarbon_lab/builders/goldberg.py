"""Goldberg-Coxeter superstructures: flat graphene patches, defects at vertices.

The knee route (:mod:`.knee`) assembles a node from straight arms and pays
its curvature in heptagons; the implicit route (:mod:`.junction`,
:mod:`.supernetwork`) lets a remesher choose the rings and gets an
amorphous wall. This module is the third route, and it is the one the
schwarzite literature has used since Mackay & Terrones (*Nature* **352**,
762, 1991): take a COARSE triangulated map of the surface whose vertices
already carry the ring sizes the topology needs, and fill every coarse
triangle with a flat patch of the triangular lattice. The dual of the
result is graphene everywhere except at the coarse vertices, so

* the defects are exactly the coarse map's non-six vertices, no more;
* they are isolated, ``m`` rings apart for a ``(m, 0)`` subdivision;
* the size is a free integer, not a list of shapes that happen to close.

This is Koshino & Aoki's construction of P-surface graphene (*Phys. Rev.
B* **93**, 041412(R), 2016: hexagonal graphene patches, one octagon where
four meet), and the Goldberg-Coxeter construction that turns an
icosahedron into C60 -- the same operation :mod:`.fullerene_mesh` performs
on the sphere -- carried onto a periodic surface of negative curvature.

Subdivision ``(m, n)`` is supported for ``n == 0`` (class I, each triangle
into ``m**2``) and ``n == m`` (class II, the root-three step composed with
class I). Chiral ``(m, n)`` would need a consistent cut of the triangular
lattice across every coarse edge and is not implemented.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND

Mesh = tuple[np.ndarray, np.ndarray]


def _offset(vertices: np.ndarray, start: int, end: int,
            box: np.ndarray | None) -> np.ndarray:
    """Vector from vertex ``start`` to vertex ``end`` under the minimum image."""
    from .fullerene_mesh import minimum_image

    return minimum_image(vertices[end] - vertices[start], box)


def _wrap(points: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Put points back in the cell on the live axes; identity for ``None``."""
    if box is None:
        return points
    edges = np.broadcast_to(np.asarray(box, dtype=float), (3,))
    out = np.array(points, dtype=float, copy=True)
    live = edges > 0.0
    out[:, live] = np.mod(out[:, live], edges[live])
    return out


_OMEGA = complex(0.5, np.sqrt(3.0) / 2.0)


def _to_eisenstein(z: complex) -> tuple[int, int]:
    """Nearest Eisenstein integer ``a + b*omega`` to ``z``, as ``(a, b)``."""
    b = z.imag / _OMEGA.imag
    return int(round(z.real - 0.5 * b)), int(round(b))


def _from_eisenstein(a: int, b: int) -> complex:
    return a + b * _OMEGA


def orient_mesh(tris) -> list[tuple[int, int, int]]:
    """Make a closed triangulation's winding consistent, face by face.

    Breadth-first over shared edges: a neighbour that runs the shared edge
    the same way round as the face already fixed is reversed. Raises if
    the surface is not orientable (an edge is met in the same sense from
    two faces both already fixed). The winding of the first face sets the
    handedness of everything, which matters only for chiral GC.
    """
    from collections import deque

    tris = [list(map(int, t)) for t in tris]
    by_edge: dict[tuple[int, int], list[int]] = {}
    for f, t in enumerate(tris):
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            by_edge.setdefault((min(a, b), max(a, b)), []).append(f)
    done = [False] * len(tris)
    for seed in range(len(tris)):
        if done[seed]:
            continue
        done[seed] = True
        queue = deque([seed])
        while queue:
            f = queue.popleft()
            t = tris[f]
            for k in range(3):
                a, b = t[k], t[(k + 1) % 3]
                for g in by_edge[(min(a, b), max(a, b))]:
                    if g == f:
                        continue
                    u = tris[g]
                    same = any(u[j] == a and u[(j + 1) % 3] == b
                               for j in range(3))
                    if done[g]:
                        if same:
                            raise ValueError("the surface is not orientable.")
                        continue
                    if same:
                        tris[g] = u[::-1]
                    done[g] = True
                    queue.append(g)
    return [tuple(t) for t in tris]


def chiral_subdivide(mesh: Mesh, m: int, n: int,
                     box: np.ndarray | None = None) -> Mesh:
    """Goldberg-Coxeter ``(m, n)`` of a closed, oriented triangulation.

    Every coarse triangle is laid in the Eisenstein lattice with its
    corners at ``0``, ``w`` and ``w*omega`` (``w = m + n*omega``), and the
    lattice's own triangles are given to the coarse face their centroid
    lies in. For ``n = 0`` (or ``m = 0``) the coarse edges run along
    lattice lines and this is :func:`subdivide`; for a chiral pair they
    cut through lattice triangles, which is why chiral GC needed a
    consistent cut. Here it is: a lattice point outside its face is
    carried into the neighbour across the edge it left by, by the rotation
    that unfolds the neighbour onto this face's plane -- a sixth root of
    unity about the shared edge, so lattice points map to lattice points
    and both faces agree on every vertex they share. A centroid exactly
    on a coarse edge (it happens, e.g. at ``(1, 1)``) goes to the face
    that runs that edge from its lower vertex index to its higher.

    Every new vertex has degree six and every coarse vertex keeps its
    degree, so the dual's census is the coarse map's, ``T * (m^2 + mn +
    n^2)`` atoms for ``T`` coarse triangles, and the handedness is
    ``(m, n)`` against ``(n, m)``.

    Parameters
    ----------
    mesh
        ``(vertices, triangles)``, closed and consistently oriented.
    m, n
        Goldberg-Coxeter indices, ``m >= 1``, ``n >= 0``.
    box
        As for :func:`subdivide`.

    Returns
    -------
    Mesh
    """
    if m < 1 or n < 0:
        raise ValueError("need m >= 1 and n >= 0.")
    if n == 0:
        return subdivide(mesh, m, box)
    vertices, tris = np.asarray(mesh[0], dtype=float), np.asarray(mesh[1])
    tris = [tuple(int(x) for x in t) for t in tris]
    w = complex(m) + n * _OMEGA
    corners = (0j, w, w * _OMEGA)
    norm = m * m + m * n + n * n
    face_of: dict[tuple[int, int], tuple[int, int]] = {}
    for f, t in enumerate(tris):
        for k in range(3):
            face_of[(t[k], t[(k + 1) % 3])] = (f, k)
    if len(face_of) != 3 * len(tris):
        tris = orient_mesh(tris)
        face_of = {}
        for f, t in enumerate(tris):
            for k in range(3):
                face_of[(t[k], t[(k + 1) % 3])] = (f, k)
        if len(face_of) != 3 * len(tris):
            raise ValueError("the coarse map could not be oriented.")
    # Barycentric coordinates in the common local frame.
    frame = np.array([[corners[1].real, corners[2].real],
                      [corners[1].imag, corners[2].imag]])
    inverse = np.linalg.inv(frame)
    eps = 1e-9

    def bary(z: complex) -> np.ndarray:
        l1, l2 = inverse @ np.array([z.real, z.imag])
        return np.array([1.0 - l1 - l2, l1, l2])

    def canon(f: int, z: complex) -> tuple:
        for _ in range(32):
            lam = bary(z)
            if lam.min() >= -eps:
                break
            k = int(np.argmin(lam))           # outside the edge opposite k
            a_loc, b_loc = (k + 1) % 3, (k + 2) % 3
            ga, gb = tris[f][a_loc], tris[f][b_loc]
            g, j = face_of[(gb, ga)]          # neighbour runs gb -> ga
            pa, pb = corners[a_loc], corners[b_loc]
            qb, qa = corners[j], corners[(j + 1) % 3]
            z = qa + (z - pa) * (qb - qa) / (pb - pa)
            z = _from_eisenstein(*_to_eisenstein(z))
            f = g
        else:
            raise ValueError("a lattice point did not settle in a face.")
        t = tris[f]
        for k in range(3):
            if abs(z - corners[k]) < 1e-6:
                return ("v", t[k])
        for k in range(3):
            if abs(lam[k]) < eps:
                a_loc, b_loc = (k + 1) % 3, (k + 2) % 3
                lo, hi = sorted((a_loc, b_loc), key=lambda c: t[c])
                along = ((z - corners[lo]) / (corners[hi] - corners[lo])).real
                return ("e", t[lo], t[hi], int(round(along * norm)))
        return ("f", f) + _to_eisenstein(z)

    ids: dict[tuple, int] = {}
    points: list[np.ndarray] = []

    def place(key: tuple, f: int, z: complex) -> int:
        if key in ids:
            return ids[key]
        if key[0] == "v":
            point = vertices[key[1]]
        elif key[0] == "e":
            lo, hi, step = key[1], key[2], key[3]
            point = vertices[lo] + _offset(vertices, lo, hi, box) * step / norm
        else:
            ff = key[1]
            t = tris[ff]
            lam = bary(_from_eisenstein(key[2], key[3]))
            point = (vertices[t[0]] + lam[1] * _offset(vertices, t[0], t[1], box)
                     + lam[2] * _offset(vertices, t[0], t[2], box))
        ids[key] = len(points)
        points.append(point)
        return ids[key]

    span = int(np.ceil(abs(w))) + 2
    out: list[tuple[int, int, int]] = []
    for f, t in enumerate(tris):
        for a in range(-span, 2 * span):
            for b in range(-span, 2 * span):
                z = _from_eisenstein(a, b)
                for tri in ((z, z + 1, z + _OMEGA),
                            (z + 1, z + 1 + _OMEGA, z + _OMEGA)):
                    lam = bary(sum(tri) / 3.0)
                    if lam.min() < -eps:
                        continue
                    if lam.min() < eps:
                        k = int(np.argmin(lam))
                        ga, gb = t[(k + 1) % 3], t[(k + 2) % 3]
                        if ga > gb:
                            continue
                    keys = [canon(f, p) for p in tri]
                    out.append(tuple(place(key, f, p)
                                     for key, p in zip(keys, tri, strict=True)))
    expected = len(tris) * norm
    if len(out) != expected:
        raise ValueError(f"GC({m},{n}) produced {len(out)} triangles, "
                         f"expected {expected}.")
    return _wrap(np.array(points), box), np.array(out, dtype=int)


def subdivide(mesh: Mesh, m: int, box: np.ndarray | None = None) -> Mesh:
    """Class-I Goldberg-Coxeter step: every triangle into ``m**2``.

    New vertices are placed linearly inside each coarse triangle, so a flat
    coarse face stays flat -- it becomes a patch of the exact triangular
    lattice, whose dual is a patch of exact graphene. Every new vertex has
    degree six; the coarse vertices keep their degree, and those are the
    only rings in the dual that are not hexagons.

    Parameters
    ----------
    mesh
        ``(vertices, triangles)``, closed and consistently oriented.
    m
        Subdivision frequency, ``>= 1``.
    box
        Orthorhombic cell edges for a periodic mesh (``0`` = not periodic
        on that axis), else ``None``.

    Returns
    -------
    Mesh
        The refined mesh, same orientation, positions wrapped into the cell.
    """
    if m < 1:
        raise ValueError("the subdivision frequency m must be >= 1.")
    vertices, tris = np.asarray(mesh[0], dtype=float), np.asarray(mesh[1])
    if m == 1:
        return vertices.copy(), tris.copy()
    points: list[np.ndarray] = [v for v in vertices]
    edge_points: dict[tuple[int, int], list[int]] = {}

    def edge_chain(a: int, b: int) -> list[int]:
        """Indices along a -> b, endpoints included, shared with b -> a."""
        key = (min(a, b), max(a, b))
        if key not in edge_points:
            lo, hi = key
            step = _offset(vertices, lo, hi, box) / m
            inner = []
            for k in range(1, m):
                points.append(vertices[lo] + k * step)
                inner.append(len(points) - 1)
            edge_points[key] = [lo] + inner + [hi]
        chain = edge_points[key]
        return chain if a == key[0] else chain[::-1]

    new_tris: list[tuple[int, int, int]] = []
    for a, b, c in tris:
        a, b, c = int(a), int(b), int(c)
        ab = _offset(vertices, a, b, box) / m
        ac = _offset(vertices, a, c, box) / m
        grid: dict[tuple[int, int], int] = {}
        side_ab, side_ac, side_bc = edge_chain(a, b), edge_chain(a, c), \
            edge_chain(b, c)
        for i in range(m + 1):          # steps along a -> b
            for j in range(m + 1 - i):  # steps along a -> c
                if j == 0:
                    grid[i, j] = side_ab[i]
                elif i == 0:
                    grid[i, j] = side_ac[j]
                elif i + j == m:
                    grid[i, j] = side_bc[j]
                else:
                    points.append(vertices[a] + i * ab + j * ac)
                    grid[i, j] = len(points) - 1
        for i in range(m):
            for j in range(m - i):
                new_tris.append((grid[i, j], grid[i + 1, j], grid[i, j + 1]))
                if i + j < m - 1:
                    new_tris.append((grid[i + 1, j], grid[i + 1, j + 1],
                                     grid[i, j + 1]))
    return _wrap(np.array(points), box), np.array(new_tris, dtype=int)


def root_three(mesh: Mesh, box: np.ndarray | None = None) -> Mesh:
    """The ``(1, 1)`` step: a vertex in every face, then every old edge flipped.

    This is Kobbelt's root-three subdivision without its smoothing, and it is
    the leapfrog operation of fullerene chemistry seen from the triangulation
    side: it triples the face count and, like :func:`subdivide`, leaves the
    coarse vertices' degrees untouched while every new vertex has degree six.
    """
    vertices, tris = np.asarray(mesh[0], dtype=float), np.asarray(mesh[1])
    n = len(vertices)
    centres = []
    for a, b, c in tris:
        centres.append(vertices[a] + (_offset(vertices, a, b, box)
                                      + _offset(vertices, a, c, box)) / 3.0)
    left: dict[tuple[int, int], int] = {}
    for face, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            left[int(u), int(v)] = face
    new_tris = []
    for (u, v), face in left.items():
        if u < v:
            other = left.get((v, u))
            if other is None:
                raise ValueError("root_three needs a closed, oriented mesh.")
            c1, c2 = n + face, n + other
            new_tris.append((u, c2, c1))
            new_tris.append((c2, v, c1))
    points = np.vstack([vertices, np.array(centres)])
    return _wrap(points, box), np.array(new_tris, dtype=int)


def goldberg_coxeter(mesh: Mesh, m: int, n: int = 0,
                     box: np.ndarray | None = None) -> Mesh:
    """Apply ``GC(m, n)``, achiral or chiral.

    ``(m, m)`` is ``(1, 1)`` composed with ``(m, 0)``, which is how the
    Goldberg-Coxeter indices multiply (as Eisenstein integers). Any other
    pair is chiral and goes through :func:`chiral_subdivide`. The face
    count grows by ``T = m**2 + m*n + n**2``.
    """
    if n not in (0, m):
        return chiral_subdivide(mesh, m, n, box)
    refined = subdivide(mesh, m, box)
    return root_three(refined, box) if n == m and m > 0 else refined


def triangulation_number(m: int, n: int) -> int:
    """``T = m**2 + m*n + n**2``: faces of the refined map per coarse face."""
    return m * m + m * n + n * n


def project_to_surface(vertices: np.ndarray, field: Callable,
                       box: np.ndarray | None, iterations: int = 8,
                       step: float = 1e-4) -> np.ndarray:
    """Newton-project points onto ``field == 0`` along its gradient.

    Used only to round a polyhedral coarse map onto the smooth surface it
    approximates; the exact construction is the flat one and is the default.
    """
    points = np.array(vertices, dtype=float, copy=True)
    for _ in range(iterations):
        value = field(points)
        grad = np.empty_like(points)
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            grad[:, axis] = (field(points + shift) - field(points - shift)) \
                / (2.0 * step)
        norm2 = np.maximum((grad ** 2).sum(axis=1), 1e-12)
        points -= (value / norm2)[:, None] * grad
    return _wrap(points, box)


def refine_remeshed(mesh: Mesh, m: int, field: Callable,
                    box: np.ndarray | float | None) -> Mesh:
    """The general route: a coarse remesh refined into graphene patches.

    The implicit builders remesh at ``sqrt(3) * bond`` and the remesher then
    chooses a ring at every vertex -- 11-21 % of them non-hexagonal. Remeshed
    ``m`` times coarser instead, there are ``m**2`` times fewer vertices to
    choose for; :func:`subdivide` then fills each coarse triangle with an
    exact lattice patch and :func:`project_to_surface` lays the result back
    on the field. The disclinations are the coarse mesh's, ``m`` rings
    apart, and the wall between them is crystalline.
    """
    if m <= 1:
        return mesh
    edges = None if box is None else np.broadcast_to(
        np.asarray(box, dtype=float), (3,)).copy()
    vertices, tris = subdivide(mesh, m, box=edges)
    return project_to_surface(vertices, field, edges), tris


def degree_census(tris: np.ndarray) -> dict[int, int]:
    """Ring census of the dual: how many mesh vertices have each degree."""
    degree: dict[int, int] = defaultdict(int)
    for tri in tris:
        for v in tri:
            degree[int(v)] += 1
    census: dict[int, int] = defaultdict(int)
    for d in degree.values():
        census[d] += 1
    return dict(sorted(census.items()))


# --------------------------------------------------------------------------
# Coarse maps. Each is written as its construction, never as a vertex table.
# --------------------------------------------------------------------------
def _truncated_octahedron(centre: np.ndarray, a: float):
    """Vertices (permutations of (0, ±1, ±2) * a/4) and hexagonal faces."""
    from itertools import permutations, product

    verts = set()
    for perm in permutations((0, 1, 2)):
        for signs in product((1, -1), repeat=3):
            verts.add(tuple(s * p for s, p in zip(signs, perm, strict=True)))
    verts = sorted(verts)
    hexagons = []
    for normal in product((1, -1), repeat=3):
        ring = [v for v in verts if np.dot(v, normal) == 3]
        # Order the six round their own centre.
        middle = np.mean(ring, axis=0)
        e1 = np.array(ring[0]) - middle
        e2 = np.cross(normal, e1)
        ring.sort(key=lambda v: np.arctan2(np.dot(np.array(v) - middle, e2),
                                           np.dot(np.array(v) - middle, e1)))
        hexagons.append((np.array(normal), ring))
    squares = []
    for axis in range(3):
        for sign in (1, -1):
            ring = [v for v in verts if v[axis] == 2 * sign]
            normal = np.zeros(3)
            normal[axis] = sign
            middle = np.mean(ring, axis=0)
            e1 = np.array(ring[0]) - middle
            e2 = np.cross(normal, e1)
            ring.sort(key=lambda v: np.arctan2(
                np.dot(np.array(v) - middle, e2),
                np.dot(np.array(v) - middle, e1)))
            squares.append((normal, ring))
    scale = a / 4.0
    return (centre, [(nrm, [centre + scale * np.array(v) for v in ring])
                     for nrm, ring in hexagons],
            [(nrm, [centre + scale * np.array(v) for v in ring])
             for nrm, ring in squares])


class _Builder:
    """Accumulates a periodic triangle mesh, merging coincident vertices."""

    def __init__(self, box: np.ndarray, tol: float = 1e-6):
        self.box = np.asarray(box, dtype=float)
        self.points: list[np.ndarray] = []
        self.keys: dict[tuple, int] = {}
        self.tris: list[tuple[int, int, int]] = []
        self.tol = tol

    def vertex(self, point: np.ndarray) -> int:
        wrapped = _wrap(np.asarray(point, dtype=float)[None], self.box)[0]
        live = self.box > 0
        key_point = wrapped.copy()
        # Points sitting on a face of the cell must key the same as their
        # image on the opposite face.
        key_point[live] = np.where(
            np.abs(key_point[live] - self.box[live]) < 1e-7, 0.0,
            key_point[live])
        key = tuple(np.round(key_point / self.tol).astype(np.int64))
        if key not in self.keys:
            self.keys[key] = len(self.points)
            self.points.append(wrapped)
        return self.keys[key]

    def fan(self, polygon: list[np.ndarray], outward: np.ndarray) -> None:
        """Fan a planar polygon from its centre, oriented along ``outward``."""
        middle = np.mean(polygon, axis=0)
        c = self.vertex(middle)
        ids = [self.vertex(p) for p in polygon]
        flip = np.dot(np.cross(polygon[0] - middle, polygon[1] - middle),
                      outward) < 0
        for k in range(len(ids)):
            a, b = ids[k], ids[(k + 1) % len(ids)]
            self.tris.append((c, b, a) if flip else (c, a, b))

    def split(self, polygon: list[np.ndarray], outward: np.ndarray) -> None:
        """Split a planar quadrilateral along one diagonal."""
        ids = [self.vertex(p) for p in polygon]
        middle = np.mean(polygon, axis=0)
        flip = np.dot(np.cross(polygon[0] - middle, polygon[1] - middle),
                      outward) < 0
        first = (ids[0], ids[1], ids[2])
        second = (ids[0], ids[2], ids[3])
        for tri in (first, second):
            self.tris.append(tri[::-1] if flip else tri)

    def mesh(self) -> Mesh:
        return np.array(self.points), np.array(self.tris, dtype=int)


def primitive_octagon_map(a: float = 4.0) -> tuple[Mesh, np.ndarray]:
    """Schwarz P as the muoctahedron {6,4|4}: hexagons, four at each vertex.

    Truncated octahedra on a body-centred cubic lattice tile space; keep
    only their hexagonal faces and the two families of octahedra are
    joined through the open square windows into two interpenetrating
    simple-cubic labyrinths. The surface between them is the P surface's
    polyhedral model, one corner octahedron and one body-centred one per
    cubic cell: 8 hexagons, 12 vertices, 24 edges, ``chi = -4``, genus 3.

    Fanning each hexagon from its centre gives a triangulation whose
    vertices are the hexagon centres (degree 6) and the polyhedron's own
    vertices (four hexagons each, so degree 8). Its dual is 12 octagons
    per cell and nothing else that is not a hexagon: ``sum(6-n) = -24``,
    which is ``6*chi`` exactly. Mackay & Terrones's original P schwarzite
    and Koshino & Aoki's P-surface graphene are both this family.
    """
    box = np.array([a, a, a])
    build = _Builder(box)
    _, hexagons, _ = _truncated_octahedron(np.zeros(3), a)
    for normal, ring in hexagons:
        # Outward from the corner octahedron, i.e. into the body-centred
        # labyrinth: one consistent side for every face of the surface.
        build.fan(ring, normal)
    return build.mesh(), box


def fold_mesh(mesh: Mesh, box: np.ndarray, period: np.ndarray) -> Mesh:
    """Fold a periodic mesh built in a supercell back onto its primitive cell.

    A coarse map whose edges reach half its own cell cannot be handled under
    the minimum image -- a displacement of exactly ``box/2`` has two
    images and either may be taken. The map is then built and refined in a
    supercell, where every edge is short, and folded afterwards: vertices
    that coincide modulo ``period`` are merged and every triangle is
    renamed. The folded mesh must carry each triangle exactly
    ``prod(box/period)`` times over, which is checked.
    """
    vertices, tris = mesh
    period = np.asarray(period, dtype=float)
    live = period > 0
    folded = vertices.copy()
    folded[:, live] = np.mod(folded[:, live], period[live])
    key_points = folded.copy()
    key_points[:, live] = np.where(
        np.abs(key_points[:, live] - period[live]) < 1e-6, 0.0,
        key_points[:, live])
    keys = [tuple(k) for k in np.round(key_points * 1e5).astype(np.int64)]
    index: dict[tuple, int] = {}
    rename = np.empty(len(vertices), dtype=int)
    points = []
    for i, key in enumerate(keys):
        if key not in index:
            index[key] = len(points)
            points.append(folded[i])
        rename[i] = index[key]
    copies = int(round(np.prod(np.asarray(box)[live] / period[live])))
    seen: dict[tuple, int] = defaultdict(int)
    out = []
    for tri in rename[np.asarray(tris)]:
        a, b, c = (int(x) for x in tri)
        rot = min((a, b, c), (b, c, a), (c, a, b))
        seen[rot] += 1
        if seen[rot] == 1:
            out.append(rot)
    if any(count != copies for count in seen.values()):
        raise ValueError("the supercell mesh does not fold onto the cell: "
                         "it is not periodic with the stated period.")
    return np.array(points), np.array(out, dtype=int)


def square_octagon_map(a: float = 4.0, vacuum: float = 0.0
                       ) -> tuple[Mesh, np.ndarray]:
    """super-square as one layer of truncated octahedra, windows open in-plane.

    A sheet of truncated octahedra on a square lattice, each joined to its
    four neighbours through its side squares and closed above and below by
    its top and bottom squares. Per cell: 8 hexagons and 2 squares, 16
    vertices, 28 edges, ``chi = -2`` -- the super-square's genus exactly.

    Hexagons are fanned (centre degree 6) and the closing squares split on
    a diagonal, so the dual is ``{5: 4, 6: N, 8: 8}``: four pentagons where
    the poles of each crossing are genuinely convex (as in the knee
    route's X node) and eight octagons in its saddles, against the knee
    route's four pentagons and sixteen heptagons. ``sum(6-n) = -12``.
    """
    # Built in a 2 x 2 supercell: the closing squares' diagonals are
    # exactly half the primitive cell, which the minimum image cannot
    # resolve. `build_goldberg` folds the refined mesh back.
    box = np.array([2 * a, 2 * a, 0.0])
    build = _Builder(box)
    for i in range(2):
        for j in range(2):
            _, hexagons, squares = _truncated_octahedron(
                np.array([i * a, j * a, 0.0]), a)
            for normal, ring in hexagons:
                build.fan(ring, normal)
            for normal, ring in squares:
                if abs(normal[2]) > 0.5:      # top and bottom close it
                    build.split(ring, normal)
    vertices, tris = build.mesh()
    vertices = vertices.copy()
    vertices[:, 2] -= vertices[:, 2].min()
    return (vertices, tris), box


#: Vanderbilt & Tersoff's C168 (*Phys. Rev. Lett.* **68**, 511, 1992),
#: Table I: the seven inequivalent atoms in units of the fcc lattice
#: constant ``a = 21.8`` Å. The other 77 atoms of a joint follow from the
#: twelve proper rotations of the tetrahedron, the second joint from
#: inversion and a ``(a/4, a/4, a/4)`` translation.
VANDERBILT_TERSOFF_C168 = np.array([
    (0.1134, 0.0050, 0.2415), (-0.0109, 0.0303, 0.2200),
    (0.0272, 0.0756, 0.2435), (-0.0984, 0.0826, 0.1704),
    (-0.0662, 0.1383, 0.1542), (-0.0743, 0.0402, 0.2146),
    (-0.1915, 0.0177, 0.1649),
])
C168_LATTICE = 21.8


def _tetrahedral_rotations() -> list[np.ndarray]:
    """The twelve proper rotations of the tetrahedron, group T."""
    from itertools import product

    out = []
    for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for signs in product((1, -1), repeat=3):
            if np.prod(signs) == 1:
                rot = np.zeros((3, 3))
                for row, col in enumerate(perm):
                    rot[row, col] = signs[row]
                out.append(rot)
    return out


def c168_atoms() -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Vanderbilt & Tersoff's C168 in its conventional cubic cell (672 atoms).

    Rebuilt from the paper's own Table I, not transcribed: 7 sites x 12
    rotations x 2 joints x 4 fcc translations, deduplicated. Measured, every
    atom comes out three-coordinate with bonds 1.388-1.454 Å, which is the
    check that the table was read correctly.

    Returns positions (Å), bonds as index pairs, and the cubic cell edges.
    """
    from scipy.spatial import cKDTree

    joint = np.array([rot @ site for rot in _tetrahedral_rotations()
                      for site in VANDERBILT_TERSOFF_C168])
    primitive = np.vstack([joint, -joint + 0.25])
    frac = np.vstack([primitive + np.array(shift) for shift in
                      ((0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0))])
    frac = np.mod(frac, 1.0)
    keys = np.round(frac * 1e4).astype(np.int64) % 10000
    _, first = np.unique(keys, axis=0, return_index=True)
    positions = frac[np.sort(first)] * C168_LATTICE
    box = np.array([C168_LATTICE] * 3)
    pairs = sorted(cKDTree(positions, boxsize=C168_LATTICE).query_pairs(1.7))
    degree = np.bincount(np.array(pairs).ravel(), minlength=len(positions))
    if len(positions) != 672 or set(degree) != {3}:
        raise RuntimeError("C168 did not rebuild as a trivalent 672-atom cell.")
    return positions, pairs, box


def _faces_of_trivalent_net(n_atoms: int, pairs, limit: int = 8
                            ) -> list[list[int]]:
    """The rings of a trivalent surface net: the shortest cycle at each angle.

    On C168 every face is the smallest cycle through each of its corners,
    which is not true of an arbitrary net -- the result is therefore checked
    by the caller against Euler (each atom in exactly three faces).
    """
    from collections import deque

    neighbours: dict[int, list[int]] = defaultdict(list)
    for i, j in pairs:
        neighbours[i].append(j)
        neighbours[j].append(i)
    faces: dict[tuple, list[int]] = {}
    for centre in range(n_atoms):
        for u in neighbours[centre]:
            for v in neighbours[centre]:
                if u >= v:
                    continue
                # Shortest path u -> v avoiding the centre.
                parent = {u: None}
                queue = deque([u])
                while queue and v not in parent:
                    node = queue.popleft()
                    for nxt in neighbours[node]:
                        if nxt != centre and nxt not in parent:
                            parent[nxt] = node
                            queue.append(nxt)
                if v not in parent:
                    continue
                path = [v]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                ring = [centre] + path
                if len(ring) <= limit:
                    faces.setdefault(tuple(sorted(ring)), ring)
    return list(faces.values())


def _c168_rings():
    positions, pairs, box = c168_atoms()
    rings = _faces_of_trivalent_net(len(positions), pairs)
    count = np.zeros(len(positions), dtype=int)
    for ring in rings:
        count[ring] += 1
    sizes = sorted({len(r) for r in rings})
    if set(count) != {3} or sizes != [6, 7]:
        raise RuntimeError("C168's faces were not perceived cleanly.")
    centres = []
    for ring in rings:
        offsets = [_offset(positions, ring[0], k, box) for k in ring]
        centres.append(_wrap((positions[ring[0]]
                              + np.mean(offsets, axis=0))[None], box)[0])
    return positions, pairs, box, rings, np.array(centres)


def diamond_heptagon_map() -> tuple[Mesh, np.ndarray]:
    """Schwarz D as Klein's {3,7} map: the coarse map behind C168.

    Every hexagon of C168 is ringed by alternating hexagons and heptagons
    (Vanderbilt & Tersoff), so it touches exactly three heptagons; those
    three are a triangle of the coarse map, whose vertices are the 96
    heptagon centres of the cubic cell -- degree seven each, 224 triangles,
    ``chi = -16``. ``GC(1, 0)`` of it is four copies of their C56, and
    ``GC(1, 1)`` gives C168 back.
    """
    positions, pairs, box, rings, centres = _c168_rings()
    hepta = [k for k, ring in enumerate(rings) if len(ring) == 7]
    index = {k: i for i, k in enumerate(hepta)}
    edge_owner: dict[tuple[int, int], int] = {}
    for k, ring in enumerate(rings):
        for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
            key = (min(a, b), max(a, b))
            edge_owner.setdefault(key, [])
            edge_owner[key].append(k)
    tris = []
    for k, ring in enumerate(rings):
        if len(ring) != 6:
            continue
        touching = set()
        for a, b in zip(ring, ring[1:] + ring[:1], strict=True):
            for other in edge_owner[(min(a, b), max(a, b))]:
                if other != k and len(rings[other]) == 7:
                    touching.add(other)
        if len(touching) != 3:
            raise RuntimeError("a C168 hexagon does not touch three heptagons.")
        tris.append(tuple(index[t] for t in sorted(touching)))
    return (centres[hepta], np.array(tris, dtype=int)), box


def diamond_c168_map() -> tuple[Mesh, np.ndarray]:
    """C168's own dual: ``GC(1, 1)`` of :func:`diamond_heptagon_map`.

    Vertices are all 320 ring centres of the cubic cell (96 of degree 7),
    triangles the 672 atoms. Subdividing it ``m`` times gives
    ``GC(m, m)`` of the Klein map without needing an orientation.
    """
    positions, pairs, box, rings, centres = _c168_rings()
    member: dict[int, list[int]] = defaultdict(list)
    for k, ring in enumerate(rings):
        for atom in ring:
            member[atom].append(k)
    tris = np.array([member[a] for a in range(len(positions))], dtype=int)
    return (centres, tris), box


#: Coarse maps by kind: ``kind -> (map builder, genus, pbc, budget,
#: supercell)``. ``supercell`` is how many primitive cells per axis the
#: map is built over before it is folded back (see :func:`fold_mesh`).
GOLDBERG_MAPS: dict[str, tuple[Callable, int, tuple[bool, bool, bool],
                               int, int]] = {
    "primitive": (primitive_octagon_map, 3, (True, True, True), -24, 1),
    "super-square": (square_octagon_map, 2, (True, True, False), -12, 2),
    # Conventional cubic cell: four primitive D cells, genus 9.
    "diamond": (diamond_heptagon_map, 9, (True, True, True), -96, 1),
}

#: Kinds whose class-II family is built from its own map, subdivided,
#: rather than by the root-three step (which needs an oriented map).
GOLDBERG_CLASS_TWO: dict[str, Callable] = {"diamond": diamond_c168_map}


def goldberg_atom_count(kind: str, m: int, n: int = 0) -> int:
    """Atoms the build will have: coarse faces per cell times ``T``.

    Exact, and free -- the coarse map is combinatorial, so this is the
    estimate the window shows before anything is relaxed.
    """
    make_map, _, _, _, supercell = GOLDBERG_MAPS[kind]
    (_, tris), box = make_map()
    cells = int(round(np.prod([supercell for edge in box if edge > 0])))
    return len(tris) // cells * triangulation_number(m, n)


def loop_subdivide(mesh: Mesh, box: np.ndarray | None = None,
                   levels: int = 1) -> Mesh:
    """Loop subdivision: GC(2, 0) topology, smooth-surface geometry.

    Topologically each step is :func:`subdivide` with ``m = 2`` -- every
    triangle into four, every new vertex of degree six, the old vertices'
    degrees kept -- so the census cannot change. Geometrically the new
    vertices go where Loop's masks put them (3/8 of each edge end, 1/8 of
    each opposite vertex) and the old ones move by Warren's weights, so
    repeated steps converge to a smooth limit surface fixed by the coarse
    map itself. That is what a flat refinement lacks: a coarse map's
    kinks and badly shaped triangles otherwise survive into the graphene
    as strain, and projecting onto an outside surface instead dragged
    vertices to 5 Å bonds where the two shapes disagreed.
    """
    from .fullerene_mesh import minimum_image

    live = None if box is None or not np.any(np.asarray(box) > 0) else box
    vertices, tris = np.array(mesh[0], dtype=float), np.asarray(mesh[1])
    for _ in range(levels):
        n = len(vertices)
        neighbours: dict[int, set[int]] = defaultdict(set)
        opposite: dict[tuple[int, int], list[int]] = defaultdict(list)
        for a, b, c in tris:
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
                neighbours[int(u)].add(int(v))
                neighbours[int(v)].add(int(u))
                opposite[(min(u, v), max(u, v))].append(int(w))
        # New edge points.
        edge_index: dict[tuple[int, int], int] = {}
        new_points = []
        for (u, v), opp in opposite.items():
            d_v = minimum_image(vertices[v] - vertices[u], live)
            point = vertices[u] + 0.5 * d_v
            if len(opp) == 2:
                d_c = minimum_image(vertices[opp[0]] - vertices[u], live)
                d_d = minimum_image(vertices[opp[1]] - vertices[u], live)
                point = vertices[u] + 0.375 * d_v + 0.125 * (d_c + d_d)
            edge_index[(u, v)] = n + len(new_points)
            new_points.append(point)
        # Moved old points (Warren's beta).
        moved = vertices.copy()
        for i in range(n):
            nb = np.fromiter(neighbours[i], dtype=int)
            k = len(nb)
            beta = 3.0 / 16.0 if k == 3 else 3.0 / (8.0 * k)
            offsets = minimum_image(vertices[nb] - vertices[i], live)
            moved[i] = vertices[i] + beta * offsets.sum(axis=0)
        out_tris = []
        for a, b, c in tris:
            ab = edge_index[(min(a, b), max(a, b))]
            bc = edge_index[(min(b, c), max(b, c))]
            ca = edge_index[(min(c, a), max(c, a))]
            out_tris += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        vertices = _wrap(np.vstack([moved, np.array(new_points)]), live)
        tris = np.array(out_tris, dtype=int)
    return vertices, tris


def taubin_smooth(mesh: Mesh, box: np.ndarray | None, iterations: int = 20,
                  lam: float = 0.5, mu: float = -0.53) -> Mesh:
    """Round a polyhedral mesh without shrinking it (Taubin's lambda/mu).

    A coarse map refined into flat patches meets itself at sharp folds --
    90 deg along a square tube's edges -- which the valence force field
    then has to bend out with bond strain. Alternating a shrinking and an
    inflating umbrella step rounds the folds and leaves the overall size
    alone. Connectivity, and so the census, is untouched.
    """
    from .fullerene_mesh import minimum_image

    vertices, tris = np.array(mesh[0], dtype=float), np.asarray(mesh[1])
    neighbours: dict[int, set[int]] = defaultdict(set)
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            neighbours[int(u)].add(int(v))
            neighbours[int(v)].add(int(u))
    index = [np.fromiter(neighbours[i], dtype=int)
             for i in range(len(vertices))]
    live = None if box is None or not np.any(np.asarray(box) > 0) else box
    for _ in range(iterations):
        for factor in (lam, mu):
            step = np.array([minimum_image(vertices[nb] - vertices[i],
                                           live).mean(axis=0)
                             for i, nb in enumerate(index)])
            vertices = vertices + factor * step
    return _wrap(vertices, live), tris


def finish_refined(mesh: Mesh, box: np.ndarray | None,
                   pbc: tuple[bool, bool, bool], budget: int,
                   bond: float = CC_BOND, relax: bool = True,
                   relax_iterations: int = 3000, vacuum: float = 12.0,
                   label: str = "structure", cell_scan: bool = False
                   ) -> Atoms:
    """Dual, rescale, relax, check the census, package -- shared tail.

    ``box`` holds the cell edges on the periodic axes (``0`` elsewhere) or
    is ``None`` for a finite structure. Non-periodic axes get ``vacuum``
    either side and the structure centred. The census is checked against
    ``budget`` and a mismatch raises: the map was not the surface it
    claimed to be.

    Returns an ``Atoms`` with ``ring_counts``, ``ring_deficit``,
    ``pentagons``, ``rings``, ``bonds``, ``geometry``, ``cell_lengths``,
    ``disclination_check`` and ``disclinations_placed`` in ``info``.
    """
    from ..analyse.curvature import disclination_check
    from .capped_cnt import geometry_report
    from .fullerene_mesh import dual_honeycomb, minimum_image, relax_shell

    vertices, tris = mesh
    live_box = None if box is None or not np.any(np.asarray(box) > 0) \
        else np.asarray(box, dtype=float)
    positions, bonds, rings = dual_honeycomb((vertices, tris), box=live_box)
    pairs = np.asarray(sorted(bonds), dtype=int)
    raw = np.linalg.norm(minimum_image(positions[pairs[:, 1]]
                                       - positions[pairs[:, 0]], live_box),
                         axis=1)
    scale = bond / float(raw.mean())
    positions = positions * scale
    live_box = None if live_box is None else live_box * scale
    if relax and live_box is not None and cell_scan:
        # A periodic cell cannot change size during the relaxation, so a
        # map whose tubes and nodes want different scales is left strained
        # by the mean-bond rescale alone. Scan the cell (on its periodic
        # axes) and keep the scale whose short relaxation fits carbon best.
        # Measured on the super-cubic frame: the best cell sat 20 % BELOW
        # the mean-bond rescale (1.31-1.49 Å there against 1.33-1.59 at
        # the rescale), so the coarse pass is wide and the fine one local.
        def score(factor: float, iterations: int):
            trial_box = live_box * factor
            trial = relax_shell(positions * factor, bonds, equilibrium=bond,
                                max_iterations=iterations, box=trial_box)
            lengths = np.linalg.norm(minimum_image(
                trial[pairs[:, 1]] - trial[pairs[:, 0]], trial_box), axis=1)
            return float(np.abs(lengths - bond).max()), factor, trial

        best = min((score(f, 600) for f in np.arange(0.60, 1.201, 0.05)),
                   key=lambda r: r[0])
        centre = best[1]
        best = min([best] + [score(f, 1500) for f in
                             np.arange(centre - 0.04, centre + 0.041, 0.01)],
                   key=lambda r: r[0])
        _, factor, positions = best
        live_box = live_box * factor
    if relax:
        positions = relax_shell(positions, bonds, equilibrium=bond,
                                max_iterations=relax_iterations, box=live_box)
    ring_counts: dict[int, int] = defaultdict(int)
    for ring in rings:
        ring_counts[len(ring)] += 1
    ring_counts = dict(sorted(ring_counts.items()))
    deficit = sum((6 - s) * c for s, c in ring_counts.items())
    if deficit != budget:
        raise RuntimeError(
            f"{label}: sum(6-n) = {deficit} against the budget {budget}; the "
            "coarse map is not the surface it claims to be.")

    cell = np.zeros(3) if live_box is None else np.array(live_box, dtype=float)
    positions = positions.copy()
    for axis in range(3):
        if not pbc[axis]:
            span = float(np.ptp(positions[:, axis]))
            cell[axis] = span + 2.0 * vacuum
            positions[:, axis] += 0.5 * cell[axis] - 0.5 * (
                positions[:, axis].min() + positions[:, axis].max())
    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  cell=list(cell), pbc=pbc)
    if any(pbc):
        atoms.wrap()
    periodic = np.where(np.array(pbc), cell, 0.0)
    metric = None if not any(pbc) else periodic
    placement = disclination_check(positions, rings, sorted(bonds),
                                   box=metric)
    atoms.info.update({
        "ring_counts": ring_counts,
        "ring_deficit": int(deficit),
        "pentagons": int(ring_counts.get(5, 0)),
        "rings": [[int(x) for x in ring] for ring in rings],
        "bonds": sorted(bonds),
        "relaxed": bool(relax),
        "cell_lengths": [round(float(x), 3) for x in cell],
        "geometry": geometry_report(positions, sorted(bonds), box=metric),
        # Pentagons in positive curvature, heptagons and octagons in
        # negative: the intrinsic check every meshed builder records.
        "disclination_check": placement,
        "disclinations_placed": placement["agreement"],
    })
    return atoms


def build_goldberg(
    kind: str = "primitive",
    m: int = 3,
    n: int = 0,
    bond: float = CC_BOND,
    vacuum: float = 12.0,
    relax: bool = True,
    relax_iterations: int = 3000,
    project: bool = False,
) -> Atoms:
    """A superstructure from flat graphene patches, defects only at map vertices.

    Parameters
    ----------
    kind
        A key of :data:`GOLDBERG_MAPS`.
    m, n
        Goldberg-Coxeter indices, ``n`` either 0 or ``m``. They set the
        size: ``T = m**2 + m*n + n**2`` triangles per coarse face, and the
        defects sit ``m`` (or ``m`` root-three) rings apart.
    bond
        C-C length (Å); the whole cell is rescaled to it before relaxing.
    vacuum
        Vacuum (Å) above and below a sheet; ignored for a 3D cell.
    relax, relax_iterations
        Valence-force-field relaxation under the minimum image. The bond
        graph is explicit, so the census is fixed before it runs.
    project
        Round the polyhedral map onto the smooth surface before taking the
        dual. Off by default: the flat patches are the construction.

    Returns
    -------
    ase.Atoms
        With the census, the budget and the measured geometry in ``info``.
    """
    if kind not in GOLDBERG_MAPS:
        raise ValueError(f"unknown kind {kind!r}; built here: "
                         f"{sorted(GOLDBERG_MAPS)}.")
    make_map, genus, pbc, budget, supercell = GOLDBERG_MAPS[kind]
    if n == m and kind in GOLDBERG_CLASS_TWO:
        coarse, box = GOLDBERG_CLASS_TWO[kind]()
        fine = subdivide(coarse, m, box)
    else:
        coarse, box = make_map()
        fine = goldberg_coxeter(coarse, m, n, box)
    if supercell > 1:
        if triangulation_number(m, n) < 4:
            raise ValueError(
                f"{kind} needs T = m^2 + mn + n^2 >= 4 (m >= 2): its coarse "
                "map has edges half the cell long, which only a refined "
                "mesh can carry onto the primitive cell.")
        box = box / np.where(box > 0, supercell, 1)
        fine = fold_mesh(fine, box * supercell, box)
    vertices, tris = fine
    if project and kind == "primitive":
        a = float(box[0])
        k = 2.0 * np.pi / a

        def field(p):
            # The P nodal surface, centred on the corner labyrinth's
            # channels: cos x + cos y + cos z = 0 passes through the
            # hexagon centres at (a/4)(1, 1, 1).
            return np.cos(k * p[:, 0]) + np.cos(k * p[:, 1]) + \
                np.cos(k * p[:, 2])
        vertices = project_to_surface(vertices, field, box)

    atoms = finish_refined((vertices, tris), box, pbc, budget, bond=bond,
                           relax=relax, relax_iterations=relax_iterations,
                           vacuum=vacuum, label=f"{kind} (genus {genus})")
    atoms.info.update({
        "builder": "goldberg",
        "structure_type": "schwarzite" if kind in ("primitive", "diamond")
        else "supernetwork",
        "kind": kind,
        "gc_indices": (int(m), int(n)),
        "triangulation_number": triangulation_number(m, n),
        "genus": int(genus),
        "euler": 2 - 2 * int(genus),
        "ring_budget": int(budget),
        "projected": bool(project),
    })
    return atoms


def describe_goldberg(atoms: Atoms) -> str:
    """One line for the window and the command line: indices, census, law."""
    info = atoms.info
    m, n = info["gc_indices"]
    census = ", ".join(f"{size}: {count}"
                       for size, count in info["ring_counts"].items())
    return (f"{info['kind']} GC({m},{n}), T = {info['triangulation_number']}: "
            f"{len(atoms)} atoms, rings {{{census}}}, sum(6-n) = "
            f"{info['ring_deficit']:+d} against genus {info['genus']}'s "
            f"{info['ring_budget']:+d}")
