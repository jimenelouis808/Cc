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
    """Apply ``GC(m, n)`` for ``n == 0`` or ``n == m``.

    ``(m, m)`` is ``(1, 1)`` composed with ``(m, 0)``, which is how the
    Goldberg-Coxeter indices multiply (as Eisenstein integers). The face
    count grows by ``T = m**2 + m*n + n**2``.
    """
    if n not in (0, m):
        raise ValueError(
            f"GC({m}, {n}) is chiral; only (m, 0) and (m, m) are built here. "
            "A chiral patch would have to cut the triangular lattice "
            "consistently across every coarse edge.")
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


#: Coarse maps by kind: ``kind -> (map builder, genus, pbc, budget,
#: supercell)``. ``supercell`` is how many primitive cells per axis the
#: map is built over before it is folded back (see :func:`fold_mesh`).
GOLDBERG_MAPS: dict[str, tuple[Callable, int, tuple[bool, bool, bool],
                               int, int]] = {
    "primitive": (primitive_octagon_map, 3, (True, True, True), -24, 1),
    "super-square": (square_octagon_map, 2, (True, True, False), -12, 2),
}


def goldberg_atom_count(kind: str, m: int, n: int = 0) -> int:
    """Atoms the build will have: coarse faces per cell times ``T``.

    Exact, and free -- the coarse map is combinatorial, so this is the
    estimate the window shows before anything is relaxed.
    """
    make_map, _, _, _, supercell = GOLDBERG_MAPS[kind]
    (_, tris), box = make_map()
    cells = int(round(np.prod([supercell for edge in box if edge > 0])))
    return len(tris) // cells * triangulation_number(m, n)


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

    from ..analyse.curvature import disclination_check
    from .capped_cnt import geometry_report
    from .fullerene_mesh import dual_honeycomb, minimum_image, relax_shell

    positions, bonds, rings = dual_honeycomb((vertices, tris), box=box)
    pairs = np.asarray(sorted(bonds), dtype=int)
    raw = np.linalg.norm(minimum_image(positions[pairs[:, 1]]
                                       - positions[pairs[:, 0]], box), axis=1)
    scale = bond / float(raw.mean())
    positions, box = positions * scale, box * scale
    if relax:
        positions = relax_shell(positions, bonds, equilibrium=bond,
                                max_iterations=relax_iterations, box=box)
    ring_counts: dict[int, int] = defaultdict(int)
    for ring in rings:
        ring_counts[len(ring)] += 1
    ring_counts = dict(sorted(ring_counts.items()))
    deficit = sum((6 - s) * c for s, c in ring_counts.items())
    if deficit != budget:
        raise RuntimeError(
            f"{kind}: sum(6-n) = {deficit} against the genus-{genus} budget "
            f"{budget}; the coarse map is not the surface it claims to be.")

    cell = np.array(box, dtype=float)
    if not pbc[2]:
        # A sheet: centre it in vacuum along z.
        height = float(np.ptp(positions[:, 2]))
        cell[2] = height + 2.0 * vacuum
        positions = positions.copy()
        positions[:, 2] += 0.5 * cell[2] - 0.5 * (positions[:, 2].min()
                                                 + positions[:, 2].max())
    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  cell=list(cell), pbc=pbc)
    atoms.wrap()
    placement = disclination_check(positions, rings, sorted(bonds),
                                   box=np.where(np.array(pbc), cell, 0.0))
    geometry = geometry_report(positions, sorted(bonds),
                               box=np.where(np.array(pbc), cell, 0.0))
    atoms.info.update({
        "builder": "goldberg",
        "structure_type": "schwarzite" if kind == "primitive"
        else "supernetwork",
        "kind": kind,
        "gc_indices": (int(m), int(n)),
        "triangulation_number": triangulation_number(m, n),
        "genus": int(genus),
        "euler": 2 - 2 * int(genus),
        "ring_budget": int(budget),
        "ring_counts": ring_counts,
        "ring_deficit": int(deficit),
        "pentagons": int(ring_counts.get(5, 0)),
        "rings": [[int(x) for x in ring] for ring in rings],
        "bonds": sorted(bonds),
        "relaxed": bool(relax),
        "projected": bool(project),
        "cell_lengths": [round(float(x), 3) for x in cell],
        "geometry": geometry,
        # The intrinsic placement check every meshed builder records:
        # octagons in negative curvature, pentagons in positive.
        "disclination_check": placement,
        "disclinations_placed": placement["agreement"],
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
