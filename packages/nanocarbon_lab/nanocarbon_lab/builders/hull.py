"""Any graph of tubes, crystalline: hull nodes, coarse defect search, GC inflation.

Three routes in this package build a superstructure wall exactly, and
each stops somewhere. The knee route needs every node to balance, so no
finite cage. The octahedral frame (:mod:`.frame`) needs every arm along
<100> or <111>, so no icosahedron, no C60, no fcc. The mesh route builds
anything and leaves 11-21 % of the rings non-hexagonal, scattered.

This module removes the restriction by splitting the problem in two:

1. **A coarse map that is right by construction.** Each tube is a prism
   on a regular ``k``-gon; each node is the **convex hull of its tubes'
   mouths**, with the mouths cut out. A hull is a closed convex surface,
   so nothing can tear, and every mouth is a facet of it, so every tube
   meets its node along a clean polygon -- at any arm angles whatsoever.
2. **The defects are optimised where there are few of them.** The coarse
   map has tens of vertices per node, so its census can be driven to the
   minimum by edge flips while it is still small (``optimise_census``),
   then frozen.
3. **Goldberg-Coxeter inflation** (:func:`.goldberg.subdivide`) turns
   every coarse triangle into a flat patch of graphene. The census does
   not change: the defects stay exactly where step 2 left them, ``m``
   rings apart, and everything between them is crystalline.

The ring budget is the graph's own ``12(V - E)``, checked at the end like
every other route here.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from ase import Atoms
from scipy.spatial import ConvexHull

from ..utils.constants import CC_BOND
from .goldberg import (
    Mesh,
    _Builder,
    finish_refined,
    loop_subdivide,
    subdivide,
    taubin_smooth,
)

#: Coarse edge length (map units) and the prism row spacing that keeps
#: the strip triangles equilateral.
_EDGE = 1.0
_ROW = _EDGE * np.sqrt(3.0) / 2.0


def _frame_for(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to ``axis`` and to each other."""
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, axis)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    p = np.cross(axis, helper)
    p /= np.linalg.norm(p)
    return p, np.cross(axis, p)


def _arms(points, edges, box):
    """Per node: list of (edge index, unit direction, sign) of its arms."""
    arms: dict[int, list] = defaultdict(list)
    vectors = []
    for index, (i, j, shift) in enumerate(edges):
        far = points[j] + (np.zeros(3) if box is None
                           else np.asarray(shift, float) * box)
        vector = far - points[i]
        vectors.append(vector)
        unit = vector / np.linalg.norm(vector)
        arms[i].append((index, unit, +1))
        arms[j].append((index, -unit, -1))
    return arms, vectors


def hull_map(points: np.ndarray, edges, box: np.ndarray | None = None,
             sides: int = 6, rows: int = 2, phase: float = 0.0
             ) -> tuple[Mesh, int, float]:
    """The coarse map: convex-hull nodes joined by ``sides``-gon prisms.

    Parameters
    ----------
    points, edges, box
        The graph: node positions (any units), edges ``(i, j, shift)``,
        and the cell for a periodic graph (``None`` for a cage).
    sides
        Polygon of every tube, so its circumference after refining by
        ``m`` is ``sides * m`` rings.
    rows
        Prism rows on the SHORTEST arm; the graph is scaled to fit.
    phase
        Rotation of every mouth polygon about its axis, as a fraction of
        one polygon step. It changes which hull triangles form.

    Returns
    -------
    (mesh, budget, scale)
        The closed coarse surface, ``12(V - E)``, and the factor applied
        to the graph's coordinates (map units per input unit).
    """
    points = np.asarray(points, dtype=float)
    arms, vectors = _arms(points, edges, box)
    radius = _EDGE / (2.0 * np.sin(np.pi / sides))
    # How far out each mouth sits: far enough that neighbouring mouths do
    # not overlap, from the tightest angle between two of the node's arms.
    reach = {}
    for node, items in arms.items():
        tightest = np.pi
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                cos = float(np.clip(np.dot(items[a][1], items[b][1]), -1, 1))
                tightest = min(tightest, np.arccos(cos))
        reach[node] = max(radius * 1.25 / np.tan(tightest / 2.0), radius)
    scale = max((reach[i] + reach[j] + rows * _ROW) / np.linalg.norm(v)
                for (i, j, _), v in zip(edges, vectors, strict=True))
    centres = points * scale
    cell = None if box is None else np.asarray(box, dtype=float) * scale
    build = _Builder(np.zeros(3) if cell is None else cell)

    angles = 2.0 * np.pi * (np.arange(sides) + phase) / sides
    mouths: dict[tuple[int, int], list[np.ndarray]] = {}
    for index, (i, j, shift) in enumerate(edges):
        unit = vectors[index] / np.linalg.norm(vectors[index])
        p, q = _frame_for(unit)
        start = centres[i] + reach[i] * unit
        far = centres[j] + (np.zeros(3) if cell is None
                            else np.asarray(shift, float) * cell)
        end = far - reach[j] * unit
        length = float(np.linalg.norm(end - start))
        count = max(1, int(round(length / _ROW)))
        # An antiprism stack, not a prism: each ring turned half a step
        # on the one below, so every strip triangle is equilateral --
        # which is what a graphene tube's own triangulation is. A prism
        # stack gives 1 : 0.87 : 1.32 right triangles, and the patches
        # refined from them come out sheared.
        def turned(k: int, p=p, q=q) -> list[np.ndarray]:
            angle = np.pi * k / sides
            return [radius * (np.cos(t + angle) * p + np.sin(t + angle) * q)
                    for t in angles]
        layers = [[start + (end - start) * k / count + v for v in turned(k)]
                  for k in range(count + 1)]
        mouths[(i, index)] = layers[0]
        mouths[(j, index)] = [v - (np.zeros(3) if cell is None else
                                   np.asarray(shift, float) * cell)
                              for v in layers[-1]]
        ids = [[build.vertex(v) for v in layer] for layer in layers]
        for k in range(count):
            for a in range(sides):
                b = (a + 1) % sides
                p0 = layers[k][a]
                out = p0 - start - np.dot(p0 - start, unit) * unit
                # Ring k+1 vertex a sits between ring k's a and b.
                tri1 = (ids[k][a], ids[k][b], ids[k + 1][a])
                tri2 = (ids[k + 1][a], ids[k][b], ids[k + 1][b])
                cross = np.cross(layers[k][b] - p0, layers[k + 1][a] - p0)
                flip = np.dot(cross, out) < 0
                for tri in (tri1, tri2):
                    build.tris.append(tri[::-1] if flip else tri)

    for node, items in arms.items():
        rims = [mouths[(node, index)] for index, _, _ in items]
        cloud = np.array([v for rim in rims for v in rim])
        owner = np.repeat(np.arange(len(rims)), sides)
        hull = ConvexHull(cloud - centres[node])
        ids = [build.vertex(v) for v in cloud]
        for simplex, equation in zip(hull.simplices, hull.equations,
                                     strict=True):
            if len(set(owner[simplex])) == 1:
                continue          # inside a mouth: that is the tube's hole
            a, b, c = simplex
            normal = np.cross(cloud[b] - cloud[a], cloud[c] - cloud[a])
            tri = (ids[a], ids[b], ids[c])
            build.tris.append(tri if np.dot(normal, equation[:3]) > 0
                              else tri[::-1])

    mesh = build.mesh()
    budget = 12 * (len(points) - len(edges))
    return mesh, budget, scale


def _degrees(tris: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(np.asarray(tris).ravel(), minlength=n)


def curvature_charges(mesh: Mesh, box: np.ndarray | None = None
                      ) -> np.ndarray:
    """Each coarse vertex's share of the ring budget: ``3 K(v) / pi``.

    ``K(v)`` is the angle deficit of the coarse polyhedral surface, so the
    shares sum to ``6 chi`` exactly (discrete Gauss-Bonnet) and say where
    the budget belongs: positive at a cage's convex corners, negative in
    the necks. A census that only counts defects ignores this and leaves
    a convex corner with no pentagon to pay for it.
    """
    from .fullerene_mesh import minimum_image

    vertices, tris = mesh
    live = None if box is None or not np.any(np.asarray(box) > 0) else box
    angle = np.zeros(len(vertices))
    for tri in tris:
        for k in range(3):
            a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            u = minimum_image(vertices[b] - vertices[a], live)
            v = minimum_image(vertices[c] - vertices[a], live)
            cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            angle[a] += np.arccos(np.clip(cos, -1.0, 1.0))
    return 3.0 * (2.0 * np.pi - angle) / np.pi


def optimise_census(mesh: Mesh, box: np.ndarray | None = None,
                    sweeps: int = 50, max_edge: float = 1.9,
                    targets: np.ndarray | None = None
                    ) -> tuple[Mesh, list[int]]:
    """Drive the coarse map's census down by edge flips, while it is small.

    The objective is ``sum (deg - 6)**2``: it falls when a 5-7 pair
    annihilates and when a 4 or an 8 splits, and a flip never changes
    ``sum(6 - deg)``, so the budget is safe. A flip is taken only when it
    strictly lowers the objective, keeps every degree at 4 or more, does
    not create an edge already present, and the new edge is no longer than
    ``max_edge`` coarse edges -- the guard that stops a flip reaching
    across a tube (see ``FLIP_MAX_EDGE`` in the remesher's notes).

    Returns the mesh and the objective after each sweep.
    """
    from .fullerene_mesh import minimum_image

    vertices, tris = np.array(mesh[0]), [tuple(int(x) for x in t)
                                         for t in mesh[1]]
    live = None if box is None or not np.any(np.asarray(box) > 0) else box
    history = []
    for _ in range(sweeps):
        degree = _degrees(np.array(tris), len(vertices))
        edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        neighbours: dict[int, set[int]] = defaultdict(set)
        for f, (a, b, c) in enumerate(tris):
            for u, v in ((a, b), (b, c), (c, a)):
                edge_faces[(min(u, v), max(u, v))].append(f)
                neighbours[u].add(v)
                neighbours[v].add(u)
        changed = False
        touched: set[int] = set()
        for (u, v), faces in edge_faces.items():
            if len(faces) != 2 or touched & set(faces):
                continue
            f1, f2 = faces
            c = next(x for x in tris[f1] if x not in (u, v))
            d = next(x for x in tris[f2] if x not in (u, v))
            if c == d or d in neighbours[c]:
                continue
            goal = (np.zeros(4) if targets is None
                    else 6.0 - np.array([targets[x] for x in (u, v, c, d)]))
            now = np.array([degree[u], degree[v], degree[c], degree[d]],
                           dtype=float)
            then = now + np.array([-1.0, -1.0, 1.0, 1.0])
            if targets is None:
                goal = np.full(4, 6.0)
            before = float(((now - goal) ** 2).sum())
            after = float(((then - goal) ** 2).sum())
            if after >= before - 1e-9 or degree[u] <= 4 or degree[v] <= 4:
                continue
            span = np.linalg.norm(minimum_image(vertices[d] - vertices[c],
                                                live))
            if span > max_edge * _EDGE:
                continue
            # Keep the orientation: the triangle that had u -> v keeps
            # its sense.
            t1 = list(tris[f1])
            k = t1.index(u)
            if t1[(k + 1) % 3] == v:
                tris[f1], tris[f2] = (u, d, c), (d, v, c)
            else:
                tris[f1], tris[f2] = (u, c, d), (c, v, d)
            for x, delta in ((u, -1), (v, -1), (c, 1), (d, 1)):
                degree[x] += delta
            touched |= {f1, f2}
            changed = True
        history.append(int(((degree - 6) ** 2).sum()))
        if not changed:
            break
    return (vertices, np.array(tris, dtype=int)), history


def settle_on_tubes(mesh: Mesh, centres: np.ndarray, edges,
                    cell: np.ndarray | None, tube_radius: float,
                    iterations: int = 30, blend_fraction: float = 0.6
                    ) -> Mesh:
    """Lay the refined map on the smooth surface of the same tubes, evenly.

    The coarse map is polyhedral -- flat hull facets and flat prism
    strips -- and the patches refined from it inherit every kink and every
    badly shaped coarse triangle, which the force field can only pay back
    in bond strain (measured: 1.2-1.7 Å on the first hull cages). The cure
    is the one `fullerene_mesh` uses for C60: put the vertices on a smooth
    surface and equalise them there. The surface is the implicit field of
    the same graph with tubes of the map's own radius; each round moves
    every vertex toward the mean of its neighbours and projects it back,
    so only the tangential part survives. Connectivity, and so the census,
    is never touched.
    """
    from . import implicit as im
    from . import remesh as rm

    live = None if cell is None or not np.any(np.asarray(cell) > 0) \
        else np.asarray(cell, dtype=float)
    segments = []
    for i, j, shift in edges:
        far = centres[j] + (np.zeros(3) if live is None
                            else np.asarray(shift, float) * live)
        segments.append((centres[i], far))
    segments = np.array(segments)
    box = live if live is not None else np.ones(3)
    pbc = tuple(bool(x) for x in (live > 0)) if live is not None \
        else (False, False, False)
    field = im.graph_field(segments, box, pbc, tube_radius,
                           blend_fraction * tube_radius)
    vertices = im.project_to_surface(field, np.array(mesh[0], dtype=float))
    if live is not None:
        vertices = _wrap_live(vertices, live)
    current = (vertices, np.asarray(mesh[1]))
    for _ in range(iterations):
        current = rm._tangential_smooth(current, field, strength=0.5,
                                        box=live)
    return current


def _wrap_live(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    out = np.array(points, dtype=float, copy=True)
    live = cell > 0
    out[:, live] = np.mod(out[:, live], cell[live])
    return out


def equilateralise(mesh: Mesh, box: np.ndarray | None = None,
                   edge: float = _EDGE, k_repel: float = 2.0,
                   iterations: int = 4000) -> Mesh:
    """Move the coarse vertices until every coarse triangle is equilateral.

    This is the step every clean coarse map in this package shares and the
    hull map lacked. The muoctahedron, C168's Klein map and the octahedral
    frames are all made of equilateral triangles, so the graphene patch
    in each is the exact lattice, unsheared, and every bit of curvature
    sits at a disclination or a fold. A coarse triangle that is not
    equilateral is refined into an affine image of the lattice -- sheared
    graphene -- and on the first hull cages that alone left bonds at
    1.19-1.66 Å whatever was done afterwards.

    Springs pull every coarse edge to ``edge``; a short-range repulsion
    keeps vertices that are not neighbours apart, so the surface cannot
    fold onto itself while it finds a deltahedral shape. Connectivity is
    untouched.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    from .fullerene_mesh import minimum_image

    live = None if box is None or not np.any(np.asarray(box) > 0) \
        else np.asarray(box, dtype=float)
    vertices, tris = np.array(mesh[0], dtype=float), np.asarray(mesh[1])
    pairs = set()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            pairs.add((min(u, v), max(u, v)))
    edges = np.array(sorted(pairs), dtype=int)
    adjacent = {tuple(e) for e in edges}
    n = len(vertices)

    def energy(flat):
        x = flat.reshape(n, 3)
        d = minimum_image(x[edges[:, 1]] - x[edges[:, 0]], live)
        length = np.linalg.norm(d, axis=1)
        e = float(((length - edge) ** 2).sum())
        grad = np.zeros_like(x)
        coeff = (2.0 * (length - edge) / np.maximum(length, 1e-12))[:, None] * d
        np.add.at(grad, edges[:, 1], coeff)
        np.add.at(grad, edges[:, 0], -coeff)
        wrapped = x if live is None else _wrap_live(x, live)
        if live is not None:
            # Clip the periodic axes only. Clipping every axis to
            # nextafter(box, 0) sends each coordinate on an axis that does
            # not repeat (box 0) to 0, which collapsed a supertube's whole
            # cross-section in the tree: far-apart vertices were found as
            # pairs and the "repulsion", with a negative gap, pulled them
            # together -- coarse edges 0.04-18 after a 0.65-1.90 input.
            on = live > 0
            wrapped[:, on] = np.minimum(wrapped[:, on],
                                        np.nextafter(live[on], 0))
        tree = cKDTree(wrapped, boxsize=live) if live is not None \
            else cKDTree(wrapped)
        for i, j in tree.query_pairs(1.4 * edge):
            if (min(i, j), max(i, j)) in adjacent:
                continue
            dd = minimum_image(x[j] - x[i], live)
            r = float(np.linalg.norm(dd))
            gap = 1.4 * edge - r
            if gap <= 0.0:
                continue
            e += k_repel * gap * gap
            g = -2.0 * k_repel * gap * dd / max(r, 1e-12)
            grad[j] += g
            grad[i] -= g
        return e, grad.ravel()

    result = minimize(energy, vertices.ravel(), jac=True, method="L-BFGS-B",
                      options={"maxiter": iterations})
    out = result.x.reshape(n, 3)
    return (out if live is None else _wrap_live(out, live)), tris


def coarse_score(tris: np.ndarray, n: int) -> tuple[int, int]:
    """(rings outside 5-7, non-hexagonal rings): lower is better, in order."""
    degree = _degrees(tris, n)
    return int(np.sum((degree < 5) | (degree > 7))), int(np.sum(degree != 6))


def best_hull_map(points, edges, box=None, rows: int = 2,
                  sides_options=(5, 6, 7), phases=(0.0, 0.25, 0.5, 0.75),
                  aware: bool = True):
    """Search tube polygon and mouth phase for the cleanest coarse census.

    Each candidate is built and optimised (both cheap: the coarse map has
    tens of vertices per node), and scored first by rings outside 5-7,
    then by how many rings are not hexagons. Returns
    ``(mesh, budget, scale, sides, phase, history)``.
    """
    best = None
    for sides in sides_options:
        for phase in phases:
            mesh, budget, scale = hull_map(points, edges, box, sides=sides,
                                           rows=rows, phase=phase)
            cell = None if box is None else np.asarray(box) * scale
            targets = curvature_charges(mesh, cell) if aware else None
            mesh, history = optimise_census(mesh, cell, targets=targets)
            score = coarse_score(mesh[1], len(mesh[0]))
            if aware:
                # Rings outside 5-7 first, then how far the census is from
                # where the curvature puts the budget.
                misfit = float(((6 - _degrees(mesh[1], len(mesh[0]))
                                 - curvature_charges(mesh, cell)) ** 2).sum())
                score = (score[0], misfit)
            if best is None or score < best[0]:
                best = (score, mesh, budget, scale, sides, phase, history)
    return best[1:]


def build_hull_superstructure(
    graph="super-icosahedron",
    m: int = 3,
    sides: int | None = None,
    rows: int = 2,
    optimise: bool = True,
    phase: float = 0.0,
    bond: float = CC_BOND,
    vacuum: float = 12.0,
    relax: bool = True,
    relax_iterations: int = 3000,
    smooth: int = 30,
    surface: bool = False,
    loop: bool = False,
    equilateral: bool = True,
) -> Atoms:
    """A crystalline tube superstructure on any graph.

    Parameters
    ----------
    graph
        A :class:`~.supernetwork.SuperGraph` or a name from that module's
        catalogue (periodic nets and cages alike). Only its topology and
        its arm directions are used; the size comes from the tubes.
    m
        Goldberg-Coxeter index of the inflation: tube circumference
        ``sides * m`` rings, defects ``m`` rings apart.
    sides, rows
        Coarse tube polygon and rows on the shortest arm. ``sides=None``
        searches polygon and phase (:func:`best_hull_map`).
    optimise
        Run :func:`optimise_census` on the coarse map first.
    phase
        Mouth rotation, as a fraction of a polygon step.
    bond, vacuum, relax, relax_iterations
        As in :func:`.frame.build_frame`.
    smooth, surface
        Rounds of smoothing of the refined mesh before the dual. With
        ``surface`` (the default) each round is a tangential Laplacian step
        projected back onto the smooth tube surface of the same graph
        (:func:`settle_on_tubes`); without it, plain Taubin rounds.
    """
    from .supernetwork import named_graph

    if isinstance(graph, str):
        graph = named_graph(graph, 1.0)
    periodic = graph.periodic
    if periodic:
        box = np.asarray(graph.cell(1.0), dtype=float)
        points = np.asarray(graph.nodes, dtype=float) * box
        live = np.where(np.array(graph.pbc), box, 0.0)
    else:
        points = np.asarray(graph.nodes, dtype=float)
        live = None
    if sides is None:
        coarse, budget, scale, sides, phase, history = best_hull_map(
            points, graph.edges, live, rows=rows)
        cell = None if live is None else live * scale
    else:
        coarse, budget, scale = hull_map(points, graph.edges, live,
                                         sides=sides, rows=rows, phase=phase)
        cell = None if live is None else live * scale
        history = []
        if optimise:
            coarse, history = optimise_census(
                coarse, cell, targets=curvature_charges(coarse, cell))
    if equilateral:
        coarse = equilateralise(coarse, cell)
    # m = 2**k goes through Loop subdivision, which refines exactly as
    # GC(2,0) does but onto the coarse map's own smooth limit surface;
    # any other m is a flat GC refinement.
    levels = int(round(np.log2(m))) if m >= 2 and 2 ** int(round(np.log2(m))) == m \
        else 0
    if loop and levels:
        fine = loop_subdivide(coarse, cell, levels=levels)
        surface = False
    else:
        fine = subdivide(coarse, m, cell)
    if smooth and surface:
        fine = settle_on_tubes(fine, points * scale, graph.edges, cell,
                               tube_radius=_EDGE / (2.0 * np.sin(np.pi / sides)),
                               iterations=smooth)
    elif smooth:
        fine = taubin_smooth(fine, cell, iterations=smooth)
    atoms = finish_refined(fine, cell, tuple(bool(x) for x in graph.pbc)
                           if periodic else (False, False, False), budget,
                           bond=bond, relax=relax,
                           relax_iterations=relax_iterations, vacuum=vacuum,
                           label=graph.name, cell_scan=periodic)
    coarse_census = defaultdict(int)
    for d in _degrees(coarse[1], len(coarse[0])):
        coarse_census[int(d)] += 1
    atoms.info.update({
        "builder": "hull",
        "structure_type": "supernetwork",
        "kind": graph.name,
        "gc_indices": (int(m), 0),
        "tube_sides": int(sides),
        "mouth_phase": float(phase),
        "arm_rows": int(rows),
        "n_nodes": len(graph.nodes),
        "n_struts": len(graph.edges),
        "ring_budget": int(budget),
        "euler": 2 * (len(graph.nodes) - len(graph.edges)),
        "coarse_census": dict(sorted(coarse_census.items())),
        "optimisation_history": history,
    })
    return atoms
