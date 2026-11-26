"""Deltahedral carriers: geodesic-sphere nodes, antiprism arms, GC inflation.

The rule every clean coarse map in this package obeys (see the hull
module's notes) is that its triangles are equilateral: refined, each one
becomes unsheared graphene and the curvature stays at the defects. This
module builds such a carrier for a tube superstructure whose arm
directions are not the bcc lattice's:

* every **node** is a geodesic sphere -- an icosahedron, class-I
  subdivided ``freq`` times and projected, near-equilateral everywhere,
  twelve vertices of degree five and the rest of degree six;
* every **arm** leaves through the star of the sphere vertex lying along
  it, and is a stack of hexagonal (or pentagonal) **antiprisms** -- each
  ring turned half a step on the last, so every strip triangle is
  equilateral -- with the residual twist needed to land on the next
  node's window spread evenly along it.

The census follows from the construction and is checked against the
graph's ``12(V - E)``: each window's rim comes out heptagonal (it loses
two triangles to the window and gains three from the tube), and the
twelve degree-five vertices of every node stay pentagons -- on a node,
which is convex, where positive curvature belongs.

The node sphere is oriented by the graph itself: for an icosahedral cage
it is the cage's own icosahedron, so the five arms at a corner lie
exactly on its two-fold axes, which are vertices of the sphere at every
even frequency.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.spatial import ConvexHull

from ..utils.constants import CC_BOND
from .goldberg import Mesh, _Builder, finish_refined, subdivide, taubin_smooth

_ROW = np.sqrt(3.0) / 2.0


def geodesic_sphere(directions: np.ndarray, freq: int) -> Mesh:
    """A geodesic sphere of unit mean edge from 12 icosahedral directions.

    ``directions`` are the icosahedron's vertices, in any frame; their
    convex hull gives the twenty faces, class-I subdivision by ``freq``
    and projection give the sphere, and it is rescaled so the mean edge is
    one coarse unit.
    """
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    hull = ConvexHull(unit)
    tris = hull.simplices.copy()
    for k, (simplex, eq) in enumerate(zip(hull.simplices, hull.equations,
                                          strict=True)):
        a, b, c = unit[simplex]
        if np.dot(np.cross(b - a, c - a), eq[:3]) < 0:
            tris[k] = simplex[::-1]
    vertices, tris = subdivide((unit, tris), freq)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    edges = set()
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    mean = np.mean([np.linalg.norm(vertices[i] - vertices[j])
                    for i, j in edges])
    return vertices / mean, tris


def _six_fold(tris: np.ndarray, n: int) -> np.ndarray:
    """Mask of the sphere vertices of degree six -- the only window sites.

    A window on a five-fold vertex rims a pentagonal mouth, and a tube
    cannot join it to a hexagonal one at the far end; the twelve five-fold
    vertices are kept closed (they are the node's pentagons).
    """
    return np.bincount(np.asarray(tris).ravel(), minlength=n) == 6


def _star(tris: np.ndarray, centre: int) -> tuple[list[int], list[int]]:
    """Triangles around ``centre`` and its ordered link (the window rim)."""
    around = [k for k, t in enumerate(tris) if centre in t]
    nxt = {}
    for k in around:
        t = list(tris[k])
        i = t.index(centre)
        nxt[t[(i + 1) % 3]] = t[(i + 2) % 3]
    start = next(iter(nxt))
    rim = [start]
    while nxt[rim[-1]] != start:
        rim.append(nxt[rim[-1]])
    return around, rim


def _arm_vectors(points: np.ndarray, edges, cell=None) -> list[np.ndarray]:
    """Each edge's vector from its start node to its end node's image.

    ``cell`` holds the cell edges for a periodic graph (an edge's integer
    shift is in cells); ``None`` for a finite one.
    """
    cell = np.zeros(3) if cell is None else np.asarray(cell, dtype=float)
    return [np.asarray(points[j], float) + np.asarray(shift, float) * cell
            - np.asarray(points[i], float) for i, j, shift in edges]


def geodesic_map(points: np.ndarray, edges, frame: np.ndarray,
                 freq: int = 4, rows: int = 2, tolerance_deg: float = 2.0,
                 cell=None) -> tuple[Mesh, int, float, dict]:
    """The coarse carrier of a finite graph of tubes on geodesic nodes.

    Parameters
    ----------
    points, edges
        The graph (finite): node positions and ``(i, j, shift)`` edges.
    frame
        Twelve icosahedral directions fixing every node sphere's
        orientation, or one such set per node, shaped ``(V, 12, 3)``
        (:func:`fit_node_frames`).
    freq
        Geodesic frequency of the node spheres.
    rows
        Antiprism rows on the shortest arm; the graph is scaled to fit.
    cell
        For a periodic graph, the cell edges (``0`` on an axis that does
        not repeat), in the units of ``points``. The map is then built in
        that cell, scaled with everything else, and returned periodic.
    tolerance_deg
        How far an arm may miss the nearest sphere vertex. An icosahedral
        cage's arms miss by zero; anything larger is refused rather than
        bent.

    Returns
    -------
    (mesh, budget, scale, report)
        ``report["box"]`` is the scaled cell (zeros when finite).
    """
    points = np.asarray(points, dtype=float)
    frame = np.asarray(frame, dtype=float)
    if frame.ndim == 2:
        frame = np.broadcast_to(frame, (len(points),) + frame.shape)
    spheres = [geodesic_sphere(f, freq) for f in frame]
    radius = float(np.mean(np.linalg.norm(spheres[0][0], axis=1)))

    vectors = _arm_vectors(points, edges, cell)
    # Which sphere vertex each arm leaves by, at each end. Keyed by
    # (edge, end): a one-node periodic net joins a node to its own image,
    # so the node alone does not name the window.
    windows: dict[int, dict[tuple[int, int], int]] = {
        i: {} for i in range(len(points))}
    worst = 0.0
    for e, (i, j, _shift) in enumerate(edges):
        d = vectors[e] / np.linalg.norm(vectors[e])
        for end, node, vec in ((0, i, d), (1, j, -d)):
            sv, st = spheres[node]
            cos = (sv / np.linalg.norm(sv, axis=1, keepdims=True)) @ vec
            cos = np.where(_six_fold(st, len(sv)), cos, -2.0)
            k = int(np.argmax(cos))
            miss = float(np.degrees(np.arccos(np.clip(cos[k], -1, 1))))
            worst = max(worst, miss)
            if miss > tolerance_deg:
                raise ValueError(
                    f"an arm misses the nearest node-sphere vertex by "
                    f"{miss:.1f} deg; the geodesic carrier at frequency "
                    f"{freq} has no window along it.")
            windows[node][(e, end)] = k

    # Scale: node spheres sit at radius `radius`; the arm between windows
    # must hold `rows` antiprism rows on the shortest edge.
    lengths = [float(np.linalg.norm(v)) for v in vectors]
    scale = (2.0 * radius + rows * _ROW) / min(lengths)
    centres = points * scale
    box = (np.zeros(3) if cell is None
           else np.asarray(cell, dtype=float) * scale)
    build = _Builder(box)
    # Rims relative to their node's centre: the far end of a periodic arm
    # is an image of its node, not the node.
    rims: dict[tuple[int, int], list[np.ndarray]] = {}

    for node in range(len(points)):
        sphere_v, sphere_t = spheres[node]
        removed: set[int] = set()
        for key, k in windows[node].items():
            around, rim = _star(sphere_t, k)
            if removed & set(around):
                raise ValueError("two windows on one node overlap; raise "
                                 "the geodesic frequency.")
            removed |= set(around)
            rims[key] = [sphere_v[r] for r in rim]
        for t, (a, b, c) in enumerate(sphere_t):
            if t in removed:
                continue
            build.tris.append(tuple(build.vertex(centres[node] + sphere_v[x])
                                    for x in (a, b, c)))

    for e, (i, _j, _shift) in enumerate(edges):
        far = centres[i] + vectors[e] * scale
        start = [centres[i] + v for v in rims[(e, 0)]]
        end = [far + v for v in rims[(e, 1)]]
        axis = far - centres[i]
        axis /= np.linalg.norm(axis)
        p = np.cross(axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(p) < 0.1:
            p = np.cross(axis, [1.0, 0.0, 0.0])
        p /= np.linalg.norm(p)
        q = np.cross(axis, p)
        c0 = np.mean(start, axis=0)
        c1 = np.mean(end, axis=0)
        n = len(start)
        if len(end) != n:
            raise ValueError("an arm joins windows of different sizes.")

        def polar(ring, centre, p=p, q=q, axis=axis):
            rel = [v - centre for v in ring]
            ang = np.array([np.arctan2(np.dot(r, q), np.dot(r, p))
                            for r in rel])
            rad = np.array([np.linalg.norm(r - np.dot(r, axis) * axis)
                            for r in rel])
            return ang, rad

        a0, r0 = polar(start, c0)
        # Orient both rims the same way round the axis.
        if ((np.angle(np.exp(1j * (a0[1] - a0[0])))) < 0):
            start = start[::-1]
            a0, r0 = polar(start, c0)
        a1, r1 = polar(end, c1)
        if ((np.angle(np.exp(1j * (a1[1] - a1[0])))) < 0):
            end = end[::-1]
            a1, r1 = polar(end, c1)
        length = float(np.linalg.norm(c1 - c0))
        count = max(1, int(round(length / _ROW)))
        step = 2.0 * np.pi / n
        # Index shift and residual twist so that ring `count`, turned half
        # a step per row from the start, lands on the end rim.
        best = None
        for shift in range(n):
            target = np.angle(np.exp(1j * (a1[shift] - a0[0]
                                          - count * step / 2.0)))
            if best is None or abs(target) < abs(best[1]):
                best = (shift, target)
        shift, residual = best
        end = end[shift:] + end[:shift]
        layers = [start]
        for k in range(1, count):
            t = k / count
            ring = []
            for a in range(n):
                ang = a0[a] + k * step / 2.0 + t * residual
                rad = (1 - t) * r0[a] + t * r1[(a + shift) % n]
                centre = c0 + t * (c1 - c0)
                ring.append(centre + rad * (np.cos(ang) * p + np.sin(ang) * q))
            layers.append(ring)
        layers.append(end)
        ids = [[build.vertex(v) for v in ring] for ring in layers]
        for k in range(count):
            for a in range(n):
                b = (a + 1) % n
                p0 = layers[k][a]
                out = p0 - c0 - np.dot(p0 - c0, axis) * axis
                tri1 = (ids[k][a], ids[k][b], ids[k + 1][a])
                tri2 = (ids[k + 1][a], ids[k][b], ids[k + 1][b])
                cross = np.cross(layers[k][b] - p0, layers[k + 1][a] - p0)
                flip = np.dot(cross, out) < 0
                for tri in (tri1, tri2):
                    build.tris.append(tri[::-1] if flip else tri)

    mesh = build.mesh()
    # Orient the node triangles consistently with the arms: outward from
    # their node centre.
    vertices, tris = mesh
    budget = 12 * (len(points) - len(edges))
    report = {"worst_arm_miss_deg": round(worst, 3), "node_radius": radius,
              "box": box}
    return (vertices, tris), budget, scale, report


def build_geodesic_superstructure(
    graph="super-icosahedron",
    m: int = 2,
    freq: int = 2,
    rows: int = 1,
    equilateral: bool = True,
    smooth: int = 10,
    tolerance_deg: float = 2.5,
    bond: float = CC_BOND,
    vacuum: float = 12.0,
    relax: bool = True,
    relax_iterations: int = 3000,
) -> Atoms:
    """A cage or periodic net of tubes on geodesic nodes, as graphene.

    Parameters
    ----------
    graph
        A :class:`~.supernetwork.SuperGraph` or catalogue name, finite or
        periodic. A periodic net is built in its own cell (vacuum on an
        axis that does not repeat) and the cell is scanned, as for the
        octahedral frames. The node spheres share one fitted frame when
        that puts every arm within ``tolerance_deg`` of a window site, and
        are fitted node by node otherwise (:func:`fit_node_frames`).
    m
        Goldberg-Coxeter index: tube circumference ``6m`` rings.
    freq, rows
        Node sphere frequency and antiprism rows on the shortest arm.
    equilateral
        Relax the coarse carrier to equal edges first
        (:func:`.hull.equilateralise`).
    tolerance_deg
        How far an arm may miss its window's sphere vertex. The frame is
        fitted first (:func:`fit_frame`); icosahedral, dodecahedral and
        cubic cages fit exactly.
    smooth, bond, vacuum, relax, relax_iterations
        As for the other carrier builders.
    """
    from .hull import equilateralise
    from .supernetwork import named_graph

    if isinstance(graph, str):
        graph = named_graph(graph, 1.0)
    points, cell = _graph_frame(graph)
    frame, miss = fit_frame(points, graph.edges, freq, cell=cell)
    per_node = miss > tolerance_deg
    if per_node:
        frame, miss = fit_node_frames(points, graph.edges, freq, cell=cell)
    coarse, budget, scale, report = geodesic_map(
        points, graph.edges, frame, freq=freq, rows=rows,
        tolerance_deg=tolerance_deg, cell=cell)
    box = report.pop("box")
    live = box if np.any(box > 0) else None
    if equilateral:
        coarse = equilateralise(coarse, live)
    fine = subdivide(coarse, m, live)
    if smooth:
        fine = taubin_smooth(fine, live, iterations=smooth)
    atoms = finish_refined(fine, live, tuple(bool(x) for x in graph.pbc),
                           budget, bond=bond, relax=relax,
                           relax_iterations=relax_iterations, vacuum=vacuum,
                           label=graph.name, cell_scan=live is not None)
    degree = np.bincount(np.asarray(coarse[1]).ravel())
    atoms.info.update({
        "builder": "geodesic",
        "structure_type": "supernetwork",
        "kind": graph.name,
        "gc_indices": (int(m), 0),
        "node_frequency": int(freq),
        "node_frames": "per node" if per_node else "shared",
        "arm_rows": int(rows),
        "n_nodes": len(points),
        "n_struts": len(graph.edges),
        "ring_budget": int(budget),
        "euler": 2 * (len(points) - len(graph.edges)),
        "coarse_census": {int(d): int((degree == d).sum())
                          for d in sorted(set(degree))},
        **report,
    })
    return atoms


def _graph_frame(graph) -> tuple[np.ndarray, np.ndarray | None]:
    """Cartesian node positions and, if periodic, the cell edges.

    A periodic :class:`~.supernetwork.SuperGraph` stores fractional nodes
    and a cell ``shape``; an axis that does not repeat gets a zero edge,
    which is what the mesh builders read as "finite along this axis".
    """
    nodes = np.asarray(graph.nodes, dtype=float)
    if not graph.periodic:
        return nodes, None
    shape = np.asarray(graph.shape, dtype=float)
    cell = np.where(np.asarray(graph.pbc, bool), shape, 0.0)
    return nodes * shape, cell


def fit_frame(points: np.ndarray, edges, freq: int,
              starts: int = 24, seed: int = 0,
              cell=None) -> tuple[np.ndarray, float]:
    """The icosahedral frame whose geodesic vertices best fit every arm.

    The worst arm's miss from its nearest sphere vertex is minimised over
    rotations of the golden icosahedron, from several starts. An
    icosahedral, dodecahedral or cubic cage fits to zero -- their edges lie
    on two-fold axes -- and anything else gets the best compromise, which
    the builder then refuses or accepts by its tolerance.

    Returns ``(frame, worst_miss_deg)``.
    """
    arms = [v / np.linalg.norm(v) for v in _arm_vectors(points, edges, cell)]
    return _fit_arms(np.array(arms + [-a for a in arms]), freq, starts, seed)


def fit_node_frames(points: np.ndarray, edges, freq: int, starts: int = 8,
                    seed: int = 0, cell=None
                    ) -> tuple[np.ndarray, float]:
    """One icosahedral frame per node, each fitted to that node's arms.

    A graph whose nodes are not all images of one another under the
    icosahedral group -- a rolled super-graphene (supertube), the
    hypercube's projection -- has no single frame that puts every arm on
    a sphere vertex: measured, the best global fits miss by 7-16 deg. But
    each node alone has three to five arms, and those fit. The arms then
    join rims of differently turned spheres, which the antiprism stacking
    already handles (it matches any two rims by an index shift and a
    residual twist).

    Returns ``(frames, worst_miss_deg)`` with ``frames`` shaped
    ``(V, 12, 3)``.
    """
    vectors = _arm_vectors(points, edges, cell)
    per_node: list[list[np.ndarray]] = [[] for _ in range(len(points))]
    for (i, j, _shift), v in zip(edges, vectors, strict=True):
        u = v / np.linalg.norm(v)
        per_node[i].append(u)
        per_node[j].append(-u)
    frames, worst = [], 0.0
    for arms in per_node:
        frame, miss = _fit_arms(np.array(arms), freq, starts, seed)
        frames.append(frame)
        worst = max(worst, miss)
    return np.array(frames), worst


def _fit_arms(arms: np.ndarray, freq: int, starts: int, seed: int
              ) -> tuple[np.ndarray, float]:
    """The rotated golden icosahedron whose sphere best fits ``arms``."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    golden = _icosahedral_frame(np.zeros((1, 3)))
    sphere, sphere_t = geodesic_sphere(golden, freq)
    dirs = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    dirs = dirs[_six_fold(sphere_t, len(sphere))]

    def worst(rotvec):
        rotated = Rotation.from_rotvec(rotvec).apply(dirs)
        best = np.max(arms @ rotated.T, axis=1)
        return float(np.degrees(np.arccos(np.clip(best.min(), -1, 1))))

    def smooth(rotvec):
        rotated = Rotation.from_rotvec(rotvec).apply(dirs)
        best = np.max(arms @ rotated.T, axis=1)
        return float(np.sum(1.0 - best))

    rng = np.random.default_rng(seed)
    candidates = [np.zeros(3)] + [Rotation.random(random_state=rng).as_rotvec()
                                  for _ in range(starts)]
    if freq > 2 and freq % 2 == 0:
        # A frequency-2 sphere's vertices are a subset of every even
        # frequency's, so its fit is always a candidate -- and the
        # optimiser on the finer sphere, with more local minima, measurably
        # misses it (dodecahedron: 0 deg at frequency 2, 6 at 4).
        coarse_frame, _ = _fit_arms(arms, 2, starts, seed)
        candidates.append(Rotation.align_vectors(coarse_frame,
                                                 golden)[0].as_rotvec())
    best = None
    for start in candidates:
        result = minimize(smooth, start, method="Nelder-Mead",
                          options={"xatol": 1e-8, "fatol": 1e-12,
                                   "maxiter": 4000})
        score = worst(result.x)
        if best is None or score < best[0]:
            best = (score, result.x)
    frame = Rotation.from_rotvec(best[1]).apply(golden)
    return frame, best[0]


def _icosahedral_frame(points: np.ndarray) -> np.ndarray:
    """Twelve icosahedral directions sharing the graph's own frame.

    For an icosahedral cage the nodes ARE such a set. Otherwise the frame
    is the standard golden-ratio icosahedron, which only helps when the
    graph's arms happen to lie on its two- or five-fold axes.
    """
    centred = points - points.mean(axis=0)
    if len(points) == 12:
        norms = np.linalg.norm(centred, axis=1)
        if np.allclose(norms, norms[0], rtol=1e-6):
            return centred / norms[:, None]
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    base = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            base += [(0.0, s1, s2 * phi), (s1, s2 * phi, 0.0),
                     (s2 * phi, 0.0, s1)]
    return np.array(base)


#: The cages and nets this carrier builds, with the frequency and
#: tolerance that fit each one: icosahedron, cube, dodecahedron and the
#: C60 of tubes put every arm exactly on a two-fold axis of a frequency-2
#: sphere; the tetrahedron and the octahedron miss by 2.1 deg.
GEODESIC_CAGES: dict[str, dict] = {
    "super-icosahedron": {"freq": 2, "tolerance_deg": 0.5},
    "super-cube": {"freq": 2, "tolerance_deg": 0.5},
    "super-dodecahedron": {"freq": 2, "tolerance_deg": 0.5},
    "superfullerene-C60": {"freq": 2, "tolerance_deg": 0.5},
    # At frequency 2 their 60-deg arms share window rims (octagons and
    # nonagons, strained); at 4 they are clean.
    "super-tetrahedron": {"freq": 4, "tolerance_deg": 2.5},
    "super-octahedron": {"freq": 4, "tolerance_deg": 2.5},
    # Periodic nets, each in its own cell. <100> arms are two-fold axes of
    # the golden icosahedron and <111> arms three-fold ones (face centres,
    # sphere vertices at frequency 3); the honeycomb's 120-deg arms fit a
    # frequency-2 sphere exactly too.
    "super-square": {"freq": 2, "tolerance_deg": 0.5},
    "super-graphene": {"freq": 2, "tolerance_deg": 0.5},
    "super-cubic": {"freq": 2, "tolerance_deg": 0.5},
    "super-diamond": {"freq": 3, "tolerance_deg": 0.5},
    # Twelve <110> arms: no frequency-2 window set avoids shared rims, and
    # at 4 they miss by 2.1 deg and come out clean -- the first exact
    # route to the fcc net, which the octahedral frames cannot build.
    "super-fcc": {"freq": 4, "tolerance_deg": 2.5},
    "super-cuboctahedron": {"freq": 4, "tolerance_deg": 2.5},
    "super-icosidodecahedron": {"freq": 2, "tolerance_deg": 0.5},
    # The projected tesseract has two kinds of node and no shared frame
    # (best 14 deg); fitted node by node, frequency 6 puts every arm on a
    # window exactly.
    "super-hypercube": {"freq": 6, "tolerance_deg": 0.5},
    "super-truncated-cube": {"freq": 4, "tolerance_deg": 2.5},
}


def geodesic_atom_count(graph: str = "super-icosahedron", m: int = 2,
                        freq: int = 2, rows: int = 1) -> int:
    """Atoms the build will have: coarse triangles times ``m**2``, exactly."""
    from .supernetwork import named_graph

    g = named_graph(graph, 1.0)
    points, cell = _graph_frame(g)
    tolerance = GEODESIC_CAGES.get(graph, {}).get("tolerance_deg", 2.5)
    frame, miss = fit_frame(points, g.edges, freq, starts=8, cell=cell)
    if miss > tolerance:
        frame, _ = fit_node_frames(points, g.edges, freq, cell=cell)
    (_, tris), _, _, _ = geodesic_map(points, g.edges, frame, freq=freq,
                                      rows=rows, tolerance_deg=tolerance,
                                      cell=cell)
    return len(tris) * m * m


def describe_geodesic(atoms: Atoms) -> str:
    """One line: cage, size, census, the law it meets."""
    info = atoms.info
    census = ", ".join(f"{k}: {v}" for k, v in info["ring_counts"].items())
    return (f"{info['kind']} on frequency-{info['node_frequency']} geodesic "
            f"nodes, GC{info['gc_indices']}: {len(atoms)} atoms, rings "
            f"{{{census}}}, sum(6-n) = {info['ring_deficit']:+d} = 12(V - E) "
            f"for V = {info['n_nodes']}, E = {info['n_struts']}; worst arm "
            f"miss {info['worst_arm_miss_deg']} deg")
