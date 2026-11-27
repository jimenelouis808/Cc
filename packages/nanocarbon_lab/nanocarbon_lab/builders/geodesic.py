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
from .goldberg import (
    Mesh,
    _Builder,
    finish_refined,
    goldberg_coxeter,
    subdivide,
    taubin_smooth,
)

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


def _six_fold(tris: np.ndarray, n: int, window: int = 6) -> np.ndarray:
    """Mask of the sphere vertices allowed as window sites.

    Both ends of an arm must rim the same polygon. With **one shared
    frame** that is automatic: the sphere is centrosymmetric, so the
    vertices nearest ``d`` and ``-d`` are antipodes of the same degree,
    and ``window = 0`` lets every vertex be a site. Differently oriented
    spheres (per-node or group frames) have no such guarantee -- one end
    of the first per-node supertube arm landed on a five-fold vertex and
    the other on a six-fold one -- so there windows open on degree-6
    vertices only (``window = 6``). Restricting the shared case as well
    is what lost super-fcc: its <110> arms fit mixed sites to 2.1 deg at
    frequency 4, six-fold ones only to 4.7.
    """
    if window == 0:
        return np.ones(n, dtype=bool)
    return np.bincount(np.asarray(tris).ravel(), minlength=n) == window


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


def _patch(tris: np.ndarray, centre: int, radius: int
           ) -> tuple[list[int], list[int]]:
    """A hexagonal window of ``radius`` rings round ``centre``, and its rim.

    ``radius = 1`` is :func:`_star`. Larger windows take every triangle
    within ``radius`` edges of the centre (all corners at distance
    ``<= radius``, one strictly inside), so the mouth grows with the
    sphere instead of staying one edge across while the node swells
    around it. The rim is the patch's boundary, ordered with the
    triangles' own winding.
    """
    if radius == 1:
        return _star(tris, centre)
    neighbours: dict[int, set[int]] = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            neighbours.setdefault(int(u), set()).add(int(v))
            neighbours.setdefault(int(v), set()).add(int(u))
    dist = {int(centre): 0}
    frontier = [int(centre)]
    for d in range(1, radius + 1):
        nxt = []
        for u in frontier:
            for v in neighbours[u]:
                if v not in dist:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    inside = [k for k, t in enumerate(tris)
              if all(int(x) in dist for x in t)
              and min(dist[int(x)] for x in t) < radius]
    directed = set()
    for k in inside:
        a, b, c = (int(x) for x in tris[k])
        directed |= {(a, b), (b, c), (c, a)}
    nxt = {u: v for u, v in directed if (v, u) not in directed}
    start = next(iter(nxt))
    rim = [start]
    while nxt[rim[-1]] != start:
        rim.append(nxt[rim[-1]])
        if len(rim) > len(nxt):
            raise ValueError("a window's rim is not one loop.")
    if len(rim) != len(nxt):
        raise ValueError("a window's rim is not one loop.")
    return inside, rim


def _arm_vectors(points: np.ndarray, edges, cell=None) -> list[np.ndarray]:
    """Each edge's vector from its start node to its end node's image.

    ``cell`` holds the cell edges for a periodic graph (an edge's integer
    shift is in cells); ``None`` for a finite one.
    """
    cell = np.zeros(3) if cell is None else np.asarray(cell, dtype=float)
    return [np.asarray(points[j], float) + np.asarray(shift, float) * cell
            - np.asarray(points[i], float) for i, j, shift in edges]


def _hermite(c0, c1, t0, t1, t):
    """Cubic Hermite point and unit tangent at ``t`` (tangents scaled by
    the chord, so a straight pair gives a straight, uniform line)."""
    chord = float(np.linalg.norm(c1 - c0))
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    point = h00 * c0 + h10 * chord * t0 + h01 * c1 + h11 * chord * t1
    d00 = 6 * t ** 2 - 6 * t
    d10 = 3 * t ** 2 - 4 * t + 1
    d01 = -6 * t ** 2 + 6 * t
    d11 = 3 * t ** 2 - 2 * t
    tangent = d00 * c0 + d10 * chord * t0 + d01 * c1 + d11 * chord * t1
    return point, tangent / np.linalg.norm(tangent)


def _stack_arm(build, start, end, t_start, t_end) -> None:
    """Stack antiprism rings from one window rim to the other.

    The arm runs along a cubic Hermite curve that leaves the first rim
    along ``t_start`` (its window's outward normal) and enters the second
    along ``t_end``, and the rings ride a rotation-minimising frame
    carried along it (double reflection). When both normals lie on the
    chord this is the straight antiprism stack exactly; when the sphere's
    window vertex misses the arm direction, the tube **bends** to leave
    its mouth square instead of shearing out of it -- which is what lets
    a node keep a low frequency (a small sphere) whatever the arm angles.
    Each ring turns half a step on the last, and the residual twist that
    lands the last ring on the far rim is spread along the arm.
    """
    start = list(start)
    end = list(end)
    n = len(start)
    if len(end) != n:
        raise ValueError("an arm joins windows of different sizes.")
    c0 = np.mean(start, axis=0)
    c1 = np.mean(end, axis=0)
    t_start = np.asarray(t_start, float) / np.linalg.norm(t_start)
    t_end = np.asarray(t_end, float) / np.linalg.norm(t_end)
    # Arc length, and the rotation-minimising frame along the curve.
    samples = 64
    ts = np.linspace(0.0, 1.0, samples + 1)
    pts, tans = zip(*(_hermite(c0, c1, t_start, t_end, t) for t in ts),
                    strict=True)
    pts, tans = np.array(pts), np.array(tans)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0),
                                                          axis=1))])
    p = start[0] - c0
    p = p - np.dot(p, tans[0]) * tans[0]
    p /= np.linalg.norm(p)
    frames = [p]
    for k in range(samples):
        v1 = pts[k + 1] - pts[k]
        c1_ = float(np.dot(v1, v1))
        r_l = p - (2.0 / c1_) * np.dot(v1, p) * v1
        t_l = tans[k] - (2.0 / c1_) * np.dot(v1, tans[k]) * v1
        v2 = tans[k + 1] - t_l
        c2 = float(np.dot(v2, v2))
        p = r_l if c2 < 1e-18 else r_l - (2.0 / c2) * np.dot(v2, r_l) * v2
        p = p - np.dot(p, tans[k + 1]) * tans[k + 1]
        p /= np.linalg.norm(p)
        frames.append(p)
    frames = np.array(frames)

    def frame_at(u):
        """Centre, tangent, p, q at arc fraction ``u``."""
        s_ = u * arc[-1]
        k = int(np.clip(np.searchsorted(arc, s_) - 1, 0, samples - 1))
        w = (s_ - arc[k]) / max(arc[k + 1] - arc[k], 1e-12)
        centre = (1 - w) * pts[k] + w * pts[k + 1]
        tangent = (1 - w) * tans[k] + w * tans[k + 1]
        tangent /= np.linalg.norm(tangent)
        pp = (1 - w) * frames[k] + w * frames[k + 1]
        pp = pp - np.dot(pp, tangent) * tangent
        pp /= np.linalg.norm(pp)
        return centre, tangent, pp, np.cross(tangent, pp)

    def polar(ring, centre, tangent, pp, qq):
        rel = [v - centre for v in ring]
        ang = np.array([np.arctan2(np.dot(r, qq), np.dot(r, pp))
                        for r in rel])
        rad = np.array([np.linalg.norm(r - np.dot(r, tangent) * tangent)
                        for r in rel])
        return ang, rad

    f0 = (c0,) + frame_at(0.0)[1:]
    f1 = (c1,) + frame_at(1.0)[1:]
    a0, r0 = polar(start, *f0)
    if np.angle(np.exp(1j * (a0[1] - a0[0]))) < 0:
        start = start[::-1]
        a0, r0 = polar(start, *f0)
    a1, r1 = polar(end, *f1)
    if np.angle(np.exp(1j * (a1[1] - a1[0]))) < 0:
        end = end[::-1]
        a1, r1 = polar(end, *f1)
    count = max(1, int(round(arc[-1] / _ROW)))
    step = 2.0 * np.pi / n
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
        centre, tangent, pp, qq = frame_at(t)
        ring = []
        for a in range(n):
            ang = a0[a] + k * step / 2.0 + t * residual
            rad = (1 - t) * r0[a] + t * r1[(a + shift) % n]
            ring.append(centre + rad * (np.cos(ang) * pp + np.sin(ang) * qq))
        layers.append(ring)
    layers.append(end)
    ids = [[build.vertex(v) for v in ring] for ring in layers]
    for k in range(count):
        centre, tangent, _, _ = frame_at(k / count)
        for a in range(n):
            b = (a + 1) % n
            p0 = layers[k][a]
            out = p0 - centre - np.dot(p0 - centre, tangent) * tangent
            tri1 = (ids[k][a], ids[k][b], ids[k + 1][a])
            tri2 = (ids[k + 1][a], ids[k][b], ids[k + 1][b])
            cross = np.cross(layers[k][b] - p0, layers[k + 1][a] - p0)
            flip = np.dot(cross, out) < 0
            for tri in (tri1, tri2):
                build.tris.append(tri[::-1] if flip else tri)


def largest_window(points: np.ndarray, edges, frame: np.ndarray, freq: int,
                   tolerance_deg: float = 2.0, cell=None, window: int = 6,
                   limit: int = 8) -> int:
    """The largest window radius whose windows do not touch, at ``freq``.

    The node is a sphere of about ``0.86 * freq`` edges radius and a
    window of radius ``r`` is a mouth about ``r`` edges across, so the
    ball-to-tube ratio is roughly ``0.86 * freq / r``: the largest ``r``
    the arm angles allow is the most compact node this frequency has.
    Windows of one node may not share a rim vertex (see
    :func:`geodesic_map`). Returns at least 1.
    """
    best = 1
    for r in range(2, limit + 1):
        try:
            geodesic_map(points, edges, frame, freq=freq, rows=1,
                         tolerance_deg=tolerance_deg, cell=cell,
                         window=window, window_radius=r)
        except ValueError:
            break
        best = r
    return best


def geodesic_map(points: np.ndarray, edges, frame: np.ndarray,
                 freq: int = 4, rows: int = 2, tolerance_deg: float = 2.0,
                 cell=None, window: int = 6, window_radius: int = 1
                 ) -> tuple[Mesh, int, float, dict]:
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
            cos = np.where(_six_fold(st, len(sv), window), cos, -2.0)
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
    normals: dict[tuple[int, int], np.ndarray] = {}

    for node in range(len(points)):
        sphere_v, sphere_t = spheres[node]
        removed: set[int] = set()
        rim_vertices: set[int] = set()
        for key, k in windows[node].items():
            around, rim = _patch(sphere_t, k, window_radius)
            # Radius-1 rims may share a vertex (the frequency-2 icosahedron
            # does, and is clean with an octagon there); a larger window
            # touching another leaves no closed row between them.
            touching = window_radius > 1 and rim_vertices & set(rim)
            if removed & set(around) or touching:
                raise ValueError("two windows on one node overlap; raise "
                                 "the geodesic frequency or shrink the "
                                 "window radius.")
            removed |= set(around)
            rim_vertices |= set(rim)
            rims[key] = [sphere_v[r] for r in rim]
            normals[key] = sphere_v[k] / np.linalg.norm(sphere_v[k])
        for t, (a, b, c) in enumerate(sphere_t):
            if t in removed:
                continue
            build.tris.append(tuple(build.vertex(centres[node] + sphere_v[x])
                                    for x in (a, b, c)))

    for e, (i, _j, _shift) in enumerate(edges):
        far = centres[i] + vectors[e] * scale
        start = [centres[i] + v for v in rims[(e, 0)]]
        end = [far + v for v in rims[(e, 1)]]
        _stack_arm(build, start, end, normals[(e, 0)], -normals[(e, 1)])

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
    n: int = 0,
    rows: int = 1,
    equilateral: bool = True,
    smooth: int = 10,
    tolerance_deg: float = 2.5,
    bond: float = CC_BOND,
    vacuum: float = 12.0,
    relax: bool = True,
    relax_iterations: int = 3000,
    seed: int = 0,
    window: int | None = None,
    window_radius: int = 1,
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
    n
        Second Goldberg-Coxeter index. Non-zero (and not ``m``) gives the
        chiral family (:func:`.goldberg.chiral_subdivide`): the same
        census, the patches turned against the coarse edges, and
        ``(m, n)`` the mirror of ``(n, m)``.
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
    window_radius
        Rings of the sphere each window takes (:func:`_patch`): the tube's
        circumference is ``6 * window_radius * m``, so for one tube
        diameter a larger window means a smaller ``m`` and a node that is
        mostly mouth instead of mostly ball. ``0`` takes the largest the
        arm angles allow (:func:`largest_window`).
    window
        Degree of the sphere vertices windows open on, by preference: 6
        (hexagonal mouths) or 5. ``None`` takes the catalogue entry's own,
        else 6. Only when no frame fits that way does a shared frame fall
        back to any vertex (see :func:`_choose_frames`), recorded as
        ``window_degree = 0``.
    seed
        Seed of the frame fit's random starts. Different seeds can land
        on different, equally good frames, and so on measurably different
        relaxed cells -- which is the construction's own noise, and the
        yardstick any comparison between two builds has to clear.
    """
    from .hull import equilateralise

    if isinstance(graph, str):
        graph = _named(graph)
    if window is None:
        window = GEODESIC_CAGES.get(graph.name, {}).get("window", 6)
    points, cell = _graph_frame(graph)
    frame, miss, per_node, window = _choose_frames(
        graph, points, cell, freq, tolerance_deg, seed, window=window)
    if window_radius == 0:
        window_radius = largest_window(points, graph.edges, frame, freq,
                                       tolerance_deg, cell, window)
    coarse, budget, scale, report = geodesic_map(
        points, graph.edges, frame, freq=freq, rows=rows,
        tolerance_deg=tolerance_deg, cell=cell, window=window,
        window_radius=window_radius)
    box = report.pop("box")
    live = box if np.any(box > 0) else None
    if equilateral:
        coarse = equilateralise(coarse, live)
    fine = goldberg_coxeter(coarse, m, n, live)
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
        "gc_indices": (int(m), int(n)),
        "node_frequency": int(freq),
        "node_frames": ("group" if per_node == "group" else
                        "per node" if per_node else "shared"),
        "frame_seed": int(seed),
        "window_degree": int(window),
        "window_radius": int(window_radius),
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


def _choose_frames(graph, points, cell, freq: int, tolerance_deg: float,
                   seed: int = 0, starts: int = 24, window: int = 6):
    """Node frames for a build, which kind they are, and the window rule.

    Tried in order, the first within ``tolerance_deg`` wins:

    1. **One shared frame, windows on degree-``window`` vertices.** Every
       rim has the same orientation, so the carrier is mirror-exact
       (super-cube's GC(2,1) and GC(1,2) agree to 2e-9), and every mouth
       is the same polygon.
    2. **Frames carried by the Wythoff group.** Exact under [5,3], whose
       every element is a symmetry of the sphere; a [4,3] mirror is not,
       turns a two-fold rim 90 deg and makes the arm pick a twist.
    3. **One frame per node**, degree-``window`` windows.
    4. **One shared frame, windows on any vertex.** Still safe, since a
       shared frame is centrosymmetric and an arm's two ends are
       antipodes of one degree; but it mixes pentagonal and hexagonal
       mouths, which measured worse wherever 1 or 3 fits (cuboctahedron
       and the truncated cages went strained or broken that way). It is
       here for super-fcc, whose <110> arms fit mixed sites to 2.1 deg
       and six-fold ones only to 4.7.

    If none fits, the per-node frames are returned and the map refuses.

    Returns ``(frame, worst_miss_deg, kind, window)`` with kind ``False``
    (shared), ``"group"`` or ``True`` (per node), and ``window = 0`` when
    any vertex may open.
    """
    frame, miss = fit_frame(points, graph.edges, freq, starts=starts,
                            seed=seed, cell=cell, window=window)
    if miss <= tolerance_deg:
        return frame, miss, False, window
    if graph.name.startswith("wythoff-"):
        from .wythoff import wythoff_frames

        group_frame, group_miss = wythoff_frames(graph.name, freq, seed=seed,
                                                 window=window)
        if group_miss <= tolerance_deg:
            return group_frame, group_miss, "group", window
    node_frame, node_miss = fit_node_frames(points, graph.edges, freq,
                                            cell=cell, seed=seed,
                                            window=window)
    if node_miss <= tolerance_deg:
        return node_frame, node_miss, True, window
    frame, miss = fit_frame(points, graph.edges, freq, starts=starts,
                            seed=seed, cell=cell, window=0)
    if miss <= tolerance_deg:
        return frame, miss, False, 0
    return node_frame, node_miss, True, window


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
              cell=None, window: int = 6) -> tuple[np.ndarray, float]:
    """The icosahedral frame whose geodesic vertices best fit every arm.

    The worst arm's miss from its nearest sphere vertex is minimised over
    rotations of the golden icosahedron, from several starts. An
    icosahedral, dodecahedral or cubic cage fits to zero -- their edges lie
    on two-fold axes -- and anything else gets the best compromise, which
    the builder then refuses or accepts by its tolerance.

    Returns ``(frame, worst_miss_deg)``.
    """
    arms = [v / np.linalg.norm(v) for v in _arm_vectors(points, edges, cell)]
    return _fit_arms(np.array(arms + [-a for a in arms]), freq, starts, seed,
                     window)


def fit_node_frames(points: np.ndarray, edges, freq: int, starts: int = 16,
                    seed: int = 0, cell=None, window: int = 6
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
        frame, miss = _fit_arms(np.array(arms), freq, starts, seed, window)
        frames.append(frame)
        worst = max(worst, miss)
    return np.array(frames), worst


def _fit_arms(arms: np.ndarray, freq: int, starts: int, seed: int,
              window: int = 6) -> tuple[np.ndarray, float]:
    """The rotated golden icosahedron whose sphere best fits ``arms``."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    golden = _icosahedral_frame(np.zeros((1, 3)))
    sphere, sphere_t = geodesic_sphere(golden, freq)
    dirs = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    dirs = dirs[_six_fold(sphere_t, len(sphere), window)]

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
        coarse_frame, _ = _fit_arms(arms, 2, starts, seed, window)
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
    "super-truncated-tetrahedron": {"freq": 4, "tolerance_deg": 2.5},
    "super-truncated-octahedron": {"freq": 4, "tolerance_deg": 2.5},
    # At frequency 4 its arms miss by 2.8-3.0 deg, and more rows make it
    # worse (broken at three); node by node at 6 they miss by 1.9.
    "super-rhombicuboctahedron": {"freq": 6, "tolerance_deg": 2.5},
    # Rolled super-graphene: no shared frame, per-node misses up to
    # 3.7 deg at frequency 4. Clean at two rows (measured on (4,4) only).
    "supertube-(4,4)": {"freq": 4, "tolerance_deg": 4.5},
    # GEODESIC_NETS. srs and lonsdaleite have nodes in several
    # orientations and fit node by node (exactly, at 16 starts; 4 missed
    # lonsdaleite's by 7 deg); bcc's eight <111> arms are three-fold axes.
    "super-srs": {"freq": 2, "tolerance_deg": 0.5},
    "super-srs-mirror": {"freq": 2, "tolerance_deg": 0.5},
    "super-bcc": {"freq": 3, "tolerance_deg": 0.5},
    "super-kagome": {"freq": 2, "tolerance_deg": 0.5},
    "super-lonsdaleite": {"freq": 3, "tolerance_deg": 0.5},
    # Wythoff cages no other catalogue holds (builders/wythoff.py).
    "wythoff-truncated-dodecahedron": {"freq": 2, "tolerance_deg": 0.5},
    "wythoff-rhombicosidodecahedron": {"freq": 2, "tolerance_deg": 0.5},
    "wythoff-truncated-icosidodecahedron": {"freq": 2, "tolerance_deg": 0.5},
    "wythoff-truncated-cuboctahedron": {"freq": 4, "tolerance_deg": 1.5},
}


def geodesic_atom_count(graph: str = "super-icosahedron", m: int = 2,
                        freq: int = 2, rows: int = 1, n: int = 0,
                        window_radius: int = 1) -> int:
    """Atoms the build will have: coarse triangles times ``m^2+mn+n^2``.

    ``window_radius = 0`` resolves to the largest that fits, as the
    builder does.
    """
    g = _named(graph)
    points, cell = _graph_frame(g)
    tolerance = GEODESIC_CAGES.get(graph, {}).get("tolerance_deg", 2.5)
    window = GEODESIC_CAGES.get(graph, {}).get("window", 6)
    frame, _, _, window = _choose_frames(g, points, cell, freq, tolerance,
                                         window=window)
    if window_radius == 0:
        window_radius = largest_window(points, g.edges, frame, freq,
                                       tolerance, cell, window)
    (_, tris), _, _, _ = geodesic_map(points, g.edges, frame, freq=freq,
                                      window=window,
                                      window_radius=window_radius,
                                      rows=rows, tolerance_deg=tolerance,
                                      cell=cell)
    return len(tris) * (m * m + m * n + n * n)


def describe_geodesic(atoms: Atoms) -> str:
    """One line: cage, size, census, the law it meets."""
    info = atoms.info
    census = ", ".join(f"{k}: {v}" for k, v in info["ring_counts"].items())
    return (f"{info['kind']} on frequency-{info['node_frequency']} geodesic "
            f"nodes, GC{info['gc_indices']}: {len(atoms)} atoms, rings "
            f"{{{census}}}, sum(6-n) = {info['ring_deficit']:+d} = 12(V - E) "
            f"for V = {info['n_nodes']}, E = {info['n_struts']}; worst arm "
            f"miss {info['worst_arm_miss_deg']} deg")


def _fold_orthohexagonal(basis: np.ndarray, c: float = 1.0) -> tuple:
    """A hexagonal lattice's sites in its orthohexagonal (a, a*sqrt3, c) cell.

    ``basis`` is in fractions of the hexagonal cell (a = 1, a2 at 60 deg
    to a1 -- not the 120 deg convention, whose special points differ). The
    orthorhombic cell holds two hexagonal ones, and it is what
    ``minimum_image`` can measure.
    """
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0) / 2.0, 0.0])
    a3 = np.array([0.0, 0.0, c])
    shape = np.array([1.0, np.sqrt(3.0), c])
    sites = []
    for i in range(-2, 3):
        for j in range(-2, 4):
            for f in basis:
                p = (f[0] + i) * a1 + (f[1] + j) * a2 + f[2] * a3
                q = p / shape
                if np.all(q >= -1e-9) and np.all(q < 1 - 1e-9):
                    sites.append(np.round(q, 12) % 1.0)
    return np.unique(np.round(sites, 10), axis=0), tuple(shape)


def _geodesic_nets() -> dict:
    """Periodic nets the geodesic carrier adds to the shared catalogue.

    Kept here rather than in ``supernetwork.SUPERLATTICES`` on purpose:
    that catalogue feeds the meshed and knee modes too, and these nets have
    only been measured on this route.
    """
    from .knee import _srs_sites
    from .supernetwork import _periodic_graph

    kagome, kshape = _fold_orthohexagonal(
        np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]))
    c = float(np.sqrt(8.0 / 3.0))
    # Wurtzite positions with a2 at 60 deg to a1, so the two special
    # points are (1/3, 1/3) and (2/3, 2/3); ideal c/a and u = 3/8 make all
    # four bonds equal.
    lon, lshape = _fold_orthohexagonal(np.array([
        [1 / 3, 1 / 3, 0.0], [2 / 3, 2 / 3, 0.5],
        [1 / 3, 1 / 3, 3 / 8], [2 / 3, 2 / 3, 7 / 8]]), c)
    srs = _srs_sites()
    return {
        # The gyroid's skeleton: three arms at 120 deg, in planes normal
        # to the four <111>. srs is chiral; the mirror is its own entry so
        # that net and wall handedness can be chosen independently.
        "super-srs": _periodic_graph(
            "super-srs", srs, (1.0, 1.0, 1.0), (True, True, True),
            "The Laves graph (gyroid skeleton), chiral."),
        "super-srs-mirror": _periodic_graph(
            "super-srs-mirror", (1.0 - srs) % 1.0, (1.0, 1.0, 1.0),
            (True, True, True), "The Laves graph, the other hand."),
        # Eight <111> arms per node: the replica of sodalite's beta cages.
        "super-bcc": _periodic_graph(
            "super-bcc", [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)], (1.0, 1.0, 1.0),
            (True, True, True), "bcu: eight <111> arms per node."),
        "super-kagome": _periodic_graph(
            "super-kagome", kagome, kshape, (True, True, False),
            "2D, four arms per node at 60 and 120 deg."),
        # Hexagonal diamond: the replica of EMT, FAU's hexagonal polytype.
        "super-lonsdaleite": _periodic_graph(
            "super-lonsdaleite", lon, lshape, (True, True, True),
            "lon: tetrahedral nodes, eclipsed along c."),
    }


#: Nets only this carrier builds (see :func:`_geodesic_nets`).
GEODESIC_NETS: dict = _geodesic_nets()


def _named(graph: str):
    from .supernetwork import named_graph
    from .wythoff import WYTHOFF_CAGES

    if graph in GEODESIC_NETS:
        return GEODESIC_NETS[graph]
    if graph in WYTHOFF_CAGES:
        return WYTHOFF_CAGES[graph]
    return named_graph(graph, 1.0)


def enantiomer(atoms: Atoms) -> Atoms:
    """The exact mirror image of a built structure, by reflecting atoms.

    Rebuilding the mirror-image graph is not the same thing: the frame fit
    and the arms' reference axes are not mirrored with the net, and the
    srs control measured the two builds 4 % apart in cell. Reflecting the
    finished atoms through the plane ``x = L/2`` (``x = 0`` for a finite
    structure) gives the enantiomer exactly -- the same bonds, rings and
    strain, every length and angle identical -- and swaps the labels that
    carry handedness: ``gc_indices`` ``(m, n) -> (n, m)`` and an srs net
    for its mirror.

    Returns a new ``Atoms``; ``info`` is copied, not shared.
    """
    import copy

    mirrored = atoms.copy()
    mirrored.info = copy.deepcopy(atoms.info)
    positions = mirrored.get_positions()
    if mirrored.pbc[0] and mirrored.cell.lengths()[0] > 0:
        positions[:, 0] = mirrored.cell.lengths()[0] - positions[:, 0]
    else:
        positions[:, 0] = -positions[:, 0]
    mirrored.set_positions(positions)
    if mirrored.pbc[0]:
        mirrored.wrap()
    info = mirrored.info
    if "gc_indices" in info:
        m, n = info["gc_indices"]
        info["gc_indices"] = (n, m)
    kind = str(info.get("kind", ""))
    if kind.endswith("-mirror"):
        info["kind"] = kind[: -len("-mirror")]
    elif kind == "super-srs":
        info["kind"] = "super-srs-mirror"
    info["enantiomer_of"] = kind
    return mirrored


def strain_energy(atoms: Atoms, k_bond: float = 40.0, k_angle: float = 15.0,
                  bond: float = CC_BOND) -> float:
    """Valence-force-field strain per atom, from the recorded bond graph.

    ``1/2 k_bond (r - bond)^2`` per bond plus ``1/2 k_angle (theta -
    120 deg)^2`` (radians) per angle, under the minimum image -- the
    relaxer's own bond and angle terms, without its repulsion. A
    comparison number for structures of one family, not an energy of
    formation, and mirror-invariant by construction.
    """
    from .fullerene_mesh import minimum_image

    pos = atoms.get_positions()
    box = None
    if any(atoms.pbc):
        box = np.where(atoms.pbc, atoms.cell.lengths(), 0.0)
    pairs = np.asarray(sorted(atoms.info["bonds"]), dtype=int)
    vec = minimum_image(pos[pairs[:, 1]] - pos[pairs[:, 0]], box)
    r = np.linalg.norm(vec, axis=1)
    energy = 0.5 * k_bond * float(((r - bond) ** 2).sum())
    neighbours: dict[int, list[np.ndarray]] = {}
    for (i, j), v in zip(pairs, vec, strict=True):
        neighbours.setdefault(int(i), []).append(v)
        neighbours.setdefault(int(j), []).append(-v)
    target = np.radians(120.0)
    for arms in neighbours.values():
        for a in range(len(arms)):
            for b in range(a + 1, len(arms)):
                c = np.dot(arms[a], arms[b]) / (np.linalg.norm(arms[a])
                                                 * np.linalg.norm(arms[b]))
                theta = np.arccos(np.clip(c, -1.0, 1.0))
                energy += 0.5 * k_angle * (theta - target) ** 2
    return energy / len(atoms)
