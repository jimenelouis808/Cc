"""Low-level triangulated-mesh engine for closed sp2 carbon shells.

A capped carbon nanotube is, topologically, an **elongated fullerene**: a
closed, genus-0 (spherical) 3-coordinate carbon network. Euler's polyhedron
formula (``V - E + F = 2``) forces such a shell to contain *exactly* 12
more pentagons than heptagons (counting every non-hexagonal ring's
``6 - size`` deficit/surplus, the sum is always ``+12``). Pentagons carry
positive Gaussian curvature (they pucker the sheet **convex**, e.g. dome
caps); heptagons carry negative curvature (**concave** saddle points, e.g.
the inside of a bend); octagons are a second-order negative-curvature
feature typically paired with two pentagons (the classic 5-8-5
reconstructed divacancy).

Rather than editing a hexagonal (honeycomb) lattice directly -- which
makes it very easy to silently build an unphysical ring pattern -- this
module works one level down, on the honeycomb's **triangulated dual**
(each honeycomb ring = one mesh vertex; each honeycomb bond = one mesh
edge shared by two triangles). On that mesh:

* a mesh vertex of degree 5 / 6 / 7 / 8 maps to a pentagon / hexagon /
  heptagon / octagon in the honeycomb,
* Euler's formula for the *triangulated* mesh (``V - E + F = 2``, all
  faces triangles) guarantees the honeycomb dual also satisfies its own
  Euler relation automatically -- there is no way to build an invalid
  ring count by construction,
* local topology edits become simple, provably-correct combinatorial
  operations:

  - :func:`edge_flip` (90 deg bond rotation) turns two degree-6 vertices
    into degree-5 and their two opposite neighbours into degree-7 --
    the textbook Stone-Wales 5-7-7-5 defect.
  - :func:`contract_edge` (merge two mesh vertices into one) turns a
    degree-6/degree-6 pair into one degree-8 vertex and drops their two
    shared neighbours to degree-5 each -- the textbook 5-8-5
    reconstructed divacancy (one octagon flanked by two pentagons).
  - the seed polyhedron (:func:`seed_capsule_mesh`) plants exactly six
    degree-5 vertices at each pole, which is exactly the disclination
    budget (6 x 60 deg = 360 deg) needed to close a hemispherical dome
    -- the two end caps.

Every operation here works on plain NumPy arrays (mesh vertices, integer
triangle index array) and is unit-tested against the Euler invariant, so
the higher-level builder in :mod:`nanocarbon_lab.builders.capped_cnt`
never has to re-derive ring statistics: it can trust the mesh.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional, Sequence

import numpy as np

Mesh = tuple[np.ndarray, np.ndarray]  # (vertices [V,3], triangles [F,3] int)


# --------------------------------------------------------------------------
# Seed polyhedron: two pentagonal poles + a stack of pentagonal-antiprism
# rings. Subdividing and taking the dual of this seed always yields exactly
# 12 pentagons (6 per pole) and otherwise hexagons -- i.e. a valid capped
# nanotube / elongated fullerene topology, for any seed size.
# --------------------------------------------------------------------------
def seed_capsule_mesh(n_rings: int, pole_gap: float = 0.85) -> Mesh:
    """Build the triangulated seed polyhedron for a capped-tube fullerene.

    Two 5-valent poles are joined by ``n_rings`` stacked pentagonal
    antiprism rings (consecutive rings alternate a 36 deg azimuthal
    offset, the standard antiprism zig-zag triangulation). ``n_rings=2``
    reproduces a regular icosahedron; larger values elongate the body.

    Parameters
    ----------
    n_rings
        Number of 5-atom rings between the poles (>= 2). Controls the
        eventual tube length (each ring is one lattice period).
    pole_gap
        Axial distance from each pole to its nearest ring, in units of the
        ring radius (1.0). The default reproduces icosahedral proportions;
        the exact value only affects the initial geometry, which is later
        cleaned up by :func:`relax_shell`.

    Returns
    -------
    (vertices, triangles)
        ``vertices`` has shape ``(2 + 5*n_rings, 3)``; ``triangles`` is an
        ``(F, 3)`` integer index array. Every vertex has degree 5 (the two
        poles and the two rings adjacent to them) or degree 6 (interior
        rings), which is exactly the disclination budget for two
        hemispherical, 6-pentagon caps.

    Raises
    ------
    ValueError
        If ``n_rings < 2``.
    """
    if n_rings < 2:
        raise ValueError("n_rings must be >= 2 (need at least one ring per pole).")

    idx: dict[tuple, int] = {}
    verts: list[np.ndarray] = []

    def add(key: tuple, p: np.ndarray) -> None:
        idx[key] = len(verts)
        verts.append(p)

    total_h = n_rings - 1
    add(("pole", 0), np.array([0.0, 0.0, total_h / 2 + pole_gap]))
    for r in range(n_rings):
        z = total_h / 2 - r
        offset = (r % 2) * (np.pi / 5)
        for k in range(5):
            ang = 2 * np.pi * k / 5 + offset
            add(("ring", r, k), np.array([np.cos(ang), np.sin(ang), z]))
    add(("pole", 1), np.array([0.0, 0.0, -(total_h / 2 + pole_gap)]))

    faces: list[tuple[int, int, int]] = []
    for k in range(5):
        a, b = idx[("ring", 0, k)], idx[("ring", 0, (k + 1) % 5)]
        faces.append((idx[("pole", 0)], a, b))
    for r in range(n_rings - 1):
        for k in range(5):
            a0, a1 = idx[("ring", r, k)], idx[("ring", r, (k + 1) % 5)]
            b0, b1 = idx[("ring", r + 1, k)], idx[("ring", r + 1, (k + 1) % 5)]
            if r % 2 == 0:
                faces.append((a0, a1, b1))
                faces.append((a0, b1, b0))
            else:
                faces.append((a0, a1, b0))
                faces.append((a1, b1, b0))
    last = n_rings - 1
    for k in range(5):
        a, b = idx[("ring", last, k)], idx[("ring", last, (k + 1) % 5)]
        faces.append((idx[("pole", 1)], b, a))

    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def subdivide_mesh(mesh: Mesh, freq: int) -> Mesh:
    """Refine every triangle of ``mesh`` into ``freq**2`` sub-triangles.

    Standard barycentric (Class I geodesic) subdivision: new vertices sit
    on a straight-line barycentric grid inside each original triangle, and
    shared edges are deduplicated so the result stays a valid manifold
    mesh. This is what controls the final atom count / tube diameter and
    hexagon density between defects -- it does **not** change which
    vertices are 5- or 6-valent (subdividing a degree-5 pole vertex still
    leaves a single degree-5 vertex at its position), so it never changes
    the pentagon/hexagon budget.

    Parameters
    ----------
    mesh
        ``(vertices, triangles)`` seed mesh.
    freq
        Subdivision frequency (>= 1). ``freq=1`` returns the mesh
        unchanged (up to vertex dedup).

    Returns
    -------
    (vertices, triangles)
        Refined mesh. Vertex coordinates are the linear (non-projected)
        barycentric interpolation of the input -- callers typically follow
        this with :func:`capsule_project` to place atoms on the intended
        physical surface.
    """
    if freq < 1:
        raise ValueError("freq must be >= 1.")
    verts, faces = mesh
    key_to_idx: dict[tuple, int] = {}
    new_verts: list[np.ndarray] = []

    def get_idx(p: np.ndarray) -> int:
        key = tuple(np.round(p, 7))
        found = key_to_idx.get(key)
        if found is not None:
            return found
        new_idx = len(new_verts)
        key_to_idx[key] = new_idx
        new_verts.append(p)
        return new_idx

    new_faces: list[tuple[int, int, int]] = []
    for i0, i1, i2 in faces:
        a, b, c = verts[i0], verts[i1], verts[i2]
        grid: dict[tuple[int, int], int] = {}
        for i in range(freq + 1):
            for j in range(freq - i + 1):
                p = (i * a + j * b + (freq - i - j) * c) / freq
                grid[(i, j)] = get_idx(p)
        for i in range(freq):
            for j in range(freq - i):
                k = freq - i - j
                v00, v10, v01 = grid[(i, j)], grid[(i + 1, j)], grid[(i, j + 1)]
                new_faces.append((v00, v10, v01))
                if k > 1:
                    new_faces.append((v10, grid[(i + 1, j + 1)], v01))

    return np.array(new_verts, dtype=float), np.array(new_faces, dtype=int)


def mesh_adjacency(faces: np.ndarray) -> dict[int, set[int]]:
    """Return ``{vertex: {neighbour vertices}}`` for a triangulated mesh."""
    nbrs: dict[int, set[int]] = defaultdict(set)
    for f in faces:
        a, b, c = (int(x) for x in f)
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return nbrs


def edge_flip(mesh: Mesh, u: int, v: int) -> Mesh:
    """Rotate the mesh edge ``(u, v)`` by 90 deg -- a Stone-Wales flip.

    ``u`` and ``v`` must share exactly two triangles ``(u, v, w1)`` and
    ``(v, u, w2)``. The edit removes edge ``u-v`` and adds edge
    ``w1-w2`` instead, dropping ``deg(u)`` and ``deg(v)`` by one each and
    raising ``deg(w1)`` and ``deg(w2)`` by one each. For an interior
    hexagonal patch (``u, v, w1, w2`` all degree 6) this produces the
    honeycomb's classic 5-7-7-5 Stone-Wales defect: two new pentagons
    (at ``u, v``) and two new heptagons (at ``w1, w2``) in the dual.

    Parameters
    ----------
    mesh
        ``(vertices, triangles)``.
    u, v
        Endpoints of the mesh edge to flip.

    Returns
    -------
    (vertices, triangles)
        Edited mesh (vertex array unchanged; triangle array updated).

    Raises
    ------
    ValueError
        If ``u, v`` do not form a valid interior edge (exactly 2 incident
        triangles).
    """
    verts, faces = mesh
    face_list = [tuple(int(x) for x in f) for f in faces]
    incident = [f for f in face_list if u in f and v in f]
    if len(incident) != 2:
        raise ValueError(
            f"edge ({u}, {v}) has {len(incident)} incident triangles; "
            "expected exactly 2 (interior edge)."
        )
    opposite = set()
    for f in incident:
        opposite |= set(f) - {u, v}
    if len(opposite) != 2:
        raise ValueError(f"edge ({u}, {v}) is degenerate (non-manifold).")
    w1, w2 = sorted(opposite)

    new_faces = [f for f in face_list if not (u in f and v in f)]
    new_faces.append((u, w1, w2))
    new_faces.append((v, w2, w1))
    return verts, np.array(new_faces, dtype=int)


def contract_edge(mesh: Mesh, u: int, v: int) -> tuple[Mesh, dict[int, int]]:
    """Merge mesh vertex ``v`` into ``u`` -- a divacancy-style contraction.

    ``u`` and ``v`` must share exactly two triangles (equivalently: they
    have exactly two common neighbours ``w1, w2``). The edit deletes the
    two triangles containing edge ``u-v``, deletes vertex ``v``, and
    rewires every other triangle that referenced ``v`` to reference ``u``
    instead. The merged vertex ends up with degree
    ``deg(u) + deg(v) - 4``; for two interior degree-6 vertices this is
    degree 8, and ``w1, w2`` each drop from degree 6 to degree 5 -- the
    honeycomb's classic 5-8-5 reconstructed divacancy (one new octagon
    flanked by two new pentagons), with every atom left 3-coordinate.

    Parameters
    ----------
    mesh
        ``(vertices, triangles)``.
    u, v
        Mesh edge to contract. ``v`` is removed; ``u`` survives at the
        midpoint of the original ``u, v`` positions.

    Returns
    -------
    (mesh, remap)
        The edited mesh, and a ``{old_vertex_index: new_vertex_index}``
        map (``v`` is absent from ``remap``; all indices above ``v``
        shift down by one).

    Raises
    ------
    ValueError
        If ``u, v`` do not form a valid interior edge.
    """
    verts, faces = mesh
    face_list = [tuple(int(x) for x in f) for f in faces]
    incident = [f for f in face_list if u in f and v in f]
    if len(incident) != 2:
        raise ValueError(
            f"edge ({u}, {v}) has {len(incident)} incident triangles; "
            "expected exactly 2 (interior edge)."
        )

    merged_faces = []
    for f in face_list:
        if u in f and v in f:
            continue
        merged_faces.append(tuple(u if x == v else x for x in f))

    new_verts = verts.copy()
    new_verts[u] = 0.5 * (verts[u] + verts[v])

    keep_mask = np.ones(len(verts), dtype=bool)
    keep_mask[v] = False
    remap_arr = -np.ones(len(verts), dtype=int)
    remap_arr[keep_mask] = np.arange(int(keep_mask.sum()))
    final_verts = new_verts[keep_mask]
    final_faces = np.array(
        [[int(remap_arr[x]) for x in f] for f in merged_faces], dtype=int
    )
    remap = {i: int(remap_arr[i]) for i in range(len(verts)) if keep_mask[i]}
    return (final_verts, final_faces), remap


def dual_honeycomb(
    mesh: Mesh,
) -> tuple[np.ndarray, set[tuple[int, int]], list[list[int]]]:
    """Take the planar/spherical dual of a triangulated mesh: the honeycomb.

    Every mesh face becomes one honeycomb atom (positioned at the
    triangle's centroid); every mesh vertex becomes one honeycomb ring,
    whose member atoms are the (cyclically ordered) centroids of that
    vertex's incident faces. A degree-5/6/7/8 mesh vertex therefore
    produces a pentagon/hexagon/heptagon/octagon honeycomb ring, and every
    honeycomb atom ends up exactly 3-coordinate (it borders exactly 3
    mesh-face edges).

    Parameters
    ----------
    mesh
        ``(vertices, triangles)`` -- must be a closed, manifold,
        triangulated mesh (e.g. from :func:`seed_capsule_mesh` +
        :func:`subdivide_mesh`, optionally edited with :func:`edge_flip` /
        :func:`contract_edge`).

    Returns
    -------
    (positions, bonds, rings)
        ``positions`` -- ``(n_atoms, 3)`` array (one row per mesh face).
        ``bonds`` -- set of ``(i, j)`` atom-index pairs with ``i < j``.
        ``rings`` -- list of atom-index lists, one per mesh vertex; the
        ring size equals the mesh vertex's degree.
    """
    verts, faces = mesh
    vert_faces: dict[int, list[int]] = defaultdict(list)
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[int(v)].append(fi)

    face_centers = verts[faces].mean(axis=1)

    def order_ring(incident: list[int]) -> list[int]:
        face_verts = {fi: set(int(x) for x in faces[fi]) for fi in incident}
        adj: dict[int, list[int]] = defaultdict(list)
        for a in incident:
            for b in incident:
                if a != b and len(face_verts[a] & face_verts[b]) == 2:
                    adj[a].append(b)
        start = incident[0]
        ring = [start]
        prev, cur = None, start
        while True:
            nxt = next(x for x in adj[cur] if x != prev)
            if nxt == start:
                break
            ring.append(nxt)
            prev, cur = cur, nxt
            if len(ring) > len(incident):
                raise RuntimeError("Non-manifold mesh: ring walk failed to close.")
        if len(ring) != len(incident):
            raise RuntimeError("Non-manifold mesh: incomplete ring.")
        return ring

    rings = [order_ring(ifaces) for ifaces in vert_faces.values()]
    bonds: set[tuple[int, int]] = set()
    for ring in rings:
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            bonds.add((a, b) if a < b else (b, a))
    return face_centers, bonds, rings


def ring_size_histogram(rings: Sequence[Sequence[int]]) -> dict[int, int]:
    """Return ``{ring_size: count}`` -- a quick Euler-invariant sanity check.

    For any closed genus-0 shell, ``sum((6 - size) * count) == 12`` must
    hold (Descartes' angular-deficit theorem); callers can assert this to
    catch a broken mesh edit early.
    """
    return dict(Counter(len(r) for r in rings))


def capsule_project(
    positions: np.ndarray,
    radius: float,
    half_length: float,
    bend_angle: float = 0.0,
) -> np.ndarray:
    """Reshape raw dual-mesh positions onto a straight or bent capsule.

    Each point is classified by its axial coordinate ``z``: points with
    ``|z| <= half_length`` are pushed radially onto a cylinder of the
    given ``radius`` (the tube body); points beyond that are pushed onto
    a sphere of the same radius centred on the nearest cap pole (the
    hemispherical end caps). This turns whatever raw shape the seed +
    subdivision produced into a clean capsule, independent of the exact
    seed proportions -- :func:`relax_shell` then cleans up bond lengths.

    Parameters
    ----------
    positions
        ``(n, 3)`` raw atom positions (axis of revolution = z).
    radius
        Target capsule radius (Å).
    half_length
        Half the straight-body length (Å); ``|z| <= half_length`` is
        treated as body, the rest as caps.
    bend_angle
        If non-zero, additionally sweep the straight body axis into a
        circular arc of this total angle (radians), Frenet-frame style
        (as in :mod:`nanocarbon_lab.builders.nanocoil`), so the returned
        capsule is a gentle elastic bend rather than a straight rod. Caps
        are rigidly carried along at the arc's two ends. ``0.0`` (default)
        keeps the tube straight.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` reshaped positions.
    """
    pos = positions.copy()
    z = pos[:, 2]
    rho = np.linalg.norm(pos[:, :2], axis=1)
    rho = np.where(rho < 1e-9, 1e-9, rho)
    out = np.empty_like(pos)

    body = np.abs(z) <= half_length
    scale = radius / rho
    out[body, 0] = pos[body, 0] * scale[body]
    out[body, 1] = pos[body, 1] * scale[body]
    out[body, 2] = z[body]
    for sign, mask in ((1.0, ~body & (z > 0)), (-1.0, ~body & (z < 0))):
        center = np.array([0.0, 0.0, sign * half_length])
        rel = pos[mask] - center
        n = np.linalg.norm(rel, axis=1, keepdims=True)
        n = np.where(n < 1e-9, 1e-9, n)
        out[mask] = center + rel / n * radius

    if bend_angle == 0.0 or half_length <= 0:
        return out

    # Sweep the whole capsule (body + caps) along a circular arc in the
    # x-z plane. Arc length equals the straight-body length (2*half_length)
    # so bond lengths in the body are preserved on average.
    arc_radius = (2.0 * half_length) / bend_angle
    theta = out[:, 2] / arc_radius  # signed arc angle for this point
    x_local = out[:, 0]
    y_local = out[:, 1]
    ct, st = np.cos(theta), np.sin(theta)
    bent = np.empty_like(out)
    bent[:, 0] = (arc_radius + x_local) * st
    bent[:, 1] = y_local
    bent[:, 2] = (arc_radius + x_local) * ct - arc_radius
    return bent


def relax_shell(
    positions: np.ndarray,
    bonds: set[tuple[int, int]],
    equilibrium: float = 1.42,
    k_bond: float = 20.0,
    k_angle: float = 6.0,
    k_repel: float = 12.0,
    repel_cutoff_factor: float = 1.55,
    repel_refresh: int = 10,
    steps: int = 2500,
    step_size: float = 0.01,
    max_disp: float = 0.03,
) -> np.ndarray:
    """Relax a closed shell to uniform sp2 bond lengths and 120 deg angles.

    Three harmonic terms, integrated with damped steepest descent, using
    **only the explicit topology passed in** (never re-derived from
    distances): a bond spring on every entry of ``bonds`` targeting
    ``equilibrium``, a 1-3 ("angle-proxy") spring on every pair of atoms
    that share a bonded neighbour targeting ``equilibrium * sqrt(3)`` (the
    ideal 1-3 distance at a 120 deg sp2 angle), and a short-range
    non-bonded repulsion. The angle term keeps the shell from folding
    under the bond term alone (pure bond springs have no resistance to
    bending); the repulsion term additionally guards floppier rings
    (heptagons, octagons) against pushing two unrelated atoms through
    each other during relaxation -- a real risk pure springs cannot rule
    out, since they only constrain *bonded* and 1-3 distances.

    Because the bond graph is fixed for the whole relaxation, this can
    never change ring topology; it only cleans up geometry. Uses the same
    spring math as
    :func:`nanocarbon_lab.relax.optimize.harmonic_pre_relax`, but takes an
    explicit edge list instead of re-guessing bonds from distances
    (guessing would be unsafe here: many non-bonded atoms in a curved
    shell sit closer together than the bond length).

    Parameters
    ----------
    positions
        ``(n, 3)`` initial atom positions (Å).
    bonds
        Set/iterable of ``(i, j)`` bonded atom-index pairs.
    equilibrium
        Target C-C bond length (Å).
    k_bond, k_angle, k_repel
        Spring / repulsion constants (arbitrary units; only their ratio to
        ``step_size`` matters).
    repel_cutoff_factor
        Non-bonded pairs closer than ``repel_cutoff_factor * equilibrium``
        repel each other. Kept below the ideal 1-3 distance
        (``sqrt(3) ~= 1.73`` x ``equilibrium``) so it never fights the
        angle term.
    repel_refresh
        Rebuild the non-bonded neighbour list every this many steps (a
        k-d tree query), rather than every step, for speed.
    steps
        Maximum steepest-descent iterations.
    step_size, max_disp
        Integration step scale and per-atom displacement clamp (Å) for
        numerical stability.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` relaxed positions.
    """
    from scipy.spatial import cKDTree

    bonds_arr = np.array(sorted(bonds), dtype=int)
    nbrs: dict[int, set[int]] = defaultdict(set)
    for a, b in bonds_arr:
        nbrs[int(a)].add(int(b))
        nbrs[int(b)].add(int(a))
    pairs13: set[tuple[int, int]] = set()
    for atom, ns in nbrs.items():
        ns = list(ns)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                pair = (ns[i], ns[j]) if ns[i] < ns[j] else (ns[j], ns[i])
                pairs13.add(pair)
    pairs13_arr = np.array(sorted(pairs13), dtype=int)
    eq13 = equilibrium * np.sqrt(3.0)
    excluded = set(bonds) | pairs13
    repel_cutoff = repel_cutoff_factor * equilibrium

    pos = positions.copy()
    repel_arr = np.empty((0, 2), dtype=int)
    for step in range(steps):
        if step % repel_refresh == 0:
            tree = cKDTree(pos)
            close = tree.query_pairs(r=repel_cutoff, output_type="ndarray")
            if len(close):
                mask = np.array(
                    [
                        (int(a), int(b)) not in excluded
                        for a, b in close
                    ]
                )
                repel_arr = close[mask] if mask.any() else np.empty((0, 2), dtype=int)
            else:
                repel_arr = np.empty((0, 2), dtype=int)

        forces = np.zeros_like(pos)
        for pair_arr, k, eq in (
            (bonds_arr, k_bond, equilibrium),
            (pairs13_arr, k_angle, eq13),
        ):
            if len(pair_arr) == 0:
                continue
            ri, rj = pos[pair_arr[:, 0]], pos[pair_arr[:, 1]]
            rij = rj - ri
            dist = np.linalg.norm(rij, axis=1)
            dist = np.where(dist < 1e-6, 1e-6, dist)
            unit = rij / dist[:, None]
            fmag = k * (dist - eq)
            np.add.at(forces, pair_arr[:, 0], fmag[:, None] * unit)
            np.add.at(forces, pair_arr[:, 1], -fmag[:, None] * unit)

        if len(repel_arr):
            ri, rj = pos[repel_arr[:, 0]], pos[repel_arr[:, 1]]
            rij = rj - ri
            dist = np.linalg.norm(rij, axis=1)
            dist = np.where(dist < 1e-6, 1e-6, dist)
            unit = rij / dist[:, None]
            push = np.clip(repel_cutoff - dist, 0.0, None)
            fmag = -k_repel * push  # always pushes i,j apart
            np.add.at(forces, repel_arr[:, 0], fmag[:, None] * unit)
            np.add.at(forces, repel_arr[:, 1], -fmag[:, None] * unit)

        disp = step_size * forces
        norms = np.linalg.norm(disp, axis=1)
        over = norms > max_disp
        if np.any(over):
            disp[over] *= (max_disp / norms[over])[:, None]
        pos += disp
    return pos


def pick_interior_edge(
    faces: np.ndarray,
    rng: np.random.Generator,
    exclude: Optional[set[int]] = None,
    required_degree: dict[int, int] | None = None,
    n_tries: int = 200,
) -> Optional[tuple[int, int]]:
    """Sample a random mesh edge suitable for :func:`edge_flip` / :func:`contract_edge`.

    Restricts to edges whose two endpoints are both degree 6 (i.e. an
    undisturbed hexagonal patch), so repeated defect placement always
    starts from clean hexagons rather than compounding onto an existing
    defect. Optionally excludes a set of vertex indices (e.g. atoms
    already used by a previous defect, to keep defects well separated).

    Parameters
    ----------
    faces
        Triangle index array of the current mesh.
    rng
        NumPy random generator (caller controls the seed).
    exclude
        Vertex indices to avoid (both endpoints and their opposite
        vertices must be clear of this set).
    required_degree
        Optional precomputed ``{vertex: degree}`` map (recomputed from
        ``faces`` if omitted -- pass it in when calling repeatedly for
        speed).
    n_tries
        Number of random candidate edges to test before giving up.

    Returns
    -------
    (u, v) or None
        A valid edge, or ``None`` if no suitable edge was found within
        ``n_tries`` attempts.
    """
    exclude = exclude or set()
    nbrs = mesh_adjacency(faces)
    deg = required_degree or {v: len(ns) for v, ns in nbrs.items()}
    vertices = np.array(list(nbrs.keys()))
    for _ in range(n_tries):
        u = int(rng.choice(vertices))
        if u in exclude or deg.get(u) != 6:
            continue
        candidates = [v for v in nbrs[u] if v not in exclude and deg.get(v) == 6]
        if not candidates:
            continue
        v = int(rng.choice(candidates))
        incident = [
            f for f in faces if u in f and v in f
        ]
        if len(incident) != 2:
            continue
        opposite = set()
        for f in incident:
            opposite |= set(int(x) for x in f) - {u, v}
        if len(opposite) != 2 or opposite & exclude:
            continue
        if any(deg.get(w) != 6 for w in opposite):
            continue
        return u, v
    return None
