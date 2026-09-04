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
from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np

Mesh = tuple[np.ndarray, np.ndarray]  # (vertices [V,3], triangles [F,3] int)


def minimum_image(delta: np.ndarray, box: float | None) -> np.ndarray:
    """Wrap displacement vectors into the shortest periodic image.

    With ``box=None`` this is the identity, so every caller can be written
    once and used for both finite and periodic structures. For a periodic
    cell it maps each component into ``[-box/2, box/2)``, which is what
    makes a bond across the cell seam measure ~1.42 Å instead of ~box.
    """
    if box is None:
        return delta
    return delta - box * np.round(delta / box)


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
    box: float | None = None,
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
    box
        Cubic cell length for a periodic mesh, else ``None``. Face
        centroids are then computed under the minimum-image convention.
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

    if box is None:
        face_centers = verts[faces].mean(axis=1)
    else:
        # Average the two other corners as offsets from the first, or a
        # triangle straddling the cell seam would place its atom in the
        # middle of the box rather than on the surface.
        anchor = verts[faces[:, 0]]
        off_b = minimum_image(verts[faces[:, 1]] - anchor, box)
        off_c = minimum_image(verts[faces[:, 2]] - anchor, box)
        face_centers = np.mod(anchor + (off_b + off_c) / 3.0, box)

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


def smooth_mesh_on_capsule(
    mesh: Mesh,
    radius: float,
    half_length: float,
    iterations: int = 60,
) -> Mesh:
    """Equalise triangle sizes by Laplacian smoothing *on the capsule surface*.

    Barycentric subdivision (:func:`subdivide_mesh`) produces triangles of
    quite unequal size -- those near an original seed vertex are much
    smaller than those near a face centre. Taking the dual of such a mesh
    gives a honeycomb whose hexagons are correspondingly unequal, which is
    a poor starting point for relaxation: on larger structures the
    optimiser cannot recover from it and the shell folds through itself.

    This alternates two cheap steps: move every mesh vertex to the
    centroid of its neighbours (Laplacian smoothing, which equalises
    spacing but shrinks the shell), then push every vertex back onto the
    capsule surface (which restores the shape). The fixed point is a
    near-uniform triangulation *of the capsule*, whose dual is a
    near-uniform honeycomb -- exactly the starting point
    :func:`relax_shell` needs.

    Topology is never touched, so ring counts are unaffected.

    Parameters
    ----------
    mesh
        ``(vertices, triangles)`` to smooth.
    radius, half_length
        Capsule geometry to project onto (see :func:`capsule_project`).
    iterations
        Smoothing sweeps. 60 is comfortably converged for the sizes this
        module builds; the cost is linear and small next to relaxation.

    Returns
    -------
    (vertices, triangles)
        Same triangles, smoothed + projected vertices.
    """
    verts, faces = mesh
    nbrs = mesh_adjacency(faces)
    neighbour_idx = [
        np.array(sorted(nbrs[i]), dtype=int) if i in nbrs else np.array([], dtype=int)
        for i in range(len(verts))
    ]
    pos = capsule_project(verts.copy(), radius, half_length)
    for _ in range(iterations):
        smoothed = pos.copy()
        for i, ns in enumerate(neighbour_idx):
            if len(ns):
                smoothed[i] = pos[ns].mean(axis=0)
        pos = capsule_project(smoothed, radius, half_length)
    return pos, faces


def radius_for_freq(freq: int, bond: float = 1.42) -> float:
    """Radius (Å) of the tube a given subdivision ``freq`` produces.

    The seed polyhedron is 5-fold symmetric, so the body circumference
    spans ``5 * freq`` hexagons, each ``sqrt(3) * bond`` wide across
    flats. Hence ``2*pi*R = 5 * freq * sqrt(3) * bond``. Radius is
    therefore **not** a free parameter: it is quantised by the lattice,
    exactly as for a real ``(n, m)`` nanotube, whose diameter is fixed by
    its chiral indices.
    """
    return 5.0 * freq * np.sqrt(3.0) * bond / (2.0 * np.pi)


def freq_for_radius(radius: float, bond: float = 1.42) -> int:
    """Smallest-error subdivision ``freq`` for a target radius (Å).

    Inverse of :func:`radius_for_freq`, rounded to the nearest realisable
    lattice. Use :func:`radius_for_freq` on the result to get the radius
    you will actually obtain.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")
    freq = int(round(2.0 * np.pi * radius / (5.0 * np.sqrt(3.0) * bond)))
    return max(1, freq)


def _valence_terms(
    bonds: Sequence[tuple[int, int]], n_atoms: int, exclude_13: bool = True
) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    """Build bond array, angle triplet array and the 1-2/1-3 exclusion set.

    ``exclude_13`` keeps 1-3 pairs out of the non-bonded repulsion, which
    is right whenever the angle term is doing that job -- for sp2 carbon
    it always is. Set it ``False`` when there is no usable angle target,
    and the repulsion holds the ligands apart instead (a VSEPR argument
    rather than a valence one). A six-coordinate metal is the case that
    needs it: its ligand-metal-ligand angles take several values at once,
    so no single ``angle_deg`` fits, and excluding 1-3 pairs while
    ``k_angle`` is zero leaves them with nothing keeping them apart at
    all.
    """
    bond_arr = np.array(sorted(bonds), dtype=int)
    nbrs: dict[int, list[int]] = defaultdict(list)
    for a, b in bond_arr:
        nbrs[int(a)].append(int(b))
        nbrs[int(b)].append(int(a))
    angles: list[tuple[int, int, int]] = []
    for centre, ns in nbrs.items():
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                angles.append((ns[i], centre, ns[j]))
    angle_arr = np.array(sorted(angles), dtype=int)

    excluded: set[tuple[int, int]] = set()
    for a, b in bond_arr:
        excluded.add((int(a), int(b)))
        excluded.add((int(b), int(a)))
    if exclude_13:
        for a, _, c in angle_arr:
            excluded.add((int(a), int(c)))
            excluded.add((int(c), int(a)))
    return bond_arr, angle_arr, excluded


def _vff_energy_gradient(
    x: np.ndarray,
    bond_arr: np.ndarray,
    angle_arr: np.ndarray,
    repel_arr: np.ndarray,
    n_atoms: int,
    r0: float,
    theta0: float,
    k_bond: float,
    k_angle: float,
    k_repel: float,
    repel_cutoff: float,
    anchors: np.ndarray | None,
    anchor_targets: np.ndarray | None,
    k_anchor: float,
    box: float | None = None,
) -> tuple[float, np.ndarray]:
    """Energy and analytic gradient of the sp2 valence force field.

    ``E = 1/2 k_bond (r - r0)^2 + 1/2 k_angle (theta - theta0)^2
         + 1/2 k_repel (d - d_cut)^2 [d < d_cut, non-bonded only]
         + 1/2 k_anchor |x - x_target|^2 [anchored atoms only]``

    Gradients are exact (verified against central finite differences to
    ~1e-9 relative error), which is what lets L-BFGS converge to genuine
    sp2 geometry instead of stalling like a fixed-step integrator.
    """
    pos = x.reshape(n_atoms, 3)
    energy = 0.0
    grad = np.zeros_like(pos)

    # --- bond stretching
    i, j = bond_arr[:, 0], bond_arr[:, 1]
    dvec = minimum_image(pos[j] - pos[i], box)
    r = np.linalg.norm(dvec, axis=1)
    r = np.where(r < 1e-9, 1e-9, r)
    dr = r - r0
    energy += 0.5 * k_bond * float(np.sum(dr**2))
    gv = (k_bond * dr / r)[:, None] * dvec
    np.add.at(grad, i, -gv)
    np.add.at(grad, j, gv)

    # --- angle bending (true angle, not a 1-3 distance proxy)
    ia, ja, ka = angle_arr[:, 0], angle_arr[:, 1], angle_arr[:, 2]
    u = minimum_image(pos[ia] - pos[ja], box)
    v = minimum_image(pos[ka] - pos[ja], box)
    lu = np.linalg.norm(u, axis=1)
    lv = np.linalg.norm(v, axis=1)
    lu = np.where(lu < 1e-9, 1e-9, lu)
    lv = np.where(lv < 1e-9, 1e-9, lv)
    uh, vh = u / lu[:, None], v / lv[:, None]
    cos_t = np.clip(np.sum(uh * vh, axis=1), -1.0 + 1e-9, 1.0 - 1e-9)
    theta = np.arccos(cos_t)
    sin_t = np.sqrt(1.0 - cos_t**2)
    dtheta = theta - theta0
    energy += 0.5 * k_angle * float(np.sum(dtheta**2))
    pref = k_angle * dtheta / sin_t
    gu = -(pref / lu)[:, None] * (vh - cos_t[:, None] * uh)
    gv2 = -(pref / lv)[:, None] * (uh - cos_t[:, None] * vh)
    np.add.at(grad, ia, gu)
    np.add.at(grad, ka, gv2)
    np.add.at(grad, ja, -(gu + gv2))

    # --- short-range non-bonded repulsion (1-2 and 1-3 pairs excluded)
    if len(repel_arr):
        p, q = repel_arr[:, 0], repel_arr[:, 1]
        dd = minimum_image(pos[q] - pos[p], box)
        rr = np.linalg.norm(dd, axis=1)
        rr = np.where(rr < 1e-9, 1e-9, rr)
        mask = rr < repel_cutoff
        if np.any(mask):
            drr = rr[mask] - repel_cutoff
            energy += 0.5 * k_repel * float(np.sum(drr**2))
            gvv = (k_repel * drr / rr[mask])[:, None] * dd[mask]
            np.add.at(grad, p[mask], -gvv)
            np.add.at(grad, q[mask], gvv)

    # --- positional restraints (used to hold an imposed bend in place)
    if anchors is not None and len(anchors):
        delta = minimum_image(pos[anchors] - anchor_targets, box)
        energy += 0.5 * k_anchor * float(np.sum(delta**2))
        grad[anchors] += k_anchor * delta

    return energy, grad.ravel()


def relax_shell(
    positions: np.ndarray,
    bonds: set[tuple[int, int]],
    equilibrium: float | Mapping[tuple[int, int], float] = 1.42,
    angle_deg: float = 120.0,
    k_bond: float = 40.0,
    k_angle: float = 15.0,
    k_repel: float = 25.0,
    repel_cutoff: float = 2.2,
    repel_skin: float = 2.0,
    anchors: np.ndarray | None = None,
    anchor_targets: np.ndarray | None = None,
    k_anchor: float = 5.0,
    box: float | None = None,
    outer_cycles: int = 3,
    max_iterations: int = 3000,
    steps: int | None = None,
    exclude_13: bool = True,
) -> np.ndarray:
    """Relax a closed sp2 shell to realistic bond lengths and bond angles.

    Minimises a three-term valence force field -- bond stretching toward
    ``equilibrium``, **true** angle bending toward ``angle_deg``, and a
    short-range non-bonded repulsion -- with L-BFGS-B and exact analytic
    gradients. Because the bond graph is passed in explicitly and held
    fixed for the whole minimisation, this can never alter ring topology;
    it only fixes geometry.

    The angle term is what makes the result physical. Bond springs alone
    (or a 1-3 *distance* proxy for angles) leave the shell free to
    pyramidalise and fold: an earlier fixed-step implementation of this
    function converged to structures with 66-164 deg bond angles and
    dozens of sub-2 Å non-bonded contacts despite near-perfect bond
    lengths. With a real angle term and a proper optimiser, a clean
    capped tube relaxes to 1.415-1.423 Å bonds and 107.8-120.0 deg angles
    -- the 107.8 deg minimum being exactly the interior angle of the cap
    pentagons, which is the correct physical answer rather than a
    numerical artefact.

    The non-bonded neighbour list is rebuilt between ``outer_cycles``
    minimisations (it cannot be updated inside a single L-BFGS run).

    Parameters
    ----------
    positions
        ``(n, 3)`` starting positions (Å). Should already be roughly on
        the intended surface -- see :func:`smooth_mesh_on_capsule`.
    bonds
        Explicit bonded pairs. Never re-derived from distances: on a
        curved shell many non-bonded atoms sit closer than a bond length.
    equilibrium, angle_deg
        sp2 targets: 1.42 Å and 120 deg for graphitic carbon.
        ``equilibrium`` also accepts a mapping from ``(i, j)`` (with
        ``i < j``, matching the keys in ``bonds``) to that bond's own rest
        length, for a structure whose bonds are not all alike -- a binary
        compound with a few homoelemental defect bonds, say, where forcing
        an M-M contact to the M-X length distorts everything around it.
    k_bond, k_angle, k_repel
        Force constants (arbitrary consistent units). ``k_angle`` must be
        a substantial fraction of ``k_bond``; too soft an angle term is
        what allows the fold-through failure described above.
    repel_cutoff
        Non-bonded pairs closer than this (Å) repel. Kept below the ideal
        1-3 distance (2.46 Å) so it never fights the angle term.
    repel_skin
        Extra radius (Å) used when building the non-bonded neighbour list,
        beyond ``repel_cutoff``. The energy term still switches on only
        inside ``repel_cutoff``; the skin exists because the list is
        **frozen for a whole L-BFGS run**, so without it two atoms that
        start further apart than the cutoff are invisible to one another
        for thousands of iterations and pass straight through each other.
        Compact shells never noticed, but a long floppy one buckles: a
        284 Å coiled tube relaxed with no skin drifted 6.5 Å per atom
        (24 Å at worst) and fused neighbouring turns together, 370
        sub-2 Å contacts between atoms 135 bonds apart. A skin, plus the
        conservative rebuild criterion below, removes the failure. Set to
        ``0`` only to reproduce that old behaviour.

        The default is deliberately modest. A larger skin means fewer
        rebuilds but a bigger pair list every iteration, and the list is
        the dominant cost: on a compact 900-atom shell, relaxation takes
        0.9 s with no skin, 1.1 s at 2 Å and 1.8 s at 5 Å, for a bit-wise
        identical result. Compact structures barely move, so they never
        trigger a rebuild at any skin and only pay the list cost; the
        structures that do move are the ones the guard is for.
    box
        Cubic cell length for a periodic structure. Every bond, angle and
        non-bonded vector is then taken under the minimum-image
        convention, and the neighbour search wraps too -- without this a
        bond across the cell seam reads as a box-length stretch and the
        optimiser tears the network apart trying to shorten it.
    anchors, anchor_targets, k_anchor
        Optional harmonic restraints pinning ``positions[anchors]`` near
        ``anchor_targets`` -- used to hold an imposed bend while the rest
        of the lattice relaxes around it.
    exclude_13
        Whether 1-3 pairs are kept out of the non-bonded repulsion. True
        (the default, and correct for sp2 carbon) leaves them to the
        angle term. Pass False together with ``k_angle=0`` when no single
        angle target fits the centre -- a six-coordinate metal -- so the
        repulsion holds its ligands apart instead. Choose
        ``repel_cutoff`` below the real 1-3 distance so the term is inert
        at equilibrium and only resists a collapse.
    outer_cycles, max_iterations
        Neighbour-list rebuilds, and L-BFGS iterations per cycle.
    steps
        Deprecated alias for ``max_iterations``, accepted so older calls
        keep working.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` relaxed positions.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    if steps is not None:
        max_iterations = int(steps)

    n_atoms = len(positions)
    bond_arr, angle_arr, excluded = _valence_terms(
        list(bonds), n_atoms, exclude_13=exclude_13)
    if isinstance(equilibrium, Mapping):
        # Keyed by the bond pair rather than positional, so a per-bond
        # target cannot silently misalign with `bond_arr`'s sorted order.
        try:
            equilibrium = np.array(
                [equilibrium[(int(a), int(b))] for a, b in bond_arr],
                dtype=float)
        except KeyError as exc:
            raise ValueError(
                f"equilibrium is missing a length for bond {exc.args[0]}; "
                "a mapping must cover every bond passed in."
            ) from exc
    theta0 = np.radians(angle_deg)
    anchor_targets = (
        np.asarray(anchor_targets, dtype=float) if anchor_targets is not None else None
    )
    anchors = np.asarray(anchors, dtype=int) if anchors is not None else None

    x = positions.ravel().astype(float).copy()
    skin = max(0.0, repel_skin)
    # A rebuild forced by the Verlet criterion is not one of the caller's
    # cycles -- it is the same cycle continuing on a fresh list -- so it
    # does not consume the budget. The hard cap only guarantees
    # termination if a structure oscillates instead of settling.
    cycles_left = max(1, outer_cycles)
    rebuilds_left = 40 * cycles_left
    while cycles_left > 0 and rebuilds_left > 0:
        reference = x.reshape(n_atoms, 3).copy()
        if box is None:
            tree = cKDTree(reference)
        else:
            # cKDTree's boxsize needs coordinates inside [0, box).
            tree = cKDTree(np.mod(reference, box), boxsize=box)
        candidates = tree.query_pairs(
            r=repel_cutoff + skin, output_type="ndarray"
        )
        if len(candidates):
            keep = [
                (int(a), int(b)) not in excluded for a, b in candidates
            ]
            repel_arr = candidates[np.array(keep)] if any(keep) else np.empty((0, 2), int)
        else:
            repel_arr = np.empty((0, 2), dtype=int)

        def _rebuild_when_stale(xk, _ref=reference):
            # Conservative Verlet criterion: a pair excluded from the list
            # can only reach the repulsion cutoff once its two atoms have
            # closed the skin between them, so the list is stale exactly
            # when the two largest displacements sum past the skin. The
            # naive "any atom moved half the skin" is far more trigger-happy
            # -- ordinary local rearrangement moves atoms about an Ångström
            # and would restart L-BFGS (discarding its history) over and
            # over, which is why this one is worth stating precisely.
            moved = np.linalg.norm(
                minimum_image(xk.reshape(n_atoms, 3) - _ref, box), axis=1
            )
            if n_atoms >= 2:
                largest = np.partition(moved, -2)[-2:]
                if largest.sum() > skin:
                    raise StopIteration
            elif moved.max() > skin:
                raise StopIteration

        result = minimize(
            _vff_energy_gradient,
            x,
            args=(
                bond_arr, angle_arr, repel_arr, n_atoms, equilibrium, theta0,
                k_bond, k_angle, k_repel, repel_cutoff,
                anchors, anchor_targets, k_anchor, box,
            ),
            jac=True,
            method="L-BFGS-B",
            callback=None if skin <= 0 else _rebuild_when_stale,
            options={
                "maxiter": max_iterations,
                "maxfun": max_iterations * 2,
                "ftol": 1e-14,
                "gtol": 1e-10,
            },
        )
        x = result.x
        if result.status == 99:  # callback stopped it: stale list, not a cycle
            rebuilds_left -= 1
        else:
            cycles_left -= 1
    return x.reshape(n_atoms, 3)


def apply_surface_roughness(
    positions: np.ndarray,
    bonds: set[tuple[int, int]],
    sigma: float,
    rng: np.random.Generator,
    equilibrium: float = 1.42,
    box: float | None = None,
    settle_iterations: int = 300,
) -> np.ndarray:
    """Corrugate a relaxed shell out of plane, the way a grown wall is.

    A perfectly relaxed sp2 shell is unrealistically smooth: real
    CVD-grown walls ripple, because they nucleate on a rough catalyst and
    freeze in long-wavelength out-of-plane undulations. This displaces
    each atom along its **local surface normal** by a Gaussian of width
    ``sigma``, then re-relaxes briefly with the displaced positions as
    weak restraints, so bond lengths recover while the corrugation
    survives.

    Displacing along the normal rather than isotropically is what makes
    this read as corrugation instead of noise: in-plane jitter just
    strains bonds and is undone by the relaxation, whereas out-of-plane
    motion is the soft direction a sheet actually has.

    Topology is untouched -- no bond is made or broken -- so ring
    statistics are identical before and after.

    Parameters
    ----------
    positions
        ``(n, 3)`` relaxed positions.
    bonds
        Explicit bond list (also used to find each atom's neighbours).
    sigma
        RMS out-of-plane displacement in Å. Around 0.1-0.3 Å reads as a
        realistically grown wall; beyond ~0.5 Å the sheet starts to
        buckle rather than ripple.
    rng
        Seeded generator, so a given roughness is reproducible.
    equilibrium, box, settle_iterations
        Passed through to :func:`relax_shell` for the settling pass.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` corrugated positions.
    """
    if sigma <= 0:
        return positions

    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in bonds:
        neighbours[int(a)].append(int(b))
        neighbours[int(b)].append(int(a))

    displaced = positions.copy()
    amplitudes = rng.normal(0.0, sigma, size=len(positions))
    for index, ns in neighbours.items():
        if len(ns) < 3:
            continue
        # Local normal from two neighbour bond vectors.
        v1 = minimum_image(positions[ns[0]] - positions[index], box)
        v2 = minimum_image(positions[ns[1]] - positions[index], box)
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        displaced[index] = positions[index] + amplitudes[index] * (normal / norm)

    # Settle with the corrugated positions as restraints: bonds recover,
    # the ripple stays. Without restraints the relaxation simply erases it.
    return relax_shell(
        displaced,
        bonds,
        equilibrium=equilibrium,
        box=box,
        anchors=np.arange(len(displaced)),
        anchor_targets=displaced,
        k_anchor=4.0,
        max_iterations=settle_iterations,
        outer_cycles=1,
    )


def pick_interior_edge(
    faces: np.ndarray,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
    required_degree: dict[int, int] | None = None,
    n_tries: int = 200,
) -> tuple[int, int] | None:
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
