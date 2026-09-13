"""Turn an implicit surface into a triangulation whose dual is a honeycomb.

Marching cubes gives a watertight triangulation of a level set, but a
useless one for this purpose: its triangles follow the sampling grid, so
vertex degrees scatter from 3 to 9. That matters more here than in
ordinary graphics, because in
:func:`nanocarbon_lab.builders.fullerene_mesh.dual_honeycomb` **a mesh
vertex of degree d becomes a carbon ring of size d**. A degree-3 vertex
is a three-membered ring; a degree-9 vertex is a nine-membered hole.
Neither exists in real sp2 carbon.

So the raw mesh is isotropically remeshed (Botsch & Kobbelt): repeatedly
split long edges, collapse short ones, flip edges toward degree 6, then
smooth tangentially and project back onto the surface. The result has
degrees concentrated on 6, with 5s where the surface is convex and 7s
where it saddles -- which is exactly the pentagon/hexagon/heptagon
distribution real curved carbon adopts, arrived at from the geometry
rather than imposed by hand.

Every operation preserves manifoldness, and :func:`mesh_statistics`
re-derives the Euler characteristic so callers can assert it rather than
trust it. The edge-collapse link condition is the subtle part: collapsing
an edge whose endpoints share more than the two opposite vertices tears
the surface, so those collapses are rejected.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .fullerene_mesh import Mesh, minimum_image
from .implicit import Field, project_to_surface


def marching_cubes_mesh(field: Field, extent: float, resolution: int = 80) -> Mesh:
    """Sample ``field`` on a cubic grid and extract its zero level set.

    Parameters
    ----------
    field
        Scalar field; the surface is where it crosses zero.
    extent
        Half-width of the sampling box. **Must clear the whole surface**
        -- if the surface reaches the box wall, marching cubes returns an
        open mesh with boundary edges, whose dual is not a closed carbon
        network. The field factories in
        :mod:`nanocarbon_lab.builders.implicit` return a safe value.
    resolution
        Grid points per axis. Higher resolves fine features but costs
        ``resolution**3`` samples; the remesher sets the final triangle
        size, so this only needs to capture the shape.

    Returns
    -------
    (vertices, triangles)
    """
    from skimage.measure import marching_cubes

    axis = np.linspace(-extent, extent, resolution)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    values = field(grid)
    spacing = float(axis[1] - axis[0])
    verts, faces, _, _ = marching_cubes(values, level=0.0, spacing=(spacing,) * 3)
    return verts - extent, faces.astype(int)


def marching_cubes_box(
    field: Field,
    lower: np.ndarray,
    upper: np.ndarray,
    spacing: float,
    max_samples: int = 40_000_000,
) -> Mesh:
    """Extract a zero level set inside an arbitrary axis-aligned box.

    :func:`marching_cubes_mesh` samples a *cube* centred on the origin,
    which is right for a junction radiating from a centre but wasteful for
    a long, flat object: a two-turn coil of 40 Å radius and 25 Å pitch fits
    in a 92 x 92 x 60 Å box, and padding that to a cube spends over a third
    of the grid on empty space. Since cost is the product of the three axis
    counts, sampling the real box is a direct saving, and it is what makes
    meshing a coil affordable at a voxel fine enough to resolve its wall.

    Parameters
    ----------
    field
        Scalar field; the surface is where it crosses zero.
    lower, upper
        Opposite corners of the sampling box (Å). Must clear the surface
        entirely -- a surface touching a wall gives an open mesh, whose
        dual is not a closed carbon network.
    spacing
        Target voxel edge (Å), applied to all three axes. The realised
        spacing per axis differs slightly because each axis takes a whole
        number of samples.
    max_samples
        Guard on total grid points. Exceeding it raises rather than
        exhausting memory.

    Returns
    -------
    (vertices, triangles)
        Vertices in the same absolute coordinates as ``lower``/``upper``.

    Raises
    ------
    ValueError
        For a degenerate box, a non-positive spacing, or a grid that would
        exceed ``max_samples``.
    """
    from skimage.measure import marching_cubes

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
        raise ValueError("lower/upper must be 3-vectors with upper > lower.")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    counts = np.maximum(4, np.ceil((upper - lower) / spacing).astype(int) + 1)
    total = int(np.prod(counts, dtype=np.int64))
    if total > max_samples:
        raise ValueError(
            f"Sampling {counts.tolist()} = {total:,} grid points exceeds the "
            f"{max_samples:,} limit. Use a coarser spacing or a smaller shape."
        )

    axes = [np.linspace(lower[i], upper[i], counts[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = field(grid)
    steps = tuple(float(a[1] - a[0]) for a in axes)
    verts, faces, _, _ = marching_cubes(values, level=0.0, spacing=steps)
    return verts + lower, faces.astype(int)


def periodic_marching_cubes_mesh(
    field: Field, cell: float, resolution: int = 64
) -> Mesh:
    """Mesh one period of a triply periodic surface, closed on the 3-torus.

    Sampling spans ``[0, cell]`` inclusive; because the field has period
    ``cell``, the two end planes carry identical values and marching cubes
    emits the same surface pattern on each. Wrapping every vertex into
    ``[0, cell)`` and welding coincident ones therefore stitches the
    ``x = cell`` face onto ``x = 0`` (and likewise in y, z), turning the
    open slab into a closed manifold on the torus -- no caps, no clipping,
    tubes simply continuing into the neighbouring cell.

    That closure is what makes the result a *schwarzite unit cell* rather
    than the sphere-clipped blob a naive extraction gives: the Euler
    characteristics come out at the textbook values (Schwarz P genus 3,
    gyroid genus 5, Schwarz D genus 9).

    Parameters
    ----------
    field
        Periodic scalar field with period ``cell``.
    cell
        Cubic cell length (Å).
    resolution
        Grid points per axis across one period. Too coarse and the weld
        fails, leaving boundary edges -- the caller must check
        :func:`mesh_statistics`.

    Returns
    -------
    (vertices, triangles)
        Vertices lie in ``[0, cell)``. Triangles may span the seam, so all
        downstream geometry must use the minimum-image convention.
    """
    from skimage.measure import marching_cubes

    axis = np.linspace(0.0, cell, resolution)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    values = field(grid)
    spacing = float(axis[1] - axis[0])
    verts, faces, _, _ = marching_cubes(values, level=0.0, spacing=(spacing,) * 3)

    wrapped = np.mod(verts, cell)
    # Weld on a quantised key: matching vertices on opposite faces come
    # from identical interpolations, so they agree to many digits.
    keys = np.round(wrapped / spacing * 1e3).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    new_faces = inverse[faces.astype(int)]
    keep = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    return wrapped[first], new_faces[keep]


def mesh_statistics(mesh: Mesh) -> dict[str, int]:
    """Euler characteristic, genus and boundary-edge count of a mesh.

    ``boundary_edges`` must be 0 for a closed surface; anything else means
    the mesh has holes (usually a sampling box that clipped the surface)
    and its dual will not be a valid carbon network. ``genus`` follows
    from ``V - E + F = 2 - 2g`` and sets the ring budget downstream:
    ``sum(6 - ring_size) = 6 * euler``.
    """
    verts, faces = mesh
    counts: Counter = Counter()
    for tri in faces:
        a, b, c = (int(x) for x in tri)
        for u, v in ((a, b), (b, c), (c, a)):
            counts[(u, v) if u < v else (v, u)] += 1
    boundary = sum(1 for n in counts.values() if n != 2)
    euler = len(verts) - len(counts) + len(faces)
    return {
        "vertices": len(verts),
        "edges": len(counts),
        "faces": len(faces),
        "euler": euler,
        "genus": (2 - euler) // 2,
        "boundary_edges": boundary,
    }


def _edge_faces(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map each undirected edge to the faces containing it."""
    mapping: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, tri in enumerate(faces):
        a, b, c = (int(x) for x in tri)
        for u, v in ((a, b), (b, c), (c, a)):
            mapping[(u, v) if u < v else (v, u)].append(index)
    return mapping


def _adjacency(faces: np.ndarray) -> dict[int, set[int]]:
    nbrs: dict[int, set[int]] = defaultdict(set)
    for tri in faces:
        a, b, c = (int(x) for x in tri)
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return nbrs


def _split_long_edges(mesh: Mesh, threshold: float, box: float | None = None) -> Mesh:
    """Subdivide every edge longer than ``threshold`` (red-green style).

    Each face is rebuilt from whichever of its three edges were marked,
    so 1, 2 or 3 marked edges give 2, 3 or 4 sub-triangles. Rebuilding
    per face (rather than splicing edge by edge) keeps orientation
    consistent and cannot leave T-junctions.
    """
    verts, faces = mesh
    lengths = {}
    for tri in faces:
        a, b, c = (int(x) for x in tri)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            if key not in lengths:
                lengths[key] = np.linalg.norm(
                    minimum_image(verts[v] - verts[u], box)
                )
    marked = {e for e, length in lengths.items() if length > threshold}
    if not marked:
        return mesh

    new_verts = list(verts)
    midpoint: dict[tuple[int, int], int] = {}
    for edge in marked:
        u, v = edge
        midpoint[edge] = len(new_verts)
        # Step from u along the *shortest* image of the edge, so a bond that
        # wraps the cell splits at its true midpoint rather than halfway
        # across the box.
        mid = verts[u] + 0.5 * minimum_image(verts[v] - verts[u], box)
        new_verts.append(mid if box is None else np.mod(mid, box))

    def mid(u: int, v: int) -> int | None:
        return midpoint.get((u, v) if u < v else (v, u))

    new_faces: list[tuple[int, int, int]] = []
    for tri in faces:
        a, b, c = (int(x) for x in tri)
        mab, mbc, mca = mid(a, b), mid(b, c), mid(c, a)
        present = [m for m in (mab, mbc, mca) if m is not None]
        if not present:
            new_faces.append((a, b, c))
        elif len(present) == 3:
            new_faces += [
                (a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)
            ]
        elif len(present) == 2:
            # Rotate the triangle so the *un-split* edge is always c->a
            # (i.e. mca is the None one), then the split is one fixed
            # pattern instead of three. Rotating (a,b,c)->(b,c,a) carries
            # the midpoints (mab,mbc,mca)->(mbc,mca,mab).
            if mab is None:
                a, b, c, mab, mbc, mca = b, c, a, mbc, mca, mab
            elif mbc is None:
                a, b, c, mab, mbc, mca = c, a, b, mca, mab, mbc
            new_faces += [(a, mab, mbc), (mab, b, mbc), (a, mbc, c)]
        else:
            if mab is not None:
                new_faces += [(a, mab, c), (mab, b, c)]
            elif mbc is not None:
                new_faces += [(b, mbc, a), (mbc, c, a)]
            else:
                new_faces += [(c, mca, b), (mca, a, b)]
    return np.array(new_verts, dtype=float), np.array(new_faces, dtype=int)


def _collapse_short_edges(
    mesh: Mesh, threshold: float, max_length: float, box: float | None = None
) -> Mesh:
    """Collapse edges shorter than ``threshold`` where it is safe to do so.

    Three guards keep the mesh manifold. The **link condition**: ``u`` and
    ``v`` may share exactly the two vertices opposite their common edge --
    collapsing across any other shared neighbour pinches the surface into
    a non-manifold point. A **length guard**: the collapse must not create
    an edge longer than ``max_length``, which would immediately be split
    again next pass, so the remesher would never converge. And a
    **locking** rule: once an edge is collapsed, its merged vertex and
    that vertex's whole one-ring are locked for the rest of the pass.

    Locking is what makes the link condition trustworthy. Adjacency is
    computed once per pass for speed, so after a collapse it is stale for
    every vertex near the merge; testing a later collapse against stale
    adjacency silently admits ones that tear the surface. (Without this,
    intermediate meshes here came out with tens of boundary edges and a
    nonsensical genus, which then happened to heal over later
    iterations -- correct by luck rather than construction.)
    """
    verts, faces = mesh
    nbrs = _adjacency(faces)
    edge_faces = _edge_faces(faces)

    dead: set[int] = set()
    locked: set[int] = set()
    remap: dict[int, int] = {}
    positions = verts.copy()

    def resolve(index: int) -> int:
        while index in remap:
            index = remap[index]
        return index

    for edge, incident in edge_faces.items():
        u, v = edge
        if u in locked or v in locked or u in dead or v in dead:
            continue
        if len(incident) != 2:
            continue
        if np.linalg.norm(
            minimum_image(positions[v] - positions[u], box)
        ) >= threshold:
            continue
        opposite = set()
        for face_index in incident:
            opposite |= {int(x) for x in faces[face_index]} - {u, v}
        if len(opposite) != 2 or nbrs[u] & nbrs[v] != opposite:
            continue  # link condition violated: collapse would pinch
        target = positions[u] + 0.5 * minimum_image(positions[v] - positions[u], box)
        if box is not None:
            target = np.mod(target, box)
        merged = (nbrs[u] | nbrs[v]) - {u, v}
        if any(
            np.linalg.norm(minimum_image(positions[w] - target, box)) > max_length
            for w in merged
        ):
            continue
        positions[u] = target
        remap[v] = u
        dead.add(v)
        nbrs[u] = merged
        for w in merged:
            nbrs[w].discard(v)
            nbrs[w].add(u)
        # Freeze this neighbourhood: the cached adjacency is now stale here.
        locked.add(u)
        locked.update(merged)

    if not dead:
        return mesh

    new_faces = []
    for tri in faces:
        a, b, c = (resolve(int(x)) for x in tri)
        if len({a, b, c}) == 3:
            new_faces.append((a, b, c))
    keep = np.array(sorted(set(range(len(positions))) - dead), dtype=int)
    reindex = -np.ones(len(positions), dtype=int)
    reindex[keep] = np.arange(len(keep))
    remapped = np.array(
        [[reindex[a], reindex[b], reindex[c]] for a, b, c in new_faces], dtype=int
    )
    return positions[keep], remapped


def _flip_edges_toward_degree_six(mesh: Mesh) -> Mesh:
    """Flip edges when doing so brings the four touched degrees nearer 6.

    Degree is ring size in the dual, so this is what drives the carbon
    network toward hexagons, leaving 5s and 7s only where curvature
    genuinely demands them.
    """
    verts, faces = mesh
    face_list = [tuple(int(x) for x in tri) for tri in faces]
    nbrs = _adjacency(faces)
    degree = {v: len(ns) for v, ns in nbrs.items()}
    edge_faces = _edge_faces(faces)

    def deviation(*values: int) -> int:
        return sum(abs(v - 6) for v in values)

    touched: set[int] = set()
    for edge, incident in edge_faces.items():
        if len(incident) != 2:
            continue
        if incident[0] in touched or incident[1] in touched:
            continue
        u, v = edge
        opposite = set()
        for face_index in incident:
            opposite |= set(face_list[face_index]) - {u, v}
        if len(opposite) != 2:
            continue
        w1, w2 = sorted(opposite)
        if w2 in nbrs[w1]:
            continue  # would duplicate an existing edge
        before = deviation(degree[u], degree[v], degree[w1], degree[w2])
        after = deviation(
            degree[u] - 1, degree[v] - 1, degree[w1] + 1, degree[w2] + 1
        )
        if after >= before:
            continue
        face_list[incident[0]] = (u, w1, w2)
        face_list[incident[1]] = (v, w2, w1)
        degree[u] -= 1
        degree[v] -= 1
        degree[w1] += 1
        degree[w2] += 1
        nbrs[u].discard(v)
        nbrs[v].discard(u)
        nbrs[w1].add(w2)
        nbrs[w2].add(w1)
        touched.update(incident)
    return verts, np.array(face_list, dtype=int)


def _tangential_smooth(
    mesh: Mesh, field: Field, strength: float = 0.5, box: float | None = None,
    max_step: float | None = None,
) -> Mesh:
    """Laplacian-smooth vertices, then project them back onto the surface.

    Smoothing equalises triangle sizes but pulls vertices off the
    surface (and shrinks a closed shape toward its centre); the
    projection step in :func:`implicit.project_to_surface` puts them
    back, so only the tangential component of the motion survives.
    """
    verts, faces = mesh
    nbrs = _adjacency(faces)
    target = verts.copy()
    for index, neighbours in nbrs.items():
        if neighbours:
            # Average the neighbours' *offsets* under the minimum image, not
            # their raw coordinates: across a periodic seam the raw mean lands
            # in the middle of the cell instead of next door.
            offsets = minimum_image(verts[list(neighbours)] - verts[index], box)
            target[index] = verts[index] + offsets.mean(axis=0)
    moved = verts + strength * (target - verts)
    projected = project_to_surface(field, moved, max_step=max_step)
    return (projected if box is None else np.mod(projected, box)), faces


def _remove_low_degree_vertices(mesh: Mesh, min_degree: int = 5) -> Mesh:
    """Collapse away vertices below ``min_degree``.

    Degree is ring size in the dual, so a degree-3 or degree-4 vertex is a
    three- or four-membered carbon ring. Those are not merely rare, they
    are chemically absurd in an sp2 sheet, and the length-based collapse
    will not touch them when their edges happen to be of normal length.
    Collapsing one of the vertex's edges deletes it outright; Euler's
    budget then redistributes into pentagons, which are perfectly
    physical.

    Same link condition and locking as :func:`_collapse_short_edges`.
    """
    verts, faces = mesh
    nbrs = _adjacency(faces)
    low = {v for v, ns in nbrs.items() if len(ns) < min_degree}
    if not low:
        return mesh
    edge_faces = _edge_faces(faces)

    dead: set[int] = set()
    locked: set[int] = set()
    remap: dict[int, int] = {}
    positions = verts.copy()

    for edge, incident in edge_faces.items():
        u, v = edge
        if not (u in low or v in low):
            continue
        if u in locked or v in locked or u in dead or v in dead:
            continue
        if len(incident) != 2:
            continue
        opposite = set()
        for face_index in incident:
            opposite |= {int(x) for x in faces[face_index]} - {u, v}
        if len(opposite) != 2 or nbrs[u] & nbrs[v] != opposite:
            continue
        # Keep the higher-degree endpoint; it is the healthier vertex.
        keep, drop = (u, v) if len(nbrs[u]) >= len(nbrs[v]) else (v, u)
        merged = (nbrs[u] | nbrs[v]) - {u, v}
        positions[keep] = 0.5 * (positions[u] + positions[v])
        remap[drop] = keep
        dead.add(drop)
        nbrs[keep] = merged
        for w in merged:
            nbrs[w].discard(drop)
            nbrs[w].add(keep)
        locked.add(keep)
        locked.update(merged)

    if not dead:
        return mesh

    def resolve(index: int) -> int:
        while index in remap:
            index = remap[index]
        return index

    new_faces = []
    for tri in faces:
        a, b, c = (resolve(int(x)) for x in tri)
        if len({a, b, c}) == 3:
            new_faces.append((a, b, c))
    keep_idx = np.array(sorted(set(range(len(positions))) - dead), dtype=int)
    reindex = -np.ones(len(positions), dtype=int)
    reindex[keep_idx] = np.arange(len(keep_idx))
    remapped = np.array(
        [[reindex[a], reindex[b], reindex[c]] for a, b, c in new_faces], dtype=int
    )
    return positions[keep_idx], remapped


def isotropic_remesh(
    mesh: Mesh,
    field: Field,
    target_edge: float,
    iterations: int = 25,
    box: float | None = None,
    anneal_sweeps: int = 80,
    anneal_temperature: float = 0.3,
    rng: np.random.Generator | None = None,
) -> Mesh:
    """Remesh to near-uniform triangles of side ``target_edge``.

    Runs the standard split / collapse / flip / smooth cycle. The 4/3 and
    4/5 thresholds are Botsch & Kobbelt's: they bracket the target
    length so an edge cannot be split and collapsed on alternate passes.

    Parameters
    ----------
    mesh
        Starting triangulation, typically from :func:`marching_cubes_mesh`.
    field
        The implicit surface, used to re-project smoothed vertices.
    target_edge
        Desired triangle side. In the dual this sets the carbon ring
        size, so it should be roughly the ring-centre spacing you want --
        about ``sqrt(3) * bond`` (2.46 Å) for graphitic carbon.
    anneal_sweeps, anneal_temperature, rng
        Passed to :func:`anneal_edge_flips` after the main loop. Set
        ``anneal_sweeps=0`` to keep the as-remeshed defect population,
        which reads as a rougher, more CVD-like wall.
    box
        Cubic cell length when remeshing a periodic surface; every distance
        and midpoint is then taken under the minimum-image convention, so
        the seam across the cell boundary is treated like any other part of
        the mesh. ``None`` for a finite surface.
    iterations
        Full cycles. The mesh shrinks toward equilibrium from whatever
        marching cubes produced, so this needs to be generous: on a Y
        junction the vertex count settles around pass 20 and the degree
        histogram stops moving by pass 25.

    Returns
    -------
    (vertices, triangles)
        A closed manifold whose vertex degrees cluster on 6.
    """
    if target_edge <= 0:
        raise ValueError("target_edge must be positive.")
    # A smoothing pass moves a vertex by at most half an edge, so a
    # projection step of one edge length is already generous. Capping it
    # there stops a vertex that drifted past the midline of a narrow gap
    # from being projected onto the *facing* sheet of the surface -- see
    # `project_to_surface`. That failure is silent (the mesh stays closed
    # and keeps its genus) and only surfaces much later, as a patch of
    # carbon fused to a wall one coil turn away.
    smooth_step = target_edge
    for _ in range(max(1, iterations)):
        mesh = _split_long_edges(mesh, (4.0 / 3.0) * target_edge, box)
        mesh = _collapse_short_edges(
            mesh, (4.0 / 5.0) * target_edge, (4.0 / 3.0) * target_edge, box
        )
        mesh = _flip_edges_toward_degree_six(mesh)
        mesh = _tangential_smooth(mesh, field, box=box, max_step=smooth_step)
    # Anneal out the dislocation pairs the greedy loop cannot escape, then
    # let the geometry follow the new connectivity.
    if anneal_sweeps > 0:
        mesh = anneal_edge_flips(
            mesh,
            rng if rng is not None else np.random.default_rng(0),
            sweeps=anneal_sweeps,
            temperature=anneal_temperature,
        )
        mesh = _tangential_smooth(mesh, field, box=box, max_step=smooth_step)

    # Final cleanup: three- and four-membered rings cannot exist in sp2
    # carbon, and length-based collapse leaves them when their edges are of
    # ordinary length. Flip afterwards to re-settle the degrees disturbed.
    for _ in range(3):
        cleaned = _remove_low_degree_vertices(mesh)
        if cleaned[0].shape == mesh[0].shape:
            break
        mesh = _flip_edges_toward_degree_six(cleaned)
        mesh = _tangential_smooth(mesh, field, box=box, max_step=smooth_step)
    return mesh


def anneal_edge_flips(
    mesh: Mesh,
    rng: np.random.Generator,
    sweeps: int = 80,
    temperature: float = 0.3,
    min_degree: int = 5,
    max_degree: int = 8,
) -> Mesh:
    """Metropolis-anneal edge flips to remove spurious dislocation pairs.

    :func:`isotropic_remesh` only ever accepts a flip that lowers the total
    degree deviation, so it stalls in a local minimum littered with
    pentagon-heptagon pairs beyond those curvature actually requires --
    measured at 39 pairs on a Y junction whose ideal is 6, and barely
    improved by running more iterations (39 -> 35 for five times the work).
    Those pairs are dislocations: topologically neutral, and real in
    CVD-grown material, but not something the caller should be stuck with.

    Accepting an occasional worsening flip with probability
    ``exp(-delta / T)`` lets the mesh climb out. Cooling linearly to zero
    leaves it in a much better minimum: on the same junction this reaches
    14 pairs, a 64% reduction, in about a second.

    Degrees are hard-clamped to ``[min_degree, max_degree]`` regardless of
    temperature. Without that clamp a hot run trades dislocations for
    three- and four-membered rings, which is a strictly worse structure --
    those cannot exist in sp2 carbon at all.

    Parameters
    ----------
    mesh
        Remeshed triangulation. Only connectivity changes; vertex
        positions are untouched, so callers usually smooth afterwards.
    rng
        Seeded generator -- annealing is stochastic, so results are
        reproducible only through this.
    sweeps
        Passes over the edge list. Temperature falls linearly to zero
        across them. 80 is comfortably converged; 0 disables annealing and
        leaves the as-remeshed defect population intact.
    temperature
        Starting temperature. Measured optimum is ~0.3: colder barely
        escapes the minimum, hotter (>=0.6) wanders and ends up worse.

    Returns
    -------
    (vertices, triangles)
        Same vertices, re-flipped triangles.
    """
    if sweeps <= 0:
        return mesh

    verts, faces = mesh
    face_list = [tuple(int(x) for x in tri) for tri in faces]
    nbrs = _adjacency(faces)
    degree = {v: len(ns) for v, ns in nbrs.items()}

    def deviation(*values: int) -> int:
        return sum(abs(v - 6) for v in values)

    for sweep in range(sweeps):
        temp = temperature * (1.0 - sweep / max(1, sweeps - 1))
        edge_faces = _edge_faces(np.array(face_list, dtype=int))
        items = list(edge_faces.items())
        rng.shuffle(items)
        touched: set[int] = set()
        for edge, incident in items:
            if len(incident) != 2:
                continue
            if incident[0] in touched or incident[1] in touched:
                continue
            u, v = edge
            opposite = set()
            for face_index in incident:
                opposite |= set(face_list[face_index]) - {u, v}
            if len(opposite) != 2:
                continue
            w1, w2 = sorted(opposite)
            if w2 in nbrs[w1]:
                continue
            if (
                degree[u] - 1 < min_degree
                or degree[v] - 1 < min_degree
                or degree[w1] + 1 > max_degree
                or degree[w2] + 1 > max_degree
            ):
                continue
            delta = deviation(
                degree[u] - 1, degree[v] - 1, degree[w1] + 1, degree[w2] + 1
            ) - deviation(degree[u], degree[v], degree[w1], degree[w2])
            if delta > 0 and (
                temp <= 0.0 or rng.random() >= np.exp(-delta / temp)
            ):
                continue
            face_list[incident[0]] = (u, w1, w2)
            face_list[incident[1]] = (v, w2, w1)
            degree[u] -= 1
            degree[v] -= 1
            degree[w1] += 1
            degree[w2] += 1
            nbrs[u].discard(v)
            nbrs[v].discard(u)
            nbrs[w1].add(w2)
            nbrs[w2].add(w1)
            touched.update(incident)

    return verts, np.array(face_list, dtype=int)


def dislocation_pairs(mesh: Mesh) -> int:
    """Count pentagon-heptagon pairs beyond what curvature requires.

    A 5-7 pair contributes nothing to the Euler budget, so the number of
    such pairs is a direct measure of how "as-grown" (rough) versus
    "annealed" (smooth) the network is.
    """
    histogram = degree_histogram(mesh)
    return min(histogram.get(5, 0), histogram.get(7, 0))


def degree_histogram(mesh: Mesh) -> dict[int, int]:
    """``{vertex degree: count}`` -- the dual's ring-size distribution."""
    nbrs = _adjacency(mesh[1])
    return dict(sorted(Counter(len(ns) for ns in nbrs.values()).items()))
