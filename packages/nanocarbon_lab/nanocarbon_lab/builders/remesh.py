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

import warnings
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
    field: Field, cell: float | np.ndarray, resolution: int = 64
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
        Cell edge length (Å): a scalar cube, or a **three-vector** for an
        orthorhombic cell. The anisotropic case is what a coil needs --
        it is periodic along its axis with period equal to the pitch, and
        merely wide in the other two. There the weld in x and y does
        nothing, because vacuum means no surface reaches those walls, and
        only the z seam does real work.
    resolution
        Grid points across the **longest** axis; the others get as many
        as they need to keep the voxels cube-shaped, so a scalar cell
        behaves exactly as before. Too coarse and the weld fails, leaving
        boundary edges -- the caller must check :func:`mesh_statistics`.

    Returns
    -------
    (vertices, triangles)
        Vertices lie in ``[0, cell)``. Triangles may span the seam, so all
        downstream geometry must use the minimum-image convention.
    """
    from skimage.measure import marching_cubes

    edges = np.broadcast_to(np.asarray(cell, dtype=float), (3,))
    if np.any(edges <= 0.0):
        raise ValueError(f"Cell edges must all be positive, got {edges}.")
    # One voxel size for all three axes, so an elongated cell is not
    # sampled finely across its short axis and coarsely across its long
    # one -- which would resolve the wall in z and lose it in x.
    voxel = float(edges.max()) / max(1, resolution - 1)
    counts = [max(2, int(round(length / voxel)) + 1) for length in edges]
    axes = [np.linspace(0.0, length, count)
            for length, count in zip(edges, counts, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = field(grid)
    spacing = tuple(float(a[1] - a[0]) for a in axes)
    verts, faces, _, _ = marching_cubes(values, level=0.0, spacing=spacing)

    wrapped = np.mod(verts, edges)
    # Weld on a quantised key: matching vertices on opposite faces come
    # from identical interpolations, so they agree to many digits.
    keys = np.round(wrapped / np.asarray(spacing) * 1e3).astype(np.int64)
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
    anneal_restarts: int = 1,
    place_curvature: bool = False,
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
    place_curvature
        Run :func:`place_disclinations` after annealing, moving the
        disclinations to where the curvature asks for them without
        changing how many there are. Measured misfit reductions of
        17-36% on the four junction kinds. Off by default because it
        changes every structure that uses it.
    anneal_sweeps, anneal_temperature, anneal_restarts, rng
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
            restarts=anneal_restarts,
        )
    if place_curvature:
        # The full refinement, not one greedy pass: placement alone
        # stalls at 47/43 on a toroid where alternating it with the
        # annealer reaches 17/15. See `refine_disclinations`.
        mesh, _log = refine_disclinations(mesh, rng)
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


#: Neighbourhood radius, in rings, over which disclination charge and
#: curvature charge are compared. One ring (7 vertices) is too small to
#: hold a disclination's worth of curvature and correlates at 0.574;
#: three (37 vertices) correlates at 0.897 but smears a junction neck
#: into its arms. Two (19 vertices, 0.814) is the working choice.
CURVATURE_WINDOW_RINGS = 2


#: Above this many mesh vertices the refinement is cut back rather than
#: run in full. Each round recomputes every window from scratch, so the
#: cost grows with the mesh and with the round count at once: a 700-vertex
#: toroid converges in minutes, while a 6300-vertex superfullerene had
#: not finished in far longer. A structure that looks like a hung window
#: is worse than one refined less.
LARGE_MESH_VERTICES = 2000


def refine_disclinations(
    mesh: Mesh,
    rng: np.random.Generator | None = None,
    cycles: int = 5,
    sweeps: int = 40,
    temperature: float = 0.3,
    pair_weight: float = 5.0,
    rounds: int = 40,
) -> tuple[Mesh, list[str]]:
    """Anneal the spurious pairs away while holding the curvature split.

    Neither half of this works alone, and the toroid is the clean
    demonstration. **Gauss-Bonnet fixes the disclination budget of a
    torus's outer half at exactly +12 and its inner half at -12, for any
    R and any r** -- in ``K dA = cos(phi) dphi dtheta`` the radii cancel,
    so the integral over the outer half is 4*pi and the budget is
    ``3/pi * 4*pi = 12``. That is where Dunlap's twelve comes from. It is
    not ``4*pi*r/e``, the formula tried first here, which depends on the
    tube radius and gave 17 or 27.

    A remeshed torus already carries that budget **exactly**: measured
    +12 outside and -12 inside straight out of the remesher. What it
    does not have is Dunlap's census, because the budget is carried by 73
    pentagons and 71 heptagons rather than 12 and 12 -- the surplus being
    neutral 5-7 pairs, which are dislocations and cost nothing to the
    budget.

    - **Census annealing alone** removes pairs and scrambles the split:
      73/71 down to 21/17, with the outer charge falling +12 -> +8,
      because nothing in ``sum(|deg - 6|)`` knows which side of the
      equator a defect belongs on.
    - **Placement alone** holds the split and stalls on the pairs: greedy
      descent cannot escape its own minimum, stopping at 47/43.

    Alternating them beats both: **73/71 -> 17/15 with the split at
    +11/-11**, converged by the fifth cycle. The stochastic pass finds
    the pairs, the exact pass puts what is left back where the curvature
    wants it, and neither undoes the other.

    Parameters
    ----------
    mesh
        Remeshed triangulation.
    rng
        Seeded generator. The annealing half is stochastic, so this is
        the only thing making a result reproducible.
    cycles
        Anneal-then-place rounds. Measured to converge by five on a
        toroid; more cost time and change nothing.
    sweeps, temperature
        Passed to :func:`anneal_edge_flips` for the stochastic half.
    pair_weight, rounds
        Passed to :func:`place_disclinations` for the exact half.

    Returns
    -------
    (mesh, log)
        The refined mesh and one line per cycle, so a caller can show
        that it converged rather than assume it.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(mesh[0]) > LARGE_MESH_VERTICES:
        # Cut back rather than refuse: some improvement, bounded time.
        # The annealing half is cheap and does most of the pair removal;
        # the exact half is what costs, so that is what shrinks.
        scale = LARGE_MESH_VERTICES / len(mesh[0])
        cycles = max(1, int(cycles * scale) or 1)
        rounds = max(4, int(rounds * scale))
        warnings.warn(
            f"{len(mesh[0])} mesh vertices is past "
            f"{LARGE_MESH_VERTICES}, where each refinement round costs more "
            f"than it returns; running {cycles} cycle(s) of {rounds} rounds "
            "instead of the full schedule. The disclinations will be better "
            "placed than unrefined and not as well as on a small cell.",
            stacklevel=2,
        )
    log: list[str] = []
    current = mesh
    previous = None
    for cycle in range(max(1, cycles)):
        current = anneal_edge_flips(current, rng, sweeps=sweeps,
                                    temperature=temperature, restarts=2)
        current, _history = place_disclinations(current, max_rounds=rounds,
                                                pair_weight=pair_weight)
        degrees = _adjacency(current[1])
        defects = sum(1 for ns in degrees.values() if len(ns) != 6)
        log.append(f"cycle {cycle + 1}: {defects} disclinations, "
                   f"misfit {curvature_misfit(current):.1f}")
        # Converged: the last cycle changed nothing at all.
        if previous is not None and np.array_equal(current[1], previous):
            break
        previous = current[1].copy()
    return current, log


def place_disclinations(
    mesh: Mesh,
    targets: np.ndarray | None = None,
    rings: int | None = None,
    max_rounds: int = 60,
    min_degree: int = 5,
    max_degree: int = 8,
    pair_weight: float = 0.0,
) -> tuple[Mesh, list[float]]:
    """Move disclinations to where the curvature wants them, by flips.

    Greedy descent on :func:`curvature_misfit`, recomputing the whole
    state every round. **That is a deliberate choice against a faster
    incremental scheme, which was written first and did not work.** A
    flip changes the neighbourhood structure, so every delta computed
    after it in the same sweep is measured against a graph that no
    longer exists: 45 flips were accepted in one sweep, each of them
    "improving", and the sweep ended worse than it started. Recomputing
    is slower per move and exact, and exactness is what this needs --
    the misfit it reports is then a real measurement rather than an
    accumulation of stale deltas.

    Each round applies every improving flip whose four vertices are far
    enough apart that they cannot interact -- more than twice the window
    radius -- so a round is still many flips, and then verifies the
    result. A round that does not lower the true misfit is rolled back
    and the descent stops.

    The flips are ordinary Stone-Wales moves, so ``sum(6 - deg)`` is
    invariant and the clamp keeps every degree in
    ``[min_degree, max_degree]``. A flip *can* still turn two hexagons
    into a 5-7 pair, so this additionally **refuses any flip that raises
    the disclination count** -- without that guard the descent buys
    misfit by manufacturing pairs to chase curvature finer than one
    disclination can represent. It moves disclinations; it does not
    create them. Separating a 5-7 pair is climb, not glide, and no
    sequence of flips will do that either.

    Parameters
    ----------
    mesh
        Triangulation to rearrange. Positions are never touched.
    targets
        Per-vertex curvature targets; computed from the mesh if omitted.
    rings
        Window radius. Defaults to :data:`CURVATURE_WINDOW_RINGS`.
    pair_weight
        Cost charged per disclination, on top of the misfit. ``0`` leaves
        the count alone and only moves what is there. Positive also
        **annihilates spurious 5-7 pairs**, which is what separates a
        remeshed torus from a Dunlap one: both carry the same net charge
        -- Gauss-Bonnet fixes the outer half of any torus at exactly +12
        and the inner at -12, whatever R and r -- but the remeshed one
        carries it as 73 pentagons and 71 heptagons rather than 12 and
        12, the surplus being neutral pairs. Annealing the pairs away by
        census alone reaches 21 and 17 and *scrambles the split*, +12
        falling to +8, because nothing in that objective knows which side
        of the equator a defect belongs on. Charging for both at once is
        the only thing that holds the split while the pairs go.
    max_rounds
        Cap on descent rounds. Each is a full recomputation.
    min_degree, max_degree
        Hard clamp, as in :func:`anneal_edge_flips`. ``max_degree=8``
        admits octagons, which belong at a junction neck where the
        curvature is most negative.

    Returns
    -------
    (mesh, history)
        The rearranged mesh and the misfit after each round, starting
        with the value it began at. The history is returned rather than
        printed so a caller can assert the descent was monotonic.
    """
    if rings is None:
        rings = CURVATURE_WINDOW_RINGS
    if targets is None:
        targets = vertex_curvature_targets(mesh)

    verts, faces = mesh
    face_list = [tuple(int(x) for x in tri) for tri in faces]
    def total_cost(faces_now) -> float:
        """Misfit plus the per-disclination charge."""
        candidate = (verts, np.asarray(faces_now, dtype=int))
        cost = curvature_misfit(candidate, targets, rings)
        if pair_weight:
            adjacency = _adjacency(candidate[1])
            cost += pair_weight * sum(
                1 for ns in adjacency.values() if len(ns) != 6)
        return cost

    history = [total_cost(face_list)]

    for _ in range(max_rounds):
        current = np.asarray(face_list, dtype=int)
        nbrs = _adjacency(current)
        degree = {v: len(ns) for v, ns in nbrs.items()}
        windows = _vertex_windows(nbrs, len(verts), rings)
        target_sum = np.array([sum(targets[w] for w in win)
                               for win in windows], dtype=float)
        charge = np.array([float(sum(6 - degree[w] for w in win))
                           for win in windows], dtype=float)

        moves = []
        for edge, incident in _edge_faces(current).items():
            if len(incident) != 2:
                continue
            u, v = edge
            opposite: set[int] = set()
            for face_index in incident:
                opposite |= set(face_list[face_index]) - {u, v}
            if len(opposite) != 2:
                continue
            w1, w2 = sorted(opposite)
            if w2 in nbrs[w1]:
                continue
            if (degree[u] - 1 < min_degree or degree[v] - 1 < min_degree
                    or degree[w1] + 1 > max_degree
                    or degree[w2] + 1 > max_degree):
                continue
            # A flip lowers degree at u and v, and charge is 6 - degree,
            # so u and v GAIN charge while w1 and w2 lose it. Reversed,
            # this maximises the misfit: 355 -> 1476 on a Y junction.
            shift: dict[int, float] = {}
            for vertex, amount in ((u, 1.0), (v, 1.0), (w1, -1.0), (w2, -1.0)):
                for y in windows[vertex]:
                    shift[y] = shift.get(y, 0.0) + amount
            delta = 0.0
            for y, amount in shift.items():
                if amount:
                    miss = charge[y] - target_sum[y]
                    delta += (miss + amount) ** 2 - miss ** 2
            # Refuse any flip that ADDS a disclination. Without this the
            # descent buys misfit by manufacturing 5-7 pairs to chase
            # curvature variations smaller than one disclination is
            # worth -- fitting noise. Measured with it off: the misfit
            # fell 17-36% while the fraction of disclinations on the
            # right side of the curvature fell 95.7% -> 85.6% on an X,
            # the two measures moving opposite ways. This is also what
            # makes the docstring's claim true rather than aspirational.
            before = sum(1 for d in (degree[u], degree[v],
                                     degree[w1], degree[w2]) if d != 6)
            after = sum(1 for d in (degree[u] - 1, degree[v] - 1,
                                    degree[w1] + 1, degree[w2] + 1)
                        if d != 6)
            if after > before:
                continue
            delta += pair_weight * (after - before)
            if delta < -1e-9:
                moves.append((delta, edge, tuple(incident), u, v, w1, w2))

        if not moves:
            break
        moves.sort(key=lambda item: item[0])

        # Apply the improving flips that cannot interact. Two are
        # independent only when their windows are disjoint, which is
        # twice the window radius apart -- not one, which was the error
        # the incremental version made.
        reach = _vertex_windows(nbrs, len(verts), 2 * rings)
        blocked: set[int] = set()
        trial = list(face_list)
        applied = 0
        for _delta, _edge, incident, u, v, w1, w2 in moves:
            if {u, v, w1, w2} & blocked:
                continue
            trial[incident[0]] = (u, w1, w2)
            trial[incident[1]] = (v, w2, w1)
            for vertex in (u, v, w1, w2):
                blocked |= reach[vertex]
            applied += 1
        if not applied:
            break

        score = total_cost(trial)
        if score >= history[-1] - 1e-9:
            break
        face_list = trial
        history.append(score)

    return (verts, np.asarray(face_list, dtype=int)), history


def curvature_misfit(mesh: Mesh, targets: np.ndarray | None = None,
                     rings: int = None) -> float:
    """How far the disclinations sit from where curvature wants them.

    The squared miss between the disclination charge in each vertex's
    neighbourhood and the curvature charge there, summed. Zero would mean
    every patch of the surface carries exactly the disclination content
    its own Gaussian curvature asks for. It is the quantity
    :func:`anneal_edge_flips` minimises under ``objective="curvature"``,
    and it is reported rather than hidden so a caller can say whether
    annealing helped instead of assuming it.

    Lower is better, and it is not normalised -- compare two meshes of
    the same surface, not two different surfaces.
    """
    if rings is None:
        rings = CURVATURE_WINDOW_RINGS
    if targets is None:
        targets = vertex_curvature_targets(mesh)
    adjacency = _adjacency(mesh[1])
    windows = _vertex_windows(adjacency, len(mesh[0]), rings)
    total = 0.0
    for win in windows:
        charge = sum(6 - len(adjacency[w]) for w in win)
        total += (charge - sum(targets[w] for w in win)) ** 2
    return float(total)


def _vertex_windows(adjacency: dict[int, set[int]], count: int, rings: int
                    ) -> list[set[int]]:
    """Every vertex's closed ``rings``-neighbourhood, by breadth-first walk.

    Symmetric by construction -- ``w`` is within ``rings`` of ``v`` exactly
    when ``v`` is within ``rings`` of ``w`` -- which is what lets a flip's
    effect be accumulated by walking the four changed vertices' own
    windows rather than every vertex's.
    """
    windows: list[set[int]] = []
    for v in range(count):
        seen = {v}
        frontier = {v}
        for _ in range(rings):
            nxt: set[int] = set()
            for x in frontier:
                nxt |= adjacency.get(x, set())
            nxt -= seen
            seen |= nxt
            frontier = nxt
        windows.append(seen)
    return windows


def vertex_curvature_targets(mesh: Mesh, smoothing: int = 2) -> np.ndarray:
    """The degree excess each vertex's own curvature asks for.

    Discrete Gauss-Bonnet, read locally. The angle deficit
    ``K(v) = 2*pi - sum(theta)`` is the discrete Gaussian curvature at a
    vertex, and it is **geometric rather than combinatorial**: a degree-5
    vertex on a flat sheet has five 72 deg angles summing to exactly
    2*pi, so its deficit is zero. That is what makes it usable as a
    target -- it says what the *surface* is doing, not what the mesh
    happens to be doing.

    Over a region, ``sum(6 - deg) = (3/pi) * integral K dA``, so the share
    belonging to one vertex is ``3 * K(v) / pi``. Summed over a closed
    mesh that returns ``6 * chi`` exactly, which is the budget the rest of
    this package checks against -- so a mesh that matched its targets
    everywhere would satisfy the Euler check automatically.

    The raw per-vertex deficit is noisy on a remeshed surface, so it is
    averaged over ``smoothing`` rings of neighbours first. The point is
    the curvature of the surface, not of one triangle fan.

    Parameters
    ----------
    mesh
        Triangulation. Only positions are read; connectivity is used for
        the angle sums and for the smoothing neighbourhood.
    smoothing
        Neighbour shells to average over. ``0`` leaves the raw deficit.

    Returns
    -------
    numpy.ndarray
        One float per vertex: the *fractional* degree excess its
        curvature justifies. Near zero on a cylinder or a flat sheet,
        positive on a cap, negative in a neck.
    """
    verts, faces = mesh
    deficit = np.full(len(verts), 2.0 * np.pi)
    for tri in faces:
        a, b, c = (verts[int(tri[i])] for i in range(3))
        for i, (p0, p1, p2) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
            e1, e2 = p1 - p0, p2 - p0
            n1 = np.linalg.norm(e1) * np.linalg.norm(e2)
            if n1 <= 0.0:
                continue
            cosine = float(np.clip(np.dot(e1, e2) / n1, -1.0, 1.0))
            deficit[int(tri[i])] -= np.arccos(cosine)

    if smoothing > 0:
        nbrs = _adjacency(faces)
        for _ in range(smoothing):
            smoothed = deficit.copy()
            for v, ring in nbrs.items():
                if ring:
                    smoothed[v] = (deficit[v] + sum(deficit[w] for w in ring)
                                   ) / (1 + len(ring))
            deficit = smoothed
    return 3.0 * deficit / np.pi


def anneal_edge_flips(
    mesh: Mesh,
    rng: np.random.Generator,
    sweeps: int = 80,
    temperature: float = 0.3,
    min_degree: int = 5,
    max_degree: int = 8,
    restarts: int = 1,
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
        across them; 0 disables annealing and leaves the as-remeshed
        defect population intact. More sweeps do not move the median much
        -- 80 and 200 both sit at 13 pairs on a Y junction -- but they
        lengthen the good tail, which is what ``restarts`` samples: the
        best of twelve was 12 at 80 sweeps and 8 at 200.
    temperature
        Starting temperature. Measured optimum is ~0.3: colder barely
        escapes the minimum, hotter (>=0.6) wanders and ends up worse.
    restarts
        Independent anneals from the same mesh, keeping whichever ends
        with the fewest dislocations. Annealing is stochastic and its
        spread is wide enough to be worth sampling: twelve runs of 200
        sweeps on one Y-junction mesh gave 8 to 15 pairs. One draw from
        that is a coin toss; the best of a few is not, and each costs
        under two seconds against a ten-second build.

    Returns
    -------
    (vertices, triangles)
        Same vertices, re-flipped triangles.
    """
    if sweeps <= 0 or restarts < 1:
        return mesh
    best = None
    best_pairs = None
    for _ in range(max(1, restarts)):
        candidate = _anneal_once(mesh, rng, sweeps, temperature,
                                 min_degree, max_degree)
        pairs = dislocation_pairs(candidate)
        if best_pairs is None or pairs < best_pairs:
            best, best_pairs = candidate, pairs
    return best


def _anneal_once(
    mesh: Mesh,
    rng: np.random.Generator,
    sweeps: int,
    temperature: float,
    min_degree: int,
    max_degree: int,
) -> Mesh:
    """One Metropolis run. See :func:`anneal_edge_flips` for the reasoning."""
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
                degree[u] - 1, degree[v] - 1,
                degree[w1] + 1, degree[w2] + 1,
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
