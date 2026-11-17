"""A capped nanotube with branches FUSED into its wall.

A branch laid against a tube at van der Waals contact is two structures
and a scene. A branch whose wall continues into the tube's is one
structure -- a Y junction -- and this module builds that one, on the
lattice, the way :mod:`fullerene_mesh` says a ring edit must be done:
as a combinatorial operation on the triangulated dual, never by
deciding from distances which atoms ought to be bonded.

**The weld.** In the dual, a tube's barrel is a triangular lattice and a
circumference of the branch is a closed lattice row of ``L = 5 * freq``
vertices. Cut the branch along one such row and throw its base cap
away: every vertex on the rim keeps the three triangles on the tip
side. Cut out of the trunk's barrel a convex lattice HEXAGON whose
perimeter is the same ``L``: on the trunk side a vertex on a straight
side of that hexagon keeps three triangles and a vertex at one of its
six corners keeps four, because the trunk turns through 240 deg round a
corner and 180 deg along a side. Sew rim to hexagon vertex for vertex.
A side vertex now has ``3 + 3 = 6`` triangles and a corner has
``4 + 3 = 7``:

* the junction carries **exactly six heptagons**, at the six corners,
  which is the number a three-arm node needs (``6 * (2 - 3) = -6``) and
  the number published Y junctions carry;
* nothing else changes degree, so the barrel stays all-hexagon right up
  to the crotch;
* the budget closes by itself. Two capped tubes carry ``+12`` each; the
  discarded base cap takes six pentagons with it and the weld adds six
  heptagons, so the fused structure is back at ``+12`` -- a sphere,
  which a tree of tubes is. The builder checks that and the heptagon
  count separately, because the first cannot see a weld that put its
  heptagons somewhere else.

The hexagon need not be regular -- only closed, which for side lengths
``(a, b, c, d, e, f)`` means ``a + b = d + e`` and ``b + c = e + f`` --
so any ``L >= 6`` has one, and the most nearly regular is used.

**The geometry** comes after the topology, as everywhere in this
package: the branch's mesh is stood on the hole, the mesh is smoothed
round the seam, the dual is taken and the same valence force field that
relaxes a plain capped tube relaxes this one. A crotch is strained --
its heptagons are 128.6 deg rings before any strain -- so the bond
window here is the junction's, not the barrel's, and the measured
range is recorded rather than assumed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.rng import make_rng
from . import centerline as cl
from . import fullerene_mesh as fm

__all__ = ["build_branched_cnt", "hexagon_sides", "weld_branch"]

Mesh = tuple[np.ndarray, np.ndarray]


def hexagon_sides(perimeter: int) -> tuple[int, ...]:
    """Side lengths of the most regular closed lattice hexagon.

    A convex hexagon on a triangular lattice with sides
    ``(a, b, c, d, e, f)`` closes iff ``a + b = d + e`` and
    ``b + c = e + f``. Among those with the given perimeter and every
    side at least 1, this returns the one whose sides are most nearly
    equal -- a regular hexagon when the perimeter is a multiple of six.
    """
    if perimeter < 6:
        raise ValueError("a lattice hexagon has a perimeter of at least 6.")
    best: tuple[float, tuple[int, ...]] | None = None
    top = perimeter - 4
    for a in range(1, top):
        for b in range(1, top):
            for c in range(1, top):
                for d in range(1, top):
                    e = a + b - d
                    f = b + c - e
                    if e < 1 or f < 1 or a + b + c + d + e + f != perimeter:
                        continue
                    sides = (a, b, c, d, e, f)
                    spread = float(np.var(sides))
                    if best is None or spread < best[0]:
                        best = (spread, sides)
    if best is None:
        raise ValueError(f"no closed lattice hexagon has perimeter "
                         f"{perimeter}.")
    return best[1]


def _ordered_neighbours(vertices: np.ndarray, faces: np.ndarray,
                        normals: np.ndarray) -> dict[int, list[int]]:
    """Each vertex's neighbours, counter-clockwise seen from outside."""
    adjacency = fm.mesh_adjacency(faces)
    ordered: dict[int, list[int]] = {}
    for vertex, around in adjacency.items():
        normal = normals[vertex]
        helper = np.array([1.0, 0.0, 0.0])
        if abs(normal @ helper) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(normal, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        ring = sorted(around, key=lambda other: np.arctan2(
            (vertices[other] - vertices[vertex]) @ e2,
            (vertices[other] - vertices[vertex]) @ e1))
        ordered[vertex] = ring
    return ordered


def _walk(ordered: dict[int, list[int]], start: int, first: int,
          sides: tuple[int, ...] | None, steps: int) -> list[int] | None:
    """Trace a closed lattice loop; ``None`` if it does not close.

    With `sides`, a convex hexagon turning LEFT by 60 deg at each corner
    (so its interior is on the left). Without, a straight line of
    `steps` edges, which on a tube's barrel is a circumference.
    """
    turns: set[int] = set()
    if sides is not None:
        total = 0
        for side in sides:
            total += side
            turns.add(total)
    loop = [start]
    previous, current = start, first
    for step in range(1, steps + 1):
        if step == steps:
            return loop if current == start else None
        if current in loop or len(ordered[current]) != 6:
            return None
        loop.append(current)
        ring = ordered[current]
        back = ring.index(previous)
        # Arriving from `previous`: straight on is three places round,
        # a 60 deg left turn is four.
        ahead = ring[(back + (4 if step in turns else 3)) % 6]
        previous, current = current, ahead
    return None


def _side_faces(faces: np.ndarray, loop: list[int],
                ordered: dict[int, list[int]], left: bool) -> np.ndarray:
    """Mask of the faces on one side of a closed loop, by flood fill."""
    count = len(loop)
    barrier = {frozenset((loop[k], loop[(k + 1) % count]))
               for k in range(count)}
    by_edge: dict[frozenset, list[int]] = defaultdict(list)
    for index, face in enumerate(faces):
        for k in range(3):
            by_edge[frozenset((int(face[k]), int(face[(k + 1) % 3])))].append(
                index)
    # The face on the wanted side of the first loop edge u -> v: its
    # third vertex sits one place round from u at v, anticlockwise for
    # the left side and clockwise for the right.
    u, v = loop[0], loop[1]
    ring = ordered[v]
    third = ring[(ring.index(u) + (5 if left else 1)) % 6]
    wanted = {u, v, third}
    seed = next(index for index in by_edge[frozenset((u, v))]
                if set(int(x) for x in faces[index]) == wanted)
    inside = np.zeros(len(faces), dtype=bool)
    stack = [seed]
    inside[seed] = True
    while stack:
        face = faces[stack.pop()]
        for k in range(3):
            edge = frozenset((int(face[k]), int(face[(k + 1) % 3])))
            if edge in barrier:
                continue
            for other in by_edge[edge]:
                if not inside[other]:
                    inside[other] = True
                    stack.append(other)
    return inside


def weld_branch(
    trunk: Mesh,
    trunk_normals: np.ndarray,
    seat: int,
    branch: Mesh,
    branch_normals: np.ndarray,
    tilt: float = 0.0,
    lean: np.ndarray | None = None,
) -> tuple[Mesh, np.ndarray, np.ndarray]:
    """Fuse `branch` into `trunk`'s wall near vertex `seat`.

    Both meshes are in real units, the branch in its own frame with its
    axis along ``+z`` and its tip at positive ``z``. `tilt` (degrees)
    leans the branch off the trunk's normal toward unit vector `lean`.

    Returns
    -------
    (mesh, normals, corners)
        The fused mesh, an outward normal per vertex, and the indices
        of the six corner vertices -- the junction's heptagons.

    Raises
    ------
    ValueError
        If no closed hexagon of the branch's circumference can be cut
        round `seat` -- it runs into a cap, or into an earlier weld.
    """
    t_vertices, t_faces = (np.asarray(trunk[0], dtype=float),
                           np.asarray(trunk[1], dtype=int))
    b_vertices, b_faces = (np.asarray(branch[0], dtype=float),
                           np.asarray(branch[1], dtype=int))
    t_order = _ordered_neighbours(t_vertices, t_faces, trunk_normals)
    b_order = _ordered_neighbours(b_vertices, b_faces, branch_normals)

    # --- the branch's rim: the lowest full circumference row.
    degree = {v: len(n) for v, n in b_order.items()}
    radius = float(np.hypot(b_vertices[:, 0], b_vertices[:, 1]).max())
    low = float(b_vertices[:, 2].min())
    rim = None
    for start in np.argsort(b_vertices[:, 2]):
        start = int(start)
        if degree[start] != 6 or b_vertices[start, 2] < low + 0.9 * radius:
            continue
        # Circumferential, anticlockwise seen from the tip.
        azimuth = np.cross(np.array([0.0, 0.0, 1.0]), b_vertices[start])
        azimuth /= np.linalg.norm(azimuth)
        first = max(b_order[start], key=lambda other: (
            (b_vertices[other] - b_vertices[start]) @ azimuth
            - 3.0 * abs(b_vertices[other][2] - b_vertices[start][2])))
        for steps in range(6, 400):
            rim = _walk(b_order, start, first, None, steps)
            if rim is not None:
                break
        if rim is not None:
            break
    if rim is None:
        raise ValueError("the branch has no closed circumference to cut.")
    perimeter = len(rim)
    base = _side_faces(b_faces, rim, b_order, left=False)
    b_keep = b_faces[~base]

    # --- the trunk's hole: a hexagon of that perimeter round the seat.
    sides = hexagon_sides(perimeter)
    hole = None
    for start in [seat, *t_order[seat]]:
        if len(t_order[start]) != 6:
            continue
        for first in t_order[start]:
            for shift in range(6):
                rolled = sides[shift:] + sides[:shift]
                loop = _walk(t_order, start, first, rolled, perimeter)
                if loop is not None:
                    hole = loop
                    break
            if hole is not None:
                break
        if hole is not None:
            break
    if hole is None:
        raise ValueError(
            f"no closed lattice hexagon of perimeter {perimeter} fits "
            f"round vertex {seat}: it meets a cap or an earlier weld.")
    removed = _side_faces(t_faces, hole, t_order, left=True)
    t_keep = t_faces[~removed]
    # A corner is where the walk turned: the vertex reached after each
    # side but the last, plus the start, which closes the sixth.
    corner_steps = np.cumsum(sides)[:-1]
    corners_old = [hole[0]] + [hole[int(step)] for step in corner_steps]

    # --- stand the branch on the hole.
    centre = t_vertices[hole].mean(axis=0)
    normal = trunk_normals[hole].mean(axis=0)
    normal /= np.linalg.norm(normal)
    heading = normal
    if tilt and lean is not None:
        sideways = lean - (lean @ normal) * normal
        if np.linalg.norm(sideways) > 1e-9:
            sideways /= np.linalg.norm(sideways)
            angle = np.radians(tilt)
            heading = np.cos(angle) * normal + np.sin(angle) * sideways
    helper = np.array([1.0, 0.0, 0.0])
    if abs(normal @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    rim_height = float(b_vertices[rim, 2].mean())
    rise = b_vertices[:, 2] - rim_height
    # Cross-sections stay parallel to the hole and slide along the
    # heading: a sheared tube, which the relaxation rounds.
    placed = (centre + np.outer(b_vertices[:, 0], e1)
              + np.outer(b_vertices[:, 1], e2)
              + np.outer(rise / max(heading @ normal, 0.3), heading))
    placed_normals = (np.outer(branch_normals[:, 0], e1)
                      + np.outer(branch_normals[:, 1], e2)
                      + np.outer(branch_normals[:, 2], normal))

    # --- sew: the rotation of the rim that lies closest to the hole.
    best_offset, best_cost = 0, np.inf
    for offset in range(perimeter):
        cost = float(sum(
            np.sum((t_vertices[hole[k]]
                    - placed[rim[(k + offset) % perimeter]]) ** 2)
            for k in range(perimeter)))
        if cost < best_cost:
            best_offset, best_cost = offset, cost
    n_trunk = len(t_vertices)
    remap = np.arange(len(b_vertices)) + n_trunk
    for k in range(perimeter):
        remap[rim[(k + best_offset) % perimeter]] = hole[k]
    vertices = np.vstack((t_vertices, placed))
    normals = np.vstack((trunk_normals, placed_normals))
    faces = np.vstack((t_keep, remap[b_keep]))

    # --- drop what nothing refers to, and renumber.
    used = np.zeros(len(vertices), dtype=bool)
    used[faces.ravel()] = True
    renumber = -np.ones(len(vertices), dtype=int)
    renumber[used] = np.arange(int(used.sum()))
    vertices, normals = vertices[used], normals[used]
    faces = renumber[faces]
    corners = renumber[np.asarray(corners_old)]

    # --- smooth the seam, so the dual does not start from a crease.
    adjacency = fm.mesh_adjacency(faces)
    seam = set(int(renumber[v]) for v in hole)
    region = set(seam)
    frontier = set(seam)
    for _ in range(3):
        frontier = {n for v in frontier for n in adjacency[v]} - region
        region |= frontier
    region_list = sorted(region)
    for _ in range(25):
        updated = vertices.copy()
        for vertex in region_list:
            around = list(adjacency[vertex])
            updated[vertex] = (0.5 * vertices[vertex]
                               + 0.5 * vertices[around].mean(axis=0))
        vertices = updated
    return (vertices, faces), normals, corners


def _capsule(freq: int, rings: int, bond: float,
             smoothing: int) -> tuple[Mesh, np.ndarray, float]:
    """A capped tube's dual mesh in angstrom, its normals and its radius."""
    mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(rings), freq)
    half = (rings - 1) / 2.0
    vertices, faces = fm.smooth_mesh_on_capsule(
        mesh, radius=1.0, half_length=half, iterations=smoothing)
    vertices = np.asarray(vertices, dtype=float)
    axis_point = np.zeros_like(vertices)
    axis_point[:, 2] = np.clip(vertices[:, 2], -half, half)
    normals = vertices - axis_point
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    radius = float(fm.radius_for_freq(freq, bond))
    return (vertices * radius, np.asarray(faces, dtype=int)), normals, radius


def build_branched_cnt(
    freq: int = 3,
    n_body_rings: int = 14,
    branches: int = 2,
    branch_freq: int = 2,
    branch_rings: tuple[int, int] = (4, 8),
    branch_tilt: float = 30.0,
    shape: cl.Shape = "arc",
    waviness: float = 0.4,
    bond: float = CC_BOND,
    smoothing_iterations: int = 60,
    relax_iterations: int = 3000,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
) -> Atoms:
    """Build a capped nanotube with branches fused into its wall.

    Parameters
    ----------
    freq, n_body_rings
        The trunk, exactly as :func:`build_capped_cnt` takes them: the
        radius is ``1.96 * freq`` A and is an output of the lattice.
    branches
        Branches to fuse. One that cannot be seated -- its hexagon would
        meet a cap or another weld -- is skipped, and
        ``info["n_branches"]`` is the count that went in.
    branch_freq, branch_rings
        The branches' lattice frequency (thinner than the trunk) and the
        range their length is drawn from, in body rings.
    branch_tilt
        Degrees a branch leans off the trunk's normal, toward the tip.
    shape, waviness
        The trunk's centreline, swept after the junctions are relaxed.

    Returns
    -------
    ase.Atoms
        One carbon structure. ``info`` carries ``ring_counts``,
        ``n_branches``, ``heptagons`` (six per branch), the measured
        ``geometry`` (bond range and angle range), ``radius``, and
        ``normals`` -- an outward unit vector per atom, for anything
        that has to be seated on the wall -- and ``part``, 0 for a
        trunk atom and 1, 2, ... for the atoms of each branch.

    Raises
    ------
    RuntimeError
        If the fused mesh does not carry the census the weld fixes.
    """
    if branch_freq > freq:
        raise ValueError("a branch may not be wider than its trunk.")
    rng = make_rng(seed)
    mesh, normals, radius = _capsule(freq, n_body_rings, bond,
                                     smoothing_iterations)
    half = float(np.abs(mesh[0][:, 2]).max()) - radius
    welded = 0
    taken: list[np.ndarray] = []
    for _ in range(branches):
        rings = int(rng.integers(branch_rings[0], branch_rings[1] + 1))
        b_mesh, b_normals, b_radius = _capsule(branch_freq, rings, bond,
                                               smoothing_iterations)
        for _try in range(40):
            vertices = mesh[0]
            # On the barrel, clear of both caps and of earlier welds.
            eligible = np.flatnonzero(
                (np.abs(vertices[:, 2]) < half - 2.0 * b_radius - 3.0)
                & (vertices[:, 2] > -half + 0.25 * 2.0 * half)
                & (np.hypot(vertices[:, 0], vertices[:, 1]) < 1.1 * radius))
            if not len(eligible):
                break
            seat = int(rng.choice(eligible))
            if any(np.linalg.norm(vertices[seat] - other)
                   < 4.0 * b_radius + 6.0 for other in taken):
                continue
            try:
                mesh, normals, _corners = weld_branch(
                    mesh, normals, seat, b_mesh, b_normals,
                    tilt=branch_tilt, lean=np.array([0.0, 0.0, 1.0]))
            except (ValueError, StopIteration):
                continue
            taken.append(vertices[seat].copy())
            welded += 1
            break

    positions, bond_set, rings_list = fm.dual_honeycomb(mesh)
    census = fm.ring_size_histogram(rings_list)
    deficit = sum((6 - size) * count for size, count in census.items())
    if deficit != 12 or census.get(7, 0) != 6 * welded:
        raise RuntimeError(
            f"the fused mesh carries {dict(sorted(census.items()))} "
            f"(deficit {deficit:+d}); {welded} welded branches fix it at "
            f"+12 with exactly {6 * welded} heptagons.")

    bonds = sorted(bond_set)
    lengths = np.array([np.linalg.norm(positions[a] - positions[b])
                        for a, b in bonds])
    positions = positions * (bond / np.median(lengths))
    positions = fm.relax_shell(positions, bond_set, equilibrium=bond,
                               max_iterations=relax_iterations)

    # --- which atoms are trunk and which belong to which branch, read
    # while the trunk is still straight along z: a branch atom is one
    # standing clear of the trunk's cylinder, and the branches are the
    # connected pieces of those.
    part = np.zeros(len(positions), dtype=int)
    around: list[list[int]] = [[] for _ in range(len(positions))]
    for a, b in bonds:
        around[a].append(b)
        around[b].append(a)
    away = np.hypot(positions[:, 0] - np.median(positions[:, 0]),
                    positions[:, 1] - np.median(positions[:, 1]))
    outside = away > radius + 1.5
    visited = np.zeros(len(positions), dtype=bool)
    label = 0
    for start in np.flatnonzero(outside):
        if visited[start]:
            continue
        visited[start] = True
        piece, stack = [int(start)], [int(start)]
        while stack:
            atom = stack.pop()
            for other in around[atom]:
                if outside[other] and not visited[other]:
                    visited[other] = True
                    piece.append(other)
                    stack.append(other)
        # A real branch stands a tube's width clear of the trunk; a
        # ripple of the relaxed wall does not, and stays trunk.
        if away[piece].max() > radius + 5.0:
            label += 1
            part[piece] = label

    if shape != "straight" and waviness > 0.0:
        centred = positions - positions.mean(axis=0)
        length = float(np.ptp(centred[:, 2]))
        control = cl.shape_control_points(shape, rng, n_points=9,
                                          amplitude=waviness, turns=1.5)
        control, _strain = cl.fit_to_strain_budget(
            control, length, radius, max_strain=cl.DEFAULT_MAX_STRAIN)
        positions = cl.sweep_along_path(positions, control)
        positions = fm.relax_shell(
            positions, bond_set, equilibrium=bond,
            anchors=np.arange(len(positions)), anchor_targets=positions,
            k_anchor=3.0, max_iterations=relax_iterations)

    # --- measure, and an outward normal per atom.
    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in bonds:
        neighbours[a].append(b)
        neighbours[b].append(a)
    lengths = np.array([np.linalg.norm(positions[a] - positions[b])
                        for a, b in bonds])
    atom_normals = np.zeros_like(positions)
    for atom, around in neighbours.items():
        p0, p1, p2 = (positions[around[0]], positions[around[1]],
                      positions[around[2]])
        normal = np.cross(p1 - p0, p2 - p0)
        atom_normals[atom] = normal / max(np.linalg.norm(normal), 1e-12)
    # One consistent sign, carried bond to bond from the tip, where
    # outward is unambiguous: along the axis.
    top = int(np.argmax(positions[:, 2]))
    if atom_normals[top] @ (positions[top] - positions.mean(axis=0)) < 0.0:
        atom_normals[top] *= -1.0
    seen = {top}
    stack = [top]
    while stack:
        atom = stack.pop()
        for other in neighbours[atom]:
            if other in seen:
                continue
            if atom_normals[other] @ atom_normals[atom] < 0.0:
                atom_normals[other] *= -1.0
            seen.add(other)
            stack.append(other)

    atoms = Atoms(f"C{len(positions)}", positions=positions)
    atoms.center(vacuum=vacuum)
    atoms.info.update({
        "structure_type": "branched_cnt",
        "ring_counts": dict(sorted(census.items())),
        "ring_deficit": int(deficit),
        "n_branches": int(welded),
        "heptagons": int(census.get(7, 0)),
        "radius": radius,
        "bonds": [list(pair) for pair in bonds],
        "normals": atom_normals.tolist(),
        "part": part.tolist(),
        "geometry": {"bond_min": float(lengths.min()),
                     "bond_max": float(lengths.max()),
                     "bond_mean": float(lengths.mean())},
    })
    return atoms
