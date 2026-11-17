"""A carbon anemone: tubes radiating from one core, as ONE closed surface.

The sea-anemone / urchin morphology of catalytically grown carbon: a
core, arms of unequal length leaving it in every direction, side
branches on some of them, and a bulb at each arm's tip where the
catalyst particle sits. The earlier model of it (``modelos_SEM``) was a
*scene* -- finished capped tubes placed round a point, touching nothing
-- which is honest as an illustration and wrong in the one place the
micrographs are clear: the arms share a wall where they meet.

Here the whole thing is a **tree**, and a tree thickened into a surface
is a sphere however many arms it has: ``V - E = 1``, so the ring budget
is ``12 * (V - E) = +12`` exactly, the same identity
:mod:`~nanocarbon_lab.builders.beads` and
:mod:`~nanocarbon_lab.builders.supernetwork` check against. The field is
the one they use -- :func:`implicit.graph_field` with one radius per
segment -- so a bulb is a zero-length segment (a sphere), an arm is a
segment, and the smooth union flares each arm into the core and into its
bulb instead of meeting them at a crease.

Three things are refused rather than returned, each because the result
would be a sound surface of the wrong shape:

* **A neck over three quarters of its bulb** gives a lozenge, not a
  bulb on a stem (the bead chain's rule).
* **An arm too short to leave a waist** between the core and its bulb.
* **Two non-adjacent parts closer than their radii plus the blend**:
  the union would merge them, which adds a handle, and the census would
  read a torus's budget. Branches are therefore drawn by rejection, and
  the independent census check at the end is what catches anything the
  geometric test missed.

The wall is the implicit route's: sound, and amorphous in the sense that
file's CLAUDE.md records -- 5-7 pairs wherever the curvature asks for
them, not a rolled lattice. The tube radius has the same floor as every
meshed tube here, about 3 A.

The bulb centres and radii are recorded in ``atoms.info`` so that a
scene can seat a particle INSIDE each bulb against a measured clearance
rather than by eye.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.rng import make_rng
from . import implicit as im
from . import remesh as rm
from .junction import MAX_GRID_RESOLUTION, _finish

__all__ = ["MAX_NECK_FRACTION", "MIN_TUBE_RADIUS", "anemone_graph",
           "build_anemone"]

#: A neck wider than this fraction of its bulb merges with it.
MAX_NECK_FRACTION = 0.75

#: The meshed route's own floor for a tube, in A: under eight rings fit
#: round anything narrower and the wall pinches.
MIN_TUBE_RADIUS = 3.0

#: Free tube that must remain between the core and a bulb, in A.
MIN_WAIST = 4.0

#: Attempts at seating one branch before giving it up.
BRANCH_TRIES = 200


def _segment_distance(p1: np.ndarray, q1: np.ndarray,
                      p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between two segments (either may be a point).

    Sampled rather than solved: the segments here are tens of A long
    and the answer feeds a clearance of several A, so a hundred points
    a side is ample and has no degenerate cases to get wrong.
    """
    t = np.linspace(0.0, 1.0, 100)[:, None]
    a = p1 + t * (q1 - p1)
    b = p2 + t * (q2 - p2)
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min())


def anemone_graph(
    arms: int = 8,
    arm_length: tuple[float, float] = (22.0, 38.0),
    branches: int = 3,
    branch_angle: float = 50.0,
    branch_fraction: tuple[float, float] = (0.4, 0.6),
    core_radius: float = 6.0,
    tube_radius: float = 3.5,
    bulb_radius: float = 7.0,
    blend: float = 2.5,
    seed: int | None = 0,
) -> dict:
    """The skeleton: segments, radii and the bulb centres.

    Arms leave the origin along Fibonacci directions -- evenly spread,
    which a random draw at this count is not -- with lengths drawn
    uniformly from ``arm_length``. A branch leaves an arm part-way along
    it, ``branch_angle`` degrees off the arm's own direction, and is
    ``branch_fraction`` of that arm's length. Branches carry no bulb.

    Returns
    -------
    dict
        ``segments`` (N, 2, 3), ``radii`` (N,), ``bulbs`` (arms, 3),
        ``n_vertices``, ``n_edges`` and ``n_branches`` actually placed
        -- which can fall short of the request, because a branch that
        would touch its neighbours is dropped rather than forced.
    """
    rng = make_rng(seed)
    index = np.arange(arms, dtype=float)
    height = 1.0 - 2.0 * (index + 0.5) / arms
    radial = np.sqrt(np.maximum(0.0, 1.0 - height**2))
    angle = index * np.pi * (3.0 - np.sqrt(5.0))
    directions = np.column_stack((radial * np.cos(angle),
                                  radial * np.sin(angle), height))
    lengths = rng.uniform(arm_length[0], arm_length[1], size=arms)
    tips = directions * lengths[:, None]

    origin = np.zeros(3)
    # (start, end, radius, owner): `owner` is the arm a part belongs to,
    # so the clearance test can skip parts that are MEANT to touch.
    parts: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    for arm in range(arms):
        parts.append((origin, tips[arm], tube_radius, arm))
        parts.append((tips[arm], tips[arm], bulb_radius, arm))

    placed = 0
    for _ in range(branches):
        for _try in range(BRANCH_TRIES):
            arm = int(rng.integers(arms))
            along = rng.uniform(0.35, 0.6)
            root = tips[arm] * along
            # A unit vector perpendicular to the arm, at a random
            # azimuth round it.
            trial = rng.normal(size=3)
            perpendicular = trial - (trial @ directions[arm]) * directions[arm]
            perpendicular /= np.linalg.norm(perpendicular)
            theta = np.radians(branch_angle)
            heading = (np.cos(theta) * directions[arm]
                       + np.sin(theta) * perpendicular)
            reach = lengths[arm] * rng.uniform(*branch_fraction)
            end = root + heading * reach
            clear = True
            for start, stop, radius, owner in parts:
                if owner == arm and radius == tube_radius:
                    continue            # its own arm: they join there
                gap = _segment_distance(root + heading * (tube_radius + blend),
                                        end, start, stop)
                if gap < radius + tube_radius + blend + 1.0:
                    clear = False
                    break
            if clear:
                parts.append((root, end, tube_radius, -1 - placed))
                placed += 1
                break

    segments = np.stack([np.stack([start, stop])
                         for start, stop, _, _ in parts])
    radii = np.array([radius for _, _, radius, _ in parts])
    # The core is a sphere at the origin: one more zero-length segment.
    segments = np.concatenate([segments, np.zeros((1, 2, 3))])
    radii = np.append(radii, core_radius)
    # A tree: the core, one tip per arm, and for each branch its root
    # (splitting an arm adds a vertex AND an edge) and its end.
    vertices = 1 + arms + 2 * placed
    edges = arms + 2 * placed
    return {"segments": segments, "radii": radii, "bulbs": tips,
            "n_vertices": vertices, "n_edges": edges, "n_branches": placed}


def build_anemone(
    arms: int = 8,
    arm_length: tuple[float, float] = (22.0, 38.0),
    branches: int = 3,
    branch_angle: float = 50.0,
    core_radius: float = 6.0,
    tube_radius: float = 3.5,
    bulb_radius: float = 7.0,
    blend: float = 2.5,
    bond: float = CC_BOND,
    grid_resolution: int = 72,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 0,
    place_curvature: bool = True,
    wall_anchor: float = 0.0,
    relax_iterations: int = 3000,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
) -> Atoms:
    """Build a carbon anemone as one closed graphitic surface.

    Parameters
    ----------
    arms
        Arms leaving the core. Each ends in a bulb.
    arm_length
        ``(shortest, longest)`` core-to-bulb-centre distance in A; each
        arm's length is drawn uniformly between them.
    branches
        Side branches to attempt. One that would touch a neighbour is
        dropped, so ``info["n_branches"]`` is the count to quote.
    branch_angle
        Degrees between a branch and the arm it leaves.
    core_radius, tube_radius, bulb_radius
        Radii in A of the central sphere, of every arm and branch, and
        of the bulb at each arm's tip. They are requests: the wall
        relaxes, and a meshed tube does not hold its radius exactly.
    blend
        Smooth-union radius where the parts meet.
    place_curvature
        On, as for the bead chain: the bulbs and the core are convex,
        so pentagons belong on them.
    seed
        For the arm lengths and the branch draw.

    Returns
    -------
    ase.Atoms
        Carbon only, non-periodic. ``info`` carries the census keys plus
        ``bulb_centres`` (arms, 3) and ``bulb_radius`` in the returned
        frame, ``n_branches`` and ``ring_budget``.

    Raises
    ------
    ValueError
        For a tube under the meshing floor, a neck too wide for its
        bulb, or an arm too short to leave a waist.
    RuntimeError
        If the surface closes around a different shape than the tree.
    """
    if arms < 2:
        raise ValueError("an anemone needs at least two arms.")
    if tube_radius < MIN_TUBE_RADIUS:
        raise ValueError(
            f"tube_radius {tube_radius:.2f} A is under the meshed route's "
            f"floor of {MIN_TUBE_RADIUS:.1f} A: under eight rings fit round "
            "the tube and the wall pinches shut.")
    if tube_radius > MAX_NECK_FRACTION * bulb_radius:
        raise ValueError(
            f"tube_radius {tube_radius:.2f} A is more than "
            f"{MAX_NECK_FRACTION:.0%} of bulb_radius {bulb_radius:.2f} A, "
            "so the tip comes out as a lozenge rather than a bulb on a "
            "stem. Narrow the tube or widen the bulb.")
    shortest = core_radius + bulb_radius + 2.0 * blend + MIN_WAIST
    if min(arm_length) < shortest:
        raise ValueError(
            f"the shortest arm, {min(arm_length):.1f} A, leaves no waist "
            f"between a {core_radius:.1f} A core and a {bulb_radius:.1f} A "
            f"bulb at blend {blend:.1f}: it needs {shortest:.1f} A.")

    graph = anemone_graph(arms=arms, arm_length=arm_length,
                          branches=branches, branch_angle=branch_angle,
                          core_radius=core_radius, tube_radius=tube_radius,
                          bulb_radius=bulb_radius, blend=blend, seed=seed)
    segments, radii = graph["segments"], graph["radii"]
    budget = 12 * (graph["n_vertices"] - graph["n_edges"])

    extent = float(np.abs(segments).max() + bulb_radius + blend + 2.0)
    box = np.array([2.0 * extent] * 3)
    field = im.graph_field(segments, box, (False, False, False),
                           tube_radius=radii, blend=blend)

    # Track the voxel to the tube, the narrowest feature -- the bead
    # chain's rule, for the bead chain's reason.
    voxel_target = min(1.4, 0.25 * tube_radius)
    needed = int(np.ceil(2.0 * extent / voxel_target))
    resolution = int(np.clip(needed, grid_resolution, MAX_GRID_RESOLUTION))
    mesh = rm.marching_cubes_mesh(field, extent, resolution=resolution)
    rng = make_rng(seed)
    mesh = rm.isotropic_remesh(
        mesh, field, target_edge=np.sqrt(3.0) * bond,
        iterations=remesh_iterations, anneal_sweeps=anneal_sweeps,
        place_curvature=place_curvature, rng=rng,
    )
    # `_finish` recentres the structure in its vacuum box, so the bulb
    # centres have to be carried through the same shift. It is measured
    # off one bulb's own atoms rather than assumed.
    atoms = _finish(
        mesh, field=field, wall_anchor=wall_anchor, bond=bond,
        relax_iterations=relax_iterations, vacuum=vacuum, roughness=0.0,
        rng=rng,
        info={
            "structure_type": "anemone",
            "arms": int(arms),
            "n_branches": int(graph["n_branches"]),
            "ring_budget": int(budget),
            "core_radius": core_radius,
            "tube_radius": tube_radius,
            "bulb_radius": bulb_radius,
            "blend": blend,
            "bond": bond,
        },
    )

    census = atoms.info.get("ring_counts", {})
    measured = sum((6 - size) * number for size, number in census.items())
    if measured != budget:
        raise RuntimeError(
            f"An anemone is a tree, so its ring budget is {budget:+d}; the "
            f"surface that came out carries {measured:+d} "
            f"({dict(sorted(census.items()))}). Two parts merged that the "
            "skeleton keeps apart -- shorten the branches, or use fewer.")

    # Where the bulbs ended up. The mesh frame has the core at the
    # origin and `_finish` recentres the structure in its vacuum box, so
    # the finished frame is the mesh frame plus one rigid shift. The
    # shift is taken between CENTROIDS -- the atoms are the mesh's
    # triangle centroids and a relaxation driven by internal forces
    # leaves their mean where it was. (The two bounding boxes were tried
    # first and are wrong by several A, because one stray atom at the
    # end of the longest arm sets a box and not a mean; three of eight
    # bulbs then "measured" a free radius under 5 A that was really the
    # distance to their own stem.)
    vertices = np.asarray(mesh[0], dtype=float)
    faces = np.asarray(mesh[1], dtype=int)
    positions = np.asarray(atoms.positions, dtype=float)
    shift = positions.mean(axis=0) - vertices[faces].mean(axis=1).mean(axis=0)
    centres, measured_radii = [], []
    for guess in graph["bulbs"] + shift:
        centre = guess
        # Each centre is then MEASURED, coarse to fine. The tip of a long
        # thin arm drifts by several A as the wall relaxes, so a capture
        # sized to the bulb alone can miss it outright -- one bulb of
        # eight came back with no atoms round its guess at all. The
        # capture therefore starts wide (where the centroid of what it
        # holds pulls the centre toward the bulb, the densest thing
        # there) and narrows to the bulb, where a least-squares sphere
        # takes over: |x|^2 = 2 c.x + (r^2 - |c|^2) is linear in c, and
        # unlike a centroid it is not pulled toward the missing patch
        # where the stem enters.
        for capture in (2.4, 1.8, 1.4, 1.2, 1.15):
            near = positions[np.linalg.norm(positions - centre, axis=1)
                             < capture * bulb_radius]
            if len(near) < 10:
                continue
            if capture > 1.3:
                centre = near.mean(axis=0)
                continue
            design = np.column_stack((2.0 * near, np.ones(len(near))))
            solution = np.linalg.lstsq(design, (near**2).sum(axis=1),
                                       rcond=None)[0]
            fitted = solution[:3]
            radius = float(np.sqrt(max(solution[3] + fitted @ fitted, 0.0)))
            # A fit through a patch that is not a sphere flies off; it
            # is accepted only while it still describes this bulb.
            if (np.linalg.norm(fitted - centre) < 0.5 * bulb_radius
                    and 0.5 * bulb_radius < radius < 1.4 * bulb_radius):
                centre = fitted
        distance = np.linalg.norm(positions - centre, axis=1)
        shell = distance[distance < 1.3 * bulb_radius]
        centres.append(centre.tolist())
        # The minimum, not the mean: what a particle must clear is the
        # wall's NEAREST approach to the centre.
        measured_radii.append(float(shell.min()) if shell.size else 0.0)
    atoms.info["bulb_centres"] = centres
    atoms.info["bulb_radii"] = measured_radii
    atoms.info["core_centre"] = shift.tolist()
    return atoms
