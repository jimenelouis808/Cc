"""Nanotube junctions (L, T, Y, X) and schwarzite fragments.

Where :func:`nanocarbon_lab.builders.capped_cnt.build_capped_cnt` builds a
tube from a hand-made seed polyhedron, this builder starts from an
**implicit surface** and lets the topology follow from the geometry:

1. :mod:`nanocarbon_lab.builders.implicit` defines the shape as a scalar
   field (a smooth union of capsules for a junction; a triply periodic
   minimal surface for a schwarzite),
2. :mod:`nanocarbon_lab.builders.remesh` meshes its zero level set and
   remeshes it isotropically so vertex degrees cluster on 6,
3. the same dual/relax machinery as the tube builder turns that mesh into
   carbon.

The pay-off is that ring statistics are *derived*, never prescribed.
Nobody tells the code that a Y junction needs heptagons; the branch is a
saddle, saddles carry negative Gaussian curvature, and negative curvature
comes out of the remesher as degree-7 vertices, which the dual renders as
heptagons. Euler's theorem then holds automatically:
``sum(6 - ring_size) = 6 * chi``, which is 12 for anything closed and
sphere-like (a capped junction, however many arms) and drops by 12 per
handle for a schwarzite fragment.

That last point is why this module does not reuse the tube builder's
hardcoded "deficit must equal 12" check: a schwarzite is *supposed* to
violate it, and the correct budget is read from the mesh's own Euler
characteristic instead.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from ase import Atoms

from ..analyse.curvature import disclination_check
from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from ..utils.rng import make_rng
from . import fullerene_mesh as fm
from . import implicit as im
from . import remesh as rm
from .capped_cnt import geometry_report

# Marching-cubes grids cost resolution**3 samples; beyond this the memory
# and time stop being worth the extra detail for these shapes.
MAX_GRID_RESOLUTION = 170

# Alternations of (relax atoms, rescale cell) for periodic structures.
CELL_RELAX_CYCLES = 6

# Smallest cell each surface can be tiled with at graphitic ring size.
# The limit scales with genus, because a higher-genus surface packs more
# channels into the same volume and its necks get correspondingly finer:
# Schwarz D (genus 9) needs more room than Schwarz P (genus 3).
#
# These were raised after measuring rather than merely checking the cell
# produced *something*. The old limits (20 / 22 / 30 Å) let a cell through
# that met the tear gate below and still had bonds well outside the sp2
# range -- Schwarz P at 24 Å relaxes to 1.328-1.560 Å, Schwarz D at 30 Å
# to 1.343-1.586 Å, both "broken" by `validation.sp2_quality`. Sweeping
# cell size at `anneal_sweeps=0`:
#
#   surface     24 Å      30 Å      36 Å      44 Å
#   primitive   broken    strained  clean     strained
#   gyroid      broken    broken    strained  strained
#   diamond     --        broken    strained  strained
#
# so the minimum for each is the smallest cell that is not "broken".
MIN_SCHWARZITE_CELL = {"primitive": 30.0, "gyroid": 36.0, "diamond": 36.0}


def build_junction(
    kind: im.JunctionKind = "Y",
    tube_radius: float = 6.0,
    arm_length: float = 22.0,
    blend: float = 4.0,
    bond: float = CC_BOND,
    grid_resolution: int = 70,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 0,
    place_curvature: bool = False,
    wall_anchor: float = 0.0,
    roughness: float = 0.0,
    relax_iterations: int = 3000,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
) -> Atoms:
    """Build a capped multi-arm carbon nanotube junction.

    Parameters
    ----------
    kind
        ``"L"`` (elbow), ``"T"``, ``"Y"`` (120 deg), ``"X"``, or
        ``"cross3d"`` (six arms along +-x, +-y, +-z).
    tube_radius
        Arm radius in Å. Note this is a *geometric* radius imposed on the
        surface, unlike the tube builder where the radius is quantised by
        the lattice; the relaxation absorbs the small mismatch.
    arm_length
        Centre-to-tip distance of each arm, in Å.
    blend
        Smooth-union radius at the branch (Å). This sets how flared the
        neck is, and therefore how many heptagons it takes. Too small and
        the surface creases; too large and the junction becomes a blob.
    bond
        Target C-C bond length (Å).
    grid_resolution
        *Minimum* marching-cubes grid points per axis. The actual
        resolution scales with the bounding box so voxels stay near a
        quarter of the tube radius, then is capped at
        :data:`MAX_GRID_RESOLUTION`; the remesher sets the final ring size
        regardless.
    remesh_iterations
        Isotropic remeshing cycles; see
        :func:`nanocarbon_lab.builders.remesh.isotropic_remesh`.
    anneal_sweeps
        Metropolis flip-annealing passes, which remove pentagon-heptagon
        pairs beyond those curvature requires (measured 39 -> 14 on a Y
        junction). **Defaults to 0, as the schwarzite already did**, and
        for the same reason -- which was measured here only after the
        default had been 80 for a long time on the strength of the ring
        census alone.

        The census is not the thing to judge it by. Annealing does remove
        stray pairs, and the wall gets *wavier* for it. Measured as each
        atom's distance from its own arm's axis over the straight barrel,
        where a cylinder has no Gaussian curvature and nothing but
        hexagons belongs (a pristine (8,8) tube reads 0.000 Å):

            kind   sweeps   wobble     bonds          rings
            L        0      1.576 Å    1.360-1.482    41/313/29
            L       80      1.782 Å    1.374-1.497    23/351/9
            T        0      1.594 Å    1.358-1.504    58/422/46
            T       80      1.648 Å    1.378-1.497    29/482/15
            Y        0      0.608 Å    1.366-1.484    50/443/38
            Y       80      1.441 Å    1.319-1.506    29/486/15
            X        0      1.313 Å    1.367-1.507    64/539/50
            X       80      1.902 Å    1.371-1.506    33/604/17

        Four kinds out of four: the as-grown wall is smoother, and on the
        Y by more than a factor of two. A spread-out population of 5-7
        pairs lets the net take up the surface's curvature everywhere at
        once; annealing most of them away leaves the survivors carrying
        all of it, and each one buckles the wall around it.

        Two costs, both real. The census reads worse, which is the thing
        that made 80 look right. And the sp2 verdict on an L or a T moves
        from "clean" to "strained", because a regular heptagon's interior
        angle is 128.6 deg before any strain at all, so more of them
        pushes ``angle_max`` toward the window's edge. Neither is the wall
        getting worse.
    wall_anchor
        Restrain each atom on the wall's own surface, along the normal
        only. **Off here, and on for the periodic coil**, because it
        pays only where the wall is actually leaving its surface. A
        junction's wall is not: its angle sums sit at 337-340 deg, well
        clear of tetrahedral. Turning it on does pull the atoms back
        (mean deviation 0.790 -> 0.350 Å on a Y, 0.838 -> 0.378 on an X)
        and costs bonds (1.366-1.484 to 1.332-1.564) and placement
        (94.3% to 92.0%) for a wall that was already sound. A thin
        coiled tube is the opposite case and gains on every measure.
    place_curvature
        Move the disclinations to where the surface's own Gaussian
        curvature asks for them, by Stone-Wales flips, after the
        remesh. It cannot change HOW MANY there are -- flips preserve
        ``sum(6 - deg)`` -- only where they sit, which is the thing the
        census objective never looked at.
    roughness
        RMS out-of-plane corrugation in Å applied after relaxation. ``0``
        leaves an ideally smooth shell; 0.1-0.3 Å looks CVD-grown.
    relax_iterations
        L-BFGS iterations for the valence force field.
    vacuum
        Vacuum padding (Å) around the structure.

    Returns
    -------
    ase.Atoms
        Finite, fully closed carbon junction. ``atoms.info`` carries
        ``ring_counts``, ``rings``, ``bonds``, ``geometry``, ``euler``,
        ``genus`` and the build parameters.

    Raises
    ------
    RuntimeError
        If the meshed surface is not closed (open boundary edges), or if
        the ring budget does not match the mesh's Euler characteristic --
        either would mean the carbon network is not a valid closed shell.
    """
    field, extent = im.junction_field(
        kind, tube_radius=tube_radius, arm_length=arm_length, blend=blend
    )
    # Grid resolution has to track the box, not stay fixed: `grid_resolution`
    # points spread over a longer-armed junction give fatter voxels, and once
    # a voxel approaches the tube radius marching cubes starts faceting the
    # arms into prisms. Keep the voxel near a target size instead, capped so
    # memory stays bounded (cost is resolution**3).
    voxel_target = min(1.4, 0.25 * tube_radius)
    needed = int(np.ceil(2.0 * extent / voxel_target))
    resolution = int(np.clip(needed, grid_resolution, MAX_GRID_RESOLUTION))
    mesh = rm.marching_cubes_mesh(field, extent, resolution=resolution)
    # Ring centres in a honeycomb sit sqrt(3)*bond apart, so that is the
    # triangle side that makes the dual come out at the right scale.
    rng = make_rng(seed)
    mesh = rm.isotropic_remesh(
        mesh, field, target_edge=np.sqrt(3.0) * bond,
        iterations=remesh_iterations, anneal_sweeps=anneal_sweeps,
        place_curvature=place_curvature, rng=rng,
    )
    return _finish(
        mesh,
        field=field,
        wall_anchor=wall_anchor,
        bond=bond,
        relax_iterations=relax_iterations,
        vacuum=vacuum,
        roughness=roughness,
        rng=rng,
        info={
            "structure_type": "junction",
            "anneal_sweeps": anneal_sweeps,
            "place_curvature": place_curvature,
            "roughness": roughness,
            "junction_kind": kind,
            "tube_radius": tube_radius,
            "arm_length": arm_length,
            "blend": blend,
            "bond": bond,
        },
    )


def build_schwarzite(
    kind: im.SchwarziteKind = "primitive",
    cell: float = 32.0,
    thickness: float = 0.0,
    bond: float = CC_BOND,
    grid_resolution: int = 64,
    anneal_sweeps: int = 0,
    place_curvature: bool = False,
    wall_anchor: float = 0.0,
    remesh_iterations: int = 25,
    roughness: float = 0.0,
    relax_iterations: int = 3000,
    seed: int | None = 0,
) -> Atoms:
    """Build a **periodic** schwarzite unit cell (negative-curvature carbon).

    One period of a triply periodic minimal surface -- Schwarz P, Schwarz D
    or the gyroid -- meshed and closed on the 3-torus, so the tubes run out
    of one face and back in the opposite one exactly as they do in the
    published structures. The returned :class:`ase.Atoms` is genuinely
    periodic (``pbc=True`` with a cubic cell), not a finite fragment.

    The surface saddles everywhere, so the network is dominated by
    **heptagons and octagons** rather than the pentagons that close a
    fullerene, and the Euler characteristic is strongly negative. The
    genus is fixed by the surface family, and the ring budget follows:

    ========== ====== ==================
    surface    genus  sum(6 - ring_size)
    ========== ====== ==================
    Schwarz P  3      -24
    gyroid     5      -48
    Schwarz D  9      -96
    ========== ====== ==================

    Parameters
    ----------
    kind
        ``"primitive"`` (Schwarz P), ``"diamond"`` (Schwarz D) or
        ``"gyroid"``.
    cell
        Cubic cell length in Å. Smaller cells curve harder, so they need
        proportionally more heptagons and strain the bonds more; below
        :data:`MIN_SCHWARZITE_CELL` for the chosen surface the channels get
        narrower than a carbon ring and the build is rejected.
    thickness
        Level-set offset; nonzero thins or thickens the channels.
    place_curvature
        Move the disclinations to where the surface's own Gaussian
        curvature asks for them, by Stone-Wales flips, after the remesh.
        It cannot change HOW MANY there are, only where they sit. On a
        schwarzite the wall is saddle everywhere, so the heptagons have
        somewhere definite to go.
    anneal_sweeps
        Metropolis flip-annealing passes. **Defaults to 0 here, unlike
        :func:`build_junction`, and raising it makes the structure
        worse.**

        On a junction, a stray pentagon-heptagon pair is genuine
        disorder and annealing them away helps (39 -> 14 on a Y). On a
        triply periodic *minimal* surface it is not disorder at all: the
        surface saddles everywhere, and 5-7 pairs are the mechanism by
        which a hexagonal net covers that Gaussian curvature. Annealing
        removes them and the remaining lattice has to stretch to cover
        the same curvature instead. Measured on Schwarz P at 24 Å, going
        from 0 to 80 sweeps cuts stray pairs from 28 to 3 and pushes the
        longest bond from 1.560 Å to 1.655 Å; the pattern held for every
        surface and cell size tried. A high stray count on these
        structures is a sign the surface is being tiled correctly, not a
        defect to be polished out.
    bond, remesh_iterations, relax_iterations
        As for :func:`build_junction`.
    grid_resolution
        Grid points across one period. Below roughly 64 the periodic weld
        can fail; the builder checks and says so rather than emitting a
        torn surface.

    Returns
    -------
    ase.Atoms
        Periodic unit cell, ``pbc=(True, True, True)``, with ``genus``,
        ``euler`` and the usual ring/geometry metadata in ``atoms.info``.

    Raises
    ------
    RuntimeError
        If the periodic weld leaves boundary edges, i.e. the mesh is not
        closed on the torus.
    """
    if cell <= 0:
        raise ValueError("cell must be positive.")
    minimum = MIN_SCHWARZITE_CELL.get(kind, 22.0)
    if cell < minimum:
        raise ValueError(
            f"cell={cell:.1f} Å is too small for the {kind!r} surface, which "
            f"needs at least {minimum:.0f} Å. Below that the channels curve "
            "harder than a graphitic net can follow: the cell still builds, "
            "but relaxes to bonds outside the sp2 range (Schwarz P at 24 Å "
            "gives 1.33-1.56 Å). Larger cells curve more gently and come out "
            "cleaner."
        )
    field, _ = im.schwarzite_field(kind, cell=cell, thickness=thickness)

    # Whether the marching-cubes grid resolves a given neck depends on how
    # the surface happens to fall between sample points, so a specific
    # (cell, resolution) pair can tear where both its neighbours are fine --
    # the gyroid at 26 Å did exactly that while 24 and 28 were clean. That
    # is a discretisation artefact rather than a physical limit, so retry
    # with a shifted grid before giving up.
    failures: list[str] = []
    for attempt, resolution in enumerate(
        (grid_resolution, grid_resolution + 8, grid_resolution + 16)
    ):
        mesh = rm.periodic_marching_cubes_mesh(field, cell, resolution=resolution)
        stats = rm.mesh_statistics(mesh)
        if stats["boundary_edges"]:
            failures.append(
                f"resolution {resolution}: periodic weld left "
                f"{stats['boundary_edges']} boundary edges"
            )
            continue
        try:
            return _finish(
                rm.isotropic_remesh(
                    mesh, field, target_edge=np.sqrt(3.0) * bond,
                    iterations=remesh_iterations, box=cell,
                    anneal_sweeps=anneal_sweeps, rng=make_rng(seed),
                    place_curvature=place_curvature,
                ),
                field=field,
                wall_anchor=wall_anchor,
                bond=bond,
                relax_iterations=relax_iterations,
                vacuum=0.0,
                box=cell,
                roughness=roughness,
                rng=make_rng(seed),
                info={
                    "structure_type": "schwarzite",
                    "anneal_sweeps": anneal_sweeps,
                    "roughness": roughness,
                    "schwarzite_kind": kind,
                    "cell": cell,
                    "thickness": thickness,
                    "bond": bond,
                    "grid_resolution": resolution,
                    "grid_retries": attempt,
                },
            )
        except RuntimeError as exc:
            failures.append(f"resolution {resolution}: {exc}")

    raise RuntimeError(
        f"Could not build a valid {kind!r} cell at cell={cell:.1f} Å after "
        f"{len(failures)} grid resolutions:\n  "
        + "\n  ".join(failures)
        + "\nTry a larger cell, where the channels are wider relative to a "
        "carbon ring."
    )


#: What the rescue tries, in order, stopping at the first wall that
#: clears tetrahedral by `RESCUE_MARGIN`. One remedy does not fit all:
#: super-cubic clears with anchor 1.0; the superfullerene gets *worse*
#: there (327.6 -> 325.8) and needs 2.0 to clear, 4.0 to clear
#: comfortably; and super-diamond at scale 60 is not an anchor problem
#: at all -- it is under-resolved, sound at voxel 0.60 and collapsed at
#: 0.83, so it needs a finer grid instead.
#:
#: Each step is a full rebuild, so only a wall that actually collapsed
#: pays, and the ladder stops as soon as one works.
RESCUE_ANCHORS = (1.0, 2.0, 4.0)

#: Seconds the rescue may spend rebuilding, in total, before it gives up
#: and names the remedy instead.
#:
#: **Atom count is the wrong measure and was tried first.** A super-fcc
#: cell is 3082 atoms and takes 1440 s; a superfullerene is 7534 atoms
#: and takes 79. An atom limit that spares the fcc from an hour and a
#: half of rebuilds also refuses the superfullerene, which could be
#: rescued three times over inside four minutes. Cost is what matters,
#: and cost is time.
RESCUE_TIME_BUDGET = 900.0

#: Degrees clear of tetrahedral the rescue wants before it stops trying.
RESCUE_MARGIN = 1.5


def rescue_collapsed_wall(atoms, rebuild, anchors=RESCUE_ANCHORS,
                          time_budget: float = RESCUE_TIME_BUDGET,
                          seconds_spent: float = 0.0):
    """Rebuild once with the wall held, if the wall came back collapsed.

    A collapsed wall is not a bad structure, it is an **impossible**
    one: an angle sum under 328.4 deg is past tetrahedral, which no
    carbon reaches. Handing it back is worse than taking the time to try
    again, and the retry is the same pattern this module already uses
    for a weld that fails -- vary one thing and re-measure.

    Only a wall that actually collapsed pays the cost, and the rebuild
    is kept only if it is genuinely better: `wall_anchor` widens bonds,
    so on a wall that was already sound it would be a loss.

    The strength is escalated rather than fixed, because one value does
    not fit all: super-cubic clears at 1.0, and the superfullerene gets
    *worse* at 1.0 (327.6 -> 325.8) before clearing at 2.0 and reaching
    330.8 at 4.0. It keeps whichever rebuild has the highest minimum
    angle sum and stops at the first sound one.
    """
    from ..analyse.hybridisation import (
        TETRAHEDRAL_SUM,
        collapsed_wall,
        hybridisation_report,
    )

    def verdict(candidate):
        return hybridisation_report(candidate.positions,
                                    candidate.info.get("bonds", []),
                                    np.asarray(candidate.cell))

    before = verdict(atoms)
    if not collapsed_wall(before):
        return atoms
    if seconds_spent > time_budget:
        # Rebuilding is worth minutes, not an hour. Say what to set.
        warnings.warn(
            f"The wall came back collapsed (angle sums start at "
            f"{before['angle_sum_min']:.1f} deg, past the 328.4 of a "
            f"tetrahedral carbon), and this cell took {seconds_spent:.0f} s "
            f"to build once, so rebuilding it is left to you. Pass "
            "wall_anchor=2, or a finer grid_resolution, and build again.",
            stacklevel=2,
        )
        return atoms
    warnings.warn(
        f"The wall came back collapsed (angle sums start at "
        f"{before['angle_sum_min']:.1f} deg, past the 328.4 of a "
        "tetrahedral carbon), so it is being rebuilt with the wall held "
        "on its own surface. Pass wall_anchor=0 to keep the collapsed "
        "one, or a larger wall_anchor if this is not enough.",
        stacklevel=2,
    )
    best, best_report, strength = atoms, before, None
    spent = float(seconds_spent)
    for strength in anchors:
        if spent > time_budget:
            break
        started = time.monotonic()
        try:
            candidate = rebuild(strength)
        except Exception:  # noqa: BLE001 - the original is still valid output
            spent += time.monotonic() - started
            continue
        spent += time.monotonic() - started
        report = verdict(candidate)
        if report["angle_sum_min"] > best_report["angle_sum_min"]:
            best, best_report = candidate, report
        # Not merely "not collapsed": stopping the moment the check
        # passes left the superfullerene at 328.4 deg, which is the
        # threshold itself to one decimal. One more step took it to
        # 330.8. A hairline pass is not a sound wall, it is a wall that
        # will read collapsed again on the next change to anything.
        if report["angle_sum_min"] >= TETRAHEDRAL_SUM + RESCUE_MARGIN:
            break
    if best is atoms:
        return atoms
    best.info["wall_rescued_from"] = round(float(before["angle_sum_min"]), 1)
    # `None` is the ladder's "try a finer grid" step, so this cannot
    # just be a float -- it names what actually worked.
    best.info["wall_rescue_step"] = (
        "finer grid" if strength is None else f"wall_anchor={strength:g}")
    return best


def _finish(
    mesh: fm.Mesh,
    bond: float,
    relax_iterations: int,
    vacuum: float,
    info: dict,
    box: float | None = None,
    roughness: float = 0.0,
    rng=None,
    pin_near: np.ndarray | None = None,
    pin_radius: float = 0.0,
    k_pin: float = 5.0,
    field=None,
    wall_anchor: float = 0.0,
) -> Atoms:
    """Shared tail: validate the mesh, take its dual, relax, package.

    ``box`` set means the structure is a periodic cell: the dual, the
    relaxation and the returned cell/pbc all switch to minimum-image.

    ``pin_near`` gives points (in mesh coordinates) whose surrounding
    atoms are harmonically restrained during relaxation, with ``pin_radius``
    setting how far that reaches and ``k_pin`` how stiffly.

    That option exists for one specific asymmetry. A closed shell's ring
    topology encodes its **curvature**, so a coiled tube relaxes back to
    the coil radius it was built at -- measured 29.4 Å against a requested
    30.0. It does not encode **torsion**: nothing in a hexagonal net fixes
    how fast the coil advances along its axis, so the pitch is a free soft
    mode and an unrestrained coil springs open, 20 Å requested to 26.7 Å
    relaxed. Pinning the caps is the boundary condition of a tube held
    between contacts and holds the axial length -- but it also fights the
    relaxation where the network is most distorted, so it is off by
    default: at ``k_pin=5`` that coil came out with 1.18-1.71 Å bonds and
    three overlapping pairs, failing the quality gate below that the free
    relaxation passes.
    """
    stats = rm.mesh_statistics(mesh)
    if stats["boundary_edges"]:
        raise RuntimeError(
            f"Meshed surface is not closed: {stats['boundary_edges']} boundary "
            "edges. The sampling box probably clipped the surface -- increase "
            "the extent, or reduce arm_length / clip_radius."
        )

    positions, bond_set, rings = fm.dual_honeycomb(mesh, box=box)
    bonds = sorted(bond_set)
    ring_counts = fm.ring_size_histogram(rings)

    # Euler's budget, read from the mesh rather than assumed: a capped
    # junction is sphere-like (chi = 2, deficit 12) but a schwarzite
    # fragment has handles and a strongly negative deficit.
    deficit = sum((6 - size) * count for size, count in ring_counts.items())
    expected = 6 * stats["euler"]
    if deficit != expected:
        raise RuntimeError(
            f"Ring deficit {deficit} does not match the mesh's Euler "
            f"characteristic (expected {expected} for chi={stats['euler']}). "
            "The dual is not a valid closed carbon network."
        )

    lengths = np.array([
        np.linalg.norm(fm.minimum_image(positions[b] - positions[a], box))
        for a, b in bonds
    ])
    scale = bond / lengths.mean()
    # Scaling a periodic cell must scale the cell with it, or the bonds
    # come out right while the lattice no longer matches them.
    positions = positions * scale
    scaled_box = None if box is None else box * scale

    if scaled_box is None:
        anchors = anchor_targets = None
        if pin_near is not None and pin_radius > 0:
            from scipy.spatial import cKDTree

            near = cKDTree(np.asarray(pin_near, dtype=float) * scale).query_ball_point(
                positions, r=pin_radius
            )
            anchors = np.array([i for i, hits in enumerate(near) if hits], dtype=int)
            if len(anchors):
                anchor_targets = positions[anchors]
            else:
                anchors = None
        if field is not None and wall_anchor > 0.0 and anchors is None:
            # Hold the wall on its own surface, along the normal only,
            # so atoms still slide within it. Free relaxation walks a
            # large share of them off the surface entirely.
            # `positions` were scaled, so the field is read at the
            # pre-scale coordinates it was built in.
            anchors = np.arange(len(positions))
            anchor_targets = positions.copy()
            positions = fm.relax_shell(
                positions, bond_set, equilibrium=bond,
                max_iterations=relax_iterations,
                anchors=anchors, anchor_targets=anchor_targets,
                anchor_normals=rm.field_normals(field, positions / scale),
                k_anchor=float(wall_anchor),
            )
        else:
            positions = fm.relax_shell(
                positions, bond_set, equilibrium=bond,
                max_iterations=relax_iterations,
                anchors=anchors, anchor_targets=anchor_targets,
                k_anchor=k_pin,
            )
    else:
        # Variable-cell relaxation. With the cell held fixed the network
        # cannot reach its natural bond length -- it is stretched or
        # compressed by whatever the initial guess was off by, and on the
        # denser surfaces that showed up as 6 Å "bonds" and torn geometry.
        # So alternate: relax the atoms at the current cell, measure how far
        # the mean bond is from equilibrium, and rescale cell and atoms
        # together by that ratio.
        for _ in range(CELL_RELAX_CYCLES):
            # The field lives in the mesh's ORIGINAL coordinates, and
            # this branch keeps rescaling cell and atoms together, so
            # the map back is the cumulative scale -- tracked rather
            # than assumed. Without this the periodic nets never got
            # the anchor at all: super-cubic came back byte-identical
            # with it on and off, still collapsed at 327.8 deg.
            extra = {}
            if field is not None and wall_anchor > 0.0:
                cumulative = float(np.mean(np.asarray(scaled_box)
                                           / np.asarray(box)))
                extra = dict(
                    anchors=np.arange(len(positions)),
                    anchor_targets=positions.copy(),
                    anchor_normals=rm.field_normals(
                        field, np.mod(positions, scaled_box) / cumulative),
                    k_anchor=float(wall_anchor),
                )
            positions = fm.relax_shell(
                positions, bond_set, equilibrium=bond,
                box=scaled_box, max_iterations=relax_iterations, **extra,
            )
            mean_bond = float(np.mean([
                np.linalg.norm(fm.minimum_image(positions[b] - positions[a], scaled_box))
                for a, b in bonds
            ]))
            if abs(mean_bond - bond) < 1e-3:
                break
            adjust = bond / mean_bond
            positions = np.mod(positions * adjust, scaled_box * adjust)
            scaled_box *= adjust

    if scaled_box is None:
        atoms = Atoms(symbols=["C"] * len(positions), positions=positions, pbc=False)
        extents = positions.max(axis=0) - positions.min(axis=0)
        atoms.set_cell(np.diag(extents + vacuum))
        center_in_cell(atoms, axes=(0, 1, 2))
    else:
        positions = np.mod(positions, scaled_box)
        atoms = Atoms(symbols=["C"] * len(positions), positions=positions, pbc=True)
        # A cell of three lengths, not one: the nets that came through
        # here first were all cubic, and a honeycomb supersheet is a wide
        # and a*sqrt(3) deep. `np.eye(3) * v` does the right thing for
        # both a scalar and a 3-vector.
        atoms.set_cell(np.eye(3) * scaled_box)
        lengths = np.broadcast_to(np.asarray(scaled_box, dtype=float), (3,))
        info = {**info, "cell": (float(lengths[0]) if len(set(lengths)) == 1
                                 else [float(v) for v in lengths])}

    if roughness > 0:
        positions = fm.apply_surface_roughness(
            positions, bond_set, roughness,
            rng if rng is not None else make_rng(0),
            equilibrium=bond, box=scaled_box,
        )

    quality = geometry_report(positions, bonds, box=scaled_box)
    # Fail loudly rather than hand back a torn network. These thresholds are
    # far outside anything strain can explain: a 1.8 Å "bond" or a sub-2 Å
    # non-bonded contact means the surface pinched through itself during
    # remeshing, not that the structure is merely strained.
    if quality["bond_max"] > 1.80 or quality["n_close_contacts"] > 0:
        raise RuntimeError(
            "Relaxed network is not physically valid: bonds span "
            f"{quality['bond_min']:.2f}-{quality['bond_max']:.2f} Å with "
            f"{quality['n_close_contacts']} non-bonded contacts under 2 Å. "
            "The surface most likely has features finer than a carbon ring -- "
            "use a larger cell / arm_length, or a bigger tube_radius."
        )

    atoms.info.update(info)
    atoms.info.update(
        {
            "euler": stats["euler"],
            "genus": stats["genus"],
            "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
            "rings": [[int(a) for a in r] for r in rings],
            "bonds": [[int(a), int(b)] for a, b in bonds],
            "geometry": quality,
            # The coil papers' one checkable structural claim, in its
            # general form: a pentagon is a +60 deg disclination and a
            # heptagon a -60 deg one, so they belong in positive and
            # negative curvature respectively. The coil tests it against
            # a torus axis, which only a coil has; this is intrinsic, so
            # every surface through _finish -- junction, schwarzite,
            # network, supernetwork -- is held to the same claim. It
            # costs 0.08 s at 1134 atoms.
            "disclination_check": disclination_check(
                positions, rings, sorted(bonds),
                box=None if box is None else np.broadcast_to(
                    np.asarray(box, dtype=float), (3,)).copy(),
            ),
        }
    )
    return atoms
