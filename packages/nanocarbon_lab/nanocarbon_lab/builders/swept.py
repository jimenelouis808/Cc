"""Curved tubes whose ring topology is *derived* from the curvature.

There are two ways to build a bent nanotube in this package, and they are
not equivalent.

:func:`nanocarbon_lab.builders.capped_cnt.build_capped_cnt` with a
``shape`` builds a straight all-hexagon tube and then **sweeps** it onto a
curved centreline. That is fast, keeps the exact cap topology, and is
right for gentle curves -- but the lattice is unchanged, so the bend is
carried entirely as elastic strain. A pure-hexagon tube bent onto an arc
*must* have a longer outer wall than inner, and no relaxation removes
that: measured on a 90 Å coil around a 5.9 Å tube, a 6.5% path strain
leaves bonds spanning 1.33-1.51 Å, and weakening the positional restraints
only lets atoms wander (1 Å drift) without recovering a single hundredth
of an Ångström of bond length.

This module takes the other route, the one already used for junctions and
schwarzites: build the curved tube as an implicit surface, mesh it, and
let :mod:`nanocarbon_lab.builders.remesh` choose vertex degrees -- which
*are* ring sizes in the dual. Curvature then buys its own topology.
Pentagons appear where the wall is compressed, heptagons where it is
stretched, exactly as in real coiled and knee-jointed nanotubes, and the
bonds come back to graphitic length instead of being held stretched.

The cost is that the caps are no longer the hand-built 6-pentagon domes
and the atom count is not a simple function of ring count; the pay-off is
a structure that is physically right rather than merely bent. Both are
kept, because for a gentle S-curve the swept route is cheaper and just as
good, and only tight curvature makes the difference matter.

Euler still holds, and is checked the same way: the tube is a closed
sphere-like surface however it is coiled, so ``sum(6 - ring_size) = 12``,
and the extra pentagons the curvature introduces are paid for by an equal
number of extra heptagons.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.rng import make_rng
from . import centerline as cl
from . import implicit as im
from . import remesh as rm
from .junction import _finish

# Voxel edge as a fraction of the tube radius. A quarter resolves the
# wall's curvature without faceting it into a prism; the remesher sets the
# final ring size regardless, so finer than this buys nothing.
VOXEL_FRACTION = 0.25

# Absolute floor on voxel size (Å). Below this the grid cost explodes
# without changing the meshed shape at nanotube dimensions.
MIN_VOXEL = 0.7


def build_swept_tube(
    path: np.ndarray,
    tube_radius: float = 6.0,
    bond: float = CC_BOND,
    voxel: float | None = None,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 80,
    roughness: float = 0.0,
    relax_iterations: int = 3000,
    vacuum: float = DEFAULT_VACUUM_1D,
    pin_ends: bool = False,
    seed: int | None = 0,
) -> Atoms:
    """Build a capped carbon tube following an arbitrary 3D centreline.

    Parameters
    ----------
    path
        ``(n, 3)`` centreline in Å. Resampled internally, so control
        points need only resolve the shape, not be evenly spaced. The
        tube is capped at both ends (the field is a capsule sweep), so the
        path's endpoints are the centres of the caps, not the tips.
    tube_radius
        Tube radius in Å. Unlike ``build_capped_cnt``, this is a geometric
        radius rather than a lattice-quantised one: the surface is made to
        this size and the remesher tiles whatever it gets.
    bond
        Target C-C bond length (Å).
    voxel
        Marching-cubes voxel edge (Å). Defaults to
        ``max(MIN_VOXEL, VOXEL_FRACTION * tube_radius)``.
    remesh_iterations, anneal_sweeps, roughness, relax_iterations, vacuum
        As for :func:`nanocarbon_lab.builders.junction.build_junction`.
    pin_ends
        Restrain the two end caps to their meshed positions during
        relaxation, the boundary condition of a tube held between
        contacts. Off by default, and worth understanding before turning
        on.

        Ring topology encodes a tube's **curvature**, so a free coil comes
        back at the radius it was built at (29.4 Å measured against a
        requested 30.0). It encodes nothing about **torsion**, so the pitch
        is a soft mode and a free coil springs open along its axis
        (20 Å requested, 26.7 Å relaxed). Pinning holds the axial length,
        but it also fights the relaxation exactly where the network is most
        distorted: at ``k_pin=5`` the same coil came out with 1.18-1.71 Å
        bonds and three overlapping atom pairs, failing the quality gate
        that the free relaxation passes cleanly. So the default is to let
        the coil find its own pitch and to *report* it, rather than to hold
        a requested number at the cost of the chemistry.
    seed
        RNG seed for flip annealing and roughness.

    Returns
    -------
    ase.Atoms
        Closed carbon tube with ``ring_counts`` reflecting the curvature,
        plus ``path_length``, ``tube_radius`` and the usual ``geometry``,
        ``euler`` and ``genus`` entries in ``atoms.info``.

    Raises
    ------
    ValueError
        For a degenerate path or radius (raised by
        :func:`nanocarbon_lab.builders.implicit.tube_along_path`).
    RuntimeError
        If the meshed surface is not closed, the ring budget does not
        match Euler, or the relaxed network fails the quality gate. A
        self-intersecting path is the usual cause: where two coils of a
        helix come within ``2 * tube_radius`` the surface merges into one
        object and stops being a tube.
    """
    path = np.asarray(path, dtype=float)
    if voxel is None:
        voxel = max(MIN_VOXEL, VOXEL_FRACTION * tube_radius)

    field, lower, upper = im.tube_along_path(
        path, radius=tube_radius, sample_spacing=min(0.4, 0.25 * voxel + 0.2)
    )
    mesh = rm.marching_cubes_box(field, lower, upper, spacing=voxel)

    rng = make_rng(seed)
    mesh = rm.isotropic_remesh(
        mesh, field, target_edge=np.sqrt(3.0) * bond,
        iterations=remesh_iterations, anneal_sweeps=anneal_sweeps, rng=rng,
    )
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return _finish(
        mesh,
        bond=bond,
        relax_iterations=relax_iterations,
        vacuum=vacuum,
        roughness=roughness,
        rng=rng,
        pin_near=path[[0, -1]] if pin_ends else None,
        pin_radius=1.2 * tube_radius,
        info={
            "structure_type": "swept_tube",
            "tube_radius": tube_radius,
            "path_length": float(steps.sum()),
            "voxel": float(voxel),
            "pin_ends": bool(pin_ends),
            "anneal_sweeps": anneal_sweeps,
            "roughness": roughness,
            "bond": bond,
        },
    )


def build_coil(
    coil_radius: float = 40.0,
    pitch: float = 25.0,
    turns: float = 2.0,
    tube_radius: float = 6.0,
    handedness: int = 1,
    taper: float = 1.0,
    bond: float = CC_BOND,
    **kwargs,
) -> Atoms:
    """Build a helical carbon nanocoil with curvature-derived ring topology.

    A convenience wrapper that lays out the helix and hands it to
    :func:`build_swept_tube`. Prefer this over
    ``build_capped_cnt(shape="helix", ...)`` whenever the coil is tight
    enough that the swept route reports strained or broken bonds -- here
    the curvature is absorbed by pentagon-heptagon pairs instead.

    Parameters
    ----------
    coil_radius, pitch, turns, handedness, taper
        Coil geometry in absolute Å; see
        :func:`nanocarbon_lab.builders.centerline.helix_control_points`.
    tube_radius
        Radius of the tube itself (Å).
    bond
        Target C-C bond length (Å).
    **kwargs
        Passed through to :func:`build_swept_tube`.

    Returns
    -------
    ase.Atoms
        The coil, with ``coil_radius``, ``pitch`` and ``turns`` recorded
        in ``atoms.info`` alongside the usual metadata.

    Raises
    ------
    ValueError
        If successive coils would overlap. Neighbouring turns are ``pitch``
        apart along the axis, so a pitch below ``2 * tube_radius`` plus a
        van der Waals gap does not describe a coil at all -- the walls
        merge into a single solid and the mesh is no longer a tube.
    """
    clearance = 2.0 * tube_radius + 3.4
    if pitch < clearance:
        raise ValueError(
            f"pitch={pitch:.1f} Å is below the {clearance:.1f} Å needed to keep "
            f"successive turns of a {tube_radius:.1f} Å tube apart (two walls "
            "plus a graphitic gap). Adjacent coils would merge into one solid."
        )
    # Dense enough that the polyline itself does not corner the helix.
    control = cl.helix_control_points(
        coil_radius, pitch, turns,
        n_points=max(64, round(turns * 48)),
        handedness=handedness, taper=taper,
    )
    atoms = build_swept_tube(control, tube_radius=tube_radius, bond=bond, **kwargs)
    # The requested dimensions size the *surface*; the carbon network then
    # finds its own equilibrium, and the two dimensions do not fare alike.
    # Ring topology encodes curvature, so the coil radius comes back
    # essentially as asked (29.4 Å measured against a requested 30.0). It
    # encodes nothing about torsion, so the pitch is a soft mode that opens
    # even with the end caps pinned. Report what came out.
    achieved = _measure_coil(atoms.get_positions())
    atoms.info.update(
        {
            "structure_type": "nanocoil_implicit",
            "coil_radius": coil_radius,
            "pitch": pitch,
            "turns": turns,
            "handedness": handedness,
            "taper": taper,
            "achieved_coil_radius": achieved[0],
            "achieved_pitch": achieved[1],
        }
    )
    return atoms


def _measure_coil(positions: np.ndarray, n_sectors: int = 24) -> tuple[float, float]:
    """Coil radius and pitch measured off the relaxed atoms.

    The coil axis is z, which is how
    :func:`nanocarbon_lab.builders.centerline.helix_control_points` lays a
    helix out and nothing downstream rotates it. The radius is then just
    the mean in-plane distance from that axis.

    Pitch is measured **per azimuthal sector**, not by unwrapping the
    azimuth along z. Unwrapping fails here for a reason worth recording:
    atoms at a given height wrap right around the tube's circumference, so
    the azimuth is multivalued in z and the unwrapped angle accumulates
    turns that the coil does not have -- it read a 20 Å pitch as 7 Å.
    Instead, atoms in one narrow sector of azimuth belong to one turn per
    visit, so their z values fall into clusters exactly one pitch apart,
    and the median gap between consecutive clusters is the pitch.

    That needs a sector to be visited **twice**, so pitch is only
    measurable above one full turn. Below that there is no second cluster
    to measure a gap to, and the returned pitch is ``nan`` rather than a
    number that looks plausible and is not: a one-turn coil of pitch 20 Å
    reported 63.9 Å from the axial-span fallback this replaces.
    """
    centred = positions - positions.mean(axis=0)
    radius = float(np.linalg.norm(centred[:, :2], axis=1).mean())

    angle = np.arctan2(centred[:, 1], centred[:, 0])
    sector = np.floor((angle + np.pi) / (2.0 * np.pi) * n_sectors).astype(int)
    gaps: list[float] = []
    for index in range(n_sectors):
        z = np.sort(centred[sector == index, 2])
        if len(z) < 4:
            continue
        # A cluster break is a z gap larger than the tube's own extent in
        # this sector; the tube diameter is a safe threshold since the
        # pitch must exceed it for the turns not to touch.
        steps = np.diff(z)
        breaks = np.flatnonzero(steps > max(3.0, 4.0 * np.median(steps)))
        if len(breaks) == 0:
            continue
        edges = np.concatenate([[0], breaks + 1, [len(z)]])
        centres = [z[edges[k]:edges[k + 1]].mean() for k in range(len(edges) - 1)]
        gaps += list(np.diff(centres))

    pitch = float(np.median(gaps)) if gaps else float("nan")
    return radius, pitch


__all__ = ["MIN_VOXEL", "VOXEL_FRACTION", "build_coil", "build_swept_tube"]
