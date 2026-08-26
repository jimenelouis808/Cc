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

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from . import fullerene_mesh as fm
from . import implicit as im
from . import remesh as rm
from .capped_cnt import geometry_report


def build_junction(
    kind: im.JunctionKind = "Y",
    tube_radius: float = 6.0,
    arm_length: float = 22.0,
    blend: float = 4.0,
    bond: float = CC_BOND,
    grid_resolution: int = 70,
    remesh_iterations: int = 25,
    relax_iterations: int = 3000,
    vacuum: float = DEFAULT_VACUUM_1D,
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
        Marching-cubes grid points per axis. Only needs to capture the
        shape -- the remesher sets the final ring size.
    remesh_iterations
        Isotropic remeshing cycles; see
        :func:`nanocarbon_lab.builders.remesh.isotropic_remesh`.
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
    mesh = rm.marching_cubes_mesh(field, extent, resolution=grid_resolution)
    # Ring centres in a honeycomb sit sqrt(3)*bond apart, so that is the
    # triangle side that makes the dual come out at the right scale.
    mesh = rm.isotropic_remesh(
        mesh, field, target_edge=np.sqrt(3.0) * bond, iterations=remesh_iterations
    )
    return _finish(
        mesh,
        bond=bond,
        relax_iterations=relax_iterations,
        vacuum=vacuum,
        info={
            "structure_type": "junction",
            "junction_kind": kind,
            "tube_radius": tube_radius,
            "arm_length": arm_length,
            "blend": blend,
            "bond": bond,
        },
    )


def build_schwarzite(
    kind: im.SchwarziteKind = "primitive",
    cell: float = 30.0,
    clip_radius: float | None = None,
    thickness: float = 0.0,
    bond: float = CC_BOND,
    grid_resolution: int = 80,
    remesh_iterations: int = 25,
    relax_iterations: int = 3000,
    vacuum: float = DEFAULT_VACUUM_1D,
) -> Atoms:
    """Build a finite fragment of a schwarzite (negative-curvature carbon).

    A triply periodic minimal surface -- Schwarz P, Schwarz D or the
    gyroid -- is clipped to a ball so the result is a closed molecule
    rather than an infinite sheet. The surface is saddle-shaped
    everywhere, so the network is dominated by **heptagons and octagons**
    instead of the pentagons that close a fullerene, and its Euler
    characteristic is strongly negative: a fragment with ``g`` handles
    obeys ``sum(6 - ring_size) = 12 * (1 - g)``.

    Parameters
    ----------
    kind
        ``"primitive"`` (Schwarz P), ``"diamond"`` (Schwarz D) or
        ``"gyroid"``.
    cell
        Period of the surface in Å. Smaller cells curve harder and need
        more heptagons.
    clip_radius
        Radius of the clipping ball (Å). Defaults to ``0.75 * cell``,
        which captures roughly one period. Larger fragments have more
        handles and take longer to relax.
    thickness
        Level-set offset; nonzero thins or thickens the channels.
    bond, grid_resolution, remesh_iterations, relax_iterations, vacuum
        As for :func:`build_junction`.

    Returns
    -------
    ase.Atoms
        Finite schwarzite fragment, with ``genus`` and ``euler`` recorded
        in ``atoms.info``.

    Notes
    -----
    This is a **fragment**, not a periodic cell: it is clipped and closed
    so it can be rendered and handed to a finite-molecule workflow. A
    genuinely periodic schwarzite would need the mesh built on a torus
    with matching boundaries, which this does not attempt.
    """
    base, _ = im.schwarzite_field(kind, cell=cell, thickness=thickness)
    # The trigonometric field is unitless; the clip is in Å. Normalise
    # before intersecting or the ball does not actually cut anything.
    base = im.normalize_to_distance(base)
    radius = clip_radius if clip_radius is not None else 0.75 * cell
    field = im.intersect_with_ball(base, radius, softness=0.10 * radius)
    extent = radius + 0.30 * cell

    mesh = rm.marching_cubes_mesh(field, extent, resolution=grid_resolution)
    mesh = rm.isotropic_remesh(
        mesh, field, target_edge=np.sqrt(3.0) * bond, iterations=remesh_iterations
    )
    return _finish(
        mesh,
        bond=bond,
        relax_iterations=relax_iterations,
        vacuum=vacuum,
        info={
            "structure_type": "schwarzite",
            "schwarzite_kind": kind,
            "cell": cell,
            "clip_radius": radius,
            "thickness": thickness,
            "bond": bond,
        },
    )


def _finish(
    mesh: fm.Mesh,
    bond: float,
    relax_iterations: int,
    vacuum: float,
    info: dict,
) -> Atoms:
    """Shared tail: validate the mesh, take its dual, relax, package."""
    stats = rm.mesh_statistics(mesh)
    if stats["boundary_edges"]:
        raise RuntimeError(
            f"Meshed surface is not closed: {stats['boundary_edges']} boundary "
            "edges. The sampling box probably clipped the surface -- increase "
            "the extent, or reduce arm_length / clip_radius."
        )

    positions, bond_set, rings = fm.dual_honeycomb(mesh)
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

    lengths = np.array([np.linalg.norm(positions[a] - positions[b]) for a, b in bonds])
    positions = positions * (bond / lengths.mean())
    positions = fm.relax_shell(
        positions, bond_set, equilibrium=bond, max_iterations=relax_iterations
    )

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions, pbc=False)
    extents = positions.max(axis=0) - positions.min(axis=0)
    atoms.set_cell(np.diag(extents + vacuum))
    center_in_cell(atoms, axes=(0, 1, 2))

    atoms.info.update(info)
    atoms.info.update(
        {
            "euler": stats["euler"],
            "genus": stats["genus"],
            "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
            "rings": [[int(a) for a in r] for r in rings],
            "bonds": [[int(a), int(b)] for a, b in bonds],
            "geometry": geometry_report(positions, bonds),
        }
    )
    return atoms
