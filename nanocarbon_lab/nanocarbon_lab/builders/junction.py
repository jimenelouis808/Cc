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
# Schwarz D (genus 9) tears at 24 Å where Schwarz P (genus 3) is fine.
# Measured by sweeping the cell and checking bond/contact statistics.
MIN_SCHWARZITE_CELL = {"primitive": 20.0, "gyroid": 22.0, "diamond": 30.0}


def build_junction(
    kind: im.JunctionKind = "Y",
    tube_radius: float = 6.0,
    arm_length: float = 22.0,
    blend: float = 4.0,
    bond: float = CC_BOND,
    grid_resolution: int = 70,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 80,
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
        junction). Set to ``0`` to keep the as-remeshed defect population,
        which reads as a rougher, more as-grown wall.
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
        iterations=remesh_iterations, anneal_sweeps=anneal_sweeps, rng=rng,
    )
    return _finish(
        mesh,
        bond=bond,
        relax_iterations=relax_iterations,
        vacuum=vacuum,
        roughness=roughness,
        rng=rng,
        info={
            "structure_type": "junction",
            "anneal_sweeps": anneal_sweeps,
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
    remesh_iterations: int = 25,
    anneal_sweeps: int = 80,
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
            f"needs at least {minimum:.0f} Å. Its channels would be narrower "
            "than a carbon ring, so the remesher pinches through the necks "
            "and the network comes out torn."
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
                ),
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
        positions = fm.relax_shell(
            positions, bond_set, equilibrium=bond, max_iterations=relax_iterations,
            anchors=anchors, anchor_targets=anchor_targets, k_anchor=k_pin,
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
            positions = fm.relax_shell(
                positions, bond_set, equilibrium=bond,
                box=scaled_box, max_iterations=relax_iterations,
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
        atoms.set_cell(np.eye(3) * scaled_box)
        info = {**info, "cell": float(scaled_box)}

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
        }
    )
    return atoms
