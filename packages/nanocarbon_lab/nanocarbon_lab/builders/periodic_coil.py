"""A coiled nanotube as a genuinely periodic cell, for DFT.

A helix of pitch ``p`` maps onto itself under a translation of ``p``
along its axis: one turn, closed on the z-torus, is a real unit cell and
not a fragment with its ends waved at. That is what makes a plane-wave
calculation of a nanocoil possible at all — the alternative is a finite
segment whose two dangling ends dominate whatever you were trying to
measure.

Two details make the difference between a cell and a torn mesh, and both
were found by measurement rather than by reasoning:

**The implicit field has to be periodic to machine precision.**
:func:`~nanocarbon_lab.builders.implicit.tube_along_path` resamples the
centreline by arc length, so unless one turn is an exact whole number of
samples the field at ``z = 0`` and ``z = p`` differ — by 0.05 Å at a
loose spacing, which is enormous next to the interpolation marching cubes
does. The weld then fails and the mesh comes back with hundreds of
boundary edges. Choosing the sample spacing commensurate with one turn
drops the mismatch to 2e-13 and the boundary edges to zero.

**The wall is not a rolled lattice, and should not be.** Sweeping a
finished all-hexagon tube along a helix stretches its outer wall: at 6.5%
path strain that is 1.51 Å bonds, and no relaxation removes them, because
a pure-hexagon tube bent onto an arc *must* have a longer outer wall.
Meshing the surface instead lets the remesher put pentagons on the
compressed inner wall and heptagons on the stretched outer one, which is
how real coiled nanotubes relieve the strain. The result looks less tidy
and is the more physical structure.

One period of a helical tube is topologically a torus, so a correct mesh
comes back at genus 1. That is the check this module runs before it hands
anything back.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND
from ..utils.rng import make_rng
from . import fullerene_mesh as fm
from . import implicit as im
from . import remesh as rm
from .capped_cnt import geometry_report

__all__ = ["build_periodic_coil", "helix_period_samples"]

#: Turns of helix sampled either side of the cell. The distance field near
#: the seam has to see the neighbouring turns, or the surface bulges where
#: the sampled centreline runs out.
_CONTEXT_TURNS = 2.0


def helix_period_samples(coil_radius: float, pitch: float,
                         target_spacing: float = 0.3) -> float:
    """Sample spacing commensurate with one turn, near ``target_spacing``.

    Returns the spacing that divides one turn's arc length a whole number
    of times. Without this the field is not periodic and the periodic
    weld has nothing to match.
    """
    arc = float(np.hypot(2.0 * np.pi * coil_radius, pitch))
    divisions = max(1, round(arc / target_spacing))
    return arc / divisions


def build_periodic_coil(
    coil_radius: float = 15.0,
    pitch: float = 8.0,
    tube_radius: float = 2.0,
    bond: float = CC_BOND,
    handedness: int = 1,
    vacuum: float = 10.0,
    resolution: int = 64,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 80,
    relax_iterations: int = 3000,
    seed: int | None = 0,
) -> Atoms:
    """Build one period of a coiled nanotube, periodic along z.

    Parameters
    ----------
    coil_radius
        Distance from the helix axis to the tube's centre (Å).
    pitch
        Rise of one full turn (Å), and therefore the cell length along z.
        Must clear two tube walls plus a graphitic gap, or successive
        turns merge into one solid.
    tube_radius
        Radius of the tube itself (Å).
    bond
        Target C-C length (Å); sets the remesh edge and the relaxation.
    handedness
        ``+1`` right-handed, ``-1`` left-handed.
    vacuum
        Padding (Å) between the coil and the cell walls in x and y. Those
        two directions are not periodic; the padding keeps the surface
        clear of them so their weld is a no-op.
    resolution
        Grid points across the longest cell axis. Too coarse and the seam
        does not weld, which is reported rather than returned.
    remesh_iterations, anneal_sweeps, relax_iterations, seed
        Passed through to the remesh and relaxation, as for the other
        implicit builders.

    Returns
    -------
    ase.Atoms
        ``pbc=(False, False, True)`` with the cell's z equal to ``pitch``.

    Raises
    ------
    ValueError
        If the pitch cannot keep successive turns apart, or if the mesh
        comes back open or at the wrong genus -- both mean the cell is
        not periodic, and a structure that is quietly not periodic is
        worse than none.
    """
    if coil_radius <= 0.0 or pitch <= 0.0 or tube_radius <= 0.0:
        raise ValueError("coil_radius, pitch and tube_radius must be positive.")
    clearance = 2.0 * tube_radius + 3.4
    if pitch < clearance:
        raise ValueError(
            f"pitch={pitch:.1f} Å is below the {clearance:.1f} Å needed to keep "
            f"successive turns of a {tube_radius:.1f} Å tube apart (two walls "
            "plus a graphitic gap). Adjacent turns would merge into one solid."
        )

    spacing = helix_period_samples(coil_radius, pitch)
    turns = 1.0 + 2.0 * _CONTEXT_TURNS
    # A whole number of samples per turn only helps if the sampling starts
    # at a turn boundary, so the context is a whole number of turns too.
    theta = np.linspace(-_CONTEXT_TURNS * 2.0 * np.pi,
                        (1.0 + _CONTEXT_TURNS) * 2.0 * np.pi,
                        int(round(turns * 4000)) + 1)
    sign = 1.0 if handedness >= 0 else -1.0
    path = np.column_stack([
        coil_radius * np.cos(theta),
        sign * coil_radius * np.sin(theta),
        pitch * theta / (2.0 * np.pi),
    ])
    field, _lower, _upper = im.tube_along_path(
        path, radius=tube_radius, sample_spacing=spacing)

    half = coil_radius + tube_radius + vacuum
    cell = np.array([2.0 * half, 2.0 * half, pitch], dtype=float)

    def centred(points: np.ndarray) -> np.ndarray:
        """The mesher samples ``[0, L]``; the helix is centred on the axis."""
        shifted = np.array(points, dtype=float, copy=True)
        shifted[..., 0] -= half
        shifted[..., 1] -= half
        return field(shifted)

    mesh = rm.periodic_marching_cubes_mesh(centred, cell, resolution=resolution)
    stats = rm.mesh_statistics(mesh)
    if stats.get("boundary_edges", 0):
        raise ValueError(
            f"The z seam did not weld: {stats['boundary_edges']} boundary edges "
            f"at resolution {resolution}. The cell is not periodic. Raise the "
            "resolution, or widen the pitch so the wall is better resolved."
        )

    rng = make_rng(seed)
    mesh = rm.isotropic_remesh(
        mesh, centred, target_edge=float(np.sqrt(3.0) * bond),
        iterations=remesh_iterations, box=cell,
        anneal_sweeps=anneal_sweeps, rng=rng,
    )
    positions, bond_set, rings = fm.dual_honeycomb(mesh, box=cell)
    positions = fm.relax_shell(
        positions, bond_set, equilibrium=bond, box=cell,
        max_iterations=relax_iterations,
    )
    positions = np.mod(positions, cell)

    # Euler is necessary and not sufficient: a mesh can come back
    # topologically consistent and still be geometrically torn, which is
    # what a 2.2 Å "bond" means. Measured on a 25 Å coil that passed the
    # ring budget and would have been handed back as a DFT cell, so this
    # check is here for a case that actually happened rather than a
    # hypothetical one. Same thresholds as the schwarzites: far outside
    # anything strain can explain.
    quality = geometry_report(positions, sorted(bond_set), box=cell)
    if quality["bond_max"] > 1.80 or quality["n_close_contacts"] > 0:
        raise ValueError(
            "The relaxed network is torn, not merely strained: bonds span "
            f"{quality['bond_min']:.2f}-{quality['bond_max']:.2f} Å with "
            f"{quality['n_close_contacts']} non-bonded contacts under 2 Å. "
            "The surface has features finer than a carbon ring somewhere -- "
            "raise the resolution, or use a thicker tube or a tighter coil."
        )

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  pbc=(False, False, True))
    atoms.set_cell(np.diag(cell))

    bonds = sorted(bond_set)
    counts: dict[int, int] = {}
    for ring in rings:
        counts[len(ring)] = counts.get(len(ring), 0) + 1
    # Euler, as the last word on whether this is a cell or a tear. On a
    # closed trivalent net, sum(6 - n) over the rings is 12(1 - genus),
    # and one period of a helical tube is a torus, so it must come to
    # zero. A mesh that pinched through itself during remeshing passes
    # every local check and fails this one.
    deficit = sum((6 - size) * count for size, count in counts.items())
    expected = 12 * (1 - int(stats.get("genus", 1)))
    if deficit != expected:
        raise ValueError(
            f"Ring budget {deficit} does not match the {expected} a genus-"
            f"{stats.get('genus')} surface owes: the mesh is torn, not merely "
            f"strained. Census {dict(sorted(counts.items()))}."
        )

    atoms.info.update({
        "structure_type": "periodic_coil",
        "coil_radius": coil_radius,
        "pitch": pitch,
        "tube_radius": tube_radius,
        "handedness": int(sign),
        "bond": bond,
        "period_axis": 2,
        "cell": [float(v) for v in cell],
        "mesh_genus": int(stats.get("genus", -1)),
        "genus": int(stats.get("genus", -1)),
        "ring_deficit": int(deficit),
        "euler": 1 - int(stats.get("genus", 1)),
        "ring_counts": {int(k): int(v) for k, v in sorted(counts.items())},
        "rings": [[int(a) for a in r] for r in rings],
        "bonds": [[int(a), int(b)] for a, b in bonds],
        "geometry": quality,
        "seed": seed,
    })
    return atoms
