"""Helical MX2 nanotubes -- a nanotube swept onto a coiled centreline.

Carbon has two routes to a coil and they are not interchangeable (see
``builders/swept.py``): sweep a finished all-hexagon tube and carry the
bend as elastic strain, or mesh the curved surface implicitly so that ring
sizes follow the curvature. The second route buys its clean bonds by
letting pentagons and heptagons appear where the surface saddles.

For a dichalcogenide only the first route is available, and that is a
chemical fact rather than a missing feature. M and X alternate around
every ring, so odd rings would force an M-M or X-X bond; the implicit
route's whole mechanism for absorbing curvature is therefore closed off.
Sweeping keeps the rolled tube's topology exactly -- every ring stays a
hexagon, every bond stays M-X -- and pays for the bend in strain.

That is the honest trade here anyway. A TMD sandwich is ~3 Å thick, so it
is far stiffer than a one-atom-thick graphene sheet, and real MoS2 tubes
are tens of nm across. Bending one is elastic deformation of a stiff tube,
not a rewiring of its lattice.

Two strains stack, and the builder reports both because they have
different cures:

    roll   h / (2R_tube)             set by the tube's own radius
    bend   R_outer * kappa           set by how tightly it is coiled

with ``kappa = R / (R^2 + (pitch/2pi)^2)`` the curvature of the helix.
Widening the tube cuts the first and raises the second, so they cannot be
minimised together -- the coil radius is the free parameter.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from ase import Atoms

from ..builders.centerline import (
    arclength_sampler,
    helix_control_points,
    sweep_along_path,
)
from ..utils.constants import DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from .materials import Phase, TMDMaterial, coordination_geometry, get_material
from .nanotube import MAX_ROLL_STRAIN, build_tmd_nanotube, chirality_of

#: Outer-wall bending strain past which the coil stops being a sensible
#: starting structure. Deliberately looser than the carbon builder's 8%
#: budget is tight, because a TMD tube is thick and any coil worth
#: rendering already spends most of its budget on the roll.
MAX_BEND_STRAIN = 0.08


def helix_curvature(coil_radius: float, pitch: float) -> float:
    """Curvature (1/Å) of a helix of the given radius and rise per turn."""
    if coil_radius <= 0:
        raise ValueError("coil_radius must be positive.")
    reduced = pitch / (2.0 * math.pi)
    return coil_radius / (coil_radius**2 + reduced**2)


def build_tmd_coil(
    material: str | TMDMaterial = "MoS2",
    n: int = 30,
    m: int = 0,
    coil_radius: float = 220.0,
    pitch: float = 90.0,
    turns: float = 0.5,
    phase: Phase = "2H",
    handedness: int = 1,
    vacuum: float = DEFAULT_VACUUM_1D,
    max_strain: float = MAX_ROLL_STRAIN,
    max_bend_strain: float = MAX_BEND_STRAIN,
) -> Atoms:
    """Coil a single-walled ``(n, m)`` MX2 nanotube onto a helix.

    One translational period of the tube is built and tiled along its
    axis, then bent onto the helix with a rotation-minimizing frame. The
    number of periods is chosen so the straight tube is as close as a
    whole period allows to the helix arc length, because
    :func:`~nanocarbon_lab.builders.centerline.sweep_along_path`
    rescales the path to the structure rather than stretching the
    structure to the path. The residual rescale is under one period, and
    the **achieved** radius and pitch are reported in ``info`` rather than
    the requested ones.

    Parameters
    ----------
    material
        Formula or :class:`~nanocarbon_lab.tmd.materials.TMDMaterial`.
    n, m
        Chiral indices of the tube being coiled, ``n >= 1``, ``0 <= m <= n``.
    coil_radius, pitch, turns
        Helix radius and rise per turn in Å, and how many turns.
    phase
        Layer phase before rolling.
    handedness
        ``+1`` right-handed, ``-1`` left-handed.
    vacuum
        Padding (Å) beyond the coil's bounding box.
    max_strain, max_bend_strain
        Warning thresholds for the roll and bend strains. The coil is
        still built past them -- both are geometric consequences of what
        was asked for, not errors -- but returning one silently would
        hide why a later relaxation moves every atom.

    Returns
    -------
    ase.Atoms
        Non-periodic, with both strains, the achieved helix geometry and
        the tube's own parameters in ``info``.

    Raises
    ------
    ValueError
        For non-positive ``turns``, or indices the tube builder rejects.

    Warns
    -----
    UserWarning
        When either strain exceeds its threshold.
    """
    if isinstance(material, str):
        material = get_material(material)
    if turns <= 0:
        raise ValueError("turns must be positive.")
    if handedness not in (1, -1):
        raise ValueError("handedness must be +1 (right-handed) or -1 (left).")

    # One period, then tile: building the long tube directly would ask
    # `_unrolled_patch` for a supercell of thousands of cells per side.
    unit = build_tmd_nanotube(material, n=n, m=m, length=1, phase=phase,
                              vacuum=vacuum, max_strain=max_strain)
    period = float(unit.info["period"])

    control = helix_control_points(coil_radius, pitch, turns,
                                   handedness=handedness)
    _, arc_length = arclength_sampler(control)
    periods = max(1, int(round(arc_length / period)))

    tube = unit.repeat((1, 1, periods))
    positions = tube.get_positions()
    positions[:, 2] -= positions[:, 2].min()
    span = float(positions[:, 2].max())

    # sweep_along_path normalises the path to the structure's own length,
    # so state what that does to the requested geometry instead of
    # quietly reporting numbers the structure does not have.
    scale = span / arc_length if arc_length > 0 else 1.0
    achieved_radius = coil_radius * scale
    achieved_pitch = pitch * scale

    curvature = helix_curvature(achieved_radius, achieved_pitch)
    outer = float(unit.info["outer_radius"])
    bend_strain = outer * curvature
    roll_strain = float(unit.info["roll_strain"])

    if bend_strain > max_bend_strain:
        warnings.warn(
            f"Coiling a ({n},{m}) {material.formula} tube at radius "
            f"{achieved_radius:.0f} Å strains its outer wall by "
            f"{bend_strain:.1%} on top of the {roll_strain:.1%} already "
            "spent on rolling. Bend strain is R_outer * kappa, so widen the "
            "coil or narrow the tube — but narrowing the tube raises the "
            "roll strain, which is why real MX2 coils are large.",
            UserWarning,
            stacklevel=2,
        )

    swept = sweep_along_path(positions, control)
    coil = Atoms(symbols=tube.get_chemical_symbols(), positions=swept,
                 pbc=False)
    extent = swept.max(axis=0) - swept.min(axis=0)
    coil.set_cell(np.diag(extent + 2.0 * vacuum))
    center_in_cell(coil, axes=(0, 1, 2))

    coil.info.update(
        {
            "structure_type": "tmd_coil",
            "material": material.formula,
            "metal": material.metal,
            "chalcogen": material.chalcogen,
            "phase": phase,
            "coordination": coordination_geometry(phase),
            "chiral_indices": (n, m),
            "chirality": chirality_of(n, m),
            "tube_radius": float(unit.info["radius"]),
            "outer_radius": outer,
            "roll_strain": roll_strain,
            "bend_strain": bend_strain,
            "total_strain": roll_strain + bend_strain,
            "curvature": curvature,
            "coil_radius": achieved_radius,
            "pitch": achieved_pitch,
            "requested_coil_radius": coil_radius,
            "requested_pitch": pitch,
            "turns": turns,
            "handedness": handedness,
            "periods": periods,
            "period": period,
            "arc_length": span,
            "a": material.a,
            "h": material.h,
            "bond_length": material.bond_length,
        }
    )
    return coil


__all__ = ["MAX_BEND_STRAIN", "build_tmd_coil", "helix_curvature"]
