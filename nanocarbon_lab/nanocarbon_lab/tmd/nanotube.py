"""Single-walled MX2 nanotubes, rolled from a monolayer.

Rolling a TMD is not rolling a graphene sheet. Graphene is one atom
thick, so wrapping it onto a cylinder costs only bond-angle bending. A
TMD is a sandwich of finite thickness ``h``, and wrapping it puts the
outer chalcogen plane on a larger circle than the inner one. The strain
is geometric and unavoidable:

    outer plane   +h / (2R)
    inner plane   -h / (2R)

For MoS2 (``h`` = 3.13 Å) a 20 Å radius tube already stretches its outer
sulphur plane by 7.8%. This is the reason real MoS2 nanotubes are tens of
nanometres across rather than the ~1 nm typical of carbon: the same
curvature costs far more. The builder reports the strain and warns when
it leaves the range where the structure is a sensible starting point.

Atoms are placed by exact rolling, without relaxation, for the reason
given in :mod:`nanocarbon_lab.tmd.slab` -- this is a starting geometry
for DFT or a Stillinger-Weber MD run, and pre-relaxing it with an
approximate field would only add error.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
from ase import Atoms

from ..utils.constants import DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from .materials import Phase, TMDMaterial, coordination_geometry, get_material
from .slab import build_tmd_layers

Chirality = Literal["armchair", "zigzag", "chiral"]

# Outer-plane strain past which a rolled TMD stops being a reasonable
# starting structure. Real MoS2 nanotubes sit well below this; above it
# the M-X bonds are distorted enough that a relaxation will move every
# atom substantially.
MAX_ROLL_STRAIN = 0.10


def chirality_of(n: int, m: int) -> Chirality:
    """Name the chirality family of an ``(n, m)`` tube."""
    if m == 0:
        return "zigzag"
    if n == m:
        return "armchair"
    return "chiral"


def tube_radius(material: TMDMaterial, n: int, m: int) -> float:
    """Radius of the metal cylinder (Å) for chiral indices ``(n, m)``."""
    return material.a * math.sqrt(n * n + n * m + m * m) / (2.0 * math.pi)


def build_tmd_nanotube(
    material: str | TMDMaterial = "MoS2",
    n: int = 20,
    m: int = 0,
    length: int = 1,
    phase: Phase = "2H",
    vacuum: float = DEFAULT_VACUUM_1D,
    max_strain: float = MAX_ROLL_STRAIN,
) -> Atoms:
    """Roll a monolayer into a single-walled ``(n, m)`` nanotube.

    Parameters
    ----------
    material
        Formula or :class:`~nanocarbon_lab.tmd.materials.TMDMaterial`.
    n, m
        Chiral indices, ``n >= 1`` and ``0 <= m <= n``. ``(n, 0)`` is
        zigzag, ``(n, n)`` armchair, anything else chiral.
    length
        Translational periods along the tube axis (>= 1).
    phase
        Layer phase before rolling.
    vacuum
        Transverse vacuum padding (Å) beyond the tube's outer surface.
    max_strain
        Outer-plane strain above which a warning is issued. The tube is
        still built -- a strained tube is a legitimate thing to want --
        but silently returning one would hide the reason a subsequent
        relaxation moves every atom.

    Returns
    -------
    ase.Atoms
        Periodic along z only, with the radius, chirality, both plane
        radii and the measured strain in ``info``.

    Raises
    ------
    ValueError
        For indices outside ``n >= 1``, ``0 <= m <= n``, or a
        non-positive length.

    Warns
    -----
    UserWarning
        When the outer-plane strain exceeds ``max_strain``.
    """
    if isinstance(material, str):
        material = get_material(material)
    if n < 1 or m < 0 or m > n:
        raise ValueError(
            f"Need n >= 1 and 0 <= m <= n; got (n, m) = ({n}, {m})."
        )
    if length < 1:
        raise ValueError("length must be >= 1.")

    radius = tube_radius(material, n, m)
    strain = material.h / (2.0 * radius)
    if strain > max_strain:
        warnings.warn(
            f"A ({n},{m}) {material.formula} tube has radius {radius:.1f} Å, so "
            f"rolling stretches its outer {material.chalcogen} plane by "
            f"{strain:.1%} and compresses the inner one by as much. Real "
            f"{material.formula} nanotubes are tens of nm across because the "
            "sandwich is ~3 Å thick and that cost scales as h/2R — raise n for "
            "a physically comfortable tube.",
            UserWarning,
            stacklevel=2,
        )

    sheet, chiral, axial = _unrolled_patch(material, phase, n, m, length)
    positions = sheet.get_positions()

    # Roll: distance along the chiral vector becomes arc length at the
    # atom's own radius, which is what makes the three planes land on
    # three different cylinders.
    origin_z = positions[:, 2].mean()
    height = positions[:, 2] - origin_z
    around = positions[:, :2] @ (chiral / np.linalg.norm(chiral))
    along = positions[:, :2] @ (axial / np.linalg.norm(axial))

    angle = around / radius
    rolled_radius = radius + height
    rolled = np.column_stack([
        rolled_radius * np.cos(angle),
        rolled_radius * np.sin(angle),
        along,
    ])

    period = float(np.linalg.norm(axial)) / length
    outer = radius + material.h / 2.0
    diameter = 2.0 * outer
    tube = Atoms(
        symbols=sheet.get_chemical_symbols(),
        positions=rolled,
        cell=np.diag([diameter + vacuum, diameter + vacuum, period * length]),
        pbc=(False, False, True),
    )
    center_in_cell(tube, axes=(0, 1, 2))

    tube.info.update(
        {
            "structure_type": "tmd_nanotube",
            "material": material.formula,
            "metal": material.metal,
            "chalcogen": material.chalcogen,
            "phase": phase,
            "coordination": coordination_geometry(phase),
            "chiral_indices": (n, m),
            "chirality": chirality_of(n, m),
            "radius": radius,
            "inner_radius": radius - material.h / 2.0,
            "outer_radius": outer,
            "diameter": diameter,
            "roll_strain": strain,
            "period": period,
            "length_cells": length,
            "a": material.a,
            "h": material.h,
            "bond_length": material.bond_length,
        }
    )
    return tube


def _unrolled_patch(material: TMDMaterial, phase: Phase, n: int, m: int,
                    length: int) -> tuple[Atoms, np.ndarray, np.ndarray]:
    """The rectangular patch of monolayer that rolls into one tube cell.

    Returns the atoms plus the chiral and axial vectors in Cartesian
    coordinates. The patch is cut from a generously oversized supercell
    rather than constructed on its own basis: the sublattice and phase
    logic then stays in :func:`~nanocarbon_lab.tmd.slab.build_tmd_layers`
    and cannot drift out of step with it.
    """
    # 60-degree basis, in which the textbook (n, m) formulas hold. The
    # slab builder uses the 120-degree convention, so the two differ by
    # b2 = a1 + a2; working in Cartesian avoids having to track that.
    a = material.a
    b1 = np.array([a, 0.0])
    b2 = np.array([a / 2.0, a * math.sqrt(3.0) / 2.0])

    chiral = n * b1 + m * b2
    divisor = math.gcd(2 * n + m, 2 * m + n)
    axial = ((2 * m + n) // divisor) * b1 - ((2 * n + m) // divisor) * b2
    axial = axial * length

    # A supercell certain to contain the whole parallelogram. The corners
    # of C + T in lattice units bound how far we must tile.
    reach = 2 * (abs(n) + abs(m) + 1) * max(1, length) + 4
    sheet = build_tmd_layers(material, n_layers=1, phase=phase,
                             nx=reach, ny=reach, vacuum=20.0)
    positions = sheet.get_positions()
    # Re-centre the tiling on the origin so the patch is surrounded.
    centre = positions[:, :2].mean(axis=0)
    positions[:, :2] -= centre
    positions[:, :2] += (chiral + axial) / 2.0
    sheet.set_positions(positions)

    basis = np.column_stack([chiral, axial])
    fractional = np.linalg.solve(basis, positions[:, :2].T).T
    tol = 1e-9
    inside = np.all((fractional >= -tol) & (fractional < 1.0 - tol), axis=1)
    if not inside.any():
        raise RuntimeError(
            "Rolling patch came out empty; the supercell did not cover the "
            "chiral cell. This is a bug, not a parameter problem."
        )
    return sheet[inside], chiral, axial


__all__ = [
    "MAX_ROLL_STRAIN",
    "Chirality",
    "build_tmd_nanotube",
    "chirality_of",
    "tube_radius",
]
