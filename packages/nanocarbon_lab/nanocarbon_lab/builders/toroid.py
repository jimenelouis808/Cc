"""Carbon toroids: the structure the disclination rule is really about.

A toroidal carbon molecule -- a nanotube bent until its two ends meet --
is genus 1, so ``sum(6 - n) = 6 * chi = 0``: **every pentagon must be
paired with a heptagon**, exactly, and no other ring size is needed. That
makes it the cleanest possible test of the claim the coil papers rest on,
because a torus is the surface they are actually describing. Its outer
equator has positive Gaussian curvature and its inner one negative, so
the pentagons belong outside and the heptagons inside, and there is
nowhere else for either to go.

Built through the implicit route -- field, periodic marching cubes,
isotropic remesh, dual -- so the ring sizes are **derived**: nobody
places a pentagon. A toroid meshed at R = 20 Å, r = 5 Å comes out with
68 pentagons and 68 heptagons, equal to the digit, against 560 hexagons,
``sum(6-n)`` exactly 0, and 93% of those disclinations on the side the
curvature calls for.

The builder is deliberately thin. `swept.build_swept_tube` already
sweeps a tube along any centreline, and a torus is simply a **closed**
centreline: the capsule sweep that would cap two free ends instead
overlaps itself and closes the surface. Writing a second field for it
would be a second thing to keep right.

**The aspect ratio is the whole of the physics here.** A torus of major
radius ``R`` and minor radius ``r`` bends its wall by ``r / R`` at the
inner equator, so a fat torus around a small hole is strained no matter
how it is relaxed -- and below ``R = 2r`` the hole closes entirely and
the shape is a sphere with a dimple. The published toroidal carbons sit
near ``R/r`` of 3 to 6; the builder refuses under 2.5 and says so.
"""

from __future__ import annotations

import warnings

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D

#: Below this ``R/r`` the hole is smaller than the tube is thick and what
#: comes out is a dimpled sphere rather than a torus. Refused.
MIN_ASPECT = 2.5

#: Published toroidal carbons sit in this band. Outside it the structure
#: is still built -- it is not wrong, only not the geometry those
#: calculations relaxed to -- and the builder says which.
LITERATURE_ASPECT = (3.0, 6.0)

#: Centreline samples. A torus closes on itself, so unlike the coil there
#: is no seam to weld and no need for the sample count to be coprime with
#: anything; it only has to resolve the curve.
PATH_SAMPLES = 400


def build_toroid(
    major_radius: float = 20.0,
    minor_radius: float = 5.0,
    bond: float = CC_BOND,
    voxel: float | None = None,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 0,
    roughness: float = 0.0,
    relax_iterations: int = 3000,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
) -> Atoms:
    """Build a closed carbon torus.

    Parameters
    ----------
    major_radius
        Radius of the ring, centre to tube axis (Å).
    minor_radius
        Radius of the tube itself (Å). Free rather than quantised,
        because the wall is meshed rather than rolled.
    anneal_sweeps
        Defaults to 0 and should stay there. On a curved surface the 5-7
        pairs **are** how a hexagonal net covers the curvature; annealing
        them away leaves the survivors to carry all of it.
    bond, roughness, relax_iterations, vacuum, seed
        As for :func:`~nanocarbon_lab.builders.swept.build_swept_tube`.

    Returns
    -------
    ase.Atoms
        A finite molecule, with the ring census, the aspect ratio and the
        disclination placement in ``atoms.info``.

    Raises
    ------
    ValueError
        If either radius is non-positive, or the aspect ratio is below
        :data:`MIN_ASPECT` -- where the hole has closed and the result is
        not a torus.

    Warns
    -----
    If the aspect ratio falls outside :data:`LITERATURE_ASPECT`, or if
    the finished census is not an equal number of pentagons and
    heptagons: a torus is genus 1, so ``sum(6-n)`` must be 0, and any
    other answer means the mesh did not close as a torus.
    """
    if major_radius <= 0 or minor_radius <= 0:
        raise ValueError("both radii must be positive.")
    aspect = major_radius / minor_radius
    if aspect < MIN_ASPECT:
        raise ValueError(
            f"R/r = {aspect:.2f} is below {MIN_ASPECT}: a {minor_radius:.1f} Å "
            f"tube bent to a {major_radius:.1f} Å ring leaves a hole smaller "
            "than the tube is thick, and what comes out is a dimpled sphere "
            "rather than a torus. Widen the ring or narrow the tube."
        )

    from .swept import build_swept_tube

    angle = np.linspace(0.0, 2.0 * np.pi, PATH_SAMPLES, endpoint=False)
    path = np.column_stack([major_radius * np.cos(angle),
                            major_radius * np.sin(angle),
                            np.zeros_like(angle)])
    # Closing the path is what turns the swept tube's two caps into one
    # continuous surface: the capsules at the join overlap instead of
    # ending.
    path = np.vstack([path, path[:1]])

    atoms = build_swept_tube(
        path, tube_radius=minor_radius, bond=bond, voxel=voxel,
        remesh_iterations=remesh_iterations, anneal_sweeps=anneal_sweeps,
        roughness=roughness, relax_iterations=relax_iterations,
        vacuum=vacuum, pin_ends=False, seed=seed,
    )

    counts = atoms.info.get("ring_counts", {})
    deficit = sum((6 - size) * count for size, count in counts.items())
    atoms.info.update({
        "builder": "toroid",
        "structure_type": "toroid",
        "major_radius": float(major_radius),
        "minor_radius": float(minor_radius),
        "aspect_ratio": round(float(aspect), 3),
        "literature_aspect": list(LITERATURE_ASPECT),
        # r/R is the bend the wall takes at the inner equator, and it is
        # the number that says whether this torus is strained.
        "inner_bend_strain": round(float(minor_radius / major_radius), 4),
    })

    if deficit != 0:
        warnings.warn(
            f"The census {dict(sorted(counts.items()))} gives sum(6-n) = "
            f"{deficit:+d}, but a torus is genus 1 and the budget is exactly "
            "0. The mesh did not close as a torus -- most likely the tube "
            "merged across the hole. Raise the aspect ratio.",
            stacklevel=2,
        )
    low, high = LITERATURE_ASPECT
    if not low <= aspect <= high:
        warnings.warn(
            f"R/r = {aspect:.2f} is outside the {low}-{high} band the "
            "published toroidal carbons sit in. The structure is not wrong "
            "-- it is simply not the geometry those calculations relaxed "
            "to, and its strain is correspondingly different.",
            stacklevel=2,
        )
    return atoms


def describe_toroid(atoms: Atoms) -> str:
    """One line: what was built and whether it obeys the budget."""
    info = atoms.info
    counts = info.get("ring_counts", {})
    census = ", ".join(f"{s}:{c}" for s, c in sorted(counts.items()))
    deficit = sum((6 - s) * c for s, c in counts.items())
    check = info.get("disclination_check") or {}
    agreement = check.get("agreement")
    placed = ("" if agreement is None
              else f", {100 * agreement:.0f}% of disclinations on the "
                   "curvature side they belong on")
    return (
        f"toroid R={info.get('major_radius', 0):.1f} r="
        f"{info.get('minor_radius', 0):.1f} Å (R/r "
        f"{info.get('aspect_ratio', 0):.2f}): {len(atoms)} atoms, rings "
        f"{census}, sum(6-n) = {deficit:+d}{placed}."
    )


__all__ = ["LITERATURE_ASPECT", "MIN_ASPECT", "PATH_SAMPLES", "build_toroid",
           "describe_toroid"]
