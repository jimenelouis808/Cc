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
    place_curvature: bool = False,
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
    place_curvature
        Move the disclinations to where the surface's own Gaussian
        curvature asks for them, by Stone-Wales flips, after the remesh.
        It cannot change HOW MANY there are, only where they sit.
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
        place_curvature=place_curvature,
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


#: Outer-wall strain a bent finished lattice may carry, as a fraction.
#: Bending can only STRETCH a polyhex -- there are no disclinations to
#: relieve it with -- so this is the same budget the nanocoil uses, and
#: for the same reason.
POLYHEX_MAX_STRAIN = 0.08

#: Above this the wall is torn rather than strained. Measured on a
#: (10,10) at 60 periods, r/R = 28.9%: the census came back with 360
#: three-membered and 120 four-membered rings, which is the tube folded
#: through itself, not a lattice under load.
POLYHEX_TEAR_STRAIN = 0.15


def build_polyhex_toroid(
    n: int = 5,
    m: int = 5,
    periods: int = 110,
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
) -> Atoms:
    """Bend a finished ``(n, m)`` lattice into a closed ring: **all hexagons**.

    The other route in this module meshes a torus implicitly, so the
    remesher picks whatever ring sizes the curvature calls for and the
    wall comes back with matched 5-7 pairs -- 68 of each at R=20, r=5.
    That is a *sound* torus and an **amorphous** one. It is not what the
    published pictures of toroidal carbon show.

    A torus is genus 1, so ``sum(6-n) = 0`` is satisfied by an
    all-hexagon net with no disclinations at all -- a **toroidal
    polyhex**, and it is reachable by bending a real nanotube until its
    ends meet. `build_cnt` returns an exact number of translational
    periods, so a ring of that circumference closes seam to seam with no
    defect at the join: measured on a (5,5) at 110 periods, 2200 atoms
    and a census of **{6: 1100}** with zero boundary edges.

    **The price is size, and it is not negotiable.** Bending a finished
    lattice can only stretch it -- there is no disclination to relieve
    the curvature with, which is exactly what the meshed route buys with
    its 5-7 pairs -- so the outer wall carries ``r / R``. A (5,5) tube is
    3.39 Å in radius, so an 8% budget puts the ring at 43 Å and 2200
    atoms. That is why the small round toroids in the literature are
    *not* polyhexes: they use knees with a pentagon outside and a
    heptagon inside, which is a third route this module does not yet have.

    Parameters
    ----------
    n, m
        Chiral indices of the tube to bend.
    periods
        Translational periods around the ring. This sets the
        circumference, so it is what picks the radius -- and the strain.
    bond, vacuum
        C-C length and the padding around the finished molecule (Å).

    Returns
    -------
    ase.Atoms
        A finite molecule, all hexagons, with the strain and the two
        radii in ``atoms.info``.

    Raises
    ------
    ValueError
        If the outer-wall strain exceeds :data:`POLYHEX_TEAR_STRAIN`,
        where the wall is torn rather than loaded.

    Warns
    -----
    Past :data:`POLYHEX_MAX_STRAIN`, because a strained polyhex is a real
    structure that no relaxation will improve -- the stretch is
    geometric, not a relaxation failure.
    """
    from .cnt import build_cnt

    if periods < 3:
        raise ValueError("a ring needs at least 3 periods.")
    unit = build_cnt(n=n, m=m, length=1.0, bond=bond, axis=2)
    period = float(unit.info["period_length"])
    tube_radius = float(unit.info["radius"])
    # 0.999 so `length` does not round up to one period too many: the
    # builder returns the smallest whole number of periods of at least
    # `length`, and asking for exactly `periods * period` can tip over.
    tube = build_cnt(n=n, m=m, length=period * periods * 0.999, bond=bond,
                     axis=2)
    circumference = float(tube.cell[2][2])
    major = circumference / (2.0 * np.pi)
    strain = tube_radius / major

    if strain > POLYHEX_TEAR_STRAIN:
        raise ValueError(
            f"A ({n},{m}) tube is {tube_radius:.2f} Å in radius, and "
            f"{periods} periods make a ring of only {major:.1f} Å, so the "
            f"outer wall would stretch by {100 * strain:.1f}% -- past the "
            f"{100 * POLYHEX_TEAR_STRAIN:.0f}% where it tears rather than "
            "loads. Bending a finished lattice can only stretch it, so "
            "raise `periods` or narrow the tube. (A small round toroid "
            "needs knees with a pentagon outside and a heptagon inside, "
            "which is what the meshed route approximates.)"
        )

    positions = tube.get_positions()
    positions[:, 0] -= positions[:, 0].mean()
    positions[:, 1] -= positions[:, 1].mean()
    angle = 2.0 * np.pi * positions[:, 2] / circumference
    bent = np.column_stack([
        (major + positions[:, 0]) * np.cos(angle),
        (major + positions[:, 0]) * np.sin(angle),
        positions[:, 1],
    ])

    atoms = Atoms(symbols=["C"] * len(bent), positions=bent,
                  pbc=(False, False, False))
    span = bent.max(axis=0) - bent.min(axis=0)
    atoms.set_cell(span + 2.0 * vacuum)
    atoms.center()

    from ..analyse.curvature import disclination_check
    from ..analyse.rings import trace_faces
    from ..utils.geometry import guess_bonds
    from .capped_cnt import geometry_report

    pairs = [(int(i), int(j)) for i, j, _ in guess_bonds(atoms)]
    faces, boundary = trace_faces(atoms, np.asarray(pairs, dtype=int),
                                  max_size=12)
    census: dict[int, int] = {}
    for face in faces:
        census[len(face)] = census.get(len(face), 0) + 1
    deficit = sum((6 - size) * count for size, count in census.items())

    atoms.info.update({
        "builder": "polyhex_toroid",
        "structure_type": "toroid",
        "route": "polyhex",
        "n": int(n), "m": int(m), "periods": int(tube.info["n_periods"]),
        "major_radius": round(float(major), 3),
        "minor_radius": round(float(tube_radius), 3),
        "aspect_ratio": round(float(major / tube_radius), 3),
        "outer_wall_strain": round(float(strain), 4),
        "circumference": round(circumference, 3),
        "bond": float(bond),
        "bonds": pairs,
        "rings": [[int(i) for i in f] for f in faces],
        "ring_counts": dict(sorted(census.items())),
        "euler": deficit // 6 if deficit else 0,
        "genus": 1,
        "boundary_walks": int(boundary),
        "geometry": geometry_report(bent, pairs),
        "disclination_check": disclination_check(bent, faces, pairs),
    })

    if boundary:
        warnings.warn(
            f"The ring traced {boundary} boundary walk(s), so the seam did "
            "not close. The tube's periodic length should divide the "
            "circumference exactly; this means it did not.",
            stacklevel=2,
        )
    if set(census) != {6}:
        warnings.warn(
            f"The census is {dict(sorted(census.items()))}, not all "
            "hexagons. Bending a finished lattice cannot change a ring, so "
            "this is the wall having torn and `trace_faces` finding the "
            "wreckage. Raise `periods`.",
            stacklevel=2,
        )
    if strain > POLYHEX_MAX_STRAIN:
        warnings.warn(
            f"The outer wall is stretched by {100 * strain:.1f}%, past the "
            f"{100 * POLYHEX_MAX_STRAIN:.0f}% budget. This is geometry, not "
            "an unfinished relaxation: an all-hexagon ring has no "
            "disclination to relieve curvature with, so nothing will "
            f"shorten those bonds. Raise `periods` (R is now {major:.1f} Å) "
            "or use the meshed route, which pays in 5-7 pairs instead.",
            stacklevel=2,
        )
    return atoms


__all__ = ["LITERATURE_ASPECT", "MIN_ASPECT", "PATH_SAMPLES",
           "POLYHEX_MAX_STRAIN", "POLYHEX_TEAR_STRAIN", "build_polyhex_toroid",
           "build_toroid", "describe_toroid"]
