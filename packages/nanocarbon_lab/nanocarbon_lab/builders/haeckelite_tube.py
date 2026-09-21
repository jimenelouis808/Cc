"""Haeckelite nanotubes: the flat lattice rolled, not re-derived.

A haeckelite sheet is graphene with pentagons and heptagons patterned
into it (:mod:`~nanocarbon_lab.builders.haeckelite`). Rolling one into a
tube is what Terrones et al. proposed alongside the flat lattices in the
paper that named them -- *New Metallic Allotropes of Planar and Tubular
Graphite*, Phys. Rev. Lett. **84**, 1716 (2000) -- and the tubular forms
are the half of that paper this framework did not have.

**A cylinder is developable, and that is the whole argument.** Its
Gaussian curvature is zero everywhere, so wrapping a flat sheet onto one
is an *isometry*: no ring has to change size, nothing has to be
re-meshed, and the census the flat lattice relaxed to is carried over
atom for atom. That is what separates this from every other curved
builder here. A sphere or a saddle has non-zero curvature and *must* pay
for it in pentagons or heptagons -- which is why the fullerene, the
schwarzite, the junction and the supernetwork all derive their rings
from a mesh. A tube pays nothing, so deriving its rings again would only
be an opportunity to get them wrong.

Three consequences worth stating, because each is checkable:

* **The budget stays zero.** A tube periodic along its axis is a torus,
  exactly as the flat periodic sheet was, so ``sum(6 - n) = 0`` on both
  sides. The builder traces the rolled tube's own faces and refuses
  anything else.
* **The circumference is exact by construction.** It is the sheet's own
  relaxed cell edge, wrapped once. There is no ``(n, m)`` quantisation
  to search for and no chiral angle to snap to: the lattice that closes
  around the tube is the lattice that was built, and the radius is
  whatever that circumference implies.
* **The radius is an output, and which way it moves is not obvious.**
  Rolling replaces every circumferential arc by its chord, shortening
  each such bond by about ``l^2 / (24 R^2)``, so the waist starts under
  compression and an all-hexagon tube duly relaxes *outwards*: the
  control case here is a (8,0), rolled at 3.13 Å and relaxed to 3.19.
  A patterned lattice does the opposite -- r57 rolled at 4.89 Å settles
  at 4.72, and at 8.00 Å settles at 7.65 -- because its pentagons and
  heptagons let the wall relieve the same compression by contracting
  the axis instead, which it does by 4.5% and 8.5% respectively. The
  chord term is 0.35% and 0.13% in those two cases, so it is not what
  is moving them. Neither radius is held; both are reported, for the
  reason :mod:`~nanocarbon_lab.builders.swept` reports achieved pitch.

What the force field here cannot price is the rehybridisation: curving
an sp2 sheet misaligns its pi orbitals, and that -- not bond strain --
is what makes a narrow tube expensive. A valence force field with bond,
angle and repulsion terms sees almost nothing of it (the chord effect at
R = 4 Å is 0.5%, which no relaxation will complain about). So the radius
floor here is taken from what has been observed rather than from what
this relaxer reports, and the docstring says so rather than implying the
number was computed.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from . import fullerene_mesh as fm
from .haeckelite import (
    ANGLE_CEILING,
    ANGLE_FLOOR,
    BOND_CEILING,
    BOND_FLOOR,
    BOND_SOFT_FLOOR,
    build_haeckelite,
)

#: Radius (Å) below which the builder refuses. The narrowest carbon
#: nanotubes ever observed sit near 2 Å, grown inside a template that
#: held them there; below it nothing has been made. This is an observed
#: limit, not one this relaxer measured -- a valence force field barely
#: notices curvature, so it would happily return a 1 Å tube.
MIN_TUBE_RADIUS = 2.0

#: Radius below which the tube is built and warned about. Under roughly
#: 3 Å -- narrower than a (5,5) -- pyramidalisation is large enough that
#: the geometry this returns is a starting point for a real calculator
#: rather than a prediction.
WARN_TUBE_RADIUS = 3.0

#: Axial-period scan, as a fraction either side of the flat sheet's own
#: period, and the step. The axis is the one direction the roll leaves
#: constrained by the cell, so it is the one that has to be re-fitted;
#: the radius re-fits itself, the atoms being free in x and y.
AXIAL_SEARCH = 0.04
AXIAL_STEP = 0.005

#: How far the scan may widen when its best candidate lands on an end.
#: **A search pinned at its own bound is reporting the bound**, not a
#: minimum, and this one pinned on every patterned lattice tried: the
#: first +-4% window returned 0.96 exactly -- its lower edge -- for r57
#: rolled either way. Rolling shortens the axial period by more than the
#: flat sheet's cell fit suggests, so the window is widened toward the
#: end that won until the minimum is interior. Past this limit the answer
#: is not a fit at all and the builder says so rather than quoting an
#: edge.
AXIAL_LIMIT = 0.25


def _score(positions: np.ndarray, pairs: np.ndarray, box: np.ndarray,
           bond: float) -> float:
    """Mean squared bond strain plus mean squared angle deviation.

    The tube's counterpart of
    :func:`~nanocarbon_lab.builders.haeckelite.strain_score`, weighted
    the same way and for the same reason: bond strain alone cannot pick
    an axial period, because a sheet under axial compression buckles
    instead of compressing its bonds and the bonds stay at 1.42 Å while
    the cell collapses. Buckling costs angle energy, and that is what
    the second term sees.
    """
    first, second = pairs[:, 0], pairs[:, 1]
    lengths = np.linalg.norm(
        fm.minimum_image(positions[second] - positions[first], box), axis=1)
    bond_term = float(np.mean(((lengths - bond) / bond) ** 2))

    table: list[list[int]] = [[] for _ in range(len(positions))]
    for a, b in pairs:
        table[a].append(b)
        table[b].append(a)
    target = math.radians(120.0)
    deviations: list[float] = []
    for centre, neighbours in enumerate(table):
        for i in range(len(neighbours)):
            for j in range(i + 1, len(neighbours)):
                u = fm.minimum_image(positions[neighbours[i]] - positions[centre], box)
                v = fm.minimum_image(positions[neighbours[j]] - positions[centre], box)
                cosine = float(np.dot(u, v)
                               / (np.linalg.norm(u) * np.linalg.norm(v)))
                deviations.append(math.acos(np.clip(cosine, -1.0, 1.0)) - target)
    angle_term = float(np.mean(np.square(deviations))) if deviations else 0.0

    from .capped_cnt import geometry_report

    report = geometry_report(positions, [tuple(p) for p in pairs.tolist()],
                             box=box)
    # A tube that has folded flat or through itself can have excellent
    # bonds and angles -- both are local, and collapse is not. The
    # framework's own rule in reverse: zero close contacts does not prove
    # a structure sound, but one close contact proves it is not.
    if int(report["n_close_contacts"]):
        return float("inf")
    return 40.0 * bond_term + 15.0 * angle_term


def roll_to_tube(positions: np.ndarray, circumference: float,
                 wrap_axis: int, axis: int) -> tuple[np.ndarray, float]:
    """Wrap a flat sheet once around a cylinder.

    ``wrap_axis`` becomes the circumference and ``axis`` the tube axis;
    the third direction was the sheet's vacuum and is consumed. Returns
    the cylindrical positions with the tube axis along z, and the radius
    the circumference implies.

    The map is arc length to angle, so it is an isometry in the limit of
    infinitely short bonds. What it is not exact about is the *chord*: a
    bond spanning an arc ``l`` comes out at ``2 R sin(l / 2R)``, shorter
    by about ``l^2 / (24 R^2)``. That compression is real and is left in
    rather than corrected, because the relaxation that follows is what
    resolves it -- by expanding the tube, which is the physical answer.
    """
    radius = circumference / (2.0 * math.pi)
    theta = 2.0 * math.pi * positions[:, wrap_axis] / circumference
    out = np.empty_like(positions)
    out[:, 0] = radius * np.cos(theta)
    out[:, 1] = radius * np.sin(theta)
    out[:, 2] = positions[:, axis]
    return out, radius


def build_haeckelite_tube(
    nx: int = 8,
    ny: int = 3,
    pattern: str = "r57",
    roll: str = "a",
    period: int = 2,
    density: float = 0.15,
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
    relax_iterations: int = 3000,
    fit_axial: bool = True,
    seed: int | None = 0,
) -> Atoms:
    """Roll a haeckelite sheet into a periodic nanotube.

    Parameters
    ----------
    nx, ny
        Repeats of graphene's 4-atom rectangular cell in the flat sheet,
        so the tube holds ``4 * nx * ny`` atoms. Whichever of the two
        ``roll`` names sets the circumference, and the other the axial
        period -- so a long thin tube is a large ``nx`` with a small
        ``ny`` rolled along ``"a"``.
    pattern, period, density, seed
        The flat lattice to roll; see
        :func:`~nanocarbon_lab.builders.haeckelite.build_haeckelite`.
        ``"none"`` gives an ordinary all-hexagon tube, which is the
        useful control rather than a degenerate case: it is the same
        code path, so anything it gets wrong is the rolling and not the
        pattern.
    roll
        ``"a"`` wraps the sheet's first cell edge, ``"b"`` the second.
        These are different tubes from the same lattice -- the pattern
        is not isotropic -- and neither is a rotation of the other.
    bond
        Target C-C length (Å).
    vacuum
        Padding (Å) between the tube and its transverse images.
    fit_axial
        Scan the axial period over :data:`AXIAL_SEARCH` either side of
        the flat sheet's own and keep the lowest-strain one. The axis is
        the single degree of freedom the roll leaves the cell holding;
        the radius needs no scan because the atoms are free to move in
        x and y and find it themselves.
    relax_iterations
        L-BFGS iterations per relaxation.

    Returns
    -------
    ase.Atoms
        ``pbc=(False, False, True)``, the tube along z, with the ring
        census, the achieved and rolled radii, the axial period and the
        measured geometry in ``atoms.info``.

    Raises
    ------
    ValueError
        If ``roll`` is not ``"a"`` or ``"b"``; if the circumference
        implies a radius below :data:`MIN_TUBE_RADIUS`; if the rolled
        tube's traced faces disagree with the flat sheet's census, which
        means the roll or the relaxation tore it; or if the relaxed
        geometry leaves the sp2 window.

    Warns
    -----
    If the radius is below :data:`WARN_TUBE_RADIUS`, or if the shortest
    bond is below ``BOND_SOFT_FLOOR`` -- the same force-field limit the
    flat builder warns about, for the same reason.

    Notes
    -----
    The census is **carried, then verified**, not re-derived. Rolling is
    an isometry onto a developable surface, so the flat lattice's rings
    are the tube's rings and re-perceiving them could only introduce an
    error; but the relaxation afterwards can tear, so the finished tube's
    faces are traced and checked against what was carried. The Euler
    budget is zero on both sides, a periodic tube being a torus exactly
    as the flat periodic sheet is.
    """
    if roll not in ("a", "b"):
        raise ValueError(f"roll must be 'a' or 'b', not {roll!r}.")

    sheet = build_haeckelite(nx=nx, ny=ny, pattern=pattern, period=period,
                             density=density, bond=bond, vacuum=vacuum,
                             relax_iterations=relax_iterations, seed=seed)
    flat_cell = sheet.get_cell().lengths()
    wrap_axis, axis = (0, 1) if roll == "a" else (1, 0)
    circumference = float(flat_cell[wrap_axis])
    axial = float(flat_cell[axis])

    radius = circumference / (2.0 * math.pi)
    if radius < MIN_TUBE_RADIUS:
        raise ValueError(
            f"Rolling the {roll!r} edge of a {nx}x{ny} {pattern!r} sheet "
            f"wraps {circumference:.2f} Å around, which is a tube of radius "
            f"{radius:.2f} Å -- below the {MIN_TUBE_RADIUS} Å at which the "
            "narrowest carbon nanotube ever observed sits, and that one was "
            "grown inside a template holding it open. Roll the other edge, "
            f"or raise n{'x' if roll == 'a' else 'y'}."
        )

    pairs = np.asarray(sheet.info["bonds"], dtype=int)[:, :2]
    positions, rolled_radius = roll_to_tube(
        sheet.get_positions(), circumference, wrap_axis, axis)

    # The axial period is the one thing the cell still holds. Scan it,
    # relaxing at each, and keep the best -- the radius looks after
    # itself, the atoms being unconstrained in x and y.
    bond_set = {tuple(p) for p in pairs.tolist()}
    scored: dict[int, tuple[float, np.ndarray]] = {}
    window = int(round(AXIAL_SEARCH / AXIAL_STEP))
    ceiling = int(round(AXIAL_LIMIT / AXIAL_STEP))

    def at(step: int) -> float:
        """Relax at ``1 + step * AXIAL_STEP`` times the flat period."""
        if step not in scored:
            factor = 1.0 + AXIAL_STEP * step
            box = np.array([0.0, 0.0, axial * factor])
            trial = positions.copy()
            trial[:, 2] *= factor
            relaxed = fm.relax_shell(trial, bond_set, equilibrium=bond,
                                     box=box, max_iterations=relax_iterations)
            scored[step] = (_score(relaxed, pairs, box, bond), relaxed)
        return scored[step][0]

    low, high = (-window, window) if fit_axial else (0, 0)
    pinned = False
    while True:
        for step in range(low, high + 1):
            at(step)
        finite = {k: v[0] for k, v in scored.items() if np.isfinite(v[0])}
        if not finite or not fit_axial:
            break
        winner = min(finite, key=lambda k: finite[k])
        # Widen toward whichever end won, and only that end: the minimum
        # is on that side, so scanning the other way buys nothing but
        # relaxations. A minimum strictly inside the window is the point
        # of the loop -- it means the scan found a turning point rather
        # than a wall.
        if winner == low and low > -ceiling:
            low = max(-ceiling, low - window)
            continue
        if winner == high and high < ceiling:
            high = min(ceiling, high + window)
            continue
        pinned = winner in (low, high) and abs(winner) >= ceiling
        break

    finite = {k: v[0] for k, v in scored.items() if np.isfinite(v[0])}
    best: np.ndarray | None = None
    best_score = float("inf")
    best_axial = axial
    if finite:
        winner = min(finite, key=lambda k: finite[k])
        best_score, best = scored[winner]
        best_axial = axial * (1.0 + AXIAL_STEP * winner)

    if pinned:
        raise ValueError(
            f"A {nx}x{ny} {pattern!r} sheet rolled along {roll!r} wants an "
            f"axial period more than {100 * AXIAL_LIMIT:.0f}% from the flat "
            "sheet's, which is not a cell fit but a sign that this lattice "
            "does not roll onto this circumference at all. Roll the other "
            "edge, or use a wider tube."
        )

    if best is None:
        raise ValueError(
            f"Every axial period tried for a {nx}x{ny} {pattern!r} tube "
            "relaxed into a structure with overlapping atoms. The sheet "
            "itself was sound, so this is the roll: the circumference is "
            f"{circumference:.2f} Å and the tube walls meet across the "
            "axis. Roll the other edge, or use a wider sheet."
        )

    positions = best
    box = np.array([0.0, 0.0, best_axial])
    achieved = float(np.mean(np.hypot(positions[:, 0], positions[:, 1])))

    width = 2.0 * (achieved + vacuum)
    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  pbc=(False, False, True))
    atoms.set_cell([width, width, best_axial])
    atoms.positions[:, 0] += 0.5 * width
    atoms.positions[:, 1] += 0.5 * width

    counts = dict(sheet.info["ring_counts"])
    from ..analyse.rings import trace_faces

    faces, boundary = trace_faces(atoms, pairs, max_size=12)
    traced: dict[int, int] = {}
    for face in faces:
        traced[len(face)] = traced.get(len(face), 0) + 1
    if boundary or traced != counts:
        raise ValueError(
            f"The flat sheet's rings are {counts}, but the rolled tube "
            f"traces {traced} with {boundary} boundary walk(s). Rolling is "
            "an isometry onto a developable surface and cannot change a "
            "single ring, so the topology is not at fault: the tube tore "
            "during relaxation. A larger n"
            f"{'x' if roll == 'a' else 'y'} gives a wider tube with less "
            "curvature to absorb."
        )

    deficit = sum((6 - size) * count for size, count in counts.items())
    if deficit != 0:
        raise ValueError(
            f"The tube's ring census {counts} gives sum(6-n) = {deficit:+d}, "
            "but a tube periodic along its axis is a torus and the budget "
            "is exactly 0. A non-zero value here means the sheet this was "
            "rolled from was not the periodic lattice it claimed to be."
        )

    from .capped_cnt import geometry_report

    geometry = geometry_report(positions, [tuple(p) for p in pairs.tolist()],
                               box=box)

    atoms.info.update({
        "builder": "haeckelite_tube",
        "structure_type": "haeckelite_tube",
        # A heptagon's interior angle is 128.6 deg before any strain, so
        # the default sp2 window calls every sound haeckelite BROKEN.
        "quality_family": "haeckelite",
        "pattern": pattern,
        "catalogue": bool(sheet.info.get("catalogue", False)),
        "roll": roll,
        "nx": int(nx),
        "ny": int(ny),
        "bonds": [(int(a), int(b)) for a, b in pairs.tolist()],
        "rings": [[int(i) for i in ring] for ring in sheet.info["rings"]],
        "ring_counts": {int(k): int(v) for k, v in sorted(counts.items())},
        "euler": deficit,
        "genus": 1,               # a periodic tube is a torus, as the sheet was
        "circumference": round(circumference, 4),
        "rolled_radius": round(rolled_radius, 4),
        "radius": round(achieved, 4),
        # The chord-vs-arc compression the roll imposes, before relaxation.
        # Quoted because it is the only part of the curvature cost this
        # force field can see; the rehybridisation it cannot see is larger.
        "roll_compression": round(bond ** 2 / (24.0 * rolled_radius ** 2), 5),
        "axial_period": round(float(best_axial), 4),
        "axial_factor": round(float(best_axial / axial), 4),
        "flat_axial_period": round(axial, 4),
        "strain_score": round(float(best_score), 6),
        "bond": float(bond),
        "seed": seed,
        "geometry": geometry,
    })

    fraction = sum(count for size, count in counts.items() if size != 6)
    atoms.info["non_hexagonal_fraction"] = round(
        fraction / max(1, len(sheet.info["rings"])), 4)

    if (geometry["bond_min"] < BOND_FLOOR
            or geometry["bond_max"] > BOND_CEILING
            or geometry["angle_min"] < ANGLE_FLOOR
            or geometry["angle_max"] > ANGLE_CEILING):
        raise ValueError(
            f"A {nx}x{ny} {pattern!r} sheet rolled along {roll!r} gives a "
            f"sound topology -- rings {dict(sorted(counts.items()))}, "
            f"sum(6-n) = 0 -- on a geometry that is not carbon: bonds "
            f"{geometry['bond_min']:.3f}-{geometry['bond_max']:.3f} Å and "
            f"angles {geometry['angle_min']:.1f}-{geometry['angle_max']:.1f} "
            f"deg, against the sp2 window {BOND_FLOOR}-{BOND_CEILING} Å and "
            f"{ANGLE_FLOOR}-{ANGLE_CEILING} deg. The flat sheet passed, so "
            f"this is the curvature: at radius {achieved:.2f} Å the wall is "
            "too tightly rolled for this lattice. Use a wider tube."
        )

    if achieved < WARN_TUBE_RADIUS:
        warnings.warn(
            f"Tube radius {achieved:.2f} Å is below {WARN_TUBE_RADIUS} Å -- "
            "narrower than a (5,5). The geometry passes every check this "
            "framework makes, but those checks are a valence force field, "
            "which barely sees curvature: the real cost of a narrow tube is "
            "sigma-pi rehybridisation, and nothing here prices it. Treat "
            "this as a starting geometry for a real calculator.",
            stacklevel=2,
        )

    if geometry["bond_min"] < BOND_SOFT_FLOOR:
        warnings.warn(
            f"Shortest bond is {geometry['bond_min']:.3f} Å, below the "
            f"{BOND_SOFT_FLOOR} Å this framework expects of sp2 carbon. One "
            "rest length for every bond is what this force field has, and a "
            "pentagon-heptagon lattice really does put some bonds under "
            "compression -- the more so once rolled. Re-relax with a proper "
            "calculator before quoting a geometry.",
            stacklevel=2,
        )
    return atoms


def describe_haeckelite_tube(atoms: Atoms) -> str:
    """One-line summary of what the roll actually produced."""
    info: dict[str, Any] = atoms.info
    counts = info.get("ring_counts", {})
    census = ", ".join(f"{size}:{count}" for size, count in sorted(counts.items()))
    return (
        f"{info.get('pattern', '?')} rolled along {info.get('roll', '?')}: "
        f"{len(atoms)} atoms, rings {census}, sum(6-n) = "
        f"{info.get('euler', 0):+d}, radius {info.get('radius', 0):.2f} Å "
        f"(rolled at {info.get('rolled_radius', 0):.2f}), axial period "
        f"{info.get('axial_period', 0):.2f} Å, "
        f"{100 * info.get('non_hexagonal_fraction', 0):.0f}% non-hexagonal."
    )


__all__ = [
    "AXIAL_SEARCH",
    "AXIAL_STEP",
    "MIN_TUBE_RADIUS",
    "WARN_TUBE_RADIUS",
    "build_haeckelite_tube",
    "describe_haeckelite_tube",
    "roll_to_tube",
]
