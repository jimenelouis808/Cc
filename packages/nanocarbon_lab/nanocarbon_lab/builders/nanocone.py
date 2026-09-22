"""Carbon nanocones: one disclination, and the angle it forces.

A cone is **developable** -- zero Gaussian curvature everywhere except
at its apex -- so unrolling it onto the plane is an isometry. That is the
whole construction here: take graphene, remove a wedge of ``N * 60``
degrees about a hexagon centre, and join the cut edges. Every bond length
survives exactly, the wall stays all-hexagon, and the apex hexagon, having
lost ``N`` of its six sectors, becomes a ``(6 - N)``-gon.

The apex angle is therefore **quantised**, not chosen:

    sin(theta / 2) = 1 - N / 6

======  ==========  ==================================
``N``   apex angle  apex ring
======  ==========  ==================================
1       112.9 deg   pentagon
2        83.6 deg   square
3        60.0 deg   triangle
4        38.9 deg   -- would need a 2-gon
5        19.2 deg   -- would need a 1-gon
======  ==========  ==================================

Those five angles are the ones Krishnan et al. observed (*Nature* **388**,
451), so this is one of the few builders here with an answer from outside
the framework to check against.

**Only ``N = 1`` is a clean structure, and the table says why.** Measured
at a 22 Å radius: 470 atoms, census **{5: 1, 6: 210}**, bonds
**1.391-1.420 Å** -- a single pentagon and nothing else out of place.
``N = 2`` and ``N = 3`` put a square or a triangle at the apex, which is
topologically sound and chemically not: the bonds there come back at 1.339
and 1.230 Å. Nature does not do this. It splits the disclination into
``N`` *separate* pentagons spread around the tip, which keeps every ring
at five and is why the 19.2 deg single-wall nanohorn exists at all. That
split is a cap-design problem, not a sector cut, and this module does not
solve it -- so ``N >= 4`` is refused rather than approximated.

**The cut must run through atoms, not between them.** Both are mirror
lines of the hexagon, so both look equally valid, and one of them is
wrong: cutting between atoms left a spurious four-membered ring at the
seam on every cone tried (``{4: 1, 5: 1, 6: 204}`` for ``N = 1`` against
``{5: 1, 6: 210}``). The census is the only thing that shows it -- the
bond lengths are identical to three decimals either way.

**The rim is open on purpose.** A nanocone is not a closed molecule; the
base is a free edge, exactly as a nanoribbon's is, and its atoms are
two-coordinate. They are recorded in ``info["rim_atoms"]`` so validation
can tell a deliberate edge from a dangling bond.

For an arbitrary apex angle there is the implicit route, and it is worth
knowing what it costs. Meshing a cone of the same 112.9 deg gave a sound
+12 budget on a **3800-atom amorphous shell**: 183 pentagons and 171
heptagons scattered over the wall instead of one pentagon at the tip.
Sound, and not a nanocone as anybody draws it.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D

#: Disclination counts this construction can place as a single apex ring.
#: Four and five would need a 2-gon and a 1-gon.
BUILDABLE = (1, 2, 3)

#: The one that is also chemistry. See the module docstring.
CLEAN = 1

#: Where to put the cut, in degrees, as candidates rather than a
#: constant. Both 0 and 30 are mirror lines of the hexagon, so both look
#: equally valid, and **which one is right depends on N**: at 0 the
#: ``N = 1`` cone comes back with a spurious four-membered ring at the
#: seam ({4: 1, 5: 1, 6: 204} against {5: 1, 6: 210}), while at 30 the
#: ``N = 2`` cone comes back with four squares instead of one
#: ({4: 4, 6: 156}, sum(6-n) = +8 against +2).
#:
#: So the builder tries them and keeps the one whose ring budget equals
#: the disclination it asked for. That test is exact -- a cone is
#: developable, so rolling cannot create or destroy a ring, and any
#: deviation is the seam failing to match -- which makes this a checked
#: choice rather than a magic number that happens to work for one case.
CUT_CANDIDATES_DEG = (30.0, 0.0, 15.0, 45.0)


def apex_angle(n_pentagons: int) -> float:
    """Apex angle in degrees for a disclination of ``n * 60`` degrees.

    ``sin(theta/2) = 1 - n/6``, which is geometry rather than a fit: the
    unrolled sector has angle ``2*pi - n*pi/3``, and a cone of half-angle
    ``alpha`` unrolls to ``2*pi*sin(alpha)``.
    """
    if not 0 <= n_pentagons < 6:
        raise ValueError("a cone's disclination is 0 to 5 pentagons.")
    return 2.0 * math.degrees(math.asin(1.0 - n_pentagons / 6.0))


def _honeycomb_disc(radius: float, bond: float) -> np.ndarray:
    """Graphene out to ``radius``, centred on a **hexagon centre**.

    With the basis at ``(0, 0)`` and ``(0, bond)`` the hexagon centre sits
    at ``(0, -bond)`` -- verified by counting neighbours, which is the
    only way to be sure: the origin of that basis is an *atom*, and
    building the wedge about an atom gives a three-fold cut that does not
    close for most ``N``.
    """
    lattice = math.sqrt(3.0) * bond
    a1 = np.array([lattice, 0.0])
    a2 = np.array([0.5 * lattice, 0.5 * math.sqrt(3.0) * lattice])
    basis = (np.array([0.0, bond]), np.array([0.0, 2.0 * bond]))
    reach = int(radius / lattice) + 3
    points = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            cell = i * a1 + j * a2
            for offset in basis:
                point = cell + offset
                if float(np.hypot(*point)) <= radius:
                    points.append(point)
    return np.asarray(points, dtype=float)


def build_nanocone(
    n_pentagons: int = CLEAN,
    radius: float = 22.0,
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
    strict: bool = True,
) -> Atoms:
    """Build a carbon nanocone by cutting a wedge out of graphene.

    Parameters
    ----------
    n_pentagons
        The disclination, in units of 60 degrees. This **is** the apex
        angle -- see :func:`apex_angle` -- so there is no angle to pass.
    radius
        Slant radius of the flat sector before rolling (Å), which is the
        cone's slant height afterwards.
    bond
        C-C length (Å). Preserved exactly: the roll is an isometry.
    vacuum
        Padding around the finished molecule (Å).
    strict
        Refuse ``n_pentagons`` of 2 or 3, whose square and triangular
        apex rings are sound topology and poor chemistry. Pass ``False``
        to build them anyway and see for yourself.

    Returns
    -------
    ase.Atoms
        A finite, **open** cone with the census, the apex ring, the rim
        and the measured geometry in ``atoms.info``.

    Raises
    ------
    ValueError
        For ``n_pentagons`` of 4 or 5, which cannot be a single apex ring
        at all; and under ``strict`` for 2 and 3.
    """
    if n_pentagons not in BUILDABLE:
        raise ValueError(
            f"A disclination of {n_pentagons} * 60 deg would make the apex a "
            f"{6 - n_pentagons}-gon, which is not a ring. Only "
            f"{list(BUILDABLE)} can be a single apex ring. Nature splits "
            "larger disclinations into separate pentagons around the tip -- "
            f"that is how the {apex_angle(5):.1f} deg nanohorn works -- and "
            "that is a cap design rather than a sector cut, which this "
            "builder does not do."
        )
    if strict and n_pentagons != CLEAN:
        raise ValueError(
            f"{n_pentagons} * 60 deg as a single apex ring gives a "
            f"{6 - n_pentagons}-gon, and the bonds in it come back at "
            f"{'1.339' if n_pentagons == 2 else '1.230'} Å -- sound "
            "topology, poor chemistry, and not what nature does (it uses "
            f"{n_pentagons} separate pentagons). Pass strict=False to build "
            "it anyway, or use n_pentagons=1, which is exact."
        )

    theta = apex_angle(n_pentagons)
    sin_half = 1.0 - n_pentagons / 6.0
    cos_half = math.sqrt(max(0.0, 1.0 - sin_half ** 2))
    kept = 2.0 * math.pi - n_pentagons * math.pi / 3.0

    flat = _honeycomb_disc(radius, bond)
    scale = 2.0 * math.pi / kept

    best = None
    for cut in CUT_CANDIDATES_DEG:
        angle = np.mod(np.arctan2(flat[:, 1], flat[:, 0])
                       - math.radians(cut), 2.0 * math.pi)
        slant = np.hypot(flat[:, 0], flat[:, 1])
        # Drop the far edge of the wedge: after rolling it lands on the
        # near one, and keeping both would double every seam atom.
        keep = angle < kept - 1e-6
        rolled = np.column_stack([
            slant[keep] * sin_half * np.cos(angle[keep] * scale),
            slant[keep] * sin_half * np.sin(angle[keep] * scale),
            -slant[keep] * cos_half,
        ])
        trial, census, rings, pairs, rim = _measure(rolled, vacuum)
        deficit = sum((6 - size) * count for size, count in census.items())
        if deficit == n_pentagons and census.get(6 - n_pentagons, 0) == 1:
            best = (cut, trial, census, rings, pairs, rim, deficit)
            break
        if best is None or abs(deficit - n_pentagons) < abs(best[6] - n_pentagons):
            best = (cut, trial, census, rings, pairs, rim, deficit)

    from .capped_cnt import geometry_report

    cut, atoms, census, rings, pairs, rim, deficit = best
    geometry = geometry_report(atoms.get_positions(), pairs)

    atoms.info.update({
        "builder": "nanocone",
        "structure_type": "nanocone",
        "n_pentagons": int(n_pentagons),
        "cut_offset_deg": float(cut),
        "apex_angle": round(theta, 2),
        "apex_ring": 6 - int(n_pentagons),
        "slant_radius": float(radius),
        "bond": float(bond),
        "bonds": pairs,
        "rings": [[int(i) for i in r] for r in rings],
        "ring_counts": dict(sorted(census.items())),
        "euler": deficit,
        "rim_atoms": rim,
        # A free edge, like a nanoribbon's. Declared so validation reads
        # two-coordinate carbon here as the edge it is.
        "terminal_atoms": rim,
        "geometry": geometry,
    })

    if deficit != n_pentagons:
        warnings.warn(
            f"The wall's sum(6-n) is {deficit:+d} against the {n_pentagons} "
            "the disclination puts in. A cone is developable, so the roll "
            "cannot create a ring: this means the seam did not match, and "
            "the usual cause is cutting between atoms rather than through "
            "them.",
            stacklevel=2,
        )
    return atoms


def _measure(positions: np.ndarray, vacuum: float):
    """Wrap positions into an Atoms and read off bonds, rings and rim."""
    from ..analyse.rings import trace_faces
    from ..utils.geometry import guess_bonds

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  pbc=(False, False, False))
    span = positions.max(axis=0) - positions.min(axis=0)
    atoms.set_cell(span + 2.0 * vacuum)
    atoms.center()

    pairs = [(int(i), int(j)) for i, j, _ in guess_bonds(atoms)]
    degree: dict[int, int] = {}
    for i, j in pairs:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    rim = sorted(i for i in range(len(atoms)) if degree.get(i, 0) < 3)

    faces, _boundary = trace_faces(atoms, np.asarray(pairs, dtype=int),
                                   max_size=12)
    census: dict[int, int] = {}
    rings = []
    for face in faces:
        # An open surface traces its own boundary as one enormous face;
        # max_size is enough for a closed shell and not for this.
        if len(face) > 8:
            continue
        rings.append(face)
        census[len(face)] = census.get(len(face), 0) + 1
    return atoms, census, rings, pairs, rim


def describe_nanocone(atoms: Atoms) -> str:
    """One line: the angle, the apex ring, and whether the wall is clean."""
    info = atoms.info
    census = info.get("ring_counts", {})
    hexagons = census.get(6, 0)
    others = {s: c for s, c in census.items() if s != 6}
    return (
        f"{info.get('n_pentagons', 0)} x 60 deg disclination: apex angle "
        f"{info.get('apex_angle', 0):.1f} deg, apex ring "
        f"{info.get('apex_ring', 0)}-membered, {len(atoms)} atoms, "
        f"{hexagons} hexagons and {others} else, "
        f"{len(info.get('rim_atoms', []))} rim atoms."
    )


__all__ = ["BUILDABLE", "CLEAN", "CUT_CANDIDATES_DEG", "apex_angle",
           "build_nanocone", "describe_nanocone"]
