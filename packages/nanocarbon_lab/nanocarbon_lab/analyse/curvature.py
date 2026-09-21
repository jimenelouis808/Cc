"""Where the disclinations sit, measured against the surface's own shape.

Two studies of single-wall coils (Popović et al., *Contemp. Mater.*
III-1 (2012) 51; Liu et al., *Nanoscale Res. Lett.* 5 (2010) 478) make
one structural claim that can be checked on a finished model without
running anything: a pentagon is a +60 deg disclination and a heptagon a
-60 deg one, so **pentagons belong where the Gaussian curvature is
positive and heptagons where it is negative**. On a coil that reads as
"pentagons on the outer equator, heptagons on the inner", and
:func:`~nanocarbon_lab.builders.periodic_coil.curvature_check` tests it
that way -- by asking which side of the torus axis each ring falls on.

That test needs an axis, so it works on a coil and on nothing else. The
claim itself is not about coils: it is about curvature, and every
surface this package meshes has some. This module states it intrinsically
so that a junction, a schwarzite, a nanotube network and a supernetwork
can all be held to it.

**The obvious way to measure it is wrong, and wrong in a way that looks
right.** The discrete Gaussian curvature of a polyhedron is the angular
defect ``2*pi - sum(angles)`` at each vertex -- but a carbon atom here is
**trivalent**, and three angles at a point can never sum past 360 deg
(the spherical triangle inequality). So that defect is never negative, at
any atom, on any structure: measured, it called Schwarz P -- a minimal
surface, negatively curved everywhere -- positive at every ring size.
What it actually measures is pyramidalisation. The curvature of a
trivalent net does not live at its atoms; it lives in the non-planar
faces.

So the surface is fitted instead. Around each atom, in a frame aligned
with the local normal, ``z = a x^2 + b x y + c y^2`` over its
two-bond neighbourhood, and the Gaussian curvature has the sign of
``4ac - b^2``. That is a property of the embedding, which is what the
claim is about.

Measured, and this is the calibration rather than an assertion:

=================  ==========  ==========  ==========
surface            pentagons   hexagons    heptagons
=================  ==========  ==========  ==========
C60                +1.00       +1.00       --
graphene            --          0.00       --
haeckelite R5,7     0.00        --          0.00
Y junction         +0.88       +0.11       **-0.75**
Schwarz P          +0.56       -0.46       **-1.00**
periodic coil      +1.00        --         **-1.00**
=================  ==========  ==========  ==========

The numbers are the mean sign of ``K`` over the rings of that size. A
sphere is positive everywhere and a plane is zero everywhere, which is
how the measure is checked against cases whose answer is known.

**The flat haeckelite row is the one that matters**, because the obvious
objection to all of this is circularity: the patch around a pentagon is
dominated by that pentagon, so does the fit simply recover the ring size
under a different name? R5,7 settles it. It is a *flat* lattice of 16
pentagons and 16 heptagons and nothing else, so every ring has genuinely
zero curvature -- and the fit returns **exactly 0.000 for both**, mean
sign and median alike. The measure reads the embedding, not the census.
Keep that control if this module is ever rewritten; without it the other
rows prove nothing.

The surfaces carrying both signs put the disclinations where the papers
say. On Schwarz P **every heptagon** is in negative curvature; a
junction's hexagons sit near zero because its arms are cylinders; and
the coil scores **100%** -- higher than its own axis-based test reports
(68% for the smooth helix, 85% for the hexagon at D/d = 3.92). The two
disagree because the axis test asks only which side of the torus axis a
ring falls on, which misreads a ring sitting near the top or bottom of
the tube where the real curvature is near zero. Where they differ, the
intrinsic one is measuring what the claim is about, and the axis test's
"wrong side" count is its own artefact rather than a defect of the
model.
"""

from __future__ import annotations

import numpy as np

#: Rings smaller than six want positive curvature, larger want negative,
#: and a hexagon wants nothing -- it is the flat case and carries no
#: expectation, so it is reported and not scored.
NEUTRAL_RING = 6

#: How many bonds out the patch reaches. Two gives 9-12 atoms on a
#: trivalent net, enough to fit six quadric coefficients; one gives three
#: neighbours and the fit is underdetermined.
DEFAULT_SHELLS = 2


def _neighbour_table(n_atoms: int, bonds) -> list[set[int]]:
    table: list[set[int]] = [set() for _ in range(n_atoms)]
    for a, b in bonds:
        table[a].add(b)
        table[b].add(a)
    return table


def local_curvature(positions: np.ndarray, bonds, box=None,
                    shells: int = DEFAULT_SHELLS) -> np.ndarray:
    """Sign-carrying Gaussian curvature at every atom.

    Returns ``4ac - b^2`` of the local quadric fit, which shares the sign
    of the Gaussian curvature and is zero on a plane or a cylinder. The
    magnitude is not a curvature in units of inverse length squared --
    only its sign and relative size are meaningful -- so use it to tell a
    dome from a saddle, not to quote a radius.

    ``box`` is a 3-vector with ``0`` on any non-periodic axis, matching
    the convention used everywhere else here; without it a patch that
    straddles a cell face is fitted to atoms a cell-length away.
    """
    from ..builders import fullerene_mesh as fm

    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    table = _neighbour_table(n, bonds)
    out = np.zeros(n)
    for centre in range(n):
        patch = {centre}
        frontier = {centre}
        for _ in range(shells):
            nxt: set[int] = set()
            for v in frontier:
                nxt |= table[v]
            patch |= nxt
            frontier = nxt
        index = sorted(patch - {centre})
        if len(index) < 6:            # six quadric coefficients to find
            continue
        rel = positions[index] - positions[centre]
        if box is not None:
            rel = fm.minimum_image(rel, np.asarray(box, dtype=float))
        # The patch's own least-spread direction is its normal. Taken
        # from the field gradient instead it would be wrong after
        # relaxation, which is the mistake tmd/curved.py records.
        _u, _s, vt = np.linalg.svd(rel - rel.mean(axis=0))
        e1, e2, normal = vt[0], vt[1], vt[2]
        x, y = rel @ e1, rel @ e2
        z = rel @ normal
        design = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
        coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
        a, b, c = coefficients[0], coefficients[1], coefficients[2]
        out[centre] = 4.0 * a * c - b * b
    return out


def disclination_check(positions: np.ndarray, rings, bonds, box=None,
                       shells: int = DEFAULT_SHELLS) -> dict:
    """Do the pentagons sit in positive curvature and the heptagons in
    negative?

    Returns per ring size the count, the mean sign of ``K`` over its
    rings and the median ``K``; plus ``agreement``, the fraction of
    **non-hexagonal** rings on the side the disclination's sign calls
    for. Hexagons are excluded from the score rather than counted as
    passes: a hexagon is the flat case and has no side to be on.

    ``agreement`` is a diagnostic, not a gate. Rings on the wrong side
    are defects of the model and are reported as defects -- the coil
    module's own rule, kept here.
    """
    curvature = local_curvature(positions, bonds, box, shells)
    per_size: dict[int, list[float]] = {}
    for ring in rings:
        per_size.setdefault(len(ring), []).append(
            float(np.mean(curvature[list(ring)])))

    sizes: dict[int, dict] = {}
    right = 0
    scored = 0
    for size, values in sorted(per_size.items()):
        array = np.asarray(values)
        sizes[size] = {
            "count": int(array.size),
            "mean_sign": round(float(np.mean(np.sign(array))), 4),
            "median": float(np.median(array)),
        }
        if size == NEUTRAL_RING:
            continue
        wanted = 1.0 if size < NEUTRAL_RING else -1.0
        right += int(np.sum(np.sign(array) == wanted))
        scored += int(array.size)
    return {
        "sizes": sizes,
        "n_scored": scored,
        "n_correct": right,
        "agreement": round(right / scored, 4) if scored else None,
        "shells": shells,
    }


__all__ = ["DEFAULT_SHELLS", "NEUTRAL_RING", "disclination_check",
           "local_curvature"]
