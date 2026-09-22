"""How sp2 or sp3 each carbon actually is, measured from the geometry.

The question "what fraction of this carbon is sp3" is asked of every real
sample in this field -- it is what distinguishes a graphitic wall from a
diamond-like one, and it is what a Raman D/G ratio or an XPS C1s
asymmetry is a proxy for. Here it can be measured directly, because the
positions are known.

The measure is the **angle sum at a three-coordinate carbon**. Its three
bond angles add to:

* **360 deg** when the carbon is planar -- ideal sp2, as in graphene,
* **328.4 deg** when it is tetrahedral -- ideal sp3, three of the 109.47
  deg angles, the fourth bond pointing out of the net.

Anything between is partial pyramidalisation, and the fraction

    sp3 character = (360 - sum(theta)) / (360 - 328.4)

runs 0 for flat and 1 for tetrahedral. A four-coordinate carbon is sp3
by construction and is reported as 1 without measuring.

**This is the same angle sum the curvature analysis warns about**, and
the warning is worth repeating here because the two uses look identical
and are not. The deficit ``2*pi - sum(theta)`` is never negative at a
trivalent vertex -- the spherical triangle inequality forbids it -- so it
cannot tell a cap from a neck and is useless as a curvature *sign*. What
it does measure, faithfully, is how far the carbon has bent out of its
own plane, and that is exactly hybridisation. A heptagon in a flat sheet
and a heptagon in a saddle both read near 0 here, correctly: neither
carbon is pyramidal.

So a negatively curved wall does **not** show up as sp3, and should not.
Curvature lives in the ring census; hybridisation lives in the angles.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

#: Angle sum of an ideal planar (sp2) carbon, in degrees.
PLANAR_SUM = 360.0

#: Angle sum of three tetrahedral (sp3) bonds, 3 * 109.471 deg.
TETRAHEDRAL_SUM = 328.4

#: Above this sp3 character a carbon is counted as sp3 in the tally.
#: Halfway is the only defensible cut when the measure is continuous,
#: and the mean character is reported beside the count so the choice of
#: cut never has to carry an argument on its own.
SP3_CUT = 0.5


def _neighbours(n_atoms: int, bonds: Sequence[Sequence[int]]
                ) -> dict[int, list[int]]:
    table: dict[int, list[int]] = defaultdict(list)
    for pair in bonds:
        i, j = int(pair[0]), int(pair[1])
        table[i].append(j)
        table[j].append(i)
    return {i: table.get(i, []) for i in range(n_atoms)}


def sp3_character(
    positions: np.ndarray,
    bonds: Sequence[Sequence[int]],
    cell: np.ndarray | None = None,
) -> np.ndarray:
    """Per-atom sp3 character, 0 (flat) to 1 (tetrahedral).

    Parameters
    ----------
    positions
        ``(n, 3)`` coordinates.
    bonds
        Pairs of indices. Only the first two entries of each are read, so
        a list carrying extra per-bond data is accepted unchanged.
    cell
        Periodic cell, if any. Bond vectors are taken under the minimum
        image convention when given, because a bond across the seam is a
        1.42 Å bond and not a cell-length one.

    Returns
    -------
    numpy.ndarray
        One value per atom. Atoms with fewer than three bonds get
        ``nan`` -- an edge atom or a dangling one has no angle sum to
        measure, and reporting 0 for it would quietly call it sp2.
    """
    positions = np.asarray(positions, dtype=float)
    table = _neighbours(len(positions), bonds)
    lattice = None if cell is None else np.asarray(cell, dtype=float)
    out = np.full(len(positions), np.nan)

    for i, ring in table.items():
        if len(ring) >= 4:
            out[i] = 1.0
            continue
        if len(ring) != 3:
            continue
        vectors = []
        for j in ring:
            delta = positions[j] - positions[i]
            if lattice is not None and np.any(lattice):
                fractional = np.linalg.solve(lattice.T, delta)
                delta = delta - np.round(fractional) @ lattice
            norm = np.linalg.norm(delta)
            if norm <= 0.0:
                break
            vectors.append(delta / norm)
        if len(vectors) != 3:
            continue
        total = 0.0
        for a, b in ((0, 1), (1, 2), (2, 0)):
            cosine = float(np.clip(np.dot(vectors[a], vectors[b]), -1.0, 1.0))
            total += math.degrees(math.acos(cosine))
        out[i] = (PLANAR_SUM - total) / (PLANAR_SUM - TETRAHEDRAL_SUM)
    return out


def hybridisation_report(
    positions: np.ndarray,
    bonds: Sequence[Sequence[int]],
    cell: np.ndarray | None = None,
) -> dict:
    """Summarise the sp2/sp3 split of a structure.

    Returns
    -------
    dict
        ``n_measured`` (atoms with three or four bonds), ``n_sp3``,
        ``sp3_fraction``, ``mean_character``, ``max_character`` and
        ``angle_sum_min``/``angle_sum_max`` in degrees. The angle sums
        are carried because they are the raw measurement and the
        character is a rescaling of them -- a reader who distrusts the
        scaling can work from the degrees.
    """
    character = sp3_character(positions, bonds, cell)
    measured = character[~np.isnan(character)]
    if not len(measured):
        return {"n_measured": 0, "n_sp3": 0, "sp3_fraction": float("nan"),
                "mean_character": float("nan"), "max_character": float("nan"),
                "angle_sum_min": float("nan"), "angle_sum_max": float("nan")}
    sums = PLANAR_SUM - measured * (PLANAR_SUM - TETRAHEDRAL_SUM)
    return {
        "n_measured": int(len(measured)),
        "n_sp3": int((measured >= SP3_CUT).sum()),
        "sp3_fraction": float((measured >= SP3_CUT).mean()),
        "mean_character": float(measured.mean()),
        "max_character": float(measured.max()),
        "angle_sum_min": float(sums.min()),
        "angle_sum_max": float(sums.max()),
    }


def describe_hybridisation(report: dict) -> str:
    """One line: the split, and the angle sums it was read from."""
    if not report.get("n_measured"):
        return "hybridisation: nothing with three bonds to measure"
    return (
        f"sp3 {100 * report['sp3_fraction']:.1f}% of "
        f"{report['n_measured']} carbons "
        f"(mean character {report['mean_character']:.2f}, angle sums "
        f"{report['angle_sum_min']:.1f}-{report['angle_sum_max']:.1f} deg; "
        f"360 is flat sp2, 328.4 tetrahedral sp3)"
    )


__all__ = ["PLANAR_SUM", "SP3_CUT", "TETRAHEDRAL_SUM", "describe_hybridisation",
           "hybridisation_report", "sp3_character"]
