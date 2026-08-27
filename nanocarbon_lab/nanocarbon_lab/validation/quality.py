"""A one-line verdict on whether a built structure's geometry is physical.

Every builder already records measured bond lengths, bond angles and
non-bonded contacts in ``atoms.info["geometry"]``. Those numbers are the
honest answer to "is this structure any good", but they are three ranges
and a count, and a reader has to know the sp2 window by heart to judge
them. This module turns them into a verdict.

It exists because the numbers alone were being misread in both
directions. A coil drawn far tighter than any real nanocoil still
reported ``0`` close contacts and looked fine, while its bonds had been
pulled to 1.66 Å -- broken, by any chemical standard. Conversely, an
intentionally CVD-rough wall looks alarming in a render and is in fact
perfectly within the sp2 range. Neither reading is available from the
render, so the verdict is stated in words next to the numbers.

The thresholds are chemistry, not taste:

* **sp2 bonds** sit at 1.42 Å in graphite. Curvature and defects move
  them: a pentagon or a strained bend reaches ~1.30-1.55 Å and the
  network is intact. Past that, the bond is no longer a sensible C-C
  bond -- 1.60 Å is beyond even the longest sp3 C-C (1.54 Å), in a
  network that is supposed to be sp2.
* **Angles** are 120 deg flat; curvature pyramidalises them. Fullerene
  pentagons run to ~108 deg and the strained rings of a schwarzite neck
  to ~100-135 deg. Outside that, the sheet has folded rather than bent.
* **Close contacts** below 2 Å between non-bonded atoms are never
  physical in a relaxed sp2 network; the graphite interlayer distance is
  3.35 Å.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

# Chemically defensible sp2 windows (Å, degrees). See module docstring.
SP2_BOND_RANGE = (1.30, 1.55)
SP2_ANGLE_RANGE = (100.0, 135.0)

Verdict = Literal["clean", "strained", "broken"]


def sp2_quality(geometry: Mapping[str, float]) -> tuple[Verdict, str]:
    """Classify a ``geometry`` report as clean, strained or broken.

    Parameters
    ----------
    geometry
        An ``atoms.info["geometry"]`` mapping, as produced by
        :func:`nanocarbon_lab.builders.capped_cnt.geometry_report`. Needs
        ``bond_min``, ``bond_max``, ``angle_min``, ``angle_max`` and
        ``n_close_contacts``.

    Returns
    -------
    (verdict, explanation)
        ``"clean"`` -- every bond and angle inside the sp2 window and no
        close contacts; the structure is publishable as physics.
        ``"strained"`` -- inside the window but near an edge (within
        0.03 Å or 5 deg): intact, and fine for artwork, but visibly bent.
        ``"broken"`` -- outside the window, or atoms overlapping. The
        explanation names which quantity failed and by how much, so the
        caller knows what to loosen.

    Notes
    -----
    This judges *geometry only*. Ring topology is checked separately by
    the Euler budget, and a structure can be topologically perfect while
    geometrically broken -- that combination is exactly what this catches.
    """
    bond_lo, bond_hi = SP2_BOND_RANGE
    angle_lo, angle_hi = SP2_ANGLE_RANGE
    b_min = float(geometry["bond_min"])
    b_max = float(geometry["bond_max"])
    a_min = float(geometry["angle_min"])
    a_max = float(geometry["angle_max"])
    clashes = int(geometry["n_close_contacts"])

    if clashes:
        return "broken", (
            f"{clashes} non-bonded atom pairs closer than 2 Å — the wall has "
            "folded through itself."
        )
    if b_max > bond_hi or b_min < bond_lo:
        worst = b_max if b_max > bond_hi else b_min
        return "broken", (
            f"bond {worst:.2f} Å is outside the sp2 range "
            f"{bond_lo:.2f}–{bond_hi:.2f} Å — lower the strain "
            "(wider coil, gentler shape, or a thinner tube)."
        )
    if a_max > angle_hi or a_min < angle_lo:
        worst = a_max if a_max > angle_hi else a_min
        return "broken", (
            f"bond angle {worst:.0f}° is outside {angle_lo:.0f}–{angle_hi:.0f}° "
            "— the sheet has folded rather than bent."
        )
    if b_max > bond_hi - 0.03 or b_min < bond_lo + 0.03 or a_max > angle_hi - 5.0:
        return "strained", (
            "bonds and angles are inside the sp2 range but near its edge — "
            "intact and fine for artwork, visibly strained as physics."
        )
    return "clean", "bonds, angles and spacings are all in the sp2 range."


__all__ = ["SP2_ANGLE_RANGE", "SP2_BOND_RANGE", "Verdict", "sp2_quality"]
