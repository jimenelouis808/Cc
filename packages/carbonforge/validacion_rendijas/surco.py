"""How deep the well of a GROOVE is, against a slit and a bare tube.

    uv run python validacion_rendijas/surco.py

`curvatura.py` settled why the superstructures lose per m2: most of
their surface is the OUTSIDE of a tube, and a convex wall holds fewer
carbons within reach of a point above it than a flat sheet of the same
areal density -- 76 % of the flat well for a (6,6) exterior.

That is a statement about one wall. It leaves the design question
open, because a superstructure's adsorbent is not one wall: where two
tubes run alongside each other they leave a GROOVE, and a point in the
crevice sees two convex walls at once. Whether that pays back the
convexity deficit is arithmetic nobody here has done, and it decides
whether the next GCMC hours go on a superstructure at all.

So three geometries are read with nothing but their shape differing --
same probe, same temperature, same carbon, same scan:

* the groove between two parallel tubes, as a function of their gap;
* the slit between two flat sheets, the reference adsorbent;
* one tube's exterior, the `curvatura.py` baseline.

Every well is SCANNED rather than evaluated at a chosen height, for
the reason `curvatura.py` gives: the minimum moves with the geometry,
and reading one geometry at another's optimum would report that
movement as a change in depth.
"""

from __future__ import annotations

import numpy as np

from carbonforge.adsorption.potentials import H2_C, feynman_hibbs
from carbonforge.builders.cnt import build_cnt
from carbonforge.builders.graphene import build_graphene_supercell

#: The temperature the rest of this folder was measured at.
TEMPERATURE = 77.0

#: The strut of the superstructures, and so the tube a groove is made of.
INDICES = 6

#: Wall-to-wall gaps to scan, in A. The graphitic contact is 3.4.
GAPS = (3.0, 3.4, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)

#: Beyond this the pair potential is negligible and the sum is cut.
CUTOFF = 20.0

#: Scan step and span for every well, shared so nothing differs but shape.
STEP = 0.02
SPAN = (0.5, 14.0)

#: The groove's scan starts ON its symmetry axis, and must. Where the
#: gap is wide the deepest site is the axis itself -- the slit the two
#: tubes make -- and a floor of 0.5 A reported that as a minimum AT the
#: floor for every gap of 7 A and over, which is the boundary rather
#: than a well. Where the gap is narrow the minimum moves outward, to
#: 4.68 A at 3.0 A, so both cases need the same open scan.
GROOVE_SPAN = (0.0, 14.0)


def scan(positions: np.ndarray, origin: np.ndarray,
         direction: np.ndarray,
         span: tuple[float, float] = SPAN) -> tuple[float, float]:
    """``(offset, energy)`` of the deepest point along a ray.

    `origin` is where the offset is measured from and `direction` a
    unit vector. The sum is over carbons within `CUTOFF`.

    `span` bounds the scan, and bounding it is load-bearing for a
    slit: scanned freely the ray leaves through the far wall and the
    deepest point it finds is the OUTSIDE of a bilayer, which is not a
    slit site at all. Measured, that reported a 3.0 A slit as -503.7 K
    when the slit holds nothing -- a number from the wrong surface.
    """
    best = (0.0, 0.0)
    for offset in np.arange(*span, STEP):
        point = origin + direction * offset
        radii = np.linalg.norm(positions - point, axis=1)
        radii = radii[radii < CUTOFF]
        if not radii.size:
            continue
        energy = float(np.sum(
            feynman_hibbs(np.maximum(radii, 1e-6), H2_C, TEMPERATURE)))
        if energy < best[1]:
            best = (float(offset), energy)
    return best


def centred(atoms) -> np.ndarray:
    positions = atoms.get_positions()
    return positions - positions.mean(axis=0)


def tube_wall() -> tuple[np.ndarray, float]:
    """A long armchair tube centred on the z axis, and its radius."""
    positions = centred(build_cnt(n=INDICES, m=INDICES, length=60.0,
                                  vacuum=20.0))
    radius = float(np.hypot(positions[:, 0], positions[:, 1]).mean())
    return positions, radius


def main() -> None:
    sheet = centred(build_graphene_supercell(nx=24, ny=14, vacuum=20.0))
    up = np.array([0.0, 0.0, 1.0])
    flat_at, flat = scan(sheet, np.zeros(3), up)

    tube, radius = tube_wall()
    out = np.array([1.0, 0.0, 0.0])
    tube_at, tube_well = scan(tube, out * radius, out)

    print(f"una pared plana (grafeno)   pozo {flat:7.1f} K "
          f"a {flat_at:.2f} A")
    print(f"una pared convexa ({INDICES},{INDICES})     pozo {tube_well:7.1f} K "
          f"a {tube_at:.2f} A   ({100 * tube_well / flat:.1f} % del plano)")
    print()
    print("| hueco (Å) | rendija (K) | a (Å) | surco (K) | a (Å) | "
          "surco / rendija | surco / pared plana |")
    print("|---|---|---|---|---|---|---|")

    for gap in GAPS:
        # Slit: two sheets, probe scanned up from the lower one.
        lower = sheet.copy()
        upper = sheet.copy()
        upper[:, 2] += gap
        slit = np.vstack((lower, upper))
        # Bounded to the inside, or the ray leaves through the far wall.
        inside = (STEP, max(gap - STEP, 2 * STEP))
        slit_at, slit_well = scan(slit, np.zeros(3), up, span=inside)

        # Groove: two parallel tubes side by side along y, probe
        # scanned outward along x from the line where they touch.
        left = tube.copy()
        right = tube.copy()
        separation = 2.0 * radius + gap
        left[:, 1] -= separation / 2.0
        right[:, 1] += separation / 2.0
        groove = np.vstack((left, right))
        groove_at, groove_well = scan(groove, np.zeros(3), out,
                                      span=GROOVE_SPAN)

        empty = slit_well == 0.0
        ratio = "--" if empty else f"{100 * groove_well / slit_well:.1f} %"
        depth = "vacía" if empty else f"{slit_well:.1f}"
        where = "--" if empty else f"{slit_at:.2f}"
        print(f"| {gap:.1f} | {depth} | {where} | {groove_well:.1f} | "
              f"{groove_at:.2f} | {ratio} | "
              f"{100 * groove_well / flat:.1f} % |")


if __name__ == "__main__":
    main()
