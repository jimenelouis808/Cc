"""How much of the well a convex wall costs, against a flat one.

    uv run python validacion_rendijas/curvatura.py

This answers the question the README was left holding. Two explanations
for the superstructures' deficit per m2 had been measured and refused;
the one that survived was that their surface is mostly the OUTSIDE of a
tube, which is convex, and a convex wall has fewer carbons within reach
of a point above it than a flat sheet of the same areal density.

That is a claim about the depth of the potential well, not about any
distance, which is why `mean_wall_distance` could not settle it. Here
one isolated wall is read at a time, so nothing but the curvature
differs: same carbon density, same probe, same temperature, and the
probe is scanned away from the surface so every wall is read at its own
minimum rather than at a distance chosen for it.

The flat limit is what makes this a measurement instead of a story: the
deficit has to vanish as the radius grows.
"""

from __future__ import annotations

import numpy as np

from carbonforge.adsorption.potentials import H2_C, feynman_hibbs
from carbonforge.builders.cnt import build_cnt
from carbonforge.builders.graphene import build_graphene_supercell

#: The temperature everything else in this folder was measured at.
TEMPERATURE = 77.0

#: Armchair indices to scan. (6,6) is the strut of the superstructures.
INDICES = (5, 6, 8, 10, 15, 20, 30)

#: Beyond this the pair potential is negligible and the sum is cut.
CUTOFF = 20.0


def well(positions: np.ndarray, direction: np.ndarray,
         start: float) -> tuple[float, float]:
    """``(distance, energy)`` of the deepest point along `direction`.

    Scanned rather than evaluated at a fixed height, because a fixed
    height would smuggle in the answer: if the minimum moved with the
    curvature, reading every wall at the flat wall's optimum would
    report that movement as a change in depth.
    """
    best = (0.0, 0.0)
    for distance in np.arange(2.0, 8.0, 0.02):
        point = direction * (start + distance)
        radii = np.linalg.norm(positions - point, axis=1)
        radii = radii[radii < CUTOFF]
        if not radii.size:
            continue
        energy = float(np.sum(
            feynman_hibbs(np.maximum(radii, 1e-6), H2_C, TEMPERATURE)))
        if energy < best[1]:
            best = (float(distance), energy)
    return best


def centred(atoms) -> np.ndarray:
    positions = atoms.get_positions()
    return positions - positions.mean(axis=0)


def main() -> None:
    sheet = centred(build_graphene_supercell(nx=20, ny=12, vacuum=20.0))
    flat_at, flat = well(sheet, np.array([0.0, 0.0, 1.0]), 0.0)
    print(f"pared plana (grafeno)        R = inf      "
          f"pozo {flat:7.1f} K a {flat_at:.2f} A")
    print()
    print("| (n,n) | R (Å) | pozo (K) | distancia (Å) | % del plano |")
    print("|---|---|---|---|---|")
    for index in INDICES:
        tube = centred(build_cnt(n=index, m=index, length=40.0, vacuum=20.0))
        radius = float(np.hypot(tube[:, 0], tube[:, 1]).mean())
        at, depth = well(tube, np.array([1.0, 0.0, 0.0]), radius)
        print(f"| ({index},{index}) | {radius:.2f} | {depth:.1f} | "
              f"{at:.2f} | {100 * depth / flat:.1f} % |")
    print()
    print(f"| plana | ∞ | {flat:.1f} | {flat_at:.2f} | 100.0 % |")


if __name__ == "__main__":
    main()
