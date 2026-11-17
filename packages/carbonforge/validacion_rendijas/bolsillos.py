"""Is a deep well distributed, or a few spots? And where are they?

    uv run python validacion_rendijas/bolsillos.py <carpeta> <nombre>...

`q_st` is a Boltzmann-weighted mean, so a handful of very deep points
carries it. That is how a pinched wall fakes a good adsorbent: a
pinched tube is a narrow slit, and `collapsed_wall`'s 328.4 deg is a
sharp threshold on a continuous quantity, so a cell a few degrees over
it may be less damaged rather than undamaged.

Two measurements separate geometry from damage, and neither is a
tolerance.

**The unweighted mean over the accessible volume.** A slit's whole
volume IS its deep site: measured at H = 6.6 A, the mean is -824 K and
the deepest -909, a ratio of 1.1. A structure whose mean is shallow
and whose extreme is deep holds its uptake in a small region, whatever
that region turns out to be:

    rendija H=6.6      mean  -824 K   deepest  -909 K   ratio  1.1
    super-cube               -208            -1440            6.9
    super-tetrahedron        -243            -1439            5.9
    super-diamond             -82            -1143           13.9

**Where those points are.** This is what tells a pocket from a pinch.
A cage vertex where three tubes meet is enclosed by three walls, so a
deep pocket there is geometry and there should be one per vertex; a
pinch lands anywhere. Measured, super-cube's deep points fall into
**7 pockets of 1.6 +- 0.8 A against its 8 vertices** -- one each, to
within the one vertex no sample happened to land in. That regularity
is not what damage looks like.

So the cages' -1440 K wells are real, and they are **tiny**: the mean
over the volume says almost all of a cage adsorbs nothing. Which is
the whole caveat of a zero-coverage limit, stated at the top of
`adsorption/henry.py`: the slope says which fills first, and only an
isotherm says which holds more. Eight pockets per cage is about
0.4 mol/kg of saturation against a slit's tens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

from carbonforge.adsorption.porosity import _framework_energy, analyse_porosity

from llenado import slit
from llenado_super import embed

#: Points within this fraction of the deepest count as deep.
DEEP_FRACTION = 0.80

#: Deep points closer than this are one pocket, in Å. Comfortably over
#: a pocket's own size and well under any net's strut length.
TOGETHER = 6.0

#: Samples for the energy field and for the porosity report.
FIELD_SAMPLES = 150_000
POROSITY_SAMPLES = 60_000

#: Vertices per cell, for the comparison that makes the pocket count
#: mean something. A cage has one node per vertex of its polyhedron;
#: the 4-cube's projection has sixteen.
VERTICES: dict[str, int] = {
    "super-tetrahedron": 4, "super-cube": 8, "super-octahedron": 6,
    "super-dodecahedron": 20, "super-icosahedron": 12,
    "super-hypercube": 16, "super-diamond": 8, "super-cubic": 1,
}


def spread(label: str, atoms: Atoms) -> None:
    """The unweighted mean against the extreme."""
    porosity = analyse_porosity(atoms, samples=POROSITY_SAMPLES,
                                area_points=200, temperature=77.0, seed=0)
    ratio = (porosity.deepest_energy / porosity.mean_energy
             if porosity.mean_energy else float("nan"))
    print(f"| {label} | {porosity.mean_energy:.0f} | "
          f"{porosity.deepest_energy:.0f} | {ratio:.1f} |", flush=True)


def pockets(label: str, atoms: Atoms) -> None:
    """How many deep regions there are, and how big each is."""
    edges = np.diag(np.asarray(atoms.cell, dtype=float))
    tree = cKDTree(np.mod(atoms.get_positions(), edges), boxsize=edges)
    rng = np.random.default_rng(0)
    points = rng.random((FIELD_SAMPLES, 3)) * edges
    energy = _framework_energy(points, tree, edges, 77.0)
    finite = np.isfinite(energy)
    deepest = energy[finite].min()
    deep = points[finite][energy[finite] < DEEP_FRACTION * deepest]
    expected = VERTICES.get(label, "?")
    if len(deep) < 2:
        print(f"| {label} | {expected} | {len(deep)} | {len(deep)} | — |")
        return
    # Cluster under the minimum image, shifting to the first point so
    # a pocket straddling the cell seam is not split in two.
    shifted = deep - deep[0]
    shifted -= edges * np.round(shifted / edges)
    labels = fcluster(linkage(shifted, "single"), TOGETHER, "distance")
    sizes = [float(np.ptp(shifted[labels == k], axis=0).max())
             for k in np.unique(labels)]
    print(f"| {label} | {expected} | {len(np.unique(labels))} | {len(deep)} "
          f"| {np.mean(sizes):.1f} ± {np.std(sizes):.1f} |", flush=True)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    names = sys.argv[2:]
    loaded = {}
    for name in names:
        source = folder / f"{name}.extxyz"
        if not source.exists():
            print(f"falta {source}")
            return 1
        loaded[name], _repeat, _gap = embed(read(str(source)))

    print("### Está el pozo repartido, o es una cola\n")
    print("| estructura | media sobre el volumen (K) | más profundo (K) | "
          "razón |")
    print("|---|---|---|---|")
    spread("rendija H=6.6 (referencia)", slit(6.6))
    for name, atoms in loaded.items():
        spread(name, atoms)

    print("\n### Dónde están esos puntos\n")
    print("| estructura | vértices | bolsillos | puntos profundos | "
          "tamaño (Å) |")
    print("|---|---|---|---|---|")
    for name, atoms in loaded.items():
        pockets(name, atoms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
