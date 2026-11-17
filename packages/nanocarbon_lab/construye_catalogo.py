"""Build every superlattice entry the catalogue has, not just the nets.

    python construye_catalogo.py <carpeta>
    python construye_catalogo.py <carpeta> --solo super-diamond

The adsorption study measured four of the catalogue's twenty-five
entries -- super-square, super-graphene, super-cubic and super-fcc --
and the gap was not deliberate: `SUPERLATTICES` holds five periodic
nets and `CAGES` holds twenty finite ones, and only the first table
had been looked at. This builds the other twenty-one with the
parameters their own presets ship, so nothing here is an invented
number.

Two things about this set change what a measurement of it means, and
both belong before any table of results.

**A cage is finite in all three directions, so the whole inter-cage
space is a PACKING choice.** A periodic net's pore is the net's own;
a cage's is the cage's interior plus whatever gap is left between
images. The handoff uses 3.4 A, the graphitic contact -- the densest
solid these objects form without touching, and the same value
`validacion_superestructuras` used -- but a Henry coefficient per gram
then depends on that choice as much as on the cage, and a different
packing is a different adsorbent.

**No cage has a crystalline route.** A node's census comes out clean
only when its arms sum to zero, and a convex polyhedron's vertex lies
on its own hull, so every edge at it points into the supporting
half-space and the sum cannot vanish. So all twenty cages are
amorphous-walled by construction -- the collapse check is the only
guard they have, and it is recorded in each json.

Five supertube indices have no preset of their own ((4,4), (5,5),
(8,8), (8,0), (10,0)); they are the same construction as (6,6) at
different indices and take its parameters, which is stated rather than
hidden.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import write

from nanocarbon_lab.analyse.hybridisation import (
    collapsed_wall,
    hybridisation_report,
)
from nanocarbon_lab.builders.supernetwork import build_supernetwork

#: ``name -> (scale, tube_radius, blend)``, from each entry's own
#: preset. The five supertube indices without a preset carry the
#: supertube preset's numbers.
SHIPPED: dict[str, tuple[float, float, float]] = {
    # The periodic net the study missed, and the one that matters most:
    # super-diamond is eight nodes of four arms, the same object the
    # crystalline route calls Schwarz D. Measuring both is the only
    # controlled comparison of the two routes on one net.
    "super-diamond": (60.0, 5.0, 4.0),
    # 4-cube projected into three dimensions, so its struts come out in
    # three different lengths. `scale` sets the shortest.
    "super-hypercube": (20.0, 3.5, 2.5),
    # Platonic cages.
    "super-tetrahedron": (24.0, 3.0, 2.0),
    "super-cube": (24.0, 3.0, 2.0),
    "super-octahedron": (24.0, 3.0, 2.0),
    "super-dodecahedron": (24.0, 3.0, 2.0),
    "super-icosahedron": (24.0, 4.0, 3.0),
    # Archimedean cages. The strut is 30 and not the Platonic 24 for a
    # measured reason: at 24/3/2 the cuboctahedron comes back broken.
    "super-truncated-tetrahedron": (30.0, 4.0, 3.0),
    "super-cuboctahedron": (30.0, 4.0, 3.0),
    "super-truncated-cube": (30.0, 4.0, 3.0),
    "super-truncated-octahedron": (30.0, 4.0, 3.0),
    "super-rhombicuboctahedron": (30.0, 4.0, 3.0),
    "super-icosidodecahedron": (30.0, 4.0, 3.0),
    # A C60 whose every bond is a tube.
    "superfullerene-C60": (14.2, 3.0, 2.0),
    # Super-graphene rolled. (6,6) and (6,0) have presets; the rest are
    # the same construction at other indices.
    "supertube-(6,6)": (14.0, 3.0, 2.0),
    "supertube-(6,0)": (14.0, 3.0, 2.0),
    "supertube-(4,4)": (14.0, 3.0, 2.0),
    "supertube-(5,5)": (14.0, 3.0, 2.0),
    "supertube-(8,8)": (14.0, 3.0, 2.0),
    "supertube-(8,0)": (14.0, 3.0, 2.0),
    "supertube-(10,0)": (14.0, 3.0, 2.0),
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    wanted = dict(SHIPPED)
    if "--solo" in sys.argv:
        only = sys.argv[sys.argv.index("--solo") + 1]
        if only not in SHIPPED:
            print(f"{only!r} no está; hay {list(SHIPPED)}")
            return 2
        wanted = {only: SHIPPED[only]}

    for name, (scale, radius, blend) in wanted.items():
        target = out / f"{name}.extxyz"
        record = out / f"{name}.json"
        if target.exists() and record.exists():
            print(f"{name}: already done", flush=True)
            continue
        started = time.time()
        try:
            atoms = build_supernetwork(name, scale=scale, tube_radius=radius,
                                       blend=blend, seed=0)
        except Exception as error:                   # noqa: BLE001
            print(f"{name}: REFUSED -- {type(error).__name__}: "
                  f"{str(error).splitlines()[0][:120]}", flush=True)
            continue

        bonds = atoms.info.get("bonds")
        report = (hybridisation_report(atoms.get_positions(), bonds,
                                       np.asarray(atoms.cell))
                  if bonds else {})
        collapsed = collapsed_wall(report) if report else None
        lengths = np.empty(0)
        if bonds:
            lengths = np.linalg.norm(atoms.get_distances(
                [i for i, _ in bonds], [j for _, j in bonds],
                mic=True, vector=True), axis=1)

        handed = atoms.copy()
        handed.info.clear()
        write(str(target), handed, format="extxyz")
        info = {k: v for k, v in atoms.info.items()
                if isinstance(v, (int, float, str, bool, list, dict))}
        record.write_text(json.dumps({
            "scale": scale, "tube_radius": radius, "blend": blend,
            "atomos": len(atoms),
            "celda": [round(float(x), 3) for x in atoms.cell.lengths()],
            "pbc": [bool(p) for p in atoms.pbc],
            "collapsed": collapsed,
            "angle_sum_min": report.get("angle_sum_min"),
            "angle_sum_max": report.get("angle_sum_max"),
            "bond_min": float(lengths.min()) if lengths.size else None,
            "bond_max": float(lengths.max()) if lengths.size else None,
            "info": info,
        }, indent=1, default=str))
        print(f"{name}: {len(atoms)} atoms, cell "
              f"{[round(float(x), 1) for x in atoms.cell.lengths()]} A, "
              f"pbc {[bool(p) for p in atoms.pbc]}, angle sum from "
              f"{report.get('angle_sum_min', float('nan')):.1f} deg, bonds "
              f"{lengths.min() if lengths.size else float('nan'):.3f}-"
              f"{lengths.max() if lengths.size else float('nan'):.3f} A, "
              f"{'COLLAPSED' if collapsed else 'sound'}, "
              f"{time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
