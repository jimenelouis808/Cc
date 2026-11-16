"""Sweep the crystalline route's shape, for adsorption rather than bonds.

    python barre_cristalinas.py <carpeta>
    python barre_cristalinas.py <carpeta> --familia diamond

The crystalline route already covers the meshed superlattice catalogue,
under different names, and that is worth writing down before anybody
ports anything:

| meshed net     | V, E, coord | budget | crystalline equivalent      |
|----------------|-------------|--------|-----------------------------|
| super-cubic    | 1, 3, 6     | -24    | Schwarz P ("primitive")     |
| super-diamond  | 8, 16, 4    | -96    | Schwarz D ("diamond")       |
| super-graphene | 4, 6, 3     | -24    | knee sheet, same name       |
| super-square   | 1, 2, 4     | -12    | knee sheet, same name       |
| super-fcc      | 4, 24, 12   | -240   | NONE -- 12 arms cannot close|

A super-cubic cell is one node of six arms, which is exactly Schwarz
P's node and budget; a super-diamond cell is eight nodes of four arms,
which is Schwarz D's. So "build super-cubic crystalline" means
`build_knee_schwarzite(kind="primitive")`, and the gyroid is a net the
meshed catalogue does not have at all.

**What the shapes need is not a port but a different objective.**
`DEFAULT_SCHWARZITE_SHAPE` and `DEFAULT_SUPERNET_SHAPE` were chosen
for the relaxed bond spread, which is the right objective for a
structure and the wrong one for an adsorbent: measured against H2 at
77 K, uptake per gram peaks for a free channel of roughly 6 to 10 A
and falls two orders of magnitude by 15 A. Schwarz P's default (20, 9)
carries a 14.61 A channel and reads 352 mol/kg/MPa, where Schwarz D's
(18, 5) carries 6.72 A and reads 12861. Same route, same wall quality,
a factor of 37 between them, and nothing but the shape differs.

So this sweeps the shape and records what an adsorption study needs to
choose on: the census and its budget, the bond window, the minimum
angle sum, and the cell. The channel itself is measured on the
carbonforge side, which owns the porosity code.
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
from nanocarbon_lab.builders.knee import (
    build_knee_schwarzite,
    build_knee_supernetwork,
)

#: ``family -> (builder kind, circumferences, arm row counts)``.
#:
#: The ranges bracket each family's own shipped default rather than
#: sharing one window, because what closes is per family: a P cell
#: wants a wide arm and a long one, a D cell the opposite (eight nodes
#: in one cube leave far less room), and the gyroid wants a fatter tube
#: than either because its arms leave at 120 deg against the diamond
#: node's 109.47.
SWEEP: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]] = {
    # = crystalline super-cubic. Default (20, 9), channel 14.61 A.
    "primitive": ("schwarzite", (12, 14, 16, 18, 20), (3, 5, 7, 9)),
    # = crystalline super-diamond. Default (18, 5), channel 6.72 A.
    "diamond": ("schwarzite", (14, 16, 18, 20, 22), (3, 5, 7)),
    # No meshed counterpart. Default (20, 5), channel 15.80 A.
    "gyroid": ("schwarzite", (16, 18, 20, 22, 24), (3, 5, 7)),
    # Sheets. Defaults (22, 5) and (12, 5).
    "super-graphene": ("sheet", (14, 18, 22, 26), (3, 5, 7)),
    "super-square": ("sheet", (12, 16, 20), (3, 5, 7)),
}


def build(family: str, route: str, k: int, rows: int):
    """Build one shape, or raise. Kept thin so the caller logs."""
    if route == "schwarzite":
        return build_knee_schwarzite(circumference=k, arm_rows=rows,
                                     kind=family)
    return build_knee_supernetwork(family, circumference=k, arm_rows=rows)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    families = dict(SWEEP)
    if "--familia" in sys.argv:
        wanted = sys.argv[sys.argv.index("--familia") + 1]
        if wanted not in SWEEP:
            print(f"familia {wanted!r} desconocida; hay {list(SWEEP)}")
            return 2
        families = {wanted: SWEEP[wanted]}

    for family, (route, circumferences, row_counts) in families.items():
        for k in circumferences:
            for rows in row_counts:
                name = f"knee-{family}_k{k}_n{rows}"
                target = out / f"{name}.extxyz"
                record = out / f"{name}.json"
                if target.exists() and record.exists():
                    print(f"{name}: already done", flush=True)
                    continue
                started = time.time()
                try:
                    atoms = build(family, route, k, rows)
                except Exception as error:           # noqa: BLE001
                    # A refusal is a result: most shapes do not close,
                    # and which ones do is the finding.
                    print(f"{name}: REFUSED -- {type(error).__name__}: "
                          f"{str(error).splitlines()[0][:110]}", flush=True)
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
                info = {kk: v for kk, v in atoms.info.items()
                        if isinstance(v, (int, float, str, bool, list, dict))}
                record.write_text(json.dumps({
                    "family": family, "route": route,
                    "circumference": k, "arm_rows": rows,
                    "atomos": len(atoms),
                    "celda": [round(float(x), 3)
                              for x in atoms.cell.lengths()],
                    "collapsed": collapsed,
                    "angle_sum_min": report.get("angle_sum_min"),
                    "bond_min": float(lengths.min()) if lengths.size else None,
                    "bond_max": float(lengths.max()) if lengths.size else None,
                    "info": info,
                }, indent=1, default=str))
                print(f"{name}: {len(atoms)} atoms, cell "
                      f"{atoms.cell.lengths()[0]:.1f} A, angle sum from "
                      f"{report.get('angle_sum_min', float('nan')):.1f} deg, "
                      f"bonds "
                      f"{lengths.min() if lengths.size else float('nan'):.3f}"
                      f"-{lengths.max() if lengths.size else float('nan'):.3f}"
                      f" A, {'COLLAPSED' if collapsed else 'sound'}, "
                      f"census {info.get('ring_counts')}, "
                      f"{time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
