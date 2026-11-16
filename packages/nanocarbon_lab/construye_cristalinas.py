"""super-cubic by the crystalline route, as the control the meshed one needs.

The meshed route's wall is amorphous and, measured, not round: the
shipped preset reads 4.68 +- 0.69 A about a requested 5.0 with a
minimum of 2.85, and the tuned cells reach 1.1-1.9 A of a 4 A axis.
Across the whole grid the Boltzmann-weighted well depth ranks almost
exactly INVERSE to the minimum angle sum, which is the signature of a
well produced by wall damage rather than by design.

`build_knee_supernetwork` places the rings by construction instead, so
its wall is crystalline and cannot be pinched. If a crystalline net
shows the same deep well, the well is a design; if it does not, the
meshed numbers were damage.

**It does not hold super-cubic.** The route carries `diamond`,
`gyroid`, `super-graphene` and `super-square` and nothing else -- the
line in nanocarbon_lab/CLAUDE.md listing super-cubic among the nets
is about the budget LAW agreeing across the catalogue, not about what
this builder can make. So the control is run on the nets it does
hold, which is no loss: super-graphene and super-square are the two
meshed nets that sit beside super-cubic in the measured table, and
diamond and gyroid are crystalline schwarzites nothing here has
measured at all.
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
from nanocarbon_lab.builders.knee import build_knee_supernetwork

#: Net to (circumference, arm rows). Each is the builder's own default
#: for that net, which is a measured minimum of the bond spread rather
#: than a round number -- see DEFAULT_SUPERNET_SHAPE.
#: `diamond` and `gyroid` are in this route's list and are NOT here:
#: they repeat in all three directions, so they are schwarzite cells
#: rather than sheets and `build_knee_schwarzite` is their builder --
#: `construye_schwarzitas.py` beside this file. Asking for them here
#: only earns a refusal on every run.
NETS = {
    "super-graphene": (22, 5),
    "super-square": (12, 5),
}

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for net, (k, rows) in NETS.items():
    name = f"knee-{net}_k{k}"
    target = out / f"{name}.extxyz"
    record = out / f"{name}.json"
    if target.exists() and record.exists():
        print(f"{name}: already done", flush=True)
        continue
    t = time.time()
    try:
        atoms = build_knee_supernetwork(net, circumference=k,
                                        arm_rows=rows)
    except Exception as e:                       # noqa: BLE001
        print(f"{name}: REFUSED -- {type(e).__name__}: "
              f"{str(e).splitlines()[0][:130]}", flush=True)
        continue
    bonds = atoms.info.get("bonds")
    rep = (hybridisation_report(atoms.get_positions(), bonds,
                                np.asarray(atoms.cell)) if bonds else {})
    collapsed = collapsed_wall(rep) if rep else None
    handed = atoms.copy()
    handed.info.clear()
    write(str(target), handed, format="extxyz")
    info = {kk: v for kk, v in atoms.info.items()
            if isinstance(v, (int, float, str, bool, list, dict))}
    record.write_text(json.dumps(
        {"net": net, "circumference": k, "arm_rows": rows, "atomos": len(atoms),
         "celda": [round(float(x), 3) for x in atoms.cell.lengths()],
         "collapsed": collapsed,
         "angle_sum_min": rep.get("angle_sum_min"),
         "info": info}, indent=1, default=str))
    print(f"{name}: {len(atoms)} atoms, cell "
          f"{atoms.cell.lengths()[0]:.1f} A, angle sum from "
          f"{rep.get('angle_sum_min', float('nan')):.1f} deg, "
          f"{'COLLAPSED' if collapsed else 'sound'}, "
          f"census {info.get('ring_counts')}, {time.time()-t:.0f} s",
          flush=True)
