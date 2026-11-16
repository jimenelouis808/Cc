"""The crystalline schwarzites: Schwarz D and the gyroid.

`build_knee_supernetwork` refuses these two -- they repeat in all
three directions, so they are schwarzite cells rather than sheets and
`build_knee_schwarzite` is their builder. They are crystalline 3D
carbon nothing in this folder has measured, and they are the honest
answer to "what else could be optimised": their walls come out at
1.347-1.489 A with no pentagon at all.
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
from nanocarbon_lab.builders.knee import build_knee_schwarzite

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
# `kind` is a KEYWORD argument and its first positional is the
# circumference, so `build_knee_schwarzite("d")` tries int("d"). The
# valid kinds are the keys of SCHWARZITE_CELLS; each takes its own
# default shape, which is a measured best rather than a round number.
for kind in ("primitive", "diamond", "gyroid"):
    name = f"knee-schwarzite-{kind}"
    target = out / f"{name}.extxyz"
    record = out / f"{name}.json"
    if target.exists() and record.exists():
        print(f"{name}: already done", flush=True)
        continue
    t = time.time()
    try:
        atoms = build_knee_schwarzite(kind=kind)
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
    info = {k: v for k, v in atoms.info.items()
            if isinstance(v, (int, float, str, bool, list, dict))}
    record.write_text(json.dumps(
        {"kind": kind, "atomos": len(atoms),
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
