"""Build the four superlattices and hand them over as extxyz.

Run inside nanocarbon_lab. carbonforge reads the files; neither package
imports the other, which is the workspace's one rule.

Parameters are the shipped presets, not invented ones -- the audit in
nanocarbon_lab/CLAUDE.md is at these numbers and super-fcc at any other
scale runs for an hour and a half.
"""
import json, sys, time
from pathlib import Path
from ase.io import write
from nanocarbon_lab.builders.supernetwork import build_supernetwork

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "super-square":   dict(scale=34.0, tube_radius=5.0, blend=4.0),
    "super-graphene": dict(scale=34.0, tube_radius=5.0, blend=4.0),
    "super-cubic":    dict(scale=34.0, tube_radius=5.0, blend=4.0),
    "super-fcc":      dict(scale=40.0, tube_radius=3.0, blend=2.0),
}

for name, kwargs in PRESETS.items():
    target = OUT / f"{name}.extxyz"
    if target.exists():
        print(f"{name}: ya está", flush=True)
        continue
    t = time.time()
    atoms = build_supernetwork(name, seed=0, **kwargs)
    info = {k: v for k, v in atoms.info.items()
            if isinstance(v, (int, float, str, bool, list, dict))}
    # The handoff file carries geometry, cell and periodicity and
    # nothing else. The builder's `info` holds the ring census and the
    # Euler budget as nested structures, which extxyz writes and then
    # cannot read back (`KeyError: '_JSON'` on the way in), so the
    # metadata goes in the json beside it, where it is readable.
    handed = atoms.copy()
    handed.info.clear()
    write(str(target), handed, format="extxyz")
    (OUT / f"{name}.json").write_text(json.dumps(
        {"preset": kwargs, "atomos": len(atoms),
         "celda": [round(float(x), 3) for x in atoms.cell.lengths()],
         "pbc": [bool(p) for p in atoms.pbc], "info": info},
        indent=1, default=str))
    print(f"{name}: {len(atoms)} átomos, celda "
          f"{[round(float(x),1) for x in atoms.cell.lengths()]}, "
          f"pbc {list(atoms.pbc)}, {time.time()-t:.0f} s", flush=True)
