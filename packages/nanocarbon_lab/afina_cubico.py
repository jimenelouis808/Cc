"""Build a grid of super-cubic cells and record whether each is sound.

    python afina_cubico.py <carpeta>

carbonforge's `validacion_rendijas/llenado_super.py` reads the extxyz
this writes. Neither package imports the other, so everything the
measurement needs about a structure's soundness has to travel in the
json beside it -- which is the point of this script rather than a
convenience.

**Why the soundness verdict is written here.** The adsorbate scan
ended on the observation that super-cubic's 20 A channel comes from
the preset 34/5/4 and not from any optimum, while the slit that beats
it wins by holding exactly one molecular layer. `scale` and
`tube_radius` are arguments, so the superlattice has the same tuning
available -- and tightening a cell is exactly what pinches a wall
shut. A pinched tube IS a narrow slit: it offers spuriously deep sites
and a high Henry coefficient for the wrong reason. Measured on this
very grid, the tightest cell that built came back with the HIGHEST
coefficient of the ten (23128 mol/kg/MPa against the best slit's 8660)
and an angle sum of 325.2 deg -- past tetrahedral, so not a
hybridisation at all. Without the verdict travelling with the file,
that is the number anybody would have quoted.

`collapsed_wall` is the check, and it is not a tolerance: no carbon is
bent further than three tetrahedral angles, 328.4 deg. The existing
quality gate cannot see it, because bonds and contacts are local and a
collapse is not.
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
from nanocarbon_lab.builders.supernetwork import FREE_TUBE, build_supernetwork

#: Smooth-union radius at the vertices. 2.0 is the value the presets
#: use for small tubes; it also sets the floor, since the builder
#: refuses `scale <= 2*(tube_radius + blend) + FREE_TUBE`.
BLEND = 2.0

#: Tube radii to try. 2.0 is included and is expected to fail: the
#: mesher's own floor is about 3 A, and a 2 A tube carries under eight
#: rings around its circumference.
RADII = (2.0, 3.0, 4.0, 5.0)

#: Offsets above each radius's own floor. Laid out this way rather than
#: as absolute scales because with BLEND = 2 the floor's channel is
#: about 4 A whatever the radius -- the range the slit optimum sits in.
OFFSETS = (0.5, 2.0, 5.0, 10.0)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    for radius in RADII:
        floor = 2.0 * (radius + BLEND) + FREE_TUBE
        for offset in OFFSETS:
            scale = floor + offset
            name = f"super-cubic_r{radius:g}_s{scale:g}"
            target = out / f"{name}.extxyz"
            record = out / f"{name}.json"
            # BOTH files, not just the structure. The json carries the
            # soundness verdict the measurement filters on, so an
            # extxyz without one is not a finished handoff -- and
            # skipping on the structure alone silently left a whole
            # grid unusable after the verdict was added.
            if target.exists() and record.exists():
                print(f"{name}: already done", flush=True)
                continue
            started = time.time()
            try:
                atoms = build_supernetwork("super-cubic", scale=scale,
                                           tube_radius=radius, blend=BLEND,
                                           seed=0)
            except Exception as error:               # noqa: BLE001
                # A refusal is a result: the tight end of this grid is
                # mostly unbuildable and that is the finding, not a
                # failed run.
                print(f"{name}: REFUSED -- {type(error).__name__}: "
                      f"{str(error).splitlines()[0][:120]}", flush=True)
                continue

            bonds = atoms.info.get("bonds")
            report = (hybridisation_report(atoms.get_positions(), bonds,
                                           np.asarray(atoms.cell))
                      if bonds else {})
            collapsed = collapsed_wall(report) if report else None

            handed = atoms.copy()
            handed.info.clear()
            write(str(target), handed, format="extxyz")
            info = {k: v for k, v in atoms.info.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
            record.write_text(json.dumps({
                "scale": scale, "tube_radius": radius, "blend": BLEND,
                "atomos": len(atoms),
                "celda": [round(float(x), 3) for x in atoms.cell.lengths()],
                # What the measurement needs and cannot work out for
                # itself without importing this package.
                "collapsed": collapsed,
                "angle_sum_min": report.get("angle_sum_min"),
                "angle_sum_max": report.get("angle_sum_max"),
                "sp3_fraction": report.get("sp3_fraction"),
                "info": info,
            }, indent=1, default=str))
            print(f"{name}: {len(atoms)} atoms, cell "
                  f"{atoms.cell.lengths()[0]:.1f} A, angle sum from "
                  f"{report.get('angle_sum_min', float('nan')):.1f} deg, "
                  f"{'COLLAPSED' if collapsed else 'sound'}, "
                  f"{time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
