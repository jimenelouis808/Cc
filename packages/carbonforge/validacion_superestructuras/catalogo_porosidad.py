"""Porosity of every superstructure in the workspace catalogue.

Seconds per structure, no simulation: which ones admit H2 at all, how
much area they offer per gram, and how wide their channels are -- the
two numbers the slit series and the GCMC runs say decide uptake at 77 K.

The catalogue (``catalogo/xyz``) holds crystals, sheets, tubes and
finite cages, and the porosity routine needs three periodic orthorhombic
axes, so each is put in a cell by the rule ``validacion_rendijas/
empotra.py`` set for the earlier series:

* a periodic axis is replicated until it clears 24 Å (the same crystal);
* a free axis gets the structure's extent plus a **3.4 Å gap** between
  images, the graphitic contact: the densest solid these objects form
  without touching. That is a packing choice, recorded per row, and it
  puts the outside of a cage or tube in the measured area, as it would
  be in a powder.

    python catalogo_porosidad.py [--catalogue ../../../catalogo/xyz] [--workers 3]

Writes ``catalogo_porosidad.json`` and a table sorted by area.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase.io import read

from carbonforge.adsorption.porosity import H2_KINETIC_DIAMETER, analyse_porosity

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "catalogo_porosidad.json"
MIN_EDGE = 24.0
GAP = 3.4
#: Chahine's rule at 77 K: about 1 wt% of H2 per 500 m2/g.
CHAHINE = 500.0


def packed(path: Path):
    """The structure in a 3-periodic orthorhombic cell, and how."""
    atoms = read(path)
    cell = np.asarray(atoms.cell, dtype=float)
    lengths = atoms.cell.lengths()
    if np.abs(cell - np.diag(lengths)).max() > 1e-6:
        raise ValueError("cell is not orthorhombic")
    positions = atoms.get_positions()
    periodic = [bool(p) and lengths[k] > 0 for k, p in enumerate(atoms.pbc)]
    new = np.array(lengths, dtype=float)
    for k in range(3):
        if not periodic[k]:
            lo, hi = positions[:, k].min(), positions[:, k].max()
            positions[:, k] += GAP / 2.0 - lo
            new[k] = hi - lo + GAP
    atoms.set_positions(positions)
    atoms.set_cell(new)
    atoms.pbc = True
    reps = [max(1, int(np.ceil(MIN_EDGE / x))) for x in new]
    how = {"periodic_axes": int(sum(periodic)), "gap": GAP, "repeat": reps}
    return (atoms.repeat(reps) if max(reps) > 1 else atoms), how


def measure(path: Path) -> dict:
    atoms, how = packed(path)
    report = analyse_porosity(atoms, samples=60_000)
    mass_g = report.framework_mass_g
    return {"file": path.name, "atoms_in_cell": len(atoms), **how,
            "density_g_cm3": mass_g / (report.cell_volume_a3 * 1e-24),
            "area_m2_g": report.accessible_area_m2_g,
            "helium_void_fraction": report.helium_void_fraction,
            "pore_volume_cm3_g": report.pore_volume_cm3_g,
            "largest_free_sphere": report.largest_free_sphere,
            "largest_included_sphere": report.largest_included_sphere,
            "admits_h2": report.largest_free_sphere >= H2_KINETIC_DIAMETER,
            "chahine_wt_pct": report.accessible_area_m2_g / CHAHINE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalogue", type=Path,
                        default=HERE.parents[2] / "catalogo" / "xyz")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    # The two-period supertubes are the same structures as the one-period
    # files at twice the atoms.
    todo = [p for p in sorted(args.catalogue.glob("*.xyz"))
            if "2periodos" not in p.name and p.name not in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        for path, future in [(p, pool.submit(measure, p)) for p in todo]:
            try:
                rows[path.name] = future.result()
            except Exception as exc:              # recorded, not fatal
                rows[path.name] = {"file": path.name, "error": str(exc)}
            RESULTS.write_text(json.dumps(rows, indent=1))
            r = rows[path.name]
            print(f"{path.name}: " + (f"{r['area_m2_g']:.0f} m2/g, channel "
                  f"{r['largest_free_sphere']:.1f} A" if "error" not in r
                  else r["error"]), flush=True)


if __name__ == "__main__":
    main()
