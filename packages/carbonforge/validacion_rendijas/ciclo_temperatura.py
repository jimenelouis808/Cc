"""Temperature-swing working capacity of the functionalised 6.6 A slits,
adsorption force field. Adsorb at 298 K and 15 kPa CO2 (single-component
estimate of a 15 % flue gas); desorb at 373 K either under a purge that
keeps 15 kPa CO2, or under pure CO2 at 0.1 MPa (the product stream).
Interpolation in log P between simulated points; mol/kg."""
from __future__ import annotations

import json

from capacidad_trabajo import loading
from funcional_co2 import FOLDER, VARIANTS

if __name__ == "__main__":
    print(f"{'rendija':10s} {'298K 15kPa':>11s} {'373K 15kPa':>11s} "
          f"{'373K 100kPa':>12s} {'TSA purga':>10s} {'TSA CO2':>8s}")
    for name in VARIANTS:
        hot = FOLDER / f"funcional-{name}-ads-373K.json"
        cold = FOLDER / f"funcional-{name}-ads.json"
        if not cold.exists():
            cold = FOLDER / f"funcional-{name}.json"   # no groups: same
        if not hot.exists():
            continue
        c = json.loads(cold.read_text())["puntos"]
        h = json.loads(hot.read_text())["puntos"]
        a, b15, b100 = loading(c, 0.015), loading(h, 0.015), loading(h, 0.1)
        print(f"{name:10s} {a:11.2f} {b15:11.2f} {b100:12.2f} "
              f"{a - b15:10.2f} {a - b100:8.2f}")
