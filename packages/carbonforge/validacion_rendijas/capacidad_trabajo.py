"""Working capacity from the pure-CO2 isotherms at 298 K (mol/kg, absolute).

Two vacuum swings, by interpolation in log P between simulated points:

* VSA: adsorb at 100 kPa, regenerate at 10 kPa (pure CO2 feed).
* Flue gas, single-component estimate: adsorb at 15 kPa (the CO2 partial
  pressure of a 15 % flue gas at 1 bar), regenerate at 1 kPa. This
  ignores N2 competition, which ``selectividad.py`` measures separately.

A temperature swing needs an isotherm at the desorption temperature and
is not computed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "isotermas_co2"
STRUCTURES = {
    "rendija-6.6": "rendija-6.6.json",
    "super-cube": "super-cube.json",
    "super-tetrahedron": "super-tetrahedron.json",
    "prístina": "funcional-pristina.json",
    "N 5 %": "funcional-N5.json",
    "OH": "funcional-OH-ads.json",
    "COOH": "funcional-COOH-ads.json",
    "NH₂": "funcional-NH2-ads.json",
}
SWINGS = {"VSA 100→10 kPa": (0.1, 0.01), "gas 15→1 kPa": (0.015, 0.001)}


def loading(points: list[dict], pressure_mpa: float) -> float:
    p = np.array([q["pressure_mpa"] for q in points])
    n = np.array([q["mol_per_kg"] for q in points])
    order = np.argsort(p)
    return float(np.interp(np.log(pressure_mpa), np.log(p[order]), n[order]))


def main() -> None:
    head = "".join(f"{s:>18s}" for s in SWINGS)
    print(f"{'estructura':20s}{head}")
    for label, file in STRUCTURES.items():
        points = json.loads((FOLDER / file).read_text())["puntos"]
        cells = []
        for high, low in SWINGS.values():
            hi, lo = loading(points, high), loading(points, low)
            cells.append(f"{hi - lo:7.2f} ({hi:.2f}-{lo:.2f})")
        print(f"{label:20s}" + "".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
