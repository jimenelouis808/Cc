"""Humid flue gas: CO2/N2/H2O in the functionalised 6.6 A slits.

15 % CO2 on a dry basis, the rest N2, plus water at a relative humidity
of the REAL saturation pressure at 298 K (3.17 kPa); total 0.1 MPa. Water
is SPC/E, whose own vapour pressure differs from water's, so the
humidity the model feels is not the one quoted -- the scan in RH is what
to read, not a single point.

GCMC from an empty cell. Water in a hydrophobic pore condenses through a
nucleation barrier, so an empty start can stay dry past the point where
the pore would fill: a dry answer at high humidity is a lower bound.

    python humedad.py corre <variant> <RH%>      (variant of funcional_co2)
    python humedad.py tabla
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict

from carbonforge.adsorption.co2 import CO2, N2, WATER, wall_parameters
from carbonforge.adsorption.mixture import run_mixture_gcmc

from funcional_co2 import HERE, slit

FOLDER = HERE / "humedad"
TEMPERATURE = 298.0
PRESSURE = 0.1
P_SAT_MPA = 3.169e-3              # water at 298.15 K (IAPWS)
DRY_CO2 = 0.15
EQUILIBRATION, PRODUCTION = 300_000, 600_000


def fractions(rh: float) -> tuple[float, float, float]:
    y_w = rh / 100.0 * P_SAT_MPA / PRESSURE
    return (DRY_CO2 * (1 - y_w), (1 - DRY_CO2) * (1 - y_w), y_w)


def run(variant: str, rh: float) -> None:
    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / f"{variant}-hr{rh:g}.json"
    if path.exists():
        return
    atoms = slit(variant)
    wall = wall_parameters(atoms, forcefield="adsorption")
    started = time.time()
    r = run_mixture_gcmc(atoms, TEMPERATURE, PRESSURE, fractions(rh),
                         species=(CO2, N2, WATER),
                         equilibration=EQUILIBRATION, production=PRODUCTION,
                         seed=0, wall=wall)
    path.write_text(json.dumps(asdict(r) | {"hr": rh,
                                            "segundos": time.time() - started},
                               indent=1))
    print(f"{variant} HR {rh:g} %: CO2 {r.mol_per_kg[0]:.3f}, H2O "
          f"{r.mol_per_kg[2]:.3f} mol/kg, S {r.selectivity:.0f}", flush=True)


def table() -> None:
    print(f"{'variante':10s} {'HR %':>5s} {'CO2':>7s} {'N2':>7s} {'H2O':>7s} "
          f"{'S CO2/N2':>12s}  (mol/kg)")
    for path in sorted(FOLDER.glob("*.json")):
        r = json.loads(path.read_text())
        name = path.stem.rsplit("-hr", 1)[0]
        c, n, w = r["mol_per_kg"]
        print(f"{name:10s} {r['hr']:5g} {c:7.3f} {n:7.3f} {w:7.3f} "
              f"{r['selectivity']:7.0f} ± {r['selectivity_error']:3.0f}  "
              f"avisos {len(r['warnings'])}")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "corre":
        run(sys.argv[2], float(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "tabla":
        table()
    else:
        print(__doc__)
        sys.exit(1)
