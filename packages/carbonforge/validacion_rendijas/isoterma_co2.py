"""CO2 isotherms at 298 K, resumable point by point.

    uv run python validacion_rendijas/isoterma_co2.py corre super-cube
    uv run python validacion_rendijas/isoterma_co2.py tabla

Three-site TraPPE CO2 with Ewald (`carbonforge.adsorption.co2`) on
super-cube, super-tetrahedron and the slit that wins CO2's Henry sweep
(6.6 A, `henry_co2.txt`), from 1 kPa to 1 MPa. Widom alone does not
converge on the cages -- four seeds of 150 000 insertions scatter by up
to a factor of two, the slope being set by deep pockets random
insertion rarely finds -- so the cages are compared by isotherm, which
explores them by moving molecules.

Each finished point is appended to `isotermas_co2/<name>.json` at once
and skipped on a rerun, so a container restart costs only the point in
flight (`lanza_co2.sh` relaunches). The upper limit is 1 MPa because
the empty-box test holds TraPPE to the Peng-Robinson reservoir within
2-3 % up to 2 MPa and drifts to 8 % low at 4 MPa.

The excess uses the helium void fraction of the framework, as for H2.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ase.io import read

from carbonforge.adsorption.co2 import run_co2_gcmc
from carbonforge.adsorption.porosity import analyse_porosity

from llenado import slit
from llenado_super import embed

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "isotermas_co2"
NAMES = ("super-cube", "super-tetrahedron", "rendija-6.6")
PRESSURES = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
TEMPERATURE = 298.0


def framework(name: str):
    if name.startswith("rendija"):
        return slit(float(name.split("-")[1]))
    return embed(read(str(HERE / "jaulas" / f"{name}.extxyz")))[0]


def run(name: str) -> None:
    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / f"{name}.json"
    stored = json.loads(path.read_text()) if path.exists() else {}
    atoms = framework(name)
    if "vacio_helio" not in stored:
        stored["vacio_helio"] = analyse_porosity(atoms).helium_void_fraction
        stored["puntos"] = []
        path.write_text(json.dumps(stored, indent=1))
    done = {p["pressure_mpa"] for p in stored["puntos"]}
    for k, pressure in enumerate(PRESSURES):
        if pressure in done:
            continue
        started = time.time()
        result = run_co2_gcmc(atoms, TEMPERATURE, pressure,
                              equilibration=100_000, production=200_000,
                              seed=k, void_fraction=stored["vacio_helio"])
        row = asdict(result) | {"segundos": time.time() - started}
        stored["puntos"].append(row)
        stored["puntos"].sort(key=lambda r: r["pressure_mpa"])
        path.write_text(json.dumps(stored, indent=1))
        print(f"{name} {pressure} MPa: {result.mol_per_kg:.3f} mol/kg "
              f"({row['segundos']:.0f} s)", flush=True)


def table() -> None:
    for name in NAMES:
        path = FOLDER / f"{name}.json"
        if not path.exists():
            continue
        stored = json.loads(path.read_text())
        print(f"== {name} (vacío de helio {stored['vacio_helio']:.3f})")
        for p in stored["puntos"]:
            print(f"  {p['pressure_mpa']:7.3g} MPa  N {p['mean_count']:8.2f}"
                  f" ± {p['count_error']:.2f}  {p['mol_per_kg']:7.3f} mol/kg "
                  f"abs  {p['excess_mol_per_kg']:7.3f} exc  "
                  f"avisos {len(p['warnings'])}")


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "corre":
        run(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "tabla":
        table()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
