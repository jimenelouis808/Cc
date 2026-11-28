"""The nine schwarzite points again, run by LAMMPS (fix gcmc).

The same three knee cells (Schwarz P, D and the gyroid, read from the
workspace catalogue as extxyz), the same 77 K, 1 / 3.8 / 10 MPa, the same
trial counts and the same helium void fraction as the in-house run in the
three ``.json`` files beside this script, so every point compares two
engines on curved walls, after ``validacion_GCMC/`` compared them on
Levesque's bundle.

    LAMMPS_EXECUTABLE=/path/to/lmp python lammps_kowalczyk.py salida/ \\
        [--catalogue ../../../catalogo/xyz] [--workers 3]

A point already in ``datos_lammps.json`` is not rerun.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ase.io import read

from carbonforge.adsorption.lammps import run_lammps_gcmc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "datos_lammps.json"
FILES = {"primitive": "knee-schwarzite-primitive_k20_r9.xyz",
         "diamond": "knee-schwarzite-diamond_k18_r5.xyz",
         "gyroid": "knee-schwarzite-gyroid_k20_r5.xyz"}
PRESSURES = (1.0, 3.8, 10.0)


def run_point(out: Path, catalogue: Path, kind: str, pressure: float) -> dict:
    """One state point; returns its row."""
    in_house = json.loads((HERE / f"{kind}.json").read_text())
    sampling = in_house["muestreo"]
    framework = read(catalogue / FILES[kind])
    result = run_lammps_gcmc(
        framework, out / f"{kind}_P{pressure:g}", 77.0, pressure,
        equilibration=sampling["equilibrado"],
        production=sampling["produccion"], seed=11,
        void_fraction=in_house["porosidad"]["helium_void_fraction"])
    return {"kind": kind, "pressure_mpa": pressure,
            "rho_o": result.absolute_density_g_cm3,
            "weight_percent": result.weight_percent,
            "excess_g_cm3": result.excess_density_g_cm3,
            "count": result.mean_count, "error": result.count_error,
            "atoms": len(framework), "warnings": result.warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--catalogue", type=Path,
                        default=HERE.parents[2] / "catalogo" / "xyz")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    done = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    have = {(r["kind"], r["pressure_mpa"]) for r in done}
    todo = [(k, p) for k in FILES for p in PRESSURES if (k, p) not in have]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_point, args.out, args.catalogue, k, p)
                   for k, p in todo]
        for future in futures:
            row = future.result()
            done.append(row)
            RESULTS.write_text(json.dumps(done, indent=1))
            print(f"{row['kind']} {row['pressure_mpa']:g} MPa: rho_o "
                  f"{row['rho_o']:.5f}, {row['weight_percent']:.2f} wt%, "
                  f"N {row['count']:.0f} +- {row['error']:.0f}", flush=True)


if __name__ == "__main__":
    main()
