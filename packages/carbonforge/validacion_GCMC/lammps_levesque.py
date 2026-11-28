"""Levesque's sixteen points again, run by LAMMPS (fix gcmc).

The same bundle, the same Feynman-Hibbs model, the same reservoir and the
same trial counts as the in-house run in ``datos.json``, so the table this
writes compares three columns: the paper, the in-house GCMC and LAMMPS.

    LAMMPS_EXECUTABLE=/path/to/lmp python lammps_levesque.py salida/ [--workers 2]

Each point is its own LAMMPS folder under ``salida/``; a point whose
result is already in ``datos_lammps.json`` is not rerun.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from carbonforge.adsorption.bundle import (
    LEVESQUE_BUNDLE_ISOTHERMS,
    levesque_bundle,
)
from carbonforge.adsorption.lammps import run_lammps_gcmc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "datos_lammps.json"

#: Trial moves per point, as the in-house validation ran them.
MOVES = {293.0: (150_000, 300_000), 77.0: (1_500_000, 1_500_000)}


def run_point(out: Path, temperature: float, pressure: float) -> dict:
    """One state point; returns its row."""
    equilibration, production = MOVES[temperature]
    result = run_lammps_gcmc(
        levesque_bundle(gap=6.0), out / f"T{temperature:g}_P{pressure:g}",
        temperature, pressure, equilibration=equilibration,
        production=production, seed=11)
    return {"temperature": temperature, "pressure_mpa": pressure,
            "rho_o": result.absolute_density_g_cm3,
            "count": result.mean_count, "error": result.count_error,
            "warnings": result.warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    done = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    have = {(r["temperature"], r["pressure_mpa"]) for r in done}
    todo = [(t, p) for t in (293.0, 77.0)
            for p, _rho, _wt in LEVESQUE_BUNDLE_ISOTHERMS[t]
            if (t, p) not in have]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_point, args.out, t, p) for t, p in todo]
        for future in futures:
            row = future.result()
            done.append(row)
            RESULTS.write_text(json.dumps(done, indent=1))
            print(f"{row['temperature']:g} K {row['pressure_mpa']:g} MPa: "
                  f"rho_o {row['rho_o']:.5f}, N {row['count']:.0f} "
                  f"+- {row['error']:.0f}", flush=True)


if __name__ == "__main__":
    main()
