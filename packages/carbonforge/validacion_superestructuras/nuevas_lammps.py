"""H2 at 77 K in the new crystalline superstructures, by LAMMPS.

The structures nanocarbon_lab added after the first six were measured:
the hexagonal D (lonsdaleite) and I-WP (bcc) knee cells, and the
interpenetrated pairs. Read from the workspace catalogue as extxyz,
measured with ``analyse_porosity`` first (a structure that admits no H2
is reported and skipped), then three GCMC points each at the pressures
of the schwarzite and slit series, with the same trial counts and the
helium void fraction for the excess.

    LAMMPS_EXECUTABLE=/path/to/lmp python nuevas_lammps.py salida/ [--workers 3]

Points already in ``nuevas_lammps.json`` are not rerun.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ase.io import read

from carbonforge.adsorption.gcmc import CUTOFF
from carbonforge.adsorption.lammps import run_lammps_gcmc
from carbonforge.adsorption.porosity import H2_KINETIC_DIAMETER, analyse_porosity

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "nuevas_lammps.json"
STRUCTURES = {
    "lonsdaleite": "knee-schwarzite-lonsdaleite_k18_r5.xyz",
    "bcc": "knee-schwarzite-bcc_k36_r13.xyz",
    "doble-diamante": "knee-schwarzite-diamond_k18_r7_double.xyz",
    "doble-giroide-geodesico": "super-srs_m1n0_f4_r1_w3_double.xyz",
    "doble-giroide-codos": "knee-schwarzite-gyroid_k20_r9_double.xyz",
    "doble-primitiva": "super-cubic_m2n0_f2_r2_w1_double.xyz",
}
PRESSURES = (1.0, 3.8, 10.0)
EQUILIBRATION = PRODUCTION = 1_500_000


def framework(catalogue: Path, name: str):
    """The cell, repeated until every edge clears twice the cutoff.

    Repeating a periodic cell does not change the crystal; it only lets
    the minimum image hold at the 12 Å cutoff.
    """
    atoms = read(catalogue / STRUCTURES[name])
    reps = [max(1, int(-(-2 * CUTOFF // x))) for x in atoms.cell.lengths()]
    return atoms.repeat(reps) if max(reps) > 1 else atoms


def porosity(catalogue: Path, name: str) -> dict:
    report = analyse_porosity(framework(catalogue, name), samples=100_000)
    return {"area_m2_g": report.accessible_area_m2_g,
            "helium_void_fraction": report.helium_void_fraction,
            "pore_volume_cm3_g": report.pore_volume_cm3_g,
            "largest_free_sphere": report.largest_free_sphere,
            "largest_included_sphere": report.largest_included_sphere}


def run_point(out: Path, catalogue: Path, name: str, pressure: float,
              helium: float) -> dict:
    atoms = framework(catalogue, name)
    result = run_lammps_gcmc(
        atoms, out / f"{name}_P{pressure:g}", 77.0, pressure,
        equilibration=EQUILIBRATION, production=PRODUCTION, seed=11,
        void_fraction=helium)
    excess_g = result.excess_density_g_cm3 * float(atoms.get_volume()) * 1e-24
    return {"name": name, "pressure_mpa": pressure, "atoms": len(atoms),
            "rho_o": result.absolute_density_g_cm3,
            "weight_percent": result.weight_percent,
            "excess_weight_percent": 100 * excess_g
            / (excess_g + result.carbon_mass_g),
            "count": result.mean_count, "error": result.count_error,
            "warnings": result.warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--catalogue", type=Path,
                        default=HERE.parents[2] / "catalogo" / "xyz")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    state = (json.loads(RESULTS.read_text()) if RESULTS.exists()
             else {"porosity": {}, "points": []})
    for name in STRUCTURES:
        if name not in state["porosity"]:
            state["porosity"][name] = porosity(args.catalogue, name)
            RESULTS.write_text(json.dumps(state, indent=1))
    have = {(r["name"], r["pressure_mpa"]) for r in state["points"]}
    todo = []
    for name, pore in state["porosity"].items():
        if pore["largest_free_sphere"] < H2_KINETIC_DIAMETER:
            print(f"{name}: largest free sphere {pore['largest_free_sphere']:.2f}"
                  f" A, under H2's {H2_KINETIC_DIAMETER} A -- no GCMC.",
                  flush=True)
            continue
        todo += [(name, p, pore["helium_void_fraction"]) for p in PRESSURES
                 if (name, p) not in have]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_point, args.out, args.catalogue, *job)
                   for job in todo]
        for future in futures:
            row = future.result()
            state["points"].append(row)
            RESULTS.write_text(json.dumps(state, indent=1))
            print(f"{row['name']} {row['pressure_mpa']:g} MPa: "
                  f"{row['weight_percent']:.2f} wt% abs, "
                  f"{row['excess_weight_percent']:.2f} exc, N {row['count']:.0f}"
                  f" +- {row['error']:.0f}", flush=True)


if __name__ == "__main__":
    main()
