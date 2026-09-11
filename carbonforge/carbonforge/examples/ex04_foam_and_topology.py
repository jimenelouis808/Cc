"""Example 04 — build a 3D disordered carbon foam, analyse its topology
with networkx and export for LAMMPS relaxation.
"""

from pathlib import Path

from carbonforge.builders import build_carbon_foam
from carbonforge.exports.lammps import write_lammps
from carbonforge.topology import coordination_numbers, ring_statistics
from carbonforge.validation import run_basic_checks


def main() -> None:
    atoms = build_carbon_foam(
        box_size=30.0, n_flakes=25, flake_radius=4.0, seed=42
    )
    print(atoms)
    print("density (g/cm^3):", atoms.info["density_g_cm3"])
    print("mean coordination:", coordination_numbers(atoms).mean())
    print("ring statistics:", ring_statistics(atoms, max_ring=8))
    print(run_basic_checks(atoms).summary())

    # Foams are pre-relaxation: force=True is acceptable for the MD starting point.
    data, inp = write_lammps(atoms, Path("out/foam"), force=True)
    print(f"LAMMPS files: {data}, {inp}")


if __name__ == "__main__":
    main()
