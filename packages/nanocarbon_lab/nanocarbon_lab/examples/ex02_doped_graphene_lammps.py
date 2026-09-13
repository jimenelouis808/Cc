"""Example 02 — N-doped graphene supercell with a Stone-Wales defect,
exported to LAMMPS for MD annealing.
"""

from pathlib import Path

from nanocarbon_lab.builders import build_graphene_supercell
from nanocarbon_lab.defects import stone_wales_defect
from nanocarbon_lab.dopants import dope_random
from nanocarbon_lab.exports.lammps import LAMMPSSettings, write_lammps


def main() -> None:
    sheet = build_graphene_supercell(nx=5, ny=5)
    sheet = stone_wales_defect(sheet, seed=0)
    sheet = dope_random(sheet, "N", concentration=0.02, seed=1)

    settings = LAMMPSSettings(nvt_temperature_k=500.0, nvt_steps=20000)
    data, inp = write_lammps(sheet, Path("out/doped_graphene"), settings=settings)
    print(f"LAMMPS data → {data}\nLAMMPS input → {inp}")


if __name__ == "__main__":
    main()
