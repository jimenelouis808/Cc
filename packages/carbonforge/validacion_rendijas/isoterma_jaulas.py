"""Real H2 isotherms on the two cages, to test the "getter" verdict.

    uv run python validacion_rendijas/isoterma_jaulas.py escribe
    uv run python validacion_rendijas/isoterma_jaulas.py corre super-cube

The Henry study found super-cube 5.6 times the best slit in initial
slope, and ESTIMATED its capacity at 0.38 mol/kg from one molecule per
vertex pocket (`bolsillos.py`). That estimate is what calls the cage a
getter rather than a store, and only an isotherm can test it: if the
loading levels off near 0.4 mol/kg the verdict stands measured; if it
keeps rising, the channels hold gas too and the verdict is wrong.

`escribe` makes one `carbonforge-isoterma/1` folder per cage under
`isotermas/`, embedded exactly as the Henry runs were (`llenado_super.
embed`), at 77 K over eight pressures from 1 Pa to 10 MPa, log-spaced,
so the low end resolves the pockets filling and the high end the
plateau. `corre` runs one folder; a finished point is never recomputed,
so a killed run resumes where it stopped (`lanza_isotermas.sh` relaunches
whatever is not running). Super-tetrahedron is the control: the other
cage the Henry study ranked, with four pockets instead of eight.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ase.io import read

from carbonforge.adsorption.campaign import (
    read_results,
    run_isotherm,
    write_isotherm_request,
)

from llenado_super import embed

HERE = Path(__file__).resolve().parent
CAGES = ("super-cube", "super-tetrahedron")

#: MPa. 1 Pa to 10 MPa, one point per decade and a last at 10 MPa: the
#: Henry estimate puts half loading near 8 Pa, so the first three points
#: bracket the pockets filling.
PRESSURES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def folder_of(name: str) -> Path:
    return HERE / "isotermas" / name


def write_all() -> None:
    for name in CAGES:
        folder = folder_of(name)
        if (folder / "isoterma.json").exists():
            print(f"{name}: ya escrita")
            continue
        cell, repeat, gap = embed(read(str(HERE / "jaulas" / f"{name}.extxyz")))
        write_isotherm_request(
            cell, folder, temperature=77.0, pressures=PRESSURES, name=name,
            seed=0, snapshots=2,
            reason=("Confirmar la saturación estimada (~0.4 mol/kg, una "
                    "molécula por bolsillo de vértice) que hace de la jaula "
                    "un captador y no un almacén."))
        print(f"{name}: {len(cell)} átomos, réplicas {repeat}, hueco "
              f"{gap} Å -> {folder}")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("escribe", "corre"):
        print(__doc__)
        return 2
    if sys.argv[1] == "escribe":
        write_all()
        return 0
    name = sys.argv[2]
    run_isotherm(folder_of(name))
    for row in read_results(folder_of(name)):
        print(row["presion_mpa"], {k: v for k, v in row.items()
                                   if "mol" in k or "carga" in k})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
