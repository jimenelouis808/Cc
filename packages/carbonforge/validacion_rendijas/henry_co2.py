"""CO2's Henry constant at 298 K on the slits and the two cages.

    uv run python validacion_rendijas/henry_co2.py rendijas
    uv run python validacion_rendijas/henry_co2.py jaulas

Three-site TraPPE CO2 (`carbonforge.adsorption.co2`), Widom insertion
with random centre and orientation, against the same frameworks the H2
study measured: AA graphene slits from `llenado.slit` across widths,
and super-cube and super-tetrahedron embedded as for the isotherms.
The slit width that wins for CO2 need not be H2's, which is the point
of the sweep. Rows go to `henry_co2.txt` as they finish.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ase.io import read

from carbonforge.adsorption.co2 import henry_co2

from llenado import slit
from llenado_super import embed

HERE = Path(__file__).resolve().parent
OUT = HERE / "henry_co2.txt"
TEMPERATURE = 298.0
SAMPLES = 120_000
WIDTHS = (6.0, 6.6, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0)


def report(label: str, atoms) -> None:
    started = time.time()
    k_h, q_st = henry_co2(atoms, TEMPERATURE, samples=SAMPLES, seed=0)
    line = (f"{label:24s} {len(atoms):6d} C   K_H {k_h:12.4g} mol/kg/MPa   "
            f"q_st {q_st:6.2f} kJ/mol   ({time.time() - started:.0f} s)")
    print(line, flush=True)
    with OUT.open("a") as handle:
        handle.write(line + "\n")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "rendijas"
    if which == "rendijas":
        for width in WIDTHS:
            report(f"rendija H={width}", slit(width))
    else:
        for name in ("super-cube", "super-tetrahedron"):
            cell, _repeat, _gap = embed(read(str(HERE / "jaulas"
                                                 / f"{name}.extxyz")))
            report(name, cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
