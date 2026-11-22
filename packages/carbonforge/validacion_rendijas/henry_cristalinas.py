"""CO2 Henry constant at 298 K of the crystalline superstructures in
``cristalinas/`` (neutral Steele walls, as the slit of henry_co2.py).
Rows already in henry_cristalinas.txt are skipped."""
from __future__ import annotations

import numpy as np

from carbonforge.adsorption.co2 import henry_co2
from carbonforge.adsorption.porosity import analyse_porosity

from selectividad import CRYSTALLINE, HERE, structure

if __name__ == "__main__":
    out = HERE / "henry_cristalinas.txt"
    done = out.read_text() if out.exists() else ""
    for name in CRYSTALLINE:
        if any(line.split()[:1] == [name] for line in done.splitlines()):
            continue
        atoms, _ = structure(name)
        k, q = zip(*[henry_co2(atoms, 298.0, samples=200_000, seed=s)
                     for s in range(2)])
        channel = analyse_porosity(atoms).largest_free_sphere
        line = (f"{name:40s} canal {channel:6.2f} A  K_H {np.mean(k):8.1f} "
                f"({min(k):.0f}-{max(k):.0f})  q_st {np.mean(q):5.2f}")
        print(line, flush=True)
        with out.open("a") as handle:
            handle.write(line + "\n")
