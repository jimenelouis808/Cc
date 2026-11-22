"""CO2 Henry constants in the functionalised 6.6 A slits with the wall
charges replaced by published ones, to see how much of the step-3 verdict
rides on the representative charges of forcefields.parameters.

COOH and OH: Tenney & Lastoskie, Environ. Prog. 25 (2006) 343, Table 3
(the extracted table has lost its minus signs; they are restored so that
each group is neutral, as the paper's sp2-preserving construction is).
NH2: Chen et al., Phys. Chem. Chem. Phys. (2026), SI Table S1 (DFT).
Any residual net charge is spread over the C_sp2 atoms, as before.
"""
from __future__ import annotations

import numpy as np

from carbonforge.adsorption.co2 import henry_co2, wall_parameters
from funcional_co2 import HERE, TEMPERATURE, slit

LITERATURE = {
    "COOH": ("Tenney 2006", {"C_O": 0.75, "O_carbonyl": -0.50,
                             "O_hydroxyl": -0.55, "H_O": 0.36,
                             "C_sp3": -0.06}),
    "OH": ("Tenney 2006", {"C_O": 0.30, "O_hydroxyl": -0.60, "H_O": 0.30}),
    "NH2": ("Chen 2026", {"C_N": -0.172, "N_amine": -0.539, "H_N": 0.301}),
}


def literature_wall(atoms, table):
    wall = wall_parameters(atoms, neutralise=False)
    types = np.asarray(wall["types"])
    charge = np.array(wall["charge"], float)
    for t, q in table.items():
        charge[types == t] = q
    net = float(charge.sum())
    sp2 = types == "C_sp2"
    charge[sp2] -= net / sp2.sum()
    return dict(wall, charge=charge), net


if __name__ == "__main__":
    out = HERE / "cargas_literatura.txt"
    done = out.read_text() if out.exists() else ""
    for name, (source, table) in LITERATURE.items():
        if any(line.split()[:1] == [name] for line in done.splitlines()):
            continue
        atoms = slit(name)
        wall, net = literature_wall(atoms, table)
        k, q = zip(*[henry_co2(atoms, TEMPERATURE, samples=120_000, seed=s,
                               wall=wall) for s in range(2)])
        line = (f"{name:5s} {source:12s} carga neta {net:+.2f} e -> 0  "
                f"K_H {np.mean(k):8.1f} ({min(k):.0f}-{max(k):.0f})  "
                f"q_st {np.mean(q):5.2f}")
        print(line, flush=True)
        with out.open("a") as handle:
            handle.write(line + "\n")
