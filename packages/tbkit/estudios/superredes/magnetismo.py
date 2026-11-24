"""Mean-field Hubbard magnetism of the superstructures with flat π bands.

A flat band at E_F makes a paramagnet unstable at any U (Stoner: the
susceptibility diverges with the DOS). The superstructures are NOT
bipartite (heptagons), so Lieb's theorem does not fix the answer and the
antiferro sublattice guess does not exist: each structure starts from
paramagnetic, ferro and random patterns, and the lowest energy wins.

U = |t| = 2.7 eV (tbkit's pi_huckel default, Yazyev 2010), below
graphene's mean-field critical U (~2.2|t|): graphene must come out
non-magnetic, and is the control. Schwarz D (gap 1.25 eV) is the second
control: a gapped π system must stay non-magnetic too.

Mean-field moments are an order parameter, not a correlated ground state.

    python magnetismo.py [name ...]   -> magnetismo.json (resumable per run)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.build import graphene

from tbkit.hamiltonian import System
from tbkit.hubbard import mean_field
from tbkit.kpoints import mesh
from tbkit.params import pi_model

from electronica import load

HERE = Path(__file__).resolve().parent
OUT = HERE / "magnetismo.json"
#: name -> k mesh
CASES = {"graphene (control)": 24, "knee-super-square": 8,
         "knee-super-graphene": 4, "schwarzite-D": 2}
GUESSES = ("paramagnetic", "ferro", "random")


def structure(name):
    if name == "graphene (control)":
        g = graphene(a=2.46, size=(2, 2, 1), vacuum=10.0)
        g.pbc = (True, True, False)
        return g
    return load(name)


def main(names) -> None:
    model = pi_model()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        atoms = structure(name)
        system = System.build(atoms, model)
        k, w = mesh(system, CASES[name])
        for guess in GUESSES:
            key = f"{name} | {guess}"
            if key in results:
                continue
            started = time.time()
            r = mean_field(system, kpts=k, weights=w, kT=0.005, guess=guess,
                           mixing=0.3, tol=1e-6, max_iter=1500)
            moments = r.populations[0] - r.populations[1]
            results[key] = {
                "structure": name, "guess": guess, "atoms": len(atoms),
                "kmesh": CASES[name], "U_eV": float(np.max(r.U)),
                "energy_eV": r.energy, "magnetization_muB": r.magnetization,
                "abs_moment_sum_muB": float(np.abs(moments).sum()),
                "max_local_moment_muB": float(np.abs(moments).max()),
                "converged": r.converged, "iterations": r.iterations,
                "seconds": time.time() - started,
                "moments": moments.round(5).tolist()}
            OUT.write_text(json.dumps(results, indent=0))
            print(f"{key:40s} E {r.energy:14.6f} eV  M {r.magnetization:+8.3f}  "
                  f"Σ|m| {np.abs(moments).sum():7.3f}  max {np.abs(moments).max():.3f}  "
                  f"conv {r.converged} ({r.iterations} it, {time.time() - started:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
