"""Critical U of the flat-band superstructures in mean field.

At U = |t| (magnetismo.py) every structure stays non-magnetic. This scan
raises U from a symmetry-breaking start (antiferro on bipartite
graphene; random on the non-bipartite nets, whose ferro start keeps the
uniform symmetry and cannot reach a staggered state) and records where a moment first appears,
against graphene's mean-field threshold (~2.23|t|, Sorella-Tosatti /
Hirsch), the closed-form check. Mean-field moments are an order
parameter, not a correlated ground state; U above ~2|t| is far beyond
the screened U of carbon (Wehling 2011: ~1.6|t| on-site, less in mean
field) and is a measure of how close to an instability the net is.

    python magnetismo_U.py [name ...]   -> magnetismo_U.json (resumable)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from tbkit.hamiltonian import System
from tbkit.hubbard import mean_field
from tbkit.kpoints import mesh
from tbkit.params import pi_model

from magnetismo import CASES, structure

HERE = Path(__file__).resolve().parent
OUT = HERE / "magnetismo_U.json"
T = 2.7
RATIOS = (1.5, 2.0, 2.5)
NAMES = ("graphene (control)", "knee-super-square", "knee-super-graphene")


def main(names) -> None:
    model = pi_model()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        system = System.build(structure(name), model)
        k, w = mesh(system, CASES[name])
        for ratio in RATIOS:
            key = f"{name} | U={ratio}|t|"
            if key in results:
                continue
            started = time.time()
            guess = "antiferro" if name.startswith("graphene") else "random"
            r = mean_field(system, U=ratio * T, kpts=k, weights=w, kT=0.005,
                           guess=guess, mixing=0.3, tol=1e-6, max_iter=1500)
            m = r.populations[0] - r.populations[1]
            results[key] = {"structure": name, "guess": guess, "U_over_t": ratio, "U_eV": ratio * T,
                            "energy_eV": r.energy, "magnetization_muB": r.magnetization,
                            "abs_moment_sum_muB": float(np.abs(m).sum()),
                            "max_local_moment_muB": float(np.abs(m).max()),
                            "converged": r.converged, "iterations": r.iterations,
                            "seconds": time.time() - started, "moments": m.round(5).tolist()}
            OUT.write_text(json.dumps(results, indent=0))
            print(f"{key:40s} M {r.magnetization:+8.3f}  Σ|m| {np.abs(m).sum():7.3f} "
                  f"max {np.abs(m).max():.3f} conv {r.converged} ({time.time() - started:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(NAMES))
