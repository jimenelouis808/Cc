"""Does one carrier per cell switch on flat-band magnetism?

Undoped, the flat band of super-square / super-graphene sits AT E_F and
U = |t| leaves everything non-magnetic (magnetismo.py). A flat band that is
PARTLY filled is the textbook route to ferromagnetism (Mielke, Tasaki):
the exchange gain beats a kinetic cost close to zero. One N or B per cell
moves the filling by one electron; this tests the rigid-band version of
that (charge = -1 / +1 per cell, no dopant potential), U = |t|, ferro and
random starts, the same k meshes as magnetismo.py.

Rigid band is the optimistic limit: a real N or B also adds a local
potential that can pull a state out of the flat band. That comes after.

    python dopaje_banda_plana.py   -> dopaje_banda_plana.json (resumable)
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
from magnetismo_k8 import half_mesh

HERE = Path(__file__).resolve().parent
OUT = HERE / "dopaje_banda_plana.json"
NAMES = ("graphene (control)", "knee-super-square", "knee-super-graphene")
CHARGES = (-1.0, +1.0)            # per cell: -1 = one extra electron (N-like)


def main(names) -> None:
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    model = pi_model()
    for name in names:
        system = System.build(structure(name), model)
        k, w = half_mesh(*mesh(system, CASES[name]))
        for q in CHARGES:
            for guess in ("ferro", "random", "paramagnetic"):
                key = f"{name} | q={q:+.0f} | {guess}"
                if key in results:
                    continue
                t0 = time.time()
                r = mean_field(system, charge=q, kpts=k, weights=w, kT=0.005, guess=guess,
                               mixing=0.3, tol=1e-6, max_iter=1500)
                m = r.populations[0] - r.populations[1]
                results[key] = {"structure": name, "charge": q, "guess": guess,
                                "energy_eV": r.energy, "magnetization_muB": r.magnetization,
                                "abs_moment_sum_muB": float(np.abs(m).sum()),
                                "max_local_moment_muB": float(np.abs(m).max()),
                                "converged": r.converged, "iterations": r.iterations,
                                "seconds": time.time() - t0, "moments": m.round(5).tolist()}
                OUT.write_text(json.dumps(results, indent=0))
                print(f"{key:48s} E {r.energy:.5f} M {r.magnetization:+.3f} "
                      f"max {np.abs(m).max():.3f} conv {r.converged} ({time.time() - t0:.0f} s)",
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(NAMES))
