"""Paramagnetic energy of super-graphene at U = 2|t| on the same halved 8x8
mesh as magnetismo_k8.py: a real magnetic solution must lie below it.

    python energia_k8.py   -> adds "paramagnetic_energy_eV" to magnetismo_k8.json
"""
from __future__ import annotations

import json

from tbkit.hamiltonian import System
from tbkit.hubbard import mean_field
from tbkit.kpoints import mesh
from tbkit.params import pi_model

from magnetismo import structure
from magnetismo_k8 import NAME, N, OUT, RATIO, T, half_mesh

system = System.build(structure(NAME), pi_model())
k, w = half_mesh(*mesh(system, N))
r = mean_field(system, U=RATIO * T, kpts=k, weights=w, kT=0.005, guess="paramagnetic",
               mixing=0.3, tol=1e-6, max_iter=500)
d = json.loads(OUT.read_text())
d["paramagnetic_energy_eV"] = r.energy
d["paramagnetic_converged"] = r.converged
d["magnetic_minus_paramagnetic_meV"] = 1000 * (d["energy_eV"] - r.energy)
OUT.write_text(json.dumps(d, indent=0))
print(r.converged, r.iterations, d["magnetic_minus_paramagnetic_meV"], "meV", flush=True)
