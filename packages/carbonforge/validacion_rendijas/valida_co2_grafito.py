"""CO2 on one graphite surface against Bottani, Bojan, Steele & Ismail,
Langmuir 10 (1994) 3805: the molecule-surface virial B_AS/A and q_st(0).

B_AS/A = integral over height of <exp(-u/kT) - 1>, averaged over the
lateral position and the orientation; q_st(0) = RT - <u e^-bu>/<e^-bu>.
The surface is atomistic AB graphite (5 layers), the same LJ the GCMC uses.
"""
from __future__ import annotations

import sys

import numpy as np
from ase import Atoms

from carbonforge.adsorption.co2 import CO2System, molecule, random_axis

R = 8.314462618e-3   # kJ/mol/K
LAYER = 3.35
A = 2.46


def graphite(nx: int = 10, ny: int = 6, layers: int = 5, top: float = 20.0):
    b = A * np.sqrt(3.0)
    base = np.array([[0, 0], [A / 2, b / 6], [A / 2, b / 2], [0, 2 * b / 3]])
    shift = np.array([0.0, b / 3])            # AB stacking
    pos = []
    for k in range(layers):
        for i in range(nx):
            for j in range(ny):
                for p in base:
                    xy = p + [i * A, j * b] + (k % 2) * shift
                    pos.append([xy[0], xy[1], top - k * LAYER])
    edges = np.array([nx * A, ny * b, 70.0])
    return Atoms("C" * len(pos), positions=pos, cell=edges, pbc=True), edges


def virial(system: CO2System, temperature: float, top: float,
           samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    heights = np.arange(2.4, 14.0, 0.05)
    beta = 1.0 / temperature
    boltz = np.empty(len(heights))
    energy = np.empty(len(heights))
    for n, h in enumerate(heights):
        u = np.empty(samples)
        for s in range(samples):
            c = np.array([rng.uniform(0, system.edges[0]),
                          rng.uniform(0, system.edges[1]), top + h])
            u[s] = system.wall_energy(molecule(c, random_axis(rng)))
        w = np.exp(-beta * np.minimum(u, 50 / beta) )
        w[u > 50 / beta] = 0.0
        boltz[n] = w.mean()
        energy[n] = (w * u).mean()
    dz = heights[1] - heights[0]
    b_area = float(np.sum(boltz - 1.0) * dz + heights[0] * -1.0)
    q = R * temperature - R * float(energy.sum() / boltz.sum())
    return b_area, q


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    atoms, edges = graphite()
    system = CO2System(atoms, edges)
    print(f"grafito {len(atoms)} C, celda {edges.round(2)}")
    for t in (173, 203, 233, 278, 298):
        b, q = virial(system, t, 20.0, samples, seed=t)
        print(f"T {t:4d} K   B_AS/A {b:9.1f} A   ln(B/A [cm]) {np.log(b*1e-8):7.3f}"
              f"   q_st(0) {q:5.2f} kJ/mol", flush=True)
