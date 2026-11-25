"""Is super-graphene's weak moment at U = 2|t| real, or k sampling?

Graphene's control moment at 2.0|t| goes 0.022 -> 0.006 -> 3e-7 uB for
12x12 -> 24x24 -> 48x48 k: a Dirac point sampled by few k makes a moment
that is not there. Super-graphene's (max 0.036 uB, on the nodes) came from
4x4 k on a linear crossing. This reruns it on 8x8, STARTING FROM the 4x4
magnetic state: a real solution stays, an artefact decays to zero.

Two economies, both exact:
- time reversal: real H, collinear spin, no field -> n(k) = n(-k), so one of
  each +-k pair is kept with double weight (64 -> 34 k points);
- checkpointing: the SCF runs in chunks of CHUNK iterations, chaining
  mean_field through its returned (mixed) populations, which is the same
  linear-mixing sequence as one long call. Every chunk is saved to
  magnetismo_k8.npz; a rerun resumes from it. A killed worker loses one chunk.

    python magnetismo_k8.py   -> magnetismo_k8.json, magnetismo_k8.npz
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tbkit.hamiltonian import System
from tbkit.hubbard import mean_field
from tbkit.kpoints import mesh
from tbkit.params import pi_model

from magnetismo import structure

HERE = Path(__file__).resolve().parent
NAME, N, RATIO, T = "knee-super-graphene", 8, 2.0, 2.7
CHECK, OUT = HERE / "magnetismo_k8.npz", HERE / "magnetismo_k8.json"
CHUNK, TOL, MAX_ITER = 10, 1e-6, 3000


def half_mesh(k: np.ndarray, w: np.ndarray):
    """One of each +-k pair (fractional coordinates), weights summed."""
    keys = {}
    for ki, wi in zip(k, w):
        key = tuple(np.round(np.mod(ki, 1.0), 8) % 1.0)
        neg = tuple(np.round(np.mod(-ki, 1.0), 8) % 1.0)
        if neg in keys:
            keys[neg][1] += wi
        else:
            keys[key] = [ki, wi]
    kk = np.array([v[0] for v in keys.values()])
    ww = np.array([v[1] for v in keys.values()])
    return kk, ww


def main() -> None:
    system = System.build(structure(NAME), pi_model())
    k, w = half_mesh(*mesh(system, N))
    if CHECK.exists():
        c = np.load(CHECK)
        pops, done, history = c["pops"], int(c["iterations"]), list(c["history"])
    else:
        start = json.loads((HERE / "magnetismo_U.json").read_text())[f"{NAME} | U={RATIO}|t|"]
        m = np.array(start["moments"])
        pops = np.array([0.5 + m / 2, 0.5 - m / 2])       # one π orbital, n0 = 1
        done, history = 0, []
    print(f"{len(k)} k points (8x8 halved by time reversal); resuming at {done}", flush=True)
    while done < MAX_ITER:
        t0 = time.time()
        r = mean_field(system, U=RATIO * T, kpts=k, weights=w, kT=0.005, guess=pops,
                       mixing=0.3, tol=TOL, max_iter=CHUNK)
        pops, done = r.populations, done + r.iterations
        history += r.history
        m = pops[0] - pops[1]
        np.savez(CHECK, pops=pops, iterations=done, history=np.array(history))
        print(f"it {done:5d}  change {r.history[-1]:.1e}  max|m| {np.abs(m).max():.5f}  "
              f"({(time.time() - t0) / r.iterations:.0f} s/it)", flush=True)
        if r.converged:
            break
    m = pops[0] - pops[1]
    OUT.write_text(json.dumps({
        "structure": NAME, "kmesh": N, "k_points_used": len(k), "U_over_t": RATIO,
        "start": "4x4 magnetic solution (magnetismo_U.json)", "energy_eV": r.energy,
        "magnetization_muB": r.magnetization, "abs_moment_sum_muB": float(np.abs(m).sum()),
        "max_local_moment_muB": float(np.abs(m).max()), "converged": r.converged,
        "iterations": done, "history": [float(h) for h in history],
        "moments": m.round(6).tolist()}, indent=0))
    print("done", r.converged, np.abs(m).max(), flush=True)


if __name__ == "__main__":
    main()
