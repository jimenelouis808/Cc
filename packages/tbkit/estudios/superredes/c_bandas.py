"""Check the π-model bands with c_bandas (s + p, fitted to GPAW PBE bands of
graphene and tubes; electronic only). With σ orbitals it sees the σ-π
mixing that the curvature of the nodes causes, which the π model cannot.

Validity of c_bandas: C-C 1.37-1.48 Å and ±3 eV of E_F. The knee
structures reach 1.35-1.49 Å at their nodes, slightly outside: stated.

    python c_bandas.py [name ...]   -> resultados_c_bandas.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from tbkit.analysis import bands
from tbkit.hamiltonian import System
from tbkit.kpoints import band_path, mesh
from tbkit.params import load_parameters

from electronica import STRUCTURES, load

HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados_c_bandas.json"
CASES = {"schwarzite-D": 3, "knee-super-square": 6, "knee-super-graphene": 4}


def main(names) -> None:
    model = load_parameters("c_bandas")
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        if name in results:
            continue
        started = time.time()
        atoms = load(name)
        system = System.build(atoms, model)
        # Eigenvalues only: 4 orbitals per atom make the eigenvectors of a
        # whole k mesh ~14 GB for Schwarz D, which killed the first run.
        # Neutral carbon fills 2 bands per atom (4 electrons, spin pairs).
        kpts, weights = mesh(system, CASES[name])
        em = bands(system, kpts)[0]                       # [k, band]
        nocc = 2 * len(atoms)
        vbm, cbm = float(em[:, nocc - 1].max()), float(em[:, nocc].min())
        ef = 0.5 * (vbm + cbm)
        grid = np.linspace(ef - 3, ef + 3, 1201)
        total = np.zeros_like(grid)
        for wk, ek in zip(weights / weights.sum(), em):
            total += 2 * wk * np.exp(-0.5 * ((grid[:, None] - ek[None]) / 0.03) ** 2
                                     ).sum(axis=1) / (0.03 * np.sqrt(2 * np.pi))
        bp = band_path(atoms, STRUCTURES[name][1], 60)
        x, sx, labels = bp.get_linear_kpoint_axis()
        ep = bands(system, bp)[0]
        vbm = max(vbm, float(ep[:, nocc - 1].max()))
        cbm = min(cbm, float(ep[:, nocc].min()))
        ef = 0.5 * (vbm + cbm)
        e = ep - ef
        near = np.any(np.abs(e) < 2.0, axis=0)
        gap = max(0.0, cbm - vbm)
        results[name] = {"atoms": len(atoms), "kmesh": CASES[name], "fermi": ef,
                         "gap": gap, "vbm": vbm, "cbm": cbm,
                         "dos": {"e_minus_ef": (grid - ef).round(4).tolist(),
                                 "note": "Gaussian 0.03 eV on the k mesh only",
                                 "states_per_ev_per_atom": (total / len(atoms)).round(5).tolist()},
                         "bands": {"k": x.round(5).tolist(), "special_k": sx.round(5).tolist(),
                                   "labels": list(labels),
                                   "e_minus_ef": e[:, near].round(5).tolist()},
                         "seconds": time.time() - started}
        OUT.write_text(json.dumps(results, indent=0))
        print(f"{name:22s} gap {gap:.3f} eV (mesh + path)  "
              f"({time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(CASES))
