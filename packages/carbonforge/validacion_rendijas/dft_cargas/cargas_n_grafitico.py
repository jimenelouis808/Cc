"""Partial charges of graphitic N and its carbon neighbours, from DFT.

Runs OUTSIDE carbonforge's environment, in one with PySCF and ASE
(``pip install pyscf ase``); carbonforge does not depend on PySCF.

Model: an H-terminated graphene flake cut round one carbon, which is
replaced by N (graphitic, substitutional, three C neighbours). The flake
has an odd electron count, as graphitic N in graphene donates one
electron to the pi system, so it is run spin-unrestricted as a doublet.
PBE/def2-SVP. Two charge schemes, because they disagree and the
disagreement is the uncertainty:

* **IAO** (intrinsic atomic orbitals, Knizia 2013): a population
  analysis that is stable in the basis, unlike Mulliken.
* **ESP fit** (CHELPG-like): charges that reproduce the electrostatic
  potential on shells at 1.4-2.0 vdW radii, total charge 0. This is the
  scheme force fields for adsorption are usually fitted with.

    python cargas_n_grafitico.py [radius_A]   -> cargas_n_grafitico.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.build import graphene
from pyscf import dft, gto, lo

HERE = Path(__file__).resolve().parent
VDW = {"H": 1.20, "C": 1.70, "N": 1.55}


def flake(radius: float):
    """Symbols and positions of an H-terminated flake with N at its centre."""
    sheet = graphene(a=2.46, size=(14, 14, 1), vacuum=10.0)
    pos = sheet.get_positions()
    centre = pos.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(pos - centre, axis=1)))
    pos = pos - pos[first]
    pos = pos[np.linalg.norm(pos[:, :2], axis=1) <= radius]
    # Drop atoms with a single neighbour until none is left.
    while True:
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        coord = ((d > 0.5) & (d < 1.6)).sum(axis=1)
        if coord.min() >= 2:
            break
        pos = pos[coord >= 2]
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    neigh = [np.flatnonzero((row > 0.5) & (row < 1.6)) for row in d]
    hydrogens = []
    for i, nb in enumerate(neigh):
        if len(nb) == 2:
            out = 2 * pos[i] - pos[nb[0]] - pos[nb[1]]
            hydrogens.append(pos[i] + 1.09 * out / np.linalg.norm(out))
    symbols = ["C"] * len(pos)
    centre_atom = int(np.argmin(np.linalg.norm(pos, axis=1)))
    symbols[centre_atom] = "N"
    return symbols + ["H"] * len(hydrogens), np.vstack([pos, hydrogens]), \
        centre_atom, [int(j) for j in neigh[centre_atom]]


def esp_charges(mol, dm, coords, symbols) -> np.ndarray:
    """Least-squares charges to the ESP on vdW shells, sum fixed at 0."""
    rng = np.random.default_rng(0)
    points = []
    for scale in (1.4, 1.6, 1.8, 2.0):
        for i, s in enumerate(symbols):
            u = rng.normal(size=(60, 3))
            u /= np.linalg.norm(u, axis=1)[:, None]
            points.append(coords[i] + scale * VDW[s] * u)
    points = np.vstack(points)
    radii = np.array([VDW[s] for s in symbols])
    dist = np.linalg.norm(points[:, None] - coords[None], axis=-1)
    points = points[np.all(dist >= 1.4 * radii[None] - 1e-9, axis=1)]
    bohr = points / 0.52917721092
    nuclear = sum(mol.atom_charge(i) / np.linalg.norm(
        bohr - mol.atom_coord(i), axis=1) for i in range(mol.natm))
    electronic = np.einsum("pij,ij->p", mol.intor("int1e_grids", grids=bohr),
                           dm)
    v = nuclear - electronic
    a = 1.0 / np.linalg.norm(bohr[:, None] - (coords / 0.52917721092)[None],
                             axis=-1)
    n = len(symbols)
    lhs = np.zeros((n + 1, n + 1))
    lhs[:n, :n] = a.T @ a
    lhs[:n, n] = lhs[n, :n] = 1.0
    rhs = np.concatenate([a.T @ v, [0.0]])
    return np.linalg.solve(lhs, rhs)[:n]


def iao_charges(mol, mf) -> np.ndarray:
    pmol = mol.copy()
    pmol.basis = "minao"
    pmol.build()
    charges = np.array([mol.atom_charge(i) for i in range(mol.natm)], float)
    for spin in range(2):
        occ = mf.mo_coeff[spin][:, mf.mo_occ[spin] > 0]
        c = lo.iao.iao(mol, occ)
        c = lo.vec_lowdin(c, mf.get_ovlp())
        proj = c.T @ mf.get_ovlp() @ occ
        pop = np.einsum("ik,ik->i", proj, proj)
        for i, (_s, _e, p0, p1) in enumerate(pmol.aoslice_by_atom()):
            charges[i] -= pop[p0:p1].sum()
    return charges


def main() -> None:
    radius = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    symbols, coords, n_index, neighbours = flake(radius)
    electrons = sum({"H": 1, "C": 6, "N": 7}[s] for s in symbols)
    mol = gto.M(atom=list(zip(symbols, coords)), basis="def2-svp",
                spin=electrons % 2, unit="Angstrom", verbose=0)
    started = time.time()
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.max_cycle = 200
    mf.conv_tol = 1e-8
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    dm = mf.make_rdm1()
    esp = esp_charges(mol, dm[0] + dm[1], coords, symbols)
    iao = iao_charges(mol, mf)
    result = {
        "flake": "".join(f"{s}{symbols.count(s)}" for s in ("C", "N", "H")),
        "radius_A": radius, "spin_2S": mol.spin,
        "method": "UKS PBE/def2-SVP",
        "seconds": time.time() - started,
        "N": {"iao": float(iao[n_index]), "esp": float(esp[n_index])},
        "C_N": {"iao": [float(iao[j]) for j in neighbours],
                "esp": [float(esp[j]) for j in neighbours]},
        "C_sp2_mean_interior": {
            "iao": float(np.mean([iao[i] for i, s in enumerate(symbols)
                                  if s == "C" and i not in neighbours
                                  and np.linalg.norm(coords[i]) < radius - 2.5])),
            "esp": float(np.mean([esp[i] for i, s in enumerate(symbols)
                                  if s == "C" and i not in neighbours
                                  and np.linalg.norm(coords[i]) < radius - 2.5])),
        },
    }
    out = HERE / f"cargas_n_grafitico_r{radius:g}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
