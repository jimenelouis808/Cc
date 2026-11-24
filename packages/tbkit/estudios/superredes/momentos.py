"""Where the mean-field moments sit: heptagon (node) atoms, their
neighbours, and the hexagonal arms, by graph distance to a heptagon.

    python momentos.py   -> momentos.json (from magnetismo_U.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from tbkit.sites import ring_atoms

from magnetismo import structure

HERE = Path(__file__).resolve().parent


def main() -> None:
    data = json.loads((HERE / "magnetismo_U.json").read_text())
    out = {}
    for key, r in data.items():
        m = np.abs(np.array(r["moments"]))
        if m.sum() < 0.1 or r["structure"].startswith("graphene"):
            continue
        atoms = structure(r["structure"])
        hept = sorted(ring_atoms(atoms, sizes=(7,))[7])
        i, j = neighbor_list("ij", atoms, 1.7)
        graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(atoms),) * 2)
        dist = shortest_path(graph, unweighted=True, indices=hept).min(axis=0)
        shells = {}
        for lo, hi, label in ((0, 0, "heptagon atoms"), (1, 2, "1-2 bonds away"),
                              (3, 5, "3-5 bonds away"), (6, 99, ">= 6 bonds away")):
            sel = (dist >= lo) & (dist <= hi)
            if sel.any():
                shells[label] = {"atoms": int(sel.sum()), "mean_abs_m": float(m[sel].mean()),
                                 "share_of_sum": float(m[sel].sum() / m.sum())}
        out[key] = {"heptagon_atoms": len(hept), "max_graph_distance": int(dist.max()),
                    "shells": shells}
        print(key)
        for k, v in shells.items():
            print(f"   {k:18s} n={v['atoms']:4d}  <|m|>={v['mean_abs_m']:.3f}  "
                  f"share={v['share_of_sum']:.2f}")
    (HERE / "momentos.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
