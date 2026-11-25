"""One audit, the same for every superstructure of the adsorption study.

Reads the extxyz files handed to carbonforge (no import across packages:
files only) and measures, with one bond rule and one ring rule for all:
rings (smallest ring through every bond angle), defects touching defects,
bonds, angles, coordination, close contacts, the sp2 verdict and the
collapsed-wall check.

    python auditoria.py <folder with the extxyz>  -> auditoria.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

from nanocarbon_lab.analyse.hybridisation import collapsed_wall, hybridisation_report
from nanocarbon_lab.validation.quality import sp2_quality

HERE = Path(__file__).resolve().parent
CUTOFF = 1.80          # Å, C-C bond
FILES = {
    "Schwarz P (crystalline)": "cristalinas/knee-schwarzite-primitive.extxyz",
    "Schwarz D (crystalline)": "cristalinas/knee-schwarzite-diamond.extxyz",
    "gyroid (crystalline)": "cristalinas/knee-schwarzite-gyroid.extxyz",
    "super-graphene (crystalline)": "cristalinas/knee-super-graphene_k22.extxyz",
    "super-square (crystalline)": "cristalinas/knee-super-square_k12_n7.extxyz",
    "super-cubic (meshed)": "superestructuras/super-cubic.extxyz",
    "super-diamond (meshed)": "superestructuras/super-diamond.extxyz",
    "super-fcc (meshed)": "superestructuras/super-fcc.extxyz",
    "super-graphene (meshed)": "superestructuras/super-graphene.extxyz",
    "super-square (meshed)": "superestructuras/super-square.extxyz",
    "super-hypercube (meshed cage)": "superestructuras/super-hypercube.extxyz",
    "super-cube (meshed cage)": "jaulas/super-cube.extxyz",
    "super-tetrahedron (meshed cage)": "jaulas/super-tetrahedron.extxyz",
}


def rings(graph: nx.Graph, largest: int = 10) -> set[frozenset]:
    """Smallest ring through each angle j-i-k (shortest j..k path avoiding i)."""
    out = set()
    for i in graph:
        nbr = list(graph[i])
        sub = graph.copy()
        sub.remove_node(i)
        for a in range(len(nbr)):
            for b in range(a + 1, len(nbr)):
                try:
                    path = nx.shortest_path(sub, nbr[a], nbr[b])
                except nx.NetworkXNoPath:
                    continue
                if len(path) + 1 <= largest:
                    out.add(frozenset(path + [i]))
    return out


def audit(atoms) -> dict:
    i, j, d, D = neighbor_list("ijdD", atoms, CUTOFF)
    keep = i < j
    bonds = list(zip(i[keep].tolist(), j[keep].tolist()))
    graph = nx.Graph(bonds)
    graph.add_nodes_from(range(len(atoms)))
    coordination = Counter(dict(graph.degree()).values())
    found = rings(graph)
    census = Counter(len(r) for r in found)
    defects = {size: [r for r in found if len(r) == size] for size in (4, 5, 7, 8, 9)}
    edge_set = lambda r: {frozenset(e) for e in graph.subgraph(r).edges}  # noqa: E731
    def touching(a, b):
        ea = [edge_set(r) for r in defects[a]]
        eb = ea if a == b else [edge_set(r) for r in defects[b]]
        n = 0
        for x in range(len(ea)):
            for y in range(len(eb)):
                if (a == b and y <= x):
                    continue
                n += bool(ea[x] & eb[y])
        return n
    # angles from the bond vectors
    vec = {}
    for a, b, v in zip(i, j, D):
        vec.setdefault(a, []).append(v)
    angles = []
    for a, vs in vec.items():
        vs = np.array(vs)
        u = vs / np.linalg.norm(vs, axis=1)[:, None]
        for x in range(len(u)):
            for y in range(x + 1, len(u)):
                angles.append(np.degrees(np.arccos(np.clip(u[x] @ u[y], -1, 1))))
    # close non-bonded contacts (< 2 Å, not bonded)
    ci, cj = neighbor_list("ij", atoms, 2.0)
    bonded = {frozenset(e) for e in bonds}
    close = sum(1 for a, b in zip(ci, cj) if a < b and frozenset((a, b)) not in bonded)
    lengths = d[keep]
    geometry = {"bond_min": float(lengths.min()), "bond_max": float(lengths.max()),
                "angle_min": float(min(angles)), "angle_max": float(max(angles)),
                "n_close_contacts": int(close)}
    verdict, why = sp2_quality(geometry)
    # Only bond vectors are imaged, all far shorter than any cell edge, so
    # the full cell is safe for a slab too.
    hyb = hybridisation_report(atoms.positions, bonds,
                               np.array(atoms.cell) if atoms.pbc.any() else None)
    return {"atoms": len(atoms), "pbc": atoms.pbc.tolist(),
            "cell_A": [round(float(x), 2) for x in atoms.cell.lengths()],
            "coordination": dict(sorted(coordination.items())),
            "rings": dict(sorted(census.items())),
            "sum_6_minus_n": int(sum((6 - s) * c for s, c in census.items())),
            "touching_5_5": touching(5, 5), "touching_7_7": touching(7, 7),
            "touching_5_7": touching(5, 7),
            "bond_mean": float(lengths.mean()), "bond_std": float(lengths.std()),
            **geometry, "sp2_verdict": verdict, "sp2_reason": why,
            "angle_sum_min": hyb["angle_sum_min"], "sp3_fraction": hyb["sp3_fraction"],
            "collapsed_wall": collapsed_wall(hyb)}


def main(folder: Path) -> None:
    out = {}
    for label, rel in FILES.items():
        r = audit(read(str(folder / rel)))
        out[label] = r
        print(f"{label:34s} {r['atoms']:5d} rings {r['rings']} 5-5 {r['touching_5_5']} "
              f"7-7 {r['touching_7_7']} bonds {r['bond_min']:.3f}-{r['bond_max']:.3f} "
              f"angles {r['angle_min']:.1f}-{r['angle_max']:.1f} {r['sp2_verdict']}"
              f"{' COLLAPSED' if r['collapsed_wall'] else ''}", flush=True)
    (HERE / "auditoria.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
