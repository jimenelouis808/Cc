"""Can the crystalline (knee) route build what today is only meshed?

For each candidate: glue knee nodes along the net's edges (finite cage:
no cell; fcc: periodic) at every circumference and arm length tried,
and report the mesh census against the Gauss-Bonnet budget.

    python prueba_rutas.py > prueba_rutas.txt
"""
from __future__ import annotations

import itertools

import numpy as np

from nanocarbon_lab.builders.knee import (
    MESH_EDGE, collapse_degree_three, glue_nodes, mesh_census)
from nanocarbon_lab.builders.supernetwork import platonic_cage, named_graph

CC = 1.42


def axes_of(graph):
    pos = np.asarray(graph.nodes, float)
    out = []
    for i in range(len(pos)):
        nbr = [b if a == i else a for a, b, _ in graph.edges if i in (a, b)]
        d = pos[nbr] - pos[i]
        out.append(d / np.linalg.norm(d, axis=1)[:, None])
    return pos, out


def try_cage(name, graph, ks=range(8, 21, 2), rows=(5, 7, 9)):
    pos, axes = axes_of(graph)
    V, E = len(pos), len(graph.edges)
    budget = 12 * (V - E)
    edge_len = np.mean([np.linalg.norm(pos[a] - pos[b]) for a, b, _ in graph.edges])
    unbalance = max(float(np.linalg.norm(a.sum(0))) for a in axes)
    print(f"\n## {name}: V={V} E={E} budget sum(6-n)={budget}, node |sum of arms|={unbalance:.2f}")
    for k, r in itertools.product(ks, rows):
        spacing = MESH_EDGE * CC * np.sqrt(3) / 2
        radius = MESH_EDGE * CC / (2 * np.sin(np.pi / k))
        bond = 2 * (r - 1) * spacing
        try:
            v, t = glue_nodes(pos / edge_len * bond, axes, k, r, radius, spacing, cell=None)
            v, t = collapse_degree_three(v, t)
            census, _, broken = mesh_census(v, t)
            deficit = sum((6 - s) * c for s, c in census.items())
            clean = (not broken) and set(census) <= {5, 6, 7} and deficit == budget
            print(f"k={k:2d} rows={r}: {dict(sorted(census.items()))} sum={deficit:+d} "
                  f"{'CLEAN' if clean else 'no'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"k={k:2d} rows={r}: fails ({str(e)[:70]})", flush=True)


def try_fcc(ks=range(8, 25, 2), rows=(5, 7, 9, 11)):
    """super-fcc: one 12-arm node per lattice point, glued periodically."""
    from nanocarbon_lab.builders.knee import collapse_degree_four, fill_node_holes, hole_size, node_mesh
    axes = np.array([v for v in itertools.product((-1, 0, 1), repeat=3)
                     if sorted(map(abs, v)) == [0, 1, 1]], float)
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    print(f"\n## super-fcc node: 12 arms, |sum| = {np.linalg.norm(axes.sum(0)):.2f}, "
          "node budget 6(2-12) = -60")
    for k, r in itertools.product(ks, rows):
        spacing = MESH_EDGE * CC * np.sqrt(3) / 2
        radius = MESH_EDGE * CC / (2 * np.sin(np.pi / k))
        try:
            v, t = node_mesh(axes, k, r, radius, spacing)
            t = fill_node_holes(t, max_size=hole_size(axes))
            v, t = collapse_degree_three(v, t)
            v, t = collapse_degree_four(v, t)
            census, _, broken = mesh_census(v, t)
            print(f"k={k:2d} rows={r}: node census {dict(sorted(census.items()))}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"k={k:2d} rows={r}: fails ({str(e)[:70]})", flush=True)


if __name__ == "__main__":
    import sys
    if "fcc" in sys.argv:
        try_fcc()
        raise SystemExit
    for solid in ("tetrahedron", "cube", "octahedron"):
        try_cage(f"super-{solid}", platonic_cage(solid, 24.0))

