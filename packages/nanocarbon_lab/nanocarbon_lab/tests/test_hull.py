"""Hull nodes on any graph: the coarse map and its census invariants."""

import numpy as np
import pytest

from nanocarbon_lab.builders import hull as H
from nanocarbon_lab.builders.supernetwork import named_graph


def _graph(name):
    graph = named_graph(name, 1.0)
    return np.asarray(graph.nodes, dtype=float), graph.edges


@pytest.mark.parametrize("name", ["super-cube", "super-tetrahedron",
                                  "super-octahedron", "super-icosahedron"])
def test_the_hull_map_meets_the_graph_budget(name):
    points, edges = _graph(name)
    (vertices, tris), budget, _ = H.hull_map(points, edges, sides=6)
    degree = np.bincount(tris.ravel(), minlength=len(vertices))
    assert int((6 - degree).sum()) == budget == 12 * (len(points) - len(edges))


def test_optimisation_never_moves_the_budget_and_never_worsens():
    points, edges = _graph("super-icosahedron")
    mesh, budget, _ = H.hull_map(points, edges, sides=6)
    before = ((np.bincount(mesh[1].ravel()) - 6) ** 2).sum()
    after_mesh, history = H.optimise_census(mesh)
    degree = np.bincount(after_mesh[1].ravel(), minlength=len(mesh[0]))
    assert int((6 - degree).sum()) == budget
    assert ((degree - 6) ** 2).sum() <= before
    assert history == sorted(history, reverse=True)


def test_curvature_charges_are_gauss_bonnet():
    points, edges = _graph("super-cube")
    mesh, budget, _ = H.hull_map(points, edges, sides=5)
    assert H.curvature_charges(mesh).sum() == pytest.approx(budget, abs=1e-6)
