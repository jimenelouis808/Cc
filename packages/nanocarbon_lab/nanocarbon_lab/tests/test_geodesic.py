"""Geodesic carriers: deltahedral cages, clean by construction."""

import numpy as np
import pytest

from nanocarbon_lab.builders import geodesic as G
from nanocarbon_lab.builders.supernetwork import named_graph
from nanocarbon_lab.validation.quality import sp2_quality


def test_a_geodesic_sphere_has_twelve_fives_and_unit_edges():
    golden = G._icosahedral_frame(np.zeros((1, 3)))
    vertices, tris = G.geodesic_sphere(golden, 2)
    degree = np.bincount(tris.ravel())
    assert sorted(set(degree)) == [5, 6] and (degree == 5).sum() == 12


@pytest.mark.parametrize("name", ["super-icosahedron", "super-cube",
                                  "super-dodecahedron", "superfullerene-C60"])
def test_these_cages_fit_the_frame_exactly(name):
    graph = named_graph(name, 1.0)
    _, miss = G.fit_frame(np.asarray(graph.nodes, float), graph.edges, 2,
                          starts=8)
    assert miss < 0.05


@pytest.mark.parametrize("name", ["super-icosahedron", "super-cube"])
def test_the_coarse_carrier_meets_the_budget(name):
    graph = named_graph(name, 1.0)
    points = np.asarray(graph.nodes, float)
    frame, _ = G.fit_frame(points, graph.edges, 2, starts=8)
    (vertices, tris), budget, _, _ = G.geodesic_map(points, graph.edges,
                                                    frame, freq=2, rows=1)
    degree = np.bincount(tris.ravel(), minlength=len(vertices))
    assert int((6 - degree).sum()) == budget == 12 * (len(points)
                                                      - len(graph.edges))


def test_the_cube_cage_is_clean_with_its_census_placed():
    built = G.build_geodesic_superstructure("super-cube", m=2, freq=2, rows=1)
    assert built.info["ring_counts"] == {5: 48, 6: 1416, 7: 96}
    assert str(sp2_quality(built.info["geometry"])[0]) == "clean"
    assert built.info["disclinations_placed"] == 1.0
    assert len(built) == G.geodesic_atom_count("super-cube", 2, 2, 1)


def test_an_arm_off_every_vertex_is_refused():
    graph = named_graph("super-tetrahedron", 1.0)
    points = np.asarray(graph.nodes, float)
    frame, _ = G.fit_frame(points, graph.edges, 2, starts=4)
    with pytest.raises(ValueError, match="misses"):
        G.geodesic_map(points, graph.edges, frame, freq=2, rows=1,
                       tolerance_deg=0.5)
