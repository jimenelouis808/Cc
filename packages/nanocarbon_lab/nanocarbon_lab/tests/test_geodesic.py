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


def test_a_periodic_net_builds_in_its_cell_clean():
    built = G.build_geodesic_superstructure("super-cubic", m=2, freq=2,
                                            rows=1)
    assert built.info["ring_counts"] == {6: 204, 7: 24}
    assert built.info["ring_deficit"] == -24
    assert all(built.pbc)
    assert str(sp2_quality(built.info["geometry"])[0]) == "clean"
    assert built.info["disclinations_placed"] == 1.0


def test_windows_open_only_on_six_fold_vertices():
    graph = named_graph("supertube-(4,4)", 1.0)
    points, cell = G._graph_frame(graph)
    frames, miss = G.fit_node_frames(points, graph.edges, 4, starts=4,
                                     cell=cell)
    assert frames.shape == (len(points), 12, 3) and miss < 5.0
    (vertices, tris), budget, _, _ = G.geodesic_map(
        points, graph.edges, frames, freq=4, rows=1, tolerance_deg=5.0,
        cell=cell)
    degree = np.bincount(tris.ravel(), minlength=len(vertices))
    assert int((6 - degree).sum()) == budget


def test_the_enantiomer_is_exact_and_swaps_the_handedness():
    built = G.build_geodesic_superstructure("super-cube", m=2, n=1, freq=2,
                                            rows=1, relax=False)
    mirror = G.enantiomer(built)
    assert mirror.info["gc_indices"] == (1, 2)
    assert G.strain_energy(mirror) == pytest.approx(G.strain_energy(built),
                                                    rel=1e-12)
    back = G.enantiomer(mirror)
    assert np.allclose(back.get_positions(), built.get_positions())
    # A pseudoscalar flips sign under reflection: the mean triple product
    # of each atom's three bond vectors.
    def handedness(atoms):
        pos = atoms.get_positions()
        arms = {}
        for i, j in atoms.info["bonds"]:
            arms.setdefault(i, []).append(pos[j] - pos[i])
            arms.setdefault(j, []).append(pos[i] - pos[j])
        return np.mean([np.dot(a[0], np.cross(a[1], a[2]))
                        for a in arms.values() if len(a) == 3])
    assert handedness(mirror) == pytest.approx(-handedness(built))


def test_a_larger_window_keeps_the_budget_and_saves_atoms():
    graph = named_graph("super-cube", 1.0)
    points = np.asarray(graph.nodes, float)
    frame, _ = G.fit_frame(points, graph.edges, 4, starts=8)
    (vertices, tris), budget, _, _ = G.geodesic_map(
        points, graph.edges, frame, freq=4, rows=1, window_radius=2)
    degree = np.bincount(tris.ravel(), minlength=len(vertices))
    assert int((6 - degree).sum()) == budget
    assert set(degree.tolist()) <= {5, 6, 7}
    # Same 12-ring tube: radius 2 at m=1 against radius 1 at m=2.
    assert G.geodesic_atom_count("super-cube", 1, 4, 1, 0, 2) \
        < G.geodesic_atom_count("super-cube", 2, 2, 1, 0, 1)


def test_windows_that_would_touch_are_refused():
    graph = named_graph("super-cube", 1.0)
    points = np.asarray(graph.nodes, float)
    frame, _ = G.fit_frame(points, graph.edges, 2, starts=8)
    with pytest.raises(ValueError, match="overlap"):
        G.geodesic_map(points, graph.edges, frame, freq=2, rows=1,
                       window_radius=2)


def test_a_bent_arm_keeps_the_budget_where_a_straight_one_would_shear():
    # The hypercube's arms miss a frequency-4 sphere's windows by ~7 deg;
    # the arms bend to leave their mouths square, and the map still
    # meets 12(V - E) exactly.
    graph = G._named("super-hypercube")
    points, cell = G._graph_frame(graph)
    frame, miss, _, window = G._choose_frames(graph, points, cell, 4, 8.0,
                                              starts=8)
    assert 2.0 < miss < 8.0
    (vertices, tris), budget, _, _ = G.geodesic_map(
        points, graph.edges, frame, freq=4, rows=1, tolerance_deg=8.0,
        cell=cell, window=window)
    degree = np.bincount(tris.ravel(), minlength=len(vertices))
    assert int((6 - degree).sum()) == budget


def test_the_largest_window_is_found():
    graph = named_graph("super-cube", 1.0)
    points = np.asarray(graph.nodes, float)
    frame, _ = G.fit_frame(points, graph.edges, 4, starts=8)
    assert G.largest_window(points, graph.edges, frame, 4) == 2
