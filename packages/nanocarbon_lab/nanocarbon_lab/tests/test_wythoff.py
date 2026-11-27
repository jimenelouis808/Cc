"""Wythoff's kaleidoscope: uniform cages from three mirrors."""

import numpy as np
import pytest

from nanocarbon_lab.builders import geodesic as G
from nanocarbon_lab.builders import wythoff as W

#: (vertices, edges, degree) of every uniform solid built, from the
#: tables -- the construction writes none of them down.
KNOWN = {
    "tetrahedron": (4, 6, 3), "octahedron": (6, 12, 4), "cube": (8, 12, 3),
    "icosahedron": (12, 30, 5), "dodecahedron": (20, 30, 3),
    "cuboctahedron": (12, 24, 4), "icosidodecahedron": (30, 60, 4),
    "truncated-tetrahedron": (12, 18, 3), "truncated-octahedron": (24, 36, 3),
    "truncated-cube": (24, 36, 3), "truncated-icosahedron": (60, 90, 3),
    "truncated-dodecahedron": (60, 90, 3),
    "rhombicuboctahedron": (24, 48, 4), "rhombicosidodecahedron": (60, 120, 4),
    "truncated-cuboctahedron": (48, 72, 3),
    "truncated-icosidodecahedron": (120, 180, 3),
}


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_every_uniform_solid_comes_out_of_three_mirrors(name):
    graph = W.wythoff_cage(name)
    assert (len(graph.nodes), len(graph.edges), graph.coordination) == KNOWN[name]
    lengths = [np.linalg.norm(graph.nodes[j] - graph.nodes[i])
               for i, j, _ in graph.edges]
    assert np.allclose(lengths, 1.0)


def test_a_euclidean_triangle_is_refused():
    with pytest.raises(ValueError, match="finite"):
        W.mirrors(6, 3)


@pytest.mark.parametrize("name", ["truncated-icosahedron",
                                  "truncated-icosidodecahedron"])
def test_group_frames_fit_the_icosahedral_solids_exactly(name):
    _, miss = W.wythoff_frames(name, 2, starts=8)
    assert miss < 0.05


def test_a_shared_frame_makes_the_carrier_mirror_exact():
    strains = [G.strain_energy(G.build_geodesic_superstructure(
        "wythoff-cube", m=m, n=n, freq=2, rows=1, tolerance_deg=0.5,
        relax=False)) for m, n in ((2, 1), (1, 2))]
    assert strains[0] == pytest.approx(strains[1], rel=1e-9)


#: Nodes per cell, coordination and edges per cell of the Euclidean
#: Wythoff nets, against the uniform honeycomb and tiling tables. The
#: [6,3] tilings sit in the orthohexagonal cell, two primitive cells.
EUCLIDEAN = {
    ("[4,3,4]", (1, 0, 0, 0)): (1, 6),
    ("[4,3,4]", (0, 1, 0, 0)): (3, 8),
    ("[4,3,4]", (1, 1, 0, 0)): (6, 5),
    ("[4,3,4]", (0, 1, 1, 0)): (12, 4),
    ("[4,3,4]", (1, 0, 1, 0)): (12, 6),
    ("[4,3,4]", (1, 1, 1, 0)): (24, 4),
    ("[4,3,4]", (1, 1, 0, 1)): (24, 5),
    ("[4,3,4]", (1, 1, 1, 1)): (48, 4),
    ("[4,4]", (1, 0, 0)): (1, 4),
    ("[4,4]", (0, 1, 1)): (4, 3),
    ("[6,3]", (0, 0, 1)): (2, 6),
    ("[6,3]", (1, 0, 0)): (4, 3),
    ("[6,3]", (0, 1, 0)): (6, 4),
    ("[6,3]", (1, 0, 1)): (12, 4),
    ("[6,3]", (1, 1, 1)): (24, 3),
}


@pytest.mark.parametrize("key", sorted(EUCLIDEAN))
def test_the_uniform_honeycombs_and_tilings_come_out_of_mirrors(key):
    from nanocarbon_lab.builders.supernetwork import _periodic_graph

    group, rings = key
    nodes, shape, pbc = W.euclidean_wythoff(group, rings)
    graph = _periodic_graph("t", nodes, shape, pbc, "")
    vertices, degree = EUCLIDEAN[key]
    assert len(graph.nodes) == vertices
    assert graph.coordination == degree
    assert 2 * len(graph.edges) == vertices * degree


def test_the_named_nets_are_on_the_geodesic_lookup():
    for name in W.WYTHOFF_NETS:
        assert G._named(name) is W.WYTHOFF_NETS[name]
    sodalite = W.WYTHOFF_NETS["wythoff-bitruncated-cubic"]
    assert sodalite.ring_budget == 12 * (12 - 24)
