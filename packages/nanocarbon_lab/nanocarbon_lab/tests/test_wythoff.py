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
