"""Octahedral frames: the minimum census for any <100>/<111> graph."""

import numpy as np
import pytest

from nanocarbon_lab.builders import frame as fr
from nanocarbon_lab.builders.goldberg import degree_census


def _coarse(kind, rows=1):
    points, edges, box, _ = fr.FRAMES[kind][0](rows)
    (vertices, tris), budget = fr.frame_map(points, edges, box)
    return vertices, tris, budget, len(points), len(edges)


@pytest.mark.parametrize("kind, census", [
    ("super-cubic", {6: 20, 7: 24}),
    ("super-square", {5: 4, 6: 20, 7: 16}),
    ("super-diamond", {6: 224, 7: 96}),
    ("cube cage", {5: 48, 6: 160, 7: 96}),
])
def test_coarse_census_is_the_minimum(kind, census):
    _, tris, budget, _, _ = _coarse(kind, rows=2)
    found = degree_census(tris)
    assert found == census
    assert sum((6 - s) * c for s, c in found.items()) == budget


@pytest.mark.parametrize("kind", sorted(fr.FRAMES))
def test_every_frame_meets_twelve_v_minus_e(kind):
    _, tris, budget, v, e = _coarse(kind)
    census = degree_census(tris)
    assert budget == 12 * (v - e)
    assert sum((6 - s) * c for s, c in census.items()) == budget
    # An octagon appears where a square arm and a hexagonal one leave by
    # neighbouring faces (the tesseract's corners): -2 at a saddle,
    # isolated like the rest.
    assert set(census) <= {5, 6, 7, 8}


def test_a_cage_corner_is_six_pentagons_and_twelve_heptagons():
    _, tris, _, v, _ = _coarse("cube cage")
    census = degree_census(tris)
    assert census[5] == 6 * v and census[7] == 12 * v


def test_an_arm_off_the_lattice_directions_is_refused():
    points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    with pytest.raises(ValueError, match="<100>"):
        fr.frame_map(points, [(0, 1, (0, 0, 0))])


def test_the_atom_count_is_exact():
    built = fr.build_frame("super-square", m=2, rows=1)
    assert len(built) == fr.frame_atom_count("super-square", 2, 1)


def test_super_square_builds_with_its_census_and_every_defect_placed():
    built = fr.build_frame("super-square", m=2, rows=1)
    assert built.info["ring_counts"][5] == 4
    assert built.info["ring_counts"][7] == 16
    assert built.info["disclinations_placed"] == 1.0
    assert tuple(built.pbc) == (True, True, False)
