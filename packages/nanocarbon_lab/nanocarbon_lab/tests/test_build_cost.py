"""What a supernetwork costs to build, said before it is built.

The geometry check a net already had passes on one that is perfectly
sound and merely enormous. A super-fcc reached from the Net box while
super-diamond's ``scale=60, tube_radius=5`` was still in the fields has
42 Å struts with 24 Å of free tube -- healthy by every test there was --
and had not finished building after ninety minutes. These tests pin the
measure that notices, and the numbers it is calibrated on.
"""
import numpy as np
import pytest

from nanocarbon_lab.builders.supernetwork import (
    BRISK_AREA,
    SLOW_AREA,
    SUPERLATTICES,
    build_cost_note,
    named_graph,
)

#: ``(net, scale, tube_radius, area, seconds)`` -- every row measured on
#: this builder, and the reason the thresholds are where they are.
#: The periodic cells and the cages are kept apart because they cost
#: differently by an order of magnitude, which is the whole finding.
PERIODIC = [
    ("super-graphene", 34.0, 5.0, 3_700, 36),
    ("super-cubic", 34.0, 5.0, 3_204, 37),
    ("super-fcc", 40.0, 3.0, 12_796, 413),
    ("super-diamond", 60.0, 5.0, 13_059, 400),
    ("supertube-(6,6)", 14.0, 3.0, 18_964, 434),
    ("super-fcc", 60.0, 5.0, 31_989, 5_400),
]
CAGES_MEASURED = [
    ("super-icosahedron", 24.0, 4.0, 18_096, 35),
    ("super-hypercube", 20.0, 3.5, 21_802, 88),
    ("superfullerene-C60", 14.2, 3.0, 24_090, 79),
]
MEASURED = PERIODIC + CAGES_MEASURED


@pytest.mark.parametrize("name, scale, radius, area, _seconds", MEASURED)
def test_wall_area_reproduces_the_measured_table(name, scale, radius, area,
                                                 _seconds):
    graph = named_graph(name, scale)
    assert graph.wall_area(scale, radius) == pytest.approx(area, rel=1e-3)


def test_cost_bands_sort_the_periodic_builds_by_what_they_took():
    """A minute, several minutes and tens of them, in that order."""
    notes = [build_cost_note(area) for _n, _s, _r, area, _t in PERIODIC]
    assert notes[0] == notes[1] == "about a minute to build"
    assert notes[2] == notes[3] == notes[4] == "several minutes to build"
    assert notes[5].startswith("tens of minutes")


def test_a_cage_is_not_priced_like_a_periodic_cell():
    """The correction that area alone got wrong.

    Every cage measured ran in 35-88 s, and the largest of them carries
    more wall than the super-diamond cell that takes seven minutes. Area
    ranked them the wrong way round; what actually costs is welding the
    cell's faces, which a cage does not do.
    """
    for _name, _scale, _radius, area, seconds in CAGES_MEASURED:
        assert seconds < 120
        assert build_cost_note(area, periodic=False) == \
            "about a minute to build"
        # The same area, in a cell that has to be welded, is not.
        assert build_cost_note(area, periodic=True) != \
            "about a minute to build"


def test_the_hypercube_cage_outweighs_the_diamond_cell_and_still_beats_it():
    """The single row that refuted the first calibration."""
    cage = named_graph("super-hypercube", 20.0).wall_area(20.0, 3.5)
    cell = named_graph("super-diamond", 60.0).wall_area(60.0, 5.0)
    assert cage > cell                       # more wall ...
    assert build_cost_note(cage, periodic=False) == "about a minute to build"
    assert build_cost_note(cell, periodic=True) == "several minutes to build"


def test_the_ninety_minute_case_is_the_one_that_reads_slow():
    """The exact settings the user reached, and the ones that work."""
    graph = named_graph("super-fcc", 60.0)
    assert graph.wall_area(60.0, 5.0) > SLOW_AREA
    # The preset the window now offers instead builds in seven minutes.
    assert BRISK_AREA < named_graph("super-fcc", 40.0).wall_area(40.0, 3.0) \
        < SLOW_AREA


def test_area_is_linear_in_radius_and_in_scale():
    """It is ``2*pi*r`` times a length, so both must scale cleanly."""
    graph = named_graph("super-cubic", 34.0)
    single = graph.wall_area(34.0, 5.0)
    assert graph.wall_area(34.0, 10.0) == pytest.approx(2.0 * single)
    # Doubling the cell doubles every strut.
    bigger = named_graph("super-cubic", 68.0)
    assert bigger.wall_area(68.0, 5.0) == pytest.approx(2.0 * single)


def test_a_denser_net_costs_more_at_the_same_cell():
    """Twelve tubes per node is the expensive end, and says so."""
    at = {name: named_graph(name, 40.0).wall_area(40.0, 5.0)
          for name in SUPERLATTICES}
    assert at["super-fcc"] == max(at.values())
    assert at["super-square"] == min(at.values())
    # The spread is the point: an eightfold difference between two
    # entries of one combobox, at identical settings.
    assert at["super-fcc"] / at["super-square"] > 5.0


def test_a_radius_that_is_not_a_radius_is_refused():
    graph = named_graph("super-cubic", 34.0)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="tube_radius"):
            graph.wall_area(34.0, bad)


def test_every_net_in_the_catalogue_can_price_itself():
    """No net may be missing the measure the window asks it for."""
    for name in SUPERLATTICES:
        area = named_graph(name, 40.0).wall_area(40.0, 4.0)
        assert np.isfinite(area) and area > 0.0
        assert build_cost_note(area)
