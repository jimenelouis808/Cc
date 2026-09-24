"""How big a supernetwork is, said before it is built.

The geometry check a net already had passes on one that is perfectly
sound and merely enormous. A super-fcc reached from the Net box while
super-diamond's ``scale=60, tube_radius=5`` was still in the fields has
42 Å struts with 24 Å of free tube -- healthy by every test there was --
and is eight times the wall of the same net at its own preset.

These tests pin the size measure, and, just as importantly, pin the
three things that were tried as a **time** measure and do not work.
"""
import numpy as np
import pytest

from nanocarbon_lab.builders.supernetwork import (
    LARGE_AREA,
    SUPERLATTICES,
    build_cost_note,
    named_graph,
)

#: ``(net, scale, tube_radius, area, atoms, seconds)`` -- every row
#: measured on this builder, single-threaded, seed 0.
PERIODIC = [
    ("super-cubic", 34.0, 5.0, 3_204, 836, 37),
    ("super-graphene", 34.0, 5.0, 3_700, 1_180, 36),
    ("super-fcc", 40.0, 3.0, 12_796, 2_982, 413),
    ("super-diamond", 60.0, 5.0, 13_059, 3_752, 400),
    ("supertube-(6,6)", 14.0, 3.0, 18_964, 5_922, 434),
    ("super-fcc", 50.0, 4.0, 21_340, 4_694, 483),
    ("super-fcc", 60.0, 5.0, 31_989, 6_862, 566),
]
CAGES_MEASURED = [
    ("super-icosahedron", 24.0, 4.0, 18_096, 4_292, 35),
    ("super-hypercube", 20.0, 3.5, 21_802, 6_494, 88),
    ("superfullerene-C60", 14.2, 3.0, 24_090, 7_534, 79),
]
MEASURED = PERIODIC + CAGES_MEASURED


@pytest.mark.parametrize("name, scale, radius, area, _atoms, _secs", MEASURED)
def test_wall_area_reproduces_the_measured_table(name, scale, radius, area,
                                                 _atoms, _secs):
    graph = named_graph(name, scale)
    assert graph.wall_area(scale, radius) == pytest.approx(area, rel=1e-3)


class TestTheMeasuresThatDoNotPredictTime:
    """Three were tried. Each is pinned here so none is tried again.

    This matters more than it looks: a hint that tells someone a
    nine-minute build will take tens of minutes is worse than a hint
    with no number at all, because it talks them out of pressing a
    button that works.
    """

    def test_area_ranks_the_cages_backwards(self):
        """More wall, a sixteenth of the time."""
        cage = next(r for r in CAGES_MEASURED if r[0] == "super-hypercube")
        cell = next(r for r in PERIODIC if r[0] == "super-diamond")
        assert cage[3] > cell[3]            # 21_802 Å² against 13_059
        assert cage[5] < cell[5] / 4.0      # 88 s against ~400

    def test_area_flattens_out_among_the_periodic_cells(self):
        """2.5x the wall for 1.4x the time, because the grid is fixed.

        ``grid_resolution`` is 72 whatever the cell is, so a larger cell
        is sampled more coarsely and does not carry proportionally more
        mesh. Any rule linear in area gets this badly wrong.
        """
        small = next(r for r in PERIODIC
                     if r[0] == "super-fcc" and r[1] == 40.0)
        large = next(r for r in PERIODIC
                     if r[0] == "super-fcc" and r[1] == 60.0)
        assert large[3] / small[3] > 2.4
        assert large[5] / small[5] < 1.5

    def test_cost_per_unit_area_is_not_monotone(self):
        """It rises and then falls, so it cannot order anything."""
        per = [secs / (area / 1000.0)
               for _n, _s, _r, area, _a, secs in PERIODIC]
        assert per != sorted(per)
        assert per != sorted(per, reverse=True)

    def test_atom_count_does_not_separate_a_cage_from_a_cell(self):
        """7534 atoms in 79 s against 2982 in 413."""
        cage = next(r for r in CAGES_MEASURED
                    if r[0] == "superfullerene-C60")
        cell = next(r for r in PERIODIC
                    if r[0] == "super-fcc" and r[1] == 40.0)
        assert cage[4] > 2 * cell[4]
        assert cage[5] < cell[5] / 4.0

    def test_the_note_claims_no_minutes(self):
        """The wording is checked, because this is the whole point."""
        for area in (1_000.0, LARGE_AREA, 100_000.0):
            for periodic in (True, False):
                note = build_cost_note(area, periodic)
                assert "minute" not in note and "hour" not in note


def test_the_note_sorts_large_nets_from_modest_ones():
    at_preset = named_graph("super-fcc", 40.0).wall_area(40.0, 3.0)
    inherited = named_graph("super-fcc", 60.0).wall_area(60.0, 5.0)
    assert build_cost_note(at_preset) == "a modest one"
    assert build_cost_note(inherited) == \
        "one of the larger nets in the catalogue"


def test_area_is_linear_in_radius_and_in_scale():
    """It is ``2*pi*r`` times a length, so both must scale cleanly."""
    graph = named_graph("super-cubic", 34.0)
    single = graph.wall_area(34.0, 5.0)
    assert graph.wall_area(34.0, 10.0) == pytest.approx(2.0 * single)
    bigger = named_graph("super-cubic", 68.0)
    assert bigger.wall_area(68.0, 5.0) == pytest.approx(2.0 * single)


def test_a_denser_net_costs_more_at_the_same_cell():
    """Twelve tubes per node is the expensive end, and says so."""
    at = {name: named_graph(name, 40.0).wall_area(40.0, 5.0)
          for name in SUPERLATTICES}
    assert at["super-fcc"] == max(at.values())
    assert at["super-square"] == min(at.values())
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
