"""Rolling a haeckelite sheet into a tube.

The tests are organised around the one claim the module rests on: a
cylinder is developable, so the roll is an isometry and the census must
survive it untouched. Everything else -- the radius, the axial period --
is an output, and what is checked about those is that they are measured
rather than asserted.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from nanocarbon_lab.builders.haeckelite_tube import (
    AXIAL_LIMIT,
    AXIAL_SEARCH,
    AXIAL_STEP,
    MIN_TUBE_RADIUS,
    build_haeckelite_tube,
    describe_haeckelite_tube,
    roll_to_tube,
)


def _quiet(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_haeckelite_tube(**kwargs)


class TestRollToTube:
    """The map itself, with no relaxation in the way."""

    def test_it_puts_every_atom_on_one_cylinder(self):
        flat = np.array([[0.0, 0.0, 0.0], [2.5, 1.0, 0.0],
                         [5.0, 3.0, 0.0], [7.5, 2.0, 0.0]])
        rolled, radius = roll_to_tube(flat, circumference=10.0,
                                      wrap_axis=0, axis=1)
        assert radius == pytest.approx(10.0 / (2 * math.pi))
        distances = np.hypot(rolled[:, 0], rolled[:, 1])
        assert distances == pytest.approx(np.full(4, radius))

    def test_the_axis_coordinate_is_carried_untouched(self):
        flat = np.array([[0.0, 1.5, 0.0], [3.0, 7.25, 0.0]])
        rolled, _ = roll_to_tube(flat, circumference=12.0,
                                 wrap_axis=0, axis=1)
        assert rolled[:, 2] == pytest.approx(flat[:, 1])

    def test_arc_length_becomes_angle(self):
        """A quarter of the circumference is a quarter turn, exactly.

        This is what makes the roll an isometry in the limit of short
        bonds, and it is the property the census survives on.
        """
        flat = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0],
                         [5.0, 0.0, 0.0]])
        rolled, radius = roll_to_tube(flat, circumference=10.0,
                                      wrap_axis=0, axis=1)
        angles = np.arctan2(rolled[:, 1], rolled[:, 0])
        assert angles[1] - angles[0] == pytest.approx(math.pi / 2)
        assert angles[2] - angles[1] == pytest.approx(math.pi / 2)


@pytest.fixture(scope="module")
def control():
    """A plain (8,0): the one case here with a published answer."""
    return _quiet(nx=8, ny=2, pattern="none", roll="a")


@pytest.fixture(scope="module")
def patterned():
    return _quiet(nx=12, ny=4, pattern="r57", roll="a")


class TestTheAllHexagonControl:
    """``pattern='none'`` is a plain nanotube, and that is the point.

    It goes down the same code path as every patterned lattice, so
    anything it gets wrong is the rolling rather than the pattern -- and
    unlike a haeckelite it has a published answer to be checked against.
    """

    def test_it_reproduces_a_real_zigzag_nanotube(self, control):
        """An 8-cell-wide roll of graphene is an (8,0), radius 3.13 Å.

        ``R = n * a / 2pi`` with ``a = sqrt(3) * 1.42``. This is the only
        number in this module with an answer outside the framework, so it
        is the one check that is not the builder marking its own work.
        """
        expected = 8 * math.sqrt(3.0) * 1.42 / (2 * math.pi)
        assert control.info["rolled_radius"] == pytest.approx(expected, abs=0.02)

    def test_it_is_all_hexagons(self, control):
        assert set(control.info["ring_counts"]) == {6}

    def test_it_relaxes_outwards(self, control):
        """The chord is shorter than the arc, so the waist starts under
        compression and an unpatterned wall answers by expanding."""
        assert control.info["radius"] > control.info["rolled_radius"]

    def test_the_geometry_is_graphitic(self, control):
        geometry = control.info["geometry"]
        assert 1.39 < geometry["bond_min"] <= geometry["bond_max"] < 1.45
        assert geometry["n_close_contacts"] == 0

    def test_it_is_periodic_along_its_axis_only(self, control):
        assert list(control.pbc) == [False, False, True]


class TestThePatternedLattice:
    def test_the_census_survives_the_roll(self, patterned):
        """Rolling is an isometry, so the tube's rings are the sheet's.

        The builder traces the finished tube's own faces and compares;
        reaching this assertion at all means that check passed. What is
        pinned here is that r57 stays the zero-hexagon lattice it is.
        """
        counts = patterned.info["ring_counts"]
        assert 6 not in counts
        assert counts[5] == counts[7]
        assert patterned.info["non_hexagonal_fraction"] == 1.0

    def test_the_euler_budget_is_zero(self, patterned):
        """A tube periodic along its axis is a torus, exactly as the flat
        periodic sheet was."""
        counts = patterned.info["ring_counts"]
        assert sum((6 - size) * count for size, count in counts.items()) == 0

    def test_rolling_moves_no_atoms(self, patterned):
        """A Stone-Wales rotation moves bonds and the roll moves nobody,
        so the count is graphene's whatever the pattern."""
        assert len(patterned) == 4 * 12 * 4

    def test_the_geometry_is_sp2(self, patterned):
        geometry = patterned.info["geometry"]
        assert geometry["bond_min"] > 1.22
        assert geometry["bond_max"] < 1.60
        assert geometry["angle_min"] > 95.0
        assert geometry["angle_max"] < 145.0
        assert geometry["n_close_contacts"] == 0

    def test_it_is_judged_as_a_haeckelite(self, patterned):
        """A heptagon's interior angle is 128.6 deg before any strain, so
        the hexagonal sp2 window calls every sound one BROKEN."""
        assert patterned.info["quality_family"] == "haeckelite"

    def test_the_description_names_what_was_built(self, patterned):
        text = describe_haeckelite_tube(patterned)
        assert "r57" in text and "rolled along a" in text


class TestTheAxialFit:
    """The axis is the one degree of freedom the cell still holds."""

    def test_the_minimum_is_not_on_a_bound(self, patterned):
        """A search pinned at its own edge is reporting the edge.

        The first version of this scan was a fixed +-4% window, and every
        patterned lattice came back at exactly 0.96 -- its lower bound.
        The window now widens toward whichever end wins, so a returned
        factor must be strictly inside it.
        """
        factor = patterned.info["axial_factor"]
        assert 1.0 - AXIAL_LIMIT < factor < 1.0 + AXIAL_LIMIT
        # Not sitting on a multiple of the window either, which is where
        # the bounds of every widening step fall.
        steps = round((factor - 1.0) / AXIAL_STEP)
        window = round(AXIAL_SEARCH / AXIAL_STEP)
        assert steps % window != 0 or steps == 0

    def test_turning_the_fit_off_keeps_the_flat_period(self):
        tube = _quiet(nx=12, ny=4, pattern="r57", roll="a", fit_axial=False)
        assert tube.info["axial_factor"] == pytest.approx(1.0)
        assert tube.info["axial_period"] == pytest.approx(
            tube.info["flat_axial_period"])


class TestTheTwoRolls:
    """``roll='a'`` and ``roll='b'`` are different tubes, not one tube
    described twice: the lattice is anisotropic, so neither is a rotation
    of the other."""

    def test_they_differ(self):
        along_a = _quiet(nx=12, ny=4, pattern="r57", roll="a")
        along_b = _quiet(nx=4, ny=12, pattern="r57", roll="b")
        assert len(along_a) == len(along_b)
        assert along_a.info["radius"] != pytest.approx(
            along_b.info["radius"], abs=0.1)
        assert along_a.info["axial_period"] != pytest.approx(
            along_b.info["axial_period"], abs=0.1)


class TestRefusals:
    def test_an_unknown_roll_is_rejected(self):
        with pytest.raises(ValueError, match="roll must be"):
            build_haeckelite_tube(roll="c")

    def test_a_tube_narrower_than_any_ever_observed_is_refused(self):
        """The floor is taken from what has been made, not from what this
        relaxer reports: a valence force field barely sees curvature and
        would happily return a 1 Å tube."""
        with pytest.raises(ValueError, match="narrowest carbon nanotube"):
            build_haeckelite_tube(nx=4, ny=2, pattern="none", roll="a")

    def test_a_narrow_tube_is_built_and_warned_about(self):
        with pytest.warns(UserWarning, match="rehybridisation"):
            build_haeckelite_tube(nx=7, ny=2, pattern="none", roll="a")


class TestTheFloorIsWhereItSaysItIs:
    def test_the_refusal_boundary_matches_the_constant(self):
        """``nx`` cells of graphene wrap to ``nx * a / 2pi``; the refusal
        must fall exactly where that crosses :data:`MIN_TUBE_RADIUS`."""
        a = math.sqrt(3.0) * 1.42
        smallest_ok = math.ceil(2 * math.pi * MIN_TUBE_RADIUS / a)
        assert smallest_ok == 6
        with pytest.raises(ValueError):
            build_haeckelite_tube(nx=smallest_ok - 1, ny=2, pattern="none")
