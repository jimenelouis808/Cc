"""Carbon toroids: the surface the disclination rule is really about.

A torus is genus 1, so ``sum(6-n) = 0`` exactly -- every pentagon paired
with a heptagon, and nothing else needed. That makes the census
*predictable* rather than merely measurable, which no other curved
builder here can say.
"""

from __future__ import annotations

import warnings

import pytest

from nanocarbon_lab.builders.toroid import (LITERATURE_ASPECT, MIN_ASPECT,
                                            build_toroid, describe_toroid)


@pytest.fixture(scope="module")
def toroid():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_toroid(major_radius=20.0, minor_radius=5.0, seed=0)


class TestTheBudgetIsZeroAndTheDisclinationsPair:
    def test_the_euler_budget_is_exactly_zero(self, toroid):
        counts = toroid.info["ring_counts"]
        assert sum((6 - size) * count for size, count in counts.items()) == 0

    def test_pentagons_and_heptagons_come_out_equal(self, toroid):
        """Not an approximation: with only 5s, 6s and 7s, a zero budget
        forces the two counts to match exactly."""
        counts = toroid.info["ring_counts"]
        assert set(counts) <= {5, 6, 7}
        assert counts[5] == counts[7]

    def test_it_has_both(self, toroid):
        """A torus that came out all-hexagon would have zero budget too,
        and would not be a torus -- it would be a mesh that failed to
        close round the hole."""
        counts = toroid.info["ring_counts"]
        assert counts[5] > 0 and counts[7] > 0


class TestTheDisclinationsSitWhereTheyBelong:
    def test_pentagons_outside_and_heptagons_inside(self, toroid):
        """The outer equator of a torus has positive Gaussian curvature
        and the inner one negative, so this is the case the coil papers
        describe with nothing else going on."""
        check = toroid.info["disclination_check"]
        assert check["sizes"][5]["mean_sign"] > 0.5
        assert check["sizes"][7]["mean_sign"] < -0.5
        assert check["agreement"] > 0.85


class TestTheGeometry:
    def test_the_wall_is_graphitic(self, toroid):
        geometry = toroid.info["geometry"]
        assert 1.25 < geometry["bond_min"]
        assert geometry["bond_max"] < 1.55
        assert geometry["n_close_contacts"] == 0

    def test_the_aspect_ratio_is_recorded_both_ways(self, toroid):
        assert toroid.info["aspect_ratio"] == pytest.approx(4.0)
        assert toroid.info["inner_bend_strain"] == pytest.approx(0.25)

    def test_the_description_states_the_budget(self, toroid):
        assert "sum(6-n) = +0" in describe_toroid(toroid)


class TestRefusals:
    def test_a_closed_hole_is_refused(self):
        """Below R/r = 2.5 the hole is smaller than the tube is thick,
        and the result is a dimpled sphere with a torus's name on it."""
        with pytest.raises(ValueError, match="dimpled sphere"):
            build_toroid(major_radius=10.0, minor_radius=5.0)

    def test_the_floor_is_where_the_constant_says(self):
        assert MIN_ASPECT == 2.5
        assert LITERATURE_ASPECT == (3.0, 6.0)

    @pytest.mark.parametrize("major, minor", [(0.0, 5.0), (20.0, 0.0),
                                              (-1.0, 5.0)])
    def test_a_non_positive_radius_is_refused(self, major, minor):
        with pytest.raises(ValueError, match="positive"):
            build_toroid(major_radius=major, minor_radius=minor)

    def test_an_unpublished_ratio_is_built_and_warned_about(self):
        """Not wrong -- simply not the geometry those calculations
        relaxed to, which is a different statement."""
        with pytest.warns(UserWarning, match="outside the"):
            build_toroid(major_radius=60.0, minor_radius=4.0, seed=0)
