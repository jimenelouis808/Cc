"""Carbon toroids: the surface the disclination rule is really about.

A torus is genus 1, so ``sum(6-n) = 0`` exactly -- every pentagon paired
with a heptagon, and nothing else needed. That makes the census
*predictable* rather than merely measurable, which no other curved
builder here can say.
"""

from __future__ import annotations

import warnings

import pytest

from nanocarbon_lab.builders.toroid import (
    LITERATURE_ASPECT,
    MIN_ASPECT,
    POLYHEX_MAX_STRAIN,
    build_polyhex_toroid,
    build_toroid,
    describe_toroid,
)


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


class TestThePolyhexRoute:
    """The crystalline ring: a finished lattice bent until its ends meet.

    A torus is genus 1, so an all-hexagon net satisfies the budget with
    no disclinations at all -- and unlike the meshed route, that is what
    the published pictures of toroidal carbon show. The price is size,
    because bending a polyhex can only stretch it.
    """

    @pytest.fixture(scope="module")
    def ring(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return build_polyhex_toroid(n=5, m=5, periods=110)

    def test_the_wall_is_all_hexagons(self, ring):
        """The whole point. The meshed route at a comparable size comes
        back with 68 pentagons and 68 heptagons; this has none."""
        assert set(ring.info["ring_counts"]) == {6}

    def test_the_seam_closed(self, ring):
        """`build_cnt` returns a whole number of translational periods, so
        a ring of that circumference meets itself exactly. A boundary walk
        would mean it did not."""
        assert ring.info["boundary_walks"] == 0

    def test_the_budget_is_still_zero(self, ring):
        counts = ring.info["ring_counts"]
        assert sum((6 - size) * count for size, count in counts.items()) == 0

    def test_the_geometry_is_graphitic(self, ring):
        geometry = ring.info["geometry"]
        assert geometry["bond_min"] > 1.30
        assert geometry["bond_max"] < 1.55
        assert geometry["n_close_contacts"] == 0

    def test_the_strain_is_the_radius_ratio(self, ring):
        """r/R, and nothing else: there is no disclination to relieve it
        with, which is exactly what the meshed route buys with its
        5-7 pairs."""
        info = ring.info
        # rel=1e-3 rather than machine precision: the two radii are
        # recorded to 3 decimals and the strain to 4, so the identity can
        # only be checked to the coarser of those roundings.
        assert info["outer_wall_strain"] == pytest.approx(
            info["minor_radius"] / info["major_radius"], rel=1e-3)
        assert info["outer_wall_strain"] <= POLYHEX_MAX_STRAIN

    def test_a_tight_ring_is_refused_rather_than_torn(self):
        """Measured at r/R = 29%: the census came back with 360 three- and
        120 four-membered rings, which is the tube folded through itself.
        That is not a strained lattice, so it is not returned as one."""
        with pytest.raises(ValueError, match="tears rather than loads"):
            build_polyhex_toroid(n=5, m=5, periods=40)

    def test_a_strained_but_sound_ring_is_built_and_warned_about(self):
        with pytest.warns(UserWarning, match="stretched"):
            build_polyhex_toroid(n=10, m=10, periods=170)

    def test_the_two_routes_disagree_about_disclinations_on_purpose(self,
                                                                    ring,
                                                                    toroid):
        """Same object, two constructions, and the difference is the whole
        reason both exist: the meshed one relieves curvature with 5-7
        pairs at any size, the polyhex one refuses to and must be wide."""
        assert set(ring.info["ring_counts"]) == {6}
        assert set(toroid.info["ring_counts"]) == {5, 6, 7}
        assert ring.info["major_radius"] > toroid.info["major_radius"]
