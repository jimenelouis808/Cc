"""Carbon nanocones: a wedge cut out of graphene, and the angle it forces.

A cone is developable, so unrolling it is an isometry -- which makes this
one of the few builders whose bond lengths are exact by construction
rather than relaxed into place, and whose apex angle has an answer from
outside the framework to check against.
"""

from __future__ import annotations

import warnings

import pytest

from nanocarbon_lab.builders.nanocone import (
    BUILDABLE,
    CLEAN,
    apex_angle,
    build_nanocone,
    describe_nanocone,
)


@pytest.fixture(scope="module")
def cone():
    return build_nanocone(n_pentagons=1, radius=22.0)


class TestTheAngleIsQuantised:
    """``sin(theta/2) = 1 - N/6``: the apex angle is not a parameter, it
    follows from the disclination. These five are the angles Krishnan et
    al. observed (*Nature* 388, 451)."""

    @pytest.mark.parametrize("pentagons, degrees", [
        (1, 112.9), (2, 83.6), (3, 60.0), (4, 38.9), (5, 19.2),
    ])
    def test_the_five_observed_cones(self, pentagons, degrees):
        assert apex_angle(pentagons) == pytest.approx(degrees, abs=0.05)

    def test_no_disclination_is_a_flat_sheet(self):
        assert apex_angle(0) == pytest.approx(180.0)

    def test_six_would_close_the_cone_to_a_line(self):
        with pytest.raises(ValueError, match="0 to 5"):
            apex_angle(6)


class TestTheCleanCone:
    """One pentagon at the apex and nothing else out of place."""

    def test_the_census_is_one_pentagon_and_hexagons(self, cone):
        census = cone.info["ring_counts"]
        assert census[5] == 1
        assert set(census) == {5, 6}
        assert census[6] > 100

    def test_the_budget_equals_the_disclination(self, cone):
        """A cone is developable, so rolling cannot create or destroy a
        ring: sum(6-n) must come out at exactly N."""
        assert cone.info["euler"] == cone.info["n_pentagons"] == 1

    def test_the_bonds_are_near_ideal_because_the_roll_is_an_isometry(
            self, cone):
        """Not relaxed -- placed. The only departure from 1.42 Å is at
        the apex ring, where the pentagon closes."""
        geometry = cone.info["geometry"]
        assert geometry["bond_max"] == pytest.approx(1.42, abs=0.005)
        assert geometry["bond_min"] > 1.35
        assert geometry["n_close_contacts"] == 0

    def test_the_angles_are_sp2(self, cone):
        geometry = cone.info["geometry"]
        assert geometry["angle_min"] > 100.0
        assert geometry["angle_max"] < 125.0

    def test_the_apex_ring_is_recorded(self, cone):
        assert cone.info["apex_ring"] == 5
        assert cone.info["apex_angle"] == pytest.approx(112.89, abs=0.05)

    def test_the_rim_is_declared_open(self, cone):
        """A nanocone is not a closed molecule. Its base atoms are
        two-coordinate, and saying so is what stops validation reading a
        deliberate edge as a dangling bond."""
        rim = cone.info["rim_atoms"]
        assert rim
        assert cone.info["terminal_atoms"] == rim

    def test_the_description_names_the_angle(self, cone):
        assert "112.9" in describe_nanocone(cone)


class TestTheSeamIsChosenNotAssumed:
    """Both 0 and 30 degrees are mirror lines of the hexagon, so both
    look valid -- and which one is right depends on N. The builder tries
    them and keeps the one whose budget comes out at N, which is an exact
    test because a developable roll cannot change a ring."""

    @pytest.mark.parametrize("pentagons", BUILDABLE)
    def test_every_buildable_cone_lands_on_its_budget(self, pentagons):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build_nanocone(n_pentagons=pentagons, radius=18.0,
                                   strict=False)
        assert atoms.info["euler"] == pentagons
        assert atoms.info["ring_counts"][6 - pentagons] == 1

    def test_the_chosen_offset_is_recorded(self, cone):
        assert cone.info["cut_offset_deg"] in (0.0, 15.0, 30.0, 45.0)


class TestRefusals:
    @pytest.mark.parametrize("pentagons", (4, 5))
    def test_a_disclination_too_large_for_one_ring(self, pentagons):
        """Four would need a 2-gon and five a 1-gon. Nature splits these
        into separate pentagons -- that is how the 19.2 deg nanohorn
        works -- and that is a cap design, not a sector cut."""
        with pytest.raises(ValueError, match="not a ring"):
            build_nanocone(n_pentagons=pentagons)

    @pytest.mark.parametrize("pentagons", (2, 3))
    def test_the_strained_apexes_are_refused_under_strict(self, pentagons):
        with pytest.raises(ValueError, match="poor chemistry"):
            build_nanocone(n_pentagons=pentagons)

    @pytest.mark.parametrize("pentagons, worst", [(2, 1.36), (3, 1.25)])
    def test_and_they_really_are_strained(self, pentagons, worst):
        """The refusal is not squeamishness: a square apex comes back with
        90 deg angles and a triangular one with 60."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build_nanocone(n_pentagons=pentagons, radius=18.0,
                                   strict=False)
        assert atoms.info["geometry"]["bond_min"] < worst
        assert atoms.info["geometry"]["angle_min"] < 95.0

    def test_only_one_is_clean(self):
        assert CLEAN == 1
        assert BUILDABLE == (1, 2, 3)
