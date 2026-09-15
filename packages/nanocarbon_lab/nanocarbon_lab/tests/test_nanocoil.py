"""Tests for the helical nanocoil builder.

This builder winds a finished ``(n, m)`` lattice along a helix, so every
ring stays a hexagon and the wall stays graphitic -- which is the whole
reason to prefer it over meshing a surface. The price is that bending a
finished lattice can only *stretch* it: nothing here relieves curvature
the way a meshed wall does with 5-7 pairs.

Two separate things therefore break it, and the tests keep them apart
because the fixes are opposite. Wall strain, ``r_tube * kappa``, is cured
by widening the coil. Turns colliding is not: a tube wider than the pitch
passes through its own next turn at **every** coil radius, and widening
only makes more of it.

The earlier version of this file asserted ``1.2 < nearest_neighbour <
1.8``, a window loose enough to accept two carbons sitting on top of each
other. Every coil it built was torn -- measured, all six cases -- and it
passed. The assertions here go through ``sp2_quality`` instead, which is
the same judgement the rest of the framework uses.
"""

from __future__ import annotations

import math

import pytest

from nanocarbon_lab.builders import build_nanocoil
from nanocarbon_lab.builders.nanocoil import TURN_CLEARANCE, _clean_coil_radius
from nanocarbon_lab.topology import coordination_numbers
from nanocarbon_lab.validation import run_basic_checks
from nanocarbon_lab.validation.quality import sp2_quality

# A (4,4) tube on a 36 Å coil at a 10 Å pitch: 7.5% wall strain, 8.8 Å of
# clearance needed against a 10 Å pitch, 1474 atoms in half a second.
CLEAN = dict(n=4, m=4, coil_radius=36.0, pitch=10.0, n_turns=1.0)


def _verdict(coil):
    return sp2_quality(coil.info["geometry"], coil.info["quality_family"])


class TestACleanCoil:
    def test_the_wall_is_graphitic(self):
        """Every ring a hexagon. That is the point of this route: the
        meshed one relieves curvature with 5-7 pairs and comes back with
        ninety non-hexagons where Euler needs twelve."""
        coil = build_nanocoil(**CLEAN)
        assert set(coil.info["ring_counts"]) == {6}

    def test_the_geometry_is_sound(self):
        coil = build_nanocoil(**CLEAN)
        verdict, why = _verdict(coil)
        assert verdict == "clean", why
        assert coil.info["geometry"]["n_close_contacts"] == 0

    def test_arc_length_matches_the_helix(self):
        coil = build_nanocoil(**CLEAN)
        expected = math.hypot(2 * math.pi * CLEAN["coil_radius"], CLEAN["pitch"])
        assert coil.info["arc_length"] == pytest.approx(expected, rel=1e-10)

    def test_coordination_is_three_away_from_the_ends(self):
        coil = build_nanocoil(**CLEAN)
        coord = coordination_numbers(coil)
        assert (coord == 3).mean() > 0.95

    def test_validation_passes(self):
        coil = build_nanocoil(**CLEAN)
        report = run_basic_checks(coil)
        assert report.ok, report.summary()

    def test_metadata_records_what_was_asked_and_what_it_cost(self):
        coil = build_nanocoil(**CLEAN)
        info = coil.info
        assert info["structure_type"] == "nanocoil"
        assert info["coil_radius"] == CLEAN["coil_radius"]
        assert info["pitch"] == CLEAN["pitch"]
        assert info["wall_strain"] == pytest.approx(0.075, abs=0.005)
        assert info["min_pitch"] == pytest.approx(
            2 * info["tube_radius"] + TURN_CLEARANCE)

    def test_one_turn_spans_the_pitch(self):
        coil = build_nanocoil(**{**CLEAN, "pitch": 12.0})
        z = coil.get_positions()[:, 2]
        assert float(z.max() - z.min()) >= 12.0 - 1.0


class TestWallStrain:
    @pytest.mark.parametrize("coil_radius", [18.0, 25.0, 30.0])
    def test_a_coil_too_tight_to_wind_is_refused(self, coil_radius):
        """These are the radii the old tests used. Every one of them tore
        the wall, and the loose tolerance let them through."""
        with pytest.raises(ValueError, match="tears the wall"):
            build_nanocoil(n=5, m=5, coil_radius=coil_radius, pitch=12.0,
                           n_turns=1.0)

    def test_the_refusal_names_the_strain_and_a_radius_that_works(self):
        with pytest.raises(ValueError, match="strained") as excinfo:
            build_nanocoil(n=5, m=5, coil_radius=25.0, pitch=12.0, n_turns=1.0)
        assert "widen the coil" in str(excinfo.value)

    def test_widening_the_coil_fixes_it(self):
        coil = build_nanocoil(n=5, m=5, coil_radius=45.0, pitch=12.0,
                              n_turns=1.0)
        assert _verdict(coil)[0] == "clean"

    def test_the_clean_radius_is_where_the_strain_crosses(self):
        """The helper inverts strain = r * R / (R^2 + c^2) for R, so the
        radius it names must be the one that gives back the budget."""
        radius, pitch, budget = 3.39, 12.0, 0.08
        wide = _clean_coil_radius(radius, pitch, budget)
        c = pitch / (2 * math.pi)
        assert radius * wide / (wide ** 2 + c ** 2) == pytest.approx(budget)


class TestTurnsColliding:
    """A (10,10) tube is 13.6 Å across, so on a 13 Å pitch it passes
    through its own next turn. Measured at R = 120 Å, where the wall
    strain is a comfortable 5.6%: 5211 overlapping pairs and 0.69 Å
    bonds. Two turns, not one, throughout: with a single turn there is no
    next turn to hit, only the two free ends, and whether those happen to
    line up depends on the radius -- measured, R = 80 Å builds and 50 and
    120 do not. The rule is about turns meeting turns, so the tests ask
    for turns.
    """

    def test_a_pitch_narrower_than_the_tube_is_refused(self):
        with pytest.raises(ValueError, match="pass through each other"):
            build_nanocoil(n=10, m=10, coil_radius=120.0, pitch=13.0,
                           n_turns=2.0)

    @pytest.mark.parametrize("coil_radius", [50.0, 80.0, 120.0])
    def test_widening_the_coil_does_not_help(self, coil_radius):
        with pytest.raises(ValueError, match="pass through each other"):
            build_nanocoil(n=10, m=10, coil_radius=coil_radius,
                           pitch=13.0, n_turns=2.0)

    def test_raising_the_pitch_does(self):
        coil = build_nanocoil(n=10, m=10, coil_radius=120.0, pitch=18.0,
                              n_turns=2.0)
        assert coil.info["geometry"]["n_close_contacts"] == 0


class TestRejections:
    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            build_nanocoil(n=5, m=5, coil_radius=0.0, pitch=12.0, n_turns=1.0)
        with pytest.raises(ValueError):
            build_nanocoil(n=5, m=5, coil_radius=25.0, pitch=12.0, n_turns=0.0)

    def test_a_coil_tighter_than_the_tube_is_refused_before_building(self):
        with pytest.raises(ValueError, match="too small"):
            build_nanocoil(n=6, m=6, coil_radius=5.0, pitch=12.0, n_turns=1.0)

    def test_stone_wales_density_bounds(self):
        with pytest.raises(ValueError):
            build_nanocoil(**{**CLEAN, "stone_wales_density": 0.5})
