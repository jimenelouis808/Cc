"""Tests for the dopant chemistry table and ring-selected placement.

Two things are being pinned here. The first is that the table is not
merely a longer element list: each entry carries a site type and a
ceiling, and the whole point is that 10% N and 10% Fe are treated
differently. The second is that pentagon placement reads the ring
metadata the builders already record, and refuses rather than guessing
when it is absent -- re-deriving rings from coordinates on a curved shell
is exactly the mistake `builders/fullerene_mesh.py` exists to prevent.
"""

from __future__ import annotations

import warnings

import pytest

from nanocarbon_lab.builders import build_capped_cnt, build_cnt, build_fullerene
from nanocarbon_lab.dopants import (
    DOPANT_CHEMISTRY,
    DOPANT_ELEMENTS,
    PLANAR_DOPANTS,
    dope_random,
    dope_rings,
    get_chemistry,
    ring_sites,
    ring_size_census,
)
from nanocarbon_lab.utils.constants import (
    COVALENT_RADII,
    MAX_COORDINATION,
    MAX_DOPING_FRACTION,
    MIN_DOPING_FRACTION,
)
from nanocarbon_lab.validation import run_basic_checks


@pytest.fixture(scope="module")
def cage():
    """C60: every carbon sits on exactly one pentagon."""
    return build_fullerene(family="C60", freq=1)


@pytest.fixture(scope="module")
def capped():
    """A capped tube: 12 pentagons in the caps, hexagons along the body."""
    return build_capped_cnt(n_body_rings=6, freq=2)


class TestChemistryTable:
    def test_the_elements_the_user_asked_for_are_all_there(self):
        for element in ("N", "B", "P", "S", "Se", "O"):
            assert element in DOPANT_CHEMISTRY

    def test_every_dopant_has_a_radius_and_a_coordination_ceiling(self):
        """The load-bearing invariant.

        An element with no covalent radius falls back to 1.80 Å in
        `guess_bonds`, which is how the whole tmd package once validated
        as isolated atoms. An element with no coordination ceiling is
        judged against carbon's, which rejects anything larger.
        """
        for element in DOPANT_ELEMENTS:
            assert element in COVALENT_RADII, element
            assert element in MAX_COORDINATION, element

    def test_only_n_and_b_are_planar(self):
        """Not a taste call: they are the only two within 0.15 Å of
        carbon, and the only two that reach tens of per cent in real
        samples."""
        assert set(PLANAR_DOPANTS) == {"N", "B"}

    def test_planar_dopants_tolerate_more_than_puckered_ones(self):
        planar = min(DOPANT_CHEMISTRY[s].max_fraction for s in PLANAR_DOPANTS)
        others = max(c.max_fraction for c in DOPANT_CHEMISTRY.values()
                     if c.site != "planar")
        assert planar > others

    def test_the_size_mismatch_matches_the_site_type(self):
        """Size is what decides whether a dopant stays in the plane, so
        the two fields must not disagree."""
        for chem in DOPANT_CHEMISTRY.values():
            if chem.site == "planar":
                assert abs(chem.size_mismatch) < 0.20, chem.symbol
            else:
                assert chem.size_mismatch > 0.10 or chem.symbol == "O", chem.symbol

    def test_an_unknown_element_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="Unsupported dopant"):
            get_chemistry("Xx")

    def test_the_interface_range_is_one_to_fifteen_per_cent(self):
        assert (MIN_DOPING_FRACTION, MAX_DOPING_FRACTION) == (0.01, 0.15)


class TestUnrealisticFractionsWarn:
    def test_ten_per_cent_iron_warns(self, capped):
        with pytest.warns(RuntimeWarning, match="physically meaningful"):
            dope_random(capped, "Fe", 0.10, seed=0)

    def test_ten_per_cent_nitrogen_does_not(self, capped):
        """The whole point of a per-element ceiling: the same fraction is
        ordinary for N and absurd for Fe."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            dope_random(capped, "N", 0.10, seed=0)

    def test_it_warns_but_still_builds(self, capped):
        """A warning, not a refusal -- metastable and purely
        computational structures are a legitimate subject."""
        with pytest.warns(RuntimeWarning):
            doped = dope_random(capped, "Fe", 0.10, seed=0)
        assert doped.get_chemical_symbols().count("Fe") > 0


class TestRingSites:
    def test_every_carbon_in_c60_is_on_a_pentagon(self, cage):
        assert len(ring_sites(cage, 5)) == len(cage)

    def test_a_capped_tube_has_exactly_twelve_pentagons_worth(self, capped):
        """Twelve pentagons of five atoms each, and no atom is on two of
        them -- the caps are far enough apart."""
        assert len(ring_sites(capped, 5)) == 60

    def test_the_census_counts_sites_not_rings(self, capped):
        """An atom sits on three rings at once, so these overlap and are
        deliberately not the ring counts in info['ring_counts']."""
        census = ring_size_census(capped)
        assert census[5] == 60
        assert census[6] == len(capped)

    def test_a_plain_tube_has_no_ring_metadata_and_says_so(self):
        """`build_cnt` does not go through the dual, so it records no
        rings -- and re-deriving them from coordinates is the thing the
        mesh machinery exists to avoid."""
        with pytest.raises(ValueError, match="no ring metadata"):
            ring_sites(build_cnt(n=6, m=6, length=8), 5)


class TestDopeRings:
    def test_it_places_on_pentagons_only(self, capped):
        doped = dope_rings(capped, "N", ring_size=5, concentration=0.2, seed=0)
        pentagon = set(ring_sites(capped, 5).tolist())
        placed = {i for i, s in enumerate(doped.get_chemical_symbols())
                  if s == "N"}
        assert placed
        assert placed <= pentagon

    def test_the_fraction_is_of_the_pentagon_sites(self, capped):
        """10% of 60 pentagon carbons is 6 atoms, not 10% of 240."""
        doped = dope_rings(capped, "N", ring_size=5, concentration=0.10, seed=0)
        assert doped.get_chemical_symbols().count("N") == 6

    def test_both_fractions_are_recorded(self, capped):
        """They differ by a factor of four here, so reporting only one
        would be read as the other."""
        doped = dope_rings(capped, "N", ring_size=5, concentration=0.10, seed=0)
        assert doped.info["doping_concentration"] == pytest.approx(0.10)
        assert doped.info["doping_concentration_overall"] == pytest.approx(0.025)

    def test_it_is_reproducible(self, capped):
        first = dope_rings(capped, "N", ring_size=5, count=6, seed=7)
        second = dope_rings(capped, "N", ring_size=5, count=6, seed=7)
        assert first.get_chemical_symbols() == second.get_chemical_symbols()

    def test_a_different_seed_picks_differently(self, capped):
        first = dope_rings(capped, "N", ring_size=5, count=6, seed=1)
        second = dope_rings(capped, "N", ring_size=5, count=6, seed=2)
        assert first.get_chemical_symbols() != second.get_chemical_symbols()

    def test_it_names_what_rings_do_exist(self, capped):
        with pytest.raises(ValueError, match="no 9-membered rings"):
            dope_rings(capped, "N", ring_size=9, count=1)

    def test_concentration_and_count_are_exclusive(self, capped):
        with pytest.raises(ValueError, match="exactly one"):
            dope_rings(capped, "N", ring_size=5)
        with pytest.raises(ValueError, match="exactly one"):
            dope_rings(capped, "N", ring_size=5, concentration=0.1, count=3)

    def test_an_unknown_dopant_is_rejected_before_any_work(self, capped):
        with pytest.raises(ValueError, match="Unsupported dopant"):
            dope_rings(capped, "Xx", ring_size=5, count=1)

    def test_the_result_still_validates(self, capped):
        doped = dope_rings(capped, "N", ring_size=5, concentration=0.10, seed=0)
        assert not run_basic_checks(doped).errors

    def test_the_host_is_untouched(self, capped):
        """Every doping function returns a new Atoms."""
        before = capped.get_chemical_symbols()
        dope_rings(capped, "N", ring_size=5, count=6, seed=0)
        assert capped.get_chemical_symbols() == before
