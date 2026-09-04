"""Tests for post-build edits to a finished MX2 structure.

The one that needs real care is the Janus split, because "which face is
this chalcogen on" has no global answer: on a rolled tube the two faces
are the inner and outer walls, so any test written against a global z
would pass on a flat layer and silently mean nothing on a tube. These
check the tube case explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.tmd import build_tmd_layers, build_tmd_nanotube
from nanocarbon_lab.tmd.modify import (
    alloy,
    antisites,
    chalcogen_vacancies,
    make_janus,
)
from nanocarbon_lab.validation.checks import run_basic_checks


@pytest.fixture(scope="module")
def monolayer():
    return build_tmd_layers("MoS2", n_layers=1, nx=3, ny=3)


@pytest.fixture(scope="module")
def tube():
    return build_tmd_nanotube("MoS2", n=30, m=0)


class TestJanus:
    def test_it_replaces_exactly_one_face(self, monolayer):
        janus = make_janus(monolayer, "Se")
        symbols = janus.get_chemical_symbols()
        assert symbols.count("Se") == symbols.count("S")
        assert symbols.count("Mo") * 1 == symbols.count("Se")

    def test_the_faces_end_up_on_opposite_sides(self, monolayer):
        janus = make_janus(monolayer, "Se")
        z = janus.get_positions()[:, 2]
        species = np.array(janus.get_chemical_symbols())
        assert z[species == "Se"].mean() > z[species == "Mo"].mean()
        assert z[species == "Mo"].mean() > z[species == "S"].mean()

    def test_it_follows_a_rolled_tube(self, tube):
        """The decisive case. On a tube the two faces are the inner and
        outer walls, so a global-z rule would mix them; the sides are
        found from each chalcogen's own outward direction instead.
        """
        janus = make_janus(tube, "Se")
        positions = janus.get_positions()
        species = np.array(janus.get_chemical_symbols())
        centre = positions[:, :2].mean(axis=0)
        radius = np.linalg.norm(positions[:, :2] - centre, axis=1)
        assert radius[species == "Se"].mean() > radius[species == "Mo"].mean()
        assert radius[species == "Mo"].mean() > radius[species == "S"].mean()
        # And cleanly separated, not merely different on average.
        assert radius[species == "Se"].min() > radius[species == "S"].max()

    def test_the_inner_face_can_be_chosen(self, tube):
        inner = make_janus(tube, "Se", side=-1)
        positions = inner.get_positions()
        species = np.array(inner.get_chemical_symbols())
        centre = positions[:, :2].mean(axis=0)
        radius = np.linalg.norm(positions[:, :2] - centre, axis=1)
        assert radius[species == "Se"].mean() < radius[species == "S"].mean()

    def test_a_bilayer_gets_both_layers(self):
        bilayer = build_tmd_layers("MoS2", n_layers=2, nx=2, ny=2)
        janus = make_janus(bilayer, "Se")
        symbols = janus.get_chemical_symbols()
        assert symbols.count("Se") == symbols.count("S")

    def test_geometry_is_untouched(self, monolayer):
        janus = make_janus(monolayer, "Se")
        assert np.allclose(janus.get_positions(), monolayer.get_positions())

    def test_the_same_chalcogen_is_rejected(self, monolayer):
        with pytest.raises(ValueError, match="already all S"):
            make_janus(monolayer, "S")

    def test_an_unknown_chalcogen_is_rejected(self, monolayer):
        with pytest.raises(ValueError, match="chalcogen must be"):
            make_janus(monolayer, "O")

    def test_a_bad_side_is_rejected(self, monolayer):
        with pytest.raises(ValueError, match="side must be"):
            make_janus(monolayer, "Se", side=0)


class TestVacancies:
    def test_it_removes_the_requested_number(self, monolayer):
        out = chalcogen_vacancies(monolayer, n_defects=3, seed=0)
        assert len(out) == len(monolayer) - 3
        assert out.get_chemical_symbols().count("Mo") == 9

    def test_only_chalcogens_are_removed(self, monolayer):
        out = chalcogen_vacancies(monolayer, n_defects=5, seed=1)
        before = monolayer.get_chemical_symbols().count("Mo")
        assert out.get_chemical_symbols().count("Mo") == before

    def test_it_is_reproducible(self, monolayer):
        first = chalcogen_vacancies(monolayer, n_defects=4, seed=7)
        second = chalcogen_vacancies(monolayer, n_defects=4, seed=7)
        assert np.allclose(first.get_positions(), second.get_positions())

    def test_a_different_seed_picks_different_sites(self, monolayer):
        first = chalcogen_vacancies(monolayer, n_defects=4, seed=0)
        second = chalcogen_vacancies(monolayer, n_defects=4, seed=99)
        assert not np.allclose(first.get_positions(), second.get_positions())

    def test_paired_removes_both_sides(self, monolayer):
        single = chalcogen_vacancies(monolayer, n_defects=2, seed=0)
        double = chalcogen_vacancies(monolayer, n_defects=2, seed=0,
                                     paired=True)
        assert len(double) < len(single)

    def test_asking_for_too_many_is_refused(self, monolayer):
        with pytest.raises(ValueError, match="only"):
            chalcogen_vacancies(monolayer, n_defects=999)

    def test_zero_is_a_copy(self, monolayer):
        out = chalcogen_vacancies(monolayer, n_defects=0)
        assert len(out) == len(monolayer)

    def test_it_is_logged(self, monolayer):
        out = chalcogen_vacancies(monolayer, n_defects=2, seed=3)
        entry = out.info["defect_log"][-1]
        assert entry["type"] == "chalcogen_vacancy"
        assert entry["seed"] == 3


class TestAlloy:
    def test_half_the_metal_is_replaced(self, monolayer):
        out = alloy(monolayer, "W", fraction=0.5, seed=0)
        symbols = out.get_chemical_symbols()
        assert symbols.count("W") + symbols.count("Mo") == 9
        assert out.info["alloy"]["achieved_fraction"] == pytest.approx(
            4 / 9, abs=1e-9)

    def test_the_achieved_fraction_is_reported_not_the_request(self, monolayer):
        """Nine metal sites cannot be split in half, and it is the
        achieved value that describes the structure."""
        out = alloy(monolayer, "W", fraction=0.5, seed=0)
        assert out.info["alloy"]["requested_fraction"] == 0.5
        assert out.info["alloy"]["achieved_fraction"] != 0.5

    def test_the_chalcogen_sublattice_can_be_alloyed(self, monolayer):
        out = alloy(monolayer, "Se", fraction=0.25, seed=0, site="chalcogen")
        assert "Se" in out.get_chemical_symbols()
        assert out.get_chemical_symbols().count("Mo") == 9

    def test_zero_and_one_are_the_endpoints(self, monolayer):
        none = alloy(monolayer, "W", fraction=0.0)
        every = alloy(monolayer, "W", fraction=1.0)
        assert none.get_chemical_symbols().count("W") == 0
        assert every.get_chemical_symbols().count("Mo") == 0

    def test_a_fraction_outside_the_range_is_rejected(self, monolayer):
        with pytest.raises(ValueError, match="fraction"):
            alloy(monolayer, "W", fraction=1.5)

    def test_an_unknown_site_is_rejected(self, monolayer):
        with pytest.raises(ValueError, match="site must be"):
            alloy(monolayer, "W", site="middle")


class TestAntisites:
    def test_a_metal_lands_on_a_chalcogen_site(self, monolayer):
        out = antisites(monolayer, n_defects=2, seed=0)
        symbols = out.get_chemical_symbols()
        assert symbols.count("Mo") == 11
        assert symbols.count("S") == 16

    def test_asking_for_too_many_is_refused(self, monolayer):
        with pytest.raises(ValueError, match="only"):
            antisites(monolayer, n_defects=999)


class TestComposability:
    def test_the_edits_stack(self, monolayer):
        """Build, make it Janus, alloy it, then knock atoms out -- each
        edit leaves a structure the next one can still read."""
        out = make_janus(monolayer, "Se")
        out = alloy(out, "W", fraction=0.3, seed=0)
        out = chalcogen_vacancies(out, n_defects=2, seed=0)
        assert set(out.get_chemical_symbols()) == {"Mo", "W", "S", "Se"}
        assert not run_basic_checks(out).errors
