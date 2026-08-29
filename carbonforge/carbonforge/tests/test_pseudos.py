"""Tests for the pseudopotential requirements helper."""

from __future__ import annotations

import pytest

from carbonforge.builders import build_cnt, build_graphene_supercell
from carbonforge.dopants import dope_random
from carbonforge.exports.pseudos import (
    check_directory,
    describe,
    pseudopotential_map,
    requirements_for,
)


class TestRequirements:
    def test_one_entry_per_element(self):
        doped = dope_random(build_cnt(10, 0, length=6), "N", 0.05, seed=0)
        requirements = requirements_for(doped)
        assert sorted(r.element for r in requirements) == ["C", "N"]

    def test_default_is_paw(self):
        requirements = requirements_for(build_graphene_supercell(2, 2))
        assert requirements[0].family == "PAW"
        assert not requirements[0].relativistic

    def test_raman_forces_norm_conserving(self):
        requirements = requirements_for(
            build_graphene_supercell(2, 2), needs_raman=True
        )
        assert requirements[0].family == "NC"
        assert "PAW" in requirements[0].reason

    def test_soc_forces_relativistic(self):
        requirements = requirements_for(
            build_graphene_supercell(2, 2), needs_soc=True
        )
        assert requirements[0].relativistic
        assert "rel-" in requirements[0].filename

    def test_raman_plus_soc_needs_both(self):
        """The narrowest case: norm-conserving AND fully relativistic."""
        requirements = requirements_for(
            build_graphene_supercell(2, 2), needs_raman=True, needs_soc=True
        )
        assert requirements[0].family == "NC"
        assert requirements[0].relativistic
        assert "nc-fr" in requirements[0].source.lower()

    def test_source_matches_family(self):
        paw = requirements_for(build_graphene_supercell(2, 2))[0]
        nc = requirements_for(build_graphene_supercell(2, 2), needs_raman=True)[0]
        assert "quantum-espresso.org" in paw.source
        assert "pseudo-dojo" in nc.source

    def test_map_for_qe_settings(self):
        doped = dope_random(build_cnt(10, 0, length=6), "N", 0.05, seed=0)
        mapping = pseudopotential_map(requirements_for(doped))
        assert set(mapping) == {"C", "N"}
        assert all(name.endswith((".UPF", ".upf")) for name in mapping.values())


class TestDescribe:
    def test_mentions_family_reason_and_source(self):
        text = describe(requirements_for(build_graphene_supercell(2, 2),
                                         needs_raman=True))
        assert "NC" in text
        assert "pseudo-dojo" in text
        assert "convergencia" in text

    def test_cutoff_guidance_is_family_dependent(self):
        """Norm-conserving needs a higher cutoff than PAW; say so."""
        paw = describe(requirements_for(build_graphene_supercell(2, 2)))
        nc = describe(requirements_for(build_graphene_supercell(2, 2),
                                       needs_raman=True))
        assert "50–60 Ry" in paw
        assert "80–100 Ry" in nc

    def test_empty_input(self):
        assert "No hay elementos" in describe([])


class TestCheckDirectory:
    def test_all_present(self, tmp_path):
        requirements = requirements_for(build_graphene_supercell(2, 2))
        (tmp_path / requirements[0].filename).touch()
        check = check_directory(tmp_path, requirements)
        assert check.ok
        assert "Todo listo" in check.summary()

    def test_missing_reported(self, tmp_path):
        requirements = requirements_for(build_graphene_supercell(2, 2))
        check = check_directory(tmp_path, requirements)
        assert not check.ok
        assert "Faltan" in check.summary()

    def test_case_insensitive_match(self, tmp_path):
        """UPF files circulate as both .UPF and .upf; QE accepts either."""
        requirements = requirements_for(build_graphene_supercell(2, 2))
        (tmp_path / requirements[0].filename.lower()).touch()
        assert check_directory(tmp_path, requirements).ok

    def test_substitute_suggested_not_assumed(self, tmp_path):
        """A different C pseudopotential is offered, never silently used."""
        requirements = requirements_for(build_graphene_supercell(2, 2))
        (tmp_path / "C_ONCV_PBE-1.2.upf").touch()
        check = check_directory(tmp_path, requirements)
        assert not check.ok
        assert "C_ONCV_PBE-1.2.upf" in check.summary()
        assert check.substitutes["C"]

    def test_other_elements_not_offered_as_substitutes(self, tmp_path):
        """'Ca...' must not be proposed as a substitute for carbon."""
        requirements = requirements_for(build_graphene_supercell(2, 2))
        (tmp_path / "Ca.pbe-spn-kjpaw_psl.1.0.0.UPF").touch()
        check = check_directory(tmp_path, requirements)
        assert "C" not in check.substitutes

    def test_missing_directory_is_reported_not_crashed(self, tmp_path):
        requirements = requirements_for(build_graphene_supercell(2, 2))
        check = check_directory(tmp_path / "nope", requirements)
        assert not check.ok
        assert "No existe" in check.summary()
