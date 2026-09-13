"""Tests for choosing a dichalcogenide by its two elements.

The table is the authority and `material_for` is deliberately a lookup
rather than a constructor: an MX2 that is not tabulated is one whose
lattice constants this package does not know, and deriving them from
covalent radii would produce a structure that looks authoritative and is
not. So the interesting assertions here are about what happens for a
pair that does not exist, and about the table's internal consistency.
"""

from __future__ import annotations

import math

import pytest

from nanocarbon_lab.tmd import build_tmd_monolayer, tmd_quality
from nanocarbon_lab.tmd.materials import (
    MATERIALS,
    available_chalcogens,
    available_metals,
    chalcogens_for,
    material_for,
    metals_for,
)
from nanocarbon_lab.tmd.quality import geometry_report
from nanocarbon_lab.utils.constants import (
    COVALENT_RADII,
    HOMOELEMENTAL_BOND,
    MAX_COORDINATION,
)
from nanocarbon_lab.validation import run_basic_checks


class TestLookupByElements:
    def test_it_finds_a_compound_by_its_two_elements(self):
        assert material_for("Mo", "S").formula == "MoS2"
        assert material_for("W", "Se").formula == "WSe2"

    def test_every_tabulated_compound_is_reachable_this_way(self):
        for material in MATERIALS.values():
            found = material_for(material.metal, material.chalcogen)
            assert found is material

    def test_an_unknown_metal_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="Unknown metal"):
            material_for("Re", "S")

    def test_an_unknown_chalcogen_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="Unknown chalcogen"):
            material_for("Mo", "O")

    def test_a_missing_pair_names_what_each_element_does_have(self):
        """Both elements are real; the compound is not. The message has
        to say which way to move, or the user is left guessing."""
        with pytest.raises(KeyError) as excinfo:
            material_for("Sn", "Te")
        message = str(excinfo.value)
        assert "No tabulated SnTe2" in message
        assert "Sn is available with" in message
        assert "Te is available with" in message

    def test_the_availability_queries_agree_with_the_table(self):
        for metal in available_metals():
            for chalcogen in chalcogens_for(metal):
                assert material_for(metal, chalcogen).formula
            assert metal in metals_for(chalcogens_for(metal)[0])

    def test_every_metal_and_chalcogen_in_the_table_is_listed(self):
        """A material added without touching the ordering lists must
        still reach the dropdowns rather than silently vanishing."""
        assert {m.metal for m in MATERIALS.values()} == set(available_metals())
        assert ({m.chalcogen for m in MATERIALS.values()}
                == set(available_chalcogens()))


class TestTableConsistency:
    @pytest.mark.parametrize("name", sorted(MATERIALS))
    def test_the_bond_length_is_physically_sensible(self, name):
        """`d` is derived from `a` and `h`, so a typo in either shows up
        here as a bond no dichalcogenide has."""
        material = MATERIALS[name]
        assert 2.2 < material.bond_length < 3.0, material.bond_length

    @pytest.mark.parametrize("name", sorted(MATERIALS))
    def test_the_van_der_waals_gap_is_a_contact_not_a_hole(self, name):
        """2.3-3.5 Å: a real vdW contact between closed-shell sheets.

        The lower end is the platinum dichalcogenides, whose gap really
        is ~2.4 Å -- that unusually strong interlayer coupling is why
        PtSe2's band gap depends so strongly on layer count, and it is
        not a data-entry error to be tidied away.
        """
        material = MATERIALS[name]
        assert 2.3 < material.vdw_gap < 3.5, material.vdw_gap

    @pytest.mark.parametrize("name", sorted(MATERIALS))
    def test_the_derived_bond_matches_the_geometry(self, name):
        material = MATERIALS[name]
        expected = math.sqrt(material.a**2 / 3.0 + material.h**2 / 4.0)
        assert material.bond_length == pytest.approx(expected)

    @pytest.mark.parametrize("name", sorted(MATERIALS))
    def test_every_element_has_the_constants_bond_detection_needs(self, name):
        """A metal missing from these tables reads as 12-coordinate, or
        as bonded to nothing at all. Both have happened."""
        material = MATERIALS[name]
        for element in (material.metal, material.chalcogen):
            assert element in COVALENT_RADII, element
            assert element in MAX_COORDINATION, element
        assert material.metal in HOMOELEMENTAL_BOND

    @pytest.mark.parametrize("name", sorted(MATERIALS))
    def test_a_metal_metal_cutoff_falls_below_the_lattice_repeat(self, name):
        """The whole reason BOND_CUTOFF_OVERRIDE exists: a cutoff above
        `a` makes every metal bond to its six in-plane neighbours."""
        from nanocarbon_lab.utils.constants import BOND_CUTOFF_OVERRIDE

        material = MATERIALS[name]
        pair = (material.metal, material.metal)
        assert BOND_CUTOFF_OVERRIDE[pair] < material.a, name


class TestEveryMaterialBuilds:
    @pytest.mark.parametrize("name", sorted(MATERIALS))
    def test_the_monolayer_is_clean_and_stoichiometric(self, name):
        material = MATERIALS[name]
        atoms = build_tmd_monolayer(name, phase=material.natural_phase)
        assert len(atoms) == 3
        assert not run_basic_checks(atoms).errors
        verdict, why = tmd_quality(geometry_report(atoms))
        assert verdict == "clean", f"{name}: {why}"
