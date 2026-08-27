"""Tests for dopants and defects."""

from __future__ import annotations

import pytest

from nanocarbon_lab.builders import build_cnt, build_graphene_supercell
from nanocarbon_lab.defects import (
    apply_random_distortion,
    introduce_vacancies,
    stone_wales_defect,
)
from nanocarbon_lab.dopants import codope, dope_directed, dope_random, substitute_atoms


class TestDopants:
    def test_random_concentration(self):
        gr = build_graphene_supercell(4, 4)
        doped = dope_random(gr, "N", 0.1, seed=42)
        n_count = sum(1 for s in doped.get_chemical_symbols() if s == "N")
        assert n_count == round(0.1 * len(gr))

    def test_reproducible_with_seed(self):
        gr = build_graphene_supercell(3, 3)
        a = dope_random(gr, "B", 0.1, seed=7)
        b = dope_random(gr, "B", 0.1, seed=7)
        assert a.get_chemical_symbols() == b.get_chemical_symbols()

    def test_zero_concentration_is_noop(self):
        gr = build_graphene_supercell(2, 2)
        out = dope_random(gr, "N", 0.0, seed=0)
        assert out.get_chemical_symbols() == gr.get_chemical_symbols()

    def test_invalid_element_raises(self):
        gr = build_graphene_supercell(2, 2)
        with pytest.raises(ValueError):
            dope_random(gr, "Si", 0.1)

    def test_substitute_out_of_range(self):
        gr = build_graphene_supercell(2, 2)
        with pytest.raises(IndexError):
            substitute_atoms(gr, [9999], "N")

    def test_directed_bulk(self):
        gr = build_graphene_supercell(4, 4)
        doped = dope_directed(gr, "N", where="bulk", count=2, seed=0)
        assert sum(1 for s in doped.get_chemical_symbols() if s == "N") == 2

    def test_codoping_mixes_species(self):
        gr = build_graphene_supercell(4, 4)
        doped = codope(gr, [("N", 0.05), ("B", 0.05)], seed=1)
        syms = doped.get_chemical_symbols()
        assert "N" in syms and "B" in syms

    def test_records_dopant_metadata(self):
        gr = build_graphene_supercell(3, 3)
        doped = dope_random(gr, "N", 0.1, seed=0)
        assert "dopants" in doped.info
        assert doped.info["dopants"][0]["element"] == "N"


class TestDefects:
    def test_monovacancy_removes_one_atom(self):
        cnt = build_cnt(6, 6, length=10)
        n0 = len(cnt)
        out = introduce_vacancies(cnt, n_defects=1, kind="mono", seed=0)
        assert len(out) == n0 - 1

    def test_divacancy_removes_two_atoms(self):
        cnt = build_cnt(6, 6, length=10)
        n0 = len(cnt)
        out = introduce_vacancies(cnt, n_defects=1, kind="di", seed=0)
        assert len(out) == n0 - 2

    def test_multiple_defects_respect_separation(self):
        cnt = build_cnt(8, 8, length=20)
        out = introduce_vacancies(cnt, n_defects=3, kind="mono",
                                  seed=0, min_separation=4.0)
        assert len(out) == len(cnt) - 3

    def test_stone_wales_preserves_atom_count(self):
        gr = build_graphene_supercell(5, 5)
        out = stone_wales_defect(gr, seed=0)
        assert len(out) == len(gr)
        assert out.info["defects"][-1]["type"] == "stone_wales"

    def test_distortion_moves_atoms(self):
        gr = build_graphene_supercell(3, 3)
        out = apply_random_distortion(gr, amplitude=0.05, seed=0)
        assert len(out) == len(gr)
        delta = (out.get_positions() - gr.get_positions())
        assert (delta ** 2).sum() > 0

    def test_distortion_rejects_huge_amplitude(self):
        gr = build_graphene_supercell(2, 2)
        with pytest.raises(ValueError):
            apply_random_distortion(gr, amplitude=0.5)
