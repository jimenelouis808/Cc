"""Tests for band paths, spectroscopy and spin-orbit setups."""

from __future__ import annotations

import numpy as np
import pytest

from carbonforge.builders import (
    build_carbon_foam,
    build_cnt,
    build_graphene,
    build_graphene_supercell,
    build_nanocoil,
    build_nanoribbon,
)
from carbonforge.calculations import (
    format_qe_kpath,
    format_siesta_bandlines,
    heaviest_element,
    infrared_setup,
    phonon_setup,
    raman_setup,
    relativistic_pseudo_name,
    soc_is_physically_relevant,
    soc_setup,
    suggest_band_path,
)
from carbonforge.calculations.spectroscopy import (
    format_dynmat_input,
    format_ph_input,
)


class TestBandPaths:
    def test_cnt_is_one_dimensional(self):
        spec = suggest_band_path(build_cnt(6, 6, length=8))
        assert spec.dimensionality == 1
        assert spec.labels[0] == "G"

    def test_cnt_zone_boundary_lies_on_periodic_axis(self):
        """The k-path must run along z, the only periodic direction."""
        spec = suggest_band_path(build_cnt(6, 6, length=8))
        boundary = spec.points[-1]
        assert abs(boundary[2]) == pytest.approx(0.5)
        assert boundary[0] == pytest.approx(0.0)
        assert boundary[1] == pytest.approx(0.0)

    def test_ribbon_zone_boundary_follows_its_own_axis(self):
        """The ribbon is periodic along y, so the path must be along y."""
        spec = suggest_band_path(build_nanoribbon(4, 3))
        boundary = spec.points[-1]
        assert abs(boundary[1]) == pytest.approx(0.5)
        assert boundary[0] == pytest.approx(0.0)

    def test_hexagonal_graphene_gives_textbook_path(self):
        spec = suggest_band_path(build_graphene())
        assert spec.path_string == "G-M-K-G"

    def test_orthogonal_supercell_differs_from_hexagonal(self):
        """The whole reason for delegating to ASE: the paths are not the same."""
        hexagonal = suggest_band_path(build_graphene())
        orthogonal = suggest_band_path(build_graphene_supercell(2, 2))
        assert hexagonal.path_string != orthogonal.path_string
        assert "K" not in orthogonal.labels

    def test_isolated_system_has_no_dispersion(self):
        coil = build_nanocoil(n=5, m=5, coil_radius=25, pitch=12, n_turns=0.3)
        spec = suggest_band_path(coil)
        assert spec.dimensionality == 0
        assert spec.labels == ["G"]
        assert spec.source == "fallback"

    def test_three_dimensional_path(self):
        foam = build_carbon_foam(box_size=20, n_flakes=3, flake_radius=3, seed=0)
        assert suggest_band_path(foam).dimensionality == 3

    def test_total_points_accounting(self):
        spec = suggest_band_path(build_graphene(), npoints_per_segment=10)
        # 3 segments for G-M-K-G.
        assert spec.total_points() == 31

    def test_qe_card_format(self):
        card = format_qe_kpath(suggest_band_path(build_graphene()))
        assert card.startswith("K_POINTS crystal_b")
        lines = [ln for ln in card.splitlines() if "!" in ln]
        assert len(lines) == 4
        # QE counts points to the *next* label, so the last one is 0.
        assert lines[-1].split()[3] == "0"

    def test_siesta_block_uses_its_own_convention(self):
        """SIESTA counts points arriving at each label, so the first is 1."""
        block = format_siesta_bandlines(suggest_band_path(build_graphene()))
        assert block.startswith("%block BandLines")
        first = block.splitlines()[1]
        assert first.split()[0] == "1"


class TestSpectroscopySpecs:
    def test_raman_implies_dielectric_response(self):
        spec = raman_setup()
        assert spec.needs_raman and spec.needs_epsil

    def test_infrared_needs_epsil_but_not_raman(self):
        spec = infrared_setup()
        assert spec.needs_epsil and not spec.needs_raman

    def test_plain_phonons_need_neither(self):
        spec = phonon_setup()
        assert not spec.needs_epsil and not spec.needs_raman

    def test_ph_input_switches(self):
        text = format_ph_input(raman_setup())
        assert "trans = .true." in text
        assert "epsil = .true." in text
        assert "lraman = .true." in text

    def test_phonon_only_omits_raman_switches(self):
        text = format_ph_input(phonon_setup())
        assert "lraman" not in text
        assert "epsil" not in text

    def test_dynmat_carries_laser_and_temperature(self):
        text = format_dynmat_input(raman_setup(laser_wavelength_nm=633.0))
        assert "633" in text
        assert "asr = 'crystal'" in text


class TestSpinOrbit:
    def test_soc_forces_noncollinear(self):
        # lspinorb without noncolin is rejected by QE; the dataclass fixes it.
        spec = soc_setup(noncolin=False)
        assert spec.noncolin is True

    def test_carbon_is_too_light_to_matter(self):
        assert not soc_is_physically_relevant(["C"])
        assert not soc_is_physically_relevant(["C", "N", "B"])

    def test_heavy_adatoms_make_soc_relevant(self):
        assert soc_is_physically_relevant(["C", "Au"])
        assert soc_is_physically_relevant(["C", "Bi"])

    def test_heaviest_element_detection(self):
        assert heaviest_element(["C", "N", "S"]) == ("S", 16)
        assert heaviest_element([]) == ("", 0)

    def test_relativistic_pseudo_renaming(self):
        assert (
            relativistic_pseudo_name("C", "C.pbe-n-kjpaw_psl.1.0.0.UPF")
            == "C.rel-pbe-n-kjpaw_psl.1.0.0.UPF"
        )

    def test_renaming_is_idempotent(self):
        already = "C.rel-pbe-n-kjpaw_psl.1.0.0.UPF"
        assert relativistic_pseudo_name("C", already) == already
