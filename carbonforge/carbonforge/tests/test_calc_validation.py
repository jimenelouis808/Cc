"""Tests for the calculation-level physics checks.

These encode the failure modes the checks exist to catch, so a regression
here means the package has started telling users that an impossible
calculation is fine.
"""

from __future__ import annotations

import pytest

from carbonforge.builders import (
    build_carbon_foam,
    build_cnt,
    build_graphene_supercell,
    build_nanoribbon,
)
from carbonforge.calculations import (
    infrared_setup,
    phonon_setup,
    raman_setup,
    soc_setup,
    suggest_band_path,
)
from carbonforge.validation.calculations import (
    check_band_path,
    check_calculation_type,
    check_full_setup,
    check_spectroscopy,
    check_spinorbit,
    is_fully_relativistic,
    is_likely_metallic,
    pseudo_family,
)

PAW = {"C": "C.pbe-n-kjpaw_psl.1.0.0.UPF"}
NC = {"C": "C_ONCV_PBE-1.2.upf"}
NC_REL = {"C": "C.rel-pbe-n-nc.UPF"}


class TestPseudoClassification:
    @pytest.mark.parametrize(
        "filename,family",
        [
            ("C.pbe-n-kjpaw_psl.1.0.0.UPF", "PAW"),
            ("C.pbe-rrkjus.UPF", "USPP"),
            ("C_ONCV_PBE-1.2.upf", "NC"),
            ("C.SG15.upf", "NC"),
            ("weird_name.upf", "unknown"),
        ],
    )
    def test_family_detection(self, filename, family):
        assert pseudo_family(filename) == family

    def test_relativistic_detection(self):
        assert is_fully_relativistic("C.rel-pbe-n-kjpaw_psl.1.0.0.UPF")
        assert not is_fully_relativistic("C.pbe-n-kjpaw_psl.1.0.0.UPF")


class TestMetallicity:
    def test_armchair_cnt_is_metallic(self):
        metallic, reason = is_likely_metallic(build_cnt(6, 6, length=6))
        assert metallic and "armchair" in reason

    def test_zigzag_multiple_of_three_is_metallic(self):
        # (9,0): n-m = 9, divisible by 3 -> metallic/near-metallic.
        metallic, _ = is_likely_metallic(build_cnt(9, 0, length=6))
        assert metallic

    def test_zigzag_semiconducting(self):
        # (10,0): n-m = 10, not divisible by 3 -> semiconductor.
        metallic, reason = is_likely_metallic(build_cnt(10, 0, length=6))
        assert not metallic and "semiconductor" in reason

    def test_graphene_is_semimetal(self):
        metallic, reason = is_likely_metallic(build_graphene_supercell(2, 2))
        assert metallic and "semimetal" in reason

    def test_zigzag_ribbon_has_metallic_edges(self):
        metallic, _ = is_likely_metallic(build_nanoribbon(4, 3, edge="zigzag"))
        assert metallic

    def test_armchair_ribbon_semiconducting(self):
        # width 6 -> 6 % 3 == 0, not the near-metallic 3p+2 family.
        metallic, _ = is_likely_metallic(build_nanoribbon(6, 3, edge="armchair"))
        assert not metallic

    def test_armchair_ribbon_3p_plus_2_family(self):
        # width 5 -> 5 % 3 == 2, gap nearly closes.
        metallic, _ = is_likely_metallic(build_nanoribbon(5, 3, edge="armchair"))
        assert metallic

    def test_unknown_structure_assumed_metallic(self):
        """Conservative default: assuming metallic only ever adds a warning."""
        foam = build_carbon_foam(box_size=20, n_flakes=3, flake_radius=3, seed=0)
        metallic, _ = is_likely_metallic(foam)
        assert metallic


class TestSpectroscopyChecks:
    def test_raman_on_metal_is_rejected(self):
        report = check_spectroscopy(build_cnt(6, 6, length=6), raman_setup(), NC)
        assert not report.ok
        assert any("gap" in e for e in report.errors)

    def test_raman_with_paw_is_rejected(self):
        report = check_spectroscopy(build_cnt(10, 0, length=6), raman_setup(), PAW)
        assert not report.ok
        assert any("PAW" in e for e in report.errors)

    def test_raman_on_semiconductor_with_nc_passes(self):
        report = check_spectroscopy(build_cnt(10, 0, length=6), raman_setup(), NC)
        assert report.ok, report.summary()

    def test_plain_phonons_work_on_metal_with_paw(self):
        """The safe path: no gap requirement, no pseudopotential restriction."""
        report = check_spectroscopy(build_cnt(6, 6, length=6), phonon_setup(), PAW)
        assert report.ok, report.summary()

    def test_infrared_off_gamma_is_rejected(self):
        spec = infrared_setup(qpoint=(0.0, 0.0, 0.5))
        report = check_spectroscopy(build_cnt(10, 0, length=6), spec, NC)
        assert not report.ok
        assert any("Γ" in e for e in report.errors)

    def test_dispersion_conflicts_with_intensities(self):
        spec = raman_setup(ldisp=True, nq=(4, 4, 1))
        report = check_spectroscopy(build_cnt(10, 0, length=6), spec, NC)
        assert not report.ok

    def test_missing_asr_warns(self):
        report = check_spectroscopy(
            build_cnt(10, 0, length=6), phonon_setup(asr="no"), NC
        )
        assert any("acústica" in w for w in report.warnings)

    def test_low_dimensional_epsilon_warning(self):
        report = check_spectroscopy(build_cnt(10, 0, length=6), raman_setup(), NC)
        assert any("vacío" in w for w in report.warnings)

    def test_unknown_pseudo_family_warns(self):
        report = check_spectroscopy(
            build_cnt(10, 0, length=6), raman_setup(), {"C": "mystery.upf"}
        )
        assert any("norm-conserving" in w for w in report.warnings)


class TestSpinOrbitChecks:
    def test_scalar_pseudos_rejected(self):
        report = check_spinorbit(build_graphene_supercell(2, 2), soc_setup(), PAW)
        assert not report.ok
        assert any("relativistas" in e for e in report.errors)

    def test_relativistic_pseudos_accepted(self):
        report = check_spinorbit(build_graphene_supercell(2, 2), soc_setup(), NC_REL)
        assert report.ok, report.summary()

    def test_light_system_warns_about_magnitude(self):
        report = check_spinorbit(build_graphene_supercell(2, 2), soc_setup(), NC_REL)
        assert any("Z⁴" in w for w in report.warnings)

    def test_disabled_soc_is_silent(self):
        report = check_spinorbit(
            build_graphene_supercell(2, 2), soc_setup(enabled=False), PAW
        )
        assert report.ok and not report.warnings


class TestCalculationTypeChecks:
    def test_vc_relax_on_slab_rejected(self):
        """The vacuum would be compressed, silently changing the system."""
        report = check_calculation_type(build_graphene_supercell(2, 2), "vc-relax")
        assert not report.ok
        assert any("2Dxy" in e for e in report.errors)

    def test_vc_relax_on_wire_suggests_z_only(self):
        report = check_calculation_type(build_cnt(6, 6, length=6), "vc-relax")
        assert not report.ok
        assert any("'z'" in e for e in report.errors)

    def test_vc_relax_accepted_with_constraint(self):
        report = check_calculation_type(
            build_graphene_supercell(2, 2), "vc-relax", cell_dofree="2Dxy"
        )
        assert report.ok, report.summary()

    def test_relax_on_slab_is_fine(self):
        assert check_calculation_type(build_graphene_supercell(2, 2), "relax").ok

    def test_fixed_occupations_on_metal_rejected(self):
        report = check_calculation_type(
            build_cnt(6, 6, length=6), "scf", occupations="fixed"
        )
        assert not report.ok

    def test_smearing_on_semiconductor_only_warns(self):
        report = check_calculation_type(
            build_cnt(10, 0, length=6), "scf", occupations="smearing"
        )
        assert report.ok
        assert report.warnings


class TestBandPathChecks:
    def test_isolated_system_warns(self):
        from carbonforge.builders import build_nanocoil

        coil = build_nanocoil(n=5, m=5, coil_radius=25, pitch=12, n_turns=0.3)
        report = check_band_path(suggest_band_path(coil))
        assert any("0D" in w for w in report.warnings)

    def test_sparse_sampling_warns(self):
        spec = suggest_band_path(build_graphene_supercell(2, 2),
                                 npoints_per_segment=3)
        assert any("angulosa" in w for w in check_band_path(spec).warnings)

    def test_standard_path_is_clean(self):
        spec = suggest_band_path(build_graphene_supercell(2, 2))
        assert check_band_path(spec).ok


class TestFullSetup:
    def test_aggregates_every_failure(self):
        """Raman + metal + PAW + SOC on light atoms: four separate problems."""
        report = check_full_setup(
            build_cnt(6, 6, length=6),
            calculation="scf",
            spectroscopy=raman_setup(),
            spinorbit=soc_setup(),
            pseudopotentials=PAW,
        )
        assert not report.ok
        assert len(report.errors) >= 3

    def test_clean_setup_passes(self):
        report = check_full_setup(
            build_cnt(10, 0, length=6),
            calculation="scf",
            spectroscopy=phonon_setup(),
            pseudopotentials=NC,
        )
        assert report.ok, report.summary()
