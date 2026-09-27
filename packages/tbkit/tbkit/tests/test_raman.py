"""Non-resonant Raman from the tight-binding model (phase E, step 1).

Validation by symmetry first -- selection rules do not depend on the
parameters: diamond has one Raman-active triplet (T2g) with an off-diagonal
tensor and ρ = 3/4; C60 has exactly 10 active frequencies, 2 polarized (Ag)
and 8 depolarized (Hg). Then consistency (linear response = finite field =
sum over states) and experiment (diamond ε∞, C60 α), with the error stated.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk, graphene

from tbkit import System, solve, xu_carbon
from tbkit.optics import (
    dielectric_constant,
    polarizability,
    polarizability_finite,
    polarizability_linear_response,
    polarizability_screened,
)
from tbkit.raman import RamanResult, invariants, raman, spectrum
from tbkit.tests.test_energy import nonorthogonal_model, ring


@pytest.fixture(scope="module")
def diamond_raman():
    return raman(bulk("C", "diamond", a=3.555), xu_carbon(), kmesh=6, kT=0.01)


class TestDiamond:
    def test_one_raman_active_triplet(self, diamond_raman):
        groups = diamond_raman.groups()
        assert len(groups) == 1 and groups[0]["degeneracy"] == 3
        assert groups[0]["frequency_cm1"] == pytest.approx(1332, rel=0.10)   # exp. 1332

    def test_t2g_tensor_is_off_diagonal_and_depolarized(self, diamond_raman):
        strongest = diamond_raman.tensors[np.argmax(diamond_raman.activities)]
        assert np.abs(np.diag(strongest)).max() < 1e-3 * np.abs(strongest).max()
        assert diamond_raman.groups()[0]["depolarization"] == pytest.approx(0.75, abs=1e-6)

    def test_dielectric_constant(self):
        eps = dielectric_constant(System.build(bulk("C", "diamond", a=3.567), xu_carbon()),
                                  kmesh=8)
        assert np.allclose(eps, eps[0, 0] * np.eye(3), atol=1e-6)            # cubic
        # Independent particles, point dipoles: 4.75 against 5.7 measured.
        assert eps[0, 0] == pytest.approx(5.7, rel=0.20)


class TestPolarizability:
    @pytest.fixture(scope="class")
    @staticmethod
    def molecule_system():
        # C4N2: an even electron count (C5N, 25 electrons, is an open shell
        # with a half-filled level and no static polarizability).
        return System.build(ring(6, ["C", "C", "N", "C", "C", "N"], seed=8),
                            nonorthogonal_model(with_nitrogen=True))

    def test_linear_response_equals_finite_field(self, molecule_system):
        lr = polarizability_linear_response(molecule_system)
        ff = polarizability_screened(molecule_system, field=0.002)
        assert lr == pytest.approx(ff, rel=1e-4, abs=1e-4)

    def test_unscreened_equals_sum_over_states(self, molecule_system):
        lr = polarizability_linear_response(molecule_system, screened=False)
        sos = polarizability_finite(solve(molecule_system))
        assert lr == pytest.approx(sos, rel=1e-8, abs=1e-8)

    def test_screening_reduces_alpha(self, molecule_system):
        screened = np.trace(polarizability_linear_response(molecule_system))
        bare = np.trace(polarizability_linear_response(molecule_system, screened=False))
        assert 0 < screened < bare

    def test_alpha_is_symmetric_positive(self, molecule_system):
        alpha = polarizability_linear_response(molecule_system)
        assert alpha == pytest.approx(alpha.T)
        assert np.linalg.eigvalsh(alpha).min() > 0

    def test_semimetal_is_refused(self):
        sheet = graphene(a=2.46, vacuum=6.0)
        sheet.pbc = (True, True, False)
        with pytest.raises(ValueError, match="resonante"):
            polarizability(System.build(sheet, xu_carbon()), kmesh=12)

    def test_near_resonance_is_refused(self):
        system = System.build(bulk("C", "diamond", a=3.567), xu_carbon())
        with pytest.raises(ValueError, match="gap"):
            polarizability(system, kmesh=6, omega=10.0)


class TestTools:
    def test_invariants(self):
        assert invariants(np.eye(3)) == pytest.approx((1.0, 0.0))
        off = np.zeros((3, 3))
        off[0, 1] = off[1, 0] = 1.0
        assert invariants(off) == pytest.approx((0.0, 3.0))

    def test_spectrum_factors(self):
        result = RamanResult(np.array([200.0, 1600.0]), np.array([1.0, 1.0]),
                             np.array([0.75, 0.75]), np.zeros((2, 3, 3)), np.eye(3), "prueba",
                             np.array([200.0, 1600.0]))
        grid, bare = spectrum(result, laser_nm=None, temperature_k=None)
        _, measured = spectrum(result, laser_nm=532, temperature_k=300)
        low, high = np.argmin(abs(grid - 200)), np.argmin(abs(grid - 1600))
        assert bare[low] == pytest.approx(bare[high], rel=1e-3)
        # Bose and 1/ν favour the low-frequency line; (ν_L - ν)⁴ the other way.
        assert measured[low] / measured[high] > 1.0

    def test_raman_task_and_cli(self, tmp_path, capsys):
        from ase.io import write

        from tbkit.cli import main

        write(tmp_path / "diamante.extxyz", bulk("C", "diamond", a=3.555))
        assert main(["raman", str(tmp_path / "diamante.extxyz"), "--model", "sp3",
                     "--kmesh", "4", "-o", str(tmp_path / "raman.csv")]) == 0
        text = capsys.readouterr().out
        assert "Raman no resonante" in text and (tmp_path / "raman.csv").exists()


@pytest.mark.slow
def test_c60_selection_rules_and_polarizability():
    """C60 (Ih): Raman-active = 2 Ag (ρ = 0) + 8 Hg (ρ = 3/4, fivefold)."""
    from ase.build import molecule
    from ase.optimize import BFGS

    from tbkit.calculator import TBCalculator

    c60 = molecule("C60")
    c60.calc = TBCalculator(xu_carbon(), kT=0.01)
    BFGS(c60, logfile=None).run(fmax=0.005)
    alpha = polarizability_linear_response(System.build(c60, xu_carbon()))
    assert np.trace(alpha) / 3 == pytest.approx(76.5, rel=0.25)     # exp. 76.5 ± 8 Å³
    result = raman(c60, xu_carbon(), kT=0.01)
    # Hg(3) is weak (~5e-5 of the strongest line, measured at 710 cm⁻¹ and weak
    # there too); inactive modes carry finite-difference noise below ~3e-6.
    groups = result.groups(tolerance=2.0, threshold=1e-5)
    noise = [g for g in result.groups(tolerance=2.0, threshold=1e-9) if g not in groups]
    top = max(g["activity"] for g in groups)
    assert max(g["activity"] for g in noise) < 0.2 * min(g["activity"] for g in groups)
    assert min(g["activity"] for g in groups) > 1e-5 * top
    polarized = [g for g in groups if g["depolarization"] < 0.05]
    depolarized = [g for g in groups if abs(g["depolarization"] - 0.75) < 0.01]
    assert len(polarized) == 2 and all(g["degeneracy"] == 1 for g in polarized)
    assert len(depolarized) == 8 and all(g["degeneracy"] == 5 for g in depolarized)
    assert len(groups) == 10
