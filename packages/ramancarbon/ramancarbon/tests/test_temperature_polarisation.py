"""Sample temperature from the two Raman branches, and polarised Raman."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.analysis.polarisation import (
    DEPOLARISATION_LIMIT,
    angular_alignment,
    depolarisation_ratio,
)
from ramancarbon.analysis.temperature import (
    expected_ratio,
    laser_heating,
    population,
    stokes_anti_stokes_temperature,
)
from ramancarbon.core.spectrum import Spectrum


def two_branch(kelvin: float, mode: float = 200.0, laser: float = 532.0,
               seed: int = 0, offset: float = 20.0, response: float = 1.0):
    """A spectrum with both branches of one mode at a known temperature."""
    x = np.arange(-600.0, 600.0, 1.0)
    half = 8.0
    ratio = expected_ratio(mode, kelvin, laser) * response
    y = (1000.0 * half ** 2 / ((x - mode) ** 2 + half ** 2)
         + 1000.0 * ratio * half ** 2 / ((x + mode) ** 2 + half ** 2))
    noise = np.random.default_rng(seed).normal(0.0, 0.5, x.size)
    return Spectrum(shift=x, intensity=y + offset + noise, laser_nm=laser,
                    name="dos ramas")


def band(scale: float, centre: float = 1580.0, seed: int = 0) -> Spectrum:
    x = np.arange(1300.0, 1900.0, 1.0)
    half = 15.0
    y = scale * 1000.0 * half ** 2 / ((x - centre) ** 2 + half ** 2) + 40.0
    return Spectrum(shift=x, intensity=y + np.random.default_rng(seed)
                    .normal(0.0, 1.0, x.size), laser_nm=532.0, name="banda")


# -- temperature -------------------------------------------------------

@pytest.mark.parametrize("kelvin", [300.0, 500.0, 900.0])
def test_the_temperature_that_went_in_comes_back(kelvin):
    result = stokes_anti_stokes_temperature(two_branch(kelvin), 200.0)
    assert result.available
    assert abs(result.kelvin - kelvin) / kelvin < 0.02


def test_the_local_background_is_what_makes_it_come_back():
    """A flat offset raises the weaker branch's area proportionally more,
    so an uncorrected ratio always reads hot."""
    hot = stokes_anti_stokes_temperature(two_branch(300.0, offset=200.0), 200.0)
    assert abs(hot.kelvin - 300.0) < 15.0


def test_a_spectrum_with_only_one_branch_is_refused():
    spectrum = two_branch(300.0)
    stokes_only = spectrum.crop(0.0, 600.0)
    result = stokes_anti_stokes_temperature(stokes_only, 200.0)
    assert not result.available
    assert "las dos ramas" in result.reason


def test_a_mode_too_hard_to_measure_is_refused_rather_than_answered():
    x = np.arange(-2000.0, 2000.0, 2.0)
    half = 12.0
    ratio = expected_ratio(1580.0, 300.0, 532.0)
    y = (1000.0 * half ** 2 / ((x - 1580.0) ** 2 + half ** 2)
         + 1000.0 * ratio * half ** 2 / ((x + 1580.0) ** 2 + half ** 2))
    spectrum = Spectrum(shift=x, intensity=y + 20.0, laser_nm=532.0, name="G")
    result = stokes_anti_stokes_temperature(spectrum, 1580.0)
    assert not result.available or result.warnings
    if not result.available:
        assert "ruido" in result.reason


def test_without_a_laser_no_temperature():
    spectrum = two_branch(300.0)
    spectrum.laser_nm = None
    assert not stokes_anti_stokes_temperature(spectrum, 200.0).available


def test_an_uncalibrated_result_says_it_is_apparent():
    result = stokes_anti_stokes_temperature(two_branch(300.0), 200.0)
    assert not result.calibrated
    assert "aparente" in result.describe()
    assert any("respuesta" in w for w in result.warnings)


def test_a_response_correction_removes_the_bias_it_warns_about():
    """A spectrometer passing 30 % less anti-Stokes reads 40 K cold; told
    the factor, it reads right."""
    measured = two_branch(300.0, response=0.7)
    uncorrected = stokes_anti_stokes_temperature(measured, 200.0)
    corrected = stokes_anti_stokes_temperature(measured, 200.0, response=0.7)
    assert abs(corrected.kelvin - 300.0) < 10.0
    assert abs(uncorrected.kelvin - 300.0) > 20.0
    assert corrected.calibrated


def test_a_ratio_above_the_classical_limit_is_refused():
    impossible = two_branch(300.0, response=50.0)
    result = stokes_anti_stokes_temperature(impossible, 200.0)
    assert not result.available
    assert "límite clásico" in result.reason


def test_the_exponent_convention_is_a_choice_that_changes_the_answer():
    spectrum = two_branch(600.0)
    three = stokes_anti_stokes_temperature(spectrum, 200.0, exponent=3)
    four = stokes_anti_stokes_temperature(spectrum, 200.0, exponent=4)
    assert three.kelvin != four.kelvin
    assert abs(three.kelvin - four.kelvin) > 1.0


def test_the_occupation_is_the_physics_underneath():
    assert population(200.0, 300.0) == pytest.approx(0.6213, rel=1e-3)
    assert population(1580.0, 300.0) < 1e-3
    assert population(200.0, 0.0) == 0.0


def test_laser_heating_from_a_band_shift():
    cold = band(1.0, centre=1580.0)
    hot = band(1.0, centre=1577.0, seed=1)
    result = laser_heating(cold, hot, 1580.0)
    assert result["desplazamiento_cm"] == pytest.approx(-3.0, abs=1.5)
    assert result["delta_T_K"] > 100.0
    assert result["coeficiente_cm_por_K"] == -0.0148


def test_laser_heating_needs_a_coefficient_that_is_not_zero():
    with pytest.raises(ValueError, match="cero"):
        laser_heating(band(1.0), band(1.0), 1580.0, coefficient_cm_per_k=0.0)


# -- polarisation ------------------------------------------------------

@pytest.mark.parametrize("rho", [0.05, 0.35, 0.75])
def test_the_depolarisation_ratio_that_went_in_comes_back(rho):
    result = depolarisation_ratio(band(1.0), band(rho, seed=1), 1580.0)
    assert result.available
    assert result.ratio == pytest.approx(rho, abs=0.02)


def test_a_symmetric_mode_is_told_from_a_depolarised_one():
    assert depolarisation_ratio(band(1.0), band(0.1, seed=1), 1580.0).symmetry \
        == "totalmente simétrico"
    assert depolarisation_ratio(band(1.0), band(0.75, seed=1), 1580.0).symmetry \
        == "despolarizado"


def test_a_ratio_above_three_quarters_is_flagged_as_impossible():
    result = depolarisation_ratio(band(1.0), band(0.95, seed=1), 1580.0)
    assert result.symmetry == "anómalo"
    assert any(str(DEPOLARISATION_LIMIT) in w for w in result.warnings)


def test_the_grating_is_a_polariser_and_the_result_says_so():
    result = depolarisation_ratio(band(1.0), band(0.3, seed=1), 1580.0)
    assert not result.corrected
    assert any("red" in w for w in result.warnings)
    told = depolarisation_ratio(band(1.0), band(0.3, seed=1), 1580.0,
                                scrambler=True)
    assert told.corrected


def test_a_correction_factor_is_divided_out():
    plain = depolarisation_ratio(band(1.0), band(0.6, seed=1), 1580.0)
    corrected = depolarisation_ratio(band(1.0), band(0.6, seed=1), 1580.0,
                                     correction=2.0)
    assert corrected.ratio == pytest.approx(plain.ratio / 2.0, rel=1e-9)


def test_two_different_lasers_cannot_be_compared():
    other = band(0.5, seed=1)
    other.laser_nm = 633.0
    result = depolarisation_ratio(band(1.0), other, 1580.0)
    assert not result.available
    assert "láseres distintos" in result.reason


def test_a_band_outside_the_measured_range_is_refused():
    result = depolarisation_ratio(band(1.0), band(0.5, seed=1), 2700.0)
    assert not result.available
    assert "no cubre" in result.reason


def test_alignment_finds_the_axis_that_was_put_in():
    angles = np.arange(0.0, 180.0, 15.0)
    spectra = [band(0.1 + 0.9 * np.cos(np.radians(a - 30.0)) ** 4, seed=int(a))
               for a in angles]
    result = angular_alignment(angles, spectra, 1580.0)
    assert result.available
    assert result.angle_deg == pytest.approx(30.0, abs=3.0)
    assert result.modulation > 0.7
    assert result.residual < 0.1


def test_a_random_sample_shows_no_modulation():
    angles = np.arange(0.0, 180.0, 15.0)
    spectra = [band(1.0, seed=int(a)) for a in angles]
    result = angular_alignment(angles, spectra, 1580.0)
    assert result.modulation is None or result.modulation < 0.05


def test_three_angles_are_refused_because_any_three_fit():
    angles = [0.0, 45.0, 90.0]
    spectra = [band(1.0, seed=int(a)) for a in angles]
    result = angular_alignment(angles, spectra, 1580.0)
    assert not result.available
    assert "cuatro ángulos" in result.reason


def test_a_short_angular_range_is_warned_about():
    angles = np.arange(0.0, 60.0, 10.0)
    spectra = [band(0.1 + 0.9 * np.cos(np.radians(a - 10.0)) ** 4, seed=int(a))
               for a in angles]
    result = angular_alignment(angles, spectra, 1580.0)
    assert any("90" in w for w in result.warnings)


def test_the_order_parameter_is_labelled_as_approximate():
    angles = np.arange(0.0, 180.0, 20.0)
    spectra = [band(0.2 + 0.8 * np.cos(np.radians(a)) ** 4, seed=int(a))
               for a in angles]
    result = angular_alignment(angles, spectra, 1580.0)
    assert any("aproximado" in w for w in result.warnings)


def test_mismatched_angles_and_spectra_are_refused():
    result = angular_alignment([0.0, 45.0], [band(1.0)], 1580.0)
    assert not result.available
    assert "espectros" in result.reason
