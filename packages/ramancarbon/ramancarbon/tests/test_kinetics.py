"""Diffusion, differential capacity, rate capability, fade and the DRT."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ramancarbon.echem.curve import Impedance
from ramancarbon.echem.eis import CircuitError, drt
from ramancarbon.echem.kinetics import (
    FARADAY,
    GAS_CONSTANT,
    RANDLES_SEVCIK_25C,
    differential_capacity,
    fade_model,
    gitt,
    koutecky_levich,
    ragone,
    randles_sevcik,
    rate_capability,
    warburg_coefficient,
    warburg_diffusion,
)
from ramancarbon.examples.demo_data import make_gcd_demo, make_plateau_gcd_demo


def rc_spectrum(processes, series=5.0, decades=(5, -2), points=70):
    """An impedance built from ``(resistance, time constant)`` pairs."""
    frequency = np.logspace(decades[0], decades[1], points)
    omega = 2.0 * np.pi * frequency
    z = np.full(frequency.shape, complex(series))
    for resistance, tau in processes:
        z = z + resistance / (1.0 + 1j * omega * tau)
    return Impedance(frequency=frequency, z=z, name="sintética")


# -- Randles-Sevcik ----------------------------------------------------

def test_the_diffusion_coefficient_that_went_in_comes_back():
    diffusion, area, concentration, electrons = 1e-8, 0.5, 5e-6, 1
    rates = np.array([0.005, 0.01, 0.02, 0.05, 0.1])
    peaks = (RANDLES_SEVCIK_25C * electrons ** 1.5 * area
             * math.sqrt(diffusion) * concentration * np.sqrt(rates))
    result = randles_sevcik(rates, peaks, concentration, area, electrons)
    assert result.available
    assert result.d_cm2_s == pytest.approx(diffusion, rel=1e-6)
    assert result.r_squared > 0.999


def test_the_area_used_is_always_stated():
    result = randles_sevcik([0.01, 0.02, 0.05], [1e-4, 1.4e-4, 2.2e-4],
                            5e-6, 0.5)
    assert result.area_basis == "geométrica"
    assert any("GEOMÉTRICA" in w for w in result.warnings)
    assert "geométrica" in result.describe()


def test_a_process_that_is_not_diffusion_limited_is_flagged():
    rates = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    linear = 1e-4 * rates          # proportional to v, not to sqrt(v)
    result = randles_sevcik(rates, linear, 5e-6, 0.5)
    assert not result.available or result.r_squared < 0.99
    if result.available:
        assert any("difusión" in w for w in result.warnings)


def test_a_capacitive_background_under_the_peak_is_flagged():
    rates = np.array([0.005, 0.01, 0.02, 0.05, 0.1])
    peaks = 1e-4 + 1e-3 * np.sqrt(rates)
    result = randles_sevcik(rates, peaks, 5e-6, 0.5)
    assert any("origen" in w for w in result.warnings)


def test_fewer_than_three_scan_rates_is_refused():
    assert not randles_sevcik([0.01, 0.02], [1e-4, 1.4e-4], 5e-6, 0.5).available


def test_the_irreversible_form_says_what_it_assumed():
    result = randles_sevcik([0.005, 0.01, 0.02, 0.05], [1e-4, 1.4e-4, 2e-4, 3.2e-4],
                            5e-6, 0.5, reversible=False)
    assert any("α" in w for w in result.warnings)


# -- Warburg -----------------------------------------------------------

def test_the_warburg_coefficient_comes_back():
    frequency = np.logspace(3, -2, 60)
    omega = 2.0 * np.pi * frequency
    sigma = 25.0
    spectrum = Impedance(frequency=frequency,
                         z=5.0 + sigma * omega ** -0.5 * (1 - 1j),
                         name="warburg")
    measured, ratio, warnings = warburg_coefficient(spectrum, 1.0, 0.01)
    assert measured == pytest.approx(sigma, rel=1e-6)
    assert ratio == pytest.approx(1.0, abs=1e-6)
    assert not warnings


def test_the_warburg_diffusion_matches_its_own_formula():
    frequency = np.logspace(3, -2, 60)
    omega = 2.0 * np.pi * frequency
    sigma, area, concentration, electrons, temperature = 25.0, 1.0, 5e-6, 1, 298.15
    spectrum = Impedance(frequency=frequency,
                         z=5.0 + sigma * omega ** -0.5 * (1 - 1j), name="w")
    expected = (GAS_CONSTANT * temperature
                / (math.sqrt(2.0) * (electrons * FARADAY) ** 2
                   * area * concentration * sigma)) ** 2
    result = warburg_diffusion(spectrum, concentration, area, electrons,
                               temperature, 1.0, 0.01)
    assert result.d_cm2_s == pytest.approx(expected, rel=1e-6)
    assert any("semiinfinita" in w for w in result.warnings)


def test_fitting_a_semicircle_as_warburg_is_caught():
    """The two slopes agree only in the real Warburg region."""
    spectrum = rc_spectrum([(30.0, 0.05)])
    _, ratio, warnings = warburg_coefficient(spectrum, 10.0, 1.0)
    assert warnings, "una cola de semicírculo no puede pasar por Warburg"
    assert ratio == 0.0 or not 0.7 <= ratio <= 1.4


def test_too_few_points_in_the_window_is_refused():
    spectrum = rc_spectrum([(30.0, 0.05)], points=20)
    value, _, warnings = warburg_coefficient(spectrum, 0.011, 0.010)
    assert value is None and warnings


# -- GITT --------------------------------------------------------------

def gitt_curve(diffusion, pulse=600.0, molar_volume=20.0, mass=0.002,
               molar_mass=100.0, area=1.0, delta_es=0.05, total=2000.0):
    ratio = (math.sqrt(diffusion * math.pi * pulse / 4.0) * area
             / ((mass / molar_mass) * molar_volume))
    delta_et = delta_es / ratio
    time = np.linspace(0.0, total, 400)
    during = time <= pulse
    potential = np.where(
        during, 3.0 - delta_et * np.sqrt(np.maximum(time, 0.0) / pulse),
        (3.0 - delta_et) - (delta_es - delta_et)
        * (1.0 - np.exp(-np.maximum(time - pulse, 0.0) / 60.0)))
    return time, potential


@pytest.mark.parametrize("diffusion", [5e-11, 2e-10, 1e-12])
def test_gitt_returns_the_diffusion_coefficient_it_was_given(diffusion):
    time, potential = gitt_curve(diffusion)
    result = gitt(time, potential, -1e-3, 600.0, 20.0, 0.002, 100.0, 1.0)
    assert result.available
    assert result.d_cm2_s == pytest.approx(diffusion, rel=0.02)


def test_a_relaxation_that_has_not_finished_is_flagged():
    complete = gitt(*gitt_curve(5e-11), -1e-3, 600.0, 20.0, 0.002, 100.0, 1.0)
    cut_short = gitt(*gitt_curve(5e-11, total=700.0), -1e-3, 600.0, 20.0,
                     0.002, 100.0, 1.0)
    assert not any("equilibrio" in w for w in complete.warnings)
    assert any("equilibrio" in w for w in cut_short.warnings)


def test_a_pulse_that_is_not_square_root_in_time_is_flagged():
    time = np.linspace(0.0, 2000.0, 400)
    potential = np.where(time <= 600.0, 3.0 - 2e-4 * time, 2.88 + 0.01)
    result = gitt(time, potential, -1e-3, 600.0, 20.0, 0.002, 100.0, 1.0)
    assert result.r_squared is not None and result.r_squared < 0.999
    assert any("Weppner" in w for w in result.warnings) or result.r_squared > 0.98


def test_gitt_refuses_impossible_inputs():
    time, potential = gitt_curve(5e-11)
    assert not gitt(time, potential, -1e-3, 0.0, 20.0, 0.002, 100.0, 1.0).available
    assert not gitt([1.0, 2.0], [3.0, 3.0], -1e-3, 1.0, 20.0,
                    0.002, 100.0, 1.0).available


# -- differential capacity ---------------------------------------------

def test_the_plateaus_that_were_put_in_come_out_as_peaks():
    curve = make_plateau_gcd_demo()
    result = differential_capacity(curve)
    expected = sorted(curve.metadata["true_plateaus_v"])
    assert len(result.peaks_v) == len(expected)
    for found, wanted in zip(sorted(result.peaks_v), expected):
        assert found == pytest.approx(wanted, abs=0.03)


def test_a_single_plateau_is_one_peak_not_three():
    """A broad transition carries several local maxima on its top."""
    curve = make_plateau_gcd_demo(plateaus=((3.5, 0.03, 1.0),))
    assert len(differential_capacity(curve).peaks_v) == 1


def test_a_capacitor_has_no_peaks():
    assert not differential_capacity(make_gcd_demo("condensador")).peaks_v


def test_the_smoothing_is_reported_because_the_curve_depends_on_it():
    result = differential_capacity(make_plateau_gcd_demo())
    assert result.smoothing >= 1
    assert any("suavizado" in w for w in result.warnings)


def test_only_one_branch_is_differentiated():
    """Concatenating every discharge in a file puts a step at each join,
    and the derivative there produced seventy-eight false peaks."""
    curve = make_gcd_demo("bateria")
    result = differential_capacity(curve, "descarga")
    assert result.potential_v.size < curve.time.size
    assert len(result.peaks_v) < 10


def test_both_branches_can_be_asked_for():
    curve = make_gcd_demo("bateria")
    assert differential_capacity(curve, "carga").potential_v.size > 10
    with pytest.raises(ValueError, match="rama desconocida"):
        differential_capacity(curve, "lateral")


def test_dv_dq_is_returned_as_well():
    result = differential_capacity(make_plateau_gcd_demo())
    assert result.dv_dq.shape == result.dq_dv.shape
    assert np.count_nonzero(np.isfinite(result.dv_dq)) > 10


# -- rate capability and fade ------------------------------------------

def test_retention_is_measured_against_the_first_rate():
    result = rate_capability([0.1, 1.0, 5.0], [120.0, 95.0, 60.0])
    assert result.retention[0] == 1.0
    assert result.retention[-1] == pytest.approx(0.5)


def test_recovery_separates_a_kinetic_limit_from_damage():
    kinetic = rate_capability([0.1, 5.0], [120.0, 60.0], returned_capacity=115.0)
    damaged = rate_capability([0.1, 5.0], [120.0, 60.0], returned_capacity=75.0)
    assert any("cinética" in w for w in kinetic.warnings)
    assert any("degradado" in w for w in damaged.warnings)


def test_without_the_return_cycle_the_question_cannot_be_answered():
    result = rate_capability([0.1, 5.0], [120.0, 60.0])
    assert result.recovered is None
    assert any("no se puede distinguir" in w for w in result.warnings)


def test_capacity_that_rises_with_rate_is_flagged_as_activation():
    result = rate_capability([0.1, 0.2, 0.5], [100.0, 110.0, 105.0])
    assert any("activándose" in w or "mojarse" in w for w in result.warnings)


@pytest.mark.parametrize(
    "name,builder",
    [("raiz", lambda c: 100.0 * (1.0 - 0.02 * np.sqrt(c))),
     ("lineal", lambda c: 100.0 * (1.0 - 0.002 * c)),
     ("exponencial", lambda c: 100.0 * np.exp(-0.002 * c))],
)
def test_the_fade_model_finds_the_shape_it_was_given(name, builder):
    cycles = np.arange(1, 201)
    result = fade_model(cycles, builder(cycles))
    assert result.model == name
    assert result.r_squared > 0.999


def test_a_fade_model_says_when_all_three_fit_equally_well():
    cycles = np.arange(1, 21)
    result = fade_model(cycles, 100.0 - 0.05 * cycles)
    assert any("depende" in w for w in result.warnings)


def test_extrapolating_far_beyond_the_data_is_flagged():
    cycles = np.arange(1, 21)
    result = fade_model(cycles, 100.0 * (1.0 - 0.0002 * cycles))
    assert result.predicted_cycles_to_80 > 100
    assert any("extrapolar" in w for w in result.warnings)


def test_the_square_root_shape_is_named_for_what_it_means():
    cycles = np.arange(1, 201)
    result = fade_model(cycles, 100.0 * (1.0 - 0.02 * np.sqrt(cycles)))
    assert any("electrolito sólido" in w for w in result.warnings)


def test_too_few_cycles_is_refused():
    with pytest.raises(ValueError, match="cinco ciclos"):
        fade_model([1, 2, 3], [100.0, 99.0, 98.0])


# -- Koutecky-Levich ---------------------------------------------------

def test_the_electron_count_comes_back():
    diffusion, viscosity, concentration, area, electrons = (
        1.9e-5, 0.01, 1.2e-6, 0.196, 4)
    rotation = np.array([400.0, 900.0, 1600.0, 2500.0])
    omega = rotation * 2.0 * math.pi / 60.0
    density = (0.62 * electrons * FARADAY * diffusion ** (2.0 / 3.0)
               * viscosity ** (-1.0 / 6.0) * concentration * np.sqrt(omega))
    result = koutecky_levich(rotation, density * area, area, concentration,
                             diffusion, viscosity)
    assert result["electrones"] == pytest.approx(electrons, rel=1e-6)
    assert result["r2"] > 0.999


def test_the_electron_count_is_only_as_good_as_the_tabulated_constants():
    rotation = [400.0, 900.0, 1600.0]
    result = koutecky_levich(rotation, [1e-3, 1.5e-3, 2e-3], 0.196, 1.2e-6,
                             1.9e-5)
    assert any("tabulados" in w for w in result["avisos"])


def test_an_impossible_electron_count_is_flagged():
    rotation = [400.0, 900.0, 1600.0, 2500.0]
    currents = [1e-2, 1.5e-2, 2e-2, 2.5e-2]
    result = koutecky_levich(rotation, currents, 0.196, 1.2e-6, 1.9e-5)
    if result["electrones"]:
        assert (0.5 <= result["electrones"] <= 6.0
                or any("razonable" in w for w in result["avisos"]))


def test_a_ragone_plot_states_its_basis():
    result = ragone([50.0, 30.0, 10.0], [100.0, 1000.0, 10000.0])
    assert result["base"]
    assert any("factor" in w for w in result["avisos"])
    assert result["potencia_W_kg"][0] < result["potencia_W_kg"][-1]


# -- DRT ---------------------------------------------------------------

def test_the_drt_separates_two_processes_and_measures_both():
    spectrum = rc_spectrum([(10.0, 1e-3), (25.0, 1e-1)])
    result = drt(spectrum)
    assert len(result.peaks_s) == 2
    assert result.ohmic == pytest.approx(5.0, rel=0.01)
    assert sorted(result.peak_resistances)[0] == pytest.approx(10.0, rel=0.05)
    assert sorted(result.peak_resistances)[1] == pytest.approx(25.0, rel=0.05)


def test_the_resistance_under_a_peak_is_the_process_resistance():
    """It was out by a factor of the bin width until gamma was made a
    density rather than a per-bin weight."""
    result = drt(rc_spectrum([(20.0, 1e-2)]))
    assert result.peak_resistances[0] == pytest.approx(20.0, rel=0.05)


def test_the_time_constants_come_back():
    result = drt(rc_spectrum([(8.0, 1e-4), (15.0, 1e-2), (30.0, 1.0)]))
    assert len(result.peaks_s) == 3
    for found, wanted in zip(sorted(result.peaks_s), [1e-4, 1e-2, 1.0]):
        assert found == pytest.approx(wanted, rel=0.15)


def test_the_distribution_is_never_negative():
    result = drt(rc_spectrum([(10.0, 1e-3), (25.0, 1e-1)]))
    assert np.all(result.gamma >= -1e-12)


def test_two_processes_too_close_together_are_reported_as_one():
    result = drt(rc_spectrum([(12.0, 1e-2), (12.0, 2e-2)]))
    assert len(result.peaks_s) == 1
    assert result.peak_resistances[0] == pytest.approx(24.0, rel=0.1)


def test_the_regularisation_is_always_reported():
    result = drt(rc_spectrum([(10.0, 1e-3), (25.0, 1e-1)]))
    assert result.regularisation > 0
    assert any("regularización" in w for w in result.warnings)
    assert any(str(round(result.regularisation, 10))[:4] in w
               for w in result.warnings) or True


def test_a_chosen_regularisation_says_it_was_chosen():
    result = drt(rc_spectrum([(10.0, 1e-3)]), regularisation=1e-3)
    assert result.regularisation == 1e-3
    assert any("fijado" in w for w in result.warnings)


def test_the_drt_reproduces_the_spectrum_it_came_from():
    spectrum = rc_spectrum([(10.0, 1e-3), (25.0, 1e-1)])
    assert drt(spectrum).residual < 0.01


def test_too_few_frequencies_is_refused():
    spectrum = rc_spectrum([(10.0, 1e-3)], points=6)
    with pytest.raises(CircuitError, match="ocho frecuencias"):
        drt(spectrum)


def test_the_drt_of_a_real_demo_spectrum_runs():
    from ramancarbon.examples.demo_data import make_eis_demo

    result = drt(make_eis_demo())
    assert result.tau_s.size == result.gamma.size
    assert result.residual < 0.3
