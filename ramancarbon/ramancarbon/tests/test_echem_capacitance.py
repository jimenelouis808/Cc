"""Capacitance measured three ways, Dunn's separation, and R_u from EIS."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.echem.capacitance import (
    capacitance_from_eis,
    compare,
    complex_capacitance,
    specific,
)
from ramancarbon.echem.curve import Electrode, Impedance
from ramancarbon.echem.cv import dunn_analysis
from ramancarbon.echem.eis import uncompensated_resistance, with_resistance_from
from ramancarbon.examples.demo_data import cv_rate_series


def _supercapacitor(series_r=2.0, charge_transfer=8.0, double_layer=5e-5,
                    tail=0.050, inductance=0.0) -> Impedance:
    """R0 − (R1|C1) − C2, which is what a supercapacitor looks like."""
    frequency = np.logspace(5, -2, 90)
    omega = 2 * np.pi * frequency
    z = (series_r + 1.0 / (1.0 / charge_transfer + 1j * omega * double_layer)
         + 1.0 / (1j * omega * tail) + 1j * omega * inductance)
    return Impedance(frequency, z,
                     electrode=Electrode(mass_mg=2.0, area_cm2=1.0))


def test_the_complex_capacitance_recovers_a_known_capacitor():
    analysis = complex_capacitance(_supercapacitor(tail=0.050))
    assert analysis.low_frequency_capacitance == pytest.approx(0.050, rel=0.01)


def test_the_relaxation_time_is_the_rc_of_the_device():
    """τ₀ at the maximum of C″ is (R_s + R_ct)·C for this circuit: 10 Ω
    times 50 mF is half a second."""
    analysis = complex_capacitance(_supercapacitor())
    assert analysis.relaxation_s == pytest.approx(0.5, rel=0.05)


def test_the_sign_convention_is_the_physical_one():
    """Z″ is stored negative for a capacitive response. Code written
    against a Nyquist plot's −Z″ returns a negative capacitance."""
    spectrum = _supercapacitor()
    assert np.all(spectrum.z.imag < 0)
    assert np.all(complex_capacitance(spectrum).real > 0)


def test_a_spectrum_that_never_reaches_the_capacitive_regime_says_so():
    frequency = np.logspace(5, 2, 40)          # stops at 100 Hz
    omega = 2 * np.pi * frequency
    z = 2.0 + 1.0 / (1.0 / 8.0 + 1j * omega * 5e-5) + 1.0 / (1j * omega * 0.05)
    analysis = complex_capacitance(Impedance(frequency, z))
    assert any("límite INFERIOR" in text for text in analysis.warnings)


def test_the_frequency_is_part_of_the_number():
    """Two laboratories disagree about the same material because one
    measured to 10 mHz and the other stopped at 100."""
    entry = capacitance_from_eis(_supercapacitor())
    assert "Hz" in entry.condition
    assert entry.per_gram == pytest.approx(25.0, rel=0.02)


def test_the_methods_are_compared_rather_than_averaged():
    electrode = Electrode(mass_mg=2.0)
    entries = [
        specific(0.100, electrode, "CV", "20 mV/s"),
        specific(0.050, electrode, "GCD", "1 mA"),
        specific(0.030, electrode, "EIS", "C′ a 0.01 Hz"),
    ]
    result = compare(entries)
    assert result.spread == pytest.approx(0.70, abs=0.01)
    assert any("difieren" in text for text in result.warnings)
    assert any("GCD" in text for text in result.warnings)


def test_the_uncompensated_resistance_comes_off_the_real_axis_crossing():
    spectrum = _supercapacitor(series_r=2.0, inductance=1e-6)
    value, how = uncompensated_resistance(spectrum)
    assert value == pytest.approx(2.0, abs=0.05)
    assert "cruce" in how


def test_a_purely_capacitive_spectrum_gives_an_upper_bound_and_says_so():
    """It never crosses the real axis, and refusing outright would be worse
    than an explicit bound."""
    value, how = uncompensated_resistance(_supercapacitor())
    assert value == pytest.approx(2.0, abs=0.05)
    assert "COTA SUPERIOR" in how


def test_the_resistance_travels_to_the_electrode():
    spectrum = _supercapacitor()
    electrode, _ = with_resistance_from(spectrum.electrode, spectrum)
    assert electrode.resistance_ohm == pytest.approx(2.0, abs=0.05)
    assert spectrum.electrode.resistance_ohm is None    # the original is left alone


def test_dunn_needs_three_rates_for_two_coefficients():
    curves = cv_rate_series("condensador")[:2]
    with pytest.raises(Exception, match="dos coeficientes"):
        dunn_analysis(curves)


def test_dunn_calls_a_capacitor_capacitive_at_every_rate():
    analysis = dunn_analysis(cv_rate_series("condensador"))
    assert min(analysis.fractions.values()) > 0.95
    assert analysis.warnings == []                      # nothing to complain of


def test_dunn_sees_a_battery_become_more_capacitive_as_it_speeds_up():
    """Diffusion has less time at a fast scan, so its share falls. A
    capacitive fraction that does NOT rise with scan rate means the
    separation is not describing what it claims to."""
    analysis = dunn_analysis(cv_rate_series("bateria"))
    rates = sorted(analysis.fractions)
    shares = [analysis.fractions[rate] for rate in rates]
    assert shares[0] < 0.7
    assert shares[-1] > shares[0] + 0.2
    assert all(b >= a - 0.02 for a, b in zip(shares, shares[1:]))


def test_the_reconstructed_capacitive_curve_is_part_of_the_total():
    analysis = dunn_analysis(cv_rate_series("bateria"))
    rate = sorted(analysis.fractions)[-1]
    capacitive = analysis.capacitive_current(rate)
    diffusive = analysis.diffusive_current(rate)
    assert capacitive.shape == analysis.potentials.shape
    assert np.mean(np.abs(capacitive)) > np.mean(np.abs(diffusive))


def test_the_electrode_carries_its_electrolyte():
    electrode = Electrode(mass_mg=2.0, electrolyte="KOH 6 M")
    assert "KOH 6 M" in electrode.describe()
