"""The least-squares engine: recovery, bounds, warnings, model selection."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.spectrum import Spectrum
from ramancarbon.models.deconvolution import build_model, compare_models
from ramancarbon.models.fitting import FitModel, PeakSpec, fit_model
from ramancarbon.models.lineshapes import bwf, gaussian, lorentzian


def _four_band(seed=7, noise=4.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(900.0, 1900.0, 1001)
    truth = (
        lorentzian(x, 1348.0, 420.0, 45.0)
        + gaussian(x, 1500.0, 90.0, 180.0)
        + lorentzian(x, 1583.0, 900.0, 28.0)
        + lorentzian(x, 1618.0, 140.0, 20.0)
        + 2.0
        + 0.004 * (x - 1400.0)
    )
    return Spectrum(x, truth + rng.normal(0.0, noise, x.size), laser_nm=532.0)


def test_recovers_known_parameters():
    spectrum = _four_band()
    model = FitModel(
        window=(1000.0, 1800.0),
        background="linear",
        peaks=[
            PeakSpec("D", "lorentzian", 1350.0, 400.0, 50.0, band="D"),
            PeakSpec("D3", "gaussian", 1500.0, 80.0, 180.0,
                     fwhm_bounds=(80.0, 320.0), band="D3"),
            PeakSpec("G", "lorentzian", 1580.0, 800.0, 30.0, band="G"),
            PeakSpec("Dp", "lorentzian", 1620.0, 120.0, 22.0, band="D'"),
        ],
    )
    result = fit_model(spectrum, model)
    assert result.success
    assert result.r_squared > 0.99
    expected = {"D": (1348.0, 45.0), "D3": (1500.0, 180.0),
                "G": (1583.0, 28.0), "Dp": (1618.0, 20.0)}
    for name, (centre, fwhm) in expected.items():
        peak = result.peak(name)
        assert peak.centre == pytest.approx(centre, abs=2.0)
        assert peak.fwhm == pytest.approx(fwhm, rel=0.1)


def test_recovers_a_breit_wigner_fano_g_minus():
    rng = np.random.default_rng(4)
    x = np.linspace(1350.0, 1750.0, 401)
    y = (bwf(x, 1540.0, 500.0, 90.0, -0.22) + lorentzian(x, 1590.0, 800.0, 20.0)
         + rng.normal(0.0, 4.0, x.size))
    spectrum = Spectrum(x, y, laser_nm=532.0)
    model = FitModel(
        window=(1360.0, 1740.0),
        background="constant",
        peaks=[
            PeakSpec("G-", "bwf", 1545.0, 450.0, 80.0,
                     fwhm_bounds=(30.0, 200.0), band="G-"),
            PeakSpec("G+", "lorentzian", 1590.0, 800.0, 22.0, band="G+"),
        ],
    )
    result = fit_model(spectrum, model)
    g_minus = result.peak("G-")
    assert g_minus.extra[0] == pytest.approx(-0.22, abs=0.03)
    assert g_minus.centre == pytest.approx(1540.0, abs=3.0)
    # The reported maximum is displaced from omega_0 by Gamma/q.
    assert g_minus.peak_position < g_minus.centre


def test_uncertainties_are_produced_and_are_small_for_a_good_fit():
    result = fit_model(_four_band(), build_model(_four_band(), "four_band"))
    for peak in result.peaks:
        assert "centre" in peak.errors
        assert peak.errors["centre"] < 12.0


def test_warns_when_a_parameter_hits_its_bound():
    spectrum = _four_band()
    model = FitModel(
        window=(1000.0, 1800.0),
        background="linear",
        peaks=[PeakSpec("G", "lorentzian", 1583.0, 900.0, 28.0,
                        fwhm_bounds=(5.0, 8.0), band="G")],
    )
    result = fit_model(spectrum, model)
    assert any("pegado a su límite" in w for w in result.warnings)


def test_warns_that_smoothing_makes_errors_optimistic():
    from ramancarbon.core.preprocess import smooth

    spectrum = smooth(_four_band(), window=9)
    result = fit_model(spectrum, build_model(spectrum, "three_band"))
    assert any("suavizado" in w for w in result.warnings)


def test_underdetermined_model_is_refused():
    x = np.linspace(1500.0, 1520.0, 8)
    spectrum = Spectrum(x, np.ones(8), laser_nm=532.0)
    model = FitModel(
        window=(1500.0, 1520.0),
        background="quadratic",
        peaks=[PeakSpec(f"P{i}", "lorentzian", 1505.0 + i, 1.0, 5.0) for i in range(4)],
    )
    with pytest.raises(ValueError, match="underdetermined"):
        fit_model(spectrum, model)


def test_model_out_of_window_is_refused():
    with pytest.raises(ValueError, match="outside the fit window"):
        FitModel(window=(1000.0, 1200.0),
                 peaks=[PeakSpec("G", "lorentzian", 1583.0, 1.0, 20.0)])


def test_model_selection_picks_the_generating_model():
    """BIC must choose the four-band model for four-band data, not the
    five-band one that always fits at least as well."""
    comparison = compare_models(_four_band())
    assert comparison.best == "four_band"
    assert comparison.results["five_band"].r_squared >= \
        comparison.results["four_band"].r_squared - 1e-9


def test_information_criteria_penalise_parameters():
    comparison = compare_models(_four_band())
    two, five = comparison.results["two_band"], comparison.results["five_band"]
    assert five.n_parameters > two.n_parameters
    assert five.r_squared > two.r_squared


def test_bwf_component_reports_both_position_conventions():
    x = np.linspace(1400.0, 1700.0, 601)
    y = bwf(x, 1560.0, 300.0, 80.0, -0.2) + 1.0
    spectrum = Spectrum(x, y, laser_nm=532.0)
    model = FitModel(
        window=(1410.0, 1690.0),
        background="constant",
        peaks=[PeakSpec("G-", "bwf", 1560.0, 300.0, 80.0, band="G-")],
    )
    peak = fit_model(spectrum, model).peak("G-")
    assert peak.peak_position != peak.centre
    assert peak.peak_height > peak.height
