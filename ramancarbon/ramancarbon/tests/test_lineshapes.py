"""Profiles, their analytic areas, and the BWF closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.models.lineshapes import (
    bwf,
    bwf_area,
    bwf_peak_height,
    bwf_peak_position,
    gaussian,
    gaussian_area,
    lorentzian,
    lorentzian_area,
    pseudo_voigt,
    pseudo_voigt_area,
    resolve_profile,
)

X = np.linspace(1000.0, 2200.0, 120001)


def _numeric_fwhm(y, x):
    above = x[y >= 0.5 * y.max()]
    return float(above[-1] - above[0])


@pytest.mark.parametrize("profile", [lorentzian, gaussian])
def test_height_and_fwhm_mean_what_they_say(profile):
    y = profile(X, 1580.0, 7.0, 25.0)
    assert y.max() == pytest.approx(7.0)
    assert _numeric_fwhm(y, X) == pytest.approx(25.0, rel=1e-3)


@pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_pseudo_voigt_fwhm_is_independent_of_eta(eta):
    y = pseudo_voigt(X, 1580.0, 1.0, 25.0, eta)
    assert _numeric_fwhm(y, X) == pytest.approx(25.0, rel=1e-3)


def test_analytic_areas_match_numeric_integration():
    assert np.trapezoid(gaussian(X, 1580.0, 3.0, 40.0), X) == pytest.approx(
        gaussian_area(3.0, 40.0), rel=1e-6
    )
    # The Lorentzian's tails make a finite window undercount; 1200 cm-1 of
    # window around a 20 cm-1 band still misses ~1 %.
    assert np.trapezoid(lorentzian(X, 1580.0, 3.0, 20.0), X) == pytest.approx(
        lorentzian_area(3.0, 20.0), rel=0.02
    )
    assert np.trapezoid(pseudo_voigt(X, 1580.0, 3.0, 20.0, 0.6), X) == pytest.approx(
        pseudo_voigt_area(3.0, 20.0, 0.6), rel=0.02
    )


@pytest.mark.parametrize("q_inverse", [0.0, -0.05, -0.1, -0.22, -0.3, 0.15])
def test_bwf_closed_forms_match_the_numerics(q_inverse):
    """The maximum is at u = 1/q, not at a root of the quadratic an earlier
    version solved; getting it wrong put the peak on the wrong side."""
    y = bwf(X, 1580.0, 1.0, 40.0, q_inverse)
    index = int(np.argmax(y))
    assert X[index] == pytest.approx(bwf_peak_position(1580.0, 40.0, q_inverse), abs=0.05)
    assert y[index] == pytest.approx(bwf_peak_height(1.0, q_inverse), rel=1e-6)


def test_bwf_reduces_to_a_lorentzian_at_zero_coupling():
    assert np.allclose(bwf(X, 1580.0, 5.0, 30.0, 0.0), lorentzian(X, 1580.0, 5.0, 30.0))


def test_bwf_maximum_exceeds_the_amplitude():
    """Using the amplitude as I_G underestimates it, and by more the
    stronger the coupling."""
    assert bwf_peak_height(1.0, -0.3) > 1.0


def test_bwf_negative_q_puts_the_tail_below_the_centre():
    assert bwf_peak_position(1580.0, 40.0, -0.2) < 1580.0
    assert bwf_peak_position(1580.0, 40.0, +0.2) > 1580.0


def test_bwf_convergent_area_reduces_to_the_lorentzian():
    assert bwf_area(3.0, 20.0, 0.0) == pytest.approx(lorentzian_area(3.0, 20.0))


@pytest.mark.parametrize(
    "alias, expected",
    [("Fano", "bwf"), ("pseudo-voigt", "pseudo_voigt"), ("LOR", "lorentzian"),
     ("gauss", "gaussian"), ("BWF", "bwf")],
)
def test_profile_aliases(alias, expected):
    assert resolve_profile(alias) == expected


def test_unknown_profile_lists_the_options():
    with pytest.raises(ValueError, match="available"):
        resolve_profile("supergaussian")
