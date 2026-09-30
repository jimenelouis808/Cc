"""How many layers, from the 002 — and not from somebody else's reflection.

`carbon_microstructure` takes four numbers the caller has already picked
out, and picking them out is where the mistake lived. The command line
took "the peak closest to 26.5 degrees" and "the peak closest to 43",
which always return a peak: on a pattern with no carbon at all that is a
stack height and a layer count computed from an iron line.
"""

from __future__ import annotations

import math

import pytest

from ramancarbon.xrd.microstructure import (
    D002_WINDOW,
    D100_WINDOW,
    carbon_from_peaks,
)

CU = 1.540598


class _Peak:
    def __init__(self, two_theta: float, fwhm: float, height: float = 100.0):
        self.two_theta = two_theta
        self.fwhm = fwhm
        self.height = height


def _angle_for(d: float, wavelength: float = CU) -> float:
    return 2.0 * math.degrees(math.asin(wavelength / (2.0 * d)))


def test_the_real_pattern_gives_back_the_number_the_user_missed() -> None:
    """The user's own CVD diffractogram: a 1.2 degree hump at 26.120."""
    peaks = [_Peak(26.120, 1.200, 86.2)]
    carbon = carbon_from_peaks(peaks, wavelength=CU)
    assert carbon.d002 == pytest.approx(3.4088, abs=1e-3)
    assert carbon.lc_nm == pytest.approx(6.7, abs=0.3)
    assert carbon.layers == pytest.approx(20, abs=1.5)
    assert carbon.graphitisation == pytest.approx(0.36, abs=0.03)


def test_a_pattern_with_no_carbon_refuses_instead_of_inventing_a_stack() -> None:
    """Iron lines only. "Nearest to 26.5" would have returned one of them."""
    peaks = [_Peak(44.67, 0.21), _Peak(65.02, 0.25), _Peak(82.33, 0.30)]
    carbon = carbon_from_peaks(peaks, wavelength=CU)
    assert carbon.d002 is None
    assert carbon.layers is None
    assert any("ninguno" in w or "no hay" in w for w in carbon.warnings)


def test_an_empty_peak_list_says_so() -> None:
    carbon = carbon_from_peaks([], wavelength=CU)
    assert carbon.d002 is None
    assert carbon.warnings


def test_the_in_plane_line_of_another_phase_is_not_used_for_la() -> None:
    """Measured: cementite at 43.89 gave L_a = 36.8 nm for a 6.7 nm stack.

    Narrow line, big Scherrer size, plausible magnitude, no meaning.
    """
    peaks = [_Peak(26.120, 1.200, 86.2), _Peak(43.890, 0.208, 235.0)]
    loose = carbon_from_peaks(peaks, wavelength=CU)
    assert loose.la_nm is not None and loose.la_nm > 20.0

    filtered = carbon_from_peaks(peaks, wavelength=CU, skip_for_la=[43.890])
    assert filtered.la_nm is None
    assert filtered.lc_nm is not None          # the stack is still measured
    assert any("otra fase ya lo explica" in w for w in filtered.warnings)


def test_a_genuine_carbon_100_still_gives_la() -> None:
    peaks = [_Peak(26.120, 1.200, 86.2), _Peak(43.300, 2.100, 40.0)]
    carbon = carbon_from_peaks(peaks, wavelength=CU, skip_for_la=[44.67])
    assert carbon.la_nm is not None
    assert carbon.la_nm < 10.0                 # broad line, small domain


def test_the_strongest_peak_in_the_window_wins_not_the_nearest() -> None:
    """A disordered 002 is a wide hump whose apex wanders from the table."""
    peaks = [_Peak(26.500, 0.10, 5.0), _Peak(25.100, 2.00, 300.0)]
    carbon = carbon_from_peaks(peaks, wavelength=CU)
    assert carbon.d002 == pytest.approx(
        CU / (2.0 * math.sin(math.radians(25.100) / 2.0)), abs=1e-3)


@pytest.mark.parametrize("d", [D002_WINDOW[0] - 0.05, D002_WINDOW[1] + 0.05])
def test_just_outside_the_window_is_outside(d: float) -> None:
    carbon = carbon_from_peaks([_Peak(_angle_for(d), 1.0)], wavelength=CU)
    assert carbon.d002 is None


@pytest.mark.parametrize("d", [D002_WINDOW[0] + 0.01, D002_WINDOW[1] - 0.01])
def test_just_inside_the_window_is_inside(d: float) -> None:
    carbon = carbon_from_peaks([_Peak(_angle_for(d), 1.0)], wavelength=CU)
    assert carbon.d002 is not None


def test_the_windows_bracket_the_spacings_they_are_for() -> None:
    from ramancarbon.xrd.microstructure import (
        GRAPHITE_D002,
        TURBOSTRATIC_D002,
    )

    assert D002_WINDOW[0] < GRAPHITE_D002 < TURBOSTRATIC_D002 < D002_WINDOW[1]
    assert D100_WINDOW[0] < 2.13 < D100_WINDOW[1]      # graphite 100
