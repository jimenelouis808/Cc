"""Peak detection, and the false-positive rate that motivated its design."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.peaks import find_peaks, measure_peak
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.models.lineshapes import lorentzian


def test_finds_known_peaks_at_the_right_positions_and_widths():
    rng = np.random.default_rng(3)
    x = np.arange(100.0, 3200.0)
    y = (
        lorentzian(x, 167.0, 120.0, 10.0)
        + lorentzian(x, 1345.0, 400.0, 35.0)
        + lorentzian(x, 1583.0, 900.0, 20.0)
        + lorentzian(x, 2680.0, 500.0, 60.0)
        + rng.normal(0.0, 3.0, x.size)
    )
    peaks = find_peaks(Spectrum(x, y, laser_nm=532.0))
    found = {round(p.position): p for p in peaks}
    for expected, width in ((167, 10.0), (1345, 35.0), (1583, 20.0), (2680, 60.0)):
        match = next((p for pos, p in found.items() if abs(pos - expected) <= 2), None)
        assert match is not None, f"missed the band at {expected} cm⁻¹"
        assert match.fwhm == pytest.approx(width, rel=0.15)


def test_pure_noise_yields_almost_no_peaks():
    """The regression that mattered: a 3-sigma height cut reported four
    phantom RBMs in every graphene spectrum. Over 40 noise-only spectra the
    threshold must keep false positives rare."""
    total = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        x = np.arange(100.0, 420.0)
        total += len(find_peaks(Spectrum(x, rng.normal(0.0, 2.0, x.size))))
    assert total <= 4, f"{total} false peaks in 40 noise-only spectra"


def test_a_weak_but_real_band_is_still_found():
    """The threshold must not be so strict that it misses real weak RBMs."""
    rng = np.random.default_rng(11)
    x = np.arange(100.0, 420.0)
    y = lorentzian(x, 250.0, 8.0, 10.0) + rng.normal(0.0, 2.0, x.size)
    peaks = find_peaks(Spectrum(x, y))
    assert any(abs(p.position - 250.0) < 4.0 for p in peaks)


def test_significance_scales_with_width_not_only_height():
    """A broad band of modest height is more detectable than a sharp spike
    of the same height; that is what the matched filter encodes."""
    x = np.arange(100.0, 900.0)
    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 2.0, x.size)
    broad = find_peaks(Spectrum(x, lorentzian(x, 500.0, 20.0, 60.0) + noise))
    narrow = find_peaks(Spectrum(x, lorentzian(x, 500.0, 20.0, 6.0) + noise))
    assert broad and broad[0].significance > 3 * max(
        (p.significance for p in narrow), default=0.0
    )


def test_measure_peak_returns_none_outside_the_data():
    s = Spectrum(np.arange(100.0, 200.0), np.ones(100))
    assert measure_peak(s, 5000.0) is None


def test_window_restricts_the_search():
    rng = np.random.default_rng(1)
    x = np.arange(100.0, 2000.0)
    y = lorentzian(x, 200.0, 300.0, 10.0) + lorentzian(x, 1580.0, 900.0, 20.0)
    y = y + rng.normal(0.0, 2.0, x.size)
    peaks = find_peaks(Spectrum(x, y), window=(100.0, 400.0))
    assert peaks and all(p.position < 400.0 for p in peaks)
