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


# -- a CCD does not have one noise level --------------------------------

def _heteroscedastic(seed=0, quiet=9.0, loud=35.0, with_2d=True):
    """A spectrum whose noise rises towards the red, as a real one does.

    Quantum efficiency falls off, so on a 532 nm measurement the scatter
    above about 2300 cm⁻¹ is routinely three times what it is under the
    G band. Everything in this package is a ratio against sigma, so a
    single sigma is wrong in both directions at once.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(100.0, 3200.0)
    y = (lorentzian(x, 1340.0, 380.0, 180.0)
         + lorentzian(x, 1586.0, 430.0, 90.0))
    if with_2d:
        y = y + lorentzian(x, 2680.0, 55.0, 300.0)
    y = y + 60.0
    level = quiet + (loud - quiet) / (1.0 + np.exp(-(x - 2350.0) / 120.0))
    y = y + rng.normal(0.0, 1.0, x.size) * level
    return Spectrum(shift=x, intensity=y, name="het", laser_nm=532.0), level


def test_the_noise_is_measured_where_it_is_not_averaged_over_the_spectrum():
    spectrum, level = _heteroscedastic()
    sigma = spectrum.local_noise()
    assert sigma.shape == spectrum.intensity.shape

    quiet = float(np.median(sigma[(spectrum.shift > 300) & (spectrum.shift < 1100)]))
    loud = float(np.median(sigma[spectrum.shift > 2600])) 
    assert quiet == pytest.approx(9.0, rel=0.45), quiet
    assert loud == pytest.approx(35.0, rel=0.45), loud
    assert loud > 2.0 * quiet, "the profile is flat; it is meant to follow the data"

    # A single number is between the two and therefore wrong everywhere.
    single = spectrum.noise_estimate()
    assert quiet < single < loud


def test_the_local_noise_does_not_collapse_at_the_ends():
    """A median filter five hundred points wide whose padding repeats one
    edge value RETURNS that edge value: half the window is real data and
    half is copies of a single number. The estimate collapsed to 2.5 at
    the last point against 30 a hundred wavenumbers earlier, the
    threshold collapsed with it, and edge noise came back as a band."""
    spectrum, _ = _heteroscedastic()
    sigma = spectrum.local_noise()
    interior = float(np.median(sigma[-600:-300]))
    edge = float(np.median(sigma[-30:]))
    assert edge == pytest.approx(interior, rel=0.5), (edge, interior)
    assert edge > 0.4 * interior


def test_despiking_does_not_flatten_the_ends_of_the_spectrum():
    """scipy's medfilt zero-pads, so within half a kernel of either end
    the running median is dragged towards zero, every point there stands
    high above "its" median, and each one is flagged as a cosmic ray and
    REPLACED by that wrong median. A despiked spectrum then ends in a run
    of identical values that are not the measurement."""
    from ramancarbon.core.preprocess import despike

    spectrum, _ = _heteroscedastic(seed=2)
    cleaned, mask = despike(spectrum)
    tail = cleaned.intensity[-12:]
    assert len(set(np.round(tail, 6))) > 6, (
        f"the tail was flattened into repeats: {np.round(tail, 1)}")
    assert mask[-12:].sum() <= 4, "nearly every point at the end was 'a spike'"

    # And in the noisy region a single global sigma made ordinary scatter
    # look like cosmic rays, because the threshold there was about 2.5
    # LOCAL sigma rather than 8.
    loud = mask[spectrum.shift > 2400].sum()
    assert loud <= 0.02 * int((spectrum.shift > 2400).sum()), loud


def test_smoothing_records_the_noise_it_started_from():
    """Every detection threshold here is a ratio against the
    point-to-point scatter, and a Savitzky-Golay filter destroys that
    scatter by construction. Unguarded, five-point smoothing took eight
    real bands to fifty-eight "peaks", most of them noise, and three of
    those were given names."""
    from ramancarbon.core.preprocess import smooth

    spectrum, _ = _heteroscedastic(seed=3)
    before = spectrum.noise_estimate()
    smoothed = smooth(spectrum, window=5, order=3)

    raw_scatter = float(np.median(np.abs(np.diff(spectrum.intensity, n=2))))
    smooth_scatter = float(np.median(np.abs(np.diff(smoothed.intensity, n=2))))
    assert smooth_scatter < raw_scatter / 1.5, "the smoothing did nothing"

    assert smoothed.metadata["noise_floor"] == pytest.approx(before, rel=1e-9)
    assert smoothed.noise_estimate() == pytest.approx(before, rel=0.02)
    assert "NO se ajusta" in smoothed.metadata["smoothed"]
    assert len(find_peaks(smoothed)) < 12


def test_a_band_wider_than_its_own_search_window_needs_an_integral_not_a_peak():
    """Prominence is measured against the minima on either side of a
    maximum, so a band that FILLS the window it is looked for in has no
    prominence inside it. The 2D envelope of a disordered carbon is three
    hundred wavenumbers wide and the 2D window is three hundred and ten;
    every "peak" found in there is a two-to-five-point noise spike while
    the band itself, eighty counts above zero across the whole window, is
    invisible to the search."""
    from ramancarbon.core.peaks import window_excess

    present, _ = _heteroscedastic(seed=4, with_2d=True)
    absent, _ = _heteroscedastic(seed=4, with_2d=False)

    inside = [p for p in find_peaks(present, window=(2490.0, 2860.0))
              if 2550.0 <= p.position <= 2800.0]
    assert not inside, (
        "this spectrum's 2D is now a detectable maximum; the test needs a "
        "broader one to still be testing what it says")

    valley = [(2130.0, 2430.0)]
    strong = window_excess(present, 2550.0, 2800.0, flanks=valley)
    nothing = window_excess(absent, 2550.0, 2800.0, flanks=valley)
    assert strong > 6.0, strong
    assert nothing < 4.5, nothing
    assert strong > 2.0 * max(nothing, 0.5)
