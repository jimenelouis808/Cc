"""Despiking, baselines, smoothing and normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.baseline import (
    asls_baseline,
    subtract_baseline,
)
from ramancarbon.core.preprocess import despike, normalise, preprocess, resample, smooth
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.models.lineshapes import lorentzian


def _synthetic(noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(200.0, 3200.0)
    background = 500.0 * np.exp(-(x - 200.0) / 1500.0) + 20.0
    peaks = lorentzian(x, 1350.0, 300.0, 50.0) + lorentzian(x, 1580.0, 800.0, 20.0)
    return x, background, peaks, background + peaks + rng.normal(0.0, noise, x.size)


def test_despike_removes_spikes_and_only_spikes():
    """The regression this replaced: a slope-based criterion carved notches
    out of the flanks of an intense, sharp G band."""
    x, _, _, y = _synthetic()
    y = y.copy()
    y[500] += 6000.0
    y[2000] += 4000.0
    cleaned, mask = despike(Spectrum(x, y, laser_nm=532.0))
    assert int(mask.sum()) == 2
    assert list(np.flatnonzero(mask)) == [500, 2000]


def test_despike_preserves_a_sharp_band_apex():
    x = np.arange(100.0, 400.0)
    y = lorentzian(x, 250.0, 500.0, 10.0) + 10.0
    cleaned, mask = despike(Spectrum(x, y))
    assert not mask.any()
    assert cleaned.intensity.max() == pytest.approx(y.max())


@pytest.mark.parametrize("method", ["asls", "polynomial", "rubberband"])
def test_baselines_recover_the_background(method):
    x, background, _, y = _synthetic()
    _, estimated = subtract_baseline(Spectrum(x, y, laser_nm=532.0), method=method)
    rms = float(np.sqrt(np.mean((estimated - background) ** 2)))
    assert rms < 0.1 * float(np.ptp(background))


def test_asls_defaults_do_not_eat_the_g_band():
    """Softer smoothing shaves ~15 % off the G height, which propagates
    straight into I_D/I_G. Guard the tuned default."""
    x, background, peaks, y = _synthetic(noise=0.0)
    corrected, _ = subtract_baseline(Spectrum(x, y, laser_nm=532.0), method="asls")
    recovered = corrected.max_in(1550.0, 1620.0)[1]
    assert recovered == pytest.approx(800.0, rel=0.05)


def test_asls_rejects_invalid_asymmetry():
    with pytest.raises(ValueError, match="asymmetry"):
        asls_baseline(np.ones(100), p=1.5)


def test_smooth_forces_an_odd_window():
    x = np.arange(0.0, 500.0)
    s = Spectrum(x, np.sin(x / 10.0))
    out = smooth(s, window=8, order=3)
    assert "window=9" in out.history[-1]


def test_smooth_refuses_a_window_longer_than_the_data():
    s = Spectrum(np.arange(0.0, 20.0), np.ones(20))
    with pytest.raises(ValueError, match="not shorter"):
        smooth(s, window=41)


def test_resample_gives_a_uniform_axis():
    axis = np.concatenate([np.arange(100.0, 200.0), np.arange(200.0, 400.0, 4.0)])
    s = Spectrum(axis, np.ones(axis.size))
    assert not s.is_uniform
    assert resample(s, step=1.0).is_uniform


def test_normalisation_does_not_change_ratios():
    """Stated in the docstring; worth a test, because it is the reason
    normalisation is safe to apply before reading off I_D/I_G."""
    x, _, peaks, _ = _synthetic(noise=0.0)
    s = Spectrum(x, peaks, laser_nm=532.0)
    before = s.max_in(1300.0, 1400.0)[1] / s.max_in(1550.0, 1620.0)[1]
    scaled = normalise(s, "g")
    after = scaled.max_in(1300.0, 1400.0)[1] / scaled.max_in(1550.0, 1620.0)[1]
    assert before == pytest.approx(after)


def test_normalise_rejects_a_flat_spectrum():
    with pytest.raises(ValueError, match="flat"):
        normalise(Spectrum(np.arange(100.0), np.ones(100)), "minmax")


def test_pipeline_records_its_whole_history():
    x, _, _, y = _synthetic()
    out, diagnostics = preprocess(
        Spectrum(x, y, laser_nm=532.0), smooth_window=7, normalise_method="g"
    )
    joined = " ".join(out.history)
    for step in ("despike", "smooth", "baseline", "normalise"):
        assert step in joined
    assert diagnostics["baseline"] is not None
