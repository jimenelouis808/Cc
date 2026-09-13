"""The Spectrum container and its invariants."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.spectrum import (
    Spectrum,
    laser_energy_ev,
    laser_wavelength_nm,
    stack_average,
)


def test_axis_is_sorted_ascending():
    s = Spectrum([300.0, 100.0, 200.0], [3.0, 1.0, 2.0])
    assert list(s.shift) == [100.0, 200.0, 300.0]
    assert list(s.intensity) == [1.0, 2.0, 3.0]


def test_duplicate_abscissae_are_averaged():
    """Stitched spectra repeat abscissae at grating boundaries."""
    s = Spectrum([100.0, 100.0, 200.0], [10.0, 20.0, 5.0])
    assert list(s.shift) == [100.0, 200.0]
    assert s.intensity[0] == pytest.approx(15.0)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="same length"):
        Spectrum([1.0, 2.0, 3.0], [1.0, 2.0])


def test_laser_energy_round_trip():
    assert laser_energy_ev(532.0) == pytest.approx(2.3305, abs=1e-3)
    assert laser_wavelength_nm(laser_energy_ev(633.0)) == pytest.approx(633.0)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_laser_rejected(bad):
    with pytest.raises(ValueError):
        laser_energy_ev(bad)


def test_covers_distinguishes_absent_from_unmeasured():
    """The distinction the whole classifier rests on."""
    wide = Spectrum(np.arange(100.0, 3000.0), np.ones(2900))
    narrow = Spectrum(np.arange(600.0, 3000.0), np.ones(2400))
    assert wide.covers(120.0, 350.0)
    assert not narrow.covers(120.0, 350.0)


def test_noise_estimate_survives_a_band():
    """The second-difference estimator must not be fooled by curvature."""
    rng = np.random.default_rng(0)
    x = np.arange(1000.0, 2000.0)
    y = 500.0 / (1.0 + ((x - 1500.0) / 15.0) ** 2) + rng.normal(0.0, 3.0, x.size)
    assert Spectrum(x, y).noise_estimate() == pytest.approx(3.0, rel=0.2)


def test_step_uses_median_not_mean():
    """A stitch gap must not distort the reported sampling."""
    axis = np.concatenate([np.arange(100.0, 200.0), np.arange(900.0, 1000.0)])
    s = Spectrum(axis, np.ones(axis.size))
    assert s.step == pytest.approx(1.0)
    assert not s.is_uniform


def test_stack_average_refuses_mixed_lasers():
    a = Spectrum(np.arange(100.0, 200.0), np.ones(100), laser_nm=532.0)
    b = Spectrum(np.arange(100.0, 200.0), np.ones(100), laser_nm=633.0)
    with pytest.raises(ValueError, match="different excitations"):
        stack_average([a, b])


def test_stack_average_reduces_noise():
    rng = np.random.default_rng(5)
    x = np.arange(100.0, 1100.0)
    spectra = [
        Spectrum(x, 100.0 + rng.normal(0.0, 10.0, x.size), laser_nm=532.0)
        for _ in range(9)
    ]
    averaged = stack_average(spectra)
    assert np.std(averaged.intensity) < 0.5 * np.std(spectra[0].intensity)


def test_history_records_every_step():
    s = Spectrum(np.arange(100.0, 200.0), np.ones(100))
    out = s.with_intensity(np.zeros(100), "paso1").with_intensity(np.ones(100), "paso2")
    assert out.history == ["paso1", "paso2"]
    assert s.history == []  # the original is untouched


def test_crop_reports_the_actual_range_when_it_fails():
    s = Spectrum(np.arange(100.0, 200.0), np.ones(100))
    with pytest.raises(ValueError, match="100.0–199.0"):
        s.crop(500.0, 600.0)
