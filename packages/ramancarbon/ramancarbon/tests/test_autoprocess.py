"""Automatic preprocessing: arPLS, stiffness selection, smoothing choice."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.baseline import (
    MIN_WIDEST_BAND_CM,
    arpls_baseline,
    auto_lambda,
    background_mask,
    cutoff_for_lambda,
    lambda_for_cutoff,
    subtract_baseline,
)
from ramancarbon.core.preprocess import auto_settings, normalise, preprocess
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.models.lineshapes import lorentzian

TRUE_DG = 500.0 / 600.0


def _fluorescent(noise=4.0, amplitude=2500.0, decay=800.0, step=1.0, seed=2):
    """A carbon spectrum on a decaying fluorescence tail."""
    rng = np.random.default_rng(seed)
    x = np.arange(120.0, 3200.0, step)
    background = amplitude * np.exp(-(x - 120.0) / decay) + 40.0
    bands = (
        lorentzian(x, 1350.0, 500.0, 70.0)
        + lorentzian(x, 1585.0, 600.0, 55.0)
        + lorentzian(x, 2700.0, 150.0, 140.0)
    )
    y = background + bands + rng.normal(0.0, noise, x.size)
    return Spectrum(x, y, laser_nm=532.0, name="fluor"), background


def _dg(spectrum, baseline):
    corrected = spectrum.intensity - baseline
    x = spectrum.shift
    d = corrected[(x > 1300) & (x < 1400)].max()
    g = corrected[(x > 1540) & (x < 1630)].max()
    return d / g


def test_cutoff_formula_round_trips():
    for period in (100.0, 350.0, 700.0, 2000.0):
        assert cutoff_for_lambda(lambda_for_cutoff(period)) == pytest.approx(period)


@pytest.mark.parametrize("method", ["asls", "arpls"])
def test_both_baselines_recover_the_ratio_under_strong_fluorescence(method):
    """The finding that decides how to report results: a smooth background
    error is nearly constant across the D-G window and largely cancels in a
    ratio, even when it is large in absolute terms."""
    spectrum, background = _fluorescent(amplitude=3000.0, decay=700.0)
    lam, _ = auto_lambda(spectrum, method=method)
    _, baseline = subtract_baseline(spectrum, method=method, lam=lam)
    absolute_error = float(np.sqrt(np.mean((baseline - background) ** 2)))
    assert absolute_error > 20.0  # the background really is badly recovered
    assert _dg(spectrum, baseline) == pytest.approx(TRUE_DG, rel=0.05)


def test_asls_is_the_default_because_it_measured_better():
    """Guards the choice against being reverted on arPLS's reputation."""
    from ramancarbon.core.preprocess import auto_settings as _auto

    spectrum, background = _fluorescent(amplitude=3000.0, decay=700.0)
    assert _auto(spectrum).baseline_method == "asls"
    lam, _ = auto_lambda(spectrum, method="asls")
    error = lambda b: float(np.sqrt(np.mean((b - background) ** 2)))  # noqa: E731
    _, asls = subtract_baseline(spectrum, method="asls", lam=lam)
    _, arpls = subtract_baseline(spectrum, method="arpls", lam=lam)
    assert error(asls) < error(arpls)


def test_auto_lambda_scales_with_the_sampling_step():
    """The cutoff is a period in POINTS, so the same sample at 1 and at
    4 cm-1 per point needs stiffnesses differing by 4^4."""
    fine, _ = _fluorescent(step=1.0)
    coarse, _ = _fluorescent(step=4.0)
    lam_fine, _ = auto_lambda(fine)
    lam_coarse, _ = auto_lambda(coarse)
    assert lam_fine / lam_coarse == pytest.approx(256.0, rel=0.5)


@pytest.mark.parametrize(
    "amplitude, decay",
    [(0.0, 1e9), (600.0, 2000.0), (2500.0, 800.0)],
)
def test_auto_lambda_recovers_the_ratio(amplitude, decay):
    spectrum, _ = _fluorescent(amplitude=amplitude, decay=decay)
    lam, _ = auto_lambda(spectrum)
    assert _dg(spectrum, arpls_baseline(spectrum.intensity, lam=lam)) == pytest.approx(
        TRUE_DG, rel=0.08
    )


def test_auto_lambda_warns_when_the_background_is_as_fast_as_the_bands():
    spectrum, _ = _fluorescent(amplitude=3000.0, decay=300.0)
    _, reason = auto_lambda(spectrum)
    assert "AVISO" in reason


def test_auto_lambda_is_quiet_on_a_benign_background():
    spectrum, _ = _fluorescent(amplitude=600.0, decay=2000.0)
    _, reason = auto_lambda(spectrum)
    assert "AVISO" not in reason


def test_background_mask_finds_band_free_points_under_fluorescence():
    """Ranking raw intensity would mark only the far end of the spectrum,
    missing the steep region near the laser line entirely."""
    spectrum, _ = _fluorescent(amplitude=3000.0, decay=600.0)
    mask = background_mask(spectrum.intensity)
    low_end = mask[: mask.size // 4].sum()
    assert low_end > 0.1 * (mask.size // 4)


def test_auto_settings_declines_to_smooth_clean_data():
    spectrum, _ = _fluorescent(noise=2.0)
    settings = auto_settings(spectrum)
    assert settings.smooth_window == 0
    assert "por encima" in settings.reasons["smooth"]


def test_auto_settings_smooths_noisy_data_but_not_past_the_narrowest_band():
    spectrum, _ = _fluorescent(noise=120.0)
    settings = auto_settings(spectrum)
    assert settings.smooth_window > 0
    # 55 cm-1 is the narrowest band; a third of it is ~18 points.
    assert settings.smooth_window <= 21


def test_auto_settings_explains_every_decision():
    spectrum, _ = _fluorescent(noise=40.0)
    settings = auto_settings(spectrum)
    assert set(settings.reasons) == {"despike", "smooth", "baseline"}
    assert all(text for text in settings.reasons.values())
    assert "λ" in settings.summary()


def test_auto_settings_feed_straight_into_preprocess():
    spectrum, _ = _fluorescent(noise=25.0)
    processed, _ = preprocess(spectrum, **auto_settings(spectrum).to_kwargs())
    assert any("baseline" in step for step in processed.history)


def test_widest_band_floor_is_applied():
    """A noisy spectrum whose broad bands escape detection must not get a
    soft baseline that eats the bands it did find."""
    spectrum, _ = _fluorescent(noise=90.0)
    lam, reason = auto_lambda(spectrum)
    assert lam >= lambda_for_cutoff(5.0 * MIN_WIDEST_BAND_CM / spectrum.step) * 0.99


def test_normalise_0_100():
    spectrum, _ = _fluorescent()
    scaled = normalise(spectrum, "0-100")
    assert float(scaled.intensity.min()) == pytest.approx(0.0)
    assert float(scaled.intensity.max()) == pytest.approx(100.0)
    assert "0-100" in scaled.history[-1]


def test_0_100_barely_changes_ratios_once_the_baseline_is_gone():
    spectrum, _ = _fluorescent()
    corrected, _ = subtract_baseline(spectrum, method="asls")
    before = _dg(corrected, np.zeros_like(corrected.intensity))
    after = _dg(normalise(corrected, "0-100"), np.zeros((corrected.shift.size,)))
    assert before == pytest.approx(after, rel=0.02)


def test_0_100_warns_when_it_would_shift_ratios():
    """Min-max subtracts an offset, and an offset does not cancel in a
    ratio. On uncorrected data that matters, so it has to say so."""
    spectrum, _ = _fluorescent()
    scaled = normalise(spectrum, "0-100")
    assert "AVISO" in scaled.history[-1]
