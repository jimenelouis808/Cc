"""Peak search and phase matching on real, awkward patterns."""
from __future__ import annotations

import pytest


class TestUnresolvedAndScale:
    """Two ways a real multiphase pattern punished a phase that is there.

    Both were found on a measured sample -- carbon over FeSe with Fe3C
    and alpha-Fe, at 1.0 to 2.8 counts -- where Fe3C matched eight peaks
    including its strongest and still scored 0.17, "descartada", while
    alpha-Fe matched two and scored 0.76.
    """

    def test_a_blended_reflection_is_not_counted_as_missing(self):
        """Fe3C's (220) at 44.57 and (031) at 44.99 sit under one measured
        peak 0.57 deg wide. Only one can take the credit; charging the
        phase for the other is charging it for the resolution."""
        from ramancarbon.xrd.search import _mark_undetectable

        match = _blended_match()
        _mark_undetectable(match, _flat_pattern())
        assert [r.hkl for r in match.unresolved] == [(2, 2, 0)]
        assert not match.missing

    def test_an_unresolved_reflection_leaves_coverage_alone(self):
        from ramancarbon.xrd.search import _mark_undetectable

        match = _blended_match()
        before = match.coverage
        _mark_undetectable(match, _flat_pattern())
        assert match.coverage >= before

    def test_the_scale_ignores_reflections_too_weak_to_set_it(self):
        """A ratio with a weak denominator is worthless: Fe3C's (020), at
        1.8 % of its strongest line, landed on the carbon 002 and gave a
        scale thirty times what its strong lines agree on."""
        from ramancarbon.xrd.search import SCALE_REFLECTION, _mark_undetectable

        match = _scale_match()
        _mark_undetectable(match, _flat_pattern())
        # With the weak match included the scale is ~0.30 and the faint
        # reflection reads as detectable; with strong lines only it is
        # ~0.010 and the reflection is correctly out of reach.
        assert SCALE_REFLECTION > 0
        assert [r.hkl for r in match.undetectable] == [(9, 9, 9)]
        assert not match.missing


def _flat_pattern():
    """A photon-starved background with real counting noise.

    Flat to machine precision would make `noise_estimate` zero and every
    reflection infinitely significant, which is a property of the fixture
    and not of the code under test.
    """
    import numpy as np

    from ramancarbon.xrd.pattern import Pattern

    two_theta = np.arange(10.0, 90.0, 0.01)
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.5, size=two_theta.size).astype(float)
    return Pattern(two_theta=two_theta, intensity=counts)


def _reflection(hkl, two_theta, intensity):
    from ramancarbon.xrd.search import Reflection

    return Reflection(hkl=hkl, d=1.0, two_theta=two_theta,
                      intensity=intensity, multiplicity=1, f_squared=1.0,
                      phase="prueba")


def _peak(two_theta, height, fwhm):
    from ramancarbon.xrd.search import XRDPeak

    return XRDPeak(two_theta=two_theta, d=1.0, height=height, fwhm=fwhm,
                   area=height * fwhm, prominence=height, significance=50.0)


def _blended_match():
    from ramancarbon.xrd.search import PhaseMatch

    peak = _peak(44.89, 1.03, 0.568)
    match = PhaseMatch(crystal=None)
    match.matched = [(_reflection((0, 3, 1), 44.99, 100.0), peak)]
    match.missing = [_reflection((2, 2, 0), 44.57, 64.1)]
    match.expected_strong = 2
    return match


def _scale_match():
    """One strong, honest match and one weak accidental one."""
    from ramancarbon.xrd.search import PhaseMatch

    match = PhaseMatch(crystal=None)
    match.matched = [
        (_reflection((0, 3, 1), 44.99, 100.0), _peak(44.89, 1.03, 0.10)),
        (_reflection((0, 2, 0), 26.41, 1.8), _peak(26.36, 0.53, 0.10)),
    ]
    # Predicted at the strong-line scale this is far under the noise;
    # at the scale the weak coincidence implies it would look findable.
    match.missing = [_reflection((9, 9, 9), 60.0, 3.0)]
    match.expected_strong = 3
    return match


class TestNormalisedPatternsInventPeaks:
    """Dividing a diffractogram by a constant is not free.

    A user sent the same measurement twice: the instrument's own .ras,
    544-1543 counts, and an export divided by 543, running 1.00-2.84.
    The first gives ten peaks and the second seventeen, and the nine
    extras have FWHM of 0.06-0.12 deg -- six to twelve points -- with
    significance 19-24 against a threshold of 19. They are ripples.

    The mechanism is the point-wise uncertainty. On counts the sigma is
    sqrt(N) at every point: 23 on the 550-count background and 39 on the
    1543-count peak. Once the file is normalised the per-point estimate
    is gone and a single number stands in for the whole pattern, so the
    noise on top of a strong reflection is judged by the background's --
    and every wobble on its flank clears the bar.

    Three of the eight reflections Fe3C "matched" on the normalised file
    were among those ripples, which is worth knowing before trusting any
    figure of merit computed from it.
    """

    def _pattern(self, scale=1.0, seed=0):
        import numpy as np

        from ramancarbon.xrd.pattern import Pattern

        two_theta = np.arange(20.0, 80.0, 0.01)
        rng = np.random.default_rng(seed)
        signal = 550.0 + 1000.0 * np.exp(
            -0.5 * ((two_theta - 44.9) / 0.25) ** 2)
        counts = rng.poisson(signal).astype(float)
        return Pattern(two_theta=two_theta, intensity=counts * scale,
                       wavelength=1.540593)

    def test_counts_get_a_point_wise_sigma_that_follows_the_signal(self):
        import numpy as np

        sigma = np.asarray(self._pattern().sigma)
        assert sigma.min() == pytest.approx(np.sqrt(550.0), rel=0.15)
        assert sigma.max() > 1.4 * sigma.min()

    def test_the_normalised_copy_finds_no_more_reflections(self):
        """One real reflection is in there. Anything past that is the
        normalisation talking."""
        from ramancarbon.xrd.search import find_peaks

        real = find_peaks(self._pattern())
        divided = find_peaks(self._pattern(scale=1.0 / 543.0))
        assert len(divided) <= len(real) + 1, (
            f"{len(divided)} picos sobre el patrón normalizado contra "
            f"{len(real)} sobre las cuentas")
