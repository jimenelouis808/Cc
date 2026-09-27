

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
