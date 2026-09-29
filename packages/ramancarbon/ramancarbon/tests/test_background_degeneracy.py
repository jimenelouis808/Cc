"""A weight fraction is a statement about the background as much as the phase.

A broad reflection from a nanocrystalline phase and a flexible polynomial
background describe the same shape, and nothing in a refinement objects:
it converges, Rwp looks good, and the phase comes back with a fraction
that reads like a measurement.

Measured on the user's real CVD pattern -- alpha-Fe, a 6 nm turbostratic
carbon, cementite and two iron selenides over 10-90 degrees -- changing
ONLY the order of the background polynomial moved the carbon from 25.9 %
to 52.4 % by weight and the cementite from 52.7 % to 16.3 %, and deleting
the carbon phase entirely cost 0.29 percentage points of Rwp. The fit
never complained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from ramancarbon.xrd.rietveld import (
    BACKGROUND_DOMINATES,
    SCALE_BACKGROUND_DEGENERATE,
    RietveldResult,
    _warn_about_the_background,
)


@dataclass
class _Crystal:
    name: str


@dataclass
class _Phase:
    crystal: _Crystal


class _Result:
    """Just enough of a RietveldResult for the checks to run on."""

    def __init__(self, background, calculated, weights, visible, correlations=None):
        self.background = np.asarray(background, dtype=float)
        self.calculated = np.asarray(calculated, dtype=float)
        self._weights = weights
        self._visible = visible
        self.correlations = correlations or {}
        self.phases = [_Phase(_Crystal(name)) for name in weights]
        self.warnings: list[str] = []

    background_share = RietveldResult.background_share

    def weight_fractions(self):
        return dict(self._weights)

    def phase_contributions(self):
        return dict(self._visible)


def _codes(result) -> str:
    return " || ".join(result.warnings)


def test_a_background_that_is_most_of_the_pattern_is_reported() -> None:
    """The measured case: the phases explained 3.2 % of the counts."""
    result = _Result(
        background=np.full(100, 96.8),
        calculated=np.full(100, 100.0),
        weights={"C": 0.52, "Fe": 0.48},
        visible={"C": 0.2, "Fe": 0.8},
    )
    _warn_about_the_background(result)
    assert "97 % del patrón calculado" in _codes(result)
    assert "orden del polinomio" in _codes(result)


def test_a_modest_background_says_nothing_about_itself() -> None:
    result = _Result(
        background=np.full(100, 30.0),
        calculated=np.full(100, 100.0),
        weights={"C": 0.5, "Fe": 0.5},
        visible={"C": 0.5, "Fe": 0.5},
    )
    _warn_about_the_background(result)
    assert "del patrón calculado" not in _codes(result)


def test_a_scale_correlated_with_the_background_is_reported() -> None:
    result = _Result(
        background=np.full(100, 10.0),
        calculated=np.full(100, 100.0),
        weights={"C": 0.5, "Fe": 0.5},
        visible={"C": 0.5, "Fe": 0.5},
        correlations={("escala_C", "fondo_c2"): -0.987},
    )
    _warn_about_the_background(result)
    assert "escala_C está atado al fondo" in _codes(result)
    assert "no está determinada por la medida" in _codes(result)


def test_a_phase_is_named_once_however_many_coefficients_it_ties_to() -> None:
    """The region wants to be elsewhere; that is ONE problem, not six."""
    result = _Result(
        background=np.full(100, 10.0),
        calculated=np.full(100, 100.0),
        weights={"C": 0.5, "Fe": 0.5},
        visible={"C": 0.5, "Fe": 0.5},
        correlations={("escala_C", f"fondo_c{i}"): -0.99 for i in range(6)},
    )
    _warn_about_the_background(result)
    assert _codes(result).count("escala_C está atado al fondo") == 1


def test_any_degenerate_pair_is_reported_not_just_scale_against_background() -> None:
    """Measured on the real pattern: zero against sample displacement, -1.0000.

    Two different functions of the angle, a constant and a cos-theta, and
    on this refinement the data could not tell them apart at all. Nothing
    was reporting it, because the first version of this check only looked
    at scale-against-background pairs.
    """
    result = _Result(
        background=np.full(100, 10.0),
        calculated=np.full(100, 100.0),
        weights={"Fe": 1.0},
        visible={"Fe": 1.0},
        correlations={("cero", "desplazamiento"): -1.0},
    )
    _warn_about_the_background(result)
    assert "cero–desplazamiento (-1.000)" in _codes(result)
    assert "NO separan" in _codes(result)


def test_one_degeneracy_in_three_costumes_is_reported_once() -> None:
    """zero, displacement and a lattice constant mutually at 0.9999."""
    result = _Result(
        background=np.full(100, 10.0),
        calculated=np.full(100, 100.0),
        weights={"Fe": 1.0},
        visible={"Fe": 1.0},
        correlations={
            ("cero", "desplazamiento"): -1.0,
            ("cero", "a[Fe]"): 0.9999,
            ("desplazamiento", "a[Fe]"): -0.9999,
        },
    )
    _warn_about_the_background(result)
    line = next(w for w in result.warnings if "NO separan" in w)
    assert line.count("–") == 2          # two pairs cover all three names


def test_a_weak_correlation_is_left_alone() -> None:
    result = _Result(
        background=np.full(100, 10.0),
        calculated=np.full(100, 100.0),
        weights={"C": 0.5, "Fe": 0.5},
        visible={"C": 0.5, "Fe": 0.5},
        correlations={("escala_C", "fondo_c2"): -0.80},
    )
    _warn_about_the_background(result)
    assert "y el fondo están correlacionados" not in _codes(result)


def test_heavy_by_weight_and_faint_in_the_pattern_is_reported() -> None:
    """Carbon against iron: Z is 6 against 26 and the scattering goes as Z²."""
    result = _Result(
        background=np.full(100, 10.0),
        calculated=np.full(100, 100.0),
        weights={"C": 0.52, "Fe": 0.48},
        visible={"C": 0.06, "Fe": 0.94},
    )
    _warn_about_the_background(result)
    assert "52 % en peso pero solo el 6 %" in _codes(result)
    assert "no es incoherente" in _codes(result).lower()


def test_the_real_numbers_from_the_users_pattern_are_caught() -> None:
    """52.4 % by weight against 17.9 % of the intensity: a ratio of 2.93.

    A threshold of 3 missed this by four hundredths, which is the wrong
    way to miss for a check that exists to explain a number the user
    already found surprising.
    """
    result = _Result(
        background=np.full(100, 96.8),
        calculated=np.full(100, 100.0),
        weights={"C_turbostratico_3.36": 0.524, "Fe_alfa": 0.208,
                 "Fe3C_cementita": 0.163, "FeSe_hexagonal": 0.079,
                 "FeSe_tetragonal": 0.026},
        visible={"C_turbostratico_3.36": 0.179, "Fe_alfa": 0.293,
                 "Fe3C_cementita": 0.328, "FeSe_hexagonal": 0.112,
                 "FeSe_tetragonal": 0.088},
    )
    _warn_about_the_background(result)
    assert "C_turbostratico_3.36: 52 % en peso pero solo el 18 %" in _codes(result)
    # alpha-Fe is 20.8 % by weight and 29.3 % of the intensity: nothing to say.
    assert "Fe_alfa:" not in _codes(result)


def test_a_phase_whose_weight_matches_its_signal_says_nothing() -> None:
    result = _Result(
        background=np.full(100, 10.0),
        calculated=np.full(100, 100.0),
        weights={"C": 0.5, "Fe": 0.5},
        visible={"C": 0.45, "Fe": 0.55},
    )
    _warn_about_the_background(result)
    assert "en peso pero solo el" not in _codes(result)


def test_a_small_phase_is_not_reported_for_being_faint() -> None:
    """2 % by weight and 0.5 % of the signal is not a finding about anything."""
    result = _Result(
        background=np.full(100, 10.0),
        calculated=np.full(100, 100.0),
        weights={"minor": 0.02, "Fe": 0.98},
        visible={"minor": 0.005, "Fe": 0.995},
    )
    _warn_about_the_background(result)
    assert "en peso pero solo el" not in _codes(result)


@pytest.mark.parametrize("constant", [BACKGROUND_DOMINATES,
                                      SCALE_BACKGROUND_DEGENERATE])
def test_the_thresholds_are_fractions(constant: float) -> None:
    assert 0.0 < constant < 1.0


def test_the_weight_over_signal_threshold_sits_between_the_measured_cases() -> None:
    from ramancarbon.xrd.rietveld import WEIGHT_OVER_SIGNAL

    assert 0.208 / 0.293 < WEIGHT_OVER_SIGNAL < 0.524 / 0.179


def test_the_degeneracy_threshold_matches_the_raman_side() -> None:
    """Same question, same number: two parameters the data do not separate."""
    from ramancarbon.models.acceptance import DEGENERATE_CORRELATION

    assert SCALE_BACKGROUND_DEGENERATE == DEGENERATE_CORRELATION


def test_background_share_of_an_empty_pattern_is_zero_not_a_crash() -> None:
    result = _Result(
        background=np.zeros(10), calculated=np.zeros(10),
        weights={}, visible={},
    )
    assert result.background_share == 0.0


def test_the_texture_stage_says_when_it_did_not_run() -> None:
    """Skipped in silence, and silence reads as "nothing to do".

    The preferred-orientation stage is the only one that touches relative
    intensities, and it is skipped whenever no phase declares a texture
    axis -- which is the default. A user whose peak HEIGHTS do not match
    while the positions do had no way to learn that the one stage for
    that never ran.
    """
    from ramancarbon.xrd.pattern import Pattern
    from ramancarbon.xrd.reference import load_library
    from ramancarbon.xrd.rietveld import auto_refine

    crystal = next(iter(load_library())).crystal
    angles = np.linspace(20.0, 60.0, 900)
    pattern = Pattern(two_theta=angles,
                      intensity=400.0 + 30.0 * np.sin(angles / 7.0),
                      name="sintético")
    without = auto_refine(pattern, [crystal])
    assert any("orientación preferente no se ha corrido" in w
               for w in without.warnings)

    with_axis = auto_refine(pattern, [crystal], preferred_axis=(0, 0, 1))
    assert not any("orientación preferente no se ha corrido" in w
                   for w in with_axis.warnings)
