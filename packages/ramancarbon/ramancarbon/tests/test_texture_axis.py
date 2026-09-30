"""The texture axis is a direction, and the program can find it itself.

Three separate things are protected here, and the second one is a bug
that had been silent since preferred orientation was added:

1. The candidates come from the phase's own reflections, reduced to
   primitive directions. That is what turns a carbon's 002 into (0 0 1)
   without anybody needing to know that (0 0 1) is systematically absent.
2. Each arm of the comparison starts FRESH. Refining the untextured
   pattern first and freeing r from there does not work: the early stages
   absorb the texture and `least_squares` then stops on xtol with r at
   exactly 1.000000.
3. The verdict has three values. On a phase with four reflections the
   ranking is right and the margins are a tenth of a per cent, so
   "accepted or rejected" either throws away a correct answer or turns
   0.1 % into a claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.xrd.pattern import Pattern
from ramancarbon.xrd.powder import candidate_axes
from ramancarbon.xrd.reference import load_library
from ramancarbon.xrd.rietveld import (
    TEXTURE_AXIS_MARGIN,
    TEXTURE_MIN_GAIN,
    PhaseModel,
    auto_refine,
    calculate_pattern,
    choose_texture_axes,
)


@pytest.fixture(scope="module")
def carbon():
    return {e.key: e.crystal for e in load_library()}["C_turbostratico_3.36"]


def _pattern(crystal, axis, r, seed: int = 7) -> Pattern:
    angles = np.linspace(15.0, 80.0, 2600)
    truth = PhaseModel(crystal=crystal, scale=0.024)
    truth.preferred_axis, truth.preferred_r = axis, r
    clean = calculate_pattern(angles, [truth], wavelength=1.540598,
                              background=[300.0, 0.0, 0.0])
    rng = np.random.default_rng(seed)
    noisy = np.maximum(rng.poisson(np.maximum(clean, 1.0)).astype(float), 1.0)
    return Pattern(two_theta=angles, intensity=noisy, name="sintético")


def test_the_carbon_offers_001_although_001_is_absent(carbon) -> None:
    """The point of deriving candidates instead of asking for them.

    A graphitic carbon shows 002 and no 001, so someone reading their own
    diffractogram concludes (001) cannot be the axis. Reducing the 002 to
    its primitive direction gives (0 0 1) anyway.
    """
    axes = candidate_axes(carbon, two_theta_range=(10.0, 90.0))
    assert (0, 0, 1) in axes
    assert axes[0] == (0, 0, 1)                 # from the strongest reflection


def test_an_axis_is_offered_once_however_many_orders_it_has(carbon) -> None:
    """002 and 004 are the same direction."""
    axes = candidate_axes(carbon, two_theta_range=(10.0, 90.0))
    assert len(axes) == len(set(axes))
    assert (0, 0, 2) not in axes and (0, 0, 4) not in axes


def test_sign_is_not_a_different_axis(carbon) -> None:
    """March-Dollase depends on cos², so (0 0 -1) is (0 0 1)."""
    axes = candidate_axes(carbon, two_theta_range=(10.0, 90.0))
    assert not any(tuple(-v for v in a) in axes for a in axes if any(a))


@pytest.mark.parametrize("axis, r", [((0, 0, 1), 0.45), ((1, 0, 0), 0.60)])
def test_the_true_axis_comes_top_of_the_ranking(carbon, axis, r) -> None:
    """The ranking is right even where the margin says it is not certain."""
    choice = choose_texture_axes(_pattern(carbon, axis, r), [carbon])[0]
    assert choice.ranking[0][0] == axis
    assert choice.axis == axis
    assert choice.verdict in ("textura", "eje ambiguo")
    assert choice.r < 1.0                       # plates, as built


def test_a_pattern_built_without_texture_is_not_given_any(carbon) -> None:
    choice = choose_texture_axes(_pattern(carbon, None, 1.0), [carbon])[0]
    assert choice.verdict == "sin textura"
    assert choice.axis is None
    assert choice.gain < TEXTURE_MIN_GAIN


def test_an_ambiguous_axis_says_so_instead_of_choosing(carbon) -> None:
    """Four reflections cannot separate the axes, and that is the answer."""
    choice = choose_texture_axes(_pattern(carbon, (0, 0, 1), 0.45), [carbon])[0]
    if choice.verdict == "eje ambiguo":
        assert not choice.accepted
        assert "no separa" in choice.reason
        assert "PROBABLE" in choice.describe()


def test_auto_reaches_a_texture_the_staged_protocol_cannot(carbon) -> None:
    """The measured failure: r stayed at exactly 1.0 through every stage.

    `auto_refine` with the true axis DECLARED came back with r = 1.000000
    and a Rwp identical to the untextured fit, because by the texture
    stage the widths and the scale had absorbed the texture and the
    solver terminated on xtol. Running the search first, from fresh
    starts, is what gets out of that basin.
    """
    pattern = _pattern(carbon, (0, 0, 1), 0.45)
    auto = auto_refine(pattern, [carbon], preferred_axis="auto")
    assert auto.phases[0].preferred_axis is not None
    assert any("eje" in w for w in auto.warnings)


def test_a_bad_string_is_refused_rather_than_ignored(carbon) -> None:
    with pytest.raises(ValueError, match="auto"):
        auto_refine(_pattern(carbon, None, 1.0), [carbon],
                    preferred_axis="vertical")


def test_the_thresholds_are_relative_fractions() -> None:
    assert 0.0 < TEXTURE_MIN_GAIN < 1.0
    assert 0.0 < TEXTURE_AXIS_MARGIN < TEXTURE_MIN_GAIN


def test_the_search_does_not_write_into_the_phases_it_is_given(carbon) -> None:
    """It is a measurement; applying it is `auto_refine`'s decision."""
    phase = PhaseModel(crystal=carbon)
    choose_texture_axes(_pattern(carbon, (0, 0, 1), 0.45), [phase])
    assert phase.preferred_axis is None
    assert phase.preferred_r == 1.0


def test_a_screening_arm_is_given_a_screening_budget() -> None:
    """The arms compare axes; they are not final refinements.

    Measured on a five-phase pattern: the untextured arm converged in 288
    evaluations, 24 per free parameter, while the same arm with one
    texture parameter added spent its whole 54 400-evaluation budget
    without converging -- six minutes for one of the forty arms a
    five-phase search runs.
    """
    from ramancarbon.xrd.rietveld import TEXTURE_ARM_ITERATIONS

    assert 24 < TEXTURE_ARM_ITERATIONS < 200


def test_a_fit_out_of_budget_hands_back_its_best_point() -> None:
    """Not wherever the optimiser happened to stop.

    `refine` tracked the best point already and restored it only when the
    USER stopped the fit. The same argument applies to running out of
    evaluations, and the consequence was measurable: a five-phase fit
    with one texture parameter free came back at Rwp 6.222 %, worse than
    the same model fitted without that parameter at all. One more free
    parameter cannot make a fit worse; where it stopped can.

    The property under test is the one that makes a starved fit usable:
    it may fail to improve, but it must not come back worse than the
    point it started from.
    """
    import numpy as np

    from ramancarbon.xrd.pattern import Pattern
    from ramancarbon.xrd.reference import load_library
    from ramancarbon.xrd.rietveld import (
        PhaseModel,
        build_parameters,
        calculate_pattern,
        free_kinds,
        refine,
    )

    crystal = next(iter(load_library())).crystal
    angles = np.linspace(20.0, 60.0, 900)
    rng = np.random.default_rng(3)
    counts = np.maximum(
        rng.poisson(400.0 + 30.0 * np.sin(angles / 7.0)).astype(float), 1.0)
    pattern = Pattern(two_theta=angles, intensity=counts, name="sintético")

    def rwp_of(models, parameters) -> float:
        background = [p.value for p in parameters if p.kind == "background"]
        calculated = calculate_pattern(
            angles, models, wavelength=pattern.wavelength,
            kalpha2_ratio=pattern.kalpha2_ratio, background=background)
        sigma = np.asarray(pattern.sigma, dtype=float)
        weights = 1.0 / np.maximum(sigma, 1e-9) ** 2
        return float(np.sqrt(
            np.sum(weights * (counts - calculated) ** 2)
            / np.sum(weights * counts ** 2)))

    model = PhaseModel(crystal=crystal)
    parameters = build_parameters([model], 6)
    for parameter in parameters:
        if parameter.name == "fondo_c0":
            parameter.value = float(np.percentile(counts, 5))
    free_kinds(parameters, ("scale", "background", "lattice", "profile_w"))
    before = rwp_of([model], parameters)

    starved = refine(pattern, [model], parameters=parameters,
                     background_order=6, max_iterations=2)
    assert np.isfinite(starved.r_wp)
    assert starved.r_wp <= before + 1e-9
