"""Rietveld refinement: round trips, constraints and honest limits."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.examples.demo_data import make_xrd_demo
from ramancarbon.xrd.powder import Profile, simulate
from ramancarbon.xrd.reference import find_phase
from ramancarbon.xrd.report import analyse_pattern, theoretical_pattern
from ramancarbon.xrd.rietveld import (
    GOF_LIMIT,
    STAGES,
    PhaseModel,
    RefinementError,
    auto_refine,
    build_parameters,
    chebyshev_background,
    free_kinds,
    r_factors,
    refine,
)


@pytest.fixture(scope="module")
def two_phase():
    """A pattern whose true answer is known exactly."""
    return simulate(
        [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")],
        two_theta_range=(15.0, 72.0),
        step=0.025,
        scales=[1.0, 0.5],
        profile=Profile(u=0.010, v=-0.003, w=0.008, eta0=0.60),
        background=300.0,
        counts_at_max=12000.0,
        seed=3,
        name="dos_fases",
    )


@pytest.fixture(scope="module")
def refined(two_phase):
    return auto_refine(
        two_phase, [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]
    )


# -- the round trip ---------------------------------------------------


def test_the_refinement_recovers_the_pattern_it_was_given(refined):
    assert refined.converged
    assert refined.gof == pytest.approx(1.0, abs=0.15)
    assert refined.r_wp < 0.08


def test_the_refinement_recovers_the_lattice_parameters(refined):
    truth = {
        "FeSe_tetragonal": (3.7734, 5.5258),
        "grafito_2H": (2.4640, 6.7110),
    }
    for phase in refined.phases:
        lattice = phase.current_crystal().lattice
        a, c = truth[phase.crystal.name]
        assert lattice.a == pytest.approx(a, rel=2e-4)
        assert lattice.c == pytest.approx(c, rel=2e-4)


def test_the_refinement_recovers_the_profile(refined):
    for phase in refined.phases:
        assert phase.profile.w == pytest.approx(0.008, abs=0.003)
        assert phase.profile.eta0 == pytest.approx(0.60, abs=0.15)


def test_symmetry_ties_a_and_b_in_a_hexagonal_cell(refined):
    """Refining them separately returned 2.46359 and 2.46451 — two numbers
    differing by four standard errors that symmetry says are one."""
    for phase in refined.phases:
        lattice = phase.current_crystal().lattice
        assert lattice.a == lattice.b
    names = [p.name for p in refined.parameters if p.kind == "lattice"]
    assert not any(name.startswith("b[") for name in names)


def test_weight_fractions_sum_to_one(refined):
    fractions = refined.weight_fractions()
    assert all(value is not None for value in fractions.values())
    assert sum(fractions.values()) == pytest.approx(1.0)


# -- the machinery ----------------------------------------------------


def test_a_chebyshev_background_of_one_term_is_a_constant():
    angles = np.linspace(10.0, 80.0, 100)
    assert np.allclose(chebyshev_background(angles, [42.0]), 42.0)


def test_r_factors_are_zero_for_a_perfect_fit():
    observed = np.array([100.0, 200.0, 300.0])
    weights = 1.0 / observed
    r_p, r_wp, _, gof = r_factors(observed, observed.copy(), weights, 1)
    assert r_p == 0.0 and r_wp == 0.0 and gof == 0.0


def test_refining_nothing_is_an_error(two_phase):
    phases = [PhaseModel(crystal=find_phase("grafito_2H"))]
    parameters = build_parameters(phases)
    with pytest.raises(RefinementError):
        refine(two_phase, phases, parameters=parameters)


def test_parameters_can_be_freed_one_group_at_a_time(two_phase):
    phases = [PhaseModel(crystal=find_phase("grafito_2H"))]
    parameters = build_parameters(phases)
    free_kinds(parameters, ("scale",))
    assert sum(1 for p in parameters if p.free) == 1
    result = refine(two_phase, phases, parameters=parameters)
    assert result.free_parameters == 1


def test_manual_and_automatic_reach_the_same_place(two_phase):
    """The staged protocol is a route to the minimum, not a different one.

    The comparison only means anything when both sides free the SAME
    parameters. An earlier version freed seven groups by hand and compared
    against the nine the staged schedule ends with, then asserted they
    agreed to 0.01 in Rwp — which they did on one version of SciPy and not
    on the next, because a fit with fewer degrees of freedom simply cannot
    reach as low an Rwp. That is arithmetic, not a bug in either path.
    """
    crystals = [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]
    staged = auto_refine(two_phase, crystals)

    # Every group the schedule ever frees, minus texture: auto_refine skips
    # that one when no axis is given, and this comparison only means
    # anything if both sides free the same parameters.
    every_group = tuple(
        kind for _, kinds in STAGES for kind in kinds if kind != "preferred"
    )
    phases = [PhaseModel(crystal=c) for c in crystals]
    parameters = build_parameters(phases)
    for parameter in parameters:
        if parameter.name == "fondo_c0":
            parameter.value = float(np.percentile(two_phase.intensity, 5))
    free_kinds(parameters, every_group)
    manual = refine(two_phase, phases, parameters=parameters)

    assert manual.free_parameters == staged.free_parameters
    assert manual.r_wp == pytest.approx(staged.r_wp, abs=0.005)


def test_a_partial_parameter_set_cannot_beat_the_full_one(two_phase):
    """Sanity on the comparison above: fewer free parameters, higher Rwp."""
    crystals = [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]
    phases = [PhaseModel(crystal=c) for c in crystals]
    parameters = build_parameters(phases)
    for parameter in parameters:
        if parameter.name == "fondo_c0":
            parameter.value = float(np.percentile(two_phase.intensity, 5))
    free_kinds(parameters, ("scale", "background"))
    partial = refine(two_phase, phases, parameters=parameters)
    assert partial.r_wp >= auto_refine(two_phase, crystals).r_wp


def test_uncertainties_are_reported_for_free_parameters(refined):
    free = [p for p in refined.parameters if p.free]
    assert free
    assert all(p.error is not None and p.error >= 0.0 for p in free)


# -- honesty ----------------------------------------------------------


def test_a_missing_phase_shows_up_as_a_bad_fit(two_phase):
    """Rietveld does not discover phases; leaving one out must not look
    like a good fit that happens to be missing something."""
    partial = auto_refine(two_phase, [find_phase("grafito_2H")])
    assert partial.gof > GOF_LIMIT
    assert any("no describe los datos" in w for w in partial.warnings)


def test_the_weight_fraction_caveat_is_always_stated(refined):
    assert any("CRISTALINA E IDENTIFICADA" in w for w in refined.warnings)


def test_a_resolution_limited_phase_refuses_to_give_a_size(refined):
    sizes = refined.crystallite_sizes(instrument_fwhm=0.06)
    assert all(value is None for value in sizes.values())


def test_a_broadened_phase_gives_a_size_back():
    """A 15 nm crystallite broadens its peaks well past the instrument, and
    the size must come back close to the one that was put in."""
    crystal = find_phase("grafito_2H")
    pattern = simulate(
        crystal,
        two_theta_range=(15.0, 75.0),
        profile=Profile(u=0.02, v=0.0, w=0.35, eta0=0.8),
        background=200.0,
        counts_at_max=9000.0,
        seed=11,
    )
    result = auto_refine(pattern, [crystal])
    size = result.crystallite_sizes(instrument_fwhm=0.06)[crystal.name]
    assert size is not None and 10.0 < size < 40.0


def test_texture_is_refined_when_an_axis_is_given():
    pattern = make_xrd_demo("MoS2_texturado", seed=2)
    result = auto_refine(pattern, [find_phase("MoS2_2H")], preferred_axis=(0, 0, 1))
    phase = result.phases[0]
    assert phase.preferred_r < 0.9, "la textura del demo es fuerte y va hacia r<1"
    assert result.gof < 3.0


def test_without_a_texture_axis_no_texture_parameter_is_freed():
    pattern = make_xrd_demo("MoS2_texturado", seed=2)
    result = auto_refine(pattern, [find_phase("MoS2_2H")])
    assert all(not p.free for p in result.parameters if p.kind == "preferred")


# -- the orchestrator -------------------------------------------------


def test_analyse_pattern_identifies_then_refines():
    pattern = make_xrd_demo("CNT_FeSe", seed=6)
    result = analyse_pattern(pattern)
    assert result.refinement is not None
    assert result.calculated is not None
    assert result.difference is not None
    assert len(result.difference) == pattern.n
    found = {m.crystal.name for m in result.search.accepted}
    assert set(pattern.metadata["phases"]) <= found


def test_the_report_puts_the_difference_curve_before_the_r_factors():
    pattern = make_xrd_demo("CNT_FeSe", seed=6)
    text = analyse_pattern(pattern).report()
    assert "Mira PRIMERO la curva diferencia" in text
    # "Rwp" also appears in the prose above; compare against the R-factor
    # block itself.
    assert text.index("Diferencia:") < text.index("Rwp  =")


def test_the_report_row_carries_the_headline_numbers():
    row = analyse_pattern(make_xrd_demo("CNT_FeSe", seed=6)).to_dict()
    assert row["fases"]
    assert row["GOF"] > 0.0
    assert any(key.startswith("w_") for key in row)


def test_refusing_to_refine_without_phases_says_why():
    angles = np.arange(10.0, 80.0, 0.02)
    flat = np.random.default_rng(1).poisson(300.0, angles.size).astype(float)
    from ramancarbon.xrd.pattern import Pattern

    result = analyse_pattern(Pattern(angles, flat, counts=True, kalpha2_ratio=0.0))
    assert result.refinement is None
    assert any("no lo descubre" in w for w in result.warnings)


def test_theoretical_pattern_returns_a_comparable_curve():
    pattern = make_xrd_demo("CNT_FeSe", seed=6)
    calculated, difference, refinement = theoretical_pattern(
        pattern, [find_phase("FeSe_tetragonal"), find_phase("grafito_2H"),
                  find_phase("Se_trigonal")]
    )
    assert calculated.shape == pattern.intensity.shape
    assert np.allclose(difference, pattern.intensity - calculated)
    assert refinement.r_wp < 0.1
