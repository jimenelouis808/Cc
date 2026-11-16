"""Tests for the zero-coverage limit.

The load-bearing one is `test_the_limit_agrees_with_the_simulation`:
everything else checks that the arithmetic is self-consistent, and
only that one checks it against the thing it claims to be the limit
**of**. It is marked slow because it runs a Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from carbonforge.adsorption.gcmc import AVOGADRO, run_gcmc
from carbonforge.adsorption.henry import (
    GAS_CONSTANT_KJ,
    HenryReport,
    henry_constant,
)
from carbonforge.adsorption.potentials import (
    ADSORBATES,
    MASS_C,
    matched_temperature,
)
from carbonforge.builders.graphene import build_graphene_supercell


def _slit(separation: float) -> Atoms:
    """An AA-stacked graphene slit, as `validacion_rendijas` builds it."""
    sheet = build_graphene_supercell(nx=10, ny=6, vacuum=10.0)
    sheet.set_cell(np.diag([sheet.cell[0, 0], sheet.cell[1, 1], separation]))
    sheet.center(axis=2)
    layers = max(1, int(np.ceil(24.0 / separation)))
    stack = sheet * (1, 1, layers)
    stack.set_pbc((True, True, True))
    stack.wrap()
    return stack


@pytest.fixture(scope="module")
def slit_report() -> HenryReport:
    return henry_constant(_slit(8.0), temperature=77.0, samples=40_000,
                          seed=0)


def test_a_slit_binds_hydrogen(slit_report: HenryReport) -> None:
    """The sign of everything, which a factor error would flip."""
    assert slit_report.boltzmann_average > 1.0
    assert slit_report.mean_energy < 0.0
    assert slit_report.henry_coefficient > 0.0
    assert slit_report.isosteric_heat > 0.0


def test_the_heat_is_a_van_der_waals_well_on_carbon(
    slit_report: HenryReport,
) -> None:
    """A few kJ/mol, which is what this pair potential is worth.

    Not a literature comparison: the bound is derived from the pair
    the simulation itself uses. A single flat wall's well is a few
    hundred kelvin, a slit sees two of them, and ``q_st = -<U> + RT``,
    so anything past the tens of kJ/mol an open metal site reaches
    would mean the sample is wrong.
    """
    assert 2.0 < slit_report.isosteric_heat < 12.0
    assert not slit_report.warnings, slit_report.warnings


def test_the_heat_follows_the_weighted_mean_energy(
    slit_report: HenryReport,
) -> None:
    """``q_st = -<U> + RT`` and nothing else, in the stated units."""
    expected = ((-slit_report.mean_energy + slit_report.temperature)
                * GAS_CONSTANT_KJ)
    assert slit_report.isosteric_heat == pytest.approx(expected)


def test_the_loading_is_linear_in_pressure(slit_report: HenryReport) -> None:
    """It is a tangent at the origin; that is the whole claim."""
    assert slit_report.loading_at(2.0) == pytest.approx(
        2.0 * slit_report.loading_at(1.0))
    assert slit_report.loading_at(0.0) == 0.0


def test_a_wider_slit_binds_less_per_gram_at_zero_coverage() -> None:
    """Two walls within reach beat one, which is the groove result.

    At 8 Å a molecule on the midplane is in range of both walls; at
    20 Å it is in range of one. The ordering is the measurement that
    makes the Henry coefficient mean what the module says it means.
    """
    narrow = henry_constant(_slit(8.0), samples=40_000, seed=0)
    wide = henry_constant(_slit(20.0), samples=40_000, seed=0)
    assert narrow.isosteric_heat > wide.isosteric_heat
    assert narrow.henry_coefficient > wide.henry_coefficient


def test_the_half_filling_pressure_inverts_the_slope(
    slit_report: HenryReport,
) -> None:
    """Langmuir, stated as the model it is."""
    saturation = 20.0
    half = slit_report.half_filling_pressure(saturation)
    assert half == pytest.approx(saturation / slit_report.henry_coefficient)
    # At the half-filling pressure a Langmuir isotherm holds half the
    # saturation, so the tangent there holds exactly the saturation.
    assert slit_report.loading_at(half) == pytest.approx(saturation)
    with pytest.raises(ValueError, match="positive saturation"):
        slit_report.half_filling_pressure(0.0)


def test_a_cell_narrower_than_twice_the_cutoff_is_refused() -> None:
    """The same refusal the GCMC makes, for the same reason."""
    with pytest.raises(ValueError, match="only image within range"):
        henry_constant(_narrow(), samples=100)


def _narrow() -> Atoms:
    """A legal structure in a cell too short for the minimum image."""
    atoms = build_graphene_supercell(nx=4, ny=3, vacuum=10.0)
    atoms.set_cell(np.diag([10.0, 10.0, 10.0]))
    atoms.set_pbc(True)
    return atoms


def test_a_triclinic_cell_is_refused() -> None:
    atoms = _slit(8.0)
    cell = np.array(atoms.cell, dtype=float)
    cell[0, 1] = 2.0
    atoms.set_cell(cell)
    with pytest.raises(ValueError, match="triclinic"):
        henry_constant(atoms, samples=100)


def test_the_result_is_reproducible_from_the_seed() -> None:
    """Nothing stochastic here is allowed to be unseeded."""
    first = henry_constant(_slit(8.0), samples=20_000, seed=7)
    again = henry_constant(_slit(8.0), samples=20_000, seed=7)
    other = henry_constant(_slit(8.0), samples=20_000, seed=8)
    assert first.henry_coefficient == again.henry_coefficient
    assert first.henry_coefficient != other.henry_coefficient


def test_a_poorly_converged_average_is_reported_not_raised() -> None:
    """Degrade, do not crash -- and say so in the result."""
    report = henry_constant(_slit(20.0), samples=200, seed=3)
    assert report.henry_coefficient > 0.0
    assert any("standard error" in w for w in report.warnings)


def test_the_temperature_enters_twice() -> None:
    """The Boltzmann factor AND the Feynman-Hibbs correction.

    So a report cached at one temperature is wrong at another, which
    is worth a test because caching it is the obvious optimisation.

    **And `q_st` is NOT the quantity to test it on**, which this test
    asserted until the energy clip was fixed. `q_st = -<U> + RT`, and
    `RT` is 0.64 kJ/mol at 77 K against 2.44 at 293, so the heat RISES
    with temperature while the well the molecule sits in gets deeper
    as it falls (-573 K against -550 K here, the weighting
    concentrating on the best sites as the temperature drops). The
    earlier assertion passed only because the unclipped overflow was
    poisoning the warm case -- a test that was green for the reason it
    was meant to catch.
    """
    cold = henry_constant(_slit(8.0), temperature=77.0, samples=40_000)
    warm = henry_constant(_slit(8.0), temperature=293.0, samples=40_000)
    assert cold.boltzmann_average > warm.boltzmann_average
    assert cold.mean_energy < warm.mean_energy
    # The heat goes the other way, because of the RT term, and both
    # stay physical.
    assert warm.isosteric_heat > cold.isosteric_heat
    assert 2.0 < cold.isosteric_heat < 12.0
    assert 2.0 < warm.isosteric_heat < 12.0


@pytest.mark.slow
def test_the_limit_agrees_with_the_simulation() -> None:
    """The one test that checks this is the limit of the GCMC.

    At 1 kPa a slit is nearly empty, so the simulation must sit on the
    tangent; by 10 kPa it has started to fill and must fall below it.
    Both halves matter -- agreement alone could be two wrong numbers,
    and the departure is what says where the linear regime ends.
    """
    atoms = _slit(8.0)
    report = henry_constant(atoms, temperature=77.0, samples=200_000, seed=0)
    carbon_kg = len(atoms) * MASS_C / AVOGADRO * 1e-3

    def tangent(pressure: float) -> float:
        return report.loading_at(pressure) * carbon_kg * AVOGADRO

    low = run_gcmc(atoms, temperature=77.0, pressure_mpa=0.001,
                   equilibration=200_000, production=200_000, seed=1)
    assert low.mean_count == pytest.approx(tangent(0.001), rel=0.10)

    higher = run_gcmc(atoms, temperature=77.0, pressure_mpa=0.01,
                      equilibration=200_000, production=200_000, seed=1)
    assert higher.mean_count < 0.9 * tangent(0.01)


def test_the_block_size_cannot_change_the_answer() -> None:
    """Memory is bounded by blocks; the result is not a function of them.

    The points are drawn in one go and only the energy evaluation is
    chunked, so this is exact rather than close -- which is the
    property that makes the block size safe to tune. Unchunked, a
    million-point sample held a million neighbour lists and the
    process was killed outright.
    """
    atoms = _slit(8.0)
    whole = henry_constant(atoms, samples=20_000, seed=0, chunk=20_000)
    split = henry_constant(atoms, samples=20_000, seed=0, chunk=997)
    assert split.henry_coefficient == pytest.approx(
        whole.henry_coefficient, rel=1e-12)
    assert split.isosteric_heat == pytest.approx(
        whole.isosteric_heat, rel=1e-12)
    assert split.henry_error == pytest.approx(whole.henry_error, rel=1e-9)


def test_an_adsorbate_below_its_critical_temperature_is_flagged() -> None:
    """77 K suits hydrogen and is condensation for methane.

    The arithmetic still returns a number -- the function degrades
    rather than raising, as everything here does -- but a coefficient
    for a fluid that is not a gas is not comparable with one for a
    fluid that is, and that has to be said in the result rather than
    in a README.
    """
    atoms = _slit(8.0)
    cold = henry_constant(atoms, temperature=77.0,
                          adsorbate=ADSORBATES["CH4"], samples=20_000)
    assert any("critical temperature" in w for w in cold.warnings)

    warm = henry_constant(atoms, temperature=matched_temperature(
        ADSORBATES["CH4"]), adsorbate=ADSORBATES["CH4"], samples=20_000)
    assert not any("critical temperature" in w for w in warm.warnings)


def test_hydrogen_at_77_K_is_not_flagged() -> None:
    """The control: the temperature this whole repository works at."""
    report = henry_constant(_slit(8.0), temperature=77.0, samples=20_000)
    assert not any("critical" in w for w in report.warnings)


def test_matching_reduced_temperature_puts_hydrogen_back_at_77() -> None:
    """The anchor of the comparison, so it cannot drift."""
    assert matched_temperature(ADSORBATES["H2"]) == pytest.approx(77.0,
                                                                  rel=1e-3)
    for name in ("Ne", "Ar", "CH4", "N2"):
        pair = ADSORBATES[name]
        matched = matched_temperature(pair)
        assert matched > pair.critical_temperature
        assert matched / pair.critical_temperature == pytest.approx(
            77.0 / ADSORBATES["H2"].critical_temperature)


def test_carbon_is_derived_from_the_two_validated_pairs() -> None:
    """And the round trip is exact, masses included.

    `with_carbon(H2_H2)` would hand in a fluid-fluid reduced mass --
    half the molecule's -- and the Feynman-Hibbs term goes as 1/mu, so
    that is a silent factor of two in the quantum correction. H2_SELF
    exists to stop it, and this pins the difference.
    """
    from carbonforge.adsorption.potentials import (
        H2_C,
        H2_H2,
        carbon_self,
        with_carbon,
    )

    back = with_carbon(ADSORBATES["H2"])
    assert back.epsilon == pytest.approx(H2_C.epsilon, rel=1e-12)
    assert back.sigma == pytest.approx(H2_C.sigma, rel=1e-12)
    assert back.reduced_mass == pytest.approx(H2_C.reduced_mass)

    carbon = carbon_self()
    assert carbon.epsilon == pytest.approx(27.989, abs=1e-3)
    assert carbon.sigma == pytest.approx(3.402, abs=1e-3)

    trap = with_carbon(H2_H2)
    assert trap.reduced_mass == pytest.approx(0.5 * back.reduced_mass)


def test_a_bigger_molecule_prefers_a_wider_slit() -> None:
    """The experiment's hypothesis, as a test on its cheapest case.

    At matched reduced temperature the only thing differing between
    two adsorbates is the molecule. Methane's sigma against carbon is
    3.57 A against hydrogen's 3.18, so the slit that suits it is
    wider -- and if that ordering ever flipped, every conclusion drawn
    from the scan would be wrong.
    """
    pair = ADSORBATES["CH4"]
    temperature = matched_temperature(pair)
    narrow = henry_constant(_slit(6.6), temperature=temperature,
                            adsorbate=pair, samples=40_000)
    wider = henry_constant(_slit(7.5), temperature=temperature,
                           adsorbate=pair, samples=40_000)
    assert wider.henry_coefficient > narrow.henry_coefficient


def test_the_weighted_mean_energy_is_physical_at_any_temperature() -> None:
    """The clip has to reach both sums, and this is why.

    An overlapping insertion has a finite but astronomic energy --
    1e20 K and up from the r^-12 term. The Boltzmann weight is
    computed from a CLIPPED energy, so weighting the unclipped value
    by it multiplies a vanishing number by an enormous one. At 77 K
    the factor is exp(-130) and the product underflows to nothing; at
    352 K it is exp(-28) and it does not, and the mean well depth came
    back as -4.6e9 kJ/mol. So the fault was invisible at the only
    temperature anything in this repository had been measured at,
    which is exactly why it needs a test at another one.
    """
    atoms = _slit(7.0)
    for name in ("H2", "Ne", "N2", "Ar", "CH4"):
        pair = ADSORBATES[name]
        report = henry_constant(atoms, temperature=matched_temperature(pair),
                                adsorbate=pair, samples=20_000)
        assert -1.0 < report.isosteric_heat < 40.0, (
            f"{name}: q_st = {report.isosteric_heat:.3g} kJ/mol")
        assert report.mean_energy < 0.0, name
        assert not report.warnings, (name, report.warnings)


def test_an_implausible_heat_is_caught_whichever_sign_it_has() -> None:
    """The guard read `heat > limit` and the bug produced -4.6e9.

    A sanity bound that only looks one way is not a sanity bound.
    """
    from carbonforge.adsorption.henry import (
        IMPLAUSIBLE_HEAT_KJ,
        HenryReport,
    )

    for heat in (IMPLAUSIBLE_HEAT_KJ + 1.0, -IMPLAUSIBLE_HEAT_KJ - 1.0):
        report = HenryReport(
            temperature=77.0, henry_coefficient=1.0, henry_error=0.0,
            isosteric_heat=heat, boltzmann_average=1.0, mean_energy=-1.0,
            samples=10)
        assert abs(report.isosteric_heat) > IMPLAUSIBLE_HEAT_KJ
