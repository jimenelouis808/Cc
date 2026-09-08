"""Size and strain, whole-pattern fitting, and carbon's own measures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ramancarbon.examples.demo_data import make_xrd_demo
from ramancarbon.xrd.lebail import (
    OVERLAP_FRACTION,
    cell_from_extraction,
    figure_of_merit,
    le_bail,
    pawley,
)
from ramancarbon.xrd.microstructure import (
    GRAPHITE_D002,
    K_LA,
    K_LC,
    TURBOSTRATIC_D002,
    agreement,
    amorphous_fraction,
    carbon_microstructure,
    compare_methods,
    halder_wagner,
    size_strain_plot,
    williamson_hall,
)
from ramancarbon.xrd.powder import scherrer
from ramancarbon.xrd.reference import find_phase
from ramancarbon.xrd.rietveld import RefinementError

CU = 1.540598
ANGLES = [20.0, 30.0, 45.0, 60.0, 75.0, 90.0]


def widths(angles, size_nm, strain, k=0.89, wavelength=CU):
    """Widths that follow beta = K.lambda/(D cos t) + 4.eps.tan t exactly."""
    out = []
    for angle in angles:
        theta = math.radians(angle) / 2.0
        beta = (0.1 * k * wavelength / (size_nm * math.cos(theta))
                + 4.0 * strain * math.tan(theta))
        out.append(math.degrees(beta))
    return out


# -- size and strain ---------------------------------------------------

def test_williamson_hall_returns_what_was_put_in():
    result = williamson_hall(ANGLES, widths(ANGLES, 25.0, 0.002), CU)
    assert result.size_nm == pytest.approx(25.0, rel=0.01)
    assert result.strain == pytest.approx(0.002, rel=0.01)
    assert result.r_squared > 0.999


@pytest.mark.parametrize(
    "method", [williamson_hall, size_strain_plot, halder_wagner])
def test_every_method_recovers_a_pure_strain(method):
    """With no size broadening, all three use the same 4.eps.tan t term."""
    result = method(ANGLES, widths(ANGLES, 1e6, 0.003), CU)
    assert result.strain == pytest.approx(0.003, rel=0.02)


@pytest.mark.parametrize(
    "method,expected",
    [(williamson_hall, 15.0), (size_strain_plot, 15.0),
     (halder_wagner, 15.0 / 0.89)],
)
def test_every_method_recovers_a_pure_size(method, expected):
    """Halder-Wagner carries no shape constant, so its size differs by K."""
    result = method(ANGLES, widths(ANGLES, 15.0, 0.0), CU)
    assert result.size_nm == pytest.approx(expected, rel=0.02)
    assert result.strain == pytest.approx(0.0, abs=1e-4)


def test_a_flat_plot_is_a_perfect_fit_not_a_catastrophic_one():
    """Pure size gives a horizontal Williamson-Hall line, where the naive
    coefficient of determination is 0/0 and came out as -8.5."""
    assert williamson_hall(ANGLES, widths(ANGLES, 15.0, 0.0), CU).r_squared \
        == pytest.approx(1.0, abs=1e-6)


def test_the_three_methods_are_all_reported_because_they_disagree():
    results = compare_methods(ANGLES, widths(ANGLES, 25.0, 0.002), CU)
    assert len(results) == 3
    assert len({r.method for r in results}) == 3
    assert "difieren" in agreement(results) or "coinciden" in agreement(results)


def test_agreement_says_when_the_model_does_not_fit():
    results = compare_methods(ANGLES, widths(ANGLES, 50.0, 0.005), CU)
    assert "intervalo" in agreement(results) or "único" in agreement(results)


def test_a_negative_intercept_is_not_reported_as_a_negative_size():
    """A sample whose broadening falls with angle breaks the additive
    Lorentzian model, and the intercept goes negative."""
    falling = [w * (1.0 - 0.01 * a) for a, w in
               zip(ANGLES, widths(ANGLES, 20.0, 0.0))]
    result = williamson_hall(ANGLES, falling, CU)
    assert result.size_nm is None or result.size_nm > 0
    if result.size_nm is None:
        assert any("negativo" in w for w in result.warnings)


def test_reflections_narrower_than_the_instrument_are_dropped_by_name():
    result = williamson_hall(ANGLES, widths(ANGLES, 25.0, 0.002), CU,
                             instrument_fwhm=5.0)
    assert not result.available
    assert any("resolución" in w for w in result.warnings)


def test_fewer_than_three_reflections_cannot_separate_two_effects():
    result = williamson_hall(ANGLES[:2], widths(ANGLES[:2], 25.0, 0.002), CU)
    assert not result.available
    assert "tres reflexiones" in result.reason


def test_a_narrow_angular_range_is_warned_about():
    angles = [20.0, 22.0, 24.0]
    result = williamson_hall(angles, widths(angles, 25.0, 0.002), CU)
    assert any("poco ángulo" in w for w in result.warnings)


def test_a_size_above_what_a_laboratory_can_measure_says_so():
    result = williamson_hall(ANGLES, widths(ANGLES, 1e6, 0.003), CU)
    assert any("resolución" in w for w in result.warnings)


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="anchuras"):
        williamson_hall(ANGLES, [0.1, 0.2], CU)


# -- carbon ------------------------------------------------------------

def test_graphite_comes_out_as_graphite():
    two_theta = 2.0 * math.degrees(math.asin(CU / (2.0 * GRAPHITE_D002)))
    result = carbon_microstructure(two_theta, 0.3, 42.4, 0.5, wavelength=CU)
    assert result.d002 == pytest.approx(GRAPHITE_D002, rel=1e-4)
    assert result.graphitisation == pytest.approx(1.0, abs=0.01)
    assert not result.turbostratic
    assert result.lc_nm and result.layers and result.layers > 1


def test_a_turbostratic_carbon_is_named_and_qualified():
    two_theta = 2.0 * math.degrees(math.asin(CU / (2.0 * 3.46)))
    result = carbon_microstructure(two_theta, 3.0, wavelength=CU)
    assert result.turbostratic
    assert result.graphitisation is None
    assert any("turbostrático" in w for w in result.warnings)


def test_a_spacing_below_graphite_is_an_instrument_error_and_says_so():
    two_theta = 2.0 * math.degrees(math.asin(CU / (2.0 * 3.30)))
    result = carbon_microstructure(two_theta, 0.5, wavelength=CU)
    assert result.graphitisation is None
    assert any("MENOR" in w for w in result.warnings)
    assert result.d002 < GRAPHITE_D002


def test_the_layer_width_uses_the_two_dimensional_constant():
    """K = 1.84 for 100, not 0.89. Using 0.89 halves every published L_a."""
    result = carbon_microstructure(two_theta_100=42.4, fwhm_100=1.0,
                                   wavelength=CU)
    assert result.la_nm == pytest.approx(
        scherrer(1.0, 42.4, CU, k=K_LA), rel=1e-9)
    assert K_LA / K_LC > 2.0


def test_a_resolution_limited_peak_gives_no_size():
    result = carbon_microstructure(26.5, 0.05, wavelength=CU,
                                   instrument_fwhm=0.08)
    assert result.lc_nm is None
    assert any("resolución" in w for w in result.warnings)


def test_the_number_of_layers_follows_from_the_two_measurements():
    result = carbon_microstructure(26.5, 1.0, wavelength=CU)
    assert result.layers == pytest.approx(result.lc_nm * 10.0 / result.d002)
    assert "capas" in result.describe()


def test_the_graphitisation_scale_runs_between_the_two_spacings():
    middle = (GRAPHITE_D002 + TURBOSTRATIC_D002) / 2.0
    two_theta = 2.0 * math.degrees(math.asin(CU / (2.0 * middle)))
    result = carbon_microstructure(two_theta, wavelength=CU)
    assert result.graphitisation == pytest.approx(0.5, abs=0.02)


# -- amorphous content -------------------------------------------------

def test_an_internal_standard_measures_what_rietveld_cannot_see():
    """Half the sample amorphous: the refinement doubles everything
    crystalline, standard included."""
    # Half the sample amorphous plus 25 % standard leaves 62.5 % of the
    # mixture crystalline, of which the standard is 0.25/0.625 = 40 %.
    refined = {"muestra": 0.6, "patron": 0.4}
    result = amorphous_fraction(refined, "patron", standard_added=0.25)
    assert result["amorfo"] == pytest.approx(0.5, abs=0.02)
    assert result["muestra"] == pytest.approx(0.5, abs=0.02)


def test_no_amorphous_content_comes_back_as_none():
    refined = {"muestra": 0.8, "patron": 0.2}
    result = amorphous_fraction(refined, "patron", standard_added=0.2)
    assert result["amorfo"] == pytest.approx(0.0, abs=1e-9)


def test_a_standard_that_is_not_in_the_refinement_is_refused():
    with pytest.raises(ValueError, match="no está entre las fases"):
        amorphous_fraction({"a": 1.0}, "corindon", 0.2)


def test_an_impossible_standard_fraction_is_refused():
    with pytest.raises(ValueError, match="entre 0 y 1"):
        amorphous_fraction({"a": 0.5, "s": 0.5}, "s", 1.5)


# -- whole-pattern fitting ---------------------------------------------

@pytest.fixture(scope="module")
def mos2():
    return make_xrd_demo("MoS2_texturado"), find_phase("MoS2_2H")


@pytest.mark.parametrize("method", [le_bail, pawley])
def test_a_whole_pattern_fit_recovers_the_cell(mos2, method):
    pattern, crystal = mos2
    result = method(pattern, crystal, cycles=15)
    cell = cell_from_extraction(result, crystal)
    assert cell["a"] == pytest.approx(crystal.lattice.a, rel=2e-3)
    assert cell["c"] == pytest.approx(crystal.lattice.c, rel=2e-3)
    assert result.gof < 2.0


@pytest.mark.parametrize("method", [le_bail, pawley])
def test_a_cell_two_per_cent_wrong_is_refined_back(mos2, method):
    pattern, crystal = mos2
    wrong = crystal.with_lattice(crystal.lattice.scaled((1.02, 1.02, 0.985)))
    result = method(pattern, wrong, cycles=20)
    cell = cell_from_extraction(result, wrong)
    assert cell["a"] == pytest.approx(crystal.lattice.a, rel=3e-3)
    assert cell["c"] == pytest.approx(crystal.lattice.c, rel=3e-3)


def test_the_two_methods_agree_where_there_is_no_overlap(mos2):
    pattern, crystal = mos2
    first = cell_from_extraction(le_bail(pattern, crystal, cycles=15), crystal)
    second = cell_from_extraction(pawley(pattern, crystal, cycles=15), crystal)
    assert first["a"] == pytest.approx(second["a"], rel=1e-3)
    assert first["c"] == pytest.approx(second["c"], rel=1e-3)


def test_the_peak_width_is_recovered_and_not_inflated(mos2):
    """It came out four times too wide until the Ka2 satellite was
    modelled and the partition given enough passes."""
    pattern, crystal = mos2
    result = pawley(pattern, crystal, cycles=15)
    assert result.profile.w == pytest.approx(0.09 ** 2, rel=0.3)


def test_a_missing_phase_shows_up_as_a_terrible_fit():
    pattern = make_xrd_demo("FeSe_dos_fases")
    result = pawley(pattern, find_phase("FeSe_tetragonal"), cycles=12)
    assert result.r_wp > 0.15, "media muestra sin modelar tiene que verse"


def test_the_overlap_of_each_reflection_is_reported(mos2):
    pattern, crystal = mos2
    result = pawley(pattern, crystal, cycles=10)
    assert all(0.0 <= line.overlap <= 1.0 for line in result.lines)
    assert len(result.overlapped()) == sum(
        1 for line in result.lines if line.overlap > OVERLAP_FRACTION)


def test_every_fit_says_its_r_factor_cannot_be_compared_with_rietveld(mos2):
    pattern, crystal = mos2
    result = le_bail(pattern, crystal, cycles=5)
    assert any("Rietveld" in w for w in result.warnings)


def test_le_bail_admits_its_even_split(mos2):
    pattern, crystal = mos2
    assert any("partes iguales" in w
               for w in le_bail(pattern, crystal, cycles=5).warnings)


def test_pawley_gives_the_intensities_their_own_uncertainties(mos2):
    pattern, crystal = mos2
    result = pawley(pattern, crystal, cycles=10)
    assert all(line.uncertainty is not None for line in result.lines)
    assert any(line.uncertainty > 0 for line in result.lines)


def test_no_intensity_is_ever_negative(mos2):
    pattern, crystal = mos2
    for method in (le_bail, pawley):
        result = method(pattern, crystal, cycles=10)
        assert all(line.intensity >= 0 for line in result.lines)


def test_a_cell_that_produces_no_reflections_is_refused(mos2):
    pattern, crystal = mos2
    tiny = crystal.with_lattice(crystal.lattice.scaled((0.05, 0.05, 0.05)))
    with pytest.raises(RefinementError, match="ninguna reflexión"):
        pawley(pattern, tiny, cycles=2)


def test_the_figure_of_merit_prefers_a_cell_that_is_right(mos2):
    pattern, crystal = mos2
    wrong = crystal.with_lattice(crystal.lattice.scaled((1.15, 1.15, 1.15)))
    good = figure_of_merit(pawley(pattern, crystal, cycles=10))
    bad = figure_of_merit(pawley(pattern, wrong, cycles=3,
                                 refine_cell=False, refine_profile=False))
    assert good > bad


def test_the_difference_curve_is_available(mos2):
    pattern, crystal = mos2
    result = pawley(pattern, crystal, cycles=8)
    assert result.difference.shape == pattern.intensity.shape
    assert np.mean(np.abs(result.difference)) < 0.2 * np.mean(pattern.intensity)
