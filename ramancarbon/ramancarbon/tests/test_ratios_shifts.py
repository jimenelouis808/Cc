"""Intensity ratios, structural conversions, and the shift analysis."""

from __future__ import annotations

import pytest

from ramancarbon.analysis.ratios import (
    TURNOVER_LD_NM,
    crystallite_size,
    defect_type,
    graphene_layers,
)
from ramancarbon.analysis.shifts import decompose_strain_doping


def test_crystallite_size_reproduces_tuinstra_koenig():
    """Cançado's area-based constant divided by Knight & White's
    height-based 4.4 nm must equal the D/G width ratio, ~3.8. Their
    agreement is the check that the area basis is the right reading."""
    la = crystallite_size(1.0, 514.5, basis="area").la_low_defect_nm
    assert la / 4.4 == pytest.approx(3.8, abs=0.2)


def test_crystallite_size_scales_as_lambda_to_the_fourth():
    a = crystallite_size(1.0, 532.0, basis="area").la_low_defect_nm
    b = crystallite_size(1.0, 633.0, basis="area").la_low_defect_nm
    assert b / a == pytest.approx((633.0 / 532.0) ** 4, rel=1e-6)


def test_defect_density_rises_with_the_ratio():
    low = crystallite_size(0.2, 532.0, basis="area").defect_density_cm2
    high = crystallite_size(1.0, 532.0, basis="area").defect_density_cm2
    assert high > low
    assert high / low == pytest.approx(5.0, rel=1e-6)


def test_both_branches_are_reported():
    """I_D/I_G is not monotonic in disorder; quoting only the low-defect
    branch can be wrong by an order of magnitude."""
    result = crystallite_size(0.5, 532.0, basis="area", g_fwhm=20.0)
    assert result.ld_low_defect_nm is not None
    assert result.ld_high_defect_nm is not None
    assert result.ld_high_defect_nm < TURNOVER_LD_NM < result.ld_low_defect_nm


def test_broad_g_selects_the_amorphous_branch():
    result = crystallite_size(0.8, 532.0, basis="area", g_fwhm=110.0)
    assert result.likely_branch == "amorphous"
    assert any("invierte" in w for w in result.warnings)


def test_narrow_g_selects_the_low_defect_branch():
    assert crystallite_size(0.3, 532.0, basis="area",
                            g_fwhm=18.0).likely_branch == "low-defect"


def test_missing_g_width_leaves_the_branch_ambiguous():
    assert crystallite_size(0.5, 532.0, basis="area").likely_branch == "ambiguous"


def test_height_basis_is_warned_about():
    result = crystallite_size(1.0, 532.0, basis="height")
    assert any("alturas" in w for w in result.warnings)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_ratio_rejected(bad):
    with pytest.raises(ValueError):
        crystallite_size(bad, 532.0)


@pytest.mark.parametrize(
    "ratio, expected",
    [(13.2, "sp3"), (7.1, "vacancy"), (3.4, "boundary"),
     (1.3, "on_site_substitutional")],
)
def test_defect_type_identification(ratio, expected):
    result = defect_type(ratio)
    assert result.best_match == expected
    assert result.confident


def test_ambiguous_defect_ratio_is_flagged():
    result = defect_type(10.0)
    assert not result.confident
    assert any("mezcla" in w for w in result.warnings)


def test_graphene_layers_prefers_width_over_ratio():
    """I_2D/I_G drops with doping alone; the 2D width does not."""
    verdict, reasons = graphene_layers(0.9, two_d_fwhm=27.0,
                                       two_d_single_lorentzian=True)
    assert verdict == "monocapa"
    assert any("dopado" in r for r in reasons)
    assert any("turbostrático" in r for r in reasons)


def test_pure_strain_decomposes_to_pure_strain():
    result = decompose_strain_doping(-10.0, -22.0)
    assert result.strain_component_g == pytest.approx(-10.0, abs=1e-6)
    assert result.doping_component_g == pytest.approx(0.0, abs=1e-6)
    assert result.strain_percent > 0  # softening means tension


def test_pure_doping_decomposes_to_pure_doping():
    result = decompose_strain_doping(10.0, 7.0)
    assert result.doping_component_g == pytest.approx(10.0, abs=1e-6)
    assert result.strain_component_g == pytest.approx(0.0, abs=1e-6)


def test_mixed_case_splits_into_both():
    result = decompose_strain_doping(5.0, -10.0)
    assert result.strain_component_g < 0 < result.doping_component_g
    assert result.strain_component_g + result.doping_component_g == pytest.approx(5.0)


def test_decomposition_warns_outside_graphene():
    result = decompose_strain_doping(5.0, 5.0, material_key="MWCNT")
    assert any("monocapa" in w for w in result.warnings)
