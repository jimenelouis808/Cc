"""The whole pipeline, on synthetic spectra of known composition."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.analysis.report import analyse
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.examples.demo_data import DEMO_KINDS, add_doping, make_demo


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("SWCNT", "SWCNT"),
        ("SWCNT_metalico", "SWCNT"),
        ("DWCNT", "DWCNT"),
        ("MWCNT", "MWCNT"),
        ("grafeno_1L", "graphene_1L"),
    ],
)
def test_classifier_identifies_each_material(kind, expected):
    result = analyse(make_demo(kind, seed=1))
    assert result.classification.best == expected


def test_classification_always_carries_its_evidence():
    result = analyse(make_demo("DWCNT", seed=2))
    assert result.classification.evidence
    assert all(e.statement for e in result.classification.evidence)


def test_dwcnt_is_recognised_by_paired_rbms():
    result = analyse(make_demo("DWCNT", seed=3))
    assert any(e.rule == "rbm_paired" for e in result.classification.evidence)
    assert [p for p in result.rbm.wall_pairs if p.plausible]


def test_metallic_g_minus_is_detected_as_breit_wigner_fano():
    result = analyse(make_demo("SWCNT_metalico", seed=4))
    assert result.g_split_diameter is not None
    assert result.g_split_diameter.parameterisation == "metallic"


def test_semiconducting_tube_is_not_called_metallic():
    result = analyse(make_demo("SWCNT", seed=5))
    assert result.g_split_diameter.parameterisation == "semiconducting"


def test_rbm_and_g_splitting_diameters_agree_for_a_single_wall():
    """Two independent routes to the same number; disagreement would mean
    the assignment is wrong."""
    result = analyse(make_demo("SWCNT", seed=6))
    rbm = result.rbm.diameters[0].diameter_nm
    g_split = result.g_split_diameter.diameter_nm
    assert abs(rbm - g_split) < 0.25


def test_absent_rbm_region_is_not_treated_as_absent_rbm():
    """The single check that stops every 400-3000 cm-1 spectrum being
    called multi-walled."""
    truncated = make_demo("SWCNT", seed=7, low=600.0)
    result = analyse(truncated)
    assert not result.rbm.covered
    assert any("RBM" in r for r in result.classification.blocked_rules)
    assert not any(e.rule == "rbm_absent" for e in result.classification.evidence)


def test_graphene_has_no_phantom_rbm():
    """Regression: noise maxima were reported as four RBM peaks."""
    result = analyse(make_demo("grafeno_1L", seed=8))
    assert result.rbm.covered
    assert not result.rbm.diameters


def test_oxide_gets_no_phantom_2d_band():
    """GO's conjugated network is broken, so there is no 2D to find; the
    fitter must not pin a component to the window edge and call it one."""
    result = analyse(make_demo("GO", seed=9))
    assert result.assignment.get("2D") is None


def test_tube_spectra_select_a_model_containing_g_minus():
    """Without a G- component the D band stretches to absorb it and
    I_D/I_G comes out several times too large."""
    result = analyse(make_demo("SWCNT_metalico", seed=10))
    assert result.comparison.best == "swcnt_full"
    assert result.id_ig < 0.35


def test_ratios_carry_both_bases():
    result = analyse(make_demo("MWCNT", seed=11))
    ratio = result.ratios["ID_IG"]
    assert ratio.available
    assert ratio.alternate is not None
    assert ratio.on_basis("height") != ratio.on_basis("area")


def test_dispersion_is_applied_at_785_nm():
    """The D band sits near 1312 cm-1 at 785 nm, not 1350."""
    result = analyse(make_demo("MWCNT", laser_nm=785.0, seed=12))
    assert result.assignment.position("D") == pytest.approx(1312.0, abs=6.0)
    assert result.classification.best == "MWCNT"


def test_analysis_without_a_laser_says_what_it_cannot_do():
    spectrum = make_demo("MWCNT", seed=13)
    spectrum.laser_nm = None
    result = analyse(spectrum)
    assert result.crystallite is None
    assert any("láser" in w for w in result.warnings)


def test_doping_shift_is_recovered():
    """A synthetic n-type signature — G up, 2D down — must come back as
    one, measured against its own undoped control."""
    pristine = make_demo("grafeno_1L", seed=14)
    doped = add_doping(pristine, delta_g=8.0, delta_2d=-12.0)
    control = analyse(pristine)
    result = analyse(doped, control=control)
    g_shift = result.shifts.shifts.get("G") or result.shifts.shifts.get("G+")
    assert g_shift.delta == pytest.approx(8.0, abs=2.5)
    assert result.shifts.shifts["2D"].delta == pytest.approx(-12.0, abs=3.0)
    assert any("tipo n" in text for text in result.shifts.interpretation)


def test_shift_analysis_always_warns_about_calibration():
    result = analyse(make_demo("grafeno_1L", seed=15))
    assert any("calibra" in w for w in result.shifts.warnings)


@pytest.mark.parametrize("kind", DEMO_KINDS)
def test_report_renders_for_every_material(kind):
    text = analyse(make_demo(kind, seed=16)).report()
    assert "IDENTIFICACIÓN" in text
    assert "COCIENTES DE INTENSIDAD" in text
    assert len(text.splitlines()) > 25


@pytest.mark.parametrize("kind", DEMO_KINDS)
def test_to_dict_is_flat_and_csv_safe(kind):
    row = analyse(make_demo(kind, seed=17)).to_dict()
    assert row["nombre"]
    for key, value in row.items():
        assert isinstance(value, (str, int, float, type(None))), key


def test_a_featureless_spectrum_does_not_crash_or_over_claim():
    rng = np.random.default_rng(0)
    x = np.arange(100.0, 3200.0)
    flat = Spectrum(x, 100.0 + rng.normal(0.0, 2.0, x.size), laser_nm=532.0,
                    name="ruido")
    result = analyse(flat)
    assert result.classification.confidence in {"baja", "insuficiente", "media"}
    assert result.report()
