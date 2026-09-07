"""Structural indices, multi-wavelength combination, and the exports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ramancarbon.analysis.export import (
    COMPONENT_COLUMNS,
    analysis_to_dict,
    component_rows,
    export_analysis,
    export_batch,
    export_components,
    export_curves,
)
from ramancarbon.analysis.indices import amorphisation_stage, compute_indices
from ramancarbon.analysis.multiwavelength import compare_excitations, measure_dispersion
from ramancarbon.analysis.report import analyse
from ramancarbon.examples.demo_data import make_demo


# -- indices -----------------------------------------------------------
def test_g_width_is_reported_and_interpreted():
    result = analyse(make_demo("MWCNT", seed=1))
    entry = result.indices.indices["gamma_G"]
    assert entry.available
    assert entry.value == pytest.approx(48.0, abs=12.0)
    assert entry.interpretation


def test_beyssac_r2_needs_a_deconvolution():
    result = analyse(make_demo("MWCNT", seed=2))
    assert result.indices.get("R2") is not None
    assert 0.0 < result.indices.get("R2") < 1.0


def test_r2_refuses_the_geothermometer():
    result = analyse(make_demo("MWCNT", seed=3))
    assert any("termómetro" in w for w in result.indices.warnings)


@pytest.mark.parametrize(
    "g_position, id_ig, expected",
    [
        (1582.0, 0.02, "etapa 1 (grafito casi perfecto)"),
        (1598.0, 1.2, "etapa 1 (grafito → nanocristalino)"),
        (1555.0, 1.0, "etapa 1–2 (frontera)"),
        (1515.0, 0.5, "etapa 2 (nanocristalino → amorfo)"),
    ],
)
def test_amorphisation_stage(g_position, id_ig, expected):
    stage, reason = amorphisation_stage(g_position, id_ig)
    assert stage == expected
    assert reason


def test_stage_2_warns_that_the_ratio_inverts():
    _, reason = amorphisation_stage(1515.0, 0.5)
    assert "MENOR" in reason and "MÁS desorden" in reason


def test_indices_degrade_without_a_fit():
    from ramancarbon.analysis.assignment import assign_bands
    from ramancarbon.core.peaks import find_peaks
    from ramancarbon.core.preprocess import preprocess

    processed, _ = preprocess(make_demo("MWCNT", seed=4))
    assignment = assign_bands(processed, find_peaks(processed))
    indices = compute_indices(assignment, fit=None)
    assert indices.indices["R2"].available is False
    assert "deconvolución" in indices.indices["R2"].reason


# -- multi-wavelength ---------------------------------------------------
def test_dispersion_of_a_two_point_measurement_is_exact():
    slope, error = measure_dispersion({532.0: 1350.0, 633.0: 1331.5})
    assert slope == pytest.approx(50.0, abs=1.0)
    assert error is None  # no spare degree of freedom, and it says so


def test_measured_d_dispersion_matches_the_database():
    at_532 = analyse(make_demo("MWCNT", laser_nm=532.0, seed=5))
    at_633 = analyse(make_demo("MWCNT", laser_nm=633.0, seed=6))
    combined = compare_excitations([at_532, at_633])
    d = combined.dispersions["D"]
    assert d.slope == pytest.approx(50.0, abs=12.0)
    assert d.consistent


def test_g_dispersion_near_zero_is_reported_as_graphitic():
    at_532 = analyse(make_demo("MWCNT", laser_nm=532.0, seed=7))
    at_633 = analyse(make_demo("MWCNT", laser_nm=633.0, seed=8))
    combined = compare_excitations([at_532, at_633])
    assert "indicio de carbono amorfo" in combined.g_dispersion_verdict


def test_crystallite_sizes_from_two_lasers_agree():
    """I_D/I_G scales as lambda^4; if the correction is right the derived
    L_a must not depend on which laser measured it."""
    at_532 = analyse(make_demo("MWCNT", laser_nm=532.0, seed=9))
    at_633 = analyse(make_demo("MWCNT", laser_nm=633.0, seed=10))
    combined = compare_excitations([at_532, at_633])
    values = list(combined.crystallite_sizes.values())
    assert len(values) == 2
    assert max(values) / min(values) < 1.6
    assert "coinciden" in combined.crystallite_verdict


def test_single_wavelength_is_refused():
    one = analyse(make_demo("MWCNT", seed=11))
    with pytest.raises(ValueError, match="distintas"):
        compare_excitations([one, one])


# -- lineshape choice ---------------------------------------------------
def test_free_eta_recovers_the_generating_lineshape():
    """A pseudo-Voigt with free eta contains both limits, so the fitted eta
    is a measurement: the synthetic D3 is Gaussian and G is Lorentzian."""
    result = analyse(make_demo("MWCNT", seed=12), profile="pseudo_voigt")
    etas = {p.name: p.extra[0] for p in result.fit.peaks if p.extra_names == ("eta",)}
    assert etas["G"] == pytest.approx(1.0, abs=0.15)
    assert etas["D3"] == pytest.approx(0.0, abs=0.25)


def test_gaussian_only_fits_worse_than_pseudo_voigt():
    gaussian = analyse(make_demo("MWCNT", seed=13), profile="gaussian")
    voigt = analyse(make_demo("MWCNT", seed=13), profile="pseudo_voigt")
    assert voigt.fit.r_squared > gaussian.fit.r_squared


def test_profile_is_recorded_in_the_result():
    assert analyse(make_demo("MWCNT", seed=14), profile="gaussian").profile == "gaussian"


def test_unknown_profile_is_rejected():
    from ramancarbon.core.preprocess import preprocess
    from ramancarbon.models.deconvolution import build_model

    processed, _ = preprocess(make_demo("MWCNT", seed=15))
    with pytest.raises(ValueError, match="unknown lineshape"):
        build_model(processed, "three_band", profile="supergaussian")


# -- export -------------------------------------------------------------
def test_component_rows_carry_area_percentages():
    result = analyse(make_demo("MWCNT", seed=16))
    rows = component_rows(result.fit)
    assert len(rows) == len(result.fit.peaks)
    assert sum(r["area_pct"] for r in rows) == pytest.approx(100.0)


def test_components_csv_has_a_provenance_header(tmp_path):
    result = analyse(make_demo("MWCNT", seed=17))
    path = export_components(result.fit, tmp_path / "c.csv", result)
    text = path.read_text(encoding="utf-8")
    assert "# laser: 532 nm" in text
    assert "# procesado:" in text
    assert ",".join(COMPONENT_COLUMNS) in text


def test_curves_csv_has_one_column_per_component(tmp_path):
    result = analyse(make_demo("MWCNT", seed=18))
    path = export_curves(result.fit, tmp_path / "curves.csv", result)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    header = lines[0].split(",")
    for peak in result.fit.peaks:
        assert peak.name in header
    assert len(lines) - 1 == result.fit.x.size


def test_curve_columns_sum_to_the_total_fit(tmp_path):
    """Component curves include the background, so summing them
    double-counts it exactly once per component - which is the arithmetic a
    user needs to know about, so verify it holds."""
    result = analyse(make_demo("MWCNT", seed=19))
    fit = result.fit
    total = sum(p.curve(fit.x) for p in fit.peaks) + fit.background
    assert np.allclose(total, fit.fitted, atol=1e-6)


def test_json_export_is_complete_and_valid(tmp_path):
    result = analyse(make_demo("MWCNT", seed=20), profile="pseudo_voigt")
    payload = analysis_to_dict(result)
    for section in ("espectro", "identificacion", "bandas", "cocientes",
                    "ajuste", "indices"):
        assert section in payload
    assert payload["ajuste"]["componentes"]
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["espectro"]["laser_nm"] == 532.0


def test_export_analysis_writes_every_file(tmp_path):
    result = analyse(make_demo("MWCNT", seed=21))
    files = export_analysis(result, tmp_path)
    names = {f.name.split("_")[-1] for f in files}
    assert names == {"informe.txt", "componentes.csv", "curvas.csv",
                     "espectro.txt", "analisis.json"}
    assert all(f.stat().st_size > 100 for f in files)


def test_batch_export_unions_the_columns(tmp_path):
    results = [analyse(make_demo(k, seed=22)) for k in ("SWCNT", "MWCNT")]
    path = export_batch(results, tmp_path / "batch.csv")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    assert len(lines) == 3
    header = lines[0]
    assert "d_RBM_nm" in header and "gamma_G" in header


def test_batch_export_refuses_an_empty_list(tmp_path):
    with pytest.raises(ValueError, match="exportar"):
        export_batch([], tmp_path / "x.csv")
