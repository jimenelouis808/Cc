"""Quality checks, batch statistics, bootstrap and interference matching."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ramancarbon.analysis.batch import (
    OUTLIER_Z,
    analyse_many,
    single_threaded_blas,
    summarise,
)
from ramancarbon.analysis.interference import (
    CORROBORATION,
    find_interferences,
    load_species,
)
from ramancarbon.analysis.quality import calibrate, check_quality, silicon_offset
from ramancarbon.analysis.report import analyse
from ramancarbon.core.peaks import find_peaks
from ramancarbon.core.preprocess import preprocess
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.examples.demo_data import make_demo
from ramancarbon.models.bootstrap import bootstrap_fit, position_of, ratio_of, width_of
from ramancarbon.models.deconvolution import build_model
from ramancarbon.models.fitting import fit_model
from ramancarbon.models.lineshapes import lorentzian


# -- quality -------------------------------------------------------------
def test_a_good_spectrum_raises_nothing():
    assert check_quality(make_demo("MWCNT", seed=1)).usable


def test_saturation_is_detected():
    """Clipping the top 3 % doubled I_D/I_G with no warning anywhere."""
    source = make_demo("MWCNT", seed=1)
    ceiling = float(np.percentile(source.intensity, 97))
    clipped = Spectrum(source.shift, np.minimum(source.intensity, ceiling),
                       laser_nm=532.0)
    report = check_quality(clipped)
    assert not report.usable
    assert any(i.key == "saturacion" for i in report.issues)
    assert report.saturated_fraction > 0.02


def test_under_resolution_is_detected():
    """At 16 cm-1 per point a true I_D/I_G of 1.0 came out as 3.9."""
    report = check_quality(make_demo("MWCNT", seed=1, step=16.0))
    assert not report.usable
    assert any(i.key == "resolucion" for i in report.issues)


def test_a_spectrum_with_no_variation_is_grave():
    x = np.arange(100.0, 3200.0)
    report = check_quality(Spectrum(x, np.full_like(x, 7.0)))
    assert any(i.key == "sin_senal" for i in report.issues)
    assert not report.usable


def test_missing_windows_are_reported_as_information_only():
    report = check_quality(
        make_demo("MWCNT", seed=1, low=800.0),
        required_windows=(("RBM", 120.0, 350.0),),
    )
    assert any(i.key == "cobertura_RBM" for i in report.issues)
    assert report.usable  # not covering a window is not a defect


def test_silicon_line_gives_the_calibration_offset():
    x = np.arange(400.0, 700.0)
    y = lorentzian(x, 522.9, 900.0, 4.0) + 20.0
    y = y + np.random.default_rng(0).normal(0.0, 2.0, x.size)
    spectrum = Spectrum(x, y, laser_nm=532.0)
    assert silicon_offset(spectrum) == pytest.approx(2.2, abs=0.3)
    assert silicon_offset(calibrate(spectrum)) == pytest.approx(0.0, abs=0.1)


def test_a_broad_feature_is_not_taken_for_silicon():
    """A wide bump near 520 is not the silicon phonon and must not be used
    to 'correct' the axis."""
    x = np.arange(400.0, 700.0)
    y = lorentzian(x, 523.0, 900.0, 60.0) + 20.0
    assert silicon_offset(Spectrum(x, y, laser_nm=532.0)) is None


def test_calibrate_refuses_to_invent_an_offset():
    with pytest.raises(ValueError, match="520.7"):
        calibrate(make_demo("grafeno_1L", seed=1, low=900.0))


# -- batch ---------------------------------------------------------------
def test_batch_finds_the_planted_outliers():
    spectra = []
    for i in range(16):
        s = make_demo("MWCNT", seed=i)
        s.name = f"p{i:02d}"
        if i in (3, 11):
            s.intensity[(s.shift > 1300) & (s.shift < 1400)] *= 2.2
        spectra.append(s)
    results, failures = analyse_many(spectra, workers=1)
    summary = summarise(results, failures=failures)
    assert summary.n_results == 16
    assert set(summary.statistics["ID_IG"].outliers) == {3, 11}


def test_robust_spread_survives_the_outliers():
    """The point of the median absolute deviation: two bad points in
    sixteen inflate the standard deviation but not the robust spread."""
    spectra = []
    for i in range(16):
        s = make_demo("MWCNT", seed=i)
        if i in (3, 11):
            s.intensity[(s.shift > 1300) & (s.shift < 1400)] *= 2.5
        spectra.append(s)
    stat = summarise(analyse_many(spectra, workers=1)[0]).statistics["ID_IG"]
    assert stat.mad_std < 0.2 * stat.std


def test_batch_warns_about_heterogeneity():
    rng = np.random.default_rng(1)
    spectra = []
    for i in range(12):
        s = make_demo("MWCNT", seed=i)
        s.intensity[(s.shift > 1300) & (s.shift < 1400)] *= 1.0 + rng.uniform(0, 1.5)
        spectra.append(s)
    summary = summarise(analyse_many(spectra, workers=1)[0])
    assert any("heterogénea" in w for w in summary.warnings)


def test_a_failing_spectrum_does_not_take_the_batch_down():
    good = [make_demo("MWCNT", seed=i) for i in range(3)]
    broken = Spectrum(np.arange(100.0, 200.0), np.ones(100), laser_nm=532.0,
                      name="corto")
    results, failures = analyse_many([*good, broken], workers=1)
    assert len(results) + len(failures) == 4


def test_blas_pinning_restores_the_environment():
    import os

    before = os.environ.get("OMP_NUM_THREADS")
    with single_threaded_blas():
        assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ.get("OMP_NUM_THREADS") == before


# -- bootstrap -----------------------------------------------------------
def test_bootstrap_intervals_exceed_the_analytic_errors():
    """The whole point: curvature-based standard errors are systematically
    too small for a derived quantity measured on correlated residuals."""
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    model = build_model(spectrum, "four_band")
    analytic = fit_model(spectrum, model).peak("G").errors["centre"]
    interval = bootstrap_fit(
        spectrum, model, {"pos_G": position_of("G")}, replicates=25
    ).get("pos_G")
    assert interval.half_width > analytic


def test_bootstrap_tracks_ratios_and_widths():
    spectrum, _ = preprocess(make_demo("MWCNT", seed=2))
    model = build_model(spectrum, "four_band")
    result = bootstrap_fit(
        spectrum,
        model,
        {"ID_IG": ratio_of("D", "G"), "gamma_G": width_of("G")},
        replicates=20,
    )
    assert result.get("ID_IG").low < result.get("ID_IG").value < result.get("ID_IG").high
    assert result.get("gamma_G").relative < 0.5


def test_bootstrap_says_what_it_does_not_cover():
    spectrum, _ = preprocess(make_demo("MWCNT", seed=3))
    result = bootstrap_fit(
        spectrum, build_model(spectrum, "three_band"),
        {"ID_IG": ratio_of("D", "G")}, replicates=15,
    )
    assert any("línea base" in w for w in result.warnings)


def test_bootstrap_is_reproducible():
    spectrum, _ = preprocess(make_demo("MWCNT", seed=4))
    model = build_model(spectrum, "three_band")
    kwargs = dict(quantities={"ID_IG": ratio_of("D", "G")}, replicates=15, seed=7)
    a = bootstrap_fit(spectrum, model, **kwargs).get("ID_IG")
    b = bootstrap_fit(spectrum, model, **kwargs).get("ID_IG")
    assert a.low == pytest.approx(b.low)


# -- interference --------------------------------------------------------
def test_interference_is_off_by_default():
    assert analyse(make_demo("SWCNT", seed=1)).interference.enabled is False


def test_a_clean_nanotube_loses_no_diameters():
    """The first version discarded four of five genuine radial breathing
    modes, calling them cobalt oxide and selenium."""
    plain = analyse(make_demo("SWCNT", seed=1))
    checked = analyse(make_demo("SWCNT", seed=1), check_interferences=True)
    assert len(checked.rbm.diameters) == len(plain.rbm.diameters)
    assert not checked.interference.excluded_from_rbm


def test_a_real_oxide_is_identified_from_several_lines():
    source = make_demo("SWCNT", seed=1)
    x = source.shift
    y = (source.intensity + lorentzian(x, 225.0, 140.0, 12.0)
         + lorentzian(x, 293.0, 190.0, 14.0) + lorentzian(x, 412.0, 90.0, 16.0))
    result = analyse(Spectrum(x, y, laser_nm=532.0, name="con óxido"),
                     check_interferences=True)
    assert "Fe2O3_hematite" in result.interference.species_present
    assert len(result.interference.excluded_from_rbm) == 2
    # and the genuine tubes survive
    assert result.rbm.diameters


def test_one_isolated_match_is_not_corroborated():
    """A coincidence in the crowded 100-400 window is likelier than the
    species, so a single line must not exclude anything."""
    source = make_demo("SWCNT", seed=1)
    x = source.shift
    y = source.intensity + lorentzian(x, 225.0, 140.0, 12.0)
    report = find_interferences(
        find_peaks(preprocess(Spectrum(x, y, laser_nm=532.0))[0]),
        spectrum_range=(90.0, 3200.0),
    )
    hematite = [m for m in report.matches if m.species.key == "Fe2O3_hematite"]
    assert hematite and not any(m.corroborated for m in hematite)
    assert not report.excluded_from_rbm


def test_unreacted_sulfur_is_called_out():
    source = make_demo("MWCNT", seed=1)
    x = source.shift
    y = (source.intensity + lorentzian(x, 152.0, 200.0, 8.0)
         + lorentzian(x, 219.0, 300.0, 8.0) + lorentzian(x, 471.0, 150.0, 9.0))
    report = find_interferences(
        find_peaks(preprocess(Spectrum(x, y, laser_nm=532.0))[0]),
        spectrum_range=(90.0, 3200.0),
    )
    assert "S8" in report.species_present
    assert any("NO se incorporó" in w for w in report.warnings)


def test_catalogue_loads_and_is_documented():
    species = load_species()
    assert len(species) >= 20
    assert all(s.source for s in species)
    assert 0.0 < CORROBORATION <= 1.0
    assert OUTLIER_Z > 0


# -- benchmarks ---------------------------------------------------------

def test_the_benchmark_runs_and_reports_something():
    """It is in the package so that a change making something four times
    slower is noticeable without anybody remembering to check."""
    from ramancarbon.benchmarks import run

    report = run(quick=True, repeats=1)
    assert len(report.measurements) >= 10
    assert all(m.seconds > 0 for m in report.measurements)
    assert {"python", "numpy", "scipy"} <= set(report.machine)
    text = report.text()
    assert "Raman" in text and "Difracción" in text and "Electroquímica" in text


def test_the_benchmark_reports_the_best_run_not_the_mean():
    """On a shared machine the mean measures whatever else was running."""
    from ramancarbon.benchmarks import measure

    calls = {"n": 0}

    def variable():
        import time

        calls["n"] += 1
        time.sleep(0.02 if calls["n"] == 1 else 0.001)
        return calls["n"]

    seconds, value = measure(variable, repeats=3)
    assert seconds < 0.015, "el mejor tiempo, no la media"
    assert value == 3


def test_the_benchmark_can_be_written_as_json(tmp_path):
    from ramancarbon.benchmarks import main

    destination = tmp_path / "tiempos.json"
    assert main(["--rapido", "--repeticiones", "1", "--json", str(destination)]) == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["medidas"] and payload["maquina"]["python"]
