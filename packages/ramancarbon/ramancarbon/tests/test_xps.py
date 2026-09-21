"""Tests for the XPS section.

Every test here protects one of the barriers in CLAUDE.md, and each of
those barriers exists because its absence produced a wrong answer that
looked right.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.examples.demo_data import make_xps_demo
from ramancarbon.xps.background import estimate_background, shirley_background
from ramancarbon.xps.calibrate import calibrate, calibrate_to_state
from ramancarbon.xps.elements import load_xps_database
from ramancarbon.xps.fitting import XPSComponent, XPSModel, fit_region
from ramancarbon.xps.io import (
    read_spe,
    read_vamas,
    read_xps,
    read_xps_text,
    write_vamas,
)
from ramancarbon.xps.lineshapes import ds, gl, profile_fwhm, window_area
from ramancarbon.xps.presets import count_model, state_model
from ramancarbon.xps.quantify import LineArea, quantify, survey_areas
from ramancarbon.xps.report import analyse_xps
from ramancarbon.xps.spectrum import XPSError, XPSSpectrum, source_energy
from ramancarbon.xps.survey import find_survey_peaks, identify
from ramancarbon.xps.tables import components_table, fit_table, read_components, write_fit


@pytest.fixture(scope="module")
def demo():
    return make_xps_demo("NCNT_FeSe", seed=11)


@pytest.fixture(scope="module")
def database():
    return load_xps_database()


# ----------------------------------------------------------------------
# the scales
# ----------------------------------------------------------------------
def test_an_auger_line_moves_233_eV_between_the_two_anodes():
    """Auger lines sit at fixed KINETIC energy. The C KLL is at 1219 eV of
    apparent binding energy with aluminium and 986 with magnesium, and
    confusing the two puts a whole element where it is not."""
    axis = np.linspace(0.0, 1400.0, 1401)
    counts = np.ones_like(axis)
    al = XPSSpectrum(axis, counts, photon_energy=source_energy("Al"))
    mg = XPSSpectrum(axis, counts, photon_energy=source_energy("Mg"))
    assert float(al.binding_of_kinetic(263.0)) == pytest.approx(1219.1, abs=0.1)
    assert float(mg.binding_of_kinetic(263.0)) == pytest.approx(986.1, abs=0.1)
    difference = float(al.binding_of_kinetic(263.0) - mg.binding_of_kinetic(263.0))
    assert difference == pytest.approx(233.0, abs=0.1)


def test_an_auger_line_has_no_binding_energy_without_a_source(database):
    group = database.element("C").auger[0]
    assert not hasattr(group, "energy_ev")
    assert group.binding_at(1486.6) == pytest.approx(1486.6 - 263.0 - 4.5)


def test_a_kinetic_scale_is_refused_without_the_photon_energy():
    spectrum = XPSSpectrum(np.linspace(280, 300, 50), np.ones(50))
    with pytest.raises(XPSError, match="energía del fotón"):
        _ = spectrum.kinetic_energy


def test_counts_per_second_cannot_be_judged_against_poisson_noise():
    """A spectrum in counts per second has had its statistics divided away.
    Without the dwell time there is no Poisson expectation, and the honest
    answer is no verdict rather than a wrong one."""
    rng = np.random.default_rng(0)
    axis = np.linspace(280.0, 300.0, 401)
    counts = rng.poisson(10000.0, 401).astype(float)
    plain = XPSSpectrum(axis, counts)
    assert plain.looks_smoothed() is None            # honest Poisson data
    rate = XPSSpectrum(axis, counts / 0.2, intensity_unit="cuentas/s")
    assert rate.accumulated_counts is None
    assert rate.looks_smoothed() is None
    with_dwell = XPSSpectrum(axis, counts / 0.2, intensity_unit="cuentas/s",
                             dwell_s=0.2)
    assert with_dwell.accumulated_counts == pytest.approx(counts)
    smoothed = XPSSpectrum(axis, np.convolve(counts, np.ones(9) / 9, "same"))
    assert "suavizado" in (smoothed.looks_smoothed() or "")


# ----------------------------------------------------------------------
# lineshapes
# ----------------------------------------------------------------------
def test_the_gl_product_reduces_to_its_two_limits():
    x = np.linspace(-10, 10, 2001)
    gaussian = np.exp(-4 * np.log(2) * (x / 1.5) ** 2)
    lorentzian = 1.0 / (1.0 + 4.0 * (x / 1.5) ** 2)
    assert np.allclose(gl(x, 0, 1, 1.5, 0.0), gaussian)
    assert np.allclose(gl(x, 0, 1, 1.5, 1.0), lorentzian)


def test_the_gl_width_parameter_is_not_the_width():
    """The product of a Gaussian and a Lorentzian is narrower than either,
    by up to 5 %. A width quoted without its lineshape is 5 % wrong and
    always in the same direction."""
    assert profile_fwhm("gl", 2.0, (0.0,)) == pytest.approx(2.0, abs=0.01)
    assert profile_fwhm("gl", 2.0, (1.0,)) == pytest.approx(2.0, abs=0.01)
    assert profile_fwhm("gl", 2.0, (0.5,)) < 1.93


def test_doniach_sunjic_is_a_lorentzian_at_zero_asymmetry():
    x = np.linspace(-20, 20, 4001)
    assert np.allclose(ds(x, 0, 1, 1.5, 0.0), 1.0 / (1.0 + (x / 0.75) ** 2))


def test_the_doniach_sunjic_area_diverges_with_its_window():
    """Its tails fall as |u|^(α−1), so there is no area over the real line.
    Any 'DS area' is a windowed area and quoting one without the window is
    meaningless — the same rule this package applies to Breit-Wigner-Fano."""
    narrow = window_area("ds", 1.0, 1.0, (0.15,), (-20, 20), 0.0, 200001)
    wide = window_area("ds", 1.0, 1.0, (0.15,), (-2000, 2000), 0.0, 200001)
    assert wide > 1.9 * narrow


# ----------------------------------------------------------------------
# backgrounds
# ----------------------------------------------------------------------
def _region_with_known_shirley(seed: int = 1):
    rng = np.random.default_rng(seed)
    axis = np.linspace(280.0, 296.0, 321)
    peaks = (20000 * gl(axis, 284.8, 1, 1.0, 0.3)
             + 4000 * gl(axis, 286.3, 1, 1.2, 0.3))
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (peaks[1:] + peaks[:-1]) * np.diff(axis))])
    background = 2000 + 1500 * cumulative / cumulative[-1]
    counts = rng.poisson(peaks + background).astype(float)
    return axis, counts, peaks, background


def test_shirley_steps_up_and_recovers_a_known_background():
    axis, counts, peaks, truth = _region_with_known_shirley()
    found = shirley_background(axis, counts)
    assert found.values[-1] > found.values[0]        # up in binding energy
    error = np.max(np.abs(found.values - truth))
    assert error < 0.02 * (truth[-1] - truth[0]) + 30
    area = np.trapezoid(found.subtract(counts), axis)
    assert area == pytest.approx(np.trapezoid(peaks, axis), rel=0.02)


def test_a_correct_shirley_raises_no_complaint():
    """One noise level below the background is where one point in six sits.
    A one-sigma test fires on every good subtraction there is."""
    axis, counts, _, _ = _region_with_known_shirley(seed=2)
    assert shirley_background(axis, counts).warnings == []


def test_the_background_is_anchored_on_data_minus_peaks():
    """With an asymmetric lineshape the peak's own tail is inside the window
    edge. Anchoring on the raw data counts it twice and leaves a systematic
    residual at the edge, where nobody looks."""
    axis = np.linspace(280.0, 296.0, 321)
    envelope = 5000 * gl(axis, 288.0, 1, 1.2, 0.3) + 900.0   # a flat tail
    counts = envelope + 2000.0
    with_peaks = shirley_background(axis, counts, envelope=envelope)
    assert with_peaks.endpoints[0] == pytest.approx(2000.0, abs=5.0)
    without = shirley_background(axis, counts)
    assert without.endpoints[0] > with_peaks.endpoints[0] + 800


def test_a_tougaard_over_a_narrow_region_says_so(demo):
    region = next(item for item in demo if item.region == "C 1s")
    found = estimate_background(region, region.range, "tougaard")
    assert any("Tougaard necesita" in text for text in found.warnings)


# ----------------------------------------------------------------------
# doublets and constraints
# ----------------------------------------------------------------------
def test_a_doublet_is_one_component_with_a_fixed_ratio(database):
    line = database.element("Fe").primary_line
    components = line.components()
    assert len(components) == 2
    assert components[0][1] == pytest.approx(711.0)
    assert components[1][1] == pytest.approx(711.0 + 13.1)
    assert components[0][2] == pytest.approx(2 / 3)
    assert components[1][2] == pytest.approx(1 / 3)


def test_every_doublet_matches_its_degeneracy(database):
    """1:2 for p, 2:3 for d, 3:4 for f. These are properties of the atom,
    not parameters, and a database entry that disagrees is a typo."""
    for line in database.all_lines():
        if line.doublet is None:
            continue
        expected = line.doublet.expected_ratio
        if expected is not None:
            assert line.doublet.ratio == pytest.approx(expected, abs=0.01), line.label


def test_a_doublet_component_keeps_its_ratio_through_a_fit(demo):
    region = next(item for item in demo if item.region == "Se 3d5/2")
    model = state_model(region, "Se 3d5/2", ["selenide", "Se0", "SeOx"])
    result = fit_region(region, model)
    for component in result.components:
        assert component.doublet is not None
        assert component.doublet.ratio == pytest.approx(2 / 3, abs=0.01)


def test_widths_are_linked_only_where_the_literature_agrees(demo):
    """Fe(III) is multiplet-broadened to 1.5-4.5 eV and Fe-Se sits at
    0.9-2.0. Tied together they fit a common width, leave a visible residual
    on both, and raise chi-squared by a factor of fifty."""
    region = next(item for item in demo if item.region == "Fe 2p3/2")
    model = state_model(region, "Fe 2p3/2", ["Fe-Se", "Fe3+"])
    assert not [link for link in model.links if link.target_parameter == "fwhm"]
    nitrogen = next(item for item in demo if item.region == "N 1s")
    tied = state_model(nitrogen, "N 1s", ["pyridinic", "pyrrolic", "graphitic"])
    assert len([link for link in tied.links
                if link.target_parameter == "fwhm"]) == 2


# ----------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------
def test_a_region_comes_back_with_the_states_that_went_in(demo):
    region = next(item for item in demo if item.region == "N 1s")
    truth = region.metadata["true_states"]
    result = fit_region(region, state_model(region, "N 1s", list(truth)))
    assert result.reduced_chi2 < 2.0
    total = sum(truth.values())
    found = {item.state: item.area_fraction for item in result.components}
    for key, share in truth.items():
        assert found[key] == pytest.approx(share / total, abs=0.04)


def test_too_few_components_shows_up_in_chi_squared_and_the_residual(demo):
    """This is the one statistic in XPS that can say 'this fit is wrong' out
    loud. R-squared sits at 0.999 for a visibly bad fit."""
    region = next(item for item in demo if item.region == "N 1s")
    poor = fit_region(region, state_model(region, "N 1s",
                                          ["pyridinic", "pyrrolic"]))
    good = fit_region(region, state_model(
        region, "N 1s", ["pyridinic", "pyrrolic", "graphitic", "N-oxide"]))
    assert poor.reduced_chi2 > 10 * good.reduced_chi2
    assert poor.durbin_watson < 0.5 < good.durbin_watson
    # Meanwhile R² barely moves: it goes from 0.95 to 0.999 while χ² moves
    # by a factor of fifty. That is why the report leads with χ² and DW.
    assert poor.r_squared > 0.95
    assert good.r_squared - poor.r_squared < 0.05


def test_the_component_count_is_taken_from_the_regions_own_shoulders(demo):
    """Ranking states by the intensity at their tabulated position picks a
    state that sits BETWEEN two real peaks, because its neighbours put
    intensity there: N-metal at 399.3 over the N-oxide at 403.5 that is in
    the spectrum as a peak of its own."""
    region = next(item for item in demo if item.region == "N 1s")
    model, notes = count_model(region, "N 1s", 4)
    chosen = {component.state for component in model.components}
    assert chosen == {"pyridinic", "pyrrolic", "graphitic", "N-oxide"}
    assert "N-metal" not in chosen
    assert any("hombro" in note for note in notes)


def test_the_number_of_components_can_be_left_to_the_region(demo):
    region = next(item for item in demo if item.region == "N 1s")
    model, _ = count_model(region, "N 1s", None)
    assert len(model.components) == 4


def test_a_fit_narrower_than_the_resolution_is_flagged(demo):
    region = next(item for item in demo if item.region == "N 1s")
    region = region.with_counts(region.counts, "prueba")
    region.pass_energy = 200.0                       # a very wide analyser
    result = fit_region(region, state_model(region, "N 1s", ["pyridinic"]))
    assert any("resolución del equipo" in text for text in result.warnings)


def test_a_component_below_the_noise_is_called_out():
    """A least-squares fit ALWAYS returns as many peaks as it is given."""
    rng = np.random.default_rng(3)
    axis = np.linspace(394.0, 406.0, 241)
    counts = rng.poisson(3000.0, axis.size).astype(float)
    flat = XPSSpectrum(axis, counts, photon_energy=1486.6, pass_energy=26.0,
                       region="N 1s")
    model = XPSModel(
        [XPSComponent(name="a", centre=399.0, height=50.0, fwhm=1.3),
         XPSComponent(name="b", centre=401.0, height=50.0, fwhm=1.3)],
        window=(394.0, 406.0), background="shirley", region_label="N 1s",
    )
    result = fit_region(flat, model)
    assert any("no llegan a" in text for text in result.warnings)


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------
def test_charge_referencing_recovers_a_known_shift():
    rng = np.random.default_rng(7)
    axis = np.linspace(278.0, 300.0, 441)
    centre = 284.8 + 2.7
    peaks = 15000 * gl(axis, centre, 1, 1.4, 0.3)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (peaks[1:] + peaks[:-1]) * np.diff(axis))])
    counts = rng.poisson(peaks + 2500 + 700 * cumulative / cumulative[-1])
    spectrum = XPSSpectrum(axis, counts.astype(float), photon_energy=1486.6,
                           pass_energy=26.0, region="C 1s")
    _, calibration = calibrate(spectrum, "C1s_adventitious")
    assert calibration.shift_ev == pytest.approx(-2.7, abs=0.05)


def test_referencing_on_a_component_beats_referencing_on_the_peak():
    """In a sample made of carbon the C 1s is an envelope, and its centroid
    is not the position of any one state. 284.8 is the adventitious value;
    the sample's own sp2 carbon is at 284.4."""
    shifted = make_xps_demo("NCNT_FeSe", seed=12, charge_shift=1.8)
    region = next(item for item in shifted if item.region == "C 1s")
    _, whole = calibrate(region, "C1s_adventitious")
    _, component = calibrate_to_state(
        region, "C 1s", "C-C sp2", ["C-C sp2", "C-O", "C=O", "O-C=O"])
    assert abs(component.shift_ev + 1.8) < 0.2
    assert abs(component.shift_ev + 1.8) < abs(whole.shift_ev + 1.8) / 3


def test_the_shift_travels_with_the_spectrum():
    """A binding energy quoted without saying what put the axis there is
    not a measurement."""
    axis = np.linspace(280.0, 300.0, 201)
    spectrum = XPSSpectrum(axis, np.ones(201) * 100.0)
    moved = spectrum.shifted(-1.25, "C 1s a 284.8 eV")
    assert moved.metadata["charge_shift_ev"] == pytest.approx(-1.25)
    assert "C 1s a 284.8 eV" in moved.history[0]


# ----------------------------------------------------------------------
# survey
# ----------------------------------------------------------------------
def test_the_survey_threshold_finds_nothing_in_pure_noise():
    """Calibrated on Poisson noise, as the Raman and XRD thresholds are."""
    axis = np.arange(0.0, 1200.5, 0.5)
    total = 0
    for seed in range(6):
        rng = np.random.default_rng(200 + seed)
        level = 2000 + 6000 * (axis / 1200.0) ** 2
        spectrum = XPSSpectrum(axis, rng.poisson(level).astype(float),
                               photon_energy=1486.6, dwell_s=0.05, sweeps=10)
        peaks, _ = find_survey_peaks(spectrum)
        total += len(peaks)
    assert total == 0


def test_the_survey_identifies_what_is_there_and_nothing_else(demo):
    result = identify(demo[0])
    assert set(result.symbols()) == {"C", "N", "O", "Fe", "Se"}


def test_cobalt_does_not_ride_in_on_irons_auger_group(demo):
    """With an aluminium anode one of iron's LMM groups lands within 2 eV of
    cobalt's 2p3/2. An iron sample must not 'contain cobalt'."""
    result = identify(demo[0])
    assert "Co" not in result.symbols()
    relegated = {item.symbol for item in result.uncorroborated}
    assert "Co" in relegated
    reason = next(item for item in result.uncorroborated if item.symbol == "Co")
    assert reason.notes and reason.confidence == "baja"


def test_an_element_needs_a_line_nobody_else_explains():
    """The exclusivity rule itself, on a spectrum made only of iron: its
    2p, its 3s and the LMM group that sits on cobalt's 2p3/2. Cobalt's main
    line matches — and it is relegated anyway, because every peak it can
    point at belongs to iron."""
    axis = np.arange(0.0, 1000.0, 0.5)
    counts = np.full(axis.size, 2000.0)
    for centre, height in ((92.0, 6000.0), (711.0, 40000.0), (724.1, 20000.0),
                           (779.1, 9000.0)):
        counts = counts + height * gl(axis, centre, 1.0, 2.0, 0.3)
    rng = np.random.default_rng(5)
    spectrum = XPSSpectrum(axis, rng.poisson(counts).astype(float),
                           photon_energy=1486.6, dwell_s=0.05, sweeps=10,
                           region="Survey")
    result = identify(spectrum, elements=["Fe", "Co"])
    assert result.symbols() == ["Fe"]
    cobalt = next(item for item in result.uncorroborated if item.symbol == "Co")
    assert any("Fe" in note for note in cobalt.notes)


def test_a_doublet_needs_its_strong_component(demo):
    """Sulphur's 2p1/2 and selenium's 3p1/2 are 0.1 eV apart. Without this
    rule every selenide contains sulphur."""
    result = identify(demo[0])
    assert "S" not in result.symbols()


# ----------------------------------------------------------------------
# quantification
# ----------------------------------------------------------------------
def test_the_transmission_correction_improves_every_element(demo):
    survey = demo[0]
    truth = survey.metadata["true_composition_at"]
    # Every element that is in the sample: dropping one renormalises the
    # rest, and then the comparison is against a composition that no longer
    # sums to a hundred.
    areas = survey_areas(survey, ["C", "N", "O", "Fe", "Se"])
    flat = quantify(areas, 1486.6, transmission="ninguna")
    corrected = quantify(areas, 1486.6, transmission="potencia", exponent=-0.65)
    # Judged on the whole composition, not element by element: an element
    # can land closer by luck with the wrong transmission, and on this
    # sample the nitrogen does. What cannot happen by luck is the total
    # error halving.
    before = sum(abs(flat.percent(name) - value) for name, value in truth.items())
    after = sum(abs(corrected.percent(name) - value)
                for name, value in truth.items())
    assert after < 0.6 * before


def test_integrating_a_survey_says_which_lines_it_swallowed(demo):
    """Fe 3p sits on Se 3d at 55 eV, so integrating the selenium window off
    a survey counts iron as selenium."""
    areas = survey_areas(demo[0], ["Se"])
    assert any("Fe 3p" in note for note in areas[0].notes)


def test_quantification_says_it_is_a_per_cent_of_what_was_detected():
    result = quantify([LineArea("C", "C 1s", 100.0),
                       LineArea("N", "N 1s", 10.0)], 1486.6)
    assert sum(item.atomic_percent for item in result.abundances) == pytest.approx(100.0)
    assert any("hidrógeno" in text for text in result.warnings)


def test_a_line_the_anode_cannot_excite_is_refused():
    """Cu 2p is at 932.6 eV of binding energy; a 900 eV source cannot reach
    it, and an area attributed to it is an area of something else."""
    with pytest.raises(XPSError, match="no se excita"):
        quantify([LineArea("Cu", "Cu 2p", 1.0)], 900.0)


# ----------------------------------------------------------------------
# files
# ----------------------------------------------------------------------
def test_a_vamas_round_trip_keeps_the_numbers_and_the_settings(tmp_path):
    axis = np.linspace(280.0, 300.0, 201)
    counts = 1000 + 8000 * gl(axis, 284.8, 1, 1.0, 0.3)
    spectrum = XPSSpectrum(axis, counts, photon_energy=1486.6, pass_energy=26.0,
                           dwell_s=0.1, sweeps=10, region="C 1s", name="demo")
    spectrum = spectrum.shifted(0.35, "C 1s adventicio")
    path = write_vamas(tmp_path / "demo.vms", spectrum)
    back = read_vamas(path)[0]
    assert np.allclose(back.binding_energy, spectrum.binding_energy)
    assert np.allclose(back.counts, spectrum.counts, atol=1e-5)
    assert back.pass_energy == pytest.approx(26.0)
    assert back.sweeps == 10
    assert back.monochromated is True


def test_a_spe_header_that_disagrees_with_itself_is_refused(tmp_path):
    """The header gives endpoints, point count AND step — one number more
    than is needed. If they disagree, the fields were not where the reader
    thought, and going on would write a plausible axis over the wrong data."""
    header = (
        "SOFH\n"
        "SofhRev: 5.0\n"
        "XraySource: Al 1486.6 mono\n"
        "NoSpectralReg: 1\n"
        "SpectralRegDef: 1 1 C1s 6 200 0.100 295.0 280.0 295.0 280.0 "
        "0.050 26.00 \"C1s\"\n"
        "EOFH\n"
    )
    path = tmp_path / "malo.spe"
    path.write_text(header + "1.0 2.0\n" * 200, encoding="latin-1")
    with pytest.raises(XPSError, match="no cuadra consigo misma"):
        read_spe(path)


def test_an_ascii_spe_reads_with_its_header(tmp_path):
    axis = np.linspace(295.0, 280.0, 151)
    counts = 2000 + 20000 * gl(axis, 284.8, 1, 1.2, 0.3)
    header = (
        "SOFH\n"
        "SofhRev: 5.0\n"
        "InstrumentModel: PHI Quantera\n"
        "XraySource: Al 1486.6 mono\n"
        "AnalyserWorkFcn: 4.470\n"
        "NoSpectralReg: 1\n"
        "SpectralRegDef: 1 1 C1s 6 151 0.100 295.0 280.0 295.0 280.0 "
        "0.050 26.00 \"C1s\"\n"
        "EOFH\n"
    )
    body = "".join(f"{x:.3f} {y:.3f}\n" for x, y in zip(axis, counts))
    path = tmp_path / "bueno.spe"
    path.write_text(header + body, encoding="latin-1")
    spectrum = read_spe(path)[0]
    assert spectrum.pass_energy == pytest.approx(26.0)
    assert spectrum.photon_energy == pytest.approx(1486.6)
    assert spectrum.work_function == pytest.approx(4.47)
    assert spectrum.counts.size == 151
    assert spectrum.binding_energy[0] == pytest.approx(280.0)


def test_an_unreadable_spe_binary_names_the_export_instead_of_guessing(tmp_path):
    header = (
        "SOFH\nSofhRev: 5.0\nXraySource: Al 1486.6 mono\nNoSpectralReg: 1\n"
        "SpectralRegDef: 1 1 C1s 6 151 0.100 295.0 280.0 295.0 280.0 "
        "0.050 26.00 \"C1s\"\nEOFH\n"
    )
    path = tmp_path / "binario.spe"
    path.write_bytes(header.encode("latin-1") + b"\x00\x01\x02\x03" * 40)
    with pytest.raises(XPSError, match="MultiPak"):
        read_spe(path)


def test_a_text_axis_defaults_to_binding_energy_and_says_so(tmp_path):
    path = tmp_path / "region.txt"
    axis = np.linspace(280.0, 300.0, 60)
    path.write_text("".join(f"{x:.2f}\t{1000.0 + x:.1f}\n" for x in axis))
    spectrum = read_xps_text(path)
    assert "aviso" in spectrum.metadata
    with pytest.raises(XPSError, match="energía del fotón"):
        read_xps_text(path, axis="kinetic")


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------
def test_a_fit_table_adds_back_up_to_its_envelope(demo):
    region = next(item for item in demo if item.region == "Se 3d5/2")
    result = fit_region(region, state_model(region, "Se 3d5/2",
                                            ["selenide", "Se0", "SeOx"]))
    table = fit_table(result)
    values = np.array([[float(cell) for cell in row] for row in table.rows])
    columns = table.columns
    envelope = values[:, columns.index("envolvente")]
    total = values[:, columns.index("fondo")]
    for component in result.components:
        total = total + values[:, columns.index(component.name)]
    assert np.allclose(envelope, total, atol=1e-6)


def test_a_components_table_carries_the_window_each_component_was_held_in(demo):
    region = next(item for item in demo if item.region == "N 1s")
    result = fit_region(region, state_model(region, "N 1s",
                                            ["pyridinic", "pyrrolic"]))
    text = components_table(result).to_csv()
    assert "398.0–399.0 eV" in text or "398.0" in text
    assert "confianza" in text


def test_a_model_exported_from_one_sample_applies_to_another(tmp_path, demo):
    """Fit the first sample carefully, then give the other eleven the same
    model, so what differs between them is the sample."""
    region = next(item for item in demo if item.region == "N 1s")
    result = fit_region(region, state_model(
        region, "N 1s", ["pyridinic", "pyrrolic", "graphitic", "N-oxide"]))
    _, parameters = write_fit(result, tmp_path / "n1s.csv")
    components, label = read_components(parameters)
    assert label == "N 1s"
    assert [item.state for item in components] == [
        "pyridinic", "pyrrolic", "graphitic", "N-oxide"]

    other = next(item for item in make_xps_demo("NCNT", seed=13)
                 if item.region == "N 1s")
    window = (min(c.centre for c in components) - 4.0,
              max(c.centre for c in components) + 4.0)
    from ramancarbon.xps.tables import heights_from

    heights_from(other, components, window)
    transferred = fit_region(other, XPSModel(components, window,
                                             region_label="N 1s"))
    shares = {item.state: item.area_fraction for item in transferred.components}
    # The second sample has no oxidised nitrogen, and that component has to
    # go to zero rather than absorb somebody else's area.
    assert shares["N-oxide"] < 0.02
    assert shares["pyridinic"] == pytest.approx(0.40, abs=0.05)


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------
def test_the_whole_analysis_recovers_the_composition_that_went_in():
    spectra = make_xps_demo("NCNT_FeSe", seed=14, charge_shift=1.8)
    truth = spectra[0].metadata["true_composition_at"]
    analysis = analyse_xps(
        spectra, reference_state=("C 1s", "C-C sp2"),
        regions={"C 1s": ["C-C sp2", "C-O", "C=O", "O-C=O"], "N 1s": 4,
                 "O 1s": 3, "Fe 2p3/2": ["Fe-Se", "Fe3+"], "Se 3d5/2": 3},
        name="prueba",
    )
    assert analysis.calibration.shift_ev == pytest.approx(-1.8, abs=0.2)
    assert set(analysis.survey.symbols()) == {"C", "N", "O", "Fe", "Se"}
    for element, expected in truth.items():
        assert analysis.composition.percent(element) == pytest.approx(
            expected, rel=0.05, abs=1.0), element
    report = analysis.report()
    assert "REFERENCIA DE CARGA" in report
    assert "COMPOSICIÓN" in report


def test_referencing_a_carbon_sample_on_its_own_c1s_is_called_out():
    spectra = make_xps_demo("NCNT_FeSe", seed=15)
    analysis = analyse_xps(spectra, reference="C1s_adventitious",
                           regions={"N 1s": 4})
    assert any("circular" in text for text in analysis.warnings)


# ----------------------------------------------------------------------
# the universal reader, and the section's own state
# ----------------------------------------------------------------------
def test_a_photoelectron_file_is_recognised_by_the_universal_reader(tmp_path):
    """A file is a file: the same import that reads a voltammogram has to
    recognise a VAMAS spectrum, or a mixed folder loses it."""
    from ramancarbon.dataio.detect import detect
    from ramancarbon.dataio.load import load

    path = write_vamas(tmp_path / "sesion.vms", make_xps_demo("NCNT", seed=16))
    found = detect(path)
    assert found.kind == "xps" and found.fmt == "vamas"
    loaded = load(path)
    assert loaded.kind == "xps"
    assert len(loaded.data) == 3            # a survey and two regions


def test_a_phi_header_is_recognised_before_the_binary_check(tmp_path):
    """A .spe is binary AFTER an ASCII header. Rejecting it as binary would
    throw away the one vendor format this package can actually read."""
    from ramancarbon.dataio.detect import detect

    path = tmp_path / "algo.spe"
    path.write_bytes(b"SOFH\nSofhRev: 5.0\nEOFH\n" + b"\x00\x01" * 50)
    assert detect(path).kind == "xps"


def test_the_section_state_runs_the_whole_thing_without_tkinter():
    """Everything the window does lives here, so it can be tested."""
    from ramancarbon.gui.xps_state import XPSSession

    session = XPSSession(name="prueba")
    session.spectra.extend(make_xps_demo("NCNT_FeSe", seed=17, charge_shift=1.5))
    session.shifted = list(session.spectra)
    session.reference_state = ("C 1s", "C-C sp2")
    session.choice_for("C 1s").states = ["C-C sp2", "C-O", "C=O", "O-C=O"]

    calibration = session.apply_reference()
    assert calibration.shift_ev == pytest.approx(-1.5, abs=0.2)
    assert session.run_survey() is not None
    assert session.fit_all() == 5
    composition = session.quantify()
    truth = session.spectra[0].metadata["true_composition_at"]
    for element, expected in truth.items():
        # 5 % relative, or 1 point absolute for the minor elements. Well
        # inside the 15 % systematic the technique carries with tabulated
        # sensitivity factors — which is the floor here, not the code.
        assert composition.percent(element) == pytest.approx(
            expected, rel=0.05, abs=1.0), element
    assert "COMPOSICIÓN" in session.report()


def test_the_section_warns_before_referencing_a_carbon_sample_circularly():
    from ramancarbon.gui.xps_state import XPSSession

    session = XPSSession()
    session.spectra.extend(make_xps_demo("NCNT", seed=18))
    session.shifted = list(session.spectra)
    session.apply_reference()
    assert any("circular" in text for _, text in session.messages)


def test_an_asymmetric_component_and_its_background_are_not_independent():
    """A Doniach–Šunjić does not decay to zero on EITHER side — it goes as
    |u|^(α−1) — so over a finite window the Shirley absorbs part of the
    tail, the fitted asymmetry comes out low, and the metallic component's
    area comes out short. Always in that direction. The test locks the
    direction and the fitter says so in its warnings."""
    from ramancarbon.core.compat import trapezoid
    from ramancarbon.xps.lineshapes import ds_gauss

    rng = np.random.default_rng(4)
    axis = np.linspace(846.0, 868.0, 221)
    peak = 60000 * ds_gauss(axis, 852.6, 1.0, 0.6, 0.30, 0.7)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (peak[1:] + peak[:-1]) * np.diff(axis))])
    counts = rng.poisson(peak + 3000 + 1200 * cumulative / cumulative[-1])
    spectrum = XPSSpectrum(axis, counts.astype(float), photon_energy=1486.6,
                           pass_energy=26.0, dwell_s=0.1, sweeps=10,
                           region="Ni 2p3/2")
    model = XPSModel(
        [XPSComponent(name="m", label="metal", centre=852.6, height=50000.0,
                      fwhm=0.6, profile="ds_gauss", centre_bounds=(851.5, 853.5),
                      fwhm_bounds=(0.2, 2.0), extra=(0.15, 0.7))],
        window=(846.0, 868.0), background="shirley", region_label="Ni 2p3/2",
    )
    result = fit_region(spectrum, model)
    component = result.components[0]
    asymmetry = component.extra[component.extra_names.index("asymmetry")]
    assert asymmetry < 0.30                      # low, never high
    assert component.area < float(trapezoid(peak, axis))
    assert any("NO son independientes" in text for text in result.warnings)


def test_a_symmetric_shape_on_a_metal_is_much_worse_than_an_asymmetric_one():
    """This is why the asymmetric profiles are here at all."""
    from ramancarbon.xps.lineshapes import ds_gauss

    rng = np.random.default_rng(6)
    axis = np.linspace(848.0, 866.0, 181)
    peak = 60000 * ds_gauss(axis, 852.6, 1.0, 0.6, 0.30, 0.7)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (peak[1:] + peak[:-1]) * np.diff(axis))])
    counts = rng.poisson(peak + 3000 + 1200 * cumulative / cumulative[-1])
    spectrum = XPSSpectrum(axis, counts.astype(float), photon_energy=1486.6,
                           pass_energy=26.0, dwell_s=0.1, sweeps=10,
                           region="Ni 2p3/2")

    def fit(component):
        return fit_region(spectrum, XPSModel(
            [component], window=(848.0, 866.0), background="shirley",
            region_label="Ni 2p3/2"))

    asymmetric = fit(XPSComponent(
        name="m", label="metal", centre=852.6, height=50000.0, fwhm=0.6,
        profile="ds_gauss", centre_bounds=(851.5, 853.5),
        fwhm_bounds=(0.2, 2.0), extra=(0.2, 0.7)))
    symmetric = fit(XPSComponent(
        name="s", label="simétrica", centre=852.6, height=50000.0, fwhm=1.0,
        profile="gl", centre_bounds=(851.5, 853.5), fwhm_bounds=(0.3, 3.0)))
    assert symmetric.reduced_chi2 > 5 * asymmetric.reduced_chi2


def test_a_photoelectron_spectrum_exports_with_what_cannot_be_recovered(tmp_path):
    """The charge shift above all: a binding energy exported without saying
    what put the axis there is not a measurement, and whoever opens the
    file has no other way to find out."""
    from ramancarbon.dataio.export import export

    spectrum = next(item for item in make_xps_demo("NCNT_FeSe", seed=19)
                    if item.region == "N 1s")
    spectrum = spectrum.shifted(-1.2, "C 1s sp² a 284.4 eV")
    text = export(spectrum, tmp_path / "n1s.csv").read_text(encoding="utf-8")
    assert "energia_enlace" in text
    assert "hν = 1486.6 eV" in text
    assert "energía de paso 26 eV" in text
    assert "-1.200 eV ya aplicado" in text


def test_a_photoelectron_spectrum_survives_a_project_round_trip(tmp_path):
    """Reopening a project has to give back something that can still be
    analysed: without the anode there is no kinetic scale and no Auger
    lines, and without the pass energy no resolution floor."""
    from ramancarbon.dataio import Project

    original = next(item for item in make_xps_demo("NCNT_FeSe", seed=20)
                    if item.region == "Fe 2p3/2")
    project = Project(name="prueba")
    project.add(original)
    back = Project.load(project.save(tmp_path / "s.rcproj")).datasets[0].to_object()
    assert np.allclose(back.binding_energy, original.binding_energy)
    assert np.allclose(back.counts, original.counts)
    assert back.photon_energy == pytest.approx(original.photon_energy)
    assert back.pass_energy == pytest.approx(original.pass_energy)
    assert back.sweeps == original.sweeps
    assert back.region == original.region


# -- per-component control in the section -------------------------------

def _xps_session():
    from ramancarbon.gui.xps_state import XPSSession

    session = XPSSession()
    spectra = make_xps_demo("NCNT_FeSe", seed=0)
    session.spectra = list(spectra)
    session.shifted = list(spectra)
    return session


def test_the_fit_window_is_a_parameter_and_can_be_set():
    """Moving the high-binding-energy limit of a C 1s by one electronvolt
    moves the carbonyl area by several per cent. That is not a defect of
    the method -- it is what "the area of a peak over a background" means
    -- so the ends have to be reachable, and they go in the report."""
    session = _xps_session()
    whole = session.window_for("C 1s")
    assert session.fit("C 1s") is not None
    wide = {row[0]: float(row[3]) for row in session.component_rows("C 1s")}

    assert session.set_window("C 1s", (whole[0], whole[1] - 2.0))
    assert session.window_for("C 1s") == (whole[0], whole[1] - 2.0)
    assert session.fit("C 1s") is not None
    narrow = {row[0]: float(row[3]) for row in session.component_rows("C 1s")}

    moved = [name for name in wide
             if name in narrow and abs(narrow[name] - wide[name])
             > 0.02 * max(wide[name], 1.0)]
    assert moved, "narrowing the window by 2 eV changed no area at all"

    # And the ends are checked, not trusted.
    assert session.set_window("C 1s", (1500.0, 1501.0)) is False
    assert session.set_window("C 1s", (284.0, 284.2)) is False
    assert session.window_for("C 1s") == (whole[0], whole[1] - 2.0), (
        "a rejected window was applied anyway")
    assert session.set_window("C 1s", None) is True
    assert session.window_for("C 1s") == whole


def test_a_component_can_be_moved_and_held():
    """The same rule as the circuit fit and the Rietveld refinement:
    holding a parameter removes a degree of freedom, so every other
    uncertainty comes out smaller and a wrong held value moves into its
    neighbours instead of showing up as a bad fit. So it has to be
    visible in the table."""
    session = _xps_session()
    assert session.fit("N 1s") is not None
    rows = session.component_rows("N 1s")
    assert rows
    name = rows[0][0]

    session.set_component("N 1s", name, fwhm=1.30, fixed=("fwhm",))
    assert session.fit("N 1s") is not None
    held = next(r for r in session.component_rows("N 1s") if r[0] == name)
    assert float(held[2]) == pytest.approx(1.30, abs=0.01)
    assert "fwhm" in held[6]

    result = session.fits["N 1s"]
    component = next(c for c in result.components if c.label == name)
    assert "fwhm" not in component.errors, (
        "a held parameter must not carry an uncertainty")

    session.reset_components("N 1s")
    assert session.choice_for("N 1s").overrides == {}


def test_moving_a_component_out_of_its_published_window_is_allowed_and_said():
    """It is the user's sample, so their number wins -- but a component
    that has left its state's published window is no longer evidence for
    that state, and the report says so rather than quietly clipping the
    value back inside."""
    session = _xps_session()
    assert session.fit("Fe 2p3/2") is not None
    rows = session.component_rows("Fe 2p3/2")
    name, centre = rows[0][0], float(rows[0][1])

    session.set_component("Fe 2p3/2", name, centre=centre + 3.0)
    result = session.fit("Fe 2p3/2")
    assert result is not None
    assert any("ventana" in text for text in session.messages[-1]) or any(
        "ventana" in text for _, text in session.messages[-6:]), (
        "nothing was said about a component outside its published window")


def test_a_component_can_be_added_where_the_residual_shows_one():
    """The operation the difference curve asks for. It is also the
    easiest way to invent a chemical state, so a hand-added component
    carries no tabulated identity."""
    session = _xps_session()
    assert session.fit("C 1s") is not None
    before = len(session.fits["C 1s"].components)

    assert session.add_component("C 1s", 288.6, 1.6, name="COOH")
    result = session.fit("C 1s")
    assert len(result.components) == before + 1
    added = next(c for c in result.components if c.label == "COOH")
    assert added.state is None, "a hand-added component claimed a state"
    assert "mano" in added.justification

    # Adding one ALWAYS lowers the residual, which is why the section has
    # a component-count comparison and not just this button.
    assert session.remove_component("C 1s", "COOH") is True
    assert session.remove_component("C 1s", "C=O (carbonilo)") is False, (
        "a tabulated component must not be deletable one at a time; those "
        "are chosen by picking states")
    assert len(session.fit("C 1s").components) == before

    # And it has to be inside the window that will be fitted.
    low, high = session.window_for("C 1s")
    session.add_component("C 1s", low - 50.0)
    session.fit("C 1s")
    assert any("fuera de la ventana" in text
               for level, text in session.messages if level == "error")


def test_a_component_can_be_given_an_asymmetric_shape():
    """A metal needs one. With symmetric shapes the fit has to cover the
    tail with something, and that something is reported as an oxide that
    is not there."""
    from ramancarbon.gui.xps_state import PROFILES

    keys = [key for key, _ in PROFILES]
    assert "ds_gauss" in keys and "gl" in keys
    assert any("asim" in text for _, text in PROFILES), (
        "the menu has to say which shapes are asymmetric")

    session = _xps_session()
    assert session.fit("Fe 2p3/2") is not None
    name = session.component_rows("Fe 2p3/2")[0][0]
    session.set_component("Fe 2p3/2", name, profile="ds_gauss")
    result = session.fit("Fe 2p3/2")
    changed = next(c for c in result.components if c.label == name)
    assert changed.profile == "ds_gauss"

    # And the Shirley/asymmetry interaction is still declared: they are
    # not independent over a finite window, and the metallic area comes
    # out short in a known direction.
    assert any("asim" in text and "Shirley" in text
               for text in result.warnings), result.warnings


def test_the_component_table_shows_the_total_width():
    """In a Doniach-Sunjic the ``fwhm`` parameter is only the Lorentzian
    part. The literature publishes the TOTAL, and it is the total that
    has to be compared against the resolution floor."""
    session = _xps_session()
    assert session.fit("C 1s") is not None
    result = session.fits["C 1s"]
    rows = {row[0]: row for row in session.component_rows("C 1s")}
    for component in result.components:
        assert float(rows[component.label][2]) == pytest.approx(
            component.true_fwhm, abs=0.005)
    asymmetric = [c for c in result.components if c.profile.startswith("ds")]
    if asymmetric:
        one = asymmetric[0]
        assert one.true_fwhm != pytest.approx(one.fwhm, abs=1e-6), (
            "this demo no longer has an asymmetric component whose total "
            "width differs from its parameter; the test needs another one")


def test_the_width_you_type_is_the_width_you_read():
    """The table shows the TOTAL width, because that is what the
    literature publishes and what has to be compared against the
    resolution floor. For every profile but the plain Gaussian and
    Lorentzian that is not the width PARAMETER -- a GL product is 4 %
    narrower than its parameter at mixing 0.3 and a Doniach-Sunjic 13 %
    wider -- so an editor that stored the typed number directly would
    change the peak when you typed the displayed value back."""
    from ramancarbon.xps.lineshapes import (XPS_PROFILES, fwhm_for_total,
                                            profile_fwhm)

    for name, spec in XPS_PROFILES.items():
        extra = tuple(spec["defaults"])
        for total in (0.8, 1.4, 3.0):
            parameter = fwhm_for_total(name, total, extra)
            # 0.5 %, which is the FORWARD function's own resolution:
            # profile_fwhm measures the half-maximum crossings on a
            # 4001-point grid spanning +-8 widths, so it is quantised at
            # about 0.4 % and no inverse of it can be tighter. On a 1.4 eV
            # component that is six thousandths of an electronvolt.
            assert profile_fwhm(name, parameter, extra) == pytest.approx(
                total, rel=6e-3), (name, total, parameter)

    # It is a real conversion, not the identity, for the shapes that need
    # one -- and ds_gauss is not scale-invariant, so a single rescaling
    # would not do.
    assert fwhm_for_total("gl", 1.4, (0.3,)) != pytest.approx(1.4, rel=1e-3)
    ratios = [fwhm_for_total("ds_gauss", t, (0.1, 0.5)) / t
              for t in (0.8, 3.0)]
    assert ratios[0] != pytest.approx(ratios[1], rel=0.05), (
        "ds_gauss looks scale-invariant here; the numeric inverse would "
        "not be needed and this test is no longer testing anything")

    # And through the section: type back what the table shows, refit,
    # and the width has not moved.
    session = _xps_session()
    assert session.fit("C 1s") is not None
    rows = session.component_rows("C 1s")
    name, shown = rows[0][0], float(rows[0][2])
    session.set_component("C 1s", name, fwhm=shown, fixed=("fwhm",))
    assert session.fit("C 1s") is not None
    again = next(r for r in session.component_rows("C 1s") if r[0] == name)
    # Within 0.02 eV: the table rounds to two decimals and the forward
    # width measurement is quantised at about 0.4 %, so a round trip
    # cannot be tighter than the last displayed digit. Without the
    # conversion this component -- the asymmetric one -- came back 4 %
    # narrower every time the displayed number was typed back.
    assert float(again[2]) == pytest.approx(shown, abs=0.02)
    assert session.fits["C 1s"].components[0].profile.startswith("ds"), (
        "this test wants the asymmetric component, where the parameter "
        "and the true width differ most")


# ----------------------------------------------------------------------
# what happens to a file that will not open
# ----------------------------------------------------------------------
def _a_vamas_file(tmp_path):
    from ramancarbon.xps.io import write_vamas

    energy = np.linspace(280.0, 300.0, 401)
    counts = 1000.0 + 5000.0 * np.exp(-0.5 * ((energy - 284.6) / 0.8) ** 2)
    spectrum = XPSSpectrum(
        binding_energy=energy, counts=counts, name="C 1s", region="C1s",
        photon_energy=1486.6, pass_energy=26.0,
    )
    return write_vamas(tmp_path / "muestra.vms", spectrum)


# The reader decodes with latin-1, so a UTF-8 byte-order mark reaches
# it as these three characters.
@pytest.mark.parametrize("preamble", ["ï»¿", "\n", "   \n\n"])
def test_a_vamas_file_is_read_even_when_the_identifier_is_not_line_one(
        tmp_path, preamble):
    """A byte-order mark or a blank line in front of the identifier used to
    send the file to the two-column text reader, which then complained
    about columns — a message that describes the wrong problem entirely."""
    from ramancarbon.xps.io import read_xps

    path = _a_vamas_file(tmp_path)
    path.write_text(preamble + path.read_text(encoding="latin-1"),
                    encoding="latin-1")
    spectra = read_xps(path)
    assert [item.region for item in spectra] == ["C1s"]


def test_a_vms_that_is_not_vamas_is_refused_by_name(tmp_path):
    """The extension claims VAMAS, so the VAMAS reader answers — quoting
    the line it found. Falling through to the text reader produced an
    error about column counts that sent the user looking in the wrong
    place."""
    from ramancarbon.xps.io import read_xps

    path = tmp_path / "raro.vms"
    path.write_text("# exportado por otra cosa\n1 2\n3 4\n", encoding="latin-1")
    with pytest.raises(XPSError) as raised:
        read_xps(path)
    assert "VAMAS" in str(raised.value)


def test_the_xps_section_shows_the_reason_a_file_would_not_open():
    """Static: the load handler must put the reader's diagnosis in front of
    the user and must not overwrite it with a count.

    This was the whole of "the XPS section does not open anything". The
    readers diagnose precisely what an unreadable file is; ``_load`` pushed
    that diagnosis into the status bar with ``flush_messages`` and then, on
    the very next line, replaced it with ``0 espectro(s) cargados``. The
    list said ``(ningún espectro)`` and no reason appeared anywhere.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "gui" / "xps_app.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    handler = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_load"
    )
    body = ast.unparse(handler)
    assert "showerror" in body, "an unreadable file has to say so in a dialog"
    assert "level == 'error'" in body, "the errors the readers logged are ignored"
    # The early return is what keeps the count from overwriting the reason.
    assert any(isinstance(node, ast.Return) for node in ast.walk(handler))


def test_a_descending_region_reads_because_that_is_the_normal_direction(tmp_path):
    """PHI writes the step SIGNED, and a photoelectron scan normally runs
    downwards in binding energy.

    The user's own file declared ``-0.05`` for a C 1s from 295 to 280 eV --
    301 points, perfectly self-consistent -- and the reader refused it for
    "declaring a step of -0.05", which reads like a corrupt header and is
    in fact the ordinary case. Every fixture in this file had been written
    with a positive step, so nothing caught it.
    """
    axis = np.linspace(295.0, 280.0, 301)
    counts = 2000 + 20000 * gl(axis, 284.8, 1, 1.2, 0.3)
    header = (
        "SOFH\n"
        "SofhRev: 5.0\n"
        "XraySource: Al 1486.6 mono\n"
        "NoSpectralReg: 1\n"
        "SpectralRegDef: 1 1 C1s 6 301 -0.050 295.0 280.0 295.0 280.0 "
        "0.050 26.00 \"C1s\"\n"
        "EOFH\n"
    )
    body = "".join(f"{x:.3f} {y:.3f}\n" for x, y in zip(axis, counts))
    path = tmp_path / "260909.112.spe"
    path.write_text(header + body, encoding="latin-1")

    spectrum = read_xps(path)[0]
    assert spectrum.counts.size == 301
    assert spectrum.pass_energy == pytest.approx(26.0)
    assert spectrum.binding_energy[0] == pytest.approx(280.0)
    assert read_spe(path)[0].counts.size == 301


def test_a_region_whose_step_really_disagrees_is_still_refused(tmp_path):
    """The self-consistency check survives the sign fix: the magnitude has
    to match, so a header whose fields were misread is still caught."""
    header = (
        "SOFH\n"
        "SofhRev: 5.0\n"
        "XraySource: Al 1486.6 mono\n"
        "NoSpectralReg: 1\n"
        "SpectralRegDef: 1 1 C1s 6 301 -0.200 295.0 280.0 295.0 280.0 "
        "0.050 26.00 \"C1s\"\n"
        "EOFH\n"
    )
    path = tmp_path / "malo.spe"
    path.write_text(header + "1.0 2.0\n" * 301, encoding="latin-1")
    with pytest.raises(XPSError, match="no cuadra consigo misma"):
        read_spe(path)


def test_a_vamas_that_quotes_a_phi_header_is_still_vamas(tmp_path):
    """A PHI Quantera writes its whole SOFH/EOFH header into the comment
    lines of the VAMAS it exports. Looking for SOFH anywhere in the first
    8 kB therefore handed a real VAMAS file to the PHI reader, which read
    the regions out of the comment and then looked for a binary block
    that is not there -- "el bloque de datos binario admite más de una
    lectura". The identifier decides; SOFH inside it is a quotation."""
    from ramancarbon.xps.io import read_xps, write_vamas

    energy = np.linspace(280.0, 300.0, 401)
    counts = 1000.0 + 5000.0 * np.exp(-0.5 * ((energy - 284.6) / 0.8) ** 2)
    spectrum = XPSSpectrum(binding_energy=energy, counts=counts, name="C 1s",
                           region="C1s", photon_energy=1486.6, pass_energy=26.0)
    path = write_vamas(tmp_path / "quantera.vms", spectrum,
                       institution="SOFH PHI Quantera SXM")
    assert [item.region for item in read_xps(path)] == ["C1s"]


def test_the_file_settles_the_work_function_when_it_states_the_window():
    """VAMAS says the abscissa is the kinetic energy as measured, so the
    work function comes off it; PHI has already taken it off. Nothing in
    the block says which, and getting it wrong moves every region by the
    whole work function -- 4.05 eV here -- in silence. The PHI header in
    the comments states each window, which settles it."""
    from ramancarbon.xps.io import _declared_for, _declared_windows

    comments = [
        "SpectralRegDef: 1 1 C1s 6 401 -0.0500 298.0000 278.0000 297.0000 "
        "279.0000 0.360000 55.00 AREA",
        "SpectralRegDef: 3 1 O1s 8 321 -0.0500 540.0000 524.0000 539.0000 "
        "525.0000 3.600000 55.00 AREA",
        "Platform: PC",
    ]
    windows = _declared_windows(comments)
    assert windows == {"c1s": (298.0, 278.0), "o1s": (540.0, 524.0)}
    assert _declared_for(windows, "C", "1s") == (298.0, 278.0)
    assert _declared_for(windows, "S", "2p") is None
    assert _declared_for({}, "C", "1s") is None


def test_a_region_that_was_scanned_is_evidence_the_survey_cannot_override():
    """A survey is a compromise -- wide range, coarse step, short dwell --
    so a minor element sits at its noise floor even when it is plainly
    there. Nitrogen in a doped carbon is the standard case, and it is also
    why the narrow region was measured. Reporting "no nitrogen" while an
    N 1s region with a 17-sigma peak is loaded in the same session is the
    program disagreeing with data it already has.
    """
    from ramancarbon.xps.survey import SurveyResult, add_region_evidence

    energy = np.linspace(391.0, 411.0, 401)
    counts = 6500.0 + 1500.0 * np.exp(-0.5 * ((energy - 400.6) / 1.0) ** 2)
    region = XPSSpectrum(binding_energy=energy, counts=counts, name="N 1s",
                         region="N 1s", photon_energy=1486.6, pass_energy=55.0)
    empty = SurveyResult(peaks=[], elements=[], uncorroborated=[],
                         unexplained=[], overlaps=[])
    out = add_region_evidence(empty, [region])
    assert [item.symbol for item in out.elements] == ["N"]
    assert out.elements[0].confidence == "alta"
    assert "alta resolución" in out.elements[0].notes[0]

    # And a region with no peak adds nothing: the sulphur region of the
    # user's own sample is flat, and the program must not invent it.
    flat = XPSSpectrum(binding_energy=np.linspace(155.0, 175.0, 401),
                       counts=np.full(401, 1100.0), name="S 2p",
                       region="S 2p", photon_energy=1486.6, pass_energy=55.0)
    assert add_region_evidence(empty, [flat]).elements == []


def test_the_states_a_sulphur_doped_carbon_needs_are_in_the_catalogue():
    """The thiophenic C–S–C at ~163.9 eV is the dominant state in an
    S-doped carbon. Without it its intensity is absorbed by the S–S and
    the sulphide, and the sulphur comes out more reduced than it is."""
    database = load_xps_database()
    keys = {state.key for state in database.states_for("S 2p3/2")}
    assert {"C-S-C", "sulfoxide", "sulfonate"} <= keys
    assert "pyridinium" in {s.key for s in database.states_for("N 1s")}
    assert "C-S" in {s.key for s in database.states_for("C 1s")}


def test_every_chemical_state_that_was_added_cites_where_it_came_from():
    """CLAUDE.md: a literature value goes in the JSON with its source.
    The entries that predate that rule are grandfathered; nothing new is."""
    import json
    from pathlib import Path

    data = json.loads(
        (Path(__file__).resolve().parents[1] / "database" / "data" /
         "xps.json").read_text(encoding="utf-8"))
    added = {"C-S", "C-S-C", "metal-sulfide", "sulfoxide", "sulfonate",
             "amine", "pyridinium", "quinone", "sulfate-O"}
    seen = set()
    for region, states in data["chemical_states"].items():
        if region.startswith("_"):
            continue
        for state in states:
            if state["key"] in added:
                seen.add(state["key"])
                assert state.get("source"), f"{region}/{state['key']} sin fuente"
    assert seen == added


class TestTheSurveyLineThatIsNotAPeak:
    """A minor dopant puts a bump of a few per cent on the background, and
    prominence cannot see it: every ripple of noise on that background is
    also a local maximum. For a line whose position the database already
    gives, the question with an answer is how much intensity is in the
    window -- the same test the Raman side runs on the 2D band."""

    @staticmethod
    def _survey(seed, fraction, level=6500.0):
        energy = np.linspace(0.0, 1100.0, 2751)
        clean = level * (1.0 + 0.35 * (energy / 1100.0) ** 2)
        for centre, height, fwhm in ((284.8, 0.9, 2.2), (531.0, 0.5, 2.4),
                                     (399.5, fraction, 2.4)):
            clean = clean + height * level * np.exp(
                -0.5 * ((energy - centre) / (fwhm / 2.3548)) ** 2)
        counts = np.random.default_rng(seed).poisson(clean).astype(float)
        return XPSSpectrum(binding_energy=energy, counts=counts,
                           name="survey", region="survey",
                           photon_energy=1486.6)

    def test_the_window_sees_what_prominence_cannot(self):
        from ramancarbon.xps.survey import find_survey_peaks, line_contrast

        spectrum = self._survey(0, 0.05)
        peaks, background = find_survey_peaks(spectrum)
        assert not any(abs(peak.binding_energy - 399.5) < 2.0
                       for peak in peaks), "prominencia ya lo veía"
        assert line_contrast(spectrum, 399.5, background) > 3.0

    def test_a_line_with_nothing_under_it_stays_quiet(self):
        """Calibration, not a lowered threshold: the same window on a
        position with no line in it must come back at nothing."""
        from ramancarbon.xps.survey import line_contrast

        spectrum = self._survey(1, 0.0)
        for empty in (150.0, 250.0, 700.0, 900.0):
            assert abs(line_contrast(spectrum, empty)) < 6.0

    def test_it_is_reported_as_tentative_and_not_as_present(self):
        from ramancarbon.xps.survey import identify

        # Strong enough for the window, not for prominence.
        found = identify(self._survey(4, 0.045))
        assert "N" not in [item.symbol for item in found.elements]
        tentative = [item.symbol for item in found.tentative]
        if tentative:                       # the band is statistical
            assert "N" in tentative
            note = found.tentative[0].notes[0]
            assert "tentativo" in note and "sigma" in note

    def test_an_element_that_is_really_there_is_not_demoted(self):
        from ramancarbon.xps.survey import identify

        found = identify(self._survey(3, 0.20))
        assert "N" in [item.symbol for item in found.elements]
        assert "N" not in [item.symbol for item in found.tentative]


class TestChemistryFiltersTheCandidates:
    """Binding energy is not a functional group.

    «O de red (óxido metálico)» at 530 eV and a quinone C=O at 531 eV are
    one electronvolt apart, and which of the two a shoulder is depends on
    something the region cannot see: whether the sample contains a metal.
    Offering both and letting least squares choose is how a nitrogen- and
    sulphur-doped carbon acquires a metal oxide it does not have -- which
    is exactly what it did on the user's own O 1s.
    """

    def test_a_state_declares_what_it_needs(self):
        database = load_xps_database()
        lattice = database.state("O 1s", "lattice")
        assert lattice.requires_any, "el óxido de red no declara requisitos"
        assert not lattice.possible_in(["C", "N", "O"])
        assert lattice.possible_in(["C", "O", "Fe"])
        # No information is not a licence to filter.
        assert lattice.possible_in(None)

    def test_the_catalogue_filters_on_composition(self):
        database = load_xps_database()
        everything = {s.key for s in
                      database.states_for("O 1s", include_satellites=False)}
        carbon = {s.key for s in
                  database.states_for("O 1s", include_satellites=False,
                                      present=["C", "N", "O"])}
        assert "lattice" in everything and "lattice" not in carbon
        assert "C-O" in carbon, "el C–O orgánico no depende de ningún metal"

    def test_the_automatic_model_does_not_offer_the_impossible(self):
        from ramancarbon.xps.presets import count_model

        energy = np.linspace(524.0, 540.0, 321)
        counts = 6500.0 + 5000.0 * np.exp(
            -0.5 * ((energy - 531.0) / 1.2) ** 2) + 3000.0 * np.exp(
            -0.5 * ((energy - 533.0) / 1.4) ** 2)
        spectrum = XPSSpectrum(binding_energy=energy, counts=counts,
                               name="O 1s", region="O 1s",
                               photon_energy=1486.6, pass_energy=55.0)
        model, notes = count_model(spectrum, "O 1s", 3,
                                   present=["C", "N", "O"])
        assert "lattice" not in {c.state for c in model.components}
        assert any("descartados por la composición" in note for note in notes)

    def test_an_explicit_choice_by_the_user_is_not_overruled(self):
        """Section 28: the constraint is inspectable and can be switched
        off. What the user names, the user gets -- the cross-check says so
        afterwards rather than the model silently disagreeing."""
        from ramancarbon.xps.presets import state_model

        energy = np.linspace(524.0, 540.0, 321)
        counts = 6500.0 + 5000.0 * np.exp(-0.5 * ((energy - 530.0) / 1.2) ** 2)
        spectrum = XPSSpectrum(binding_energy=energy, counts=counts,
                               name="O 1s", region="O 1s",
                               photon_energy=1486.6, pass_energy=55.0)
        model = state_model(spectrum, "O 1s", ["lattice"])
        assert [c.state for c in model.components] == ["lattice"]


def test_a_region_cannot_check_itself_but_two_regions_can():
    """Section 25. A C 1s fit puts a C–N component at 285.9 eV whether or
    not the sample has nitrogen: the shoulder is there, C–O sits on top of
    C–N, and least squares has no opinion. What decides is the N 1s."""
    from ramancarbon.gui.xps_state import XPSSession

    session = XPSSession()
    energy = np.linspace(278.0, 298.0, 401)
    counts = 3000.0 + 20000.0 * np.exp(-0.5 * ((energy - 284.6) / 0.9) ** 2) \
        + 4000.0 * np.exp(-0.5 * ((energy - 286.0) / 1.2) ** 2)
    session.spectra.append(XPSSpectrum(
        binding_energy=energy, counts=counts, name="C 1s", region="C 1s",
        photon_energy=1486.6, pass_energy=55.0))
    session.shifted = list(session.spectra)
    session.choice_for("C 1s").states = ["C-C sp2", "C-N"]
    assert session.fit("C 1s") is not None

    verdicts = dict((text, verdict) for verdict, text in session.cross_checks())
    assert any(verdict == "incoherente" and "C–N" in text
               for text, verdict in verdicts.items()), \
        "un C–N sin nitrógeno en la muestra tiene que salir como incoherente"
