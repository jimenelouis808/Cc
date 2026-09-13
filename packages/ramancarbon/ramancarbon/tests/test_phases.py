"""Identification of non-carbon sample phases (FeSe, Se, carbides)."""

from __future__ import annotations

import pytest

from ramancarbon import analyse
from ramancarbon.analysis.phases import (
    CORROBORATION,
    PhaseReport,
    find_phases,
    load_families,
    load_phases,
)
from ramancarbon.core.peaks import PeakMeasurement
from ramancarbon.examples.demo_data import make_demo


def pk(position: float, height: float = 100.0, fwhm: float = 8.0) -> PeakMeasurement:
    return PeakMeasurement(
        position=position,
        height=height,
        fwhm=fwhm,
        area=height * (fwhm or 10.0),
        prominence=height,
        snr=30.0,
        significance=40.0,
    )


# -- database ---------------------------------------------------------


def test_catalogue_loads_and_is_self_consistent():
    phases = load_phases()
    families = {f.key for f in load_families()}
    assert phases
    for phase in phases:
        assert phase.family in families
        for line in phase.strong:
            assert phase.band_at(line) is not None, f"{phase.key}: {line} no catalogada"
        for line in phase.discriminating:
            assert phase.band_at(line) is not None
        for band in phase.bands:
            low, high = band.window
            assert low < band.position < high


def test_iron_is_catalogued_without_bands():
    """Metallic iron has no first-order Raman modes and the file says so."""
    iron = next(p for p in load_phases() if p.key == "Fe_alpha")
    assert iron.raman_silent
    assert iron.bands == ()


def test_every_phase_names_a_diffraction_check():
    for phase in load_phases():
        assert phase.xrd_advice()


# -- matching ---------------------------------------------------------


def test_tetragonal_fese_is_identified_and_resolved():
    peaks = [pk(181, 200), pk(196, 120), pk(1580, 400, 40)]
    report = find_phases(peaks, spectrum_range=(80.0, 3000.0))
    best = report.identifications[0]
    assert best.phase.key == "FeSe_tetragonal"
    assert best.corroborated
    verdict = next(v for v in report.families if v.family.key == "FeSe")
    assert verdict.resolved
    assert verdict.best is not None and verdict.best.phase.key == "FeSe_tetragonal"


def test_a_single_isolated_line_is_not_an_identification():
    """One coincidence in the RBM window is likelier than the phase."""
    report = find_phases([pk(196, 100)], spectrum_range=(80.0, 3000.0))
    assert not report.found_anything
    assert all(not i.corroborated for i in report.identifications)


def test_clean_nanotube_spectrum_yields_no_corroborated_phase():
    """The regression that killed the first interference matcher: genuine
    radial breathing modes must not be reassigned to invented phases."""
    peaks = [pk(150), pk(168), pk(190), pk(214), pk(265), pk(1580, 400, 20)]
    report = find_phases(peaks, spectrum_range=(100.0, 3000.0))
    assert not report.found_anything
    assert report.excluded_from_rbm == []


def test_lines_outside_the_measured_range_do_not_count_against_a_phase():
    narrow = find_phases([pk(237, 150)], spectrum_range=(200.0, 400.0))
    wide = find_phases([pk(237, 150), pk(143, 60)], spectrum_range=(80.0, 400.0))
    # Out of range, 143 cm-1 cannot be held against the phase — but a lone
    # line is still not corroboration. Measure the region and find it, and
    # the identification stands.
    assert not narrow.found_anything
    assert wide.found_anything


def test_weak_lines_alone_never_report_a_phase():
    """A minor line landing on a peak while every strong line is missing is
    arithmetic, not evidence."""
    report = find_phases([pk(148, 100)], spectrum_range=(80.0, 3000.0))
    assert all(i.strong_found > 0 for i in report.identifications)


# -- polymorphs -------------------------------------------------------


def test_an_upper_width_bound_alone_never_corroborates():
    """"Narrower than 15 cm-1" is true of every radial breathing mode, so it
    is not evidence of anything. Letting it count turned the 254 cm-1 RBM of
    the clean single-wall demo into monoclinic selenium."""
    report = find_phases([pk(254, 150, fwhm=9.0)], spectrum_range=(90.0, 400.0))
    assert not report.found_anything


def test_band_width_separates_amorphous_from_monoclinic_selenium():
    broad = find_phases([pk(252, 150, fwhm=30.0)], spectrum_range=(80.0, 400.0))
    narrow = find_phases([pk(254, 150, fwhm=6.0), pk(112, 60)],
                         spectrum_range=(80.0, 400.0))
    assert {i.phase.key for i in broad.identifications if i.corroborated} == {
        "Se_amorphous"
    }
    assert {i.phase.key for i in narrow.identifications if i.corroborated} == {
        "Se_monoclinic"
    }


def test_unmeasurable_width_reports_rather_than_decides():
    report = find_phases([pk(252, 150, fwhm=None)], spectrum_range=(80.0, 400.0))
    amorphous = next(i for i in report.identifications if i.phase.key == "Se_amorphous")
    assert amorphous.width_ok is True
    assert not amorphous.width_confirmed
    assert "no se pudo comprobar" in amorphous.width_note


def test_family_is_reported_even_when_the_polymorph_is_not():
    """Composition without discrimination must name the composition."""
    report = find_phases([pk(220, 150), pk(145, 90)], spectrum_range=(80.0, 400.0))
    verdict = next((v for v in report.families if v.family.key == "FeSe"), None)
    assert verdict is not None
    assert verdict.present


def test_unresolved_family_asks_a_diffraction_question():
    """FeSe2 and beta-FeSe overlap; the report must hand the call to XRD."""
    report = find_phases(
        [pk(176, 200), pk(245, 90), pk(148, 70)], spectrum_range=(80.0, 400.0)
    )
    assert report.xrd_questions
    assert any("2θ" in q for q in report.xrd_questions)


# -- coupling to the diameter analysis --------------------------------


def test_phase_lines_are_kept_out_of_the_diameter_analysis():
    """Without this the FeSe modes become five confident, fictitious tube
    diameters. That failure is the reason the scan is on by default."""
    spectrum = make_demo("MWCNT_FeSe", laser_nm=532.0, seed=3)

    without = analyse(spectrum, auto_preprocess=True, check_phases=False)
    assert len(without.rbm.diameters) >= 3, "el fallo que se quiere evitar"

    with_scan = analyse(spectrum, auto_preprocess=True)
    assert with_scan.rbm.diameters == []
    assert with_scan.classification.label.startswith("Nanotubo de pared multiple")
    assert with_scan.phases is not None and with_scan.phases.found_anything


def test_scan_is_on_by_default_and_can_be_switched_off():
    spectrum = make_demo("MWCNT", laser_nm=532.0, seed=1)
    assert analyse(spectrum).phases.enabled
    off = analyse(spectrum, check_phases=False).phases
    assert isinstance(off, PhaseReport) and not off.enabled
    assert "desactivada" in off.summary()


def test_clean_demo_spectra_stay_free_of_phase_calls():
    for kind in ("SWCNT", "DWCNT", "MWCNT", "grafeno_1L", "GO"):
        result = analyse(make_demo(kind, laser_nm=532.0, seed=2), auto_preprocess=True)
        assert not result.phases.found_anything, f"{kind}: falso positivo"


def test_report_mentions_the_phases_and_the_abundance_caveat():
    result = analyse(make_demo("MWCNT_FeSe", seed=3), auto_preprocess=True)
    text = result.report(verbose=False)
    assert "FASES NO CARBONOSAS" in text
    assert "FeSe" in text
    assert "NO son fracciones másicas" in text


def test_selenium_segregation_is_called_out():
    result = analyse(make_demo("MWCNT_FeSe", seed=3), auto_preprocess=True)
    assert any("NO está en la red" in w for w in result.phases.warnings)


def test_row_export_carries_the_phase_keys():
    row = analyse(make_demo("MWCNT_FeSe", seed=3), auto_preprocess=True).to_dict()
    assert "FeSe_tetragonal" in row["fases"]


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_corroboration_threshold_is_a_fraction(fraction):
    assert 0.0 < CORROBORATION <= 1.0
