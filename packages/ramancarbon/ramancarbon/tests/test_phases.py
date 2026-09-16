"""Identification of non-carbon sample phases (FeSe, Se, carbides)."""

from __future__ import annotations

import pytest

from ramancarbon import analyse
from ramancarbon.analysis.phases import (
    CORROBORATION,
    PhaseReport,
    find_phases,
    load_families,
    elements_of,
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


class TestTheRBMImpostors:
    """The case the catalogue exists for, end to end.

    A carbon spectrum from a sample decorated with FeSe: the two FeSe
    lines at 180 and 195 cm-1 and elemental selenium's at 237 and 250 all
    fall inside the radial-breathing-mode window, and converting them with
    omega = A/d + B gives four believable nanotube diameters, all false.
    """

    def test_every_impostor_is_kept_out_of_the_diameter_calculation(self):
        peaks = [
            pk(180.2), pk(195.4), pk(237.0), pk(250.5),
            pk(1350.0), pk(1580.0),
        ]
        report = find_phases(peaks)
        assert report.excluded_from_rbm == [180.2, 195.4, 237.0, 250.5]
        found = {i.phase.key for i in report.identifications if i.corroborated}
        assert {"FeSe_tetragonal", "Se_trigonal"} <= found
        # Only the carbon bands are left over.
        assert [p.position for p in report.unmatched_peaks] == [1350.0, 1580.0]

    def test_a_lone_catalogued_line_is_reported_rather_than_converted(self):
        """One selenium line and nothing to corroborate it.

        Too thin to name the phase -- the rule that needs corroboration is
        right -- and far too strong to turn into a 1.0 nm nanotube without
        a word, which is what happened before: the peak went through in
        silence because it was not a confirmed identification.
        """
        report = find_phases([pk(237.0), pk(1350.0), pk(1580.0)])
        assert report.excluded_from_rbm == []
        assert [position for position, _ in report.rbm_suspects] == [237.0]
        assert any("sin que nada la corrobore" in w for w in report.warnings)

    def test_the_catalogue_covers_the_rbm_window_beyond_the_selenides(self):
        """Anatase is the worst impostor of the lot: its 144 cm-1 E_g is
        the strongest and narrowest line in the catalogue that lands in
        the window, and it converts to a 1.6 nm tube."""
        report = find_phases([pk(144.0), pk(399.0), pk(515.0),
                              pk(639.0), pk(1350.0), pk(1580.0)])
        found = {i.phase.key for i in report.identifications if i.corroborated}
        assert "TiO2_anatase" in found
        assert 144.0 in report.excluded_from_rbm


# -- precursor phases: N, P, B, Cl, S ----------------------------------


def test_the_catalogue_covers_the_precursor_heteroatoms():
    """Every heteroatom the synthesis puts in has a phase that carries it.

    A coverage test. The point of these entries is not that the precursor
    is interesting but that an unreacted one is indistinguishable, in the
    carbon region, from the doped carbon you meant to make.
    """
    phases = load_phases()
    formulas = " ".join(p.formula for p in phases)
    for element in ("N", "P", "B", "Cl", "S", "Se"):
        assert element in formulas, element
    keys = {p.key for p in phases}
    assert {"melamine", "urea", "g_C3N4", "P_red", "H3BO3", "B2O3_glassy",
            "BN_hexagonal", "NH4Cl_salmiac", "S8_orthorhombic"} <= keys
    for phase in phases:
        assert phase.source, phase.key
        assert phase.confidence in {"high", "medium", "low"}, phase.key


def test_boron_nitride_is_separated_from_the_d_band_by_width_alone():
    """h-BN has one Raman band, at 1366 cm-1, inside the D band's range.

    Position cannot separate them and never will. What separates them is
    that the E2g of a h-BN crystal is ~10 cm-1 wide and a disordered
    carbon's D band is 50-150, so the catalogue puts the whole
    identification on a width rule.
    """
    sharp = find_phases([pk(1366.0, 300, fwhm=11.0), pk(1580.0, 400, 40.0)])
    assert "BN_hexagonal" in {
        i.phase.key for i in sharp.identifications if i.corroborated
    }

    broad = find_phases([pk(1358.0, 300, fwhm=70.0), pk(1580.0, 400, 40.0)])
    assert "BN_hexagonal" not in {
        i.phase.key for i in broad.identifications if i.corroborated
    }


def test_carbon_nitride_is_caught_below_the_carbon_region():
    """g-C3N4 has bands at 1233, 1310 and 1570 -- on top of the D and the
    G. A sample made from melamine that is really carbon nitride gives a
    carbon-looking spectrum and an I_D/I_G that means nothing. The bands
    that give it away are at 707 and 750, where carbon has nothing."""
    peaks = [pk(707.0, 300), pk(750.0, 200), pk(1233.0, 250),
             pk(1310.0, 300), pk(1570.0, 350)]
    report = find_phases(peaks, spectrum_range=(100.0, 3000.0))
    best = report.identifications[0]
    assert best.phase.key == "g_C3N4" and best.corroborated

    # Without the low-frequency pair it is just a disordered carbon.
    blind = find_phases([pk(1310.0, 300), pk(1570.0, 350)],
                        spectrum_range=(1000.0, 3000.0))
    assert "g_C3N4" not in {
        i.phase.key for i in blind.identifications if i.corroborated
    }


def test_sulfur_needs_its_line_at_473():
    """S8's other two lines, 153 and 219, sit inside the RBM window. Two
    coincidences there are what a clean nanotube looks like, so they are
    not allowed to identify sulfur on their own."""
    rbm_only = find_phases([pk(153.0), pk(219.0), pk(1580.0, 400, 40.0)],
                           spectrum_range=(100.0, 3000.0))
    assert "S8_orthorhombic" not in {
        i.phase.key for i in rbm_only.identifications if i.corroborated
    }
    complete = find_phases([pk(153.0), pk(219.0), pk(473.0, 400),
                            pk(1580.0, 400, 40.0)],
                           spectrum_range=(100.0, 3000.0))
    assert "S8_orthorhombic" in {
        i.phase.key for i in complete.identifications if i.corroborated
    }


def test_magnetite_is_not_reported_as_manganese_dioxide():
    """Fe3O4 sits at 668/540/310 and beta-MnO2 at 665/535: every line of
    the manganese oxide lands within tolerance of an iron one. Both match
    and both corroborate, so the report has to prefer the phase that
    explains more peaks and explains them closer."""
    peaks = [pk(310.0, 120), pk(540.0, 150), pk(668.0, 400)]
    report = find_phases(peaks, spectrum_range=(100.0, 1200.0))
    called = {i.phase.key for i in report.identifications if i.corroborated}
    assert called == {"Fe3O4_magnetite"}


def test_manganese_oxides_are_still_told_apart_from_each_other():
    """The rule above must not silence a real manganese oxide. All three
    share a band between 640 and 670, and what separates them is the rest:
    374 for the spinel, 535 for pyrolusite, 575 for birnessite."""
    for key, extra in (("Mn3O4_hausmannite", 374.0),
                       ("MnO2_beta", 535.0),
                       ("MnO2_birnessite", 575.0)):
        phase = next(p for p in load_phases() if p.key == key)
        peaks = [pk(phase.strong[0], 400), pk(extra, 200)]
        report = find_phases(peaks, spectrum_range=(100.0, 1200.0))
        called = {i.phase.key for i in report.identifications if i.corroborated}
        assert key in called, (key, called)


def test_a_phase_whose_strong_lines_you_did_not_measure_is_not_confirmed():
    """The mirror of the RBM rule, at the other end of the axis.

    g-C3N4's bands at 1233, 1310 and 1570 are the ones it shares with a
    disordered carbon; the ones that identify it are at 707 and 750. A
    spectrum that starts at 1000 cm-1 matches two of the three shared
    bands and none of the exclusive ones. That is not an identification --
    but it is not an absence either, and the report says which.
    """
    report = find_phases([pk(1310.0, 300), pk(1570.0, 350)],
                         spectrum_range=(1000.0, 3000.0))
    assert not any(i.corroborated for i in report.identifications)
    lead = next(i for i in report.identifications if i.phase.key == "g_C3N4")
    assert lead.strong_out_of_range
    assert any("ninguna de sus líneas fuertes" in w and "707" in w
               for w in report.warnings)


# -- saying what the sample is made of ---------------------------------


def test_a_formula_gives_up_its_elements():
    from ramancarbon.analysis.phases import elements_of

    assert elements_of("Fe3C") == {"Fe", "C"}
    assert elements_of("FeSe") == {"Fe", "Se"}       # not F + e + S + e
    assert elements_of("FeOOH") == {"Fe", "O", "H"}
    assert elements_of("C3H6N6") == {"C", "H", "N"}
    assert elements_of("") == set()


def test_declaring_the_elements_removes_the_impossible_phases():
    """A manganese oxide cannot be in a sample with no manganese. With
    three dozen phases catalogued those impossible matches are most of
    the noise in a real report."""
    peaks = [pk(p) for p in (119, 212, 273, 372, 477, 585, 621)]
    peaks += [pk(1355, 2000, 60), pk(1583, 2300, 50)]

    everything = find_phases(peaks, spectrum_range=(100.0, 3000.0))
    named = {i.phase.key for i in everything.identifications}
    assert {"Mn3O4_hausmannite", "Co3O4_spinel"} & named

    declared = find_phases(peaks, spectrum_range=(100.0, 3000.0),
                           elements=["C", "Fe", "Se"])
    for ident in declared.identifications:
        assert elements_of(ident.phase.formula) <= {"C", "Fe", "Se"}
    assert declared.elements == ["C", "Fe", "Se"]


def test_declaring_the_elements_makes_the_rbm_warning_louder_not_quieter():
    """The case the module exists for, and the one the caution silenced.

    Cementite's bands at 212 and 280 cm-1 are inside the RBM window. Its
    entry is LOW confidence — because cementite's Raman cross-section is
    poor, not because the positions are doubtful — so the rule that only
    a high-confidence phase may raise a suspect said nothing at all about
    a CVD sample grown on an iron catalyst, which is exactly the sample
    this program is for. Once the user has said there is iron, the line is
    worth raising, and the report states the confidence rather than hiding
    behind it.
    """
    peaks = [pk(212), pk(273), pk(1355, 2000, 60), pk(1583, 2300, 50)]

    quiet = find_phases(peaks, spectrum_range=(100.0, 3000.0))
    assert not any("Cementita" in label for _, label in quiet.rbm_suspects)

    loud = find_phases(peaks, spectrum_range=(100.0, 3000.0),
                       elements=["C", "Fe", "Se"])
    flagged = [label for position, label in loud.rbm_suspects if position == 212]
    assert flagged and "Cementita" in flagged[0]
    assert "confianza de la referencia: low" in flagged[0]


def test_a_clean_nanotube_is_still_not_interrupted_without_a_composition():
    """The caution has to stay where there is no composition to justify
    dropping it, or every real breathing mode draws a flag."""
    report = find_phases([pk(150), pk(168), pk(190), pk(214), pk(265),
                          pk(1580, 400, 20)],
                         spectrum_range=(100.0, 3000.0))
    assert not report.rbm_suspects
    assert not report.found_anything
