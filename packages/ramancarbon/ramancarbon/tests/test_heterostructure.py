"""Oxide + dichalcogenide composites (MoO₃@MoSe₂, MoO₂@MoSe₂)."""

from __future__ import annotations

from ramancarbon.analysis.heterostructure import (
    HeterostructureResult,
    analyse_heterostructure,
    expected_oxides,
    find_oxides,
    load_oxides,
)
from ramancarbon.analysis.tmd import analyse_tmd
from ramancarbon.core.peaks import PeakMeasurement
from ramancarbon.examples.demo_data import TMD_OXIDE_DEMOS, make_tmd_demo


def pk(position: float, height: float = 100.0, fwhm: float = 8.0) -> PeakMeasurement:
    return PeakMeasurement(
        position=position,
        height=height,
        fwhm=fwhm,
        area=height * fwhm,
        prominence=height,
        snr=30.0,
        significance=40.0,
    )


FULL = (100.0, 1100.0)


# -- catalogue --------------------------------------------------------


def test_catalogue_is_self_consistent():
    for oxide in load_oxides():
        assert oxide.bands
        for line in oxide.strong:
            assert any(abs(line - b) < 1e-6 for b in oxide.bands)
        for line in oxide.signature:
            assert any(abs(line - b) < 1e-6 for b in oxide.strong), oxide.key


def test_the_search_is_restricted_to_chemically_possible_oxides():
    """A molybdenum sample cannot produce a tungsten oxide."""
    keys = {o.key for o in expected_oxides("MoSe2")}
    assert "MoO3_alpha" in keys
    assert not any(k.startswith("WO") for k in keys)


def test_conducting_oxides_are_flagged_as_such():
    by_key = {o.key: o for o in load_oxides()}
    assert by_key["MoO2"].is_metallic
    assert not by_key["MoO3_alpha"].is_metallic


# -- detection --------------------------------------------------------


def test_moo3_is_identified_from_its_signature_pair():
    peaks = [pk(158), pk(291), pk(666), pk(819, 400), pk(995, 200)]
    matches = find_oxides(peaks, "MoSe2", spectrum_range=FULL)
    assert matches[0].oxide.key == "MoO3_alpha"
    assert matches[0].corroborated


def test_overlapping_lines_alone_never_identify_an_oxide():
    """MoO2 has bands at 203 and 228; MoSe2's own modes are at 240 and 287.
    Without the signature gate every MoSe2 spectrum contains MoO2."""
    peaks = [pk(203), pk(228)]
    matches = find_oxides(peaks, "MoSe2", spectrum_range=FULL)
    assert not any(m.corroborated for m in matches)
    moo2 = next(m for m in matches if m.oxide.key == "MoO2")
    assert "exclusivas" in moo2.reason


def test_a_short_spectrum_cannot_rule_an_oxide_out():
    result = analyse_heterostructure([pk(240, 500)], "MoSe2", spectrum_range=(100.0, 500.0))
    assert not result.found_anything
    assert any("no significa nada" in w for w in result.warnings)


def test_amorphous_oxide_needs_a_broad_band():
    narrow = find_oxides([pk(950, 200, fwhm=8.0)], "MoSe2", spectrum_range=FULL)
    broad = find_oxides([pk(950, 200, fwhm=60.0)], "MoSe2", spectrum_range=FULL)
    assert not any(
        m.corroborated for m in narrow if m.oxide.key == "MoOx_amorphous"
    )
    assert any(m.corroborated for m in broad if m.oxide.key == "MoOx_amorphous")


# -- oxidation index --------------------------------------------------


def test_the_index_refuses_a_contaminated_denominator():
    """MoO3 has bands at 246 and 291 where MoSe2's modes are. Measuring the
    host there measures the oxide and the ratio comes out as 1 regardless."""
    peaks = [pk(158), pk(291, 300), pk(666), pk(819, 400), pk(995, 200)]
    result = analyse_heterostructure(
        peaks,
        "MoSe2",
        mode_positions={"A1g": 241.0, "E2g": 288.0},
        spectrum_range=FULL,
    )
    assert result.oxidation_index is None
    assert any("caen encima de bandas del óxido" in w for w in result.warnings)


def test_the_index_names_the_mode_it_used():
    """MoO2 leaves both MoSe2 modes clear, so the index is computable —
    against the taller of the two, and the report says which."""
    peaks = [
        pk(203), pk(228, 80), pk(240, 500), pk(287, 200),
        pk(495, 300), pk(589), pk(744, 250),
    ]
    result = analyse_heterostructure(
        peaks,
        "MoSe2",
        mode_positions={"A1g": 240.0, "E2g": 287.0},
        spectrum_range=FULL,
    )
    assert result.index_mode == "A1g"
    assert result.oxidation_index is not None


# -- interface --------------------------------------------------------


def test_no_shift_is_reported_as_no_interface_evidence():
    result = analyse_heterostructure(
        [pk(240, 500)],
        "MoSe2",
        mode_positions={"A1g": 240.0, "E2g": 287.0},
        reference_positions={"A1g": 240.0, "E2g": 287.0},
        spectrum_range=FULL,
    )
    assert "sin evidencia" in result.interface_verdict
    assert "no ve topología" in result.interface_reason


def test_out_of_plane_only_shift_reads_as_doping():
    result = analyse_heterostructure(
        [pk(243, 500)],
        "MoSe2",
        mode_positions={"A1g": 243.0, "E2g": 287.2},
        reference_positions={"A1g": 240.0, "E2g": 287.0},
        spectrum_range=FULL,
    )
    assert "dopado" in result.interface_verdict


def test_both_modes_moving_together_reads_as_strain():
    result = analyse_heterostructure(
        [pk(243, 500)],
        "MoSe2",
        mode_positions={"A1g": 243.0, "E2g": 290.0},
        reference_positions={"A1g": 240.0, "E2g": 287.0},
        spectrum_range=FULL,
    )
    assert "deformación" in result.interface_verdict


def test_a_large_shift_suggests_checking_calibration_first():
    result = analyse_heterostructure(
        [pk(246, 500)],
        "MoSe2",
        mode_positions={"A1g": 246.0, "E2g": 287.0},
        reference_positions={"A1g": 240.0, "E2g": 287.0},
        spectrum_range=FULL,
    )
    assert any("520.7" in w for w in result.warnings)


# -- notation ---------------------------------------------------------


def test_core_shell_notation_is_only_used_when_asserted():
    result = analyse_heterostructure(
        [pk(158), pk(666), pk(819, 400), pk(995, 200), pk(291)],
        "MoSe2",
        spectrum_range=FULL,
    )
    assert result.composition() == "MoO₃ + MoSe₂"
    assert result.composition(topology="core_shell") == "MoO₃@MoSe₂"


# -- end to end -------------------------------------------------------


def test_demo_composites_are_identified_end_to_end():
    for i, (material, layers, oxide) in enumerate(TMD_OXIDE_DEMOS):
        spectrum = make_tmd_demo(
            material, layers, high=1100.0, oxide=oxide, seed=20 + i
        )
        result = analyse_tmd(spectrum)
        assert result.material == material
        assert result.oxides is not None
        assert oxide in {o.key for o in result.oxides.oxides_present}, spectrum.name


def test_a_clean_dichalcogenide_reports_no_oxide():
    result = analyse_tmd(make_tmd_demo("MoSe2", "bulk", high=1100.0, seed=5))
    assert not result.oxides.found_anything


def test_metallic_oxide_warns_about_the_electrochemistry_tab():
    spectrum = make_tmd_demo("MoSe2", "bulk", high=1100.0, oxide="MoO2", seed=7)
    result = analyse_tmd(spectrum)
    assert any("CONDUCTOR" in w for w in result.oxides.warnings)


def test_laser_oxidation_is_always_raised_when_an_oxide_is_found():
    spectrum = make_tmd_demo("MoS2", "bulk", high=1100.0, oxide="MoO3_alpha", seed=9)
    result = analyse_tmd(spectrum)
    assert any("LÁSER" in w for w in result.oxides.warnings)


def test_scan_can_be_switched_off():
    spectrum = make_tmd_demo("MoSe2", "bulk", high=1100.0, oxide="MoO3_alpha", seed=7)
    off = analyse_tmd(spectrum, check_oxides=False).oxides
    assert isinstance(off, HeterostructureResult) and not off.enabled
    assert "desactivada" in off.summary()


def test_row_export_carries_the_composition():
    spectrum = make_tmd_demo("MoSe2", "bulk", high=1100.0, oxide="MoO3_alpha", seed=7)
    row = analyse_tmd(spectrum).to_dict()
    assert "MoO₃" in row["composicion"]
    assert row["oxidos"] == "MoO3_alpha"
