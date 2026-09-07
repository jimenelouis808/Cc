"""Dichalcogenides, and models with a chosen number of peaks per region."""

from __future__ import annotations

import pytest

from ramancarbon.analysis.tmd import (
    RESOLUTION_LIMIT,
    analyse_tmd,
    count_layers,
    identify_material,
    tmd_materials,
)
from ramancarbon.core.peaks import find_peaks
from ramancarbon.core.preprocess import preprocess
from ramancarbon.examples.demo_data import TMD_DEMOS, make_demo, make_tmd_demo
from ramancarbon.models.deconvolution import (
    D_REGION_LADDER,
    G_REGION_LADDER,
    build_region_model,
)
from ramancarbon.models.fitting import fit_model


# -- TMD -----------------------------------------------------------------
@pytest.mark.parametrize("material, layers", TMD_DEMOS)
def test_material_is_identified(material, layers):
    result = analyse_tmd(make_tmd_demo(material, layers, seed=1))
    assert result.material == material


@pytest.mark.parametrize("layers", ["1", "2", "3", "bulk"])
def test_mos2_layer_count_round_trips(layers):
    """The demo places the two modes at the separation the database says
    that layer count has, so recovering it is a genuine round trip."""
    result = analyse_tmd(make_tmd_demo("MoS2", layers, seed=2))
    assert result.layers == layers


def test_separation_is_what_counts_layers():
    one = analyse_tmd(make_tmd_demo("MoS2", "1", seed=3))
    bulk = analyse_tmd(make_tmd_demo("MoS2", "bulk", seed=3))
    assert bulk.separation > one.separation + 4.0


def test_wse2_falls_back_to_the_b12g_mode():
    """Its two modes are nearly degenerate, so the separation method cannot
    work and the code must say so rather than produce a number."""
    material = next(m for m in tmd_materials() if m.key == "WSe2")
    assert not material.counts_layers_by_separation
    result = analyse_tmd(make_tmd_demo("WSe2", "2", seed=4))
    assert result.separation is None
    assert result.layers == "≥2"
    assert "B¹₂g" in result.layer_reason


def test_a_monolayer_has_no_b12g():
    result = analyse_tmd(make_tmd_demo("WSe2", "1", seed=5))
    assert result.layers.startswith("1")


def test_separation_outside_every_range_is_refused():
    material = next(m for m in tmd_materials() if m.key == "MoS2")
    layers, reason = count_layers(material, 15.0)
    assert layers is None and "DEBAJO" in reason
    layers, reason = count_layers(material, 40.0)
    assert layers is None and "satura" in reason


def test_metallic_phase_is_detected_from_the_j_modes():
    result = analyse_tmd(make_tmd_demo("MoS2", "1", seed=6, phase_1t=True))
    assert result.phase == "1T'"
    assert "J1" in result.phase_reason


def test_absence_of_j_modes_is_stated_as_weak_evidence():
    result = analyse_tmd(make_tmd_demo("MoS2", "1", seed=7))
    assert result.phase == "2H"
    assert "NO descarta" in result.phase_reason


def test_coarse_sampling_is_flagged_for_tmd():
    """The layer-count boundaries are 2-3 cm-1 apart, so resolution matters
    far more here than for carbon."""
    result = analyse_tmd(make_tmd_demo("MoS2", "2", seed=8, step=RESOLUTION_LIMIT * 2))
    assert any("resolución" in w or "paso de muestreo" in w for w in result.warnings)


def test_a_carbon_spectrum_is_not_called_a_tmd():
    result = analyse_tmd(make_demo("MWCNT", seed=9))
    assert result.material is None
    assert "no se reconoce" in " ".join(result.warnings)


def test_identification_scores_are_ordered():
    spectrum, _ = preprocess(make_tmd_demo("WS2", "1", seed=10))
    scores = identify_material(find_peaks(spectrum, min_distance_cm=4.0,
                                          min_fwhm_cm=1.5))
    assert scores[0][0] == "WS2"
    assert scores == sorted(scores, key=lambda item: -item[1])


def test_every_tmd_entry_is_sourced():
    for material in tmd_materials():
        assert material.notes
        if material.counts_layers_by_separation:
            assert material.separation_source


def test_tmd_result_is_flat_for_a_table():
    row = analyse_tmd(make_tmd_demo("MoS2", "2", seed=11)).to_dict()
    for key, value in row.items():
        assert isinstance(value, (str, int, float, type(None))), key


# -- configurable peaks per region ---------------------------------------
@pytest.mark.parametrize("n_d, n_g", [(1, 1), (2, 1), (3, 2), (4, 2), (3, 3), (5, 3)])
def test_region_model_has_the_requested_component_count(n_d, n_g):
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    model = build_region_model(spectrum, n_d=n_d, n_g=n_g)
    assert len(model.peaks) == n_d + n_g


def test_the_usual_convention_gives_the_sadezky_five_band_model():
    """Three in D and two in G is D4 + D + D3 + G + D'."""
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    model = build_region_model(spectrum, n_d=3, n_g=2)
    assert {p.band for p in model.peaks} == {"D4", "D", "D3", "G", "D'"}


def test_named_components_come_in_the_documented_order():
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    for count in (1, 2, 3):
        model = build_region_model(spectrum, n_d=count, n_g=1)
        named = {p.band for p in model.peaks if p.band} - {"G"}
        assert named == set(D_REGION_LADDER[:count])
    for count in (1, 2, 3):
        model = build_region_model(spectrum, n_d=1, n_g=count)
        named = {p.band for p in model.peaks if p.band} - {"D"}
        assert named == set(G_REGION_LADDER[:count])


def test_extra_components_are_unnamed_and_stay_unnamed():
    """A component with no name has no physical interpretation, so it must
    not be given a band key that would feed the ratios or the classifier."""
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    model = build_region_model(spectrum, n_d=5, n_g=4)
    unnamed = [p for p in model.peaks if p.band is None]
    assert len(unnamed) == (5 - len(D_REGION_LADDER)) + (4 - len(G_REGION_LADDER))
    assert all(p.name.startswith(("Dx", "Gx")) for p in unnamed)


def test_extra_components_do_not_stack_on_the_named_ones():
    """Only the unnamed ones are placed by this code. Named bands can be
    genuinely close — G⁻ and G sit 12 cm⁻¹ apart in the database — so the
    spacing rule applies to what the filler puts down, not to everything."""
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    model = build_region_model(spectrum, n_d=5, n_g=3)
    centres = sorted(p.centre for p in model.peaks)
    for peak in model.peaks:
        if peak.band is not None:
            continue
        nearest = min(
            abs(peak.centre - other) for other in centres if other != peak.centre
        )
        assert nearest > 25.0, f"{peak.name} sits {nearest:.0f} cm⁻¹ from a neighbour"


def test_region_model_fits_and_improves_with_components():
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    simple = fit_model(spectrum, build_region_model(spectrum, n_d=1, n_g=1))
    rich = fit_model(spectrum, build_region_model(spectrum, n_d=3, n_g=2))
    assert rich.r_squared > simple.r_squared
    assert rich.durbin_watson > simple.durbin_watson


@pytest.mark.parametrize("n_d, n_g", [(0, 2), (3, 0), (-1, 1)])
def test_at_least_one_component_per_region(n_d, n_g):
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    with pytest.raises(ValueError, match="al menos una"):
        build_region_model(spectrum, n_d=n_d, n_g=n_g)


def test_absurd_component_counts_are_refused():
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    with pytest.raises(ValueError, match="más de lo que"):
        build_region_model(spectrum, n_d=8, n_g=8)


def test_region_model_defaults_to_pseudo_voigt():
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))
    model = build_region_model(spectrum, n_d=3, n_g=2)
    assert all(p.profile == "pseudo_voigt" for p in model.peaks)
