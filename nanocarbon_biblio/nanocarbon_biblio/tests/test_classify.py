"""Classifier tests, concentrated on the disambiguation guards.

These are the rules most likely to be broken by a well-meaning lexicon edit, so
they are pinned here.
"""

from __future__ import annotations

import pandas as pd

from nanocarbon_biblio.classify import classify_record, crosstab, sample_for_validation, to_dataframe
from nanocarbon_biblio.records import Record


def _record(title: str, abstract: str = "", keywords: str = "") -> Record:
    return Record(key="t", source="test", title=title, abstract=abstract, keywords=keywords)


def test_nitrogen_and_defect_detection() -> None:
    labels = classify_record(_record(
        "Nitrogen-doped carbon nanotubes with Stone-Wales defects",
        "Pyridinic nitrogen and monovacancies were characterised by XPS.",
    ))
    assert labels["dopant"] == "nitrogen"
    assert "stone_wales" in labels["defect"] and "vacancy" in labels["defect"]
    assert labels["study_type"] == "experimental"


def test_p_doped_does_not_mean_phosphorus_without_a_guard() -> None:
    """The worst false positive in this literature: p-type read as phosphorus."""
    labels = classify_record(_record(
        "p-doped silicon nanowires", "We study p-doped and n-type silicon transport."
    ))
    assert "phosphorus" not in labels["dopant"]


def test_p_doped_does_mean_phosphorus_when_guarded() -> None:
    labels = classify_record(_record(
        "P-doped carbon nanotubes",
        "Phosphorus was introduced into the lattice; P-doped samples were prepared.",
    ))
    assert "phosphorus" in labels["dopant"]


def test_combined_study_type_needs_both_methods() -> None:
    labels = classify_record(_record(
        "Boron-doped carbon nanofibers",
        "Samples were prepared by CVD and characterised by Raman spectroscopy. "
        "Density functional theory was used to compute the formation energy.",
    ))
    assert labels["study_type"] == "combined"
    assert "dft" in labels["method_theory"]
    assert "raman" in labels["method_experiment"]


def test_host_ambiguity_flags_a_supported_non_carbon_dopant() -> None:
    labels = classify_record(_record(
        "N-doped TiO2 supported on multi-walled carbon nanotubes",
        "Nitrogen-doped TiO2 was deposited on MWCNTs for photocatalysis.",
    ))
    assert labels["dopant_host_ambiguous"] is True


def test_host_ambiguity_does_not_flag_a_genuine_doped_nanotube() -> None:
    labels = classify_record(_record(
        "Nitrogen-doped carbon nanotubes decorated with TiO2",
        "N-doped carbon nanotubes were grown by CVD and decorated with TiO2 nanoparticles.",
    ))
    assert labels["dopant_host_ambiguous"] is False


def test_3d_assembly_and_hybrid_flags() -> None:
    labels = classify_record(_record(
        "Nitrogen-doped carbon nanotube sponges hybridised with graphene",
        "A three-dimensional network of N-doped CNTs and reduced graphene oxide.",
    ))
    assert labels["is_3d_assembly"] is True
    assert labels["is_hybrid"] is True
    assert labels["mentions_graphene"] is True


def test_codoping_is_its_own_label() -> None:
    labels = classify_record(_record(
        "N,S-codoped carbon nanofibers", "Nitrogen and sulfur co-doped fibers for ORR."
    ))
    assert "codoped" in labels["dopant"]
    assert "nitrogen" in labels["dopant"] and "sulfur" in labels["dopant"]


def test_crosstab_explodes_multi_valued_facets() -> None:
    frame = pd.DataFrame([
        {"dopant": "nitrogen|sulfur", "application": "orr_fuelcell|supercapacitor"},
        {"dopant": "nitrogen", "application": "orr_fuelcell"},
    ])
    table = crosstab(frame)
    assert table.loc["nitrogen", "orr_fuelcell"] == 2
    assert table.loc["sulfur", "supercapacitor"] == 1


def test_dataframe_and_validation_sample_are_deterministic() -> None:
    records = [
        _record(f"Nitrogen-doped carbon nanotubes study {i}", "Prepared by CVD.")
        for i in range(20)
    ]
    for i, record in enumerate(records):
        record.key, record.year = f"k{i}", 2000 + i
        record.labels.update(classify_record(record))
    frame = to_dataframe(records)
    first = sample_for_validation(frame, n=5)
    second = sample_for_validation(frame, n=5)
    assert list(first.key) == list(second.key)
