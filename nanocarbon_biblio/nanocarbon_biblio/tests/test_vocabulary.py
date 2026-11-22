"""Tests for the vocabulary reference and the query-coverage check.

The coverage check is only worth having if it fires on a real hole, so the
central test introduces one deliberately. The rest pin the regex expander,
whose job is to turn patterns into words a human can read.
"""

from __future__ import annotations

from pathlib import Path

from nanocarbon_biblio.vocabulary import (
    BY_DESIGN_ABSENT, coverage_report, facet_dictionary, readable_terms, write_dictionary,
)

QUERIES = Path(__file__).resolve().parents[2] / "queries"


def _query_text() -> str:
    return "".join(
        (QUERIES / name).read_text(encoding="utf-8")
        for name in ("scopus.md", "wos.md")
        if (QUERIES / name).exists()
    )


def test_expander_produces_words_not_regex_fragments() -> None:
    """`dop(?:ed|ing|ant)` must read as doped/doping/dopant, not as four pieces."""
    terms = readable_terms([(r"\bnitrogen[\s\-]?dop(?:ed|ing|ant)", None, False)])
    assert "nitrogen doped" in terms
    assert "nitrogen doping" in terms
    assert "nitrogen dopant" in terms
    for fragment in ("dop", "ed", "ing", "ant"):
        assert fragment not in terms


def test_expander_drops_connectives_and_keeps_phrases() -> None:
    terms = readable_terms([(r"\bdoped with boron\b", None, False)])
    assert terms == ["doped with boron"]
    assert readable_terms([(r"\bpyridinic\b|\bpyrrolic\b", None, False)]) == [
        "pyridinic", "pyrrolic"
    ]


def test_dictionary_covers_every_facet_and_label() -> None:
    dictionary = facet_dictionary()
    assert set(dictionary) >= {"dopant", "defect", "application", "morphology"}
    assert all(terms for labels in dictionary.values() for terms in labels.values())
    assert "pyridinic" in dictionary["dopant"]["nitrogen"]


def test_the_real_query_covers_every_retrieval_label() -> None:
    """The shipped query must be able to retrieve what the classifier labels."""
    rows = coverage_report(_query_text())
    uncovered = [f"{r['facet']}/{r['label']}" for r in rows if not r["covered"]]
    assert not uncovered, f"la consulta no puede recuperar: {uncovered}"


def test_coverage_check_fires_on_a_real_hole() -> None:
    """A checker that can only say 'fine' is worthless: prove it can say 'not fine'.

    Scoring the shipped lexicon against a query that mentions no dopant at all
    must report the dopant labels as unreachable.
    """
    rows = coverage_report("TITLE-ABS-KEY('carbon nanotube*')")
    uncovered = [r for r in rows if not r["covered"]]
    assert len(uncovered) > 5
    assert any(r["facet"] == "dopant" for r in uncovered)
    # The host facet is still satisfied by that query.
    assert any(r["label"] == "carbon_1d" and r["covered"] for r in rows)


def test_by_design_absences_are_not_reported_as_holes() -> None:
    rows = coverage_report("TITLE-ABS-KEY('carbon nanotube*')")
    graphene = [r for r in rows if f"{r['facet']}/{r['label']}" in BY_DESIGN_ABSENT]
    assert graphene and all(r["covered"] for r in graphene)
    assert all(r["by_design"] for r in graphene)


def test_write_dictionary_is_self_describing(tmp_path: Path) -> None:
    path = write_dictionary(tmp_path / "DICCIONARIO.md", query_text=_query_text())
    text = path.read_text(encoding="utf-8")
    assert "Generado automáticamente" in text
    assert "No lo edites a mano" in text
    assert "## Cobertura de la consulta" in text
    assert "pyridinic" in text
    # The scope decision must be explained, not silently passed.
    assert "PROTOCOL.md §2.1" in text


def test_dictionary_is_committed_and_current() -> None:
    """The checked-in dictionary must match what the lexicon generates today."""
    shipped = QUERIES / "DICCIONARIO.md"
    assert shipped.exists(), "falta queries/DICCIONARIO.md; corre `cli vocab`"
    text = shipped.read_text(encoding="utf-8")
    for label in facet_dictionary()["dopant"]:
        assert f"**{label}**" in text, f"{label} no está en el diccionario publicado"
