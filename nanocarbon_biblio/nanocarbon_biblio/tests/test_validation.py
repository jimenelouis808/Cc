"""Tests for the Methods-section evidence: query recall and coder agreement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanocarbon_biblio.records import Record
from nanocarbon_biblio.validation import (
    GoldItem, cohens_kappa, load_gold_standard, score_agreement, score_recall,
)


def _rec(key: str, title: str, doi: str = "", source: str = "scopus") -> Record:
    return Record(key=key, source=source, title=title, doi=doi, year=2010)


def test_kappa_punishes_agreement_that_is_only_class_bias() -> None:
    """Two coders who both always say 'experimental' have learned nothing."""
    truth = ["a"] * 8 + ["b"] * 2
    assert cohens_kappa(truth, ["a"] * 10) == 0.0          # 80% accurate, kappa 0
    assert cohens_kappa(truth, truth) == 1.0
    assert cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) == -1.0
    assert cohens_kappa([], []) == 0.0
    assert cohens_kappa(["a"], ["a", "b"]) == 0.0          # mismatched lengths


def test_recall_matches_by_doi_then_by_title() -> None:
    records = [
        _rec("s1", "Nitrogen-doped carbon nanotubes for oxygen reduction", "10.1/aaa"),
        _rec("w1", "Stone-Wales defects in single walled carbon nanotubes", source="wos"),
    ]
    gold = [
        GoldItem(doi="10.1/AAA", title="wrong title entirely", verified=True),
        GoldItem(title="Stone-Wales defects in single-walled carbon nanotubes", verified=True),
        GoldItem(title="A paper that is simply not in the corpus at all", verified=True),
    ]
    report = score_recall(records, gold)
    assert report.n_found == 2
    assert report.relative_recall == 2 / 3
    assert report.rows.loc[0, "matched_by"] == "doi"          # DOI beats a wrong title
    assert report.rows.loc[1, "matched_by"].startswith("title")
    assert len(report.missing()) == 1
    assert report.by_source == {"scopus": 1, "wos": 1}


def test_recall_flags_unverified_gold_entries() -> None:
    report = score_recall([], [GoldItem(title="Something long enough to match on", verified=False)])
    assert report.unverified == 1
    assert report.relative_recall == 0.0


def test_gold_loader_skips_unfilled_template_rows(tmp_path: Path, capsys) -> None:
    path = tmp_path / "gold.csv"
    path.write_text(
        "doi,title,year,why,verified\n"
        "10.1/x,A real entry,2010,seminal,TRUE\n"
        ",,,PENDIENTE de rellenar,FALSE\n",
        encoding="utf-8",
    )
    items = load_gold_standard(path)
    assert len(items) == 1
    assert items[0].verified is True
    assert "1 entradas usables, 1 filas" in capsys.readouterr().out


def test_agreement_scores_categorical_and_multilabel_facets() -> None:
    sheet = pd.DataFrame({
        "study_type": ["experimental", "theoretical", "combined", "experimental"],
        "manual_study_type": ["experimental", "theoretical", "experimental", "experimental"],
        "dopant": ["nitrogen", "nitrogen|sulfur", "boron", ""],
        "manual_dopant": ["nitrogen", "nitrogen|sulfur", "", "boron"],
    })
    results = score_agreement(sheet)
    assert results["study_type"]["n_coded"] == 4
    assert results["study_type"]["accuracy"] == 0.75
    assert "kappa" in results["study_type"]
    assert not results["study_type"]["per_class"].empty
    # The multi-label facet gets Jaccard, not kappa: two of three coded rows match.
    assert results["dopant"]["n_coded"] == 3
    assert 0.0 < results["dopant"]["mean_jaccard"] < 1.0
    assert "kappa" not in results["dopant"]


def test_agreement_ignores_uncoded_rows() -> None:
    sheet = pd.DataFrame({
        "study_type": ["experimental", "theoretical"],
        "manual_study_type": ["experimental", ""],
    })
    results = score_agreement(sheet)
    assert results["study_type"]["n_coded"] == 1
