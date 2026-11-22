"""Tests for the Methods-section evidence: query recall and coder agreement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanocarbon_biblio.records import Record
from nanocarbon_biblio.validation import (
    GoldItem, cohens_kappa, load_gold_standard, merge_gold, read_reference_export,
    score_agreement, score_recall, write_gold_standard,
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


# --------------------------------------------------------------------- seeding

BIBTEX = """@article{iijima1991,
  title = {Helical microtubules of graphitic carbon},
  author = {Iijima, Sumio},
  journal = {Nature},
  year = {1991},
  doi = {10.1038/354056a0}
}
@article{noDoi,
  title = {A paper with no DOI at all},
  year = {1998}
}
@article{junk,
  author = {Nobody}
}
"""

RIS = """TY  - JOUR
TI  - Nitrogen-doped carbon nanotube arrays
AU  - Gong, Kuanping
PY  - 2009
DO  - 10.1126/science.1168049
ER  - 
TY  - JOUR
TI  - Helical microtubules of graphitic carbon
PY  - 1991
DO  - 10.1038/354056a0
ER  - 
"""


def test_bibtex_reader_extracts_the_three_fields(tmp_path: Path) -> None:
    path = tmp_path / "library.bib"
    path.write_text(BIBTEX, encoding="utf-8")
    items = read_reference_export(path)
    # The entry with neither title nor DOI cannot be matched, so it is dropped.
    assert len(items) == 2
    assert items[0].doi_key == "10.1038/354056a0"
    assert items[0].year == 1991
    assert items[0].title == "Helical microtubules of graphitic carbon"
    assert items[1].doi_key == "" and items[1].year == 1998
    assert all(not item.verified for item in items)


def test_ris_reader_handles_zotero_style_exports(tmp_path: Path) -> None:
    path = tmp_path / "zotero.ris"
    path.write_text(RIS, encoding="utf-8")
    items = read_reference_export(path)
    assert [i.year for i in items] == [2009, 1991]
    assert items[0].doi_key == "10.1126/science.1168049"


def test_reference_export_falls_back_to_the_database_loaders(raw_dir: Path) -> None:
    """A Scopus search can seed the gold standard directly."""
    items = read_reference_export(raw_dir / "scopus_chunk1.csv")
    assert len(items) == 4
    assert any(i.doi_key == "10.1016/j.carbon.2005.01.001" for i in items)


def test_merge_keeps_human_annotations_across_sittings(tmp_path: Path) -> None:
    """Harvesting from a second source must not wipe the 'why' already written."""
    annotated = [GoldItem(doi="10.1038/354056a0", title="Helical microtubules",
                          year=1991, why="seminal", verified=True)]
    bib = tmp_path / "l.bib"
    bib.write_text(BIBTEX, encoding="utf-8")
    merged, added = merge_gold(annotated, read_reference_export(bib))
    assert added == 1                       # only the no-DOI entry is new
    assert merged[0].why == "seminal"       # untouched
    assert merged[0].verified is True


def test_merge_deduplicates_by_title_when_dois_are_absent() -> None:
    a = [GoldItem(title="Defect structure of multi-walled carbon nanotubes")]
    b = [GoldItem(title="Defect structure of multi-walled carbon nanotubes")]
    merged, added = merge_gold(a, b)
    assert added == 0 and len(merged) == 1


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    items = [GoldItem(doi="10.1/x", title="A title", year=2010, why="porque sí", verified=True)]
    path = write_gold_standard(items, tmp_path / "gold.csv")
    back = load_gold_standard(path)
    assert len(back) == 1
    assert back[0].why == "porque sí"
    assert back[0].verified is True
    assert back[0].year == 2010
