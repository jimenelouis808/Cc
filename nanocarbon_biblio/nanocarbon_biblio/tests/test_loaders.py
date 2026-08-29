"""Loader tests, including the round-trip guarantees the R bridge depends on."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanocarbon_biblio.loaders import load_any, load_directory, load_scopus_csv, load_wos_plaintext


def test_wos_plaintext_parses_records(raw_dir: Path) -> None:
    records = load_wos_plaintext(raw_dir / "wos_chunk1.txt")
    assert len(records) == 3
    first = records[0]
    assert first.year == 2005
    assert first.doi == "10.1016/j.carbon.2005.01.001"
    assert first.cited_by == 145
    # Continuation lines must be folded into one title.
    assert first.title == "Nitrogen-doped carbon nanotubes: synthesis and electronic structure"
    # DE and ID are pooled into `keywords`.
    assert "pyridinic nitrogen" in first.keywords and "DEFECTS" in first.keywords


def test_wos_raw_block_is_preserved_for_reexport(raw_dir: Path) -> None:
    """The stored block must still be parseable by convert2df: tags plus ER."""
    records = load_wos_plaintext(raw_dir / "wos_chunk1.txt")
    raw = records[0].raw
    assert raw.startswith("PT J")
    assert raw.rstrip().endswith("ER")
    # Cited references survive verbatim - this is what RPYS and co-citation need.
    assert "IIJIMA S, 1991, NATURE" in raw
    assert "FN Clarivate" not in raw  # file header must not leak into a record


def test_record_without_doi_still_loads(raw_dir: Path) -> None:
    records = load_wos_plaintext(raw_dir / "wos_chunk1.txt")
    undated = [r for r in records if r.uid == "WOS:000400000100003"][0]
    assert undated.doi_key == ""
    assert undated.year == 2018


def test_scopus_csv_parses_records(raw_dir: Path) -> None:
    records = load_scopus_csv(raw_dir / "scopus_chunk1.csv")
    assert len(records) == 4
    assert records[0].source_title == "Carbon"
    assert records[0].n_references == 2
    assert records[3].doi_key == ""  # blank DOI column


def test_load_any_detects_format_by_content(raw_dir: Path, tmp_path: Path) -> None:
    """A WoS file with a .csv extension must still load as WoS."""
    mislabelled = tmp_path / "actually_wos.csv"
    mislabelled.write_text((raw_dir / "wos_chunk1.txt").read_text(encoding="utf-8"), encoding="utf-8")
    records = load_any(mislabelled)
    assert len(records) == 3
    assert records[0].source == "wos"


def test_load_directory_concatenates(raw_dir: Path) -> None:
    records = load_directory(raw_dir)
    assert len(records) == 7
    assert {r.source for r in records} == {"scopus", "wos"}


def test_scopus_loader_rejects_wrong_format(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.csv"
    bogus.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no Scopus 'Title' column"):
        load_scopus_csv(bogus)
