"""Deduplication tests, including the cases that make naive dedup fail."""

from __future__ import annotations

from pathlib import Path

from nanocarbon_biblio.dedupe import deduplicate, identified_by_source, overlap_table
from nanocarbon_biblio.loaders import load_directory
from nanocarbon_biblio.records import Record


def test_cross_database_duplicate_is_merged(raw_dir: Path) -> None:
    """The Terrones paper appears in both databases with different casing."""
    result = deduplicate(load_directory(raw_dir))
    assert result.n_input == 7
    assert len(result.unique) == 6
    merged = [r for r in result.unique if r.doi_key == "10.1016/j.carbon.2005.01.001"]
    assert len(merged) == 1
    assert merged[0].labels["sources"] == ["scopus", "wos"]


def test_overlap_table_partitions_the_corpus(raw_dir: Path) -> None:
    result = deduplicate(load_directory(raw_dir))
    overlap = overlap_table(result)
    assert overlap["both"] == 1
    assert overlap["scopus_only"] == 3
    assert overlap["wos_only"] == 2
    assert overlap["scopus_only"] + overlap["wos_only"] + overlap["both"] == overlap["total_unique"]


def test_distinct_dois_are_never_merged_on_title_similarity() -> None:
    """Part I and part II of the same study share almost the whole title."""
    records = [
        Record(key="a", source="scopus", year=2020, doi="10.1/aaa",
               title="Nitrogen doped carbon nanotubes for oxygen reduction part I"),
        Record(key="b", source="scopus", year=2020, doi="10.1/bbb",
               title="Nitrogen doped carbon nanotubes for oxygen reduction part II"),
    ]
    result = deduplicate(records, title_threshold=80.0)
    assert len(result.unique) == 2


def test_titles_merge_when_doi_is_missing() -> None:
    """No DOI on either side: fuzzy title is the only signal available."""
    records = [
        Record(key="a", source="scopus", year=2001,
               title="Defect structure of multi-walled carbon nanotubes grown by CVD"),
        Record(key="b", source="wos", year=2002,
               title="Defect structure of multi walled carbon nanotubes grown by CVD"),
    ]
    result = deduplicate(records)
    assert len(result.unique) == 1
    assert result.unique[0].labels["sources"] == ["scopus", "wos"]


def test_year_window_bounds_the_comparison() -> None:
    """Same title 5 years apart is an erratum or a different work, not a duplicate."""
    records = [
        Record(key="a", source="scopus", year=2001,
               title="Defect structure of multi-walled carbon nanotubes grown by CVD"),
        Record(key="b", source="wos", year=2010,
               title="Defect structure of multi-walled carbon nanotubes grown by CVD"),
    ]
    assert len(deduplicate(records, year_window=1).unique) == 2


def test_representative_prefers_a_record_with_an_abstract() -> None:
    records = [
        Record(key="a", source="scopus", year=2020, doi="10.1/x", title="A doped nanotube study"),
        Record(key="b", source="wos", year=2020, doi="10.1/x", title="A doped nanotube study",
               abstract="A long enough abstract to be considered usable by the classifier rules."),
    ]
    result = deduplicate(records)
    assert len(result.unique) == 1
    assert result.unique[0].key == "b"


def test_subset_titles_do_not_merge() -> None:
    """A short title contained in a longer one is a different paper.

    token_set_ratio scores this pair 100 because the intersection equals the
    shorter token set, which is why the matcher uses token-sort semantics
    instead. Both records carry the same author and no DOI, so nothing else
    would stop the merge.
    """
    records = [
        Record(key="a", source="scopus", year=2015, authors="Zhang, L.",
               title="Nitrogen doped carbon nanotubes"),
        Record(key="b", source="wos", year=2015, authors="Zhang, L.",
               title="Nitrogen doped carbon nanotubes for the oxygen reduction "
                     "reaction in alkaline media"),
    ]
    assert len(deduplicate(records).unique) == 2


def test_word_order_and_punctuation_still_merge() -> None:
    """What the fuzzy pass is actually for: the same work, typeset differently."""
    records = [
        Record(key="a", source="scopus", year=2015, authors="Zhang, L.",
               title="Defect structure of multi-walled carbon nanotubes grown by CVD"),
        Record(key="b", source="wos", year=2015, authors="Zhang, L",
               title="Defect structure of multi walled carbon nanotubes grown by C.V.D."),
    ]
    assert len(deduplicate(records).unique) == 1


def test_identified_by_source_counts_before_deduplication() -> None:
    """PRISMA's identification row: raw retrieval, overlap counted twice."""
    records = [
        Record(key="scopus:f:0", source="scopus", year=2015, doi="10.1/x", title="A shared paper about doped nanotubes"),
        Record(key="wos:f:0", source="wos", year=2015, doi="10.1/x", title="A shared paper about doped nanotubes"),
        Record(key="scopus:f:1", source="scopus", year=2016, doi="10.1/y", title="A scopus only paper about vacancies"),
    ]
    result = deduplicate(records)
    assert identified_by_source(result) == {"scopus": 2, "wos": 1}
    assert sum(identified_by_source(result).values()) == result.n_input
