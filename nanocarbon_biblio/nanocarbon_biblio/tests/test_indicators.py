"""Tests for the RQ2 and RQ3 indicators.

These are the numbers that would end up as claims in the manuscript, so the
edge cases that would quietly corrupt them are pinned here: ties in the citation
distribution, single-document years, and facets present on only one side of the
theory/experiment split.
"""

from __future__ import annotations

import pandas as pd

from nanocarbon_biblio.indicators import (
    annotate_records, combined_share_trend, dopant_lag_table, gap_matrix, spearman,
)
from nanocarbon_biblio.records import Record


def _rec(key: str, year: int, cited: int) -> Record:
    return Record(key=key, source="t", year=year, cited_by=cited)


def test_spearman_handles_the_standard_cases() -> None:
    assert spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == 1.0
    assert spearman([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == -1.0
    assert spearman([1, 2, 3, 4, 5], [7, 7, 7, 7, 7]) == 0.0  # undefined -> 0
    assert spearman([1, 2], [1, 2]) == 0.0                     # too few points


def test_citation_percentile_uses_mid_ranks_for_ties() -> None:
    """All-tied years must not put every document at the 100th percentile."""
    tied = [_rec(f"k{i}", 2020, 7) for i in range(12)]
    annotate_records(tied)
    assert all(r.labels["citation_percentile_year"] == 50.0 for r in tied)
    assert not any(r.labels["top_decile_year"] for r in tied)


def test_normalised_citations_rank_within_the_year() -> None:
    records = [_rec(f"m{i}", 2020, c) for i, c in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])]
    annotate_records(records)
    best, worst = records[-1], records[0]
    assert best.labels["cnorm_year"] > 1.0
    assert best.labels["top_decile_year"] is True
    assert worst.labels["top_decile_year"] is False
    assert worst.labels["citation_percentile_year"] < best.labels["citation_percentile_year"]


def test_thin_years_are_flagged_unreliable() -> None:
    """A mean over two documents is not a baseline; say so rather than hide it."""
    records = [_rec("a", 1993, 5), _rec("b", 1993, 9)]
    annotate_records(records, min_year_n=5)
    assert all(r.labels["citations_reliable"] is False for r in records)


def test_combined_share_trend_drops_thin_years() -> None:
    rows = []
    for year in (2000, 2001):
        rows.append({"year": year, "study_type": "combined"})  # 1 doc: dropped
    for year in (2010, 2011, 2012):
        rows += [{"year": year, "study_type": "combined"}] * 3
        rows += [{"year": year, "study_type": "experimental"}] * 3
    result = combined_share_trend(pd.DataFrame(rows), min_year_n=5)
    assert result["n_years"] == 3
    assert 2000 not in result["table"].index


def test_dopant_lag_sign_distinguishes_prediction_from_explanation() -> None:
    """Positive lag = theory first; negative = experiment first."""
    rows = (
        [{"year": 1995, "dopant": "boron", "study_type": "theoretical"}] * 3
        + [{"year": 2005, "dopant": "boron", "study_type": "experimental"}] * 3
        + [{"year": 2010, "dopant": "sulfur", "study_type": "experimental"}] * 3
        + [{"year": 2015, "dopant": "sulfur", "study_type": "theoretical"}] * 3
    )
    table = dopant_lag_table(pd.DataFrame(rows), k=3).set_index("dopant")
    assert table.loc["boron", "lag_first"] == 10       # theory predicted it
    assert table.loc["sulfur", "lag_first"] == -5      # experiment came first


def test_dopant_lag_is_none_when_one_side_is_missing() -> None:
    rows = [{"year": 2000, "dopant": "selenium", "study_type": "theoretical"}] * 4
    table = dopant_lag_table(pd.DataFrame(rows)).set_index("dopant")
    assert pd.isna(table.loc["selenium", "lag_first"])
    assert table.loc["selenium", "n_experiment"] == 0


def test_gap_matrix_identifies_predicted_but_unrealised_cells() -> None:
    rows = (
        [{"dopant": "silicon", "application": "battery",
          "study_type": "theoretical", "cnorm_year": 2.0}] * 3
        + [{"dopant": "nitrogen", "application": "orr_fuelcell",
            "study_type": "experimental", "cnorm_year": 1.0}] * 5
        + [{"dopant": "boron", "application": "sensor",
            "study_type": "combined", "cnorm_year": 1.0}] * 4
    )
    table = gap_matrix(pd.DataFrame(rows), min_theory=2)
    by_cell = table.set_index(["dopant", "application"])
    assert by_cell.loc[("silicon", "battery"), "status"] == "theory_only"
    assert by_cell.loc[("nitrogen", "orr_fuelcell"), "status"] == "experiment_only"
    assert by_cell.loc[("boron", "sensor"), "status"] == "covered"
    # Only the theory-only cell scores, and it sorts to the top.
    assert table.iloc[0]["status"] == "theory_only"
    assert (table[table.status != "theory_only"].gap_score == 0).all()


def test_gap_matrix_threshold_suppresses_one_off_predictions() -> None:
    """A single theoretical paper must not manufacture a research gap."""
    rows = [{"dopant": "selenium", "application": "battery",
             "study_type": "theoretical", "cnorm_year": 1.0}]
    table = gap_matrix(pd.DataFrame(rows), min_theory=2)
    assert (table.status == "theory_only").sum() == 0
