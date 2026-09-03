"""Deduplication of a merged Scopus + WoS corpus.

Two passes, in this order:

1. **DOI exact.** Cheap, near-perfect precision. Normalised via
   :func:`~nanocarbon_biblio.records.normalise_doi`.
2. **Fuzzy title within a year window.** Catches the 10–20 % of records that
   carry no DOI (older papers, some conference proceedings) and the ones where
   one database recorded a wrong DOI. Blocked by publication year ±
   ``year_window`` so the comparison stays tractable on corpora of tens of
   thousands of records.

The output keeps one *representative* per cluster and records which databases
contributed, which is what feeds the Scopus/WoS Venn diagram — a number worth
reporting in its own right (see ``docs/PROTOCOL.md`` §7).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process

from .records import Record

__all__ = ["DedupeResult", "deduplicate", "overlap_table", "identified_by_source"]

# Preference order when choosing the survivor of a duplicate cluster. Scopus
# first because its abstracts are more consistently complete; change freely,
# but change it *before* the final run and record the choice in the protocol.
_SOURCE_PRIORITY = {"scopus": 0, "wos": 1}

#: Query rows scored per rapidfuzz call. Bounds peak matrix memory to roughly
#: _CHUNK_ROWS x block size bytes (uint8), which stays small even for a year
#: holding tens of thousands of records.
_CHUNK_ROWS = 2048


@dataclass(slots=True)
class DedupeResult:
    """Outcome of a deduplication run.

    Attributes
    ----------
    unique:
        One representative :class:`~nanocarbon_biblio.records.Record` per work.
        Each carries ``labels["sources"]`` (sorted list of contributing
        databases) and ``labels["duplicate_keys"]``.
    clusters:
        ``{representative_key: [all record keys in the cluster]}``.
    n_input:
        Number of records fed in, for the PRISMA flow diagram.
    """

    unique: list[Record]
    clusters: dict[str, list[str]]
    n_input: int

    @property
    def n_removed(self) -> int:
        """Records dropped as duplicates."""
        return self.n_input - len(self.unique)


class _UnionFind:
    """Minimal union-find over record keys."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """Return the cluster root of ``item``, adding it if unseen."""
        parent = self._parent.setdefault(item, item)
        if parent != item:
            parent = self.find(parent)
            self._parent[item] = parent
        return parent

    def union(self, a: str, b: str) -> None:
        """Merge the clusters containing ``a`` and ``b``."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def groups(self) -> dict[str, list[str]]:
        """Return ``{root: members}`` for every known item."""
        out: dict[str, list[str]] = defaultdict(list)
        for item in self._parent:
            out[self.find(item)].append(item)
        return dict(out)


def _pick_representative(records: list[Record]) -> Record:
    """Choose the survivor of a duplicate cluster.

    Prefers, in order: a record with an abstract, the configured source
    priority, then the one with more cited references (a proxy for a complete
    ``CR`` block, which the citation analyses depend on).
    """
    return sorted(
        records,
        key=lambda r: (
            not r.has_abstract(),
            _SOURCE_PRIORITY.get(r.source, 9),
            -(r.n_references or 0),
        ),
    )[0]


def deduplicate(
    records: list[Record],
    *,
    title_threshold: float = 92.0,
    year_window: int = 1,
    min_title_len: int = 25,
) -> DedupeResult:
    """Deduplicate ``records`` by DOI then fuzzy title.

    Parameters
    ----------
    title_threshold:
        ``rapidfuzz.fuzz.token_set_ratio`` cutoff, 0–100. 92 is deliberately
        conservative: below ~88 this starts merging genuinely different papers
        from the same group ("N-doped CNTs for ORR" parts I and II). Tune it on
        your own corpus and report the value.
    year_window:
        Records are only compared when their years differ by at most this much.
        1 absorbs the online-first/issue-year mismatch between the two databases.
    min_title_len:
        Titles shorter than this (normalised) are matched by DOI only. Very
        short titles fuzzy-match each other far too easily.

    Returns
    -------
    DedupeResult
    """
    uf = _UnionFind()
    by_key: dict[str, Record] = {}
    for rec in records:
        by_key[rec.key] = rec
        uf.find(rec.key)

    # Pass 1 — DOI.
    by_doi: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        doi = rec.doi_key
        if doi:
            by_doi[doi].append(rec.key)
    for keys in by_doi.values():
        for other in keys[1:]:
            uf.union(keys[0], other)

    # Pass 2 — fuzzy title, blocked by year.
    by_year: dict[int | None, list[Record]] = defaultdict(list)
    for rec in records:
        if len(rec.title_key) >= min_title_len:
            by_year[rec.year].append(rec)

    undated = by_year.get(None, [])
    years = sorted(y for y in by_year if y is not None)
    # Each year's records are the queries; the candidates are that year, the
    # following `year_window` years, and every undated record. Comparing
    # queries against candidates rather than a merged block against itself
    # halves the work and stops each pair being scored twice.
    blocks: list[tuple[list[Record], list[Record]]] = []
    for year in years:
        queries = by_year[year]
        candidates = list(queries)
        for offset in range(1, year_window + 1):
            candidates.extend(by_year.get(year + offset, []))
        candidates.extend(undated)
        if len(candidates) > 1:
            blocks.append((queries, candidates))
    # Undated records are compared against each other in their own block; they
    # ride along in every dated block but are never queries there.
    if len(undated) > 1:
        blocks.append((undated, undated))

    for queries, candidates in blocks:
        candidate_titles = [r.title_sort_key for r in candidates]
        # Score in row chunks: a single year can hold thousands of records, and
        # a full queries x candidates matrix would be needlessly large.
        for start in range(0, len(queries), _CHUNK_ROWS):
            chunk = queries[start:start + _CHUNK_ROWS]
            scores = process.cdist(
                [r.title_sort_key for r in chunk], candidate_titles,
                scorer=fuzz.ratio, score_cutoff=title_threshold,
                workers=-1, dtype=np.uint8,
            )
            # Pull the surviving pairs out in C. Walking the whole matrix in
            # Python costs O(n^2) interpreter steps per block and dominated the
            # runtime; the matches themselves are a tiny fraction of the cells.
            for i, j in np.argwhere(scores >= title_threshold):
                left, right = chunk[i], candidates[j]
                if left.key == right.key:
                    continue
                # Two different DOIs is strong evidence of two different works;
                # trust the DOIs over the title similarity.
                if left.doi_key and right.doi_key and left.doi_key != right.doi_key:
                    continue
                # A title match alone is not enough when a DOI is missing on one
                # side: near-identical titles are common in this literature
                # (same group, same template, different dopant loading). The
                # first author's surname is stable across both databases, so
                # disagreement there vetoes the merge.
                if (
                    left.surname_key and right.surname_key
                    and left.surname_key != right.surname_key
                ):
                    continue
                uf.union(left.key, right.key)

    clusters = uf.groups()
    unique: list[Record] = []
    out_clusters: dict[str, list[str]] = {}
    for members in clusters.values():
        group = [by_key[k] for k in members]
        rep = _pick_representative(group)
        rep.labels["sources"] = sorted({r.source for r in group})
        rep.labels["n_duplicates"] = len(group)
        rep.labels["duplicate_keys"] = sorted(r.key for r in group if r.key != rep.key)
        unique.append(rep)
        out_clusters[rep.key] = sorted(members)

    unique.sort(key=lambda r: (r.year or 0, r.title_key))
    return DedupeResult(unique=unique, clusters=out_clusters, n_input=len(records))


def identified_by_source(result: DedupeResult) -> dict[str, int]:
    """Records retrieved per database **before** deduplication.

    PRISMA's identification row wants the raw retrieval count from each
    database, not the post-deduplication split — those differ by exactly the
    number of works both databases held. Derived from the cluster membership,
    since every record key is prefixed with its source.
    """
    counts: dict[str, int] = defaultdict(int)
    for members in result.clusters.values():
        for key in members:
            counts[key.split(":", 1)[0]] += 1
    return dict(counts)


def overlap_table(result: DedupeResult) -> dict[str, int]:
    """Counts for the Scopus/WoS Venn diagram.

    Returns a mapping with keys ``"scopus_only"``, ``"wos_only"``, ``"both"``,
    ``"total_unique"`` and ``"duplicates_removed"``. Report these verbatim in
    Methods: the exclusive fractions justify searching both databases.
    """
    counts: dict[str, int] = defaultdict(int)
    for rec in result.unique:
        sources = tuple(rec.labels.get("sources", [rec.source]))
        if sources == ("scopus",):
            counts["scopus_only"] += 1
        elif sources == ("wos",):
            counts["wos_only"] += 1
        elif set(sources) >= {"scopus", "wos"}:
            counts["both"] += 1
        else:
            counts["other_only"] += 1
    counts["total_unique"] = len(result.unique)
    counts["duplicates_removed"] = result.n_removed
    return dict(counts)
