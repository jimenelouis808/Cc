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

from rapidfuzz import fuzz, process

from .records import Record

__all__ = ["DedupeResult", "deduplicate", "overlap_table"]

# Preference order when choosing the survivor of a duplicate cluster. Scopus
# first because its abstracts are more consistently complete; change freely,
# but change it *before* the final run and record the choice in the protocol.
_SOURCE_PRIORITY = {"scopus": 0, "wos": 1}


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

    years = sorted(y for y in by_year if y is not None)
    for year in years:
        block: list[Record] = []
        for offset in range(0, year_window + 1):
            block.extend(by_year.get(year + offset, []))
        # Records with no year are compared against every block: cheap insurance,
        # since undated records are rare in these exports.
        block.extend(by_year.get(None, []))
        if len(block) < 2:
            continue
        titles = [r.title_key for r in block]
        matches = process.cdist(
            titles, titles, scorer=fuzz.token_set_ratio,
            score_cutoff=title_threshold, workers=-1,
        )
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                if matches[i][j] < title_threshold:
                    continue
                left, right = block[i], block[j]
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
