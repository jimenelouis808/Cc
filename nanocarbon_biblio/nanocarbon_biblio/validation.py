"""Methods-section evidence: query recall and classifier agreement.

Two things a bibliometric review is routinely asked for and rarely supplies:

1. **Did the search actually find the literature?** :func:`score_recall` runs a
   known-item test against a gold standard you assemble by hand, and reports
   relative recall per database. This is the table that turns "we searched
   Scopus and WoS" into a defensible claim (``docs/PROTOCOL.md`` §4).
2. **Are the classification rules any good?** :func:`score_agreement` reads back
   a coding sheet a human filled in and computes Cohen's kappa, accuracy and
   per-class precision/recall. A review that reports "rules validated on 100
   stratified records, kappa = 0.87" is in a different credibility class from
   one that does not.

Neither needs a network connection or an API key.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from .records import Record, normalise_doi, normalise_title

__all__ = [
    "GoldItem",
    "RecallReport",
    "load_gold_standard",
    "score_recall",
    "cohens_kappa",
    "score_agreement",
    "GOLD_TEMPLATE_COLUMNS",
]

#: Columns of the gold-standard file. ``doi`` is preferred; ``title`` + ``year``
#: is the fallback for older records that predate DOI assignment.
GOLD_TEMPLATE_COLUMNS = ("doi", "title", "year", "why", "verified")

_TITLE_MATCH_THRESHOLD = 88.0


@dataclass(slots=True)
class GoldItem:
    """One paper the search is required to retrieve.

    Attributes
    ----------
    why:
        Free text saying *why* this paper belongs in the gold standard —
        seminal, a specific dopant, a specific morphology, your own group's.
        Forces the set to be built deliberately rather than from memory.
    verified:
        Whether a human has checked the DOI and title against the actual record.
        An unverified gold standard measures your memory, not your query.
    """

    doi: str = ""
    title: str = ""
    year: int | None = None
    why: str = ""
    verified: bool = False

    @property
    def doi_key(self) -> str:
        """Normalised DOI, or ``""``."""
        return normalise_doi(self.doi)

    @property
    def title_key(self) -> str:
        """Normalised title."""
        return normalise_title(self.title)

    def label(self) -> str:
        """Short human-readable identifier for reports."""
        return self.title[:70] or self.doi or "(sin identificar)"


@dataclass(slots=True)
class RecallReport:
    """Outcome of a known-item test.

    Attributes
    ----------
    rows:
        One row per gold item: whether it was found and by which databases.
    unverified:
        Gold items not marked ``verified``. Reported loudly — they weaken the
        whole exercise.
    """

    rows: pd.DataFrame
    n_gold: int
    n_found: int
    unverified: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    @property
    def relative_recall(self) -> float:
        """Fraction of the gold standard the corpus contains, 0–1.

        ``docs/PROTOCOL.md`` §4 sets the target at 0.95. Below that, the query
        has a vocabulary hole: find it rather than adding the missing papers by
        hand, because whatever the hole is, it is also hiding papers you have
        never heard of.
        """
        return self.n_found / self.n_gold if self.n_gold else 0.0

    def missing(self) -> pd.DataFrame:
        """Gold items the corpus failed to retrieve — the diagnostic list."""
        return self.rows[~self.rows["found"]].reset_index(drop=True)


def load_gold_standard(path: str | Path) -> list[GoldItem]:
    """Read a gold-standard CSV.

    Expects the columns in :data:`GOLD_TEMPLATE_COLUMNS`; extras are ignored and
    missing ones default to empty. Rows with neither a DOI nor a title are
    skipped, since they cannot be matched against anything.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    items: list[GoldItem] = []
    skipped = 0
    for row in frame.to_dict(orient="records"):
        doi = str(row.get("doi", "")).strip()
        title = str(row.get("title", "")).strip()
        if not doi and not title:
            # A template prompt that has not been filled in yet. Counted and
            # reported rather than dropped in silence: a gold standard of two
            # entries measures nothing, and the user should see that.
            skipped += 1
            continue
        year_raw = str(row.get("year", "")).strip()
        try:
            year = int(float(year_raw)) if year_raw else None
        except ValueError:
            year = None
        items.append(GoldItem(
            doi=doi, title=title, year=year,
            why=str(row.get("why", "")).strip(),
            verified=str(row.get("verified", "")).strip().lower() in {"1", "true", "yes", "sí", "si", "x"},
        ))
    if skipped:
        print(
            f"  {Path(path).name}: {len(items)} entradas usables, {skipped} filas "
            "sin DOI ni título (plantilla sin rellenar)."
        )
    if items and not any(item.verified for item in items):
        print(
            "  AVISO: ninguna entrada está marcada como verificada. Un conjunto "
            "de oro sin verificar mide tu memoria, no tu consulta."
        )
    return items


def score_recall(records: list[Record], gold: list[GoldItem]) -> RecallReport:
    """Test whether ``records`` contains every paper in ``gold``.

    Matching is by normalised DOI first, then by fuzzy title above
    ``_TITLE_MATCH_THRESHOLD``. A gold item with no DOI is matched on title
    alone, which is why the gold standard should carry DOIs wherever they exist.

    Returns
    -------
    RecallReport
        Put ``rows`` in the supplementary material and
        ``relative_recall`` in Methods. Run it once per query variant: the
        difference between the precision arm and the sensitive arm is exactly
        what justifies choosing one.
    """
    by_doi: dict[str, list[Record]] = {}
    for record in records:
        key = record.doi_key
        if key:
            by_doi.setdefault(key, []).append(record)
    titles = [(r, r.title_key) for r in records if r.title_key]

    rows: list[dict[str, object]] = []
    source_counts: Counter[str] = Counter()
    found = 0
    for item in gold:
        hits: list[Record] = []
        how = ""
        if item.doi_key and item.doi_key in by_doi:
            hits = by_doi[item.doi_key]
            how = "doi"
        elif item.title_key:
            best_score, best = 0.0, None
            for record, title in titles:
                score = fuzz.ratio(item.title_key, title)
                if score > best_score:
                    best_score, best = score, record
            if best is not None and best_score >= _TITLE_MATCH_THRESHOLD:
                hits, how = [best], f"title ({best_score:.0f})"
        sources = sorted({r.source for r in hits})
        if hits:
            found += 1
            source_counts["+".join(sources)] += 1
        rows.append({
            "title": item.label(),
            "doi": item.doi_key,
            "year": item.year,
            "why": item.why,
            "verified": item.verified,
            "found": bool(hits),
            "matched_by": how,
            "sources": "+".join(sources),
        })
    return RecallReport(
        rows=pd.DataFrame(rows),
        n_gold=len(gold),
        n_found=found,
        unverified=sum(1 for item in gold if not item.verified),
        by_source=dict(source_counts),
    )


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two label sequences.

    Kappa corrects observed agreement for the agreement expected by chance,
    which raw accuracy does not: two coders who both label everything
    "experimental" in a corpus that is 80 % experimental agree 80 % of the time
    and have learned nothing.

    Returns 0.0 when the sequences are empty or mismatched in length, and 1.0
    when both are constant and identical (perfect agreement, chance undefined).
    """
    if len(a) != len(b) or not a:
        return 0.0
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(
        (count_a[label] / n) * (count_b[label] / n)
        for label in set(count_a) | set(count_b)
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def _per_class(truth: list[str], predicted: list[str]) -> pd.DataFrame:
    """Precision, recall and F1 per class, plus support."""
    labels = sorted(set(truth) | set(predicted))
    rows = []
    for label in labels:
        tp = sum(1 for t, p in zip(truth, predicted) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, predicted) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, predicted) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "label": label, "support": tp + fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3),
        })
    return pd.DataFrame(rows)


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity of two ``|``-joined multi-label strings."""
    set_a = {t for t in str(a or "").split("|") if t}
    set_b = {t for t in str(b or "").split("|") if t}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def score_agreement(
    sheet: pd.DataFrame | str | Path,
    *,
    categorical: tuple[str, ...] = ("study_type",),
    multilabel: tuple[str, ...] = ("dopant", "defect", "application"),
) -> dict[str, object]:
    """Compare a human-coded sheet against the rule output.

    ``sheet`` is the CSV downloaded from the GUI's validation tab, with the
    ``manual_*`` columns filled in. Only rows where the manual column is
    non-empty are scored, so a partially coded sheet still works.

    For categorical facets, reports Cohen's kappa, accuracy and per-class
    precision/recall/F1. For multi-label facets — where a record can carry two
    dopants — reports mean Jaccard similarity and exact-set-match rate instead,
    because kappa is not defined on sets.

    Returns
    -------
    dict
        ``{"study_type": {...}, "dopant": {...}, "n_coded": int, ...}``. The
        kappa figures go in Methods; the per-class table goes in the
        supplementary material and tells you *which* rule to fix.
    """
    frame = sheet if isinstance(sheet, pd.DataFrame) else pd.read_csv(sheet, dtype=str, keep_default_na=False)
    results: dict[str, object] = {"n_rows": int(len(frame))}

    coded_any = 0
    for column in categorical:
        manual = f"manual_{column}"
        if column not in frame.columns or manual not in frame.columns:
            continue
        subset = frame[frame[manual].astype(str).str.strip() != ""]
        if subset.empty:
            results[column] = {"n_coded": 0, "note": f"Sin filas codificadas en {manual}."}
            continue
        predicted = subset[column].fillna("").astype(str).str.strip().tolist()
        truth = subset[manual].astype(str).str.strip().tolist()
        coded_any = max(coded_any, len(subset))
        accuracy = sum(1 for t, p in zip(truth, predicted) if t == p) / len(truth)
        results[column] = {
            "n_coded": len(subset),
            "kappa": cohens_kappa(truth, predicted),
            "accuracy": round(accuracy, 4),
            "per_class": _per_class(truth, predicted),
            "confusion": pd.crosstab(
                pd.Series(truth, name="manual"), pd.Series(predicted, name="reglas")
            ),
        }

    for column in multilabel:
        manual = f"manual_{column}"
        if column not in frame.columns or manual not in frame.columns:
            continue
        subset = frame[frame[manual].astype(str).str.strip() != ""]
        if subset.empty:
            results[column] = {"n_coded": 0, "note": f"Sin filas codificadas en {manual}."}
            continue
        pairs = list(zip(
            subset[manual].astype(str).tolist(),
            subset[column].fillna("").astype(str).tolist(),
        ))
        coded_any = max(coded_any, len(subset))
        results[column] = {
            "n_coded": len(subset),
            "mean_jaccard": round(sum(_jaccard(t, p) for t, p in pairs) / len(pairs), 4),
            "exact_set_match": round(
                sum(1 for t, p in pairs if _jaccard(t, p) == 1.0) / len(pairs), 4
            ),
            "note": "Multietiqueta: kappa no está definida sobre conjuntos.",
        }

    results["n_coded"] = coded_any
    results["interpretation"] = (
        "Kappa: <0.40 pobre · 0.40-0.60 moderada · 0.60-0.80 sustancial · "
        ">0.80 casi perfecta (Landis y Koch). Reporta el valor, el n y quién codificó."
    )
    return results
