"""Indicators for the review's two original research questions.

Everything here is computed offline from the corpus itself. Nothing calls an
external API, so it runs the same way in a year's time as it does today.

**RQ2 — theory↔experiment coupling.** Does theory predict, or document? The
share of *combined* studies over time measures how coupled the two halves of the
field are, and :func:`dopant_lag_table` measures, per dopant, how many years
passed between the first computational study and the first experimental
realisation.

**RQ3 — coverage gaps.** :func:`gap_matrix` scores every (dopant × application)
cell, singling out the ones with theoretical work and no experimental follow-up.
Those cells are the evidence-backed version of a "future perspectives" section.

Citation normalisation is deliberately modest about what it claims — read
:func:`annotate_records` before reporting any of it.
"""

from __future__ import annotations

import bisect
from collections import defaultdict

import pandas as pd

from .records import Record

__all__ = [
    "annotate_records",
    "combined_share_trend",
    "dopant_lag_table",
    "gap_matrix",
    "spearman",
]

#: Study types that count as containing a computational component.
_THEORY_TYPES = frozenset({"theoretical", "combined"})
#: Study types that count as containing an experimental component.
_EXPERIMENT_TYPES = frozenset({"experimental", "combined"})


def annotate_records(records: list[Record], *, min_year_n: int = 5) -> list[Record]:
    """Add citation indicators to each record's labels, in place.

    Adds four keys:

    ``cnorm_year``
        Citations divided by the mean citations of corpus documents published
        the same year. 1.0 means "average for its year *within this corpus*".
    ``citation_percentile_year``
        Percentile rank of the citation count within its publication year, 0–100.
    ``top_decile_year``
        True when the record is in the top 10 % of its year.
    ``citations_reliable``
        False when fewer than ``min_year_n`` documents share the year, which
        makes both the mean and the percentile unstable.

    .. warning::
       This is **corpus-normalised, not field-normalised.** A true field
       indicator (MNCS, CNCI, category percentiles) compares a paper against
       every paper in its subject category worldwide, which needs a baseline
       this package does not have offline. What is computed here is valid for
       ranking documents *inside* this review's corpus — which is what the
       review actually needs — but must never be reported as CNCI or MNCS. If a
       reviewer asks for field-normalised figures, take them from SciVal or
       InCites and say where they came from.

    Raw citation counts are not an alternative: they measure age. A 2005 paper
    has had twenty years to accumulate citations and a 2024 paper has not, so
    any unnormalised ranking is a ranking by publication date.
    """
    by_year: dict[int | None, list[Record]] = defaultdict(list)
    for record in records:
        by_year[record.year].append(record)

    for year, group in by_year.items():
        counts = [r.cited_by if r.cited_by is not None else 0 for r in group]
        n = len(group)
        mean = sum(counts) / n if n else 0.0
        ordered = sorted(counts)
        reliable = year is not None and n >= min_year_n
        for record, count in zip(group, counts):
            # Mid-rank percentile: strictly-below plus half the ties. Counting
            # ties with <= would put every document at the 100th percentile in a
            # year where they all have the same citation count.
            n_below = bisect.bisect_left(ordered, count)
            n_equal = bisect.bisect_right(ordered, count) - n_below
            rank = n_below + 0.5 * n_equal
            record.labels["cnorm_year"] = round(count / mean, 3) if mean > 0 else 0.0
            record.labels["citation_percentile_year"] = round(100.0 * rank / n, 1) if n else 0.0
            record.labels["top_decile_year"] = bool(n >= 10 and record.labels["citation_percentile_year"] >= 90.0)
            record.labels["citations_reliable"] = reliable
    return records


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation of two equal-length sequences.

    Implemented here rather than pulled from SciPy: one correlation does not
    justify the dependency. Ties get average ranks. Returns 0.0 when either
    sequence is constant, where the coefficient is undefined.
    """
    if len(x) != len(y) or len(x) < 3:
        return 0.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(list(x)), ranks(list(y))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(num / (dx * dy), 4) if dx > 0 and dy > 0 else 0.0


def combined_share_trend(frame: pd.DataFrame, *, min_year_n: int = 5) -> dict[str, object]:
    """Measure whether the share of combined theory+experiment studies is rising.

    Years with fewer than ``min_year_n`` documents are dropped: a share computed
    over one or two documents is 0 % or 100 % by chance, and those years
    otherwise dominate the correlation.

    Returns the per-year table plus a Spearman correlation of the combined share
    against the year. A positive coefficient is the quantitative form of "theory
    and experiment have grown more coupled" — the claim RQ2 is asking about.
    """
    work = frame.dropna(subset=["year"]).copy()
    work["year"] = work["year"].astype(int)
    counts = work.groupby(["year", "study_type"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    dense = counts[totals >= min_year_n]
    if dense.empty:
        return {"table": dense, "spearman_rho": 0.0, "n_years": 0,
                "note": "No year reaches the minimum document count."}
    share = dense.div(dense.sum(axis=1), axis=0)
    combined = share["combined"] if "combined" in share.columns else pd.Series(0.0, index=share.index)
    rho = spearman(list(share.index.astype(float)), list(combined.astype(float)))
    return {
        "table": share.round(4),
        "spearman_rho": rho,
        "n_years": int(len(share)),
        "note": (
            "Spearman de la cuota de estudios 'combined' contra el año. "
            "Positivo = teoría y experimento más acoplados con el tiempo. "
            "No es una prueba causal; descríbelo como asociación."
        ),
    }


def _first_year_at_k(years: list[int], k: int) -> int | None:
    """Year in which the ``k``-th document of a group appeared."""
    ordered = sorted(years)
    return ordered[k - 1] if len(ordered) >= k else None


def dopant_lag_table(
    frame: pd.DataFrame,
    *,
    facet: str = "dopant",
    k: int = 3,
) -> pd.DataFrame:
    """Years between the first computational and the first experimental study.

    For every label of ``facet`` (dopant by default), reports the first year and
    the year the ``k``-th document appeared, separately for the theoretical and
    the experimental sides, plus the lag between them.

    Parameters
    ----------
    k:
        Robust anchor. A single early record can be a misclassification or a
        stray mention, and anchoring a decade-long lag claim on one document is
        indefensible. ``lag_at_k`` is the number to report; ``lag_first`` is
        there to show how much the choice of anchor matters.

    Returns
    -------
    pandas.DataFrame
        One row per label. A positive lag means theory came first — the
        prediction-then-realisation pattern. A negative lag means experiment came
        first and theory followed to explain it. Both are real and the mix
        across dopants is the interesting result.

    Notes
    -----
    ``n_theory`` and ``n_experiment`` are there to be read alongside the lag: a
    lag computed from three theoretical documents is a curiosity, not a finding.
    Rows are sorted by total documents so the well-supported ones come first.
    """
    if facet not in frame.columns:
        raise KeyError(f"column {facet!r} not in frame")
    work = frame.dropna(subset=["year"]).copy()
    work["year"] = work["year"].astype(int)
    work = work[work[facet].fillna("") != ""]
    work[facet] = work[facet].str.split("|")
    work = work.explode(facet)

    rows: list[dict[str, object]] = []
    for label, group in work.groupby(facet):
        theory = group[group.study_type.isin(_THEORY_TYPES)].year.tolist()
        experiment = group[group.study_type.isin(_EXPERIMENT_TYPES)].year.tolist()
        first_t = min(theory) if theory else None
        first_e = min(experiment) if experiment else None
        kth_t = _first_year_at_k(theory, k)
        kth_e = _first_year_at_k(experiment, k)
        rows.append({
            facet: label,
            "n_docs": len(group),
            "n_theory": len(theory),
            "n_experiment": len(experiment),
            "first_theory": first_t,
            "first_experiment": first_e,
            "lag_first": (first_e - first_t) if (first_t and first_e) else None,
            f"year_at_{k}_theory": kth_t,
            f"year_at_{k}_experiment": kth_e,
            "lag_at_k": (kth_e - kth_t) if (kth_t and kth_e) else None,
        })
    table = pd.DataFrame(rows)
    return table.sort_values("n_docs", ascending=False).reset_index(drop=True)


def gap_matrix(
    frame: pd.DataFrame,
    *,
    rows: str = "dopant",
    cols: str = "application",
    min_theory: int = 2,
) -> pd.DataFrame:
    """Score every (dopant × application) cell for coverage and for gaps.

    One row per cell, with the document count, the theoretical and experimental
    counts, the mean corpus-normalised citations, and a status:

    ``theory_only``
        At least ``min_theory`` computational studies and **no** experimental
        one. These are the cells worth writing a perspectives section about:
        somebody predicted it and nobody has made it yet.
    ``experiment_only``
        Experimental work with no computational study. The mirror-image gap —
        a mechanism nobody has modelled.
    ``covered``
        Both sides present.

    ``gap_score`` ranks the ``theory_only`` cells by ``n_theory × mean cnorm``,
    so a cell predicted by well-cited theory outranks one predicted once in
    passing. Every other status scores 0.

    Notes
    -----
    A document with two dopants and two applications contributes to four cells,
    so the counts sum to more than the corpus size. Say so in the caption.

    An empty cell is **not** by itself a research gap: it may be physically
    uninteresting, or simply outside the vocabulary the rules recognise. Read
    the top-ranked cells before claiming any of them, and check that the
    absence is not an artefact of the lexicon.
    """
    for column in (rows, cols):
        if column not in frame.columns:
            raise KeyError(f"column {column!r} not in frame")
    work = frame.copy()
    if "cnorm_year" not in work.columns:
        work["cnorm_year"] = 0.0
    work = work[(work[rows].fillna("") != "") & (work[cols].fillna("") != "")]
    work[rows] = work[rows].str.split("|")
    work = work.explode(rows)
    work[cols] = work[cols].str.split("|")
    work = work.explode(cols)

    records: list[dict[str, object]] = []
    for (row_label, col_label), group in work.groupby([rows, cols]):
        n_theory = int(group.study_type.isin(_THEORY_TYPES).sum())
        n_experiment = int(group.study_type.isin(_EXPERIMENT_TYPES).sum())
        mean_cnorm = float(group.cnorm_year.fillna(0).mean())
        if n_theory >= min_theory and n_experiment == 0:
            status, score = "theory_only", round(n_theory * max(mean_cnorm, 0.1), 3)
        elif n_experiment > 0 and n_theory == 0:
            status, score = "experiment_only", 0.0
        else:
            status, score = "covered", 0.0
        records.append({
            rows: row_label, cols: col_label, "n_docs": len(group),
            "n_theory": n_theory, "n_experiment": n_experiment,
            "mean_cnorm": round(mean_cnorm, 3), "status": status,
            "gap_score": score,
        })
    table = pd.DataFrame(records)
    if table.empty:
        return table
    return table.sort_values(["gap_score", "n_docs"], ascending=False).reset_index(drop=True)
