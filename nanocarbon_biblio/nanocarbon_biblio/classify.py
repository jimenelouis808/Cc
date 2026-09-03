"""Lexicon-based classification of records into review facets.

Produces the label columns that the R side joins onto the bibliometrix data
frame, which is what lets you slice every biblioshiny analysis by dopant,
defect type, morphology, application, and — most importantly — by
theoretical vs experimental study type (RQ2 in ``docs/PROTOCOL.md``).

These are **rules, not a classifier**. They are transparent, auditable and
reproducible, which is exactly what a systematic review needs, but their error
rate is not zero. Validate them: :func:`sample_for_validation` draws a
stratified sample for manual coding, and the agreement you measure on it is a
number that belongs in your Methods section.
"""

from __future__ import annotations

import random
import re
from typing import Any

import pandas as pd

from .lexicons import HOST_MATERIALS, SUPPORT_VERBS, compile_facets
from .records import Record

__all__ = [
    "classify_record",
    "classify_all",
    "to_dataframe",
    "crosstab",
    "sample_for_validation",
    "codoping_dopants",
]

_RULES = compile_facets()
_HOST_RE = re.compile(HOST_MATERIALS, re.IGNORECASE)
_SUPPORT_RE = re.compile(SUPPORT_VERBS, re.IGNORECASE)
_DOPED_CARBON_RE = re.compile(
    r"\bdoped carbon\b|\bdoped (?:carbon )?nano(?:tube|fib)"
    r"|\b(?:nitrogen|boron|phosphor\w+|sul[fp]h?ur|N|B|P|S)[\s\-]doped "
    r"(?:multi|single|double)?[\s\-]?wall\w*[\s\-]?(?:carbon )?nanotube",
    re.IGNORECASE,
)

#: Facets whose value is a set of labels rather than a single label.
_MULTI_FACETS = (
    "dopant", "defect", "doping_mode", "method_theory",
    "method_experiment", "morphology", "application", "host",
)


# --------------------------------------------------------------------------
# Co-doping element lists.
#
# "Nitrogen and sulfur co-doped carbon nanofibers" names two dopants without
# ever writing "nitrogen-doped" or "sulfur-doped", so the per-element rules in
# lexicons.py miss both. Matching the *list* is the fix, and it must be tight:
# a loose proximity window would read "oxygen reduction over nitrogen co-doped
# carbon" as oxygen doping. So the gap between elements is allowed to contain
# nothing but other element names and separators.
# --------------------------------------------------------------------------

_ELEMENT_TOKENS: dict[str, tuple[str, ...]] = {
    "nitrogen": (r"(?i:nitrogen)", r"N"),
    "boron": (r"(?i:boron)", r"B"),
    "phosphorus": (r"(?i:phosphor(?:us|ous))", r"P"),
    "sulfur": (r"(?i:sul[fp]h?ur)", r"S"),
    "fluorine": (r"(?i:fluorine)", r"F"),
    "silicon": (r"(?i:silicon)", r"Si"),
    "oxygen": (r"(?i:oxygen)", r"O"),
    "selenium": (r"(?i:selenium)", r"Se"),
    "halogen_other": (r"(?i:chlorine|bromine|iodine)", r"Cl", r"Br", r"I"),
    "transition_metal": (r"(?i:iron|cobalt|nickel|manganese)", r"Fe", r"Co", r"Ni", r"Mn"),
}

_TOKEN_TO_LABEL: dict[str, str] = {
    token: label for label, tokens in _ELEMENT_TOKENS.items() for token in tokens
}
_ANY_TOKEN = "|".join(_TOKEN_TO_LABEL)
_SEP = r"(?:\s*(?:,|/|&|\+|-|\band\b)\s*)"

_CODOPE_LIST_RE = re.compile(
    rf"\b((?:{_ANY_TOKEN})(?:{_SEP}(?:{_ANY_TOKEN})){{1,3}})[\s\-]*co[\s\-]?dop"
)
_DOPED_WITH_RE = re.compile(
    rf"(?:co[\s\-]?)?doped\s+with\s+((?:{_ANY_TOKEN})(?:{_SEP}(?:{_ANY_TOKEN})){{0,3}})"
)
_SINGLE_TOKEN_RE = re.compile(rf"(?:{_ANY_TOKEN})")


def _elements_in(fragment: str) -> set[str]:
    """Map every element token inside a matched list fragment to its label."""
    labels: set[str] = set()
    for match in _SINGLE_TOKEN_RE.finditer(fragment):
        text = match.group(0)
        for token, label in _TOKEN_TO_LABEL.items():
            if re.fullmatch(token, text):
                labels.add(label)
                break
    return labels


def codoping_dopants(text: str) -> set[str]:
    """Return dopant labels named by a co-doping element list.

    Catches ``"nitrogen and sulfur co-doped"``, ``"N,S-codoped"`` and
    ``"doped with boron and nitrogen"`` — phrasings where no single-element
    rule fires but two dopants are unambiguously named.
    """
    found: set[str] = set()
    for pattern in (_CODOPE_LIST_RE, _DOPED_WITH_RE):
        for match in pattern.finditer(text):
            found |= _elements_in(match.group(1))
    return found


#: Rules regrouped by label, so a label that has already fired can skip its
#: remaining alternatives. "nitrogen" alone carries eight patterns, and once one
#: matches the other seven cannot change the outcome — evaluating them anyway
#: was most of the classification cost on a large corpus.
_RULES_BY_LABEL: dict[str, dict[str, list]] = {
    facet: {
        label: [rule for rule in rules if rule.label == label]
        for label in dict.fromkeys(rule.label for rule in rules)
    }
    for facet, rules in _RULES.items()
}


def _match_facet(facet: str, text: str) -> list[str]:
    """Return the sorted distinct labels of ``facet`` that fire on ``text``.

    Short-circuits per label: the first matching pattern settles that label.
    """
    hits: list[str] = []
    for label, rules in _RULES_BY_LABEL[facet].items():
        if any(rule.matches(text) for rule in rules):
            hits.append(label)
    return sorted(hits)


def _study_type(theory: list[str], experiment: list[str]) -> str:
    """Collapse the two method facets into one categorical study type.

    Returns one of ``"theoretical"``, ``"experimental"``, ``"combined"`` or
    ``"unclear"``. ``"combined"`` studies are the ones RQ2 cares about most:
    their share over time measures how coupled the two halves of the field are.
    """
    if theory and experiment:
        return "combined"
    if theory:
        return "theoretical"
    if experiment:
        return "experimental"
    return "unclear"


def classify_record(record: Record) -> dict[str, Any]:
    """Classify one record and return its label dictionary.

    The returned keys become columns in :func:`to_dataframe`. Multi-valued
    facets appear twice: as a ``|``-joined string (``dopant``) for reading and
    joining in R, and as a count (``n_dopant``) for quick filtering.

    Notes
    -----
    ``dopant_host_ambiguous`` flags the review's worst false positive: a paper
    where a *non-carbon* phase is the doped material and the nanocarbon is only
    a support (e.g. "N-doped TiO2 supported on MWCNTs"). It fires when a host
    material and a support verb are both present and no explicit "doped
    carbon nanotube"-style phrase appears. Records so flagged **must be screened
    by hand** — the heuristic is deliberately noisy in the safe direction.
    """
    text = record.text_blob()
    labels: dict[str, Any] = {}
    for facet in _MULTI_FACETS:
        hits = set(_match_facet(facet, text))
        if facet == "dopant":
            hits |= codoping_dopants(text)
        ordered = sorted(hits)
        labels[facet] = "|".join(ordered)
        labels[f"n_{facet}"] = len(ordered)

    theory = _match_facet("method_theory", text)
    experiment = _match_facet("method_experiment", text)
    labels["study_type"] = _study_type(theory, experiment)

    hosts = labels["host"].split("|") if labels["host"] else []
    labels["is_hybrid"] = "carbon_1d" in hosts and "graphene_2d" in hosts
    labels["mentions_graphene"] = "graphene_2d" in hosts
    labels["is_3d_assembly"] = bool(
        {"sponge_aerogel_foam", "forest_vacnt", "fiber_yarn", "film_buckypaper", "network_junction"}
        & set(labels["morphology"].split("|") if labels["morphology"] else [])
    )
    labels["has_dopant"] = labels["n_dopant"] > 0
    labels["has_defect"] = labels["n_defect"] > 0

    labels["dopant_host_ambiguous"] = bool(
        _HOST_RE.search(text) and _SUPPORT_RE.search(text) and not _DOPED_CARBON_RE.search(text)
    )
    labels["no_abstract"] = not record.has_abstract()
    return labels


def classify_all(records: list[Record]) -> list[Record]:
    """Classify every record in place and return the same list.

    Existing keys in ``record.labels`` (e.g. the ``sources`` written by
    :mod:`nanocarbon_biblio.dedupe`) are preserved.
    """
    for record in records:
        record.labels.update(classify_record(record))
    return records


def to_dataframe(records: list[Record]) -> pd.DataFrame:
    """Build the label table joined onto the bibliometrix data frame in R.

    The ``key`` column is the join key; ``doi`` and ``uid`` are carried along so
    the join can fall back to a DOI match if keys ever drift.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "key": record.key,
            "source": record.source,
            "sources": "|".join(record.labels.get("sources", [record.source])),
            "doi": record.doi_key,
            "uid": record.uid,
            "year": record.year,
            "doc_type": record.doc_type,
            "cited_by": record.cited_by,
            "title": record.title,
        }
        row.update({k: v for k, v in record.labels.items() if k not in {"sources", "duplicate_keys"}})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "key" in frame.columns:
        frame = frame.sort_values(["year", "key"], na_position="last").reset_index(drop=True)
    return frame


def crosstab(
    frame: pd.DataFrame,
    rows: str = "dopant",
    cols: str = "application",
    *,
    value: str | None = None,
) -> pd.DataFrame:
    """Cross-tabulate two multi-valued facets, exploding ``|``-joined labels.

    This builds the review's headline figure: the dopant × application matrix
    (see ``docs/PROTOCOL.md`` §2.3). A record doped with both N and S and aimed
    at both ORR and supercapacitors contributes to all four cells — which is the
    correct behaviour for a coverage map, and must be stated in the caption
    since row and column sums exceed the corpus size.

    Parameters
    ----------
    value:
        Column to average within each cell instead of counting, e.g.
        ``"cited_by"`` for a citation-weighted map. Prefer a field-normalised
        citation column over raw counts.
    """
    for column in (rows, cols):
        if column not in frame.columns:
            raise KeyError(f"column {column!r} not in frame; have {list(frame.columns)[:12]}")
    work = frame[[rows, cols] + ([value] if value else [])].copy()
    work = work[(work[rows].fillna("") != "") & (work[cols].fillna("") != "")]
    work[rows] = work[rows].str.split("|")
    work = work.explode(rows)
    work[cols] = work[cols].str.split("|")
    work = work.explode(cols)
    if value:
        table = work.pivot_table(index=rows, columns=cols, values=value, aggfunc="mean")
    else:
        table = work.pivot_table(index=rows, columns=cols, aggfunc="size", fill_value=0)
    return table


def sample_for_validation(
    frame: pd.DataFrame,
    n: int = 100,
    *,
    stratify: str = "study_type",
    seed: int = 20260829,
) -> pd.DataFrame:
    """Draw a stratified random sample for manual validation of the rules.

    Code the sample by hand, compute agreement (Cohen's kappa against the rule
    output), and report it. A review that states "rule-based classification
    validated on 100 stratified records, kappa = 0.87" is in a different
    credibility class from one that does not.

    The ``seed`` defaults to a fixed value so the sample is reproducible; change
    it only if you need a genuinely independent second sample.
    """
    if stratify not in frame.columns:
        return frame.sample(n=min(n, len(frame)), random_state=seed)
    rng = random.Random(seed)
    groups = list(frame.groupby(stratify, dropna=False))
    per_group = max(1, n // max(1, len(groups)))
    parts = [
        group.sample(n=min(per_group, len(group)), random_state=rng.randrange(1 << 30))
        for _, group in groups
    ]
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return sample.head(n).reset_index(drop=True)
