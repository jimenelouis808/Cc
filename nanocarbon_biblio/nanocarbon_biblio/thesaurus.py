"""Keyword harmonisation for biblioshiny.

Terminology in this field drifts badly: ``nitrogen-doped carbon nanotube`` →
``N-doped CNT`` → ``N-CNT`` → ``NCNT``. Left alone, a co-word map splits one
concept across four nodes and every cluster is wrong. A thesaurus is not
optional here.

Output format is the one biblioshiny expects for its *synonyms* file: one group
per line, terms separated by ``;``, **the first term is the label that replaces
the rest**.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from .records import Record

__all__ = ["extract_keywords", "suggest_synonyms", "write_thesaurus", "SEED_GROUPS"]

_WS = re.compile(r"\s+")

#: Hand-curated starting point. These are the merges you would otherwise
#: discover the hard way, after the first co-word map comes out unreadable.
SEED_GROUPS: list[list[str]] = [
    ["nitrogen-doped carbon nanotube", "n-doped carbon nanotube", "n-doped cnt",
     "nitrogen doped carbon nanotubes", "ncnt", "n-cnt", "n-doped mwcnt",
     "nitrogen-doped cnts", "n-doped carbon nanotubes"],
    ["boron-doped carbon nanotube", "b-doped carbon nanotube", "b-doped cnt",
     "boron doped carbon nanotubes", "b-cnt"],
    ["carbon nanotube", "carbon nanotubes", "cnt", "cnts", "nanotube", "nanotubes"],
    ["multi-walled carbon nanotube", "multiwalled carbon nanotube", "mwcnt",
     "mwcnts", "mwnt", "multi walled carbon nanotubes", "multi-wall carbon nanotube"],
    ["single-walled carbon nanotube", "singlewalled carbon nanotube", "swcnt",
     "swcnts", "swnt", "single wall carbon nanotube"],
    ["carbon nanofiber", "carbon nanofibre", "carbon nanofibers", "carbon nanofibres",
     "carbon nano-fiber", "cnf", "cnfs"],
    ["density functional theory", "dft", "dft calculations", "first-principles",
     "first principles", "ab initio", "ab-initio calculations"],
    ["molecular dynamics", "md simulation", "molecular dynamics simulation",
     "molecular-dynamics", "md simulations"],
    ["stone-wales defect", "stone wales defect", "sw defect", "stone-wales defects",
     "5-7-7-5 defect", "pentagon-heptagon defect"],
    ["vacancy", "vacancies", "vacancy defect", "monovacancy", "single vacancy",
     "mono-vacancy", "atomic vacancy"],
    ["divacancy", "divacancies", "double vacancy", "di-vacancy"],
    ["oxygen reduction reaction", "orr", "oxygen reduction", "oxygen-reduction reaction"],
    ["hydrogen evolution reaction", "her", "hydrogen evolution"],
    ["oxygen evolution reaction", "oer", "oxygen evolution"],
    ["x-ray photoelectron spectroscopy", "xps", "x ray photoelectron spectroscopy",
     "photoelectron spectroscopy"],
    ["raman spectroscopy", "raman", "raman spectra", "raman scattering", "raman analysis"],
    ["chemical vapor deposition", "chemical vapour deposition", "cvd", "cvd growth",
     "catalytic chemical vapor deposition", "ccvd"],
    ["transmission electron microscopy", "tem", "hrtem",
     "high-resolution transmission electron microscopy"],
    ["supercapacitor", "supercapacitors", "electrochemical capacitor", "super capacitor"],
    ["lithium-ion battery", "lithium ion battery", "li-ion battery", "lithium-ion batteries"],
    ["graphene oxide", "go", "graphite oxide"],
    ["reduced graphene oxide", "rgo", "reduced graphite oxide"],
    ["vertically aligned carbon nanotube", "vacnt", "aligned carbon nanotube",
     "vertically-aligned carbon nanotubes", "carbon nanotube forest"],
    ["carbon nanotube sponge", "cnt sponge", "carbon nanotube aerogel", "cnt aerogel"],
    ["buckypaper", "bucky paper", "carbon nanotube paper", "cnt film"],
    ["co-doping", "codoping", "co-doped", "codoped", "dual doping", "dual-doped"],
    ["heteroatom doping", "heteroatom-doped", "heteroatom doped", "heteroatoms"],
    ["defect engineering", "defect-engineering", "defect engineered"],
    ["single-atom catalyst", "single atom catalyst", "sac", "single-atom catalysts"],
]


def _norm(term: str) -> str:
    """Lowercase, strip and collapse whitespace in a keyword."""
    return _WS.sub(" ", str(term).strip().lower())


def extract_keywords(records: list[Record]) -> Counter[str]:
    """Count normalised keywords across ``records``.

    Author keywords and Keywords Plus are pooled — biblioshiny lets you analyse
    them separately (``DE`` vs ``ID``), but for building a thesaurus you want
    every spelling variant that occurs anywhere.
    """
    counts: Counter[str] = Counter()
    for record in records:
        for raw in str(record.keywords or "").split(";"):
            term = _norm(raw)
            if len(term) > 2:
                counts[term] += 1
    return counts


def suggest_synonyms(
    records: list[Record],
    *,
    min_count: int = 5,
    threshold: float = 88.0,
    max_terms: int = 3000,
) -> list[list[str]]:
    """Propose synonym groups from the corpus's own keyword distribution.

    Greedy single-pass clustering: terms are visited most-frequent first, and
    any not-yet-assigned term scoring above ``threshold`` against the seed term
    joins its group. The most frequent term becomes the group label.

    **These are suggestions, not results.** Read every group before using it —
    ``n-doped`` and ``p-doped`` score high against each other and are opposites.
    Merge the reviewed output with :data:`SEED_GROUPS` and version the file.
    """
    counts = extract_keywords(records)
    terms = [term for term, count in counts.most_common(max_terms) if count >= min_count]
    assigned: set[str] = set()
    groups: list[list[str]] = []
    for term in terms:
        if term in assigned:
            continue
        group = [term]
        assigned.add(term)
        for other in terms:
            if other in assigned:
                continue
            if fuzz.token_set_ratio(term, other) >= threshold:
                group.append(other)
                assigned.add(other)
        if len(group) > 1:
            groups.append(group)
    return groups


def write_thesaurus(
    groups: list[list[str]],
    path: str | Path,
    *,
    include_seed: bool = True,
) -> Path:
    """Write a biblioshiny synonyms file.

    Each line is ``label;variant;variant;...``. Load it in biblioshiny under
    *Data → Filters* (or pass it as the ``synonyms`` argument of
    ``bibliometrix::termExtraction``). Duplicate variants across groups are
    dropped, keeping the first occurrence, because biblioshiny applies the
    substitutions in file order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_groups = ([[_norm(t) for t in g] for g in SEED_GROUPS] if include_seed else []) + \
                 [[_norm(t) for t in g] for g in groups]
    seen: set[str] = set()
    lines: list[str] = []
    for group in all_groups:
        cleaned = [t for t in dict.fromkeys(group) if t and t not in seen]
        if len(cleaned) < 2:
            continue
        seen.update(cleaned)
        lines.append(";".join(cleaned))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
