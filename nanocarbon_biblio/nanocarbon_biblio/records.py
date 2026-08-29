"""Canonical record model shared by every loader.

The guiding principle of this package: **never rebuild the bibliometrix data
frame in Python**. Cited references (``CR``) are what make co-citation,
bibliographic coupling and RPYS possible, and reconstructing them from a
half-parsed export is a reliable way to corrupt them.

Instead, each loader keeps the *raw* record exactly as the database exported it
(``raw`` for tagged WoS blocks, the original CSV row for Scopus) alongside a
handful of normalised fields used only for deduplication and classification.
Exporters then write back a **filtered copy in the original format**, and
``bibliometrix::convert2df`` does the parsing it was designed to do.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Record",
    "normalise_doi",
    "normalise_title",
    "normalise_surname",
]

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
    "doi ",
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Return ``text`` with combining marks removed (NFKD fold)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise_doi(raw: str | None) -> str:
    """Normalise a DOI for exact matching.

    Lowercases, strips resolver prefixes and surrounding whitespace. Returns an
    empty string when no usable DOI is present, which callers must treat as
    "unknown" rather than as a matchable value.
    """
    if not raw:
        return ""
    doi = str(raw).strip().lower()
    for prefix in _DOI_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    doi = doi.strip().rstrip(".")
    # A DOI always starts with the "10." registrant prefix; anything else is noise
    # such as "[No DOI available]" that some exports write into the column.
    return doi if doi.startswith("10.") else ""


def normalise_title(raw: str | None) -> str:
    """Normalise a title into a comparison key.

    Accents folded, case dropped, every non-alphanumeric run collapsed to a
    single space. Deliberately aggressive: Scopus and WoS disagree on hyphens,
    subscripts, Greek letters and trailing periods for the very same article.
    """
    if not raw:
        return ""
    title = _strip_accents(str(raw)).lower()
    title = _NON_ALNUM.sub(" ", title)
    return _WS.sub(" ", title).strip()


def normalise_surname(raw: str | None) -> str:
    """Extract and normalise the first author's surname.

    Handles both export conventions: ``"Terrones, M."`` (surname first, Scopus
    and WoS ``AU``) and ``"M. Terrones"`` (given name first, some BibTeX).
    """
    if not raw:
        return ""
    first = str(raw).split(";")[0].strip()
    if not first:
        return ""
    surname = first.split(",")[0] if "," in first else first.split()[-1]
    surname = _strip_accents(surname).lower()
    return _NON_ALNUM.sub("", surname)


@dataclass(slots=True)
class Record:
    """One bibliographic record, database-agnostic.

    Attributes
    ----------
    key:
        Stable within-run identifier, ``"<source>:<index>"``. Used to join the
        Python-side labels back onto the bibliometrix data frame in R.
    source:
        ``"scopus"`` or ``"wos"`` (or another loader name).
    raw:
        The record exactly as exported. For WoS this is the tagged text block
        including its ``ER`` terminator; for Scopus it is the original CSV row
        as a dict. This is what gets written back out.
    """

    key: str
    source: str
    title: str = ""
    abstract: str = ""
    keywords: str = ""
    authors: str = ""
    year: int | None = None
    doi: str = ""
    source_title: str = ""
    doc_type: str = ""
    cited_by: int | None = None
    n_references: int | None = None
    uid: str = ""
    raw: Any = None
    labels: dict[str, Any] = field(default_factory=dict)

    @property
    def doi_key(self) -> str:
        """Normalised DOI, or ``""`` when absent."""
        return normalise_doi(self.doi)

    @property
    def title_key(self) -> str:
        """Normalised title used for fuzzy matching."""
        return normalise_title(self.title)

    @property
    def surname_key(self) -> str:
        """Normalised first-author surname."""
        return normalise_surname(self.authors)

    def text_blob(self) -> str:
        """Concatenated title + abstract + keywords, for lexicon classification.

        Case is preserved on purpose: several dopant patterns (``N-doped`` vs
        ``n-type``) are only separable with case information.
        """
        return " \n ".join(part for part in (self.title, self.abstract, self.keywords) if part)

    def has_abstract(self) -> bool:
        """True when an abstract usable for classification is present."""
        text = (self.abstract or "").strip()
        return len(text) >= 40 and text.lower() not in {"[no abstract available]", "no abstract available"}
