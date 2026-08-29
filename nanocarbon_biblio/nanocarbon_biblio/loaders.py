"""Loaders for raw Scopus and Web of Science exports.

Every loader returns ``list[Record]`` and preserves the original record in
``Record.raw`` so that :mod:`nanocarbon_biblio.exporters` can write a filtered
copy in the *native* format for ``bibliometrix::convert2df``.

Supported inputs
----------------
============================  =========================================
Format                        Function
============================  =========================================
Scopus CSV export             :func:`load_scopus_csv`
WoS tagged plain text (ISI)   :func:`load_wos_plaintext`
WoS tab-delimited export      :func:`load_wos_tabbed`
any of the above, autodetect  :func:`load_any`
============================  =========================================
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from .records import Record

__all__ = [
    "load_scopus_csv",
    "load_wos_plaintext",
    "load_wos_tabbed",
    "load_any",
    "load_directory",
    "WOS_HEADER",
    "WOS_FOOTER",
]

WOS_HEADER = "FN Clarivate Analytics Web of Science\nVR 1.0"
WOS_FOOTER = "EF"

# Scopus column names have drifted over the years; accept every spelling seen.
_SCOPUS_COLUMNS: dict[str, tuple[str, ...]] = {
    "title": ("Title", "Document Title"),
    "abstract": ("Abstract",),
    "authors": ("Authors", "Author full names", "Author Names"),
    "year": ("Year", "Publication Year"),
    "doi": ("DOI",),
    "source_title": ("Source title", "Source Title"),
    "doc_type": ("Document Type", "Document type"),
    "cited_by": ("Cited by", "Citations"),
    "uid": ("EID", "Scopus EID"),
    "author_keywords": ("Author Keywords", "Author keywords"),
    "index_keywords": ("Index Keywords", "Index keywords"),
    "references": ("References",),
}

_WOS_TAG = re.compile(r"^([A-Z][A-Z0-9]) (.*)$")


def _pick(row: dict, names: tuple[str, ...]) -> str:
    """Return the first non-empty value among ``names`` in ``row``."""
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() and str(value) != "nan":
            return str(value).strip()
    return ""


def _to_int(value: str) -> int | None:
    """Best-effort integer parse; ``None`` when the field is absent or dirty."""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def load_scopus_csv(path: str | Path, source: str = "scopus") -> list[Record]:
    """Load a Scopus CSV export.

    The CSV is read with ``dtype=str`` so that identifiers, years and page
    ranges survive untouched — pandas would otherwise turn ``"2020"`` into a
    float and DOIs with leading zeros into something unusable on re-export.

    Raises
    ------
    ValueError
        If the file has no recognisable Scopus title column, which almost always
        means the wrong export format (RIS or BibTeX) was saved with a .csv name.
    """
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="replace")
    if not any(col in frame.columns for col in _SCOPUS_COLUMNS["title"]):
        raise ValueError(
            f"{path.name}: no Scopus 'Title' column found. Columns seen: "
            f"{list(frame.columns)[:8]}. Re-export from Scopus as CSV."
        )
    if not any(col in frame.columns for col in _SCOPUS_COLUMNS["references"]):
        # Not fatal, but it silently destroys every citation-based analysis.
        print(
            f"WARNING {path.name}: no 'References' column. Co-citation, "
            "bibliographic coupling and RPYS will be impossible. Re-export "
            "from Scopus with the 'References' field ticked."
        )

    records: list[Record] = []
    for idx, row in enumerate(frame.to_dict(orient="records")):
        keywords = "; ".join(
            part for part in (
                _pick(row, _SCOPUS_COLUMNS["author_keywords"]),
                _pick(row, _SCOPUS_COLUMNS["index_keywords"]),
            ) if part
        )
        refs = _pick(row, _SCOPUS_COLUMNS["references"])
        records.append(
            Record(
                key=f"{source}:{path.stem}:{idx}",
                source=source,
                title=_pick(row, _SCOPUS_COLUMNS["title"]),
                abstract=_pick(row, _SCOPUS_COLUMNS["abstract"]),
                keywords=keywords,
                authors=_pick(row, _SCOPUS_COLUMNS["authors"]),
                year=_to_int(_pick(row, _SCOPUS_COLUMNS["year"])),
                doi=_pick(row, _SCOPUS_COLUMNS["doi"]),
                source_title=_pick(row, _SCOPUS_COLUMNS["source_title"]),
                doc_type=_pick(row, _SCOPUS_COLUMNS["doc_type"]),
                cited_by=_to_int(_pick(row, _SCOPUS_COLUMNS["cited_by"])),
                n_references=len([r for r in refs.split(";") if r.strip()]) or None,
                uid=_pick(row, _SCOPUS_COLUMNS["uid"]),
                raw=row,
            )
        )
    return records


def _parse_wos_block(block: str) -> dict[str, str]:
    """Parse one tagged WoS record into ``{tag: value}``.

    Continuation lines (three leading spaces) are joined with a single space,
    except for ``CR`` and ``C1`` where each continuation is a separate entry and
    is joined with ``"; "`` instead.
    """
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in block.splitlines():
        match = _WOS_TAG.match(line)
        if match:
            current = match.group(1)
            fields.setdefault(current, []).append(match.group(2).strip())
        elif line.startswith("   ") and current:
            fields[current].append(line.strip())
    multi = {"CR", "C1", "AU", "AF", "EM", "FU", "OI", "RI"}
    return {
        tag: ("; " if tag in multi else " ").join(v for v in values if v)
        for tag, values in fields.items()
    }


def load_wos_plaintext(path: str | Path, source: str = "wos") -> list[Record]:
    """Load a Web of Science tagged plain-text export (``FN``/``VR``/``ER``).

    This is the most faithful WoS format and the one to prefer: it carries the
    full ``CR`` cited-reference block, which the tab-delimited export truncates
    in some subscriptions.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    # Records are terminated by a line containing exactly "ER".
    blocks = [b.strip("\n") for b in re.split(r"^ER\s*$", text, flags=re.MULTILINE)]
    records: list[Record] = []
    saw_cr = False
    for idx, block in enumerate(blocks):
        if "PT " not in block and "TI " not in block:
            continue  # header preamble or trailing "EF"
        # Drop any leading FN/VR header lines that belong to the file, not the record.
        lines = [
            ln for ln in block.splitlines()
            if not ln.startswith(("FN ", "VR ", "EF"))
        ]
        clean = "\n".join(lines).strip("\n")
        if not clean:
            continue
        fields = _parse_wos_block(clean)
        if not fields.get("TI"):
            continue
        saw_cr = saw_cr or bool(fields.get("CR"))
        keywords = "; ".join(p for p in (fields.get("DE", ""), fields.get("ID", "")) if p)
        records.append(
            Record(
                key=f"{source}:{path.stem}:{idx}",
                source=source,
                title=fields.get("TI", ""),
                abstract=fields.get("AB", ""),
                keywords=keywords,
                authors=fields.get("AU", ""),
                year=_to_int(fields.get("PY", "")),
                doi=fields.get("DI", ""),
                source_title=fields.get("SO", ""),
                doc_type=fields.get("DT", ""),
                cited_by=_to_int(fields.get("TC", "")),
                n_references=_to_int(fields.get("NR", "")),
                uid=fields.get("UT", ""),
                raw=clean + "\nER\n",
            )
        )
    if records and not saw_cr:
        print(
            f"WARNING {path.name}: no CR (cited references) in any record. "
            "Re-export from WoS choosing 'Full Record and Cited References'."
        )
    return records


def load_wos_tabbed(path: str | Path, source: str = "wos") -> list[Record]:
    """Load a Web of Science tab-delimited export.

    Kept for convenience, but :func:`load_wos_plaintext` is preferred — see that
    function's note on cited references.
    """
    path = Path(path)
    frame = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        quoting=csv.QUOTE_NONE, encoding="utf-8-sig", encoding_errors="replace",
    )
    frame.columns = [c.strip() for c in frame.columns]
    if "TI" not in frame.columns:
        raise ValueError(f"{path.name}: not a WoS tab-delimited export (no 'TI' column).")
    records: list[Record] = []
    for idx, row in enumerate(frame.to_dict(orient="records")):
        keywords = "; ".join(p for p in (str(row.get("DE", "")), str(row.get("ID", ""))) if p.strip())
        records.append(
            Record(
                key=f"{source}:{path.stem}:{idx}",
                source=source,
                title=str(row.get("TI", "")),
                abstract=str(row.get("AB", "")),
                keywords=keywords,
                authors=str(row.get("AU", "")),
                year=_to_int(str(row.get("PY", ""))),
                doi=str(row.get("DI", "")),
                source_title=str(row.get("SO", "")),
                doc_type=str(row.get("DT", "")),
                cited_by=_to_int(str(row.get("TC", ""))),
                n_references=_to_int(str(row.get("NR", ""))),
                uid=str(row.get("UT", "")),
                raw=row,
            )
        )
    return records


def load_any(path: str | Path) -> list[Record]:
    """Autodetect the export format of ``path`` and load it.

    Detection is by content, not extension: a WoS plain-text file saved as
    ``.csv`` still loads correctly.
    """
    path = Path(path)
    head = path.read_text(encoding="utf-8-sig", errors="replace")[:4000]
    if head.lstrip().startswith("FN ") or re.search(r"^PT [A-Z]", head, flags=re.MULTILINE):
        return load_wos_plaintext(path)
    first_line = head.splitlines()[0] if head.splitlines() else ""
    if "\t" in first_line and "TI" in first_line.split("\t"):
        return load_wos_tabbed(path)
    return load_scopus_csv(path)


def load_directory(directory: str | Path, pattern: str = "*") -> list[Record]:
    """Load every export file under ``directory`` matching ``pattern``.

    Files that fail to parse are reported and skipped rather than aborting the
    run — a single malformed chunk out of twenty should not cost you the batch.
    """
    directory = Path(directory)
    records: list[Record] = []
    for path in sorted(directory.rglob(pattern)):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
            continue
        try:
            loaded = load_any(path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"SKIP {path.name}: {exc}")
            continue
        print(f"  loaded {len(loaded):>6} records from {path.name}")
        records.extend(loaded)
    return records
