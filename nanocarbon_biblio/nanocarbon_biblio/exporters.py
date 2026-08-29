"""Write the deduplicated, classified corpus back out for R / bibliometrix.

The bridge to R rests on one decision, and it is the most important design
choice in this package: **the filtered corpus is written back in its native
export format**, not as a reconstructed data frame. ``convert2df`` then parses
Scopus CSV and WoS tagged text exactly as it was written to, so the cited
reference field (``CR``), the affiliation field (``C1``) and every quirk of
bibliometrix's own parsing are preserved. Python contributes the *labels*, in a
separate table joined on a key.

Rebuilding ``CR`` in Python would quietly break co-citation, bibliographic
coupling and RPYS — the three analyses that make the review more than a word
cloud.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .classify import to_dataframe
from .dedupe import DedupeResult, overlap_table
from .loaders import WOS_FOOTER, WOS_HEADER
from .records import Record

__all__ = [
    "export_scopus_csv",
    "export_wos_plaintext",
    "export_wos_tabbed",
    "export_labels",
    "export_bundle",
]


def _count_true(frame: pd.DataFrame, column: str) -> int:
    """Count truthy values in a boolean column, tolerating a missing column."""
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def export_scopus_csv(records: list[Record], path: str | Path) -> Path:
    """Write Scopus-sourced records back as a Scopus CSV.

    Columns are the union of every input file's columns, in first-seen order,
    so heterogeneous export chunks concatenate safely. Missing cells are written
    empty, which ``convert2df(dbsource = "scopus", format = "csv")`` tolerates.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.raw for r in records if isinstance(r.raw, dict)]
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    frame = pd.DataFrame(rows, columns=columns).fillna("")
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def export_wos_plaintext(records: list[Record], path: str | Path) -> Path:
    """Write WoS-sourced records back as a tagged plain-text file.

    Reproduces the ``FN``/``VR`` header and ``EF`` footer that
    ``convert2df(dbsource = "wos", format = "plaintext")`` looks for. Each
    record's stored block already ends with its ``ER`` terminator.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [r.raw for r in records if isinstance(r.raw, str)]
    body = "\n".join(block.rstrip("\n") for block in blocks)
    path.write_text(f"{WOS_HEADER}\n{body}\n{WOS_FOOTER}\n", encoding="utf-8")
    return path


def export_wos_tabbed(records: list[Record], path: str | Path) -> Path:
    """Write WoS records that came from a tab-delimited export back as TSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.raw for r in records if isinstance(r.raw, dict)]
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    frame = pd.DataFrame(rows).fillna("")
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8")
    return path


def export_labels(records: list[Record], path: str | Path) -> Path:
    """Write the Python-side label table joined onto the R data frame.

    One row per unique record. Boolean columns are written as ``TRUE``/``FALSE``
    so R reads them as logicals rather than as character.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = to_dataframe(records)
    for column in frame.columns:
        if frame[column].dtype == bool:
            frame[column] = frame[column].map({True: "TRUE", False: "FALSE"})
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def export_bundle(
    result: DedupeResult,
    outdir: str | Path,
    *,
    query_note: str = "",
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the complete hand-off bundle for the R side.

    Produces, under ``outdir``:

    ==========================  ===============================================
    ``scopus_kept.csv``         Scopus records that survived, native format
    ``wos_kept.txt``            WoS tagged records that survived, native format
    ``wos_kept.tsv``            WoS tab-delimited records, if any were loaded
    ``labels.csv``              Python classification labels, joined on ``key``
    ``manifest.json``           PRISMA counts, overlap table, timestamp
    ==========================  ===============================================

    Both databases' files are written even when a record was deduplicated
    across them: the representative goes to its own source's file, so the union
    is exactly the unique corpus with no record written twice.

    Returns the manifest dict, which is also written to disk. Put its numbers
    straight into the PRISMA flow diagram.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    unique = result.unique

    scopus = [r for r in unique if r.source == "scopus"]
    wos_tagged = [r for r in unique if r.source == "wos" and isinstance(r.raw, str)]
    wos_tabbed = [r for r in unique if r.source == "wos" and isinstance(r.raw, dict)]

    written: dict[str, str] = {}
    if scopus:
        written["scopus_csv"] = str(export_scopus_csv(scopus, outdir / "scopus_kept.csv"))
    if wos_tagged:
        written["wos_plaintext"] = str(export_wos_plaintext(wos_tagged, outdir / "wos_kept.txt"))
    if wos_tabbed:
        written["wos_tabbed"] = str(export_wos_tabbed(wos_tabbed, outdir / "wos_kept.tsv"))
    written["labels"] = str(export_labels(unique, outdir / "labels.csv"))

    frame = to_dataframe(unique)
    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query_note": query_note,
        "prisma": {
            "records_identified": result.n_input,
            "duplicates_removed": result.n_removed,
            "records_screened": len(unique),
            "records_without_abstract": _count_true(frame, "no_abstract"),
            "flagged_host_ambiguous": _count_true(frame, "dopant_host_ambiguous"),
        },
        "overlap": overlap_table(result),
        "counts_by_source": {
            "scopus_records_written": len(scopus),
            "wos_records_written": len(wos_tagged) + len(wos_tabbed),
        },
        "study_type": frame["study_type"].value_counts().to_dict() if "study_type" in frame else {},
        "files": written,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
