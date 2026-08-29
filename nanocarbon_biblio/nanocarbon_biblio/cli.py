"""Command-line entry point — the scriptable equivalent of the GUI.

Anything the Streamlit app does, this does headlessly and reproducibly::

    python -m nanocarbon_biblio.cli run --raw data/raw --out data/processed
    python -m nanocarbon_biblio.cli thesaurus --raw data/raw --out queries/thesaurus.txt

Use the GUI to explore and to make screening decisions; use the CLI for the
final run that goes into the paper, and commit the command you used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classify import classify_all, crosstab, to_dataframe
from .dedupe import DedupeResult, deduplicate, overlap_table
from .exporters import export_bundle
from .loaders import load_directory
from .thesaurus import suggest_synonyms, write_thesaurus

__all__ = ["main", "build_parser"]


def _cmd_run(args: argparse.Namespace) -> int:
    """Load → deduplicate → classify → export. Returns a process exit code."""
    print(f"Loading exports from {args.raw} …")
    records = load_directory(args.raw)
    if not records:
        print(f"ERROR: no readable exports under {args.raw}", file=sys.stderr)
        return 1
    print(f"  {len(records)} records loaded")

    print("Deduplicating …")
    result = deduplicate(
        records, title_threshold=args.title_threshold, year_window=args.year_window
    )
    print(f"  {len(result.unique)} unique, {result.n_removed} duplicates removed")
    print(f"  overlap: {overlap_table(result)}")

    print("Classifying …")
    classify_all(result.unique)
    frame = to_dataframe(result.unique)

    if args.require_topic:
        keep = frame[frame.has_dopant.astype(bool) | frame.has_defect.astype(bool)]
        keys = set(keep.key)
        dropped = len(result.unique) - len(keys)
        result = DedupeResult(
            unique=[r for r in result.unique if r.key in keys],
            clusters=result.clusters,
            n_input=result.n_input,
        )
        print(f"  screening: dropped {dropped} records with no dopant or defect signal")

    manifest = export_bundle(result, args.out, query_note=args.note)
    print(json.dumps(manifest["prisma"], indent=2))
    print(f"Wrote bundle to {args.out}")

    if args.crosstab:
        table = crosstab(to_dataframe(result.unique))
        table.to_csv(Path(args.out) / "crosstab_dopant_application.csv")
        print(f"Wrote crosstab to {Path(args.out) / 'crosstab_dopant_application.csv'}")
    return 0


def _cmd_thesaurus(args: argparse.Namespace) -> int:
    """Suggest synonym groups and write a biblioshiny thesaurus file."""
    records = load_directory(args.raw)
    if not records:
        print(f"ERROR: no readable exports under {args.raw}", file=sys.stderr)
        return 1
    groups = suggest_synonyms(records, min_count=args.min_count, threshold=args.threshold)
    path = write_thesaurus(groups, args.out, include_seed=True)
    print(f"Wrote {path} — {len(groups)} suggested groups plus the curated seed.")
    print("REVIEW IT BY HAND before using it: 'n-doped' and 'p-doped' score high and are opposites.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed so tests can invoke subcommands)."""
    parser = argparse.ArgumentParser(
        prog="nanocarbon_biblio",
        description="Scopus + WoS → deduplicated, classified corpus → bibliometrix.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="full pipeline: load, dedupe, classify, export")
    run.add_argument("--raw", default="data/raw", help="directory of raw exports")
    run.add_argument("--out", default="data/processed", help="output directory")
    run.add_argument("--title-threshold", type=float, default=92.0,
                     help="fuzzy title cutoff, 0-100 (default 92)")
    run.add_argument("--year-window", type=int, default=1,
                     help="year tolerance when matching titles (default 1)")
    run.add_argument("--require-topic", action="store_true",
                     help="drop records where no dopant and no defect rule fired")
    run.add_argument("--crosstab", action="store_true",
                     help="also write the dopant x application matrix")
    run.add_argument("--note", default="", help="free-text note stored in manifest.json")
    run.set_defaults(func=_cmd_run)

    thes = sub.add_parser("thesaurus", help="suggest a biblioshiny synonyms file")
    thes.add_argument("--raw", default="data/raw")
    thes.add_argument("--out", default="queries/thesaurus.txt")
    thes.add_argument("--min-count", type=int, default=5)
    thes.add_argument("--threshold", type=float, default=88.0)
    thes.set_defaults(func=_cmd_thesaurus)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
