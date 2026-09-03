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
from .demo import DemoConfig, generate_demo_corpus
from .dedupe import DedupeResult, deduplicate, overlap_table
from .exporters import export_bundle
from .indicators import combined_share_trend, dopant_lag_table, gap_matrix, annotate_records
from .loaders import load_directory
from .prisma import counts_from_manifest, write_svg
from .validation import load_gold_standard, score_agreement, score_recall
from .thesaurus import suggest_synonyms, write_thesaurus

__all__ = ["main", "build_parser"]


def _cmd_run(args: argparse.Namespace) -> int:
    """Load → deduplicate → classify → export. Returns a process exit code."""
    raw = Path(args.raw).resolve()
    out = Path(args.out).resolve()
    if out == raw or raw in out.parents:
        print(
            f"ERROR: --out ({out}) is inside --raw ({raw}). The next run would "
            "re-ingest this run's own output. Use a directory outside --raw.",
            file=sys.stderr,
        )
        return 2

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
    annotate_records(result.unique)
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

    final = to_dataframe(result.unique)
    outdir = Path(args.out)
    if args.crosstab:
        crosstab(final).to_csv(outdir / "crosstab_dopant_application.csv")
        print(f"Wrote crosstab to {outdir / 'crosstab_dopant_application.csv'}")

    if args.indicators:
        trend = combined_share_trend(final)
        trend["table"].to_csv(outdir / "rq2_study_type_share.csv")
        dopant_lag_table(final).to_csv(outdir / "rq2_dopant_lag.csv", index=False)
        gaps = gap_matrix(final)
        gaps.to_csv(outdir / "rq3_gap_matrix.csv", index=False)
        print(
            f"RQ2: Spearman rho = {trend['spearman_rho']} for the combined share "
            f"across {trend['n_years']} years"
        )
        if not gaps.empty:
            n_gap = int((gaps.status == "theory_only").sum())
            print(f"RQ3: {n_gap} theory-only cells (predicted, not yet realised)")
        counts = counts_from_manifest(manifest)
        write_svg(counts, outdir / "prisma_flow.svg",
                  title="Defectos y dopaje en nanocarbonos 1D — flujo PRISMA 2020")
        problems = counts.check_consistency()
        print(f"Wrote RQ2/RQ3 tables and prisma_flow.svg to {outdir}")
        if problems:
            print("PRISMA inconsistente: " + "; ".join(problems), file=sys.stderr)
        else:
            print("PRISMA: falta el recuento del cribado manual; regenera con "
                  "`nanocarbon_biblio prisma --excluded N` cuando lo tengas.")
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    """Known-item test: does the corpus contain the gold standard?"""
    gold = load_gold_standard(args.gold)
    if not gold:
        print(f"ERROR: {args.gold} has no usable entries (needs a doi or a title).",
              file=sys.stderr)
        return 1
    records = load_directory(args.raw)
    if not records:
        print(f"ERROR: no readable exports under {args.raw}", file=sys.stderr)
        return 1

    report = score_recall(records, gold)
    print(f"\nRecall relativo: {report.relative_recall:.3f} "
          f"({report.n_found}/{report.n_gold})")
    print(f"Encontrados por base: {report.by_source}")
    if report.unverified:
        print(f"AVISO: {report.unverified} entradas del conjunto de oro sin verificar.")
    missing = report.missing()
    if not missing.empty:
        print(f"\nNO recuperados ({len(missing)}) — cada uno señala un agujero de vocabulario:")
        for row in missing.to_dict(orient="records"):
            print(f"  · {row['title']}  [{row['why']}]")
    if report.relative_recall < 0.95:
        print("\nPor debajo del objetivo 0.95. Arregla la consulta, no añadas los "
              "que faltan a mano: el mismo agujero esconde trabajos que no conoces.")
    if args.out:
        report.rows.to_csv(args.out, index=False)
        print(f"\nTabla completa en {args.out}")
    return 0


def _cmd_agreement(args: argparse.Namespace) -> int:
    """Score a human-coded validation sheet against the rule output."""
    results = score_agreement(args.sheet)
    if not results.get("n_coded"):
        print("ERROR: la hoja no tiene ninguna columna manual_* rellenada.", file=sys.stderr)
        return 1
    for facet, payload in results.items():
        if not isinstance(payload, dict):
            continue
        print(f"\n=== {facet} (n = {payload.get('n_coded', 0)}) ===")
        for key in ("kappa", "accuracy", "mean_jaccard", "exact_set_match", "note"):
            if key in payload:
                print(f"  {key}: {payload[key]}")
        if "per_class" in payload:
            print(payload["per_class"].to_string(index=False))
    print(f"\n{results['interpretation']}")
    return 0


def _cmd_prisma(args: argparse.Namespace) -> int:
    """Draw the PRISMA 2020 flow diagram from a manifest."""
    counts = counts_from_manifest(
        args.manifest, excluded_screening=args.excluded, included=args.included
    )
    problems = counts.check_consistency()
    path = write_svg(counts, args.out, title=args.title)
    print(f"Wrote {path}")
    print(f"  identificados {counts.identified} − duplicados {counts.duplicates_removed} "
          f"= cribados {counts.screened} → incluidos {counts.resolved_included()}")
    if problems:
        print("INCONSISTENTE: " + "; ".join(problems), file=sys.stderr)
        return 1
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


def _cmd_demo(args: argparse.Namespace) -> int:
    """Write a synthetic corpus so the pipeline can be exercised without Scopus."""
    summary = generate_demo_corpus(
        args.out, DemoConfig(n_works=args.n_works, seed=args.seed)
    )
    print(json.dumps(summary, indent=2))
    print(
        "\nSynthetic data — for exercising the pipeline and learning the GUI only.\n"
        f"Next: python -m nanocarbon_biblio.cli run --raw {args.out} --out data/processed"
    )
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
    run.add_argument("--indicators", action="store_true",
                     help="also write the RQ2 (theory/experiment lag) and RQ3 (gap matrix) tables")
    run.add_argument("--note", default="", help="free-text note stored in manifest.json")
    run.set_defaults(func=_cmd_run)

    thes = sub.add_parser("thesaurus", help="suggest a biblioshiny synonyms file")
    thes.add_argument("--raw", default="data/raw")
    thes.add_argument("--out", default="queries/thesaurus.txt")
    thes.add_argument("--min-count", type=int, default=5)
    thes.add_argument("--threshold", type=float, default=88.0)
    thes.set_defaults(func=_cmd_thesaurus)

    demo = sub.add_parser(
        "demo", help="generate a synthetic corpus for testing and for learning the GUI"
    )
    demo.add_argument("--out", default="data/raw/demo")
    demo.add_argument("--n-works", type=int, default=1200)
    demo.add_argument("--seed", type=int, default=20260830)
    demo.set_defaults(func=_cmd_demo)

    recall = sub.add_parser(
        "recall", help="known-item test of the query against a gold standard"
    )
    recall.add_argument("--raw", default="data/raw")
    recall.add_argument("--gold", default="queries/gold_standard.csv")
    recall.add_argument("--out", default="", help="optional CSV for the full table")
    recall.set_defaults(func=_cmd_recall)

    agree = sub.add_parser(
        "agreement", help="Cohen's kappa of a coded validation sheet vs the rules"
    )
    agree.add_argument("--sheet", required=True, help="CSV with the manual_* columns filled in")
    agree.set_defaults(func=_cmd_agreement)

    pr = sub.add_parser("prisma", help="draw the PRISMA 2020 flow diagram as SVG")
    pr.add_argument("--manifest", default="data/processed/manifest.json")
    pr.add_argument("--out", default="results/prisma_flow.svg")
    pr.add_argument("--excluded", type=int, default=None,
                    help="records excluded during manual screening")
    pr.add_argument("--included", type=int, default=None,
                    help="final included count (derived from --excluded if omitted)")
    pr.add_argument("--title", default="Flujo PRISMA 2020")
    pr.set_defaults(func=_cmd_prisma)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
