"""End-to-end CLI test: raw exports in, R-ready bundle out."""

from __future__ import annotations

import json
from pathlib import Path

from nanocarbon_biblio.cli import main


def test_run_pipeline_end_to_end(raw_dir: Path, tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "processed"
    code = main([
        "run", "--raw", str(raw_dir), "--out", str(outdir),
        "--require-topic", "--crosstab", "--note", "cli test",
    ])
    assert code == 0
    assert (outdir / "manifest.json").exists()
    assert (outdir / "crosstab_dopant_application.csv").exists()
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query_note"] == "cli test"
    assert "loaded" in capsys.readouterr().out


def test_run_reports_failure_on_empty_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["run", "--raw", str(empty), "--out", str(tmp_path / "out")]) == 1


def test_thesaurus_subcommand_writes_a_file(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "thesaurus.txt"
    assert main(["thesaurus", "--raw", str(raw_dir), "--out", str(out), "--min-count", "1"]) == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 10
    assert all(";" in line for line in lines)
    # The curated seed must be present.
    assert any(line.startswith("carbon nanotube;") for line in lines)


def test_run_refuses_an_output_directory_inside_the_input(raw_dir: Path, capsys) -> None:
    """The mistake that would re-ingest this run's output on the next run."""
    assert main(["run", "--raw", str(raw_dir), "--out", str(raw_dir / "out")]) == 2
    assert "is inside" in capsys.readouterr().err


def test_demo_subcommand_writes_both_exports(tmp_path: Path) -> None:
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--n-works", "50"]) == 0
    assert (out / "scopus_demo_1991-2025.csv").exists()
    assert (out / "wos_demo_1991-2025.txt").exists()


def test_recall_subcommand_reports_the_known_item_test(raw_dir: Path, tmp_path: Path, capsys) -> None:
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "doi,title,year,why,verified\n"
        "10.1016/j.carbon.2005.01.001,Nitrogen-doped carbon nanotubes,2005,seminal,TRUE\n"
        "10.9/missing,A paper the search never retrieves,2011,control,TRUE\n",
        encoding="utf-8",
    )
    out = tmp_path / "recall.csv"
    assert main(["recall", "--raw", str(raw_dir), "--gold", str(gold), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Recall relativo: 0.500" in printed
    assert "NO recuperados" in printed
    assert out.exists()


def test_prisma_subcommand_fails_loudly_on_bad_arithmetic(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "prisma": {
            "records_identified": 200, "records_identified_by_source": {"scopus": 100, "wos": 100},
            "duplicates_removed": 10, "records_screened": 150,
            "records_without_abstract": 0, "flagged_host_ambiguous": 0,
        },
        "overlap": {},
    }), encoding="utf-8")
    code = main(["prisma", "--manifest", str(manifest), "--out", str(tmp_path / "f.svg")])
    assert code == 1
    assert "INCONSISTENTE" in capsys.readouterr().err
