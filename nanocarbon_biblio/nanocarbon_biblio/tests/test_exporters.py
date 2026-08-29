"""Export tests — the round trip that the whole R bridge depends on.

If these break, ``convert2df`` silently loses cited references and every
citation-based analysis in the review becomes wrong without erroring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nanocarbon_biblio.classify import classify_all
from nanocarbon_biblio.dedupe import deduplicate
from nanocarbon_biblio.exporters import export_bundle, export_wos_plaintext
from nanocarbon_biblio.loaders import load_directory, load_wos_plaintext


def _bundle(raw_dir: Path, outdir: Path) -> dict:
    result = deduplicate(load_directory(raw_dir))
    classify_all(result.unique)
    return export_bundle(result, outdir, query_note="test run")


def test_bundle_writes_every_expected_file(raw_dir: Path, tmp_path: Path) -> None:
    outdir = tmp_path / "processed"
    manifest = _bundle(raw_dir, outdir)
    for name in ("scopus_kept.csv", "wos_kept.txt", "labels.csv", "manifest.json"):
        assert (outdir / name).exists(), name
    assert manifest["prisma"]["records_identified"] == 7
    assert manifest["prisma"]["duplicates_removed"] == 1
    assert manifest["overlap"]["both"] == 1


def test_wos_export_round_trips_through_the_loader(raw_dir: Path, tmp_path: Path) -> None:
    """Re-loading the written WoS file must give back the same records."""
    outdir = tmp_path / "processed"
    _bundle(raw_dir, outdir)
    reloaded = load_wos_plaintext(outdir / "wos_kept.txt")
    original = load_directory(raw_dir)
    wos_kept = {r.uid for r in reloaded}
    assert wos_kept <= {r.uid for r in original if r.source == "wos"}
    # The header/footer convert2df looks for must be present exactly once.
    text = (outdir / "wos_kept.txt").read_text(encoding="utf-8")
    assert text.startswith("FN Clarivate Analytics Web of Science\nVR 1.0\n")
    assert text.rstrip().endswith("EF")
    assert text.count("FN Clarivate") == 1


def test_cited_references_survive_the_round_trip(raw_dir: Path, tmp_path: Path) -> None:
    """CR is the field that cannot be reconstructed. It must pass through verbatim."""
    outdir = tmp_path / "processed"
    _bundle(raw_dir, outdir)
    text = (outdir / "wos_kept.txt").read_text(encoding="utf-8")
    assert "CR IIJIMA S, 1991, NATURE, V354, P56" in text

    scopus = pd.read_csv(outdir / "scopus_kept.csv", dtype=str, keep_default_na=False)
    assert "References" in scopus.columns
    assert scopus["References"].str.contains("Iijima").any()
    assert scopus["References"].str.contains("Stone").any()


def test_multiline_cited_references_keep_their_continuation_lines(
    raw_dir: Path, tmp_path: Path
) -> None:
    """A CR block spanning several lines must be re-emitted with its indentation.

    convert2df splits CR on the continuation indent; losing it collapses two
    cited references into one malformed string.
    """
    records = load_wos_plaintext(raw_dir / "wos_chunk1.txt")
    out = tmp_path / "all_wos.txt"
    export_wos_plaintext(records, out)
    text = out.read_text(encoding="utf-8")
    assert "CR IIJIMA S, 1991, NATURE, V354, P56" in text
    assert "   STONE AJ, 1986, CHEM PHYS LETT, V128, P501" in text
    # And the folded title's continuation line too.
    assert "   electronic structure" in text
    assert len(load_wos_plaintext(out)) == len(records)


def test_no_record_is_written_to_both_databases(raw_dir: Path, tmp_path: Path) -> None:
    """The union must be exactly the unique corpus, with nothing counted twice."""
    outdir = tmp_path / "processed"
    manifest = _bundle(raw_dir, outdir)
    written = manifest["counts_by_source"]
    assert written["scopus_records_written"] + written["wos_records_written"] == \
        manifest["prisma"]["records_screened"]


def test_labels_are_joinable_and_boolean_columns_are_r_friendly(
    raw_dir: Path, tmp_path: Path
) -> None:
    outdir = tmp_path / "processed"
    _bundle(raw_dir, outdir)
    labels = pd.read_csv(outdir / "labels.csv", dtype=str, keep_default_na=False)
    assert labels.key.is_unique
    assert set(labels.dopant_host_ambiguous.unique()) <= {"TRUE", "FALSE"}
    assert {"study_type", "dopant", "defect", "morphology", "application"} <= set(labels.columns)
    # Every record must carry a join key on at least one of DOI or title.
    assert (labels.doi.str.len().gt(0) | labels.title.str.len().gt(0)).all()


def test_manifest_is_valid_json_with_prisma_counts(raw_dir: Path, tmp_path: Path) -> None:
    outdir = tmp_path / "processed"
    _bundle(raw_dir, outdir)
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query_note"] == "test run"
    assert manifest["prisma"]["flagged_host_ambiguous"] == 1  # the N-doped TiO2 paper
    assert sum(manifest["study_type"].values()) == manifest["prisma"]["records_screened"]
