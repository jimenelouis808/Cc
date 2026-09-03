"""Tests for the PRISMA flow diagram.

A flow diagram is checked by subtraction, so the arithmetic guard matters more
than the drawing.
"""

from __future__ import annotations

import json
from pathlib import Path

from nanocarbon_biblio.prisma import PrismaCounts, counts_from_manifest, render_svg, write_svg


def _manifest() -> dict:
    return {
        "prisma": {
            "records_identified": 4621,
            "records_identified_by_source": {"scopus": 2377, "wos": 2244},
            "duplicates_removed": 1626,
            "records_screened": 2995,
            "records_without_abstract": 3,
            "flagged_host_ambiguous": 118,
        },
        "overlap": {"scopus_only": 750, "wos_only": 620, "both": 1625},
        "counts_by_source": {"scopus_records_written": 2375, "wos_records_written": 620},
    }


def test_identification_uses_pre_deduplication_counts() -> None:
    """PRISMA wants raw retrieval per database, not the post-dedup split."""
    counts = counts_from_manifest(_manifest())
    assert counts.scopus == 2377 and counts.wos == 2244
    assert counts.identified == 4621
    assert counts.identified - counts.duplicates_removed == counts.screened
    assert counts.check_consistency() == []


def test_older_manifests_reconstruct_identification_from_the_overlap() -> None:
    manifest = _manifest()
    del manifest["prisma"]["records_identified_by_source"]
    counts = counts_from_manifest(manifest)
    # scopus_only + both, wos_only + both - the raw counts, not the written ones.
    assert counts.scopus == 750 + 1625
    assert counts.wos == 620 + 1625


def test_inconsistent_arithmetic_is_reported_not_drawn_silently() -> None:
    bad = PrismaCounts(scopus=100, wos=100, duplicates_removed=10, screened=150)
    problems = bad.check_consistency()
    assert len(problems) == 1 and "190" in problems[0]
    assert "⚠" in render_svg(bad)  # the warning reaches the figure itself


def test_included_is_derived_from_the_exclusion_count() -> None:
    counts = counts_from_manifest(_manifest(), excluded_screening=41)
    assert counts.resolved_included() == 2995 - 41
    explicit = counts_from_manifest(_manifest(), included=2000)
    assert explicit.resolved_included() == 2000


def test_pending_decisions_are_labelled_not_invented() -> None:
    svg = render_svg(counts_from_manifest(_manifest()))
    assert "pendiente" in svg
    assert "n = 2995" in svg


def test_write_svg_produces_a_standalone_file(tmp_path: Path) -> None:
    path = write_svg(counts_from_manifest(_manifest(), excluded_screening=41),
                     tmp_path / "flow.svg", title="Prueba")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<svg xmlns=")
    assert text.rstrip().endswith("</svg>")
    assert "Prueba" in text
    assert "http" not in text.replace('xmlns="http://www.w3.org/2000/svg"', "")


def test_counts_from_manifest_accepts_a_path(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert counts_from_manifest(path).identified == 4621
