"""Tests for the synthetic corpus and the guards it exposed.

The demo corpus doubles as a regression harness: it knows its own ground truth,
so the pipeline's deduplication can be scored against it rather than eyeballed.
"""

from __future__ import annotations

from pathlib import Path

from nanocarbon_biblio.classify import classify_all, to_dataframe
from nanocarbon_biblio.dedupe import deduplicate, overlap_table
from nanocarbon_biblio.demo import DemoConfig, generate_demo_corpus
from nanocarbon_biblio.loaders import load_directory


def test_demo_corpus_is_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    generate_demo_corpus(a, DemoConfig(n_works=120))
    generate_demo_corpus(b, DemoConfig(n_works=120))
    for name in ("scopus_demo_1991-2025.csv", "wos_demo_1991-2025.txt"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_pipeline_recovers_the_known_overlap(tmp_path: Path) -> None:
    """Deduplication is scored against ground truth, not inspected by eye."""
    raw = tmp_path / "raw"
    summary = generate_demo_corpus(raw, DemoConfig(n_works=600))
    truth = summary["true_overlap"]

    result = deduplicate(load_directory(raw))
    overlap = overlap_table(result)

    # Allow a handful of misses out of 600: some synthetic titles are near
    # duplicates by construction. A large drift means the matcher regressed.
    assert abs(len(result.unique) - summary["n_works"]) <= 8
    for bucket in ("both", "scopus_only", "wos_only"):
        assert abs(overlap[bucket] - truth[bucket]) <= 8, bucket


def test_demo_corpus_contains_the_hard_cases(tmp_path: Path) -> None:
    """The noise the classifier is supposed to survive must actually be there."""
    raw = tmp_path / "raw"
    generate_demo_corpus(raw, DemoConfig(n_works=800))
    result = deduplicate(load_directory(raw))
    classify_all(result.unique)
    frame = to_dataframe(result.unique)

    assert frame.dopant_host_ambiguous.sum() > 0, "no host-ambiguous records generated"
    assert frame.is_3d_assembly.sum() > 0, "no 3D assembly records generated"
    assert (frame.study_type == "combined").sum() > 0, "no combined theory+experiment records"
    assert frame.dopant.str.contains("codoped", na=False).sum() > 0, "no co-doped records"

    # The p-type noise must NOT be read as phosphorus doping.
    ptype = frame[frame.title.str.contains("p-doped silicon", case=False, na=False)]
    assert len(ptype) > 0, "no p-type decoys generated"
    assert not ptype.dopant.str.contains("phosphorus", na=False).any()


def test_every_record_carries_cited_references(tmp_path: Path) -> None:
    """Without CR the whole R-side citation analysis is dead; the demo must have it."""
    raw = tmp_path / "raw"
    generate_demo_corpus(raw, DemoConfig(n_works=100))
    records = load_directory(raw)
    assert all(r.n_references and r.n_references > 0 for r in records)
