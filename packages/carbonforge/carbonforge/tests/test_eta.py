"""utils.eta: time left from the completion times of the files a job writes."""

from __future__ import annotations

import os

import pytest

from carbonforge.utils import eta


def test_estimate_from_recent_intervals():
    finished = [0, 100, 200, 210, 220, 230]          # steps got faster
    e = eta.estimate(finished, total=10, now=230, window=3)
    assert e["seconds_per_step"] == pytest.approx(10.0)
    assert e["remaining_seconds"] == pytest.approx(40.0)
    assert not e["stale"]


def test_step_in_progress_counts_and_stale_runs_are_flagged():
    finished = [0, 60, 120]
    assert eta.estimate(finished, 5, now=150)["remaining_seconds"] == pytest.approx(90.0)
    assert "¿detenido?" in eta.suffix(finished, 5, now=120 + 3600)
    assert eta.suffix([0], 5, now=10) == ""              # nothing to measure yet


def test_from_files_reads_modification_times(tmp_path):
    paths = []
    for k, t in enumerate((1000.0, 1060.0, 1120.0)):
        p = tmp_path / f"cache.{k}.json"
        p.write_text("{}")
        os.utime(p, (t, t))
        paths.append(p)
    assert "quedan ~2 min" in eta.from_files(paths, 5, now=1120.0)


def test_durations():
    assert eta.duration(45) == "45 s"
    assert eta.duration(3 * 3600 + 5 * 60) == "3 h 05 min"


def test_gpaw_scf_note(tmp_path):
    log = tmp_path / "gpaw.txt"
    log.write_text("header\niter:   1  10:00:01  -100.0\niter:   2  10:00:05  -101.0\n")
    assert eta.gpaw_scf_note(log) == " · SCF iteración 2"
    assert eta.gpaw_scf_note(tmp_path / "missing.txt") == ""
