"""tbkit.progress: the estimate is the remaining count times the recent time per step."""

from __future__ import annotations

import io
import json
import time

import pytest

from tbkit import progress as pg


def test_estimate_uses_recent_steps(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(pg.time, "time", lambda: clock[0])
    bar = pg.Progress(10, "prueba", stream=io.StringIO(), window=3)
    for dt in (10.0, 10.0, 2.0, 2.0, 2.0):            # steps get faster
        clock[0] += dt
        bar.step()
    assert bar.seconds_per_step == pytest.approx(2.0)   # only the last three count
    assert bar.remaining_seconds == pytest.approx(10.0)
    assert "quedan ~10 s" in bar.line()


def test_resumed_steps_count_for_fraction_not_rate(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(pg.time, "time", lambda: clock[0])
    bar = pg.Progress(10, stream=io.StringIO(), done=8)
    assert bar.remaining_seconds is None and "8/10 (80 %)" in bar.line()
    clock[0] += 60
    bar.step()
    assert bar.remaining_seconds == pytest.approx(60.0)


def test_status_file_and_stale_detection(tmp_path, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(pg.time, "time", lambda: clock[0])
    path = tmp_path / "a" / "progreso.json"
    bar = pg.Progress(4, "fd", status=path, stream=io.StringIO())
    clock[0] += 120
    bar.step()
    data = json.loads(path.read_text())
    assert data["done"] == 1 and data["seconds_per_step"] == pytest.approx(120)
    assert not pg.read_status(path)["stale"]
    clock[0] += 3600                                 # an hour without news
    assert pg.read_status(path)["stale"]
    assert [d["label"] for d in pg.status_files(tmp_path)] == ["fd"]


def test_durations():
    assert pg.duration(45) == "45 s"
    assert pg.duration(12 * 60) == "12 min"
    assert pg.duration(3 * 3600 + 5 * 60) == "3 h 05 min"
    assert pg.duration(50 * 3600) == "2 d 2 h"
