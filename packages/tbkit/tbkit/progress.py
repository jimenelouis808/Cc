"""How far a long calculation got and when it will end: a counter with an estimated time.

Every long loop in tbkit (finite-difference Hessians, Raman tensors, double-resonance
sums, reference sets) can report through :class:`Progress`::

    with Progress(len(jobs), "pares 2.33 eV", status=folder / "progreso.json") as bar:
        for job in jobs:
            ...
            bar.step()

It prints one line, rewritten in place on a terminal (or one line every ``every``
seconds in a log), with the count, the elapsed time and the time left::

    pares 2.33 eV: 12/26 (46 %) · 38 min · quedan ~44 min (≈ 10:52)

and, with ``status``, keeps a small JSON file up to date that ``tbkit estado`` reads, so
a calculation running in the background (or on another machine, in a shared folder)
can be watched from a terminal or the window.

**How the estimate is made.** The time left is the remaining count times the mean time
per step over the most recent ``window`` steps (twenty by default): a running mean of
the whole history lags when the steps get faster or slower part way through (another
job finishing, a laser with more resonant states), and the last step alone jumps. The
first estimate appears after the first step; steps that were already done when the
loop started (a resumed calculation: ``done=``) count for the fraction but not for the
rate. ``tbkit estado`` marks a status file *detenido?* when it has not been updated
for three times the mean step (a killed process cannot say so itself).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Self


def duration(seconds: float) -> str:
    """'45 s', '12 min', '3 h 05 min', '2 d 4 h'."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours, minutes = divmod(round(minutes), 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h"


class Progress:
    """Counter with an estimated time left. See the module note."""

    def __init__(self, total: int, label: str = "", status: str | Path | None = None,
                 stream=None, every: float = 30.0, window: int = 20, done: int = 0,
                 quiet: bool = False):
        self.total = int(total)
        self.label = label
        self.status = Path(status) if status else None
        self.stream = sys.stderr if stream is None else stream
        self.every = every
        self.done = int(done)
        self.quiet = quiet
        self.started = time.time()
        self._last = self.started
        self._steps: deque[float] = deque(maxlen=window)
        self._printed = 0.0
        self._tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.note = ""
        self._write()

    # ------------------------------------------------------------------ state
    @property
    def seconds_per_step(self) -> float | None:
        return sum(self._steps) / len(self._steps) if self._steps else None

    @property
    def remaining_seconds(self) -> float | None:
        rate = self.seconds_per_step
        return None if rate is None else rate * max(0, self.total - self.done)

    def line(self) -> str:
        fraction = f" ({100 * self.done / self.total:.0f} %)" if self.total else ""
        text = f"{self.label + ': ' if self.label else ''}{self.done}/{self.total}{fraction}"
        text += f" · {duration(time.time() - self.started)}"
        left = self.remaining_seconds
        if self.done >= self.total:
            text += " · terminado"
        elif left is not None:
            end = time.strftime("%H:%M", time.localtime(time.time() + left))
            text += f" · quedan ~{duration(left)} (≈ {end})"
        return text + (f" · {self.note}" if self.note else "")

    # ------------------------------------------------------------------ update
    def step(self, n: int = 1, note: str = "") -> None:
        now = time.time()
        if n > 0:
            per = (now - self._last) / n
            self._steps.extend([per] * min(n, self._steps.maxlen or n))
        self._last = now
        self.done += n
        if note:
            self.note = note
        self._write()

    def _write(self) -> None:
        now = time.time()
        if self.status is not None:
            data = {"label": self.label, "done": self.done, "total": self.total,
                    "started": self.started, "updated": now, "pid": os.getpid(),
                    "seconds_per_step": self.seconds_per_step,
                    "remaining_seconds": self.remaining_seconds, "note": self.note}
            self.status.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self.status)
        if self.quiet:
            return
        if self._tty:
            self.stream.write("\r" + self.line() + "\033[K")
            self.stream.flush()
        elif now - self._printed >= self.every or self.done >= self.total:
            self.stream.write(self.line() + "\n")
            self.stream.flush()
            self._printed = now

    def close(self) -> None:
        if self._tty and not self.quiet:
            self.stream.write("\n")
            self.stream.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def callback(self):
        """``progress(done, total)`` for functions that take one (``resonant_raman``)."""
        def report(done: int, total: int) -> None:
            self.total = total
            self.step(done - self.done)
        return report


def read_status(path: str | Path) -> dict:
    """A status file with its line and whether it looks stopped."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    now = time.time()
    rate = data.get("seconds_per_step")
    finished = data["done"] >= data["total"]
    stale = (not finished and rate is not None
             and now - data["updated"] > max(3 * rate, 300))
    left = data.get("remaining_seconds")
    if left is not None and not finished:
        left = max(0.0, left - (now - data["updated"]))
    text = f"{data['label'] or Path(path).parent.name}: {data['done']}/{data['total']}"
    if data["total"]:
        text += f" ({100 * data['done'] / data['total']:.0f} %)"
    text += f" · {duration(now - data['started'])}"
    if finished:
        text += " · terminado"
    elif stale:
        text += f" · detenido? (sin noticias hace {duration(now - data['updated'])})"
    elif left is not None:
        end = time.strftime("%H:%M", time.localtime(now + left))
        text += f" · quedan ~{duration(left)} (≈ {end})"
    if data.get("note"):
        text += f" · {data['note']}"
    return {**data, "line": text, "stale": stale, "finished": finished, "path": str(path)}


def status_files(folder: str | Path, pattern: str = "*progreso*.json") -> list[dict]:
    """Every status file under ``folder``, most recently updated first."""
    out = []
    for path in Path(folder).rglob(pattern):
        try:
            out.append(read_status(path))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(out, key=lambda d: -d["updated"])
