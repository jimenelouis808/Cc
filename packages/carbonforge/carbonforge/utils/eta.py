"""Time left for a calculation, from the files it has already written.

carbonforge's jobs are external programs (GPAW scripts, pw.x, LAMMPS) that cannot be
asked how far they are; what they leave behind can. A GPAW vibration run writes one
``cache.*.json`` per displacement, a Raman run one ``alpha.*.json`` per polarizability:
their modification times are the moments each step ended. The time left is then the
remaining count times the mean interval between the most recent ``window`` of them
(twenty: a mean over the whole run lags when the steps speed up or slow down, the last
interval alone jumps). With fewer than two finished steps there is nothing to measure
and only the elapsed time is given; nor is anything said when the files came less
than a second apart (copied or written by a test, not computed).

A run whose newest file is much older than its usual step (three steps, and at least
five minutes) is marked "¿detenido?": a process that was killed cannot say so itself.
"""

from __future__ import annotations

import time
from pathlib import Path


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


def estimate(finished: list[float], total: int, now: float | None = None,
             window: int = 20) -> dict:
    """``{"seconds_per_step", "remaining_seconds", "end", "stale"}`` from the completion
    times (Unix seconds) of the steps done so far, out of ``total``."""
    now = time.time() if now is None else now
    times = sorted(finished)
    out = {"seconds_per_step": None, "remaining_seconds": None, "end": None, "stale": False}
    if len(times) >= 2:
        recent = times[-(window + 1):]
        rate = (recent[-1] - recent[0]) / (len(recent) - 1)
        out["seconds_per_step"] = rate
        left = max(0, total - len(times))
        if left:
            # the step in progress started at the last completion
            remaining = max(0.0, rate * left - (now - times[-1]))
            out["remaining_seconds"] = remaining
            out["end"] = time.strftime("%H:%M", time.localtime(now + remaining))
            out["stale"] = now - times[-1] > max(3 * rate, 300)
    return out


def suffix(finished: list[float], total: int, now: float | None = None) -> str:
    """' · quedan ~44 min (≈ 10:52)', ' · ¿detenido? …' or '' — to append to a count."""
    e = estimate(finished, total, now)
    now = time.time() if now is None else now
    if e["stale"]:
        return f" · ¿detenido? (sin archivos nuevos hace {duration(now - max(finished))})"
    if e["remaining_seconds"] is None or e["seconds_per_step"] < 1.0:
        return ""                         # files written in under a second: not a run
    return f" · quedan ~{duration(e['remaining_seconds'])} (≈ {e['end']})"


def from_files(paths, total: int, now: float | None = None) -> str:
    """:func:`suffix` with the modification times of ``paths``."""
    times = []
    for p in paths:
        try:
            times.append(Path(p).stat().st_mtime)
        except OSError:
            continue
    return suffix(times, total, now)


def gpaw_scf_note(txt: Path, tail_bytes: int = 20000) -> str:
    """' · SCF iteración 12' from the end of a GPAW text log, or ''. GPAW does not know
    in advance how many iterations a density takes; the current one says it is alive."""
    import re

    path = Path(txt)
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - tail_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    hits = re.findall(r"^iter:\s+(\d+)", text, flags=re.MULTILINE)
    return f" · SCF iteración {hits[-1]}" if hits else ""
