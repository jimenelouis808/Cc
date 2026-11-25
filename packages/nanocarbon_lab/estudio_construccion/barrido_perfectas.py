"""Can Schwarz P and the knee super-square come out CLEAN, not strained?

The audit (auditoria.py) finds the shipped P cell (k=20, 9 rows) and
super-square (k=12, 7 rows) "strained": sound, but near the edge of the
sp2 window. Every shape whose mesh census is exact is built, relaxed with
the builder's own force field, and judged with the same sp2 verdict.
Resumable: one line per shape in barrido_perfectas.jsonl.

    python barrido_perfectas.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from nanocarbon_lab.builders.knee import (
    build_knee_schwarzite, build_knee_supernetwork, clean_schwarzite_shapes)
from nanocarbon_lab.validation.quality import sp2_quality

HERE = Path(__file__).resolve().parent
OUT = HERE / "barrido_perfectas.jsonl"


def shapes():
    for rows, k in clean_schwarzite_shapes(kind="primitive", rows_candidates=(5, 7, 9, 11)):
        yield "primitive", k, rows
    for k in (12, 20):
        for rows in (5, 7, 9, 11):
            yield "super-square", k, rows


def main() -> None:
    done = set()
    if OUT.exists():
        done = {(r["kind"], r["k"], r["rows"]) for r in map(json.loads, OUT.read_text().splitlines())}
    for kind, k, rows in shapes():
        if (kind, k, rows) in done:
            continue
        t0 = time.time()
        row = {"kind": kind, "k": k, "rows": rows}
        try:
            atoms = (build_knee_schwarzite(circumference=k, arm_rows=rows, kind=kind)
                     if kind == "primitive" else
                     build_knee_supernetwork(kind, circumference=k, arm_rows=rows))
            g = atoms.info["geometry"]
            verdict, why = sp2_quality(g)
            row.update(atoms=len(atoms), rings=atoms.info.get("ring_counts"),
                       cell=atoms.info.get("cell_lengths"),
                       bond_min=g["bond_min"], bond_max=g["bond_max"],
                       angle_min=g["angle_min"], angle_max=g["angle_max"],
                       contacts=g["n_close_contacts"], verdict=verdict, reason=why)
        except Exception as e:  # noqa: BLE001
            row.update(error=str(e)[:200])
        row["seconds"] = round(time.time() - t0, 1)
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
