"""Schwarz D cells of growing arm length (the clean shapes: k=18, 5/7/9 rows),
written as extxyz for tbkit's gap-vs-size scan.

    python escala_D.py   -> escala_D/D_k18_n{5,7,9}.extxyz
"""
from __future__ import annotations

from pathlib import Path

from ase.io import write

from nanocarbon_lab.builders.knee import build_knee_schwarzite
from nanocarbon_lab.validation.quality import sp2_quality

HERE = Path(__file__).resolve().parent
(HERE / "escala_D").mkdir(exist_ok=True)
for rows in (5, 7, 9):
    target = HERE / "escala_D" / f"D_k18_n{rows}.extxyz"
    if target.exists():
        continue
    a = build_knee_schwarzite(circumference=18, arm_rows=rows, kind="diamond")
    verdict, _ = sp2_quality(a.info["geometry"])
    a.info = {"rows": rows, "verdict": verdict, "cell": float(a.cell.lengths()[0])}
    write(str(target), a)
    print(rows, len(a), verdict, round(float(a.cell.lengths()[0]), 2), flush=True)
