"""Analizar muchos espectros y volcar una tabla comparativa.

    python -m ramancarbon.examples.ex05_lote
"""

from __future__ import annotations

from pathlib import Path

from ramancarbon.examples.demo_data import DEMO_KINDS, make_demo
from ramancarbon.gui.state import Session


def main() -> None:
    out = Path("salida/ex05")
    session = Session()
    for index, kind in enumerate(DEMO_KINDS):
        session.add(make_demo(kind, seed=index))

    ok, bad = session.analyse_all()
    print(f"{ok} espectros analizados, {bad} con error.\n")

    columns, rows = session.results_table()
    keep = ["nombre", "material", "confianza", "ID_IG", "I2D_IG", "ID_IDp", "modelo"]
    indices = [columns.index(c) for c in keep if c in columns]
    widths = [max(len(keep[i]), *(len(row[j]) for row in rows))
              for i, j in enumerate(indices)]
    print("  ".join(name.ljust(w) for name, w in zip(keep, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(row[j].ljust(w) for j, w in zip(indices, widths)))

    path = session.export_table(out / "resultados.csv")
    print(f"\nTabla completa en {path}")


if __name__ == "__main__":
    main()
