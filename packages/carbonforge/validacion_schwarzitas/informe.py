"""Schwarzite uptake against Kowalczyk, and against the Chahine rule.

Kowalczyk is **not** a closed case the way Levesque is, and the report
says so rather than presenting a percentage. His numbers live in
figures; the only ones in his text are "~3 wt% at 3.8 MPa and 77 K" for
the primitive, diamond and gyroid, and 6 wt% for the icosahedral. And a
"primitive schwarzite" is a family rather than a file: the uptake per
gram follows the surface area, which follows the cell size and the tube
radius, and ours need not be his.

So the comparison that carries weight here is the **internal** one. The
Chahine rule -- about 1 wt% of hydrogen per 500 m2/g at 77 K -- is an
empirical regularity across activated carbons, and four structures of
three different topologies either follow it or they do not. It would
not catch a 10 % error; it would catch a factor of two, which is the
size of error a new framework class invites.
"""
from __future__ import annotations

import json
import pathlib
import sys

BASE = pathlib.Path(sys.argv[1])
#: m2/g from `analyse_porosity`, geometric and not BET.
AREA = {"primitive": 2543, "diamond": 1717, "gyroid": 2534}
#: The only figures Kowalczyk puts in his text, all at 3.8 MPa and 77 K.
KOWALCZYK = 3.0
CHAHINE = 500.0


def main() -> int:
    print(f"{'estructura':<11} {'P (MPa)':>8} {'wt% total':>10} "
          f"{'wt% /C':>8} {'rho_o':>9} {'N':>8}")
    rows: dict[str, list[dict]] = {}
    for kind in ("primitive", "diamond", "gyroid"):
        path = BASE / f"iso_{kind}" / "resultados.json"
        if not path.exists():
            continue
        points = json.loads(path.read_text())["puntos"]
        rows[kind] = points
        for row in points:
            print(f"{kind:<11} {row['presion_mpa']:8.1f} "
                  f"{row['wt_pct_total']:10.2f} "
                  f"{row['wt_pct_por_carbono']:8.2f} "
                  f"{row['densidad_absoluta_g_cm3']:9.5f} "
                  f"{row['n_medio']:8.0f}")
            for warning in row.get("avisos", []):
                print(f"            aviso: {warning}")

    print()
    print("Contra la regla de Chahine (~1 wt% por 500 m2/g a 77 K):")
    print(f"{'estructura':<11} {'area':>7} {'esperado':>9} "
          f"{'a 3.8 MPa':>10} {'razon':>7}")
    for kind, points in rows.items():
        at38 = next((r for r in points if abs(r["presion_mpa"] - 3.8) < 1e-6),
                    None)
        expected = AREA[kind] / CHAHINE
        if at38 is None:
            print(f"{kind:<11} {AREA[kind]:7d} {expected:9.1f} "
                  f"{'pendiente':>10}")
            continue
        got = at38["wt_pct_total"]
        print(f"{kind:<11} {AREA[kind]:7d} {expected:9.1f} {got:10.2f} "
              f"{got / expected:7.2f}")

    print()
    print(f"Kowalczyk cita ~{KOWALCZYK:.1f} wt% a 3.8 MPa y 77 K para las "
          f"tres, leido de una figura.")
    print("Una schwarzita 'primitive' es una familia, no un fichero: el "
          "area por gramo depende del tamano de celda y del radio de tubo, "
          "y las nuestras no tienen por que ser las suyas. Esto es una "
          "comprobacion de plausibilidad de clase, no una validacion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
