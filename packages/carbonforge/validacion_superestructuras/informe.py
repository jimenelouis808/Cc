"""Collect every hydrogen isotherm measured so far into one comparison.

The point of putting the schwarzites and the superstructures in one table
is the last column.  Excess uptake divided by the Chahine rule's
prediction (1 wt% per 500 m2/g at 77 K) is uptake per unit of area, so it
separates "this framework has a lot of surface" from "this framework uses
its surface well" -- and those two answers are what decide whether a
family is worth pursuing.

Absolute and excess are both printed because for a very open framework
they say different things: the absolute number counts every molecule in
the cell, most of which would be there as compressed gas without any
carbon, while the excess subtracts exactly that.  The excess is the one
to quote, and it is recomputed here from the excess MASS rather than
rescaled from the absolute percentage -- wt% is not linear in the
adsorbed mass, so rescaling is wrong by a few per cent at these loadings.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHWARZITES = HERE.parent / "validacion_schwarzitas"
CHAHINE_M2_G_PER_WT = 500.0   # 1 wt% per 500 m2/g at 77 K
REPORT_PRESSURE_MPA = 3.8     # the point every family has in common


def excess_weight_percent(point: dict, porosity: dict) -> float:
    """Excess uptake as a percentage of total mass, from the excess mass.

    `100 * m_exc / (m_exc + m_C)`, Kowalczyk's eqn 5 convention.  Taking
    the absolute percentage and scaling it by the density ratio gives a
    different (wrong) number, because the denominator moves too.
    """
    volume_cm3 = porosity["cell_volume_a3"] * 1e-24
    excess_mass = point["densidad_exceso_g_cm3"] * volume_cm3
    framework = porosity["framework_mass_g"]
    return 100.0 * excess_mass / (excess_mass + framework)


def rows() -> list[dict]:
    """One row per framework, at the common reporting pressure."""
    sources = [
        ("Schwarz D", SCHWARZITES / "diamond.json"),
        ("Schwarz P", SCHWARZITES / "primitive.json"),
        ("giroide", SCHWARZITES / "gyroid.json"),
        ("super-grafeno", HERE / "supergrafeno.json"),
        ("supertubo", HERE / "supertubo.json"),
        ("superfulereno", HERE / "superfulereno.json"),
    ]
    out = []
    for name, path in sources:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        poro = data["porosidad"]
        for point in data["puntos"]:
            if abs(point["presion_mpa"] - REPORT_PRESSURE_MPA) > 1e-6:
                continue
            area = poro["accessible_area_m2_g"]
            excess = excess_weight_percent(point, poro)
            out.append({
                "nombre": name,
                "area_m2_g": area,
                "esfera_libre_a": poro["largest_free_sphere"],
                "vacio_helio": poro["helium_void_fraction"],
                "wt_absoluto": point["wt_pct_total"],
                "wt_exceso": excess,
                "razon_chahine": excess / (area / CHAHINE_M2_G_PER_WT),
                "avisos": point["avisos"],
            })
    return out


def main() -> None:
    """Print the comparison, narrowest pore first."""
    data = sorted(rows(), key=lambda r: r["esfera_libre_a"])
    if not data:
        print("No hay ninguna isoterma medida todavía.")
        return
    print(f"Adsorción de H2 a 77 K y {REPORT_PRESSURE_MPA:g} MPa\n")
    head = (f"{'marco':16s} {'área':>7s} {'libre':>7s} {'He':>6s} "
            f"{'wt abs':>7s} {'wt exc':>7s} {'/Chahine':>9s}")
    print(head)
    print("-" * len(head))
    for r in data:
        print(f"{r['nombre']:16s} {r['area_m2_g']:7.0f} {r['esfera_libre_a']:7.2f} "
              f"{r['vacio_helio']:6.3f} {r['wt_absoluto']:7.2f} "
              f"{r['wt_exceso']:7.2f} {r['razon_chahine']:9.2f}")
    flagged = [r for r in data if r["avisos"]]
    if flagged:
        print("\nPuntos con avisos del muestreo (no citar sin repetirlos):")
        for r in flagged:
            for warning in r["avisos"]:
                print(f"  {r['nombre']}: {warning}")


if __name__ == "__main__":
    main()
