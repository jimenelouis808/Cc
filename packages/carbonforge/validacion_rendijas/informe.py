"""The slit series, and everything else measured, against pore width.

This is the series the rest of the study could not provide. In the six
frameworks measured before it -- three schwarzites and three
superstructures -- area, pore width and curvature all moved together, so
no difference between them could be assigned to any one of the three. A
graphene slit holds the chemistry and the curvature fixed, and its
accessible area is constant across the whole series (2670 m2/g, which is
graphene's own textbook value to 1.5%), so width is the only variable
left.

The width quoted is the FREE one -- the largest sphere that can cross the
cell, as `analyse_porosity` measures it -- not the centre-to-centre
separation of the carbon planes, which is larger by about a carbon
diameter. The free width is what the other frameworks are described by,
so it is the only one that puts them on the same axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CHAHINE_M2_G_PER_WT = 500.0

HERE = Path(__file__).resolve().parent
SCHWARZITES = HERE.parent / "validacion_schwarzitas"
SUPERS = HERE.parent / "validacion_superestructuras"


def excess_weight_percent(point: dict, porosity: dict) -> float:
    """Excess as a percentage of total mass, from the excess mass."""
    volume_cm3 = porosity["cell_volume_a3"] * 1e-24
    excess = point["densidad_exceso_g_cm3"] * volume_cm3
    framework = porosity["framework_mass_g"]
    return 100.0 * excess / (excess + framework)


def read_folder(path: Path, label: str) -> list[dict]:
    """Every pressure of one isotherm, with its porosity carried along."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    porosity = data["porosidad"]
    out = []
    for point in data["puntos"]:
        out.append({
            "marco": label,
            "canal_a": porosity["largest_free_sphere"],
            "area_m2_g": porosity["accessible_area_m2_g"],
            "vacio_helio": porosity["helium_void_fraction"],
            "presion_mpa": point["presion_mpa"],
            "wt_absoluto": point["wt_pct_total"],
            "wt_exceso": excess_weight_percent(point, porosity),
            "n_medio": point["n_medio"],
            "avisos": point["avisos"],
        })
    return out


def collect(slits: Path, others: list[tuple[str, Path]]) -> list[dict]:
    """The slit series plus whatever other frameworks are given."""
    rows: list[dict] = []
    for path in sorted(slits.glob("rendija_*.json"),
                       key=lambda p: float(p.stem.split("_")[1])):
        rows += read_folder(path, f"rendija {float(path.stem.split('_')[1]):g} A")
    for folder in sorted(slits.glob("iso_*"),
                         key=lambda p: float(p.name.split("_")[1])):
        separation = float(folder.name.split("_")[1])
        rows += read_folder(folder / "resultados.json", f"rendija {separation:g} A")
    for label, path in others:
        rows += read_folder(path, label)
    return rows


def table(rows: list[dict], pressure: float) -> None:
    """One line per framework at one pressure, narrowest channel first."""
    chosen = [r for r in rows if abs(r["presion_mpa"] - pressure) < 1e-9]
    if not chosen:
        print(f"  (ningún punto a {pressure:g} MPa todavía)")
        return
    print(f"\n{pressure:g} MPa")
    head = (f"  {'marco':16s} {'canal':>7s} {'área':>7s} {'He':>6s} "
            f"{'N':>8s} {'wt abs':>7s} {'wt exc':>7s} {'/Chahine':>9s}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in sorted(chosen, key=lambda r: r["canal_a"]):
        ratio = r["wt_exceso"] / (r["area_m2_g"] / CHAHINE_M2_G_PER_WT)
        flag = "  <- aviso" if r["avisos"] else ""
        print(f"  {r['marco']:16s} {r['canal_a']:7.2f} {r['area_m2_g']:7.0f} "
              f"{r['vacio_helio']:6.3f} {r['n_medio']:8.1f} "
              f"{r['wt_absoluto']:7.2f} {r['wt_exceso']:7.2f} {ratio:9.2f}{flag}")


def main() -> int:
    """Print the comparison at each pressure the series holds."""
    slits = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    others = [
        ("Schwarz D", SCHWARZITES / "diamond.json"),
        ("Schwarz P", SCHWARZITES / "primitive.json"),
        ("giroide", SCHWARZITES / "gyroid.json"),
        ("súper-grafeno", SUPERS / "supergrafeno.json"),
        ("superfulereno", SUPERS / "superfulereno.json"),
        ("supertubo", SUPERS / "supertubo.json"),
    ]
    rows = collect(slits, others)
    print("Adsorción de H2 a 77 K, ordenado por ancho de canal libre")
    for pressure in (1.0, 3.8, 10.0):
        table(rows, pressure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
