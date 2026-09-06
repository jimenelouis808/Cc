"""Consultar la base de datos de literatura y ver el efecto del láser.

Todos los números que este programa usa vienen de archivos JSON editables
con su fuente al lado. Este ejemplo los recorre y muestra por qué la
corrección por dispersión no es opcional.

    python -m ramancarbon.examples.ex06_base_datos
"""

from __future__ import annotations

from ramancarbon.core.spectrum import laser_energy_ev
from ramancarbon.database import load_database


def main() -> None:
    db = load_database()
    print(db.summary(), "\n")

    lasers = (325.0, 488.0, 532.0, 633.0, 785.0, 1064.0)
    keys = ("D", "G", "D'", "2D", "D+D'")
    print("Posición de cada banda según el láser (cm⁻¹):\n")
    print("banda   " + "".join(f"{v:>9g}nm" for v in lasers))
    for key in keys:
        band = db.band(key)
        row = "".join(
            f"{band.position_at(laser_energy_ev(v)):>11.1f}" for v in lasers
        )
        print(f"{key:<8s}{row}")

    print("\nLa D se mueve ~50 cm⁻¹/eV y la 2D el doble, porque es su")
    print("sobretono. Comparar posiciones medidas con láseres distintos sin")
    print("corregir esto produce «desplazamientos» que son pura aritmética.\n")

    print("Tipos de defecto según I_D/I_D':")
    for key, value in db.defect_type_ratios.items():
        if isinstance(value, (int, float)) and not key.startswith("_"):
            print(f"  {key:<26s} {value}")
    print(f"  fuente: {db.defect_type_ratios['source']}")


if __name__ == "__main__":
    main()
