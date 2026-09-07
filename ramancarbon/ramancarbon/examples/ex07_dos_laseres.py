"""Combinar el mismo material medido a 532 y 633 nm.

Tres cosas que con un solo láser no se pueden hacer.

    python -m ramancarbon.examples.ex07_dos_laseres
"""

from __future__ import annotations

from ramancarbon import analyse
from ramancarbon.analysis.multiwavelength import compare_excitations
from ramancarbon.examples.demo_data import make_demo


def main() -> None:
    results = [
        analyse(make_demo("MWCNT", laser_nm=laser, seed=1))
        for laser in (532.0, 633.0)
    ]

    print("Lo que ve cada láser por separado:\n")
    for result in results:
        print(f"  {result.raw.laser_nm:g} nm:  I_D/I_G = {result.id_ig:.3f}   "
              f"D en {result.assignment.position('D'):.1f} cm⁻¹")
    print(
        "\nEl cociente cambia un factor 2 entre los dos, y no porque la muestra\n"
        "sea distinta: I_D/I_G escala como λ⁴. Comparar valores medidos a\n"
        "láseres distintos sin corregir eso no significa nada.\n"
    )

    print(compare_excitations(results).summary())


if __name__ == "__main__":
    main()
