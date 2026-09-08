"""Electroquímica: por qué el mecanismo va antes que la cifra.

    python -m ramancarbon.examples.ex13_echem
"""

from __future__ import annotations

from ramancarbon.echem.report import analyse_sample
from ramancarbon.examples.demo_data import (
    ECHEM_DEMOS,
    cv_rate_series,
    make_cv_demo,
    make_gcd_demo,
)


def main() -> None:
    print("Los tres electrodos de abajo dan capacitancias parecidas. Solo uno")
    print("de ellos debería informarse en faradios.\n")

    print(f"{'electrodo':>20s} {'mecanismo':>18s} {'C (F/g)':>10s} "
          f"{'q (mAh/g)':>11s}  informar como")
    print("─" * 88)
    for kind in ECHEM_DEMOS:
        result = analyse_sample(
            kind,
            cv=make_cv_demo(kind, seed=1),
            rate_series=cv_rate_series(kind, seed=2),
            gcd=make_gcd_demo(kind, seed=1),
        )
        capacitance = result.gcd.specific_f_per_g or 0.0
        capacity = result.gcd.capacity_mah_per_g or 0.0
        print(
            f"{kind:>20s} {result.storage.mechanism:>18s} {capacitance:10.1f} "
            f"{capacity:11.2f}  {result.storage.report_as}"
        )

    print()
    battery = analyse_sample(
        "batería",
        cv=make_cv_demo("bateria", seed=1),
        rate_series=cv_rate_series("bateria", seed=2),
        gcd=make_gcd_demo("bateria", seed=1),
    )
    print(battery.storage.summary())

    print("\n" + "─" * 68)
    print("Y la comprobación cruzada que solo aparece con dos medidas:\n")
    for warning in battery.warnings:
        if "veces la de la curva galvanostática" in warning:
            print("  ⚠ " + warning)

    print("\n" + "─" * 68)
    print("La b que importa es la del PICO, no la mediana de la ventana:\n")
    rates = battery.rates
    for value in rates.b_values:
        print(f"  {value}")
    print("\n  Un electrodo de batería está limitado por difusión solo donde")
    print("  ocurre su proceso redox; en el resto la doble capa mantiene b≈1.")


if __name__ == "__main__":
    main()
