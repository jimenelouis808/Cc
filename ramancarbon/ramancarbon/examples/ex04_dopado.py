"""Detectar dopado comparando contra un control, y separarlo de la deformación.

El punto importante: la banda G SUBE tanto con electrones como con huecos,
así que ΔG por sí sola no dice el signo del dopado. Solo la banda 2D lo
distingue — sube con huecos y baja con electrones.

    python -m ramancarbon.examples.ex04_dopado
"""

from __future__ import annotations

from ramancarbon import analyse
from ramancarbon.analysis.shifts import decompose_strain_doping
from ramancarbon.examples.demo_data import add_doping, make_demo


def main() -> None:
    pristine = make_demo("grafeno_1L", seed=2)

    # Dopado tipo n sintético: G arriba, 2D abajo.
    doped = add_doping(pristine, delta_g=8.0, delta_2d=-12.0, extra_disorder=0.4)

    control = analyse(pristine)
    result = analyse(doped, control=control)

    print("Contra el control medido en la misma sesión:")
    print(result.shifts.summary())

    print("\n\nY el mismo cálculo a mano, para ver la descomposición sola:")
    for label, (dg, d2d) in {
        "deformación pura": (-10.0, -22.0),
        "dopado puro (huecos)": (10.0, 7.0),
        "mezcla": (5.0, -10.0),
    }.items():
        print(f"\n{label}:")
        print(decompose_strain_doping(dg, d2d).summary())


if __name__ == "__main__":
    main()
