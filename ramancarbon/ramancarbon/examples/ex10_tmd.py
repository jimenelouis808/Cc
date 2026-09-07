"""Dicalcogenuros: contar capas por la separación E₂g–A₁g.

    python -m ramancarbon.examples.ex10_tmd
"""

from __future__ import annotations

from ramancarbon.analysis.tmd import analyse_tmd, tmd_materials
from ramancarbon.examples.demo_data import make_tmd_demo


def main() -> None:
    print("Al apilar capas, el modo E₂g (en el plano) se ABLANDA y el A₁g")
    print("(fuera del plano) se ENDURECE. La separación crece de forma")
    print("monótona, y al ser una diferencia cualquier error común de")
    print("calibración se cancela.\n")

    print(f"{'capas':>8s} {'E₂g':>9s} {'A₁g':>9s} {'separación':>12s} {'deducido':>10s}")
    for layers in ("1", "2", "3", "bulk"):
        result = analyse_tmd(make_tmd_demo("MoS2", layers, seed=1))
        print(
            f"{layers:>8s} {result.positions['E2g']:9.2f} "
            f"{result.positions['A1g']:9.2f} {result.separation:12.2f} "
            f"{str(result.layers):>10s}"
        )

    print("\n\nDonde el método NO funciona:\n")
    for material in tmd_materials():
        if material.counts_layers_by_separation:
            continue
        print(f"  {material.label}")
        print(f"    {material.notes}\n")

    result = analyse_tmd(make_tmd_demo("WSe2", "2", seed=2))
    print(f"  WSe₂ bicapa → {result.layers}: {result.layer_reason}")

    print("\n\nFase metálica 1T′:\n")
    for flag in (False, True):
        result = analyse_tmd(make_tmd_demo("MoS2", "1", seed=3, phase_1t=flag))
        print(f"  {'con' if flag else 'sin'} modos J → fase {result.phase}")
        print(f"    {result.phase_reason}\n")


if __name__ == "__main__":
    main()
