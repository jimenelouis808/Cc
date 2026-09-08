"""Difracción: identificar fases y refinar por Rietveld.

    python -m ramancarbon.examples.ex12_drx
"""

from __future__ import annotations

from ramancarbon.examples.demo_data import make_xrd_demo
from ramancarbon.xrd.reference import find_phase
from ramancarbon.xrd.report import analyse_pattern


def main() -> None:
    print("Una fase es una ESTRUCTURA, no una lista de picos. Cada posición")
    print("e intensidad sale del factor de estructura, así que refinar la")
    print("celda mueve los picos como los movería la física.\n")

    pattern = make_xrd_demo("CNT_FeSe", seed=3)
    print(pattern.describe())
    print(f"Fases que se metieron: {pattern.metadata['phases']}\n")

    result = analyse_pattern(pattern)
    for match in result.search.accepted:
        print(f"  {match.summary().splitlines()[0]}")

    refinement = result.refinement
    print(f"\nRwp = {100 * refinement.r_wp:.2f} %   GOF = {refinement.gof:.3f}")
    print("\nCelda refinada frente a la de referencia:")
    for phase in refinement.phases:
        reference = find_phase(phase.crystal.name).lattice
        refined = phase.current_crystal().lattice
        print(
            f"  {phase.crystal.name:18s} a: {reference.a:.5f} → {refined.a:.5f} Å"
            f"   c: {reference.c:.5f} → {refined.c:.5f} Å"
        )

    print("\nFracciones en peso (de la parte CRISTALINA E IDENTIFICADA):")
    for name, fraction in refinement.weight_fractions().items():
        print(f"  {name:18s} {100 * fraction:5.1f} %")

    print("\n" + "─" * 68)
    print("Y lo que Raman deja abierto, esto lo cierra:\n")
    two_phases = make_xrd_demo("FeSe_dos_fases", seed=2)
    found = analyse_pattern(two_phases, refine=False)
    for match in found.search.accepted:
        crystal = match.crystal
        lines = crystal.notes.split(".")[0]
        print(f"  {crystal.name:18s} {crystal.space_group:10s} — {lines}")
    print("\n  El 101 del tetragonal está en 28.6° y el del hexagonal en 32.2°:")
    print("  catorce veces la anchura de un pico. No admite discusión.")


if __name__ == "__main__":
    main()
