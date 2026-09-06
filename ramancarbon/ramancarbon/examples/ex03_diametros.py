"""Diámetros por RBM, quirialidades compatibles y emparejamiento de paredes.

    python -m ramancarbon.examples.ex03_diametros
"""

from __future__ import annotations

from ramancarbon.analysis.diameter import (
    assign_chirality,
    compare_parameterisations,
    find_wall_pairs,
    rbm_diameter_with_spread,
)


def main() -> None:
    omega = 158.0
    print(f"RBM a {omega:g} cm⁻¹ según cada parametrización de la literatura:")
    for key, estimate in compare_parameterisations(omega).items():
        print(f"  {key:24s} d = {estimate.diameter_nm:.3f} nm")
    print("\nEsa dispersión es la incertidumbre sistemática que se esconde")
    print("detrás de un único diámetro citado sin decir qué relación se usó.\n")

    estimate = rbm_diameter_with_spread(omega)
    print(f"Valor elegido: {estimate}")
    print(f"Fuente: {estimate.source}\n")

    print(f"(n,m) compatibles con d = {estimate.diameter_nm:.3f} nm:")
    for candidate in assign_chirality(estimate.diameter_nm)[:8]:
        print(f"  {candidate}")
    print("\nEl diámetro por sí solo NO fija (n,m): hace falta además la")
    print("condición de resonancia, que requiere medir con varios láseres.\n")

    print("¿Son paredes concéntricas? (prueba de doble pared)")
    for pair in find_wall_pairs([158.0, 178.0, 265.0, 291.0])[:4]:
        print(f"  {pair}")


if __name__ == "__main__":
    main()
