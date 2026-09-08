"""Tamaño, deformación y celda: lo que un difractograma dice de más.

    python -m ramancarbon.examples.ex17_microestructura

Un pico ensanchado no distingue por sí solo entre cristalitos pequeños y
red deformada: hacen falta varias reflexiones, porque las dos cosas
dependen del ángulo de forma distinta. Y para refinar una celda no hace
falta conocer la estructura.
"""

from __future__ import annotations

import math

from ramancarbon.examples.demo_data import make_xrd_demo
from ramancarbon.xrd.lebail import cell_from_extraction, le_bail, pawley
from ramancarbon.xrd.microstructure import (
    agreement,
    amorphous_fraction,
    carbon_microstructure,
    compare_methods,
)
from ramancarbon.xrd.reference import find_phase

CU = 1.540598


def anchuras(angulos, tamano_nm, deformacion, k=0.89):
    """β = Kλ/(D cos θ) + 4ε tan θ, en grados."""
    salida = []
    for angulo in angulos:
        theta = math.radians(angulo) / 2.0
        beta = (0.1 * k * CU / (tamano_nm * math.cos(theta))
                + 4.0 * deformacion * math.tan(theta))
        salida.append(math.degrees(beta))
    return salida


def main() -> None:
    angulos = [20.0, 30.0, 45.0, 60.0, 75.0, 90.0]

    print("1. Un solo pico no separa tamaño de deformación\n")
    for tamano, epsilon in ((25.0, 0.0), (25.0, 0.002), (60.0, 0.004)):
        w = anchuras(angulos, tamano, epsilon)
        print(f"   D = {tamano:4.0f} nm, ε = {100 * epsilon:.1f} %  →  "
              f"anchuras {w[0]:.3f}° a 20° y {w[-1]:.3f}° a 90°")
    print("   Las tres muestras tienen anchuras parecidas a ángulo bajo y")
    print("   muy distintas a ángulo alto. Ahí está la separación.\n")

    print("2. Los tres métodos, sobre una muestra con las dos cosas")
    w = anchuras(angulos, 25.0, 0.002)
    resultados = compare_methods(angulos, w, CU)
    for resultado in resultados:
        print(f"   {resultado.describe()}")
    print(f"   → {agreement(resultados)}")
    print("   Los datos se generaron con D = 25 nm y ε = 0.20 %: el primero")
    print("   los recupera porque es su propio modelo; los otros dos suponen")
    print("   otra convolución y por eso difieren. La diferencia ES el dato.\n")

    print("3. Un carbono, con sus propias medidas")
    for espaciado, nombre in ((3.354, "grafito"), (3.39, "parcialmente grafitizado"),
                              (3.47, "turbostrático")):
        dos_theta = 2.0 * math.degrees(math.asin(CU / (2.0 * espaciado)))
        carbono = carbon_microstructure(dos_theta, 1.2, 42.4, 2.0, wavelength=CU)
        print(f"   {nombre:26s} {carbono.describe()}")
        for aviso in carbono.warnings[:1]:
            print(f"       · {aviso[:70]}…")
    print()

    print("4. Refinar la celda sin saber la estructura")
    patron = make_xrd_demo("MoS2_texturado")
    cristal = find_phase("MoS2_2H")
    mal = cristal.with_lattice(cristal.lattice.scaled((1.02, 1.02, 0.985)))
    print(f"   celda de partida (2 % mal): a = {mal.lattice.a:.4f} Å, "
          f"c = {mal.lattice.c:.4f} Å")
    for metodo, nombre in ((le_bail, "Le Bail"), (pawley, "Pawley")):
        resultado = metodo(patron, mal, cycles=15)
        celda = cell_from_extraction(resultado, mal)
        print(f"   {nombre:8s} a = {celda['a']:.5f} Å, c = {celda['c']:.5f} Å   "
              f"{resultado.describe()}")
    print(f"   verdad     a = {cristal.lattice.a:.5f} Å, "
          f"c = {cristal.lattice.c:.5f} Å")
    print("   Sin ningún átomo: solo la celda, la simetría y el perfil.\n")

    print("5. Lo que Rietveld no puede ver")
    refinado = {"muestra": 0.6, "patron": 0.4}
    resultado = amorphous_fraction(refinado, "patron", standard_added=0.25)
    print(f"   refinado: {refinado}, con un 25 % de patrón añadido")
    print(f"   → amorfo: {100 * resultado['amorfo']:.0f} % de la muestra")
    print("   Las fracciones de un Rietveld suman 100 % siempre, porque son")
    print("   de la parte cristalina modelada. El amorfo solo se mide")
    print("   añadiendo una cantidad conocida de algo cristalino y viendo")
    print("   cuánto cree el refinamiento que hay.")


if __name__ == "__main__":
    main()
