"""Un mapa Raman: de 480 espectros a una imagen que dice algo.

    python -m ramancarbon.examples.ex16_mapas

El mapa de demostración lleva dentro, a propósito, todo lo que hace
difícil un mapa de verdad: un gradiente de fluorescencia sobre la muestra,
rayos cósmicos, píxeles perdidos y una frontera que no está alineada con
la rejilla del barrido.
"""

from __future__ import annotations

import numpy as np

from ramancarbon.examples.demo_data import make_map_demo
from ramancarbon.mapping import (
    band_intensity,
    band_position,
    band_ratio,
    coverage,
    despike_map,
    kmeans,
    mcr_als,
    pca,
    suggested_components,
)


def main() -> None:
    mapa = make_map_demo("dos_fases", seed=1)
    print(mapa.describe())
    aviso = mapa.sampling_warning()
    if aviso:
        print(f"  aviso: {aviso}")
    print()

    print("1. Rayos cósmicos, antes que nada")
    limpio, canales = despike_map(mapa)
    print(f"   {canales} canales sustituidos de {mapa.intensity.size}.")
    print("   Un pico de un canal en un píxel es la mayor excursión del cubo")
    print("   entero: sin quitarlo sale como primera componente principal.\n")

    print("2. Lo que se integra, con y sin línea base local")
    con = band_intensity(limpio, 1500, 1660)
    sin = band_intensity(limpio, 1500, 1660, baseline=False)
    print(f"   con línea base : {np.nanmedian(con.finite):9.0f} cuentas·cm⁻¹")
    print(f"   sin línea base : {np.nanmedian(sin.finite):9.0f}")
    izq, der = np.nanmean(sin.values[:, :5]), np.nanmean(sin.values[:, -5:])
    print(f"   Sin corregir, el mapa crece {100 * (der - izq) / izq:.0f} % de")
    print("   izquierda a derecha siguiendo la fluorescencia, no la muestra.\n")

    print("3. Los mapas que se publican")
    razon = band_ratio(limpio, (1280, 1420), (1500, 1660))
    print(f"   {razon.describe()}")
    print(f"     izquierda {np.nanmean(razon.values[:, :5]):.2f}   "
          f"derecha {np.nanmean(razon.values[:, -5:]):.2f}")
    posicion = band_position(limpio, 1500, 1660)
    print(f"   {posicion.describe()}")
    print(f"     izquierda {np.nanmean(posicion.values[:, :5]):.1f} cm⁻¹   "
          f"derecha {np.nanmean(posicion.values[:, -5:]):.1f} cm⁻¹")
    print(f"     {len(np.unique(np.round(posicion.finite, 3)))} valores "
          f"distintos con un paso espectral de "
          f"{np.median(np.diff(limpio.shift)):.0f} cm⁻¹: la interpolación")
    print("     parabólica evita que el mapa salga en terrazas.\n")

    print("4. Cuántas cosas hay aquí")
    componentes = pca(limpio, 6)
    print(f"   {componentes.describe()}")
    print("   varianza por componente: "
          + ", ".join(f"{100 * v:.1f} %" for v in componentes.explained[:5]))
    print(f"   → sugeridas: {suggested_components(componentes)}")
    print("   Una componente principal es una dirección de varianza, no una")
    print("   sustancia. Para leerla como espectro hace falta MCR.\n")

    print("5. Qué píxeles se parecen")
    grupos = kmeans(limpio, k=2, seed=0)
    print(f"   {grupos.describe()}")
    for aviso in grupos.warnings:
        print(f"     · {aviso[:72]}…")
    etiquetas = grupos.labels
    print(f"     izquierda: grupo {np.bincount(etiquetas[:, :6][etiquetas[:, :6] >= 0]).argmax()}"
          f"   derecha: grupo {np.bincount(etiquetas[:, -6:][etiquetas[:, -6:] >= 0]).argmax()}\n")

    print("6. Qué espectros suman esto")
    resuelto = mcr_als(limpio, 2)
    print(f"   {resuelto.describe()}")
    for indice in range(2):
        espectro = resuelto.component_spectrum(indice)
        maximo = espectro.shift[int(np.argmax(espectro.intensity))]
        izquierda = np.nanmean(resuelto.scores[:, :6, indice])
        derecha = np.nanmean(resuelto.scores[:, -6:, indice])
        print(f"   componente {indice + 1}: máximo en {maximo:.0f} cm⁻¹, "
              f"peso izq {izquierda:.0f} / der {derecha:.0f}")
    print("   Las dos se reparten el mapa, que es lo que se puso.")
    for aviso in resuelto.warnings[:2]:
        print(f"     · {aviso[:72]}…")

    print("\n7. Y el caso en que casi todo el mapa es nada")
    hojuela = despike_map(make_map_demo("hojuela", seed=2))[0]
    mascara, fraccion = coverage(hojuela, 1500, 1660)
    print(f"   {mascara.label}: {100 * fraccion:.0f} % del mapa")
    razon = band_ratio(hojuela, (1280, 1420), (1500, 1660))
    print(f"   I_D/I_G enmascarado en {razon.masked} de {hojuela.n_pixels} "
          "píxeles")
    print("   Sin enmascarar, los píxeles más brillantes del mapa de cocientes")
    print("   estarían todos en la parte donde no hay muestra.")


if __name__ == "__main__":
    main()
