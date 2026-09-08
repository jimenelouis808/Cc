"""Controles de programa especializado: ligaduras, exclusiones, anclas.

    python -m ramancarbon.examples.ex19_controles
"""

from __future__ import annotations

import numpy as np

from ramancarbon.core.baseline import anchor_baseline, suggest_anchors
from ramancarbon.core.history import History, Preferences
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.models.fitting import FitModel, PeakSpec, fit_model

X = np.arange(1100.0, 1800.0, 1.0)


def lorentziana(centro, altura, anchura):
    media = anchura / 2.0
    return altura * media ** 2 / ((X - centro) ** 2 + media ** 2)


def espectro(con_rayo: bool = False) -> Spectrum:
    y = (lorentziana(1350.0, 300.0, 60.0) + lorentziana(1580.0, 500.0, 45.0)
         + lorentziana(1620.0, 120.0, 45.0) + 20.0)
    y = y + np.random.default_rng(0).normal(0.0, 2.0, X.size)
    if con_rayo:
        y[np.abs(X - 1450.0) < 4] += 4000.0
    return Spectrum(shift=X, intensity=y, laser_nm=532.0, name="muestra")


def modelo(**extra) -> FitModel:
    return FitModel(
        peaks=[PeakSpec("D", centre=1350.0, height=300.0, fwhm=60.0),
               PeakSpec("G", centre=1580.0, height=500.0, fwhm=45.0),
               PeakSpec("D'", centre=1620.0, height=120.0, fwhm=50.0)],
        window=(1150.0, 1750.0), **extra,
    )


def main() -> None:
    muestra = espectro()

    print("1. Ligar un parámetro a otro\n")
    libre = fit_model(muestra, modelo())
    ligado = fit_model(muestra, modelo(links=("D'.fwhm = G.fwhm",)))
    for nombre, r in (("libre", libre), ("ligado", ligado)):
        anchuras = "  ".join(f"{p.name}:{p.fwhm:5.1f}" for p in r.peaks)
        print(f"   {nombre:7s} {anchuras}   {r.n_parameters} parámetros, "
              f"R² = {r.r_squared:.5f}")
    print("   Ligar no es un atajo: quita un parámetro libre, así que las")
    print("   incertidumbres bajan. Eso solo vale si la ligadura es física,")
    print("   y por eso va escrita en el informe:")
    print(f"   → {[linea for linea in ligado.summary().splitlines() if 'Ligaduras' in linea][0]}\n")

    desplazado = fit_model(muestra, modelo(links=("D'.centre = G.centre + 40",)))
    separacion = desplazado.peak("D'").centre - desplazado.peak("G").centre
    print(f"   Y con desplazamiento fijo: D' − G = {separacion:.6f} cm⁻¹\n")

    print("2. Excluir un trozo del ajuste\n")
    sucia = espectro(con_rayo=True)
    estropeado = fit_model(sucia, modelo())
    limpio = fit_model(sucia, modelo(), exclude=[(1440.0, 1460.0)])
    print(f"   con el rayo dentro: D en {estropeado.peak('D').centre:7.1f} cm⁻¹, "
          f"R² = {estropeado.r_squared:.4f}")
    print(f"   excluido          : D en {limpio.peak('D').centre:7.1f} cm⁻¹, "
          f"R² = {limpio.r_squared:.4f}")
    dentro = np.abs(limpio.x - 1450.0) < 4
    print(f"   El residuo SIGUE cubriendo lo excluido "
          f"(máximo {np.max(np.abs(limpio.residual[dentro])):.0f}): es lo que")
    print("   permite ver si la exclusión estaba justificada. Y queda escrita:")
    print(f"   → {[linea for linea in limpio.summary().splitlines() if 'Excluido' in linea][0]}\n")

    print("3. Línea base a mano\n")
    from ramancarbon.examples.demo_data import make_demo

    con_fondo = make_demo("MWCNT", fluorescence=700.0, seed=1)
    anclas = suggest_anchors(con_fondo.shift, con_fondo.intensity, 8)
    print(f"   anclas sugeridas: {[round(a) for a in anclas]}")
    for tipo in ("pchip", "spline", "linear"):
        base = anchor_baseline(con_fondo.shift, con_fondo.intensity, anclas,
                               kind=tipo)
        corregido = con_fondo.intensity - base
        print(f"   {tipo:7s} → G a {corregido[int(np.argmin(np.abs(con_fondo.shift - 1580)))]:.0f}, "
              f"residuo mínimo {corregido.min():.0f}")
    print("   El valor de cada ancla es una mediana local con la pendiente")
    print("   quitada: exacta sobre un fondo inclinado e inmune a una púa.\n")

    print("4. Deshacer y preferencias\n")
    historia = History({"base": "asls", "normalizar": "no"})
    historia.push({"base": "snip", "normalizar": "no"}, "cambiar a SNIP")
    historia.push({"base": "snip", "normalizar": "max"}, "normalizar al máximo")
    print(f"   {len(historia)} estados: {historia.labels()}")
    print(f"   deshacer → {historia.undo()}")
    print(f"   deshacer → {historia.undo()}")
    print(f"   rehacer  → {historia.redo()}")

    preferencias = Preferences()
    preferencias.set("laser_nm", 633.0)
    preferencias.set("plot_preset", "acs")
    print(f"\n   preferencias: láser {preferencias.get('laser_nm'):g} nm, "
          f"figura «{preferencias.get('plot_preset')}»")
    print(f"   se guardarían en {preferencias.path or 'el directorio de configuración'}")
    print("   Un archivo ilegible no impide arrancar: se usan las de fábrica")
    print("   y se dice por qué.")


if __name__ == "__main__":
    main()
