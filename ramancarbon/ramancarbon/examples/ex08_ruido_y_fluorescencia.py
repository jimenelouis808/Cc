"""Preprocesar un espectro ruidoso y muy fluorescente, automático y a mano.

    python -m ramancarbon.examples.ex08_ruido_y_fluorescencia
"""

from __future__ import annotations

import numpy as np

from ramancarbon.core.baseline import cutoff_for_lambda, subtract_baseline
from ramancarbon.core.preprocess import auto_settings, normalise, preprocess
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.examples.demo_data import make_demo


def main() -> None:
    # Una muestra fea: mucho ruido y una cola de fluorescencia enorme.
    clean = make_demo("MWCNT", seed=4, noise=1.0, fluorescence=0.0)
    rng = np.random.default_rng(0)
    x = clean.shift
    background = 4000.0 * np.exp(-(x - x[0]) / 700.0) + 60.0
    dirty = Spectrum(
        x,
        clean.intensity + background + rng.normal(0.0, 45.0, x.size),
        laser_nm=532.0,
        name="feo",
    )

    print("Espectro de partida: ruido σ ≈ 45, fondo de 4000 cuentas.\n")

    chosen = auto_settings(dirty)
    print(chosen.summary())
    print(
        f"\nEse λ corresponde a un corte de "
        f"{cutoff_for_lambda(chosen.baseline_lam) * dirty.step:.0f} cm⁻¹: la línea "
        "base sigue todo lo más lento que eso y nada más rápido.\n"
    )

    processed, _ = preprocess(dirty, **chosen.to_kwargs())

    truth = clean.max_in(1300, 1400)[1] / clean.max_in(1540, 1630)[1]
    recovered = processed.max_in(1300, 1400)[1] / processed.max_in(1540, 1630)[1]
    print(f"I_D/I_G verdadero  : {truth:.3f}")
    print(f"I_D/I_G recuperado : {recovered:.3f}")
    print(f"error              : {abs(recovered - truth) / truth * 100:.1f} %\n")

    print("Sensibilidad del cociente a la rigidez, sobre estos mismos datos:")
    for factor in (100.0, 10.0, 1.0, 0.1, 0.01):
        lam = chosen.baseline_lam * factor
        corrected, _ = subtract_baseline(dirty, method="asls", lam=lam)
        value = corrected.max_in(1300, 1400)[1] / corrected.max_in(1540, 1630)[1]
        print(f"  λ = {lam:8.1e}  →  I_D/I_G = {value:.3f}")
    print(
        "\nDos órdenes de magnitud arriba y abajo de λ mueven el cociente muy\n"
        "poco. No es casualidad ni suerte: el error de una línea base suave es\n"
        "casi constante a lo largo de la ventana 1300–1600 cm⁻¹ y se cancela en\n"
        "gran medida al dividir. Las ÁREAS ABSOLUTAS no tienen esa suerte y sí\n"
        "dependen mucho del fondo — razón para publicar cocientes y no áreas\n"
        "cuando la fluorescencia es fuerte.\n"
    )

    scaled = normalise(processed, "0-100")
    print(f"\nNormalizado 0–100: {scaled.intensity.min():.1f} … "
          f"{scaled.intensity.max():.1f}")
    print(f"  {scaled.history[-1]}")


if __name__ == "__main__":
    main()
