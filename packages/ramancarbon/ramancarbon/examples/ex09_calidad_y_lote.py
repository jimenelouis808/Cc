"""Comprobar la calidad de la medida, y analizar un lote con estadística.

Dos fallos que no se ven en el informe final si nadie los busca.

    python -m ramancarbon.examples.ex09_calidad_y_lote
"""

from __future__ import annotations

import numpy as np

from ramancarbon.analysis.batch import analyse_many, summarise
from ramancarbon.analysis.quality import check_quality
from ramancarbon.analysis.report import analyse
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.examples.demo_data import make_demo


def main() -> None:
    print("═══ 1. Detector saturado ═══\n")
    good = make_demo("MWCNT", seed=1)
    ceiling = float(np.percentile(good.intensity, 97))
    clipped = Spectrum(
        good.shift, np.minimum(good.intensity, ceiling), laser_nm=532.0,
        name="saturado",
    )
    print(f"I_D/I_G sin saturar : {analyse(good).id_ig:.3f}")
    print(f"I_D/I_G saturado    : {analyse(clipped).id_ig:.3f}")
    print("\nEl mismo material, el doble de cociente. La saturación aplana")
    print("primero la banda más intensa, que es la G, así que infla todo lo")
    print("que se mide contra ella. Y en el informe no se nota.\n")
    print(check_quality(clipped).summary())

    print("\n\n═══ 2. Muestreo demasiado grueso ═══\n")
    for step in (1.0, 4.0, 16.0):
        result = analyse(make_demo("MWCNT", seed=1, step=step))
        print(f"  paso {step:5.1f} cm⁻¹  →  I_D/I_G = {result.id_ig:.3f}")
    print("\n" + check_quality(make_demo("MWCNT", seed=1, step=16.0)).summary())

    print("\n\n═══ 3. Un mapa heterogéneo ═══\n")
    rng = np.random.default_rng(3)
    spectra = []
    for i in range(20):
        s = make_demo("MWCNT", seed=i)
        s.name = f"punto{i:02d}"
        # dos puntos sobre algo distinto
        if i in (5, 13):
            s.intensity[(s.shift > 1300) & (s.shift < 1400)] *= 2.4
        else:
            s.intensity[(s.shift > 1300) & (s.shift < 1400)] *= 1 + rng.normal(0, 0.04)
        spectra.append(s)

    results, failures = analyse_many(spectra, workers=1)
    print(summarise(results, failures=failures).summary())
    print("\nLa media y la mediana de I_D/I_G no coinciden, y los dos puntos")
    print("que las separan están identificados por número. Publicar la media")
    print("sin mirarlos sería publicar el promedio de dos cosas distintas.")


if __name__ == "__main__":
    main()
