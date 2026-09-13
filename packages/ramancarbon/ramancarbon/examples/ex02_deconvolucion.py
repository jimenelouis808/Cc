"""Deconvolución de la región D–G y elección del número de componentes.

Muestra el punto central de la deconvolución: añadir componentes SIEMPRE
mejora el R², así que el R² no puede decidir cuántas hacen falta. El
criterio de información sí, porque penaliza cada parámetro extra.

    python -m ramancarbon.examples.ex02_deconvolucion
"""

from __future__ import annotations

from ramancarbon.core.preprocess import preprocess
from ramancarbon.examples.demo_data import make_demo
from ramancarbon.models.deconvolution import build_model, compare_models
from ramancarbon.models.fitting import fit_model


def main() -> None:
    spectrum, _ = preprocess(make_demo("MWCNT", seed=1))

    print("R² de cada modelo (fíjate en que nunca baja al añadir componentes):")
    for preset in ("two_band", "three_band", "four_band", "five_band"):
        result = fit_model(spectrum, build_model(spectrum, preset=preset))
        print(f"  {preset:12s} R² = {result.r_squared:.6f}  "
              f"({result.n_parameters} parámetros)")

    print("\nY ahora con el criterio de información, que sí penaliza:")
    comparison = compare_models(spectrum)
    print(comparison.summary())

    print("\nComponentes del modelo ganador:")
    print(comparison.results[comparison.best].summary())


if __name__ == "__main__":
    main()
