"""XPS: la referencia de carga decide todo lo que venga después.

    python -m ramancarbon.examples.ex20_xps
"""

from __future__ import annotations

from ramancarbon.examples.demo_data import make_xps_demo
from ramancarbon.xps.calibrate import calibrate, calibrate_to_state
from ramancarbon.xps.presets import compare_counts
from ramancarbon.xps.report import analyse_xps

CHARGE = 1.8
REGIONS = {
    "C 1s": ["C-C sp2", "C-O", "C=O", "O-C=O"],
    "N 1s": 4,
    "O 1s": 3,
    "Fe 2p3/2": ["Fe-Se", "Fe3+"],
    "Se 3d5/2": 3,
}


def main() -> None:
    spectra = make_xps_demo("NCNT_FeSe", seed=21, charge_shift=CHARGE)
    truth = spectra[0].metadata["true_composition_at"]
    carbon = next(item for item in spectra if item.region == "C 1s")

    print("1. LA REFERENCIA DE CARGA")
    print(f"   La muestra se está cargando {CHARGE:+.1f} eV. Hay dos formas de")
    print("   quitarlo, y en una muestra HECHA de carbono no valen lo mismo.\n")
    _, whole = calibrate(carbon, "C1s_adventitious")
    _, component = calibrate_to_state(
        carbon, "C 1s", "C-C sp2", REGIONS["C 1s"])
    for label, calibration in (("el pico C 1s entero", whole),
                               ("la componente sp² ajustada", component)):
        error = calibration.shift_ev + CHARGE
        print(f"   contra {label:<28s} → queda {error:+.2f} eV sin quitar")
    print("\n   El pico C 1s de este material es la suma de cuatro estados, y")
    print("   su centro no es la posición de ninguno. Además, 284.8 eV es el")
    print("   valor del carbono ADVENTICIO: el sp² de la muestra está en 284.4.\n")

    print("2. CUÁNTAS COMPONENTES TIENE EL N 1s")
    nitrogen = next(item for item in spectra if item.region == "N 1s")
    nitrogen = nitrogen.shifted(component.shift_ev, "sp² a 284.4 eV")
    comparisons, verdict = compare_counts(nitrogen, "N 1s", (2, 3, 4, 5))
    print(f"   {'n':>3s} {'χ²_red':>9s} {'DW':>6s}  {'parámetros':>10s}")
    for item in comparisons:
        print(f"   {item.n:3d} {item.reduced_chi2:9.2f} "
              f"{item.durbin_watson or float('nan'):6.2f} {item.n_parameters:10d}")
    print(f"\n   {verdict}\n")

    print("3. EL ANÁLISIS COMPLETO")
    analysis = analyse_xps(spectra, reference_state=("C 1s", "C-C sp2"),
                           regions=REGIONS, name="NCNT@FeSe")
    print(f"   elementos: {', '.join(analysis.survey.symbols())}")
    print(f"   {'elemento':>9s} {'medido':>9s} {'puesto':>9s}")
    for item in analysis.composition.abundances:
        print(f"   {item.element:>9s} {item.atomic_percent:8.1f} % "
              f"{truth[item.element]:8.1f} %")

    print("\n4. LOS ESTADOS DEL NITRÓGENO, que es para lo que se hace esto")
    region = analysis.region("N 1s")
    for item in region.components:
        print(f"   {item.label:<28s} {item.centre:7.2f} eV  "
              f"{100 * item.area_fraction:5.1f} %")
    print("\n   Raman ve que la red está desordenada y no puede decir en qué")
    print("   está enlazado ese nitrógeno. Esto sí, y por eso está aquí.")
    print("\n   (Espectros SINTÉTICOS: comprueban las fórmulas, no la")
    print("   exactitud sobre medidas reales.)")


if __name__ == "__main__":
    main()
