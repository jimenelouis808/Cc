"""Export every number the CV report quotes, from the five .mpr files.

Run as::

    uv run python -I analisis_CV/exportar.py <carpeta con los .mpr>

Conditions, all of them from the user and none from the files: glassy
carbon 3 mm, 20 µg of active material, 0.5 M KOH, Ag/AgCl in saturated
KCl, platinum wire counter electrode, sample S1 (the 900 °C FeSe/C
already analysed here by Raman and by diffraction).

**The comment block of an .mpr is not read.** EC-Lab carries it from one
experiment to the next and these were never edited, so it describes
somebody else's measurement.
"""
import glob
import json
import sys

import numpy as np

from ramancarbon.echem.biologic import read_eclab
from ramancarbon.echem.curve import Electrode
from ramancarbon.echem.cv import (analyse_cv, analyse_rate_study, capacitance,
                                  compare_current_laws,
                                  coulombic_efficiency, drift_check,
                                  dunn_analysis, elapsed_time,
                                  integrate_charge, leakage_from_rates,
                                  reversible_capacitance, without_parasitic)
from ramancarbon.echem.evaluate import classify_storage

OUT = "/home/user/Cc/packages/ramancarbon/analisis_CV"
DIAMETER_CM = 0.30
AREA_CM2 = np.pi * (DIAMETER_CM / 2.0) ** 2
MASS_MG = 0.020
ELECTROLYTE = "KOH 0.5 M"
REFERENCE = "Ag/AgCl_sat"

#: pH of 0.5 M KOH. Not pOH = 0.30: the mean activity coefficient of KOH
#: at that concentration is about 0.73, so a(OH⁻) ≈ 0.37 and pOH = 0.44.
#: It matters because the RHE conversion carries 59 mV per pH unit and
#: the whole question about this window is where its positive limit sits
#: relative to 1.23 V.
PH = 13.56

#: Acquisition order, which is the file numbering. It is not recoverable
#: from the records themselves, and everything about degradation depends
#: on it.
ORDER = ("01", "02", "03", "04", "05")


def load(folder):
    electrode = Electrode(mass_mg=MASS_MG, area_cm2=AREA_CM2,
                          reference=REFERENCE, ph=PH, electrolyte=ELECTROLYTE,
                          label="carbono vítreo 3 mm, 20 µg, S1 900 °C")
    curves, notes = [], []
    for tag in ORDER:
        matches = [p for p in glob.glob(f"{folder}/*CV_multpit*.mpr")
                   if f"_{tag}_" in p]
        if not matches:
            raise SystemExit(f"falta el archivo {tag}")
        curve, warnings = read_eclab(matches[0], electrode=electrode)
        curve.name = f"CV_{tag}"
        curves.append(curve)
        notes.append({"archivo": f"CV_multpit_{tag}",
                      "puntos": curve.n,
                      "avisos_del_lector": list(warnings)})
    return curves, notes


def cycle_table(curves):
    rows = []
    for curve in curves:
        whole = curve.cycles()
        complete = {c.name for c in curve.complete_cycles()}
        for index, cycle in enumerate(whole, start=1):
            time, origin = elapsed_time(cycle)
            oxidation, absolute, span = integrate_charge(cycle)
            reduction = absolute - oxidation
            rows.append({
                "archivo": curve.name,
                "mv_s": 1e3 * curve.scan_rate,
                "ciclo": index,
                "completo": cycle.name in complete,
                "puntos": cycle.n,
                "duracion_s": float(time[-1]),
                "ventana_v": cycle.span,
                "recorrido_v": cycle.traversed,
                "q_ox_c": oxidation,
                "q_red_c": reduction,
                "eficiencia": reduction / oxidation if oxidation > 0 else None,
                "c_lazo_f_g": capacitance(cycle)[0].specific_f_per_g,
                "c_rama_f_g": capacitance(cycle)[1].specific_f_per_g,
                "c_red_f_g": reversible_capacitance(cycle).specific_f_per_g,
                "origen_del_tiempo": origin,
            })
    return rows


def main(folder):
    curves, notes = load(folder)

    # One file per reporting cycle, for plotting elsewhere.
    for curve in curves:
        cycle, _ = curve.reporting_cycle()
        time, _ = elapsed_time(cycle)
        table = np.column_stack([time, cycle.potential, cycle.current,
                                 cycle.current / (1e-3 * MASS_MG)])
        np.savetxt(f"{OUT}/{curve.name}_ciclo.csv", table, delimiter=",",
                   header="t_s,E_V_vs_ref,I_A,I_A_por_g", comments="")

    # Every cycle of file 01, which is where the fade is visible.
    first = curves[0]
    with open(f"{OUT}/CV_01_todos_los_ciclos.csv", "w") as handle:
        handle.write("ciclo,E_V_vs_ref,I_A\n")
        for index, cycle in enumerate(first.cycles(), start=1):
            for potential, current in zip(cycle.potential, cycle.current):
                handle.write(f"{index},{potential:.6g},{current:.6g}\n")

    rows = cycle_table(curves)
    with open(f"{OUT}/ciclos.csv", "w") as handle:
        keys = [k for k in rows[0] if k != "origen_del_tiempo"]
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(
                "" if row[k] is None else
                (f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]))
                for k in keys) + "\n")

    series = curves[1:]  # 02..05, the rate series proper
    study = analyse_rate_study(curves)
    dunn = dunn_analysis(series)
    leak = leakage_from_rates(series)
    drift = drift_check(curves)
    verdict = classify_storage(cv=analyse_cv(curves[-1]), rates=study)

    # The Dunn separation at the fastest rate, for the figure.
    fast = series[-1]
    cycle, _ = fast.reporting_cycle()
    anodic, cathodic, exceeded = dunn.capacitive_inside(cycle)
    np.savetxt(f"{OUT}/dunn_50mVs.csv",
               np.column_stack([dunn.potentials, dunn.k1, dunn.k2,
                                dunn.r_squared, anodic, cathodic,
                                exceeded.astype(float)]),
               delimiter=",",
               header="E_V,k1_A_por_Vs,k2_A_por_sqrtVs,R2,i_cap_anodica_A,"
                      "i_cap_catodica_A,no_definida", comments="")

    reference_curve = curves[0]
    rhe, rhe_note = reference_curve.electrode.to_rhe(reference_curve.potential)

    # The one test of the b-value that is not circular: subtract the
    # parasitic current MEASURED FROM THE CHARGE BALANCE, which never
    # looked at how i scales with ν, and see whether b comes back to 1.
    corrected, correction_note = (
        without_parasitic(series, leak.leakage_a) if leak else (None, "")
    )
    corrected_study = analyse_rate_study(corrected) if corrected else None
    corrected_dunn = dunn_analysis(corrected) if corrected else None

    summary = {
        "condiciones": {
            "electrodo_de_trabajo": "carbono vítreo, 3 mm de diámetro",
            "material": "S1, FeSe/C tratado a 900 °C",
            "masa_activa_mg": MASS_MG,
            "area_geometrica_cm2": AREA_CM2,
            "electrolito": ELECTROLYTE,
            "referencia": "Ag/AgCl, KCl saturado",
            "contraelectrodo": "alambre de Pt",
            "ph": PH,
            "conversion_a_rhe": rhe_note,
            "ventana_v_vs_referencia": [float(reference_curve.potential.min()),
                                        float(reference_curve.potential.max())],
            "ventana_v_vs_rhe": [float(rhe.min()), float(rhe.max())],
            "sobrepotencial_oer_en_el_limite_positivo_v": float(rhe.max() - 1.23),
            "nota": "nada de esto sale de los .mpr: su bloque de comentarios "
                    "lo arrastra EC-Lab de un experimento al siguiente y no "
                    "se lee",
        },
        "orden_de_medida": list(ORDER),
        "por_archivo": [
            {
                "archivo": curve.name,
                "mv_s": 1e3 * curve.scan_rate,
                "ciclos": len(curve.cycles()),
                "ciclos_completos": len(curve.complete_cycles()),
                "ciclo_informado": curve.reporting_cycle()[0].name,
                "ventana_v": curve.span,
                "eficiencia_coulombica": coulombic_efficiency(curve),
                "c_lazo_f_g": capacitance(curve)[0].specific_f_per_g,
                "c_rama_f_g": capacitance(curve)[1].specific_f_per_g,
                "c_reduccion_f_g": reversible_capacitance(curve).specific_f_per_g,
                "c_reduccion_mf_cm2":
                    1e3 * reversible_capacitance(curve).specific_f_per_cm2,
            }
            for curve in curves
        ],
        "fuga": None if leak is None else {
            "i_par_a": leak.leakage_a,
            "ordenada_c": leak.offset_c,
            "r2": leak.r_squared,
            "c_sin_fuga_f": leak.capacitance_farads,
            "c_sin_fuga_f_g": curves[0].electrode.specific(
                leak.capacitance_farads, "mass"),
            "fraccion_de_q_ox": {f"{1e3 * r:.0f}": s
                                 for r, s in sorted(leak.shares.items())},
        },
        "deriva": None if drift is None else {
            "mv_s": 1e3 * drift.rate,
            "primera_f": drift.first,
            "ultima_f": drift.last,
            "perdida_pct": drift.fade_percent,
            "archivos": list(drift.names),
        },
        "leyes_de_i_nu": [
            {
                "e_v": law.potential,
                "b": law.b,
                "r2": {k: law.r_squared[k] for k in law.r_squared},
                "empate": not law.decisive,
                "gana": law.best,
                "c_fuga_f": law.capacitance_f,
                "i_par_a": law.leakage_a,
            }
            for law in compare_current_laws(series)
        ],
        "b": [{"e_v": b.potential, "b": b.b, "error": b.b_error,
               "r2": b.r_squared, "lectura": b.interpretation}
              for b in study.b_values],
        "dunn": {
            "fraccion_capacitiva": {f"{1e3 * r:.0f}": f
                                    for r, f in sorted(dunn.fractions.items())},
            "potenciales_con_r2_bajo": int(np.count_nonzero(dunn.r_squared < 0.95)),
            "potenciales": int(dunn.r_squared.size),
            "avisos": list(dunn.warnings),
        },
        "trasatti": {
            "total_f": study.trasatti_total,
            "externa_f": study.trasatti_outer,
        },
        "mecanismo": {
            "veredicto": verdict.mechanism,
            "etiqueta": verdict.label,
            "confianza": verdict.confidence,
            "informar_como": verdict.report_as,
            "a_favor": list(verdict.evidence),
            "en_contra": list(verdict.against),
        },
        "sin_la_fuga": None if corrected_study is None else {
            "que_se_hizo": correction_note,
            "b": [{"e_v": b.potential, "b": b.b, "error": b.b_error}
                  for b in corrected_study.b_values],
            "dunn": {f"{1e3 * r:.0f}": f
                     for r, f in sorted(corrected_dunn.fractions.items())},
            "lectura": "restar una constante única sobrecorrige en el extremo "
                       "negativo (b pasa de 1, que es imposible) y se queda "
                       "corta en el positivo. Los dos errores son mayores que "
                       "el efecto que se quiere medir: con estos datos el b y "
                       "el reparto de Dunn no se pueden obtener",
        },
        "avisos_de_la_serie": list(study.warnings),
        "metadatos_de_los_archivos": notes,
    }
    with open(f"{OUT}/resumen.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)

    with open(f"{OUT}/estudio_de_velocidad.txt", "w") as handle:
        handle.write(study.summary() + "\n\n" + dunn.summary() + "\n")

    print(study.summary())
    print()
    print(dunn.summary())
    print()
    print(verdict)


if __name__ == "__main__":
    main(sys.argv[1])
