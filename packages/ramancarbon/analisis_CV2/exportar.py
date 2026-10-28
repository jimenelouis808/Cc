"""The 5–100 mV/s series: everything the report quotes.

Driven through :class:`EchemSession`, the same path the window uses, so
that what this writes and what the window shows cannot drift apart.

    uv run python -I analisis_CV2/exportar.py <archivo .mpr> ...

Conditions, all from the user: glassy carbon 3 mm, 20 µg of active
material, KOH 3 M, Ag/AgCl in saturated KCl, platinum counter, sample S1 (FeSe/C
at 900 °C). The comment block of an .mpr is not read.
"""
import json
import sys

import numpy as np

from ramancarbon.echem.cv import (capacitance, compare_current_laws,
                                  coulombic_efficiency, elapsed_time,
                                  fit_full_current, integrate_charge,
                                  reversible_capacitance)
from ramancarbon.gui.echem_state import EchemSession

OUT = "/home/user/Cc/packages/ramancarbon/analisis_CV2"
MASS_MG = 0.020

#: The electrolyte, as a key of the table in ``echem.json``. **KOH 3 M,
#: not 0.5 M**: the first pass of this analysis used the wrong one, and
#: the cost is 57 mV on the RHE scale and therefore 57 mV on the OER
#: overpotential, which is the number the whole reading of the positive
#: limit rests on.
ELECTROLYTE = "KOH_3M"


def build(paths):
    session = EchemSession(name="S1 900 °C")
    session.electrode.mass_mg = MASS_MG      # lo único que no es defecto
    session.electrode.use_electrolyte(ELECTROLYTE)
    for path in paths:
        if not session.load_cv(path, into_series=True):
            raise SystemExit(session.messages[-1][1])
    session.rate_series.sort(key=lambda c: c.scan_rate)
    for curve in session.rate_series:
        curve.name = f"{1e3 * curve.scan_rate:.0f} mV/s"
    session.cv = session.rate_series[-1]
    session.retag_electrode()
    return session


def main(paths):
    session = build(paths)
    result = session.analyse()
    electrode = session.electrode
    rhe, rhe_note = electrode.to_rhe(session.cv.potential)

    for curve in session.rate_series:
        cycle, _ = curve.reporting_cycle()
        time, _ = elapsed_time(cycle)
        np.savetxt(f"{OUT}/ciclo_{1e3 * curve.scan_rate:.0f}mVs.csv",
                   np.column_stack([time, cycle.potential, cycle.current,
                                    cycle.current / (1e-3 * MASS_MG)]),
                   delimiter=",", comments="",
                   header="t_s,E_V_vs_AgAgCl,I_A,I_A_por_g")

    rates = []
    for curve in session.rate_series:
        cycle, _ = curve.reporting_cycle()
        time, _ = elapsed_time(cycle)
        oxidation, absolute, _ = integrate_charge(cycle)
        loop, branch = capacitance(curve)
        reduction = reversible_capacitance(curve)
        rates.append({
            "mv_s": 1e3 * curve.scan_rate,
            "ciclos": len(curve.cycles()),
            "duracion_ciclo_s": float(time[-1]),
            "q_ox_c": oxidation,
            "q_red_c": absolute - oxidation,
            "eficiencia": coulombic_efficiency(curve),
            "c_lazo_f_g": loop.specific_f_per_g,
            "c_rama_f_g": branch.specific_f_per_g,
            "c_red_f_g": reduction.specific_f_per_g,
            "c_red_mf_cm2": 1e3 * reduction.specific_f_per_cm2,
        })

    leak = result.rates.leakage
    summary = {
        "condiciones": {
            "material": "S1, FeSe/C a 900 °C",
            "electrodo_de_trabajo": "carbono vítreo, 3 mm (disco)",
            "masa_activa_mg": MASS_MG,
            "area_geometrica_cm2": electrode.area_cm2,
            "electrolito": electrode.electrolyte,
            "ph": electrode.ph,
            "referencia": "Ag/AgCl, KCl saturado",
            "contraelectrodo": "alambre de Pt",
            "conversion_a_rhe": rhe_note,
            "ventana_v_vs_referencia": [float(session.cv.potential.min()),
                                        float(session.cv.potential.max())],
            "ventana_v_vs_rhe": [float(rhe.min()), float(rhe.max())],
            "sobrepotencial_oer_v": float(rhe.max() - 1.23),
            "abanico_de_velocidades": (max(r["mv_s"] for r in rates)
                                       / min(r["mv_s"] for r in rates)),
        },
        "por_velocidad": rates,
        "fuga": None if leak is None else {
            "i_par_a": leak.leakage_a, "ordenada_c": leak.offset_c,
            "r2": leak.r_squared,
            "fraccion_de_q_ox": {f"{1e3 * r:.0f}": s
                                 for r, s in sorted(leak.shares.items())},
        },
        "b": [{"e_v": b.potential, "b": b.b, "error": b.b_error,
               "r2": b.r_squared, "lectura": b.interpretation}
              for b in result.rates.b_values],
        "leyes_de_dos_terminos": [
            {"e_v": law.potential, "b": law.b, "gana": law.best,
             "empate": not law.decisive,
             "r2": dict(law.r_squared)}
            for law in compare_current_laws(session.rate_series)
        ],
        "modelo_de_tres_terminos": [
            {"e_v": m.potential, "e_rhe": m.potential + float(rhe[0]
                                                              - session.cv.potential[0]),
             "k_cap_f": m.capacitance_f, "k_cap_error": m.errors[0],
             "k_diff": m.diffusive, "k_diff_error": m.errors[1],
             "i_par_a": m.leakage_a, "i_par_error": m.errors[2],
             "r2": m.r_squared, "gl": m.degrees_of_freedom,
             "determinados": m.determined()}
            for m in fit_full_current(session.rate_series)
        ],
        "dunn": {
            "fraccion_capacitiva": {f"{1e3 * r:.0f}": f
                                    for r, f in sorted(result.dunn.fractions.items())},
            "avisos": list(result.dunn.warnings),
        },
        "mecanismo": {
            "veredicto": result.storage.mechanism,
            "etiqueta": result.storage.label,
            "informar_como": result.storage.report_as,
            "a_favor": list(result.storage.evidence),
            "en_contra": list(result.storage.against),
        },
        "avisos_de_la_serie": list(result.rates.warnings),
    }
    with open(f"{OUT}/resumen.json", "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)
    session.save_tables(OUT)
    with open(f"{OUT}/estudio_de_velocidad.txt", "w") as handle:
        handle.write(result.rates.summary() + "\n\n" + result.dunn.summary() + "\n")
    print(result.rates.summary())


if __name__ == "__main__":
    main(sys.argv[1:])
