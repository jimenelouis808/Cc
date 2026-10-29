"""Los tres PEIS en H2SO4 0.5 M: 800, 900 y 1020 °C.

Pasa por :class:`EchemSession`, el mismo camino que usa la ventana, para
que lo que escribe esto y lo que enseña la ventana no puedan separarse.

    uv run python -I analisis_PEIS/exportar.py \
        <800 °C>.mpr <900 °C>.mpr <1020 °C>.mpr

Condiciones, todas del usuario: carbón de otro material carbonoso, H2SO4
0.5 M, tres temperaturas de pirólisis. Lo que el usuario NO dio:

- **La masa.** Así que no hay ni un F/g en ninguna tabla. Una
  capacitancia específica sin masa medida es el número equivocado
  exactamente en la proporción que se suponga.
- **El área.** Se usa el disco de carbón vítreo de 3 mm que el usuario
  dice usar casi siempre (0.0707 cm²), y va dicho en cada tabla que es
  una SUPOSICIÓN. Es área GEOMÉTRICA: la electroquímicamente activa de
  un carbón poroso es de diez a mil veces mayor.
- **La referencia.** Se deja Ag/AgCl en KCl saturado, que es la que usó
  en los CV. Para la impedancia da igual salvo para situar el potencial
  de polarización: la medida es de 10 mV alrededor de un punto fijo y
  ninguna magnitud que sale de aquí depende de la escala.

El bloque de comentarios del .mpr no se lee.
"""
import json
import sys
from pathlib import Path

import numpy as np

from ramancarbon.echem.capacitance import complex_capacitance
from ramancarbon.echem.eis import (AUTO_CIRCUIT, analyse_eis, brug_capacitance,
                                   compare_circuits, draw_circuit, drt,
                                   fit_circuit, read_shape, recommend_circuits)
from ramancarbon.gui.echem_state import EchemSession

OUT = Path("/home/user/Cc/packages/ramancarbon/analisis_PEIS")

ELECTROLYTE = "H2SO4_0.5M"
DISC_MM = 3.0

#: Label for each file, in the order they are passed. The pyrolysis
#: temperature IS the variable of this experiment, so it is the legend.
LABELS = ("800 °C", "900 °C", "1020 °C")

#: The circuit every sample is ALSO fitted with, on top of the one its
#: own shape asks for. Two different questions: «what describes this
#: spectrum best» is per sample, and «how do the three compare» needs one
#: model for all three or the parameters are not the same parameters.
COMMON = "R0-(R1|Q1)-T1"

#: And the one for the sample whose spectrum goes inductive at the top.
#: Leaving the L out of THAT one is not a tidiness question: measured on
#: the 900 °C file, R0 absorbs the inductance and T1.Ri collapses from
#: 0.79 Ω (± 13 %) to 2·10⁻⁵ Ω (± 3.5·10⁷ %).
COMMON_WITH_L = "L0-R0-(R1|Q1)-T1"

CANDIDATES = ("edlc_cpe", "randles_cpe", "zarc", "poroso", "supercondensador",
              "dos_semicirculos", "randles_wo", "poroso_con_transferencia",
              "supercondensador_completo")


def build(paths):
    session = EchemSession(name="carbón en H2SO4 0.5 M")
    session.electrode.use_electrolyte(ELECTROLYTE)
    session.electrode.use_disc(DISC_MM)
    session.electrode.mass_mg = None
    session.circuit = AUTO_CIRCUIT
    for path, label in zip(paths, LABELS):
        if not session.add_eis_to_series(path, label=label):
            raise SystemExit(session.messages[-1][1])
    return session


def _fit_row(label, fit):
    row = {"muestra": label, "circuito": fit.circuit,
           "residuo_pct": fit.residual_pct, "chi2": fit.chi_squared,
           "convergido": bool(fit.converged)}
    for name, value in fit.values().items():
        row[name] = value
        error = fit.errors.get(name)
        row[f"{name}_error"] = error
        row[f"{name}_pct"] = (100.0 * error / abs(value)
                              if error is not None and value else None)
    return row


def main(paths):
    session = build(paths)
    OUT.mkdir(parents=True, exist_ok=True)
    area = session.electrode.area_cm2

    comparison = session.eis_comparison_rows()
    samples = []
    fits = []
    for spectrum in session.eis_series:
        label = spectrum.name
        shape = read_shape(spectrum)
        suggestions, _ = recommend_circuits(spectrum, shape)
        analysis = analyse_eis(spectrum, circuit=AUTO_CIRCUIT)
        cap = complex_capacitance(spectrum)
        ranking = compare_circuits(spectrum, list(CANDIDATES))
        common = fit_circuit(spectrum,
                             COMMON_WITH_L if shape.inductive_top else COMMON)
        fits.append(_fit_row(label, common))
        if analysis.fit is not None and analysis.fit.circuit != common.circuit:
            fits.append(_fit_row(label + " (propio)", analysis.fit))
        relaxation = drt(spectrum)

        stored = analysis.storage_capacitance
        # The arc's CPE under BOTH accepted relations, because they rest
        # on different physics and the literature uses them
        # interchangeably. Brug needs R_s as well, and since
        # R_s ∥ R_ct < R_ct it always returns the smaller number.
        arc_brug = None
        values = common.values()
        if "Q1.Q" in values and "R1.R" in values and "R0.R" in values:
            arc_brug = brug_capacitance(
                values["Q1.Q"], values["Q1.n"], values["R0.R"],
                values["R1.R"])
        samples.append({
            "muestra": label,
            "archivo": Path(spectrum.metadata.get("archivo", "")).name,
            "n_frecuencias": spectrum.n,
            "f_hz": [float(spectrum.frequency.max()),
                     float(spectrum.frequency.min())],
            "forma": {
                "r_serie_ohm": shape.series_resistance,
                "r_serie_como": shape.series_how,
                "inductivo_arriba": shape.inductive_top,
                "apice_arco_hz": shape.arc_apex_hz,
                "codo_hz": shape.knee_hz,
                "codo_como": shape.knee_how,
                "fase_f_min_grados": shape.low_phase_deg,
                "cola_capacitiva": shape.capacitive_tail,
                "decadas_45": shape.fortyfive_decades,
                "notas": list(shape.notes),
            },
            "candidatos": [{"nombre": s.name, "circuito": s.circuit,
                            "porque": s.why} for s in suggestions],
            "ranking": [{"circuito": c.circuit, "nombre": c.name,
                         "chi2": c.fit.chi_squared if c.fit else None,
                         "residuo_pct": c.fit.residual_pct if c.fit else None,
                         "libres": c.free_parameters,
                         "determinados": c.determined, "bic": c.bic}
                        for c in ranking.ranked()],
            "elegido": analysis.fit.circuit if analysis.fit else None,
            "comun": common.circuit,
            "kk": {"pasa": bool(analysis.kk.passes),
                   "residuo_rms_pct": 100.0 * analysis.kk.rms_residual,
                   "rachas": analysis.kk.runs_ratio} if analysis.kk else None,
            "capacitancia": {
                "c_prima_mf": 1e3 * cap.low_frequency_capacitance,
                "c_prima_f_hz": float(cap.frequency[0]),
                "c_serie_mf": 1e3 * float(cap.series[0]),
                "fase_grados": cap.low_frequency_phase_deg,
                "tau0_s": cap.relaxation_s,
                "f0_hz": cap.relaxation_frequency_hz,
                "mf_cm2": 1e3 * cap.low_frequency_capacitance / area,
                "elemento_mf": 1e3 * stored.farads if stored else None,
                "elemento_metodo": stored.method if stored else None,
                "elemento_pasos": list(stored.steps) if stored else [],
                "arco_mf": (1e3 * analysis.arc_capacitance.farads
                            if analysis.arc_capacitance else None),
                "arco_metodo": (analysis.arc_capacitance.method
                                if analysis.arc_capacitance else None),
                "arco_pasos": (list(analysis.arc_capacitance.steps)
                               if analysis.arc_capacitance else []),
                "arco_brug_mf": 1e3 * arc_brug.farads if arc_brug else None,
                "arco_brug_metodo": arc_brug.method if arc_brug else None,
                "arco_brug_pasos": list(arc_brug.steps) if arc_brug else [],
                "comprobacion": analysis.storage_check,
                "avisos": list(cap.warnings),
            },
            "r_ion_ohm": analysis.ionic_resistance_ohm,
            "r_ct_ohm": analysis.charge_transfer_resistance,
            "drt": {
                "lambda": relaxation.regularisation,
                "residuo_pct": 100.0 * relaxation.residual,
                "ohmica_ohm": relaxation.ohmic,
                "picos": [{"tau_s": t, "r_ohm": r, "c_f": t / r if r else None}
                          for t, r in zip(relaxation.peaks_s,
                                          relaxation.peak_resistances)],
                "avisos": list(relaxation.warnings),
            },
            "dibujo": draw_circuit(common.circuit, common.values()),
            "avisos": list(analysis.warnings),
        })

        # per-sample spectrum, measured and fitted, with the residual
        slug = label.replace(" ", "").replace("°", "")
        header = ("f_hz,z_real_ohm,z_imag_ohm,modulo_ohm,fase_grados,"
                  "z_real_ajuste_ohm,z_imag_ajuste_ohm,residuo_pct,"
                  "c_prima_f,c_segunda_f")
        order = np.argsort(spectrum.frequency)[::-1]
        caps = {float(f): (float(r), float(i)) for f, r, i
                in zip(cap.frequency, cap.real, cap.imaginary)}
        lines = [header]
        for index in order:
            f = float(spectrum.frequency[index])
            z = spectrum.z[index]
            zf = complex(common.fitted[index])
            residual = 100.0 * abs(z - zf) / abs(z)
            cr, ci = caps[f]
            lines.append(
                f"{f:.6g},{z.real:.6g},{z.imag:.6g},{abs(z):.6g},"
                f"{np.degrees(np.angle(z)):.6g},{zf.real:.6g},{zf.imag:.6g},"
                f"{residual:.6g},{cr:.6g},{ci:.6g}")
        (OUT / f"espectro_{slug}.csv").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")

        # per-sample DRT
        lines = ["tau_s,gamma_ohm,dentro_del_rango"]
        low, high = relaxation.measured_s
        for tau, gamma in zip(relaxation.tau_s, relaxation.gamma):
            lines.append(f"{tau:.6g},{gamma:.6g},{int(low <= tau <= high)}")
        (OUT / f"drt_{slug}.csv").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")

        text = analysis.summary() + "\n\n" + relaxation.describe() + "\n" \
            + relaxation.interpret() + "\n\n" + ranking.describe()
        (OUT / f"informe_{slug}.txt").write_text(text, encoding="utf-8")

    # the comparison table
    (OUT / "comparacion.csv").write_text(
        "\n".join(",".join(row) for row in comparison) + "\n",
        encoding="utf-8")

    # the fitted parameters, one row per (sample, circuit)
    keys = []
    for row in fits:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [",".join(keys)]
    for row in fits:
        lines.append(",".join(
            "" if row.get(key) is None else
            (f"{row[key]:.6g}" if isinstance(row.get(key), float)
             else str(row.get(key)))
            for key in keys))
    (OUT / "ajustes.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # the capacitances, every basis and every method side by side
    lines = ["muestra,c_prima_mf,c_serie_mf,c_elemento_mf,c_arco_hm_mf,"
             "c_arco_brug_mf,mf_cm2_geometrica,fase_grados,tau0_s,f0_hz,f_hz"]
    for sample in samples:
        c = sample["capacitancia"]
        lines.append(
            f"{sample['muestra']},{c['c_prima_mf']:.6g},{c['c_serie_mf']:.6g},"
            + (f"{c['elemento_mf']:.6g}" if c["elemento_mf"] else "")
            + "," + (f"{c['arco_mf']:.6g}" if c["arco_mf"] else "")
            + "," + (f"{c['arco_brug_mf']:.6g}" if c["arco_brug_mf"] else "")
            + f",{c['mf_cm2']:.6g},{c['fase_grados']:.6g},"
            + (f"{c['tau0_s']:.6g}" if c["tau0_s"] else "")
            + "," + (f"{c['f0_hz']:.6g}" if c["f0_hz"] else "")
            + f",{c['c_prima_f_hz']:.6g}")
    (OUT / "capacitancias.csv").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")

    summary = {
        "condiciones": {
            "electrolito": ELECTROLYTE,
            "ph": session.electrode.ph,
            "referencia": session.electrode.reference,
            "area_cm2": area,
            "area_suposicion": (
                f"disco de carbón vítreo de {DISC_MM:g} mm; el usuario no "
                "dio el área para estas medidas"),
            "masa_mg": None,
            "masa_nota": (
                "no dada, así que no hay F/g en ninguna tabla de este "
                "análisis"),
            "circuito_comun": COMMON,
            "circuito_comun_con_l": COMMON_WITH_L,
        },
        "muestras": samples,
        "comparacion": [list(row) for row in comparison],
    }
    (OUT / "resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print("escritos en", OUT)
    for row in comparison:
        print("  " + " | ".join(f"{value:>16s}" for value in row))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
