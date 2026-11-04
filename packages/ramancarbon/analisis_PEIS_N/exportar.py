"""Cuatro PEIS de carbones Fe/Fe3C según el precursor de nitrógeno.

Pasa por :class:`EchemSession`, el mismo camino que usa la ventana, para
que lo que escribe esto y lo que enseña la ventana no puedan separarse.

    uv run python -I analisis_PEIS_N/exportar.py E.mpr EA.mpr EB.mpr EP.mpr

Las cuatro muestras son carbones con alto contenido de Fe y Fe3C,
pirolizados de etanol como fuente de carbono. Lo que cambia entre ellas
es el precursor que aporta el NITRÓGENO:

===  ============================  ==================================
E    etanol solo                   el control, sin nitrógeno
EA   etanol + acetonitrilo         CH3CN, nitrilo, N sp
EB   etanol + bencilamina          PhCH2NH2, amina primaria, N sp3
EP   etanol + piridina             C5H5N, N aromático ya en un anillo
===  ============================  ==================================

Condiciones, todas dadas por el usuario: **H2SO4 0.5 M**, **disco de
carbón vítreo de 3 mm** (0.0707 cm²), **Ag/AgCl saturado** y
**0.909 mg** de material (20 uL de una tinta de 5 mg en 110 uL). El
electrolito no entra en ninguna impedancia medida, pero sí en el pH, y
del pH sale la conversión a RHE — que aquí resulta ser la diferencia
entre medir una doble capa y medir una reacción.

**EB está DOS veces, a dos potenciales, y esa pareja es el control
interno de todo el análisis.** El primer espectro se midió a −0.19 V vs
RHE, por debajo del equilibrio del hidrógeno, y no pasa Kramers-Kronig;
el segundo, a +0.54 V, en la misma ventana que las otras tres, lo pasa.
Misma muestra, misma celda (R_s 85.01 y 85.22 ohm), seis veces menos
corriente y el doble de capacitancia. Eso convierte «EB no pasa KK» de
una sospecha sobre la muestra en una medida sobre el potencial.

Lo que sigue siendo una suposición, y se dice en cada tabla:

- **La masa es NOMINAL.** Sale del volumen de tinta depositado, no de
  pesar el electrodo seco. Y son 12.9 mg/cm2, de diez a cien veces la
  carga habitual sobre un carbón vítreo: la película es gruesa, y eso
  explica por sí solo que ninguna muestra llegue al régimen capacitivo.
- **El área ELECTROQUÍMICAMENTE ACTIVA.** Los 0.0707 cm² son el área
  GEOMÉTRICA del disco: la activa de un carbón poroso es de diez a mil
  veces mayor, así que los mF/cm² sirven para comparar estas muestras
  entre sí y para nada más.

**Y los cuatro precursores se midieron a POTENCIALES DISTINTOS.** Los
nombres dicen «PEIS 0 V», que en EC-Lab es 0 V *frente al potencial de
circuito abierto*, así que el continuo de cada medida es el OCP de su
propia muestra. Una resistencia de transferencia de carga depende
exponencialmente del sobrepotencial, así que entre dos muestras medidas
a 0.9 V de distancia no se puede atribuir a la química lo que puede ser
el punto de polarización. Con el segundo EB, tres de las cuatro quedan
ya dentro de 0.18 V entre sí.

El bloque de comentarios del .mpr no se lee.
"""
import json
import sys
from pathlib import Path

import numpy as np

from ramancarbon.echem.biologic import read_mpr
from ramancarbon.echem.capacitance import complex_capacitance
from ramancarbon.echem.eis import (AUTO_CIRCUIT, analyse_eis, brug_capacitance,
                                   compare_circuits, draw_circuit, drt,
                                   fit_circuit, hsu_mansfeld_capacitance,
                                   read_shape, recommend_circuits)
from ramancarbon.gui.echem_state import EchemSession

OUT = Path("/home/user/Cc/packages/ramancarbon/analisis_PEIS_N")

#: Dados por el usuario. El pH de `echem.json` es 0.3 y sale de la
#: ACTIVIDAD, no de -log c: el primer protón del sulfúrico está
#: disociado del todo y el segundo sólo en parte (Ka2 = 0.012).
ELECTROLYTE = "H2SO4_0.5M"
DISC_MM = 3.0

#: CONFIRMADA por el usuario. De ella depende la conversión a RHE, y de
#: la conversión depende si una muestra está por encima o por debajo del
#: equilibrio del hidrógeno — que aquí resulta ser la diferencia entre
#: medir una doble capa y medir una reacción.
REFERENCE = "Ag/AgCl_sat"

#: La masa de material sobre el electrodo, del protocolo del usuario:
#: tinta de 5 mg de muestra en 100 uL de isopropanol y 10 uL de Nafion,
#: de la que se depositan 20 uL. 20/110 x 5 mg = 0.909 mg. Es NOMINAL:
#: sale del volumen depositado, no de pesar el electrodo, y una gota
#: que se sale del disco se lleva masa que esta cuenta no descuenta.
MASS_MG = 0.909

#: La etiqueta de cada archivo, en el orden en que se pasan. El PRECURSOR
#: es la variable de este experimento, así que es la leyenda — salvo en
#: EB, que va DOS veces porque es la misma muestra a dos potenciales, y
#: esa pareja es el control interno de todo el análisis.
LABELS = ("E (etanol)", "EA (+acetonitrilo)", "EB (+bencilamina) −0.19 V",
          "EB (+bencilamina) +0.54 V", "EP (+piridina)")

#: Qué aporta cada precursor, para el informe. El entorno del nitrógeno
#: en el precursor NO es el que acaba en el carbón —la pirólisis lo
#: rehace— pero es lo único que distingue a estas cuatro síntesis.
PRECURSORS = {
    "E (etanol)": "sin nitrógeno; el control",
    "EA (+acetonitrilo)": "CH3CN: nitrilo, N sp, relación C/N baja",
    "EB (+bencilamina) −0.19 V":
        "PhCH2NH2: amina primaria sp3 sobre un anillo. MEDIDA POR DEBAJO "
        "DEL EQUILIBRIO DEL HIDRÓGENO",
    "EB (+bencilamina) +0.54 V":
        "PhCH2NH2: amina primaria sp3 sobre un anillo. La MISMA muestra, "
        "en la ventana de las otras tres",
    "EP (+piridina)": "C5H5N: N aromático ya dentro de un anillo",
}

#: El nombre de archivo de cada muestra. No sale de la etiqueta porque
#: las dos de EB empiezan igual y se pisarían los CSV.
SLUGS = {
    "E (etanol)": "E",
    "EA (+acetonitrilo)": "EA",
    "EB (+bencilamina) −0.19 V": "EB_cat",
    "EB (+bencilamina) +0.54 V": "EB_an",
    "EP (+piridina)": "EP",
}

#: El circuito con el que se ajustan TODAS, encima del que su propia
#: forma pida. Dos preguntas distintas: «qué describe mejor este
#: espectro» es por muestra, y «cómo se comparan las cuatro» necesita un
#: modelo común o los parámetros no son los mismos parámetros. Es el que
#: ganó el barrido del catálogo entero en las cuatro, y el que dibuja la
#: figura: dos arcos, 0.71 / 0.54 / 5.79 / 5.11 % de residuo medio,
#: contra el 2.4 / 3.2 / 9.7 / 17.8 % que daba el automático por
#: espectro — que en EP deja la curva ajustada sin seguir a los puntos.
COMMON = "R0-(R1|Q1)-(R2|Q2)"

#: Y el mismo con la inductancia del cableado, que se ajusta ADEMÁS para
#: la muestra cuyo extremo alto se va inductivo. No sustituye al común:
#: con dos arcos la L no cambia el residuo de E (0.706 % con y sin) y
#: vuelve indeterminada ahí, así que meterla en el modelo común costaría
#: un parámetro sin medida y haría que las cuatro ya no llevaran el
#: mismo circuito. Donde sí paga es en EA, 0.541 → 0.497 % con los ocho
#: parámetros determinados.
COMMON_WITH_L = "L0-R0-(R1|Q1)-(R2|Q2)"

#: El circuito FÍSICO del electrodo poroso, que se ajusta además del
#: común: es el único que devuelve la resistencia iónica del poro, que
#: es la que limita la potencia. Donde la devuelve determinada es una
#: medida; donde no, se dice.
POROUS = "R0-(R1|Q1)-T1"
POROUS_WITH_L = "L0-R0-(R1|Q1)-T1"

CANDIDATES = ("edlc_cpe", "randles_cpe", "zarc", "poroso", "supercondensador",
              "dos_semicirculos", "randles_wo", "poroso_con_transferencia",
              "supercondensador_completo")


def dc_potential(path: str) -> float:
    """El continuo al que se midió el espectro, leído del archivo.

    Es el dato que decide si dos espectros son comparables, y no está en
    ninguna de las magnitudes que el análisis de impedancia produce.
    """
    column = read_mpr(path).columns.get("<Ewe>/V")
    return float(np.mean(column)) if column is not None else float("nan")


def build(paths):
    session = EchemSession(name="carbones Fe/Fe3C por precursor de N")
    session.electrode.use_electrolyte(ELECTROLYTE)
    session.electrode.reference = REFERENCE
    session.electrode.use_disc(DISC_MM)
    session.electrode.mass_mg = MASS_MG
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
    biases = {label: dc_potential(path)
              for label, path in zip(LABELS, paths)}
    # To RHE, which is the scale on which «is this above or below the
    # hydrogen equilibrium» is a question with an answer. The package
    # refuses the conversion without a pH rather than guessing one.
    rhe = {}
    for label, value in biases.items():
        converted, why = session.electrode.to_reference([value], "RHE")
        rhe[label] = float(converted[0]) if converted is not None else None
    rhe_note = why

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
        common = fit_circuit(spectrum, COMMON)
        fits.append(_fit_row(label, common))
        # The wiring inductance as an EXTRA row where the shape shows
        # one, never in place of the common circuit: four samples
        # compared through four circuits are not compared.
        if shape.inductive_top:
            fits.append(_fit_row(label + " (con L)",
                                 fit_circuit(spectrum, COMMON_WITH_L)))
        # And the porous model, always, because it is the only one that
        # returns the pore's ionic resistance. Where that comes back
        # determined it is a measurement; where it does not, the table
        # says so rather than leaving the column empty.
        porous = fit_circuit(spectrum,
                             POROUS_WITH_L if shape.inductive_top else POROUS)
        fits.append(_fit_row(label + " (poroso)", porous))
        if analysis.fit is not None and analysis.fit.circuit not in (
                common.circuit, porous.circuit):
            fits.append(_fit_row(label + " (automático)", analysis.fit))
        relaxation = drt(spectrum)

        stored = analysis.storage_capacitance
        # Both arcs of the COMMON circuit, both relations, so the whole
        # capacitance table comes out of one model instead of mixing the
        # auto-chosen circuit's arc with the common one's. The slow arc
        # is the one that stores: its tau says whether the measurement
        # ever reached it, and in two of the four it did not.
        values = common.values()
        arcs = []
        for index in (1, 2):
            q, n = f"Q{index}.Q", f"Q{index}.n"
            r = f"R{index}.R"
            if not {q, n, r, "R0.R"} <= set(values):
                continue
            hm = hsu_mansfeld_capacitance(values[q], values[n], values[r])
            brug = brug_capacitance(values[q], values[n], values["R0.R"],
                                    values[r])
            arcs.append({
                "arco": index,
                "r_ohm": values[r],
                "hm_mf": 1e3 * hm.farads,
                "brug_mf": 1e3 * brug.farads,
                "tau_s": values[r] * hm.farads,
                "f_apice_hz": (1.0 / (2.0 * np.pi * values[r] * hm.farads)
                               if hm.farads else None),
                "pasos_hm": list(hm.steps),
                "pasos_brug": list(brug.steps),
            })
        arc_brug = None
        if arcs:
            arc_brug = brug_capacitance(
                values["Q1.Q"], values["Q1.n"], values["R0.R"],
                values["R1.R"])
        samples.append({
            "muestra": label,
            "precursor": PRECURSORS[label],
            "archivo": Path(spectrum.metadata.get("archivo", "")).name,
            "potencial_continuo_v": biases[label],
            "potencial_rhe_v": rhe.get(label),
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
                # F/g, which the mass finally allows. It is a LOWER
                # bound twice over: C' at 0.01 Hz has not reached the
                # capacitive regime, and the mass is the whole deposit
                # whether or not the ion got to all of it.
                "f_g": cap.low_frequency_capacitance / (1e-3 * MASS_MG),
                "f_g_elemento": (stored.farads / (1e-3 * MASS_MG)
                                 if stored else None),
                "elemento_mf": 1e3 * stored.farads if stored else None,
                "elemento_metodo": stored.method if stored else None,
                "elemento_pasos": list(stored.steps) if stored else [],
                "arco_mf": (1e3 * analysis.arc_capacitance.farads
                            if analysis.arc_capacitance else None),
                "arco_metodo": (analysis.arc_capacitance.method
                                if analysis.arc_capacitance else None),
                "arco_pasos": (list(analysis.arc_capacitance.steps)
                               if analysis.arc_capacitance else []),
                "arcos_del_comun": arcs,
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

        slug = SLUGS[label]
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

        lines = ["tau_s,gamma_ohm,dentro_del_rango"]
        low, high = relaxation.measured_s
        for tau, gamma in zip(relaxation.tau_s, relaxation.gamma):
            lines.append(f"{tau:.6g},{gamma:.6g},{int(low <= tau <= high)}")
        (OUT / f"drt_{slug}.csv").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")

        text = (f"{label} — {PRECURSORS[label]}\n"
                f"medido a {biases[label]:+.4f} V vs {REFERENCE}"
                + (f" = {rhe[label]:+.4f} V vs RHE" if rhe[label] is not None
                   else "")
                + " (continuo, del archivo)\n\n"
                + analysis.summary() + "\n\n" + relaxation.describe() + "\n"
                + relaxation.interpret() + "\n\n" + ranking.describe())
        (OUT / f"informe_{slug}.txt").write_text(text, encoding="utf-8")

    (OUT / "comparacion.csv").write_text(
        "\n".join(",".join(row) for row in comparison) + "\n",
        encoding="utf-8")

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

    lines = ["muestra,potencial_continuo_v,potencial_rhe_v,c_prima_mf,c_serie_mf,"
             "c_elemento_mf,c_arco_hm_mf,c_arco_brug_mf,"
             "c_arco_rapido_hm_mf,tau_arco_rapido_s,"
             "c_arco_lento_hm_mf,tau_arco_lento_s,mf_cm2_geometrica,"
             "f_g_c_prima,f_g_elemento,fase_grados,tau0_s,f0_hz,f_hz"]
    for sample in samples:
        c = sample["capacitancia"]
        # Sorted by tau: which of the two arcs is the slow one is a
        # property of the fit, not of the order the parameters are
        # named in, and the slow one is the one that stores.
        arcs = sorted(c["arcos_del_comun"], key=lambda a: a["tau_s"])
        fast = arcs[0] if arcs else None
        slow = arcs[-1] if len(arcs) > 1 else None
        lines.append(
            f"{sample['muestra']},{sample['potencial_continuo_v']:.6g},"
            + (f"{sample['potencial_rhe_v']:.6g}," if sample['potencial_rhe_v']
               is not None else ",") +
            f"{c['c_prima_mf']:.6g},{c['c_serie_mf']:.6g},"
            + (f"{c['elemento_mf']:.6g}" if c["elemento_mf"] else "")
            + "," + (f"{c['arco_mf']:.6g}" if c["arco_mf"] else "")
            + "," + (f"{c['arco_brug_mf']:.6g}" if c["arco_brug_mf"] else "")
            + "," + (f"{fast['hm_mf']:.6g},{fast['tau_s']:.6g}" if fast
                     else ",")
            + "," + (f"{slow['hm_mf']:.6g},{slow['tau_s']:.6g}" if slow
                     else ",")
            + f",{c['mf_cm2']:.6g},{c['f_g']:.6g},"
            + (f"{c['f_g_elemento']:.6g}" if c["f_g_elemento"] else "")
            + f",{c['fase_grados']:.6g},"
            + (f"{c['tau0_s']:.6g}" if c["tau0_s"] else "")
            + "," + (f"{c['f0_hz']:.6g}" if c["f0_hz"] else "")
            + f",{c['c_prima_f_hz']:.6g}")
    (OUT / "capacitancias.csv").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")

    summary = {
        "condiciones": {
            "electrolito": session.electrode.electrolyte,
            "electrolito_nota": (
                "dado por el usuario. No entra en ninguna impedancia "
                "medida; entra por el pH, y del pH sale la conversión a "
                "RHE"),
            "ph": session.electrode.ph,
            "ph_nota": (
                "de la base del paquete, y sale de la ACTIVIDAD y no de "
                "-log c: el primer protón del sulfúrico está disociado "
                "del todo y el segundo sólo en parte"),
            "referencia": session.electrode.reference,
            "referencia_nota": (
                "el usuario no la nombró en esta serie; se usa la de su "
                "serie anterior. Toda la columna RHE depende de ello"),
            "area_cm2": area,
            "area_suposicion": (
                f"disco de carbón vítreo de {DISC_MM:g} mm, dado por el "
                "usuario. Es área GEOMÉTRICA: la electroquímicamente "
                "activa de un carbón poroso es de diez a mil veces mayor"),
            "masa_mg": MASS_MG,
            "masa_nota": (
                f"{MASS_MG:g} mg, del protocolo del usuario: 20 uL de una "
                "tinta de 5 mg de muestra en 100 uL de isopropanol y 10 uL "
                "de Nafion, o sea 20/110 x 5 mg. Es NOMINAL: sale del "
                "volumen depositado y no de pesar el electrodo seco"),
            "carga_mg_cm2": MASS_MG / area,
            "carga_nota": (
                "de diez a cien veces la carga habitual sobre un carbón "
                "vítreo. Una película gruesa tiene una resistencia iónica "
                "en profundidad que el barrido no atraviesa a 0.01 Hz, y "
                "eso explica por sí solo que ninguna muestra llegue al "
                "régimen capacitivo"),
            "potenciales_continuos_v": biases,
            "potenciales_rhe_v": rhe,
            "potenciales_rhe_como": rhe_note,
            "potenciales_nota": (
                "los cuatro espectros se midieron a potenciales distintos "
                "—0 V frente al OCP de cada muestra— así que una diferencia "
                "en la resistencia de transferencia de carga puede ser el "
                "punto de polarización y no la química"),
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
        print("  " + " | ".join(f"{value:>18s}" for value in row))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
