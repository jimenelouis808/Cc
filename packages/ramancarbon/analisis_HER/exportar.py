"""HER on FeSe@C (N- and Se-doped carbon) pyrolysed at 900 °C, in
0.5 M H2SO4.

Usage::

    uv run python analisis_HER/exportar.py <dir con los .mpr>

Two linear sweeps of the same electrode, at 1 and 5 mV/s, 20 µg on a
3 mm glassy-carbon disc against Ag/AgCl (KCl sat.).

Read the warnings before the numbers. Both curves report a NEGATIVE
onset overpotential, which is impossible, and neither carries an
uncompensated resistance — and at the benchmark the resistance that
would account for the whole reported overpotential is one an ordinary
cell reaches. Those two together mean the overpotentials below are not
yet a measurement of the catalyst.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from ramancarbon.echem.curve import Electrode
from ramancarbon.echem.evaluate import BENCHMARK_DENSITIES
from ramancarbon.gui.echem_state import EchemSession
from ramancarbon.gui.figures_echem import CompositeFigure
from ramancarbon.gui.theme import PALETTES

OUT = Path(__file__).resolve().parent

MASS_MG = 0.020
DISC_MM = 3.0
ELECTROLYTE = "H2SO4_0.5M"
REFERENCE = "Ag/AgCl_sat"

#: Uncompensated resistance, in ohm, from the impedance of THIS cell.
#: The three spectra give 1.877, 1.917 and 1.928; the correction at
#: 10 mA/cm² is 1.3 mV, so it changes little and that is itself the
#: result: the overpotential is the electrode's.
RESISTANCE_OHM = 1.9

#: Legends for the three spectra, which the file names do not give.
PEIS_LABELS = {
    "PEIS_1_inicial": "inicial (OCV)",
    "PEIS_2_despues_OCV": "tras LSV (OCV)",
    "PEIS_3_despues_m0.35V": "tras LSV (−0.35 V)",
}

#: Scan rate in V/s by what the file name says. Not guessed from the
#: data: a rate typed wrong is a rate wrong in everything downstream,
#: and these files do carry it in their own sweep, so the two are
#: cross-checked below.
RATES = {"1mV": 0.001, "5mV": 0.005}


def rate_for(path: Path) -> float:
    for key, value in RATES.items():
        if key in path.name:
            return value
    raise SystemExit(f"no sé a qué velocidad se midió {path.name}")


def build(folder: Path) -> EchemSession:
    session = EchemSession(name="FeSe@C (N,Se) 900 °C")
    session.electrode = Electrode(
        mass_mg=MASS_MG,
        area_cm2=float(np.pi * (0.1 * DISC_MM / 2.0) ** 2),
        electrolyte=ELECTROLYTE,
        reference=REFERENCE,
        # From the three impedance spectra of this same cell: 1.88, 1.88
        # and 1.93 ohm. Measured, not assumed, which is the whole point
        # of having run them.
        resistance_ohm=RESISTANCE_OHM,
    )
    session.reaction = "HER"
    for path in sorted(folder.glob("LSV*.mpr")):
        rate = rate_for(path)
        if not session.add_lsv_to_series(path, scan_rate=rate):
            print(f"  no se pudo leer {path.name}")
    # The holds, when they are in the same folder.
    for path in sorted(folder.glob("CP_*.mpr")):
        session.add_hold(path, label=path.stem)
    # And the impedance of the same cell, which is where the resistance
    # above came from and what says the electrode changed rather than
    # the cell.
    for path in sorted(folder.glob("PEIS*.mpr")):
        session.add_eis_to_series(path, label=PEIS_LABELS.get(
            path.stem, path.stem))
    return session


def main(folder: Path) -> None:
    session = build(folder)
    curves = session.lsv_curves()
    if not curves:
        raise SystemExit("no se cargó ninguna curva")
    print(f"{len(curves)} curvas, pH {session.electrode.ph}")
    # What makes the two sweeps not a comparison of scan rate alone.
    # Here it is one thing and it is large: the 5 mV/s method began at
    # the open-circuit potential, so it opens 483 mV anodic of the
    # 1 mV/s one, which began at 0 V as asked.
    for note in session.lsv_series_notes():
        print(f"  aviso: {note}")

    # The rate the file NAME claims against the rate the sweep actually
    # ran at, read from the clock in the file. A `Voltammogram` carries
    # no time column, so this goes back to the raw record: the potential
    # step per recorded POINT is not the scan rate and comes out at 1 mV
    # for both of these files, which would pass a check that is not one.
    from ramancarbon.echem.biologic import read_mpr

    checks = []
    for path, curve in zip(sorted(folder.glob("LSV*.mpr")), curves):
        columns = read_mpr(path).columns
        time = np.asarray(columns["time/s"], dtype=float)
        potential = np.asarray(columns["Ewe/V"], dtype=float)
        step = np.diff(time)
        measured = float(np.median(np.diff(potential) / np.where(
            step == 0, np.nan, step)))
        checks.append((path.name, 1e3 * curve.scan_rate, measured))

    rows = session.lsv_rate_rows()
    lines = ["# HER: FeSe@C dopado con N y Se, 900 °C, en H2SO4 0.5 M",
             ""]
    lines.append(f"{MASS_MG * 1e3:.0f} µg sobre carbón vítreo de {DISC_MM:g} mm "
                 f"({session.electrode.area_cm2:.5g} cm²), referencia "
                 f"{REFERENCE}, pH {session.electrode.ph:g}.")
    lines.append("")
    lines.append("## Sobrepotenciales por velocidad de barrido")
    lines.append("")
    lines.append("v (mV/s) | " + " | ".join(
        f"η a {level:g} mA/cm² (mV)" for level in BENCHMARK_DENSITIES)
        + " | Tafel (mV/dec) | η de inicio (mV)")
    lines.append(" | ".join(["---"] * (len(BENCHMARK_DENSITIES) + 3)))
    for row in rows:
        lines.append(" | ".join(row))
    lines.append("")
    lines.append("La fila «dispersión» es el resultado, no un detalle: una "
                 "curva de polarización se supone que ES el estado "
                 "estacionario, así que dos velocidades tienen que caer una "
                 "sobre otra.")
    lines.append("")
    lines.append("## Las ventanas que se barrieron de verdad")
    lines.append("")
    lines.append("Leídas de la columna de potencial COMANDADO, no de la "
                 "medida: la medida abre en el potencial de reposo hasta que "
                 "el lazo de control tira de la celda.")
    lines.append("")
    lines.append("v (mV/s) | ventana comandada (V vs ref.) | puntos "
                 "descartados | potencial de reposo (V)")
    lines.append(" | ".join(["---"] * 4))
    for curve in curves:
        window = curve.metadata.get("ventana_comandada_v")
        skip = curve.metadata.get("puntos_de_asentamiento")
        rest = curve.metadata.get("potencial_de_reposo_v")
        lines.append(
            f"{1e3 * curve.scan_rate:g} | "
            + (f"{window[0]:+.4f} a {window[1]:+.4f}" if window else "—")
            + f" | {skip or 0} | "
            + (f"{rest:+.4f}" if rest is not None else "—"))
    lines.append("")
    (OUT / "tablas.md").write_text("\n".join(lines), encoding="utf-8")

    with (OUT / "velocidades.csv").open("w", newline="",
                                        encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["v_mV_s"]
                        + [f"eta_{level:g}_mA_cm2_mV"
                           for level in BENCHMARK_DENSITIES]
                        + ["tafel_mV_dec", "eta_inicio_mV"])
        writer.writerows(rows)

    report: list[str] = []
    for curve, analysis in zip(curves, session.lsv_analyses()):
        if analysis is None:
            continue
        report.append("=" * 72)
        report.append(f"{curve.name}  ({1e3 * curve.scan_rate:g} mV/s)")
        window = curve.metadata.get("ventana_comandada_v")
        if window:
            report.append(f"  ventana COMANDADA: {window[0]:+.4f} a "
                          f"{window[1]:+.4f} V vs la referencia")
        for note in curve.metadata.get("avisos", []):
            report.append(f"  ⚠ {note}")
        report.append(analysis.summary())
        report.append("")
    report.append("=" * 72)
    report.append("Velocidad declarada contra la medida en el reloj:")
    for name, declared, measured in checks:
        report.append(f"  {name}: {declared:g} mV/s declarados, "
                      f"{1e3 * measured:+.4g} mV/s medidos")
    (OUT / "informe.txt").write_text("\n".join(report), encoding="utf-8")

    # The curves themselves, as drawn.
    for curve, analysis in zip(curves, session.lsv_analyses()):
        if analysis is None or analysis.curve_j is None:
            continue
        stem = f"{1e3 * curve.scan_rate:g}mVs"
        with (OUT / f"curva_{stem}.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["j_mA_cm2", "eta_V"])
            for value, over in zip(np.abs(analysis.curve_j),
                                   analysis.curve_eta):
                writer.writerow([f"{value:.8g}", f"{over:.8g}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    session.figure_technique = "catalysis"
    spec = CompositeFigure(
        panels=["her_medida", "her_velocidades", "her_polarizacion",
                "her_tafel", "her_retencion", "eis_bode"],
        columns=2, width_in=7.2, panel_height_in=2.9, dpi=600)
    session.composite = spec
    figure = plt.figure(figsize=spec.figsize())
    for problem in session.draw_figure(figure, PALETTES["claro"]):
        print("  panel con problema:", problem)
    for suffix in (".png", ".pdf"):
        figure.savefig(OUT / f"her{suffix}", dpi=spec.dpi, bbox_inches="tight")
    plt.close(figure)
    session.save_figure_data(OUT / "figura")

    summary = {
        "condiciones": {
            "electrolito": session.electrode.electrolyte,
            "ph": session.electrode.ph,
            "referencia": REFERENCE,
            "masa_mg": MASS_MG,
            "area_cm2": session.electrode.area_cm2,
            "reaccion": session.reaction,
            "resistencia_no_compensada_ohm": None,
        },
        "curvas": [],
    }
    for curve, analysis in zip(curves, session.lsv_analyses()):
        if analysis is None:
            continue
        summary["curvas"].append({
            "velocidad_v_s": curve.scan_rate,
            "eta_por_densidad": {
                str(level): value
                for level, value in (analysis.overpotentials or {}).items()},
            "eta_inicio_v": analysis.onset_overpotential,
            "tafel_mv_dec": (analysis.tafel.slope_mv_per_decade
                             if analysis.tafel and analysis.tafel.valid
                             else None),
            "tafel_decadas": (analysis.tafel.decades
                              if analysis.tafel else None),
            "j0_ma_cm2": (analysis.tafel.exchange_current_density
                          if analysis.tafel and analysis.tafel.valid
                          else None),
            "j_max_ma_cm2": float(np.max(np.abs(analysis.curve_j)))
            if analysis.curve_j is not None else None,
            "avisos": analysis.warnings,
        })
    (OUT / "resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito en {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
