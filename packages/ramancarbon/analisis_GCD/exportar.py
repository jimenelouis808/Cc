"""Analyse the five GCPL files of the FeSe sample pyrolysed at 900 °C.

Usage::

    uv run python analisis_GCD/exportar.py <dir con los .mpr>

The files are five current steps of one schedule, run back to back over
sixteen hours, in KOH 3 M on a 3 mm glassy-carbon disc with 20 µg
deposited. The fifth one charges at 60 µA and discharges at 20, which
was a mistake in the schedule: such a pair has no single current
density, so it is analysed, reported, and kept out of the two plots
whose x axis is a current.

Nothing here draws matplotlib by hand. The panels come from the GUI's
own catalogue, so the figure and the window cannot drift apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ramancarbon.echem.curve import Electrode
from ramancarbon.echem.kinetics import fade_model
from ramancarbon.gui.echem_state import EchemSession, _discharge_current
from ramancarbon.gui.figures_echem import CompositeFigure
from ramancarbon.gui.theme import PALETTES

OUT = Path(__file__).resolve().parent

#: The electrode, as the user described it.
MASS_MG = 0.020
DISC_MM = 3.0
ELECTROLYTE = "KOH_3M"
REFERENCE = "Ag/AgCl_sat"

#: The order the schedule ran in, which is NOT the order of the
#: currents: the files are consecutive in the clock of the potentiostat,
#: so the lowest current is also the least degraded. That confounds rate
#: with degradation and the report says so.
SCHEDULE = ("GCPL_01", "GCPL_02", "GCPL_03", "GCPL_04", "GCPL_05")


def build(folder: Path) -> EchemSession:
    session = EchemSession(name="FeSe 900 °C")
    session.electrode = Electrode(
        mass_mg=MASS_MG,
        area_cm2=float(np.pi * (0.1 * DISC_MM / 2.0) ** 2),
        electrolyte=ELECTROLYTE,
        reference=REFERENCE,
    )
    for path in sorted(folder.glob("*.mpr")):
        if not session.add_gcd_to_series(path):
            print(f"  no se pudo leer {path.name}")
    return session


def tables(session: EchemSession) -> str:
    lines = ["# GCD: FeSe 900 °C en KOH 3 M", ""]
    lines.append(f"Masa activa {MASS_MG * 1e3:.0f} µg sobre un disco de "
                 f"carbón vítreo de {DISC_MM:g} mm "
                 f"({session.electrode.area_cm2:.5g} cm²), "
                 f"referencia {REFERENCE}, ventana −0.6 a +0.4 V.")
    lines.append("")
    lines.append("## Un punto por corriente")
    lines.append("")
    header = ("corriente | C (µF) | C (F/g) | C (µF/cm²) | R (Ω) | "
              "eficiencia (%) | fuga (µA) | asimétrica")
    lines.append(header)
    lines.append(" | ".join(["---"] * 8))
    for row in session.gcd_rows():
        lines.append(" | ".join(row))
    lines.append("")
    lines.append("## Lo que mil ciclos le hacen a cada uno")
    lines.append("")
    lines.append("corriente | ciclos | C inicial (µF) | C final (µF) | "
                 "retención (%) | pérdida (%/ciclo) | eficiencia media (%) | "
                 "forma")
    lines.append(" | ".join(["---"] * 8))
    for row in session.gcd_cycling_rows():
        lines.append(" | ".join(row))
    lines.append("")

    rows = session.gcd_interpolated()
    if rows:
        lines.append("## El archivo asimétrico, comparado donde se puede")
        lines.append("")
        lines.append("curva | corriente de descarga (mA) | medido (F/g) | "
                     "interpolado de los simétricos (F/g) | diferencia")
        lines.append(" | ".join(["---"] * 5))
        for label, milliamps, measured, estimate in rows:
            lines.append(
                f"{label} | {milliamps:g} | {measured:.4g} | "
                f"{estimate:.4g} | {100 * (measured / estimate - 1):+.1f} %")
        lines.append("")
    return "\n".join(lines)


def main(folder: Path) -> None:
    session = build(folder)
    curves = session.gcd_curves()
    if not curves:
        raise SystemExit("no se cargó ninguna curva")
    print(f"{len(curves)} curvas")

    OUT.mkdir(exist_ok=True)
    (OUT / "tablas.md").write_text(tables(session), encoding="utf-8")

    # The per-curve table, as CSV.
    import csv
    with (OUT / "serie.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corriente", "C_uF", "C_F_g", "C_uF_cm2", "R_ohm",
                         "eficiencia_pct", "fuga_uA", "asimetrica"])
        writer.writerows(session.gcd_rows())
    with (OUT / "ciclado.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corriente", "ciclos", "C_inicial_uF", "C_final_uF",
                         "retencion_pct", "perdida_pct_ciclo",
                         "eficiencia_media_pct", "forma"])
        writer.writerows(session.gcd_cycling_rows())

    # Every cycle of every file, which is where the content is.
    for curve, cycling in zip(curves, session.gcd_cyclings()):
        if cycling is None or not cycling.cycles:
            continue
        stem = session.gcd_label(curve).replace("/", "_").replace(" ", "")
        with (OUT / f"ciclos_{stem}.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ciclo", "q_C", "C_F", "eficiencia"])
            for index, number in enumerate(cycling.cycles):
                efficiency = (cycling.efficiency[index]
                              if index < len(cycling.efficiency) else "")
                writer.writerow([number, f"{cycling.capacity_c[index]:.8g}",
                                 f"{cycling.capacitance_f[index]:.8g}",
                                 f"{efficiency:.6g}" if efficiency != ""
                                 else ""])

    # The report, with everything the analysis refuses to state silently.
    report: list[str] = []
    for curve, analysis, cycling in zip(curves, session.gcd_analyses(),
                                        session.gcd_cyclings()):
        if analysis is None:
            continue
        report.append("=" * 70)
        report.append(f"{session.gcd_label(curve)}  —  {curve.name}")
        report.append(analysis.summary())
        if analysis.resistance_how:
            report.append(f"  R: {analysis.resistance_how}")
        if cycling is not None and cycling.cycles:
            report.append("")
            report.append(cycling.summary())
        report.append("")
    (OUT / "informe.txt").write_text("\n".join(report), encoding="utf-8")

    # The figure, from the GUI's own catalogue.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    session.figure_technique = "gcd"
    spec = CompositeFigure(
        panels=["gcd_series", "gcd_velocidad", "gcd_ciclado",
                "gcd_eficiencia", "gcd_fuga", "gcd_ragone"],
        columns=2, width_in=7.2, panel_height_in=2.9, dpi=600)
    session.composite = spec
    figure = plt.figure(figsize=spec.figsize())
    problems = session.draw_figure(figure, PALETTES["claro"])
    for problem in problems:
        print("  panel con problema:", problem)
    for suffix in (".png", ".pdf"):
        figure.savefig(OUT / f"gcd_serie{suffix}", dpi=spec.dpi,
                       bbox_inches="tight")
    plt.close(figure)

    # And the numbers behind every panel.
    session.save_figure_data(OUT / "figura")

    summary = {
        "condiciones": {
            "electrolito": ELECTROLYTE,
            "referencia": REFERENCE,
            "masa_mg": MASS_MG,
            "area_cm2": session.electrode.area_cm2,
            "ventana_v": [-0.6, 0.4],
            "politica_asimetria": session.gcd_asymmetry,
        },
        "curvas": [],
    }
    for curve, analysis, cycling in zip(curves, session.gcd_analyses(),
                                        session.gcd_cyclings()):
        if analysis is None:
            continue
        entry = {
            "etiqueta": session.gcd_label(curve),
            "corriente_descarga_a": _discharge_current(curve),
            "corriente_carga_a": analysis.charge_current_a,
            "asimetrica": analysis.asymmetric,
            "capacitancia_f": analysis.capacitance_f,
            "especifica_f_g": analysis.specific_f_per_g,
            "especifica_uf_cm2": analysis.specific_uf_per_cm2,
            "resistencia_ohm": analysis.resistance_ohm,
            "resistencia_como": analysis.resistance_how,
            "eficiencia_culombica": analysis.coulombic_efficiency,
            "fuga_a": analysis.leakage_a,
            "energia_wh_kg": analysis.energy_wh_per_kg,
            "potencia_w_kg": analysis.power_w_per_kg,
            "potencia_maxima_w_kg": analysis.max_power_w_per_kg,
            "avisos": analysis.warnings,
        }
        if cycling is not None and cycling.cycles:
            entry["ciclos"] = len(cycling.cycles)
            entry["retencion"] = cycling.retention
            try:
                entry["forma_desvanecimiento"] = fade_model(
                    cycling.cycles, cycling.capacity_c).model
            except (ValueError, ZeroDivisionError):
                entry["forma_desvanecimiento"] = None
        summary["curvas"].append(entry)
    summary["interpolacion"] = [
        {"etiqueta": label, "mA": milliamps, "medido_f_g": measured,
         "interpolado_f_g": estimate}
        for label, milliamps, measured, estimate
        in session.gcd_interpolated()
    ]
    (OUT / "resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito en {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
