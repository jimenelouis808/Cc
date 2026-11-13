"""Analyse the twenty-one EC-Lab files of the electrode in 0.5 M H2SO4.

Usage::

    uv run python analisis_H2SO4/exportar.py <dir con los .mpr> [masa_mg] [area_cm2]

The mass and the area are optional because they are not in the files and
were not given with them. Everything that does not need them is reported
either way; F/g, mAh/g, Wh/kg and the ECSA appear only when they are
passed, and the report says which ones are missing rather than quoting a
capacitance per gram of nothing.

The headers of the files name a different cell from the one that was
measured. The conditions below are the ones the user stated, and they
override the headers: 0.5 M H2SO4, Ag/AgCl in saturated KCl. The pH is
not stated anywhere and is deduced from the molarity by the package's own
electrolyte table.

Nothing here draws matplotlib by hand. The panels come from the GUI's own
catalogue, so a figure and the window cannot drift apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ramancarbon.echem.biologic import read_eclab
from ramancarbon.echem.curve import Electrode
from ramancarbon.echem.cv import analyse_rate_study
from ramancarbon.echem.eis import analyse_eis
from ramancarbon.echem.gcd import cycling_from_curve
from ramancarbon.echem.kinetics import fade_model
from ramancarbon.echem.report import analyse_sample
from ramancarbon.gui.echem_state import EchemSession
from ramancarbon.gui.figures_echem import CompositeFigure
from ramancarbon.gui.theme import PALETTES

OUT = Path(__file__).resolve().parent

#: The cell, as the user stated it. The file headers say otherwise and
#: the user's statement wins: a header travels with a template.
ELECTROLYTE = "H2SO4_0.5M"
REFERENCE = "Ag/AgCl_sat"

#: The galvanostatic files in the order the schedule ran them, which is
#: the order of their names and NOT the order of their currents. It
#: decides the sign of the degradation bias on the rate retention and
#: nothing inside a .mpr records it.
GCD_SCHEDULE = (
    "GCPL__prueba_ventana_1-0.2_C01",
    "GCPL__prueba_ventana_1-0_01_GCPL_C01",
    "GCPL__prueba_ventana_1-0_02_GCPL_C01",
    "GCPL__prueba_ventana_1-0_03_GCPL_C01",
    "GCPL__prueba_ventana_1-0_04_GCPL_C01",
)


def classify(folder: Path) -> dict[str, list[Path]]:
    """Split the folder into the four experiments it holds."""
    groups: dict[str, list[Path]] = {
        "cv": [], "ecsa": [], "gcd": [], "peis": [], "vacios": []}
    for path in sorted(folder.glob("*.mpr")):
        stem = path.stem
        if stem.lower().startswith("cv_ecsa"):
            groups["ecsa"].append(path)
        elif stem.lower().startswith("cv_multp"):
            groups["cv"].append(path)
        elif "peis" in stem.lower():
            groups["peis"].append(path)
        elif stem.lower().startswith("gcpl"):
            groups["gcd"].append(path)
    return groups


def read_group(paths: list[Path], electrode: Electrode) -> tuple[list, list[str]]:
    """Read a group, keeping the files that are actually readable."""
    curves, problems = [], []
    for path in paths:
        try:
            curve, _ = read_eclab(path)
        except Exception as exc:  # noqa: BLE001 - the reason is the output
            problems.append(f"{path.name}: {exc}")
            continue
        curve.electrode = electrode
        curve.name = path.stem
        curves.append(curve)
    return curves, problems


#: The three windows, in volts against the reference, and what the water
#: stability window says about each. Written down rather than measured
#: because they are what was PROGRAMMED; the measured excursions differ
#: from them by the IR drop, which is a different number.
WINDOWS = (
    ("CV, serie ancha", -0.600, 1.200),
    ("CV, ECSA", -0.050, 0.100),
    ("GCD", -0.200, 1.000),
)

#: Thermodynamic stability of water at 25 degrees, in volts vs RHE.
WATER_V = (0.0, 1.229)


def window_table(electrode: Electrode) -> list[str]:
    """Each window on the RHE scale, against the water window.

    This is the whole reason the pH had to be deduced: a window quoted
    against Ag/AgCl says nothing about whether it splits water, and the
    same numbers in two electrolytes are two different experiments.
    """
    lines = [
        "## Las ventanas frente al RHE "
        f"(el agua es estable entre {WATER_V[0]:.0f} y {WATER_V[1]:.3f} V)",
        "",
        "| experimento | vs Ag/AgCl sat. | vs RHE | margen inferior | "
        "margen superior |",
        "|---|---|---|---|---|",
    ]
    for name, low, high in WINDOWS:
        values, _ = electrode.to_reference([low, high], "RHE")
        if values is None:
            lines.append(f"| {name} | {low:+.3f} … {high:+.3f} V | "
                         "no convertible | | |")
            continue
        bottom = float(values[0]) - WATER_V[0]
        top = WATER_V[1] - float(values[1])
        lines.append(
            f"| {name} | {low:+.3f} … {high:+.3f} V | "
            f"{float(values[0]):+.3f} … {float(values[1]):+.3f} V | "
            f"{bottom:+.3f} V | {top:+.3f} V |"
        )
    lines.append("")
    lines.append("Un margen negativo es electrólisis: por debajo de 0 V se "
                 "desprende hidrógeno y por encima de 1.229 V, oxígeno.")
    lines.append("")
    return lines


def tables(summary: dict, electrode: Optional[Electrode] = None) -> str:
    """``tablas.md``, built from the same structure as ``resumen.json``."""
    out: list[str] = ["# Tablas — electrodo en H2SO4 0.5 M", ""]
    conditions = summary["condiciones"]
    out.append(
        f"Electrolito {conditions['electrolito']}, pH {conditions['ph']} "
        "(deducido de la molaridad), referencia Ag/AgCl KCl sat. "
        "Masa activa "
        + ("no dada" if conditions["masa_mg"] is None
           else f"{conditions['masa_mg']:g} mg")
        + ", área "
        + ("no dada" if conditions["area_cm2"] is None
           else f"{conditions['area_cm2']:g} cm²") + "."
    )
    out.append("")
    if electrode is not None:
        out.extend(window_table(electrode))

    rate = summary.get("velocidad")
    if rate:
        out.append("## Capacidad de velocidad del GCD")
        out.append("")
        head = (f"Ciclo representativo: {rate['politica']}. Retención "
                f"{100 * rate['retencion']:.1f} % a "
                f"{rate['razon_corriente']:.3g}× la corriente")
        if rate.get("retencion_sin_corregir_ir"):
            head += (f"; sin descontar la caída IR saldría "
                     f"{100 * rate['retencion_sin_corregir_ir']:.1f} %, así "
                     "que esa diferencia es la resistencia de la celda y no "
                     "el electrodo")
        out.append(head + ".")
        out.append("")
        out.append("| I (mA) | C (µF) | ventana (V) | IR (mV) | "
                   "IR / excursión | ± entre ciclos | retención interna |")
        out.append("|---|---|---|---|---|---|---|")
        for point in rate["puntos"]:
            drop, window = point["ir_v"], point["ventana_v"]
            total = window + drop
            out.append(
                f"| {1e3 * abs(point['corriente_a']):.4g} | "
                f"{1e6 * point['capacitancia_f']:.4g} | {window:.4g} | "
                f"{1e3 * drop:.4g} | {100 * drop / total:.1f} % | "
                f"{100 * point['dispersion']:.1f} % | "
                f"{100 * point['retencion_interna']:.1f} % |"
            )
        out.append("")
        if rate["m"] is not None:
            out.append(
                f"`C = a·I^-m` con m = {rate['m']:.3f} ± "
                f"{rate['m_error']:.3f}, R² = {rate['r2']:.3f}."
            )
        fade = rate.get("desvanecimiento_serie")
        if fade and fade > 1.01:
            out.append(
                f"Los archivos GANAN un {100 * (fade - 1):.1f} % compuesto "
                "dentro de sus propios ciclos: activación, no degradación."
            )
        out.append("")

    if summary.get("peis"):
        out.append("## Impedancia")
        out.append("")
        out.append("| archivo | E (V vs Ag/AgCl) | amplitud | R_s (Ω) | "
                   "C′(0.1 Hz) (mF) | codo (Hz) | Kramers-Kronig | circuito |")
        out.append("|---|---|---|---|---|---|---|---|")
        for row in summary["peis"]:
            capacitance = row.get("capacitancia_f")
            out.append(
                f"| `{row['archivo']}` | {row['potencial_cd_v']:+.4f} | "
                f"{row['amplitud_mv']:.1f} mV | {row['rs_ohm']:.4g} | "
                + (f"{1e3 * capacitance:.4g}" if capacitance else "—")
                + f" | {row['codo_hz']:.4g} | "
                + ("sí" if row["kk"] else "**no**")
                + f" | `{row['circuito']}` |"
            )
        out.append("")
        out.append("La R_ct no está en la tabla a propósito: ninguno de los "
                   "espectros cierra el semicírculo, así que la que devuelva "
                   "un ajuste la pone el modelo y no la medida.")
        out.append("")

    ecsa = summary.get("ecsa")
    if ecsa and ecsa.get("cdl_f"):
        low, high = ecsa["ecsa_intervalo_cm2"] or (ecsa["ecsa_cm2"],) * 2
        out.append("## C_dl y ECSA (serie estrecha)")
        out.append("")
        out.append(
            f"C_dl = {1e6 * ecsa['cdl_f']:.4g} µF, ECSA = "
            f"{ecsa['ecsa_cm2']:.3g} cm² (entre {low:.3g} y {high:.3g} según "
            "la Cs que se adopte: ese factor de tres es irreducible y pasa "
            "entero al área)."
        )
        out.append("")
        out.append("| E (V) | b | error | R² |")
        out.append("|---|---|---|---|")
        for entry in ecsa["b"]:
            out.append(f"| {entry['potencial_v']:+.3f} | {entry['b']:.3f} | "
                       f"{entry['error']:.3f} | {entry['r2']:.4f} |")
        out.append("")

    if summary.get("archivos_ilegibles"):
        out.append("## Archivos ilegibles")
        out.append("")
        for bad in summary["archivos_ilegibles"]:
            out.append(f"- `{bad}`")
        out.append("")
    return "\n".join(out) + "\n"


def main(folder: Path, mass_mg: Optional[float] = None,
         area_cm2: Optional[float] = None) -> None:
    electrode = Electrode(mass_mg=mass_mg, area_cm2=area_cm2,
                          electrolyte=ELECTROLYTE, reference=REFERENCE)
    print(electrode.describe())
    print(f"pH deducido de la molaridad: {electrode.ph}")

    groups = classify(folder)
    problems: list[str] = []
    cv_curves, bad = read_group(groups["cv"], electrode)
    problems += bad
    ecsa_curves, bad = read_group(groups["ecsa"], electrode)
    problems += bad
    gcd_curves, bad = read_group(groups["gcd"], electrode)
    problems += bad
    peis_curves, bad = read_group(groups["peis"], electrode)
    problems += bad
    cv_curves.sort(key=lambda c: c.scan_rate)
    ecsa_curves.sort(key=lambda c: c.scan_rate)
    order = {name: n for n, name in enumerate(GCD_SCHEDULE)}
    gcd_curves.sort(key=lambda c: order.get(c.name, 99))
    for group, curves in (("CV", cv_curves), ("ECSA", ecsa_curves),
                          ("GCD", gcd_curves), ("PEIS", peis_curves)):
        print(f"  {group}: {len(curves)} archivos")
    for problem in problems:
        print(f"  ILEGIBLE {problem}")

    summary: dict[str, object] = {
        "condiciones": {
            "electrolito": electrode.electrolyte,
            "ph": electrode.ph,
            "referencia": REFERENCE,
            "masa_mg": mass_mg,
            "area_cm2": area_cm2,
        },
        "archivos_ilegibles": problems,
    }

    # -- the main sample: the wide-window CVs, the GCD series, the first
    # -- impedance. One report, because they are one electrode.
    result = analyse_sample(
        name="electrodo en H2SO4 0.5 M",
        cv=cv_curves[0] if cv_curves else None,
        rate_series=cv_curves or None,
        gcd=gcd_curves[-1] if gcd_curves else None,
        gcd_series=gcd_curves or None,
        eis=peis_curves[0] if peis_curves else None,
    )
    (OUT / "informe.txt").write_text(result.report(), encoding="utf-8")
    print(f"  escrito informe.txt ({len(result.report().splitlines())} líneas)")

    # -- the narrow-window series is a C_dl measurement and nothing else,
    # -- so it gets its own rate study with non_faradaic set.
    ecsa = None
    if len(ecsa_curves) >= 2:
        ecsa = analyse_rate_study(ecsa_curves, non_faradaic=True)
        (OUT / "informe_ecsa.txt").write_text(ecsa.summary(), encoding="utf-8")
        summary["ecsa"] = {
            "velocidades_mv_s": [1e3 * r for r in ecsa.rates],
            "cdl_f": ecsa.cdl_farads,
            "ecsa_cm2": ecsa.ecsa_cm2,
            "ecsa_intervalo_cm2": ecsa.ecsa_range_cm2,
            "b": [{"potencial_v": b.potential, "b": b.b, "error": b.b_error,
                   "r2": b.r_squared} for b in ecsa.b_values],
            "avisos": ecsa.warnings,
        }

    # -- every impedance on its own: three states of the same cell.
    peis_rows = []
    peis_curves.sort(key=lambda c: (0 if "incial" in c.name.lower() else 1,
                                    c.metadata.get("potencial_cd_v") or 0.0))
    for curve in peis_curves:
        analysis = analyse_eis(curve, circuit="auto")
        (OUT / f"informe_peis_{curve.name[:28]}.txt").write_text(
            analysis.summary(), encoding="utf-8")
        peis_rows.append({
            "archivo": curve.name,
            "potencial_cd_v": curve.metadata.get("potencial_cd_v"),
            "amplitud_mv": curve.metadata.get("amplitud_mv"),
            "rs_ohm": analysis.series_resistance,
            "rs_como": analysis.series_resistance_how,
            "rct_ohm": analysis.charge_transfer_resistance,
            "r_ionica_ohm": analysis.ionic_resistance_ohm,
            "capacitancia_f": analysis.capacitance_f,
            "codo_hz": analysis.knee_frequency,
            "circuito": analysis.fit.circuit if analysis.fit else None,
            "forma": analysis.shape.describe() if analysis.shape else None,
            "kk": analysis.kk.passes if analysis.kk else None,
            "avisos": analysis.warnings,
        })
    summary["peis"] = peis_rows

    # -- the galvanostatic series, through the session so the figures come
    # -- from the window's own catalogue.
    session = EchemSession(name="electrodo en H2SO4 0.5 M")
    session.electrode = electrode
    for name in GCD_SCHEDULE:
        match = next((p for p in groups["gcd"] if p.stem == name), None)
        if match and not session.add_gcd_to_series(match):
            print(f"  no se pudo cargar {match.name}")
    for path in groups["gcd"]:
        if path.stem not in GCD_SCHEDULE:
            session.add_gcd_to_series(path)
    session.analyse()

    rate = session.gcd_rate_capability()
    if rate is not None:
        (OUT / "informe_velocidad.txt").write_text(rate.summary(),
                                                   encoding="utf-8")
        summary["velocidad"] = {
            "politica": rate.policy,
            "retencion": rate.retention,
            "razon_corriente": rate.current_ratio,
            "m": rate.exponent,
            "m_error": rate.exponent_error,
            "r2": rate.r_squared,
            "desvanecimiento_serie": rate.series_fade,
            "retencion_sin_corregir_ir": rate.retention_uncorrected,
            "retencion_corregida": rate.retention_corrected,
            "ohmico": rate.ohmic,
            "puntos": [
                {"corriente_a": p.current_a, "capacitancia_f": p.capacitance_f,
                 "ventana_v": p.window_v, "ir_v": p.ir_drop_v,
                 "ciclos": p.cycles, "dispersion": p.spread,
                 "retencion_interna": p.within_retention, "orden": p.order,
                 "especifica_f_g": p.specific_f_per_g}
                for p in rate.points
            ],
            "avisos": rate.warnings,
        }

    # -- csv tables
    def write_csv(name: str, header: str, rows: list[str]) -> None:
        (OUT / name).write_text("\n".join([header, *rows]) + "\n",
                                encoding="utf-8")

    if cv_curves:
        rows = []
        for curve in cv_curves:
            from ramancarbon.echem.cv import analyse_cv
            one = analyse_cv(curve)
            loop = one.capacitance_loop
            anodic = next((p for p in one.peaks if p.branch == "anódico"), None)
            cathodic = next((p for p in one.peaks if p.branch == "catódico"), None)
            rows.append(",".join([
                f"{1e3 * curve.scan_rate:g}",
                f"{1e3 * loop.farads:.6g}" if loop else "",
                f"{one.peak_separation:.6g}" if one.peak_separation else "",
                f"{anodic.potential:.6g}" if anodic else "",
                f"{1e6 * anodic.current:.6g}" if anodic else "",
                f"{cathodic.potential:.6g}" if cathodic else "",
                f"{1e6 * cathodic.current:.6g}" if cathodic else "",
                one.reversibility,
            ]))
        write_csv("cv_velocidades.csv",
                  "velocidad_mv_s,C_mF,dEp_V,Epa_V,ipa_uA,Epc_V,ipc_uA,reversibilidad", rows)

    if rate is not None:
        rows = [",".join([
            f"{1e3 * abs(p.current_a):g}",
            f"{1e6 * p.capacitance_f:.6g}",
            f"{p.window_v:.6g}",
            f"{1e3 * p.ir_drop_v:.6g}",
            str(p.cycles),
            f"{100 * p.spread:.4g}",
            f"{100 * p.within_retention:.4g}" if p.within_retention else "",
            str(p.order),
        ]) for p in rate.points]
        write_csv("gcd_velocidad.csv",
                  "I_mA,C_uF,ventana_V,IR_mV,ciclos,dispersion_pct,"
                  "retencion_interna_pct,orden", rows)

    rows = []
    for curve in gcd_curves:
        cycling = cycling_from_curve(curve)
        if not cycling.cycles:
            continue
        shape = ""
        try:
            shape = fade_model(cycling.cycles, cycling.capacity_c).model
        except (ValueError, ZeroDivisionError):
            shape = ""
        rows.append(",".join([
            curve.name,
            f"{1e3 * abs(float(np.median(cycling.current_a))):g}",
            str(len(cycling.cycles)),
            f"{1e6 * cycling.capacitance_f[0]:.6g}",
            f"{1e6 * cycling.capacitance_f[-1]:.6g}",
            f"{100 * cycling.retention:.4g}" if cycling.retention else "",
            f"{100 * cycling.fade_per_cycle:.4g}" if cycling.fade_per_cycle else "",
            f"{100 * float(np.mean(cycling.efficiency)):.4g}"
            if cycling.efficiency else "",
            shape,
        ]))
    write_csv("gcd_ciclado.csv",
              "archivo,I_mA,ciclos,C_inicial_uF,C_final_uF,retencion_pct,"
              "perdida_pct_ciclo,eficiencia_media_pct,forma", rows)

    for curve in peis_curves:
        rows = [",".join([f"{f:.8g}", f"{z.real:.8g}", f"{-z.imag:.8g}"])
                for f, z in zip(curve.frequency, curve.z)]
        write_csv(f"peis_{curve.name[:28]}.csv", "f_Hz,Zre_ohm,menosZim_ohm",
                  rows)

    # -- figures, from the catalogue
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(panels, stem, technique, height_in=2.9, columns=2):
        session.figure_technique = technique
        spec = CompositeFigure(panels=list(panels), columns=columns,
                               width_in=7.2, panel_height_in=height_in, dpi=600)
        session.composite = spec
        figure = plt.figure(figsize=spec.figsize())
        for problem in session.draw_figure(figure, PALETTES["claro"]):
            print("  panel con problema:", problem)
        for suffix in (".png", ".pdf"):
            figure.savefig(OUT / f"{stem}{suffix}", dpi=spec.dpi,
                           bbox_inches="tight")
        plt.close(figure)
        print(f"  escrita {stem}.{{png,pdf}}")

    session.rate_series = list(cv_curves)
    session.cv = cv_curves[0] if cv_curves else None
    session.eis = peis_curves[0] if peis_curves else None
    session.result = result

    draw(["gcd_series", "gcd_velocidad", "gcd_ciclado", "gcd_resistencia"],
         "gcd", "gcd")
    draw(["gcd_retencion_velocidad", "gcd_capacidad_eficiencia"],
         "gcd_velocidad", "gcd", height_in=3.2)
    if len(cv_curves) >= 2:
        draw(["cv_serie", "cv_capacitancia"], "cv", "rates", height_in=3.2)
    if peis_curves:
        draw(["eis_nyquist", "eis_bode"], "peis", "eis", height_in=3.2)

    (OUT / "tablas.md").write_text(tables(summary, electrode), encoding="utf-8")
    print("  escrito tablas.md")

    summary["avisos"] = result.warnings
    (OUT / "resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"escrito en {OUT}")


if __name__ == "__main__":
    mass = float(sys.argv[2]) if len(sys.argv) > 2 else None
    area = float(sys.argv[3]) if len(sys.argv) > 3 else None
    main(Path(sys.argv[1]).resolve(), mass, area)
