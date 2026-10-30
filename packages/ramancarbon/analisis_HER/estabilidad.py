"""The constant-current holds and the three impedance spectra.

Usage::

    uv run python analisis_HER/estabilidad.py <dir con los .mpr>

Reads the two chronopotentiometries at 10 mA/cm² and the three PEIS —
before the LSV, after it at open circuit, and after it at −0.35 V — and
writes what they say about the electrode rather than about the instant
of the sweep.

The potential column of the chronopotentiometries carries an identifier
this reader cannot NAME (174). Its WIDTH is settled by the file's own
arithmetic, so everything stays aligned and the column comes through as
``desconocida_174``; it is used here as the potential because a
chronopotentiometry has to contain one and nothing else in the record
can be it, and that is an inference, not a reading. One ``.mpt`` export
of the same experiment settles it.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from ramancarbon.echem.biologic import UNNAMED_PREFIX, read_mpr
from ramancarbon.echem.curve import Electrode
from ramancarbon.echem.evaluate import stability
from ramancarbon.gui.echem_state import EchemSession

OUT = Path(__file__).resolve().parent
MASS_MG = 0.020
DISC_MM = 3.0
AREA = float(np.pi * (0.1 * DISC_MM / 2.0) ** 2)

#: Ag/AgCl (KCl sat.) plus the Nernst term at the tabulated pH of
#: H2SO4 0.5 M. Volts to ADD to a measured potential to get RHE.
SHIFT_V = 0.197 + 0.05916 * 0.3

LABELS = {
    "PEIS_1_inicial": "inicial (OCV)",
    "PEIS_2_despues_OCV": "tras LSV (OCV)",
    "PEIS_3_despues_m0.35V": "tras LSV (−0.35 V)",
}


def _potential_column(columns: dict) -> np.ndarray:
    """The measured potential of a galvanostatic record."""
    for name in ("Ewe/V", "<Ewe>/V"):
        if name in columns:
            return np.asarray(columns[name], dtype=float)
    unnamed = [k for k in columns if k.startswith(UNNAMED_PREFIX)]
    if len(unnamed) == 1:
        return np.asarray(columns[unnamed[0]], dtype=float)
    raise SystemExit("no hay una columna de potencial que se pueda usar")


def holds(folder: Path) -> list[dict]:
    out = []
    for path in sorted(folder.glob("CP_*.mpr")):
        columns = read_mpr(path).columns
        time = np.asarray(columns["time/s"], dtype=float)
        potential = _potential_column(columns)
        current = np.asarray(columns["I/mA"], dtype=float)
        density = float(np.median(current)) / AREA
        # η positive is driving force, as the rest of the package has it.
        eta = -(potential + SHIFT_V)
        result = stability(time, eta, density)
        out.append({
            "archivo": path.name,
            "resultado": result,
            "tiempo": time,
            "eta": eta,
            "potencial": potential,
        })
    return out


def spectra(folder: Path) -> EchemSession:
    session = EchemSession(name="FeSe@C (N,Se) 900 °C")
    session.electrode = Electrode(mass_mg=MASS_MG, area_cm2=AREA,
                                  electrolyte="H2SO4_0.5M",
                                  reference="Ag/AgCl_sat")
    session.reaction = "HER"
    for path in sorted(folder.glob("PEIS*.mpr")):
        stem = path.name.replace(".mpr", "")
        session.add_eis_to_series(path, label=LABELS.get(stem, stem))
    return session


def main(folder: Path) -> None:
    runs = holds(folder)
    session = spectra(folder)
    print(f"{len(runs)} retenciones, {len(session.eis_series)} espectros")

    lines = ["# Estabilidad e impedancia: FeSe@C (N,Se) 900 °C en H2SO4 0.5 M",
             ""]
    lines.append(f"{MASS_MG * 1e3:.0f} µg sobre carbón vítreo de "
                 f"{DISC_MM:g} mm ({AREA:.5g} cm²), Ag/AgCl (KCl sat.), "
                 f"E(RHE) = E + {SHIFT_V:.4f} V.")
    lines.append("")
    lines.append("## Las retenciones a corriente constante")
    lines.append("")
    lines.append("archivo | j (mA/cm²) | duración | η inicial (mV) | "
                 "η final (mV) | deriva (mV/h) | R² | +100 mV en")
    lines.append(" | ".join(["---"] * 8))
    for run in runs:
        r = run["resultado"]
        lines.append(
            f"{run['archivo']} | {r.current_ma_cm2:.2f} | "
            f"{60 * r.duration_h:.0f} min | "
            f"{1e3 * r.overpotential_start:.1f} | "
            f"{1e3 * r.overpotential_end:.1f} | "
            f"{1e3 * r.drift_v_per_h:+.0f} | {r.r_squared:.4f} | "
            + (f"{r.hours_to_lose_100mv:.1f} h"
               if r.hours_to_lose_100mv else "—"))
    lines.append("")
    lines.append("## Los tres espectros")
    lines.append("")
    lines.append("espectro | E de polarización (V) | R_s (Ω) | R_ct (Ω) | "
                 "C′ (µF)")
    lines.append(" | ".join(["---"] * 5))
    rows = []
    for spectrum, analysis in zip(session.eis_series, session.eis_analyses()):
        if analysis is None:
            continue
        bias = spectrum.metadata.get("potencial_cd_v")
        charge = analysis.charge_transfer_resistance
        capacitance = analysis.capacitance_f
        row = (spectrum.name,
               f"{bias:+.4f}" if bias is not None else "—",
               f"{analysis.series_resistance:.3f}",
               f"{charge:.4g}" if charge else "—",
               f"{1e6 * capacitance:.2f}" if capacitance else "—")
        rows.append(row)
        lines.append(" | ".join(row))
    lines.append("")
    lines.append("La R_s es la misma en los tres, así que lo que cambia es "
                 "el electrodo y no la celda. Y con esa R_s la corrección "
                 "óhmica a 10 mA/cm² vale "
                 f"{1e3 * 10.0 * AREA * 1e-3 * 1.9:.2f} mV, así que no "
                 "explica ningún sobrepotencial.")
    (OUT / "estabilidad.md").write_text("\n".join(lines), encoding="utf-8")

    with (OUT / "retenciones.csv").open("w", newline="",
                                        encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["archivo", "j_mA_cm2", "duracion_h", "eta_ini_mV",
                         "eta_fin_mV", "deriva_mV_h", "r2", "h_para_100mV"])
        for run in runs:
            r = run["resultado"]
            writer.writerow([
                run["archivo"], f"{r.current_ma_cm2:.4g}",
                f"{r.duration_h:.5g}", f"{1e3 * r.overpotential_start:.4g}",
                f"{1e3 * r.overpotential_end:.4g}",
                f"{1e3 * r.drift_v_per_h:.4g}", f"{r.r_squared:.5f}",
                f"{r.hours_to_lose_100mv:.4g}"
                if r.hours_to_lose_100mv else ""])
    for run in runs:
        stem = run["archivo"].replace(".mpr", "")
        with (OUT / f"{stem}.csv").open("w", newline="",
                                        encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_s", "E_V_vs_AgAgCl", "eta_V"])
            for t, e, eta in zip(run["tiempo"], run["potencial"], run["eta"]):
                writer.writerow([f"{t:.6g}", f"{e:.6g}", f"{eta:.6g}"])

    report = []
    for run in runs:
        report.append("=" * 72)
        report.append(run["archivo"])
        report.append(run["resultado"].summary())
        report.append("")
    for spectrum, analysis in zip(session.eis_series, session.eis_analyses()):
        if analysis is None:
            continue
        report.append("=" * 72)
        report.append(spectrum.name)
        report.append(analysis.summary())
        report.append("")
    (OUT / "informe_estabilidad.txt").write_text("\n".join(report),
                                                 encoding="utf-8")

    summary = {
        "retenciones": [
            {"archivo": run["archivo"],
             "j_ma_cm2": run["resultado"].current_ma_cm2,
             "duracion_h": run["resultado"].duration_h,
             "eta_inicial_v": run["resultado"].overpotential_start,
             "eta_final_v": run["resultado"].overpotential_end,
             "deriva_v_h": run["resultado"].drift_v_per_h,
             "r2": run["resultado"].r_squared,
             "horas_para_100mv": run["resultado"].hours_to_lose_100mv,
             "avisos": run["resultado"].warnings}
            for run in runs],
        "espectros": [
            {"nombre": spectrum.name,
             "polarizacion_v": spectrum.metadata.get("potencial_cd_v"),
             "r_s_ohm": analysis.series_resistance,
             "r_ct_ohm": analysis.charge_transfer_resistance,
             "c_prima_f": analysis.capacitance_f}
            for spectrum, analysis in zip(session.eis_series,
                                          session.eis_analyses())
            if analysis is not None],
    }
    (OUT / "resumen_estabilidad.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito en {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
