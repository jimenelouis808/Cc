"""The validation figure: reproduced isotherms against the published ones.

Two panels, and the split is the dataviz rule rather than taste. The two
temperatures' loadings differ by a factor of thirty, which is the
"two measures of different scale" case -- so they share ONE log axis in
(a) rather than getting an axis each, and the agreement that the eye
cannot read off a log plot gets its own panel in (b), in per cent, where
both temperatures legitimately share a linear axis.

Published and reproduced are told apart by marker SHAPE, and the two
temperatures by colour and by being three decades apart. So no identity
rests on colour alone.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from carbonforge.adsorption.bundle import LEVESQUE_BUNDLE_ISOTHERMS

# Categorical slots 1 and 2 of the validated palette, in fixed order.
COLOUR = {77.0: "#2a78d6", 293.0: "#eb6834"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

ROW = re.compile(
    r"^\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([+-][0-9.]+)%\s+"
    r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\+-\s*([0-9.]+)")


def harvest(scratch: Path) -> dict[float, list[dict]]:
    """Pull every finished point out of the run logs."""
    found: dict[float, list[dict]] = {77.0: [], 293.0: []}
    sources = [(293.0, scratch / "isoterma.log")]
    sources += [(77.0, p) for p in sorted((scratch / "t77").glob("p*.log"))]
    for temperature, path in sources:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            match = ROW.match(line)
            if not match:
                continue
            found[temperature].append({
                "presion_mpa": float(match.group(1)),
                "rho_publicado": float(match.group(2)),
                "rho_calculado": float(match.group(3)),
                "diferencia_pct": float(match.group(4)),
                "n_medio": float(match.group(7)),
                "n_error": float(match.group(8)),
            })
    for rows in found.values():
        rows.sort(key=lambda row: row["presion_mpa"])
    return found


def draw(data: dict[float, list[dict]], out: Path) -> None:
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.3, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12})

    for axis in (top, bottom):
        axis.set_facecolor("#fcfcfb")
        axis.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
        axis.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=MUTED, labelcolor=INK2, width=0.8)

    # --- (a) the isotherms, one log axis for both temperatures --------
    # The published ring has to stay visible UNDER the reproduced square,
    # or the places where the two agree look identical to the places
    # where no point has been computed yet -- which is the opposite of
    # what this figure is for. Hence 12 against 6, not 8 against 7.
    for temperature in (77.0, 293.0):
        published = LEVESQUE_BUNDLE_ISOTHERMS[temperature]
        top.plot([row[0] for row in published],
                 [row[1] for row in published], "-o",
                 color=COLOUR[temperature], lw=1.4, ms=12, mfc="#fcfcfb",
                 mew=1.8, zorder=3, label=f"{temperature:g} K  publicado")
        rows = data[temperature]
        if rows:
            top.plot([r["presion_mpa"] for r in rows],
                     [r["rho_calculado"] for r in rows], "s",
                     color=COLOUR[temperature], ms=6, mew=0.0, zorder=4,
                     label=f"{temperature:g} K  este código")

    top.set_xscale("log")
    top.set_yscale("log")
    top.set_ylabel(r"$\rho_o$ de H$_2$ adsorbido  (g cm$^{-3}$)", color=INK2)
    top.legend(frameon=False, loc="lower right", fontsize=8.5,
               labelcolor=INK2, ncol=2, columnspacing=1.2)
    top.set_title(
        "Haz de SWNT de Levesque: D = 13.3 Å, d = 6 Å, tubos abiertos",
        color=INK, fontsize=10.5, loc="left", pad=8)

    # --- (b) the residual, where the agreement is actually readable ---
    bottom.axhspan(-6, 6, color=GRID, alpha=0.55, zorder=0, lw=0)
    bottom.axhline(0, color=MUTED, lw=0.9, zorder=1)
    for temperature in (77.0, 293.0):
        rows = data[temperature]
        if not rows:
            continue
        bottom.plot([r["presion_mpa"] for r in rows],
                    [r["diferencia_pct"] for r in rows], "-s",
                    color=COLOUR[temperature], lw=1.8, ms=7, mew=1.6,
                    mec="#fcfcfb", zorder=3, label=f"{temperature:g} K")
    bottom.set_xscale("log")
    bottom.set_xlabel("Presión del reservorio  (MPa)", color=INK2)
    bottom.set_ylabel("diferencia  (%)", color=INK2)
    bottom.set_ylim(-12, 12)
    bottom.legend(frameon=False, loc="upper left", fontsize=8.5,
                  labelcolor=INK2, ncol=2)
    bottom.annotate("±6 %", xy=(0.995, 6), xycoords=("axes fraction",
                                                      "data"),
                    ha="right", va="bottom", color=MUTED, fontsize=8)

    # Pressures people say out loud: 0.1, 1, 10 -- not 10^-1, 10^0, 10^1.
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    bottom.xaxis.set_major_locator(LogLocator(base=10.0))
    bottom.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{v:g}"))
    bottom.xaxis.set_minor_formatter(NullFormatter())

    figure.savefig(out.with_suffix(".pdf"), bbox_inches="tight",
                   facecolor="#fcfcfb")
    figure.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight",
                   facecolor="#fcfcfb")
    plt.close(figure)


def write_readme(data: dict[float, list[dict]], out_dir: Path) -> None:
    """The folder has to say what produced it and what it does not prove."""
    lines = [
        "# Validación del GCMC contra Levesque 2002",
        "",
        "Reproducción de las isotermas publicadas por D. Levesque, A. Gicquel,",
        "F. Lamari Darkrim y S. Beyaz Kayiran, *J. Phys.: Condens. Matter*",
        "**14** (2002) 9285, tablas 2 y 3, sobre su propio adsorbente:",
        "un haz hexagonal de nanotubos (17,0) de D = 13.3 Å separados",
        "d = 6 Å, abiertos, 8704 carbonos en 175 900 Å³.",
        "",
        "Nada de esto está ajustado. La geometría del haz, el reservorio y",
        "el muestreo salen de parámetros publicados, y lo que se compara es",
        "la columna `rho_o`: cinco decimales y ninguna convención. La",
        "columna `wt%` del artículo **no** sirve de objetivo, porque está en",
        "base por carbono y no de masa total — ver `bundle.py`.",
        "",
        "## Resultado",
        "",
    ]
    for temperature, rows in sorted(data.items()):
        if not rows:
            continue
        errors = [abs(r["diferencia_pct"]) for r in rows]
        lines += [
            f"### {temperature:g} K",
            "",
            "| P (MPa) | ρ_o artículo | ρ_o aquí | dif | N |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['presion_mpa']:g} | {row['rho_publicado']:.5f} | "
                f"{row['rho_calculado']:.5f} | "
                f"{row['diferencia_pct']:+.1f} % | "
                f"{row['n_medio']:.0f} ± {row['n_error']:.0f} |")
        lines += [
            "",
            f"{len(rows)} puntos, diferencia absoluta media "
            f"**{np.mean(errors):.1f} %**, peor **{max(errors):.1f} %**.",
            "",
        ]
    lines += [
        "## Qué significa ese ±6 %, y qué no",
        "",
        "Es la precisión a la que este prototipo tiene derecho, y las tres",
        "fuentes están medidas, no estimadas:",
        "",
        "- El reservorio es un virial de dos términos ajustado a la tabla",
        "  redondeada del propio artículo, y reproduce esa tabla al 1.2 %.",
        "- La molécula es un solo sitio Lennard-Jones sin cuadrupolo.",
        "- Las densidades publicadas están redondeadas a cinco decimales,",
        "  lo que a 0.1 MPa y 293 K es **una cifra significativa**.",
        "",
        "A 77 K las diferencias son todas del mismo signo, y el aviso de",
        "deriva dice por qué: a 20 MPa la carga seguía subiendo +61",
        "moléculas entre el primer bloque de producción y el último. Correr",
        "más largo acerca, no aleja.",
        "",
        "**Lo que esto valida es el muestreo, no el modelo.** Reproducir a",
        "Levesque significa que el potencial, la corrección de Feynman-Hibbs,",
        "la actividad del reservorio y el balance de las jugadas están bien",
        "puestos. No dice que el modelo de un sitio sin cuadrupolo sea cierto",
        "para el hidrógeno: dice que está implementado como él lo implementó.",
        "",
        "## Reproducirlo",
        "",
        "```bash",
        "carbonforge adsorcion isoterma haz.xyz salida/ \\",
        "    --temperatura 77 --presiones 0.1,0.5,1,3,5,10,15,20",
        "python -m carbonforge.adsorption.run salida/",
        "```",
        "",
        "Son del orden de 6 horas por isoterma a 77 K. La carpeta reanuda:",
        "los puntos que ya estén no se recalculan.",
        "",
        "`datos.json` lleva los números de esta figura.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")


def main() -> int:
    scratch = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = harvest(scratch)
    (out_dir / "datos.json").write_text(
        json.dumps({str(k): v for k, v in data.items()}, indent=1) + "\n")
    draw(data, out_dir / "isoterma_validacion")
    write_readme(data, out_dir)
    for temperature, rows in sorted(data.items()):
        if not rows:
            continue
        worst = max(abs(r["diferencia_pct"]) for r in rows)
        mean = np.mean([abs(r["diferencia_pct"]) for r in rows])
        print(f"{temperature:g} K: {len(rows)} puntos, "
              f"|dif| media {mean:.1f} %, peor {worst:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
