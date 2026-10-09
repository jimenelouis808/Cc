"""Reports of finished calculations: one HTML file, the data as CSV, figures for a paper.

``tbkit report FOLDER`` (or :func:`build`) reads what a recipe left in a folder and
writes, next to it or where ``--out`` says:

* ``NAME.html``: one self-contained page (interactive charts, tables, the method and
  its limits; no internet needed, nothing to install). Each chart and table has a
  CSV button.
* ``NAME_datos/``: every chart and table as CSV (one file each, header row, comma
  separated, decimal point) plus ``indice.json``: for Origin, Excel, Igor, gnuplot…
* ``NAME_figuras/`` (needs matplotlib, the ``plot`` extra): every chart as PDF and SVG
  (vector, fonts embedded as TrueType) and PNG at 600 dpi, sized for one journal
  column (85 mm) or two (178 mm), with the Okabe-Ito palette (safe for colour-blind
  readers and in greyscale).

What a folder holds is recognised by its files (:data:`ADAPTERS`): the doped-coil
Raman/IR recipe (``report.json`` with ``pristine``), its GPAW check
(``report.json`` with ``gpaw``), the electronic comparison (``recipes/electronic_compare``),
any JSON file (``validation/*.json`` and the like, laid out as tables), and any
spectrum tbkit writes (``.npz`` with a
``grid``, or the CSV of ``tbkit raman/ir --out`` and the GUI's «Exportar CSV…»).
A new recipe gets a report by adding an adapter that turns its files into a
:class:`Report`; the page, the CSV and the figures come from that alone.

Every number on the page is read from the files: nothing is typed by hand.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Okabe & Ito (2008): distinguishable with every common colour-vision deficiency.
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
#: Journal column widths, inches (85 and 178 mm: ACS, RSC, Elsevier, Springer).
WIDTHS_IN = {"simple": 3.35, "doble": 7.0}


@dataclass
class Series:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass
class Figure:
    id: str
    title: str
    xlabel: str
    ylabel: str
    series: list[Series]
    kind: str = "line"                  # line | sticks | points | scatter (parity)
    caption: str = ""
    marks: list[tuple[float, str]] = field(default_factory=list)
    xlim: tuple[float, float] | None = None
    invert_x: bool = False


@dataclass
class Table:
    id: str
    title: str
    columns: list[str]
    rows: list[list]
    note: str = ""
    digits: list[int | None] | None = None


@dataclass
class Section:
    title: str
    blocks: list = field(default_factory=list)      # str (paragraph), list[str], Figure, Table

    def text(self, value: str) -> Section:
        self.blocks.append(value)
        return self

    def items(self, values: list[str]) -> Section:
        self.blocks.append(list(values))
        return self

    def add(self, block) -> Section:
        self.blocks.append(block)
        return self


@dataclass
class Report:
    title: str
    subtitle: str = ""
    eyebrow: str = "Reporte · tbkit"
    meta: dict = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    footer: str = ""

    def section(self, title: str) -> Section:
        self.sections.append(Section(title))
        return self.sections[-1]

    def figures(self) -> list[Figure]:
        return [b for s in self.sections for b in s.blocks if isinstance(b, Figure)]

    def tables(self) -> list[Table]:
        return [b for s in self.sections for b in s.blocks if isinstance(b, Table)]


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def _rounded(values, digits: int = 5) -> list:
    out = []
    for v in np.asarray(values, dtype=float).ravel():
        out.append(None if not np.isfinite(v) else float(f"{v:.{digits}g}"))
    return out


def _jsonable(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def to_dict(report: Report) -> dict:
    sections = []
    for sec in report.sections:
        blocks = []
        for b in sec.blocks:
            if isinstance(b, str):
                blocks.append({"type": "text", "text": b})
            elif isinstance(b, list):
                blocks.append({"type": "list", "items": b})
            elif isinstance(b, Figure):
                blocks.append({"type": "figure", "id": b.id, "title": b.title, "kind": b.kind,
                               "xlabel": b.xlabel, "ylabel": b.ylabel, "caption": b.caption,
                               "xlim": list(b.xlim) if b.xlim else None,
                               "marks": [{"x": x, "label": t} for x, t in b.marks],
                               "series": [{"name": s.name, "x": _rounded(s.x), "y": _rounded(s.y)}
                                          for s in b.series]})
            elif isinstance(b, Table):
                blocks.append({"type": "table", "id": b.id, "title": b.title,
                               "columns": b.columns, "note": b.note, "digits": b.digits,
                               "rows": [[_jsonable(v) for v in row] for row in b.rows]})
        sections.append({"title": sec.title, "blocks": blocks})
    return {"title": report.title, "subtitle": report.subtitle, "eyebrow": report.eyebrow,
            "meta": report.meta, "footer": report.footer, "sections": sections}


def write_html(report: Report, path: str | Path) -> Path:
    template = (Path(__file__).with_name("report_template.html")).read_text(encoding="utf-8")
    data = json.dumps(to_dict(report), ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")               # never close the <script> early
    title = report.title.replace("<", "&lt;")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__TITLE__", title).replace("__DATA__", data),
                    encoding="utf-8")
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_") or "x"


def export_data(report: Report, folder: str | Path) -> Path:
    """Every figure and table as CSV, plus ``indice.json`` (file, title, columns)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for fig in report.figures():
        name = f"figura_{_slug(fig.id)}.csv"
        header, cols = [], []
        for s in fig.series:
            header += [f"{s.name} | {fig.xlabel}", f"{s.name} | {fig.ylabel}"]
            cols += [np.asarray(s.x, float), np.asarray(s.y, float)]
        shared = all(len(c) == len(cols[0]) and np.array_equal(c, cols[0]) for c in cols[::2])
        if shared and len(fig.series) > 1:               # one x column, then one per series
            header = [fig.xlabel] + [s.name for s in fig.series]
            cols = [cols[0]] + cols[1::2]
        n = max(len(c) for c in cols)
        with (folder / name).open("w", newline="", encoding="utf-8-sig") as handle:
            w = csv.writer(handle)
            w.writerow(header)
            for k in range(n):
                w.writerow(["" if k >= len(c) or not np.isfinite(c[k]) else f"{c[k]:.8g}"
                            for c in cols])
        index.append({"archivo": name, "titulo": fig.title, "tipo": "figura",
                      "columnas": header})
    for tab in report.tables():
        name = f"tabla_{_slug(tab.id)}.csv"
        with (folder / name).open("w", newline="", encoding="utf-8-sig") as handle:
            w = csv.writer(handle)
            w.writerow(tab.columns)
            for row in tab.rows:
                w.writerow(["" if v is None else (f"{v:.8g}" if isinstance(v, float) else v)
                            for v in row])
        index.append({"archivo": name, "titulo": tab.title, "tipo": "tabla",
                      "columnas": tab.columns})
    (folder / "indice.json").write_text(json.dumps({"reporte": report.title, "archivos": index},
                                                   ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return folder


def journal_style(width: str = "simple") -> dict:
    """matplotlib rc for a journal figure: 7 pt sans, thin axes, ticks inside, no frame
    on the legend, fonts embedded as TrueType (Type 42) in PDF, text kept in SVG."""
    from cycler import cycler

    return {
        "figure.figsize": (WIDTHS_IN[width], WIDTHS_IN[width] * (0.68 if width == "simple"
                                                                 else 0.40)),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "axes.prop_cycle": cycler(color=list(OKABE_ITO)),
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.default": "regular",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    }


def _mpl(text: str) -> str:
    """Unicode super/subscripts the journal fonts lack -> mathtext (cm⁻¹ -> cm$^{-1}$)."""
    for uni, tex in (("⁻¹", "$^{-1}$"), ("⁻²", "$^{-2}$"), ("₂", "$_2$"), ("₃", "$_3$")):
        text = text.replace(uni, tex)
    return text


def export_figures(report: Report, folder: str | Path, formats=("pdf", "svg", "png"),
                   width: str = "simple", dpi: int = 600) -> list[Path]:
    """Each figure of the report as a journal figure (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context(journal_style(width)):
        for fig in report.figures():
            f, ax = plt.subplots()
            for i, s in enumerate(fig.series):
                if fig.kind == "sticks":
                    ax.vlines(s.x, 0, s.y, lw=0.8, label=_mpl(s.name),
                              color=OKABE_ITO[i % len(OKABE_ITO)])
                elif fig.kind in ("points", "scatter"):
                    ax.plot(s.x, s.y, "o", ms=3, mew=0, label=_mpl(s.name))
                else:
                    ax.plot(s.x, s.y, label=_mpl(s.name))
            if fig.kind == "scatter":
                lo = min(min(np.min(s.x), np.min(s.y)) for s in fig.series)
                hi = max(max(np.max(s.x), np.max(s.y)) for s in fig.series)
                pad = 0.04 * (hi - lo)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=0.5, color="0.5")
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_aspect("equal")
            for x, label in fig.marks:
                ax.axvline(x, ls=":", lw=0.5, color="0.4")
                ax.annotate(_mpl(label), (x, 1), xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", fontsize=6, color="0.3")
            if fig.xlim:
                ax.set_xlim(*fig.xlim)
            if fig.invert_x:
                ax.invert_xaxis()
            if fig.kind in ("line", "sticks") and min(np.min(s.y) for s in fig.series) >= 0:
                ax.set_ylim(bottom=0)
            ax.set_xlabel(_mpl(fig.xlabel))
            ax.set_ylabel(_mpl(fig.ylabel))
            if len(fig.series) > 1:
                ax.legend()
            for fmt in formats:
                out = folder / f"{_slug(fig.id)}.{fmt}"
                f.savefig(out, dpi=dpi if fmt == "png" else None)
                written.append(out)
            plt.close(f)
    return written


def build(source: str | Path, out: str | Path | None = None, data: bool = True,
          figures: tuple = ("pdf", "svg", "png"), width: str = "simple") -> dict:
    """Recognise ``source``, write the HTML page, the CSV folder and (if matplotlib is
    installed and ``figures``) the figure folder. Returns the paths written."""
    source = Path(source)
    report = recognise(source)
    out = Path(out) if out else (source if source.is_dir() else source.parent) / \
        f"reporte_{_slug(source.stem if source.is_file() else source.name)}.html"
    written = {"html": write_html(report, out)}
    if data:
        written["datos"] = export_data(report, out.with_name(out.stem + "_datos"))
    if figures:
        try:
            written["figuras"] = export_figures(report, out.with_name(out.stem + "_figuras"),
                                                figures, width)
        except ImportError:
            written["figuras"] = "matplotlib no está instalado (extra 'plot'): sin figuras"
    return written


def open_result(source: str | Path, out: str | Path | None = None, browser: bool = True) -> Path:
    """The page for ``source`` (anything :func:`recognise` takes), written to ``out`` or to
    a fresh temporary folder, so opening a file under ``validation/`` never writes there;
    then shown in the default browser. No CSV or figures: that is :func:`build`."""
    import tempfile
    import webbrowser

    source = Path(source)
    if out is None:
        out = Path(tempfile.mkdtemp(prefix="tbkit_")) / \
            f"{_slug(source.stem if source.is_file() else source.name)}.html"
    path = write_html(recognise(source), out)
    if browser:
        webbrowser.open(path.resolve().as_uri())
    return path


# --------------------------------------------------------------------------
# Adapters: files of a recipe -> Report
# --------------------------------------------------------------------------

_HC_KB = 1.4387769                      # cm·K
NAMES = {"pristine": "sin dopar", "N": "N grafítico", "amine": "–NH₂ + H"}


def _cross_section(activity, w, laser_ev, temperature=300.0):
    nu = laser_ev * 8065.544
    return activity * (nu - w) ** 4 / w / (1 - np.exp(-_HC_KB * w / temperature))


def doped_raman(folder: Path) -> Report:
    """``out/doped_raman`` (tbkit.recipes.doped_raman): Raman, D estimate, IR."""
    r = json.loads((folder / "report.json").read_text())
    spectra = np.load(folder / "spectra.npz")
    names = [n for n in ("pristine", "N", "amine") if n in r]
    lasers = r["lasers_eV"]
    rep = Report("Coil dopada: Raman, banda D e IR",
                 "Raman resonante de primer orden en Γ, estimación de la banda D por el carácter "
                 "de respiración de anillos e IR perpendicular al eje, con un solo modelo para "
                 "todas las estructuras.",
                 meta={"carpeta": str(folder), "receta": "tbkit.recipes.doped_raman",
                       "láseres (eV)": ", ".join(f"{x:g}" for x in lasers)})

    sec = rep.section("Raman")
    grid = spectra["grid"]
    for laser in lasers:
        tag = f"{laser:.2f}"
        series = [Series(NAMES.get(n, n), grid, spectra[f"{n}_{tag}"]) for n in names
                  if f"{n}_{tag}" in spectra.files]
        top = max(float(s.y.max()) for s in series)
        for s in series:
            s.y = s.y / top
        sec.add(Figure(f"raman_{tag}", f"Raman a {1239.84193 / laser:.0f} nm ({laser:g} eV)",
                       "Desplazamiento Raman (cm⁻¹)", "Intensidad (norm.)", series,
                       xlim=(900, 1800),
                       caption="Misma escala para todas las estructuras: 1 = máximo de todas "
                               "con ese láser. FWHM 10 cm⁻¹."))
    rows = []
    for laser in lasers:
        tag = f"{laser:.2f}"
        base = None
        for n in names:
            e = r[n]
            w = np.array(e["raman_frequencies_cm1"])
            a = _cross_section(np.array(e[f"activities_{tag}"]), w, laser)
            b = np.array(e["breathing_B"])
            g, d = (w > 1500) & (w < 1700), (w > 1100) & (w < 1450)
            total = float(a.sum())
            base = base or total
            rows.append([f"{laser:g}", NAMES.get(n, n), float((a * w)[g].sum() / a[g].sum()),
                         float((a * b * w)[d].sum() / (a * b)[d].sum()),
                         float((a * b)[d].sum() / a[g].sum()), total / base])
    sec.add(Table("raman_resumen", "Bandas por estructura y láser",
                  ["láser (eV)", "estructura", "centroide G (cm⁻¹)", "D estimada (cm⁻¹)",
                   "I_D/I_G estimado", "intensidad total / sin dopar"], rows,
                  note="G: centroide de la intensidad en 1500–1700 cm⁻¹. D estimada: centroide "
                       "de la intensidad pesada por el carácter de respiración B en 1100–1450 "
                       "cm⁻¹; no incluye la doble resonancia.", digits=[None, None, 1, 0, 3, 2]))

    sec = rep.section("Estimación de la banda D")
    sec.text("La banda D real es una doble resonancia del modo A₁′ en K, que un cálculo de "
             "primer orden en Γ no contiene. Lo que sí da es de qué estaría hecha: la intensidad "
             "Raman de los modos en que los anillos respiran.")
    tag = "2.33" if 2.33 in lasers else f"{lasers[0]:.2f}"
    series = [Series(NAMES.get(n, n), grid, spectra[f"{n}_{tag}_breathing"]) for n in names
              if f"{n}_{tag}_breathing" in spectra.files]
    if series:
        sec.add(Figure("banda_d", f"Intensidad × respiración (B) a {tag} eV",
                       "Desplazamiento Raman (cm⁻¹)", "Intensidad × B", series,
                       xlim=(900, 1800)))

    if "ir_grid" in spectra.files and any(f"{n}_ir" in spectra.files for n in names):
        sec = rep.section("IR")
        ir_grid = spectra["ir_grid"]
        series = [Series(NAMES.get(n, n), ir_grid, spectra[f"{n}_ir"]) for n in names
                  if f"{n}_ir" in spectra.files]
        sec.add(Figure("ir", "IR, luz polarizada perpendicular al eje", "Número de onda (cm⁻¹)",
                       "Absorción (km/mol por cm⁻¹)", series, xlim=(50, 3500),
                       caption="Cargas de Born del mismo estado fundamental; componentes x, y. "
                               "FWHM 10 cm⁻¹."))
        rows = []
        for n in names:
            ir = r[n].get("ir")
            if ir:
                rows.append([NAMES.get(n, n), float(sum(ir["intensities_km_mol"])),
                             float(ir["sum_rule_residual_e"])]
                            + [f"{s['frequency_cm1']:.0f} ({s['intensity_km_mol']:.0f})"
                               for s in ir["strongest"][:3]])
        sec.add(Table("ir_resumen", "IR por estructura",
                      ["estructura", "total (km/mol)", "residuo regla de suma (e)",
                       "1.ª línea cm⁻¹ (km/mol)", "2.ª", "3.ª"], rows, digits=[None, 0, 5]))

    checks = [(n, json.loads((folder / n / "check.json").read_text())) for n in names
              if (folder / n / "check.json").exists()]
    if checks:
        sec = rep.section("Comprobaciones")
        rows = [[NAMES.get(n, n), c["embedded_cm1"], c["full_cm1"],
                 c["full_cm1"] - c["embedded_cm1"], c["share_on_core"]]
                for n, cs in checks for c in cs]
        sec.add(Table("comprobacion_hessiana", "Hessiana local frente al modelo completo",
                      ["estructura", "modo local (cm⁻¹)", "modelo completo (cm⁻¹)",
                       "diferencia (cm⁻¹)", "peso en el cambio"], rows,
                      digits=[None, 1, 1, 1, 2]))
    sec = rep.section("Qué no es este cálculo")
    sec.items([("Raman de primer orden en Γ con α de partículas independientes: sin doble "
                "resonancia (ni D real ni 2D)."),
               ("Las energías de resonancia son las del modelo: compárense los láseres entre "
                "sí, no con un espectro medido."),
               "IR solo perpendicular al eje (la componente axial necesita la fase de Berry).",
               "Frecuencias del modelo, sin escalar."])
    rep.footer = "Generado por tbkit.report desde los archivos de la carpeta; ningún número " \
                 "se escribió a mano."
    return rep


def doped_gpaw(folder: Path) -> Report:
    """``out/doped_gpaw`` (tbkit.recipes.doped_gpaw): tbkit against GPAW on chosen modes."""
    r = json.loads((folder / "report.json").read_text())
    names = [n for n in ("pristine", "N", "amine") if n in r and r[n].get("modes")]
    rep = Report("tbkit frente a GPAW en los modos clave",
                 "Frecuencia (cociente de Rayleigh) e intensidad IR de GPAW-PBE a lo largo de "
                 "los modos de tbkit, en la geometría PBE de cada estructura.",
                 meta={"carpeta": str(folder), "GPAW": r.get("gpaw", "")})
    sec = rep.section("Frecuencias")
    sec.add(Figure("paridad_frecuencias", "GPAW frente a tbkit", "tbkit (cm⁻¹)", "GPAW (cm⁻¹)",
                   [Series(NAMES.get(n, n), np.array([m["tb_cm1"] for m in r[n]["modes"]]),
                           np.array([m["pbe_cm1"] for m in r[n]["modes"]])) for n in names],
                   kind="scatter",
                   caption="Línea discontinua: igualdad. El valor de GPAW es exacto si el modo "
                           "de tbkit es propio de PBE y una cota superior si no."))
    rows = [[NAMES.get(n, n), m["tb_cm1"], m["pbe_cm1"], m["pbe_over_tb"], m["tb_ir_km_mol"],
             m["pbe_ir_km_mol"], ", ".join(m["why"])] for n in names for m in r[n]["modes"]]
    sec.add(Table("modos", "Modo por modo",
                  ["estructura", "tbkit (cm⁻¹)", "GPAW (cm⁻¹)", "GPAW/tbkit", "IR tbkit (km/mol)",
                   "IR GPAW (km/mol)", "por qué se eligió"], rows,
                  digits=[None, 1, 1, 3, 1, 1, None]))
    sec = rep.section("Intensidad IR")
    sec.add(Figure("paridad_ir", "IR: GPAW frente a tbkit", "tbkit (km/mol)", "GPAW (km/mol)",
                   [Series(NAMES.get(n, n), np.array([m["tb_ir_km_mol"] for m in r[n]["modes"]]),
                           np.array([m["pbe_ir_km_mol"] for m in r[n]["modes"]])) for n in names],
                   kind="scatter"))
    geo = [[NAMES.get(n, n), r[n]["geometry_tb_vs_pbe"]["bond_rms_A"],
            r[n]["geometry_tb_vs_pbe"]["bond_mean_A"], r[n]["geometry_tb_vs_pbe"]["bond_max_abs_A"]]
           for n in names if "geometry_tb_vs_pbe" in r[n]]
    if geo:
        rep.section("Geometría").add(Table(
            "geometria", "Enlaces tbkit − PBE", ["estructura", "RMS (Å)", "media (Å)",
                                                  "máx. |Δ| (Å)"], geo, digits=[None, 4, 4, 4]))
    rep.footer = "Generado por tbkit.report; ningún número se escribió a mano."
    return rep


def spectrum_file(path: Path) -> Report:
    """Any spectrum tbkit writes: ``.npz`` with ``grid`` (and other grids by their
    prefix, e.g. ``ir_grid`` for ``*_ir``), or a CSV whose first column is x."""
    rep = Report(f"Espectro {path.stem}", meta={"archivo": str(path)})
    sec = rep.section("Curvas")
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        grids = {k: data[k] for k in data.files if k == "grid" or k.endswith("_grid")}
        if not grids:
            raise ValueError(f"{path.name} no tiene 'grid'.")
        for gname, g in grids.items():
            prefix = gname[:-len("grid")]
            series = [Series(k, g, data[k]) for k in data.files
                      if k not in grids and data[k].ndim == 1 and len(data[k]) == len(g)
                      and (not prefix or k.endswith(prefix.rstrip("_")))]
            if series:
                sec.add(Figure(_slug(gname), gname, "x", "y", series))
    else:
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().strip().lstrip("#").split(",")
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        sec.add(Figure(_slug(path.stem), path.stem, header[0], "intensidad",
                       [Series(h.strip() or f"columna {k}", values[:, 0], values[:, k])
                        for k, h in enumerate(header[1:], start=1)]))
    return rep


def electronic(folder: Path) -> Report:
    """``out/electronic`` (tbkit.recipes.electronic_compare): bands and DOS, tbkit vs GPAW."""
    r = json.loads((folder / "report.json" if folder.is_dir() else folder).read_text())
    folder = folder if folder.is_dir() else folder.parent
    systems = r.get("systems", {})
    rep = Report("Estructura electrónica: tbkit frente a GPAW",
                 "tbkit (xu_chn, SCC) y GPAW (PBE, LCAO dzp) en la misma geometría; nada se "
                 "ajustó a estos datos. Energías respecto al nivel de Fermi de cada método.",
                 meta={"carpeta": str(folder), "ensanchamiento DOS (eV)": r.get("sigma_eV")})
    dos = np.load(folder / "dos.npz") if (folder / "dos.npz").exists() else None
    coils = [n for n in ("pristine", "N", "amine") if n in systems]
    if coils:
        sec = rep.section("Coil: densidad de estados y bandas")
        rows = []
        for n in coils:
            e = systems[n]
            rows.append([NAMES.get(n, n), e["gap_gpaw_eV"], e["gap_tb_eV"],
                         e["dos_correlation_-2_+2"], e["dos_correlation_-6_+4"],
                         e["states_within_1eV"]["gpaw"], e["states_within_1eV"]["tb"],
                         e.get("delta_dos_correlation_-3_+3")])
        metals = [[NAMES.get(n, n), systems[n]["electrons_per_cell"],
                   *systems[n]["half_filled_band_gpaw_eV"], *systems[n]["half_filled_band_tb_eV"]]
                  for n in coils if "half_filled_band_gpaw_eV" in systems[n]]
        if metals:
            sec.add(Table("banda_semillena", "Banda semillena (número impar de electrones)",
                          ["estructura", "electrones por celda", "GPAW mín. (eV)", "GPAW máx. (eV)",
                           "tbkit mín. (eV)", "tbkit máx. (eV)"], metals,
                          note="Con un número impar de electrones por celda una banda queda con "
                               "un solo electrón: los dos métodos dan un metal. Lo que los distingue "
                               "es el ancho y la posición de esa banda (respecto a E_F).",
                          digits=[None, 0, 3, 3, 3, 3]))
        sec.add(Table("resumen_coil", "Resumen por estructura",
                      ["estructura", "gap GPAW (eV)", "gap tbkit (eV)", "r DOS ±2 eV",
                       "r DOS −6…+4 eV", "estados ±1 eV GPAW", "estados ±1 eV tbkit",
                       "r ΔDOS (dopado − sin dopar)"], rows,
                      note="r: correlación de Pearson entre las dos curvas en esa ventana. "
                           "Estados: integral de la DOS en ±1 eV del nivel de Fermi, por celda.",
                      digits=[None, 3, 3, 2, 2, 1, 1, 2]))
        if dos is not None:
            grid = dos["grid"]
            for n in coils:
                sec.add(Figure(f"dos_{n}", f"DOS, {NAMES.get(n, n)}", "E − E_F (eV)",
                               "estados / eV / celda",
                               [Series("GPAW (PBE)", grid, dos[f"{n}_gpaw"]),
                                Series("tbkit (xu_chn)", grid, dos[f"{n}_tb"])],
                               marks=[(0.0, "E_F")]))
            for n in ("N", "amine"):
                if n in coils and "pristine" in coils:
                    sec.add(Figure(f"delta_dos_{n}", f"ΔDOS: {NAMES.get(n, n)} − sin dopar",
                                   "E − E_F (eV)", "estados / eV / celda",
                                   [Series("GPAW (PBE)", grid, dos[f"{n}_gpaw"] - dos["pristine_gpaw"]),
                                    Series("tbkit (xu_chn)", grid, dos[f"{n}_tb"] - dos["pristine_tb"])],
                                   marks=[(0.0, "E_F")], xlim=(-3.0, 3.0),
                                   caption="Los estados que añade el dopante, en cada método."))
        for n in (c for c in coils if (folder / c / "gpaw.json").exists()):
            series = []
            for (label, key), offset in zip((("GPAW (PBE)", "gpaw"), ("tbkit (xu_chn)", "tb")),
                                            (-0.005, 0.005)):
                d = json.loads((folder / n / f"{key}.json").read_text())
                kz = np.array([k[2] for k in d["kpts"]]) + offset
                xs, ys = [], []
                for x, row in zip(kz, d["bands_minus_fermi"]):
                    for e in row:
                        if abs(e) <= 1.0:
                            xs.append(x)
                            ys.append(e)
                series.append(Series(label, np.array(xs), np.array(ys)))
            sec.add(Figure(f"bandas_{n}", f"Bandas Γ–Z, {NAMES.get(n, n)}", "k_z (unidades de 2π/c)",
                           "E − E_F (eV)", series, kind="points",
                           caption="Niveles a ±1 eV del nivel de Fermi en cada k, de Γ (0) a Z (0.5); "
                                   "GPAW un poco a la izquierda y tbkit a la derecha de cada k."))
    if "graphene" in systems:
        g = systems["graphene"]
        sec = rep.section("Grafeno: la parte de carbono de todos los estados de la coil")
        rows = [["π en M, debajo de E_F (eV)", g["gpaw"]["pi_M_below_eV"], g["tb"]["pi_M_below_eV"]],
                ["π* en M, encima de E_F (eV)", g["gpaw"]["pi_M_above_eV"], g["tb"]["pi_M_above_eV"]],
                ["ħv_F (eV·Å)", g["gpaw"]["fermi_velocity_eV_A"], g["tb"]["fermi_velocity_eV_A"]],
                ["nivel más bajo en Γ (eV)", min(g["gpaw"]["gamma_eV"]), min(g["tb"]["gamma_eV"])]]
        sec.add(Table("grafeno", "Puntos de alta simetría", ["magnitud", "GPAW", "tbkit"], rows,
                      note="Niveles a −25…+12 eV del nivel de Fermi. ħv_F: cuerda de un paso del "
                           "camino antes de K (subestima la pendiente en ambos por igual).",
                      digits=[None, 2, 2]))
        series = []
        for label, key in (("GPAW (PBE)", "gpaw"), ("tbkit (xu_chn)", "tb")):
            if not (folder / "graphene" / f"{key}.json").exists():
                continue
            d = json.loads((folder / "graphene" / f"{key}.json").read_text())
            xs, ys = [], []
            for x, row in enumerate(d["bands_minus_fermi"]):
                for e in row:
                    if -12.0 <= e <= 8.0:
                        xs.append(x)
                        ys.append(e)
            series.append(Series(label, np.array(xs, dtype=float), np.array(ys)))
        if series:
            sec.add(Figure("bandas_grafeno", "Bandas del grafeno, Γ–M–K–Γ", "punto del camino (Γ 0, M 30, K 47, Γ 89)",
                           "E − E_F (eV)", series, kind="points"))
    mol = r.get("molecules")
    if mol:
        sec = rep.section("Moléculas: niveles de Kohn-Sham")
        summary = [[{"train": "en el ajuste", "test": "apartadas"}[k.split("_")[1]], v["n"],
                    v["gap_mean_error_eV"], v["gap_rms_error_eV"], v["gap_correlation"],
                    v["occupied_rms_vs_homo_median_eV"]]
                   for k, v in mol.items() if k.startswith("summary_")]
        sec.add(Table("moleculas_resumen", "Gap HOMO-LUMO y niveles ocupados",
                      ["moléculas", "n", "error medio del gap (eV)", "RMS del gap (eV)",
                       "r del gap", "RMS niveles ocupados vs HOMO (mediana, eV)"], summary,
                      note="Los niveles de 16 de las 22 moléculas fueron objetivo del ajuste de "
                           "xu_chn; las 6 apartadas son la prueba.",
                      digits=[None, 0, 2, 2, 3, 2]))
        rows = [[m["molecule"], "ajuste" if m["role"] == "train" else "apartada", m["gap_gpaw_eV"],
                 m["gap_tb_eV"], m["occupied_rms_vs_homo_eV"]] for m in mol["molecules"]]
        sec.add(Table("moleculas", "Por molécula", ["molécula", "papel", "gap GPAW (eV)",
                                                    "gap tbkit (eV)", "RMS ocupados vs HOMO (eV)"],
                      rows, digits=[None, None, 2, 2, 2]))
        sec.add(Figure("paridad_gap", "Gap HOMO-LUMO: tbkit frente a GPAW", "GPAW (eV)", "tbkit (eV)",
                       [Series(lbl, np.array([m["gap_gpaw_eV"] for m in mol["molecules"] if m["role"] == role]),
                               np.array([m["gap_tb_eV"] for m in mol["molecules"] if m["role"] == role]))
                        for role, lbl in (("train", "en el ajuste"), ("test", "apartadas"))],
                       kind="scatter"))
    rep.footer = "Generado por tbkit.report desde recipes/electronic_compare; ningún número se escribió a mano."
    return rep


def _json_of(p: Path):
    """The JSON a source stands for (a folder's report.json, or the file), or None."""
    path = p / "report.json" if p.is_dir() else p
    if path.suffix.lower() != ".json" or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_electronic(p: Path) -> bool:
    data = _json_of(p)
    return isinstance(data, dict) and "electronic_compare" in str(data.get("what", ""))


# --------------------------------------------------------------------------
# Any JSON: the validation files and whatever a recipe leaves, as tables
# --------------------------------------------------------------------------

def _scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _cell(value):
    if _scalar(value):
        return value
    if isinstance(value, list) and len(value) <= 6 and all(_scalar(v) for v in value):
        return ", ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in value)
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 80 else text[:77] + "…"


def _records_table(tid: str, title: str, rows: list[dict]) -> Table:
    columns: list[str] = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    return Table(tid, title, columns, [[_cell(row.get(c)) for c in columns] for row in rows])


def _json_into(rep: Report, value, path: list[str], depth: int = 0) -> None:
    """Scalars of a dict -> one key/value table; a list of dicts or a dict of flat dicts
    -> one table (a row each); a long list of numbers -> a curve; anything deeper -> its
    own section (to four levels)."""
    title = " › ".join(path) if path else "Contenido"
    tid = _slug("_".join(path) or "contenido")
    if isinstance(value, list):
        if value and all(isinstance(v, dict) for v in value):
            rep.section(title).add(_records_table(tid, title, value))
        elif len(value) > 6 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                    for v in value):
            rep.section(title).add(Figure(tid, title, "índice", path[-1] if path else "valor",
                                          [Series(path[-1] if path else "valor",
                                                  np.arange(len(value), dtype=float),
                                                  np.array(value, dtype=float))]))
        else:
            rep.section(title).text(str(_cell(value)))
        return
    if not isinstance(value, dict):
        rep.section(title).text(str(_cell(value)))
        return
    flat = {k: v for k, v in value.items() if _scalar(v) or (isinstance(v, list) and len(v) <= 6
                                                             and all(_scalar(x) for x in v))}
    nested = {k: v for k, v in value.items() if k not in flat}
    sec = None
    if flat:
        sec = rep.section(title)
        sec.add(Table(tid, title, ["clave", "valor"], [[k, _cell(v)] for k, v in flat.items()]))
    flat_children = {k: v for k, v in nested.items() if isinstance(v, dict) and v and
                     all(_scalar(x) or isinstance(x, list) and len(x) <= 6 for x in v.values())}
    if len(flat_children) >= 2:
        sec = sec or rep.section(title)
        sec.add(_records_table(tid + "_tabla", title,
                               [{"": k, **v} for k, v in flat_children.items()]))
        nested = {k: v for k, v in nested.items() if k not in flat_children}
    for key, child in nested.items():
        if depth >= 4:
            (sec or rep.section(title)).text(f"{key}: {_cell(child)}")
            continue
        _json_into(rep, child, [*path, str(key)], depth + 1)


def json_file(path: Path) -> Report:
    """Any JSON file tbkit or a recipe writes (``validation/*.json``, a ``report.json``…),
    laid out as tables and curves; a recognised recipe gets its own adapter instead."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rep = Report(path.stem, "Contenido del archivo, tal como está guardado.",
                 meta={"archivo": str(path)}, eyebrow="Archivo · tbkit")
    _json_into(rep, data, [])
    rep.footer = "Vista genérica de tbkit.report: cada tabla es una parte del JSON, sin cambiar números."
    return rep


def _is_doped_raman(p: Path) -> bool:
    if not (p.is_dir() and (p / "report.json").exists() and (p / "spectra.npz").exists()):
        return False
    return "pristine" in json.loads((p / "report.json").read_text())


def _is_doped_gpaw(p: Path) -> bool:
    return p.is_dir() and (p / "report.json").exists() and \
        "gpaw" in json.loads((p / "report.json").read_text())


#: (test, builder): the first whose test passes makes the report.
ADAPTERS = [
    (_is_electronic, electronic),
    (_is_doped_raman, doped_raman),
    (_is_doped_gpaw, doped_gpaw),
    (lambda p: p.is_file() and p.suffix.lower() in (".npz", ".csv"), spectrum_file),
    (lambda p: p.is_file() and p.suffix.lower() == ".json", json_file),
]


def recognise(source: Path) -> Report:
    for test, builder in ADAPTERS:
        if test(source):
            return builder(source)
    raise ValueError(f"No reconozco {source}: carpeta de doped_raman, doped_gpaw o "
                     f"electronic_compare, un archivo .json, o un espectro .npz/.csv de tbkit.")
