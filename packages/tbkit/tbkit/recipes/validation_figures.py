"""Figures and tables of everything tbkit validated against GPAW: ``docs/GRAFICOS.md``.

Run from packages/tbkit (needs matplotlib, which the package does not depend on)::

    python -m tbkit.recipes.validation_figures          # every section
    python -m tbkit.recipes.validation_figures bandas   # only the band structures

Band structures are drawn as continuous lines, every band of every method, GPAW in
black and the tight-binding sets in colour, each aligned to its own reference level (the
gap centre of an insulator, E_F of a metal: the convention of ``band_validation``). GPAW
bands are read from the files kept in ``parameters/references`` (``bands_gpaw``,
``interlayer_gpaw``, ``tm_crystals``); the TB bands are computed here, with the k mesh and
smearing of the recipe that validated each set. The other sections plot and tabulate the
numbers already stored in ``validation/*.json``: nothing is refitted.

Bands are sorted by energy at each k, so two bands that cross show as two touching lines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIGURES = Path("docs/figuras")
DOCUMENT = Path("docs/GRAFICOS.md")
REFS = Path("tbkit/parameters/references")
VALIDATION = Path("validation")
DPI = 150

#: colours (light chart theme; the categorical order is fixed, never cycled)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
GPAW_STYLE = {"color": INK, "lw": 1.7, "zorder": 3}

CARBON = (("graphene", "Grafeno"), ("8,0", "(8,0)"), ("10,0", "(10,0)"), ("9,0", "(9,0)"),
          ("5,5", "(5,5)"), ("6,6", "(6,6)"))
CARBON_MODELS = (("c_bandas", "c_bandas"), ("sp3", "Xu (sp³)"), ("tang", "tang_carbon"))
LAYERS = (("AB@3.35", "Bicapa AB, 3,35 Å"), ("AA@3.35", "Bicapa AA, 3,35 Å"),
          ("graphite@3.35", "Grafito AB"), ("ABC@3.35", "Tricapa ABC (fuera del ajuste)"))
LAYER_MODELS = (("c_capas", "c_capas"), ("xu_capas", "xu_capas"))
MO = (("MoS2/bands_x0.96", "MoS₂ −4 %"), ("MoS2/bands", "MoS₂ relajado"),
      ("MoS2/bands_x1.04", "MoS₂ +4 %"), ("Mo_bcc/bands", "Mo bcc"))


# ---------------------------------------------------------------- drawing
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    return plt


def _band_panel(ax, x, ticks, ref, curves, window, title):
    """``ref`` and each curve's eps: (nk, nbands), already aligned to 0."""
    for pos, _ in ticks:
        ax.axvline(pos, color=GRID, lw=0.8, zorder=0)
    ax.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    lo, hi = window
    for j in range(ref.shape[1]):
        band = ref[:, j]
        if band.max() >= lo - 1 and band.min() <= hi + 1:
            ax.plot(x, band, **GPAW_STYLE)
    for (eps, colour) in curves:
        for j in range(eps.shape[1]):
            band = eps[:, j]
            if band.max() >= lo - 1 and band.min() <= hi + 1:
                ax.plot(x, band, color=colour, lw=1.1, zorder=2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(lo, hi)
    ax.set_xticks([p for p, _ in ticks], [lab.replace("G", "Γ") for _, lab in ticks])
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5)


def _legend(fig, labels):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], **{**GPAW_STYLE, "label": "GPAW PBE (referencia)"})]
    handles += [Line2D([], [], color=c, lw=1.6, label=lab) for lab, c in labels]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.0))


def _save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return f"figuras/{path.name}"


def _aligned(data):
    from .band_validation import reference_level

    eps = np.asarray(data["bands"], dtype=float)
    n = int(data["n_occ"])
    return eps - reference_level(eps, n, float(data["fermi_eV"]))


def _md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _f(value, digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


# ---------------------------------------------------------------- band sections
def carbon_bands(plt) -> str:
    from . import band_validation as bv

    images = []
    for (model, label), colour in zip(CARBON_MODELS, SERIES, strict=False):
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
        for ax, (name, title) in zip(axes.flat, CARBON, strict=True):
            slug = name.replace(",", "_")
            ref = json.loads((REFS / "bands_gpaw" / f"{slug}.json").read_text())
            _, x, ticks = bv.kpath(name, bv.geometry(name))
            tb = json.loads(bv.run_tb(name, model).read_text())
            _band_panel(ax, x, ticks, _aligned(ref), [(_aligned(tb), colour)], (-3.0, 3.0),
                        title)
        for ax in axes[:, 0]:
            ax.set_ylabel("E − E_ref (eV)")
        _legend(fig, [(label, colour)])
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
        images.append((label, _save(fig, f"bandas_carbono_{model}")))
        plt.close(fig)
    data = json.loads((VALIDATION / "bands_tb_vs_gpaw.json").read_text())["systems"]
    rows = []
    for name, title in CARBON:
        methods = data[name]["methods"]
        row = [title, _f(methods["gpaw_pw"].get("gap_eV"))]
        for model, _ in CARBON_MODELS:
            m = methods[f"tb_{model}"]
            row += [_f(m.get("band_rms_1eV")), _f(m.get("gap_eV"))]
        rows.append(row)
    header = ["Sistema", "gap GPAW (eV)"]
    for _, lab in CARBON_MODELS:
        header += [f"{lab}: RMS ±1 eV", f"{lab}: gap"]
    return ("## Bandas de grafeno y nanotubos\n\n"
            "GPAW PBE ondas planas (400 eV) frente a tres conjuntos de carbono. (8,0), (5,5) y "
            "grafeno a tres constantes de red se usaron para ajustar `c_bandas`; (10,0), (9,0) y "
            "(6,6) quedaron fuera del ajuste. Nanotubos de Γ a Z.\n\n"
            + "".join(f"### GPAW frente a {label}\n\n![Bandas con {label}]({image})\n\n"
                      for label, image in images)
            + _md_table(header, rows) +
            "\n\nRMS de bandas en eV, dentro de ±1 eV del nivel de referencia "
            "(`validation/bands_tb_vs_gpaw.json`).\n")


def layer_bands(plt) -> str:
    from . import interlayer as il
    from . import interlayer_fit as ilf
    from ..params import load_parameters

    models = {m: load_parameters(m) for m, _ in LAYER_MODELS}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (name, title) in zip(axes.flat, LAYERS, strict=True):
        ref = json.loads((REFS / "interlayer_gpaw" / f"{il._slug(name)}.json").read_text())
        _, x, ticks = il.kpath(il.geometry(name))
        curves = [(_aligned(ilf.tb_bands(name, models[m])), c)
                  for (m, _), c in zip(LAYER_MODELS, SERIES, strict=False)]
        _band_panel(ax, x, ticks, _aligned(ref), curves, (-3.0, 3.0), title)
    for ax in axes[:, 0]:
        ax.set_ylabel("E − E_F (eV)")
    _legend(fig, [(lab, c) for (_, lab), c in zip(LAYER_MODELS, SERIES, strict=False)])
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    image = _save(fig, "bandas_capas")
    plt.close(fig)
    report = json.loads((VALIDATION / "interlayer_tb_vs_gpaw.json").read_text())
    rows = []
    for name, entry in report["bands"].items():
        row = [name, entry["role"]]
        for m, _ in LAYER_MODELS:
            e = entry.get(m, {})
            row += [_f(e.get("band_rms_eV")), _f(e.get("k_levels_rms_eV"), 3)]
        rows.append(row)
    header = ["Sistema", "papel"]
    for _, lab in LAYER_MODELS:
        header += [f"{lab}: RMS bandas", f"{lab}: niveles en K"]
    text = ("## Bandas de bicapas y grafito\n\n"
            "GPAW PBE (bandas sin D3, que no cambia los autovalores) frente a `c_capas` y "
            "`xu_capas`, los dos con salto entre capas. ABC queda fuera del ajuste.\n\n"
            f"![Bandas de bicapas y grafito]({image})\n\n"
            + _md_table(header, rows) +
            "\n\nRMS en eV: bandas dentro de ±3 eV de E_F; niveles en K dentro de ±1,5 eV "
            "(`validation/interlayer_tb_vs_gpaw.json`).\n")
    return text + "\n" + binding_curves(plt, report)


def binding_curves(plt, report) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True, sharey=True)
    for ax, stack in zip(axes, ("AB", "AA"), strict=True):
        points = sorted((float(k.split("@")[1]), v) for k, v in report["binding"].items()
                        if k.startswith(stack + "@"))
        d = [p[0] for p in points]
        ax.plot(d, [p[1]["pbe_d3_meV"] for p in points], color=INK, lw=1.7,
                label="GPAW PBE+D3(BJ)")
        ax.plot(d, [p[1]["pbe_meV"] for p in points], color=MUTED, lw=1.2, ls=(0, (4, 2)),
                label="GPAW PBE sin D3")
        ax.plot(d, [p[1]["xu_capas_meV"] for p in points], color=SERIES[1], lw=1.4,
                label="xu_capas")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"Apilamiento {stack}", loc="left")
        ax.set_xlabel("distancia entre capas (Å)")
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("energía de enlace (meV/átomo)")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(-35, 60)
    image = _save(fig, "enlace_capas")
    plt.close(fig)
    rows = [[s, _f(e["pbe_d3"]["d_eq_A"]), _f(e["xu_capas"]["d_eq_A"]),
             _f(e["pbe_d3"]["binding_meV"], 1), _f(e["xu_capas"]["binding_meV"], 1)]
            for s, e in report["equilibrium"].items()]
    return ("### Energía de enlace entre capas\n\n"
            f"![Energía de enlace entre capas]({image})\n\n"
            + _md_table(["Apilamiento", "d_eq PBE+D3 (Å)", "d_eq xu_capas (Å)",
                         "enlace PBE+D3 (meV/át.)", "enlace xu_capas (meV/át.)"], rows) + "\n")


def mo_bands(plt) -> str:
    from ..hamiltonian import System
    from ..kpoints import mesh
    from ..params import load_parameters
    from ..scc import _solve_shifted_k, self_consistent
    from .xu_family import band_report, load_band_targets

    model = load_parameters("xu_chnosmo")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), constrained_layout=True)
    rows = []
    for ax, (key, title) in zip(axes, MO, strict=True):
        path = REFS / "tm_crystals" / f"{key}.json"
        target = load_band_targets([path])[0]
        system = System.build(target.atoms, model)
        k0, w0 = mesh(system, target.mesh)
        result = self_consistent(system, kT=0.05, tol=1e-8, kpts=k0, weights=w0, max_iter=800)
        eps = _solve_shifted_k(system, result.shift, 0.0, 0.05, target.kpts, None).energies[0]
        n = int(round(system.electrons)) // 2
        tb = _aligned({"bands": eps, "n_occ": n, "fermi_eV": float(result.solution.fermi)})
        metal = key.startswith("Mo_bcc")
        bp = target.atoms.cell.bandpath("GHNGPH" if metal else "GMKG", npoints=len(target.kpts),
                                        pbc=target.atoms.pbc)
        assert np.allclose(bp.kpts, target.kpts, atol=1e-6), key
        x, sx, labels = bp.get_linear_kpoint_axis()
        _band_panel(ax, x, list(zip(sx, labels)), target.bands, [(tb, SERIES[0])],
                    (-3.0, 3.0) if not metal else (-5.0, 3.0), title)
        r = band_report(model, target)
        names = {i: lab for i, lab in zip([int(np.argmin(np.abs(x - s))) for s in sx], labels)}

        def where(i):
            return names.get(i, "Q" if not metal and i > 50 else f"k{i}").replace("G", "Γ")

        if not metal:
            rows.append([title, _f(r["gap_gpaw_eV"]), _f(r["gap_tb_eV"]),
                         f"{where(r['vbm_k_gpaw'])} / {where(r['cbm_k_gpaw'])}",
                         f"{where(r['vbm_k_tb'])} / {where(r['cbm_k_tb'])}", _f(r["rms_eV"])])
        else:
            rows.append([title, "metal", "metal", "—", "—", _f(r["rms_eV"])])
    axes[0].set_ylabel("E − E_ref (eV)")
    _legend(fig, [("xu_chnosmo", SERIES[0])])
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9))
    image = _save(fig, "bandas_mo")
    plt.close(fig)
    return ("## Bandas de MoS₂ y Mo\n\n"
            "GPAW PBE ondas planas (500 eV, Mo con 6 electrones, sin espín-órbita) frente a "
            "`xu_chnosmo`. Todas las deformaciones de MoS₂ y el Mo bcc entraron en el ajuste: "
            "no hay cristal apartado.\n\n"
            f"![Bandas de MoS2 y Mo]({image})\n\n"
            + _md_table(["Sistema", "gap GPAW (eV)", "gap TB (eV)",
                         "máx. valencia / mín. conducción GPAW", "ídem TB",
                         "RMS bandas ±3 eV (eV)"], rows) +
            "\n\nQ: el mínimo de conducción entre K y Γ. El RMS incluye el peso de los bordes "
            "del ajuste desactivado (peso 1).\n")


# ---------------------------------------------------------------- other sections
def parity(ax, x, y, colour, label=None, marker="o"):
    ax.scatter(x, y, s=22, color=colour, edgecolor=SURFACE, linewidth=0.6, zorder=3,
               label=label, marker=marker)


def _diagonal(ax, values):
    lo, hi = min(values), max(values)
    pad = 0.05 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=0.8, zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)


def xps(plt) -> str:
    data = json.loads((VALIDATION / "xps_tb_vs_gpaw.json").read_text())
    panels = (("C", "xu_chno/hole", "C 1s"), ("N", "xu_chno/hole", "N 1s"),
              ("O", "xu_chno/hole", "O 1s"), ("Se", "xu_chnose/hole", "Se 3d"))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    for ax, (element, key, title) in zip(axes, panels, strict=True):
        rows = [r for r in data["rows"] if r["element"] == element and key in r]
        g = [r["gpaw"] for r in rows]
        t = [r[key]["total"] for r in rows]
        parity(ax, g, t, SERIES[0])
        _diagonal(ax, g + t)
        ax.set_title(f"{title} ({key.split('/')[0]})", loc="left")
        ax.set_xlabel("desplazamiento GPAW ΔSCF (eV)")
    axes[0].set_ylabel("desplazamiento TB, hueco (eV)")
    image = _save(fig, "xps")
    plt.close(fig)
    rows = []
    for k, c in data["calibration"].items():
        element, set_name, method = k.split("/")
        rows.append([element, set_name, method, c["sites"], _f(c["pearson_r_raw"], 3),
                     _f(c["mae_raw_eV"]), _f(c.get("pearson_r_calibrated"), 3),
                     _f(c.get("mae_test_calibrated_eV", c.get("mae_test_raw_eV")))])
    return ("## XPS: desplazamientos de núcleo\n\n"
            "TB (método del hueco, sin calibrar) frente a GPAW ΔSCF con setups de hueco de "
            "núcleo; desplazamientos respecto a CH₄, NH₃, H₂O y H₂Se. La diagonal es el "
            "acuerdo perfecto.\n\n"
            f"![XPS: TB frente a GPAW]({image})\n\n"
            + _md_table(["Elemento", "conjunto", "método", "sitios", "r sin calibrar",
                         "MAE sin calibrar (eV)", "r calibrado", "MAE prueba (eV)"], rows) +
            "\n\n(`validation/xps_tb_vs_gpaw.json`.)\n")


def molecular_gaps(plt) -> str:
    data = json.loads((VALIDATION / "electronic_tb_vs_gpaw.json").read_text())["molecules"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    values = []
    for role, colour, label, marker in (("train", SERIES[0], "ajuste", "o"),
                                        ("test", SERIES[1], "fuera del ajuste", "s")):
        ms = [m for m in data["molecules"] if m["role"] == role]
        g = [m["gap_gpaw_eV"] for m in ms]
        t = [m["gap_tb_eV"] for m in ms]
        values += g + t
        parity(ax, g, t, colour, label, marker)
    _diagonal(ax, values)
    ax.set_xlabel("gap HOMO-LUMO GPAW (eV)")
    ax.set_ylabel("gap HOMO-LUMO TB xu_chn (eV)")
    ax.legend(loc="upper left")
    image = _save(fig, "gaps_moleculas")
    plt.close(fig)
    rows = [[role, s["n"], _f(s["gap_mean_error_eV"]), _f(s["gap_rms_error_eV"]),
             _f(s["gap_correlation"], 3)]
            for role, s in (("ajuste", data["summary_train"]), ("fuera", data["summary_test"]))]
    return ("## Gaps de moléculas\n\n"
            f"![Gaps HOMO-LUMO]({image})\n\n"
            + _md_table(["grupo", "moléculas", "error medio (eV)", "RMS (eV)", "correlación"],
                        rows) + "\n\n(`validation/electronic_tb_vs_gpaw.json`.)\n")


def crystals(plt) -> str:
    data = json.loads((VALIDATION / "crystals_tb_vs_gpaw.json").read_text())["crystals"]
    names = [f"{c['crystal']} ({c['set']})" for c in data]
    rnd = [np.mean([p["force_rmse"] for p in c["points"] if p["label"].startswith("rnd")])
           for c in data]
    scale = [np.mean([p["force_rms_dft"] for p in c["points"] if p["label"].startswith("rnd")])
             for c in data]
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    y = np.arange(len(names))[::-1]
    ax.barh(y, rnd, height=0.6, color=SERIES[0])
    ax.set_yticks(y, names)
    ax.set_xlabel("error RMS de fuerzas TB − GPAW, puntos desplazados (eV/Å)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    image = _save(fig, "cristales_fuerzas")
    plt.close(fig)
    rows = []
    for c, r, s in zip(data, rnd, scale, strict=True):
        de = [abs(p["dE_model"] - p["dE_dft"]) for p in c["points"] if p["label"] != "relaxed"]
        rows.append([c["crystal"], c["set"], c["atoms"], _f(r), _f(s), _f(max(de))])
    return ("## Cristales y láminas dopadas\n\n"
            "Cada conjunto en su cristal, en geometrías GPAW (LCAO dzp, PBE): relajada, ±2 % y "
            "tres desplazamientos aleatorios.\n\n"
            f"![Fuerzas en cristales]({image})\n\n"
            + _md_table(["cristal", "conjunto", "átomos", "RMS fuerzas (eV/Å)",
                         "|F| GPAW típico (eV/Å)", "máx. |ΔE TB − ΔE GPAW| (eV)"], rows) +
            "\n\n(`validation/crystals_tb_vs_gpaw.json`.)\n")


def spin(plt) -> str:
    data = json.loads((VALIDATION / "spin_tb_vs_gpaw.json").read_text())["systems"]
    rows, g, t, ts, labels = [], [], [], [], []
    for name, entry in data.items():
        for state, s in entry["states"].items():
            if s["gpaw"].get("relative_eV", 0.0) == 0.0 and state != "NM":
                continue
            rg = s["gpaw"].get("relative_eV")
            rt = s["tb"].get("relative_eV")
            rs = s.get("tb_scaled", {}).get("relative_eV")
            rows.append([name, state, _f(rg, 3), _f(rt, 3), _f(rs, 3)])
            if None not in (rg, rt, rs):
                g.append(rg), t.append(rt), ts.append(rs), labels.append(f"{name} {state}")
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    parity(ax, g, t, SERIES[0], "W del átomo libre")
    parity(ax, g, ts, SERIES[1], "W escalado (λ)", "s")
    _diagonal(ax, g + t + ts)
    ax.set_xlabel("energía relativa GPAW (eV)")
    ax.set_ylabel("energía relativa TB (eV)")
    ax.legend(loc="upper left")
    image = _save(fig, "espin")
    plt.close(fig)
    return ("## Magnetismo: SCC con espín\n\n"
            "Energía de cada estado respecto al fundamental de GPAW (cintas zigzag, "
            "triangulenos, H sobre grafeno, O₂, CH₂). λ se ajustó solo a zgnr6.\n\n"
            f"![Espín: TB frente a GPAW]({image})\n\n"
            + _md_table(["sistema", "estado", "GPAW (eV)", "TB (eV)", "TB escalado (eV)"],
                        rows) + "\n\n(`validation/spin_tb_vs_gpaw.json`.)\n")


def xtb_table() -> str:
    data = json.loads((VALIDATION / "xtb_vs_tbkit.json").read_text())["molecules"]
    rows = []
    for f, entry in data.items():
        for method in ("tb", "GFN1-xTB", "GFN2-xTB"):
            m = entry[method]
            rows.append([f, entry["set"] if method == "tb" else method,
                         _f(m["force_rmse_rnd_median"]), _f(m["relative_energy_mae"], 3),
                         _f(m["gap_mae"])])
    return ("## tbkit frente a xTB, ambos contra GPAW\n\n"
            "Distancia a PBE (GFN no se ajustó a PBE: no es una medida de exactitud de "
            "GFN).\n\n"
            + _md_table(["referencias", "método", "RMS fuerzas, mediana (eV/Å)",
                         "MAE energías relativas (eV)", "MAE gap (eV)"], rows) +
            "\n\n(`validation/xtb_vs_tbkit.json`.)\n")


def infrared(plt) -> str:
    data = json.loads((VALIDATION / "ir_tb_vs_gpaw.json").read_text())
    names = list(data)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), constrained_layout=True)
    grid = np.linspace(200, 3800, 3601)
    rows = []

    def spectrum(freq, intensity, fwhm=20.0):
        g = fwhm / 2
        return sum(i * g ** 2 / ((grid - f) ** 2 + g ** 2) for f, i in zip(freq, intensity))

    for ax, name in zip(axes.flat, names, strict=False):
        d = data[name]
        ref = spectrum(d["freq"], d["gpaw"])
        tb = spectrum(d["freq"], d["onsite"])
        top = max(ref.max(), tb.max()) or 1.0
        ax.plot(grid, ref / top, color=INK, lw=1.4)
        ax.plot(grid, tb / top, color=SERIES[0], lw=1.1)
        ax.set_title(name, loc="left")
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID, lw=0.5)
        g, t = np.asarray(d["gpaw"]), np.asarray(d["onsite"])
        strong = g > 0.05 * g.max()
        rows.append([name, int(strong.sum()), _f(float(np.corrcoef(g, t)[0, 1]), 3),
                     _f(float(t.sum() / g.sum()))])
    for ax in axes[1]:
        ax.set_xlabel("número de onda (cm⁻¹)")
    _legend(fig, [("tbkit (cargas + dipolos intraatómicos)", SERIES[0])])
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    image = _save(fig, "infrarrojo")
    plt.close(fig)
    return ("## Infrarrojo de moléculas\n\n"
            "Intensidades IR en las frecuencias GPAW de cada modo (las mismas para los dos), "
            "ensanchadas con lorentzianas de 20 cm⁻¹; cada panel normalizado a su máximo.\n\n"
            f"![Espectros IR]({image})\n\n"
            + _md_table(["molécula", "modos intensos GPAW", "correlación de intensidades",
                         "suma TB / suma GPAW"], rows) +
            "\n\n(`validation/ir_tb_vs_gpaw.json`.)\n")


def coil_modes(plt) -> str:
    data = json.loads((VALIDATION / "coil204_doped_gpaw.json").read_text())
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    values = []
    for (name, label), colour, marker in zip((("pristine", "sin dopar"), ("N", "N grafítico"),
                                              ("amine", "amina")), SERIES, "os^", strict=False):
        modes = data["modes"].get(name, [])
        t = [m["tb_cm1"] for m in modes]
        g = [m["pbe_cm1"] for m in modes]
        values += t + g
        parity(ax, g, t, colour, label, marker)
    _diagonal(ax, values)
    ax.set_xlabel("frecuencia PBE (cm⁻¹)")
    ax.set_ylabel("frecuencia tbkit xu_chn (cm⁻¹)")
    ax.legend(loc="upper left")
    image = _save(fig, "coil_modos")
    plt.close(fig)
    rows = [[k, _f(v["sum_tb_km_mol"], 0), _f(v["sum_pbe_km_mol"], 0), v["modes_over_5"],
             v["within_factor_2"]] for k, v in data["ir_band_sums"].items()]
    scale = data["scale_pbe_over_tb"]
    return ("## Coil de 204 átomos: modos frente a PBE\n\n"
            f"Frecuencia PBE de cada modo TB como cociente de Rayleigh con fuerzas GPAW (cota "
            f"superior si el modo TB no es un modo PBE). Escala PBE/TB: red de carbono "
            f"{_f(scale['carbon_lattice_1000_1800']['mean'], 3)}, modos G "
            f"{_f(scale['G_modes']['mean'], 3)}.\n\n"
            f"![Modos de la coil]({image})\n\n"
            + _md_table(["estructura", "suma IR TB (km/mol)", "suma IR PBE (km/mol)",
                         "modos > 5 km/mol", "dentro de un factor 2"], rows) +
            "\n\n(`validation/coil204_doped_gpaw.json`.)\n")


def frequency_table() -> str:
    data = json.loads((VALIDATION / "xtb_vs_tbkit.json").read_text())["frequencies"]["summary"]
    names = {"tb": "tbkit", "tb_scaled": "tbkit escalado", "GFN1-xTB": "GFN1-xTB",
             "GFN2-xTB": "GFN2-xTB"}
    rows = [[names.get(k, k), v["molecules"], _f(v["median_rms_cm1"], 0), _f(v["mean_rms_cm1"], 0)]
            for k, v in data.items()]
    return ("## Frecuencias de moléculas\n\n"
            "RMS frente a las frecuencias GPAW, cada método relajando desde la geometría "
            "GPAW.\n\n"
            + _md_table(["método", "moléculas", "RMS mediana (cm⁻¹)", "RMS media (cm⁻¹)"], rows)
            + "\n\n(`validation/xtb_vs_tbkit.json`.)\n")


def _top_legend(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.0))
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9))


def _bars(ax, groups, series, colours, labels, horizontal=False):
    """Grouped bars: ``series`` is a list (one per method) of values per group."""
    n = len(series)
    width = 0.8 / n
    pos = np.arange(len(groups))
    for k, (values, colour, label) in enumerate(zip(series, colours, labels, strict=True)):
        offset = pos - 0.4 + width * (k + 0.5)
        if horizontal:
            ax.barh(offset, values, height=width * 0.9, color=colour, label=label)
        else:
            ax.bar(offset, values, width=width * 0.9, color=colour, label=label)
    if horizontal:
        ax.set_yticks(pos, groups)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.5)
    else:
        ax.set_xticks(pos, groups, rotation=45, ha="right")
        ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def crystal_energies(plt) -> str:
    data = json.loads((VALIDATION / "crystals_tb_vs_gpaw.json").read_text())["crystals"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    rows = []
    for ax, kind, title in ((axes[0], "x", "deformación ±2 %"),
                            (axes[1], "rnd", "desplazamientos aleatorios")):
        g, t = [], []
        for c in data:
            for p in c["points"]:
                if p["label"].startswith(kind):
                    g.append(p["dE_dft"] / c["atoms"] * 1000)
                    t.append(p["dE_model"] / c["atoms"] * 1000)
        parity(ax, g, t, SERIES[0])
        _diagonal(ax, g + t)
        ax.set_title(title, loc="left")
        ax.set_xlabel("ΔE GPAW (meV/átomo)")
        rows.append([title, len(g), _f(float(np.mean(np.abs(np.subtract(t, g)))), 1)])
    axes[0].set_ylabel("ΔE TB (meV/átomo)")
    image = _save(fig, "cristales_energias")
    plt.close(fig)
    return ("### Energías de los cristales\n\n"
            "Energía de cada punto respecto a la celda relajada, por átomo, los 13 cristales "
            "juntos.\n\n"
            f"![Energías en cristales]({image})\n\n"
            + _md_table(["puntos", "n", "MAE (meV/átomo)"], rows) + "\n")


def spin_profiles(plt) -> str:
    data = json.loads((VALIDATION / "spin_tb_vs_gpaw.json").read_text())["systems"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True, sharey=True)
    for ax, name in zip(axes, ("zgnr4", "zgnr6", "zgnr8"), strict=True):
        s = data[name]["states"]["AFM"]
        n_c = len(s["gpaw"]["moments"]) - 2           # the two edge H last
        x = np.arange(1, n_c + 1)
        ax.plot(x, s["gpaw"]["moments"][:n_c], color=INK, lw=1.7, label="GPAW PBE")
        ax.plot(x, s["tb"]["moments"][:n_c], color=SERIES[0], lw=1.2, label="TB, W atómico")
        ax.plot(x, s["tb_scaled"]["moments"][:n_c], color=SERIES[1], lw=1.2,
                label="TB, W escalado")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"{name}, antiferromagnético", loc="left")
        ax.set_xlabel("átomo de C a lo ancho de la cinta")
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("momento (μB)")
    axes[0].legend(loc="upper center")
    image = _save(fig, "espin_cintas")
    plt.close(fig)
    return ("### Momentos a lo ancho de las cintas zigzag\n\n"
            "Momento de Mulliken (GPAW: el de la celda de cada átomo) del estado "
            "antiferromagnético, de un borde al otro.\n\n"
            f"![Momentos en cintas zigzag]({image})\n")


def mo_crystals(plt) -> str:
    from ..params import load_parameters
    from .xu_family import crystal_validation, load_crystal_points

    model = load_parameters("xu_chnosmo")
    refs = load_crystal_points([REFS / "tm_crystals" / "MoS2", REFS / "tm_crystals" / "Mo_bcc"])
    report = crystal_validation(model, refs)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    rows = []
    for ax, name, title in ((axes[0], "MoS2", "MoS₂ (energía por celda de 3 átomos)"),
                            (axes[1], "Mo_bcc", "Mo bcc (deformación solo en el plano a, b)")):
        points = report[name]["points"]
        strain = [(0.98, "x0.98"), (1.0, "relaxed"), (1.02, "x1.02")]
        x = [s for s, _ in strain]
        ax.plot(x, [points[k]["delta_e_gpaw"] for _, k in strain], color=INK, lw=1.7,
                marker="o", ms=5, label="GPAW PBE")
        ax.plot(x, [points[k]["delta_e_tb"] for _, k in strain], color=SERIES[0], lw=1.3,
                marker="s", ms=5, label="xu_chnosmo")
        ax.set_title(title, loc="left")
        ax.set_xlabel("a / a₀")
        ax.grid(color=GRID, lw=0.5)
        for label, pt in points.items():
            rows.append([name, label, _f(pt["force_rms"]), _f(pt["force_rms_gpaw"]),
                         _f(pt["delta_e_tb"], 3), _f(pt["delta_e_gpaw"], 3)])
    axes[0].set_ylabel("E − E(a₀) (eV)")
    axes[0].legend(loc="upper center")
    image = _save(fig, "mo_energias")
    plt.close(fig)
    return ("### Energía y fuerzas de MoS₂ y Mo con xu_chnosmo\n\n"
            "Solo tres puntos por cristal (−2 %, 0, +2 %): las líneas los unen, no son una "
            "curva calculada. La red de MoS₂ sale demasiado rígida, y en Mo bcc la celda de "
            "GPAW no es un mínimo de tbkit (las dos deformaciones bajan la energía): "
            "`xu_chnosmo` no sirve para la energía de la red del metal.\n\n"
            f"![Energía de MoS2 y Mo]({image})\n\n"
            + _md_table(["cristal", "punto", "RMS fuerzas TB − GPAW (eV/Å)",
                         "|F| GPAW (eV/Å)", "ΔE TB (eV)", "ΔE GPAW (eV)"], rows) + "\n")


def graphene_strain_bands(plt) -> str:
    from . import band_validation as bv

    cases = (("graphene@2.42", "graphene_a2.42", "a = 2,42 Å"),
             ("graphene", "graphene", "a = 2,467 Å"),
             ("graphene@2.52", "graphene_a2.52", "a = 2,52 Å"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True)
    for ax, (name, slug, title) in zip(axes, cases, strict=True):
        ref = json.loads((REFS / "bands_gpaw" / f"{slug}.json").read_text())
        _, x, ticks = bv.kpath(name, bv.geometry(name))
        tb = json.loads(bv.run_tb(name, "c_bandas").read_text())
        _band_panel(ax, x, ticks, _aligned(ref), [(_aligned(tb), SERIES[0])], (-9.0, 4.0),
                    f"Grafeno, {title}")
    axes[0].set_ylabel("E − E_F (eV)")
    _legend(fig, [("c_bandas", SERIES[0])])
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9))
    image = _save(fig, "bandas_grafeno_deformado")
    plt.close(fig)
    data = json.loads((VALIDATION / "c_bandas_fit.json").read_text())["systems"]
    rows = []
    for name, _, title in cases:
        e = data[name]
        rows.append([title, _f(e["c_bandas"]["band_rms_eV"]), _f(e["xu_carbon"]["band_rms_eV"]),
                     _f(e["c_bandas"]["fermi_velocity_gpaw_eV_A"]),
                     _f(e["c_bandas"]["fermi_velocity_eV_A"]),
                     _f(e["xu_carbon"]["fermi_velocity_eV_A"])])
    return ("### Grafeno deformado: el ajuste de c_bandas\n\n"
            "Las tres constantes de red con las que se ajustaron las leyes de distancia de "
            "`c_bandas`; ventana del ajuste −9 a +3 eV.\n\n"
            f"![Grafeno a tres constantes de red]({image})\n\n"
            + _md_table(["grafeno", "RMS c_bandas (eV)", "RMS Xu (eV)", "v_F GPAW (eV·Å)",
                         "v_F c_bandas", "v_F Xu"], rows) +
            "\n\n(`validation/c_bandas_fit.json`.)\n")


METHODS = (("tb", "tbkit"), ("tb_scaled", "tbkit escalado"), ("GFN1-xTB", "GFN1-xTB"),
           ("GFN2-xTB", "GFN2-xTB"))


def frequencies_by_file(plt) -> str:
    data = json.loads((VALIDATION / "xtb_vs_tbkit.json").read_text())["frequencies"]["per_file"]
    files = list(data)
    series = [[float(np.median([m[key] for m in data[f].values() if key in m])) for f in files]
              for key, _ in METHODS]
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    _bars(ax, [f.replace("gpaw_", "").replace("_hessians.json", "")
               .replace("_frequencies.json", "").replace(".json", "") for f in files],
          series, SERIES, [lab for _, lab in METHODS])
    ax.set_ylabel("RMS de frecuencias, mediana (cm⁻¹)")
    _top_legend(fig, ax, 4)
    image = _save(fig, "frecuencias_por_archivo")
    plt.close(fig)
    return ("### Frecuencias por archivo de referencia\n\n"
            f"![Frecuencias por archivo]({image})\n")


def xtb_layers(plt) -> str:
    data = json.loads((VALIDATION / "xtb_vs_tbkit.json").read_text())["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True, sharey=True)
    for ax, stack in zip(axes, ("AB", "AA"), strict=True):
        pts = sorted((float(k.split("@")[1]), v) for k, v in data.items()
                     if k.startswith(stack + "@"))
        d = [p[0] for p in pts]
        ax.plot(d, [p[1]["pbe_d3"] for p in pts], color=INK, lw=1.7, label="GPAW PBE+D3(BJ)")
        for (key, label), colour in zip((("xu_capas", "xu_capas"), ("GFN1-xTB", "GFN1-xTB"),
                                         ("GFN2-xTB", "GFN2-xTB")), SERIES, strict=False):
            ax.plot(d, [p[1][key] for p in pts], color=colour, lw=1.3, label=label)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"Apilamiento {stack}", loc="left")
        ax.set_xlabel("distancia entre capas (Å)")
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("energía de enlace (meV/átomo)")
    axes[0].legend(loc="upper right")
    image = _save(fig, "capas_xtb")
    plt.close(fig)
    return ("### Bicapas: tbkit y xTB frente a PBE+D3\n\n"
            f"![Bicapas con xTB]({image})\n")


def timing(plt) -> str:
    data = json.loads((VALIDATION / "xtb_vs_tbkit.json").read_text())["timing"]
    rows = data["rows"]
    n = [r["atoms"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 4), constrained_layout=True)
    for (key, label), colour in zip((("tb", "tbkit xu_chn"), ("GFN1-xTB", "GFN1-xTB"),
                                     ("GFN2-xTB", "GFN2-xTB")), SERIES, strict=False):
        ax.plot(n, [r[key] for r in rows], color=colour, lw=1.4, marker="o", ms=4, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("átomos")
    ax.set_ylabel("segundos por energía + fuerzas")
    ax.grid(color=GRID, lw=0.5, which="both")
    ax.legend(loc="upper left")
    image = _save(fig, "tiempos")
    plt.close(fig)
    return ("### Tiempo de cálculo\n\n"
            f"{data['note'].capitalize()}.\n\n![Tiempo de cálculo]({image})\n")


def not_adopted(plt) -> str:
    parts = ["## Medido, no adoptado\n\nComparaciones con GPAW que llevaron a no adoptar "
             "un método. El error que cuenta es el de validación cruzada o el de las "
             "estructuras apartadas, no el de entrenamiento.\n"]
    # Born charges
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    rows = []
    for ax, f in zip(axes, ("chn", "chno"), strict=True):
        d = json.loads((VALIDATION / f"ir_charge_models_{f}.json").read_text())
        models = list(d["born_fits"])
        tb = d["born_fits"][models[0]]["rms_tb_e"]
        train = [d["born_fits"][m]["rms_train_e"] for m in models]
        cv = [d["born_fits"][m]["rms_cv_e"] for m in models]
        names = {"bond_flux": "flujo de enlace", "environment_flux": "entorno (SOAP)",
                 "both": "ambos"}
        _bars(ax, [names.get(m, m) for m in models], [train, cv], SERIES[:2],
              ["entrenamiento", "validación cruzada"])
        ax.axhline(tb, color=INK, lw=1.4, label="tbkit sin corrección")
        ax.set_title(f"cargas de Born, xu_{f}", loc="left")
        ax.set_ylabel("RMS frente a GPAW (e)")
        held = d["ir_on_tb_modes_heldout"]
        for m in ["tb"] + models:
            rows.append([f"xu_{f}", names.get(m, "tbkit"), _f(held["median_abs_log10"][m], 3),
                         _f(held["within_factor_2"][m], 2)])
    _top_legend(fig, axes[0], 3)
    image = _save(fig, "born_modelos")
    plt.close(fig)
    parts.append("### Cargas de Born corregidas para el IR\n\n"
                 f"![Modelos de carga]({image})\n\n"
                 + _md_table(["conjunto", "modelo", "|log10 I_TB/I_GPAW| mediana, modos "
                              "apartados", "fracción dentro de un factor 2"], rows) + "\n")
    # MACE
    mace = json.loads((VALIDATION / "mace_mp_vs_gpaw.json").read_text())["molecules_chno_hessians"]
    tbm = json.loads((VALIDATION / "xtb_vs_tbkit.json").read_text())["frequencies"]["per_file"]
    tbm = tbm.get("gpaw_chno_hessians.json", {})
    mols = [m for m in mace["small"] if m in tbm]
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    _bars(ax, mols, [[tbm[m]["tb"] for m in mols], [mace["small"][m] for m in mols],
                     [mace["medium"][m] for m in mols]], SERIES[:3],
          ["tbkit xu_chno", "MACE-MP-0 small", "MACE-MP-0 medium"])
    ax.set_ylabel("RMS de frecuencias frente a GPAW (cm⁻¹)")
    _top_legend(fig, ax, 3)
    image = _save(fig, "mace")
    plt.close(fig)
    parts.append("### MACE-MP-0 sin ajuste fino\n\n"
                 f"![MACE frente a tbkit]({image})\n")
    # Δ-learning
    d = json.loads((VALIDATION / "delta_learning.json").read_text())["tang_carbon_on_gpaw_carbon_env"]
    groups = list(d["groups"])
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    _bars(ax, [g + (" (apartado)" if d["groups"][g]["held_out"] else "") for g in groups],
          [[d["groups"][g]["tb_force_rmse"] for g in groups],
           [d["groups"][g]["tb_delta_force_rmse"] for g in groups]], SERIES[:2],
          ["tang_carbon", "tang_carbon + Δ (SOAP)"])
    ax.set_ylabel("RMS de fuerzas frente a GPAW (eV/Å)")
    _top_legend(fig, ax, 2)
    image = _save(fig, "delta")
    plt.close(fig)
    cv = d["cv_force_rmse"]
    parts.append("### Corrección Δ lineal (SOAP)\n\n"
                 f"![Corrección delta]({image})\n\n"
                 f"Validación cruzada por grupos, RMS de fuerzas con el mejor ridge: "
                 f"{_f(min(cv.values()))} eV/Å.\n")
    # Se with d
    d = json.loads((VALIDATION / "xu_chnose_d_vs_xu_chnose.json").read_text())
    held = d["frequencies_held_out"]
    drift = {k: v for k, v in d["relax_max_drift_A"].items() if k != "order"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    names = list(held)
    _bars(axes[0], names, [[held[m]["xu_chnose"] for m in names],
                           [held[m]["xu_chnose_d"] for m in names]], SERIES[:2],
          ["xu_chnose (sin d)", "xu_chnose_d (con d)"])
    axes[0].set_ylabel("RMS de frecuencias, moléculas apartadas (cm⁻¹)")
    _top_legend(fig, axes[0], 2)
    names = list(drift)
    _bars(axes[1], names, [[drift[m][1] for m in names], [drift[m][0] for m in names]],
          SERIES[:2], ["xu_chnose (sin d)", "xu_chnose_d (con d)"])
    axes[1].set_ylabel("deriva máxima al relajar desde GPAW (Å)")
    image = _save(fig, "se_d")
    plt.close(fig)
    parts.append("### Se con orbitales d (publicado como conjunto aparte)\n\n"
                 "Con d mejora el Se hipervalente (deriva de DMSeO y SeO₂) y empeoran las "
                 "frecuencias de varias moléculas apartadas.\n\n"
                 f"![Se con y sin d]({image})\n")
    return "\n".join(parts)


def spectra_header(plt) -> str:
    return ("## Espectros y resultados calculados\n\n"
            "Raman, IR y XPS que tbkit ya calculó, con GPAW donde hay referencia.\n")


def graphene_raman(plt) -> str:
    data = json.loads((VALIDATION / "graphene_2d_gpaw.json").read_text())
    grid = np.asarray(data["spectra"]["grid"])
    lasers = [k for k in data["spectra"] if k != "grid"]
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    for laser, colour in zip(lasers, SERIES, strict=False):
        y = np.asarray(data["spectra"][laser])
        ax.plot(grid, y / y.max(), color=colour, lw=1.3, label=f"láser {laser.replace('.', ',')} eV")
    ax.set_xlabel("desplazamiento Raman (cm⁻¹)")
    ax.set_ylabel("intensidad (normalizada)")
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(loc="upper right")
    image = _save(fig, "raman_grafeno_2d")
    plt.close(fig)
    rows = [[_f(r["laser_ev"]), _f(r["g_frequency"], 0), _f(r["2D_position"], 0),
             _f(r["2D'_position"], 0), _f(r["2D_intensity"] / r["g_intensity"], 1)]
            for r in data["results"]]
    return ("### Grafeno: bandas 2D y 2D' por doble resonancia\n\n"
            "Fonones GPAW PBE (LCAO dzp, 6×6) con electrones del modelo π; la dispersión de "
            "la 2D con el láser es la de la doble resonancia.\n\n"
            f"![Raman 2D del grafeno]({image})\n\n"
            + _md_table(["láser (eV)", "G (cm⁻¹)", "2D (cm⁻¹)", "2D' (cm⁻¹)", "I(2D)/I(G)"],
                        rows) + "\n\n(`validation/graphene_2d_gpaw.json`.)\n")


def nanotube_raman(plt) -> str:
    data = json.loads((VALIDATION / "cnt_xu_carbon.json").read_text())
    d = np.array([t["diameter_nm"] for t in data])
    order = np.argsort(d)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(d[order], [data[i]["rbm"] for i in order], color=SERIES[0], lw=1.4, marker="o",
            ms=5, label="tbkit (Xu)")
    ax.plot(d[order], [data[i]["rbm_araujo"] for i in order], color=INK, lw=1.4,
            label="Araujo 2008 (experimental)")
    ax.plot(d[order], [data[i]["rbm_jorio"] for i in order], color=MUTED, lw=1.2,
            ls=(0, (4, 2)), label="Jorio 2001 (experimental)")
    ax.set_xlabel("diámetro (nm)")
    ax.set_ylabel("RBM (cm⁻¹)")
    ax.set_title("modo radial (RBM)", loc="left")
    ax.legend(loc="upper right")
    ax.grid(color=GRID, lw=0.5)
    ax = axes[1]
    ax.plot(d[order], [data[i]["g_plus"] for i in order], color=SERIES[0], lw=1.4, marker="o",
            ms=5, label="G⁺")
    ax.plot(d[order], [data[i]["g_minus"] for i in order], color=SERIES[1], lw=1.4, marker="s",
            ms=5, label="G⁻")
    ax.set_xlabel("diámetro (nm)")
    ax.set_ylabel("frecuencia (cm⁻¹)")
    ax.set_title("banda G", loc="left")
    ax.legend(loc="lower right")
    ax.grid(color=GRID, lw=0.5)
    image = _save(fig, "raman_nanotubos")
    plt.close(fig)
    rows = [[t["tube"], "metálico" if t["kind"] == "metal" else "semiconductor",
             _f(t["diameter_nm"], 3), _f(t["rbm"], 0), _f(t["rbm_araujo"], 0), _f(t["g_plus"], 0),
             _f(t["g_minus"], 0)] for t in (data[i] for i in order)]
    return ("### Nanotubos: RBM y G con Xu\n\n"
            "Frecuencias de Γ de tbkit (Xu, sin escalar) frente a las relaciones experimentales "
            "RBM(d). La G⁻ de los tubos metálicos depende de la malla k (anomalía de Kohn).\n\n"
            f"![RBM y G de nanotubos]({image})\n\n"
            + _md_table(["tubo", "tipo", "d (nm)", "RBM (cm⁻¹)", "RBM Araujo", "G⁺", "G⁻"],
                        rows) + "\n\n(`validation/cnt_xu_carbon.json`.)\n")


def coil_raman(plt) -> str:
    data = json.loads((VALIDATION / "coil204_doped_raman.json").read_text())["summary_by_laser_eV"]
    names = {"pristine": "sin dopar", "N": "N grafítico", "amine": "amina"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for (key, ylabel), ax in zip((("G_centroid", "centroide G (cm⁻¹)"),
                                  ("D_centroid", "centroide de la ventana D (cm⁻¹)"),
                                  ("D_over_G", "I(D)/I(G)")), axes, strict=True):
        for (name, label), colour, marker in zip(names.items(), SERIES, "os^", strict=False):
            lasers = sorted(data[name], key=float)
            ax.plot([float(x) for x in lasers], [data[name][x][key] for x in lasers],
                    color=colour, lw=1.4, marker=marker, ms=5, label=label)
        ax.set_xlabel("energía del láser (eV)")
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, lw=0.5)
    axes[0].legend(loc="best")
    image = _save(fig, "raman_coil")
    plt.close(fig)
    unc = json.loads((VALIDATION / "coil204_raman_uncertainty.json").read_text())["eta"]
    rows = []
    for name, label in names.items():
        for eta in sorted(unc[name], key=float):
            r = unc[name][eta].get("2.33", {})
            rows.append([label, eta.replace(".", ","), _f(r.get("G_centroid"), 0),
                         _f(r.get("D_centroid"), 0), _f(r.get("ID_IG"), 3)])
    return ("### Coil de 204 átomos: Raman resonante\n\n"
            "tbkit `xu_chn`, Raman resonante a tres láseres. No hay una banda D separada: "
            "la «ventana D» es la intensidad entre 1250 y 1450 cm⁻¹.\n\n"
            f"![Raman de la coil]({image})\n\n"
            "Sensibilidad al ensanchamiento η (láser 2,33 eV):\n\n"
            + _md_table(["estructura", "η (eV)", "G (cm⁻¹)", "ventana D (cm⁻¹)", "I(D)/I(G)"],
                        rows) +
            "\n\n(`validation/coil204_doped_raman.json`, `coil204_raman_uncertainty.json`.)\n")


def coil_ir(plt) -> str:
    data = json.loads((VALIDATION / "coil204_doped_raman.json").read_text())["ir"]["by_structure"]
    names = {"pristine": "sin dopar", "N": "N grafítico", "amine": "amina"}
    bands = list(next(iter(data.values()))["band_fractions"])
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    _bars(ax, [b + " cm⁻¹" for b in bands],
          [[data[n]["band_fractions"][b] for b in bands] for n in names], SERIES[:3],
          list(names.values()))
    ax.set_ylabel("fracción de la intensidad IR")
    _top_legend(fig, ax, 3)
    image = _save(fig, "ir_coil")
    plt.close(fig)
    rows = [[label, _f(data[n]["total_km_mol"], 0),
             ", ".join(f"{f:.0f} ({i:.0f})" for f, i in data[n]["strongest_cm1_km_mol"][:5])]
            for n, label in names.items()]
    return ("### Coil de 204 átomos: infrarrojo\n\n"
            f"![IR de la coil]({image})\n\n"
            + _md_table(["estructura", "intensidad total (km/mol)",
                         "modos más intensos: cm⁻¹ (km/mol)"], rows) +
            "\n\n(`validation/coil204_doped_raman.json`.)\n")


def xps_spectra(plt) -> str:
    data = json.loads((VALIDATION / "xps_tb_vs_gpaw.json").read_text())["rows"]
    systems = ("coronene", "coronene_N_graphitic", "coronene_N_pyridinic", "coronene_epoxide",
               "coronene_diol")
    fig, axes = plt.subplots(1, len(systems), figsize=(14, 3.4), constrained_layout=True)
    grid = np.linspace(-4, 4, 801)
    sigma = 0.35

    def curve(values):
        return sum(np.exp(-0.5 * ((grid - v) / sigma) ** 2) for v in values)

    for ax, name in zip(axes, systems, strict=True):
        rows = [r for r in data if r["system"] == name and r["element"] == "C"]
        g = [r["gpaw"] for r in rows]
        t = [r["xu_chno/hole"]["total"] for r in rows]
        ax.plot(grid, curve(g), color=INK, lw=1.6, label="GPAW ΔSCF")
        ax.plot(grid, curve(t), color=SERIES[0], lw=1.3, label="tbkit (hueco)")
        ax.set_title({"coronene": "coroneno", "coronene_N_graphitic": "coroneno, N grafítico",
                      "coronene_N_pyridinic": "coroneno, N piridínico",
                      "coronene_epoxide": "coroneno, epóxido",
                      "coronene_diol": "coroneno, diol"}[name], loc="left")
        ax.set_xlabel("desplazamiento C 1s (eV)")
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID, lw=0.5)
    _top_legend(fig, axes[0], 2)
    image = _save(fig, "xps_espectros")
    plt.close(fig)
    return ("### XPS C 1s de láminas (coroneno)\n\n"
            "Un pico gaussiano (σ 0,35 eV) por sitio inequivalente calculado, sin multiplicidad: "
            "las alturas no son las de un espectro medido, las posiciones sí.\n\n"
            f"![Espectros C 1s]({image})\n")


def coil_sites(plt) -> str:
    n = json.loads((VALIDATION / "coil204_N_graphitic.json").read_text())["comparison"]
    a = json.loads((VALIDATION / "coil204_amine_tb.json").read_text())["gpaw_check"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    _bars(axes[0], [f"sitio {r['site']} ({r['rings']})" for r in n],
          [[r["gpaw_relative_eV"] for r in n], [r["tb_relative_eV"] for r in n]],
          [INK, SERIES[0]], ["GPAW", "tbkit xu_chn"])
    axes[0].set_ylabel("energía relativa (eV)")
    axes[0].set_title("N grafítico en la coil", loc="left")
    names = list(a)
    _bars(axes[1], names, [[a[k]["gpaw"] for k in names], [a[k]["tb"] for k in names]],
          [INK, SERIES[0]], ["GPAW", "tbkit xu_chn"])
    axes[1].set_title("amina (-NH₂ + H) en la coil", loc="left")
    _top_legend(fig, axes[0], 2)
    image = _save(fig, "coil_sitios")
    plt.close(fig)
    return ("### Coil: sitio preferido del dopante\n\n"
            "Energía de cada sitio respecto al más estable, GPAW en la geometría TB.\n\n"
            f"![Sitios del dopante]({image})\n")


SECTIONS = {"bandas": ("carbon_bands", "graphene_strain_bands", "layer_bands", "mo_bands",
                       "mo_crystals"),
            "otros": ("crystals", "crystal_energies", "molecular_gaps", "infrared",
                      "frequency_table", "frequencies_by_file", "coil_modes", "spin",
                      "spin_profiles", "xps", "xtb_table", "xtb_layers", "timing",
                      ),
            "espectros": ("spectra_header", "graphene_raman", "nanotube_raman", "coil_raman",
                          "coil_ir", "xps_spectra", "coil_sites"),
            "no_adoptado": ("not_adopted",)}


def build(only: str | None = None) -> Path:
    plt = _plt()
    parts = ["# Validación frente a GPAW: gráficos y tablas\n",
             "Generado por `python -m tbkit.recipes.validation_figures` desde las referencias "
             "GPAW de `tbkit/parameters/references` y los resultados de `validation/`. No "
             "editar a mano.\n"]
    for section, functions in SECTIONS.items():
        if only and section != only:
            continue
        for name in functions:
            fn = globals()[name]
            parts.append(fn() if name.endswith("_table") else fn(plt))
            print(name, "hecho", flush=True)
    if only is None:
        DOCUMENT.write_text("\n".join(parts), encoding="utf-8")
    return DOCUMENT


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("section", nargs="?", choices=sorted(SECTIONS),
                        help="solo una sección (las figuras; el documento se escribe con todas)")
    args = parser.parse_args(argv)
    print(build(args.section))


if __name__ == "__main__":
    main()
