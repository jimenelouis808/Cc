"""Figures of the coil's results WITH their uncertainties (the first versions stay in paginas/).

Run from packages/tbkit:

    uv run python resultados/coil204/incertidumbre/hacer_figuras.py datos     # needs out/
    uv run python resultados/coil204/incertidumbre/hacer_figuras.py figuras   # datos/ only

``datos`` reads what was already computed (nothing new): the resonant-Raman tensors with
η = 0.05 and 0.2 cached by ``doped_raman eta`` (the 40 strongest modes of each structure),
the η = 0.1 ones, ``validation/coil204_raman_uncertainty.json`` (η and model spread of the band
centroids) and the GPAW mode check of the bundle (PBE/tbkit frequency ratios). It writes
CSV/JSON into ``datos/``. ``figuras`` draws from ``datos/`` alone into ``figuras/`` (PDF, PNG
600 dpi, one journal column), never touching ``paginas/``.

What each uncertainty is:
* η (the width of the intermediate states of the resonant Raman, 0.05–0.2 eV): moves the
  relative heights, hardly the positions;
* the model (xu_chn, xu_carbon, tang_carbon; pristine coil only): positions and I_D/I_G;
* the frequency scale against PBE: the PBE/tbkit ratio of the checked modes (GPAW, ± along
  each tbkit mode), whose spread is the uncertainty of any scaled position.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
DATA, FIGS = HERE / "datos", HERE / "figuras"
SYSTEMS = {"pristine": "sin dopar", "N": "N grafítico", "amine": "–NH₂ + H"}
ETAS = (0.05, 0.1, 0.2)
GRID = np.linspace(900.0, 1800.0, 901)


# --------------------------------------------------------------------------- data

def _curve(frequencies, activities, laser_ev):
    from tbkit.raman import RamanResult, spectrum

    r = RamanResult(np.asarray(frequencies), np.asarray(activities), np.zeros(len(activities)),
                    None, None, "", np.asarray(frequencies))
    _, y = spectrum(r, GRID, fwhm=10.0, laser_nm=1239.84193 / laser_ev)
    return y


def extract() -> None:
    """From out/ (cached tensors only) and the validation file into datos/."""
    from tbkit.recipes import doped_raman as dr

    DATA.mkdir(parents=True, exist_ok=True)
    for name in SYSTEMS:
        full = dr.raman(name)                                # η = 0.1, every mode (cached)
        top = dr.strongest_modes(name)                       # the 40 the η scan used
        pick = [list(full.mode_indices).index(m) for m in top]
        columns = {"desplazamiento_cm1": GRID}
        for li, laser in enumerate(dr.LASERS_EV):
            columns[f"eta0.1_{laser:.2f}"] = _curve(full.frequencies[pick],
                                                    full.activities[li][pick], laser)
        for eta in (0.05, 0.2):
            r = dr.raman_eta(name, eta)                      # cached tensors
            for li, laser in enumerate(dr.LASERS_EV):
                columns[f"eta{eta:g}_{laser:.2f}"] = _curve(r.frequencies, r.activities[li],
                                                            laser)
        header = ",".join(columns)
        np.savetxt(DATA / f"raman_eta_{name}.csv", np.column_stack(list(columns.values())),
                   delimiter=",", header=header, comments="")
        print(f"{name}: espectros con η 0.05, 0.1 y 0.2 (40 modos más fuertes)")
    uncertainty = json.loads((BUNDLE.parents[1] / "validation" /
                              "coil204_raman_uncertainty.json").read_text())
    (DATA / "centroides_eta_modelo.json").write_text(json.dumps(uncertainty, indent=1))
    gpaw = json.loads((BUNDLE / "validacion_gpaw_modos" / "report.json").read_text())
    rows = [{"sistema": n, "modo": m["mode"], "tb_cm1": m["tb_cm1"], "pbe_cm1": m["pbe_cm1"],
             "pbe_sobre_tb": m["pbe_over_tb"], "por": "+".join(m["why"])}
            for n in SYSTEMS for m in gpaw[n]["modes"] if m.get("pbe_over_tb")]
    (DATA / "cociente_pbe_tbkit.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"cocientes PBE/tbkit: {len(rows)} modos")


# --------------------------------------------------------------------------- figures

def _style(width="simple", height=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tbkit.report import WIDTHS_IN, journal_style

    style = journal_style(width)
    if height is not None:
        style["figure.figsize"] = (WIDTHS_IN[width], height)
    plt.rcParams.update(style)
    return plt


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=600 if ext == "png" else None)
    print(f"figura: {name}")


def fig_raman_eta() -> None:
    """Per laser: the three structures stacked, η = 0.1 line, η 0.05–0.2 band; each curve
    normalised to its own G maximum (the shape is what η changes)."""
    from tbkit.report import OKABE_ITO

    for laser in (1.96, 2.33, 2.54):
        plt = _style(height=3.6)
        fig, axes = plt.subplots(3, 1, sharex=True)
        for ax, (name, label), color in zip(axes, SYSTEMS.items(), OKABE_ITO, strict=False):
            data = np.genfromtxt(DATA / f"raman_eta_{name}.csv", delimiter=",", names=True)
            x = data["desplazamiento_cm1"]
            g = (x > 1500) & (x < 1700)
            curves = []
            for eta in ETAS:
                y = data[_col(data, eta, laser)]
                curves.append(y / y[g].max())
            low, high = np.min(curves, axis=0), np.max(curves, axis=0)
            ax.fill_between(x, low, high, color=color, alpha=0.25, linewidth=0,
                            label="η 0.05–0.2 eV")
            ax.plot(x, curves[1], color=color, lw=0.9, label="η 0.1 eV")
            ax.set_ylim(0, 1.15)
            ax.set_yticks([0, 0.5, 1])
            ax.text(0.02, 0.86, label, transform=ax.transAxes, fontsize=6.5)
        axes[0].legend(loc="upper right", fontsize=6)
        axes[1].set_ylabel("Intensidad / máx. de G")
        axes[-1].set_xlabel("Desplazamiento Raman (cm$^{-1}$)")
        axes[-1].set_xlim(900, 1800)
        fig.subplots_adjust(hspace=0.08)
        _save(fig, f"raman_banda_eta_{laser:.2f}".replace(".", "_"))
        plt.close(fig)


def _col(data, eta, laser):
    """genfromtxt drops the dots of the header names: find the column of (η, laser)."""
    want = f"eta{eta:g}_{laser:.2f}".replace(".", "")
    for name in data.dtype.names:
        if name.replace("_", "") == want.replace("_", ""):
            return name
    raise KeyError(want)


def fig_centroids() -> None:
    """G and D (first-order estimate) centroids and I_D/I_G per structure and laser: marker at
    η = 0.1 eV, bar over η 0.05–0.2 eV."""
    from tbkit.report import OKABE_ITO

    u = json.loads((DATA / "centroides_eta_modelo.json").read_text())["eta"]
    lasers = ("1.96", "2.33", "2.54")
    for key, label, name in (("G_centroid", "Centroide de G (cm$^{-1}$)", "centroide_G_eta"),
                             ("D_centroid", "D estimada (cm$^{-1}$)", "centroide_D_eta"),
                             ("ID_IG", "I_D/I_G (estimación de primer orden)", "id_ig_eta")):
        plt = _style()
        fig, ax = plt.subplots()
        for k, ((sys_key, sys_label), color) in enumerate(zip(SYSTEMS.items(), OKABE_ITO,
                                                             strict=False)):
            x = np.array([float(l) for l in lasers]) + (k - 1) * 0.025
            mid = np.array([u[sys_key]["0.1"][l][key] for l in lasers])
            values = np.array([[u[sys_key][e][l][key] for e in ("0.05", "0.1", "0.2")]
                               for l in lasers])
            err = np.vstack([mid - values.min(axis=1), values.max(axis=1) - mid])
            ax.errorbar(x, mid, yerr=err, fmt="o", ms=4, color=color, capsize=2, lw=0.8,
                        label=sys_label)
        ax.set_xlabel(r"$\hbar\omega_L$ (eV)")
        ax.set_ylabel(_label(label))
        ax.set_xticks([1.96, 2.33, 2.54])
        ax.legend(fontsize=6)
        _save(fig, name)
        plt.close(fig)


def _label(text):
    return text.replace("I_D/I_G", "$I_D/I_G$")


def fig_models() -> None:
    """Pristine coil, three models: G and D centroids and I_D/I_G per laser (η = 0.1 eV)."""
    from tbkit.report import OKABE_ITO

    m = json.loads((DATA / "centroides_eta_modelo.json").read_text())["model_pristine"]
    names = {"xu_chn": "xu_chn (SCC)", "xu_carbon": "Xu (carbono)", "tang_carbon": "Tang"}
    lasers = ("1.96", "2.33", "2.54")
    plt = _style("doble", height=2.0)
    fig, axes = plt.subplots(1, 3)
    for ax, (key, label) in zip(axes, (("G_centroid", "Centroide de G (cm$^{-1}$)"),
                                      ("D_centroid", "D estimada (cm$^{-1}$)"),
                                      ("ID_IG", "I_D/I_G")), strict=True):
        for k, ((model, model_label), color) in enumerate(zip(names.items(), OKABE_ITO,
                                                             strict=False)):
            x = np.array([float(l) for l in lasers]) + (k - 1) * 0.025
            ax.plot(x, [m[model][l][key] for l in lasers], "o", ms=4, color=color,
                    label=model_label)
        ax.set_xlabel(r"$\hbar\omega_L$ (eV)")
        ax.set_ylabel(_label(label))
        ax.set_xticks([1.96, 2.33, 2.54])
    axes[0].legend(fontsize=6)
    fig.subplots_adjust(wspace=0.45)
    _save(fig, "modelos_sin_dopar")
    plt.close(fig)


def fig_scale() -> None:
    """PBE/tbkit frequency ratio of every GPAW-checked mode against its tbkit frequency: the
    spread is the uncertainty of a scaled position (median line)."""
    from tbkit.report import OKABE_ITO

    rows = json.loads((DATA / "cociente_pbe_tbkit.json").read_text())
    plt = _style()
    fig, ax = plt.subplots()
    ratios = np.array([r["pbe_sobre_tb"] for r in rows])
    median = float(np.median(ratios))
    ax.axhline(1.0, color="0.6", lw=0.6)
    ax.axhline(median, color="0.25", lw=0.8, ls="--")
    ax.text(2300, median + 0.002, f"mediana {median:.3f}", fontsize=6, va="bottom")
    for (key, label), color in zip(SYSTEMS.items(), OKABE_ITO, strict=False):
        mine = [r for r in rows if r["sistema"] == key]
        ax.plot([r["tb_cm1"] for r in mine], [r["pbe_sobre_tb"] for r in mine], "o", ms=4,
                color=color, label=label)
    ax.set_xlabel("Frecuencia tbkit (cm$^{-1}$)")
    ax.set_ylabel("PBE / tbkit")
    ax.legend(fontsize=6)
    _save(fig, "cociente_pbe_tbkit")
    plt.close(fig)
    summary = {"mediana": median, "min": float(ratios.min()), "max": float(ratios.max()),
               "n": len(ratios)}
    (DATA / "cociente_resumen.json").write_text(json.dumps(summary, indent=1))


def figures() -> None:
    fig_raman_eta()
    fig_centroids()
    fig_models()
    fig_scale()


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "figuras"
    if step == "datos":
        extract()
    elif step == "figuras":
        figures()
    else:
        raise SystemExit("uso: hacer_figuras.py datos|figuras")
