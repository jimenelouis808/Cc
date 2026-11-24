"""Figures of electronica.py's results: π bands near E_F and DOS (English)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID, LINE = "#1c2126", "#5d6771", "#dcdfe2", "#0f6b74"
TITLES = {"knee-super-graphene": "super-graphene (crystalline)",
          "knee-super-square": "super-square (crystalline)",
          "schwarzite-P": "Schwarz P", "schwarzite-D": "Schwarz D",
          "gyroid": "gyroid", "meshed-super-graphene": "super-graphene (meshed)"}
LABEL = {"G": "Γ"}


def main() -> None:
    r = json.loads((HERE / "resultados.json").read_text())
    (HERE / "figuras").mkdir(exist_ok=True)
    names = [n for n in TITLES if n in r]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharey=True)
    for ax, name in zip(axes.ravel(), names):
        d = r[name]["bands"]
        x, e = np.array(d["k"]), np.array(d["e_minus_ef"])
        if r[name]["gap"] > 0.05:
            # The π solver puts E_F inside the gap but not at its centre.
            e = e + r[name]["fermi"] - 0.5 * (r[name]["homo"] + r[name]["lumo"])
        for band in e.T:
            ax.plot(x, band, color=LINE, lw=0.9)
        for s in d["special_k"]:
            ax.axvline(s, color=GRID, lw=0.8, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(d["special_k"], [LABEL.get(lb, lb) for lb in d["labels"]])
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-1.0, 1.0)
        gap = r[name]["gap"]
        ax.set_title(f"{TITLES[name]}\n{r[name]['atoms']} C · "
                     + (f"gap {gap:.2f} eV" if gap > 0.05 else "gapless"),
                     color=INK, fontsize=10.5, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("E − E$_F$ (eV)", color=INK)
    fig.suptitle("π bands of the carbon superstructures near the Fermi level / mid-gap "
                 "(π Hückel, t = −2.7 eV)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "figuras" / "bandas.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    styles = ["-", "-", "--", "-", ":", "-.", "-", "--"]
    for k, name in enumerate(r):
        dd = r[name]["dos"]
        label = TITLES.get(name, name)
        ax.plot(dd["e_minus_ef"], dd["states_per_ev_per_atom"], styles[k % 8],
                lw=1.3, label=label)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("E − E$_F$ (eV)")
    ax.set_ylabel("π DOS (states / eV / atom)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("π density of states (σ = 0.03 eV)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "figuras" / "dos.png", dpi=170)
    plt.close(fig)
    compare(r)
    magnetism()


def compare(r) -> None:
    """π model against c_bandas (s + p) on the same path."""
    path = HERE / "resultados_c_bandas.json"
    if not path.exists():
        return
    c = json.loads(path.read_text())
    names = [n for n in ("schwarzite-D", "knee-super-square", "knee-super-graphene") if n in c]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.6), sharey=True,
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        for source, colour, lw, label in ((r[name], "#9aa3ab", 0.8, "π Hückel"),
                                          (c[name], LINE, 1.1, "c_bandas (s+p)")):
            d = source["bands"]
            x, e = np.array(d["k"]), np.array(d["e_minus_ef"])
            if source is r[name] and r[name]["gap"] > 0.05:
                # The π solver puts E_F at the LUMO edge of a gapped system;
                # align both models on mid-gap instead.
                e = e + r[name]["fermi"] - 0.5 * (r[name]["homo"] + r[name]["lumo"])
            for i, band in enumerate(e.T):
                ax.plot(x, band, color=colour, lw=lw, label=label if i == 0 else None)
        d = c[name]["bands"]
        for s in d["special_k"]:
            ax.axvline(s, color=GRID, lw=0.8, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(d["special_k"], [LABEL.get(lb, lb) for lb in d["labels"]])
        ax.set_xlim(d["k"][0], d["k"][-1])
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{TITLES[name]}\ngap: π {r[name]['gap']:.2f} eV · "
                     f"c_bandas {c[name]['gap']:.2f} eV", color=INK, fontsize=10.5, loc="left")
    axes[0, 0].set_ylabel("E − E$_F$ (eV)", color=INK)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=9,
               loc="upper right", ncol=2)
    fig.suptitle("π model against s+p tight binding (c_bandas), zero at mid-gap / E$_F$",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "figuras" / "bandas_c_bandas.png", dpi=170)
    plt.close(fig)


def magnetism() -> None:
    """Mean-field moment per atom against U/|t| (magnetismo_U.json)."""
    path = HERE / "magnetismo_U.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = dict.fromkeys(v["structure"] for v in data.values())
    marks = {"graphene (control)": ("o", MUTED), "knee-super-square": ("s", LINE),
             "knee-super-graphene": ("^", "#b5651d")}
    for name in names:
        pts = sorted((v["U_over_t"], v["abs_moment_sum_muB"] / len(v["moments"]))
                     for v in data.values() if v["structure"] == name)
        u, m = zip(*pts)
        mk, col = marks.get(name, ("o", INK))
        ax.plot(u, m, marker=mk, color=col, lw=1.3, label=TITLES.get(name, name))
    ax.axvline(2.23, color=GRID, lw=1, ls=":")
    ax.text(2.25, ax.get_ylim()[1] * 0.92, "graphene MF\nthreshold 2.23|t|",
            color=MUTED, fontsize=8)
    ax.set_xlabel("U / |t|  (t = −2.7 eV)")
    ax.set_ylabel("mean |m| per atom (μ$_B$)")
    ax.set_title("Mean-field Hubbard: staggered moment against U", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "figuras" / "magnetismo_U.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
