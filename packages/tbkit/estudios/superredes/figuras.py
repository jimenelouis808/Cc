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
    fig.suptitle("π bands of the carbon superstructures near the Fermi level "
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


if __name__ == "__main__":
    main()
