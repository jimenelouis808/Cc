"""The two CV figures, drawn from the files exportar.py wrote.

House convention, as in analisis_FeSe/figure.py: measured data as the
thing the eye lands on, models as lines over it, and any comparison the
figure exists for kept on ONE scale.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, NullFormatter,
                               NullLocator)

OUT = "/home/user/Cc/packages/ramancarbon/analisis_CV"

# Same validated categorical set as the Rietveld figure, so the two
# figures of one paper do not use two palettes.
RATE_COLOURS = {5.0: "#2a78d6", 10.0: "#1baf7a", 20.0: "#eda100",
                50.0: "#eb6834"}
MEASURED = "#1a1a19"
MODEL = "#d4213d"
SECOND = "#6250d6"
GREY = "#8a8a84"

with open(f"{OUT}/resumen.json") as handle:
    SUMMARY = json.load(handle)
FILES = {row["archivo"]: row for row in SUMMARY["por_archivo"]}

#: E(RHE) − E(medido), from the conditions block. The window matters
#: only once it is on this scale: 1.23 V is where oxygen evolution
#: becomes thermodynamically possible, and the positive limit is past it.
TO_RHE = (SUMMARY["condiciones"]["ventana_v_vs_rhe"][0]
          - SUMMARY["condiciones"]["ventana_v_vs_referencia"][0])
OER = 1.23 - TO_RHE   # the OER equilibrium, back on the measured scale


def rhe_axis(axis, label=True):
    """A second x axis in volts against RHE, with the OER line on it."""
    axis.axvline(OER, color=MODEL, linewidth=0.9, linestyle=(0, (4, 2)),
                 zorder=1)
    top = axis.secondary_xaxis(
        "top", functions=(lambda e: e + TO_RHE, lambda e: e - TO_RHE))
    top.set_xlabel("E / V vs RHE" if label else "", fontsize=8.5)
    top.tick_params(which="both", direction="in", labelsize=8)
    return top


def cycles_table():
    rows = np.genfromtxt(f"{OUT}/ciclos.csv", delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    return rows


def panel_curves(axis):
    """(a) The reporting cycle of each file, normalised per gram."""
    for name in ("CV_02", "CV_03", "CV_04", "CV_05", "CV_01"):
        table = np.loadtxt(f"{OUT}/{name}_ciclo.csv", delimiter=",", skiprows=1)
        rate = FILES[name]["mv_s"]
        early = name == "CV_01"
        axis.plot(table[:, 1], table[:, 3],
                  color=RATE_COLOURS[round(rate, 0)],
                  linewidth=1.5 if not early else 1.1,
                  linestyle="--" if early else "-",
                  label=f"{rate:.0f} mV/s" + (" (antes)" if early else ""),
                  zorder=3 if not early else 2)
    axis.axhline(0.0, color=GREY, linewidth=0.6, zorder=1)
    axis.set_xlabel("E / V vs Ag/AgCl")
    axis.set_ylabel("i / A g$^{-1}$")
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.38 * (high - low))
    axis.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2,
                columnspacing=1.0, handlelength=1.6)
    rhe_axis(axis)
    axis.annotate(
        "equilibrio\ndel O$_2$\n1.23 V vs RHE", xy=(OER, 0.0),
        xytext=(OER - 0.30, 0.55 * axis.get_ylim()[0]), fontsize=6.8,
        color=MODEL, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=MODEL, linewidth=0.8,
                        shrinkA=2, shrinkB=2))
    axis.text(0.98, 0.04, "(a)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def panel_fade(axis):
    """(b) Every cycle of the 10-cycle file: the electrode fading."""
    data = np.loadtxt(f"{OUT}/CV_01_todos_los_ciclos.csv", delimiter=",",
                      skiprows=1)
    labels = data[:, 0].astype(int)
    highest = labels.max()
    for cycle in range(1, highest + 1):
        mask = labels == cycle
        shade = 0.82 * (cycle - 1) / max(highest - 1, 1)
        axis.plot(data[mask, 1], 1e6 * data[mask, 2],
                  color=plt.cm.plasma(0.08 + 0.78 * shade),
                  linewidth=0.8, zorder=2)
    axis.axhline(0.0, color=GREY, linewidth=0.6, zorder=1)
    axis.set_xlabel("E / V vs Ag/AgCl")
    axis.set_ylabel("i / µA")
    rows = cycles_table()
    first = rows[(rows["archivo"] == "CV_01") & (rows["ciclo"] == 1)]
    ninth = rows[(rows["archivo"] == "CV_01") & (rows["ciclo"] == 9)]
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.40 * (high - low))
    axis.text(
        0.03, 0.97,
        "10 ciclos a 50 mV/s\n"
        f"C$_{{red}}$  {float(first['c_red_f_g'][0]):.2f} → "
        f"{float(ninth['c_red_f_g'][0]):.2f} F/g   (−1 %)\n"
        f"C$_{{lazo}}$ {float(first['c_lazo_f_g'][0]):.2f} → "
        f"{float(ninth['c_lazo_f_g'][0]):.2f} F/g   (−8 %)\n"
        "lo que cae es la fuga, no lo almacenado",
        transform=axis.transAxes, va="top", ha="left", fontsize=7.2,
        linespacing=1.45)
    axis.annotate("el 1.º arranca en el\npotencial de circuito abierto",
                  xy=(-0.255, 4.0), xytext=(-0.52, -8.0), fontsize=6.8,
                  color=GREY, ha="left",
                  arrowprops=dict(arrowstyle="-|>", color=GREY,
                                  linewidth=0.8, shrinkA=2, shrinkB=2))
    axis.text(0.98, 0.04, "(b)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def panel_leak(axis):
    """(c) The charge balance: Q_red flat, Q_ox linear in the duration."""
    rows = cycles_table()
    keep = rows["completo"] & (rows["archivo"] != "CV_01")
    chosen = {}
    for row in rows[keep]:
        chosen[row["archivo"]] = row      # last complete cycle of each
    duration = np.array([r["duracion_s"] for r in chosen.values()])
    oxidation = np.array([r["q_ox_c"] for r in chosen.values()])
    reduction = np.array([r["q_red_c"] for r in chosen.values()])
    order = np.argsort(duration)
    duration, oxidation, reduction = (duration[order], oxidation[order],
                                      reduction[order])

    leak = SUMMARY["fuga"]
    net = oxidation - reduction
    grid = np.linspace(0.0, 1.08 * duration.max(), 50)
    axis.plot(grid, 1e6 * (leak["ordenada_c"] + leak["i_par_a"] * grid),
              color=MODEL, linewidth=1.3, zorder=2,
              label=(r"$\langle i_{par}\rangle t$, "
                     f"{1e6 * leak['i_par_a']:.2f} µA, "
                     f"R² = {leak['r2']:.3f}"))
    axis.axhline(1e6 * reduction.mean(), color=GREY, linewidth=1.0,
                 linestyle=":", zorder=1)
    axis.plot(duration, 1e6 * oxidation, "o", color=MEASURED, markersize=5,
              zorder=4, label=r"$Q_{ox}$")
    axis.plot(duration, 1e6 * reduction, "s", color=SECOND, markersize=5,
              zorder=4, label=r"$Q_{red}$ (lo que vuelve)")
    axis.plot(duration, 1e6 * net, "^", color=MODEL, markersize=5.5,
              zorder=4, label=r"$\Delta Q = Q_{ox} - Q_{red}$")
    for rate, x, y in zip([50, 20, 10, 5], duration, net):
        axis.annotate(f"{rate} mV/s", (x, 1e6 * y), textcoords="offset points",
                      xytext=(-7, 7), fontsize=7, color=GREY, ha="right")
    axis.set_xlabel("duración del ciclo / s")
    axis.set_ylabel("carga / µC")
    axis.set_xlim(0.0, 1.14 * duration.max())
    axis.set_ylim(0.0, 1.30e6 * oxidation.max())
    axis.legend(frameon=False, fontsize=7.0, loc="upper left", ncol=2,
                columnspacing=1.0, handlelength=1.4)
    axis.text(0.98, 0.04, "(c)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def panel_rate(axis):
    """(d) The two readings of the same series, on one scale."""
    series = [FILES[f"CV_0{n}"] for n in (2, 3, 4, 5)]
    rates = np.array([row["mv_s"] for row in series])
    loop = np.array([row["c_lazo_f_g"] for row in series])
    honest = np.array([row["c_reduccion_f_g"] for row in series])

    axis.plot(rates, loop, "o-", color=MEASURED, markersize=5, linewidth=1.3,
              label="lazo cerrado ÷ 2νΔV  (lleva la fuga)")
    axis.plot(rates, honest, "s-", color=SECOND, markersize=5, linewidth=1.3,
              label="carga de reducción ÷ ΔV  (sin la fuga)")

    early = FILES["CV_01"]
    axis.plot([early["mv_s"]], [early["c_reduccion_f_g"]], "s",
              color=SECOND, markersize=5, markerfacecolor="white",
              markeredgewidth=1.3, zorder=4)
    fade = 100.0 * (1.0 - FILES["CV_05"]["c_reduccion_f_g"]
                    / early["c_reduccion_f_g"])
    axis.annotate(
        f"{fade:.0f} % perdido entre\nla primera y la última\nmedida a 50 mV/s",
        xy=(50.0, FILES["CV_05"]["c_reduccion_f_g"]),
        xytext=(13.5, 4.1), fontsize=7, color=MODEL, ha="left",
        arrowprops=dict(arrowstyle="-|>", color=MODEL, linewidth=0.9,
                        shrinkA=2, shrinkB=3))
    axis.plot([50.0, 50.0],
              [early["c_reduccion_f_g"], FILES["CV_05"]["c_reduccion_f_g"]],
              color=MODEL, linewidth=1.0, zorder=3)
    axis.set_xscale("log")
    axis.set_xlabel("ν / mV s$^{-1}$")
    axis.set_ylabel("C / F g$^{-1}$")
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks([5, 10, 20, 50])
    axis.set_xticklabels(["5", "10", "20", "50"])
    axis.set_ylim(3.6, 12.6)
    axis.legend(frameon=False, fontsize=7.5, loc="upper right")
    axis.text(0.98, 0.04, "(d)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def figure_one():
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    panel_curves(axes[0, 0])
    panel_fade(axes[0, 1])
    panel_leak(axes[1, 0])
    panel_rate(axes[1, 1])
    for axis in axes.flat:
        if axis.get_xscale() == "linear":
            axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        axis.tick_params(which="both", direction="in", top=True, right=True,
                         labelsize=8)
        axis.set_xlabel(axis.get_xlabel(), fontsize=8.5)
        axis.set_ylabel(axis.get_ylabel(), fontsize=8.5)
    figure.tight_layout(pad=0.9)
    figure.savefig(f"{OUT}/cv_capacitancia.png", dpi=600)
    figure.savefig(f"{OUT}/cv_capacitancia.pdf")
    plt.close(figure)


def panel_laws(axis, lower):
    """b(E) against the leak the same data imply, and the three R²."""
    laws = SUMMARY["leyes_de_i_nu"]
    potential = np.array([row["e_v"] for row in laws])
    b = np.array([row["b"] for row in laws])
    leak = 1e6 * np.array([row["i_par_a"] for row in laws])

    axis.plot(potential, b, "o-", color=MEASURED, markersize=4.5,
              linewidth=1.3, label="b de $i = a\\,\\nu^{b}$")
    axis.axhline(1.0, color=GREY, linewidth=0.7, linestyle=":")
    axis.axhline(0.5, color=GREY, linewidth=0.7, linestyle=":")
    axis.text(potential[-1], 1.012, "b = 1, superficial", fontsize=7,
              color=GREY, ha="right")
    axis.text(potential[-1], 0.512, "b = 0.5, difusivo", fontsize=7,
              color=GREY, ha="right")
    twin = axis.twinx()
    twin.plot(potential, leak, "s--", color=MODEL, markersize=4.5,
              linewidth=1.2, label="$i_{par}$ de $i = k_{cap}\\nu + i_{par}$")
    twin.tick_params(axis="y", colors=MODEL, labelsize=8)
    # No legend: the two axes are already colour-coded, and a legend here
    # has nowhere to sit that is not on top of one of the two curves.
    axis.set_ylabel("b   de   $i = a\\,\\nu^{b}$", color=MEASURED)
    twin.set_ylabel("$i_{par}$ / µA   de   $i = k_{cap}\\nu + i_{par}$",
                    color=MODEL, fontsize=8.5)
    axis.set_ylim(0.38, 1.08)
    twin.set_ylim(0.0, 2.6)
    axis.text(0.015, 0.90, "(a)", transform=axis.transAxes, ha="left",
              fontsize=9, fontweight="bold")

    keys = [("potencia", "$a\\,\\nu^{b}$", MEASURED),
            ("dunn", "$k_1\\nu + k_2\\sqrt{\\nu}$", SECOND),
            ("fuga", "$k_{cap}\\nu + i_{par}$", MODEL)]
    for key, label, colour in keys:
        lower.plot(potential, [row["r2"][key] for row in laws], "o-",
                   color=colour, markersize=4, linewidth=1.2, label=label)
    lower.set_xlabel("E / V vs Ag/AgCl")
    lower.set_ylabel("R² del ajuste de i(ν)")
    lower.set_ylim(0.970, 1.0005)
    lower.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center")
    lower.text(0.98, 0.08, "(b)", transform=lower.transAxes, ha="right",
               fontsize=9, fontweight="bold")


def panel_dunn(axis):
    """The Dunn separation at 50 mV/s, with where it does not hold marked."""
    table = np.loadtxt(f"{OUT}/dunn_50mVs.csv", delimiter=",", skiprows=1)
    potential, _, _, r_squared, anodic, cathodic, exceeded = table.T
    measured = np.loadtxt(f"{OUT}/CV_05_ciclo.csv", delimiter=",", skiprows=1)

    axis.plot(measured[:, 1], 1e6 * measured[:, 2], color=MEASURED,
              linewidth=1.2, zorder=3, label="medido, 50 mV/s")
    order = np.argsort(potential)
    axis.fill_between(potential[order], 1e6 * anodic[order],
                      1e6 * cathodic[order], color=SECOND, alpha=0.22,
                      zorder=2, label="capacitivo según Dunn ($k_1\\nu$)")
    bad = exceeded > 0.5
    if bad.any():
        axis.plot(potential[bad], 1e6 * anodic[bad], "x", color=MODEL,
                  markersize=4.5, zorder=4,
                  label=f"separación no definida ({int(bad.sum())} de {bad.size})")
    axis.axhline(0.0, color=GREY, linewidth=0.6, zorder=1)
    rhe_axis(axis, label=False)
    axis.set_xlabel("E / V vs Ag/AgCl")
    axis.set_ylabel("i / µA")
    axis.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=3,
                columnspacing=1.4, handlelength=1.6)
    axis.set_ylim(-26.0, 14.0)
    axis.text(0.98, 0.04, "(c)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def figure_two():
    figure = plt.figure(figsize=(7.2, 7.0))
    grid = figure.add_gridspec(3, 1, height_ratios=[1.15, 0.85, 1.4],
                               hspace=0.55)
    upper = figure.add_subplot(grid[0])
    middle = figure.add_subplot(grid[1], sharex=upper)
    lower = figure.add_subplot(grid[2])
    panel_laws(upper, middle)
    panel_dunn(lower)
    upper.tick_params(labelbottom=False)
    for axis in (upper, middle, lower):
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        axis.tick_params(which="both", direction="in", top=True, labelsize=8)
        axis.set_xlabel(axis.get_xlabel(), fontsize=8.5)
        axis.set_ylabel(axis.get_ylabel(), fontsize=8.5)
    figure.savefig(f"{OUT}/cv_cinetica.png", dpi=600, bbox_inches="tight")
    figure.savefig(f"{OUT}/cv_cinetica.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    figure_one()
    figure_two()
    print("escritas cv_capacitancia.{png,pdf} y cv_cinetica.{png,pdf}")
