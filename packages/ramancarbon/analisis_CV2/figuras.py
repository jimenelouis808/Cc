"""The figure for the 5–100 mV/s series, from what exportar.py wrote."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, NullFormatter, NullLocator

OUT = "/home/user/Cc/packages/ramancarbon/analisis_CV2"
RATE_COLOURS = {5: "#2a78d6", 10: "#1baf7a", 20: "#eda100", 50: "#eb6834",
                100: "#d4213d", 200: "#6250d6"}
MEASURED, MODEL, SECOND, GREY = "#1a1a19", "#d4213d", "#6250d6", "#8a8a84"

with open(f"{OUT}/resumen.json") as handle:
    S = json.load(handle)
COND = S["condiciones"]
TO_RHE = COND["ventana_v_vs_rhe"][0] - COND["ventana_v_vs_referencia"][0]
OER = 1.23 - TO_RHE


def rhe_axis(axis):
    axis.axvline(OER, color=MODEL, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
    top = axis.secondary_xaxis(
        "top", functions=(lambda e: e + TO_RHE, lambda e: e - TO_RHE))
    top.set_xlabel("E / V vs RHE", fontsize=8.5)
    top.tick_params(which="both", direction="in", labelsize=8)


def panel_curves(axis):
    for row in S["por_velocidad"]:
        rate = int(round(row["mv_s"]))
        table = np.loadtxt(f"{OUT}/ciclo_{rate}mVs.csv", delimiter=",", skiprows=1)
        axis.plot(table[:, 1], table[:, 3], color=RATE_COLOURS[rate],
                  linewidth=1.4, label=f"{rate} mV/s", zorder=3)
    axis.axhline(0.0, color=GREY, linewidth=0.6, zorder=1)
    axis.set_xlabel("E / V vs Ag/AgCl")
    axis.set_ylabel("i / A g$^{-1}$")
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.40 * (high - low))
    axis.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3,
                columnspacing=1.0, handlelength=1.5)
    rhe_axis(axis)
    axis.text(0.98, 0.04, "(a)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def panel_capacitance(axis):
    rows = S["por_velocidad"]
    rates = np.array([r["mv_s"] for r in rows])
    axis.plot(rates, [r["c_lazo_f_g"] for r in rows], "o-", color=MEASURED,
              markersize=5, linewidth=1.3, label="lazo ÷ 2νΔV  (con la fuga)")
    axis.plot(rates, [r["c_red_f_g"] for r in rows], "s-", color=SECOND,
              markersize=5, linewidth=1.3, label="carga de reducción ÷ ΔV")
    twin = axis.twinx()
    twin.plot(rates, [r["eficiencia"] for r in rows], "^--", color=MODEL,
              markersize=5, linewidth=1.1)
    twin.set_ylabel("Q$_{red}$/Q$_{ox}$", color=MODEL, fontsize=8.5)
    twin.tick_params(axis="y", colors=MODEL, labelsize=8)
    twin.set_ylim(0.35, 1.02)
    axis.set_xscale("log")
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(rates)
    axis.set_xticklabels([f"{r:.0f}" for r in rates])
    axis.set_xlabel(r"$\nu$ / mV s$^{-1}$")
    axis.set_ylabel("C / F g$^{-1}$")
    axis.set_ylim(2.8, 13.2)
    axis.legend(frameon=False, fontsize=7.5, loc="upper right")
    axis.annotate("Q$_{red}$/Q$_{ox}$", xy=(rates[1], rows[1]["eficiencia"]),
                  xycoords=("data", "axes fraction"), xytext=(8, -28),
                  textcoords="offset points", fontsize=7.5, color=MODEL)
    axis.text(0.98, 0.04, "(b)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def panel_charge(axis):
    rows = S["por_velocidad"]
    duration = np.array([r["duracion_ciclo_s"] for r in rows])
    oxidation = 1e6 * np.array([r["q_ox_c"] for r in rows])
    reduction = 1e6 * np.array([r["q_red_c"] for r in rows])
    leak = S["fuga"]
    grid = np.linspace(0.0, 1.1 * duration.max(), 50)
    axis.plot(grid, 1e6 * (leak["ordenada_c"] + leak["i_par_a"] * grid),
              color=MODEL, linewidth=1.3, zorder=2,
              label=(r"$\langle i_{par}\rangle t$, "
                     f"{1e6 * leak['i_par_a']:.2f} µA, R² = {leak['r2']:.3f}"))
    axis.axhline(reduction.mean(), color=GREY, linewidth=1.0,
                 linestyle=":", zorder=1,
                 label=r"$\langle Q_{red}\rangle$")
    axis.plot(duration, oxidation, "o", color=MEASURED, markersize=5,
              zorder=4, label=r"$Q_{ox}$")
    axis.plot(duration, reduction, "s", color=SECOND, markersize=5,
              zorder=4, label=r"$Q_{red}$")
    axis.plot(duration, oxidation - reduction, "^", color=MODEL,
              markersize=5.5, zorder=4, label=r"$\Delta Q$")
    # Only the ends and the middle: six labels on points that bunch up
    # at short durations is a smudge, and the axis already says what the
    # duration is.
    for row, x, y in zip(rows, duration, oxidation - reduction):
        if row["mv_s"] not in (5, 20, 200):
            continue
        offset = (8, -2) if x < 60 else (-7, 8)
        axis.annotate(f"{row['mv_s']:.0f} mV/s", (x, y),
                      textcoords="offset points", xytext=offset, fontsize=7,
                      color=GREY, ha="left" if x < 60 else "right")
    axis.set_xlabel("duración del ciclo / s")
    axis.set_ylabel("carga / µC")
    axis.set_xlim(0.0, 1.14 * duration.max())
    axis.set_ylim(0.0, 1.30 * oxidation.max())
    axis.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=2,
                columnspacing=0.9, handlelength=1.4)
    axis.text(0.98, 0.04, "(c)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def panel_three_terms(axis):
    rows = S["modelo_de_tres_terminos"]
    potential = np.array([r["e_v"] for r in rows])
    axis.errorbar(potential, [1e6 * r["k_cap_f"] for r in rows],
                  yerr=[1e6 * r["k_cap_error"] for r in rows],
                  fmt="o-", color=MEASURED, markersize=4.5, linewidth=1.2,
                  capsize=2.5, label=r"$k_{cap}$ / µF")
    axis.errorbar(potential, [1e6 * r["k_diff"] for r in rows],
                  yerr=[1e6 * r["k_diff_error"] for r in rows],
                  fmt="s-", color=SECOND, markersize=4.5, linewidth=1.2,
                  capsize=2.5, label=r"$k_{diff}$ / µA (V/s)$^{-1/2}$")
    twin = axis.twinx()
    twin.errorbar(potential, [1e6 * r["i_par_a"] for r in rows],
                  yerr=[1e6 * r["i_par_error"] for r in rows],
                  fmt="^--", color=MODEL, markersize=4.5, linewidth=1.1,
                  capsize=2.5)
    twin.axhline(0.0, color=MODEL, linewidth=0.6, linestyle=":")
    twin.set_ylabel(r"$i_{par}$ / µA", color=MODEL, fontsize=8.5)
    twin.tick_params(axis="y", colors=MODEL, labelsize=8)
    twin.set_ylim(-2.2, 2.2)
    axis.set_xlabel("E / V vs Ag/AgCl")
    axis.set_ylabel("k$_{cap}$, k$_{diff}$")
    axis.set_ylim(0.0, 66.0)
    axis.legend(frameon=False, fontsize=7.5, loc="upper left")
    rhe_axis(axis)
    axis.text(0.98, 0.04, "(d)", transform=axis.transAxes, ha="right",
              fontsize=9, fontweight="bold")


def main():
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    panel_curves(axes[0, 0])
    panel_capacitance(axes[0, 1])
    panel_charge(axes[1, 0])
    panel_three_terms(axes[1, 1])
    for axis in axes.flat:
        if axis.get_xscale() == "linear":
            axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        axis.tick_params(which="both", direction="in", labelsize=8)
        axis.set_xlabel(axis.get_xlabel(), fontsize=8.5)
        axis.set_ylabel(axis.get_ylabel(), fontsize=8.5)
    figure.tight_layout(pad=1.0)
    figure.savefig(f"{OUT}/cv_serie.png", dpi=600)
    figure.savefig(f"{OUT}/cv_serie.pdf")
    print("escrita cv_serie.{png,pdf}")


if __name__ == "__main__":
    main()
