"""Las figuras de los tres PEIS, desde lo que escribió exportar.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, NullFormatter

OUT = Path("/home/user/Cc/packages/ramancarbon/analisis_PEIS")
COLOURS = {"800 °C": "#2a78d6", "900 °C": "#1baf7a", "1020 °C": "#d4213d"}
MODEL, GREY, INK = "#1a1a19", "#8a8a84", "#4a4a46"

S = json.loads((OUT / "resumen.json").read_text(encoding="utf-8"))


def slug(label):
    return label.replace(" ", "").replace("°", "")


def spectrum(label):
    return np.genfromtxt(OUT / f"espectro_{slug(label)}.csv", delimiter=",",
                         names=True)


def tidy(axis):
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8)
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())


def tag(axis, letter, where=(0.03, 0.94)):
    axis.text(*where, letter, transform=axis.transAxes, ha="left", va="top",
              fontsize=9, fontweight="bold")


def panel_nyquist(axis):
    """The whole spectrum: four decades of |Z|, so log-log."""
    for sample in S["muestras"]:
        label = sample["muestra"]
        table = spectrum(label)
        axis.plot(table["z_real_ohm"], -table["z_imag_ohm"], "o",
                  color=COLOURS[label], markersize=3.2,
                  markerfacecolor="none", markeredgewidth=0.9, zorder=3,
                  label=label)
        axis.plot(table["z_real_ajuste_ohm"], -table["z_imag_ajuste_ohm"],
                  "-", color=COLOURS[label], linewidth=1.2, zorder=2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Z′ / Ω")
    axis.set_ylabel("−Z″ / Ω")
    # The 45° line a distributed RC follows, as a reference and not as a
    # fit: on log-log it is a slope of one.
    axis.plot([8.0, 4e3], [8.0, 4e3], linestyle=(0, (4, 2)), color=GREY,
              linewidth=0.9, zorder=1)
    axis.text(1.3e3, 2.0e3, "45°", color=GREY, fontsize=7.5)
    axis.legend(frameon=False, fontsize=7.5, loc="upper left",
                handlelength=1.2, borderpad=0.2)
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8)
    tag(axis, "(a)", (0.72, 0.22))


def panel_zoom(axis):
    """The high-frequency corner, where the pore and the arc are.

    Linear and square, which is the only way a Nyquist plot can be read:
    the shape IS the result, and an aspect ratio that is not one turns a
    45° line into whatever angle the axes happen to give.
    """
    # Bounded by the series resistances, not by the data: the corner is
    # where the pore and the arc live, and one decade of frequency more
    # puts the capacitive tail on the axis and flattens everything else.
    limit = 6.0 * max(s["forma"]["r_serie_ohm"] for s in S["muestras"])
    for sample in S["muestras"]:
        label = sample["muestra"]
        table = spectrum(label)
        keep = -table["z_imag_ohm"] <= limit
        axis.plot(table["z_real_ohm"][keep], -table["z_imag_ohm"][keep], "o",
                  color=COLOURS[label], markersize=3.6,
                  markerfacecolor="none", markeredgewidth=0.9, zorder=3)
        axis.plot(table["z_real_ajuste_ohm"][keep],
                  -table["z_imag_ajuste_ohm"][keep], "-",
                  color=COLOURS[label], linewidth=1.2, zorder=2,
                  label=f"{label}: {sample['comun']}")
        # R_s, measured from the data and not from the fit.
        axis.plot([sample["forma"]["r_serie_ohm"]], [0.0], "v",
                  color=COLOURS[label], markersize=5, zorder=4)
    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Z′ / Ω")
    axis.set_ylabel("−Z″ / Ω")
    axis.plot([0.0, limit], [0.0, limit], linestyle=(0, (4, 2)), color=GREY,
              linewidth=0.9, zorder=1)
    axis.text(0.80 * limit, 0.84 * limit, "pendiente 45°", color=GREY,
              fontsize=7, rotation=45, ha="center", va="center")
    axis.legend(frameon=False, fontsize=6.8, loc="upper left",
                handlelength=1.3, borderpad=0.2)
    axis.text(0.97, 0.05, "▼ R$_s$ medida de los datos",
              transform=axis.transAxes, ha="right", fontsize=7, color=INK)
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8)
    tag(axis, "(b)", (0.03, 0.72))


def panel_bode(axis):
    """|Z| and the phase, with the knee marked where it was measured."""
    phase = axis.twinx()
    for sample in S["muestras"]:
        label = sample["muestra"]
        table = spectrum(label)
        axis.plot(table["f_hz"], table["modulo_ohm"], "-",
                  color=COLOURS[label], linewidth=1.3, zorder=3)
        phase.plot(table["f_hz"], table["fase_grados"], linestyle=(0, (3, 2)),
                   color=COLOURS[label], linewidth=1.1, zorder=2)
        knee = sample["forma"]["codo_hz"]
        if knee:
            phase.plot([knee], [-45.0], "o", color=COLOURS[label],
                       markersize=5, markeredgecolor="white",
                       markeredgewidth=0.8, zorder=5)
    phase.axhline(-45.0, color=GREY, linewidth=0.7, linestyle=(0, (1, 2)),
                  zorder=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("f / Hz")
    axis.set_ylabel("|Z| / Ω  (continua)")
    phase.set_ylabel("fase / grados  (discontinua)", fontsize=8.5)
    phase.set_ylim(-95.0, 5.0)
    phase.set_yticks([-90, -68, -45, -22, 0])
    phase.tick_params(which="both", direction="in", labelsize=8)
    axis.tick_params(which="both", direction="in", top=True, labelsize=8)
    axis.yaxis.set_minor_formatter(NullFormatter())
    phase.text(0.97, 0.56, "● codo: fase = −45°", transform=phase.transAxes,
               ha="right", va="bottom", fontsize=7, color=INK)
    tag(axis, "(c)", (0.03, 0.96))


def panel_capacitance(axis):
    """C′(f) and C″(f): the data, with no circuit anywhere near them."""
    second = axis.twinx()
    for sample in S["muestras"]:
        label = sample["muestra"]
        table = spectrum(label)
        order = np.argsort(table["f_hz"])
        f = table["f_hz"][order]
        axis.plot(f, 1e6 * table["c_prima_f"][order], "-",
                  color=COLOURS[label], linewidth=1.4, zorder=3, label=label)
        second.plot(f, 1e6 * table["c_segunda_f"][order],
                    linestyle=(0, (3, 2)), color=COLOURS[label],
                    linewidth=1.0, zorder=2)
        f0 = sample["capacitancia"]["f0_hz"]
        if f0:
            peak = float(np.interp(f0, f, 1e6 * table["c_segunda_f"][order]))
            second.plot([f0], [peak], "s", color=COLOURS[label], markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    axis.set_xscale("log")
    axis.set_xlabel("f / Hz")
    axis.set_ylabel("C′ / µF  (continua)")
    second.set_ylabel("C″ / µF  (discontinua)", fontsize=8.5)
    second.tick_params(which="both", direction="in", labelsize=8)
    axis.tick_params(which="both", direction="in", top=True, labelsize=8)
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.legend(frameon=False, fontsize=7.5, loc="upper right",
                handlelength=1.2, borderpad=0.2)
    second.text(0.97, 0.62, "■ máximo de C″ → τ₀",
                transform=second.transAxes, ha="right", fontsize=7, color=INK,
                va="top")
    tag(axis, "(d)", (0.03, 0.96))


def panel_drt(axis):
    """γ(τ) over the measured range of time constants only."""
    floor = 0.0
    for sample in S["muestras"]:
        label = sample["muestra"]
        table = np.genfromtxt(OUT / f"drt_{slug(label)}.csv", delimiter=",",
                              names=True)
        inside = table["dentro_del_rango"] > 0
        gamma = table["gamma_ohm"][inside]
        peak = float(gamma.max()) or 1.0
        axis.plot(table["tau_s"][inside], gamma / peak, "-",
                  color=COLOURS[label], linewidth=1.3, zorder=3,
                  label=f"{label}  (γ$_{{máx}}$ = {peak:.0f} Ω)")
        for peak in sample["drt"]["picos"]:
            axis.plot([peak["tau_s"]], [floor], "^", color=COLOURS[label],
                      markersize=5, clip_on=False, zorder=5)
    axis.set_xscale("log")
    # NORMALISED, and linear. The three γ differ by an order of
    # magnitude, so on a shared linear axis the tallest flattens the
    # other two; and on a log axis the sparseness of the non-negative
    # solution — most bins exactly zero — draws as a comb of spikes that
    # says nothing. What the panel is for is WHERE the weight sits.
    axis.set_ylim(0.0, 1.18)
    axis.set_xlabel("τ / s")
    axis.set_ylabel("γ(τ) / γ$_{máx}$")
    axis.legend(frameon=False, fontsize=7.5, loc="upper left",
                handlelength=1.2, borderpad=0.2)
    axis.text(0.03, 0.56,
              "▲ picos resueltos. Son una distribución\ntroceada: un poro "
              "no tiene τ propia",
              transform=axis.transAxes, ha="left", va="top", fontsize=7,
              color=INK)
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8)
    tag(axis, "(e)", (0.03, 0.62))


def panel_residual(axis):
    """Where the common circuit fails, frequency by frequency."""
    for sample in S["muestras"]:
        label = sample["muestra"]
        table = spectrum(label)
        axis.plot(table["f_hz"], table["residuo_pct"], "o-",
                  color=COLOURS[label], markersize=2.8, linewidth=1.0,
                  zorder=3,
                  label=f"{label}: {sample['comun']} ({table['residuo_pct'].mean():.2f} %)")
    axis.axhline(0.0, color=GREY, linewidth=0.6, zorder=1)
    axis.set_xscale("log")
    axis.set_xlabel("f / Hz")
    axis.set_ylabel("|Z − Z$_{aj}$| / |Z|   /  %")
    axis.legend(frameon=False, fontsize=6.8, loc="upper right",
                handlelength=1.3, borderpad=0.2)
    low, high = axis.get_ylim()
    axis.set_ylim(low, high + 0.35 * (high - low))
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     labelsize=8)
    tag(axis, "(f)", (0.03, 0.96))


def main():
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 9.4))
    panel_nyquist(axes[0][0])
    panel_zoom(axes[0][1])
    panel_bode(axes[1][0])
    panel_capacitance(axes[1][1])
    panel_drt(axes[2][0])
    panel_residual(axes[2][1])
    figure.tight_layout(h_pad=1.8, w_pad=2.4)
    for suffix in ("png", "pdf"):
        figure.savefig(OUT / f"peis_serie.{suffix}", dpi=220 if suffix == "png"
                       else None, bbox_inches="tight")
    plt.close(figure)
    print("escritas", OUT / "peis_serie.png")


if __name__ == "__main__":
    main()
