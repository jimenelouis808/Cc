"""At what pressure does each geometry fill? The zero-coverage limit.

    uv run python validacion_rendijas/llenado.py

`curvatura.py` and `surco.py` measured the depth of a well. Depth is
not the design answer, because a deeper well also means a framework
that is already full at the pressure anybody would run it at: what a
designer picks between is **how much is adsorbed at the pressure
available**, and at low pressure that is the initial slope of the
isotherm.

That slope is the Henry coefficient, it is a Widom average over an
empty cell, and it costs seconds where an isotherm costs days
(`adsorption/henry.py`). It is the **same** reservoir and the same
Feynman-Hibbs-corrected pair the grand canonical run uses, so it is
that simulation's own low-pressure limit rather than a second model;
`tests/test_adsorption_henry.py` pins the agreement against a GCMC at
1 kPa and the departure from it by 10 kPa.

Two geometries, one knob each, nothing else differing:

* the **slit** series `construye.py` already builds, whose wall is one
  atom thick and flat -- the reference adsorbent;
* the **tube bundle** `adsorption/bundle.py` builds, whose adsorption
  sites are the grooves between neighbouring tubes. It is the groove
  adsorbent in its idealised form -- all groove, no node -- and unlike
  any superstructure in this repository it has a published isotherm
  behind it, which is what `validacion_GCMC/` reproduces.

So this is a comparison between a flat-walled and a groove-bearing
adsorbent of the same chemistry, both of them validated geometries,
with the pore width scanned on both sides. What it is NOT is a
measurement of super-graphene or a supertube: those carry nodes, and a
node is neither groove nor slit.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from carbonforge.adsorption.bundle import levesque_bundle
from carbonforge.adsorption.henry import henry_constant
from carbonforge.adsorption.porosity import analyse_porosity
from carbonforge.builders.graphene import build_graphene_supercell

#: Everything in this folder is measured at 77 K.
TEMPERATURE = 77.0

#: Insertion attempts. 200 000 keeps the Widom average's own error
#: under a per cent on a slit; the wide cells need it most, so the
#: report prints the error rather than assuming it.
SAMPLES = 200_000

#: Twice the 12 A cutoff, which every cell here must clear.
MIN_EDGE = 24.0

#: Slit wall separations. The series `construye.py` ships is 6, 7, 8,
#: 10, 14 and 20; this one is finer between 6 and 8 because the Henry
#: coefficient has a MAXIMUM in there and the coarse series straddles
#: it. `K_H` goes as `exp(q_st/RT)` and `RT` is 0.64 kJ/mol at 77 K, so
#: a well 1.8 kJ/mol deeper is a factor of 16: a scan too coarse to
#: resolve the optimum does not merely blur it, it reports a different
#: adsorbent.
SEPARATIONS = (6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.5, 8.0, 10.0, 14.0, 20.0)

#: Wall-to-wall gaps between neighbouring tubes. The reference tried
#: 3.4 (close packed) and 6.0 and found 6.0 the better adsorbent of the
#: two, which is true and is not the optimum: this scan puts it near
#: 3.9 A, where `K_H` is five times its value at 6.0. Resolving that
#: needed the same fine steps the slits did.
GAPS = (3.4, 3.6, 3.8, 3.9, 4.0, 4.2, 4.5, 5.0, 6.0, 8.0, 10.0)

#: A second seed on every point. The deep sites are a small fraction of
#: a bundle's cell, so its Widom average is noisier than a slit's and
#: its own standard error turned out to understate that: seeds agree to
#: 1 % on a slit and to 6 % on a bundle near its peak. Two seeds is the
#: cheapest honest error bar, and it is reported rather than averaged
#: away.
SEEDS = (0, 1)


def slit(separation: float) -> Atoms:
    """One AA-stacked graphene slit, exactly as `construye.py` writes it."""
    sheet = build_graphene_supercell(nx=10, ny=6, vacuum=10.0)
    sheet.set_cell(np.diag([sheet.cell[0, 0], sheet.cell[1, 1], separation]))
    sheet.center(axis=2)
    layers = max(1, int(np.ceil(MIN_EDGE / separation)))
    stack = sheet * (1, 1, layers)
    stack.set_pbc((True, True, True))
    stack.wrap()
    return stack


def measure(atoms: Atoms, label: str, knob: float) -> dict:
    """Henry coefficient, heat and the channel the probe can cross."""
    runs = [henry_constant(atoms, temperature=TEMPERATURE,
                           samples=SAMPLES, seed=seed) for seed in SEEDS]
    coefficients = [run.henry_coefficient for run in runs]
    # The spread between seeds, not the average's own error: on a
    # bundle the first understates the second.
    spread = 0.5 * (max(coefficients) - min(coefficients))
    # The channel, so a number can be read against a width rather than
    # against a name. Cheap next to the Widom sum at these sizes.
    porosity = analyse_porosity(atoms, samples=20_000, area_points=200,
                                temperature=298.0, seed=0)
    return {
        "geometria": label,
        "mando": knob,
        "atomos": len(atoms),
        "canal_a": porosity.largest_free_sphere,
        "k_h": float(np.mean(coefficients)),
        "k_h_error": float(spread),
        "q_st": runs[0].isosteric_heat,
        "avisos": sorted({w for run in runs for w in run.warnings}),
    }


def main() -> None:
    rows = []
    for separation in SEPARATIONS:
        rows.append(measure(slit(separation), "rendija", separation))
    for gap in GAPS:
        rows.append(measure(levesque_bundle(gap=gap), "haz de tubos", gap))

    print(f"H2 a {TEMPERATURE:.0f} K, limite de cobertura cero, "
          f"{SAMPLES} inserciones\n")
    print("| geometría | mando (Å) | átomos | canal (Å) | "
          "K_H (mol/kg/MPa) | q_st (kJ/mol) |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['geometria']} | {row['mando']:.1f} | "
              f"{row['atomos']} | {row['canal_a']:.2f} | "
              f"{row['k_h']:.1f} ± {row['k_h_error']:.1f} | "
              f"{row['q_st']:.2f} |")

    # The two questions the table answers, and they do not have the
    # same answer -- which is the result.
    for label in ("rendija", "haz de tubos"):
        family = [r for r in rows if r["geometria"] == label]
        slope = max(family, key=lambda r: r["k_h"])
        well = max(family, key=lambda r: r["q_st"])
        print(f"\n{label}:")
        print(f"  mayor pendiente inicial a {slope['mando']:.1f} Å: "
              f"K_H = {slope['k_h']:.0f} ± {slope['k_h_error']:.0f} "
              f"mol/kg/MPa")
        print(f"  pozo más profundo a {well['mando']:.1f} Å: "
              f"q_st = {well['q_st']:.2f} kJ/mol")

    deepest = max(rows, key=lambda r: r["q_st"])
    steepest = max(rows, key=lambda r: r["k_h"])
    print(f"\nPozo más profundo de todo: {deepest['geometria']} a "
          f"{deepest['mando']:.1f} Å ({deepest['q_st']:.2f} kJ/mol).")
    print(f"Mayor pendiente de todo:   {steepest['geometria']} a "
          f"{steepest['mando']:.1f} Å ({steepest['k_h']:.0f} mol/kg/MPa).")
    if deepest["geometria"] != steepest["geometria"]:
        print("No son la misma geometría: K_H es POR GRAMO de armazón, y "
              "una geometría puede ofrecer\nsitios mejores y menos de "
              "ellos por gramo. Los dos números responden a preguntas "
              "distintas.")

    for row in rows:
        for warning in row["avisos"]:
            print(f"  aviso ({row['geometria']} {row['mando']:.1f}): "
                  f"{warning}")


if __name__ == "__main__":
    main()
