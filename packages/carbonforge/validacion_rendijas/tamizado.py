"""H2/D2 quantum sieving at zero coverage: every structure of the study.

The low-pressure equilibrium selectivity is the ratio of Henry
constants, S(D2/H2) = K_H(D2) / K_H(H2). Both isotopes have the same
Lennard-Jones potential; D2 differs only by its mass, through the
Feynman-Hibbs correction (carbonforge.adsorption.potentials.D2_SELF).
H2 and D2 are sampled on the SAME insertion points (same seed), so the
ratio's noise is far below each constant's.

Second-order Feynman-Hibbs is a correction, not path integrals: where
it is large (narrow pores, low T) the selectivity it gives is an upper
bound; the q_st difference between the isotopes shows how large the
correction is.

    python tamizado.py [samples]     -> tamizado.txt (rows skipped if done)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.io import read

from carbonforge.adsorption.bundle import levesque_bundle
from carbonforge.adsorption.henry import henry_constant
from carbonforge.adsorption.potentials import D2_SELF

from llenado import slit
from llenado_super import embed

HERE = Path(__file__).resolve().parent
OUT = HERE / "tamizado.txt"
TEMPERATURES = (77.0, 40.0)
SEEDS = (0, 1)


def structures():
    for h in (5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 7.0, 8.0, 10.0):
        yield f"slit H={h:g}", lambda h=h: slit(h)
    for gap in (3.4, 4.0, 6.0):
        yield f"SWCNT bundle gap={gap:g}", lambda gap=gap: levesque_bundle(gap=gap)
    from galeria import STRUCTURES
    for name, (src, _style) in STRUCTURES.items():
        if src == "bundle":
            continue
        yield name, lambda src=src: embed(read(str(HERE / src)))[0]


def main(samples: int) -> None:
    # Exact keys, not substrings: "super-graphene | 77 K" is inside
    # "knee-super-graphene | 77 K", and a substring test skipped it.
    done = ({line.split(" | K_H")[0].strip() for line in OUT.read_text().splitlines()}
            if OUT.exists() else set())
    for name, build in structures():
        for t in TEMPERATURES:
            key = f"{name} | {t:g} K"
            if key in done:
                continue
            atoms = build()
            kh, kd, qh, qd = [], [], [], []
            for seed in SEEDS:
                h = henry_constant(atoms, t, samples=samples, seed=seed)
                d = henry_constant(atoms, t, samples=samples, seed=seed,
                                   adsorbate=D2_SELF)
                kh.append(h.henry_coefficient)
                kd.append(d.henry_coefficient)
                qh.append(h.isosteric_heat)
                qd.append(d.isosteric_heat)
            ratios = np.array(kd) / np.array(kh)
            line = (f"{key:40s} | K_H(H2) {np.mean(kh):11.4g} | K_H(D2) {np.mean(kd):11.4g}"
                    f" | S {ratios.mean():7.3f} ({ratios.min():.3f}-{ratios.max():.3f})"
                    f" | q_st H2 {np.mean(qh):6.2f} D2 {np.mean(qd):6.2f} kJ/mol")
            print(line, flush=True)
            with OUT.open("a") as handle:
                handle.write(line + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500_000)
