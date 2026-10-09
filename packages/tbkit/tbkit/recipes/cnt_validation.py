"""Pristine carbon nanotubes under a tbkit model: RBM(d) and the G modes.

Run::

    python -m tbkit.recipes.cnt_validation OUT.json [--model xu_carbon] [--tubes 10,0 6,6 ...]

For each tube (ASE's ``nanotube`` builder, one translational cell, 1D
periodic, 10 Å of vacuum around it): the axial period is found by minimising
the energy (positions relaxed at each length; the calculator has no stress),
then the Γ modes are computed (:func:`tbkit.modes.vibrations`) and classified:

* only modes invariant under the tube's pure rotation C_g, g = gcd(n, m)
  (A symmetry, :func:`tbkit.modes.rotational_symmetry`), are candidates;
* **RBM**: the A mode with the largest overlap with a pure in-phase radial
  breathing (:func:`tbkit.modes.breathing_overlap`);
* **G⁺ / G⁻**: of the A modes above 1300 cm⁻¹ with optical character
  (each atom against its neighbours, :func:`tbkit.modes.optical_character`,
  > 0.8: the modes folded from graphene's Γ, not from its M point), the most
  axial and the most circumferential one.

References (each with its source, never fitted to):

* ω_RBM = 227.0/d cm⁻¹ (d in nm): the environment-free limit of Araujo et al.,
  Phys. Rev. B 77, 241403 (2008);
* ω_RBM = 248/d: isolated tubes on Si/SiO₂, Jorio et al., Phys. Rev. Lett. 86,
  1118 (2001);
* G⁺ ≈ 1591 cm⁻¹ and G⁻ = 1591 - C/d² (C = 47.7 semiconducting, 79.5
  metallic), Jorio et al., Phys. Rev. B 65, 155412 (2002). For metallic tubes
  the measured G⁻ is the LO mode softened by a Kohn anomaly, which a TB model
  without it will not reproduce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RBM_ARAUJO = 227.0
RBM_JORIO = 248.0
G_PLUS = 1591.0
G_MINUS_C = {"semiconductor": 47.7, "metal": 79.5}

DEFAULT_TUBES = ("8,0", "10,0", "12,0", "14,0", "17,0", "5,5", "6,6", "8,8", "10,10", "8,2",
                 "6,3")


def build(n: int, m: int, bond: float = 1.42, vacuum: float = 10.0):
    from ase.build import nanotube

    tube = nanotube(n, m, length=1, bond=bond, vacuum=vacuum)
    tube.pbc = [False, False, True]
    return tube


def diameter_nm(n: int, m: int, bond: float = 1.42) -> float:
    return np.sqrt(3) * bond * np.sqrt(n * n + n * m + m * m) / np.pi / 10.0


def kind(n: int, m: int) -> str:
    return "metal" if (n - m) % 3 == 0 else "semiconductor"


def relax_tube(tube, model, kmesh: int, kT: float, fmax: float = 0.005):
    """Axial period by an energy scan (quadratic fit), positions by BFGS."""
    from ase.optimize import BFGS

    from ..calculator import TBCalculator

    def relaxed(scale):
        trial = tube.copy()
        trial.set_cell(tube.cell.array * np.array([1.0, 1.0, scale])[:, None], scale_atoms=True)
        trial.calc = TBCalculator(model, kpts=kmesh, kT=kT)
        BFGS(trial, logfile=None).run(fmax=fmax, steps=500)
        return trial, trial.get_potential_energy()

    scales = np.array([0.99, 1.0, 1.01])
    energies = [relaxed(s)[1] for s in scales]
    a, b, _ = np.polyfit(scales, energies, 2)
    best = float(np.clip(-b / (2 * a), 0.97, 1.03))
    tube, _ = relaxed(best)
    return tube, best


def classify(vib, order: int):
    from ..modes import (
        breathing_overlap,
        cylindrical_character,
        optical_character,
        rotational_symmetry,
    )

    character = cylindrical_character(vib)
    symmetric = rotational_symmetry(vib, order) > 0.9
    optical = optical_character(vib) > 0.8
    f = vib.frequencies

    def best(mask, score):
        candidates = np.flatnonzero(mask & symmetric)
        return int(candidates[np.argmax(score[candidates])]) if len(candidates) else None

    rbm = best(f > 50, breathing_overlap(vib))
    g_axial = best((f > 1300) & optical, character["axial"])
    g_circ = best((f > 1300) & optical, character["tangential"])
    return rbm, g_axial, g_circ, character


def run(tubes, model_name: str = "xu_carbon", kmesh: int = 16, kT: float = 0.03,
        fmax: float = 0.005) -> list[dict]:
    from ..modes import vibrations
    from ..params import load_parameters
    from ..progress import Progress

    model = load_parameters(model_name)
    rows = []
    bar = Progress(len(tubes), "nanotubos (relajar + modos en Γ)")
    for spec in tubes:
        n, m = (int(x) for x in spec.split(","))
        tube, scale = relax_tube(build(n, m), model, kmesh, kT, fmax)
        vib = vibrations(tube, model, kmesh=kmesh, kT=kT)
        rbm, g_axial, g_circ, character = classify(vib, int(np.gcd(n, m)))
        radius = np.mean(np.linalg.norm(tube.positions[:, :2] - tube.positions[:, :2].mean(0),
                                        axis=1))
        d = 2 * radius / 10.0
        f = vib.frequencies
        g = sorted([float(f[g_axial]), float(f[g_circ])]) if None not in (g_axial, g_circ) \
            else [float("nan")] * 2
        row = {"tube": f"({n},{m})", "kind": kind(n, m), "atoms": len(tube),
               "diameter_nm": d, "axial_scale": scale,
               "rbm": float(f[rbm]), "rbm_breathing_share": float(character["radial"][rbm]),
               "g_axial": float(f[g_axial]), "g_circumferential": float(f[g_circ]),
               "g_plus": g[1], "g_minus": g[0], "g_split": g[1] - g[0],
               "g_split_reference": G_MINUS_C[kind(n, m)] / d ** 2,
               "g_minus_is": "axial" if f[g_axial] < f[g_circ] else "circunferencial",
               "rbm_araujo": RBM_ARAUJO / d, "rbm_jorio": RBM_JORIO / d,
               "g_minus_reference": G_PLUS - G_MINUS_C[kind(n, m)] / d ** 2,
               "warnings": vib.warnings}
        rows.append(row)
        print(f"{row['tube']:8s} {row['kind'][:4]} d = {d:.3f} nm  RBM {row['rbm']:.0f} "
              f"(227/d {row['rbm_araujo']:.0f}, 248/d {row['rbm_jorio']:.0f})  "
              f"G+ {row['g_plus']:.0f}  G- {row['g_minus']:.0f} ({row['g_minus_is']}) "
              f"ΔG {row['g_split']:.0f} (ref {row['g_split_reference']:.0f})", flush=True)
        bar.step(note=row["tube"])
    bar.close()
    return rows


def table(rows: list[dict]) -> str:
    lines = ["| tubo | tipo | d (nm) | RBM TB | 227/d | error | 248/d | error | G⁺ | G⁻ (dir.) | "
             "ΔG TB | ΔG ref |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e1 = 100 * (r["rbm"] - r["rbm_araujo"]) / r["rbm_araujo"]
        e2 = 100 * (r["rbm"] - r["rbm_jorio"]) / r["rbm_jorio"]
        lines.append(f"| {r['tube']} | {'metálico' if r['kind'] == 'metal' else 'semic.'} | "
                     f"{r['diameter_nm']:.3f} | {r['rbm']:.0f} | {r['rbm_araujo']:.0f} | "
                     f"{e1:+.1f} % | {r['rbm_jorio']:.0f} | {e2:+.1f} % | {r['g_plus']:.0f} | "
                     f"{r['g_minus']:.0f} ({r['g_minus_is'][:4]}.) | {r['g_split']:.0f} | "
                     f"{r['g_split_reference']:.0f} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path,
                        help="archivo JSON de resultados (la tabla se imprime)")
    parser.add_argument("--model", default="xu_carbon",
                        help="conjunto TB (nombre o archivo)")
    parser.add_argument("--tubes", nargs="*", default=list(DEFAULT_TUBES),
                        help="quiralidades n,m separadas por espacios (p. ej. 8,0 5,5)")
    parser.add_argument("--kmesh", type=int, default=16,
                        help="puntos k a lo largo del tubo (metálicos necesitan más)")
    parser.add_argument("--kT", type=float, default=0.03,
                        help="temperatura electrónica (eV): los tubos metálicos la necesitan")
    parser.add_argument("--fmax", type=float, default=0.005,
                        help="fuerza máxima (eV/Å) de la relajación; estricta porque siguen "
                             "las frecuencias")
    args = parser.parse_args(argv)
    rows = run(args.tubes, args.model, args.kmesh, args.kT, args.fmax)
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(table(rows))


if __name__ == "__main__":
    main()
