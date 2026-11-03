"""Spin constants W of each element, from GPAW's all-electron atom: never fitted.

Run from packages/tbkit::

    python -m tbkit.recipes.spin_constants      # writes tbkit/parameters/spin_constants.json

The spin-polarised SCC (:mod:`tbkit.spin`, spin-DFTB of Köhler et al., Chem. Phys. 309,
23 (2005)) adds ``½ Σ_A Σ_ll' W_A,ll' m_Al m_Al'`` to the energy, with ``m_Al`` the
Mulliken magnetisation (n↑ - n↓) of shell l of atom A. Its constants are the definition

    W_ll' = ½ (∂ε_l↑/∂n_l'↑ - ∂ε_l↑/∂n_l'↓)

on the free, neutral, spin-unpolarised atom (each valence shell's occupation split equally
between the spins), by central differences of ±``DELTA`` electrons, with GPAW's
all-electron radial solver (PBE): the same source and functional as the Hubbard U of the
sets (``dε/dn`` of the valence level). For carbon this gives W_ss, W_sp, W_pp =
-0.0306, -0.0251, -0.0227 Ha, the published PBE values (-0.031, -0.025, -0.023 Ha).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from ..settings import Param, add_arguments, apply

OUT = Path("tbkit/parameters/spin_constants.json")
#: Transition metals: s, p and d valence shells (the p one empty in the free atom).
D_ELEMENTS = ("Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mo", "Pd", "Ag")
ELEMENTS = ("H", "B", "C", "N", "O", "P", "S", "Se") + D_ELEMENTS
DELTA = 0.05          # electrons
XC = "PBE"
NAMES = "spd"


def _atom(symbol: str, f_lsn=None):
    from gpaw.atom.aeatom import AllElectronAtom

    atom = AllElectronAtom(symbol, xc=XC, spinpol=True, log=None)
    if f_lsn is not None:
        atom.f_lsn = f_lsn
    return atom


def _paramagnetic(symbol: str) -> dict:
    f = copy.deepcopy(_atom(symbol).f_lsn)
    for ang in f:
        for i in range(len(f[ang][0])):
            total = f[ang][0][i] + f[ang][1][i]
            f[ang][0][i] = f[ang][1][i] = total / 2
    if 2 in valence_shells(symbol):
        # the valence p of a transition metal (4p of Fe, 5p of Mo) is empty in the free atom
        # and absent from GPAW's configuration: add it, at the principal number of the s
        n_s = len(f[0][0])
        while len(f[1][0]) + 1 < n_s:
            f[1][0].append(0.0)
            f[1][1].append(0.0)
    return f


def valence_shells(symbol: str) -> tuple[int, ...]:
    """Angular momenta of the model's valence shells: s for H, s, p and d for the
    transition metals of ``D_ELEMENTS``, s and p otherwise."""
    if symbol == "H":
        return (0,)
    return (0, 1, 2) if symbol in D_ELEMENTS else (0, 1)


def _levels(symbol: str, f_lsn: dict, shells) -> dict:
    from ase.units import Hartree

    atom = _atom(symbol, f_lsn)
    atom.run()
    return {ang: float(atom.get_channel(l=ang, s=0).e_n[len(f_lsn[ang][0]) - 1] * Hartree)
            for ang in shells}


def spin_constants(symbol: str) -> np.ndarray:
    """W (eV) over the valence shells, rows/columns in the order of :func:`valence_shells`."""
    shells = valence_shells(symbol)
    f0 = _paramagnetic(symbol)
    out = np.zeros((len(shells), len(shells)))
    for j, lp in enumerate(shells):
        slope = {}
        for spin in (0, 1):
            if f0[lp][spin][-1] >= DELTA:            # central differences
                plus, minus = copy.deepcopy(f0), copy.deepcopy(f0)
                plus[lp][spin][-1] += DELTA
                minus[lp][spin][-1] -= DELTA
                e_plus, e_minus = _levels(symbol, plus, shells), _levels(symbol, minus, shells)
                slope[spin] = {ang: (e_plus[ang] - e_minus[ang]) / (2 * DELTA)
                               for ang in shells}
            else:                                    # an empty shell: one-sided, 2nd order
                one, two = copy.deepcopy(f0), copy.deepcopy(f0)
                one[lp][spin][-1] += DELTA
                two[lp][spin][-1] += 2 * DELTA
                e0, e1, e2 = (_levels(symbol, f, shells) for f in (f0, one, two))
                slope[spin] = {ang: (-3 * e0[ang] + 4 * e1[ang] - e2[ang]) / (2 * DELTA)
                               for ang in shells}
        for i, ang in enumerate(shells):
            out[i, j] = 0.5 * (slope[0][ang] - slope[1][ang])
    return 0.5 * (out + out.T)          # symmetric up to the finite-difference error


def build() -> dict:
    import gpaw

    source = (f"GPAW {gpaw.__version__} aeatom {XC}, átomo neutro sin polarizar, "
              f"W_ll' = ½(∂ε_l↑/∂n_l'↑ − ∂ε_l↑/∂n_l'↓) por diferencias centrales de "
              f"±{DELTA} e (tbkit.recipes.spin_constants)")
    table = {}
    for symbol in ELEMENTS:
        w = spin_constants(symbol)
        shells = valence_shells(symbol)
        table[symbol] = {f"{NAMES[a]}{NAMES[b]}": {"value": round(float(w[i, j]), 5),
                                                   "unit": "eV", "source": source}
                         for i, a in enumerate(shells) for j, b in enumerate(shells) if j >= i}
    return {
        "name": "constantes de espín W (spin-DFTB)",
        "reference": "definición de Köhler, Seifert y Frauenheim, Chem. Phys. 309, 23 (2005); "
                     "valores calculados aquí con el átomo de GPAW",
        "system": "cualquier conjunto s/p de tbkit con estos elementos (tbkit.spin)",
        "validity": "constantes del átomo libre: la polarización de espín de un sistema enlazado "
                    "se describe al nivel de spin-DFTB (Mulliken por capa), no de DFT de espín",
        "units": {"energy": "eV"},
        "spin_constants": table,
    }


PARAMS = [Param("DELTA", "paso de ocupación de las diferencias centrales", "e", "",
                "convergencia")]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=None)
    data = build()
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(json.dumps({k: {b: v["value"] for b, v in t.items()}
                      for k, t in data["spin_constants"].items()}, indent=1))


if __name__ == "__main__":
    main()
