"""Core-valence Coulomb constants J of each element and core level, from GPAW's atom.

Run from packages/tbkit::

    python -m tbkit.recipes.core_constants     # writes tbkit/parameters/core_holes.json

The core hole of :mod:`tbkit.xps` (method ``"hole"``) is a +1 charge on the ionised atom
that the valence electrons feel: on that atom with the constant J, on the others through
the same Klopman-Ohno γ as the charge SCC. J is, like the Hubbard U of the sets, a
derivative of the free atom's eigenvalues, never a fitted number:

    J = ∂ε_valence/∂n_core

(how much the valence p level rises per core electron: the Coulomb repulsion between a core
and a valence electron of the same atom, positive; taking one out lowers the level by J), by central
differences of ±``DELTA`` electrons on GPAW's all-electron radial solver (PBE, spin
paired). Valence level: p for every element listed (2p of B-O, 3p of P and S, 4p of Se).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from ..settings import Param, add_arguments, apply

OUT = Path("tbkit/parameters/core_holes.json")
#: element -> (core level name, angular momentum, index in GPAW's list of that l)
LEVELS = {"B": ("1s", 0, 0), "C": ("1s", 0, 0), "N": ("1s", 0, 0), "O": ("1s", 0, 0),
          "P": ("2p", 1, 0), "S": ("2p", 1, 0), "Se": ("3d", 2, 0)}
DELTA = 0.05
XC = "PBE"


def _valence_p(symbol: str, f_lsn: dict) -> float:
    from ase.units import Hartree
    from gpaw.atom.aeatom import AllElectronAtom

    atom = AllElectronAtom(symbol, xc=XC, spinpol=False, log=None)
    atom.f_lsn = f_lsn
    atom.run()
    return float(atom.get_channel(l=1).e_n[len(f_lsn[1][0]) - 1] * Hartree)


def core_coulomb(symbol: str) -> float:
    from gpaw.atom.aeatom import AllElectronAtom

    _, ang, i = LEVELS[symbol]
    base = AllElectronAtom(symbol, xc=XC, spinpol=False, log=None).f_lsn
    fewer, more = copy.deepcopy(base), copy.deepcopy(base)
    fewer[ang][0][i] -= DELTA
    more[ang][0][i] += DELTA
    return (_valence_p(symbol, more) - _valence_p(symbol, fewer)) / (2 * DELTA)


def build() -> dict:
    import gpaw

    source = (f"GPAW {gpaw.__version__} aeatom {XC} (sin espín): J = ∂ε_p/∂n_núcleo por "
              f"diferencias centrales de ±{DELTA} e (tbkit.recipes.core_constants)")
    table = {el: {"level": level, "J": {"value": round(core_coulomb(el), 4), "unit": "eV",
                                          "source": source}}
             for el, (level, _, _) in LEVELS.items()}
    return {"name": "constantes de Coulomb núcleo-valencia (huecos de núcleo, XPS)",
            "reference": "definición de este trabajo (tbkit.xps, método 'hole'); valores del "
                         "átomo libre con GPAW",
            "system": "tbkit.xps con cualquier conjunto SCC que tenga el elemento",
            "validity": "átomo libre: la interacción del hueco con la valencia propia; con "
                        "otros átomos, la γ de Klopman-Ohno del SCC",
            "units": {"J": "eV"}, "core_holes": table}


PARAMS = [Param("DELTA", "paso de ocupación de las diferencias centrales", "e", "",
                "convergencia")]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=None)
    data = build()
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print({el: v["J"]["value"] for el, v in data["core_holes"].items()})


if __name__ == "__main__":
    main()
