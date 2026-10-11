"""GPAW (PBE) force constants of graphene, for the double-resonance Raman.

Run (needs GPAW; tens of minutes on four cores)::

    python -m tbkit.recipes.graphene_phonons OUT.json --workers 4

A 6x6 supercell at the experimental bond (1.42 Å), LCAO dzp, PBE, Fermi-Dirac
0.1 eV, k 3x3; ±0.01 Å displacements of the two atoms of one cell. The result
(:class:`tbkit.graphene.GraphenePhonons` as JSON) goes to
``parameters/references/gpaw_graphene_phonons.json``: DFT phonons for
:mod:`tbkit.graphene` without GPAW installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..settings import Param, add_arguments, apply

SETTINGS = {"mode": "lcao", "basis": "dzp", "xc": "PBE", "h": 0.2, "kpts": (3, 3, 1),
            "smearing_ev": 0.1, "a_cc": 1.42, "supercell": 6, "delta": 0.01, "vacuum": 15.0}

PARAMS = [
    Param("SETTINGS", "GPAW y fonones: modo, base, funcional, rejilla h (Å), malla k, "
          "ensanchamiento (eV), enlace a_cc (Å), supercelda N×N, desplazamiento δ (Å); "
          "vacuum solo se registra (la celda usa 15 Å)",
          why="los valores con que se hizo la referencia incluida "
              "(parameters/references/gpaw_graphene_phonons.json); a_cc experimental, no el "
              "de la red PBE",
          group="cálculo"),
]


class _Factory:
    """Picklable GPAW factory."""

    def __call__(self):
        from gpaw import GPAW, FermiDirac

        return GPAW(mode=SETTINGS["mode"], basis=SETTINGS["basis"], xc=SETTINGS["xc"],
                    h=SETTINGS["h"], kpts=SETTINGS["kpts"], symmetry="off",
                    occupations=FermiDirac(SETTINGS["smearing_ev"]), txt=None)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path,
                        help="archivo JSON de constantes de fuerza que se escribe")
    parser.add_argument("--workers", type=int, default=4,
                        help="procesos GPAW en paralelo (cada uno, un cálculo; vigila la memoria)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=args.out.parent)
    import gpaw

    from tbkit.graphene import GraphenePhonons

    phonons = GraphenePhonons.from_calculator(
        _Factory(), SETTINGS["a_cc"], n=SETTINGS["supercell"], delta=SETTINGS["delta"],
        source=f"GPAW {gpaw.__version__} {SETTINGS['xc']} {SETTINGS['mode'].upper()} "
               f"{SETTINGS['basis']}, supercelda {SETTINGS['supercell']}x{SETTINGS['supercell']}, "
               f"a_cc = {SETTINGS['a_cc']} Å",
        workers=args.workers)
    data = phonons.to_dict()
    data["settings"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in SETTINGS.items()}
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, q in phonons.special_points().items():
        print(name, [round(float(f), 1) for f in phonons.modes(q)[0]])


if __name__ == "__main__":
    main()
