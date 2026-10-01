"""H···H contact repulsion from GPAW: two H2 molecules end to end.

Run (GPAW, a minute)::

    python -m tbkit.recipes.hh_contact references OUT.json   # recompute the curve
    python -m tbkit.recipes.hh_contact term REF.json          # the term's parameters
    python -m tbkit.recipes.hh_contact install                # add it to the xu_ch* sets

The xu_ch* sets had no H-H interaction at all, so hydrogens of different
groups could come too close: phenylboronic acid went planar (GPAW twists the
B(OH)2 23°; ortho H to hydroxyl H 1.95 Å in the model, 2.13 in GPAW). The
collinear H-H···H-H dimer isolates that contact: its energy against the
separated molecules (PBE, LCAO dzp, h 0.2 Å, 5 Å vacuum) is the repulsion the
model lacks. Only its repulsive wall is used (1.5 and 1.7 Å): beyond 1.9 Å
PBE gives a shallow minimum of -0.02 eV that is mostly basis-set
superposition, not a physical attraction worth modelling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOND = 0.75                                    # Å, H2 in the dimer
DISTANCES = (1.5, 1.7, 1.9, 2.1, 2.3, 2.6, 3.0)
#: X-H bond lengths (Å) whose hydrogens count as bonded to X, from 0.15 to 0.35 Å past them.
XH = {"C": 1.09, "N": 1.01, "O": 0.97, "B": 1.19, "S": 1.34, "P": 1.42, "Se": 1.47}
REFERENCE = Path(__file__).resolve().parents[1] / "parameters" / "references" / "gpaw_h2h2.json"


def references(out: Path) -> None:
    from ase import Atoms
    from gpaw import GPAW, __version__

    def energy(atoms):
        atoms.center(vacuum=5.0)
        atoms.calc = GPAW(mode="lcao", basis="dzp", xc="PBE", h=0.2, txt=None,
                          convergence={"density": 1e-6})
        return atoms.get_potential_energy()

    def dimer(d):
        return Atoms("H4", positions=[[0, 0, 0], [BOND, 0, 0], [BOND + d, 0, 0],
                                      [2 * BOND + d, 0, 0]])

    far = energy(dimer(6.0))
    curve = [[d, energy(dimer(d)) - far] for d in DISTANCES]
    out.write_text(json.dumps({"settings": {"code": "GPAW", "gpaw_version": __version__,
                                            "mode": "lcao", "basis": "dzp", "xc": "PBE",
                                            "h": 0.2, "vacuum": 5.0, "h2_bond": BOND},
                               "unit": {"distance": "Å", "energy": "eV"},
                               "note": "H-H···H-H colineal, energía frente a 6 Å",
                               "curve": curve}, indent=1, ensure_ascii=False),
                   encoding="utf-8")


def term_parameters(path: Path = REFERENCE) -> dict:
    """v0 at r0 = 1.5 Å and ρ from the wall (1.5 and 1.7 Å) of the GPAW curve."""
    curve = dict((round(d, 2), e) for d, e in json.loads(path.read_text())["curve"])
    rho = 0.2 / np.log(curve[1.5] / curve[1.7])
    return {"v0": round(float(curve[1.5]), 5), "r0": 1.5, "rho": round(float(rho), 5),
            "tail": [2.0, 2.4], "h2": [0.9, 1.1],
            "bonds": {x: [round(r + 0.15, 2), round(r + 0.35, 2)] for x, r in XH.items()}}


def term(path: Path = REFERENCE):
    from ..repulsive import HHContactTerm

    p = term_parameters(path)
    return HHContactTerm(p["v0"], p["r0"], p["rho"], tuple(p["tail"]),
                         {k: tuple(v) for k, v in p["bonds"].items()}, tuple(p["h2"]))


SETS = ("xu_chn", "xu_chno", "xu_chnob", "xu_chnos", "xu_chnop", "xu_chnose")


def install(reference: Path = REFERENCE) -> None:
    """Put the term second in each set's repulsion (where a refit puts it), once."""
    root = Path(__file__).resolve().parents[1] / "parameters"
    entry = dict(term(reference).to_dict(), unit="eV, Å",
                 source="repulsión H···H de GPAW (H2···H2 colineal, PBE, LCAO dzp; "
                        "receta tbkit.recipes.hh_contact); no ajustada")
    for name in SETS:
        path = root / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        terms = data["repulsive"]["terms"]
        present = [k for k, t in enumerate(terms) if t["type"] == "hh_contact"]
        if present:
            if terms[present[0]] == entry:
                continue
            terms[present[0]] = entry           # updated parameters, same place
        else:
            terms.insert(1, entry)
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(name, "instalado")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("references").add_argument("out", type=Path)
    sub.add_parser("term").add_argument("reference", type=Path, nargs="?", default=REFERENCE)
    sub.add_parser("install")
    args = parser.parse_args(argv)
    if args.stage == "references":
        references(args.out)
    elif args.stage == "install":
        install()
    else:
        print(json.dumps(term_parameters(args.reference), indent=1))


if __name__ == "__main__":
    main()
