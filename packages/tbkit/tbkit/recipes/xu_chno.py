"""C/H/N/O on top of Xu's carbon: the ``xu_chno`` parameter set.

The ``xu_chn`` recipe (:mod:`tbkit.recipes.xu_chn`, machinery in
:mod:`tbkit.recipes.xu_family`) extended with oxygen: on-site energies of O,
hoppings and pair repulsions C-O, O-H, N-O and O-O, fitted together with the
H and N parameters (which start from their ``xu_chn`` values) to the GPAW
references of both sets. Hubbard U and intra-atomic dipole of O computed
with GPAW's atom like the others (U_O = 13.48 eV, = DFTB mio's 0.4954 Ha;
d_O = 0.355 Å). Three-membered rings (epoxide, oxirane, aziridine,
cyclopropane) get the acute-angle correction of
:class:`tbkit.repulsive.AcuteAngleTerm`, zero for every angle ≥ 80° (so
graphene, nanotubes, fullerenes and aromatics keep Xu's results exactly),
fitted with the pair repulsion. O-O is needed although peroxides are rare in functionalised
carbon: the two oxygens of a carboxyl or nitro group are 2.2 Å apart, inside
the hopping range.

Run (after :mod:`tbkit.recipes.chno_references`)::

    python -m tbkit.recipes.xu_chno gpaw_chn.json gpaw_chno.json OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import xu_family
from .xu_chn import EXPERIMENTAL_ALPHA, HUBBARD_U, ONSITE_DIPOLE
from .xu_family import XuFamily

CHNO = XuFamily(
    name="C/H/N/O: Xu (C-C) + H, N, O ajustados a GPAW",
    recipe="tbkit.recipes.xu_chno",
    heteroatoms=("H", "N", "O"),
    pairs={("C", "H"): {"r0": 1.09, "rc_rep": 1.75},
           ("N", "H"): {"r0": 1.01, "rc_rep": 1.65},
           ("C", "N"): {"r0": 1.47, "rc_rep": 2.2},
           ("N", "N"): {"r0": 1.45, "rc_rep": 2.1},
           ("C", "O"): {"r0": 1.43, "rc_rep": 2.1},
           ("O", "H"): {"r0": 0.97, "rc_rep": 1.6},
           ("N", "O"): {"r0": 1.40, "rc_rep": 2.0},
           ("O", "O"): {"r0": 1.47, "rc_rep": 1.9}},
    hubbard_u={**HUBBARD_U, "O": 13.4802},
    onsite_dipole={**ONSITE_DIPOLE, "O": 0.3550},
    system=("moléculas C/H/N/O de capa cerrada: las de xu_chn más alcoholes, éteres, "
            "epóxidos, aldehídos, cetonas, ácidos carboxílicos, ésteres, amidas, furano, "
            "nitro; motivos de óxido de grafeno (epóxido e hidroxilos basales)"),
    validity_notes=("anillos de tres miembros con la corrección de ángulos agudos ajustada a "
                    "barridos de apertura del anillo: epóxido basal sobre coroneno C-C 0.03 Å, "
                    "C-O 0.10 Å; oxirano y aziridina C-C ~0.12 Å; ciclopropano C-C +0.19 Å con "
                    "un mínimo poco profundo; el biciclobutano (excluido del ajuste) se abre; "
                    "los C-C y C-N simples junto a un heteroátomo se desvían ~0.08 Å"),
    experimental_alpha=EXPERIMENTAL_ALPHA,
    acute={"powers": 2, "theta0_degrees": 80.0, "r1": 1.7, "rm": 2.0,
           "elements": ("C", "N", "O")},
    h_tail=(0.40, 0.60),
    held_out={"bicyclobutane": ("tensión extrema (dos anillos de tres miembros fusionados), "
                                "nada representativa del carbono funcionalizado; en el "
                                "entrenamiento aportaba la mitad del error de fuerzas")},
)


def warm_start() -> np.ndarray:
    """Initial vector: xu_chn's fitted values where they exist, guesses for O."""
    from ..params import PARAMETER_DIR

    fitted = json.loads((PARAMETER_DIR / "xu_chn.json").read_text(encoding="utf-8"))
    known = fitted["fit"]["parameters"]
    guess = CHNO.initial_guess()
    names = CHNO.parameter_names()
    return np.array([known.get(name, value) for name, value in zip(names, guess, strict=True)])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta H, N y O a GPAW sobre el C de Xu.")
    parser.add_argument("references", type=Path, nargs="+")
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hessians", type=Path, default=None,
                        help="hessianas GPAW (frequency_references) para ajustar la curvatura")
    parser.add_argument("--hessian-weight", type=float, default=0.1)
    args = parser.parse_args(argv)
    xu_family.run(CHNO, args.references, args.out, workers=args.workers, x0=warm_start(),
                  hessians=args.hessians, hessian_weight=args.hessian_weight)


if __name__ == "__main__":
    main()
