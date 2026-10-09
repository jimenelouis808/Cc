"""Fit H and N to GPAW on top of Xu's carbon: the ``xu_chn`` parameter set.

Carbon-carbon is Xu, Wang, Chan and Ho's model, untouched (on-site
energies, hoppings and embedded repulsion): it sets the energy zero and
everything else is fitted around it. The fit adds

* on-site energies of H (s) and N (s, p);
* Slater-Koster hoppings C-H, N-H, C-N and N-N, in Xu's own GSP form
  ``v0 (r0/d)^n exp{n[-(d/rc)^nc + (r0/rc)^nc]}`` with ``nc = 6.5`` and ``rc``
  and the smooth tail scaled from Xu's by ``r0_pair / r0_CC``; ``v0`` per bond
  type and one ``n`` per pair are fitted;
* Hubbard U of H, C and N: not fitted, computed as ``dε/dn`` of the free
  atom's valence level with GPAW's all-electron atom (PBE, spin-paired) --
  the DFTB definition; the values agree with DFTB's mio set to 1e-4 Ha;
* pair repulsions C-H, N-H, C-N, N-N, ``Σ_k c_k (rc - d)^k`` (k = 3...7).

The ground state is self-consistent (DFTB2-like, ``scc: true``): charge
transfer between C and N depends on the environment (graphitic, pyridinic,
pyrrolic), which a fixed on-site energy cannot follow.

Stage 1 (electronic): least squares on the Kohn-Sham levels of the
training structures -- all occupied valence levels, the LUMO and (lightly)
the LUMO+1, the frontier ones weighted double; empty levels matter for the
polarizability, as fitting a few empty bands does in NRL-TB
(Papaconstantopoulos et al., Encycl. Condens. Matter Phys. 2024) -- with one
common shift between model and DFT levels (both are absolute for finite
molecules; the shift is Xu's arbitrary zero). Stage 2: levels, forces and
energies together, the pair repulsion (linear in its coefficients) solved
exactly at each step.

The machinery is :mod:`tbkit.recipes.xu_family`; this module is its C/H/N
instance and keeps the names the tests and the shipped set refer to.

Run::

    python -m tbkit.recipes.xu_chn REFERENCES.json OUT.json

``REFERENCES.json`` from :mod:`tbkit.recipes.chn_references`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from . import xu_family
from .xu_family import (  # noqa: F401  (public names of this recipe)
    POWERS,
    XuFamily,
    bond_lengths,
    bond_ranges,
    electronic_energy_and_forces,
    level_residuals,
    model_levels,
    relaxed_bond_errors,
    validation_table,
)
from .xu_family import selection as _selection  # noqa: F401

#: Hubbard U (eV) from GPAW's all-electron atom, PBE, spin-paired,
#: U = dε/dn of the valence level by central differences (±0.05 e);
#: GPAW 25.7. DFTB mio: H 0.4195, C 0.3647, N 0.4309 Ha -- the same.
HUBBARD_U = {"H": 11.4154, "C": 9.9228, "N": 11.7247}

#: Intra-atomic dipole |⟨2s|r|2p⟩| (Å) of the free atom, GPAW's all-electron
#: atom, PBE (``tbkit.references.gpaw_onsite_dipole``); optics only.
ONSITE_DIPOLE = {"C": 0.4952, "N": 0.4131}

#: Mean static polarizabilities (Å³), experiment: T. M. Miller, "Atomic and
#: molecular polarizabilities", CRC Handbook of Chemistry and Physics. Only
#: molecules with a well-established value; a comparison, not a fit target.
EXPERIMENTAL_ALPHA = {"CH4": 2.593, "C2H6": 4.47, "C2H4": 4.252, "C6H6": 10.32, "N2": 1.7403}

#: Per pair: reference length r0 (a typical single bond, Å) and the repulsion
#: cutoff (beyond the longest stretched bond of the training data, short of
#: second neighbours).
CHN = XuFamily(
    name="C/H/N: Xu (C-C) + H, N ajustados a GPAW",
    recipe="tbkit.recipes.xu_chn",
    heteroatoms=("H", "N"),
    pairs={("C", "H"): {"r0": 1.09, "rc_rep": 1.75},
           ("N", "H"): {"r0": 1.01, "rc_rep": 1.65},
           ("C", "N"): {"r0": 1.47, "rc_rep": 2.2},
           ("N", "N"): {"r0": 1.45, "rc_rep": 2.1}},
    hubbard_u=HUBBARD_U,
    onsite_dipole=ONSITE_DIPOLE,
    system=("moléculas C/H/N de capa cerrada: hidrocarburos saturados, insaturados y "
            "aromáticos; aminas, iminas, nitrilos, N piridínico y pirrólico"),
    validity_notes=("falla en anillos tensos (aziridina); los C-C y C-N simples junto a un "
                    "heteroátomo se desvían ~0.08 Å"),
    crystal_notes=("N grafítico y piridínico (N3V) en grafeno, grafano y CNT (8,0) con N: "
                   "error de fuerzas en el mínimo de GPAW 0,5-1,2 veces el molecular, C-N "
                   "a menos de 0,02 Å; N sustitucional en diamante: C-N -0,08 Å (1,52 "
                   "frente a 1,60, como los C-N simples de las moléculas) y fuerzas 1,4-1,7 "
                   "veces el molecular (ambos con espín apareado: el centro P1 real es de "
                   "capa abierta); la red de TB sale 0,5-1 % más corta que la de GPAW-dzp "
                   "(el grafano, igual)"),
    experimental_alpha=EXPERIMENTAL_ALPHA,
)
PAIRS = CHN.pairs


def parameter_names() -> list[str]:
    return CHN.parameter_names()


def initial_guess() -> np.ndarray:
    return CHN.initial_guess()


def build_model(x, repulsive: Optional[object] = None):
    """Xu's carbon plus H and N from the parameter vector ``x``."""
    return CHN.build_model(x, repulsive)


def repulsion_from_coefficients(coefficients):
    return CHN.repulsion_from_coefficients(coefficients)


def fit_levels(refs, x0=None):
    return xu_family.fit_levels(CHN, refs, x0)


def fit_repulsion(model, refs, energy_weight: float = 3.0, ridge: float = 1e-8):
    """Linear least squares for the pair coefficients (electronic part fixed)."""
    return xu_family.fit_repulsion(CHN, model, refs, energy_weight, ridge)


def fit_joint(refs, x0, workers: int = 4, max_nfev: int = 400, verbose: bool = True,
              **weights):
    return xu_family.fit_joint(CHN, refs, x0, workers, max_nfev, verbose, **weights)


def validate(model, structures, shift: float, relax: bool = True) -> dict:
    return xu_family.validate(CHN, model, structures, shift, relax)


def run(references: Path, out: Path, verbose: bool = True, workers: int = 4) -> dict:
    return xu_family.run(CHN, [Path(references)], out, verbose, workers)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta H y N a GPAW sobre el C de Xu.")
    parser.add_argument("references", type=Path,
                        help="archivo(s) JSON de referencias GPAW")
    parser.add_argument("out", type=Path,
                        help="archivo de parámetros que se escribe (nunca sobre uno incluido)")
    parser.add_argument("--workers", type=int, default=4,
                        help="procesos en paralelo para evaluar las referencias")
    args = parser.parse_args(argv)
    run(args.references, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
