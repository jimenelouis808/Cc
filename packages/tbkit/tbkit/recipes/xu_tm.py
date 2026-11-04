"""Transition metals with d orbitals on top of the xu_ch* sets: Mo first (``xu_chnosmo``).

Run from packages/tbkit (after ``recipes/tm_references Mo``)::

    python -m tbkit.recipes.xu_tm Mo tbkit/parameters/references/gpaw_mo.json OUT.json \\
        --workers 4

Same machinery as :mod:`tbkit.recipes.xu_bsp` (:class:`~tbkit.recipes.xu_family.XuFamily`):
the base set is held fixed (xu_chnos, which has the C and S of MoCS and MoS₂) and the metal
is added with s, p and d orbitals (``d_elements``): its on-site s, p and d energies, the
Slater-Koster integrals of every pair with it (sd, pd and dd included) and the pair
repulsions are fitted to GPAW (PBE, LCAO dzp, the 6-electron Mo setup) on closed-shell
molecules. The Hubbard U is the free atom's dε_d/dn_d (computed, not fitted).

Molecules alone leave the solid unconstrained (the first fit made MoS₂ metallic), so the
GPAW bands of :mod:`tbkit.recipes.tm_crystals` join the fit with ``--bands`` (MoS₂ relaxed
and strained, Mo bcc), the way ``c_bandas`` was fitted for carbon; ``--held-out-bands``
(MoCS) are reported only, the test. ``--onsite-window`` keeps each on-site energy within
that many eV of the free atom's level.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from . import xu_family
from .xu_bsp import _pairs
from .xu_family import XuFamily

#: Typical Mo bond lengths in the GPAW references (Å): the centre of each distance law.
R0 = {"Mo": {"H": 1.70, "C": 2.10, "N": 1.98, "O": 1.80, "S": 2.20, "Mo": 1.94}}
#: dε_d/dn_d of the free atom, GPAW aeatom PBE, 6-electron Mo (eV)
HUBBARD_U = {"Mo": 7.886}

FAMILIES = {
    "Mo": XuFamily(
        name="C/H/N/O/S/Mo: xu_chnos + Mo (s, p, d) ajustado a GPAW",
        recipe="tbkit.recipes.xu_tm",
        heteroatoms=("Mo",),
        pairs=_pairs("Mo", R0["Mo"]),
        hubbard_u=HUBBARD_U,
        onsite_dipole={},
        d_elements=("Mo",),
        validity_notes=("Mo ajustado en moléculas de capa cerrada (Mo(VI) d0 con oxo, hidroxo, "
                        "metilo, amido, tiolato, sulfuro e hidruro; Mo(CO)6; Mo2); sólidos y "
                        "monocapas (MoS2, MoCS) son extrapolación hasta su validación; sin "
                        "espín-órbita; Mo junto con B, P o Se no está cubierto"),
        system="xu_chnos más molibdeno: óxidos, sulfuros, carbonilos, carburos de Mo",
        h_tail=(0.40, 0.60),
        base="xu_chnos"),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta un metal de transición (s, p, d) a "
                                                 "GPAW sobre un conjunto xu_ch*.")
    parser.add_argument("element", choices=sorted(FAMILIES))
    parser.add_argument("references", type=Path, nargs="+",
                        help="archivo(s) JSON de referencias GPAW (tm_references)")
    parser.add_argument("out", type=Path,
                        help="archivo de parámetros que se escribe (nunca sobre uno incluido)")
    parser.add_argument("--workers", type=int, default=4,
                        help="procesos en paralelo para evaluar las referencias")
    parser.add_argument("--bands", type=Path, nargs="*", default=[],
                        help="bandas GPAW de cristales (tm_crystals) que entran en el ajuste")
    parser.add_argument("--held-out-bands", type=Path, nargs="*", default=[],
                        help="bandas GPAW que solo se comparan (prueba)")
    parser.add_argument("--band-weight", type=float, default=1.0,
                        help="peso de cada cristal en el ajuste conjunto")
    parser.add_argument("--band-window", type=float, default=3.0,
                        help="ventana de bandas alrededor del centro del gap o E_F, eV")
    parser.add_argument("--onsite-window", type=float, default=None,
                        help="las energías on-site quedan a ± tanto (eV) del nivel del átomo libre")
    xu_family.add_fit_arguments(parser)
    args = parser.parse_args(argv)
    family = FAMILIES[args.element]
    if args.onsite_window is not None:
        family = dataclasses.replace(family, onsite_window=args.onsite_window)
    xu_family.run(family, args.references, args.out, workers=args.workers,
                  fit=xu_family.fit_options(args), bands=args.bands,
                  held_out_bands=args.held_out_bands, band_weight=args.band_weight,
                  band_window=args.band_window)


if __name__ == "__main__":
    main()
