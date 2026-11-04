"""Transition metals with d orbitals on top of the xu_ch* sets: Mo first (``xu_chnosmo``).

Run from packages/tbkit (after ``recipes/tm_references Mo``)::

    python -m tbkit.recipes.xu_tm Mo tbkit/parameters/references/gpaw_mo.json OUT.json \\
        --workers 4 --onsite-window 8 --bands T/MoS2/bands*.json T/Mo_bcc/bands.json \\
        --held-out-bands T/MoCS/bands.json T/MoCS/bands_x0.96.json T/MoCS/bands_x1.04.json \\
        --crystals T/MoS2 T/Mo_bcc --held-out-crystals T/MoCS

(``T`` = ``tbkit/parameters/references/tm_crystals``, from :mod:`tbkit.recipes.tm_crystals`;
the shipped ``xu_chnosmo.json``.)

Same machinery as :mod:`tbkit.recipes.xu_bsp` (:class:`~tbkit.recipes.xu_family.XuFamily`):
the base set is held fixed (xu_chnos, which has the C and S of MoCS and MoS₂) and the metal
is added with s, p and d orbitals (``d_elements``): its on-site s, p and d energies, the
Slater-Koster integrals of every pair with it (sd, pd and dd included) and the pair
repulsions are fitted to GPAW (PBE, LCAO dzp, the 6-electron Mo setup) on closed-shell
molecules. The Hubbard U is the free atom's dε_d/dn_d (computed, not fitted).

Molecules alone leave the solid unconstrained (the first fit made MoS₂ metallic), so the
GPAW bands of :mod:`tbkit.recipes.tm_crystals` join the fit with ``--bands`` (MoS₂ relaxed
and strained, Mo bcc), the way ``c_bandas`` was fitted for carbon; ``--held-out-bands``
(MoCS) are reported only, the test. ``--crystals`` adds their energy and force points
(relaxed, ±2 %, rattled) to forces and energies: bands alone left the crystal repulsion
unconstrained (MoS₂ forces 1.1 eV/Å at the GPAW minimum). ``--onsite-window`` keeps each on-site energy within
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
#: Mo-Mo is centred on the metallic bond (bcc nearest neighbours, 2.73 Å), not on the Mo2
#: quintuple bond (1.94 Å): the GSP law decays past rc = 2.18 r0/1.536, and from the dimer
#: that is 2.75 Å, which left no Mo-Mo hopping at the 3.18 Å of MoS₂.
R0 = {"Mo": {"H": 1.70, "C": 2.10, "N": 1.98, "O": 1.80, "S": 2.20, "Mo": 2.73}}
#: Mo-Mo hopping tail (Å). Xu's tail scaled from the dimer's 1.94 Å ended at 3.09-3.28 Å,
#: right on the Mo-Mo distance of MoS₂ (a = 3.18 Å) and the second neighbours of bcc Mo
#: (3.16 Å): the d-d hopping that sets the MoS₂ gap switched off between -2 % and +2 %
#: strain (the first band fit: gap 1.49 eV at 0.98 a, metallic at 1.02 a). The tail starts
#: past both and ends before the third bcc shell (4.47 Å).
MO_MO_TAIL = (3.90, 4.20)


#: S-S long-range hopping (Å): xu_chnos's S-S law is 0.02 eV at 3.15 Å, so the S sheets of
#: MoS₂ (S-S 3.12 Å across the layer, 3.18 Å in the plane) had no coupling; the TMD
#: tight-binding models need it (Cappelluti et al., PRB 88, 075409, 2013). Switched on
#: between 2.6 and 2.9 Å, past every S-S bond (2.05 Å) of the base, off at 4.0 Å, before
#: the next S-S shell of MoS₂ (4.45 Å).
LONG_RANGE = {("S", "S"): {"r0": 3.15, "onset": (2.6, 2.9), "tail": (3.7, 4.0)}}


def _mo_pairs() -> dict:
    pairs = _pairs("Mo", R0["Mo"])
    pairs[("Mo", "Mo")]["tail"] = MO_MO_TAIL
    return pairs


#: dε_d/dn_d of the free atom, GPAW aeatom PBE, 6-electron Mo (eV)
HUBBARD_U = {"Mo": 7.886}

FAMILIES = {
    "Mo": XuFamily(
        name="C/H/N/O/S/Mo: xu_chnos + Mo (s, p, d) ajustado a GPAW",
        recipe="tbkit.recipes.xu_tm",
        heteroatoms=("Mo",),
        pairs=_mo_pairs(),
        hubbard_u=HUBBARD_U,
        onsite_dipole={},
        d_elements=("Mo",),
        long_range=LONG_RANGE,
        validity_notes=("Mo ajustado en moléculas de capa cerrada (Mo(VI) d0 con oxo, hidroxo, "
                        "metilo, amido, tiolato, sulfuro e hidruro; Mo(CO)6; Mo2) y, con "
                        "--crystals/--bands, en MoS2 2H y Mo bcc (MoCS Janus apartado como "
                        "prueba); la energía frente a la deformación de la red sale 3-10 veces "
                        "más rígida que GPAW (MoS2 ±2 %: 0,13/0,48 eV frente a 0,05/0,04): no "
                        "relajar la celda; el gap de MoS2 sale indirecto (GPAW directo en K); "
                        "sin espín-órbita; Mo junto con B, P o Se no está cubierto"),
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
    parser.add_argument("--band-edge-weight", type=float, default=1.0,
                        help="peso extra de las dos bandas del borde y de los gaps (directo en "
                             "cada k y fundamental) en aislantes")
    parser.add_argument("--crystals", type=Path, nargs="*", default=[],
                        help="carpetas de tm_crystals cuyas energías y fuerzas entran en el ajuste")
    parser.add_argument("--held-out-crystals", type=Path, nargs="*", default=[],
                        help="carpetas de tm_crystals que solo se comparan (prueba)")
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
                  band_window=args.band_window, band_edge_weight=args.band_edge_weight,
                  crystals=args.crystals,
                  held_out_crystals=args.held_out_crystals)


if __name__ == "__main__":
    main()
