"""B, S and P on top of ``xu_chno``: the ``xu_chnob``, ``xu_chnos`` and ``xu_chnop`` sets.

Each adds one element to ``xu_chno`` -- kept fixed, numbers, repulsion and
level shift included (:attr:`tbkit.recipes.xu_family.XuFamily.base`) -- so a
molecule without B, S or P gives exactly the ``xu_chno`` result. Fitted per
element: its s and p on-site energies, the hoppings and pair repulsions of
every bond it makes with H, C, N, O and itself (all pairs are required: a
missing hopping law would silently be zero). Hubbard U and ⟨ns|r|np⟩ are
computed with GPAW's all-electron atom (PBE), like the others; the U agree
with DFTB's 3ob/matsci sets (B 0.2961, P 0.2894, S 0.3288 Ha).

One element per set: B with S or P in the same structure is not covered
(rare in doped carbon); B-N, N-S and N-P co-doping are.

Run (after :mod:`tbkit.recipes.bsp_references`)::

    python -m tbkit.recipes.xu_bsp B gpaw_b.json OUT.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import xu_family
from .xu_family import XuFamily

#: Free atom, GPAW aeatom PBE spin-paired (``tbkit.references.gpaw_hubbard_u``,
#: ``gpaw_onsite_dipole``): U = dε_p/dn (eV), d = |⟨ns|r|np⟩| (Å).
HUBBARD_U = {"B": 8.0573, "S": 8.9465, "P": 7.8736}
ONSITE_DIPOLE = {"B": 0.6208, "S": 0.5491, "P": 0.6162}


def _pairs(element: str, r0: dict, margin: float = 0.55) -> dict:
    """Pairs of ``element`` with H, C, N, O and itself; repulsion cutoff r0 + margin
    (beyond stretched bonds, short of second neighbours)."""
    return {(other, element) if other != element else (element, element):
            {"r0": d, "rc_rep": round(d + margin, 2)} for other, d in r0.items()}


FAMILIES = {
    "B": XuFamily(
        name="C/H/N/O/B: xu_chno + B ajustado a GPAW",
        recipe="tbkit.recipes.xu_bsp",
        heteroatoms=("B",),
        pairs=_pairs("B", {"H": 1.19, "C": 1.57, "N": 1.44, "O": 1.37, "B": 1.70}),
        hubbard_u={"B": HUBBARD_U["B"]},
        onsite_dipole={"B": ONSITE_DIPOLE["B"]},
        validity_notes="B junto con S o P en la misma estructura no está cubierto",
        system=("xu_chno más boro: boranos, ésteres y ácidos bóricos y borónicos, "
                "amino-borano, borazina, B-N en grafeno"),
        base="xu_chno"),
    "S": XuFamily(
        name="C/H/N/O/S: xu_chno + S ajustado a GPAW",
        recipe="tbkit.recipes.xu_bsp",
        heteroatoms=("S",),
        pairs=_pairs("S", {"H": 1.34, "C": 1.82, "N": 1.65, "O": 1.45, "S": 2.05}),
        hubbard_u={"S": HUBBARD_U["S"]},
        onsite_dipole={"S": ONSITE_DIPOLE["S"]},
        validity_notes=("S junto con B o P en la misma estructura no está cubierto; los C-S "
                        "simples con carbono sp3 se desvían 0.06-0.09 Å (S=O, S-S, S-H, S-N y "
                        "C-S aromático a menos de 0.04 Å)"),
        system=("xu_chno más azufre: tioles, sulfuros, disulfuros, tiofeno, sulfóxidos, "
                "sulfonas, ácidos sulfónicos, sulfonamidas"),
        base="xu_chno"),
    "P": XuFamily(
        name="C/H/N/O/P: xu_chno + P ajustado a GPAW",
        recipe="tbkit.recipes.xu_bsp",
        heteroatoms=("P",),
        pairs=_pairs("P", {"H": 1.42, "C": 1.85, "N": 1.70, "O": 1.60, "P": 2.22}),
        hubbard_u={"P": HUBBARD_U["P"]},
        onsite_dipole={"P": ONSITE_DIPOLE["P"]},
        validity_notes="P junto con B o S en la misma estructura no está cubierto",
        system=("xu_chno más fósforo: fosfinas, óxidos de fosfina, ácidos fosfórico y "
                "fosfónicos, fosfatos, fosfinina"),
        base="xu_chno"),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta B, S o P a GPAW sobre xu_chno.")
    parser.add_argument("element", choices=sorted(FAMILIES))
    parser.add_argument("references", type=Path, nargs="+")
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    xu_family.run(FAMILIES[args.element], args.references, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
