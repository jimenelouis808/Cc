"""B, S, P and Se on top of ``xu_chno``: ``xu_chnob``, ``xu_chnos``, ``xu_chnop``, ``xu_chnose``.

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
import dataclasses
from pathlib import Path

from . import xu_family
from .xu_family import XuFamily

#: Free atom, GPAW aeatom PBE spin-paired (``tbkit.references.gpaw_hubbard_u``,
#: ``gpaw_onsite_dipole``): U = dε_p/dn (eV), d = |⟨ns|r|np⟩| (Å).
HUBBARD_U = {"B": 8.0573, "S": 8.9465, "P": 7.8736, "Se": 8.1685}
ONSITE_DIPOLE = {"B": 0.6208, "S": 0.5491, "P": 0.6162, "Se": 0.5916}


def _pairs(element: str, r0: dict, margin: float = 0.55, long_rc: dict | None = None) -> dict:
    """Pairs of ``element`` with H, C, N, O and itself; repulsion cutoff r0 + margin
    (beyond stretched bonds, short of second neighbours), or ``long_rc[other]`` Å
    kept as given (``keep_rc``) where data show the model needs repulsion further
    out."""
    pairs = {}
    for other, d in r0.items():
        key = (other, element) if other != element else (element, element)
        pairs[key] = {"r0": d, "rc_rep": round(d + margin, 2)}
        if long_rc and other in long_rc:
            pairs[key].update(rc_rep=long_rc[other], keep_rc=True)
    return pairs


R0 = {"B": {"H": 1.19, "C": 1.57, "N": 1.44, "O": 1.37, "B": 1.70},
      "S": {"H": 1.34, "C": 1.82, "N": 1.65, "O": 1.45, "S": 2.05},
      "P": {"H": 1.42, "C": 1.85, "N": 1.70, "O": 1.60, "P": 2.22},
      "Se": {"H": 1.47, "C": 1.95, "N": 1.85, "O": 1.65, "Se": 2.33}}

FAMILIES = {
    "B": XuFamily(
        name="C/H/N/O/B: xu_chno + B ajustado a GPAW",
        recipe="tbkit.recipes.xu_bsp",
        heteroatoms=("B",),
        # B-H repulsion out to 2.4 Å: methyl H atoms of B(OCH3)3 collapsed onto the
        # O atoms, 2.1 Å from B, with nothing repulsive there (active learning data)
        pairs=_pairs("B", R0["B"],
                     long_rc={"H": 2.4}),
        hubbard_u={"B": HUBBARD_U["B"]},
        onsite_dipole={"B": ONSITE_DIPOLE["B"]},
        validity_notes=('B junto con S o P en la misma estructura no está cubierto; enlaces de B a menos de 0.03 Å de GPAW; ésteres B(OCH3)n corregidos con active learning (antes colapsaban; RMS 949 -> 95 cm⁻¹), aunque al relajar giran sus metilos (desplazamiento ~1 Å, sin colapso); ácidos borónicos arílicos (PhB(OH)2): corregidos con el contacto H···H (torsión del B(OH)2 336° frente a 337° de GPAW; antes quedaba plano, 0,6 Å)'),
        crystal_notes=("B grafítico en grafeno: fuerzas 1,4 veces el error molecular, B-C "
                       "+0,027 Å; h-BN: red 1,1 % larga (2,53 frente a 2,50 Å) y 10-15 % más "
                       "blanda en distorsiones"),
        system=("xu_chno más boro: boranos, ésteres y ácidos bóricos y borónicos, "
                "amino-borano, borazina, B-N en grafeno"),
        h_tail=(0.40, 0.60),
        base="xu_chno"),
    "S": XuFamily(
        name="C/H/N/O/S: xu_chno + S ajustado a GPAW",
        recipe="tbkit.recipes.xu_bsp",
        heteroatoms=("S",),
        pairs=_pairs("S", R0["S"]),
        hubbard_u={"S": HUBBARD_U["S"]},
        onsite_dipole={"S": ONSITE_DIPOLE["S"]},
        validity_notes=("S junto con B o P en la misma estructura no está cubierto; los C-S "
                        "simples con carbono sp3 se desvían 0.06-0.09 Å (S=O, S-S, S-H, S-N y "
                        "C-S aromático a menos de 0.04 Å)"),
        crystal_notes=("S en grafeno: fuerzas 0,5-0,6 veces el error molecular, S-C +0,03 Å, "
                       "S 0,06 Å más alto sobre la hoja"),
        system=("xu_chno más azufre: tioles, sulfuros, disulfuros, tiofeno, sulfóxidos, "
                "sulfonas, ácidos sulfónicos, sulfonamidas"),
        h_tail=(0.40, 0.60),
        base="xu_chno"),
    "P": XuFamily(
        name="C/H/N/O/P: xu_chno + P ajustado a GPAW",
        recipe="tbkit.recipes.xu_bsp",
        heteroatoms=("P",),
        # P-H repulsion out to 2.4 Å, for the same collapse in PO(OCH3)3
        pairs=_pairs("P", R0["P"],
                     long_rc={"H": 2.4}),
        hubbard_u={"P": HUBBARD_U["P"]},
        onsite_dipole={"P": ONSITE_DIPOLE["P"]},
        validity_notes=('P junto con B o S en la misma estructura no está cubierto; el C-P de P(V) (ácidos fosfónicos, óxidos de fosfina) sale 0.05-0.10 Å largo, el de P(III) 0.04-0.06 Å; P-O, P=O, P-P, P-N, P-H y C-P aromático a menos de 0.025 Å; ésteres P(OCH3)n corregidos con active learning (antes colapsaban); ácidos con P-OH (fosfórico, fosfónicos): el OH gira hacia el otro O al relajar, 0,4-0,9 Å; causa: la base mínima (GPAW con base sz también hunde el giro del OH: -0,35 eV a 180° frente a -0,23 con dzp; en TB es todo electrónico); en MePO3H2 los d del P también importan en DFT; úsense sus frecuencias O-H con cautela'),
        crystal_notes=("P en grafeno: fuerzas 0,3-0,6 veces el error molecular, P-C exacto "
                       "(0,001 Å) pero P 0,12 Å más bajo sobre la hoja: ángulos C-P-C "
                       "demasiado abiertos"),
        system=("xu_chno más fósforo: fosfinas, óxidos de fosfina, ácidos fosfórico y "
                "fosfónicos, fosfatos, fosfinina"),
        h_tail=(0.40, 0.60),
        base="xu_chno"),
    "Se": XuFamily(
        name="C/H/N/O/Se: xu_chno + Se ajustado a GPAW",
        recipe="tbkit.recipes.xu_bsp",
        heteroatoms=("Se",),
        pairs=_pairs("Se", R0["Se"]),
        hubbard_u={"Se": HUBBARD_U["Se"]},
        onsite_dipole={"Se": ONSITE_DIPOLE["Se"]},
        validity_notes=('Se junto con B, S o P en la misma estructura no está cubierto; enlaces de Se a menos de 0.08 Å; ácidos seleníninicos (Se(=O)OH): el OH gira hacia el otro O al relajar, 0,4-0,9 Å; causa: la base mínima (en el giro rígido la repulsión no aporta y la parte electrónica pone 240-300° 0,1 eV bajo el mínimo; GPAW con base mínima sz hace lo mismo, -0,06 eV, y con dzp +0,03); no son los d del Se (GPAW sin ellos no gira); úsense sus frecuencias O-H con cautela'),
        crystal_notes=("Se en grafeno: fuerzas ~1 vez el error molecular, Se-C +0,06 Å, Se "
                        "0,07 Å más alto sobre la hoja"),
        system=("xu_chno más selenio: selenoles, selenuros, diselenuros, selenofeno, "
                "selenóxidos, ácidos selenínicos, Se-N"),
        h_tail=(0.40, 0.60),
        base="xu_chno"),
}


def angular_variant(family: XuFamily, element: str, ligands, powers: int = 3) -> XuFamily:
    """``family`` plus angle stiffness at ``element`` between ``ligands`` only.

    All-ligand stiffness (every X-Y-Z angle at the centre) was tried and broke
    divalent Se (H2Se 64 -> 289 cm⁻¹): the angles the minimal basis gets wrong
    are the O-X-O of hypervalent centres, so the term can be restricted to them.
    Bonds count from 0.15 Å past the typical length, fading out by 0.40 Å.
    """
    bonds = {o: (round(R0[element][o] + 0.15, 2), round(R0[element][o] + 0.40, 2))
             for o in ligands}
    return dataclasses.replace(family, angular={"centre": element, "powers": powers,
                                                "bonds": bonds})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta B, S o P a GPAW sobre xu_chno.")
    parser.add_argument("element", choices=sorted(FAMILIES))
    parser.add_argument("references", type=Path, nargs="+",
                        help="archivo(s) JSON de referencias GPAW")
    parser.add_argument("out", type=Path,
                        help="archivo de parámetros que se escribe (nunca sobre uno incluido)")
    parser.add_argument("--workers", type=int, default=4,
                        help="procesos en paralelo para evaluar las referencias")
    parser.add_argument("--hessians", type=Path, default=None,
                        help="hessianas GPAW (frequency_references) para ajustar la curvatura")
    parser.add_argument("--hessian-weight", type=float, default=0.1,
                        help="peso de las Hessianas GPAW frente a energías y fuerzas en el ajuste")
    parser.add_argument("--angular-ligands", nargs="+", default=None,
                        help="añade rigidez angular en el heteroátomo solo entre estos "
                             "ligandos (p. ej. O: ángulos O-X-O de centros hipervalentes)")
    parser.add_argument("--angular-powers", type=int, default=3,
                        help="potencias del término angular de --angular-ligands")
    xu_family.add_fit_arguments(parser)
    args = parser.parse_args(argv)
    family = FAMILIES[args.element]
    if args.angular_ligands:
        family = angular_variant(family, args.element, args.angular_ligands,
                                 args.angular_powers)
    xu_family.run(family, args.references, args.out, workers=args.workers,
                  hessians=args.hessians, hessian_weight=args.hessian_weight,
                  fit=xu_family.fit_options(args))


if __name__ == "__main__":
    main()
