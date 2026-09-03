"""Measure and judge the geometry of a built MX2 structure.

The carbon side of this package has :func:`nanocarbon_lab.validation.
quality.sp2_quality`, which turns measured bond and angle ranges into a
one-word verdict. TMDs need their own because almost none of the sp2
thresholds transfer: the bond is metal-chalcogen at 2.4 Å rather than
C-C at 1.42, the metal is six-coordinate rather than three, and there is
no "ring size" to speak of.

What does transfer is the reason for having it at all. A rolled tube can
have every atom correctly coordinated, no overlaps, and bonds stretched
7% out of range; the numbers say so and the render does not.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms

Verdict = Literal["clean", "strained", "broken"]

#: Fractional deviation from the ideal M-X bond that still counts as a
#: sound starting structure. Rolling a 30 Å-radius MoS2 tube lands inside
#: this; rolling a 10 Å one does not, which is the distinction worth
#: drawing.
STRAIN_CLEAN = 0.02
STRAIN_INTACT = 0.06

#: Any two atoms closer than this are overlapping, whatever the species.
HARD_MIN_DISTANCE = 1.6


def geometry_report(atoms: Atoms) -> dict[str, float | int]:
    """Measured bond lengths, coordination and contacts for an MX2 model.

    Bonds are found by a distance cutoff halfway between the M-X bond and
    the next shell, which is unambiguous for these structures: the
    shortest non-bonded distance in a TMD (chalcogen to chalcogen across
    the layer, 3.1 Å) is far outside any stretched M-X bond.
    """
    from ase.neighborlist import neighbor_list

    metal = atoms.info.get("metal")
    chalcogen = atoms.info.get("chalcogen")
    ideal = float(atoms.info.get("bond_length", 2.4))
    if metal is None or chalcogen is None:
        raise ValueError(
            "geometry_report needs atoms.info['metal'] and ['chalcogen']; "
            "pass a structure built by nanocarbon_lab.tmd."
        )

    cutoff = ideal * 1.25
    first, second, distance = neighbor_list("ijd", atoms, cutoff=cutoff)
    symbols = np.array(atoms.get_chemical_symbols())

    is_bond = (
        ((symbols[first] == metal) & (symbols[second] == chalcogen))
        | ((symbols[first] == chalcogen) & (symbols[second] == metal))
    )
    bonds = distance[is_bond]

    coordination = np.zeros(len(atoms), dtype=int)
    np.add.at(coordination, first[is_bond], 1)

    metal_mask = symbols == metal
    chalcogen_mask = symbols == chalcogen
    n_metal = int(metal_mask.sum())
    n_chalcogen = int(chalcogen_mask.sum())

    # Overlaps: *any* pair closer than the hard minimum, bonded or not. An
    # M-X pair at 0.4 Å is not a short bond, it is two atoms on top of each
    # other, and excusing it because the species are bonding partners was
    # exactly how a folded structure could read as merely strained.
    close = distance[distance < HARD_MIN_DISTANCE]

    return {
        "bond_min": float(bonds.min()) if bonds.size else float("nan"),
        "bond_mean": float(bonds.mean()) if bonds.size else float("nan"),
        "bond_max": float(bonds.max()) if bonds.size else float("nan"),
        "bond_ideal": ideal,
        "n_bonds": int(bonds.size // 2),
        "metal_coordination_min": int(coordination[metal_mask].min())
        if n_metal else 0,
        "metal_coordination_max": int(coordination[metal_mask].max())
        if n_metal else 0,
        "chalcogen_coordination_min": int(coordination[chalcogen_mask].min())
        if n_chalcogen else 0,
        "chalcogen_coordination_max": int(coordination[chalcogen_mask].max())
        if n_chalcogen else 0,
        "n_metal": n_metal,
        "n_chalcogen": n_chalcogen,
        "stoichiometry": (n_chalcogen / n_metal) if n_metal else float("nan"),
        "n_close_contacts": int(close.size // 2),
    }


def tmd_quality(report: dict, expect_stoichiometric: bool = True
                ) -> tuple[Verdict, str]:
    """Turn a :func:`geometry_report` into a verdict and a reason.

    Judges the bond-length spread against the material's own ideal rather
    than an absolute window, because that ideal runs from 2.40 Å for MoS2
    to 2.73 Å for MoTe2 and a fixed range would call one of them wrong.

    ``expect_stoichiometric`` should be ``False`` for a deliberately
    terminated nanoribbon. A metal-terminated MoS2 edge really is
    sulphur-poor -- Mo12S20 rather than Mo12S24 -- and that is the point
    of asking for it, not a broken cut.
    """
    ideal = report["bond_ideal"]
    contacts = int(report["n_close_contacts"])
    if contacts:
        return "broken", (
            f"{contacts} non-bonded atom pairs closer than "
            f"{HARD_MIN_DISTANCE} Å — the layer has folded through itself."
        )

    ratio = report["stoichiometry"]
    if expect_stoichiometric and not np.isnan(ratio) and abs(ratio - 2.0) > 0.1:
        return "broken", (
            f"stoichiometry X/M = {ratio:.2f}, not MX2 — the cut has left "
            "the structure off-composition."
        )

    worst = max(abs(report["bond_max"] - ideal), abs(ideal - report["bond_min"]))
    strain = worst / ideal
    if strain > STRAIN_INTACT:
        return "broken", (
            f"M–X bonds deviate up to {strain:.1%} from {ideal:.2f} Å "
            f"({report['bond_min']:.2f}–{report['bond_max']:.2f} Å). For a "
            "rolled tube this means the radius is too small: the strain goes "
            "as h/2R, so raise the chiral index."
        )
    if strain > STRAIN_CLEAN:
        return "strained", (
            f"M–X bonds deviate up to {strain:.1%} from {ideal:.2f} Å — "
            "intact, and a usable starting structure, but a relaxation will "
            "move things noticeably."
        )
    return "clean", (
        f"M–X bonds within {strain:.1%} of {ideal:.2f} Å, coordination and "
        "stoichiometry as expected."
    )


__all__ = [
    "HARD_MIN_DISTANCE",
    "STRAIN_CLEAN",
    "STRAIN_INTACT",
    "Verdict",
    "geometry_report",
    "tmd_quality",
]
