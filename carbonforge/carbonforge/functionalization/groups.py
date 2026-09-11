"""Library of functional groups, defined by their internal geometry.

Each group is described in a **local frame** whose origin is the carbon it
attaches to and whose +z axis points along the attachment direction (out of
the edge, or normal to the basal plane). :mod:`carbonforge.functionalization.attach`
maps that frame onto a real site.

Geometries are built from tabulated bond lengths and angles rather than
hardcoded coordinates, so every number is inspectable and attributable. They
are **idealised starting geometries**: real functional groups relax, rotate
about their single bonds, and interact with neighbours. Feed the result to a
relaxation before drawing conclusions — :func:`carbonforge.relax.harmonic_pre_relax`
for a quick cleanup, a real calculator for anything quantitative.

Bond lengths (Å) and angles (degrees) follow standard organic values; see
:data:`BOND_LENGTHS` for the table and its per-entry justification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

#: Bond lengths in Å. Standard single/double/triple values for the relevant
#: hybridisations; these set the group geometry before relaxation.
BOND_LENGTHS: dict[str, float] = {
    "C-H": 1.09,     # sp3 C-H
    "C-C": 1.50,     # sp2 ring C to sp2 substituent C (e.g. carboxyl)
    "C-N": 1.47,     # sp3 C-N single (amine, nitro)
    "C-N#": 1.43,    # ring C to nitrile carbon
    "C#N": 1.16,     # nitrile triple bond
    "C-O": 1.43,     # C-O single (hydroxyl, epoxide, ether)
    "C=O": 1.23,     # carbonyl double bond
    "C-OH": 1.36,    # carboxyl C to its hydroxyl O (shorter: conjugation)
    "C-S": 1.82,     # C-S single (thiol)
    "N-H": 1.01,
    "N-O": 1.22,     # nitro N-O, both equivalent by resonance
    "O-H": 0.97,
    "S-H": 1.34,
}

#: Bond angles in degrees.
ANGLES: dict[str, float] = {
    "tetrahedral": 109.5,
    "trigonal": 120.0,
    "C-O-H": 108.5,   # hydroxyl
    "C-N-H": 110.0,   # amine, slightly pyramidal
    "O-N-O": 125.0,   # nitro, widened by O-O repulsion
    "C-S-H": 96.0,    # thiol; S uses more p character, so nearer 90 deg
}


def _spherical(length: float, polar_deg: float, azimuth_deg: float) -> np.ndarray:
    """Return a displacement of ``length`` at the given angles from +z."""
    polar = math.radians(polar_deg)
    azimuth = math.radians(azimuth_deg)
    return length * np.array(
        [
            math.sin(polar) * math.cos(azimuth),
            math.sin(polar) * math.sin(azimuth),
            math.cos(polar),
        ]
    )


@dataclass(frozen=True)
class FunctionalGroup:
    """A functional group in its local attachment frame.

    Attributes
    ----------
    key
        Short identifier used by the CLI and GUI.
    name
        Human-readable Spanish name.
    formula
        Conventional chemical notation, e.g. ``"-NH2"``.
    symbols
        Chemical symbols of the group's atoms, in order.
    positions
        Local coordinates, shape ``(n, 3)``. The anchor carbon sits at the
        origin and the group extends along +z.
    valence_used
        How many bonds the group consumes on the anchor carbon. ``1`` for a
        normal substituent; ``2`` for a carbonyl (a double bond) and for the
        bridging epoxide.
    bridging
        True when the group bonds to **two** adjacent carbons rather than
        one (only the epoxide here).
    note
        Chemistry worth knowing before using it.
    """

    key: str
    name: str
    formula: str
    symbols: tuple[str, ...]
    positions: np.ndarray
    valence_used: int = 1
    bridging: bool = False
    note: str = ""

    def __len__(self) -> int:
        return len(self.symbols)


def _hydrogen() -> FunctionalGroup:
    return FunctionalGroup(
        key="H",
        name="Hidrógeno",
        formula="-H",
        symbols=("H",),
        positions=np.array([[0.0, 0.0, BOND_LENGTHS["C-H"]]]),
        note="Pasivación simple de un borde. No aporta química, solo satura.",
    )


def _hydroxyl() -> FunctionalGroup:
    oxygen = np.array([0.0, 0.0, BOND_LENGTHS["C-O"]])
    # The C-O-H angle is measured at O between the O->C direction (-z) and
    # O->H, so H sits at (180 - angle) from +z.
    hydrogen = oxygen + _spherical(
        BOND_LENGTHS["O-H"], 180.0 - ANGLES["C-O-H"], 0.0
    )
    return FunctionalGroup(
        key="OH",
        name="Hidroxilo",
        formula="-OH",
        symbols=("O", "H"),
        positions=np.array([oxygen, hydrogen]),
        note=(
            "Grupo típico del óxido de grafeno. En el plano basal convierte "
            "el carbono en sp3 y arruga la lámina localmente."
        ),
    )


def _amine() -> FunctionalGroup:
    nitrogen = np.array([0.0, 0.0, BOND_LENGTHS["C-N"]])
    polar = 180.0 - ANGLES["C-N-H"]
    hydrogens = [
        nitrogen + _spherical(BOND_LENGTHS["N-H"], polar, azimuth)
        for azimuth in (60.0, -60.0)
    ]
    return FunctionalGroup(
        key="NH2",
        name="Amina",
        formula="-NH2",
        symbols=("N", "H", "H"),
        positions=np.array([nitrogen, *hydrogens]),
        note=(
            "Amina primaria: dona densidad electrónica al sistema π. "
            "El nitrógeno es ligeramente piramidal y rota libremente en "
            "torno al enlace C-N, así que esta es una de varias "
            "conformaciones equivalentes."
        ),
    )


def _nitro() -> FunctionalGroup:
    nitrogen = np.array([0.0, 0.0, BOND_LENGTHS["C-N"]])
    # Careful: O-N-O is the angle *between the two oxygens*, not the angle to
    # the anchor carbon. The bisector of O-N-O points along +z (away from the
    # carbon), so each N->O sits at half that angle from +z. Using the
    # 180 - angle form that suits a C-X-Y angle would fold the oxygens back
    # onto the sheet, where they land on top of the neighbouring ring carbons.
    half = ANGLES["O-N-O"] / 2.0
    oxygens = [
        nitrogen + _spherical(BOND_LENGTHS["N-O"], half, azimuth)
        for azimuth in (0.0, 180.0)
    ]
    return FunctionalGroup(
        key="NO2",
        name="Nitro",
        formula="-NO2",
        symbols=("N", "O", "O"),
        positions=np.array([nitrogen, *oxygens]),
        note=(
            "Fuertemente atractor de electrones: lo contrario de la amina. "
            "Los dos oxígenos son equivalentes por resonancia, de ahí que "
            "compartan la misma distancia N-O."
        ),
    )


def _nitrile() -> FunctionalGroup:
    carbon = np.array([0.0, 0.0, BOND_LENGTHS["C-N#"]])
    nitrogen = carbon + np.array([0.0, 0.0, BOND_LENGTHS["C#N"]])
    return FunctionalGroup(
        key="CN",
        name="Nitrilo",
        formula="-C≡N",
        symbols=("C", "N"),
        positions=np.array([carbon, nitrogen]),
        note="Grupo lineal, fuertemente atractor. El C es sp.",
    )


def _carboxyl() -> FunctionalGroup:
    carbon = np.array([0.0, 0.0, BOND_LENGTHS["C-C"]])
    # sp2 carbon: the two oxygens sit at 120 deg from the C-C bond, in a plane.
    polar = 180.0 - ANGLES["trigonal"]
    o_double = carbon + _spherical(BOND_LENGTHS["C=O"], polar, 0.0)
    o_single = carbon + _spherical(BOND_LENGTHS["C-OH"], polar, 180.0)
    # The acidic H continues roughly outward from the single-bonded O.
    direction = o_single - carbon
    direction = direction / np.linalg.norm(direction)
    hydrogen = o_single + BOND_LENGTHS["O-H"] * direction
    return FunctionalGroup(
        key="COOH",
        name="Carboxilo",
        formula="-COOH",
        symbols=("C", "O", "O", "H"),
        positions=np.array([carbon, o_double, o_single, hydrogen]),
        note=(
            "Grupo ácido, muy común en bordes de óxido de grafeno oxidado. "
            "La posición del H es aproximada: en la práctica forma puentes "
            "de hidrógeno con lo que tenga cerca."
        ),
    )


def _aldehyde() -> FunctionalGroup:
    carbon = np.array([0.0, 0.0, BOND_LENGTHS["C-C"]])
    polar = 180.0 - ANGLES["trigonal"]
    oxygen = carbon + _spherical(BOND_LENGTHS["C=O"], polar, 0.0)
    hydrogen = carbon + _spherical(BOND_LENGTHS["C-H"], polar, 180.0)
    return FunctionalGroup(
        key="CHO",
        name="Aldehído",
        formula="-CHO",
        symbols=("C", "O", "H"),
        positions=np.array([carbon, oxygen, hydrogen]),
        note="Carbonilo terminal, intermedio habitual en la oxidación.",
    )


def _amide() -> FunctionalGroup:
    carbon = np.array([0.0, 0.0, BOND_LENGTHS["C-C"]])
    polar = 180.0 - ANGLES["trigonal"]
    oxygen = carbon + _spherical(BOND_LENGTHS["C=O"], polar, 0.0)
    nitrogen = carbon + _spherical(BOND_LENGTHS["C-N"], polar, 180.0)
    direction = nitrogen - carbon
    direction = direction / np.linalg.norm(direction)
    # Amide N is planar (conjugated with C=O), so the H's stay near that plane.
    perpendicular = np.array([0.0, 1.0, 0.0])
    hydrogens = [
        nitrogen + BOND_LENGTHS["N-H"] * (
            math.cos(math.radians(60.0)) * direction
            + sign * math.sin(math.radians(60.0)) * perpendicular
        )
        for sign in (1.0, -1.0)
    ]
    return FunctionalGroup(
        key="CONH2",
        name="Amida",
        formula="-CONH2",
        symbols=("C", "O", "N", "H", "H"),
        positions=np.array([carbon, oxygen, nitrogen, *hydrogens]),
        note=(
            "El nitrógeno amídico es plano, no piramidal, porque conjuga con "
            "el carbonilo. Esa planaridad es la diferencia con una amina."
        ),
    )


def _carbonyl() -> FunctionalGroup:
    return FunctionalGroup(
        key="O",
        name="Carbonilo",
        formula="=O",
        symbols=("O",),
        positions=np.array([[0.0, 0.0, BOND_LENGTHS["C=O"]]]),
        valence_used=2,
        note=(
            "Doble enlace: consume dos valencias, así que solo tiene sentido "
            "en un carbono de borde con coordinación 2."
        ),
    )


def _thiol() -> FunctionalGroup:
    sulfur = np.array([0.0, 0.0, BOND_LENGTHS["C-S"]])
    hydrogen = sulfur + _spherical(
        BOND_LENGTHS["S-H"], 180.0 - ANGLES["C-S-H"], 0.0
    )
    return FunctionalGroup(
        key="SH",
        name="Tiol",
        formula="-SH",
        symbols=("S", "H"),
        positions=np.array([sulfur, hydrogen]),
        note=(
            "El azufre es grande: el enlace C-S (1.82 Å) distorsiona la red "
            "más que N u O, y casi siempre necesita relajación local."
        ),
    )


def _methyl() -> FunctionalGroup:
    carbon = np.array([0.0, 0.0, BOND_LENGTHS["C-C"]])
    polar = 180.0 - ANGLES["tetrahedral"]
    hydrogens = [
        carbon + _spherical(BOND_LENGTHS["C-H"], polar, azimuth)
        for azimuth in (0.0, 120.0, 240.0)
    ]
    return FunctionalGroup(
        key="CH3",
        name="Metilo",
        formula="-CH3",
        symbols=("C", "H", "H", "H"),
        positions=np.array([carbon, *hydrogens]),
        note="Sustituyente alquílico simple, débilmente donador.",
    )


def _epoxide() -> FunctionalGroup:
    # Bridges two adjacent carbons: attach.py places it over their midpoint.
    # Height above the midpoint follows from the C-O length and half the C-C
    # distance: h = sqrt(d_CO^2 - (d_CC/2)^2).
    half_cc = 1.42 / 2.0
    height = math.sqrt(BOND_LENGTHS["C-O"] ** 2 - half_cc ** 2)
    return FunctionalGroup(
        key="epoxy",
        name="Epóxido",
        formula="-O- (puente)",
        symbols=("O",),
        positions=np.array([[0.0, 0.0, height]]),
        valence_used=1,
        bridging=True,
        note=(
            "Puente sobre dos carbonos vecinos del plano basal, el grupo más "
            "característico del óxido de grafeno. Convierte ambos carbonos en "
            "sp3 y arruga la lámina: cuenta con relajarla."
        ),
    )


#: All available groups, keyed by their short identifier.
GROUPS: dict[str, FunctionalGroup] = {
    group.key: group
    for group in (
        _hydrogen(),
        _hydroxyl(),
        _amine(),
        _nitro(),
        _nitrile(),
        _carboxyl(),
        _aldehyde(),
        _amide(),
        _carbonyl(),
        _thiol(),
        _methyl(),
        _epoxide(),
    )
}

#: Groups containing nitrogen, which is what most N-doping studies want.
NITROGEN_GROUPS: tuple[str, ...] = ("NH2", "NO2", "CN", "CONH2")

#: Groups that only make sense on an edge carbon (they need a free valence,
#: or consume two).
EDGE_ONLY_GROUPS: tuple[str, ...] = ("O",)

#: Groups that bridge two carbons.
BRIDGING_GROUPS: tuple[str, ...] = ("epoxy",)


def get_group(key: str) -> FunctionalGroup:
    """Look up a group by key, with a helpful error listing the options."""
    try:
        return GROUPS[key]
    except KeyError:
        raise ValueError(
            f"Grupo funcional desconocido: '{key}'. "
            f"Disponibles: {', '.join(sorted(GROUPS))}."
        ) from None


def describe_groups() -> str:
    """Return a formatted table of every available group."""
    lines = [f"{'clave':8s} {'fórmula':14s} {'átomos':7s} nombre", "-" * 60]
    for key in sorted(GROUPS):
        group = GROUPS[key]
        lines.append(
            f"{group.key:8s} {group.formula:14s} {len(group):<7d} {group.name}"
        )
    return "\n".join(lines)
