"""Functional groups as internal coordinates, so they can be rebuilt.

A hydroxyl is not a fixed set of Cartesian coordinates. It is "an oxygen
on the surface, a hydrogen on the oxygen at 108 degrees" -- and the
moment it is written that way, swapping the oxygen for sulphur gives a
thiol **with the right geometry**, because the two bond lengths are
recomputed rather than carried over. Store Cartesians instead and the
same swap leaves a sulphur sitting at oxygen's 1.43 Å instead of its own
1.82 Å, which is a 0.4 Å error in the one bond the group is defined by.

So a group here is a small Z-matrix: each atom names its parent, the
angle to the parent's own bond direction, and a dihedral. Only the
**angles** are stored. Every length comes from
:data:`~nanocarbon_lab.utils.constants.COVALENT_RADII`, which reproduces
real single bonds to within 0.04 Å across the elements this deals with
(C-O 1.420 against 1.43, C-S 1.810 against 1.82, O-H 0.970 against 0.97),
with a factor for bond order -- 0.86 for a double bond puts C=O at
1.221 against the literature 1.23.

That is what makes the library composable rather than a fixed menu.
:func:`substitute` swaps elements and the geometry follows.

Substitution is restricted to **the same valence**, and not out of
tidiness: an -OH whose oxygen becomes nitrogen is not a group with a
different element, it is a group with a missing bond. Oxygen goes to S,
Se and Te; nitrogen to P and As; carbon to Si and Ge; hydrogen to the
halogens. Anything else is refused with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..utils.constants import COVALENT_RADII

#: Multiplier on the single-bond length per bond order. A double bond is
#: about 0.86 of a single one and a triple about 0.78 -- which puts C=O
#: at 1.22 Å against the literature 1.23, and C#N at 1.15 against 1.16.
BOND_ORDER_SCALE = {1.0: 1.0, 1.5: 0.93, 2.0: 0.86, 3.0: 0.78}

#: How many bonds each element forms. Substitution may only stay within a
#: value, because changing it changes how many neighbours the atom needs
#: and the rest of the group no longer attaches.
VALENCE: dict[str, int] = {
    "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1,
    "O": 2, "S": 2, "Se": 2, "Te": 2,
    "N": 3, "P": 3, "As": 3, "B": 3,
    "C": 4, "Si": 4, "Ge": 4,
}


def bond_length(first: str, second: str, order: float = 1.0) -> float:
    """Covalent bond length (Å) between two elements at a bond order.

    Raises
    ------
    KeyError
        If either element has no covalent radius. Guessing one would put
        the whole group at the wrong distance from the surface.
    """
    try:
        base = COVALENT_RADII[first] + COVALENT_RADII[second]
    except KeyError as exc:
        raise KeyError(
            f"No covalent radius for {exc.args[0]!r}; add one to "
            "utils.constants.COVALENT_RADII before using it in a group."
        ) from None
    return base * BOND_ORDER_SCALE.get(float(order), 1.0)


@dataclass(frozen=True)
class GroupAtom:
    """One atom of a functional group, in internal coordinates.

    Attributes
    ----------
    symbol
        Element.
    parent
        Index of the atom it bonds to *within the group*, or ``-1`` for
        the atom that bonds to the surface.
    order
        Bond order to the parent; sets the length via
        :data:`BOND_ORDER_SCALE`.
    angle
        For an atom with a parent: the **bond angle** in degrees, as it
        is quoted in the literature -- the angle grandparent-parent-this,
        so a tetrahedral centre is 109.5 and a trigonal one 120. For the
        root atom (``parent == -1``) there is no grandparent, so this is
        instead the tilt away from the surface normal, and ``0`` stands
        the group straight up.
    dihedral
        Degrees of rotation about the parent's bond axis. Separates the
        two hydrogens of an amine, or the two oxygens of a carboxyl.
    """

    symbol: str
    parent: int = -1
    order: float = 1.0
    angle: float = 0.0
    dihedral: float = 0.0


@dataclass(frozen=True)
class FunctionalGroup:
    """A graftable group: its atoms, and what it does to the site.

    Attributes
    ----------
    name
        Registry key.
    atoms
        Z-matrix entries. ``atoms[0]`` must be the root (``parent == -1``)
        and is the atom bonded to the surface.
    site_hybridisation
        What the anchor atom becomes: ``"sp3"`` for a single bond (the
        site puckers out of the sheet) or ``"sp2"`` for a double one (it
        stays planar). Recorded rather than acted on, since relaxing the
        host is the caller's decision.
    bridging
        ``True`` for a group that spans **two** adjacent surface atoms,
        such as an epoxide. Those are placed differently and cannot be
        built from a single anchor.
    note
        One line on what the group is chemically, shown by the CLI.
    """

    name: str
    atoms: tuple[GroupAtom, ...]
    site_hybridisation: str = "sp3"
    bridging: bool = False
    note: str = ""

    @property
    def formula(self) -> str:
        """Hill-ish formula of the group itself, e.g. ``"CO2H"``."""
        counts: dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
        parts = []
        for symbol in sorted(counts, key=lambda s: (s != "C", s)):
            count = counts[symbol]
            parts.append(symbol if count == 1 else f"{symbol}{count}")
        return "".join(parts)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def elements(self) -> tuple[str, ...]:
        """Distinct elements, in first-appearance order."""
        seen: list[str] = []
        for atom in self.atoms:
            if atom.symbol not in seen:
                seen.append(atom.symbol)
        return tuple(seen)


def substitute(group: FunctionalGroup,
               swaps: dict[str, str],
               name: str | None = None) -> FunctionalGroup:
    """Swap elements, keeping the geometry and rebuilding the lengths.

    ``substitute(HYDROXYL, {"O": "S"})`` is a thiol, with the C-S bond at
    1.81 Å rather than oxygen's 1.42 -- because only the angles were ever
    stored and the lengths are recomputed on placement.

    Parameters
    ----------
    group
        The group to modify (not mutated).
    swaps
        ``{from_element: to_element}``, applied to every matching atom.
    name
        Name for the result; defaults to the original with the swap
        appended, e.g. ``"hydroxyl[O->S]"``.

    Raises
    ------
    ValueError
        If a swap changes valence -- an -OH whose oxygen became nitrogen
        would be a group with a missing bond, not a variant -- or if
        either element is unknown.
    """
    for source, target in swaps.items():
        if source not in VALENCE or target not in VALENCE:
            unknown = source if source not in VALENCE else target
            raise ValueError(
                f"Unknown element {unknown!r} for substitution. Known: "
                f"{', '.join(sorted(VALENCE))}."
            )
        if VALENCE[source] != VALENCE[target]:
            raise ValueError(
                f"{source} forms {VALENCE[source]} bonds and {target} forms "
                f"{VALENCE[target]}, so swapping one for the other leaves the "
                f"group with the wrong number of bonds rather than a variant "
                f"of it. Same-valence swaps for {source}: "
                f"{', '.join(viable_swaps(source))}."
            )
        if target not in COVALENT_RADII:
            raise ValueError(
                f"No covalent radius for {target!r}; the group's bond "
                "lengths could not be rebuilt."
            )

    atoms = tuple(replace(atom, symbol=swaps.get(atom.symbol, atom.symbol))
                  for atom in group.atoms)
    if name is None:
        tag = ",".join(f"{a}->{b}" for a, b in swaps.items())
        name = f"{group.name}[{tag}]"
    return replace(group, name=name, atoms=atoms)


def build_positions(group: FunctionalGroup,
                    anchor: np.ndarray,
                    normal: np.ndarray,
                    surface_symbol: str = "C",
                    twist: float = 0.0) -> tuple[list[str], np.ndarray]:
    """Place a group's atoms in space from its internal coordinates.

    Parameters
    ----------
    group
        The group to place. Bridging groups are rejected here; they span
        two sites and are placed by their own routine.
    anchor
        Position of the surface atom the group bonds to.
    normal
        Outward unit normal at that site. The root atom goes along it.
    surface_symbol
        Element of the anchor, so the first bond length is right: a group
        on a carbon wall and the same group on an MX2 sulphur sit at
        different distances.
    twist
        Degrees of rotation of the whole group about ``normal``. A group
        joined to a surface by one single bond really does rotate freely
        about it, so this is a genuine degree of freedom rather than a
        convenience -- and without it the placement inherits whichever
        arbitrary tangent the frame happened to pick, which decides
        whether a group's own hydrogen points at the wall. The caller
        picks the rotamer with the most clearance.

    Returns
    -------
    (symbols, positions)
        Ready to append to an :class:`ase.Atoms`.
    """
    if group.bridging:
        raise ValueError(
            f"{group.name!r} bridges two surface atoms; use the bridging "
            "placement rather than build_positions."
        )

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    anchor = np.asarray(anchor, dtype=float)

    # A frame at the site: normal plus any two perpendicular tangents.
    # The dihedral reference is arbitrary for the root atom, so a stable
    # arbitrary choice is exactly right here.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, helper)
    tangent /= np.linalg.norm(tangent)
    binormal = np.cross(normal, tangent)
    if twist:
        angle = np.radians(twist)
        tangent, binormal = (np.cos(angle) * tangent + np.sin(angle) * binormal,
                            -np.sin(angle) * tangent + np.cos(angle) * binormal)

    positions = np.zeros((len(group.atoms), 3), dtype=float)
    # Direction of the bond *into* each atom, needed as the reference
    # axis for its own children.
    axes = np.zeros((len(group.atoms), 3), dtype=float)

    for index, atom in enumerate(group.atoms):
        if atom.parent < 0:
            length = bond_length(surface_symbol, atom.symbol, atom.order)
            direction = _tilt(normal, tangent, binormal,
                              atom.angle, atom.dihedral)
            positions[index] = anchor + length * direction
            axes[index] = direction
        else:
            parent = atom.parent
            if parent >= index:
                raise ValueError(
                    f"{group.name!r}: atom {index} names parent {parent}, "
                    "which is not placed yet. Z-matrix entries must follow "
                    "their parents."
                )
            length = bond_length(group.atoms[parent].symbol, atom.symbol,
                                 atom.order)
            axis = axes[parent]
            perp = np.cross(axis, normal)
            if np.linalg.norm(perp) < 1e-8:
                perp = np.cross(axis, tangent)
            perp /= np.linalg.norm(perp)
            second = np.cross(axis, perp)
            # `atom.angle` is the bond angle at the parent, measured the
            # way chemistry quotes it. `_tilt` works from the parent
            # bond's *continuation*, so a 109.5 deg centre is a 70.5 deg
            # kink -- convert here rather than making every group
            # definition carry the supplement.
            direction = _tilt(axis, perp, second, 180.0 - atom.angle,
                              atom.dihedral)
            positions[index] = positions[parent] + length * direction
            axes[index] = direction

    return [atom.symbol for atom in group.atoms], positions


def build_bridging_positions(group: FunctionalGroup,
                             first: np.ndarray,
                             second: np.ndarray,
                             normal: np.ndarray,
                             surface_symbol: str = "C"
                             ) -> tuple[list[str], np.ndarray]:
    """Place a group that spans two adjacent surface atoms.

    An epoxide is not a group on a carbon, it is a group on a **bond**:
    the oxygen sits over the midpoint, equidistant from both carbons, and
    the triangle it makes with them is what the geometry has to satisfy.
    So the height is derived rather than chosen -- with the two anchors
    ``s`` apart and each bond of length ``L``, the apex is
    ``sqrt(L^2 - (s/2)^2)`` above the midpoint, and nothing else puts
    both bonds at their correct length at once.

    Parameters
    ----------
    group
        A group with ``bridging=True``.
    first, second
        Positions of the two surface atoms. ``second`` must already be
        the minimum-image copy nearest ``first``, or a bridge across a
        cell seam would be built across the whole cell instead.
    normal
        Outward direction at the bond, usually the mean of the two site
        normals. Only its component perpendicular to the bond is used.
    surface_symbol
        Element of the anchors, setting both bond lengths.

    Returns
    -------
    (symbols, positions)

    Raises
    ------
    ValueError
        If the group does not bridge, or if the two anchors are further
        apart than twice the bond length -- then no apex exists and the
        group cannot span them. A caller placing many bridges should
        treat that as "this bond is too long for this group", not as an
        error in the structure.
    """
    if not group.bridging:
        raise ValueError(
            f"{group.name!r} attaches to a single atom; use build_positions."
        )

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    along = second - first
    separation = float(np.linalg.norm(along))
    if separation < 1e-8:
        raise ValueError("The two bridged atoms are at the same position.")
    along /= separation

    root = group.atoms[0]
    length = bond_length(surface_symbol, root.symbol, root.order)
    half = 0.5 * separation
    if length <= half:
        raise ValueError(
            f"{group.name!r} cannot bridge atoms {separation:.2f} Å apart: its "
            f"{surface_symbol}-{root.symbol} bond is {length:.2f} Å, so the two "
            "bonds cannot meet above the midpoint."
        )

    # Only the part of the normal perpendicular to the bond can lift the
    # apex; the parallel part would slide it along the bond and make the
    # two bond lengths differ.
    normal = np.asarray(normal, dtype=float)
    perpendicular = normal - np.dot(normal, along) * along
    norm = np.linalg.norm(perpendicular)
    if norm < 1e-8:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, along)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(along, helper)
        norm = np.linalg.norm(perpendicular)
    perpendicular /= norm

    apex = 0.5 * (first + second) + np.sqrt(length**2 - half**2) * perpendicular

    positions = np.zeros((len(group.atoms), 3), dtype=float)
    axes = np.zeros((len(group.atoms), 3), dtype=float)
    positions[0] = apex
    axes[0] = perpendicular

    for index, atom in enumerate(group.atoms[1:], start=1):
        parent = atom.parent
        if parent >= index or parent < 0:
            raise ValueError(
                f"{group.name!r}: atom {index} names parent {parent}, which is "
                "not placed yet. Z-matrix entries must follow their parents."
            )
        axis = axes[parent]
        reference = np.cross(axis, along)
        if np.linalg.norm(reference) < 1e-8:
            reference = np.cross(axis, perpendicular)
        reference /= np.linalg.norm(reference)
        direction = _tilt(axis, reference, np.cross(axis, reference),
                          180.0 - atom.angle, atom.dihedral)
        positions[index] = positions[parent] + bond_length(
            group.atoms[parent].symbol, atom.symbol, atom.order) * direction
        axes[index] = direction

    return [atom.symbol for atom in group.atoms], positions


def _tilt(axis: np.ndarray, first: np.ndarray, second: np.ndarray,
          angle_deg: float, dihedral_deg: float) -> np.ndarray:
    """A unit vector ``angle_deg`` from ``axis``, rotated by ``dihedral``.

    ``angle_deg`` here is the deflection from ``axis``, **not** a bond
    angle: 0 continues straight on. :func:`build_positions` converts a
    quoted bond angle into it. ``first`` and ``second`` are any two unit
    vectors completing an orthonormal frame with ``axis``; which of them
    is which only sets where the dihedral is measured from.
    """
    angle = np.radians(angle_deg)
    dihedral = np.radians(dihedral_deg)
    perpendicular = np.cos(dihedral) * first + np.sin(dihedral) * second
    return np.cos(angle) * axis + np.sin(angle) * perpendicular


# --------------------------------------------------------------- registry

def _z(symbol: str, parent: int = -1, order: float = 1.0,
       angle: float = 0.0, dihedral: float = 0.0) -> GroupAtom:
    return GroupAtom(symbol, parent, order, angle, dihedral)


#: The groups that ship. Every one is written against O, N, S or C, so
#: `substitute` reaches the heavier congeners without new definitions:
#: hydroxyl becomes a thiol or selenol, amine becomes a phosphine,
#: methyl becomes a silyl.
GROUPS: dict[str, FunctionalGroup] = {
    "hydroxyl": FunctionalGroup(
        "hydroxyl",
        (_z("O"), _z("H", parent=0, angle=108.5)),
        note="-OH. The commonest oxidised site on a carbon nanotube, and "
             "what makes an oxidised tube disperse in water. Swap O for S "
             "to get a thiol.",
    ),
    "carboxyl": FunctionalGroup(
        "carboxyl",
        (_z("C"),
         _z("O", parent=0, order=2.0, angle=121.0),
         # Order 1.5, not 1.0: the hydroxyl oxygen of an acid is
         # conjugated with the carbonyl, which is why benzoic acid's
         # C-OH is 1.34 Å and not an alcohol's 1.43. A plain single bond
         # here was the one length in the library off by more than
         # 0.04 Å (1.42 against 1.34); 1.5 puts it at 1.32.
         _z("O", parent=0, order=1.5, angle=121.0, dihedral=180.0),
         _z("H", parent=2, angle=109.0)),
        note="-COOH. The handle of oxidised nanotube chemistry: it is what "
             "amide and ester couplings attach to, and it dominates the "
             "open ends and defect sites of an acid-treated tube.",
    ),
    "carbonyl": FunctionalGroup(
        "carbonyl",
        (_z("O", order=2.0),),
        site_hybridisation="sp2",
        note="=O, a ketone. Double-bonded, so the site stays planar rather "
             "than puckering -- the one common group that does not "
             "rehybridise its carbon to sp3.",
    ),
    "amine": FunctionalGroup(
        "amine",
        (_z("N"),
         _z("H", parent=0, angle=112.0),
         _z("H", parent=0, angle=112.0, dihedral=120.0)),
        note="-NH2. The other half of amide coupling, and the usual route "
             "to attaching a biomolecule. Swap N for P to get a phosphine.",
    ),
    "thiol": FunctionalGroup(
        "thiol",
        (_z("S"), _z("H", parent=0, angle=97.0)),
        note="-SH. On MX2 this is the ligand chemistry that heals a "
             "chalcogen vacancy; on carbon it is the handle for binding "
             "gold.",
    ),
    "methyl": FunctionalGroup(
        "methyl",
        (_z("C"),
         _z("H", parent=0, angle=109.5),
         _z("H", parent=0, angle=109.5, dihedral=120.0),
         _z("H", parent=0, angle=109.5, dihedral=240.0)),
        note="-CH3. The simplest alkyl, and the test case for whether a "
             "grafting density is sterically possible at all.",
    ),
    "nitro": FunctionalGroup(
        "nitro",
        (_z("N"),
         _z("O", parent=0, order=2.0, angle=127.0),
         _z("O", parent=0, order=2.0, angle=127.0, dihedral=180.0)),
        note="-NO2. Strongly electron-withdrawing; used to tune the work "
             "function of a carbon surface.",
    ),
    "aldehyde": FunctionalGroup(
        "aldehyde",
        (_z("C"),
         _z("O", parent=0, order=2.0, angle=121.0),
         _z("H", parent=0, angle=121.0, dihedral=180.0)),
        note="-CHO. The oxidation step between an alcohol and a carboxyl.",
    ),
    "fluorine": FunctionalGroup(
        "fluorine",
        (_z("F"),),
        note="-F. Fluorination is the one addition that goes to high "
             "coverage on a carbon sheet, all the way to CF stoichiometry.",
    ),
    "epoxide": FunctionalGroup(
        "epoxide",
        (_z("O"),),
        bridging=True,
        note="A single oxygen bridging two adjacent surface atoms. With "
             "the hydroxyl it is what the basal plane of graphene oxide is "
             "actually made of.",
    ),
}


def get_group(name: str) -> FunctionalGroup:
    """Look up a shipped group by name.

    Raises
    ------
    ValueError
        Listing what is available, since a typo is far likelier than a
        genuinely missing group.
    """
    try:
        return GROUPS[name]
    except KeyError:
        raise ValueError(
            f"Unknown group {name!r}. Available: {', '.join(sorted(GROUPS))}."
        ) from None


def describe(group: FunctionalGroup, surface_symbol: str = "C") -> str:
    """A one-line human summary, including the anchor bond it will make."""
    root = group.atoms[0]
    length = bond_length(surface_symbol, root.symbol, root.order)
    kind = "bridging two sites" if group.bridging else (
        f"{surface_symbol}-{root.symbol} {length:.2f} Å")
    plural = "atom" if group.n_atoms == 1 else "atoms"
    return (f"{group.name}: {group.formula}, {group.n_atoms} {plural}, {kind}, "
            f"site becomes {group.site_hybridisation}. {group.note}")


def viable_swaps(element: str) -> tuple[str, ...]:
    """Elements ``element`` may be replaced by, keeping the valence."""
    if element not in VALENCE:
        return ()
    return tuple(sorted(other for other, valence in VALENCE.items()
                        if valence == VALENCE[element] and other != element
                        and other in COVALENT_RADII))


def as_dict(group: FunctionalGroup) -> dict[str, Any]:
    """Serialisable form, for recording what was grafted."""
    return {
        "name": group.name,
        "formula": group.formula,
        "n_atoms": group.n_atoms,
        "elements": list(group.elements()),
        "bridging": group.bridging,
        "site_hybridisation": group.site_hybridisation,
    }


__all__ = [
    "BOND_ORDER_SCALE",
    "GROUPS",
    "VALENCE",
    "FunctionalGroup",
    "GroupAtom",
    "as_dict",
    "bond_length",
    "build_positions",
    "describe",
    "get_group",
    "substitute",
    "viable_swaps",
]
