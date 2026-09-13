"""Assigning classical force-field types from local chemistry.

A reactive potential like AIREBO needs no atom types: it works out bonding
from geometry. A classical force field does not — every atom needs a type
carrying its Lennard-Jones parameters and its partial charge, and those
depend on the atom's chemical environment. A carbon next to a graphitic
nitrogen is not the same atom as one deep in the basal plane, and giving both
the same charge throws away exactly the effect a doping study is about.

This module walks the bond graph and assigns types from what it finds. It is
where carbonforge has something to offer that a generic tool does not: the
structure already knows it has a pyridinic nitrogen or a hydroxyl group,
because carbonforge put them there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ase import Atoms

from ..topology.graph import build_bond_graph


@dataclass(frozen=True)
class AtomType:
    """One classical force-field type.

    Attributes
    ----------
    name
        Identifier used in the LAMMPS data file comments.
    element
        Underlying chemical element.
    description
        What chemical environment this type represents.
    """

    name: str
    element: str
    description: str


#: Every type this module can assign, keyed by name.
ATOM_TYPES: dict[str, AtomType] = {
    # --- electrode carbon ------------------------------------------------
    "C_sp2": AtomType("C_sp2", "C", "Carbono grafítico basal (3 vecinos C)"),
    "C_edge": AtomType("C_edge", "C", "Carbono de borde (2 vecinos)"),
    "C_N": AtomType("C_N", "C", "Carbono vecino de un nitrógeno"),
    "C_O": AtomType("C_O", "C", "Carbono unido a oxígeno (sp3 en óxido)"),
    "C_sp3": AtomType("C_sp3", "C", "Carbono sp3 (4 vecinos)"),
    # --- nitrogen configurations ----------------------------------------
    "N_graph": AtomType("N_graph", "N", "N grafítico: sustituye un C, 3 vecinos"),
    "N_pyri": AtomType("N_pyri", "N", "N piridínico: borde de vacante, 2 vecinos"),
    "N_pyrr": AtomType("N_pyrr", "N", "N pirrólico: 2 vecinos C más un H"),
    "N_amine": AtomType("N_amine", "N", "N de amina (-NH2)"),
    "N_nitro": AtomType("N_nitro", "N", "N de nitro (-NO2)"),
    # --- oxygen groups ---------------------------------------------------
    "O_hydroxyl": AtomType("O_hydroxyl", "O", "O de hidroxilo (-OH)"),
    "O_epoxy": AtomType("O_epoxy", "O", "O de epóxido (puente)"),
    "O_carbonyl": AtomType("O_carbonyl", "O", "O de carbonilo (=O)"),
    "O_carboxyl": AtomType("O_carboxyl", "O", "O de carboxilo"),
    # --- hydrogen --------------------------------------------------------
    "H_C": AtomType("H_C", "H", "H unido a carbono"),
    "H_O": AtomType("H_O", "H", "H de hidroxilo o carboxilo"),
    "H_N": AtomType("H_N", "H", "H unido a nitrógeno"),
    # --- other -----------------------------------------------------------
    "S_thiol": AtomType("S_thiol", "S", "S de tiol (-SH)"),
    "B_sub": AtomType("B_sub", "B", "Boro sustitucional"),
    "P_sub": AtomType("P_sub", "P", "Fósforo sustitucional"),
}


@dataclass
class TypingResult:
    """Per-atom type assignment plus what could not be resolved."""

    types: list[str] = field(default_factory=list)
    unresolved: list[tuple[int, str]] = field(default_factory=list)

    @property
    def unique_types(self) -> list[str]:
        """Distinct types present, in a stable order."""
        return sorted(set(self.types))

    def counts(self) -> dict[str, int]:
        return {t: self.types.count(t) for t in self.unique_types}

    def summary(self) -> str:
        lines = [f"{len(self.unique_types)} tipos asignados:"]
        for name, count in sorted(self.counts().items()):
            spec = ATOM_TYPES.get(name)
            note = f" — {spec.description}" if spec else ""
            lines.append(f"  {name:12s} x{count}{note}")
        if self.unresolved:
            lines.append(f"\n{len(self.unresolved)} átomo(s) sin tipar:")
            for index, reason in self.unresolved[:10]:
                lines.append(f"  átomo {index}: {reason}")
            if len(self.unresolved) > 10:
                lines.append(f"  ... y {len(self.unresolved) - 10} más")
        return "\n".join(lines)


def _neighbour_elements(graph, symbols: list[str], index: int) -> list[str]:
    if index not in graph:
        return []
    return sorted(symbols[j] for j in graph.neighbors(index))


def assign_types(atoms: Atoms) -> TypingResult:
    """Assign a force-field type to every atom from its bonding environment.

    Parameters
    ----------
    atoms
        Structure to type. Bonds are inferred from covalent radii.

    Returns
    -------
    TypingResult
        A type per atom, plus a list of atoms whose environment did not match
        any known pattern — reported rather than silently given a default,
        since a wrong type means a wrong charge.

    Notes
    -----
    Nitrogen is classified by coordination, which distinguishes graphitic
    (three carbon neighbours, substituting in the plane) from pyridinic (two,
    on a vacancy rim) from pyrrolic (two carbons plus a hydrogen). Those are
    the three configurations that separate in N 1s XPS and they carry
    different charges, so lumping them together would defeat the purpose.
    """
    graph = build_bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()
    result = TypingResult()

    for index, symbol in enumerate(symbols):
        neighbours = _neighbour_elements(graph, symbols, index)
        degree = len(neighbours)
        assigned: Optional[str] = None

        if symbol == "C":
            if "N" in neighbours:
                assigned = "C_N"
            elif "O" in neighbours:
                assigned = "C_O"
            elif degree >= 4:
                assigned = "C_sp3"
            elif degree == 3:
                assigned = "C_sp2"
            elif degree == 2:
                assigned = "C_edge"

        elif symbol == "N":
            carbons = neighbours.count("C")
            oxygens = neighbours.count("O")
            hydrogens = neighbours.count("H")
            if oxygens >= 2:
                assigned = "N_nitro"
            elif hydrogens >= 2:
                assigned = "N_amine"
            elif carbons == 3:
                assigned = "N_graph"
            elif carbons == 2 and hydrogens == 1:
                assigned = "N_pyrr"
            elif carbons == 2:
                assigned = "N_pyri"
            elif carbons == 1 and hydrogens >= 1:
                assigned = "N_amine"

        elif symbol == "O":
            carbons = neighbours.count("C")
            hydrogens = neighbours.count("H")
            if hydrogens >= 1:
                assigned = "O_hydroxyl"
            elif carbons >= 2:
                # Two carbons and no hydrogen is the epoxide bridge.
                assigned = "O_epoxy"
            elif carbons == 1:
                assigned = "O_carbonyl"

        elif symbol == "H":
            if "O" in neighbours:
                assigned = "H_O"
            elif "N" in neighbours:
                assigned = "H_N"
            elif "C" in neighbours:
                assigned = "H_C"

        elif symbol == "S":
            assigned = "S_thiol"
        elif symbol == "B":
            assigned = "B_sub"
        elif symbol == "P":
            assigned = "P_sub"

        if assigned is None:
            result.unresolved.append((
                index,
                f"{symbol} con {degree} vecino(s) ({', '.join(neighbours) or 'ninguno'}): "
                "entorno no reconocido",
            ))
            # Fall back to the plain element type so the run can proceed, but
            # the atom stays on the unresolved list.
            assigned = f"{symbol}_sp2" if symbol == "C" else f"{symbol}_sub"
        result.types.append(assigned)

    return result


def refine_carboxyl_oxygens(atoms: Atoms, types: list[str]) -> list[str]:
    """Retag carboxyl oxygens, which the generic rules call carbonyl.

    A carboxyl group has one double-bonded and one hydroxyl oxygen on the
    same carbon. The generic pass sees them separately; this pass notices the
    pair and marks both as carboxyl, which carry different charges from an
    isolated ketone oxygen.
    """
    graph = build_bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()
    refined = list(types)

    for index, symbol in enumerate(symbols):
        if symbol != "C" or index not in graph:
            continue
        oxygen_neighbours = [
            j for j in graph.neighbors(index) if symbols[j] == "O"
        ]
        if len(oxygen_neighbours) != 2:
            continue
        # Two oxygens on one carbon: a carboxyl (or an ester).
        for j in oxygen_neighbours:
            refined[j] = "O_carboxyl"
    return refined
