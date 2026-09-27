"""Which orbitals sit on which atom: the index map every matrix uses."""

from __future__ import annotations

from dataclasses import dataclass

from ase import Atoms

from .params import TBModel


@dataclass(frozen=True)
class Orbital:
    atom: int
    element: str
    name: str            # "s", "px", "py", "pz"


@dataclass
class Basis:
    """The model's orbitals on one structure, in matrix order.

    Atoms whose element the model does not describe (H in a π model) carry
    no orbitals: :attr:`atoms` lists the ones that do.
    """

    orbitals: list[Orbital]
    atoms: list[int]
    first: dict[int, int]            # atom -> index of its first orbital

    @property
    def size(self) -> int:
        return len(self.orbitals)

    def of_atom(self, atom: int) -> range:
        start = self.first[atom]
        stop = start
        while stop < self.size and self.orbitals[stop].atom == atom:
            stop += 1
        return range(start, stop)

    @classmethod
    def build(cls, atoms: Atoms, model: TBModel) -> "Basis":
        orbitals: list[Orbital] = []
        included: list[int] = []
        first: dict[int, int] = {}
        for index, element in enumerate(atoms.get_chemical_symbols()):
            names = model.orbitals.get(element)
            if not names:
                continue
            included.append(index)
            first[index] = len(orbitals)
            orbitals += [Orbital(index, element, name) for name in names]
        if not orbitals:
            raise ValueError(f"El modelo '{model.name}' no describe ningún elemento de "
                             f"{atoms.get_chemical_formula()}.")
        return cls(orbitals, included, first)
