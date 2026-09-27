"""H(k) and S(k) of a structure under a :class:`~tbkit.params.TBModel`.

Molecules and flakes (no periodic axis) use k = 0 and real matrices.
Periodic structures sum over the images the neighbour list finds:
``H_ij(k) = sum_R h_ij(R) exp(i k·R)``, with k Cartesian in 1/Å (2π
included) and R the lattice vector of the image of j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .basis import Basis
from .params import TBModel
from .slater_koster import block


@dataclass
class Bond:
    i: int
    j: int
    vector: np.ndarray       # r_j + R - r_i, Å
    shift: np.ndarray        # R in cell coordinates (integers)


@dataclass
class System:
    """A structure prepared for a model: basis, bonds and real-space blocks.

    Build once (:meth:`build`), then evaluate :meth:`hamiltonian` at any k.
    """

    atoms: Atoms
    model: TBModel
    basis: Basis
    bonds: list[Bond]
    h_blocks: list[np.ndarray]
    s_blocks: list[np.ndarray]
    onsite: np.ndarray

    @property
    def periodic(self) -> bool:
        return bool(self.atoms.get_pbc().any())

    @property
    def electrons(self) -> float:
        """Neutral electron count of the orbitals in the model."""
        coordination = self.coordination()
        return float(sum(self.model.electrons_of(self.atoms[a].symbol, coordination[a])
                         for a in self.basis.atoms))

    def coordination(self) -> dict[int, int]:
        """Neighbours of each atom within 1.8 Å (all elements, H included)."""
        i, _ = neighbor_list("ij", self.atoms, 1.8, self_interaction=False)
        counts = np.bincount(i, minlength=len(self.atoms))
        return {a: int(counts[a]) for a in range(len(self.atoms))}

    @classmethod
    def build(cls, atoms: Atoms, model: TBModel) -> "System":
        problems = model.check()
        if problems:
            raise ValueError("Modelo incompleto:\n" + "\n".join(problems))
        basis = Basis.build(atoms, model)
        symbols = atoms.get_chemical_symbols()
        cutoff = model.cutoff()
        bonds, h_blocks, s_blocks = [], [], []
        if cutoff > 0:
            ii, jj, dd, ss = neighbor_list("ijDS", atoms, cutoff, self_interaction=False)
            for i, j, vec, shift in zip(ii, jj, dd, ss, strict=True):
                if i not in basis.first or j not in basis.first:
                    continue
                oa, ob = model.orbitals[symbols[i]], model.orbitals[symbols[j]]
                hb = block(model, model.hopping, symbols[i], oa, symbols[j], ob, vec)
                sb = (block(model, model.overlap, symbols[i], oa, symbols[j], ob, vec)
                      if not model.orthogonal else None)
                if not hb.any() and (sb is None or not sb.any()):
                    continue
                bonds.append(Bond(int(i), int(j), np.asarray(vec), np.asarray(shift)))
                h_blocks.append(hb)
                s_blocks.append(sb)
        onsite = np.array([model.onsite[o.element][o.name[0]] for o in basis.orbitals])
        return cls(atoms, model, basis, bonds, h_blocks, s_blocks, onsite)

    def kpoint_cartesian(self, fractional) -> np.ndarray:
        """Fractional k (in units of the reciprocal vectors) to Cartesian 1/Å."""
        return np.asarray(fractional, dtype=float) @ self.atoms.cell.reciprocal() * 2 * np.pi

    def hamiltonian(self, k_fractional=(0.0, 0.0, 0.0),
                    extra_onsite: Optional[np.ndarray] = None
                    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """``(H, S)`` at ``k`` (fractional). ``S`` is None for an orthogonal model.

        ``extra_onsite`` adds a diagonal (per orbital) term, for spin or charge
        potentials.
        """
        n = self.basis.size
        k = np.asarray(k_fractional, dtype=float)
        complex_needed = self.periodic and bool(np.any(k))
        dtype = complex if complex_needed else float
        h = np.zeros((n, n), dtype=dtype)
        s = None if self.model.orthogonal else np.eye(n, dtype=dtype)
        for bond, hb, sb in zip(self.bonds, self.h_blocks, self.s_blocks, strict=True):
            phase = np.exp(2j * np.pi * float(k @ bond.shift)) if complex_needed else 1.0
            ri = self.basis.of_atom(bond.i)
            rj = self.basis.of_atom(bond.j)
            h[ri.start:ri.stop, rj.start:rj.stop] += hb * phase
            if s is not None and sb is not None:
                s[ri.start:ri.stop, rj.start:rj.stop] += sb * phase
        diagonal = self.onsite if extra_onsite is None else self.onsite + extra_onsite
        h[np.diag_indices(n)] += diagonal
        return h, s
