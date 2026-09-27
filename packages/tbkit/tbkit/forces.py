"""Total energies and forces.

Energy (Mermin free energy, what the forces are the derivative of):

    F = Σ_{s,k,n} w_k f_n ε_n  -  T S_el  +  E_rep  (+ E_SCC)

``T S_el = -kT Σ w g [f ln f + (1-f) ln(1-f)]`` is the electronic entropy of
the Fermi-Dirac occupations; it vanishes as kT -> 0.

Forces from the band energy (Hellmann-Feynman for a non-orthogonal basis):

    ∂E_band/∂R = Σ_k w Re Tr[ρ(k) ∂H(k)/∂R - W(k) ∂S(k)/∂R]

with ``ρ = Σ f c c†`` and the energy-weighted ``W = Σ f ε c c†``. Only the
two-centre blocks depend on positions (on-site energies do not), so the
derivative is a sum over bonds; each block's derivative with respect to its
bond vector is taken by central differences (step 1e-5 Å, exact to
O(1e-10) for the smooth laws used). The tests check the result against
finite differences of the energy itself.

Not variational, hence without forces here: the mean-field Hubbard
solution (its potential depends on populations in a way this expression
does not cover).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .hamiltonian import System
from .slater_koster import block
from .solver import Solution

_STEP = 1e-5


def entropy_term(solution: Solution) -> float:
    """``T S_el`` (eV) of the occupations, ``kT`` from the solution."""
    degeneracy = 2.0 if solution.nspin == 1 else 1.0
    f = np.clip(solution.occupations / degeneracy, 1e-300, 1 - 1e-16)
    g = np.where((f > 1e-14) & (f < 1 - 1e-14),
                 f * np.log(f) + (1 - f) * np.log(1 - f), 0.0)
    return float(-solution.kT * degeneracy * np.einsum("k,skb->", solution.weights, g))


def band_free_energy(solution: Solution) -> float:
    return solution.band_energy() - entropy_term(solution)


def _block_derivatives(system: System, bond, table) -> np.ndarray:
    """``∂block/∂vector``: shape (3, n_i, n_j)."""
    symbols = system.atoms.get_chemical_symbols()
    oa, ob = system.model.orbitals[symbols[bond.i]], system.model.orbitals[symbols[bond.j]]
    out = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = _STEP
        plus = block(system.model, table, symbols[bond.i], oa, symbols[bond.j], ob,
                     bond.vector + step)
        minus = block(system.model, table, symbols[bond.i], oa, symbols[bond.j], ob,
                      bond.vector - step)
        out.append((plus - minus) / (2 * _STEP))
    return np.array(out)


def band_forces(solution: Solution, energy_weighted: Optional[list] = None) -> np.ndarray:
    """Forces (eV/Å) from the band energy, on every atom of the structure.

    ``energy_weighted`` may replace W(k) (the SCC correction does); by default
    ``W = Σ f ε c c†``.
    """
    system = solution.system
    forces = np.zeros((len(system.atoms), 3))
    rho, weighted = [], []
    for k in range(len(solution.kpts)):
        r = np.zeros((system.basis.size,) * 2, dtype=solution.vectors.dtype)
        w = np.zeros_like(r)
        for s in range(solution.nspin):
            c = solution.vectors[s, k]
            f = solution.occupations[s, k]
            r += (c * f) @ c.conj().T
            w += (c * (f * solution.energies[s, k])) @ c.conj().T
        rho.append(r)
        weighted.append(w)
    if energy_weighted is not None:
        weighted = energy_weighted
    phases_needed = system.periodic
    for bond in system.bonds:
        dh = _block_derivatives(system, bond, system.model.hopping)
        ds = (_block_derivatives(system, bond, system.model.overlap)
              if not system.model.orthogonal else None)
        ri, rj = system.basis.of_atom(bond.i), system.basis.of_atom(bond.j)
        gradient = np.zeros(3)
        for k, wk in enumerate(solution.weights):
            phase = np.exp(2j * np.pi * float(solution.kpts[k] @ bond.shift)) \
                if phases_needed else 1.0
            # Tr[ρ ∂H] restricted to this block: Σ_{μ∈i, ν∈j} ρ_νμ ∂H_μν.
            rho_block = rho[k][rj.start:rj.stop, ri.start:ri.stop]
            term = np.einsum("nm,amn->a", rho_block, dh)
            if ds is not None:
                w_block = weighted[k][rj.start:rj.stop, ri.start:ri.stop]
                term = term - np.einsum("nm,amn->a", w_block, ds)
            gradient += wk * np.real(phase * term)
        forces[bond.i] += gradient          # vector = r_j + R - r_i
        forces[bond.j] -= gradient
    return forces


def energy_and_forces(solution: Solution, need_forces: bool = True
                      ) -> tuple[float, Optional[np.ndarray], dict]:
    """Free energy, forces and the parts, for a non-SCC solution.

    Raises if the model has no repulsive term: without one the "energy"
    would bind atoms without limit, and forces would collapse the structure.
    """
    system = solution.system
    repulsive = system.model.repulsive
    if repulsive is None:
        raise ValueError(f"El modelo '{system.model.name}' no tiene parte repulsiva: sirve "
                         "para estructura electrónica, no para energías totales ni fuerzas.")
    if solution.nspin == 2 and solution.info.get("hubbard"):
        raise ValueError("Fuerzas con Hubbard de campo medio: no implementadas.")
    e_band = solution.band_energy()
    ts = entropy_term(solution)
    e_rep, f_rep = repulsive.energy_and_forces(system.atoms)
    parts = {"band": e_band, "entropy_TS": ts, "repulsive": e_rep}
    forces = band_forces(solution) + f_rep if need_forces else None
    return e_band - ts + e_rep, forces, parts
