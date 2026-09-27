"""Self-consistent charges: electrostatics on top of the tight-binding model.

Charge flowing between atoms changes the potential it flows into. In the
second-order (DFTB2-like) form used here, each atom's net population
change ``Δq_A`` (electrons) shifts the energies of every orbital by

    V_A = sum_B γ_AB Δq_B,    H_μν += ½ S_μν (V_A + V_B)   (μ on A, ν on B)

with γ from the Klopman-Ohno interpolation
``γ_AB = e² / sqrt(r² + (e²/Ū)²)``, ``Ū = (U_A + U_B)/2``: the Hubbard U on
the same atom, bare Coulomb at long range. ``e²/(4πε0) = 14.3996 eV Å``.

Without it a tight-binding model over-transfers charge to electronegative
atoms (nothing pushes back); with it, dopant charges come out screened.
Finite systems only: a periodic system would need an Ewald sum, which is
not implemented, and is refused rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import populations
from .hamiltonian import System
from .hubbard import _neutral, _orbital_u
from .solver import Solution, _fermi_dirac, fermi_level

#: e² / (4 π ε0), eV Å.
COULOMB = 14.399645


@dataclass
class SCCResult:
    solution: Solution
    charges: dict[int, float]          # net charge per atom, e (+ = electrons lost)
    iterations: int
    converged: bool
    energy: float
    history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        state = "convergido" if self.converged else "SIN CONVERGER"
        top = max(self.charges.items(), key=lambda kv: abs(kv[1]))
        return (f"SCC ({state}, {self.iterations} iteraciones): E = {self.energy:.6f} eV; "
                f"carga máxima {top[1]:+.4f} e en el átomo {top[0]}")


def gamma_matrix(system: System, U=None) -> np.ndarray:
    """γ between the atoms of the model (Klopman-Ohno), eV."""
    atoms = system.basis.atoms
    u_orbital = _orbital_u(system, U)
    u_atom = np.array([u_orbital[system.basis.first[a]] for a in atoms])
    if np.any(u_atom <= 0):
        raise ValueError("SCC necesita U > 0 en todos los elementos del modelo.")
    positions = system.atoms.get_positions()[atoms]
    r = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    ubar = 0.5 * (u_atom[:, None] + u_atom[None, :])
    return COULOMB / np.sqrt(r ** 2 + (COULOMB / ubar) ** 2)


def self_consistent(system: System, U=None, charge: float = 0.0, kT: float = 0.01,
                    mixing: float = 0.3, tol: float = 1e-7, max_iter: int = 500) -> SCCResult:
    """Solve with self-consistent Mulliken charges (finite systems)."""
    if system.periodic:
        raise ValueError("SCC en sistemas periódicos necesita una suma de Ewald (no "
                         "implementada). Usa un fragmento finito.")
    gamma = gamma_matrix(system, U)
    atoms = system.basis.atoms
    n0 = _neutral(system)
    owner = np.array([atoms.index(o.atom) for o in system.basis.orbitals])
    neutral_atom = np.array([n0[system.basis.of_atom(a).start:system.basis.of_atom(a).stop].sum()
                             for a in atoms])
    dq = np.zeros(len(atoms))
    history: list[float] = []
    solution = None
    converged = False
    for iteration in range(1, max_iter + 1):
        shift = gamma @ dq                      # per atom, eV
        solution = _solve_shifted(system, shift[owner], charge, kT)
        pops = populations(solution).sum(axis=0)
        new = np.array([pops[system.basis.of_atom(a).start:system.basis.of_atom(a).stop].sum()
                        for a in atoms]) - neutral_atom
        change = float(np.abs(new - dq).max())
        history.append(change)
        dq = (1 - mixing) * dq + mixing * new
        if change < tol:
            dq = new
            converged = True
            break
    # The band energy counts V_A on the whole population q0 + Δq; the
    # second-order energy is E0 + ½ Δq γ Δq.
    energy = solution.band_energy() - float((gamma @ dq) @ neutral_atom) \
        - 0.5 * float(dq @ gamma @ dq)
    charges = {a: float(-q) for a, q in zip(atoms, dq, strict=True)}
    return SCCResult(solution, charges, iteration, converged, energy, history)


def _solve_shifted(system: System, shift: np.ndarray, charge: float, kT: float) -> Solution:
    """Solve at Γ with ``H_μν += ½ S_μν (V_μ + V_ν)`` (just the diagonal when S = 1)."""
    from scipy.linalg import eigh

    h, s = system.hamiltonian((0, 0, 0))
    if s is None:
        h = h + np.diag(shift)
        e, c = eigh(h)
    else:
        h = h + 0.5 * s * (shift[:, None] + shift[None, :])
        e, c = eigh(h, s)
    return _package(system, e, c, s, charge, kT)


def _package(system, e, c, s, charge, kT):
    energies = e[None, None, :]
    weights = np.ones(1)
    electrons = system.electrons - charge
    mu = fermi_level(energies, weights, electrons, kT, 2.0)
    occupations = 2.0 * _fermi_dirac(energies, mu, kT)
    return Solution(system, np.zeros((1, 3)), weights, energies, c[None, None], occupations,
                    mu, electrons, kT, None if s is None else s[None])
