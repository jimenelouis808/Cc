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

from dataclasses import dataclass, field as dataclass_field
from typing import Optional

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
    history: list[float] = dataclass_field(default_factory=list)
    #: Converged Δq (electrons per model atom), γ and orbital shifts (eV),
    #: kept for the forces.
    dq: np.ndarray = dataclass_field(default=None, repr=False)
    gamma: np.ndarray = dataclass_field(default=None, repr=False)
    shift: np.ndarray = dataclass_field(default=None, repr=False)

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


def anderson(inputs: list[np.ndarray], residuals: list[np.ndarray], mixing: float
             ) -> np.ndarray:
    """Next input by Anderson (Pulay) mixing of the last few iterations.

    Minimises the linear combination of past residuals ``r = out - in`` and
    steps ``mixing`` along the combined residual; with one iteration it is
    linear mixing. The standard accelerator of SCF loops.
    """
    x, r = inputs[-1], residuals[-1]
    if len(inputs) == 1:
        return x + mixing * r
    dx = np.array([inputs[i + 1] - inputs[i] for i in range(len(inputs) - 1)])
    dr = np.array([residuals[i + 1] - residuals[i] for i in range(len(residuals) - 1)])
    coefficients, *_ = np.linalg.lstsq(dr.T, r, rcond=None)
    x_bar = x - dx.T @ coefficients
    r_bar = r - dr.T @ coefficients
    return x_bar + mixing * r_bar


def self_consistent(system: System, U=None, charge: float = 0.0, kT: float = 0.01,
                    mixing: float = 0.3, tol: float = 1e-9, max_iter: int = 500,
                    history_length: int = 6, field: Optional[np.ndarray] = None,
                    initial_dq: Optional[np.ndarray] = None) -> SCCResult:
    """Solve with self-consistent Mulliken charges (finite systems).

    Parameters
    ----------
    U
        Hubbard U per element (eV); defaults to the model's.
    kT
        Fermi-Dirac width, eV.
    mixing, history_length
        Anderson mixing step and how many iterations it combines.
    tol
        Convergence on the largest change of an atomic charge between input
        and output, e. Forces need a tight value: their error is of the order
        of the residual charge times the potential it creates.
    field
        Uniform external electric field (V/Å): adds ``e E·(R_A - R_centre)``
        to the orbital energies of atom A, so the induced charges screen it
        (polarizability with local fields at the monopole level).
    initial_dq
        Starting Δq (e.g. the zero-field solution, for a small field).
    """
    if system.periodic:
        raise ValueError("SCC en sistemas periódicos necesita una suma de Ewald (no "
                         "implementada). Usa un fragmento finito.")
    gamma = gamma_matrix(system, U)
    atoms = system.basis.atoms
    n0 = _neutral(system)
    owner = np.array([atoms.index(o.atom) for o in system.basis.orbitals])
    neutral_atom = np.array([n0[system.basis.of_atom(a).start:system.basis.of_atom(a).stop].sum()
                             for a in atoms])
    external = np.zeros(len(atoms))
    if field is not None:
        positions = system.atoms.get_positions()[atoms]
        external = (positions - positions.mean(axis=0)) @ np.asarray(field, dtype=float)
    dq_in = np.zeros(len(atoms)) if initial_dq is None else np.array(initial_dq, dtype=float)
    inputs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    history: list[float] = []
    converged = False
    for iteration in range(1, max_iter + 1):
        solution = _solve_shifted(system, (gamma @ dq_in + external)[owner], charge, kT)
        pops = populations(solution).sum(axis=0)
        dq_out = np.array([pops[system.basis.of_atom(a).start:system.basis.of_atom(a).stop].sum()
                           for a in atoms]) - neutral_atom
        residual = dq_out - dq_in
        change = float(np.abs(residual).max())
        history.append(change)
        if change < tol:
            converged = True
            break
        inputs = (inputs + [dq_in])[-history_length:]
        residuals = (residuals + [residual])[-history_length:]
        dq_in = anderson(inputs, residuals, mixing)
    shift_in = gamma @ dq_in
    # The exact functional of the density the solution has:
    # Σ f ε counts V_in on the whole population q0 + Δq_out (the external
    # potential, if any, stays in: it is part of the energy in the field).
    energy = solution.band_energy() - float(shift_in @ (neutral_atom + dq_out)) \
        + 0.5 * float(dq_out @ gamma @ dq_out)
    charges = {a: float(-q) for a, q in zip(atoms, dq_out, strict=True)}
    return SCCResult(solution, charges, iteration, converged, energy, history, dq_out, gamma,
                     shift_in[owner])


def energy_and_forces(result: SCCResult, need_forces: bool = True):
    """Free energy and forces of a converged SCC solution (DFTB2-like).

    ``E = Tr[ρ H0] + ½ Δq γ Δq - T S + E_rep`` and

        ∂E/∂R = Tr[ρ ∂H0] - Tr[(W - ½ ρ∘(V_μ + V_ν)) ∂S] + ½ Σ Δq_A Δq_B ∂γ_AB + ∂E_rep

    with W built from the eigenvalues of the full (shifted) Hamiltonian;
    the ½ ρ∘(V+V) term accounts for the shift riding on S.
    """
    from .forces import band_forces, entropy_term

    solution = result.solution
    system = solution.system
    if system.model.repulsive is None:
        raise ValueError(f"El modelo '{system.model.name}' no tiene parte repulsiva.")
    ts = entropy_term(solution)
    e_rep, f_rep = system.model.repulsive.energy_and_forces(system.atoms)
    energy = result.energy - ts + e_rep
    if not need_forces:
        return energy, None, {"scc": result.energy, "entropy_TS": ts, "repulsive": e_rep}
    c = solution.vectors[0, 0]
    f = solution.occupations[0, 0]
    rho = (c * f) @ c.T
    weighted = (c * (f * solution.energies[0, 0])) @ c.T
    if not system.model.orthogonal:
        weighted = weighted - 0.5 * rho * (result.shift[:, None] + result.shift[None, :])
    forces = band_forces(solution, energy_weighted=[weighted]) + f_rep
    # Second-order term: γ depends on the distances between model atoms.
    atoms = system.basis.atoms
    positions = system.atoms.get_positions()[atoms]
    vectors = positions[:, None] - positions[None, :]
    r = np.linalg.norm(vectors, axis=-1)
    # γ = C / sqrt(r² + a²)  =>  dγ/dr = -C r / (r² + a²)^{3/2} = -r γ³ / C².
    dgamma = -r * result.gamma ** 3 / COULOMB ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(r[..., None] > 0, vectors / r[..., None], 0.0)
    pair = np.outer(result.dq, result.dq) * dgamma
    gradient = np.einsum("ab,abx->ax", pair, unit)       # ∂E2/∂R_A
    for index, atom in enumerate(atoms):
        forces[atom] -= gradient[index]
    return energy, forces, {"scc": result.energy, "entropy_TS": ts, "repulsive": e_rep}


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
